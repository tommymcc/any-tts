//! Grouped-Query Attention (GQA) layer.
//!
//! GQA is the attention variant used by Qwen 2 / Qwen 2.5 / Qwen 3
//! family models. It reduces KV-cache memory by sharing key/value heads
//! across multiple query heads.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::kv_cache::{KvCache, KvCacheState};
use crate::tensor_utils::{apply_rotary_emb, cpu_flash_attention, KernelMode, RmsNorm};

/// Configuration for a GQA layer, extracted from model config.
#[derive(Debug, Clone)]
pub struct GqaConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    /// The kernels the layer runs; see [`KernelMode`].
    pub kernels: KernelMode,
}

impl GqaConfig {
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        rms_norm_eps: f64,
    ) -> Self {
        let head_dim = hidden_size / num_attention_heads;
        Self {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            attention_bias: false,
            kernels: KernelMode::Composite,
        }
    }

    /// Create a GqaConfig with an explicit head_dim (for models where
    /// head_dim != hidden_size / num_attention_heads, e.g. code predictor).
    pub fn with_head_dim(
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        rms_norm_eps: f64,
    ) -> Self {
        Self {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            attention_bias: false,
            kernels: KernelMode::Composite,
        }
    }

    pub fn with_attention_bias(mut self, attention_bias: bool) -> Self {
        self.attention_bias = attention_bias;
        self
    }

    /// Choose the kernels the layer runs. [`KernelMode::Fused`] takes the
    /// half-width RoPE tables of [`crate::tensor_utils::precompute_rope_half_freqs`].
    pub fn with_kernels(mut self, kernels: KernelMode) -> Self {
        self.kernels = kernels;
        self
    }
}

/// Grouped-Query Attention with rotary position embeddings.
pub struct GroupedQueryAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kernels: KernelMode,
    kv_cache: KvCache,
}

impl GroupedQueryAttention {
    /// Load attention weights from a VarBuilder.
    ///
    /// Expected weight names under `vb`:
    /// - `q_proj.weight` — (num_heads * head_dim, hidden_size)
    /// - `k_proj.weight` — (num_kv_heads * head_dim, hidden_size)
    /// - `v_proj.weight` — (num_kv_heads * head_dim, hidden_size)
    /// - `o_proj.weight` — (hidden_size, num_heads * head_dim)
    pub fn load(config: &GqaConfig, vb: VarBuilder) -> Result<Self> {
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;

        let q_proj = if config.attention_bias {
            candle_nn::linear(config.hidden_size, q_dim, vb.pp("q_proj"))?
        } else {
            candle_nn::linear_no_bias(config.hidden_size, q_dim, vb.pp("q_proj"))?
        };
        let k_proj = if config.attention_bias {
            candle_nn::linear(config.hidden_size, kv_dim, vb.pp("k_proj"))?
        } else {
            candle_nn::linear_no_bias(config.hidden_size, kv_dim, vb.pp("k_proj"))?
        };
        let v_proj = if config.attention_bias {
            candle_nn::linear(config.hidden_size, kv_dim, vb.pp("v_proj"))?
        } else {
            candle_nn::linear_no_bias(config.hidden_size, kv_dim, vb.pp("v_proj"))?
        };
        let o_proj = candle_nn::linear_no_bias(q_dim, config.hidden_size, vb.pp("o_proj"))?;

        // Qwen3 models have per-head QK-norm; Qwen2 models do not.
        let q_norm = RmsNorm::load(config.head_dim, config.rms_norm_eps, vb.pp("q_norm"))
            .ok()
            .map(|norm| norm.with_kernels(config.kernels));
        let k_norm = RmsNorm::load(config.head_dim, config.rms_norm_eps, vb.pp("k_norm"))
            .ok()
            .map(|norm| norm.with_kernels(config.kernels));

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            kernels: config.kernels,
            kv_cache: KvCache::new(),
        })
    }

    /// Forward pass with rotary embeddings and optional KV-cache.
    ///
    /// * `x`     — (batch, seq_len, hidden_size)
    /// * `cos`   — precomputed RoPE cos: (max_seq, head_dim) from
    ///   [`crate::tensor_utils::precompute_rope_freqs`] with
    ///   [`KernelMode::Composite`], or (max_seq, head_dim / 2) from
    ///   [`crate::tensor_utils::precompute_rope_half_freqs`] with
    ///   [`KernelMode::Fused`]
    /// * `sin`   — the matching sin table
    /// * `start_pos` — position offset for KV-cache (0 for full sequence)
    /// * `mask`  — optional causal attention mask
    pub fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        start_pos: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        if start_pos != self.kv_cache.len() {
            candle_core::bail!(
                "attention position {start_pos} differs from KV cache length {}",
                self.kv_cache.len()
            )
        }
        match self.kernels {
            KernelMode::Composite => {
                if cos.dim(1)? != self.head_dim || sin.dims() != cos.dims() {
                    candle_core::bail!(
                        "composite attention takes (max_seq, {}) RoPE tables from \
                         precompute_rope_freqs, got cos {:?} and sin {:?}",
                        self.head_dim,
                        cos.dims(),
                        sin.dims()
                    )
                }
                self.forward_composite(x, cos, sin, start_pos, mask)
            }
            KernelMode::Fused => {
                let (q, k, v) = self.normed_rotated_qkv(x, cos, sin, start_pos)?;
                let (k, v) = self.kv_cache.append(&k, &v)?;
                self.attention_output(&q, &k, &v, mask)
            }
        }
    }

    /// Forward pass over a whole sequence that neither reads nor fills the
    /// KV-cache, for models that run every position at once. Only
    /// [`KernelMode::Fused`] layers have it; `cos` and `sin` are as for
    /// [`Self::forward`].
    pub fn forward_uncached(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        if self.kernels != KernelMode::Fused {
            candle_core::bail!("uncached attention needs KernelMode::Fused")
        }
        let (q, k, v) = self.normed_rotated_qkv(x, cos, sin, 0)?;
        self.attention_output(&q, &k, &v, mask)
    }

    /// [`Self::forward`] with [`KernelMode::Composite`], the original
    /// elementwise implementation that every other model runs.
    fn forward_composite(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        start_pos: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, heads, seq_len, head_dim)
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply QK-norm if present (Qwen3-style)
        let q = if let Some(ref qn) = self.q_norm {
            qn.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref kn) = self.k_norm {
            kn.forward(&k)?
        } else {
            k
        };

        // Apply RoPE
        let cos_slice = cos
            .narrow(0, start_pos, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin_slice = sin
            .narrow(0, start_pos, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let (q, k) = apply_rotary_emb(&q, &k, &cos_slice, &sin_slice)?;

        // Update KV-cache
        let (k, v) = self.kv_cache.append(&k, &v)?;

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let attn_output =
            if let Some(cpu_attn_output) = cpu_flash_attention(&q, &k, &v, mask, scale)? {
                cpu_attn_output
            } else if seq_len == 1
                && mask.is_none()
                && matches!(x.device(), candle_core::Device::Metal(_))
            {
                candle_nn::ops::sdpa(&q.contiguous()?, &k, &v, None, false, scale, 1.0)?
            } else {
                // Repeat KV heads to match Q heads (GQA expansion)
                // Use repeat_interleave semantics: [h0,h0,h1,h1,...] not repeat: [h0..h7,h0..h7]
                // so that Q head i pairs with KV head i/repeat_factor. Broadcasting costs one
                // strided copy on CUDA, where `repeat` costs a transpose and cat per dimension.
                let repeat_factor = self.num_heads / self.num_kv_heads;
                let kv_len = k.dim(2)?;
                let k = if repeat_factor > 1 {
                    k.unsqueeze(2)?
                        .broadcast_as((
                            batch,
                            self.num_kv_heads,
                            repeat_factor,
                            kv_len,
                            self.head_dim,
                        ))?
                        .reshape((batch, self.num_heads, kv_len, self.head_dim))?
                } else {
                    k
                };
                let v = if repeat_factor > 1 {
                    v.unsqueeze(2)?
                        .broadcast_as((
                            batch,
                            self.num_kv_heads,
                            repeat_factor,
                            kv_len,
                            self.head_dim,
                        ))?
                        .reshape((batch, self.num_heads, kv_len, self.head_dim))?
                } else {
                    v
                };

                // Dense CPU GEMM (including Accelerate) has stricter matrix
                // and batch stride requirements than fused attention.
                let (k, v) = if x.device().is_cpu() {
                    (k.contiguous()?, v.contiguous()?)
                } else {
                    (k, v)
                };

                let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scale as f64, 0.0)?;
                let attn_weights = if let Some(mask) = mask {
                    attn_weights.broadcast_add(mask)?
                } else {
                    attn_weights
                };

                let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                attn_weights.matmul(&v)?
            };

        // Reshape back: (batch, num_heads, seq_len, head_dim) → (batch, seq_len, hidden)
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        // Output projection
        self.o_proj.forward(&attn_output)
    }

    /// Q, K and V projected from `x`, QK-normed (Qwen3-style, when the layer
    /// has the weights) and rotated with fused RoPE, as heads-first
    /// (batch, heads, seq_len, head_dim) views.
    fn normed_rotated_qkv(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        start_pos: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _) = x.dims3()?;
        if cos.dim(1)? * 2 != self.head_dim || sin.dims() != cos.dims() {
            candle_core::bail!(
                "fused attention takes (max_seq, {}) RoPE tables from precompute_rope_half_freqs, \
                 got cos {:?} and sin {:?}",
                self.head_dim / 2,
                cos.dims(),
                sin.dims()
            )
        }

        // Project Q, K, V to (batch, seq_len, heads, head_dim)
        let q = self
            .q_proj
            .forward(x)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k =
            self.k_proj
                .forward(x)?
                .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v =
            self.v_proj
                .forward(x)?
                .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;

        // QK-norm normalises each head, so it can run before the transpose,
        // while its input is contiguous for the fused kernel.
        let q = if let Some(ref qn) = self.q_norm {
            qn.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref kn) = self.k_norm {
            kn.forward(&k)?
        } else {
            k
        };

        let cos = cos.narrow(0, start_pos, seq_len)?;
        let sin = sin.narrow(0, start_pos, seq_len)?;
        let q = candle_nn::rotary_emb::rope_thd(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope_thd(&k.contiguous()?, &cos, &sin)?;
        Ok((q.transpose(1, 2)?, k.transpose(1, 2)?, v.transpose(1, 2)?))
    }

    /// Attention over heads-first Q, K and V, then the output projection.
    fn attention_output(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, _, seq_len, _) = q.dims4()?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let attn_output = if let Some(cpu_attn_output) = cpu_flash_attention(q, k, v, mask, scale)?
        {
            cpu_attn_output
        } else if seq_len == 1
            && mask.is_none()
            && matches!(q.device(), candle_core::Device::Metal(_))
        {
            candle_nn::ops::sdpa(&q.contiguous()?, k, v, None, false, scale, 1.0)?
        } else {
            self.grouped_attention(q, k, v, mask, scale)?
        };

        // Reshape back: (batch, num_heads, seq_len, head_dim) → (batch, seq_len, hidden)
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        // Output projection
        self.o_proj.forward(&attn_output)
    }

    /// Dense attention that shares each KV head with its group of Q heads.
    ///
    /// Q head `i` pairs with KV head `i / group` (repeat_interleave semantics).
    /// Viewing Q as (batch, kv_heads, group * seq_len, head_dim) lets each KV
    /// head serve its whole group in one matrix product, so K and V are never
    /// copied out to every Q head.
    fn grouped_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
    ) -> Result<Tensor> {
        let (batch, _, seq_len, _) = q.dims4()?;
        let kv_len = k.dim(2)?;
        let group = self.num_heads / self.num_kv_heads;

        // Scaling Q, not the scores, touches head_dim values per position
        // instead of kv_len, and writes Q out contiguously in the same kernel.
        let q = q.affine(scale as f64, 0.0)?.reshape((
            batch,
            self.num_kv_heads,
            group * seq_len,
            self.head_dim,
        ))?;
        let k = matmul_operand(k)?;
        let v = matmul_operand(v)?;

        let attn_weights = q.matmul(&k.t()?)?;
        let attn_weights = if let Some(mask) = mask {
            attn_weights
                .reshape((batch, self.num_heads, seq_len, kv_len))?
                .broadcast_add(mask)?
                .reshape((batch, self.num_kv_heads, group * seq_len, kv_len))?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        attn_weights
            .matmul(&v)?
            .reshape((batch, self.num_heads, seq_len, self.head_dim))
    }

    /// Clear the KV-cache (call between independent sequences).
    pub fn clear_cache(&mut self) {
        self.kv_cache.clear();
    }

    /// Snapshot the current KV-cache so another caller can restore it later.
    pub fn cache_state(&self) -> Option<(Tensor, Tensor)> {
        self.kv_cache.tensors().ok().flatten()
    }

    /// Restore a previously captured KV-cache.
    pub fn set_cache_state(&mut self, kv_cache: Option<(Tensor, Tensor)>) {
        self.kv_cache.restore_tensors(kv_cache);
    }

    /// Capture a branch-safe prefix without copying the KV tensors.
    pub fn snapshot_cache(&self) -> Option<KvCacheState> {
        self.kv_cache.snapshot()
    }

    /// Restore a branch-safe prefix captured by [`Self::snapshot_cache`].
    pub fn restore_cache(&mut self, state: Option<KvCacheState>) {
        self.kv_cache.restore(state);
    }
}

/// A heads-first (batch, heads, seq, head_dim) K or V that matrix products
/// can read.
///
/// CPU GEMM (including Accelerate) needs the whole tensor contiguous. The
/// CUDA and Metal products need only each head's rows packed, which the KV
/// cache's capacity-strided views already are; the heads-first transpose of
/// a fresh projection is not, and is copied.
fn matmul_operand(x: &Tensor) -> Result<Tensor> {
    let stride = x.stride();
    let rows_packed = stride[3] == 1 && stride[2] == x.dim(3)?;
    if x.device().is_cpu() || !rows_packed {
        x.contiguous()
    } else {
        Ok(x.clone())
    }
}

impl std::fmt::Debug for GroupedQueryAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupedQueryAttention")
            .field("num_heads", &self.num_heads)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("has_cache", &!self.kv_cache.is_empty())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_utils::{precompute_rope_freqs, precompute_rope_half_freqs};
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    #[test]
    fn incremental_and_dense_attention_match_full_causal_attention() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 32;
        let mut weights = HashMap::new();
        for (index, name) in [
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
        ]
        .iter()
        .enumerate()
        {
            let values: Vec<f32> = (0..hidden * hidden)
                .map(|i| ((i + index) as f32 * 0.17).sin() * 0.05)
                .collect();
            weights.insert(
                name.to_string(),
                Tensor::from_vec(values, (hidden, hidden), &device)?,
            );
        }
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let config = GqaConfig::new(hidden, 2, 2, 128, 10000., 1e-6);
        let mut full = GroupedQueryAttention::load(&config, vb.clone())?;
        let mut dense = GroupedQueryAttention::load(&config, vb.clone())?;
        let mut incremental = GroupedQueryAttention::load(&config, vb)?;
        let (cos, sin) =
            crate::tensor_utils::precompute_rope_freqs(16, 128, 10000., &device, DType::F32)?;
        let values: Vec<f32> = (0..2 * 6 * hidden)
            .map(|i| (i as f32 * 0.1).cos())
            .collect();
        let input = Tensor::from_vec(values, (2, 6, hidden), &device)?;
        let mask: Vec<f32> = (0..6)
            .flat_map(|q| (0..6).map(move |k| if k > q { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let rank_two_mask = Tensor::from_vec(mask, (6, 6), &device)?;
        let mask = rank_two_mask.broadcast_as((2, 1, 6, 6))?.contiguous()?;
        let expected = full.forward(&input, &cos, &sin, 0, Some(&mask))?;
        // Rank-two masks deliberately select the dense fallback. Its cache
        // must work with multiple batches and with Accelerate's stride rules.
        let dense_output = dense.forward(&input, &cos, &sin, 0, Some(&rank_two_mask))?;
        let dense_difference = (&dense_output - &expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            dense_difference < 1e-5,
            "dense cache attention error: {dense_difference}"
        );
        let mut chunks = Vec::new();
        for position in 0..6 {
            chunks.push(incremental.forward(
                &input.narrow(1, position, 1)?,
                &cos,
                &sin,
                position,
                None,
            )?);
        }
        let actual = Tensor::cat(&chunks, 1)?;
        let difference = (&actual - &expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(difference < 1e-5, "cached attention error: {difference}");
        assert!(incremental
            .forward(&input.narrow(1, 0, 1)?, &cos, &sin, 0, None)
            .is_err());
        Ok(())
    }

    const HIDDEN: usize = 32;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const SEQ: usize = 5;

    /// Qwen3-style grouped weights, with QK-norm.
    fn grouped_weights() -> Result<HashMap<String, Tensor>> {
        let device = Device::Cpu;
        let mut weights = HashMap::new();
        for (index, (name, rows)) in [
            ("q_proj.weight", HEADS * HEAD_DIM),
            ("k_proj.weight", KV_HEADS * HEAD_DIM),
            ("v_proj.weight", KV_HEADS * HEAD_DIM),
            ("o_proj.weight", HIDDEN),
        ]
        .iter()
        .enumerate()
        {
            let values: Vec<f32> = (0..rows * HIDDEN)
                .map(|i| ((i + 7 * index) as f32 * 0.13).sin() * 0.1)
                .collect();
            weights.insert(
                name.to_string(),
                Tensor::from_vec(values, (*rows, HIDDEN), &device)?,
            );
        }
        for name in ["q_norm.weight", "k_norm.weight"] {
            let values: Vec<f32> = (0..HEAD_DIM).map(|i| 0.5 + i as f32 * 0.1).collect();
            weights.insert(
                name.to_string(),
                Tensor::from_vec(values, HEAD_DIM, &device)?,
            );
        }
        Ok(weights)
    }

    fn grouped_attention_layer(
        weights: &HashMap<String, Tensor>,
        kernels: KernelMode,
    ) -> Result<GroupedQueryAttention> {
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
        let config = GqaConfig::with_head_dim(HIDDEN, HEADS, KV_HEADS, HEAD_DIM, 64, 1e4, 1e-6)
            .with_kernels(kernels);
        GroupedQueryAttention::load(&config, vb)
    }

    fn grouped_input() -> Result<Tensor> {
        let values: Vec<f32> = (0..2 * SEQ * HIDDEN)
            .map(|i| (i as f32 * 0.07).cos())
            .collect();
        Tensor::from_vec(values, (2, SEQ, HIDDEN), &Device::Cpu)
    }

    /// No mask, a causal (1, 1, seq, seq) mask, and a (batch, 1, seq, kv) mask
    /// that differs per batch.
    fn grouped_masks() -> Result<Vec<Option<Tensor>>> {
        let causal: Vec<f32> = (0..SEQ)
            .flat_map(|q| (0..SEQ).map(move |k| if k > q { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let causal = Tensor::from_vec(causal, (1, 1, SEQ, SEQ), &Device::Cpu)?;
        let per_batch: Vec<f32> = (0..2 * SEQ * SEQ)
            .map(|i| {
                if i % 7 == 3 {
                    f32::NEG_INFINITY
                } else {
                    (i as f32 * 0.3).sin()
                }
            })
            .collect();
        let per_batch = Tensor::from_vec(per_batch, (2, 1, SEQ, SEQ), &Device::Cpu)?;
        Ok(vec![None, Some(causal), Some(per_batch)])
    }

    /// The original attention, op for op: QK-norm after the transpose,
    /// full-width RoPE, every KV head copied out to its Q heads, and the
    /// scale applied to the scores.
    fn repeated_kv_reference(
        weights: &HashMap<String, Tensor>,
        input: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let device = Device::Cpu;
        let (cos, sin) = precompute_rope_freqs(HEAD_DIM, 64, 1e4, &device, DType::F32)?;
        let project = |name: &str, n: usize| -> Result<Tensor> {
            let w = &weights[&format!("{name}_proj.weight")];
            Linear::new(w.clone(), None)
                .forward(input)?
                .reshape((2, SEQ, n, HEAD_DIM))?
                .transpose(1, 2)
        };
        let norm = |x: Tensor, name: &str| -> Result<Tensor> {
            RmsNorm::new(weights[name].clone(), 1e-6).forward(&x)
        };
        let q = norm(project("q", HEADS)?, "q_norm.weight")?;
        let k = norm(project("k", KV_HEADS)?, "k_norm.weight")?;
        let v = project("v", KV_HEADS)?;
        let (cos, sin) = (
            cos.narrow(0, 0, SEQ)?.unsqueeze(0)?.unsqueeze(0)?,
            sin.narrow(0, 0, SEQ)?.unsqueeze(0)?.unsqueeze(0)?,
        );
        let (q, k) = apply_rotary_emb(&q, &k, &cos, &sin)?;
        let group = HEADS / KV_HEADS;
        let repeat = |x: &Tensor| -> Result<Tensor> {
            x.unsqueeze(2)?
                .broadcast_as((2, KV_HEADS, group, SEQ, HEAD_DIM))?
                .reshape((2, HEADS, SEQ, HEAD_DIM))?
                .contiguous()
        };
        let (k, v) = (repeat(&k)?, repeat(&v)?);
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale as f64, 0.0)?;
        let scores = match mask {
            Some(mask) => scores.broadcast_add(mask)?,
            None => scores,
        };
        candle_nn::ops::softmax_last_dim(&scores)?
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((2, SEQ, HEADS * HEAD_DIM))?
            .apply(&Linear::new(weights["o_proj.weight"].clone(), None))
    }

    fn max_difference(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?.abs()?.max_all()?.to_scalar::<f32>()
    }

    #[test]
    fn fused_grouped_attention_matches_repeated_kv_heads_with_masks() -> Result<()> {
        let weights = grouped_weights()?;
        let input = grouped_input()?;
        let (cos, sin) = precompute_rope_half_freqs(HEAD_DIM, 64, 1e4, &Device::Cpu, DType::F32)?;
        let uncached = grouped_attention_layer(&weights, KernelMode::Fused)?;
        for mask in grouped_masks()? {
            let expected = repeated_kv_reference(&weights, &input, mask.as_ref())?;
            let actual = uncached.forward_uncached(&input, &cos, &sin, mask.as_ref())?;
            let difference = max_difference(&actual, &expected)?;
            assert!(
                difference < 1e-5,
                "uncached grouped attention error {difference}"
            );
            // The cached path reads the KV cache's capacity-strided views.
            let mut cached = grouped_attention_layer(&weights, KernelMode::Fused)?;
            let actual = cached.forward(&input, &cos, &sin, 0, mask.as_ref())?;
            let difference = max_difference(&actual, &expected)?;
            assert!(
                difference < 1e-5,
                "cached grouped attention error {difference}"
            );
        }
        Ok(())
    }

    #[test]
    fn composite_attention_runs_the_original_ops() -> Result<()> {
        let weights = grouped_weights()?;
        let input = grouped_input()?;
        let (cos, sin) = precompute_rope_freqs(HEAD_DIM, 64, 1e4, &Device::Cpu, DType::F32)?;
        for mask in grouped_masks()? {
            let mut layer = grouped_attention_layer(&weights, KernelMode::Composite)?;
            let actual = layer.forward(&input, &cos, &sin, 0, mask.as_ref())?;
            let expected = repeated_kv_reference(&weights, &input, mask.as_ref())?;
            assert_eq!(
                actual.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?
            );
        }
        Ok(())
    }

    #[test]
    fn attention_rejects_rope_tables_for_the_other_kernels() -> Result<()> {
        let weights = grouped_weights()?;
        let input = grouped_input()?;
        let device = Device::Cpu;
        let full = precompute_rope_freqs(HEAD_DIM, 64, 1e4, &device, DType::F32)?;
        let half = precompute_rope_half_freqs(HEAD_DIM, 64, 1e4, &device, DType::F32)?;
        let mut fused = grouped_attention_layer(&weights, KernelMode::Fused)?;
        let error = fused
            .forward(&input, &full.0, &full.1, 0, None)
            .unwrap_err();
        assert!(
            error.to_string().contains("precompute_rope_half_freqs"),
            "{error}"
        );
        let mut composite = grouped_attention_layer(&weights, KernelMode::Composite)?;
        let error = composite
            .forward(&input, &half.0, &half.1, 0, None)
            .unwrap_err();
        assert!(
            error.to_string().contains("precompute_rope_freqs"),
            "{error}"
        );
        assert!(composite
            .forward_uncached(&input, &full.0, &full.1, None)
            .is_err());
        Ok(())
    }
}
