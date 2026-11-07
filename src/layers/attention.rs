//! Grouped-Query Attention (GQA) layer.
//!
//! GQA is the attention variant used by Qwen 2 / Qwen 2.5 / Qwen 3
//! family models. It reduces KV-cache memory by sharing key/value heads
//! across multiple query heads.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::kv_cache::{KvCache, KvCacheState};
use crate::tensor_utils::{apply_rotary_emb, cpu_flash_attention, RmsNorm};

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
        }
    }

    pub fn with_attention_bias(mut self, attention_bias: bool) -> Self {
        self.attention_bias = attention_bias;
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
        let q_norm = RmsNorm::load(config.head_dim, config.rms_norm_eps, vb.pp("q_norm")).ok();
        let k_norm = RmsNorm::load(config.head_dim, config.rms_norm_eps, vb.pp("k_norm")).ok();

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
            kv_cache: KvCache::new(),
        })
    }

    /// Forward pass with rotary embeddings and optional KV-cache.
    ///
    /// * `x`     — (batch, seq_len, hidden_size)
    /// * `cos`   — (max_seq, head_dim) precomputed RoPE cos, or the
    ///   (max_seq, head_dim / 2) table of [`crate::tensor_utils::precompute_rope_half_freqs`],
    ///   which selects the fused RoPE kernel
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
        let (q, k, v) = self.project(x, cos, sin, start_pos)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;
        self.attend(&q, &k, &v, mask)
    }

    /// Forward pass over a whole sequence that neither reads nor fills the
    /// KV-cache, for models that run every position at once.
    pub fn forward_uncached(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (q, k, v) = self.project(x, cos, sin, 0)?;
        // Heads-first views of the projections, which matrix products can't read.
        let (k, v) = if x.device().is_cpu() {
            (k, v)
        } else {
            (k.contiguous()?, v.contiguous()?)
        };
        self.attend(&q, &k, &v, mask)
    }

    /// Q, K and V after QK-norm and RoPE, as (batch, heads, seq_len, head_dim).
    fn project(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        start_pos: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _) = x.dims3()?;

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

        // Apply QK-norm if present (Qwen3-style). It normalises each head, so
        // it can run before the transpose, while the input is contiguous.
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
        let (q, k) = if cos.dim(1)? * 2 == self.head_dim {
            let cos = cos.narrow(0, start_pos, seq_len)?;
            let sin = sin.narrow(0, start_pos, seq_len)?;
            let q = candle_nn::rotary_emb::rope_thd(&q.contiguous()?, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope_thd(&k.contiguous()?, &cos, &sin)?;
            (q.transpose(1, 2)?, k.transpose(1, 2)?)
        } else {
            let cos_slice = cos
                .narrow(0, start_pos, seq_len)?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            let sin_slice = sin
                .narrow(0, start_pos, seq_len)?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            apply_rotary_emb(
                &q.transpose(1, 2)?,
                &k.transpose(1, 2)?,
                &cos_slice,
                &sin_slice,
            )?
        };
        Ok((q, k, v.transpose(1, 2)?))
    }

    /// Attention over heads-first Q, K and V, then the output projection.
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
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

        // Dense CPU GEMM (including Accelerate) has stricter matrix
        // and batch stride requirements than fused attention.
        let (k, v) = if q.device().is_cpu() {
            (k.contiguous()?, v.contiguous()?)
        } else {
            (k.clone(), v.clone())
        };

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

    #[test]
    fn grouped_uncached_attention_matches_repeated_kv_heads() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, heads, kv_heads, head_dim, seq) = (32, 4, 2, 8, 5);
        let mut weights = HashMap::new();
        for (index, (name, rows)) in [
            ("q_proj.weight", heads * head_dim),
            ("k_proj.weight", kv_heads * head_dim),
            ("v_proj.weight", kv_heads * head_dim),
            ("o_proj.weight", hidden),
        ]
        .iter()
        .enumerate()
        {
            let values: Vec<f32> = (0..rows * hidden)
                .map(|i| ((i + 7 * index) as f32 * 0.13).sin() * 0.1)
                .collect();
            weights.insert(
                name.to_string(),
                Tensor::from_vec(values, (*rows, hidden), &device)?,
            );
        }
        for name in ["q_norm.weight", "k_norm.weight"] {
            let values: Vec<f32> = (0..head_dim).map(|i| 0.5 + i as f32 * 0.1).collect();
            weights.insert(
                name.to_string(),
                Tensor::from_vec(values, head_dim, &device)?,
            );
        }
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &device);
        let config = GqaConfig::with_head_dim(hidden, heads, kv_heads, head_dim, 64, 1e4, 1e-6);
        let attention = GroupedQueryAttention::load(&config, vb)?;
        let values: Vec<f32> = (0..2 * seq * hidden)
            .map(|i| (i as f32 * 0.07).cos())
            .collect();
        let input = Tensor::from_vec(values, (2, seq, hidden), &device)?;
        let (cos, sin) =
            crate::tensor_utils::precompute_rope_freqs(head_dim, 64, 1e4, &device, DType::F32)?;
        let (half_cos, half_sin) = crate::tensor_utils::precompute_rope_half_freqs(
            head_dim,
            64,
            1e4,
            &device,
            DType::F32,
        )?;
        let actual = attention.forward_uncached(&input, &half_cos, &half_sin, None)?;

        // Reference: Qwen3 attention with every KV head copied out to its Q heads.
        let project = |name: &str, n: usize| -> Result<Tensor> {
            let w = &weights[&format!("{name}_proj.weight")];
            input
                .broadcast_matmul(&w.t()?)?
                .reshape((2, seq, n, head_dim))?
                .transpose(1, 2)
        };
        let norm = |x: Tensor, name: &str| -> Result<Tensor> {
            let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
            x.broadcast_div(&(variance + 1e-6)?.sqrt()?)?
                .broadcast_mul(&weights[name])
        };
        let q = norm(project("q", heads)?, "q_norm.weight")?;
        let k = norm(project("k", kv_heads)?, "k_norm.weight")?;
        let v = project("v", kv_heads)?;
        let (cos, sin) = (
            cos.narrow(0, 0, seq)?.unsqueeze(0)?.unsqueeze(0)?,
            sin.narrow(0, 0, seq)?.unsqueeze(0)?.unsqueeze(0)?,
        );
        let (q, k) = apply_rotary_emb(&q.contiguous()?, &k.contiguous()?, &cos, &sin)?;
        let repeat = |x: &Tensor| -> Result<Tensor> {
            let parts: Vec<Tensor> = (0..heads)
                .map(|head| x.narrow(1, head / (heads / kv_heads), 1))
                .collect::<Result<_>>()?;
            Tensor::cat(&parts, 1)?.contiguous()
        };
        let (k, v) = (repeat(&k)?, repeat(&v.contiguous()?)?);
        let scores = (q.matmul(&k.t()?)? / (head_dim as f64).sqrt())?;
        let expected = candle_nn::ops::softmax_last_dim(&scores)?
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((2, seq, heads * head_dim))?
            .broadcast_matmul(&weights["o_proj.weight"].t()?)?;
        let difference = (&actual - &expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(difference < 1e-5, "grouped attention error: {difference}");
        Ok(())
    }
}
