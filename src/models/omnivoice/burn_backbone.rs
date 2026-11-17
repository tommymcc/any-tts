//! OmniVoice's transformer layers, final norm and audio head on burn, for AMD
//! GPUs through ROCm (`rocm` feature).
//!
//! The text embeddings stay on candle's CPU device, which keeps the 620 MB
//! text embedding table out of VRAM. The text part of a pass doesn't change
//! between generation steps, so [`super::model`] computes it once per chunk
//! and hands it to [`BurnBackbone::prepare_pass`]. The audio embeddings (33 MB)
//! live here: they change every step, and summing them on the CPU took a tenth
//! of each step. Each layer matches [`crate::layers::TransformerBlock`]
//! with fused kernels: pre-norm, grouped-query attention with a per-head RMSNorm
//! on Q and K and split-half RoPE, and a SiLU MLP. Weights run in F32.

use burn::tensor::activation::{log_softmax, silu};
use burn::tensor::module::linear;
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
use candle_core::DType;
use candle_nn::VarBuilder;

use super::config::OmniVoiceLlmConfig;
use crate::burn_device::guard;
use crate::error::TtsError;
use crate::tensor_utils::precompute_rope_half_freqs;

/// The layers, the final norm and the audio head, on a burn device.
pub(super) struct BurnBackbone {
    device: Device,
    layers: Vec<Layer>,
    norm: Tensor<1>,
    /// The audio head, (hidden, codebooks * audio_vocab).
    head: Tensor<2>,
    /// (codebooks * audio_vocab, hidden): one table per codebook, stacked.
    audio_embeddings: Tensor<2>,
    /// RoPE tables, (max_positions, head_dim / 2).
    rope_cos: Tensor<2>,
    rope_sin: Tensor<2>,
    shape: Shape,
}

#[derive(Clone, Copy)]
struct Shape {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

/// One decoder layer. Projection weights are (in, out), as burn's linear takes them.
struct Layer {
    input_norm: Tensor<1>,
    q: Tensor<2>,
    k: Tensor<2>,
    v: Tensor<2>,
    o: Tensor<2>,
    q_norm: Tensor<1>,
    k_norm: Tensor<1>,
    post_norm: Tensor<1>,
    gate: Tensor<2>,
    up: Tensor<2>,
    down: Tensor<2>,
}

/// One pass's inputs that stay fixed through a chunk's generation steps; see
/// [`BurnBackbone::prepare_pass`].
pub(super) struct PassInputs {
    text: Option<Tensor<3>>,
    /// (1, seq, 1): 1 at audio positions.
    audio_mask: Tensor<3>,
    audio_positions: Vec<u32>,
    causal: bool,
}

impl BurnBackbone {
    /// Copies the layers, `llm.norm` and `audio_heads` from `vb`, which must be
    /// an F32 VarBuilder on the CPU, to `device`.
    pub(super) fn load(
        llm: &OmniVoiceLlmConfig,
        head_size: usize,
        vb: &VarBuilder,
        device: Device,
    ) -> Result<Self, TtsError> {
        let shape = Shape {
            heads: llm.num_attention_heads,
            kv_heads: llm.num_key_value_heads,
            head_dim: llm.head_dim(),
            eps: llm.rms_norm_eps,
        };
        let hidden = llm.hidden_size;
        let (q_size, kv_size) = (shape.heads * shape.head_dim, shape.kv_heads * shape.head_dim);
        let vector = |vb: &VarBuilder, name: &str, size: usize| -> Result<Tensor<1>, TtsError> {
            let values = vb.pp(name).get(size, "weight")?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            Ok(Tensor::from_data(TensorData::new(values, [size]), &device))
        };
        // Stored (out, in), as PyTorch does.
        let matrix = |vb: &VarBuilder, name: &str, out: usize, inp: usize| -> Result<Tensor<2>, TtsError> {
            let values = vb
                .pp(name)
                .get((out, inp), "weight")?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            Ok(Tensor::<2>::from_data(TensorData::new(values, [out, inp]), &device).transpose())
        };
        // The layers' projections, optionally in F16; see [`project`].
        let half = std::env::var_os("OMNIVOICE_BURN_F16").is_some();
        let projection = |vb: &VarBuilder, name: &str, out: usize, inp: usize| -> Result<Tensor<2>, TtsError> {
            let weight = matrix(vb, name, out, inp)?;
            Ok(if half { weight.transpose().cast(burn::tensor::DType::F16).transpose() } else { weight })
        };
        let load = || -> Result<Self, TtsError> {
            let mut layers = Vec::with_capacity(llm.num_hidden_layers);
            for index in 0..llm.num_hidden_layers {
                let vb = vb.pp(format!("llm.layers.{index}"));
                layers.push(Layer {
                    input_norm: vector(&vb, "input_layernorm", hidden)?,
                    q: projection(&vb, "self_attn.q_proj", q_size, hidden)?,
                    k: projection(&vb, "self_attn.k_proj", kv_size, hidden)?,
                    v: projection(&vb, "self_attn.v_proj", kv_size, hidden)?,
                    o: projection(&vb, "self_attn.o_proj", hidden, q_size)?,
                    q_norm: vector(&vb, "self_attn.q_norm", shape.head_dim)?,
                    k_norm: vector(&vb, "self_attn.k_norm", shape.head_dim)?,
                    post_norm: vector(&vb, "post_attention_layernorm", hidden)?,
                    gate: projection(&vb, "mlp.gate_proj", llm.intermediate_size, hidden)?,
                    up: projection(&vb, "mlp.up_proj", llm.intermediate_size, hidden)?,
                    down: projection(&vb, "mlp.down_proj", hidden, llm.intermediate_size)?,
                });
            }
            let norm = vector(vb, "llm.norm", hidden)?;
            let head = matrix(vb, "audio_heads", head_size, hidden)?;
            // Looked up by row, so not transposed.
            let audio_embeddings = matrix(vb, "audio_embeddings", head_size, hidden)?.transpose();
            let (cos, sin) = precompute_rope_half_freqs(
                shape.head_dim,
                llm.max_position_embeddings,
                llm.rope_theta(),
                &candle_core::Device::Cpu,
                DType::F32,
            )?;
            let table = |t: candle_core::Tensor| -> Result<Tensor<2>, TtsError> {
                let (rows, cols) = t.dims2()?;
                let values = t.flatten_all()?.to_vec1::<f32>()?;
                Ok(Tensor::from_data(TensorData::new(values, [rows, cols]), &device))
            };
            Ok(Self {
                device: device.clone(),
                layers,
                norm,
                head,
                audio_embeddings,
                rope_cos: table(cos)?,
                rope_sin: table(sin)?,
                shape,
            })
        };
        guard("loading OmniVoice's backbone onto the GPU", load).map_err(TtsError::ModelError)?
    }

    /// Audio logits, (batch, logits_len, codebooks * audio_vocab) on the CPU,
    /// for the last `logits_len` positions of `hidden`, (batch, seq, hidden)
    /// F32 on the CPU. Attention sees every position unless `causal`.
    pub(super) fn forward(
        &self,
        hidden: &candle_core::Tensor,
        logits_len: usize,
        causal: bool,
    ) -> Result<candle_core::Tensor, TtsError> {
        let started = std::time::Instant::now();
        let (batch, seq, _) = hidden.dims3()?;
        let input = self.upload(hidden)?;
        let run = || -> Result<(Vec<f32>, usize), String> {
            let logits = self.logits(input, logits_len, causal);
            let [_, _, size] = logits.dims();
            let values = logits.into_data().try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            Ok((values, size))
        };
        let (values, size) = guard("running OmniVoice's backbone on the GPU", run)
            .and_then(|result| result)
            .map_err(TtsError::RuntimeError)?;
        if std::env::var_os("OMNIVOICE_TIMINGS").is_some() {
            eprintln!(
                "omnivoice timing: burn backbone batch {batch} seq {seq} logits {logits_len}: total {:.3}s",
                started.elapsed().as_secs_f64()
            );
        }
        Ok(candle_core::Tensor::from_vec(
            values,
            (batch, logits_len, size),
            &candle_core::Device::Cpu,
        )?)
    }

    /// The parts of one pass's input that stay fixed through a chunk's
    /// generation steps, copied to the GPU once: `text`, (1, seq, hidden) on
    /// the CPU, is the text embeddings already zeroed at audio positions (or
    /// `None` where every position is audio), and `audio_mask` marks the audio
    /// positions. Attention sees every position unless `causal`.
    pub(super) fn prepare_pass(
        &self,
        text: Option<&candle_core::Tensor>,
        audio_mask: &[u32],
        causal: bool,
    ) -> Result<PassInputs, TtsError> {
        let seq = audio_mask.len();
        let text = text.map(|t| self.upload(t)).transpose()?;
        let mask: Vec<f32> = audio_mask.iter().map(|&m| m as f32).collect();
        let audio_mask_gpu = guard("copying the audio mask to the GPU", || {
            Tensor::<3>::from_data(TensorData::new(mask, [1, seq, 1]), &self.device)
        })
        .map_err(TtsError::RuntimeError)?;
        Ok(PassInputs { text, audio_mask: audio_mask_gpu, audio_positions: audio_mask.to_vec(), causal })
    }

    /// The hidden states, (1, seq, hidden), for one pass at the current step:
    /// the text part plus, at audio positions, the sum over codebooks of each
    /// codebook's embedding of `ids`, (codebooks, seq) codebook-major.
    fn embed(&self, pass: &PassInputs, ids: &[u32], codebooks: usize, vocab: usize) -> Tensor<3> {
        let seq = pass.audio_positions.len();
        // Text positions look up row 0 of each codebook's table, which the
        // mask then zeroes, as candle's embedding does.
        let rows: Vec<i32> = ids
            .chunks_exact(seq)
            .enumerate()
            .flat_map(|(codebook, ids)| {
                ids.iter()
                    .zip(&pass.audio_positions)
                    .map(move |(&id, &audio)| (id * audio) as i32 + (codebook * vocab) as i32)
            })
            .collect();
        let rows = Tensor::<1, Int>::from_data(TensorData::new(rows, [codebooks * seq]), &self.device);
        let [_, hidden] = self.audio_embeddings.dims();
        let audio = self
            .audio_embeddings
            .clone()
            .select(0, rows)
            .reshape([codebooks, seq, hidden])
            .sum_dim(0)
            * pass.audio_mask.clone();
        match &pass.text {
            Some(text) => text.clone() + audio,
            None => audio,
        }
    }

    /// One generation step's greedy picks, without bringing the logits back to
    /// the CPU: runs the conditional pass (and the unconditional one, for
    /// classifier-free guidance at `scale`), takes log-softmax over each
    /// codebook's vocabulary, combines them as
    /// log_softmax(c + scale (c - u)), adds `penalty` (one value per vocabulary
    /// entry), and returns each (codebook, position)'s best token and its
    /// log-probability, codebook-major, for the last `logits_len` positions.
    /// Each pass is its fixed inputs and the step's audio ids, (codebooks,
    /// seq) codebook-major.
    pub(super) fn predict(
        &self,
        cond: (&PassInputs, &[u32]),
        uncond: Option<(&PassInputs, &[u32])>,
        logits_len: usize,
        codebooks: usize,
        scale: f64,
        penalty: &[f32],
    ) -> Result<(Vec<u32>, Vec<f32>), TtsError> {
        let started = std::time::Instant::now();
        let vocab = penalty.len();
        let penalty = penalty.to_vec();
        let run = || -> Result<(Vec<u32>, Vec<f32>), String> {
            let log_probs = |logits: Tensor<3>| {
                log_softmax(logits.reshape([logits_len, codebooks, vocab]).swap_dims(0, 1), 2)
            };
            let pass = |(inputs, ids): (&PassInputs, &[u32])| {
                let hidden = self.embed(inputs, ids, codebooks, vocab);
                log_probs(self.logits(hidden, logits_len, inputs.causal))
            };
            let c = pass(cond);
            let guided = match uncond {
                Some(uncond) => {
                    let u = pass(uncond);
                    log_softmax(c.clone() + (c - u).mul_scalar(scale), 2)
                }
                None => c,
            };
            let penalty = Tensor::<1>::from_data(TensorData::new(penalty, [vocab]), &self.device);
            let guided = guided + penalty.reshape([1, 1, vocab]);
            let tokens = guided.clone().argmax(2).into_data();
            let scores = guided.max_dim(2).into_data();
            let tokens = tokens.iter::<i64>().map(|t| t as u32).collect();
            let scores = scores.try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            Ok((tokens, scores))
        };
        let picks = guard("running OmniVoice's backbone on the GPU", run)
            .and_then(|result| result)
            .map_err(TtsError::RuntimeError)?;
        if std::env::var_os("OMNIVOICE_TIMINGS").is_some() {
            eprintln!(
                "omnivoice timing: burn step seq {}{} logits {logits_len}: total {:.3}s",
                cond.0.audio_positions.len(),
                uncond.map(|(p, _)| format!(" + {}", p.audio_positions.len())).unwrap_or_default(),
                started.elapsed().as_secs_f64()
            );
        }
        Ok(picks)
    }

    /// Copies hidden states, (batch, seq, hidden) on the CPU, to the GPU.
    fn upload(&self, hidden: &candle_core::Tensor) -> Result<Tensor<3>, TtsError> {
        let (batch, seq, width) = hidden.dims3()?;
        let values = hidden.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        guard("copying hidden states to the GPU", || {
            Tensor::<3>::from_data(TensorData::new(values, [batch, seq, width]), &self.device)
        })
        .map_err(TtsError::RuntimeError)
    }

    /// The layers, the final norm and the audio head: logits (batch,
    /// logits_len, codebooks * audio_vocab) for the last `logits_len`
    /// positions, left on the GPU.
    fn logits(&self, x: Tensor<3>, logits_len: usize, causal: bool) -> Tensor<3> {
        let [_, seq, _] = x.dims();
        let cos = self.rope_cos.clone().narrow(0, 0, seq);
        let sin = self.rope_sin.clone().narrow(0, 0, seq);
        let x = self
            .layers
            .iter()
            .fold(x, |x, layer| layer.forward(x, &cos, &sin, self.shape, causal));
        let x = x.narrow(1, seq - logits_len, logits_len);
        let x = rms_norm(x, &self.norm, self.shape.eps);
        linear(x, self.head.clone(), None)
    }
}

impl Layer {
    fn forward(&self, x: Tensor<3>, cos: &Tensor<2>, sin: &Tensor<2>, shape: Shape, causal: bool) -> Tensor<3> {
        let [batch, seq, _] = x.dims();
        let Shape { heads, kv_heads, head_dim, eps } = shape;
        let h = rms_norm(x.clone(), &self.input_norm, eps);

        let q = project(h.clone(), &self.q).reshape([batch, seq, heads, head_dim]);
        let k = project(h.clone(), &self.k).reshape([batch, seq, kv_heads, head_dim]);
        let v = project(h, &self.v).reshape([batch, seq, kv_heads, head_dim]);
        let q = rope(rms_norm(q, &self.q_norm, eps), cos, sin).swap_dims(1, 2);
        let k = rope(rms_norm(k, &self.k_norm, eps), cos, sin).swap_dims(1, 2);
        let v = v.swap_dims(1, 2);
        let a = attention(q, k, v, heads / kv_heads, causal)
            .swap_dims(1, 2)
            .reshape([batch, seq, heads * head_dim]);
        let x = x + project(a, &self.o);

        let h = rms_norm(x.clone(), &self.post_norm, eps);
        let gate = silu(project(h.clone(), &self.gate));
        let up = project(h, &self.up);
        x + project(gate * up, &self.down)
    }
}

/// Grouped-query attention, written out as softmax(Q Kᵀ / √d) V instead of
/// calling burn's `attention`.
///
/// q is (batch, heads, seq, head_dim); k and v are (batch, kv_heads, seq,
/// head_dim), and query head i uses KV head i / group, as repeat_interleave
/// does. The query heads that share a KV head are stacked along the rows, so K
/// and V are never copied per head.
///
/// burn's `attention` is avoided on purpose: when burn's `autotune` feature is
/// on (parakeet-rs turns it on), it only lets its materialising fallback
/// compete for sequences no longer than the head dimension, and on GPUs without
/// matrix cores (RDNA2) the flash kernel it picks instead is about 10 times
/// slower at OmniVoice's lengths (7.9 ms against 0.8 ms per layer at 512
/// tokens). This is what that fallback computes. See audio-berry's
/// docs/amd-gpu.md.
fn attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, group: usize, causal: bool) -> Tensor<4> {
    let [batch, heads, seq, head_dim] = q.dims();
    let kv_heads = heads / group;
    let device = q.device();
    // Scaling Q rather than the scores saves a pass over the (seq x seq) scores.
    let q = q
        .mul_scalar(1.0 / (head_dim as f64).sqrt())
        .reshape([batch, kv_heads, group * seq, head_dim]);
    // The keys are padded to a multiple of 8 so the score rows vectorise (at
    // 981 keys every softmax kernel ran element by element). The padding
    // repeats the last key, so it can't raise a row's max, and its rows of V
    // (below) are zero, so it adds nothing.
    let keys = seq.div_ceil(8) * 8;
    let pad = keys - seq;
    let k = if pad == 0 {
        k
    } else {
        let last = k.clone().narrow(2, seq - 1, 1).expand([batch, kv_heads, pad, head_dim]);
        Tensor::cat(vec![k, last], 2)
    };
    // V with a column of ones (and three zeros, to keep rows a multiple of
    // 4): P V then also gives each row's sum of P, the softmax denominator,
    // instead of a separate reduction and a divide over the scores.
    let v = Tensor::cat(
        vec![
            v,
            Tensor::ones([batch, kv_heads, seq, 1], &device),
            Tensor::zeros([batch, kv_heads, seq, 3], &device),
        ],
        3,
    );
    let v = if pad == 0 {
        v
    } else {
        Tensor::cat(vec![v, Tensor::zeros([batch, kv_heads, pad, head_dim + 4], &device)], 2)
    };
    let scores = q.matmul(k.swap_dims(2, 3));
    let scores = if causal {
        // True where key position > query position (burn's tril_mask marks
        // what lies outside the lower triangle). Row r of the stacked rows is
        // query position r % seq.
        let future = Tensor::<2, Bool>::tril_mask([seq, keys], 0, &device)
            .unsqueeze_dim::<3>(0)
            .expand([group, seq, keys])
            .reshape([1, 1, group * seq, keys]);
        scores.mask_fill(future, f32::NEG_INFINITY)
    } else {
        scores
    };
    let max = scores.clone().max_dim(3);
    let weighted = (scores - max).exp().matmul(v);
    let out = weighted.clone().narrow(3, 0, head_dim) / weighted.narrow(3, head_dim, 1);
    out.reshape([batch, heads, seq, head_dim])
}

/// x W, with W (in, out). With F16 weights (`OMNIVOICE_BURN_F16`), x is cast
/// to F16 for the matmul and the result back to F32; norms, attention and the
/// residual stream stay F32.
fn project(x: Tensor<3>, weight: &Tensor<2>) -> Tensor<3> {
    if weight.dtype() == burn::tensor::DType::F16 {
        linear(x.cast(burn::tensor::DType::F16), weight.clone(), None).cast(burn::tensor::DType::F32)
    } else {
        linear(x, weight.clone(), None)
    }
}

/// RMSNorm over the last dimension, with a weight of that size.
fn rms_norm<const D: usize>(x: Tensor<D>, weight: &Tensor<1>, eps: f64) -> Tensor<D> {
    let mut shape = [1usize; D];
    shape[D - 1] = weight.dims()[0];
    let rms = x.clone().powi_scalar(2).mean_dim(D - 1).add_scalar(eps).sqrt().recip();
    x * rms * weight.clone().reshape(shape)
}

/// Split-half RoPE on (batch, seq, heads, head_dim), with (seq, head_dim / 2) tables.
fn rope(x: Tensor<4>, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<4> {
    let [batch, seq, heads, dim] = x.dims();
    let half = dim / 2;
    let cos = cos.clone().reshape([1, seq, 1, half]);
    let sin = sin.clone().reshape([1, seq, 1, half]);
    let x1 = x.clone().slice([0..batch, 0..seq, 0..heads, 0..half]);
    let x2 = x.slice([0..batch, 0..seq, 0..heads, half..dim]);
    let r1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let r2 = x2 * cos + x1 * sin;
    Tensor::cat(vec![r1, r2], 3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::ops::AttentionModuleOptions;

    /// The written-out attention against burn's own, with grouped KV heads,
    /// with and without the causal mask. Needs a ROCm GPU; skipped without one.
    #[test]
    fn attention_matches_burns() {
        let Ok(device) = crate::burn_device::rocm_device(0) else {
            eprintln!("no ROCm device; skipping");
            return;
        };
        for seq in [37, 40] {
            let (batch, heads, kv_heads, dim) = (2, 4, 2, 16);
            let values = |n: usize, seed: usize| -> Vec<f32> {
                (0..n).map(|i| (((i + seed) * 2654435761) % 1000) as f32 / 250.0 - 2.0).collect()
            };
            let tensor = |shape: [usize; 4], seed: usize| {
                let n = shape.iter().product();
                Tensor::<4>::from_data(TensorData::new(values(n, seed), shape), &device)
            };
            let q = tensor([batch, heads, seq, dim], 1);
            let k = tensor([batch, kv_heads, seq, dim], 2);
            let v = tensor([batch, kv_heads, seq, dim], 3);
            let group = heads / kv_heads;
            let repeat = |t: Tensor<4>| {
                t.unsqueeze_dim::<5>(2).expand([batch, kv_heads, group, seq, dim]).reshape([batch, heads, seq, dim])
            };
            for causal in [false, true] {
                let options = AttentionModuleOptions { is_causal: causal, ..Default::default() };
                let want = burn::tensor::module::attention(q.clone(), repeat(k.clone()), repeat(v.clone()), None, None, None, options);
                let got = attention(q.clone(), k.clone(), v.clone(), group, causal);
                let want: Vec<f32> = want.into_data().try_to_vec::<f32>().unwrap();
                let got: Vec<f32> = got.into_data().try_to_vec::<f32>().unwrap();
                let diff = want.iter().zip(&got).fold(0.0f32, |d, (a, b)| d.max((a - b).abs()));
                assert!(diff < 1e-5, "seq {seq} causal {causal}: max |diff| {diff}");
            }
        }
    }
}
