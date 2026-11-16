//! OmniVoice's transformer layers, final norm and audio head on burn, for AMD
//! GPUs through ROCm (`rocm` feature).
//!
//! The embeddings stay on candle's CPU device, which keeps the 620 MB text
//! embedding table out of VRAM: [`super::model`] sums them as before and hands
//! the hidden states here. Each layer matches [`crate::layers::TransformerBlock`]
//! with fused kernels: pre-norm, grouped-query attention with a per-head RMSNorm
//! on Q and K and split-half RoPE, and a SiLU MLP. Weights run in F32.

use burn::tensor::activation::silu;
use burn::tensor::module::{attention, linear};
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Device, Tensor, TensorData};
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
        let load = || -> Result<Self, TtsError> {
            let mut layers = Vec::with_capacity(llm.num_hidden_layers);
            for index in 0..llm.num_hidden_layers {
                let vb = vb.pp(format!("llm.layers.{index}"));
                layers.push(Layer {
                    input_norm: vector(&vb, "input_layernorm", hidden)?,
                    q: matrix(&vb, "self_attn.q_proj", q_size, hidden)?,
                    k: matrix(&vb, "self_attn.k_proj", kv_size, hidden)?,
                    v: matrix(&vb, "self_attn.v_proj", kv_size, hidden)?,
                    o: matrix(&vb, "self_attn.o_proj", hidden, q_size)?,
                    q_norm: vector(&vb, "self_attn.q_norm", shape.head_dim)?,
                    k_norm: vector(&vb, "self_attn.k_norm", shape.head_dim)?,
                    post_norm: vector(&vb, "post_attention_layernorm", hidden)?,
                    gate: matrix(&vb, "mlp.gate_proj", llm.intermediate_size, hidden)?,
                    up: matrix(&vb, "mlp.up_proj", llm.intermediate_size, hidden)?,
                    down: matrix(&vb, "mlp.down_proj", hidden, llm.intermediate_size)?,
                });
            }
            let norm = vector(vb, "llm.norm", hidden)?;
            let head = matrix(vb, "audio_heads", head_size, hidden)?;
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
        let (batch, seq, width) = hidden.dims3()?;
        let values = hidden.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let prepared = started.elapsed();
        let run = || -> Result<(Vec<f32>, usize), String> {
            let x = Tensor::<3>::from_data(TensorData::new(values, [batch, seq, width]), &self.device);
            let cos = self.rope_cos.clone().narrow(0, 0, seq);
            let sin = self.rope_sin.clone().narrow(0, 0, seq);
            let x = self
                .layers
                .iter()
                .fold(x, |x, layer| layer.forward(x, &cos, &sin, self.shape, causal));
            let x = x.narrow(1, seq - logits_len, logits_len);
            let x = rms_norm(x, &self.norm, self.shape.eps);
            let logits = linear(x, self.head.clone(), None);
            let [_, _, size] = logits.dims();
            let values = logits.into_data().try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            Ok((values, size))
        };
        let (values, size) = guard("running OmniVoice's backbone on the GPU", run)
            .and_then(|result| result)
            .map_err(TtsError::RuntimeError)?;
        if std::env::var_os("OMNIVOICE_TIMINGS").is_some() {
            eprintln!(
                "omnivoice timing: burn backbone batch {batch} seq {seq} logits {logits_len}: host {:.3}s, total {:.3}s",
                prepared.as_secs_f64(),
                started.elapsed().as_secs_f64()
            );
        }
        Ok(candle_core::Tensor::from_vec(
            values,
            (batch, logits_len, size),
            &candle_core::Device::Cpu,
        )?)
    }
}

impl Layer {
    fn forward(&self, x: Tensor<3>, cos: &Tensor<2>, sin: &Tensor<2>, shape: Shape, causal: bool) -> Tensor<3> {
        let [batch, seq, _] = x.dims();
        let Shape { heads, kv_heads, head_dim, eps } = shape;
        let h = rms_norm(x.clone(), &self.input_norm, eps);

        let q = linear(h.clone(), self.q.clone(), None).reshape([batch, seq, heads, head_dim]);
        let k = linear(h.clone(), self.k.clone(), None).reshape([batch, seq, kv_heads, head_dim]);
        let v = linear(h, self.v.clone(), None).reshape([batch, seq, kv_heads, head_dim]);
        let q = rope(rms_norm(q, &self.q_norm, eps), cos, sin).swap_dims(1, 2);
        let k = rope(rms_norm(k, &self.k_norm, eps), cos, sin).swap_dims(1, 2);
        let v = v.swap_dims(1, 2);
        // Q head i uses KV head i / group, as repeat_interleave does.
        let group = heads / kv_heads;
        let repeat = |t: Tensor<4>| {
            t.unsqueeze_dim::<5>(2)
                .expand([batch, kv_heads, group, seq, head_dim])
                .reshape([batch, heads, seq, head_dim])
        };
        let options = AttentionModuleOptions { scale: None, softcap: None, is_causal: causal };
        let a = attention(q, repeat(k), repeat(v), None, None, options)
            .swap_dims(1, 2)
            .reshape([batch, seq, heads * head_dim]);
        let x = x + linear(a, self.o.clone(), None);

        let h = rms_norm(x.clone(), &self.post_norm, eps);
        let gate = silu(linear(h.clone(), self.gate.clone(), None));
        let up = linear(h, self.up.clone(), None);
        x + linear(gate * up, self.down.clone(), None)
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
