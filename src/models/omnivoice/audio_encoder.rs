//! Turns reference audio into codec tokens: a DAC acoustic encoder and a HuBERT
//! semantic encoder, joined and quantized. It's only needed while preparing a
//! voice, so the model loads it on demand.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, VarBuilder};

use crate::layers::conv::Conv1d;

use super::audio_processing::resample;
use super::audio_tokenizer::{snake1d, DacResidualUnit};
use super::config::{OmniVoiceAudioTokenizerConfig, OmniVoiceSemanticConfig};

/// Padding added before HuBERT, in 16 kHz samples.
const SEMANTIC_PAD: usize = 160;
const GROUP_NORM_EPS: f64 = 1e-5;

struct HubertFeatureLayer {
    conv: Conv1d,
    norm: Option<(Tensor, Tensor)>,
}

impl HubertFeatureLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv.forward(x)?;
        if let Some((weight, bias)) = &self.norm {
            // One group per channel, so each channel normalizes over time.
            let mean = h.mean_keepdim(2)?;
            let centred = h.broadcast_sub(&mean)?;
            let variance = centred.sqr()?.mean_keepdim(2)?;
            h = centred
                .broadcast_div(&(variance + GROUP_NORM_EPS)?.sqrt()?)?
                .broadcast_mul(weight)?
                .broadcast_add(bias)?;
        }
        h.gelu_erf()
    }
}

struct HubertAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl HubertAttention {
    fn load(config: &OmniVoiceSemanticConfig, vb: VarBuilder) -> Result<Self> {
        let size = config.hidden_size;
        Ok(Self {
            q_proj: candle_nn::linear(size, size, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear(size, size, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear(size, size, vb.pp("v_proj"))?,
            out_proj: candle_nn::linear(size, size, vb.pp("out_proj"))?,
            num_heads: config.num_attention_heads,
            head_dim: size / config.num_attention_heads,
        })
    }

    fn split(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, length, _) = x.dims3()?;
        x.reshape((batch, length, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, length, size) = x.dims3()?;
        let q = self.split(&self.q_proj.forward(x)?)?;
        let k = self.split(&self.k_proj.forward(x)?)?;
        let v = self.split(&self.v_proj.forward(x)?)?;
        let scores =
            (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * (self.head_dim as f64).powf(-0.5))?;
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, length, size))?;
        self.out_proj.forward(&out)
    }
}

struct HubertLayer {
    attention: HubertAttention,
    layer_norm: LayerNorm,
    intermediate: Linear,
    output: Linear,
    final_layer_norm: LayerNorm,
}

impl HubertLayer {
    fn load(config: &OmniVoiceSemanticConfig, vb: VarBuilder) -> Result<Self> {
        let size = config.hidden_size;
        let feed_forward = vb.pp("feed_forward");
        Ok(Self {
            attention: HubertAttention::load(config, vb.pp("attention"))?,
            layer_norm: candle_nn::layer_norm(size, config.layer_norm_eps, vb.pp("layer_norm"))?,
            intermediate: candle_nn::linear(
                size,
                config.intermediate_size,
                feed_forward.pp("intermediate_dense"),
            )?,
            output: candle_nn::linear(
                config.intermediate_size,
                size,
                feed_forward.pp("output_dense"),
            )?,
            final_layer_norm: candle_nn::layer_norm(
                size,
                config.layer_norm_eps,
                vb.pp("final_layer_norm"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self
            .layer_norm
            .forward(&x.add(&self.attention.forward(x)?)?)?;
        let fed = self
            .output
            .forward(&self.intermediate.forward(&h)?.gelu_erf()?)?;
        self.final_layer_norm.forward(&h.add(&fed)?)
    }
}

struct Hubert {
    feature_layers: Vec<HubertFeatureLayer>,
    projection_norm: LayerNorm,
    projection: Linear,
    position_weight: Tensor,
    position_bias: Tensor,
    position_padding: usize,
    position_groups: usize,
    layer_norm: LayerNorm,
    layers: Vec<HubertLayer>,
}

impl Hubert {
    fn load(config: &OmniVoiceSemanticConfig, vb: VarBuilder) -> Result<Self> {
        let mut feature_layers = Vec::with_capacity(config.conv_dim.len());
        let extractor = vb.pp("feature_extractor.conv_layers");
        for (index, &out_channels) in config.conv_dim.iter().enumerate() {
            let in_channels = if index == 0 {
                1
            } else {
                config.conv_dim[index - 1]
            };
            let layer = extractor.pp(index.to_string());
            let conv = Conv1d::load(
                in_channels,
                out_channels,
                config.conv_kernel[index],
                config.conv_stride[index],
                0,
                1,
                1,
                false,
                layer.pp("conv"),
            )?;
            let norm = if index == 0 {
                let norm = layer.pp("layer_norm");
                Some((
                    norm.get(out_channels, "weight")?
                        .reshape((1, out_channels, 1))?,
                    norm.get(out_channels, "bias")?
                        .reshape((1, out_channels, 1))?,
                ))
            } else {
                None
            };
            feature_layers.push(HubertFeatureLayer { conv, norm });
        }

        let last = *config.conv_dim.last().ok_or_else(|| {
            candle_core::Error::Msg("HuBERT has no convolution layers".to_string())
        })?;
        let size = config.hidden_size;
        let projection = vb.pp("feature_projection");

        // Stored weight-normalized along the kernel dimension.
        let kernel = config.num_conv_pos_embeddings;
        let groups = config.num_conv_pos_embedding_groups;
        let position = vb.pp("encoder.pos_conv_embed.conv");
        let parts = position.pp("parametrizations.weight");
        let magnitude = parts.get((1, 1, kernel), "original0")?;
        let direction = parts.get((size, size / groups, kernel), "original1")?;
        let norm = direction.sqr()?.sum_keepdim((0, 1))?.sqrt()?;
        let position_weight = direction.broadcast_mul(&magnitude.broadcast_div(&norm)?)?;
        let position_bias = position.get(size, "bias")?.reshape((1, size, 1))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(HubertLayer::load(
                config,
                vb.pp(format!("encoder.layers.{index}")),
            )?);
        }

        Ok(Self {
            feature_layers,
            projection_norm: candle_nn::layer_norm(
                last,
                config.layer_norm_eps,
                projection.pp("layer_norm"),
            )?,
            projection: candle_nn::linear(last, size, projection.pp("projection"))?,
            position_weight,
            position_bias,
            position_padding: kernel / 2,
            position_groups: groups,
            layer_norm: candle_nn::layer_norm(
                size,
                config.layer_norm_eps,
                vb.pp("encoder.layer_norm"),
            )?,
            layers,
        })
    }

    /// The mean of the embedding output and every layer's output.
    fn mean_hidden_state(&self, input: &Tensor) -> Result<Tensor> {
        let mut h = input.clone();
        for layer in &self.feature_layers {
            h = layer.forward(&h)?;
        }
        let h = self.projection.forward(
            &self
                .projection_norm
                .forward(&h.transpose(1, 2)?.contiguous()?)?,
        )?;

        let channels_first = h.transpose(1, 2)?.contiguous()?;
        let length = channels_first.dim(2)?;
        let position = channels_first
            .conv1d(
                &self.position_weight,
                self.position_padding,
                1,
                1,
                self.position_groups,
            )?
            .broadcast_add(&self.position_bias)?
            .narrow(2, 0, length)?
            .gelu_erf()?
            .transpose(1, 2)?
            .contiguous()?;
        let mut h = self.layer_norm.forward(&h.add(&position)?)?;

        let mut sum = h.clone();
        for layer in &self.layers {
            h = layer.forward(&h)?;
            sum = sum.add(&h)?;
        }
        sum / (self.layers.len() + 1) as f64
    }
}

struct SemanticResidualUnit {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl SemanticResidualUnit {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&x.elu(1.0)?)?;
        let h = self.conv2.forward(&h.elu(1.0)?)?;
        x.add(&h)
    }
}

struct SemanticBlock {
    units: Vec<SemanticResidualUnit>,
    conv: Conv1d,
}

struct SemanticEncoder {
    conv: Conv1d,
    blocks: Vec<SemanticBlock>,
}

impl SemanticEncoder {
    fn load(config: &OmniVoiceAudioTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let size = config.semantic_model_config.hidden_size;
        let conv = Conv1d::load(
            size,
            size,
            config.kernel_size,
            1,
            config.kernel_size / 2,
            1,
            1,
            false,
            vb.pp("conv"),
        )?;
        let mut in_channels = size;
        let mut blocks = Vec::with_capacity(config.strides.len());
        for (index, &stride) in config.strides.iter().enumerate() {
            let out_channels = size * config.channel_ratios[index];
            let block = vb.pp(format!("conv_blocks.{index}"));
            let mut units = Vec::with_capacity(config.block_dilations.len());
            for (unit_index, &dilation) in config.block_dilations.iter().enumerate() {
                let unit = block.pp(format!("res_units.{unit_index}"));
                units.push(SemanticResidualUnit {
                    conv1: Conv1d::load(
                        in_channels,
                        in_channels,
                        config.unit_kernel_size,
                        1,
                        (config.unit_kernel_size - 1) / 2 * dilation,
                        dilation,
                        1,
                        false,
                        unit.pp("conv1"),
                    )?,
                    conv2: Conv1d::load(
                        in_channels,
                        in_channels,
                        1,
                        1,
                        0,
                        1,
                        1,
                        false,
                        unit.pp("conv2"),
                    )?,
                });
            }
            let kernel = if stride == 1 { 3 } else { 2 * stride };
            let conv = Conv1d::load(
                in_channels,
                out_channels,
                kernel,
                stride,
                (kernel - 1) / 2,
                1,
                1,
                true,
                block.pp("conv"),
            )?;
            blocks.push(SemanticBlock { units, conv });
            in_channels = out_channels;
        }
        Ok(Self { conv, blocks })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv.forward(x)?;
        for block in &self.blocks {
            for unit in &block.units {
                h = unit.forward(&h)?;
            }
            h = block.conv.forward(&h)?;
        }
        Ok(h)
    }
}

struct DacEncoderBlock {
    res_unit1: DacResidualUnit,
    res_unit2: DacResidualUnit,
    res_unit3: DacResidualUnit,
    snake1_alpha: Tensor,
    conv1: Conv1d,
}

struct DacEncoder {
    conv1: Conv1d,
    blocks: Vec<DacEncoderBlock>,
    snake1_alpha: Tensor,
    conv2: Conv1d,
}

impl DacEncoder {
    fn load(config: &OmniVoiceAudioTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let acoustic = &config.acoustic_model_config;
        let base = acoustic.encoder_hidden_size;
        let conv1 = Conv1d::load(1, base, 7, 1, 3, 1, 1, true, vb.pp("conv1"))?;
        let mut blocks = Vec::with_capacity(acoustic.downsampling_ratios.len());
        let mut channels = base;
        for (index, &stride) in acoustic.downsampling_ratios.iter().enumerate() {
            let block = vb.pp(format!("block.{index}"));
            blocks.push(DacEncoderBlock {
                res_unit1: DacResidualUnit::load(channels, 1, block.pp("res_unit1"))?,
                res_unit2: DacResidualUnit::load(channels, 3, block.pp("res_unit2"))?,
                res_unit3: DacResidualUnit::load(channels, 9, block.pp("res_unit3"))?,
                snake1_alpha: block.pp("snake1").get((1, channels, 1), "alpha")?,
                conv1: Conv1d::load(
                    channels,
                    channels * 2,
                    2 * stride,
                    stride,
                    stride.div_ceil(2),
                    1,
                    1,
                    true,
                    block.pp("conv1"),
                )?,
            });
            channels *= 2;
        }
        Ok(Self {
            conv1,
            blocks,
            snake1_alpha: vb.pp("snake1").get((1, channels, 1), "alpha")?,
            conv2: Conv1d::load(
                channels,
                acoustic.hidden_size,
                3,
                1,
                1,
                1,
                1,
                true,
                vb.pp("conv2"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv1.forward(x)?;
        for block in &self.blocks {
            h = block.res_unit1.forward(&h)?;
            h = block.res_unit2.forward(&h)?;
            h = block.res_unit3.forward(&h)?;
            h = block.conv1.forward(&snake1d(&h, &block.snake1_alpha)?)?;
        }
        self.conv2.forward(&snake1d(&h, &self.snake1_alpha)?)
    }
}

struct Quantizer {
    embed: Tensor,
    embed_square_sum: Tensor,
    project_in: Linear,
    project_out: Linear,
}

impl Quantizer {
    fn load(config: &OmniVoiceAudioTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let embed = vb
            .pp("codebook")
            .get((config.codebook_size, config.codebook_dim), "embed")?;
        Ok(Self {
            embed_square_sum: embed.sqr()?.sum_keepdim(1)?.transpose(0, 1)?.contiguous()?,
            embed,
            project_in: candle_nn::linear(
                config.semantic_hidden_size(),
                config.codebook_dim,
                vb.pp("project_in"),
            )?,
            project_out: candle_nn::linear(
                config.codebook_dim,
                config.semantic_hidden_size(),
                vb.pp("project_out"),
            )?,
        })
    }

    /// Picks the nearest code per frame, and returns the codes and the
    /// residual left for the next quantizer. `residual` is (frames, size).
    fn encode(&self, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        let projected = self.project_in.forward(residual)?;
        let square_sum = projected.sqr()?.sum_keepdim(1)?;
        let similarity = (projected.matmul(&self.embed.t()?.contiguous()?)? * 2.0)?;
        let distance = square_sum
            .broadcast_add(&self.embed_square_sum)?
            .sub(&similarity)?;
        let codes = distance.neg()?.argmax(D::Minus1)?;
        let chosen = self.embed.index_select(&codes, 0)?;
        let rest = residual.sub(&self.project_out.forward(&chosen)?)?;
        Ok((codes, rest))
    }
}

pub struct OmniVoiceAudioEncoder {
    acoustic_encoder: DacEncoder,
    semantic_model: Hubert,
    semantic_encoder: SemanticEncoder,
    fc: Linear,
    quantizers: Vec<Quantizer>,
    sample_rate: u32,
    semantic_sample_rate: u32,
    semantic_stride: usize,
    downsampling_ratios: Vec<usize>,
    pad: usize,
    device: Device,
}

impl OmniVoiceAudioEncoder {
    pub fn load(
        config: &OmniVoiceAudioTokenizerConfig,
        num_quantizers: usize,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let size = config.semantic_hidden_size();
        let mut quantizers = Vec::with_capacity(num_quantizers);
        for index in 0..num_quantizers {
            quantizers.push(Quantizer::load(
                config,
                vb.pp(format!("quantizer.quantizers.{index}")),
            )?);
        }
        Ok(Self {
            acoustic_encoder: DacEncoder::load(config, vb.pp("acoustic_encoder"))?,
            semantic_model: Hubert::load(&config.semantic_model_config, vb.pp("semantic_model"))?,
            semantic_encoder: SemanticEncoder::load(config, vb.pp("encoder_semantic"))?,
            fc: candle_nn::linear(size, size, vb.pp("fc"))?,
            quantizers,
            sample_rate: config.sample_rate,
            semantic_sample_rate: config.semantic_sample_rate,
            semantic_stride: config.semantic_downsample_factor(),
            downsampling_ratios: config.acoustic_model_config.downsampling_ratios.clone(),
            pad: config.hop_length() / 2,
            device: device.clone(),
        })
    }

    /// Features from the semantic model, shaped (1, frames, size).
    pub fn semantic_features(&self, samples: &[f32]) -> Result<Tensor> {
        let resampled = resample(samples, self.sample_rate, self.semantic_sample_rate);
        let mut padded = vec![0.0f32; SEMANTIC_PAD];
        padded.extend_from_slice(&resampled);
        padded.resize(padded.len() + SEMANTIC_PAD, 0.0);
        let input = Tensor::from_vec(
            padded,
            (1, 1, resampled.len() + 2 * SEMANTIC_PAD),
            &self.device,
        )?;
        let features = self.semantic_model.mean_hidden_state(&input)?;
        let frames = features.dim(1)?;
        let kept: Vec<u32> = (0..frames as u32).step_by(self.semantic_stride).collect();
        let kept = Tensor::from_vec(kept, frames.div_ceil(self.semantic_stride), &self.device)?;
        features.index_select(&kept, 1)
    }

    pub fn acoustic_features(&self, samples: &[f32], semantic_frames: usize) -> Result<Tensor> {
        let input = Tensor::from_slice(samples, (1, 1, samples.len()), &self.device)?;
        let frames = self
            .downsampling_ratios
            .iter()
            .fold(samples.len(), |length, &stride| {
                (length + 2 * stride.div_ceil(2) - (2 * stride - 1) - 1) / stride + 1
            });
        if frames == semantic_frames {
            self.acoustic_encoder.forward(&input)
        } else {
            self.acoustic_encoder
                .forward(&input.pad_with_zeros(2, self.pad, self.pad)?)
        }
    }

    pub fn semantic_encoded(&self, features: &Tensor) -> Result<Tensor> {
        self.semantic_encoder
            .forward(&features.transpose(1, 2)?.contiguous()?)
    }

    /// Encodes mono audio at the codec's sample rate into codes laid out by
    /// codebook, returning them and the frame count.
    pub fn encode(&self, samples: &[f32]) -> Result<(Vec<u32>, usize)> {
        let features = self.semantic_features(samples)?;
        let semantic = self.semantic_encoded(&features)?;
        let acoustic = self.acoustic_features(samples, semantic.dim(2)?)?;
        let joined = Tensor::cat(&[&acoustic, &semantic], 1)?;
        let mut residual = self
            .fc
            .forward(&joined.transpose(1, 2)?.contiguous()?)?
            .squeeze(0)?;
        let frames = residual.dim(0)?;
        let mut codes = Vec::with_capacity(frames * self.quantizers.len());
        for quantizer in &self.quantizers {
            let (layer, rest) = quantizer.encode(&residual)?;
            codes.extend(layer.to_dtype(DType::U32)?.to_vec1::<u32>()?);
            residual = rest;
        }
        Ok((codes, frames))
    }
}
