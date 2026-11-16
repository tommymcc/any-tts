//! OmniVoice's audio decoder (codes to waveform) on burn, for AMD GPUs through
//! ROCm (`rocm` feature). It computes what [`super::audio_tokenizer`] does on
//! candle: the residual quantizer's codebook lookups and projections, `fc2`,
//! and the DAC decoder (an input conv, five upsampling blocks of a transposed
//! conv and three dilated residual units, Snake activations, an output conv).
//!
//! The convolutions are written as matmuls (see [`conv1d`] and
//! [`conv_transpose1d`]) rather than with burn's conv ops, whose speed on
//! GPUs without matrix cores depends on whether burn's `autotune` feature is
//! on. `OMNIVOICE_BURN_CONV=burn` switches to burn's ops, for comparison.

use burn::tensor::module::{
    conv1d as burn_conv1d, conv_transpose1d as burn_conv_transpose1d, embedding, linear,
};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{Device, Int, Tensor, TensorData};
use candle_core::DType;
use candle_nn::VarBuilder;

use super::config::OmniVoiceAudioTokenizerConfig;
use crate::burn_device::guard;
use crate::error::TtsError;

pub(super) struct BurnAudioDecoder {
    device: Device,
    /// Per quantizer: the codebook, (codebook_size, codebook_dim), and
    /// `project_out` as (codebook_dim, hidden) and its bias.
    quantizers: Vec<(Tensor<2>, Tensor<2>, Tensor<1>)>,
    /// `fc2` as (decoder_hidden, hidden) and its bias.
    fc2: (Tensor<2>, Tensor<1>),
    conv_in: Conv,
    blocks: Vec<Block>,
    snake_out: Snake,
    conv_out: Conv,
    burn_conv: bool,
}

/// x + sin²(αx) / α, per channel. Holds α and 1 / (α + 1e-9), each (1, C, 1).
struct Snake {
    alpha: Tensor<3>,
    inv_alpha: Tensor<3>,
}

/// A stride-1 conv1d: weight (out, in, kernel), bias (out).
struct Conv {
    weight: Tensor<3>,
    bias: Tensor<1>,
    padding: usize,
    dilation: usize,
}

/// A transposed conv1d: weight (in, out, kernel), bias (out).
struct ConvTranspose {
    weight: Tensor<3>,
    bias: Tensor<1>,
    stride: usize,
    padding: usize,
    output_padding: usize,
}

struct ResidualUnit {
    snake1: Snake,
    conv1: Conv,
    snake2: Snake,
    conv2: Conv,
}

struct Block {
    snake: Snake,
    upsample: ConvTranspose,
    units: [ResidualUnit; 3],
}

impl BurnAudioDecoder {
    /// Copies the decoder's weights from `vb`, the audio tokenizer's F32
    /// VarBuilder on the CPU, to `device`.
    pub(super) fn load(
        config: &OmniVoiceAudioTokenizerConfig,
        num_quantizers: usize,
        vb: &VarBuilder,
        device: Device,
    ) -> Result<Self, TtsError> {
        let dac = &config.acoustic_model_config;
        let tensor =
            |vb: &VarBuilder, name: &str, shape: &[usize]| -> Result<TensorData, TtsError> {
                let values = vb
                    .get(shape, name)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                Ok(TensorData::new(values, shape.to_vec()))
            };
        let load = || -> Result<Self, TtsError> {
            let t1 = |vb: &VarBuilder, name: &str, n: usize| -> Result<Tensor<1>, TtsError> {
                Ok(Tensor::from_data(tensor(vb, name, &[n])?, &device))
            };
            let t2 =
                |vb: &VarBuilder, name: &str, shape: [usize; 2]| -> Result<Tensor<2>, TtsError> {
                    Ok(Tensor::from_data(tensor(vb, name, &shape)?, &device))
                };
            let t3 =
                |vb: &VarBuilder, name: &str, shape: [usize; 3]| -> Result<Tensor<3>, TtsError> {
                    Ok(Tensor::from_data(tensor(vb, name, &shape)?, &device))
                };
            let snake = |vb: VarBuilder, channels: usize| -> Result<Snake, TtsError> {
                let alpha = t3(&vb, "alpha", [1, channels, 1])?;
                let inv_alpha = alpha.clone().add_scalar(1e-9).recip();
                Ok(Snake { alpha, inv_alpha })
            };
            let conv = |vb: VarBuilder,
                        inp: usize,
                        out: usize,
                        kernel: usize,
                        dilation: usize|
             -> Result<Conv, TtsError> {
                Ok(Conv {
                    weight: t3(&vb, "weight", [out, inp, kernel])?,
                    bias: t1(&vb, "bias", out)?,
                    padding: (kernel - 1) / 2 * dilation,
                    dilation,
                })
            };

            let mut quantizers = Vec::with_capacity(num_quantizers);
            for index in 0..num_quantizers {
                let vb = vb.pp(format!("quantizer.quantizers.{index}"));
                let codebook = t2(
                    &vb.pp("codebook"),
                    "embed",
                    [config.codebook_size, config.codebook_dim],
                )?;
                // Stored (out, in), as PyTorch does.
                let project = t2(
                    &vb.pp("project_out"),
                    "weight",
                    [dac.decoder_hidden_size, config.codebook_dim],
                )?;
                let bias = t1(&vb.pp("project_out"), "bias", dac.decoder_hidden_size)?;
                quantizers.push((codebook, project.transpose(), bias));
            }
            let fc2 = (
                t2(
                    &vb.pp("fc2"),
                    "weight",
                    [dac.hidden_size, dac.decoder_hidden_size],
                )?
                .transpose(),
                t1(&vb.pp("fc2"), "bias", dac.hidden_size)?,
            );

            let vb = vb.pp("acoustic_decoder");
            let conv_in = conv(
                vb.pp("conv1"),
                dac.hidden_size,
                dac.decoder_hidden_size,
                7,
                1,
            )?;
            let mut blocks = Vec::with_capacity(dac.upsampling_ratios.len());
            for (index, &stride) in dac.upsampling_ratios.iter().enumerate() {
                let vb = vb.pp(format!("block.{index}"));
                let (inp, out) = (
                    dac.decoder_hidden_size >> index,
                    dac.decoder_hidden_size >> (index + 1),
                );
                let unit = |name: &str, dilation: usize| -> Result<ResidualUnit, TtsError> {
                    let vb = vb.pp(name);
                    Ok(ResidualUnit {
                        snake1: snake(vb.pp("snake1"), out)?,
                        conv1: conv(vb.pp("conv1"), out, out, 7, dilation)?,
                        snake2: snake(vb.pp("snake2"), out)?,
                        conv2: conv(vb.pp("conv2"), out, out, 1, 1)?,
                    })
                };
                blocks.push(Block {
                    snake: snake(vb.pp("snake1"), inp)?,
                    upsample: ConvTranspose {
                        weight: t3(&vb.pp("conv_t1"), "weight", [inp, out, 2 * stride])?,
                        bias: t1(&vb.pp("conv_t1"), "bias", out)?,
                        stride,
                        padding: stride.div_ceil(2),
                        output_padding: stride % 2,
                    },
                    units: [
                        unit("res_unit1", 1)?,
                        unit("res_unit2", 3)?,
                        unit("res_unit3", 9)?,
                    ],
                });
            }
            let channels = dac.decoder_hidden_size >> dac.upsampling_ratios.len();
            Ok(Self {
                device: device.clone(),
                quantizers,
                fc2,
                conv_in,
                blocks,
                snake_out: snake(vb.pp("snake1"), channels)?,
                conv_out: conv(vb.pp("conv2"), channels, 1, 7, 1)?,
                burn_conv: std::env::var("OMNIVOICE_BURN_CONV").is_ok_and(|v| v == "burn"),
            })
        };
        guard("loading OmniVoice's audio decoder onto the GPU", load)
            .map_err(TtsError::ModelError)?
    }

    /// The waveform for `codes`, (num_quantizers, len) row-major: len frames,
    /// each upsampled to the hop length.
    pub(super) fn decode(&self, codes: &[u32], len: usize) -> Result<Vec<f32>, TtsError> {
        let num_quantizers = self.quantizers.len();
        if codes.len() != num_quantizers * len {
            return Err(TtsError::RuntimeError(format!(
                "OmniVoice audio decoder: expected {num_quantizers} x {len} codes, got {}",
                codes.len()
            )));
        }
        let codes: Vec<i64> = codes.iter().map(|&c| i64::from(c)).collect();
        let run = || -> Result<Vec<f32>, String> {
            let codes = Tensor::<2, Int>::from_data(
                TensorData::new(codes, [num_quantizers, len]),
                &self.device,
            );
            // Sum over quantizers of project_out(codebook[code]), (1, len, hidden).
            let mut summed: Option<Tensor<3>> = None;
            for (index, (codebook, project, bias)) in self.quantizers.iter().enumerate() {
                let indices = codes.clone().narrow(0, index, 1);
                let vectors = embedding(codebook.clone(), indices);
                let projected = linear(vectors, project.clone(), Some(bias.clone()));
                summed = Some(match summed {
                    Some(sum) => sum + projected,
                    None => projected,
                });
            }
            let summed = summed.ok_or("OmniVoice's audio decoder has no quantizers")?;
            let x = linear(summed, self.fc2.0.clone(), Some(self.fc2.1.clone())).swap_dims(1, 2);
            let mut x = self.conv(&self.conv_in, x);
            for block in &self.blocks {
                x = self.conv_transpose(&block.upsample, block.snake.forward(x));
                for unit in &block.units {
                    let h = self.conv(&unit.conv1, unit.snake1.forward(x.clone()));
                    let h = self.conv(&unit.conv2, unit.snake2.forward(h));
                    x = x + h;
                }
            }
            let x = self
                .conv(&self.conv_out, self.snake_out.forward(x))
                .clamp(-1.0, 1.0);
            x.into_data()
                .try_to_vec::<f32>()
                .map_err(|e| format!("{e:?}"))
        };
        guard("running OmniVoice's audio decoder on the GPU", run)
            .and_then(|result| result)
            .map_err(TtsError::RuntimeError)
    }

    fn conv(&self, conv: &Conv, x: Tensor<3>) -> Tensor<3> {
        if self.burn_conv {
            let options = ConvOptions::new([1], [conv.padding], [conv.dilation], 1);
            burn_conv1d(x, conv.weight.clone(), Some(conv.bias.clone()), options)
        } else {
            conv1d(x, conv)
        }
    }

    fn conv_transpose(&self, conv: &ConvTranspose, x: Tensor<3>) -> Tensor<3> {
        if self.burn_conv {
            let options = ConvTransposeOptions::new(
                [conv.stride],
                [conv.padding],
                [conv.output_padding],
                [1],
                1,
            );
            burn_conv_transpose1d(x, conv.weight.clone(), Some(conv.bias.clone()), options)
        } else {
            conv_transpose1d(x, conv)
        }
    }
}

impl Snake {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let s = (x.clone() * self.alpha.clone()).sin();
        x + s.clone() * s * self.inv_alpha.clone()
    }
}

/// A stride-1, "same"-padded conv1d on (batch, in, len) as one matmul: the
/// kernel's shifted views of the zero-padded input are stacked along the
/// channels, (batch, kernel * in, len), and multiplied by the weight laid out
/// as (out, kernel * in).
fn conv1d(x: Tensor<3>, conv: &Conv) -> Tensor<3> {
    let [batch, channels, len] = x.dims();
    let [out, _, kernel] = conv.weight.dims();
    let weight = conv
        .weight
        .clone()
        .swap_dims(1, 2)
        .reshape([1, out, kernel * channels]);
    let bias = conv.bias.clone().reshape([1, out, 1]);
    if kernel == 1 {
        return weight.matmul(x) + bias;
    }
    let pad = Tensor::<3>::zeros([batch, channels, conv.padding], &x.device());
    let padded = Tensor::cat(vec![pad.clone(), x, pad], 2);
    let shifted = (0..kernel)
        .map(|j| padded.clone().narrow(2, j * conv.dilation, len))
        .collect();
    weight.matmul(Tensor::cat(shifted, 1)) + bias
}

/// A transposed conv1d on (batch, in, len) whose kernel is twice its stride s,
/// as DAC's are, as one matmul and an overlap-add. The matmul gives every
/// input frame's 2s output samples, (batch, out, 2, s, len); output frame n of
/// the unpadded result, s samples long, is the first half of input frame n's
/// plus the second half of frame n - 1's. The padding then crops the ends.
fn conv_transpose1d(x: Tensor<3>, conv: &ConvTranspose) -> Tensor<3> {
    let [batch, _, len] = x.dims();
    let [inp, out, kernel] = conv.weight.dims();
    let stride = conv.stride;
    assert_eq!(
        kernel,
        2 * stride,
        "conv_transpose1d here needs a kernel of twice the stride"
    );
    let weight = conv
        .weight
        .clone()
        .reshape([1, inp, out * kernel])
        .swap_dims(1, 2);
    let parts = weight.matmul(x).reshape([batch, out, 2, stride, len]);
    let device = parts.device();
    let zeros = Tensor::<5>::zeros([batch, out, 1, stride, 1], &device);
    let first = Tensor::cat(vec![parts.clone().narrow(2, 0, 1), zeros.clone()], 4);
    let second = Tensor::cat(vec![zeros, parts.narrow(2, 1, 1)], 4);
    let full = (first + second)
        .reshape([batch, out, stride, len + 1])
        .swap_dims(2, 3)
        .reshape([batch, out, (len + 1) * stride]);
    let out_len = (len - 1) * stride + kernel + conv.output_padding - 2 * conv.padding;
    full.narrow(2, conv.padding, out_len) + conv.bias.clone().reshape([1, out, 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i + seed) * 2654435761) % 1000) as f32 / 500.0 - 1.0)
            .collect()
    }

    fn max_diff(a: Tensor<3>, b: Tensor<3>) -> f32 {
        assert_eq!(a.dims(), b.dims());
        let a: Vec<f32> = a.into_data().try_to_vec::<f32>().unwrap();
        let b: Vec<f32> = b.into_data().try_to_vec::<f32>().unwrap();
        a.iter()
            .zip(&b)
            .fold(0.0f32, |d, (x, y)| d.max((x - y).abs()))
    }

    /// The matmul convolutions against burn's conv ops, at DAC's settings.
    /// Needs a ROCm GPU; skipped without one.
    #[test]
    fn convs_match_burns() {
        let Ok(device) = crate::burn_device::rocm_device(0) else {
            eprintln!("no ROCm device; skipping");
            return;
        };
        let t3 = |shape: [usize; 3], seed| {
            Tensor::<3>::from_data(
                TensorData::new(values(shape.iter().product(), seed), shape),
                &device,
            )
        };
        let t1 = |n, seed| Tensor::<1>::from_data(TensorData::new(values(n, seed), [n]), &device);
        for (kernel, dilation) in [(7, 1), (7, 3), (7, 9), (1, 1)] {
            let conv = Conv {
                weight: t3([6, 5, kernel], 1),
                bias: t1(6, 2),
                padding: (kernel - 1) / 2 * dilation,
                dilation,
            };
            let x = t3([2, 5, 40], 3);
            let want = burn_conv1d(
                x.clone(),
                conv.weight.clone(),
                Some(conv.bias.clone()),
                ConvOptions::new([1], [conv.padding], [dilation], 1),
            );
            let diff = max_diff(conv1d(x, &conv), want);
            assert!(
                diff < 1e-4,
                "conv1d kernel {kernel} dilation {dilation}: max |diff| {diff}"
            );
        }
        for stride in [8, 5, 4, 2, 3] {
            let conv = ConvTranspose {
                weight: t3([5, 6, 2 * stride], 4),
                bias: t1(6, 5),
                stride,
                padding: stride.div_ceil(2),
                output_padding: stride % 2,
            };
            let x = t3([2, 5, 11], 6);
            let options =
                ConvTransposeOptions::new([stride], [conv.padding], [conv.output_padding], [1], 1);
            let want = burn_conv_transpose1d(
                x.clone(),
                conv.weight.clone(),
                Some(conv.bias.clone()),
                options,
            );
            let diff = max_diff(conv_transpose1d(x, &conv), want);
            assert!(
                diff < 1e-4,
                "conv_transpose1d stride {stride}: max |diff| {diff}"
            );
        }
    }

    /// The whole decoder against candle's on the CPU, with OmniVoice's
    /// weights from the Hugging Face cache (or `OMNIVOICE_AUDIO_TOKENIZER_DIR`),
    /// and how long each takes. Ignored by default: it needs the 800 MB
    /// checkpoint and a ROCm GPU.
    #[test]
    #[ignore]
    fn decoder_matches_candle() {
        use super::super::audio_tokenizer::OmniVoiceAudioTokenizerDecoder;
        use std::time::Instant;
        let dir = std::env::var_os("OMNIVOICE_AUDIO_TOKENIZER_DIR")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                let snapshots =
                    dirs_home().join(".cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots");
                std::fs::read_dir(snapshots)
                    .ok()?
                    .flatten()
                    .map(|e| e.path().join("audio_tokenizer"))
                    .find(|p| p.is_dir())
            });
        let Some(dir) = dir else {
            eprintln!("no OmniVoice audio tokenizer found; skipping");
            return;
        };
        let device = crate::burn_device::rocm_device(0).expect("ROCm device");
        let config = OmniVoiceAudioTokenizerConfig::from_bytes(
            std::fs::read(dir.join("config.json")).unwrap(),
        )
        .unwrap();
        let cpu = candle_core::Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &cpu)
        }
        .unwrap();
        // 75 frames is 3 s; OMNIVOICE_DECODER_TEST_FRAMES picks another length.
        let len = std::env::var("OMNIVOICE_DECODER_TEST_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(75);
        let quantizers = 8;
        let candle_decoder =
            OmniVoiceAudioTokenizerDecoder::load(&config, quantizers, vb.clone(), &cpu).unwrap();
        let burn_decoder = BurnAudioDecoder::load(&config, quantizers, &vb, device).unwrap();
        let codes: Vec<u32> = (0..quantizers * len)
            .map(|i| ((i * 2654435761) % config.codebook_size) as u32)
            .collect();

        let started = Instant::now();
        let tensor =
            candle_core::Tensor::from_vec(codes.clone(), (1, quantizers, len), &cpu).unwrap();
        let want = candle_decoder
            .decode(&tensor)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let candle_time = started.elapsed();
        let mut burn_times = Vec::new();
        let mut got = Vec::new();
        for _ in 0..3 {
            let started = Instant::now();
            got = burn_decoder.decode(&codes, len).unwrap();
            burn_times.push(started.elapsed());
        }
        assert_eq!(want.len(), got.len());
        let diff = want
            .iter()
            .zip(&got)
            .fold(0.0f32, |d, (a, b)| d.max((a - b).abs()));
        eprintln!(
            "{len} frames, {} samples: candle CPU {candle_time:.2?}, burn {burn_times:.2?}, max |diff| {diff:.1e}",
            got.len()
        );
        assert!(diff < 1e-3, "max |diff| {diff}");
    }

    fn dirs_home() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
    }
}
