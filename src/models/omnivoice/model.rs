use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module};
use tracing::info;
use unicode_general_category::{get_general_category, GeneralCategory};

use crate::audio::AudioSamples;
use crate::config::{ModelFiles, TtsConfig};
use crate::error::TtsError;
use crate::layers::attention::GqaConfig;
use crate::layers::transformer::TransformerBlock;
use crate::tensor_utils::{precompute_rope_freqs, RmsNorm};
use crate::tokenizer::TextTokenizer;
use crate::traits::{ModelInfo, ReferenceAudio, SynthesisRequest, TtsModel};

use super::audio_encoder::OmniVoiceAudioEncoder;
use super::audio_processing::remove_silence;
use super::audio_tokenizer::OmniVoiceAudioTokenizerDecoder;
use super::config::{OmniVoiceAudioTokenizerConfig, OmniVoiceConfig};

const DEFAULT_NUM_STEPS: usize = 32;
const DEFAULT_GUIDANCE_SCALE: f64 = 2.0;
const DEFAULT_T_SHIFT: f64 = 0.1;
const DEFAULT_LAYER_PENALTY: f32 = 5.0;
const DEFAULT_POSITION_TEMPERATURE: f32 = 5.0;
const DEFAULT_CLASS_TEMPERATURE: f32 = 0.0;
const DEFAULT_AUDIO_CHUNK_DURATION: f32 = 15.0;
const DEFAULT_AUDIO_CHUNK_THRESHOLD: f32 = 30.0;
/// References quieter than this are scaled up to it for encoding.
const REFERENCE_MIN_RMS: f32 = 0.1;

struct OmniVoiceGenerationConfig {
    num_step: usize,
    guidance_scale: f64,
    t_shift: f64,
    layer_penalty_factor: f32,
    position_temperature: f32,
    class_temperature: f32,
    denoise: bool,
    audio_chunk_duration: f32,
    audio_chunk_threshold: f32,
}

impl OmniVoiceGenerationConfig {
    fn from_request(request: &SynthesisRequest) -> Self {
        Self {
            num_step: DEFAULT_NUM_STEPS,
            guidance_scale: request.cfg_scale.unwrap_or(DEFAULT_GUIDANCE_SCALE),
            t_shift: DEFAULT_T_SHIFT,
            layer_penalty_factor: DEFAULT_LAYER_PENALTY,
            position_temperature: env_override_f32(
                "OMNIVOICE_POSITION_TEMPERATURE",
                DEFAULT_POSITION_TEMPERATURE,
            ),
            class_temperature: request
                .temperature
                .unwrap_or(DEFAULT_CLASS_TEMPERATURE as f64) as f32,
            denoise: true,
            audio_chunk_duration: env_override_f32(
                "OMNIVOICE_AUDIO_CHUNK_DURATION",
                DEFAULT_AUDIO_CHUNK_DURATION,
            ),
            audio_chunk_threshold: env_override_f32(
                "OMNIVOICE_AUDIO_CHUNK_THRESHOLD",
                DEFAULT_AUDIO_CHUNK_THRESHOLD,
            ),
        }
    }
}

struct InferenceInputs {
    input_ids: Vec<u32>,
    audio_mask: Vec<u8>,
    seq_len: usize,
    target_len: usize,
}

#[derive(Clone)]
struct GeneratedAudioTokens {
    tokens: Vec<u32>,
    len: usize,
}

/// A prepared voice: the reference as codec tokens, with its text and level.
struct VoicePrompt {
    source: ReferenceAudio,
    tokens: GeneratedAudioTokens,
    text: String,
    rms: f32,
}

/// A voice prompt, or nothing when generating without a reference.
#[derive(Clone, Copy, Default)]
struct PromptView<'a> {
    text: Option<&'a str>,
    tokens: Option<&'a GeneratedAudioTokens>,
}

impl<'a> From<&'a VoicePrompt> for PromptView<'a> {
    fn from(prompt: &'a VoicePrompt) -> Self {
        Self {
            text: Some(&prompt.text),
            tokens: Some(&prompt.tokens),
        }
    }
}

struct OmniVoiceBackbone {
    config: OmniVoiceConfig,
    text_embeddings: Embedding,
    audio_embeddings: Embedding,
    audio_head: Linear,
    norm: RmsNorm,
    layers: Vec<TransformerBlock>,
    rope_cos: Tensor,
    rope_sin: Tensor,
    codebook_layer_offsets: Tensor,
}

impl OmniVoiceBackbone {
    fn load(
        config: &OmniVoiceConfig,
        vb: candle_nn::VarBuilder,
        device: &Device,
        dtype: DType,
    ) -> Result<Self, TtsError> {
        let llm = &config.llm_config;
        let gqa_config = GqaConfig::with_head_dim(
            llm.hidden_size,
            llm.num_attention_heads,
            llm.num_key_value_heads,
            llm.head_dim(),
            llm.max_position_embeddings,
            llm.rope_theta(),
            llm.rms_norm_eps,
        );

        let text_embeddings =
            candle_nn::embedding(llm.vocab_size, llm.hidden_size, vb.pp("llm.embed_tokens"))?;
        let audio_embeddings = candle_nn::embedding(
            config.num_audio_codebook * config.audio_vocab_size,
            llm.hidden_size,
            vb.pp("audio_embeddings"),
        )?;
        let audio_head = candle_nn::linear_no_bias(
            llm.hidden_size,
            config.num_audio_codebook * config.audio_vocab_size,
            vb.pp("audio_heads"),
        )?;
        let norm = RmsNorm::load(llm.hidden_size, llm.rms_norm_eps, vb.pp("llm.norm"))?;

        let mut layers = Vec::with_capacity(llm.num_hidden_layers);
        for layer_index in 0..llm.num_hidden_layers {
            layers.push(TransformerBlock::load(
                &gqa_config,
                llm.intermediate_size,
                vb.pp(format!("llm.layers.{}", layer_index)),
            )?);
        }

        let (rope_cos, rope_sin) = precompute_rope_freqs(
            gqa_config.head_dim,
            llm.max_position_embeddings,
            llm.rope_theta(),
            device,
            dtype,
        )?;
        // Stored as integers, which can't load through a low-precision float
        // VarBuilder: candle has no CUDA conversion from BF16 back to U32.
        let offsets: Vec<u32> = (0..config.num_audio_codebook)
            .map(|layer| (layer * config.audio_vocab_size) as u32)
            .collect();
        let codebook_layer_offsets =
            Tensor::from_vec(offsets, (1, config.num_audio_codebook, 1), device)?;

        Ok(Self {
            config: config.clone(),
            text_embeddings,
            audio_embeddings,
            audio_head,
            norm,
            layers,
            rope_cos,
            rope_sin,
            codebook_layer_offsets,
        })
    }

    fn forward(
        &mut self,
        input_ids: &Tensor,
        audio_mask_u32: &Tensor,
        audio_mask_f32: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, TtsError> {
        for layer in &mut self.layers {
            layer.clear_cache();
        }

        let text_ids = input_ids.i((.., 0, ..))?;
        let text_embeds = self.text_embeddings.forward(&text_ids)?;

        let shifted_audio_ids = input_ids
            .broadcast_mul(&audio_mask_u32.unsqueeze(1)?)?
            .broadcast_add(&self.codebook_layer_offsets)?;
        let audio_embeds = self.audio_embeddings.forward(&shifted_audio_ids)?.sum(1)?;

        let audio_mask = audio_mask_f32.to_dtype(text_embeds.dtype())?.unsqueeze(2)?;
        let text_mask = audio_mask.neg()?.add(&Tensor::ones_like(&audio_mask)?)?;
        let mut hidden = text_embeds
            .broadcast_mul(&text_mask)?
            .add(&audio_embeds.broadcast_mul(&audio_mask)?)?;

        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rope_cos, &self.rope_sin, 0, attention_mask)?;
        }
        hidden = self.norm.forward(&hidden)?;

        let (batch, seq_len, _) = hidden.dims3()?;
        let logits_flat = self.audio_head.forward(&hidden)?;
        logits_flat
            .reshape((
                batch,
                seq_len,
                self.config.num_audio_codebook,
                self.config.audio_vocab_size,
            ))?
            .transpose(1, 2)
            .map_err(Into::into)
    }
}

pub struct OmniVoiceModel {
    config: OmniVoiceConfig,
    files: ModelFiles,
    frame_rate: usize,
    compute_device: Device,
    compute_dtype: DType,
    audio_device: Device,
    tokenizer: TextTokenizer,
    backbone: Mutex<OmniVoiceBackbone>,
    causal_mask_cache: Mutex<HashMap<usize, Tensor>>,
    audio_tokenizer_config: OmniVoiceAudioTokenizerConfig,
    audio_tokenizer: OmniVoiceAudioTokenizerDecoder,
    voice_prompt: Mutex<Option<Arc<VoicePrompt>>>,
    mask_token_penalty: Tensor,
}

impl TtsModel for OmniVoiceModel {
    fn load(config: TtsConfig) -> Result<Self, TtsError> {
        let compute_device = config.device.resolve()?;
        let dtype_override = env_override_dtype("OMNIVOICE_DTYPE");
        let mut compute_dtype = dtype_override.unwrap_or_else(|| config.dtype.to_candle());
        if matches!(compute_device, Device::Cpu) && compute_dtype == DType::BF16 {
            info!("BF16 is not supported on CPU; falling back to F32 for OmniVoice");
            compute_dtype = DType::F32;
        } else if matches!(compute_device, Device::Metal(_)) {
            if dtype_override.is_none() && compute_dtype != DType::F32 {
                info!(
                    "OmniVoice Metal F16/BF16 generation is numerically unstable; preferring F32"
                );
                compute_dtype = DType::F32;
            } else if compute_dtype == DType::BF16 {
                info!("BF16 is not supported on Metal for OmniVoice; preferring F32");
                compute_dtype = DType::F32;
            }
        }

        let files = config.resolve_files()?;
        let config_bytes = files
            .config
            .as_ref()
            .expect("validated by resolve_files")
            .read_bytes()?;
        let model_config = OmniVoiceConfig::from_bytes(config_bytes.as_ref())?;
        let tokenizer = TextTokenizer::from_asset(
            files
                .tokenizer
                .as_ref()
                .expect("validated by resolve_files"),
        )?;
        let audio_tokenizer_config_bytes = files
            .speech_tokenizer_config
            .as_ref()
            .expect("validated by resolve_files")
            .read_bytes()?;
        let audio_tokenizer_config =
            OmniVoiceAudioTokenizerConfig::from_bytes(audio_tokenizer_config_bytes.as_ref())?;

        let main_vb =
            ModelFiles::load_safetensors_vb(&files.weights, compute_dtype, &compute_device)?;
        let backbone =
            OmniVoiceBackbone::load(&model_config, main_vb, &compute_device, compute_dtype)?;

        let audio_device = if matches!(compute_device, Device::Metal(_)) {
            Device::Cpu
        } else {
            compute_device.clone()
        };
        // Upstream runs the codec in F32 whatever the backbone uses.
        let audio_vb = ModelFiles::load_safetensors_vb(
            &files.speech_tokenizer_weights,
            DType::F32,
            &audio_device,
        )?;
        let audio_tokenizer = OmniVoiceAudioTokenizerDecoder::load(
            &audio_tokenizer_config,
            model_config.num_audio_codebook,
            audio_vb,
            &audio_device,
        )?;

        let mut mask_penalty = vec![0.0f32; model_config.audio_vocab_size];
        if (model_config.audio_mask_id as usize) < mask_penalty.len() {
            mask_penalty[model_config.audio_mask_id as usize] = f32::NEG_INFINITY;
        }
        let mask_token_penalty = Tensor::new(mask_penalty.as_slice(), &compute_device)?
            .reshape((1, 1, 1, model_config.audio_vocab_size))?;

        info!(
            "Loading OmniVoice on {:?} (audio decoder on {:?})",
            compute_device, audio_device
        );

        Ok(Self {
            config: model_config,
            files,
            frame_rate: audio_tokenizer_config.frame_rate(),
            compute_device,
            compute_dtype,
            audio_device,
            tokenizer,
            backbone: Mutex::new(backbone),
            causal_mask_cache: Mutex::new(HashMap::new()),
            audio_tokenizer_config,
            audio_tokenizer,
            voice_prompt: Mutex::new(None),
            mask_token_penalty,
        })
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<AudioSamples, TtsError> {
        let speed = request.speed.unwrap_or(1.0);
        if !speed.is_finite() || speed <= 0.0 {
            return Err(TtsError::ModelError(format!(
                "OmniVoice speaking speed must be above zero, not {speed}"
            )));
        }
        if request.max_tokens == Some(0)
            || request.temperature.is_some_and(|v| !v.is_finite())
            || request.cfg_scale.is_some_and(|v| !v.is_finite())
        {
            return Err(TtsError::ConfigError(
                "Generation limits must be positive and sampling parameters finite".into(),
            ));
        }
        if request.voice.is_some() {
            return Err(TtsError::ModelError(
                "OmniVoice does not use named voices. Use `instruct` for voice design instead."
                    .to_string(),
            ));
        }
        if request.voice_embedding.is_some() {
            return Err(TtsError::ModelError(
                "Native OmniVoice does not support reusable voice embeddings yet.".to_string(),
            ));
        }

        let gen_config = OmniVoiceGenerationConfig::from_request(request);
        let normalized_language = self.normalize_language(request.language.as_deref());
        let normalized_instruct = normalize_instruct(request.instruct.as_deref(), &request.text);
        let prompt = request
            .reference_audio
            .as_ref()
            .map(|reference| self.prepare_voice(reference))
            .transpose()?;
        let view = prompt.as_deref().map(PromptView::from).unwrap_or_default();
        let target_len = request.max_tokens.unwrap_or_else(|| {
            estimate_target_tokens_with_reference(
                &request.text,
                view.text,
                view.tokens.map(|t| t.len),
                speed,
            )
        });
        let token_chunks = self.generate_token_chunks(
            &request.text,
            target_len,
            normalized_language.as_deref(),
            normalized_instruct.as_deref(),
            &gen_config,
            view,
            speed,
        )?;
        let samples = self.decode_token_chunks(&token_chunks)?;
        let samples = post_process_audio(
            &samples,
            self.sample_rate(),
            prompt.as_deref().map(|p| p.rms),
        );

        Ok(AudioSamples::new(samples, self.sample_rate()))
    }

    fn sample_rate(&self) -> u32 {
        self.audio_tokenizer.sample_rate()
    }

    fn supported_languages(&self) -> Vec<String> {
        vec![
            "auto".to_string(),
            "en".to_string(),
            "zh".to_string(),
            "ja".to_string(),
            "ko".to_string(),
            "de".to_string(),
            "fr".to_string(),
            "es".to_string(),
            "pt".to_string(),
            "ru".to_string(),
            "it".to_string(),
        ]
    }

    fn supported_voices(&self) -> Vec<String> {
        Vec::new()
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "OmniVoice".to_string(),
            variant: format!(
                "native ({})",
                self.files
                    .config
                    .as_ref()
                    .map(|path| path.display_name())
                    .unwrap_or_else(|| "unresolved".to_string())
            ),
            parameters: 600_000_000,
            sample_rate: self.sample_rate(),
            languages: self.supported_languages(),
            voices: self.supported_voices(),
        }
    }
}

impl OmniVoiceModel {
    /// Encodes a reference, reusing the last result for the same reference,
    /// since a book asks for the same voice thousands of times.
    fn prepare_voice(&self, reference: &ReferenceAudio) -> Result<Arc<VoicePrompt>, TtsError> {
        let mut cached = self.voice_prompt.lock().map_err(|_| {
            TtsError::RuntimeError("OmniVoice voice prompt mutex was poisoned".to_string())
        })?;
        if let Some(prompt) = cached.as_ref() {
            if prompt.source.samples == reference.samples
                && prompt.source.sample_rate == reference.sample_rate
                && prompt.source.transcript == reference.transcript
            {
                return Ok(prompt.clone());
            }
        }
        let prompt = Arc::new(self.encode_voice(reference)?);
        *cached = Some(prompt.clone());
        Ok(prompt)
    }

    fn encode_voice(&self, reference: &ReferenceAudio) -> Result<VoicePrompt, TtsError> {
        let transcript = reference
            .transcript
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .ok_or_else(|| {
                TtsError::ConfigError(
                    "OmniVoice needs a transcript of the reference audio".to_string(),
                )
            })?;
        if reference.sample_rate != self.sample_rate() {
            return Err(TtsError::ConfigError(format!(
                "OmniVoice needs reference audio at {} Hz, got {} Hz",
                self.sample_rate(),
                reference.sample_rate
            )));
        }
        let (samples, rms) = prepare_reference_samples(
            &reference.samples,
            self.sample_rate(),
            self.audio_tokenizer_config.hop_length(),
        )?;

        let vb = ModelFiles::load_safetensors_vb(
            &self.files.speech_tokenizer_weights,
            DType::F32,
            &self.audio_device,
        )?;
        let encoder = OmniVoiceAudioEncoder::load(
            &self.audio_tokenizer_config,
            self.config.num_audio_codebook,
            vb,
            &self.audio_device,
        )?;
        let (tokens, len) = encoder.encode(&samples)?;
        Ok(VoicePrompt {
            source: reference.clone(),
            tokens: GeneratedAudioTokens { tokens, len },
            text: add_final_punctuation(transcript),
            rms,
        })
    }

    fn causal_attention_mask(&self, seq_len: usize) -> Result<Tensor, TtsError> {
        if let Some(mask) = self
            .causal_mask_cache
            .lock()
            .map_err(|_| {
                TtsError::RuntimeError("OmniVoice mask cache mutex was poisoned".to_string())
            })?
            .get(&seq_len)
        {
            return Ok(mask.clone());
        }

        let mask = build_causal_attention_mask(seq_len, &self.compute_device, self.compute_dtype)?;
        self.causal_mask_cache
            .lock()
            .map_err(|_| {
                TtsError::RuntimeError("OmniVoice mask cache mutex was poisoned".to_string())
            })?
            .insert(seq_len, mask.clone());
        Ok(mask)
    }

    fn normalize_language(&self, language: Option<&str>) -> Option<String> {
        let language = language?.trim();
        if language.is_empty() || matches!(language.to_ascii_lowercase().as_str(), "auto" | "none")
        {
            return None;
        }

        let lower = language.to_ascii_lowercase();
        let mapped = match lower.as_str() {
            "english" => "en",
            "chinese" => "zh",
            "japanese" => "ja",
            "korean" => "ko",
            "german" => "de",
            "french" => "fr",
            "spanish" => "es",
            "portuguese" => "pt",
            "russian" => "ru",
            "italian" => "it",
            _ => lower.as_str(),
        };
        Some(mapped.to_string())
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_token_chunks(
        &self,
        text: &str,
        target_len: usize,
        language: Option<&str>,
        instruct: Option<&str>,
        gen_config: &OmniVoiceGenerationConfig,
        prompt: PromptView,
        speed: f64,
    ) -> Result<Vec<GeneratedAudioTokens>, TtsError> {
        let chunk_threshold =
            (gen_config.audio_chunk_threshold.max(0.0) * self.frame_rate as f32).round() as usize;
        let single = |prompt: PromptView| {
            self.generate_audio_tokens(
                text,
                target_len,
                language,
                instruct,
                gen_config,
                prompt.text,
                prompt.tokens,
            )
            .map(|tokens| vec![tokens])
        };
        if gen_config.audio_chunk_duration <= 0.0
            || chunk_threshold == 0
            || target_len <= chunk_threshold
        {
            return single(prompt);
        }

        let chunks = chunk_text_for_target(
            text,
            target_len,
            self.frame_rate,
            gen_config.audio_chunk_duration,
        );
        if chunks.len() <= 1 {
            return single(prompt);
        }

        info!(
            "OmniVoice chunking long request into {} chunks (estimated {} audio tokens)",
            chunks.len(),
            target_len
        );

        let mut generated = Vec::with_capacity(chunks.len());
        // Without a reference, the first chunk becomes the voice for the rest.
        let first_chunk;
        let (chunk_text, chunk_tokens, remaining) = match prompt.tokens {
            Some(tokens) => (prompt.text, tokens, &chunks[..]),
            None => {
                first_chunk = self.generate_audio_tokens(
                    &chunks[0],
                    estimate_target_tokens_with_reference(&chunks[0], None, None, speed),
                    language,
                    instruct,
                    gen_config,
                    None,
                    None,
                )?;
                generated.push(first_chunk.clone());
                (Some(chunks[0].as_str()), &first_chunk, &chunks[1..])
            }
        };
        for chunk in remaining {
            let chunk_target_len = estimate_target_tokens_with_reference(
                chunk,
                chunk_text,
                Some(chunk_tokens.len),
                speed,
            );
            generated.push(self.generate_audio_tokens(
                chunk,
                chunk_target_len,
                language,
                instruct,
                gen_config,
                chunk_text,
                Some(chunk_tokens),
            )?);
        }

        Ok(generated)
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_audio_tokens(
        &self,
        text: &str,
        target_len: usize,
        language: Option<&str>,
        instruct: Option<&str>,
        gen_config: &OmniVoiceGenerationConfig,
        ref_text: Option<&str>,
        ref_audio_tokens: Option<&GeneratedAudioTokens>,
    ) -> Result<GeneratedAudioTokens, TtsError> {
        let inputs = self.prepare_inference_inputs(
            text,
            target_len,
            language,
            instruct,
            gen_config.denoise,
            ref_text,
            ref_audio_tokens,
        )?;

        let cond_len = inputs.seq_len;
        let target_len = inputs.target_len;
        let mask_id = self.config.audio_mask_id;
        let total_masked = self.config.num_audio_codebook * target_len;
        let schedule = build_schedule(total_masked, gen_config.num_step, gen_config.t_shift);
        let use_causal_mask = std::env::var_os("OMNIVOICE_CAUSAL_MASK").is_some();
        let cond_attention_mask = if use_causal_mask {
            Some(self.causal_attention_mask(cond_len)?)
        } else {
            None
        };
        let uncond_attention_mask = if use_causal_mask {
            Some(self.causal_attention_mask(target_len)?)
        } else {
            None
        };

        let mut cond_input_ids = inputs.input_ids.clone();
        let mut uncond_input_ids = vec![mask_id; self.config.num_audio_codebook * target_len];

        let cond_audio_mask_u32_values: Vec<u32> =
            inputs.audio_mask.iter().copied().map(u32::from).collect();
        let cond_audio_mask_f32_values: Vec<f32> =
            inputs.audio_mask.iter().copied().map(f32::from).collect();
        let cond_audio_mask_u32 =
            Tensor::new(cond_audio_mask_u32_values.as_slice(), &self.compute_device)?
                .reshape((1, cond_len))?;
        let cond_audio_mask_f32 =
            Tensor::new(cond_audio_mask_f32_values.as_slice(), &self.compute_device)?
                .reshape((1, cond_len))?;
        let uncond_audio_mask_u32 =
            Tensor::ones((1, target_len), DType::U32, &self.compute_device)?;
        let uncond_audio_mask_f32 =
            Tensor::ones((1, target_len), DType::F32, &self.compute_device)?;
        let mut tokens = vec![mask_id; self.config.num_audio_codebook * target_len];
        let mut rng = SimpleRng::new(omnivoice_seed());
        let layer_penalties: Vec<f32> = (0..self.config.num_audio_codebook)
            .map(|layer| layer as f32 * gen_config.layer_penalty_factor)
            .collect();
        let mut step_trace = Vec::new();

        for (step_index, step_k) in schedule.iter().copied().enumerate() {
            if step_k == 0 {
                continue;
            }

            let cond_input_ids_tensor = Tensor::new(
                cond_input_ids.as_slice(),
                &self.compute_device,
            )?
            .reshape((1, self.config.num_audio_codebook, cond_len))?;
            let cond_logits = self
                .backbone
                .lock()
                .map_err(|_| {
                    TtsError::RuntimeError("OmniVoice model mutex was poisoned".to_string())
                })?
                .forward(
                    &cond_input_ids_tensor,
                    &cond_audio_mask_u32,
                    &cond_audio_mask_f32,
                    cond_attention_mask.as_ref(),
                )?
                .to_dtype(DType::F32)?
                .i((0..1, .., cond_len - target_len..cond_len, ..))?;
            let u_logits =
                if gen_config.guidance_scale != 0.0 {
                    let uncond_input_ids_tensor =
                        Tensor::new(uncond_input_ids.as_slice(), &self.compute_device)?
                            .reshape((1, self.config.num_audio_codebook, target_len))?;
                    let logits = self
                        .backbone
                        .lock()
                        .map_err(|_| {
                            TtsError::RuntimeError("OmniVoice model mutex was poisoned".to_string())
                        })?
                        .forward(
                            &uncond_input_ids_tensor,
                            &uncond_audio_mask_u32,
                            &uncond_audio_mask_f32,
                            uncond_attention_mask.as_ref(),
                        )?
                        .to_dtype(DType::F32)?;
                    Some(logits)
                } else {
                    None
                };
            let (pred_tokens_flat, mut scores_flat) = self.predict_tokens_with_scoring(
                &cond_logits,
                u_logits.as_ref(),
                gen_config,
                &mut rng,
            )?;
            maybe_dump_debug_first_step(
                &cond_logits,
                u_logits.as_ref(),
                &pred_tokens_flat,
                &scores_flat,
            )?;
            for (layer_scores, penalty) in scores_flat
                .chunks_exact_mut(target_len)
                .zip(layer_penalties.iter().copied())
            {
                for score in layer_scores {
                    *score -= penalty;
                }
            }

            let chosen_positions = select_positions(
                &scores_flat,
                &tokens,
                mask_id,
                step_k,
                gen_config.position_temperature,
                &mut rng,
            );
            maybe_record_step_trace(
                &mut step_trace,
                step_index,
                step_k,
                &chosen_positions,
                &pred_tokens_flat,
                target_len,
            );
            for flat_index in chosen_positions {
                tokens[flat_index] = pred_tokens_flat[flat_index];
                let layer = flat_index / target_len;
                let pos = flat_index % target_len;
                cond_input_ids[index_2d(layer, cond_len - target_len + pos, cond_len)] =
                    pred_tokens_flat[flat_index];
                uncond_input_ids[index_2d(layer, pos, target_len)] = pred_tokens_flat[flat_index];
            }
        }

        maybe_dump_step_trace(&step_trace)?;
        maybe_dump_debug_tokens(&tokens, self.config.num_audio_codebook, target_len)?;

        Ok(GeneratedAudioTokens {
            tokens,
            len: target_len,
        })
    }

    fn decode_token_chunks(
        &self,
        token_chunks: &[GeneratedAudioTokens],
    ) -> Result<Vec<f32>, TtsError> {
        if token_chunks.is_empty() {
            return Ok(Vec::new());
        }

        let mut audio_chunks = Vec::with_capacity(token_chunks.len());
        for chunk in token_chunks {
            let token_tensor = Tensor::new(chunk.tokens.as_slice(), &self.audio_device)?
                .reshape((1, self.config.num_audio_codebook, chunk.len))?;
            let waveform = self.audio_tokenizer.decode(&token_tensor)?;
            let waveform = waveform.squeeze(1)?.squeeze(0)?.to_device(&Device::Cpu)?;
            audio_chunks.push(waveform.to_vec1::<f32>()?);
        }

        Ok(cross_fade_chunks(
            &audio_chunks,
            self.sample_rate() as usize,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_inference_inputs(
        &self,
        text: &str,
        target_len: usize,
        language: Option<&str>,
        instruct: Option<&str>,
        denoise: bool,
        ref_text: Option<&str>,
        ref_audio_tokens: Option<&GeneratedAudioTokens>,
    ) -> Result<InferenceInputs, TtsError> {
        let mut style_text = String::new();
        if denoise && ref_audio_tokens.is_some() {
            style_text.push_str("<|denoise|>");
        }
        style_text.push_str("<|lang_start|>");
        style_text.push_str(language.unwrap_or("None"));
        style_text.push_str("<|lang_end|>");
        style_text.push_str("<|instruct_start|>");
        style_text.push_str(instruct.unwrap_or("None"));
        style_text.push_str("<|instruct_end|>");

        let text_prompt = format!("<|text_start|>{}<|text_end|>", combine_text(text, ref_text));
        let style_ids = self.tokenizer.encode(&style_text)?;
        let text_ids = self.tokenizer.encode(&text_prompt)?;
        let style_len = style_ids.len();
        let text_len = text_ids.len();
        let ref_len = ref_audio_tokens.map(|tokens| tokens.len).unwrap_or(0);
        let seq_len = style_len + text_len + ref_len + target_len;

        let mut input_ids =
            vec![self.config.audio_mask_id; self.config.num_audio_codebook * seq_len];
        for layer in 0..self.config.num_audio_codebook {
            for (index, token_id) in style_ids.iter().copied().enumerate() {
                input_ids[index_2d(layer, index, seq_len)] = token_id;
            }
            for (index, token_id) in text_ids.iter().copied().enumerate() {
                input_ids[index_2d(layer, style_len + index, seq_len)] = token_id;
            }
        }

        if let Some(ref_tokens) = ref_audio_tokens {
            let expected_len = self.config.num_audio_codebook * ref_tokens.len;
            if ref_tokens.tokens.len() != expected_len {
                return Err(TtsError::RuntimeError(format!(
                    "OmniVoice reference token length mismatch: expected {}, got {}",
                    expected_len,
                    ref_tokens.tokens.len()
                )));
            }
            let ref_start = style_len + text_len;
            for layer in 0..self.config.num_audio_codebook {
                for pos in 0..ref_tokens.len {
                    input_ids[index_2d(layer, ref_start + pos, seq_len)] =
                        ref_tokens.tokens[index_2d(layer, pos, ref_tokens.len)];
                }
            }
        }

        let mut audio_mask = vec![0u8; seq_len];
        let audio_start = seq_len - target_len - ref_len;
        for audio_flag in audio_mask.iter_mut().skip(audio_start) {
            *audio_flag = 1;
        }

        Ok(InferenceInputs {
            input_ids,
            audio_mask,
            seq_len,
            target_len,
        })
    }

    fn predict_tokens_with_scoring(
        &self,
        c_logits: &Tensor,
        u_logits: Option<&Tensor>,
        gen_config: &OmniVoiceGenerationConfig,
        rng: &mut SimpleRng,
    ) -> Result<(Vec<u32>, Vec<f32>), TtsError> {
        let mut log_probs = guided_log_probs(c_logits, u_logits, gen_config.guidance_scale)?;

        log_probs = log_probs.broadcast_add(&self.mask_token_penalty)?;

        if gen_config.class_temperature > 0.0 {
            let flat = log_probs.flatten_all()?.to_vec1::<f32>()?;
            let (_batch, codebooks, seq_len, vocab_size) = log_probs.dims4()?;
            let top_k = ((vocab_size as f32 * 0.1).ceil() as usize).max(1);
            let mut pred_tokens = vec![0u32; codebooks * seq_len];
            let mut scores = vec![f32::NEG_INFINITY; codebooks * seq_len];

            for layer in 0..codebooks {
                for pos in 0..seq_len {
                    let start = ((layer * seq_len) + pos) * vocab_size;
                    let mut candidates: Vec<(f32, usize)> = flat[start..start + vocab_size]
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(token, score)| (score, token))
                        .collect();
                    retain_best(&mut candidates, top_k);
                    let mut best = (f32::NEG_INFINITY, 0usize);
                    for (score, token) in candidates.iter().copied() {
                        let noisy = score / gen_config.class_temperature + rng.next_gumbel();
                        if noisy > best.0 {
                            best = (noisy, token);
                        }
                    }
                    pred_tokens[index_2d(layer, pos, seq_len)] = best.1 as u32;
                    scores[index_2d(layer, pos, seq_len)] = candidates[0].0;
                }
            }

            Ok((pred_tokens, scores))
        } else {
            let pred_tokens = log_probs.argmax(candle_core::D::Minus1)?;
            let scores = log_probs.max(candle_core::D::Minus1)?;
            Ok((
                pred_tokens.flatten_all()?.to_vec1::<u32>()?,
                scores.flatten_all()?.to_vec1::<f32>()?,
            ))
        }
    }
}

fn guided_log_probs(
    conditional: &Tensor,
    unconditional: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor, TtsError> {
    let c = log_softmax_last_dim(conditional)?;
    if scale == 0.0 {
        return Ok(c);
    }
    let unconditional = unconditional.ok_or_else(|| {
        TtsError::ModelError("OmniVoice guidance requires unconditional logits".into())
    })?;
    let u = log_softmax_last_dim(unconditional)?;
    let guidance = c.broadcast_sub(&u)?;
    log_softmax_last_dim(&c.broadcast_add(&(guidance * scale)?)?)
}

fn retain_best(candidates: &mut Vec<(f32, usize)>, count: usize) {
    let compare = |left: &(f32, usize), right: &(f32, usize)| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    };
    if count == 0 {
        candidates.clear();
        return;
    }
    if candidates.len() > count {
        candidates.select_nth_unstable_by(count - 1, compare);
        candidates.truncate(count);
    }
    candidates.sort_unstable_by(compare);
}

fn build_schedule(total_masked: usize, num_steps: usize, t_shift: f64) -> Vec<usize> {
    let timesteps = get_time_steps(num_steps, t_shift);
    let mut remaining = total_masked;
    let mut schedule = Vec::with_capacity(num_steps);
    for step in 0..num_steps {
        let count = if step == num_steps - 1 {
            remaining
        } else {
            let delta = (timesteps[step + 1] - timesteps[step]).max(0.0);
            let count = ((total_masked as f64) * delta).ceil() as usize;
            count.min(remaining)
        };
        schedule.push(count);
        remaining = remaining.saturating_sub(count);
    }
    schedule
}

/// Evenly spaced times, warped towards the start. Calculated in F32 as
/// upstream does, since rounding here moves tokens between steps.
fn get_time_steps(num_step: usize, t_shift: f64) -> Vec<f64> {
    let step = 1.0f32 / num_step.max(1) as f32;
    let shift = t_shift as f32;
    let shift_minus_one = (t_shift - 1.0) as f32;
    (0..=num_step)
        .map(|index| {
            let t = if index < num_step.div_ceil(2) {
                step * index as f32
            } else {
                1.0 - step * (num_step - index) as f32
            };
            f64::from(shift * t / (1.0 + shift_minus_one * t))
        })
        .collect()
}

fn build_causal_attention_mask(
    seq_len: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor, TtsError> {
    let mut values = vec![0.0f32; seq_len * seq_len];
    for query_index in 0..seq_len {
        for key_index in query_index + 1..seq_len {
            values[index_2d(query_index, key_index, seq_len)] = -1e4;
        }
    }
    Ok(Tensor::new(values.as_slice(), device)?
        .reshape((1, 1, seq_len, seq_len))?
        .to_dtype(dtype)?)
}

fn select_positions(
    scores: &[f32],
    current_tokens: &[u32],
    mask_id: u32,
    count: usize,
    temperature: f32,
    rng: &mut SimpleRng,
) -> Vec<usize> {
    let mut candidates = Vec::new();
    for (index, score) in scores.iter().copied().enumerate() {
        if !score.is_finite() {
            continue;
        }
        let value = if temperature > 0.0 {
            score / temperature + rng.next_gumbel()
        } else {
            score
        };
        if current_tokens[index] != mask_id {
            continue;
        }
        candidates.push((value, index));
    }
    retain_best(&mut candidates, count);
    candidates
        .into_iter()
        .take(count)
        .map(|(_, index)| index)
        .collect()
}

fn omnivoice_seed() -> u64 {
    std::env::var("OMNIVOICE_SEED")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        })
}

fn env_override_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_override_dtype(name: &str) -> Option<DType> {
    let value = std::env::var(name).ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "f32" => Some(DType::F32),
        "f16" => Some(DType::F16),
        "bf16" => Some(DType::BF16),
        _ => None,
    }
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        // Consecutive seeds would otherwise start out nearly identical.
        let mixed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Self {
            state: (mixed ^ (mixed >> 31)).max(1),
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    fn next_gumbel(&mut self) -> f32 {
        let unit = self.next_f32().clamp(0.0, 1.0 - 1e-7);
        let inner = -(unit + 1e-10).ln() + 1e-10;
        -inner.ln()
    }
}

fn log_softmax_last_dim(logits: &Tensor) -> Result<Tensor, TtsError> {
    Ok(candle_nn::ops::log_softmax(
        &logits.contiguous()?,
        candle_core::D::Minus1,
    )?)
}

fn maybe_dump_debug_tokens(
    tokens: &[u32],
    num_audio_codebook: usize,
    target_len: usize,
) -> Result<(), TtsError> {
    let Some(path) = std::env::var_os("OMNIVOICE_DEBUG_TOKEN_DUMP") else {
        return Ok(());
    };

    let layers: Vec<Vec<u32>> = tokens
        .chunks(target_len)
        .take(num_audio_codebook)
        .map(|layer| layer.to_vec())
        .collect();

    let payload = serde_json::json!({
        "num_audio_codebook": num_audio_codebook,
        "target_len": target_len,
        "layers": layers,
    });

    std::fs::write(&path, serde_json::to_vec_pretty(&payload)?)?;
    Ok(())
}

fn maybe_dump_debug_first_step(
    cond_logits: &Tensor,
    uncond_logits: Option<&Tensor>,
    pred_tokens: &[u32],
    scores: &[f32],
) -> Result<(), TtsError> {
    let Some(path) = std::env::var_os("OMNIVOICE_DEBUG_FIRST_STEP_DUMP") else {
        return Ok(());
    };

    let payload = build_first_step_debug_payload(cond_logits, uncond_logits, pred_tokens, scores)?;
    std::fs::write(&path, serde_json::to_vec_pretty(&payload)?)?;
    std::env::remove_var("OMNIVOICE_DEBUG_FIRST_STEP_DUMP");
    Ok(())
}

fn maybe_record_step_trace(
    step_trace: &mut Vec<serde_json::Value>,
    step_index: usize,
    step_k: usize,
    chosen_positions: &[usize],
    pred_tokens_flat: &[u32],
    target_len: usize,
) {
    if std::env::var_os("OMNIVOICE_DEBUG_STEP_TRACE").is_none() {
        return;
    }

    let chosen: Vec<serde_json::Value> = chosen_positions
        .iter()
        .copied()
        .map(|flat_index| {
            serde_json::json!({
                "flat_index": flat_index,
                "layer": flat_index / target_len,
                "position": flat_index % target_len,
                "token": pred_tokens_flat[flat_index],
            })
        })
        .collect();
    step_trace.push(serde_json::json!({
        "step_index": step_index,
        "count": step_k,
        "chosen": chosen,
    }));
}

fn maybe_dump_step_trace(step_trace: &[serde_json::Value]) -> Result<(), TtsError> {
    let Some(path) = std::env::var_os("OMNIVOICE_DEBUG_STEP_TRACE") else {
        return Ok(());
    };

    let payload = serde_json::json!({
        "steps": step_trace,
    });
    std::fs::write(&path, serde_json::to_vec_pretty(&payload)?)?;
    Ok(())
}

fn build_first_step_debug_payload(
    cond_logits: &Tensor,
    uncond_logits: Option<&Tensor>,
    pred_tokens: &[u32],
    scores: &[f32],
) -> Result<serde_json::Value, TtsError> {
    let (_batch, codebooks, seq_len, vocab_size) = cond_logits.dims4()?;
    let cond_flat = cond_logits.flatten_all()?.to_vec1::<f32>()?;
    let uncond_flat = uncond_logits
        .map(|x| x.flatten_all()?.to_vec1::<f32>())
        .transpose()?;
    let pred_tokens_flat = pred_tokens;
    let scores_flat = scores;
    let position_count = seq_len.min(4);

    let mut layers = Vec::with_capacity(codebooks);
    for layer in 0..codebooks {
        let mut positions = Vec::with_capacity(position_count);
        for pos in 0..position_count {
            let start = ((layer * seq_len) + pos) * vocab_size;
            let end = start + vocab_size;
            positions.push(serde_json::json!({
                "position": pos,
                "cond_top": top_k_summary(&cond_flat[start..end], 10),
                "uncond_top": uncond_flat.as_ref().map(|values| top_k_summary(&values[start..end], 10)),
                "pred_token": pred_tokens_flat[index_2d(layer, pos, seq_len)],
                "score": scores_flat[index_2d(layer, pos, seq_len)],
            }));
        }
        layers.push(serde_json::json!({
            "layer": layer,
            "positions": positions,
        }));
    }

    Ok(serde_json::json!({
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "layers": layers,
    }))
}

fn top_k_summary(values: &[f32], count: usize) -> Vec<serde_json::Value> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|left, right| right.1.total_cmp(&left.1));
    indexed
        .into_iter()
        .take(count)
        .map(|(token, logit)| {
            serde_json::json!({
                "token": token,
                "logit": logit,
            })
        })
        .collect()
}

fn estimate_target_tokens_with_reference(
    text: &str,
    ref_text: Option<&str>,
    ref_audio_tokens: Option<usize>,
    speed: f64,
) -> usize {
    const LOW_THRESHOLD: f64 = 50.0;
    const BOOST_STRENGTH: f64 = 3.0;
    let (ref_text, ref_tokens) = match (ref_text, ref_audio_tokens) {
        (Some(text), Some(tokens)) if !text.is_empty() => (text, tokens),
        _ => ("Nice to meet you.", 25),
    };
    let ref_weight = text_weight(ref_text);
    if ref_tokens == 0 || ref_weight == 0.0 {
        return 1;
    }
    let mut estimate = text_weight(text) / (ref_weight / ref_tokens as f64);
    if estimate < LOW_THRESHOLD {
        estimate = LOW_THRESHOLD * (estimate / LOW_THRESHOLD).powf(1.0 / BOOST_STRENGTH);
    }
    if speed != 1.0 {
        estimate /= speed;
    }
    (estimate as usize).max(1)
}

fn text_weight(text: &str) -> f64 {
    text.chars().map(char_weight).sum()
}

/// Relative speaking time of a character, where a Latin letter is 1.0.
fn char_weight(ch: char) -> f64 {
    const CJK: f64 = 3.0;
    const HANGUL: f64 = 2.5;
    const KANA: f64 = 2.2;
    const SYLLABIC: f64 = 3.0;
    const INDIC: f64 = 1.8;
    const THAI_LAO: f64 = 1.5;
    const KHMER_MYANMAR: f64 = 1.8;
    const ABJAD: f64 = 1.5;
    const ALPHABET: f64 = 1.0;
    const PUNCTUATION: f64 = 0.5;
    const SPACE: f64 = 0.2;
    const DIGIT: f64 = 3.5;
    const MARK: f64 = 0.0;
    // Each entry covers code points up to its limit, above the previous entry.
    const RANGES: &[(u32, f64)] = &[
        (0x058F, ALPHABET),
        (0x08FF, ABJAD),
        (0x0DFF, INDIC),
        (0x0EFF, THAI_LAO),
        (0x0FFF, INDIC),
        (0x109F, KHMER_MYANMAR),
        (0x10FF, ALPHABET),
        (0x11FF, HANGUL),
        (0x139F, SYLLABIC),
        (0x177F, ALPHABET),
        (0x17FF, KHMER_MYANMAR),
        (0x18FF, ALPHABET),
        (0x19DF, INDIC),
        (0x19FF, KHMER_MYANMAR),
        (0x1C7F, INDIC),
        (0x1CBF, ALPHABET),
        (0x1CFF, INDIC),
        (0x1EFF, ALPHABET),
        (0x30FF, KANA),
        (0x312F, CJK),
        (0x318F, HANGUL),
        (0x9FFF, CJK),
        (0xA4CF, SYLLABIC),
        (0xA7FF, ALPHABET),
        (0xA82F, INDIC),
        (0xA87F, ALPHABET),
        (0xA95F, INDIC),
        (0xA97F, HANGUL),
        (0xA9DF, INDIC),
        (0xA9FF, KHMER_MYANMAR),
        (0xAA5F, INDIC),
        (0xAA7F, KHMER_MYANMAR),
        (0xAAFF, INDIC),
        (0xAB2F, SYLLABIC),
        (0xABBF, ALPHABET),
        (0xABFF, INDIC),
        (0xD7AF, HANGUL),
        (0xFAFF, CJK),
        (0xFDFF, ABJAD),
        (0xFE6F, ALPHABET),
        (0xFEFF, ABJAD),
        (0xFFEF, ALPHABET),
    ];
    use GeneralCategory as Category;
    let code = ch as u32;
    if ch.is_ascii_alphabetic() {
        return ALPHABET;
    }
    if code == 0x20 {
        return SPACE;
    }
    if code == 0x0640 {
        return MARK;
    }
    match get_general_category(ch) {
        Category::NonspacingMark | Category::SpacingMark | Category::EnclosingMark => MARK,
        Category::ConnectorPunctuation
        | Category::DashPunctuation
        | Category::OpenPunctuation
        | Category::ClosePunctuation
        | Category::InitialPunctuation
        | Category::FinalPunctuation
        | Category::OtherPunctuation
        | Category::MathSymbol
        | Category::CurrencySymbol
        | Category::ModifierSymbol
        | Category::OtherSymbol => PUNCTUATION,
        Category::SpaceSeparator | Category::LineSeparator | Category::ParagraphSeparator => SPACE,
        Category::DecimalNumber | Category::LetterNumber | Category::OtherNumber => DIGIT,
        _ => {
            let index = RANGES.partition_point(|(limit, _)| *limit < code);
            match RANGES.get(index) {
                Some((_, weight)) => *weight,
                None if code > 0x20000 => CJK,
                None => ALPHABET,
            }
        }
    }
}

fn chunk_text_for_target(
    text: &str,
    target_len: usize,
    frame_rate: usize,
    chunk_duration: f32,
) -> Vec<String> {
    let char_count = text.chars().count().max(1);
    let avg_tokens_per_char = target_len as f32 / char_count as f32;
    let chunk_len = (chunk_duration.max(0.1) * frame_rate.max(1) as f32 / avg_tokens_per_char)
        .max(1.0)
        .floor() as usize;
    chunk_text_punctuation(text, chunk_len, Some(3))
}

fn chunk_text_punctuation(
    text: &str,
    chunk_len: usize,
    min_chunk_len: Option<usize>,
) -> Vec<String> {
    let chunk_len = chunk_len.max(1);
    let mut sentences: Vec<Vec<char>> = Vec::new();
    let mut current_sentence: Vec<char> = Vec::new();

    for token in text.chars() {
        if current_sentence.is_empty()
            && !sentences.is_empty()
            && (is_split_punctuation(token) || is_closing_mark(token))
        {
            sentences.last_mut().expect("checked non-empty").push(token);
        } else {
            current_sentence.push(token);
            if is_split_punctuation(token)
                && !(token == '.' && sentence_ends_with_abbreviation(&current_sentence))
            {
                sentences.push(std::mem::take(&mut current_sentence));
            }
        }
    }
    if !current_sentence.is_empty() {
        sentences.push(current_sentence);
    }

    let mut merged_chunks: Vec<Vec<char>> = Vec::new();
    let mut current_chunk: Vec<char> = Vec::new();
    for sentence in sentences {
        if current_chunk.len() + sentence.len() <= chunk_len {
            current_chunk.extend(sentence);
        } else {
            if !current_chunk.is_empty() {
                merged_chunks.push(current_chunk);
            }
            current_chunk = sentence;
        }
    }
    if !current_chunk.is_empty() {
        merged_chunks.push(current_chunk);
    }

    let final_chunks = if let Some(min_chunk_len) = min_chunk_len {
        let first_chunk_short = merged_chunks
            .first()
            .is_some_and(|chunk| chunk.len() < min_chunk_len);
        let mut final_chunks: Vec<Vec<char>> = Vec::new();
        for (index, chunk) in merged_chunks.into_iter().enumerate() {
            if index == 1 && first_chunk_short {
                if let Some(previous) = final_chunks.last_mut() {
                    previous.extend(chunk);
                } else {
                    final_chunks.push(chunk);
                }
            } else if chunk.len() >= min_chunk_len || final_chunks.is_empty() {
                final_chunks.push(chunk);
            } else if let Some(previous) = final_chunks.last_mut() {
                previous.extend(chunk);
            }
        }
        final_chunks
    } else {
        merged_chunks
    };

    final_chunks
        .into_iter()
        .map(|chunk| chunk.into_iter().collect::<String>().trim().to_string())
        .filter(|chunk| !chunk.is_empty())
        .collect()
}

fn is_split_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '.' | ',' | ';' | ':' | '!' | '?' | '。' | '，' | '；' | '：' | '！' | '？'
    )
}

fn is_closing_mark(ch: char) -> bool {
    matches!(
        ch,
        '"' | '\'' | ')' | ']' | '>' | '）' | '》' | '」' | '】' | '”' | '’'
    )
}

fn sentence_ends_with_abbreviation(sentence: &[char]) -> bool {
    let text = sentence.iter().collect::<String>();
    let Some(last_word) = text.split_whitespace().last() else {
        return false;
    };
    is_abbreviation(last_word)
}

fn is_abbreviation(word: &str) -> bool {
    matches!(
        word,
        "Mr."
            | "Mrs."
            | "Ms."
            | "Dr."
            | "Prof."
            | "Sr."
            | "Jr."
            | "Rev."
            | "Fr."
            | "Hon."
            | "Pres."
            | "Gov."
            | "Capt."
            | "Gen."
            | "Sen."
            | "Rep."
            | "Col."
            | "Maj."
            | "Lt."
            | "Cmdr."
            | "Sgt."
            | "Cpl."
            | "Co."
            | "Corp."
            | "Inc."
            | "Ltd."
            | "Est."
            | "Dept."
            | "St."
            | "Ave."
            | "Blvd."
            | "Rd."
            | "Mt."
            | "Ft."
            | "No."
            | "Jan."
            | "Feb."
            | "Mar."
            | "Apr."
            | "Aug."
            | "Sep."
            | "Sept."
            | "Oct."
            | "Nov."
            | "Dec."
            | "i.e."
            | "e.g."
            | "vs."
            | "Vs."
            | "Etc."
            | "approx."
            | "fig."
            | "def."
    )
}

fn combine_text(text: &str, ref_text: Option<&str>) -> String {
    let mut full_text = if let Some(ref_text) = ref_text.filter(|text| !text.trim().is_empty()) {
        format!("{} {}", ref_text.trim(), text.trim())
    } else {
        text.trim().to_string()
    };

    full_text = full_text
        .chars()
        .filter(|ch| *ch != '\n' && *ch != '\r')
        .collect();
    full_text = full_text.replace('（', "(").replace('）', ")");

    let mut collapsed = String::with_capacity(full_text.len());
    let mut previous_was_space = false;
    for ch in full_text.chars() {
        if ch == ' ' || ch == '\t' {
            if !previous_was_space {
                collapsed.push(' ');
            }
            previous_was_space = true;
        } else {
            collapsed.push(ch);
            previous_was_space = false;
        }
    }

    let chars: Vec<char> = collapsed.chars().collect();
    let mut normalized = String::with_capacity(collapsed.len());
    for (index, ch) in chars.iter().copied().enumerate() {
        if ch.is_whitespace() {
            let prev_is_cjk = index > 0 && is_cjk(chars[index - 1]);
            let next_is_cjk = chars.get(index + 1).copied().is_some_and(is_cjk);
            if prev_is_cjk || next_is_cjk {
                continue;
            }
        }
        normalized.push(ch);
    }

    normalized
}

fn normalize_instruct(instruct: Option<&str>, text: &str) -> Option<String> {
    let instruct = instruct?.trim();
    if instruct.is_empty() {
        return None;
    }
    let prefer_zh = text.chars().any(is_cjk) || instruct.chars().any(is_cjk);
    let items: Vec<String> = instruct
        .split([',', '，'])
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| {
            if item.chars().any(is_cjk) {
                item.to_string()
            } else {
                item.to_ascii_lowercase()
            }
        })
        .collect();
    if items.is_empty() {
        None
    } else if prefer_zh {
        Some(items.join("，"))
    } else {
        Some(items.join(", "))
    }
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF)
}

fn cross_fade_chunks(chunks: &[Vec<f32>], sample_rate: usize) -> Vec<f32> {
    if chunks.len() <= 1 {
        return chunks.first().cloned().unwrap_or_default();
    }

    let total_n = (0.3 * sample_rate as f32) as usize;
    let fade_n = total_n / 3;
    let silence_n = fade_n;
    let mut merged = chunks[0].clone();

    for chunk in chunks.iter().skip(1) {
        let fout_n = fade_n.min(merged.len());
        if fout_n > 0 {
            let start = merged.len() - fout_n;
            for index in 0..fout_n {
                merged[start + index] *= linspace_weight(1.0, 0.0, index, fout_n);
            }
        }

        merged.extend(std::iter::repeat_n(0.0, silence_n));

        let mut fade_in = chunk.clone();
        let fin_n = fade_n.min(fade_in.len());
        if fin_n > 0 {
            for (index, sample) in fade_in.iter_mut().take(fin_n).enumerate() {
                *sample *= linspace_weight(0.0, 1.0, index, fin_n);
            }
        }
        merged.extend(fade_in);
    }

    merged
}

fn linspace_weight(start: f32, end: f32, index: usize, count: usize) -> f32 {
    if count <= 1 {
        start
    } else {
        start + (end - start) * index as f32 / (count - 1) as f32
    }
}

/// Removes long pauses, sets the level, and fades and pads the edges. With a
/// reference, quiet references give matching quiet output, as upstream does.
fn post_process_audio(samples: &[f32], sample_rate: u32, ref_rms: Option<f32>) -> Vec<f32> {
    let mut samples = remove_silence(samples, sample_rate, 500, 100, 100);
    match ref_rms {
        Some(rms) if rms < REFERENCE_MIN_RMS => {
            for sample in samples.iter_mut() {
                *sample = *sample * rms / REFERENCE_MIN_RMS;
            }
        }
        Some(_) => {}
        None => {
            let peak = samples.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
            if peak > 1e-6 {
                for sample in samples.iter_mut() {
                    *sample = *sample / peak * 0.5;
                }
            }
        }
    }
    if samples.is_empty() {
        return samples;
    }

    let fade = (sample_rate as usize / 10).min(samples.len() / 2);
    for index in 0..fade {
        samples[index] *= linspace_weight(0.0, 1.0, index, fade);
        let tail = samples.len() - fade + index;
        samples[tail] *= linspace_weight(1.0, 0.0, index, fade);
    }

    let pad = sample_rate as usize / 10;
    let mut padded = Vec::with_capacity(samples.len() + pad * 2);
    padded.extend(std::iter::repeat_n(0.0, pad));
    padded.extend(samples);
    padded.extend(std::iter::repeat_n(0.0, pad));
    padded
}

/// Scales a quiet reference up and shortens its pauses, returning the samples
/// to encode and the original level.
fn prepare_reference_samples(
    samples: &[f32],
    sample_rate: u32,
    hop_length: usize,
) -> Result<(Vec<f32>, f32), TtsError> {
    let power: f64 = samples
        .iter()
        .map(|sample| f64::from(*sample).powi(2))
        .sum();
    let rms = (power / samples.len().max(1) as f64).sqrt() as f32;
    let scaled: Vec<f32> = if rms > 0.0 && rms < REFERENCE_MIN_RMS {
        samples
            .iter()
            .map(|sample| *sample * REFERENCE_MIN_RMS / rms)
            .collect()
    } else {
        samples.to_vec()
    };
    let mut prepared = remove_silence(&scaled, sample_rate, 200, 100, 200);
    prepared.truncate(prepared.len() / hop_length * hop_length);
    if prepared.is_empty() {
        return Err(TtsError::ConfigError(
            "The reference audio is empty once silence is removed".to_string(),
        ));
    }
    Ok((prepared, rms))
}

fn add_final_punctuation(text: &str) -> String {
    let text = text.trim();
    const ENDINGS: &str = "!\"'),.:;?]}‘’“”………、。】！），：；？";
    match text.chars().last() {
        Some(last) if !ENDINGS.contains(last) => {
            let stop = if text.chars().any(is_cjk) { '。' } else { '.' };
            format!("{text}{stop}")
        }
        _ => text.to_string(),
    }
}

fn index_2d(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_divides_the_estimated_length() {
        // The reference's own text takes the reference's own length at normal speed.
        let text = "A sentence long enough that the short-text boost does not apply to it.";
        let estimate =
            |speed| estimate_target_tokens_with_reference(text, Some(text), Some(200), speed);
        assert_eq!(estimate(1.0), 200);
        assert_eq!(estimate(2.0), 100);
        assert_eq!(estimate(0.5), 400);
        assert_eq!(estimate(1.6), 125);
    }

    #[test]
    fn unguided_scoring_needs_no_unconditional_pass() -> Result<(), TtsError> {
        let conditional = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
        let actual = guided_log_probs(&conditional, None, 0.)?;
        assert_eq!(
            actual.to_vec1::<f32>()?,
            log_softmax_last_dim(&conditional)?.to_vec1::<f32>()?
        );
        assert!(guided_log_probs(&conditional, None, 2.).is_err());
        Ok(())
    }

    #[test]
    fn schedule_matches_the_python_package() {
        // Values from omnivoice.models.omnivoice with num_step=32 and t_shift=0.1.
        let short: Vec<usize> = vec![
            3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8, 8, 9, 11, 12, 14, 16, 18, 22, 26,
            32, 41, 53, 72, 103, 143,
        ];
        let long: Vec<usize> = vec![
            8, 9, 9, 10, 10, 11, 12, 13, 13, 14, 16, 17, 18, 20, 22, 24, 27, 30, 33, 37, 42, 49,
            56, 66, 78, 95, 116, 147, 192, 261, 375, 570,
        ];
        assert_eq!(
            build_schedule(82 * 8, DEFAULT_NUM_STEPS, DEFAULT_T_SHIFT),
            short
        );
        assert_eq!(
            build_schedule(300 * 8, DEFAULT_NUM_STEPS, DEFAULT_T_SHIFT),
            long
        );
    }

    #[test]
    fn partial_selection_preserves_sorted_top_values_and_ties() {
        let mut candidates = vec![(2., 3), (5., 8), (5., 1), (0., 4), (3., 9)];
        retain_best(&mut candidates, 3);
        assert_eq!(candidates, vec![(5., 1), (5., 8), (3., 9)]);
        retain_best(&mut candidates, 0);
        assert!(candidates.is_empty());
    }

    #[test]
    fn length_estimate_matches_the_python_package() {
        // Values from OmniVoice._estimate_target_tokens with its default reference.
        for (text, expected) in [
            ("Hi.", 22),
            ("Hello world, this is a test of the estimate.", 64),
            ("今天天气很好。", 43),
            ("Ünïcode — “quoted” 1865!", 52),
        ] {
            let actual = estimate_target_tokens_with_reference(text, None, None, 1.0);
            assert_eq!(actual, expected, "estimate for {text}");
        }
        // Values from RuleDurationEstimator.calculate_total_weight.
        for (ch, expected) in [
            ('a', 1.0),
            (' ', 0.2),
            ('\u{a0}', 0.2),
            ('.', 0.5),
            ('—', 0.5),
            ('€', 0.5),
            ('1', 3.5),
            ('Ⅷ', 3.5),
            ('\u{301}', 0.0),
            ('今', 3.0),
            ('あ', 2.2),
            ('한', 2.5),
            ('ب', 1.5),
            ('क', 1.8),
            ('ก', 1.5),
        ] {
            let weight = f64::from(char_weight(ch));
            assert!(
                (weight - expected).abs() < 1e-6,
                "weight of {ch:?}: {weight}"
            );
        }
    }

    #[test]
    fn combine_text_matches_upstream_cleanup() {
        assert_eq!(
            combine_text("  Hello\nworld\t again  ", Some("Reference.")),
            "Reference. Helloworld again"
        );
        assert_eq!(combine_text("你 好 （test）", None), "你好(test)");
    }

    #[test]
    fn post_processing_fades_out_to_silence() {
        let out = post_process_audio(&vec![1.0f32; 24_000], 24_000, None);
        let (pad, fade) = (2400, 2400);
        assert_eq!(out.len(), 24_000 + 2 * pad);
        assert_eq!(out[pad], 0.0);
        // The fade-out starts at full level and ends at silence.
        assert!((out[pad + 24_000 - fade] - 0.5).abs() < 1e-3);
        assert_eq!(out[pad + 24_000 - 1], 0.0);
    }

    #[test]
    fn chunk_text_punctuation_keeps_abbreviations() {
        let chunks = chunk_text_punctuation("Dr. Smith arrived. He left.", 18, Some(3));
        assert_eq!(chunks, vec!["Dr. Smith arrived.", "He left."]);
    }

    #[test]
    fn nearby_seeds_give_unrelated_draws() {
        assert_ne!(SimpleRng::new(0).next_u32(), SimpleRng::new(1).next_u32());
        let first = SimpleRng::new(1000).next_f32();
        let second = SimpleRng::new(1001).next_f32();
        assert!((first - second).abs() > 1e-3, "{first} and {second}");
    }

    #[test]
    fn cross_fade_chunks_adds_official_gap() {
        let merged = cross_fade_chunks(&[vec![1.0; 10], vec![1.0; 10]], 30);
        assert_eq!(merged.len(), 23);
        assert_eq!(&merged[10..13], &[0.0, 0.0, 0.0]);
    }

    /// Compares every stage with values dumped by scripts/omnivoice_parity.py.
    /// Run with OMNIVOICE_ORACLE set to the dump, and with the model downloaded.
    #[test]
    #[ignore = "Requires OmniVoice model weights, a GPU and an oracle dump"]
    fn matches_python_oracle() -> Result<(), TtsError> {
        let path = std::env::var("OMNIVOICE_ORACLE").expect("set OMNIVOICE_ORACLE");
        let oracle = candle_core::safetensors::load(&path, &Device::Cpu)?;
        let bytes = std::fs::read(&path).unwrap();
        let (_, header) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
        let meta = header.metadata().as_ref().unwrap();
        let floats = |name: &str| oracle[name].flatten_all()?.to_vec1::<f32>();
        let ints = |name: &str| oracle[name].flatten_all()?.to_vec1::<i32>();
        let max_diff = |a: &[f32], b: &[f32]| {
            assert_eq!(a.len(), b.len());
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };

        // Stages without the model.
        let hop = 960;
        let loaded = floats("loaded")?;
        let rms: f64 = meta["ref_rms"].parse().unwrap();
        let (prepared, actual_rms) = prepare_reference_samples(&loaded, 24_000, hop)?;
        let expected_prepared = floats("silence_removed")?;
        println!("rms {actual_rms} vs {rms}");
        assert!((f64::from(actual_rms) - rms).abs() < 1e-5);
        println!(
            "prepared {} vs {} samples, max diff {}",
            prepared.len(),
            expected_prepared.len(),
            max_diff(&prepared, &expected_prepared[..prepared.len()])
        );
        assert_eq!(prepared.len(), expected_prepared.len() / hop * hop);
        assert!(max_diff(&prepared, &expected_prepared[..prepared.len()]) < 1e-4);
        let resampled = super::super::audio_processing::resample(&prepared, 24_000, 16_000);
        let diff = max_diff(&resampled, &floats("resampled")?);
        println!("resampled max diff {diff}");
        assert!(diff < 1e-4);

        let weights = floats("char_weights")?;
        let mut wrong: Vec<u32> = Vec::new();
        for code in 0..0x10000u32 {
            if let Some(ch) = char::from_u32(code) {
                if (char_weight(ch) as f32 - weights[code as usize]).abs() > 1e-5 {
                    wrong.push(code);
                }
            }
        }
        println!(
            "{} of 63488 characters weigh differently: {:x?}",
            wrong.len(),
            &wrong[..wrong.len().min(20)]
        );
        assert!(wrong.iter().all(|code| *code >= 0x0300), "{wrong:x?}");

        let schedules = ints("schedules")?;
        for (index, t_len) in [82usize, 300, 749].iter().enumerate() {
            let actual: Vec<i32> = build_schedule(t_len * 8, DEFAULT_NUM_STEPS, DEFAULT_T_SHIFT)
                .into_iter()
                .map(|count| count as i32)
                .collect();
            assert_eq!(
                actual,
                schedules[index * 32..(index + 1) * 32],
                "schedule for {t_len}"
            );
        }

        let ref_text = meta["ref_text"].as_str();
        let text = meta["text"].as_str();
        let frames = oracle["codes"].dim(2)?;
        let estimate =
            estimate_target_tokens_with_reference(text, Some(ref_text), Some(frames), 1.0);
        assert_eq!(estimate.to_string(), meta["estimate"]);
        let chunked = ints("chunked")?;
        for (expected, text) in chunked.iter().zip([
            "Hi.".to_string(),
            "\u{201c}Curtsey as you\u{2019}re falling\u{2014}Do you see?\u{201d}".to_string(),
            "It was 1865, and the Rabbit had 3 watches. ".repeat(12),
        ]) {
            let actual =
                estimate_target_tokens_with_reference(&text, Some(ref_text), Some(281), 1.0);
            assert_eq!(actual as i32, *expected, "estimate for {text}");
        }

        // Stages with the model.
        let model = OmniVoiceModel::load(
            TtsConfig::new(crate::models::ModelType::OmniVoice).with_preferred_runtime(),
        )?;
        println!("backbone dtype {:?}", model.compute_dtype);
        let vb = ModelFiles::load_safetensors_vb(
            &model.files.speech_tokenizer_weights,
            DType::F32,
            &model.audio_device,
        )?;
        let encoder = OmniVoiceAudioEncoder::load(
            &model.audio_tokenizer_config,
            model.config.num_audio_codebook,
            vb,
            &model.audio_device,
        )?;
        let features = encoder.semantic_features(&prepared)?;
        let diff = max_diff(
            &features.flatten_all()?.to_vec1::<f32>()?,
            &floats("semantic_accurate")?,
        );
        println!("semantic features max diff {diff}");
        assert!(diff < 1e-3);
        let semantic = encoder.semantic_encoded(&features)?;
        let diff = max_diff(
            &semantic.flatten_all()?.to_vec1::<f32>()?,
            &floats("e_semantic")?,
        );
        println!("semantic encoder max diff {diff}");
        let acoustic = encoder.acoustic_features(&prepared, semantic.dim(2)?)?;
        let diff = max_diff(
            &acoustic.flatten_all()?.to_vec1::<f32>()?,
            &floats("acoustic_accurate")?,
        );
        println!("acoustic encoder max diff {diff}");
        assert!(diff < 1e-2);
        let (codes, code_frames) = encoder.encode(&prepared)?;
        drop(encoder);
        let expected_codes = ints("codes")?;
        assert_eq!(code_frames, frames);
        let accurate_codes = ints("codes_accurate")?;
        let same = codes
            .iter()
            .zip(&accurate_codes)
            .filter(|(a, b)| **a as i32 == **b)
            .count();
        println!("reference codes: {same} of {} match", codes.len());
        assert!(same as f32 / codes.len() as f32 > 0.99);

        // From here on, use Python's codes so later stages are compared alone.
        let ref_tokens = GeneratedAudioTokens {
            tokens: expected_codes.iter().map(|code| *code as u32).collect(),
            len: frames,
        };
        let inputs = model.prepare_inference_inputs(
            text,
            estimate,
            None,
            None,
            true,
            Some(ref_text),
            Some(&ref_tokens),
        )?;
        let expected_ids = ints("input_ids")?;
        let actual_ids: Vec<i32> = inputs.input_ids.iter().map(|id| *id as i32).collect();
        assert_eq!(actual_ids, expected_ids, "input ids");
        let expected_mask = ints("audio_mask")?;
        let actual_mask: Vec<i32> = inputs
            .audio_mask
            .iter()
            .map(|flag| i32::from(*flag))
            .collect();
        assert_eq!(actual_mask, expected_mask, "audio mask");

        let device = &model.compute_device;
        let (seq_len, codebooks) = (inputs.seq_len, model.config.num_audio_codebook);
        let cond_ids =
            Tensor::new(inputs.input_ids.as_slice(), device)?.reshape((1, codebooks, seq_len))?;
        let mask_u32: Vec<u32> = inputs.audio_mask.iter().copied().map(u32::from).collect();
        let mask_f32: Vec<f32> = inputs.audio_mask.iter().copied().map(f32::from).collect();
        let cond = model.backbone.lock().unwrap().forward(
            &cond_ids,
            &Tensor::new(mask_u32.as_slice(), device)?.reshape((1, seq_len))?,
            &Tensor::new(mask_f32.as_slice(), device)?.reshape((1, seq_len))?,
            None,
        )?;
        let first = cond.to_dtype(DType::F32)?.i((
            0,
            ..,
            seq_len - estimate..seq_len - estimate + 4,
            ..,
        ))?;
        let actual = first.flatten_all()?.to_vec1::<f32>()?;
        let expected = floats("cond_logits_0")?;
        let diff = max_diff(&actual, &expected);
        let mean: f32 = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / actual.len() as f32;
        println!("first conditional logits: max diff {diff}, mean diff {mean}");

        let config = OmniVoiceGenerationConfig {
            num_step: DEFAULT_NUM_STEPS,
            guidance_scale: DEFAULT_GUIDANCE_SCALE,
            t_shift: DEFAULT_T_SHIFT,
            layer_penalty_factor: DEFAULT_LAYER_PENALTY,
            position_temperature: 0.0,
            class_temperature: DEFAULT_CLASS_TEMPERATURE,
            denoise: true,
            audio_chunk_duration: DEFAULT_AUDIO_CHUNK_DURATION,
            audio_chunk_threshold: DEFAULT_AUDIO_CHUNK_THRESHOLD,
        };
        let started = std::time::Instant::now();
        let generated = model.generate_audio_tokens(
            text,
            estimate,
            None,
            None,
            &config,
            Some(ref_text),
            Some(&ref_tokens),
        )?;
        println!(
            "generated {} frames in {:.1}s",
            generated.len,
            started.elapsed().as_secs_f32()
        );
        let expected_tokens = ints("det_tokens")?;
        let same = generated
            .tokens
            .iter()
            .zip(&expected_tokens)
            .filter(|(a, b)| **a as i32 == **b)
            .count();
        println!(
            "deterministic tokens: {same} of {} match",
            expected_tokens.len()
        );

        let oracle_tokens = GeneratedAudioTokens {
            tokens: expected_tokens.iter().map(|code| *code as u32).collect(),
            len: estimate,
        };
        let decoded = model.decode_token_chunks(&[oracle_tokens])?;
        let diff = max_diff(&decoded, &floats("decoded")?);
        println!("decoded max diff {diff}");
        assert!(diff < 1e-2);
        let post = post_process_audio(&floats("decoded")?, 24_000, Some(rms as f32));
        let expected_post = floats("post_processed")?;
        println!("post-processed {} vs {}", post.len(), expected_post.len());
        assert_eq!(post.len(), expected_post.len());
        let diff = max_diff(&post, &expected_post);
        println!("post-processed max diff {diff}");
        assert!(diff < 1e-4);
        Ok(())
    }
}
