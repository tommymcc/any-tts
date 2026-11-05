use serde::Deserialize;

use crate::error::TtsError;

#[derive(Debug, Clone, Deserialize)]
pub struct OmniVoiceConfig {
    pub audio_mask_id: u32,
    pub audio_vocab_size: usize,
    pub num_audio_codebook: usize,
    #[serde(default)]
    pub audio_codebook_weights: Vec<f32>,
    pub llm_config: OmniVoiceLlmConfig,
}

impl OmniVoiceConfig {
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self, TtsError> {
        let mut config: Self = serde_json::from_slice(bytes.as_ref())?;
        if config.audio_codebook_weights.is_empty() {
            config.audio_codebook_weights = vec![8.0, 8.0, 6.0, 6.0, 4.0, 4.0, 2.0, 2.0];
        }
        Ok(config)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmniVoiceLlmConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    #[serde(default)]
    pub rope_parameters: Option<OmniVoiceRopeConfig>,
}

impl OmniVoiceLlmConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|rope| rope.rope_theta)
            .unwrap_or(1_000_000.0)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmniVoiceRopeConfig {
    pub rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmniVoiceAudioTokenizerConfig {
    pub sample_rate: u32,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub downsample_factor: usize,
    pub semantic_sample_rate: u32,
    pub kernel_size: usize,
    pub unit_kernel_size: usize,
    pub strides: Vec<usize>,
    pub channel_ratios: Vec<usize>,
    pub block_dilations: Vec<usize>,
    pub acoustic_model_config: OmniVoiceDacConfig,
    pub semantic_model_config: OmniVoiceSemanticConfig,
}

impl OmniVoiceAudioTokenizerConfig {
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self, TtsError> {
        Ok(serde_json::from_slice(bytes.as_ref())?)
    }

    pub fn hop_length(&self) -> usize {
        self.acoustic_model_config
            .upsampling_ratios
            .iter()
            .product()
    }

    /// Size of the joined acoustic and semantic features.
    pub fn semantic_hidden_size(&self) -> usize {
        self.acoustic_model_config.hidden_size + self.semantic_model_config.hidden_size
    }

    /// How many semantic frames make one codec frame.
    pub fn semantic_downsample_factor(&self) -> usize {
        let hop = self.hop_length() as f64;
        let ratio = f64::from(self.sample_rate) / f64::from(self.semantic_sample_rate);
        ((hop / ratio / self.downsample_factor as f64) as usize).max(1)
    }

    pub fn frame_rate(&self) -> usize {
        let hop_length = self.hop_length();
        (self.sample_rate as usize + hop_length.saturating_sub(1)) / hop_length.max(1)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmniVoiceDacConfig {
    pub hidden_size: usize,
    pub decoder_hidden_size: usize,
    pub encoder_hidden_size: usize,
    pub downsampling_ratios: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmniVoiceSemanticConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_conv_pos_embeddings: usize,
    pub num_conv_pos_embedding_groups: usize,
    pub layer_norm_eps: f64,
    pub conv_dim: Vec<usize>,
    pub conv_kernel: Vec<usize>,
    pub conv_stride: Vec<usize>,
}
