#[cfg(test)]
use std::sync::{Mutex, OnceLock};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    path::Path,
};

use crate::models::ggml_asr_executor::GgmlAsrPreparedAudio;
use crate::models::ggml_family_registry::{
    COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID, COHERE_TRANSCRIBE_DECODE_POLICY_ID,
    COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID, COHERE_TRANSCRIBE_TOKENIZER_ID,
    WHISPER_AUDIO_FRONTEND_ID, WHISPER_DECODE_POLICY_ID, WHISPER_GGML_ARCHITECTURE_ID,
    WHISPER_TOKENIZER_ID,
};
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};
use crate::models::{
    cohere::COHERE_TRANSCRIBE_MODEL_FAMILY,
    whisper::{WHISPER_MODEL_FAMILY, whisper_log_mel_spectrogram_16khz_mono_v0},
};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const RESERVED_OASR_MAGIC: &[u8; 4] = b"OASR";
const GGUF_VERSION_V3: u32 = 3;
const GGUF_TYPE_STRING: i32 = 8;
const GGUF_TYPE_ARRAY: i32 = 9;
const GGML_TYPE_F32: i32 = 0;
const GGML_TYPE_F16: i32 = 1;
const GGML_TYPE_I32: i32 = 26;
const GGUF_DEFAULT_ALIGNMENT: usize = 32;
const OPENASR_MODEL_ID_KEY: &str = "openasr.model.id";
const WHISPER_GRAPH_ARCHITECTURE: &str = "whisper";
const WHISPER_DEFAULT_HIDDEN_SIZE: usize = 8;
const WHISPER_DEFAULT_MELS: usize = 4;
const WHISPER_DEFAULT_POSITIONAL_FRAMES: usize = 128;
const WHISPER_DEFAULT_TOKEN_VOCAB: usize = 64;
const WHISPER_MLP_EXPANSION_FACTOR: u64 = 4;
pub const WHISPER_TINY_ENCODER_SMOKE_AUDIO_SAMPLES: usize = 480;
pub const WHISPER_TINY_ENCODER_SMOKE_MEL_HOP_SAMPLES: usize = 160;
const WHISPER_EXPECTED_SAMPLE_RATE_HZ: u32 = 16_000;
const WHISPER_EXPECTED_CHANNELS: u16 = 1;
const WHISPER_REAL_MEL_SOURCE_LABEL: &str = "whisper-log-mel-frontend-v0";
const COHERE_GRAPH_ARCHITECTURE: &str = "cohere-transcribe";
const TINY_WHISPER_SYNTHETIC_EOS_TOKEN_ID: u32 = 101;
const TINY_WHISPER_REAL_SMOKE_MODEL_PACK_RELATIVE_PATH: &str = "tmp/whisper-tiny.en-hf-gguf.oasr";
const TINY_WHISPER_REAL_SMOKE_AUDIO_RELATIVE_PATH: &str =
    "tmp/audio/librispeech/8461-278226-0010.wav";
const TINY_WHISPER_SYNTHETIC_EXPECTED_TEXT: &str = "hi";
const WHISPER_REQUIRED_TENSOR_ANCHORS_FOR_SKELETON: &[&str] = &[
    "model.encoder.conv1.weight",
    "model.encoder.conv2.weight",
    "model.encoder.embed_positions.weight",
    "model.decoder.embed_tokens.weight",
    "model.decoder.embed_positions.weight",
    "model.encoder.layers.0.self_attn.q_proj.weight",
    "model.decoder.layers.0.self_attn.q_proj.weight",
    "model.decoder.layers.0.encoder_attn.q_proj.weight",
];

#[cfg(test)]
pub(crate) fn with_forced_cpu_backend_for_test<T>(run: impl FnOnce() -> T) -> T {
    static BACKEND_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let lock = BACKEND_ENV_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().expect("backend env lock");
    let previous = std::env::var_os("OPENASR_GGML_BACKEND");
    #[expect(unsafe_code, reason = "test-only process env override")]
    unsafe {
        std::env::set_var("OPENASR_GGML_BACKEND", "cpu");
    }
    let result = run();
    match previous {
        Some(value) => {
            #[expect(unsafe_code, reason = "test-only process env restore")]
            unsafe {
                std::env::set_var("OPENASR_GGML_BACKEND", value);
            }
        }
        None => {
            #[expect(unsafe_code, reason = "test-only process env restore")]
            unsafe {
                std::env::remove_var("OPENASR_GGML_BACKEND");
            }
        }
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TinyWhisperEncoderSmokeShape {
    pub mel_bins: usize,
    pub mel_frames: usize,
    pub output_frames: usize,
    pub hidden_size: usize,
}

impl TinyWhisperEncoderSmokeShape {
    pub fn output_elements(self) -> usize {
        self.output_frames.saturating_mul(self.hidden_size)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TinyWhisperMelSmokeInput {
    pub source_label: &'static str,
    pub mel_bins: usize,
    pub mel_frames: usize,
    pub values_f32: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhisperExecutionFailureStage {
    MetadataPreflight,
    TensorBindingPreflight,
    MelFeature,
    EncoderPrelude,
    EncoderGraph,
    EncoderExecuted,
    DecoderTokenizerPending,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TinyGgufFixtureSpec {
    pub metadata: BTreeMap<String, String>,
    pub metadata_string_arrays: BTreeMap<String, Vec<String>>,
    pub tensor_names: Vec<String>,
    tensor_dims: BTreeMap<String, Vec<u64>>,
    tensor_types: BTreeMap<String, i32>,
}

impl TinyGgufFixtureSpec {
    pub fn new(metadata: BTreeMap<String, String>) -> Self {
        let tensor_names = vec!["fixture.tensor".to_string()];
        let tensor_dims = tensor_names
            .iter()
            .map(|name| (name.clone(), vec![1]))
            .collect::<BTreeMap<_, _>>();
        let tensor_types = tensor_names
            .iter()
            .map(|name| (name.clone(), GGML_TYPE_F32))
            .collect::<BTreeMap<_, _>>();
        Self {
            metadata,
            metadata_string_arrays: BTreeMap::new(),
            tensor_names,
            tensor_dims,
            tensor_types,
        }
    }

    pub fn whisper_oasr_v1_non_streaming_cpu(model_id: impl Into<String>) -> Self {
        let model_id = model_id.into();
        let mut metadata = BTreeMap::new();
        metadata.insert(OPENASR_MODEL_ID_KEY.to_string(), model_id);
        metadata.insert(
            OASR_METADATA_KEY_PACKAGE_VERSION.to_string(),
            OASR_PACKAGE_VERSION_V1.to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_MODEL_FAMILY.to_string(),
            WHISPER_MODEL_FAMILY.to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_MODEL_ARCHITECTURE.to_string(),
            WHISPER_GGML_ARCHITECTURE_ID.to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_AUDIO_FRONTEND.to_string(),
            WHISPER_AUDIO_FRONTEND_ID.to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_DECODE_POLICY.to_string(),
            WHISPER_DECODE_POLICY_ID.to_string(),
        );
        metadata.insert(
            "openasr.tokenizer.id".to_string(),
            WHISPER_TOKENIZER_ID.to_string(),
        );
        Self::new(metadata)
    }

    pub fn whisper_oasr_v1_graph_ready_for_runtime_fail_closed(
        model_id: impl Into<String>,
    ) -> Self {
        Self::whisper_oasr_v1_encoder_graph_one_layer(model_id)
    }

    pub fn whisper_oasr_v1_encoder_graph_one_layer(model_id: impl Into<String>) -> Self {
        Self::whisper_oasr_v1_encoder_graph_layers(model_id, 1, 1)
    }

    pub fn whisper_oasr_v1_encoder_graph_layers(
        model_id: impl Into<String>,
        encoder_layers: usize,
        decoder_layers: usize,
    ) -> Self {
        Self::whisper_oasr_v1_non_streaming_cpu(model_id)
            .with_whisper_graph_metadata(
                encoder_layers,
                decoder_layers,
                WHISPER_DEFAULT_HIDDEN_SIZE,
                WHISPER_DEFAULT_MELS,
            )
            .with_whisper_layer_count(encoder_layers, decoder_layers)
            .with_whisper_encoder_graph_tensors(encoder_layers, decoder_layers)
    }

    pub fn whisper_oasr_v1_encoder_graph_missing_tensor(
        model_id: impl Into<String>,
        tensor_name: &str,
    ) -> Self {
        Self::whisper_oasr_v1_encoder_graph_one_layer(model_id)
            .with_whisper_missing_required_tensor(tensor_name)
    }

    pub fn whisper_oasr_v1_encoder_graph_shape_mismatch(
        model_id: impl Into<String>,
        tensor_name: impl Into<String>,
        dims: impl IntoIterator<Item = u64>,
    ) -> Self {
        Self::whisper_oasr_v1_encoder_graph_one_layer(model_id)
            .with_whisper_required_tensor_shape_mismatch(tensor_name, dims)
    }

    pub fn whisper_oasr_v1_encoder_graph_type_mismatch(
        model_id: impl Into<String>,
        tensor_name: impl Into<String>,
    ) -> Self {
        Self::whisper_oasr_v1_encoder_graph_one_layer(model_id)
            .with_tensor_type(tensor_name, GGML_TYPE_I32)
    }

    pub fn whisper_oasr_v1_encoder_graph_unsupported_primitive(
        model_id: impl Into<String>,
    ) -> Self {
        Self::whisper_oasr_v1_encoder_graph_one_layer(model_id)
            .with_metadata("whisper.encoder.context_length", "1")
            .with_whisper_required_tensor_shape_mismatch(
                "model.encoder.embed_positions.weight",
                [1_u64, WHISPER_DEFAULT_HIDDEN_SIZE as u64],
            )
    }

    pub fn whisper_oasr_v1_encoder_graph_layer_count_mismatch(
        model_id: impl Into<String>,
        encoder_layers: usize,
        decoder_layers: usize,
    ) -> Self {
        Self::whisper_oasr_v1_encoder_graph_one_layer(model_id)
            .with_whisper_layer_count_mismatch(encoder_layers, decoder_layers)
    }

    pub fn cohere_oasr_v1_non_streaming_cpu(model_id: impl Into<String>) -> Self {
        let model_id = model_id.into();
        let mut metadata = BTreeMap::new();
        metadata.insert(OPENASR_MODEL_ID_KEY.to_string(), model_id);
        metadata.insert(
            OASR_METADATA_KEY_PACKAGE_VERSION.to_string(),
            OASR_PACKAGE_VERSION_V1.to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_MODEL_FAMILY.to_string(),
            COHERE_TRANSCRIBE_MODEL_FAMILY.to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_MODEL_ARCHITECTURE.to_string(),
            COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID.to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_AUDIO_FRONTEND.to_string(),
            COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID.to_string(),
        );
        metadata.insert(
            OASR_METADATA_KEY_DECODE_POLICY.to_string(),
            COHERE_TRANSCRIBE_DECODE_POLICY_ID.to_string(),
        );
        metadata.insert(
            "openasr.tokenizer.id".to_string(),
            COHERE_TRANSCRIBE_TOKENIZER_ID.to_string(),
        );
        Self::new(metadata)
            .with_string_array_metadata(
                "tokenizer.ggml.tokens",
                [
                    "<|startofcontext|>",
                    "<|startoftranscript|>",
                    "<|emo:undefined|>",
                    "<|en|>",
                    "<|pnc|>",
                    "<|noitn|>",
                    "<|notimestamp|>",
                    "<|nodiarize|>",
                    "<|endoftext|>",
                    "▁fixture9",
                    "▁fixture10",
                    "▁fixture11",
                    "▁fixture12",
                    "▁fixture13",
                    "▁fixture14",
                    "▁fixture15",
                    "▁fixture16",
                    "▁fixture17",
                    "▁fixture18",
                    "▁fixture19",
                    "▁fixture20",
                    "▁fixture21",
                    "▁fixture22",
                    "▁fixture23",
                    "▁fixture24",
                    "▁fixture25",
                    "▁fixture26",
                    "▁fixture27",
                    "▁fixture28",
                    "▁fixture29",
                    "▁fixture30",
                    "▁fixture31",
                ],
            )
            .with_metadata("tokenizer.ggml.model", "llama")
    }

    pub fn cohere_oasr_v1_runtime_ready(model_id: impl Into<String>) -> Self {
        Self::cohere_oasr_v1_non_streaming_cpu(model_id)
            .with_cohere_graph_metadata(2, 2, 16, 2, 8, 32, 5, 32, 32)
            .with_cohere_runtime_tensors_with_layers(2, 2)
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn with_string_array_metadata(
        mut self,
        key: impl Into<String>,
        values: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.metadata_string_arrays
            .insert(key.into(), values.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_whisper_graph_metadata(
        mut self,
        encoder_layers: usize,
        decoder_layers: usize,
        embedding_length: usize,
        encoder_mels_count: usize,
    ) -> Self {
        let encoder_attention_heads = if embedding_length.is_multiple_of(6) {
            6
        } else if embedding_length.is_multiple_of(4) {
            4
        } else if embedding_length.is_multiple_of(2) {
            2
        } else {
            1
        };
        let encoder_context_length = WHISPER_DEFAULT_POSITIONAL_FRAMES;
        self.metadata.insert(
            "general.architecture".to_string(),
            WHISPER_GRAPH_ARCHITECTURE.to_string(),
        );
        self.metadata.insert(
            "whisper.encoder.block_count".to_string(),
            encoder_layers.to_string(),
        );
        self.metadata.insert(
            "whisper.decoder.block_count".to_string(),
            decoder_layers.to_string(),
        );
        self.metadata.insert(
            "whisper.decoder.embedding_length".to_string(),
            embedding_length.to_string(),
        );
        self.metadata.insert(
            "whisper.decoder.attention.head_count".to_string(),
            encoder_attention_heads.to_string(),
        );
        self.metadata.insert(
            "whisper.decoder.context_length".to_string(),
            WHISPER_DEFAULT_POSITIONAL_FRAMES.to_string(),
        );
        self.metadata.insert(
            "whisper.vocab_size".to_string(),
            WHISPER_DEFAULT_TOKEN_VOCAB.to_string(),
        );
        self.metadata.insert(
            "whisper.encoder.embedding_length".to_string(),
            embedding_length.to_string(),
        );
        self.metadata.insert(
            "whisper.encoder.attention.head_count".to_string(),
            encoder_attention_heads.to_string(),
        );
        self.metadata.insert(
            "whisper.encoder.context_length".to_string(),
            encoder_context_length.to_string(),
        );
        self.metadata.insert(
            "whisper.encoder.mels_count".to_string(),
            encoder_mels_count.to_string(),
        );
        self
    }

    pub fn with_cohere_graph_metadata(
        mut self,
        encoder_layers: usize,
        decoder_layers: usize,
        encoder_d_model: usize,
        encoder_heads: usize,
        encoder_head_dim: usize,
        encoder_ffn_dim: usize,
        encoder_conv_kernel: usize,
        vocab_size: usize,
        n_mels: usize,
    ) -> Self {
        let decoder_d_model = encoder_d_model;
        let decoder_heads = encoder_heads;
        let decoder_head_dim = encoder_head_dim;
        let decoder_ffn_dim = encoder_ffn_dim;
        self.metadata.insert(
            "general.architecture".to_string(),
            COHERE_GRAPH_ARCHITECTURE.to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.vocab_size".to_string(),
            vocab_size.to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.encoder.n_layers".to_string(),
            encoder_layers.to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.encoder.d_model".to_string(),
            encoder_d_model.to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.encoder.n_heads".to_string(),
            encoder_heads.to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.encoder.head_dim".to_string(),
            encoder_head_dim.to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.encoder.ffn_dim".to_string(),
            encoder_ffn_dim.to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.encoder.conv_kernel".to_string(),
            encoder_conv_kernel.to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.decoder.n_layers".to_string(),
            decoder_layers.to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.decoder.d_model".to_string(),
            decoder_d_model.to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.decoder.n_heads".to_string(),
            decoder_heads.to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.decoder.head_dim".to_string(),
            decoder_head_dim.to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.decoder.ffn_dim".to_string(),
            decoder_ffn_dim.to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.decoder.max_ctx".to_string(),
            "32".to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.decoder.start_token_id".to_string(),
            "13764".to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.audio.sample_rate".to_string(),
            "16000".to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.audio.n_mels".to_string(),
            n_mels.to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.audio.n_fft".to_string(),
            "400".to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.audio.hop_length".to_string(),
            "160".to_string(),
        );
        self.metadata.insert(
            "cohere_transcribe.audio.win_length".to_string(),
            "400".to_string(),
        );
        self
    }

    pub fn with_cohere_runtime_tensors_with_layers(
        mut self,
        encoder_layers: usize,
        decoder_layers: usize,
    ) -> Self {
        let encoder_d_model = self
            .metadata
            .get("cohere_transcribe.encoder.d_model")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(16);
        let encoder_heads = self
            .metadata
            .get("cohere_transcribe.encoder.n_heads")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(2);
        let encoder_head_dim = self
            .metadata
            .get("cohere_transcribe.encoder.head_dim")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(8);
        let encoder_ffn_dim = self
            .metadata
            .get("cohere_transcribe.encoder.ffn_dim")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(32);
        let encoder_conv_kernel = self
            .metadata
            .get("cohere_transcribe.encoder.conv_kernel")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(5);
        let decoder_d_model = self
            .metadata
            .get("cohere_transcribe.decoder.d_model")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(16);
        let decoder_ffn_dim = self
            .metadata
            .get("cohere_transcribe.decoder.ffn_dim")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(32);
        let vocab_size = self
            .metadata
            .get("cohere_transcribe.vocab_size")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(32);
        let decoder_max_ctx = self
            .metadata
            .get("cohere_transcribe.decoder.max_ctx")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(32);
        let n_mels = self
            .metadata
            .get("cohere_transcribe.audio.n_mels")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(8);
        let n_fft = self
            .metadata
            .get("cohere_transcribe.audio.n_fft")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(400);
        let win_length = self
            .metadata
            .get("cohere_transcribe.audio.win_length")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(400);
        let fft_bins = n_fft / 2 + 1;
        let pre_conv_channels = 256_u64;
        let subsampled_mels = cohere_fixture_conv_out_dim(n_mels, 3, 2, 1);
        let subsampled_mels = cohere_fixture_conv_out_dim(subsampled_mels, 3, 2, 1);
        let subsampled_mels = cohere_fixture_conv_out_dim(subsampled_mels, 3, 2, 1);
        let pre_out_width = pre_conv_channels.saturating_mul(subsampled_mels.max(1));

        self = self
            .with_tensor_shape("fe.mel_fb", [fft_bins, n_mels])
            .with_tensor_shape("fe.window", [win_length])
            .with_tensor_shape("enc.pre.conv.0.weight", [3_u64, 3_u64, 1_u64, 4_u64])
            .with_tensor_shape("enc.pre.conv.0.bias", [4_u64])
            .with_tensor_shape("enc.pre.conv.2.weight", [3_u64, 3_u64, 1_u64, 4_u64])
            .with_tensor_shape("enc.pre.conv.2.bias", [4_u64])
            .with_tensor_shape("enc.pre.conv.3.weight", [1_u64, 1_u64, 4_u64, 256_u64])
            .with_tensor_shape("enc.pre.conv.3.bias", [256_u64])
            .with_tensor_shape("enc.pre.conv.5.weight", [3_u64, 3_u64, 1_u64, 256_u64])
            .with_tensor_shape("enc.pre.conv.5.bias", [256_u64])
            .with_tensor_shape("enc.pre.conv.6.weight", [1_u64, 1_u64, 256_u64, 256_u64])
            .with_tensor_shape("enc.pre.conv.6.bias", [256_u64])
            .with_tensor_shape("enc.pre.out.weight", [pre_out_width, encoder_d_model])
            .with_tensor_shape("enc.pre.out.bias", [encoder_d_model])
            .with_tensor_shape("enc.proj.weight", [decoder_d_model, encoder_d_model])
            .with_tensor_shape("enc.proj.bias", [decoder_d_model])
            .with_tensor_shape("dec.emb.weight", [vocab_size, decoder_d_model])
            .with_tensor_shape("dec.pos.weight", [decoder_max_ctx, decoder_d_model])
            .with_tensor_shape("dec.emb_ln.weight", [decoder_d_model])
            .with_tensor_shape("dec.emb_ln.bias", [decoder_d_model])
            .with_tensor_shape("dec.out_ln.weight", [decoder_d_model])
            .with_tensor_shape("dec.out_ln.bias", [decoder_d_model])
            .with_tensor_shape("dec.head.weight", [decoder_d_model, vocab_size])
            .with_tensor_shape("dec.head.bias", [vocab_size]);

        for layer_idx in 0..encoder_layers {
            let prefix = format!("enc.blk.{layer_idx}.");
            self = self
                .with_tensor_shape(format!("{prefix}ff1.norm.weight"), [encoder_d_model])
                .with_tensor_shape(format!("{prefix}ff1.norm.bias"), [encoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}ff1.up.weight"),
                    [encoder_d_model, encoder_ffn_dim],
                )
                .with_tensor_shape(format!("{prefix}ff1.up.bias"), [encoder_ffn_dim])
                .with_tensor_shape(
                    format!("{prefix}ff1.down.weight"),
                    [encoder_ffn_dim, encoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}ff1.down.bias"), [encoder_d_model])
                .with_tensor_shape(format!("{prefix}attn.norm.weight"), [encoder_d_model])
                .with_tensor_shape(format!("{prefix}attn.norm.bias"), [encoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}attn.q.weight"),
                    [encoder_d_model, encoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}attn.q.bias"), [encoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}attn.k.weight"),
                    [encoder_d_model, encoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}attn.k.bias"), [encoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}attn.v.weight"),
                    [encoder_d_model, encoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}attn.v.bias"), [encoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}attn.out.weight"),
                    [encoder_d_model, encoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}attn.out.bias"), [encoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}attn.pos.weight"),
                    [encoder_d_model, encoder_d_model],
                )
                .with_tensor_shape(
                    format!("{prefix}attn.pos_bias_u"),
                    [encoder_heads, encoder_head_dim],
                )
                .with_tensor_shape(
                    format!("{prefix}attn.pos_bias_v"),
                    [encoder_heads, encoder_head_dim],
                )
                .with_tensor_shape(format!("{prefix}conv.norm.weight"), [encoder_d_model])
                .with_tensor_shape(format!("{prefix}conv.norm.bias"), [encoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}conv.pw1.weight"),
                    [encoder_d_model * 2, encoder_d_model, 1_u64],
                )
                .with_tensor_shape(format!("{prefix}conv.pw1.bias"), [encoder_d_model * 2])
                .with_tensor_shape(
                    format!("{prefix}conv.dw.weight"),
                    [encoder_d_model, 1_u64, encoder_conv_kernel],
                )
                .with_tensor_shape(format!("{prefix}conv.dw.bias"), [encoder_d_model])
                .with_tensor_shape(format!("{prefix}conv.bn.weight"), [encoder_d_model])
                .with_tensor_shape(format!("{prefix}conv.bn.bias"), [encoder_d_model])
                .with_tensor_shape(format!("{prefix}conv.bn.mean"), [encoder_d_model])
                .with_tensor_shape(format!("{prefix}conv.bn.var"), [encoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}conv.pw2.weight"),
                    [encoder_d_model, encoder_d_model, 1_u64],
                )
                .with_tensor_shape(format!("{prefix}conv.pw2.bias"), [encoder_d_model])
                .with_tensor_shape(format!("{prefix}ff2.norm.weight"), [encoder_d_model])
                .with_tensor_shape(format!("{prefix}ff2.norm.bias"), [encoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}ff2.up.weight"),
                    [encoder_d_model, encoder_ffn_dim],
                )
                .with_tensor_shape(format!("{prefix}ff2.up.bias"), [encoder_ffn_dim])
                .with_tensor_shape(
                    format!("{prefix}ff2.down.weight"),
                    [encoder_ffn_dim, encoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}ff2.down.bias"), [encoder_d_model])
                .with_tensor_shape(format!("{prefix}out_norm.weight"), [encoder_d_model])
                .with_tensor_shape(format!("{prefix}out_norm.bias"), [encoder_d_model]);
        }

        for layer_idx in 0..decoder_layers {
            let prefix = format!("dec.blk.{layer_idx}.");
            self = self
                .with_tensor_shape(format!("{prefix}attn_ln.weight"), [decoder_d_model])
                .with_tensor_shape(format!("{prefix}attn_ln.bias"), [decoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}attn_q.weight"),
                    [decoder_d_model, decoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}attn_q.bias"), [decoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}attn_k.weight"),
                    [decoder_d_model, decoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}attn_k.bias"), [decoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}attn_v.weight"),
                    [decoder_d_model, decoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}attn_v.bias"), [decoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}attn_o.weight"),
                    [decoder_d_model, decoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}attn_o.bias"), [decoder_d_model])
                .with_tensor_shape(format!("{prefix}cross_ln.weight"), [decoder_d_model])
                .with_tensor_shape(format!("{prefix}cross_ln.bias"), [decoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}cross_q.weight"),
                    [decoder_d_model, decoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}cross_q.bias"), [decoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}cross_k.weight"),
                    [decoder_d_model, decoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}cross_k.bias"), [decoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}cross_v.weight"),
                    [decoder_d_model, decoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}cross_v.bias"), [decoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}cross_o.weight"),
                    [decoder_d_model, decoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}cross_o.bias"), [decoder_d_model])
                .with_tensor_shape(format!("{prefix}ffn_ln.weight"), [decoder_d_model])
                .with_tensor_shape(format!("{prefix}ffn_ln.bias"), [decoder_d_model])
                .with_tensor_shape(
                    format!("{prefix}ffn_up.weight"),
                    [decoder_d_model, decoder_ffn_dim],
                )
                .with_tensor_shape(format!("{prefix}ffn_up.bias"), [decoder_ffn_dim])
                .with_tensor_shape(
                    format!("{prefix}ffn_down.weight"),
                    [decoder_ffn_dim, decoder_d_model],
                )
                .with_tensor_shape(format!("{prefix}ffn_down.bias"), [decoder_d_model]);
        }

        self
    }

    pub fn with_tensor_names(mut self, tensor_names: impl IntoIterator<Item = String>) -> Self {
        self.tensor_names = dedup_tensor_names(tensor_names);
        self.reconcile_tensor_dims_with_names();
        self
    }

    pub fn with_added_tensor(mut self, tensor_name: impl Into<String>) -> Self {
        let tensor_name = tensor_name.into();
        self.tensor_names.push(tensor_name.clone());
        self.tensor_names = dedup_tensor_names(self.tensor_names);
        self.tensor_dims
            .entry(tensor_name.clone())
            .or_insert_with(|| vec![1]);
        self.tensor_types
            .entry(tensor_name)
            .or_insert(GGML_TYPE_F32);
        self
    }

    pub fn without_tensor(mut self, tensor_name: &str) -> Self {
        self.tensor_names.retain(|name| name != tensor_name);
        self.tensor_dims.remove(tensor_name);
        self.tensor_types.remove(tensor_name);
        self
    }

    pub fn with_tensor_alias(
        mut self,
        canonical_name: &str,
        alias_name: impl Into<String>,
    ) -> Self {
        let alias_name = alias_name.into();
        let canonical_shape = self
            .tensor_dims
            .remove(canonical_name)
            .unwrap_or_else(|| vec![1]);
        let canonical_type = self
            .tensor_types
            .remove(canonical_name)
            .unwrap_or(GGML_TYPE_F32);
        self.tensor_names.retain(|name| name != canonical_name);
        self.tensor_names.push(alias_name.clone());
        self.tensor_names = dedup_tensor_names(self.tensor_names);
        self.tensor_dims.insert(alias_name.clone(), canonical_shape);
        self.tensor_types.insert(alias_name, canonical_type);
        self.reconcile_tensor_dims_with_names();
        self
    }

    pub fn with_tensor_shape(
        mut self,
        tensor_name: impl Into<String>,
        dims: impl IntoIterator<Item = u64>,
    ) -> Self {
        let tensor_name = tensor_name.into();
        self.tensor_dims
            .insert(tensor_name.clone(), dims.into_iter().collect());
        if !self.tensor_names.contains(&tensor_name) {
            self.tensor_names.push(tensor_name.clone());
            self.tensor_names = dedup_tensor_names(self.tensor_names);
        }
        self.tensor_types
            .entry(tensor_name)
            .or_insert(GGML_TYPE_F32);
        self.reconcile_tensor_dims_with_names();
        self
    }

    pub fn with_tensor_f16(self, tensor_name: impl Into<String>) -> Self {
        self.with_tensor_type(tensor_name, GGML_TYPE_F16)
    }

    pub fn with_tensor_f32(self, tensor_name: impl Into<String>) -> Self {
        self.with_tensor_type(tensor_name, GGML_TYPE_F32)
    }

    pub fn with_tensor_type(mut self, tensor_name: impl Into<String>, ggml_type: i32) -> Self {
        let tensor_name = tensor_name.into();
        self.tensor_types.insert(tensor_name.clone(), ggml_type);
        self.tensor_dims
            .entry(tensor_name.clone())
            .or_insert_with(|| vec![1]);
        if !self.tensor_names.contains(&tensor_name) {
            self.tensor_names.push(tensor_name);
            self.tensor_names = dedup_tensor_names(self.tensor_names);
        }
        self.reconcile_tensor_dims_with_names();
        self
    }

    pub fn with_whisper_missing_required_tensor(self, tensor_name: &str) -> Self {
        self.without_tensor(tensor_name)
    }

    pub fn with_whisper_required_tensor_alias(
        self,
        canonical_name: &str,
        alias_name: impl Into<String>,
    ) -> Self {
        self.with_tensor_alias(canonical_name, alias_name)
    }

    pub fn with_whisper_required_tensor_shape_mismatch(
        self,
        tensor_name: impl Into<String>,
        dims: impl IntoIterator<Item = u64>,
    ) -> Self {
        self.with_tensor_shape(tensor_name, dims)
    }

    pub fn with_whisper_layer_count_mismatch(
        self,
        encoder_layers: usize,
        decoder_layers: usize,
    ) -> Self {
        self.with_whisper_layer_count(encoder_layers, decoder_layers)
    }

    pub fn with_whisper_layer_count(self, encoder_layers: usize, decoder_layers: usize) -> Self {
        self.with_metadata("whisper.encoder.block_count", encoder_layers.to_string())
            .with_metadata("whisper.decoder.block_count", decoder_layers.to_string())
    }

    pub fn with_whisper_encoder_graph_tensors(
        mut self,
        encoder_layers: usize,
        decoder_layers: usize,
    ) -> Self {
        let mut names = BTreeSet::new();
        for name in whisper_required_tensor_anchors_for_layers(encoder_layers, decoder_layers) {
            names.insert(name);
        }
        for name in whisper_required_gguf_binding_tensors(encoder_layers, decoder_layers) {
            names.insert(name);
        }
        self.tensor_names = names.into_iter().collect();
        self.reconcile_tensor_dims_with_names();
        self.apply_whisper_tensor_shape_defaults();
        self
    }

    pub fn with_whisper_preflight_tensors(
        self,
        encoder_layers: usize,
        decoder_layers: usize,
    ) -> Self {
        self.with_whisper_encoder_graph_tensors(encoder_layers, decoder_layers)
    }

    fn reconcile_tensor_dims_with_names(&mut self) {
        self.tensor_dims
            .retain(|name, _| self.tensor_names.iter().any(|tensor| tensor == name));
        self.tensor_types
            .retain(|name, _| self.tensor_names.iter().any(|tensor| tensor == name));
        for name in &self.tensor_names {
            self.tensor_dims
                .entry(name.clone())
                .or_insert_with(|| vec![1]);
            self.tensor_types
                .entry(name.clone())
                .or_insert(GGML_TYPE_F32);
        }
    }

    fn apply_whisper_tensor_shape_defaults(&mut self) {
        let encoder_layers = parse_metadata_usize(&self.metadata, "whisper.encoder.block_count", 1);
        let decoder_layers = parse_metadata_usize(&self.metadata, "whisper.decoder.block_count", 1);
        let encoder_hidden = parse_metadata_usize(
            &self.metadata,
            "whisper.encoder.embedding_length",
            WHISPER_DEFAULT_HIDDEN_SIZE,
        );
        let decoder_hidden = parse_metadata_usize(
            &self.metadata,
            "whisper.decoder.embedding_length",
            encoder_hidden,
        );
        let encoder_mels = parse_metadata_usize(
            &self.metadata,
            "whisper.encoder.mels_count",
            WHISPER_DEFAULT_MELS,
        );
        let encoder_hidden_u64 = encoder_hidden as u64;
        let decoder_hidden_u64 = decoder_hidden as u64;
        let encoder_mels_u64 = encoder_mels as u64;
        let mlp_hidden_u64 = encoder_hidden_u64.saturating_mul(WHISPER_MLP_EXPANSION_FACTOR);
        let decoder_mlp_hidden_u64 =
            decoder_hidden_u64.saturating_mul(WHISPER_MLP_EXPANSION_FACTOR);

        self.set_dims_if_present(
            &["model.encoder.conv1.weight", "encoder.conv1.weight"],
            vec![3, encoder_mels_u64, encoder_hidden_u64],
        );
        self.set_dims_if_present(
            &["model.encoder.conv1.bias", "encoder.conv1.bias"],
            vec![encoder_hidden_u64],
        );
        self.set_dims_if_present(
            &["model.encoder.conv2.weight", "encoder.conv2.weight"],
            vec![3, encoder_hidden_u64, encoder_hidden_u64],
        );
        self.set_dims_if_present(
            &["model.encoder.conv2.bias", "encoder.conv2.bias"],
            vec![encoder_hidden_u64],
        );
        self.set_dims_if_present(
            &[
                "model.encoder.embed_positions.weight",
                "encoder.positional_embedding",
            ],
            vec![WHISPER_DEFAULT_POSITIONAL_FRAMES as u64, encoder_hidden_u64],
        );
        self.set_dims_if_present(
            &[
                "model.decoder.embed_positions.weight",
                "decoder.positional_embedding",
            ],
            vec![WHISPER_DEFAULT_POSITIONAL_FRAMES as u64, decoder_hidden_u64],
        );
        self.set_dims_if_present(
            &[
                "model.decoder.embed_tokens.weight",
                "decoder.token_embedding.weight",
            ],
            vec![WHISPER_DEFAULT_TOKEN_VOCAB as u64, decoder_hidden_u64],
        );

        for layer_idx in 0..encoder_layers {
            let prefix = format!("model.encoder.layers.{layer_idx}.");
            self.set_dim_if_present(
                format!("{prefix}self_attn.q_proj.weight"),
                vec![encoder_hidden_u64, encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn.q_proj.bias"),
                vec![encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn.k_proj.weight"),
                vec![encoder_hidden_u64, encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn.k_proj.bias"),
                vec![encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn.v_proj.weight"),
                vec![encoder_hidden_u64, encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn.v_proj.bias"),
                vec![encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn.out_proj.weight"),
                vec![encoder_hidden_u64, encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn.out_proj.bias"),
                vec![encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn_layer_norm.weight"),
                vec![encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn_layer_norm.bias"),
                vec![encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}fc1.weight"),
                vec![mlp_hidden_u64, encoder_hidden_u64],
            );
            self.set_dim_if_present(format!("{prefix}fc1.bias"), vec![mlp_hidden_u64]);
            self.set_dim_if_present(
                format!("{prefix}fc2.weight"),
                vec![encoder_hidden_u64, mlp_hidden_u64],
            );
            self.set_dim_if_present(format!("{prefix}fc2.bias"), vec![encoder_hidden_u64]);
            self.set_dim_if_present(
                format!("{prefix}final_layer_norm.weight"),
                vec![encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}final_layer_norm.bias"),
                vec![encoder_hidden_u64],
            );

            let alias_prefix = format!("encoder.blocks.{layer_idx}.");
            self.set_dim_if_present(
                format!("{alias_prefix}attn.query.weight"),
                vec![encoder_hidden_u64, encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{alias_prefix}attn.key.weight"),
                vec![encoder_hidden_u64, encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{alias_prefix}attn.value.weight"),
                vec![encoder_hidden_u64, encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{alias_prefix}attn.out.weight"),
                vec![encoder_hidden_u64, encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{alias_prefix}mlp.0.weight"),
                vec![mlp_hidden_u64, encoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{alias_prefix}mlp.2.weight"),
                vec![encoder_hidden_u64, mlp_hidden_u64],
            );
        }

        for layer_idx in 0..decoder_layers {
            let prefix = format!("model.decoder.layers.{layer_idx}.");
            self.set_dim_if_present(
                format!("{prefix}self_attn.q_proj.weight"),
                vec![decoder_hidden_u64, decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn.q_proj.bias"),
                vec![decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn.k_proj.weight"),
                vec![decoder_hidden_u64, decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn.k_proj.bias"),
                vec![decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn.v_proj.weight"),
                vec![decoder_hidden_u64, decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn.v_proj.bias"),
                vec![decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn.out_proj.weight"),
                vec![decoder_hidden_u64, decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn.out_proj.bias"),
                vec![decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn_layer_norm.weight"),
                vec![decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}self_attn_layer_norm.bias"),
                vec![decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}encoder_attn.q_proj.weight"),
                vec![decoder_hidden_u64, decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}encoder_attn.q_proj.bias"),
                vec![decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}encoder_attn.k_proj.weight"),
                vec![decoder_hidden_u64, decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}encoder_attn.k_proj.bias"),
                vec![decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}encoder_attn.v_proj.weight"),
                vec![decoder_hidden_u64, decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}encoder_attn.v_proj.bias"),
                vec![decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}encoder_attn.out_proj.weight"),
                vec![decoder_hidden_u64, decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}encoder_attn.out_proj.bias"),
                vec![decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}encoder_attn_layer_norm.weight"),
                vec![decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}encoder_attn_layer_norm.bias"),
                vec![decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}fc1.weight"),
                vec![decoder_mlp_hidden_u64, decoder_hidden_u64],
            );
            self.set_dim_if_present(format!("{prefix}fc1.bias"), vec![decoder_mlp_hidden_u64]);
            self.set_dim_if_present(
                format!("{prefix}fc2.weight"),
                vec![decoder_hidden_u64, decoder_mlp_hidden_u64],
            );
            self.set_dim_if_present(format!("{prefix}fc2.bias"), vec![decoder_hidden_u64]);
            self.set_dim_if_present(
                format!("{prefix}final_layer_norm.weight"),
                vec![decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{prefix}final_layer_norm.bias"),
                vec![decoder_hidden_u64],
            );

            let alias_prefix = format!("decoder.blocks.{layer_idx}.");
            self.set_dim_if_present(
                format!("{alias_prefix}attn.query.weight"),
                vec![decoder_hidden_u64, decoder_hidden_u64],
            );
            self.set_dim_if_present(
                format!("{alias_prefix}cross_attn.query.weight"),
                vec![decoder_hidden_u64, decoder_hidden_u64],
            );
        }

        for name in &self.tensor_names {
            if self
                .tensor_dims
                .get(name)
                .is_some_and(|dims| dims.as_slice() != [1_u64])
            {
                continue;
            }
            if name.ends_with(".bias") {
                let hidden = if name.starts_with("model.decoder.") || name.starts_with("decoder.") {
                    decoder_hidden_u64
                } else {
                    encoder_hidden_u64
                };
                self.tensor_dims.insert(name.clone(), vec![hidden]);
                continue;
            }
            if name.ends_with(".weight") {
                let hidden = if name.starts_with("model.decoder.") || name.starts_with("decoder.") {
                    decoder_hidden_u64
                } else {
                    encoder_hidden_u64
                };
                self.tensor_dims.insert(name.clone(), vec![hidden, hidden]);
            }
        }
    }

    fn set_dims_if_present(&mut self, names: &[&str], dims: Vec<u64>) {
        for name in names {
            if self
                .tensor_names
                .iter()
                .any(|tensor_name| tensor_name == name)
            {
                self.tensor_dims.insert((*name).to_string(), dims.clone());
            }
        }
    }

    fn set_dim_if_present(&mut self, name: impl Into<String>, dims: Vec<u64>) {
        let name = name.into();
        if self
            .tensor_names
            .iter()
            .any(|tensor_name| tensor_name == &name)
        {
            self.tensor_dims.insert(name, dims);
        }
    }
}

fn dedup_tensor_names(tensor_names: impl IntoIterator<Item = String>) -> Vec<String> {
    tensor_names
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn parse_metadata_usize(metadata: &BTreeMap<String, String>, key: &str, fallback: usize) -> usize {
    metadata
        .get(key)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(fallback)
}

pub fn tiny_whisper_encoder_smoke_shape(
    hidden_size: usize,
    mel_bins: usize,
) -> TinyWhisperEncoderSmokeShape {
    let mel_frames = WHISPER_TINY_ENCODER_SMOKE_AUDIO_SAMPLES
        .div_ceil(WHISPER_TINY_ENCODER_SMOKE_MEL_HOP_SAMPLES);
    let conv1_output = whisper_conv_output_frames(mel_frames, 3, 1, 1, 1)
        .expect("conv1 smoke-shape inference must stay valid");
    let conv2_output = whisper_conv_output_frames(conv1_output, 3, 2, 1, 1)
        .expect("conv2 smoke-shape inference must stay valid");
    TinyWhisperEncoderSmokeShape {
        mel_bins,
        mel_frames,
        output_frames: conv2_output,
        hidden_size,
    }
}

pub fn tiny_whisper_encoder_smoke_shape_for_default_fixture() -> TinyWhisperEncoderSmokeShape {
    tiny_whisper_encoder_smoke_shape(WHISPER_DEFAULT_HIDDEN_SIZE, WHISPER_DEFAULT_MELS)
}

pub fn tiny_whisper_encoder_smoke_prepared_audio() -> GgmlAsrPreparedAudio {
    let samples = (0..WHISPER_TINY_ENCODER_SMOKE_AUDIO_SAMPLES)
        .map(|index| {
            let centered = (index % 17) as i32 - 8;
            centered as f32 / 16.0
        })
        .collect::<Vec<_>>();
    GgmlAsrPreparedAudio::mono_16khz(samples)
}

pub fn tiny_whisper_encoder_smoke_real_mel_input(
    prepared_audio: &GgmlAsrPreparedAudio,
    mel_bins: usize,
) -> Result<TinyWhisperMelSmokeInput, String> {
    if prepared_audio.sample_rate_hz != WHISPER_EXPECTED_SAMPLE_RATE_HZ {
        return Err(format!(
            "sample_rate_hz={} (expected {WHISPER_EXPECTED_SAMPLE_RATE_HZ})",
            prepared_audio.sample_rate_hz
        ));
    }
    if prepared_audio.channels != WHISPER_EXPECTED_CHANNELS {
        return Err(format!(
            "channels={} (expected {WHISPER_EXPECTED_CHANNELS})",
            prepared_audio.channels
        ));
    }
    if prepared_audio.samples_f32.is_empty() {
        return Err("samples_f32 is empty".to_string());
    }
    if prepared_audio
        .samples_f32
        .iter()
        .any(|sample| !sample.is_finite())
    {
        return Err("samples_f32 contains non-finite values".to_string());
    }
    let target_frames = prepared_audio
        .samples_f32
        .len()
        .max(1)
        .div_ceil(WHISPER_TINY_ENCODER_SMOKE_MEL_HOP_SAMPLES);
    let mel = whisper_log_mel_spectrogram_16khz_mono_v0(
        &prepared_audio.samples_f32,
        mel_bins,
        target_frames,
    )
    .map_err(|error| format!("real mel frontend failed: {error}"))?;
    let shape = mel.layout().shape();
    if shape.len() != 3 || shape[0] != 1 || shape[1] != mel_bins {
        return Err(format!(
            "real mel frontend returned invalid shape {:?}, expected [1, {}, *]",
            shape, mel_bins
        ));
    }
    let mel_frames = shape[2];
    let mel_values = mel.data();
    if mel_values.iter().any(|value| !value.is_finite()) {
        return Err("real mel frontend produced non-finite values".to_string());
    }
    let mut values_f32 = vec![0.0_f32; mel_bins * mel_frames];
    for frame_idx in 0..mel_frames {
        for mel_idx in 0..mel_bins {
            values_f32[frame_idx * mel_bins + mel_idx] =
                mel_values[mel_idx * mel_frames + frame_idx];
        }
    }
    Ok(TinyWhisperMelSmokeInput {
        source_label: WHISPER_REAL_MEL_SOURCE_LABEL,
        mel_bins,
        mel_frames,
        values_f32,
    })
}

pub fn tiny_whisper_encoder_smoke_real_mel_input_for_default_fixture()
-> Result<TinyWhisperMelSmokeInput, String> {
    tiny_whisper_encoder_smoke_real_mel_input(
        &tiny_whisper_encoder_smoke_prepared_audio(),
        WHISPER_DEFAULT_MELS,
    )
}

pub fn tiny_whisper_decoder_tokenizer_fixture_json_bytes_v0() -> &'static [u8] {
    br#"{
        "version":"1.0",
        "added_tokens":[
            {"id":100,"content":"<|notimestamps|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":101,"content":"<|endoftext|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
        ],
        "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"h":0,"i":1},"merges":[]},
        "post_processor":{"special_tokens":{"<|notimestamps|>":{"ids":[100]},"<|endoftext|>":{"ids":[101]}}}
    }"#
}

pub const fn tiny_whisper_decoder_synthetic_eos_token_id_v0() -> u32 {
    TINY_WHISPER_SYNTHETIC_EOS_TOKEN_ID
}

pub fn tiny_whisper_decoder_synthetic_top1_tokens_eot_path_v0() -> Vec<u32> {
    vec![tiny_whisper_decoder_synthetic_eos_token_id_v0()]
}

pub fn tiny_whisper_decoder_synthetic_top1_tokens_text_path_v0() -> Vec<u32> {
    vec![0, 1, tiny_whisper_decoder_synthetic_eos_token_id_v0()]
}

pub fn tiny_whisper_decoder_synthetic_top1_tokens_no_eot_path_v0() -> Vec<u32> {
    vec![0, 1, 0, 1, 0, 1]
}

pub const fn tiny_whisper_decoder_synthetic_expected_text_v0() -> &'static str {
    TINY_WHISPER_SYNTHETIC_EXPECTED_TEXT
}

pub fn whisper_tiny_real_native_smoke_command_v0() -> String {
    format!(
        "cargo run -p openasr-cli -- transcribe {} --backend native --model-pack {} --format text",
        TINY_WHISPER_REAL_SMOKE_AUDIO_RELATIVE_PATH,
        TINY_WHISPER_REAL_SMOKE_MODEL_PACK_RELATIVE_PATH
    )
}

pub fn run_tiny_whisper_decoder_synthetic_loop_v0(
    top1_tokens: &[u32],
    eos_token_id: u32,
    max_steps: usize,
) -> Vec<u32> {
    let mut generated = Vec::new();
    for token in top1_tokens.iter().copied().take(max_steps) {
        if token == eos_token_id {
            break;
        }
        generated.push(token);
        if synthetic_decoder_repetition_loop_detected_v0(&generated) {
            break;
        }
    }
    generated
}

fn synthetic_decoder_repetition_loop_detected_v0(tokens: &[u32]) -> bool {
    for n in 3..=16 {
        let needed = n * 2;
        if tokens.len() < needed {
            continue;
        }
        let first = &tokens[tokens.len() - needed..tokens.len() - n];
        let second = &tokens[tokens.len() - n..];
        if first == second {
            return true;
        }
    }
    false
}

pub fn assert_tiny_whisper_mel_input_shape_and_finite(
    mel_input: &TinyWhisperMelSmokeInput,
    expected: TinyWhisperEncoderSmokeShape,
) {
    assert_eq!(
        mel_input.mel_bins, expected.mel_bins,
        "mel bin mismatch: expected {}, got {}",
        expected.mel_bins, mel_input.mel_bins
    );
    assert_eq!(
        mel_input.mel_frames, expected.mel_frames,
        "mel frame mismatch: expected {}, got {}",
        expected.mel_frames, mel_input.mel_frames
    );
    assert_eq!(
        mel_input.values_f32.len(),
        mel_input.mel_bins * mel_input.mel_frames,
        "mel value count mismatch: expected {}, got {}",
        mel_input.mel_bins * mel_input.mel_frames,
        mel_input.values_f32.len()
    );
    assert!(
        mel_input.values_f32.iter().all(|value| value.is_finite()),
        "mel values contain non-finite values"
    );
}

pub fn assert_tiny_whisper_encoder_output_shape_and_finite(
    values: &[f32],
    expected: TinyWhisperEncoderSmokeShape,
) {
    assert_eq!(
        values.len(),
        expected.output_elements(),
        "encoder output length mismatch: expected {} (frames={} hidden={}), got {}",
        expected.output_elements(),
        expected.output_frames,
        expected.hidden_size,
        values.len()
    );
    assert!(
        values.iter().all(|value| value.is_finite()),
        "encoder output contains non-finite values"
    );
}

pub fn classify_whisper_execution_failure_stage(message: &str) -> WhisperExecutionFailureStage {
    if message.contains("missing required GGUF metadata key")
        || message.contains("metadata '")
        || message.contains("tokenizer is missing required key")
        || message.contains("requires adapter")
    {
        return WhisperExecutionFailureStage::MetadataPreflight;
    }
    if message.contains("encoder prelude graph executed")
        && message.contains("encoder graph executed")
    {
        return WhisperExecutionFailureStage::EncoderExecuted;
    }
    if message.contains("missing required GGUF tensor")
        || message.contains("failed binding validation")
        || message.contains("shape=")
        || message.contains("type '")
    {
        return WhisperExecutionFailureStage::TensorBindingPreflight;
    }
    if message.contains("prepared audio is invalid")
        || message.contains("mel/input preparation seam failed")
        || message.contains("mel feature extraction failed")
        || message.contains("real mel frontend")
        || message.contains("sample_rate_hz=")
        || message.contains("channels=")
        || message.contains("samples_f32")
        || message.contains("non-finite")
    {
        return WhisperExecutionFailureStage::MelFeature;
    }
    if message.contains("full whisper encoder/decoder graph is not implemented yet")
        || message.contains("graph path is not implemented yet")
        || message.contains("decoder/tokenizer path is not implemented yet")
        || message.contains("decoder loop + tokenizer integration are not implemented yet")
        || message.contains("whisper greedy decode reached max_generated_tokens=")
    {
        return WhisperExecutionFailureStage::DecoderTokenizerPending;
    }
    if message.contains("encoder prelude primitive") || message.contains("encoder prelude graph") {
        return WhisperExecutionFailureStage::EncoderPrelude;
    }
    if message.contains("encoder graph primitive")
        || message.contains("encoder graph execution failed")
        || message.contains("encoder graph binding seam")
    {
        return WhisperExecutionFailureStage::EncoderGraph;
    }
    WhisperExecutionFailureStage::Unknown
}

fn whisper_conv_output_frames(
    input_frames: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    let padded = input_frames.checked_add(padding.checked_mul(2)?)?;
    let receptive = dilation
        .checked_mul(kernel_size.saturating_sub(1))?
        .checked_add(1)?;
    if padded < receptive {
        return None;
    }
    let numer = padded.checked_sub(receptive)?;
    let output = numer.checked_div(stride)?.checked_add(1)?;
    (output > 0).then_some(output)
}

fn whisper_required_tensor_anchors_for_layers(
    encoder_layers: usize,
    decoder_layers: usize,
) -> Vec<String> {
    let mut names = WHISPER_REQUIRED_TENSOR_ANCHORS_FOR_SKELETON
        .iter()
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    if encoder_layers > 0 {
        names.push(format!(
            "model.encoder.layers.{}.self_attn.q_proj.weight",
            encoder_layers.saturating_sub(1)
        ));
    }
    if decoder_layers > 0 {
        names.push(format!(
            "model.decoder.layers.{}.self_attn.q_proj.weight",
            decoder_layers.saturating_sub(1)
        ));
    }
    names
}

fn whisper_required_gguf_binding_tensors(
    encoder_layers: usize,
    decoder_layers: usize,
) -> Vec<String> {
    let mut names = BTreeSet::from([
        "model.encoder.conv1.weight".to_string(),
        "model.encoder.conv1.bias".to_string(),
        "model.encoder.conv2.weight".to_string(),
        "model.encoder.conv2.bias".to_string(),
        "model.encoder.embed_positions.weight".to_string(),
        "model.encoder.layer_norm.weight".to_string(),
        "model.encoder.layer_norm.bias".to_string(),
        "model.decoder.embed_positions.weight".to_string(),
        "model.decoder.embed_tokens.weight".to_string(),
        "model.decoder.layer_norm.weight".to_string(),
        "model.decoder.layer_norm.bias".to_string(),
    ]);

    let encoder_suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.out_proj.weight",
        "self_attn.out_proj.bias",
        "self_attn_layer_norm.weight",
        "self_attn_layer_norm.bias",
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
        "final_layer_norm.weight",
        "final_layer_norm.bias",
    ];
    for layer_idx in 0..encoder_layers {
        for suffix in encoder_suffixes {
            names.insert(format!("model.encoder.layers.{layer_idx}.{suffix}"));
        }
    }

    let decoder_suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.out_proj.weight",
        "self_attn.out_proj.bias",
        "self_attn_layer_norm.weight",
        "self_attn_layer_norm.bias",
        "encoder_attn.q_proj.weight",
        "encoder_attn.q_proj.bias",
        "encoder_attn.k_proj.weight",
        "encoder_attn.k_proj.bias",
        "encoder_attn.v_proj.weight",
        "encoder_attn.v_proj.bias",
        "encoder_attn.out_proj.weight",
        "encoder_attn.out_proj.bias",
        "encoder_attn_layer_norm.weight",
        "encoder_attn_layer_norm.bias",
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
        "final_layer_norm.weight",
        "final_layer_norm.bias",
    ];
    for layer_idx in 0..decoder_layers {
        for suffix in decoder_suffixes {
            names.insert(format!("model.decoder.layers.{layer_idx}.{suffix}"));
        }
    }

    names.into_iter().collect()
}

pub fn write_tiny_gguf_runtime_source(
    path: impl AsRef<Path>,
    spec: &TinyGgufFixtureSpec,
) -> io::Result<()> {
    let tensor_entries = spec
        .tensor_names
        .iter()
        .map(|tensor_name| {
            let dims = spec
                .tensor_dims
                .get(tensor_name)
                .cloned()
                .filter(|dims| !dims.is_empty())
                .unwrap_or_else(|| vec![1_u64]);
            let ggml_type = spec
                .tensor_types
                .get(tensor_name)
                .copied()
                .unwrap_or(GGML_TYPE_F32);
            TinyGgufTensorEntry {
                name: tensor_name.clone(),
                dims,
                ggml_type,
            }
        })
        .collect::<Vec<_>>();

    let mut bytes = Vec::new();
    bytes.extend_from_slice(GGUF_MAGIC);
    bytes.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    bytes.extend_from_slice(&(tensor_entries.len() as u64).to_le_bytes());
    bytes.extend_from_slice(
        &((spec.metadata.len() + spec.metadata_string_arrays.len()) as u64).to_le_bytes(),
    );
    for (key, value) in &spec.metadata {
        push_gguf_string(&mut bytes, key);
        bytes.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        push_gguf_string(&mut bytes, value);
    }
    for (key, values) in &spec.metadata_string_arrays {
        push_gguf_string(&mut bytes, key);
        bytes.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        bytes.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            push_gguf_string(&mut bytes, value);
        }
    }

    let tensor_payload_sizes = tensor_entries
        .iter()
        .map(|tensor| payload_size_for_tensor(tensor.ggml_type, &tensor.dims))
        .collect::<Vec<_>>();

    let mut running_offset: u64 = 0;
    for (tensor_index, tensor) in tensor_entries.iter().enumerate() {
        push_gguf_string(&mut bytes, &tensor.name);
        bytes.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in &tensor.dims {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&tensor.ggml_type.to_le_bytes());
        bytes.extend_from_slice(&running_offset.to_le_bytes());
        running_offset = align_up_u64(
            running_offset + tensor_payload_sizes[tensor_index],
            GGUF_DEFAULT_ALIGNMENT as u64,
        );
    }

    let aligned_length = align_up(bytes.len(), GGUF_DEFAULT_ALIGNMENT);
    bytes.resize(aligned_length, 0);
    for (tensor_index, tensor) in tensor_entries.iter().enumerate() {
        let payload = deterministic_tensor_payload(tensor, tensor_index);
        bytes.extend_from_slice(&payload);
        debug_assert_eq!(payload.len() as u64, tensor_payload_sizes[tensor_index]);
        let next_aligned = align_up(bytes.len(), GGUF_DEFAULT_ALIGNMENT);
        bytes.resize(next_aligned, 0);
    }

    fs::write(path, bytes)
}

pub fn write_reserved_oasr_container(path: impl AsRef<Path>) -> io::Result<()> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(RESERVED_OASR_MAGIC);
    bytes.extend_from_slice(b"fixture-reserved-container");
    fs::write(path, bytes)
}

/// Writes `contents` to `path` plus an adjacent, matching LOCAL-catalog
/// signature manifest signed with the public, non-secret local-dev catalog
/// key (`catalog_security::CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID` /
/// `LOCAL_CATALOG_DEV_SIGNING_KEY_SEED_HEX`). A local `file://`/filesystem
/// catalog source now requires a valid sidecar signature just like a
/// production HTTPS catalog does (see `registry::load_model_catalog`), so any
/// test that loads a local catalog fixture must go through this helper (or
/// deliberately omit/break the sidecar to exercise the fail-closed path).
///
/// Signs for the exact `file://<path>` catalog_url the caller will pass as
/// `catalog_url`/`--catalog-url`. Call again (bumping `epoch` is not required,
/// only monotonic-or-equal) after any in-place mutation of `path`'s contents:
/// a stale sidecar is treated as tampering, not a no-op.
pub fn write_local_dev_signed_catalog(path: &Path, contents: &str, epoch: u64) {
    fs::write(path, contents).expect("write local catalog test fixture");
    let catalog_url = format!("file://{}", path.display());
    let manifest = crate::catalog_security::render_catalog_signature_manifest(
        contents,
        &catalog_url,
        epoch,
        crate::catalog_security::CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID,
        crate::catalog_security::LOCAL_CATALOG_DEV_SIGNING_KEY_SEED_HEX,
    )
    .expect("sign local catalog test fixture with the dev key");
    let signature_path = path.with_file_name(crate::catalog_security::CATALOG_SIGNATURE_FILE_NAME);
    fs::write(signature_path, manifest).expect("write local catalog signature test fixture");
}

fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment > 0);
    (value + alignment - 1) & !(alignment - 1)
}

fn align_up_u64(value: u64, alignment: u64) -> u64 {
    debug_assert!(alignment > 0);
    (value + alignment - 1) & !(alignment - 1)
}

fn cohere_fixture_conv_out_dim(input: u64, kernel: u64, stride: u64, padding: u64) -> u64 {
    input
        .saturating_add(padding.saturating_mul(2))
        .saturating_sub(kernel)
        .checked_div(stride.max(1))
        .unwrap_or(0)
        .saturating_add(1)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TinyGgufTensorEntry {
    name: String,
    dims: Vec<u64>,
    ggml_type: i32,
}

fn payload_size_for_tensor(ggml_type: i32, dims: &[u64]) -> u64 {
    let elements = dims
        .iter()
        .fold(1_u64, |elements, dim| elements.saturating_mul(*dim));
    match ggml_type {
        GGML_TYPE_F16 => elements.saturating_mul(2),
        GGML_TYPE_F32 => elements.saturating_mul(4),
        _ => elements.saturating_mul(4),
    }
}

fn deterministic_tensor_payload(tensor: &TinyGgufTensorEntry, tensor_index: usize) -> Vec<u8> {
    let num_elements = tensor
        .dims
        .iter()
        .fold(1_u64, |acc, dim| acc.saturating_mul(*dim));
    let seed = deterministic_tensor_seed(&tensor.name, tensor_index);
    match tensor.ggml_type {
        GGML_TYPE_F32 => deterministic_f32_payload(&tensor.name, seed, num_elements),
        GGML_TYPE_F16 => deterministic_f16_payload(seed, num_elements),
        _ => vec![0_u8; payload_size_for_tensor(tensor.ggml_type, &tensor.dims) as usize],
    }
}

fn deterministic_tensor_seed(tensor_name: &str, tensor_index: usize) -> u64 {
    let mut hash = 14_695_981_039_346_656_037_u64;
    for byte in tensor_name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(1_099_511_628_211_u64);
    }
    hash ^ ((tensor_index as u64).wrapping_mul(2_862_933_555_777_941_757_u64))
}

fn deterministic_f32_payload(tensor_name: &str, seed: u64, num_elements: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((num_elements as usize).saturating_mul(4));
    for index in 0..num_elements {
        let value = deterministic_f32_value(tensor_name, seed, index, num_elements);
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn deterministic_f32_value(tensor_name: &str, seed: u64, index: u64, num_elements: u64) -> f32 {
    if matches!(tensor_name, "fe.window" | "audio.mel_window") {
        if num_elements <= 1 {
            return 1.0;
        }
        let phase = index as f32 / (num_elements - 1) as f32;
        return (std::f32::consts::PI * phase).sin().powi(2).max(1.0e-3);
    }
    if matches!(tensor_name, "fe.mel_fb" | "audio.mel_filters") {
        let bucket = (seed.wrapping_add(index.wrapping_mul(17)) % 31) as f32;
        return (bucket + 1.0) / 64.0;
    }
    if tensor_name.ends_with(".bn.var") {
        let bucket = (seed.wrapping_add(index.wrapping_mul(13)) % 19) as f32;
        return 0.5 + bucket / 32.0;
    }
    let mixed = seed
        .wrapping_add(index.wrapping_mul(1_103_515_245_u64))
        .wrapping_add(12_345);
    let centered = (mixed % 2_049_u64) as i32 - 1_024;
    centered as f32 / 256.0
}

fn deterministic_f16_payload(seed: u64, num_elements: u64) -> Vec<u8> {
    const F16_FINITE_PATTERN: [u16; 8] = [
        0x3C00, // 1.0
        0x3800, // 0.5
        0x4000, // 2.0
        0xBC00, // -1.0
        0x3555, // ~0.333
        0x3A00, // 0.75
        0x3400, // 0.25
        0xC000, // -2.0
    ];
    let mut bytes = Vec::with_capacity((num_elements as usize).saturating_mul(2));
    for index in 0..num_elements {
        let pattern_idx = (seed.wrapping_add(index) as usize) % F16_FINITE_PATTERN.len();
        bytes.extend_from_slice(&F16_FINITE_PATTERN[pattern_idx].to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_gguf_metadata, read_gguf_tensor_index};
    use std::fs;
    use tempfile::NamedTempFile;

    #[test]
    fn whisper_graph_ready_fixture_includes_anchor_and_binding_tensors() {
        let spec =
            TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-runtime-fixture");
        assert_eq!(
            spec.metadata
                .get("general.architecture")
                .map(String::as_str),
            Some("whisper")
        );
        assert!(
            spec.tensor_names
                .contains(&"model.encoder.layers.0.self_attn.q_proj.weight".to_string())
        );
        assert!(
            spec.tensor_names
                .contains(&"model.decoder.layers.0.encoder_attn.q_proj.bias".to_string())
        );
        assert!(
            spec.tensor_names
                .contains(&"model.decoder.layers.0.fc1.weight".to_string())
        );
        assert!(
            spec.tensor_names
                .contains(&"model.encoder.layers.0.self_attn.k_proj.bias".to_string())
        );
    }

    #[test]
    fn whisper_fixture_supports_tensor_alias_and_missing_scenarios() {
        let canonical = "model.decoder.embed_tokens.weight";
        let alias = "model.decoder.token_embedding.weight";
        let spec =
            TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-runtime-fixture")
                .with_tensor_alias(canonical, alias)
                .without_tensor("model.encoder.conv1.weight");

        assert!(!spec.tensor_names.contains(&canonical.to_string()));
        assert!(spec.tensor_names.contains(&alias.to_string()));
        assert!(
            !spec
                .tensor_names
                .contains(&"model.encoder.conv1.weight".to_string())
        );
    }

    #[test]
    fn whisper_fixture_can_model_layer_tensor_mismatch() {
        let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_layer_count_mismatch(
            "whisper-runtime-fixture",
            2,
            2,
        );

        assert!(
            !spec
                .tensor_names
                .contains(&"model.encoder.layers.1.self_attn.q_proj.weight".to_string())
        );
        assert!(
            !spec
                .tensor_names
                .contains(&"model.decoder.layers.1.self_attn.q_proj.weight".to_string())
        );
    }

    #[test]
    fn whisper_graph_ready_fixture_sets_prelude_tensor_shapes() {
        let spec =
            TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-runtime-fixture");
        assert_eq!(
            spec.tensor_dims.get("model.encoder.conv1.weight"),
            Some(&vec![3_u64, 4_u64, 8_u64])
        );
        assert_eq!(
            spec.tensor_dims.get("model.encoder.conv2.bias"),
            Some(&vec![8_u64])
        );
        assert_eq!(
            spec.tensor_dims.get("model.encoder.embed_positions.weight"),
            Some(&vec![128_u64, 8_u64])
        );
        assert_eq!(
            spec.tensor_dims.get("model.encoder.layers.0.fc1.weight"),
            Some(&vec![32_u64, 8_u64])
        );
        assert_eq!(
            spec.tensor_dims.get("model.encoder.layers.0.fc2.weight"),
            Some(&vec![8_u64, 32_u64])
        );
    }

    #[test]
    fn whisper_fixture_helpers_cover_missing_alias_shape_and_layer_mismatch() {
        let spec =
            TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-runtime-fixture")
                .with_whisper_missing_required_tensor("model.encoder.conv1.weight")
                .with_whisper_required_tensor_alias(
                    "model.decoder.embed_tokens.weight",
                    "model.decoder.token_embedding.weight",
                )
                .with_whisper_required_tensor_shape_mismatch("model.encoder.conv2.bias", [2_u64])
                .with_whisper_layer_count_mismatch(2, 3);

        assert!(
            !spec
                .tensor_names
                .contains(&"model.encoder.conv1.weight".to_string())
        );
        assert!(
            spec.tensor_names
                .contains(&"model.decoder.token_embedding.weight".to_string())
        );
        assert_eq!(
            spec.tensor_dims.get("model.encoder.conv2.bias"),
            Some(&vec![2])
        );
        assert_eq!(
            spec.metadata
                .get("whisper.encoder.block_count")
                .map(String::as_str),
            Some("2")
        );
        assert_eq!(
            spec.metadata
                .get("whisper.decoder.block_count")
                .map(String::as_str),
            Some("3")
        );
    }

    #[test]
    fn whisper_fixture_type_mismatch_helper_marks_tensor_non_float() {
        let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_type_mismatch(
            "whisper-runtime-fixture",
            "model.encoder.conv1.bias",
        );
        assert_eq!(
            spec.tensor_types.get("model.encoder.conv1.bias"),
            Some(&GGML_TYPE_I32)
        );
    }

    #[test]
    fn cohere_runtime_ready_fixture_sets_graph_metadata_and_required_tensors() {
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        assert_eq!(
            spec.metadata
                .get("general.architecture")
                .map(String::as_str),
            Some("cohere-transcribe")
        );
        assert_eq!(
            spec.metadata
                .get("cohere_transcribe.encoder.n_layers")
                .map(String::as_str),
            Some("2")
        );
        assert!(
            spec.tensor_names.contains(&"fe.mel_fb".to_string()),
            "frontend mel filter must exist"
        );
        assert!(
            spec.tensor_names
                .contains(&"enc.blk.1.conv.pw2.weight".to_string()),
            "second encoder layer tensor must exist"
        );
        assert!(
            spec.tensor_names
                .contains(&"dec.blk.1.cross_o.bias".to_string()),
            "second decoder layer tensor must exist"
        );
        assert_eq!(spec.tensor_dims.get("fe.window"), Some(&vec![400_u64]));
        assert_eq!(
            spec.tensor_dims.get("dec.pos.weight"),
            Some(&vec![32_u64, 16_u64])
        );
    }

    #[test]
    fn cohere_runtime_ready_fixture_roundtrips_through_gguf_index() {
        let file = NamedTempFile::new().expect("temp file");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");

        write_tiny_gguf_runtime_source(file.path(), &spec).expect("write fixture");
        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");

        assert_eq!(
            index.get("fe.mel_fb").map(|tensor| tensor.dims.clone()),
            spec.tensor_dims.get("fe.mel_fb").cloned()
        );
        assert_eq!(
            index
                .get("enc.proj.weight")
                .map(|tensor| tensor.dims.clone()),
            spec.tensor_dims.get("enc.proj.weight").cloned()
        );
        assert_eq!(
            index
                .get("dec.blk.0.ffn_up.weight")
                .map(|tensor| tensor.dims.clone()),
            spec.tensor_dims.get("dec.blk.0.ffn_up.weight").cloned()
        );
    }

    #[test]
    fn tiny_gguf_writer_roundtrips_string_array_metadata() {
        let file = NamedTempFile::new().expect("temp file");
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");

        write_tiny_gguf_runtime_source(file.path(), &spec).expect("write fixture");
        let metadata = read_gguf_metadata(file.path()).expect("read metadata");

        assert_eq!(metadata.get_string("tokenizer.ggml.model"), Some("llama"));
        assert_eq!(
            metadata.get_string_array("tokenizer.ggml.tokens"),
            spec.metadata_string_arrays
                .get("tokenizer.ggml.tokens")
                .map(Vec::as_slice)
        );
    }

    #[test]
    fn tiny_whisper_encoder_smoke_helpers_produce_small_deterministic_shape() {
        let shape = tiny_whisper_encoder_smoke_shape_for_default_fixture();
        assert_eq!(shape.mel_bins, WHISPER_DEFAULT_MELS);
        assert_eq!(shape.hidden_size, WHISPER_DEFAULT_HIDDEN_SIZE);
        assert_eq!(shape.mel_frames, 3);
        assert_eq!(shape.output_frames, 2);
        assert_eq!(shape.output_elements(), 16);

        let audio = tiny_whisper_encoder_smoke_prepared_audio();
        assert_eq!(audio.sample_rate_hz, 16_000);
        assert_eq!(audio.channels, 1);
        assert_eq!(
            audio.samples_f32.len(),
            WHISPER_TINY_ENCODER_SMOKE_AUDIO_SAMPLES
        );
        assert!(audio.samples_f32.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn tiny_whisper_encoder_real_mel_helper_is_deterministic_and_finite() {
        let shape = tiny_whisper_encoder_smoke_shape_for_default_fixture();
        let mel_a = tiny_whisper_encoder_smoke_real_mel_input_for_default_fixture()
            .expect("real mel input for smoke fixture");
        let mel_b = tiny_whisper_encoder_smoke_real_mel_input_for_default_fixture()
            .expect("real mel input for smoke fixture");

        assert_tiny_whisper_mel_input_shape_and_finite(&mel_a, shape);
        assert_tiny_whisper_mel_input_shape_and_finite(&mel_b, shape);
        assert_eq!(mel_a, mel_b, "real mel helper must stay deterministic");
    }

    #[test]
    fn tiny_whisper_encoder_output_assertion_checks_shape_and_finite() {
        let shape = tiny_whisper_encoder_smoke_shape_for_default_fixture();
        let output = vec![0.25_f32; shape.output_elements()];
        assert_tiny_whisper_encoder_output_shape_and_finite(&output, shape);
    }

    #[test]
    fn tiny_whisper_decoder_tokenizer_fixture_is_stable_json() {
        let fixture = tiny_whisper_decoder_tokenizer_fixture_json_bytes_v0();
        let parsed = serde_json::from_slice::<serde_json::Value>(fixture).expect("valid json");
        assert_eq!(
            parsed
                .pointer("/model/vocab/h")
                .and_then(serde_json::Value::as_u64),
            Some(0)
        );
        assert_eq!(
            parsed
                .pointer("/post_processor/special_tokens/<|endoftext|>/ids/0")
                .and_then(serde_json::Value::as_u64),
            Some(101)
        );
    }

    #[test]
    fn synthetic_decoder_loop_stops_immediately_on_eot_path() {
        let generated = run_tiny_whisper_decoder_synthetic_loop_v0(
            &tiny_whisper_decoder_synthetic_top1_tokens_eot_path_v0(),
            tiny_whisper_decoder_synthetic_eos_token_id_v0(),
            8,
        );
        assert!(generated.is_empty(), "eot path should emit no text tokens");
    }

    #[test]
    fn synthetic_decoder_loop_emits_text_tokens_then_stops_on_eot() {
        let generated = run_tiny_whisper_decoder_synthetic_loop_v0(
            &tiny_whisper_decoder_synthetic_top1_tokens_text_path_v0(),
            tiny_whisper_decoder_synthetic_eos_token_id_v0(),
            8,
        );
        assert_eq!(generated, vec![0, 1]);
        assert_eq!(tiny_whisper_decoder_synthetic_expected_text_v0(), "hi");
    }

    #[test]
    fn synthetic_decoder_no_eot_path_stops_on_max_steps_and_stays_fail_closed() {
        let generated = run_tiny_whisper_decoder_synthetic_loop_v0(
            &tiny_whisper_decoder_synthetic_top1_tokens_no_eot_path_v0(),
            tiny_whisper_decoder_synthetic_eos_token_id_v0(),
            4,
        );
        assert_eq!(generated, vec![0, 1, 0, 1]);
        assert_eq!(
            classify_whisper_execution_failure_stage(
                "whisper greedy decode reached max_generated_tokens=4 before EOT"
            ),
            WhisperExecutionFailureStage::DecoderTokenizerPending
        );
    }

    #[test]
    fn tiny_whisper_real_native_smoke_command_is_stable() {
        let command = whisper_tiny_real_native_smoke_command_v0();
        assert!(command.contains(TINY_WHISPER_REAL_SMOKE_MODEL_PACK_RELATIVE_PATH));
        assert!(command.contains(TINY_WHISPER_REAL_SMOKE_AUDIO_RELATIVE_PATH));
        assert!(command.contains("--backend native"));
    }

    #[test]
    fn whisper_execution_stage_classifier_distinguishes_fail_closed_boundaries() {
        assert_eq!(
            classify_whisper_execution_failure_stage(
                "whisper ggml executor missing required GGUF metadata key 'general.architecture'"
            ),
            WhisperExecutionFailureStage::MetadataPreflight
        );
        assert_eq!(
            classify_whisper_execution_failure_stage(
                "whisper ggml executor tensor 'model.encoder.conv2.bias' failed binding validation: shape=[2] (expected rank-1)"
            ),
            WhisperExecutionFailureStage::TensorBindingPreflight
        );
        assert_eq!(
            classify_whisper_execution_failure_stage(
                "whisper ggml executor mel/input preparation seam failed: sample_rate_hz=8000 (expected 16000)"
            ),
            WhisperExecutionFailureStage::MelFeature
        );
        assert_eq!(
            classify_whisper_execution_failure_stage(
                "whisper ggml executor encoder prelude primitive 'ggml_conv_1d' is unsupported: unavailable"
            ),
            WhisperExecutionFailureStage::EncoderPrelude
        );
        assert_eq!(
            classify_whisper_execution_failure_stage(
                "whisper ggml executor encoder graph primitive 'encoder.self_attn.qk_attention' is unsupported: unavailable"
            ),
            WhisperExecutionFailureStage::EncoderGraph
        );
        assert_eq!(
            classify_whisper_execution_failure_stage(
                "whisper ggml executor decoder/tokenizer path is not implemented yet: encoder prelude graph executed by 'x' (output_hidden_shape=2x8, input_mel_shape=3x4); encoder graph executed by 'y' (layers=1, output_hidden_shape=2x8); decoder loop + tokenizer integration are not implemented yet"
            ),
            WhisperExecutionFailureStage::EncoderExecuted
        );
        assert_eq!(
            classify_whisper_execution_failure_stage(
                "whisper ggml executor decoder/tokenizer path is not implemented yet: decoder loop + tokenizer integration are not implemented yet"
            ),
            WhisperExecutionFailureStage::DecoderTokenizerPending
        );
    }

    #[test]
    fn tiny_gguf_writer_persists_shape_and_f16_tensor_type() {
        let file = NamedTempFile::new().expect("temp file");
        let spec =
            TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-runtime-fixture")
                .with_tensor_f16("model.encoder.conv1.weight");

        write_tiny_gguf_runtime_source(file.path(), &spec).expect("write fixture");
        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");

        let conv1 = index
            .get("model.encoder.conv1.weight")
            .expect("conv1 tensor must exist");
        assert_eq!(conv1.dims, vec![3, 4, 8]);
        assert_eq!(conv1.ggml_type, GGML_TYPE_F16);
        assert_eq!(conv1.size_bytes, 192);
    }

    #[test]
    fn tiny_gguf_writer_emits_deterministic_f32_and_f16_payloads() {
        let file_a = NamedTempFile::new().expect("temp file");
        let file_b = NamedTempFile::new().expect("temp file");
        let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-runtime")
            .with_tensor_shape("fixture.tensor", [4_u64])
            .with_tensor_shape("model.encoder.conv1.weight", [4_u64])
            .with_tensor_f16("model.encoder.conv1.weight");

        write_tiny_gguf_runtime_source(file_a.path(), &spec).expect("write fixture");
        write_tiny_gguf_runtime_source(file_b.path(), &spec).expect("write fixture");

        let bytes_a = fs::read(file_a.path()).expect("read fixture");
        let bytes_b = fs::read(file_b.path()).expect("read fixture");
        assert_eq!(bytes_a, bytes_b, "fixture bytes must be deterministic");

        let index = read_gguf_tensor_index(file_a.path()).expect("read tensor index");
        let data_start = index.data_section_offset_bytes() as usize;
        let fixture_tensor = index.get("fixture.tensor").expect("fixture tensor exists");
        let fixture_offset = data_start + fixture_tensor.offset_bytes as usize;
        let fixture_bytes =
            &bytes_a[fixture_offset..fixture_offset + fixture_tensor.size_bytes as usize];
        for chunk in fixture_bytes.chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            assert!(value.is_finite(), "payload must stay finite");
        }

        let conv1 = index
            .get("model.encoder.conv1.weight")
            .expect("conv1 tensor exists");
        let conv1_offset = data_start + conv1.offset_bytes as usize;
        let conv1_bytes = &bytes_a[conv1_offset..conv1_offset + conv1.size_bytes as usize];
        assert!(
            conv1_bytes.chunks_exact(2).any(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                matches!(
                    bits,
                    0x3C00 | 0x3800 | 0x4000 | 0xBC00 | 0x3555 | 0x3A00 | 0x3400 | 0xC000
                )
            }),
            "f16 payload should contain finite deterministic pattern bits"
        );
    }
}
