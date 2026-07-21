//! Per-architecture GGUF/`.oasr` hparam key constants and schema tables.
//!
//! Each architecture declares its canonical required-hparam key set as a flat
//! `&[&str]` table — the "architecture is data" minimum: onboarding a new
//! variant of an existing architecture means listing keys here, not writing
//! code. Aliases (legacy key names carrying the same value) and optional
//! consistency-check keys are resolved by the per-arch runtime contract; the
//! schema records only the canonical keys every package must provide.
//!
//! A richer typed/optional schema (cf. llama.cpp `LLM_KV`) can replace these
//! slices once an architecture actually needs type-specific or optional
//! hparams; today every hparam is a required `u64`, so a key list is the
//! honest representation.

// ── Whisper hparam keys ───────────────────────────────────────────────────────

pub(crate) const WHISPER_ENCODER_BLOCK_COUNT_KEY: &str = "whisper.encoder.block_count";
pub(crate) const WHISPER_DECODER_BLOCK_COUNT_KEY: &str = "whisper.decoder.block_count";
pub(crate) const WHISPER_ENCODER_CONTEXT_LENGTH_KEY: &str = "whisper.encoder.context_length";
pub(crate) const WHISPER_ENCODER_EMBEDDING_LENGTH_KEY: &str = "whisper.encoder.embedding_length";
pub(crate) const WHISPER_ENCODER_HEAD_COUNT_KEY: &str = "whisper.encoder.attention.head_count";
pub(crate) const WHISPER_DECODER_EMBEDDING_LENGTH_KEY: &str = "whisper.decoder.embedding_length";
pub(crate) const WHISPER_DECODER_HEAD_COUNT_KEY: &str = "whisper.decoder.attention.head_count";
pub(crate) const WHISPER_DECODER_CONTEXT_LENGTH_KEY: &str = "whisper.decoder.context_length";
pub(crate) const WHISPER_ENCODER_MELS_COUNT_KEY: &str = "whisper.encoder.mels_count";
pub(crate) const WHISPER_VOCAB_SIZE_KEY: &str = "whisper.vocab_size";

pub(crate) static WHISPER_HPARAM_SCHEMA: &[&str] = &[
    WHISPER_ENCODER_BLOCK_COUNT_KEY,
    WHISPER_ENCODER_EMBEDDING_LENGTH_KEY,
    WHISPER_ENCODER_HEAD_COUNT_KEY,
    WHISPER_ENCODER_CONTEXT_LENGTH_KEY,
    WHISPER_ENCODER_MELS_COUNT_KEY,
    WHISPER_DECODER_BLOCK_COUNT_KEY,
    WHISPER_DECODER_EMBEDDING_LENGTH_KEY,
    WHISPER_DECODER_HEAD_COUNT_KEY,
    WHISPER_DECODER_CONTEXT_LENGTH_KEY,
    WHISPER_VOCAB_SIZE_KEY,
];

// ── Qwen3-ASR hparam keys ────────────────────────────────────────────────────

pub(crate) const QWEN3_ARCHITECTURE_VALUE: &str = "qwen3-asr";
pub(crate) const QWEN3_SAMPLE_RATE_KEY: &str = "qwen3-asr.sample_rate";
pub(crate) const QWEN3_MELS_COUNT_KEY: &str = "qwen3-asr.n_mels";
pub(crate) const QWEN3_N_FFT_KEY: &str = "qwen3-asr.n_fft";
pub(crate) const QWEN3_WIN_LENGTH_KEY: &str = "qwen3-asr.win_length";
pub(crate) const QWEN3_HOP_LENGTH_KEY: &str = "qwen3-asr.hop_length";
pub(crate) const QWEN3_AUDIO_LAYERS_KEY: &str = "qwen3-asr.audio.n_layers";
pub(crate) const QWEN3_AUDIO_D_MODEL_KEY: &str = "qwen3-asr.audio.d_model";
pub(crate) const QWEN3_AUDIO_HEADS_KEY: &str = "qwen3-asr.audio.n_heads";
pub(crate) const QWEN3_LLM_LAYERS_KEY: &str = "qwen3-asr.llm.n_layers";
pub(crate) const QWEN3_LLM_D_MODEL_KEY: &str = "qwen3-asr.llm.d_model";
pub(crate) const QWEN3_LLM_HEADS_KEY: &str = "qwen3-asr.llm.n_heads";
pub(crate) const QWEN3_LLM_KV_HEADS_KEY: &str = "qwen3-asr.llm.n_kv_heads";
pub(crate) const QWEN3_LLM_HEAD_DIM_KEY: &str = "qwen3-asr.llm.head_dim";
pub(crate) const QWEN3_LLM_VOCAB_SIZE_KEY: &str = "qwen3-asr.llm.vocab_size";
pub(crate) const QWEN3_LLM_MAX_POSITIONS_KEY: &str = "qwen3-asr.llm.max_pos";
pub(crate) const QWEN3_AUDIO_START_TOKEN_ID_KEY: &str = "qwen3-asr.audio_start_token_id";
pub(crate) const QWEN3_AUDIO_END_TOKEN_ID_KEY: &str = "qwen3-asr.audio_end_token_id";
pub(crate) const QWEN3_AUDIO_PAD_TOKEN_ID_KEY: &str = "qwen3-asr.audio_pad_token_id";
pub(crate) const QWEN3_EOS_TOKEN_ID_KEY: &str = "qwen3-asr.eos_token_id";
pub(crate) const QWEN3_PAD_TOKEN_ID_KEY: &str = "qwen3-asr.pad_token_id";

pub(crate) static QWEN3_ASR_HPARAM_SCHEMA: &[&str] = &[
    QWEN3_SAMPLE_RATE_KEY,
    QWEN3_MELS_COUNT_KEY,
    QWEN3_N_FFT_KEY,
    QWEN3_WIN_LENGTH_KEY,
    QWEN3_HOP_LENGTH_KEY,
    QWEN3_AUDIO_LAYERS_KEY,
    QWEN3_AUDIO_D_MODEL_KEY,
    QWEN3_AUDIO_HEADS_KEY,
    QWEN3_LLM_LAYERS_KEY,
    QWEN3_LLM_D_MODEL_KEY,
    QWEN3_LLM_HEADS_KEY,
    QWEN3_LLM_KV_HEADS_KEY,
    QWEN3_LLM_HEAD_DIM_KEY,
    QWEN3_LLM_VOCAB_SIZE_KEY,
    QWEN3_LLM_MAX_POSITIONS_KEY,
    QWEN3_AUDIO_START_TOKEN_ID_KEY,
    QWEN3_AUDIO_END_TOKEN_ID_KEY,
    QWEN3_AUDIO_PAD_TOKEN_ID_KEY,
    QWEN3_EOS_TOKEN_ID_KEY,
    QWEN3_PAD_TOKEN_ID_KEY,
];

// ── Cohere Transcribe hparam keys ────────────────────────────────────────────

pub(crate) const COHERE_TRANSCRIBE_ARCHITECTURE_VALUE: &str = "cohere-transcribe";
pub(crate) const COHERE_TRANSCRIBE_VOCAB_SIZE_KEY: &str = "cohere_transcribe.vocab_size";
pub(crate) const COHERE_TRANSCRIBE_ENCODER_LAYERS_KEY: &str = "cohere_transcribe.encoder.n_layers";
pub(crate) const COHERE_TRANSCRIBE_ENCODER_D_MODEL_KEY: &str = "cohere_transcribe.encoder.d_model";
pub(crate) const COHERE_TRANSCRIBE_ENCODER_HEADS_KEY: &str = "cohere_transcribe.encoder.n_heads";
pub(crate) const COHERE_TRANSCRIBE_ENCODER_HEAD_DIM_KEY: &str =
    "cohere_transcribe.encoder.head_dim";
pub(crate) const COHERE_TRANSCRIBE_ENCODER_FFN_DIM_KEY: &str = "cohere_transcribe.encoder.ffn_dim";
pub(crate) const COHERE_TRANSCRIBE_ENCODER_CONV_KERNEL_KEY: &str =
    "cohere_transcribe.encoder.conv_kernel";
pub(crate) const COHERE_TRANSCRIBE_DECODER_LAYERS_KEY: &str = "cohere_transcribe.decoder.n_layers";
pub(crate) const COHERE_TRANSCRIBE_DECODER_D_MODEL_KEY: &str = "cohere_transcribe.decoder.d_model";
pub(crate) const COHERE_TRANSCRIBE_DECODER_HEADS_KEY: &str = "cohere_transcribe.decoder.n_heads";
pub(crate) const COHERE_TRANSCRIBE_DECODER_HEAD_DIM_KEY: &str =
    "cohere_transcribe.decoder.head_dim";
pub(crate) const COHERE_TRANSCRIBE_DECODER_FFN_DIM_KEY: &str = "cohere_transcribe.decoder.ffn_dim";
pub(crate) const COHERE_TRANSCRIBE_DECODER_MAX_CONTEXT_KEY: &str =
    "cohere_transcribe.decoder.max_ctx";
pub(crate) const COHERE_TRANSCRIBE_DECODER_START_TOKEN_ID_KEY: &str =
    "cohere_transcribe.decoder.start_token_id";
pub(crate) const COHERE_TRANSCRIBE_AUDIO_SAMPLE_RATE_KEY: &str =
    "cohere_transcribe.audio.sample_rate";
pub(crate) const COHERE_TRANSCRIBE_AUDIO_MELS_COUNT_KEY: &str = "cohere_transcribe.audio.n_mels";
pub(crate) const COHERE_TRANSCRIBE_AUDIO_N_FFT_KEY: &str = "cohere_transcribe.audio.n_fft";
pub(crate) const COHERE_TRANSCRIBE_AUDIO_HOP_LENGTH_KEY: &str =
    "cohere_transcribe.audio.hop_length";
pub(crate) const COHERE_TRANSCRIBE_AUDIO_WIN_LENGTH_KEY: &str =
    "cohere_transcribe.audio.win_length";

pub(crate) static COHERE_TRANSCRIBE_HPARAM_SCHEMA: &[&str] = &[
    COHERE_TRANSCRIBE_VOCAB_SIZE_KEY,
    COHERE_TRANSCRIBE_ENCODER_LAYERS_KEY,
    COHERE_TRANSCRIBE_ENCODER_D_MODEL_KEY,
    COHERE_TRANSCRIBE_ENCODER_HEADS_KEY,
    COHERE_TRANSCRIBE_ENCODER_HEAD_DIM_KEY,
    COHERE_TRANSCRIBE_ENCODER_FFN_DIM_KEY,
    COHERE_TRANSCRIBE_ENCODER_CONV_KERNEL_KEY,
    COHERE_TRANSCRIBE_DECODER_LAYERS_KEY,
    COHERE_TRANSCRIBE_DECODER_D_MODEL_KEY,
    COHERE_TRANSCRIBE_DECODER_HEADS_KEY,
    COHERE_TRANSCRIBE_DECODER_HEAD_DIM_KEY,
    COHERE_TRANSCRIBE_DECODER_FFN_DIM_KEY,
    COHERE_TRANSCRIBE_DECODER_MAX_CONTEXT_KEY,
    COHERE_TRANSCRIBE_DECODER_START_TOKEN_ID_KEY,
    COHERE_TRANSCRIBE_AUDIO_SAMPLE_RATE_KEY,
    COHERE_TRANSCRIBE_AUDIO_MELS_COUNT_KEY,
    COHERE_TRANSCRIBE_AUDIO_N_FFT_KEY,
    COHERE_TRANSCRIBE_AUDIO_HOP_LENGTH_KEY,
    COHERE_TRANSCRIBE_AUDIO_WIN_LENGTH_KEY,
];

// ── parakeet-ctc (FastConformer-CTC) hparam schema (goal-1) ──────────────────
pub(crate) static PARAKEET_CTC_HPARAM_SCHEMA: &[&str] = &[
    "parakeet.n_layers",
    "parakeet.hidden_size",
    "parakeet.n_heads",
    "parakeet.head_dim",
    "parakeet.ffn_dim",
    "parakeet.conv_kernel",
    "parakeet.n_mels",
    "parakeet.subsampling_factor",
    "parakeet.subsampling_channels",
    "parakeet.vocab_size",
    "ctc.blank_token_id",
];

// ── parakeet-tdt (FastConformer + Token-and-Duration Transducer) ─────────────
pub(crate) static PARAKEET_TDT_HPARAM_SCHEMA: &[&str] = &[
    "parakeet-tdt.n_layers",
    "parakeet-tdt.hidden_size",
    "parakeet-tdt.n_heads",
    "parakeet-tdt.head_dim",
    "parakeet-tdt.ffn_dim",
    "parakeet-tdt.conv_kernel",
    "parakeet-tdt.n_mels",
    "parakeet-tdt.subsampling_factor",
    "parakeet-tdt.subsampling_channels",
    "parakeet-tdt.scale_input",
    "parakeet-tdt.vocab_size",
    "parakeet-tdt.blank_token_id",
    "parakeet-tdt.pred_hidden",
    "parakeet-tdt.pred_layers",
    "parakeet-tdt.joint_hidden",
    "parakeet-tdt.n_durations",
    "parakeet-tdt.durations",
    "parakeet-tdt.max_symbols_per_step",
];

// ── sensevoice (FunAudioLLM/SenseVoiceSmall, SAN-M/CTC) hparam schema ────────
pub(crate) static SENSEVOICE_HPARAM_SCHEMA: &[&str] = &[
    "sensevoice.n_layers",
    "sensevoice.tp_layers",
    "sensevoice.d_model",
    "sensevoice.n_heads",
    "sensevoice.ffn_dim",
    "sensevoice.fsmn_kernel",
    "sensevoice.feature_dim",
    "sensevoice.vocab_size",
    "ctc.blank_token_id",
];

// ── firered-aed (FireRedTeam/FireRedASR-AED-L, Conformer AED) hparam schema ──
// Keys mirror `models::firered_aed::runtime_contract`, the required-metadata
// parser the install gate dispatches to for this architecture.
pub(crate) static FIRERED_AED_HPARAM_SCHEMA: &[&str] = &[
    "firered.encoder.n_layers",
    "firered.encoder.d_model",
    "firered.encoder.n_heads",
    "firered.encoder.head_dim",
    "firered.encoder.ffn_dim",
    "firered.encoder.conv_kernel",
    "firered.encoder.subsample_channels",
    "firered.encoder.subsample_out_dim",
    "firered.encoder.feature_dim",
    "firered.encoder.pe_len",
    "firered.decoder.n_layers",
    "firered.decoder.ffn_dim",
    "firered.decoder.pe_len",
    "firered.vocab_size",
    "firered.sos_token_id",
    "firered.eos_token_id",
    "firered.pad_token_id",
];

// ── firered-llm (FireRedTeam/FireRedASR2-LLM, Conformer+Adapter+Qwen2) ───────
// Keys mirror `models::firered_llm::runtime_contract`'s three parsers
// (encoder subset reuses firered-aed's own `firered.encoder.*` schema -- see
// that module's doc comment for why the full firered-aed schema above is NOT
// reused as-is: this family has no AED decoder branch).
pub(crate) static FIRERED_LLM_HPARAM_SCHEMA: &[&str] = &[
    "firered.encoder.n_layers",
    "firered.encoder.d_model",
    "firered.encoder.n_heads",
    "firered.encoder.head_dim",
    "firered.encoder.ffn_dim",
    "firered.encoder.conv_kernel",
    "firered.encoder.subsample_channels",
    "firered.encoder.subsample_out_dim",
    "firered.encoder.feature_dim",
    "firered.encoder.pe_len",
    "firered_llm.adapter.downsample_rate",
    "firered_llm.adapter.llm_dim",
    "firered_llm.llm.n_layers",
    "firered_llm.llm.d_model",
    "firered_llm.llm.n_heads",
    "firered_llm.llm.n_kv_heads",
    "firered_llm.llm.head_dim",
    "firered_llm.llm.ffn_dim",
    "firered_llm.llm.vocab_size",
    "firered_llm.llm.max_positions",
    "firered_llm.llm.chatml_im_start_token_id",
    "firered_llm.llm.chatml_im_end_token_id",
    "firered_llm.llm.endoftext_token_id",
    "firered_llm.llm.speech_token_id",
];

// Keys mirror `models::moss_transcribe_diarize::runtime_contract`'s three
// parsers.
pub(crate) static MOSS_TD_HPARAM_SCHEMA: &[&str] = &[
    "moss_td.encoder.n_layers",
    "moss_td.encoder.d_model",
    "moss_td.encoder.n_heads",
    "moss_td.encoder.ffn_dim",
    "moss_td.encoder.n_mels",
    "moss_td.encoder.max_source_positions",
    "moss_td.adaptor.merge_size",
    "moss_td.adaptor.input_dim",
    "moss_td.llm.n_layers",
    "moss_td.llm.d_model",
    "moss_td.llm.ffn_dim",
    "moss_td.llm.n_heads",
    "moss_td.llm.n_kv_heads",
    "moss_td.llm.head_dim",
    "moss_td.llm.vocab_size",
    "moss_td.llm.max_positions",
    "moss_td.llm.audio_start_token_id",
    "moss_td.llm.audio_end_token_id",
    "moss_td.llm.audio_pad_token_id",
];

// ── mimo-asr (XiaomiMiMo/MiMo-V2.5-ASR + MiMo-Audio-Tokenizer) hparam schema ─
// Keys mirror `models::mimo_asr::runtime_contract`'s parsers. Deliberately
// checks only the top-level presence gate (block counts + special-token ids);
// the parsers themselves enforce the finer cross-field invariants (skip_layer_id
// range, qkv_bias/qk_norm shape, head_dim*n_heads==d_model, ...).
pub(crate) static MIMO_ASR_HPARAM_SCHEMA: &[&str] = &[
    "mimo.llm.block_count",
    "mimo.llm.embedding_length",
    "mimo.llm.attention.head_count",
    "mimo.llm.attention.head_count_kv",
    "mimo.llm.attention.key_length",
    "mimo.llm.feed_forward_length",
    "mimo.llm.vocab_size",
    "mimo.llm.context_length",
    "mimo.llm.attention.layer_norm_rms_epsilon",
    "mimo.llm.rope.freq_base",
    "mimo.llm.attention.qkv_bias",
    "mimo.llm.attention.qk_norm",
    "mimo.audio.channels",
    "mimo.audio.group_size",
    "mimo.inlocal.block_count",
    "mimo.inlocal.embedding_length",
    "mimo.inlocal.attention.head_count",
    "mimo.inlocal.attention.head_dim",
    "mimo.inlocal.feed_forward_length",
    "mimo.inlocal.full_attention",
    "mimo.inlocal.rope.freq_base",
    "mimo.tok.block_count",
    "mimo.tok.embedding_length",
    "mimo.tok.attention.head_count",
    "mimo.tok.feed_forward_length",
    "mimo.tok.encoder.skip_layer_id",
    "mimo.tok.conv.kernel_size",
    "mimo.tok.conv1.stride",
    "mimo.tok.conv2.stride",
    "mimo.tok.down_sample.stride",
    "mimo.tok.rope.freq_base",
    "mimo.tok.rvq.num_quantizers_packed",
    "mimo.tok.rvq.codebook_sizes",
    "mimo.mel.sample_rate",
    "mimo.mel.n_fft",
    "mimo.mel.hop_length",
    "mimo.mel.win_length",
    "mimo.mel.n_mels",
    "mimo.mel.log_clip",
    "mimo.special.eos_id",
    "mimo.special.im_start_id",
    "mimo.special.im_end_id",
    "mimo.special.sosp_id",
    "mimo.special.eosp_id",
    "mimo.special.empty_id",
    "mimo.special.eot_id",
    "mimo.special.eostm_id",
];

// ── wav2vec2-ctc (facebook/wav2vec2-base-960h) hparam schema ─────────────────
pub(crate) static WAV2VEC2_CTC_HPARAM_SCHEMA: &[&str] = &[
    "wav2vec2.n_layers",
    "wav2vec2.hidden_size",
    "wav2vec2.n_heads",
    "wav2vec2.head_dim",
    "wav2vec2.ffn_dim",
    "wav2vec2.vocab_size",
    "wav2vec2.num_conv_pos_embeddings",
    "wav2vec2.num_conv_pos_embedding_groups",
    "ctc.blank_token_id",
];

// ── X-ASR Zipformer transducer (GilgameshWind/X-ASR-zh-en) ───────────────────
pub(crate) static XASR_ZIPFORMER_HPARAM_SCHEMA: &[&str] = &[
    "xasr.num_stacks",
    "xasr.num_encoder_layers",
    "xasr.encoder_dims",
    "xasr.query_head_dims",
    "xasr.value_head_dims",
    "xasr.num_heads",
    "xasr.cnn_module_kernels",
    "xasr.left_context_len",
    "xasr.downsampling_factors",
    "xasr.feature_dim",
    "xasr.decode_chunk_len",
    "xasr.joiner_dim",
    "xasr.decoder_context_size",
    "xasr.vocab_size",
    "xasr.blank_id",
];

// ── dolphin (WeNet E-Branchformer encoder + Transformer decoder + CTC head) ───
// Char tokenizer, CTC/attention joint decode (WeNet attention_rescoring). The
// keys mirror `models::dolphin::runtime_contract`, which is the required-metadata
// parser the install gate dispatches to for this architecture.
pub(crate) static DOLPHIN_HPARAM_SCHEMA: &[&str] = &[
    "dolphin.encoder.n_layers",
    "dolphin.encoder.d_model",
    "dolphin.encoder.n_heads",
    "dolphin.encoder.head_dim",
    "dolphin.encoder.ffn_dim",
    "dolphin.encoder.cgmlp_units",
    "dolphin.encoder.cgmlp_kernel",
    "dolphin.encoder.merge_kernel",
    "dolphin.encoder.feature_dim",
    // `max_ctx` (encoder + decoder): tensor length is authoritative when a
    // baked position-table tensor exists, so these two keys are conditionally
    // (not unconditionally) required -- see
    // `models::dolphin::runtime_contract::resolve_position_table_max_ctx`.
    "dolphin.encoder.max_ctx",
    "dolphin.decoder.n_layers",
    "dolphin.decoder.n_heads",
    "dolphin.decoder.ffn_dim",
    "dolphin.decoder.max_ctx",
    "dolphin.vocab_size",
    "dolphin.sos_token_id",
    "dolphin.eos_token_id",
    "ctc.blank_token_id",
];

// ── moonshine (UsefulSensors, raw-waveform conv-stem + RoPE seq2seq) ──────────
pub(crate) static MOONSHINE_HPARAM_SCHEMA: &[&str] = &[
    "moonshine.vocab_size",
    "moonshine.d_model",
    "moonshine.encoder.n_layers",
    "moonshine.decoder.n_layers",
    "moonshine.n_heads",
    "moonshine.head_dim",
    "moonshine.rotary_dim",
    "moonshine.rope_theta",
    "moonshine.encoder.ffn_dim",
    "moonshine.decoder.ffn_dim",
    "moonshine.decoder.max_ctx",
    "moonshine.decoder.bos_token_id",
    "moonshine.decoder.eos_token_id",
    "moonshine.audio.sample_rate",
];
