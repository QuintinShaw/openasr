//! Convert a local Dolphin WeNet checkpoint (exported `full.safetensors` +
//! `units.txt` char vocab, with `global_cmvn` folded into the encoder tensors)
//! into an OpenASR `.oasr` (GGUF-v0) runtime pack at fp16, q8_0, or q4_k.
//!
//! Naming contract: the encoder/decoder/CTC tensors are stored under their
//! **exact WeNet state-dict names**. At fp16 they keep raw element order (only
//! f32 -> f16 for the rank>=2 weight matrices/convs; the 1-D biases/norms and the
//! CMVN vectors stay f32). This keeps the runtime executor trivial and, crucially,
//! lets it feed the already parity-verified `encoder_graph::encode()` byte-for-byte
//! the same tensor buffers the raw-safetensors parity harness used — the only delta
//! is the fp16 rounding of the weights.
//!
//! Quantization (q8_0/q4_k) targets only the rank-2 `.weight` projection/embed/
//! output matrices. ggml block-quantizes along ne0 (the contiguous axis), so those
//! matrices are stored with their dims **reversed** (`[out, in]` -> `[in, out]`)
//! exactly as the xasr/cohere importers do: that puts the row-major-innermost `in`
//! dimension on ne0, block-aligns it (`in % 32`/`in % 256`), and groups each row's
//! input features into a superblock. The runtime never trusts the stored dims for
//! layout — it re-declares each graph tensor and only consumes the dequantized f32
//! by element count — so a reversed-dim q-tensor dequantizes to the *same* raw
//! element order an fp16 tensor would (round-trip is order-preserving); the only
//! delta is the quantization rounding. Convs (rank>2), position tables, the CMVN
//! vectors, 1-D biases/norms, and the mel filterbank are never quantized (their
//! ne0 is either tiny/unaligned or numerically sensitive) and keep their fp16-mode
//! representation.
//!
//! Baked into the pack:
//!   * every `encoder.*` / `decoder.*` / `ctc.*` tensor,
//!   * the `context_module.*` native deep-biasing hotword tensors (the
//!     `context_extractor` BiLSTM + word embedding, `context_encoder`,
//!     `biasing_layer` cross-attention, `combiner`, `norm_aft_combiner`) --
//!     everything the inference-time fusion in
//!     `models::dolphin::hotword_context` needs. Only the training-only aux CTC
//!     head over hotword context (`context_module.context_decoder*`) is dropped:
//!     it is never exercised at inference (see
//!     `models::dolphin::hotword_context` for the upstream reference),
//!   * the global CMVN mean/istd (already present as `encoder.global_cmvn.*`),
//!   * a kaldi/HTK mel filterbank (`dolphin.mel_filters`) reconstructed from the
//!     `train.yaml` fbank config for the later frontend phase,
//!   * the char tokenizer (`tokenizer.ggml.tokens`, ids in `units.txt` order) —
//!     the full special-token block, so every advertised `<REGION>` dialect code
//!     resolves at request time (no single region baked as the default; the
//!     importer asserts each advertised code's prefix builds against the vocab),
//!   * the runtime scalar contract keys the install gate validates.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::arch::{
    DOLPHIN_AUDIO_FRONTEND_ID, DOLPHIN_DECODE_POLICY_ID, DOLPHIN_GGML_ARCHITECTURE_ID,
    DOLPHIN_MODEL_FAMILY, DOLPHIN_TOKENIZER_ID,
};
use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    read_gguf_tensor_index, write_gguf_file_v0,
};
use crate::models::audio_frontend::mel::{FilterbankConfig, MelPointOrder, MelScale, filterbank};
use crate::models::dolphin::language::{
    DOLPHIN_CN_DIALECT_CODES, DOLPHIN_DEFAULT_LANGUAGE_CODE,
    DOLPHIN_MULTILINGUAL_CATALOG_LANGUAGES, build_dolphin_decode_prefix,
    build_dolphin_multilingual_decode_prefix,
};
use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, decode_safetensors_payload_as_f32, encode_f16_bits_le,
    validate_error, validate_output_pack_extension,
};
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};
use crate::models::pack_quant::{PackQuant, classify_quant_tensor};
use crate::nn::half::f32_to_f16_bits;

// --- E-Branchformer / Transformer configuration -------------------------------
// Every structural hparam (layer counts, d_model, head count/dim, FFN/cgMLP
// widths, conv kernels, decoder max context) is read directly off the
// checkpoint's OWN tensor shapes by `derive_dolphin_architecture` below, not
// hardcoded -- the small.cn (768/12L), cn-dialect-base and multilingual base
// (512/8L) and multilingual small (768/12L, but with a *different*
// encoder FFN width than small.cn: 1536 vs 3072) checkpoints all differ on at
// least one of these axes, so a fixed per-size const block would silently
// mismatch a new size. `FEATURE_DIM` is the one frontend choice this importer
// itself fixes (the mel filterbank it reconstructs), and is cross-checked
// against the checkpoint's own CMVN vector length instead of assumed.
const FEATURE_DIM: usize = 80;

// fbank config from `train.yaml` (`fbank_conf`): 25 ms window, 10 ms shift, 80
// mel bins, 16 kHz. Kaldi rounds the 400-sample window up to the next power of
// two for the FFT.
const SAMPLE_RATE_HZ: u32 = 16_000;
const FRAME_LENGTH_MS: u32 = 25;
const FRAME_SHIFT_MS: u32 = 10;
const FFT_SIZE: usize = 512;
const MEL_LOW_HZ: f32 = 20.0;

/// WeNet state-dict namespaces baked into the runtime pack (in order). The
/// hotword deep-biasing module (`context_module.*`) is included here too --
/// only its training-only aux CTC head (`DROPPED_TENSOR_PREFIXES`) is dropped.
const RUNTIME_TENSOR_PREFIXES: [&str; 4] = ["encoder.", "decoder.", "ctc.", "context_module."];
/// Training-only aux tensors under `context_module.*`: a CTC head over the
/// hotword context embeddings, used only to regularize training. Inference
/// (the deep-biasing fusion in `models::dolphin::hotword_context`) never reads
/// these, so they are dropped to keep the pack lean.
const DROPPED_TENSOR_PREFIXES: [&str; 2] = [
    "context_module.context_decoder.",
    "context_module.context_decoder_ctc_linear.",
];

/// fp16 weights (rank>=2), f32 for 1-D vectors + CMVN + mel filterbank; q8_0 for
/// the rank-2 `.weight` matrices (ne0 % 32 == 0); q4_k for the rank-2 `.weight`
/// matrices whose ne0 % 256 == 0 (else q8_0). See `PackQuant`.
pub type DolphinQuantizationMode = PackQuant;

/// Which decode-prefix scheme this checkpoint's vocab uses. The cn-dialect
/// family (small.cn, cn-dialect-base) fixes the OWSM `<lang>` slot at `<zh>`
/// and only the `<region>` slot varies (Chinese province dialects); the
/// multilingual family (dolphin-small, dolphin-base) varies BOTH slots across
/// 40 languages. Baked into the pack as `dolphin.language.scheme` so the
/// executor picks the matching prefix builder without re-deriving it from the
/// vocab shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DolphinLanguageScheme {
    /// Fixed `<zh>` language token, per-code Chinese dialect `<region>`.
    #[default]
    CnDialect,
    /// Per-code `<lang>` AND `<region>`, spanning the 40-language table.
    Multilingual,
}

impl DolphinLanguageScheme {
    /// The `dolphin.language.scheme` metadata value this variant writes.
    pub fn label(self) -> &'static str {
        match self {
            Self::CnDialect => "cn_dialect",
            Self::Multilingual => "multilingual",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DolphinImportRequest {
    /// The exported full state dict (`full.safetensors`, all-f32).
    pub safetensors_path: PathBuf,
    /// The char vocab (`units.txt`, `token<space>id` per line, id order).
    pub units_path: PathBuf,
    /// Output `.oasr` runtime pack path.
    pub output_path: PathBuf,
    pub model_id: String,
    pub language_scheme: DolphinLanguageScheme,
    pub quantization: DolphinQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DolphinImportResult {
    pub output_path: PathBuf,
    pub tensor_count: usize,
    pub vocab_size: usize,
    pub blank_token_id: u32,
}

pub fn convert_local_dolphin_wenet_source_to_runtime_pack(
    request: &DolphinImportRequest,
) -> Result<DolphinImportResult, LocalSourceImportError> {
    validate_output_pack_extension(&request.output_path)?;
    let vocab_tokens = read_units_txt(&request.units_path)?;
    let vocab_size = vocab_tokens.len();
    let safetensors = SafetensorsFile::open(&request.safetensors_path)?;

    let architecture = derive_dolphin_architecture(&safetensors, vocab_size, &vocab_tokens)?;
    match request.language_scheme {
        DolphinLanguageScheme::CnDialect => {
            assert_every_advertised_dialect_code_resolves(&vocab_tokens)?
        }
        DolphinLanguageScheme::Multilingual => {
            assert_every_advertised_multilingual_code_resolves(&vocab_tokens)?
        }
    }

    let mut tensors = build_runtime_tensors(&safetensors, request.quantization)?;
    tensors.push(build_mel_filterbank_tensor());
    // Synthesize the sinusoidal position table(s) the checkpoint itself did
    // not bake as a state-dict buffer (see `sinusoidal_pos_table_max_ctx`).
    // The encoder table is CnDialect-only: that family's WeNet-trained
    // encoder attention consumes the simple non-centered `RelPositionalEncoding`
    // sliced from this table (`encoder_graph::attention_branch`'s "sdpa fold, no
    // rel_shift" path). The multilingual family's ESPnet-trained encoder uses
    // `RelPositionalEncodingV1` instead -- a centered table ESPnet itself never
    // bakes (computed fresh per forward call) -- so the runtime computes that
    // one at graph-build time from the request's own frame count instead
    // (`encoder_graph::dolphin_relative_positional_table`); baking a
    // fixed-`max_ctx`-sized version of it here would be both wrong-shaped (this
    // table's layout depends on the runtime frame count, not a fixed ceiling)
    // and dead weight. The decoder's table is unaffected either way: both
    // families' Transformer decoder uses the same plain absolute
    // `PositionalEncoding`.
    if request.language_scheme == DolphinLanguageScheme::CnDialect
        && safetensors.tensor("encoder.embed.pos_enc.pe").is_none()
    {
        tensors.push(synthesized_position_table_tensor(
            "encoder.embed.pos_enc.pe",
            architecture.encoder_d_model,
            architecture.encoder_max_ctx,
        ));
    }
    if safetensors.tensor("decoder.embed.1.pe").is_none() {
        tensors.push(synthesized_position_table_tensor(
            "decoder.embed.1.pe",
            architecture.encoder_d_model,
            architecture.decoder_max_ctx,
        ));
    }

    let metadata = dolphin_runtime_gguf_metadata(request, &vocab_tokens, &architecture);
    write_gguf_file_v0(&request.output_path, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "dolphin GGUF writer failed for '{}': {error}",
            request.output_path.display()
        ))
    })?;

    let index = read_gguf_tensor_index(&request.output_path).map_err(|error| {
        validate_error(format!(
            "dolphin import produced an unreadable tensor index: {error}"
        ))
    })?;
    Ok(DolphinImportResult {
        output_path: request.output_path.clone(),
        tensor_count: index.tensors().len(),
        vocab_size,
        blank_token_id: architecture.blank_token_id,
    })
}

/// Parse `units.txt` (`token<space>id`, one per line) into an id-ordered token
/// list. WeNet char tokens never contain a space, so the id is the trailing
/// whitespace-delimited field.
fn read_units_txt(path: &std::path::Path) -> Result<Vec<String>, LocalSourceImportError> {
    let text = std::fs::read_to_string(path).map_err(|source| LocalSourceImportError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut by_id: BTreeMap<usize, String> = BTreeMap::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        let (token, id_str) = line.rsplit_once(char::is_whitespace).ok_or_else(|| {
            validate_error(format!(
                "dolphin units.txt line {} is not '<token> <id>': {line:?}",
                line_no + 1
            ))
        })?;
        let id: usize = id_str.trim().parse().map_err(|error| {
            validate_error(format!(
                "dolphin units.txt line {} has an unparseable id {id_str:?}: {error}",
                line_no + 1
            ))
        })?;
        if by_id.insert(id, token.to_string()).is_some() {
            return Err(validate_error(format!(
                "dolphin units.txt has a duplicate id {id}"
            )));
        }
    }
    let count = by_id.len();
    let mut tokens = Vec::with_capacity(count);
    for expected_id in 0..count {
        let token = by_id.remove(&expected_id).ok_or_else(|| {
            validate_error(format!(
                "dolphin units.txt is missing token id {expected_id}"
            ))
        })?;
        tokens.push(token);
    }
    if tokens.is_empty() {
        return Err(validate_error("dolphin units.txt produced an empty vocab"));
    }
    Ok(tokens)
}

/// Fail closed unless every advertised dialect recognition code
/// (`DOLPHIN_CN_DIALECT_CODES` plus the bare `zh` default) builds a full decode
/// prefix against the shipped vocab -- i.e. its `<REGION>` token and the shared
/// `<sos>/<zh>/<asr>/<notimestamp>` control tokens are all present. This is the
/// producer-side guard for the CRITICAL invariant that the picker never advertises
/// a region the executor cannot honor: if a future checkpoint's `units.txt` drops
/// a region token, the import fails here rather than shipping a pack whose picker
/// lies. It runs on the full baked vocab (no single region is baked as *the*
/// default; the region is selected per request at decode time). Checked against
/// this family's OWN dialect-code list, not the model-agnostic
/// `REGISTERED_DIALECT_CODES` union -- see `DOLPHIN_CN_DIALECT_CODES`'s doc
/// comment for why the two lists diverge.
fn assert_every_advertised_dialect_code_resolves(
    vocab_tokens: &[String],
) -> Result<(), LocalSourceImportError> {
    for &code in DOLPHIN_CN_DIALECT_CODES
        .iter()
        .chain(std::iter::once(&DOLPHIN_DEFAULT_LANGUAGE_CODE))
    {
        build_dolphin_decode_prefix(vocab_tokens, Some(code)).map_err(|error| {
            validate_error(format!(
                "dolphin pack vocab cannot honor advertised dialect code {code:?}: {error}"
            ))
        })?;
    }
    Ok(())
}

/// The multilingual counterpart of `assert_every_advertised_dialect_code_resolves`:
/// fail closed unless every catalog-advertised language code (the
/// `dolphin-small`/`dolphin-base` `languages` list) builds a full decode
/// prefix -- both its `<lang>` AND `<region>` tokens, plus the shared control
/// tokens -- against the shipped vocab.
fn assert_every_advertised_multilingual_code_resolves(
    vocab_tokens: &[String],
) -> Result<(), LocalSourceImportError> {
    for &code in DOLPHIN_MULTILINGUAL_CATALOG_LANGUAGES {
        build_dolphin_multilingual_decode_prefix(vocab_tokens, Some(code)).map_err(|error| {
            validate_error(format!(
                "dolphin pack vocab cannot honor advertised language code {code:?}: {error}"
            ))
        })?;
    }
    Ok(())
}

/// Structural hparams read directly off the checkpoint's own tensor shapes
/// (never hardcoded), so one importer handles every current and future
/// Dolphin size (base/small/medium/large, cn-dialect or multilingual) without
/// a new per-size const block. See the module-level note above `FEATURE_DIM`.
#[derive(Debug, Clone, Copy)]
struct DolphinArchitecture {
    encoder_n_layers: usize,
    encoder_d_model: usize,
    encoder_n_heads: usize,
    encoder_head_dim: usize,
    encoder_ffn_dim: usize,
    encoder_cgmlp_units: usize,
    encoder_cgmlp_kernel: usize,
    encoder_merge_kernel: usize,
    encoder_max_ctx: usize,
    decoder_n_layers: usize,
    decoder_n_heads: usize,
    decoder_ffn_dim: usize,
    decoder_max_ctx: usize,
    sos_token_id: u32,
    eos_token_id: u32,
    blank_token_id: u32,
}

/// Derive [`DolphinArchitecture`] from the checkpoint, failing closed on any
/// missing tensor or an internally-inconsistent shape (e.g. encoder/decoder
/// width disagreeing, or a head_dim that does not evenly divide d_model).
/// Cross-checked against small.cn (768 d_model/12 heads/12 layers, encoder FFN
/// == cgMLP == 3072), the multilingual small/base checkpoints (which tie
/// decoder width to encoder width and decoder head_dim to encoder head_dim in
/// every observed variant, but let the encoder FFN width diverge from the
/// cgMLP width), and the 140M base tier (512/8/6, FFN == cgMLP == 2048).
fn derive_dolphin_architecture(
    safetensors: &SafetensorsFile,
    vocab_size: usize,
    vocab_tokens: &[String],
) -> Result<DolphinArchitecture, LocalSourceImportError> {
    let shape = |name: &str| -> Result<Vec<u64>, LocalSourceImportError> {
        safetensors
            .tensor(name)
            .map(|tensor| tensor.shape.clone())
            .ok_or_else(|| validate_error(format!("dolphin checkpoint missing tensor '{name}'")))
    };
    let expect = |name: &str, actual: &[u64], want: &[u64]| -> Result<(), LocalSourceImportError> {
        if actual == want {
            Ok(())
        } else {
            Err(validate_error(format!(
                "dolphin checkpoint tensor '{name}' shape {actual:?} != expected {want:?}"
            )))
        }
    };
    let rank1 = |name: &str| -> Result<usize, LocalSourceImportError> {
        let dims = shape(name)?;
        match dims.as_slice() {
            [d0] => Ok(*d0 as usize),
            _ => Err(validate_error(format!(
                "dolphin checkpoint tensor '{name}' has shape {dims:?}, expected rank 1"
            ))),
        }
    };
    let rank2_dim0 = |name: &str| -> Result<usize, LocalSourceImportError> {
        let dims = shape(name)?;
        match dims.as_slice() {
            [d0, _] => Ok(*d0 as usize),
            _ => Err(validate_error(format!(
                "dolphin checkpoint tensor '{name}' has shape {dims:?}, expected rank 2"
            ))),
        }
    };
    let last_dim = |name: &str| -> Result<usize, LocalSourceImportError> {
        let dims = shape(name)?;
        dims.last().map(|d| *d as usize).ok_or_else(|| {
            validate_error(format!(
                "dolphin checkpoint tensor '{name}' has an empty shape"
            ))
        })
    };

    let encoder_d_model = rank1("encoder.after_norm.weight")?;
    let cmvn_len = rank1("encoder.global_cmvn.mean")?;
    if cmvn_len != FEATURE_DIM {
        return Err(validate_error(format!(
            "dolphin checkpoint CMVN vector has {cmvn_len} elements, expected the {FEATURE_DIM}-mel frontend"
        )));
    }
    expect(
        "ctc.ctc_lo.weight",
        &shape("ctc.ctc_lo.weight")?,
        &[vocab_size as u64, encoder_d_model as u64],
    )?;
    expect(
        "decoder.output_layer.weight",
        &shape("decoder.output_layer.weight")?,
        &[vocab_size as u64, encoder_d_model as u64],
    )?;
    let decoder_d_model = rank1("decoder.after_norm.weight")?;
    if decoder_d_model != encoder_d_model {
        return Err(validate_error(format!(
            "dolphin checkpoint decoder d_model {decoder_d_model} != encoder d_model {encoder_d_model}"
        )));
    }

    let pos_bias = shape("encoder.encoders.0.attn.pos_bias_u")?;
    let (encoder_n_heads, encoder_head_dim) = match pos_bias.as_slice() {
        [heads, head_dim] => (*heads as usize, *head_dim as usize),
        _ => {
            return Err(validate_error(format!(
                "dolphin checkpoint 'encoder.encoders.0.attn.pos_bias_u' has shape {pos_bias:?}, expected rank 2"
            )));
        }
    };
    if encoder_n_heads * encoder_head_dim != encoder_d_model {
        return Err(validate_error(format!(
            "dolphin checkpoint encoder heads {encoder_n_heads} * head_dim {encoder_head_dim} != d_model {encoder_d_model}"
        )));
    }
    // The decoder never stores a relative-position bias tensor to read a head
    // count off directly; every observed Dolphin checkpoint (small.cn,
    // cn-dialect-base, multilingual small/base) ties the decoder's head_dim to
    // the encoder's, so derive decoder_n_heads from that instead of a guess.
    if !decoder_d_model.is_multiple_of(encoder_head_dim) {
        return Err(validate_error(format!(
            "dolphin checkpoint decoder d_model {decoder_d_model} is not a multiple of the encoder head_dim {encoder_head_dim}"
        )));
    }
    let decoder_n_heads = decoder_d_model / encoder_head_dim;

    let encoder_ffn_dim = rank2_dim0("encoder.encoders.0.feed_forward.w_1.weight")?;
    let encoder_cgmlp_units = rank2_dim0("encoder.encoders.0.cgmlp.channel_proj1.0.weight")?;
    let encoder_cgmlp_kernel = last_dim("encoder.encoders.0.cgmlp.csgu.conv.weight")?;
    let encoder_merge_kernel = last_dim("encoder.encoders.0.depthwise_conv_fusion.weight")?;
    let encoder_max_ctx =
        sinusoidal_pos_table_max_ctx(safetensors, "encoder.embed.pos_enc.pe", encoder_d_model)?;
    let decoder_ffn_dim = rank2_dim0("decoder.decoders.0.feed_forward.w_1.weight")?;
    let decoder_max_ctx =
        sinusoidal_pos_table_max_ctx(safetensors, "decoder.embed.1.pe", decoder_d_model)?;

    let layer_count = |prefix: &str, joint: &str| -> usize {
        let mut seen = BTreeSet::new();
        for tensor in &safetensors.header().tensors {
            if let Some(rest) = tensor.name.strip_prefix(prefix)
                && let Some((idx, _)) = rest.split_once(joint)
                && let Ok(index) = idx.parse::<usize>()
            {
                seen.insert(index);
            }
        }
        seen.len()
    };
    let encoder_n_layers = layer_count("encoder.encoders.", ".");
    if encoder_n_layers == 0 {
        return Err(validate_error(
            "dolphin checkpoint has no 'encoder.encoders.N.*' layer tensors",
        ));
    }
    let decoder_n_layers = layer_count("decoder.decoders.", ".");
    if decoder_n_layers == 0 {
        return Err(validate_error(
            "dolphin checkpoint has no 'decoder.decoders.N.*' layer tensors",
        ));
    }

    // Hotword deep-biasing module: only checked when present (older exports
    // without a trained context module still import; the executor's
    // `supports_phrase_bias` degrades to reporting no hotword capability for
    // those packs via the tensor-index probe, not a hard import failure).
    if safetensors
        .tensor("context_module.context_extractor.word_embedding.weight")
        .is_some()
    {
        expect(
            "context_module.context_extractor.word_embedding.weight",
            &shape("context_module.context_extractor.word_embedding.weight")?,
            &[vocab_size as u64, encoder_d_model as u64],
        )?;
        expect(
            "context_module.context_extractor.sen_rnn.weight_ih_l0",
            &shape("context_module.context_extractor.sen_rnn.weight_ih_l0")?,
            &[4 * encoder_d_model as u64, encoder_d_model as u64],
        )?;
        expect(
            "context_module.context_extractor.sen_rnn.weight_ih_l1",
            &shape("context_module.context_extractor.sen_rnn.weight_ih_l1")?,
            &[4 * encoder_d_model as u64, 2 * encoder_d_model as u64],
        )?;
        expect(
            "context_module.biasing_layer.linear_q.weight",
            &shape("context_module.biasing_layer.linear_q.weight")?,
            &[encoder_d_model as u64, encoder_d_model as u64],
        )?;
        expect(
            "context_module.combiner.weight",
            &shape("context_module.combiner.weight")?,
            &[encoder_d_model as u64, encoder_d_model as u64],
        )?;
    }

    let blank_token_id = required_special_token_id(vocab_tokens, "<blank>")?;
    let sos_token_id = required_special_token_id(vocab_tokens, "<sos>")?;
    let eos_token_id = required_special_token_id(vocab_tokens, "<eos>")?;

    Ok(DolphinArchitecture {
        encoder_n_layers,
        encoder_d_model,
        encoder_n_heads,
        encoder_head_dim,
        encoder_ffn_dim,
        encoder_cgmlp_units,
        encoder_cgmlp_kernel,
        encoder_merge_kernel,
        encoder_max_ctx,
        decoder_n_layers,
        decoder_n_heads,
        decoder_ffn_dim,
        decoder_max_ctx,
        sos_token_id,
        eos_token_id,
        blank_token_id,
    })
}

/// Default sinusoidal position-table length ESPnet/WeNet's `PositionalEncoding`
/// classes use when the checkpoint does not bake the table as a state-dict
/// buffer (see `sinusoidal_pos_table_max_ctx`): every Dolphin checkpoint
/// observed so far -- baked (small.cn, cn-dialect-base) or not (dolphin-small,
/// dolphin-base) -- uses this exact value, and it is a decode-length ceiling
/// choice independent of encoder/decoder width, not an architecture property.
const DEFAULT_SINUSOIDAL_POS_TABLE_MAX_CTX: usize = 5000;

/// The sinusoidal position table's length, from the checkpoint's own baked
/// tensor when present, else the documented default. The cn-dialect WeNet
/// export bakes this table as a state-dict buffer (`encoder.embed.pos_enc.pe`/
/// `decoder.embed.1.pe`); the multilingual ESPnet export does not (its
/// `RelPositionalEncoding`/`PositionalEncoding` compute the sinusoid on the
/// fly in `forward()` instead of registering it as a buffer) -- either way the
/// values are the deterministic textbook formula
/// (`build_sinusoidal_position_table`), verified byte-for-byte against the
/// cn-dialect-base checkpoint's own baked table, so a checkpoint that omits it
/// gets the identical table synthesized at import time instead of failing
/// closed on a tensor that was never going to exist.
fn sinusoidal_pos_table_max_ctx(
    safetensors: &SafetensorsFile,
    name: &str,
    expected_d_model: usize,
) -> Result<usize, LocalSourceImportError> {
    let Some(tensor) = safetensors.tensor(name) else {
        return Ok(DEFAULT_SINUSOIDAL_POS_TABLE_MAX_CTX);
    };
    match tensor.shape.as_slice() {
        [_, max_ctx, d_model] if *d_model == expected_d_model as u64 => Ok(*max_ctx as usize),
        shape => Err(validate_error(format!(
            "dolphin checkpoint '{name}' has shape {shape:?}, expected [1, max_ctx, {expected_d_model}]"
        ))),
    }
}

/// Build the deterministic sinusoidal position table WeNet/ESPnet bake as
/// `encoder.embed.pos_enc.pe` / `decoder.embed.1.pe`:
/// `pe[pos, 2i] = sin(pos * exp(-2i/d_model * ln(10000)))`,
/// `pe[pos, 2i+1] = cos(pos * exp(-2i/d_model * ln(10000)))`. Verified
/// byte-for-byte (< 1e-6 abs diff, float32 rounding only) against the
/// cn-dialect-base checkpoint's own baked table before being trusted as a
/// synthesized substitute for a checkpoint that does not bake one.
fn build_sinusoidal_position_table(d_model: usize, max_positions: usize) -> Vec<f32> {
    let mut table = vec![0.0_f32; max_positions * d_model];
    for pos in 0..max_positions {
        let row = &mut table[pos * d_model..(pos + 1) * d_model];
        let mut i = 0;
        while i < d_model {
            let div_term = (-((i as f64) / (d_model as f64)) * 10000.0_f64.ln()).exp();
            let angle = pos as f64 * div_term;
            row[i] = angle.sin() as f32;
            if i + 1 < d_model {
                row[i + 1] = angle.cos() as f32;
            }
            i += 2;
        }
    }
    table
}

/// Wrap `build_sinusoidal_position_table` as the `[1, max_positions, d_model]`
/// runtime tensor the encoder/decoder graphs expect at `name`, in fp16 mode --
/// matching `make_runtime_tensor`'s convention for every other rank>=2 tensor
/// (only 1-D vectors stay f32), so a synthesized table is byte-layout
/// consistent with a checkpoint-baked one.
fn synthesized_position_table_tensor(
    name: &str,
    d_model: usize,
    max_positions: usize,
) -> GgufWriteTensor {
    let table = build_sinusoidal_position_table(d_model, max_positions);
    let bits: Vec<u16> = table.iter().copied().map(f32_to_f16_bits).collect();
    GgufWriteTensor {
        name: name.to_string(),
        dims: vec![1, max_positions as u64, d_model as u64],
        tensor_type: GgufWriteTensorType::F16,
        data: encode_f16_bits_le(bits),
    }
}

/// The id of a required control token (`<blank>`/`<sos>`/`<eos>`). WeNet vocabs
/// place these at different ids across Dolphin variants (front-loaded at
/// 0/2/3 for the char-vocab cn-dialect models; appended after ~40k BPE pieces
/// for the multilingual models), so this looks the token up by content rather
/// than assuming a fixed id.
fn required_special_token_id(
    vocab_tokens: &[String],
    token: &str,
) -> Result<u32, LocalSourceImportError> {
    token_id_for_content(vocab_tokens, token)
        .ok_or_else(|| validate_error(format!("dolphin checkpoint vocab has no '{token}' token")))
}

/// First vocab id whose token content is exactly `content` (mirrors
/// `language::token_id_for_content`; duplicated here to keep the importer free
/// of a dependency on the dialect-only prefix builder's private helper).
fn token_id_for_content(vocab: &[String], content: &str) -> Option<u32> {
    vocab
        .iter()
        .position(|token| token == content)
        .map(|index| index as u32)
}

/// Emit every `encoder.*` / `decoder.*` / `ctc.*` tensor under its exact WeNet
/// name. At fp16, raw element order is preserved (dims == the safetensors shape):
/// rank>=2 weights become f16, everything else (1-D biases/norms, CMVN) stays f32.
/// At q8_0/q4_k, the rank-2 `.weight` matrices are block-quantized (dims reversed
/// so the contiguous `in` axis lands on ne0); all other tensors keep their fp16
/// representation.
fn build_runtime_tensors(
    safetensors: &SafetensorsFile,
    quantization: DolphinQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for tensor in &safetensors.header().tensors {
        let name = tensor.name.as_str();
        if DROPPED_TENSOR_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
        {
            continue;
        }
        if !RUNTIME_TENSOR_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
        {
            continue;
        }
        if !seen.insert(name.to_string()) {
            return Err(validate_error(format!(
                "dolphin import mapped duplicate destination tensor '{name}'"
            )));
        }
        let values = decode_safetensors_payload_as_f32(
            &tensor.name,
            &tensor.dtype,
            safetensors.tensor_data(tensor)?,
        )?;
        out.push(make_runtime_tensor(
            name.to_string(),
            tensor.shape.clone(),
            values,
            quantization,
        )?);
    }
    if out.is_empty() {
        return Err(validate_error(
            "dolphin import found no encoder/decoder/ctc tensors in the checkpoint",
        ));
    }
    Ok(out)
}

/// Choose the block-quant type for a rank-2 `.weight` matrix, or `None` to keep
/// its fp16-mode representation. ggml quantizes along ne0, which after the dim
/// reversal (`[out, in]` -> `[in, out]`) is `in` (the safetensors innermost /
/// row-major-contiguous dim). q4_k needs a 256-superblock (`in % 256`), q8_0 a
/// 32-block (`in % 32`); an unaligned `in` falls back to fp16. Only 2-D `.weight`
/// tensors qualify -- convs (rank>2), position tables, and 1-D/CMVN vectors are
/// never quantized.
fn dolphin_quant_type_for_tensor(
    name: &str,
    shape: &[u64],
    quantization: DolphinQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if quantization == DolphinQuantizationMode::Fp16 {
        return None;
    }
    if !name.ends_with(".weight") || shape.len() != 2 {
        return None;
    }
    // Reversed ne0 == the safetensors innermost (last) dim == `in`.
    let ne0 = *shape.last()?;
    classify_quant_tensor(ne0, quantization)
}

/// Build one runtime tensor. Quantizable rank-2 `.weight` matrices are block-
/// quantized with **reversed dims** (ne0 = the contiguous `in` axis); the runtime
/// re-declares its own graph shapes and consumes only the dequantized f32 by
/// element count, so the reversal is transparent to it. Everything else keeps the
/// fp16-mode layout: f16 for rank>=2, f32 for 1-D vectors (biases, norms, the CMVN
/// mean/istd, the rel-pos bias), name-preserving raw element order in both cases.
fn make_runtime_tensor(
    name: String,
    dims: Vec<u64>,
    values: Vec<f32>,
    quantization: DolphinQuantizationMode,
) -> Result<GgufWriteTensor, LocalSourceImportError> {
    if let Some(qtype) = dolphin_quant_type_for_tensor(&name, &dims, quantization) {
        let mut reversed = dims.clone();
        reversed.reverse();
        let data =
            quantize_f32_to_ggml_tensor_data(qtype, &reversed, &values).map_err(|error| {
                validate_error(format!(
                    "dolphin quantization failed for '{name}' ({qtype:?}): {error}"
                ))
            })?;
        return Ok(GgufWriteTensor {
            name,
            dims: reversed,
            tensor_type: qtype,
            data,
        });
    }
    if dims.len() >= 2 {
        let bits: Vec<u16> = values.iter().copied().map(f32_to_f16_bits).collect();
        Ok(GgufWriteTensor {
            name,
            dims,
            tensor_type: GgufWriteTensorType::F16,
            data: encode_f16_bits_le(bits),
        })
    } else {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(GgufWriteTensor {
            name,
            dims,
            tensor_type: GgufWriteTensorType::F32,
            data: bytes,
        })
    }
}

/// Kaldi/HTK mel filterbank `[n_mels, fft_bins]` (peak-normalized triangles, no
/// Slaney area norm) reconstructed from the `train.yaml` fbank config, via the
/// shared [`crate::models::audio_frontend::mel`] `MelScale::Kaldi` construction
/// (same mel-domain math `crate::models::kaldi_fbank`'s runtime engine uses).
/// Stored for the later frontend phase; NOT exercised by the encoder-from-pack
/// load path (which is fed the CMVN'd golden features directly), so it is not
/// yet parity verified against a golden fbank.
fn build_mel_filterbank_tensor() -> GgufWriteTensor {
    let fft_bins = FFT_SIZE / 2 + 1;
    let high_hz = (SAMPLE_RATE_HZ as f32) / 2.0;
    let filters = filterbank(FilterbankConfig {
        scale: MelScale::Kaldi,
        sample_rate_hz: SAMPLE_RATE_HZ as f32,
        n_fft: FFT_SIZE,
        n_mels: FEATURE_DIM,
        fmin: MEL_LOW_HZ,
        fmax: high_hz,
        // Not consulted by `MelScale::Kaldi` (mel-domain edges, no round-trip).
        mel_point_order: MelPointOrder::SpanTimesIndexFirst,
    });
    let mut bytes = Vec::with_capacity(filters.len() * 4);
    for value in &filters {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    GgufWriteTensor {
        name: "dolphin.mel_filters".to_string(),
        dims: vec![FEATURE_DIM as u64, fft_bins as u64],
        tensor_type: GgufWriteTensorType::F32,
        data: bytes,
    }
}

fn dolphin_runtime_gguf_metadata(
    request: &DolphinImportRequest,
    vocab_tokens: &[String],
    architecture: &DolphinArchitecture,
) -> BTreeMap<String, GgufWriteValue> {
    let vocab_size = vocab_tokens.len();
    let mut metadata = BTreeMap::new();
    let mut put_str = |key: &str, value: &str| {
        metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
    };
    put_str("general.architecture", DOLPHIN_GGML_ARCHITECTURE_ID);
    put_str(OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1);
    put_str(OASR_METADATA_KEY_MODEL_FAMILY, DOLPHIN_MODEL_FAMILY);
    put_str(
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        DOLPHIN_GGML_ARCHITECTURE_ID,
    );
    put_str(OASR_METADATA_KEY_AUDIO_FRONTEND, DOLPHIN_AUDIO_FRONTEND_ID);
    put_str(OASR_METADATA_KEY_DECODE_POLICY, DOLPHIN_DECODE_POLICY_ID);
    put_str(GGML_TOKENIZER_ID_KEY, DOLPHIN_TOKENIZER_ID);
    put_str("openasr.model.id", &request.model_id);
    put_str("dolphin.tokenizer.model", "char");
    // Selects the decode-prefix builder at runtime (executor.rs): the
    // cn-dialect family's fixed-`<zh>` builder, or the multilingual family's
    // per-code `<lang>` + `<region>` builder. Absent on a pack predating this
    // key (none exist yet -- both currently-published dolphin packs are
    // cn-dialect), the executor's reader defaults to `cn_dialect`.
    put_str("dolphin.language.scheme", request.language_scheme.label());

    let mut put_u32 = |key: &str, value: u32| {
        metadata.insert(key.to_string(), GgufWriteValue::U32(value));
    };
    put_u32(
        "dolphin.encoder.n_layers",
        architecture.encoder_n_layers as u32,
    );
    put_u32(
        "dolphin.encoder.d_model",
        architecture.encoder_d_model as u32,
    );
    put_u32(
        "dolphin.encoder.n_heads",
        architecture.encoder_n_heads as u32,
    );
    put_u32(
        "dolphin.encoder.head_dim",
        architecture.encoder_head_dim as u32,
    );
    put_u32(
        "dolphin.encoder.ffn_dim",
        architecture.encoder_ffn_dim as u32,
    );
    put_u32(
        "dolphin.encoder.cgmlp_units",
        architecture.encoder_cgmlp_units as u32,
    );
    put_u32(
        "dolphin.encoder.cgmlp_kernel",
        architecture.encoder_cgmlp_kernel as u32,
    );
    put_u32(
        "dolphin.encoder.merge_kernel",
        architecture.encoder_merge_kernel as u32,
    );
    put_u32("dolphin.encoder.feature_dim", FEATURE_DIM as u32);
    put_u32(
        "dolphin.encoder.max_ctx",
        architecture.encoder_max_ctx as u32,
    );
    put_u32(
        "dolphin.decoder.n_layers",
        architecture.decoder_n_layers as u32,
    );
    put_u32(
        "dolphin.decoder.n_heads",
        architecture.decoder_n_heads as u32,
    );
    put_u32(
        "dolphin.decoder.ffn_dim",
        architecture.decoder_ffn_dim as u32,
    );
    put_u32(
        "dolphin.decoder.max_ctx",
        architecture.decoder_max_ctx as u32,
    );
    put_u32("dolphin.vocab_size", vocab_size as u32);
    put_u32("dolphin.sos_token_id", architecture.sos_token_id);
    put_u32("dolphin.eos_token_id", architecture.eos_token_id);
    put_u32("ctc.blank_token_id", architecture.blank_token_id);
    // fbank frontend config (the mel filterbank contract for the later phase).
    put_u32("dolphin.audio.sample_rate", SAMPLE_RATE_HZ);
    put_u32("dolphin.audio.n_fft", FFT_SIZE as u32);
    put_u32("dolphin.audio.frame_length_ms", FRAME_LENGTH_MS);
    put_u32("dolphin.audio.frame_shift_ms", FRAME_SHIFT_MS);
    put_u32("dolphin.audio.n_mels", FEATURE_DIM as u32);

    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufWriteValue::StringArray(vocab_tokens.to_vec()),
    );
    metadata
}

#[cfg(test)]
mod tests {
    use super::*;
    // `DolphinQuantizationMode` is now a type alias for the shared `PackQuant`;
    // `use`-importing a bare variant has to name the real enum, not the alias.
    use PackQuant::Fp16;

    fn string_metadata(metadata: &BTreeMap<String, GgufWriteValue>, key: &str) -> Option<String> {
        match metadata.get(key) {
            Some(GgufWriteValue::String(value)) => Some(value.clone()),
            _ => None,
        }
    }

    fn u32_metadata(metadata: &BTreeMap<String, GgufWriteValue>, key: &str) -> Option<u32> {
        match metadata.get(key) {
            Some(GgufWriteValue::U32(value)) => Some(*value),
            _ => None,
        }
    }

    fn fixture_request() -> DolphinImportRequest {
        DolphinImportRequest {
            safetensors_path: PathBuf::from("/tmp/dolphin/full.safetensors"),
            units_path: PathBuf::from("/tmp/dolphin/units.txt"),
            output_path: PathBuf::from("/tmp/dolphin-out.oasr"),
            model_id: "dolphin-cn-dialect-small".to_string(),
            quantization: DolphinQuantizationMode::Fp16,
            language_scheme: DolphinLanguageScheme::CnDialect,
        }
    }

    /// The `small.cn` architecture (768 d_model/12 heads/12 layers), matching
    /// the historical hardcoded consts this fixture exercised before the
    /// import became shape-derived.
    fn small_cn_architecture() -> DolphinArchitecture {
        DolphinArchitecture {
            encoder_n_layers: 12,
            encoder_d_model: 768,
            encoder_n_heads: 12,
            encoder_head_dim: 64,
            encoder_ffn_dim: 3072,
            encoder_cgmlp_units: 3072,
            encoder_cgmlp_kernel: 31,
            encoder_merge_kernel: 31,
            encoder_max_ctx: 5000,
            decoder_n_layers: 12,
            decoder_n_heads: 12,
            decoder_ffn_dim: 3072,
            decoder_max_ctx: 5000,
            sos_token_id: 2,
            eos_token_id: 3,
            blank_token_id: 0,
        }
    }

    #[test]
    fn runtime_metadata_declares_dolphin_selection_and_contract_keys() {
        let tokens: Vec<String> = (0..18173).map(|i| format!("t{i}")).collect();
        let metadata =
            dolphin_runtime_gguf_metadata(&fixture_request(), &tokens, &small_cn_architecture());
        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_MODEL_FAMILY),
            Some(DOLPHIN_MODEL_FAMILY.to_string())
        );
        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_MODEL_ARCHITECTURE),
            Some(DOLPHIN_GGML_ARCHITECTURE_ID.to_string())
        );
        assert_eq!(
            string_metadata(&metadata, GGML_TOKENIZER_ID_KEY),
            Some(DOLPHIN_TOKENIZER_ID.to_string())
        );
        assert_eq!(u32_metadata(&metadata, "dolphin.vocab_size"), Some(18173));
        assert_eq!(u32_metadata(&metadata, "ctc.blank_token_id"), Some(0));
        assert_eq!(
            u32_metadata(&metadata, "dolphin.encoder.d_model"),
            Some(768)
        );
    }

    /// The full special-token block every advertised dialect code needs
    /// (control tokens + one `<REGION>` token per this family's dialect code
    /// plus `<CN>`).
    fn vocab_with_all_dialect_tokens() -> Vec<String> {
        let mut tokens: Vec<String> = ["<sos>", "<eos>", "<asr>", "<zh>", "<notimestamp>", "<CN>"]
            .iter()
            .map(|token| token.to_string())
            .collect();
        for &code in DOLPHIN_CN_DIALECT_CODES {
            let region = crate::models::dolphin::language::dolphin_region_token_for_code(code)
                .expect("registered code maps to a region token");
            tokens.push(region.to_string());
        }
        tokens
    }

    #[test]
    fn advertised_dialect_codes_resolve_against_full_vocab() {
        // A vocab carrying every region + control token clears the producer guard.
        let vocab = vocab_with_all_dialect_tokens();
        assert_every_advertised_dialect_code_resolves(&vocab).expect("all codes resolve");
    }

    #[test]
    fn missing_region_token_fails_the_import_guard() {
        // Drop `<SICHUAN>` (an advertised region) and the import must fail closed
        // rather than ship a pack whose picker advertises a region it can't honor.
        let mut vocab = vocab_with_all_dialect_tokens();
        vocab.retain(|token| token != "<SICHUAN>");
        let error = assert_every_advertised_dialect_code_resolves(&vocab)
            .expect_err("missing region must fail the guard");
        let message = error.to_string();
        assert!(
            message.contains("zh-sichuan"),
            "guard error should name the unhonored code, got: {message}"
        );
    }

    #[test]
    fn quant_type_selection_matches_alignment_and_mode() {
        // fp16 mode never quantizes.
        assert_eq!(
            dolphin_quant_type_for_tensor("decoder.output_layer.weight", &[18173, 768], Fp16),
            None
        );
        // q8_0: any rank-2 `.weight` whose in-dim (last, -> reversed ne0) is 32-aligned.
        assert_eq!(
            dolphin_quant_type_for_tensor(
                "decoder.output_layer.weight",
                &[18173, 768],
                DolphinQuantizationMode::Q8_0
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        // q4_k: in-dim 256-aligned -> Q4_K; only 32- (not 256-) aligned -> Q8_0.
        assert_eq!(
            dolphin_quant_type_for_tensor(
                "encoder.encoders.0.feed_forward.w_2.weight",
                &[768, 3072],
                DolphinQuantizationMode::Q4_K
            ),
            Some(GgufWriteTensorType::Q4_K)
        );
        assert_eq!(
            dolphin_quant_type_for_tensor(
                "some.proj.weight",
                &[10, 96],
                DolphinQuantizationMode::Q4_K
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        // Unaligned in-dim, non-`.weight`, rank!=2, and 1-D tensors are never quantized.
        assert_eq!(
            dolphin_quant_type_for_tensor(
                "some.proj.weight",
                &[10, 100],
                DolphinQuantizationMode::Q8_0
            ),
            None
        );
        assert_eq!(
            dolphin_quant_type_for_tensor(
                "encoder.embed.pos_enc.pe",
                &[1, 5000, 768],
                DolphinQuantizationMode::Q8_0
            ),
            None
        );
        assert_eq!(
            dolphin_quant_type_for_tensor(
                "encoder.embed.conv.0.bias",
                &[768],
                DolphinQuantizationMode::Q8_0
            ),
            None
        );
        assert_eq!(
            dolphin_quant_type_for_tensor(
                "encoder.embed.conv.2.weight",
                &[768, 768, 3, 3],
                DolphinQuantizationMode::Q8_0
            ),
            None
        );
    }

    #[test]
    fn fp16_tensor_layout_is_unchanged_by_the_quant_plumbing() {
        // A rank-2 weight in fp16 mode stays f16 with raw (non-reversed) dims.
        let weight = make_runtime_tensor("x.weight".to_string(), vec![4, 2], vec![1.0; 8], Fp16)
            .expect("fp16 weight");
        assert_eq!(weight.tensor_type, GgufWriteTensorType::F16);
        assert_eq!(weight.dims, vec![4, 2]);
        assert_eq!(weight.data.len(), 8 * 2); // f16 = 2 bytes/elem, no reversal.
        // A 1-D vector stays f32.
        let bias = make_runtime_tensor("x.bias".to_string(), vec![4], vec![1.0; 4], Fp16)
            .expect("fp16 bias");
        assert_eq!(bias.tensor_type, GgufWriteTensorType::F32);
        assert_eq!(bias.dims, vec![4]);
    }

    #[test]
    fn quantized_weight_reverses_dims_to_block_align_ne0() {
        // in-dim 768 (256-aligned) -> q4_k, dims reversed so ne0 == 768.
        let weight = make_runtime_tensor(
            "decoder.output_layer.weight".to_string(),
            vec![18173, 768],
            vec![0.1; 18173 * 768],
            DolphinQuantizationMode::Q4_K,
        )
        .expect("quantized weight");
        assert_eq!(weight.tensor_type, GgufWriteTensorType::Q4_K);
        assert_eq!(weight.dims, vec![768, 18173]);
    }

    #[test]
    fn mel_filterbank_has_expected_shape_and_is_bounded() {
        let tensor = build_mel_filterbank_tensor();
        let fft_bins = FFT_SIZE / 2 + 1;
        assert_eq!(tensor.dims, vec![FEATURE_DIM as u64, fft_bins as u64]);
        assert_eq!(tensor.data.len(), FEATURE_DIM * fft_bins * 4);
        // Peak-normalized triangles sit within [0, 1].
        for chunk in tensor.data.chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            assert!(
                (0.0..=1.0001).contains(&value),
                "mel weight out of range: {value}"
            );
        }
    }

    /// Exact reimplementation of this importer's pre-shared-mel-module
    /// `build_mel_filterbank_tensor`/`hz_to_mel` (the version that shipped
    /// before it was switched to `crate::models::audio_frontend::mel`'s
    /// `MelScale::Kaldi` construction), kept only here to pin the baked
    /// `dolphin.mel_filters` tensor to a byte-identical value across the
    /// migration.
    fn reference_pre_refactor_build_mel_filterbank_tensor() -> GgufWriteTensor {
        fn hz_to_mel(hz: f32) -> f32 {
            1127.0 * (1.0 + hz / 700.0).ln()
        }
        let fft_bins = FFT_SIZE / 2 + 1;
        let high_hz = (SAMPLE_RATE_HZ as f32) / 2.0;
        let mel_low = hz_to_mel(MEL_LOW_HZ);
        let mel_high = hz_to_mel(high_hz);
        let mel_delta = (mel_high - mel_low) / (FEATURE_DIM as f32 + 1.0);

        let mut filters = vec![0.0_f32; FEATURE_DIM * fft_bins];
        for mel_idx in 0..FEATURE_DIM {
            let left = mel_low + (mel_idx as f32) * mel_delta;
            let center = mel_low + (mel_idx as f32 + 1.0) * mel_delta;
            let right = mel_low + (mel_idx as f32 + 2.0) * mel_delta;
            for (bin_idx, cell) in filters
                .iter_mut()
                .skip(mel_idx * fft_bins)
                .take(fft_bins)
                .enumerate()
            {
                let hz = (bin_idx as f32) * (SAMPLE_RATE_HZ as f32) / (FFT_SIZE as f32);
                let mel = hz_to_mel(hz);
                let rising = (mel - left) / (center - left);
                let falling = (right - mel) / (right - center);
                *cell = rising.min(falling).max(0.0);
            }
        }
        let mut bytes = Vec::with_capacity(filters.len() * 4);
        for value in &filters {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        GgufWriteTensor {
            name: "dolphin.mel_filters".to_string(),
            dims: vec![FEATURE_DIM as u64, fft_bins as u64],
            tensor_type: GgufWriteTensorType::F32,
            data: bytes,
        }
    }

    #[test]
    fn mel_filterbank_tensor_is_byte_identical_to_pre_refactor_impl() {
        let expected = reference_pre_refactor_build_mel_filterbank_tensor();
        let actual = build_mel_filterbank_tensor();
        assert_eq!(expected.dims, actual.dims);
        assert_eq!(expected.tensor_type, actual.tensor_type);
        assert_eq!(expected.data, actual.data);
    }
}
