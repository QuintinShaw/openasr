//! Convert a local `ibm-granite/granite-speech-4.1-2b` HF source (sharded
//! safetensors + `config.json`) into an OpenASR `.oasr` (GGUF-v0) runtime pack.
//!
//! fp16-only this pass (q8_0/q4_k rungs are a follow-up). Every tensor keeps
//! its original HF name verbatim (`encoder.layers.0.attn.
//! to_q.weight`, `projector.query`, `language_model.model.layers.0....`)
//! rather than remapping to a family-local convention: the encoder/projector
//! ggml graphs (`encoder_graph.rs`/`qformer.rs`) already load by these exact
//! names, and there is now exactly one Granite Speech checkpoint shape to
//! support, so a remap layer would only add an indirection with no current
//! second caller -- with one forced exception, `remap_tensor_name`: ggml
//! tensor names are capped at 63 bytes (`GGML_MAX_NAME`), and the Q-Former's
//! deeply nested `projector.qformer.encoder.layer.{i}....` names overflow
//! that (up to 72 bytes), so just that one prefix is shortened to
//! `projector.qf.{i}.` in the pack (see `remap_tensor_name`'s doc comment).
//! 1-D tensors (norms, biases, the BatchNorm stats) are stored F32; every 2-D+
//! matmul/conv weight is stored F16. Rank-2 tensors are written with ggml
//! dims (`[ne0=in, ne1=out]`): safetensors carries PyTorch row-major
//! `[out, in]` bytes, which are already the correct flat layout for ggml
//! `mul_mat`, so the converter only **relabels** the two extents (no byte
//! transpose). Rank-3+ tensors (depthwise conv kernels, `projector.query`)
//! keep their source shape verbatim. The four Granite-architecture scaling
//! scalars (`attention_multiplier`, `embedding_multiplier`,
//! `residual_multiplier`, `logits_scaling`) and every other
//! decoder/encoder/projector shape hparam land in GGUF metadata as
//! stringified numbers (this writer's `GgufWriteValue` has no native float
//! variant; `insert_metadata` accepts any `ToString`, so this is the
//! established convention other families already use for non-integer hparams).
//!
//! The converter writes through the shared `PackEnvelope`/`OasrPackWriter`
//! seam and returns the writer's `VerifiedPack`; the runtime contract in
//! `runtime_contract.rs` re-proves the exact same metadata keys and the full
//! three-stage tensor set at pack admission, so a pack this writer produced
//! and a pack an attacker trimmed cannot reach the executor through the same
//! door.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::VerifiedPack;
use crate::ggml_runtime::{GgufWriteTensor, GgufWriteTensorType, GgufWriteValue};
use crate::models::local_source_import::{
    LocalSourceImportError, ShardedSafetensorsSource, encode_f16_bits_le, load_gpt2_bpe_merges,
    read_source_json_file, validate_error, validate_output_pack_extension,
};
use crate::models::oasr_metadata::{
    OasrPackWriter, PackEnvelope, insert_metadata, insert_metadata_string_array,
};
use crate::models::pack_quant::{QuantizedAxis, TensorQuantizationContract, TensorRole};
use crate::nn::half::f32_to_f16_bits;

use crate::arch::GRANITE_SPEECH_GGML_ARCHITECTURE_ID;

const SOURCE_CONFIG_JSON: &str = "config.json";
const SOURCE_INDEX_JSON: &str = "model.safetensors.index.json";
const SOURCE_VOCAB_JSON: &str = "vocab.json";
const SOURCE_ADDED_TOKENS_JSON: &str = "added_tokens.json";
const SOURCE_MERGES_TXT: &str = "merges.txt";

/// `tokenizer.ggml.*` metadata keys (mirrors `qwen::package_import`'s
/// constants of the same name -- both write the same stock GPT2-BPE shape).
pub(crate) const TOKENIZER_GGML_MODEL_KEY: &str = "tokenizer.ggml.model";
pub(crate) const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";
pub(crate) const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
pub(crate) const TOKENIZER_GGML_MERGES_KEY: &str = "tokenizer.ggml.merges";

pub(crate) const AUDIO_ENCODER_TENSOR_NAME_PREFIXES: &[&str] = &["encoder.", "projector."];

pub(crate) const TENSOR_QUANTIZATION_CONTRACT: TensorQuantizationContract =
    TensorQuantizationContract::SemanticRolesV1 {
        model_architecture: GRANITE_SPEECH_GGML_ARCHITECTURE_ID,
        classify: classify_granite_speech_quant_tensor_role,
        quantized_axis: QuantizedAxis::First,
    };

fn classify_granite_speech_quant_tensor_role(name: &str) -> TensorRole {
    if name.ends_with(".weight")
        && AUDIO_ENCODER_TENSOR_NAME_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
    {
        TensorRole::AcousticEncoderMatrix
    } else if name.ends_with(".weight") {
        TensorRole::TextDecoderMatrix
    } else {
        TensorRole::NonQuantizable
    }
}

#[derive(Debug, Clone)]
pub struct GraniteSpeechImportRequest {
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraniteSpeechImportResult {
    pub output_path: PathBuf,
    pub verified_pack: VerifiedPack,
    pub tensor_count: usize,
}

#[derive(Debug, Deserialize)]
struct GraniteSpeechEncoderConfigJson {
    input_dim: usize,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
    dim_head: usize,
    feedforward_mult: usize,
    conv_kernel_size: usize,
    conv_expansion_factor: usize,
    context_size: usize,
    max_pos_emb: usize,
    output_dim: usize,
}

#[derive(Debug, Deserialize)]
struct GraniteSpeechProjectorConfigJson {
    hidden_size: usize,
    encoder_hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    cross_attention_frequency: usize,
}

#[derive(Debug, Deserialize)]
struct GraniteTextConfigJson {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    intermediate_size: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    attention_multiplier: f32,
    embedding_multiplier: f32,
    residual_multiplier: f32,
    logits_scaling: f32,
    #[serde(default)]
    rope_theta: f64,
}

#[derive(Debug, Deserialize)]
struct GraniteSpeechConfigJson {
    audio_token_index: u32,
    downsample_rate: usize,
    window_size: usize,
    encoder_config: GraniteSpeechEncoderConfigJson,
    projector_config: GraniteSpeechProjectorConfigJson,
    text_config: GraniteTextConfigJson,
}

/// `Conv1d`/`Linear` weight ranks (>=2) go F16; everything else (1-D norms,
/// biases, BatchNorm running stats, the `projector.query` [1,3,1024] parameter
/// treated as rank>=2 too since it is a genuine matmul-adjacent operand, not a
/// bias) goes F32. Mirrors the force_f32-on-1-D convention every other
/// converter in this crate uses.
fn tensor_type_for_rank(shape: &[u64]) -> GgufWriteTensorType {
    if shape.len() >= 2 {
        GgufWriteTensorType::F16
    } else {
        GgufWriteTensorType::F32
    }
}

/// `[out, in]` safetensors row-major -> ggml `[in, out]` on-disk dims: reverse
/// the two extents, keep the flat byte layout. Same convention every other
/// importer in this crate uses for 2-D `Linear` / embedding weights so the
/// keep-quantized zero-copy path can bind pack tensors without a reshape.
fn ggml_dims_for_pack(shape: &[u64]) -> Vec<u64> {
    if shape.len() == 2 {
        vec![shape[1], shape[0]]
    } else {
        shape.to_vec()
    }
}

fn make_write_tensor(name: String, shape: Vec<u64>, values: Vec<f32>) -> GgufWriteTensor {
    let dims = ggml_dims_for_pack(&shape);
    let tensor_type = tensor_type_for_rank(&dims);
    match tensor_type {
        GgufWriteTensorType::F32 => {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            GgufWriteTensor {
                name,
                dims,
                tensor_type: GgufWriteTensorType::F32,
                data: bytes,
            }
        }
        _ => {
            let bits: Vec<u16> = values.iter().copied().map(f32_to_f16_bits).collect();
            GgufWriteTensor {
                name,
                dims,
                tensor_type: GgufWriteTensorType::F16,
                data: encode_f16_bits_le(bits),
            }
        }
    }
}

/// Tensor name prefixes this converter carries into the `.oasr` pack. Every
/// other top-level key in the checkpoint (there are none known today, but a
/// future LoRA adapter or auxiliary head should not silently sneak in) is
/// dropped, not passed through blind.
const CARRIED_PREFIXES: [&str; 3] = ["encoder.", "projector.", "language_model."];

/// GGUF/ggml tensor names are capped at `GGML_MAX_NAME - 1` (63 bytes,
/// `ggml_set_name` silently truncates past that, which then desyncs from the
/// GGUF writer's own by-name lookup and aborts the process -- see
/// `ggml_set_tensor_type`/`gguf.cpp`). Every other carried tensor name is
/// already under that bound; only the Q-Former's deeply nested
/// `projector.qformer.encoder.layer.{i}....` names overflow it (up to 72
/// bytes), so this shortens just that one segment. This is a pack-format
/// naming choice, independent of the HF names the `parity` dev harness reads
/// directly off the source safetensors (no ggml tensor-name limit applies
/// there, so that harness keeps using the literal HF names).
fn remap_tensor_name(name: &str) -> String {
    const LONG_PREFIX: &str = "projector.qformer.encoder.layer.";
    match name.strip_prefix(LONG_PREFIX) {
        Some(rest) => format!("projector.qf.{rest}"),
        None => name.to_string(),
    }
}

fn should_carry_tensor(name: &str) -> bool {
    // `BatchNorm1d`'s `num_batches_tracked` is an I64 training-step counter
    // with no forward-pass value (the encoder graph never reads it; the
    // BatchNorm fold only needs weight/bias/running_mean/running_var).
    if name.ends_with(".num_batches_tracked") {
        return false;
    }
    name == "lm_head.weight"
        || CARRIED_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
}

fn build_runtime_tensors(
    source: &ShardedSafetensorsSource,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut names: Vec<&String> = source
        .tensor_names()
        .filter(|n| should_carry_tensor(n))
        .collect();
    names.sort();
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let (shape, values) = source.read_f32(
            name,
            crate::models::granite_speech::runtime_contract::GRANITE_SPEECH_CONTRACT_FAMILY,
        )?;
        out.push(make_write_tensor(remap_tensor_name(name), shape, values));
    }
    Ok(out)
}

/// Dense `[0..vocab_size)` token array: `vocab.json` (token -> id) reversed
/// into id order, with `added_tokens.json` (the `<|audio|>` placeholder and
/// any other tokens outside the base BPE vocab) filling in the remaining
/// slots up to `vocab_size`. Mirrors `tokenizer::GraniteSpeechTokenizer::
/// from_source_files`'s loading logic exactly (that constructor stays the
/// dev/test path reading these same two files directly; this is the
/// production path baking the same array into the pack).
fn load_granite_speech_vocab_tokens(
    source_root: &Path,
    vocab_size: usize,
) -> Result<Vec<String>, LocalSourceImportError> {
    let vocab: BTreeMap<String, u32> = read_source_json_file(source_root, SOURCE_VOCAB_JSON)?;
    let mut tokens = vec![None::<String>; vocab_size];
    for (token, id) in vocab {
        if let Some(slot) = tokens.get_mut(id as usize) {
            *slot = Some(token);
        }
    }
    let added_path = source_root.join(SOURCE_ADDED_TOKENS_JSON);
    let added_tokens: Option<BTreeMap<String, u32>> = std::fs::read(&added_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok());
    if let Some(added) = added_tokens {
        for (token, id) in added {
            if let Some(slot) = tokens.get_mut(id as usize) {
                *slot = Some(token);
            }
        }
    }
    tokens
        .into_iter()
        .enumerate()
        .map(|(id, token)| {
            token.ok_or_else(|| {
                validate_error(format!(
                    "granite-speech tokenizer is missing token for id {id}"
                ))
            })
        })
        .collect()
}

/// Loads the GPT-2 BPE merge list through the shared local-source loader,
/// then fails closed when the source carries no merges: the shared helper
/// tolerates an absent file (some families ship without one), but granite's
/// prompt encode/decode cannot build a tokenizer without its merge table, so
/// an empty result rejects the source instead of writing a broken pack.
fn load_granite_speech_merges(source_root: &Path) -> Result<Vec<String>, LocalSourceImportError> {
    let merges = load_gpt2_bpe_merges(
        source_root,
        super::runtime_contract::GRANITE_SPEECH_CONTRACT_FAMILY,
    )?;
    if merges.is_empty() {
        return Err(validate_error(format!(
            "granite-speech source must carry a non-empty {SOURCE_MERGES_TXT}; the pack tokenizer cannot encode prompts without its BPE merge table"
        )));
    }
    Ok(merges)
}

fn granite_speech_runtime_gguf_metadata(
    config: &GraniteSpeechConfigJson,
    request: &GraniteSpeechImportRequest,
    vocab_tokens: &[String],
    merges: &[String],
) -> BTreeMap<String, GgufWriteValue> {
    let mut metadata = BTreeMap::new();
    insert_metadata(&mut metadata, "openasr.model.id", request.model_id.as_str());
    insert_metadata(
        &mut metadata,
        "granite_speech.audio_token_index",
        config.audio_token_index,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.downsample_rate",
        config.downsample_rate as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.window_size",
        config.window_size as u32,
    );

    let e = &config.encoder_config;
    insert_metadata(
        &mut metadata,
        "granite_speech.encoder.input_dim",
        e.input_dim as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.encoder.hidden_dim",
        e.hidden_dim as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.encoder.num_layers",
        e.num_layers as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.encoder.num_heads",
        e.num_heads as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.encoder.dim_head",
        e.dim_head as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.encoder.feedforward_mult",
        e.feedforward_mult as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.encoder.conv_kernel_size",
        e.conv_kernel_size as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.encoder.conv_expansion_factor",
        e.conv_expansion_factor as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.encoder.context_size",
        e.context_size as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.encoder.max_pos_emb",
        e.max_pos_emb as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.encoder.output_dim",
        e.output_dim as u32,
    );

    let p = &config.projector_config;
    insert_metadata(
        &mut metadata,
        "granite_speech.projector.hidden_size",
        p.hidden_size as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.projector.encoder_hidden_size",
        p.encoder_hidden_size as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.projector.num_hidden_layers",
        p.num_hidden_layers as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.projector.num_attention_heads",
        p.num_attention_heads as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.projector.intermediate_size",
        p.intermediate_size as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.projector.cross_attention_frequency",
        p.cross_attention_frequency as u32,
    );

    let t = &config.text_config;
    insert_metadata(
        &mut metadata,
        "granite_speech.decoder.hidden_size",
        t.hidden_size as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.decoder.num_hidden_layers",
        t.num_hidden_layers as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.decoder.num_attention_heads",
        t.num_attention_heads as u32,
    );
    // HF's `GraniteAttention` reads an explicit `head_dim` config key when
    // present, falling back to `hidden_size / num_attention_heads` only when
    // absent (see `modeling_granite.py`); write the resolved value explicitly
    // rather than have the runtime re-derive it, so a future checkpoint that
    // *does* set `head_dim` separately from `hidden_size/num_heads` (the
    // qwen3-in-moss precedent for this exact class of bug) is not silently
    // computed wrong.
    insert_metadata(
        &mut metadata,
        "granite_speech.decoder.head_dim",
        (t.hidden_size / t.num_attention_heads.max(1)) as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.decoder.num_key_value_heads",
        t.num_key_value_heads as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.decoder.intermediate_size",
        t.intermediate_size as u32,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.decoder.vocab_size",
        t.vocab_size as u32,
    );
    // The four Granite scaling scalars + rms_norm_eps + rope_theta: stored as
    // stringified floats (see module doc -- `GgufWriteValue` has no F32/F64
    // variant, and `insert_metadata` is generic over `ToString`).
    insert_metadata(
        &mut metadata,
        "granite_speech.decoder.rms_norm_eps",
        t.rms_norm_eps,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.decoder.rope_theta",
        t.rope_theta,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.decoder.attention_multiplier",
        t.attention_multiplier,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.decoder.embedding_multiplier",
        t.embedding_multiplier,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.decoder.residual_multiplier",
        t.residual_multiplier,
    );
    insert_metadata(
        &mut metadata,
        "granite_speech.decoder.logits_scaling",
        t.logits_scaling,
    );

    insert_metadata(
        &mut metadata,
        TOKENIZER_GGML_MODEL_KEY,
        TOKENIZER_GGML_MODEL_VALUE_GPT2,
    );
    insert_metadata_string_array(&mut metadata, TOKENIZER_GGML_TOKENS_KEY, vocab_tokens);
    insert_metadata_string_array(&mut metadata, TOKENIZER_GGML_MERGES_KEY, merges);

    metadata
}

pub fn convert_local_granite_speech_source_to_runtime_pack(
    request: &GraniteSpeechImportRequest,
) -> Result<GraniteSpeechImportResult, LocalSourceImportError> {
    validate_output_pack_extension(&request.output_root)?;
    let config: GraniteSpeechConfigJson =
        read_source_json_file(&request.source_root, SOURCE_CONFIG_JSON)?;
    let source = ShardedSafetensorsSource::open(
        &request.source_root,
        SOURCE_INDEX_JSON,
        super::runtime_contract::GRANITE_SPEECH_CONTRACT_FAMILY,
    )?;

    let tensors = build_runtime_tensors(&source)?;
    let vocab_tokens =
        load_granite_speech_vocab_tokens(&request.source_root, config.text_config.vocab_size)?;
    let merges = load_granite_speech_merges(&request.source_root)?;
    let metadata = granite_speech_runtime_gguf_metadata(&config, request, &vocab_tokens, &merges);

    let verified = OasrPackWriter::write(
        &request.output_root,
        PackEnvelope::asr(GRANITE_SPEECH_GGML_ARCHITECTURE_ID),
        metadata,
        &tensors,
    )
    .map_err(|error| {
        validate_error(format!(
            "granite-speech OASR writer failed for '{}': {error}",
            request.output_root.display()
        ))
    })?;

    let tensor_count = verified.preflight().tensor_index().tensors().len();
    Ok(GraniteSpeechImportResult {
        output_path: request.output_root.clone(),
        verified_pack: verified,
        tensor_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::GgufTensorDataReader;

    const SOURCE_ROOT_ENV: &str = "OPENASR_GRANITE_SPEECH_SOURCE_ROOT";

    fn source_root() -> Option<PathBuf> {
        let path = PathBuf::from(std::env::var_os(SOURCE_ROOT_ENV)?);
        path.join(SOURCE_INDEX_JSON).exists().then_some(path)
    }

    #[test]
    fn ggml_dims_for_pack_reverses_rank2_only() {
        // Linear / embedding: safetensors [out, in] -> ggml [in, out].
        assert_eq!(ggml_dims_for_pack(&[1024, 160]), vec![160, 1024]);
        assert_eq!(ggml_dims_for_pack(&[100353, 2048]), vec![2048, 100353]);
        // Square is a no-op on the extents.
        assert_eq!(ggml_dims_for_pack(&[1024, 1024]), vec![1024, 1024]);
        // Bias / norm stays 1-D; depthwise conv / projector.query stay rank-3+.
        assert_eq!(ggml_dims_for_pack(&[1024]), vec![1024]);
        assert_eq!(ggml_dims_for_pack(&[1024, 1, 31]), vec![1024, 1, 31]);
        assert_eq!(ggml_dims_for_pack(&[1, 3, 1024]), vec![1, 3, 1024]);
    }

    #[test]
    fn merges_loader_rides_the_shared_track_and_fails_closed_on_empty() {
        let source_root = tempfile::tempdir().expect("tempdir");
        // Absent merges.txt: the shared loader tolerates it, granite must not.
        let error = load_granite_speech_merges(source_root.path())
            .expect_err("absent merges.txt must fail closed");
        assert!(
            error.to_string().contains("merges.txt"),
            "unexpected error: {error}"
        );
        // Comment + blank lines drop; real merge lines survive trimmed.
        std::fs::write(
            source_root.path().join(SOURCE_MERGES_TXT),
            "#version: 0.2\n\nĠ t\nĠa b\n",
        )
        .expect("write merges.txt");
        let merges = load_granite_speech_merges(source_root.path()).expect("load merges");
        assert_eq!(merges, vec!["Ġ t".to_string(), "Ġa b".to_string()]);
        // A header-only file is still an empty merge table: fail closed.
        std::fs::write(
            source_root.path().join(SOURCE_MERGES_TXT),
            "#version: 0.2\n",
        )
        .expect("write header-only merges.txt");
        assert!(
            load_granite_speech_merges(source_root.path()).is_err(),
            "a merge table with no merges must fail closed"
        );
    }

    /// End-to-end converter smoke test + sampled tensor parity: converts the
    /// real checkpoint to a scratch `.oasr`, then re-reads a handful of
    /// tensors from every carried segment (encoder / projector / decoder) and
    /// diffs them against the source safetensors, dequantized through the
    /// same F16 round-trip the pack stores. `#[ignore]`: needs the local
    /// 4.6 GB checkpoint under `tmp/` (not committed).
    #[test]
    #[ignore = "requires local 4.6GB granite-speech-4.1-2b weights under tmp/ (not committed)"]
    fn granite_speech_converter_round_trips_sampled_tensors() {
        let Some(source_root) = source_root() else {
            eprintln!("skip: set {SOURCE_ROOT_ENV} to a local granite-speech source tree");
            return;
        };
        let output_root = std::env::temp_dir().join("granite_speech_converter_test.oasr");
        let _ = std::fs::remove_file(&output_root);
        let request = GraniteSpeechImportRequest {
            source_root: source_root.clone(),
            output_root: output_root.clone(),
            model_id: "ibm-granite/granite-speech-4.1-2b".to_string(),
        };
        let result =
            convert_local_granite_speech_source_to_runtime_pack(&request).expect("convert");
        assert!(
            result.tensor_count > 900,
            "expected ~954 tensors, got {}",
            result.tensor_count
        );

        let source = ShardedSafetensorsSource::open(
            &source_root,
            SOURCE_INDEX_JSON,
            crate::models::granite_speech::runtime_contract::GRANITE_SPEECH_CONTRACT_FAMILY,
        )
        .expect("open source");
        let reader = GgufTensorDataReader::from_path(&output_root).expect("reader");

        let sample_names = [
            "encoder.input_linear.weight",
            "encoder.layers.7.attn.rel_pos_emb.weight",
            "encoder.out.bias",
            "projector.query",
            "projector.linear.weight",
            "projector.qformer.encoder.layer.0.crossattention.output.LayerNorm.weight",
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.embed_tokens.weight",
        ];
        for name in sample_names {
            let (shape, expected) = source
                .read_f32(
                    name,
                    crate::models::granite_speech::runtime_contract::GRANITE_SPEECH_CONTRACT_FAMILY,
                )
                .unwrap_or_else(|e| {
                panic!("source tensor '{name}' missing (a `lm_head.weight` may be tied to the embedding on this checkpoint -- adjust sample list if so): {e}")
            });
            // Pack stores ggml dims (`[in, out]` for rank-2); the flat f32
            // payload is still the safetensors row-major byte order.
            let pack_shape = ggml_dims_for_pack(&shape);
            let packed_name = remap_tensor_name(name);
            let actual = reader
                .host_tensor_f32_copy_dequantized_by_name(&packed_name, &pack_shape)
                .unwrap_or_else(|e| panic!("read back '{packed_name}' (from '{name}'): {e}"));
            assert_eq!(actual.len(), expected.len(), "'{name}' length mismatch");
            let is_f32 = shape.len() < 2;
            // F16 has ~10 mantissa bits (~4.9e-4 relative rounding step); F32
            // passthrough tensors should be bit-exact modulo the bf16-source
            // upcast that already happened before either path. Elements near
            // zero blow up a pure relative metric (F16's rounding step is
            // absolute near its denormal range, not relative to a near-zero
            // value), so gate small-magnitude elements on absolute error and
            // everything else on relative error -- still tight enough to catch
            // a genuine wiring bug (wrong tensor, wrong shape, transposed data),
            // which would blow either bound by orders of magnitude.
            let rel_tol = if is_f32 { 1.0e-6 } else { 5.0e-3 };
            let abs_tol = if is_f32 { 1.0e-6 } else { 5.0e-4 };
            let small_magnitude_floor = 1.0e-2_f32;
            let mut max_rel = 0.0f32;
            let mut max_abs_small = 0.0f32;
            for (a, e) in actual.iter().zip(expected.iter()) {
                let d = (a - e).abs();
                if e.abs() > small_magnitude_floor {
                    max_rel = max_rel.max(d / e.abs());
                } else {
                    max_abs_small = max_abs_small.max(d);
                }
            }
            assert!(
                max_rel < rel_tol,
                "'{name}' max relative diff {max_rel:.3e} exceeds {rel_tol:.3e}"
            );
            assert!(
                max_abs_small < abs_tol,
                "'{name}' max absolute diff (near-zero elements) {max_abs_small:.3e} exceeds {abs_tol:.3e}"
            );
        }

        // Tokenizer metadata: the GGUF-backed constructor must produce the
        // same encode/decode behavior as the dev/test source-files
        // constructor for the same checkpoint (both build the same dense
        // token array + merge list, just from different files).
        let gguf_metadata =
            crate::ggml_runtime::read_gguf_metadata(&output_root).expect("gguf metadata");
        let gguf_tokenizer =
            super::super::tokenizer::GraniteSpeechTokenizer::from_gguf_metadata(&gguf_metadata)
                .expect("tokenizer from gguf metadata");
        let source_tokenizer =
            super::super::tokenizer::GraniteSpeechTokenizer::from_source_files(&source_root)
                .expect("tokenizer from source files");
        let text = "The quick brown fox jumps over the lazy dog.";
        let gguf_ids = gguf_tokenizer
            .encode_prompt_text(text)
            .expect("gguf encode");
        let source_ids = source_tokenizer
            .encode_prompt_text(text)
            .expect("source encode");
        assert_eq!(
            gguf_ids, source_ids,
            "gguf and source-files tokenizers must encode identically"
        );
        assert_eq!(
            gguf_tokenizer
                .decode_text_token_ids(&gguf_ids)
                .expect("gguf decode"),
            text
        );
    }
}
