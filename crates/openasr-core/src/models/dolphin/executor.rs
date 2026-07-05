//! Dolphin `small.cn` dedicated executor: the full end-to-end transcribe path.
//!
//! Pipeline (all from the `.oasr` pack): kaldi-fbank [`frontend`] + the checkpoint's
//! global CMVN -> the parity-verified E-Branchformer [`encoder_graph`] ->
//! CTC/attention [`joint_decode`] (CTC prefix-beam over the CTC head, rescored by
//! the Transformer [`decoder_graph`]) -> char detokenize. The executor fails closed
//! with typed errors on a bad pack and never fabricates a transcript.
//!
//! [`frontend`]: super::frontend
//! [`joint_decode`]: super::joint_decode
//! [`encoder_graph`]: super::encoder_graph
//! [`decoder_graph`]: super::decoder_graph

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::api::backend::{Segment, Transcription};
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgufMetadata, GgufTensorDataReadError,
    GgufTensorDataReader, RequestBackendPreference, request_backend_override,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrExecutor,
};

use super::decoder_graph::DolphinDecoderConfig;
use super::encoder_graph::{DolphinEncoderConfig, DolphinEncoderOutput, encode};
use super::frontend::{DolphinFbankFrontend, NUM_MEL_BINS, apply_global_cmvn};
use super::joint_decode::{DolphinJointDecodeConfig, detokenize_char_tokens, joint_decode};
use super::language::build_dolphin_decode_prefix;
use super::runtime_contract::parse_dolphin_execution_metadata;

/// Encoder weight namespace baked into the pack under exact WeNet names.
const ENCODER_TENSOR_PREFIX: &str = "encoder.";
/// Sentinels proving the pack baked the encoder + CTC head namespaces (cheap
/// index probe, no dequantization).
const ENCODER_SENTINEL_TENSORS: [&str; 3] = [
    "encoder.embed.pos_enc.pe",
    "encoder.after_norm.weight",
    "ctc.ctc_lo.weight",
];

/// Global CMVN vectors baked in the pack (checkpoint's own `encoder.global_cmvn`).
const CMVN_MEAN_TENSOR: &str = "encoder.global_cmvn.mean";
const CMVN_ISTD_TENSOR: &str = "encoder.global_cmvn.istd";

/// Pack metadata keys the decode reads (mirrors the importer's writes). The
/// decode prefix is no longer read from the pack: it is built per request from the
/// vocab + the requested language code (see [`build_dolphin_decode_prefix`]), so a
/// single pack can honor any advertised dialect region rather than one baked one.
const EOS_TOKEN_ID_KEY: &str = "dolphin.eos_token_id";
const BLANK_TOKEN_ID_KEY: &str = "ctc.blank_token_id";
const TOKENIZER_TOKENS_KEY: &str = "tokenizer.ggml.tokens";

/// CTC prefix-beam width used for joint decode (WeNet default).
const DOLPHIN_BEAM_SIZE: usize = 10;

/// Rescoring combination weight. The reference `attention_rescoring` decode selects
/// purely by attention score over the CTC n-best (`ctc_weight = 0.0`); the model's
/// `0.3` is the *training* loss weight (`model_conf.ctc_weight`), a different knob.
/// Kept `0.0` so the runtime reproduces the golden reference decode.
pub(crate) const DOLPHIN_REFERENCE_RESCORE_CTC_WEIGHT: f32 = 0.0;

/// Load every tensor in the pack, dequantized to f32 and keyed by its exact WeNet
/// name -- the provider shape the encoder/decoder/CTC graphs consume. Reading each
/// tensor at its own stored dims keeps this layout-agnostic (each graph re-declares
/// its own ggml shapes).
pub(crate) fn load_dolphin_runtime_weights_from_pack(
    reader: &GgufTensorDataReader,
) -> Result<HashMap<String, Vec<f32>>, GgufTensorDataReadError> {
    let mut weights = HashMap::new();
    for tensor in reader.tensor_index().tensors() {
        let values = reader.host_tensor_f32_copy_dequantized_by_name(&tensor.name, &tensor.dims)?;
        weights.insert(tensor.name.clone(), values);
    }
    Ok(weights)
}

/// Load only the `encoder.*` tensors from the pack (the encoder-from-pack parity
/// path; the full transcribe path uses [`load_dolphin_runtime_weights_from_pack`]).
pub(crate) fn load_dolphin_encoder_weights_from_pack(
    reader: &GgufTensorDataReader,
) -> Result<HashMap<String, Vec<f32>>, GgufTensorDataReadError> {
    let mut weights = HashMap::new();
    for tensor in reader.tensor_index().tensors() {
        if !tensor.name.starts_with(ENCODER_TENSOR_PREFIX) {
            continue;
        }
        let values = reader.host_tensor_f32_copy_dequantized_by_name(&tensor.name, &tensor.dims)?;
        weights.insert(tensor.name.clone(), values);
    }
    Ok(weights)
}

/// Run the verified E-Branchformer encoder graph on weights loaded from the pack.
/// `features` is the CMVN'd `[frames_in, feature_dim]` log-mel input (frame-major,
/// mel bin innermost), matching the golden `logmel_feats_cmvn` fixture the raw
/// safetensors parity harness uses.
pub(crate) fn encode_dolphin_encoder_from_pack(
    reader: &GgufTensorDataReader,
    features: &[f32],
    frames_in: usize,
    backend: GgmlCpuGraphBackend,
) -> Result<DolphinEncoderOutput, String> {
    let weights = load_dolphin_encoder_weights_from_pack(reader)
        .map_err(|error| format!("dolphin encoder weight load failed: {error}"))?;
    let config = DolphinEncoderConfig::small_cn();
    encode(&config, &weights, features, frames_in, backend)
        .map_err(|error| format!("dolphin encoder graph failed: {error}"))
}

/// A rescored joint-decode hypothesis, detokenized for reporting.
#[derive(Debug, Clone)]
pub(crate) struct DolphinScoredText {
    pub text: String,
    pub ctc_score: f32,
    pub attention_score: f32,
    pub combined_score: f32,
}

/// End-to-end transcription output plus the diagnostics the harness reports.
#[derive(Debug, Clone)]
pub(crate) struct DolphinPipelineOutput {
    /// Best (rescored) transcript.
    pub text: String,
    pub best_token_ids: Vec<u32>,
    /// CTC greedy transcript (pre-rescoring), for comparison.
    pub ctc_greedy_text: String,
    /// Rescored n-best, best-first.
    pub scored_nbest: Vec<DolphinScoredText>,
    /// Normalized recognition code the decode prefix selected (`zh`, `zh-sichuan`,
    /// ...), surfaced so the executor reports the language it actually decoded.
    pub resolved_language: String,
}

/// Resolve the ggml backend for a Dolphin request. Fail-closed to the golden,
/// parity-validated CPU path; a GPU backend engages only when the request
/// explicitly asks for accelerated execution (`--execution-target accelerated`,
/// which the bench-suite maps `OPENASR_GGML_BACKEND=metal` onto), never on the
/// Auto default. Mirrors the xasr encoder policy so what runs is what was asked
/// for, with no silent downgrade.
///
/// Perf note (AB-measured on M1, best-of-5, 2.38 s Sichuan clip; see
/// `perf/PERFORMANCE.md`): unlike xasr's chunked encoder -- where every Metal
/// config loses to CPU -- this 0.4B E-Branchformer is wide enough per step that
/// Metal is ~1.45x FASTER (RTF 0.47 vs CPU 0.68, warm) at comparable peak RSS,
/// and reproduces the golden transcript on the clip. Metal stays an opt-in rather
/// than the default only because its fp16 numerics are not golden-validated
/// (the parity gate is CPU bit-exact); it is the recommended accelerated path.
fn dolphin_runtime_backend() -> GgmlCpuGraphBackend {
    if matches!(
        request_backend_override(),
        Some(RequestBackendPreference::Accelerated)
    ) {
        GgmlCpuGraphConfig::resolve_runtime_backend()
    } else {
        GgmlCpuGraphBackend::Cpu
    }
}

/// Dequantized runtime weights for one pack, shared behind an `Arc` so the
/// process-level pool can hand the same immutable table to every call.
pub(crate) type DolphinRuntimeWeights = HashMap<String, Vec<f32>>;

/// Process-level pool of dequantized weights keyed by pack path. The fp16 pack is
/// ~738 MB on disk and dequantizes to ~1.5 GB of host f32; doing that per request
/// costs ~0.4 s (18% of the single-utterance wall on M1). Caching it lets warm
/// calls skip the reload+dequant and reuse the same immutable table, mirroring the
/// xasr process runtime pool. Keyed by path only: the dequantized weights are
/// host-side f32 and backend-independent, so CPU and Metal runs share one entry.
static DOLPHIN_WEIGHTS_POOL: OnceLock<Mutex<HashMap<PathBuf, Arc<DolphinRuntimeWeights>>>> =
    OnceLock::new();

/// Fetch the pack's dequantized weights from the pool, loading+dequantizing (via
/// the already-resolved `reader`) and caching on a miss. The dequant runs outside
/// the pool lock so concurrent first callers for distinct packs don't serialize.
pub(crate) fn cached_dolphin_runtime_weights(
    cache_key: &Path,
    reader: &GgufTensorDataReader,
) -> Result<Arc<DolphinRuntimeWeights>, String> {
    let pool = DOLPHIN_WEIGHTS_POOL.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(weights) = pool
        .lock()
        .expect("dolphin weights pool lock")
        .get(cache_key)
    {
        return Ok(weights.clone());
    }
    let weights = Arc::new(
        load_dolphin_runtime_weights_from_pack(reader)
            .map_err(|error| format!("dolphin runtime weight load failed: {error}"))?,
    );
    pool.lock()
        .expect("dolphin weights pool lock")
        .insert(cache_key.to_path_buf(), weights.clone());
    Ok(weights)
}

/// The complete Dolphin transcribe pipeline over 16 kHz mono PCM (`samples` in
/// `[-1, 1]`): fbank + CMVN -> encoder -> CTC/attention joint decode -> detokenize.
/// Loads the pack's weights from `reader` each call (the uncached path the parity
/// harness drives); the executor uses [`cached_dolphin_runtime_weights`] +
/// [`run_dolphin_pipeline`] to reuse weights across requests.
pub(crate) fn transcribe_dolphin_pcm(
    reader: &GgufTensorDataReader,
    metadata: &GgufMetadata,
    samples: &[f32],
    ctc_weight: f32,
    backend: GgmlCpuGraphBackend,
    language: Option<&str>,
) -> Result<DolphinPipelineOutput, String> {
    let weights = load_dolphin_runtime_weights_from_pack(reader)
        .map_err(|error| format!("dolphin runtime weight load failed: {error}"))?;
    run_dolphin_pipeline(&weights, metadata, samples, ctc_weight, backend, language)
}

/// Run the fbank+CMVN -> encoder -> joint-decode -> detokenize pipeline over
/// already-loaded `weights`. Split out from [`transcribe_dolphin_pcm`] so the
/// executor can reuse pooled weights across requests without re-dequantizing.
pub(crate) fn run_dolphin_pipeline(
    weights: &DolphinRuntimeWeights,
    metadata: &GgufMetadata,
    samples: &[f32],
    ctc_weight: f32,
    backend: GgmlCpuGraphBackend,
    language: Option<&str>,
) -> Result<DolphinPipelineOutput, String> {
    let tokens = metadata
        .get_string_array(TOKENIZER_TOKENS_KEY)
        .ok_or_else(|| format!("dolphin pack is missing the '{TOKENIZER_TOKENS_KEY}' vocab"))?;
    // Build the `<sos> <zh> <region> <asr> <notimestamp>` prefix per request from
    // the pack vocab; fail closed (typed) on an unknown code or a missing region.
    let prefix = build_dolphin_decode_prefix(tokens, language)
        .map_err(|error| format!("dolphin decode prefix build failed: {error}"))?;
    let eos_token_id = metadata
        .get_u32(EOS_TOKEN_ID_KEY)
        .ok_or_else(|| format!("dolphin pack is missing '{EOS_TOKEN_ID_KEY}'"))?;
    let blank_token_id = metadata
        .get_u32(BLANK_TOKEN_ID_KEY)
        .ok_or_else(|| format!("dolphin pack is missing '{BLANK_TOKEN_ID_KEY}'"))?;

    // Frontend: kaldi fbank -> global CMVN (the exact tensor the encoder consumes).
    let mut features = DolphinFbankFrontend::new()
        .compute(samples)
        .map_err(|error| format!("dolphin fbank frontend failed: {error}"))?;
    let cmvn_mean = weights
        .get(CMVN_MEAN_TENSOR)
        .ok_or_else(|| format!("dolphin pack is missing '{CMVN_MEAN_TENSOR}'"))?;
    let cmvn_istd = weights
        .get(CMVN_ISTD_TENSOR)
        .ok_or_else(|| format!("dolphin pack is missing '{CMVN_ISTD_TENSOR}'"))?;
    apply_global_cmvn(&mut features.data, NUM_MEL_BINS, cmvn_mean, cmvn_istd)
        .map_err(|error| format!("dolphin global CMVN failed: {error}"))?;

    // Encoder (parity-verified).
    let encoder_config = DolphinEncoderConfig::small_cn();
    let encoder = encode(
        &encoder_config,
        weights,
        &features.data,
        features.n_frames,
        backend,
    )
    .map_err(|error| format!("dolphin encoder graph failed: {error}"))?;

    // CTC/attention joint decode.
    let decoder_config = DolphinDecoderConfig::small_cn();
    let decode_config = DolphinJointDecodeConfig {
        beam_size: DOLPHIN_BEAM_SIZE,
        ctc_weight,
        prompt_prefix: prefix.token_ids,
        eos_token_id,
        blank_token_id,
    };
    let decoded = joint_decode(
        &decoder_config,
        weights,
        &encoder.encoder_out,
        encoder.frames,
        &decode_config,
        backend,
    )
    .map_err(|error| format!("dolphin joint decode failed: {error}"))?;

    let text = detokenize_char_tokens(&decoded.best_token_ids, tokens);
    let ctc_greedy_text = detokenize_char_tokens(&decoded.ctc_greedy_token_ids, tokens);
    let scored_nbest = decoded
        .scored_nbest
        .iter()
        .map(|hyp| DolphinScoredText {
            text: detokenize_char_tokens(&hyp.token_ids, tokens),
            ctc_score: hyp.ctc_score,
            attention_score: hyp.attention_score,
            combined_score: hyp.combined_score,
        })
        .collect();

    Ok(DolphinPipelineOutput {
        text,
        best_token_ids: decoded.best_token_ids,
        ctc_greedy_text,
        scored_nbest,
        resolved_language: prefix.resolved_language,
    })
}

/// Dedicated `GgmlAsrExecutor` for the Dolphin family (DedicatedRuntimeExecutorV1).
#[derive(Debug, Clone, Default)]
pub(crate) struct DolphinGgmlExecutor;

impl GgmlAsrExecutor for DolphinGgmlExecutor {
    fn executor_id(&self) -> &'static str {
        crate::arch::DOLPHIN_EXECUTOR_COMPONENT_ID
    }

    fn supports_phrase_bias(&self) -> bool {
        // Hotword biasing rides the `context_module.*` tensors, which the importer
        // intentionally drops in this phase; report unsupported honestly.
        false
    }

    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        let fail = |reason: String| GgmlAsrExecutionError::ExecutorFailed {
            executor_id: crate::arch::DOLPHIN_EXECUTOR_COMPONENT_ID,
            adapter_id: request.selected_family.adapter_id,
            reason,
        };
        // Gate-0: validate the runtime source and load its metadata + tensor index.
        let preflight = request
            .resolve_runtime_source_preflight()
            .map_err(|error| fail(error.to_string()))?;
        // Fail closed on an incomplete pack (missing runtime scalar keys).
        parse_dolphin_execution_metadata(&preflight.metadata)
            .map_err(|error| fail(format!("dolphin runtime metadata contract failed: {error}")))?;
        // Confirm the encoder + CTC namespaces are actually baked before decoding.
        for sentinel in ENCODER_SENTINEL_TENSORS {
            if preflight.tensor_index.get(sentinel).is_none() {
                return Err(fail(format!(
                    "dolphin pack is missing required tensor '{sentinel}'"
                )));
            }
        }

        let backend = dolphin_runtime_backend();
        let reader = GgufTensorDataReader::from_runtime_source(&preflight.runtime_source)
            .map_err(|error| fail(format!("dolphin pack tensor reader failed: {error}")))?;
        // Reuse dequantized weights across requests (pool keyed by pack path); the
        // ~0.4 s reload+dequant is paid once, later requests are compute-only.
        let weights =
            cached_dolphin_runtime_weights(&request.runtime_source_path, &reader).map_err(fail)?;
        // Thread the request language into the decode prefix builder; an
        // unsupported code / missing region token fails closed here (typed).
        let output = run_dolphin_pipeline(
            &weights,
            &preflight.metadata,
            &request.prepared_audio.samples_f32,
            DOLPHIN_REFERENCE_RESCORE_CTC_WEIGHT,
            backend,
            request.request_options.language.as_deref(),
        )
        .map_err(fail)?;

        let duration = request.prepared_audio.samples_f32.len() as f32
            / request.prepared_audio.sample_rate_hz.max(1) as f32;
        let segments = if output.text.is_empty() {
            Vec::new()
        } else {
            vec![Segment {
                start: 0.0,
                end: duration,
                text: output.text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            }]
        };
        Ok(GgmlAsrExecutionResult {
            transcription: Transcription {
                text: output.text,
                segments,
                longform: None,
                // Surface the region/language the prefix actually selected (the
                // model does not detect it, but the selection is a genuine input);
                // the transcribe layer prefers this per the SpecifyOnly mode.
                language: Some(output.resolved_language),
            },
            carry_context: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::dolphin::package_import::{
        DolphinImportRequest, DolphinQuantizationMode,
        convert_local_dolphin_wenet_source_to_runtime_pack,
    };
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;

    const FIXTURE_ROOT: &str =
        "/Volumes/QuintinDocument/openasr-dev/openasr/tmp/publish/dolphin-cn-dialect-small";

    /// Golden `attention_rescoring` transcript (manifest `text_nospecial`): the
    /// model's own joint-decode output for the Sichuan clip. This is the parity
    /// target -- the human ground-truth WSC transcript differs by one homophone
    /// (河 vs 和), a model-accuracy gap, not an implementation gap.
    const REFERENCE_RESCORING_TEXT: &str = "学校和底下好多那种野生枸杞";
    /// Human ground-truth transcript (manifest `reference_transcript_wsc`).
    const REFERENCE_WSC_TEXT: &str = "学校河底下好多那种野生枸杞";
    /// Reference CTC greedy transcript (manifest `ctc_greedy_search.text`).
    const REFERENCE_CTC_GREEDY_TEXT: &str = "学校火底下好多那种野生枸杞";

    fn root() -> PathBuf {
        PathBuf::from(FIXTURE_ROOT)
    }

    // --- minimal little-endian f32 .npy reader (mirrors parity.rs) -------------
    fn load_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
        let bytes = std::fs::read(path).expect("read npy");
        assert_eq!(&bytes[..6], b"\x93NUMPY", "npy magic");
        let major = bytes[6];
        let header_len = if major == 1 {
            u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize
        } else {
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize
        };
        let header_start = if major == 1 { 10 } else { 12 };
        let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
            .expect("npy header utf8");
        assert!(header.contains("'<f4'"), "expected <f4 npy, got {header}");
        assert!(
            header.contains("'fortran_order': False"),
            "expected C order"
        );
        let shape_start = header.find("'shape':").expect("shape key");
        let paren = header[shape_start..].find('(').unwrap() + shape_start;
        let close = header[paren..].find(')').unwrap() + paren;
        let shape: Vec<usize> = header[paren + 1..close]
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        let data_start = header_start + header_len;
        let values: Vec<f32> = bytes[data_start..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (shape, values)
    }

    fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        actual
            .iter()
            .zip(expected)
            .fold(0.0f32, |m, (a, e)| m.max((a - e).abs()))
    }

    fn relative_max_diff(actual: &[f32], expected: &[f32]) -> f32 {
        let max = max_abs_diff(actual, expected);
        let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if scale > 0.0 { max / scale } else { max }
    }

    /// Char-level edit distance (Levenshtein) over Unicode scalar values.
    fn char_edit_distance(a: &str, b: &str) -> usize {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        let mut prev: Vec<usize> = (0..=b.len()).collect();
        let mut cur = vec![0usize; b.len() + 1];
        for (i, &ca) in a.iter().enumerate() {
            cur[0] = i + 1;
            for (j, &cb) in b.iter().enumerate() {
                let cost = usize::from(ca != cb);
                cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        prev[b.len()]
    }

    fn char_error_rate(hypothesis: &str, reference: &str) -> f32 {
        let ref_len = reference.chars().count();
        if ref_len == 0 {
            return if hypothesis.is_empty() { 0.0 } else { 1.0 };
        }
        char_edit_distance(hypothesis, reference) as f32 / ref_len as f32
    }

    /// Produce-if-absent the fp16 `.oasr` pack at its stable path, exactly once
    /// per process, and hand every caller the same path. The two heavy `#[ignore]`
    /// tests below (a producer + a consumer) share this so a fresh checkout runs
    /// the convert exactly once and later callers reuse the result.
    ///
    /// The write is atomic: the pack is built into a uniquely-named temp file in
    /// the packs dir and then `rename`d into place. Same-directory rename is
    /// atomic on the local fs, so a reader that opens the stable path never
    /// observes a half-written or missing pack -- the path always resolves to a
    /// complete pack (the previous one, or the freshly renamed one), and a reader
    /// holding an fd keeps reading its complete inode across the swap. This is
    /// what removes the earlier producer/consumer race (the old producer did
    /// `remove_file` + in-place rewrite, opening a window where the consumer read
    /// an absent/torn pack); the `dolphin-pack` nextest test-group additionally
    /// serializes the two so they never even overlap. Returns `None` when the
    /// local checkpoint is absent (the tests skip).
    fn ensure_dolphin_pack(root: &Path) -> Option<PathBuf> {
        static PACK: OnceLock<Option<PathBuf>> = OnceLock::new();
        PACK.get_or_init(|| produce_dolphin_pack_atomic(root))
            .clone()
    }

    fn produce_dolphin_pack_atomic(root: &Path) -> Option<PathBuf> {
        let pack = root.join("packs/dolphin-cn-dialect-small-fp16.oasr");
        if pack.exists() {
            return Some(pack);
        }
        let safetensors = root.join("weights/full.safetensors");
        let units = root.join("src/units.txt");
        if !safetensors.exists() || !units.exists() {
            return None;
        }
        let packs_dir = pack.parent().expect("pack has a parent dir");
        std::fs::create_dir_all(packs_dir).expect("create packs dir");
        // Reserve a uniquely-named temp `.oasr` path in the same dir (the `.oasr`
        // suffix keeps the importer's output-extension gate happy; the unique name
        // means two concurrent producers in distinct processes never collide).
        // The GGUF writer refuses to clobber an existing file, so drop the empty
        // reservation file and let the writer create it fresh; `TempPath` still
        // cleans it up on an early return, and `persist` publishes it with an
        // atomic same-dir rename.
        let temp_path = tempfile::Builder::new()
            .prefix(".dolphin-pack-")
            .suffix(".oasr")
            .tempfile_in(packs_dir)
            .expect("create temp pack")
            .into_temp_path();
        std::fs::remove_file(&temp_path).expect("clear temp reservation");
        convert_local_dolphin_wenet_source_to_runtime_pack(&DolphinImportRequest {
            safetensors_path: safetensors,
            units_path: units,
            output_path: temp_path.to_path_buf(),
            model_id: "dolphin-cn-dialect-small".to_string(),
            quantization: DolphinQuantizationMode::Fp16,
        })
        .expect("dolphin import");
        temp_path.persist(&pack).expect("publish dolphin pack");
        Some(pack)
    }

    /// Produce the fp16 `.oasr` pack from the local WeNet checkpoint and assert
    /// the encoder-from-pack matches the golden `encoder_out` within an fp16
    /// quantization tolerance. This is the convert+load gate: the pack loads, the
    /// encoder weights bind under their WeNet names, and the verified encoder
    /// graph reproduces the golden output (the f32-exact bit-level gate stays in
    /// `parity::dolphin_encoder_parity`).
    ///
    /// `#[ignore]`: needs the 1.7 GB checkpoint under `tmp/publish` (never
    /// committed). Run with:
    /// `cargo test -p openasr-core dolphin_encoder_from_pack_parity -- --ignored --nocapture`
    #[test]
    #[ignore = "requires local Dolphin checkpoint + golden under tmp/publish (not committed)"]
    fn dolphin_encoder_from_pack_parity() {
        let root = root();
        let Some(pack) = ensure_dolphin_pack(&root) else {
            eprintln!("skip: dolphin checkpoint/units not present under {root:?}");
            return;
        };

        // The produced pack must clear the fail-closed install gate (adapter
        // selection + the dolphin runtime-metadata contract) exactly as
        // `openasr pull` would enforce it.
        crate::validate_native_runtime_model_pack_contract(&pack)
            .expect("dolphin pack must pass the native install gate");

        // Vocab is a property of the produced pack (the char tokenizer table the
        // importer baked from `units.txt`).
        let pack_metadata = crate::ggml_runtime::read_gguf_metadata(&pack).expect("pack metadata");
        let vocab_size = pack_metadata
            .get_string_array(TOKENIZER_TOKENS_KEY)
            .expect("pack carries the tokenizer vocab")
            .len();
        assert_eq!(vocab_size, 18173);

        let (in_shape, features) = load_npy_f32(&root.join("golden/logmel_feats_cmvn.npy"));
        assert_eq!(in_shape.len(), 3, "expected (1,T,80), got {in_shape:?}");
        let frames_in = in_shape[1];

        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let output = encode_dolphin_encoder_from_pack(
            &reader,
            &features,
            frames_in,
            GgmlCpuGraphBackend::Cpu,
        )
        .expect("encode");

        let (_, golden_out) = load_npy_f32(&root.join("golden/encoder_out.npy"));
        let max = max_abs_diff(&output.encoder_out, &golden_out);
        let rel = relative_max_diff(&output.encoder_out, &golden_out);
        eprintln!("dolphin encoder-from-pack (fp16): max abs {max:.3e}  rel {rel:.3e}");

        // fp16-weight tolerance: the graph itself is bit-exact (proven by the
        // raw-f32 `dolphin_encoder_parity`); the only delta here is fp16 rounding
        // of the rank>=2 weights through 12 E-Branchformer blocks. Measured on the
        // committed golden: relative max diff ~3e-4 (abs ~2.4e-3). The gate sits an
        // order of magnitude above that so thread-order/fp16 jitter is fine, but an
        // algorithmic/layout bug (which blows this up by orders of magnitude) still
        // trips it.
        assert!(
            rel < 3.0e-3,
            "encoder-from-pack relative max diff {rel:.3e} exceeds the fp16 tolerance 3e-3"
        );
    }

    /// M1 CPU-vs-Metal x with/without weight-reuse AB harness. One config per
    /// invocation (selected by env) so `peak_rss_bytes` (process-global
    /// `ru_maxrss` high-water) is isolated per process; the driver script runs it
    /// 4x. Prints a machine-greppable `DOLPHIN_AB ...` line with best-of-N RTF and
    /// peak RSS. Never asserts a timing number (host-dependent); it only measures.
    ///
    /// Env: `OPENASR_DOLPHIN_AB_BACKEND=cpu|metal` (default cpu),
    /// `OPENASR_DOLPHIN_AB_REUSE=0|1` (default 0 = reload+dequant each run),
    /// `OPENASR_DOLPHIN_AB_RUNS=<n>` (default 3).
    #[test]
    #[ignore = "perf AB harness: requires local Dolphin checkpoint + golden clip under tmp/publish"]
    fn dolphin_perf_ab() {
        use std::time::{Duration, Instant};
        let root = root();
        let clip = root.join("golden/clip_sichuan.wav");
        let Some(pack) = ensure_dolphin_pack(&root) else {
            eprintln!("skip: dolphin checkpoint/units not present under {root:?}");
            return;
        };
        if !clip.exists() {
            eprintln!("skip: golden clip not present at {clip:?}");
            return;
        }
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            &clip,
            "dolphin AB",
            "clip_sichuan.wav",
        )
        .expect("load clip");
        let audio_s = samples.len() as f64 / 16_000.0;
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = crate::ggml_runtime::read_gguf_metadata(&pack).expect("metadata");

        let backend = match std::env::var("OPENASR_DOLPHIN_AB_BACKEND").as_deref() {
            Ok("metal") | Ok("gpu") => GgmlCpuGraphBackend::Metal,
            _ => GgmlCpuGraphBackend::Cpu,
        };
        let reuse = matches!(
            std::env::var("OPENASR_DOLPHIN_AB_REUSE").as_deref(),
            Ok("1")
        );
        let runs: usize = std::env::var("OPENASR_DOLPHIN_AB_RUNS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(3)
            .max(1);
        let ctc_weight = DOLPHIN_REFERENCE_RESCORE_CTC_WEIGHT;

        // Reuse == load+dequant the ~1.5 GB f32 weights once, reuse across runs
        // (the pooled executor path). No-reuse == reload+dequant every run (the
        // cold per-request cost). Best-of-N wall time isolates the reuse delta.
        let preloaded =
            reuse.then(|| load_dolphin_runtime_weights_from_pack(&reader).expect("weights"));
        let mut best = Duration::MAX;
        let mut text = String::new();
        for _ in 0..runs {
            let started = Instant::now();
            let output = if let Some(weights) = preloaded.as_ref() {
                run_dolphin_pipeline(
                    weights,
                    &metadata,
                    &samples,
                    ctc_weight,
                    backend,
                    Some("zh-sichuan"),
                )
            } else {
                let weights =
                    load_dolphin_runtime_weights_from_pack(&reader).expect("weights reload");
                run_dolphin_pipeline(
                    &weights,
                    &metadata,
                    &samples,
                    ctc_weight,
                    backend,
                    Some("zh-sichuan"),
                )
            }
            .expect("dolphin pipeline");
            best = best.min(started.elapsed());
            text = output.text;
        }
        let rtf = best.as_secs_f64() / audio_s;
        let peak_rss_mb = crate::metrics::peak_rss_bytes()
            .map(|bytes| bytes as f64 / 1.0e6)
            .unwrap_or(0.0);
        eprintln!(
            "DOLPHIN_AB backend={backend:?} reuse={reuse} runs={runs} audio={audio_s:.2}s \
             best={best:?} RTF={rtf:.3} peak_rss={peak_rss_mb:.0}MB text={text}"
        );
    }

    /// Full end-to-end joint-decode harness: read the Sichuan clip, run
    /// fbank+CMVN -> encoder -> CTC/attention rescoring from the produced `.oasr`
    /// pack, print the transcript + CER, and assert the rescored transcript
    /// reproduces the golden `attention_rescoring` output exactly (CER 0).
    ///
    /// `#[ignore]`: needs the checkpoint/golden under `tmp/publish` (not committed).
    /// Run with:
    /// `cargo test -p openasr-core dolphin_joint_decode_end_to_end -- --ignored --nocapture`
    #[test]
    #[ignore = "requires local Dolphin checkpoint + golden clip under tmp/publish (not committed)"]
    fn dolphin_joint_decode_end_to_end() {
        let root = root();
        let clip = root.join("golden/clip_sichuan.wav");
        let Some(pack) = ensure_dolphin_pack(&root) else {
            eprintln!("skip: dolphin checkpoint/units not present under {root:?}");
            return;
        };
        if !clip.exists() {
            eprintln!("skip: golden clip not present at {clip:?}");
            return;
        }

        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            &clip,
            "dolphin end-to-end harness",
            "clip_sichuan.wav",
        )
        .expect("load clip");

        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = crate::ggml_runtime::read_gguf_metadata(&pack).expect("metadata");

        // Reference-faithful decode: attention-only selection over the CTC n-best
        // (ctc_weight 0.0), the WeNet `attention_rescoring` default. The Sichuan
        // clip is decoded under the `zh-sichuan` prefix -- the same
        // `<sos> <zh> <SICHUAN> <asr> <notimestamp>` ids the pack used to bake --
        // so the golden transcript stays bit-exact through the per-code builder.
        let output = transcribe_dolphin_pcm(
            &reader,
            &metadata,
            &samples,
            DOLPHIN_REFERENCE_RESCORE_CTC_WEIGHT,
            GgmlCpuGraphBackend::Cpu,
            Some("zh-sichuan"),
        )
        .expect("dolphin transcribe");

        let cer_vs_rescoring = char_error_rate(&output.text, REFERENCE_RESCORING_TEXT);
        let cer_vs_wsc = char_error_rate(&output.text, REFERENCE_WSC_TEXT);

        eprintln!("== Dolphin CTC/attention joint decode (end-to-end) ==");
        eprintln!("transcript (rescored) : {}", output.text);
        eprintln!("reference (rescoring) : {REFERENCE_RESCORING_TEXT}");
        eprintln!("reference (human WSC) : {REFERENCE_WSC_TEXT}");
        eprintln!("ctc greedy            : {}", output.ctc_greedy_text);
        eprintln!("ctc greedy (reference): {REFERENCE_CTC_GREEDY_TEXT}");
        eprintln!(
            "CER vs rescoring ref  : {:.4}  ({} edits / {} chars)",
            cer_vs_rescoring,
            char_edit_distance(&output.text, REFERENCE_RESCORING_TEXT),
            REFERENCE_RESCORING_TEXT.chars().count()
        );
        eprintln!(
            "CER vs human WSC ref  : {:.4}  ({} edits / {} chars)",
            cer_vs_wsc,
            char_edit_distance(&output.text, REFERENCE_WSC_TEXT),
            REFERENCE_WSC_TEXT.chars().count()
        );
        eprintln!("rescored n-best (best-first):");
        for hyp in &output.scored_nbest {
            eprintln!(
                "  combined {:8.3}  attn {:8.3}  ctc {:8.3}  {}",
                hyp.combined_score, hyp.attention_score, hyp.ctc_score, hyp.text
            );
        }

        // Also report what the task-mentioned 0.3 rescoring weight would pick, to
        // show the training-vs-decode ctc_weight distinction concretely.
        let output_03 = transcribe_dolphin_pcm(
            &reader,
            &metadata,
            &samples,
            0.3,
            GgmlCpuGraphBackend::Cpu,
            Some("zh-sichuan"),
        )
        .expect("dolphin transcribe (ctc_weight 0.3)");
        eprintln!(
            "with ctc_weight 0.3   : {}  (CER vs rescoring ref {:.4})",
            output_03.text,
            char_error_rate(&output_03.text, REFERENCE_RESCORING_TEXT)
        );

        // Sanity: the CTC greedy path reproduces the reference greedy transcript.
        assert_eq!(
            output.ctc_greedy_text, REFERENCE_CTC_GREEDY_TEXT,
            "CTC greedy transcript diverged from the reference"
        );
        // Parity: the rescored transcript reproduces the golden attention_rescoring
        // output exactly (the 河/和 homophone gap to the human WSC transcript is a
        // model-accuracy artifact the reference decode shares).
        assert_eq!(
            output.text, REFERENCE_RESCORING_TEXT,
            "rescored transcript diverged from the golden attention_rescoring output"
        );
        assert_eq!(
            cer_vs_rescoring, 0.0,
            "CER against the rescoring reference must be 0"
        );
        // The decoded region/language is surfaced honestly (not None).
        assert_eq!(output.resolved_language, "zh-sichuan");
    }
}
