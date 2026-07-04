//! Dolphin dedicated executor skeleton (convert + load phase).
//!
//! What is wired here: loading the E-Branchformer encoder weights **from the
//! `.oasr` pack** and running the already parity-verified `encoder_graph::encode`
//! on them (see `encode_dolphin_encoder_from_pack`, exercised by the from-pack
//! parity test). The CTC-prefix-beam + attention-rescoring joint decode and the
//! kaldi fbank frontend are NOT wired yet, so the `GgmlAsrExecutor::execute`
//! transcription entry point validates the pack loads + binds the encoder, then
//! fails closed with a typed error (never fabricates a transcript).

#![allow(dead_code)]

use std::collections::HashMap;

use crate::ggml_runtime::{GgufTensorDataReadError, GgufTensorDataReader};
use crate::models::ggml_asr_executor::{
    GgmlAsrExecutionError, GgmlAsrExecutionRequest, GgmlAsrExecutionResult, GgmlAsrExecutor,
};

use super::encoder_graph::{DolphinEncoderConfig, DolphinEncoderOutput, encode};
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

/// Load every `encoder.*` tensor from the pack, dequantized to f32 and keyed by
/// its exact WeNet state-dict name — the provider shape `encoder_graph::encode`
/// consumes. Reading each tensor at its own stored dims makes this
/// layout-agnostic (the encoder graph re-declares its own ggml shapes).
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
) -> Result<DolphinEncoderOutput, String> {
    let weights = load_dolphin_encoder_weights_from_pack(reader)
        .map_err(|error| format!("dolphin encoder weight load failed: {error}"))?;
    let config = DolphinEncoderConfig::small_cn();
    encode(&config, &weights, features, frames_in)
        .map_err(|error| format!("dolphin encoder graph failed: {error}"))
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
        // Confirm the encoder + CTC namespaces are actually baked before claiming
        // anything about decode support.
        for sentinel in ENCODER_SENTINEL_TENSORS {
            if preflight.tensor_index.get(sentinel).is_none() {
                return Err(fail(format!(
                    "dolphin pack is missing required tensor '{sentinel}'"
                )));
            }
        }
        // Convert + load phase only: the CTC-prefix-beam + attention-rescoring
        // joint decode and the kaldi fbank frontend are not wired yet. Fail closed
        // rather than fabricate a transcript.
        Err(fail(
            "dolphin transcription is not implemented yet (convert+load phase): the CTC/attention \
             joint decode and fbank frontend are not wired. The encoder runs from the pack via the \
             internal encode-from-pack path; end-to-end decode lands in a later phase."
                .to_string(),
        ))
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

    const FIXTURE_ROOT: &str =
        "/Volumes/QuintinDocument/openasr-dev/openasr/tmp/publish/dolphin-cn-dialect-small";

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
        let safetensors = root.join("weights/full.safetensors");
        let units = root.join("src/units.txt");
        if !safetensors.exists() || !units.exists() {
            eprintln!("skip: dolphin checkpoint/units not present under {root:?}");
            return;
        }

        let pack = root.join("packs/dolphin-cn-dialect-small-fp16.oasr");
        std::fs::create_dir_all(pack.parent().unwrap()).expect("create packs dir");
        let _ = std::fs::remove_file(&pack);
        let result = convert_local_dolphin_wenet_source_to_runtime_pack(&DolphinImportRequest {
            safetensors_path: safetensors,
            units_path: units,
            output_path: pack.clone(),
            model_id: "dolphin-cn-dialect-small".to_string(),
            quantization: DolphinQuantizationMode::Fp16,
        })
        .expect("dolphin import");
        eprintln!(
            "wrote {} ({} tensors, vocab {}, blank {})",
            result.output_path.display(),
            result.tensor_count,
            result.vocab_size,
            result.blank_token_id
        );
        assert_eq!(result.vocab_size, 18173);

        // The produced pack must clear the fail-closed install gate (adapter
        // selection + the dolphin runtime-metadata contract) exactly as
        // `openasr pull` would enforce it.
        crate::validate_native_runtime_model_pack_contract(&pack)
            .expect("dolphin pack must pass the native install gate");

        let (in_shape, features) = load_npy_f32(&root.join("golden/logmel_feats_cmvn.npy"));
        assert_eq!(in_shape.len(), 3, "expected (1,T,80), got {in_shape:?}");
        let frames_in = in_shape[1];

        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let output =
            encode_dolphin_encoder_from_pack(&reader, &features, frames_in).expect("encode");

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
}
