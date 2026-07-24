//! The FireRedASR2-LLM Adapter: 2x frame-stacking + `Linear -> ReLU -> Linear`
//! (upstream `fireredasr2/models/module/adapter.py`, 32 lines, fully
//! transcribed here), run as a small ggml graph.
//!
//! This used to be a plain host-side computation (see git history): the two
//! weight matrices are small (~22M params total, ~88MB f32), so a hand-rolled
//! `mul_mat` graph looked like unneeded plumbing for a one-shot-per-utterance
//! op. Measurement on real packs proved that wrong -- the host path called
//! `host_tensor_f32_copy_dequantized_by_name`, fully dequantizing both weight
//! matrices to f32 on every `execute` (88MB, scalar Rust), then ran the
//! ~2.76B multiply-adds (137 output frames x (3584x2048 + 3584x3584) for the
//! q4_k pack's dims) through a hand-written scalar double loop with no SIMD.
//! That measured at 2868ms -- 18.4% of the whole q4_k/Metal `execute` call,
//! more than the entire 16-layer Conformer encoder (1420ms) it follows. A
//! ggml graph instead keeps the weights quantized (`ggml_mul_mat` dequantizes
//! on the fly, fused into the SIMD/Accelerate-backed kernel) and runs the
//! whole forward as two `mul_mat`s.
//!
//! Frame-stacking is free here: the encoder's output rows are token-major
//! `[frame][hidden]` contiguous f32, i.e. exactly ggml's `ne0=hidden,
//! ne1=frame_count` convention (hidden fastest-varying/contiguous). Since
//! `downsample_rate` adjacent frames' hidden vectors are already back-to-back
//! in memory, `ggml_reshape_2d` into `ne0=hidden*downsample_rate,
//! ne1=frame_count/downsample_rate` produces exactly the "concatenate
//! `downsample_rate` adjacent frames into one wider row" the upstream
//! `adapter.py::forward` does -- no permute/cont required. The on-disk weight
//! tensors are already stored `[input_width, output_width]` (ggml's
//! `mul_mat` lhs convention), so they're used directly via
//! `GgmlLoadedTensor::as_graph_tensor`, matching every other projection in
//! this crate (e.g. `firered_aed::encoder_graph`'s conformer projections).
//!
//! Runs on its own small [`GgmlCpuGraphRunner`] + [`GgmlLoadedWeightContext`]
//! (a second small mmap of the same `.oasr` GGUF the encoder already opened --
//! cheap, page-cache-backed), reusing the encoder's exact backend/thread
//! policy ([`firered_encoder_graph_config`]) since the adapter is the very
//! next stage in the same pipeline and should follow the same Auto-Metal/CPU
//! choice the encoder made.

use std::path::Path;

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlLoadedTensor, GgmlLoadedWeightContext,
};
use crate::models::firered_aed::graph_config::firered_encoder_graph_config;

use super::tensor_names::{
    ADAPTER_LINEAR1_BIAS, ADAPTER_LINEAR1_WEIGHT, ADAPTER_LINEAR2_BIAS, ADAPTER_LINEAR2_WEIGHT,
};

const ADAPTER_ENC_ROWS_TENSOR_NAME: &str = "firered_llm_adapter_enc_rows";

#[derive(Debug, Error)]
pub(crate) enum FireRedLlmAdapterError {
    #[error("firered-llm adapter runtime failed at '{step}': {source}")]
    GraphBuildFailed {
        step: &'static str,
        #[source]
        source: GgmlCpuGraphError,
    },
    #[error("firered-llm adapter is missing tensor '{name}'")]
    MissingTensor { name: &'static str },
    #[error(
        "firered-llm adapter encoder rows shape is invalid: frame_count={frame_count} encoder_d_model={encoder_d_model} values_len={values_len}"
    )]
    InvalidEncoderRowsShape {
        frame_count: usize,
        encoder_d_model: usize,
        values_len: usize,
    },
    #[error(
        "firered-llm adapter has zero output frames (frame_count={frame_count} < downsample_rate={downsample_rate})"
    )]
    NoOutputFrames {
        frame_count: usize,
        downsample_rate: usize,
    },
    #[error("firered-llm adapter graph execution failed: {reason}")]
    GraphExecutionFailed { reason: String },
    #[error("firered-llm adapter output contains non-finite values")]
    NonFiniteValues,
    #[error("firered-llm adapter shape overflowed")]
    ShapeOverflow,
}

fn map_err(step: &'static str, source: GgmlCpuGraphError) -> FireRedLlmAdapterError {
    FireRedLlmAdapterError::GraphBuildFailed { step, source }
}

fn tensor(
    loaded: &GgmlLoadedWeightContext,
    name: &'static str,
) -> Result<GgmlLoadedTensor, FireRedLlmAdapterError> {
    loaded
        .tensor(name)
        .ok_or(FireRedLlmAdapterError::MissingTensor { name })
}

/// Owns the adapter's own small ggml runner + mmap'd weight context across
/// calls, mirroring `FireRedEncoderGraphRuntime`'s shape (own runtime, single-
/// shot graph per call -- the adapter runs exactly once per utterance so
/// there is no incremental state to persist across calls).
pub(crate) struct FireRedLlmAdapterGraphRuntime {
    runner: GgmlCpuGraphRunner,
    _loaded: GgmlLoadedWeightContext,
    linear1_weight: GgmlLoadedTensor,
    linear1_bias: GgmlLoadedTensor,
    linear2_weight: GgmlLoadedTensor,
    linear2_bias: GgmlLoadedTensor,
}

impl FireRedLlmAdapterGraphRuntime {
    pub(crate) fn new(runtime_path: &Path) -> Result<Self, FireRedLlmAdapterError> {
        let runner = GgmlCpuGraphRunner::new(firered_encoder_graph_config())
            .map_err(|source| map_err("runner_init", source))?;
        let loaded = runner
            .load_gguf_weight_context(runtime_path)
            .map_err(|source| map_err("load_gguf_weight_context", source))?;
        let linear1_weight = tensor(&loaded, ADAPTER_LINEAR1_WEIGHT)?;
        let linear1_bias = tensor(&loaded, ADAPTER_LINEAR1_BIAS)?;
        let linear2_weight = tensor(&loaded, ADAPTER_LINEAR2_WEIGHT)?;
        let linear2_bias = tensor(&loaded, ADAPTER_LINEAR2_BIAS)?;
        Ok(Self {
            runner,
            _loaded: loaded,
            linear1_weight,
            linear1_bias,
            linear2_weight,
            linear2_bias,
        })
    }

    /// Run the Adapter over a full utterance's encoder output. Upstream
    /// (`adapter.py::forward`): drop the trailing `seq_len % downsample_rate`
    /// frames, reshape adjacent `downsample_rate` frames into one wider row,
    /// `linear1 -> relu -> linear2`. Returns (token-major output rows,
    /// `output_frame_count`); `output_frame_count = frame_count /
    /// downsample_rate` (integer division, matching upstream's truncation,
    /// not rounding).
    pub(crate) fn run(
        &mut self,
        encoder_rows: &[f32],
        frame_count: usize,
        encoder_d_model: usize,
        downsample_rate: usize,
        llm_dim: usize,
    ) -> Result<(Vec<f32>, usize), FireRedLlmAdapterError> {
        let expected_len = frame_count.checked_mul(encoder_d_model).ok_or(
            FireRedLlmAdapterError::InvalidEncoderRowsShape {
                frame_count,
                encoder_d_model,
                values_len: encoder_rows.len(),
            },
        )?;
        if encoder_rows.len() != expected_len {
            return Err(FireRedLlmAdapterError::InvalidEncoderRowsShape {
                frame_count,
                encoder_d_model,
                values_len: encoder_rows.len(),
            });
        }
        if encoder_rows.iter().any(|value| !value.is_finite()) {
            return Err(FireRedLlmAdapterError::NonFiniteValues);
        }

        let downsample_rate = downsample_rate.max(1);
        let output_frame_count = frame_count / downsample_rate;
        if output_frame_count == 0 {
            return Err(FireRedLlmAdapterError::NoOutputFrames {
                frame_count,
                downsample_rate,
            });
        }
        let stacked_width = encoder_d_model
            .checked_mul(downsample_rate)
            .ok_or(FireRedLlmAdapterError::ShapeOverflow)?;
        // Truncate trailing `frame_count % downsample_rate` frames by only
        // uploading the frames that form a complete stacked group -- matches
        // upstream's `seq_len // downsample_rate * downsample_rate` slice.
        let valid_frame_count = output_frame_count
            .checked_mul(downsample_rate)
            .ok_or(FireRedLlmAdapterError::ShapeOverflow)?;
        let valid_len = valid_frame_count
            .checked_mul(encoder_d_model)
            .ok_or(FireRedLlmAdapterError::ShapeOverflow)?;

        let mut graph = self.runner.start_graph();
        let enc_rows = graph
            .new_tensor_2d_f32(
                encoder_d_model,
                valid_frame_count,
                ADAPTER_ENC_ROWS_TENSOR_NAME,
            )
            .map_err(|source| map_err("ggml_new_tensor_2d(enc_rows)", source))?;
        graph
            .set_input(enc_rows)
            .map_err(|source| map_err("ggml_set_input(enc_rows)", source))?;

        let stacked = graph
            .reshape_2d(enc_rows, stacked_width, output_frame_count)
            .map_err(|source| map_err("ggml_reshape_2d(adapter_stack)", source))?;

        let mut hidden = graph
            .mul_mat(self.linear1_weight.as_graph_tensor(), stacked)
            .map_err(|source| map_err("ggml_mul_mat(adapter_linear1)", source))?;
        hidden = graph
            .add(hidden, self.linear1_bias.as_graph_tensor())
            .map_err(|source| map_err("ggml_add(adapter_linear1_bias)", source))?;
        hidden = graph
            .relu(hidden)
            .map_err(|source| map_err("ggml_relu(adapter)", source))?;

        let mut output = graph
            .mul_mat(self.linear2_weight.as_graph_tensor(), hidden)
            .map_err(|source| map_err("ggml_mul_mat(adapter_linear2)", source))?;
        output = graph
            .add(output, self.linear2_bias.as_graph_tensor())
            .map_err(|source| map_err("ggml_add(adapter_linear2_bias)", source))?;

        graph
            .set_output(output)
            .map_err(|source| map_err("ggml_set_output(adapter)", source))?;
        graph
            .prepare_outputs_for_upload(&[output])
            .map_err(|source| map_err("ggml_prepare_outputs(adapter)", source))?;
        graph
            .set_f32_slice(
                enc_rows,
                &encoder_rows[..valid_len],
                ADAPTER_ENC_ROWS_TENSOR_NAME,
            )
            .map_err(|source| map_err("ggml_set_f32_slice(enc_rows)", source))?;

        let expected_output_len = output_frame_count
            .checked_mul(llm_dim)
            .ok_or(FireRedLlmAdapterError::ShapeOverflow)?;
        let rows = graph
            .compute_output_f32(output, expected_output_len)
            .map_err(|error| FireRedLlmAdapterError::GraphExecutionFailed {
                reason: error.to_string(),
            })?;
        if rows.iter().any(|value| !value.is_finite()) {
            return Err(FireRedLlmAdapterError::NonFiniteValues);
        }

        Ok((rows, output_frame_count))
    }
}

#[cfg(test)]
mod tests {
    //! Numeric-parity fixture: a tiny toy weight set run through the ggml
    //! graph must reproduce the exact values the old scalar host
    //! implementation was verified against (see this module's git history for
    //! the removed `run_firered_llm_adapter` unit tests this was lifted
    //! from), plus a real-pack byte-level diff against the previous
    //! implementation's output on the actual jfk.wav utterance.
    use super::*;
    use crate::ggml_runtime::GgufTensorDataReader;
    use crate::ggml_runtime::read_gguf_metadata;
    use crate::models::firered_aed::encoder_graph::FireRedEncoderGraphRuntime;
    use crate::models::firered_aed::frontend::{FireRedFbankFrontend, apply_cmvn};
    use crate::models::firered_llm::runtime_contract::{
        parse_firered_llm_adapter_metadata, parse_firered_llm_encoder_metadata,
    };

    fn dev_pack_path() -> Option<std::path::PathBuf> {
        match crate::testing::external_test_fixture_path(
            "OPENASR_FIRERED_LLM_Q4_PACK",
            "FireRed2 LLM q4 .oasr pack",
        ) {
            Ok(path) => Some(path),
            Err(skip) => {
                eprintln!("skipping: {skip}");
                None
            }
        }
    }

    fn jfk_wav_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
    }

    /// Reference scalar implementation (byte-for-byte the removed production
    /// code): dequantizes both weight matrices once via
    /// `host_tensor_f32_copy_dequantized_by_name` and runs the same
    /// stack -> linear1 -> relu -> linear2 math in plain Rust. Kept only as a
    /// test oracle for the numeric-parity check below.
    fn reference_scalar_adapter(
        reader: &GgufTensorDataReader,
        encoder_rows: &[f32],
        frame_count: usize,
        encoder_d_model: usize,
        downsample_rate: usize,
        llm_dim: usize,
    ) -> (Vec<f32>, usize) {
        let stacked_width = encoder_d_model * downsample_rate;
        let linear1_weight = reader
            .host_tensor_f32_copy_dequantized_by_name(
                ADAPTER_LINEAR1_WEIGHT,
                &[stacked_width as u64, llm_dim as u64],
            )
            .expect("linear1 weight");
        let linear1_bias = reader
            .host_tensor_f32_copy_dequantized_by_name(ADAPTER_LINEAR1_BIAS, &[llm_dim as u64])
            .expect("linear1 bias");
        let linear2_weight = reader
            .host_tensor_f32_copy_dequantized_by_name(
                ADAPTER_LINEAR2_WEIGHT,
                &[llm_dim as u64, llm_dim as u64],
            )
            .expect("linear2 weight");
        let linear2_bias = reader
            .host_tensor_f32_copy_dequantized_by_name(ADAPTER_LINEAR2_BIAS, &[llm_dim as u64])
            .expect("linear2 bias");

        let output_frame_count = frame_count / downsample_rate;
        let mut output = Vec::with_capacity(output_frame_count * llm_dim);
        for out_frame in 0..output_frame_count {
            let src_start = out_frame * downsample_rate * encoder_d_model;
            let stacked_row = &encoder_rows[src_start..src_start + stacked_width];
            let mut hidden_row = vec![0.0_f32; llm_dim];
            for (out_idx, out_value) in hidden_row.iter_mut().enumerate() {
                let row = &linear1_weight[out_idx * stacked_width..(out_idx + 1) * stacked_width];
                let acc: f32 = stacked_row.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
                *out_value = (acc + linear1_bias[out_idx]).max(0.0);
            }
            let mut out_row = vec![0.0_f32; llm_dim];
            for (out_idx, out_value) in out_row.iter_mut().enumerate() {
                let row = &linear2_weight[out_idx * llm_dim..(out_idx + 1) * llm_dim];
                let acc: f32 = hidden_row.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
                *out_value = acc + linear2_bias[out_idx];
            }
            output.extend_from_slice(&out_row);
        }
        (output, output_frame_count)
    }

    #[test]
    #[ignore = "requires the private ~4.7GB dev-only firered2-llm-q4_k.oasr pack; numeric parity \
                against the removed scalar host implementation on the real jfk.wav utterance"]
    fn ggml_graph_adapter_matches_reference_scalar_implementation_on_jfk_wav() {
        let Some(pack_path) = dev_pack_path() else {
            return;
        };
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }

        let metadata_view = read_gguf_metadata(&pack_path).expect("read gguf metadata");
        let encoder_metadata =
            parse_firered_llm_encoder_metadata(&metadata_view).expect("parse encoder metadata");
        let adapter_metadata =
            parse_firered_llm_adapter_metadata(&metadata_view).expect("parse adapter metadata");

        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            jfk_wav_path(),
            "adapter parity test",
            "adapter parity test",
        )
        .expect("load jfk.wav");

        let reader = GgufTensorDataReader::from_path(&pack_path).expect("open tensor reader");
        let feature_dim_shape = [encoder_metadata.feature_dim as u64];
        let neg_mean = reader
            .host_tensor_f32_copy_dequantized_by_name("frontend.cmvn.neg_mean", &feature_dim_shape)
            .expect("neg_mean");
        let inv_stddev = reader
            .host_tensor_f32_copy_dequantized_by_name(
                "frontend.cmvn.inv_stddev",
                &feature_dim_shape,
            )
            .expect("inv_stddev");
        let frontend = FireRedFbankFrontend::new();
        let mut features = frontend.compute(&samples).expect("compute fbank");
        apply_cmvn(&mut features.data, features.n_mels, &neg_mean, &inv_stddev)
            .expect("apply cmvn");

        let mut encoder_runtime = FireRedEncoderGraphRuntime::new(&pack_path, encoder_metadata)
            .expect("build encoder runtime");
        let encoder_output = encoder_runtime
            .encode(&features.data, features.n_frames)
            .expect("encode");

        let (reference_rows, reference_frame_count) = reference_scalar_adapter(
            &reader,
            &encoder_output.rows,
            encoder_output.frame_count,
            encoder_metadata.d_model,
            adapter_metadata.downsample_rate,
            adapter_metadata.llm_dim,
        );

        let mut adapter_runtime =
            FireRedLlmAdapterGraphRuntime::new(&pack_path).expect("build adapter runtime");
        let (ggml_rows, ggml_frame_count) = adapter_runtime
            .run(
                &encoder_output.rows,
                encoder_output.frame_count,
                encoder_metadata.d_model,
                adapter_metadata.downsample_rate,
                adapter_metadata.llm_dim,
            )
            .expect("run ggml adapter");

        assert_eq!(ggml_frame_count, reference_frame_count);
        assert_eq!(ggml_rows.len(), reference_rows.len());
        let mut max_abs_diff = 0.0_f32;
        for (a, b) in ggml_rows.iter().zip(reference_rows.iter()) {
            max_abs_diff = max_abs_diff.max((a - b).abs());
        }
        eprintln!(
            "adapter parity: frames={ggml_frame_count} llm_dim={} max_abs_diff={max_abs_diff:.3e}",
            adapter_metadata.llm_dim
        );
        // Both paths run the same math in f32; q4_k mul_mat's on-the-fly
        // dequant kernel and the reference's `host_tensor_f32_copy_dequantized_by_name`
        // use the same block-wise q4_k dequant formula, so this should be
        // near bit-identical modulo summation-order rounding (ggml's SIMD
        // `mul_mat` accumulates in a different order than the scalar
        // straight-line sum above). 1e-2 covers that reordering across a
        // 3584-wide dot product without masking a real regression (a wrong
        // graph -- e.g. a transposed weight or a dropped bias -- would be off
        // by orders of magnitude, not fractions of a percent).
        assert!(
            max_abs_diff < 1.0e-2,
            "adapter ggml graph diverged from the reference scalar implementation: \
             max_abs_diff={max_abs_diff}"
        );
    }
}
