//! Qwen3-ASR inference-side dynamic LoRA (OADP Phase 0).
//!
//! The generic resolver, fail-closed validation, and per-(adapter, base) cache
//! live in [`crate::models::lora_adapter`]; this module keeps only the
//! Qwen-specific pieces: the per-layer slot bundle threaded into the LLM decoder
//! graph (`QwenLayerLoraSlots`), the LoRA-target predicate, and the thin
//! model-named entry points.
//!
//! B is pre-scaled by `alpha/rank` at load time (same as Moonshine) so the
//! in-graph side branch is exactly `y = W@x + B_scaled@(A@x)` with two `mul_mat`
//! + one `add` — no extra scale node.

use std::path::Path;
use std::sync::Arc;

use crate::adapter_pack::is_qwen3_asr_lora_target_tensor_name;
use crate::models::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;
use crate::models::lora_adapter::{
    LoraResolveError, ResolvedLoraAdapter, adapter_cache_fingerprint, resolve_lora_adapter,
};

pub(crate) use crate::models::lora_adapter::{
    LoraSlot as QwenLoraSlot, new_lora_slot_tensors as new_qwen_lora_slot,
};
// Only the LoRA-target type is named outside this module in tests; the inference
// path flows `lora_adapter::LoraTarget` directly via `ResolvedLoraAdapter::target`.
#[cfg(test)]
pub(crate) use crate::models::lora_adapter::LoraTarget as QwenLoraTarget;

/// Qwen-named alias for the shared resolved adapter / error types.
pub(crate) type QwenLoraAdapter = ResolvedLoraAdapter;
pub(crate) type QwenLoraError = LoraResolveError;

const QWEN_LORA_ALLOWED_TARGETS: &str =
    "blk.<n>.{attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down}.weight";

/// Optional LoRA slots for one LLM decoder layer. All `None` = no adapter.
#[derive(Default, Clone, Copy)]
pub(crate) struct QwenLayerLoraSlots {
    pub attn_q: Option<QwenLoraSlot>,
    pub attn_k: Option<QwenLoraSlot>,
    pub attn_v: Option<QwenLoraSlot>,
    pub attn_output: Option<QwenLoraSlot>,
    pub ffn_gate: Option<QwenLoraSlot>,
    pub ffn_up: Option<QwenLoraSlot>,
    pub ffn_down: Option<QwenLoraSlot>,
}

/// Cache-key component: empty string when no adapter is active.
pub(crate) fn qwen_adapter_cache_fingerprint(adapter: Option<&QwenLoraAdapter>) -> String {
    adapter_cache_fingerprint(adapter)
}

/// Resolve the active adapter for a qwen execution. Returns `Ok(None)` when no
/// adapter is configured. Fail-closed on every mismatch class.
pub(crate) fn resolve_qwen_lora_adapter(
    request_adapter_path: Option<&Path>,
    preflight: &GgmlAsrRuntimeSourcePreflight,
) -> Result<Option<Arc<QwenLoraAdapter>>, QwenLoraError> {
    resolve_lora_adapter(
        request_adapter_path,
        preflight,
        is_qwen3_asr_lora_target_tensor_name,
        "qwen3-asr LLM",
        QWEN_LORA_ALLOWED_TARGETS,
    )
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn qwen_lora_adapter_for_test(
    fingerprint: String,
    targets: Vec<(String, QwenLoraTarget)>,
) -> QwenLoraAdapter {
    crate::models::lora_adapter::lora_adapter_for_test(fingerprint, targets)
}

// ── integration tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::adapter_pack::{
        LoraAdapterDtype, LoraAdapterWriteRequest, LoraAdapterWriteTarget, base_pack_model_id,
        file_sha256_hex, qwen3_asr_lora_targetable_tensors, write_lora_adapter_pack,
    };
    use crate::testing::with_forced_cpu_backend_for_test;
    use crate::{
        GgmlAsrBackendPreference, GgmlAsrExecutionError, GgmlAsrExecutionOptions,
        GgmlAsrExecutionRequest, GgmlAsrExecutor, GgmlAsrPreparedAudio,
        qwen3_asr_runtime_descriptor_v1,
    };

    /// LCG for deterministic non-zero floats (no rand crate).
    /// Generates values in (-1, 1) \ {0}.
    fn lcg_f32_sequence(seed: u64, count: usize) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                // LCG parameters from Numerical Recipes.
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // Map to (-1, 1).  Avoid zero by clamping the tiny range.
                let raw = (state as i32) as f32 / i32::MAX as f32;
                if raw.abs() < 1e-6 { 0.1 } else { raw }
            })
            .collect()
    }

    fn locate_model() -> Option<PathBuf> {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp"));
        let candidates = [
            home.join(".openasr/models/qwen3-asr-0.6b/q4_k/qwen3-asr-0.6b-q4_k.oasr"),
            home.join(".openasr/models/qwen3-asr-0.6b/q8_0/qwen3-asr-0.6b-q8_0.oasr"),
            PathBuf::from("tmp/models/qwen3-asr-0.6b-q4_k.oasr"),
        ];
        candidates.into_iter().find(|p| p.exists())
    }

    fn locate_audio() -> Option<PathBuf> {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."));
        let workspace_dir = manifest_dir
            .ancestors()
            .nth(2)
            .map(|p| p.to_path_buf())
            .unwrap_or(manifest_dir);
        let candidates = [
            workspace_dir.join("tmp/audio/librispeech/1089-134691-0014.wav"),
            workspace_dir.join("tmp/audio/generated/frank_read_english_16k.wav"),
        ];
        candidates.into_iter().find(|p| p.exists())
    }

    fn run_transcription(
        runtime_path: &std::path::Path,
        audio_path: &std::path::Path,
        adapter_path: Option<PathBuf>,
    ) -> Result<String, String> {
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(audio_path, "test", "clip")
            .map_err(|e| e.to_string())?;
        let executor = crate::models::qwen::ggml_executor::Qwen3AsrGgmlExecutor::default();
        let request = GgmlAsrExecutionRequest {
            runtime_source_path: runtime_path.to_path_buf(),
            runtime_source_preflight: None,
            selected_family: qwen3_asr_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(samples),
            request_options: GgmlAsrExecutionOptions {
                adapter_path,
                ..Default::default()
            },
            backend_preference: GgmlAsrBackendPreference::CpuOnly,
        };
        match executor.execute(&request) {
            Ok(result) => Ok(result.transcription.text.trim().to_string()),
            Err(GgmlAsrExecutionError::ExecutorFailed { reason, .. })
                if reason.contains("reached max_generated_tokens") =>
            {
                // Treat budget-hit as a valid (partial) result for the test.
                Ok("[max_tokens_hit]".to_string())
            }
            Err(e) => Err(e.to_string()),
        }
    }

    #[test]
    #[ignore = "heavy: real ggml inference with a model pack (~70s); run with `--run-ignored all`"]
    fn qwen_lora_adapter_changes_transcription() {
        let runtime_path = match locate_model() {
            Some(p) => p,
            None => {
                eprintln!(
                    "qwen_lora_adapter_changes_transcription: \
                     no qwen3-asr-0.6b pack found — skipping"
                );
                return;
            }
        };
        let audio_path = match locate_audio() {
            Some(p) => p,
            None => {
                eprintln!(
                    "qwen_lora_adapter_changes_transcription: \
                     no audio clip found under tmp/audio — skipping"
                );
                return;
            }
        };

        with_forced_cpu_backend_for_test(|| {
            // ── baseline (no adapter) ────────────────────────────────────────
            let base_text = run_transcription(&runtime_path, &audio_path, None)
                .expect("baseline transcription must succeed");
            eprintln!(
                "qwen_lora_adapter_changes_transcription: base = {:?}",
                base_text
            );
            assert!(
                !base_text.is_empty() || base_text == "[max_tokens_hit]",
                "base_text must be non-empty or token-budget hit"
            );

            // ── build adapter targeting first blk layer ──────────────────────
            let base_model_id = base_pack_model_id(&runtime_path).expect("model id");
            let base_sha = file_sha256_hex(&runtime_path).expect("sha256");
            // Pick one LoRA-targetable tensor from layer 0.
            let targetable =
                qwen3_asr_lora_targetable_tensors(&runtime_path).expect("targetable tensors");
            // Filter to layer 0 q-projection specifically for a strong, deterministic
            // signal: all remaining targets get zero A/B (only one target active).
            let target_entry = targetable
                .iter()
                .find(|(name, _)| name.contains("blk.0.attn_q.weight"))
                .or_else(|| targetable.first())
                .expect("at least one targetable tensor");
            let (target_name, target_dims) = target_entry;
            let [input_dim, output_dim] = *target_dims;
            let rank: usize = 8;
            // Non-zero LCG values for A; non-zero LCG values for B.
            let a_values = lcg_f32_sequence(0xDEADBEEF, input_dim * rank);
            let b_values = lcg_f32_sequence(0xCAFEBABE, rank * output_dim);

            let temp = tempfile::tempdir().expect("tempdir");
            let adapter_path = temp.path().join("test_lora.oadp");
            write_lora_adapter_pack(&LoraAdapterWriteRequest {
                output_path: adapter_path.clone(),
                id: "test-qwen-lora".to_string(),
                base_model_id: base_model_id.clone(),
                base_pack_sha256: base_sha.clone(),
                rank: rank as u32,
                alpha: rank as u32,
                dtype: LoraAdapterDtype::F32,
                min_openasr_version: "0.1.0".to_string(),
                targets: vec![LoraAdapterWriteTarget {
                    base_tensor: target_name.clone(),
                    input_dim,
                    output_dim,
                    a_values,
                    b_values,
                }],
            })
            .expect("write adapter pack");

            // ── run WITH adapter ─────────────────────────────────────────────
            let adapted_text =
                run_transcription(&runtime_path, &audio_path, Some(adapter_path.clone()))
                    .expect("adapted transcription must succeed");
            eprintln!(
                "qwen_lora_adapter_changes_transcription: adapted = {:?}",
                adapted_text
            );

            // The LoRA side-branch must change the output.
            assert_ne!(
                base_text, adapted_text,
                "LoRA adapter with non-zero A and B must change transcription output; \
                 got base={base_text:?}, adapted={adapted_text:?}"
            );
            assert!(
                !adapted_text.is_empty() || adapted_text == "[max_tokens_hit]",
                "adapted_text must be non-empty or token-budget hit"
            );

            // ── zero-B adapter → must equal base (no-op proof) ──────────────
            let zero_b_path = temp.path().join("zero_b_lora.oadp");
            let a_values_noop = lcg_f32_sequence(0xDEADBEEF, input_dim * rank);
            write_lora_adapter_pack(&LoraAdapterWriteRequest {
                output_path: zero_b_path.clone(),
                id: "test-qwen-lora-zerob".to_string(),
                base_model_id,
                base_pack_sha256: base_sha,
                rank: rank as u32,
                alpha: rank as u32,
                dtype: LoraAdapterDtype::F32,
                min_openasr_version: "0.1.0".to_string(),
                targets: vec![LoraAdapterWriteTarget {
                    base_tensor: target_name.clone(),
                    input_dim,
                    output_dim,
                    a_values: a_values_noop,
                    b_values: vec![0.0; rank * output_dim],
                }],
            })
            .expect("write zero-B adapter pack");

            let zerob_text = run_transcription(&runtime_path, &audio_path, Some(zero_b_path))
                .expect("zero-B transcription must succeed");
            eprintln!(
                "qwen_lora_adapter_changes_transcription: zerob = {:?}",
                zerob_text
            );
            assert_eq!(
                base_text, zerob_text,
                "zero-B LoRA adapter must produce same output as no adapter; \
                 got base={base_text:?}, zerob={zerob_text:?}"
            );
        });
    }

    fn run_transcription_with_backend(
        runtime_path: &std::path::Path,
        audio_path: &std::path::Path,
        adapter_path: Option<PathBuf>,
        backend: GgmlAsrBackendPreference,
    ) -> Result<String, String> {
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(audio_path, "test", "clip")
            .map_err(|e| e.to_string())?;
        let executor = crate::models::qwen::ggml_executor::Qwen3AsrGgmlExecutor::default();
        let request = GgmlAsrExecutionRequest {
            runtime_source_path: runtime_path.to_path_buf(),
            runtime_source_preflight: None,
            selected_family: qwen3_asr_runtime_descriptor_v1(),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(samples),
            request_options: GgmlAsrExecutionOptions {
                adapter_path,
                ..Default::default()
            },
            backend_preference: backend,
        };
        match executor.execute(&request) {
            Ok(result) => Ok(result.transcription.text.trim().to_string()),
            Err(GgmlAsrExecutionError::ExecutorFailed { reason, .. })
                if reason.contains("reached max_generated_tokens") =>
            {
                Ok("[max_tokens_hit]".to_string())
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Validate the qwen LoRA adapter path on the **Metal** (GPU) backend —
    /// production's default — not just the CPU path the other tests force. This
    /// closes a real coverage gap: every other adapter test runs CPU-only.
    /// Skips cleanly when no pack/audio or no GPU backend is available.
    #[test]
    #[ignore = "heavy: real Metal-vs-CPU parity with a model pack + GPU (~70s); run with `--run-ignored all`"]
    fn qwen_lora_metal_adapter_matches_cpu() {
        let runtime_path = match locate_model() {
            Some(p) => p,
            None => {
                eprintln!("qwen_lora_metal_adapter_matches_cpu: no qwen3-asr-0.6b pack — skipping");
                return;
            }
        };
        let audio_path = match locate_audio() {
            Some(p) => p,
            None => {
                eprintln!("qwen_lora_metal_adapter_matches_cpu: no audio — skipping");
                return;
            }
        };

        // Probe Metal first; skip if this host has no GPU-class backend.
        if let Err(e) = run_transcription_with_backend(
            &runtime_path,
            &audio_path,
            None,
            GgmlAsrBackendPreference::Accelerated,
        ) {
            eprintln!("qwen_lora_metal_adapter_matches_cpu: Metal unavailable ({e}) — skipping");
            return;
        }

        // Build a deterministic non-trivial adapter on layer-0 q-proj + a zero-B
        // no-op variant.
        let base_model_id = base_pack_model_id(&runtime_path).expect("model id");
        let base_sha = file_sha256_hex(&runtime_path).expect("sha256");
        let targetable =
            qwen3_asr_lora_targetable_tensors(&runtime_path).expect("targetable tensors");
        let (target_name, dims) = targetable
            .iter()
            .find(|(n, _)| n.contains("blk.0.attn_q.weight"))
            .or_else(|| targetable.first())
            .expect("at least one targetable tensor");
        let [input_dim, output_dim] = *dims;
        let rank = 8usize;
        let temp = tempfile::tempdir().expect("tempdir");
        let write = |path: &std::path::Path, b_values: Vec<f32>, id: &str| {
            write_lora_adapter_pack(&LoraAdapterWriteRequest {
                output_path: path.to_path_buf(),
                id: id.to_string(),
                base_model_id: base_model_id.clone(),
                base_pack_sha256: base_sha.clone(),
                rank: rank as u32,
                alpha: rank as u32,
                dtype: LoraAdapterDtype::F32,
                min_openasr_version: "0.1.0".to_string(),
                targets: vec![LoraAdapterWriteTarget {
                    base_tensor: target_name.clone(),
                    input_dim,
                    output_dim,
                    a_values: lcg_f32_sequence(0xDEAD_BEEF, input_dim * rank),
                    b_values,
                }],
            })
            .expect("write adapter pack");
        };
        let nonzero = temp.path().join("nonzero.oadp");
        let zero_b = temp.path().join("zero_b.oadp");
        write(
            &nonzero,
            lcg_f32_sequence(0xCAFE_BABE, rank * output_dim),
            "metal-nonzero",
        );
        write(&zero_b, vec![0.0; rank * output_dim], "metal-zerob");

        let metal = GgmlAsrBackendPreference::Accelerated;
        let cpu = GgmlAsrBackendPreference::CpuOnly;
        let m_base = run_transcription_with_backend(&runtime_path, &audio_path, None, metal)
            .expect("metal base");
        let m_nonzero = run_transcription_with_backend(
            &runtime_path,
            &audio_path,
            Some(nonzero.clone()),
            metal,
        )
        .expect("metal nonzero");
        let m_zerob =
            run_transcription_with_backend(&runtime_path, &audio_path, Some(zero_b), metal)
                .expect("metal zero-b");
        let c_nonzero =
            run_transcription_with_backend(&runtime_path, &audio_path, Some(nonzero), cpu)
                .expect("cpu nonzero");

        eprintln!("metal base    = {m_base:?}");
        eprintln!("metal nonzero = {m_nonzero:?}");
        eprintln!("metal zero-b  = {m_zerob:?}");
        eprintln!("cpu   nonzero = {c_nonzero:?}");

        // (1) Plumbing correct on Metal: a zero-B adapter is a no-op.
        assert_eq!(
            m_base, m_zerob,
            "Metal zero-B adapter must equal Metal base (additive no-op)"
        );
        // (2) Adapter is actually applied on Metal: non-trivial A/B changes output.
        assert_ne!(
            m_base, m_nonzero,
            "Metal non-zero adapter must change the transcription"
        );
        // (3) Cross-backend parity: CPU and Metal agree on the adapted output.
        assert_eq!(
            c_nonzero, m_nonzero,
            "Metal+adapter must match CPU+adapter (cross-backend parity)"
        );
    }
}
