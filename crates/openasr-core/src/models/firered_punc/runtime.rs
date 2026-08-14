//! FireRedPunc runtime: load an `.oasr` pack and punctuate text.
//!
//! Ties the pack contract, WordPiece tokenizer, and BERT graph together and
//! implements [`crate::punctuation::PunctuationClassifier`], so the punctuation
//! post-processing stage can drive it exactly like the unit-test mock. The
//! graph needs `&mut` for a forward, so it sits behind a `RefCell` to keep the
//! classifier trait's `&self`.

#[cfg(test)]
use crate::ggml_runtime::load_runtime_source_metadata_and_tensor_index_from_source;
use std::cell::RefCell;
#[cfg(test)]
use std::path::Path;

use crate::ggml_runtime::{GgufRuntimeSourcePreflight, build_runtime_tensor_reader_from_preflight};
use crate::punctuation::{
    PunctuationClassifier, PunctuationError, PunctuationRestoreConfig, restore_punctuation,
};

use super::config::{FireRedPuncExecutionMetadata, TOKENIZER_GGML_TOKENS_KEY};
use super::graph::{FireRedPuncGraph, FireRedPuncGraphError, argmax_labels_per_position};
use super::runtime_contract::parse_and_validate_firered_punc_metadata;
use super::tokenizer::{FireRedPuncTokenizer, is_cjk_char};
use super::weights::{FireRedPuncWeights, load_firered_punc_weights};

/// Whether `text` contains at least one Han ideograph (per the tokenizer's
/// BERT `is_cjk_char` definition -- single source of truth, do not fork it).
///
/// This is the per-segment language gate for [`FireRedPuncRuntime::punctuate`]:
/// the checkpoint's label space is five full-width Chinese marks and its
/// training data is Chinese-only, so a segment with no Han ideograph is
/// outside its training domain. Skipping such segments keeps the stage from
/// planting Chinese punctuation into e.g. all-English FireRed output --
/// honest no-op over cross-language mislabeling.
pub(crate) fn segment_qualifies_for_chinese_punctuation(text: &str) -> bool {
    text.chars().any(is_cjk_char)
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum FireRedPuncRuntimeError {
    #[error("firered-punc pack read failed: {0}")]
    Read(String),
    #[error("firered-punc pack metadata invalid: {0}")]
    Metadata(String),
    #[error("firered-punc pack is missing '{0}'")]
    MissingMetadata(&'static str),
    #[error("firered-punc tokenizer build failed: {0}")]
    Tokenizer(String),
    #[error("firered-punc weight load failed: {0}")]
    Weights(String),
    #[error("firered-punc graph build failed: {0}")]
    Graph(#[from] FireRedPuncGraphError),
    #[error("firered-punc system-memory capacity failed: {0}")]
    Capacity(String),
}

pub(crate) struct FireRedPuncRuntime {
    metadata: FireRedPuncExecutionMetadata,
    tokenizer: FireRedPuncTokenizer,
    graph: RefCell<FireRedPuncGraph>,
    construction_requested_peak_bytes: u64,
}

impl FireRedPuncRuntime {
    pub(crate) fn quote_candidate_system_memory(
        preflight: &GgufRuntimeSourcePreflight,
    ) -> Result<
        crate::models::system_memory_owner::SystemMemoryAllocationQuote,
        FireRedPuncRuntimeError,
    > {
        let gguf = preflight.metadata.as_ref();
        let tensor_index = preflight.tensor_index.as_ref();
        let metadata = parse_and_validate_firered_punc_metadata(gguf)
            .map_err(|error| FireRedPuncRuntimeError::Metadata(error.to_string()))?;
        let mut quote = crate::models::prepared_runtime_cache::PreparedRuntimeQuoteBuilder::new::<
            Self,
        >(preflight.runtime_source.content_id());
        let map_capacity = |error: crate::models::system_memory_owner::SystemMemoryOwnerError| {
            FireRedPuncRuntimeError::Capacity(error.to_string())
        };
        quote
            .add_tokenizer_metadata(gguf, false)
            .map_err(map_capacity)?;
        quote
            .add_stable_owned_bytes(
                FireRedPuncGraph::quoted_retained_system_memory_bytes(metadata.layers)
                    .map_err(FireRedPuncRuntimeError::Capacity)?,
                "firered-punc graph handles",
            )
            .map_err(map_capacity)?;
        quote.observe_transient_bytes(
            FireRedPuncWeights::quoted_staging_system_memory_bytes(tensor_index, &metadata)
                .map_err(FireRedPuncRuntimeError::Capacity)?,
            "firered-punc 1-D f32 staging",
        );
        quote.finish().map_err(map_capacity)
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add(
            self.tokenizer.retained_system_memory_bytes()?,
            "firered-punc tokenizer",
        )?;
        bytes.add(
            self.graph.borrow().retained_system_memory_bytes()?,
            "firered-punc graph handles",
        )?;
        Ok(bytes.finish())
    }

    pub(crate) fn try_allocate_inside_parent_candidate(
        quote: crate::models::system_memory_owner::SystemMemoryAllocationQuote,
        preflight: &GgufRuntimeSourcePreflight,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<crate::models::system_memory_owner::SystemMemoryOwner<Self>, FireRedPuncRuntimeError>
    {
        match crate::models::system_memory_owner::SystemMemoryOwner::try_allocate_transaction(
            quote,
            || {
                let runtime = Self::from_preflight(preflight, backend)?;
                let retained = runtime
                    .retained_system_memory_bytes()
                    .map_err(FireRedPuncRuntimeError::Capacity)?;
                let requested_peak = retained
                    .checked_add(runtime.construction_requested_peak_bytes)
                    .ok_or_else(|| {
                        FireRedPuncRuntimeError::Capacity(
                            "firered-punc construction peak overflowed".to_string(),
                        )
                    })?;
                Ok::<_, FireRedPuncRuntimeError>(
                    crate::models::system_memory_owner::SystemMemoryAllocationOutcome::new(
                        runtime,
                        requested_peak,
                        retained,
                    ),
                )
            },
        ) {
            Ok(owner) => Ok(owner),
            Err(crate::models::system_memory_owner::SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
            Err(crate::models::system_memory_owner::SystemMemoryAllocationTransactionError::Capacity(error)) => {
                Err(FireRedPuncRuntimeError::Capacity(error.to_string()))
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn from_pack(
        path: &Path,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, FireRedPuncRuntimeError> {
        let runtime_source = crate::validate_ggml_runtime_source_path(path)
            .map_err(|error| FireRedPuncRuntimeError::Read(error.to_string()))?;
        let preflight = load_runtime_source_metadata_and_tensor_index_from_source(&runtime_source)
            .map_err(|error| FireRedPuncRuntimeError::Read(error.to_string()))?;
        Self::from_preflight(&preflight, backend)
    }

    /// Build from one already-validated immutable mapping. Metadata and
    /// tensor bytes must always share this source; policy-owned callers also
    /// use its content id to prove a delayed build still targets the pack that
    /// was selected during preparation.
    pub(crate) fn from_preflight(
        preflight: &GgufRuntimeSourcePreflight,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, FireRedPuncRuntimeError> {
        let reader = build_runtime_tensor_reader_from_preflight(preflight)
            .map_err(|error| FireRedPuncRuntimeError::Read(error.to_string()))?;
        let gguf = preflight.metadata.as_ref();
        let metadata = parse_and_validate_firered_punc_metadata(gguf)
            .map_err(|error| FireRedPuncRuntimeError::Metadata(error.to_string()))?;
        let tokens = gguf.get_string_array(TOKENIZER_GGML_TOKENS_KEY).ok_or(
            FireRedPuncRuntimeError::MissingMetadata(TOKENIZER_GGML_TOKENS_KEY),
        )?;
        let tokenizer = FireRedPuncTokenizer::new(tokens.to_vec())
            .map_err(|error| FireRedPuncRuntimeError::Tokenizer(error.to_string()))?;
        let weights = load_firered_punc_weights(&reader, &metadata)
            .map_err(|error| FireRedPuncRuntimeError::Weights(error.to_string()))?;
        let construction_requested_peak_bytes = weights
            .retained_system_memory_bytes()
            .map_err(FireRedPuncRuntimeError::Capacity)?;
        let graph = FireRedPuncGraph::new_from_preflight(&weights, metadata, backend, preflight)?;
        Ok(Self {
            metadata,
            tokenizer,
            graph: RefCell::new(graph),
            construction_requested_peak_bytes,
        })
    }

    /// Restore punctuation on `text` (finalize-only, Chinese full-width marks).
    ///
    /// Segments with no Han ideograph are returned verbatim without running
    /// the classifier: see [`segment_qualifies_for_chinese_punctuation`].
    pub(crate) fn punctuate(&self, text: &str) -> Result<String, PunctuationError> {
        if !segment_qualifies_for_chinese_punctuation(text) {
            return Ok(text.to_string());
        }
        restore_punctuation(
            text,
            &self.tokenizer,
            self,
            PunctuationRestoreConfig::default(),
        )
    }
}

impl PunctuationClassifier for FireRedPuncRuntime {
    fn predict_window_labels(
        &self,
        content_token_ids: &[u32],
    ) -> Result<Vec<usize>, PunctuationError> {
        if content_token_ids.is_empty() {
            return Ok(Vec::new());
        }
        // The upstream FireRedPunc `add_cls()` contract prepends one [CLS]
        // token and does not append [SEP]. An extra bidirectional-attention
        // token changes the content logits, even though its own prediction is
        // discarded.
        let ids = model_input_ids(self.tokenizer.cls_id(), content_token_ids);

        let logits = self
            .graph
            .borrow_mut()
            .forward(&ids)
            .map_err(|error| PunctuationError::Classifier(error.to_string()))?;
        let per_position =
            argmax_labels_per_position(&logits, self.metadata.label_count, ids.len());
        // Drop the [CLS] prediction and return labels aligned to content.
        Ok(per_position[1..=content_token_ids.len()].to_vec())
    }
}

fn model_input_ids(cls_id: u32, content_token_ids: &[u32]) -> Vec<u32> {
    let mut ids = Vec::with_capacity(content_token_ids.len() + 1);
    ids.push(cls_id);
    ids.extend_from_slice(content_token_ids);
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_input_matches_upstream_cls_only_contract() {
        assert_eq!(model_input_ids(101, &[7, 8]), vec![101, 7, 8]);
    }

    #[test]
    fn han_gate_rejects_segments_without_han_ideographs() {
        // Regression guard for the FireRed English-transcript bug: FireRed is
        // FixedMultilingual (its `Transcription.language` is always `None`),
        // so an all-English segment reaches the punctuation stage and must be
        // filtered by content, not by language metadata.
        assert!(!segment_qualifies_for_chinese_punctuation(
            "THIS IS A LIBRIVOX RECORDING"
        ));
        assert!(!segment_qualifies_for_chinese_punctuation(""));
        // Half/full-width Latin, digits, and CJK punctuation without any Han
        // ideograph do not qualify either (is_cjk_char is Han-blocks only).
        assert!(!segment_qualifies_for_chinese_punctuation(
            "ｈｅｌｌｏ 123 。"
        ));
    }

    #[test]
    fn han_gate_accepts_chinese_and_mixed_segments() {
        assert!(segment_qualifies_for_chinese_punctuation("你好世界"));
        // One Han ideograph is enough: mixed zh/en segments must still be
        // punctuated (full-width marks throughout is the correct GB/T 15834
        // treatment of Chinese text with embedded Latin).
        assert!(segment_qualifies_for_chinese_punctuation("打开 hello 模式"));
    }

    /// Real-weights parity: only runs when `OPENASR_FIRERED_PUNC_REAL_PACK`
    /// points at a converted FireRedPunc `.oasr` pack. Left env-gated (like the
    /// hymt2 / qwen real-pack tests) so the default suite stays weight-free; the
    /// true upstream parity is exercised at publish time.
    #[test]
    #[ignore = "host-local: needs OPENASR_FIRERED_PUNC_REAL_PACK"]
    fn real_pack_punctuates_readme_example() {
        let pack = crate::testing::external_test_fixture_path(
            "OPENASR_FIRERED_PUNC_REAL_PACK",
            "FireRedPunc runtime pack",
        )
        .expect("OPENASR_FIRERED_PUNC_REAL_PACK");
        let runtime = FireRedPuncRuntime::from_pack(
            &pack,
            crate::ggml_runtime::GgmlCpuGraphConfig::runtime_default().backend,
        )
        .expect("load real pack");
        let out = runtime.punctuate("你好世界").expect("punctuate");
        assert_eq!(out, "你好世界。", "upstream README golden");

        // The Han gate short-circuits before the classifier: an all-English
        // segment must come back byte-for-byte unchanged even with real
        // weights loaded.
        let english = "THIS IS A LIBRIVOX RECORDING";
        let out = runtime.punctuate(english).expect("punctuate english");
        assert_eq!(out, english, "no-Han segment passes through verbatim");
    }

    #[test]
    #[ignore = "host-local: needs OPENASR_FIRERED_PUNC_REAL_PACK and OPENASR_AUX_BENCH_TEXT"]
    fn firered_punc_aux_text_benchmark() {
        let pack = crate::testing::external_test_fixture_path(
            "OPENASR_FIRERED_PUNC_REAL_PACK",
            "FireRedPunc runtime pack",
        )
        .expect("OPENASR_FIRERED_PUNC_REAL_PACK");
        let text_path = crate::testing::external_test_fixture_path(
            "OPENASR_AUX_BENCH_TEXT",
            "private auxiliary-model benchmark transcript",
        )
        .expect("OPENASR_AUX_BENCH_TEXT");
        let text = std::fs::read_to_string(text_path).expect("read benchmark transcript");
        let text = text.trim();
        assert!(!text.is_empty(), "benchmark transcript must not be empty");
        let backend = match std::env::var("OPENASR_AUX_BENCH_BACKEND")
            .unwrap_or_else(|_| "cpu".to_string())
            .as_str()
        {
            "cpu" => crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
            "metal" => crate::ggml_runtime::GgmlCpuGraphBackend::Metal,
            "gpu" => crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            value => panic!("unsupported OPENASR_AUX_BENCH_BACKEND '{value}'"),
        };
        let execution_placement = crate::GgmlExecutionTelemetryCollector::new();
        let _execution_placement_guard = execution_placement.install();
        let runtime = FireRedPuncRuntime::from_pack(&pack, backend).expect("load punc runtime");
        let run = || runtime.punctuate(text).expect("punctuate benchmark text");

        let mut output = run();
        let seconds = (0..5)
            .map(|_| {
                let started = std::time::Instant::now();
                output = run();
                started.elapsed().as_secs_f64()
            })
            .collect::<Vec<_>>();
        let output_sha256 = crate::testing::benchmark_sha256_bytes([output.as_bytes()]);
        let (median_seconds, seconds) = crate::testing::benchmark_median_seconds(seconds);
        let observed = execution_placement.snapshot();
        let memory = crate::metrics::process_memory_snapshot();
        eprintln!(
            "AUX_MODEL_BENCH model=fireredpunc backend={backend:?} chars={} median_seconds={median_seconds:.6} current_rss_bytes={:?} peak_rss_bytes={:?} current_phys_footprint_bytes={:?} peak_phys_footprint_bytes={:?} observed_compute_nodes={:?} output_sha256={output_sha256} runs={seconds:?}",
            text.chars().count(),
            memory.current_rss_bytes,
            memory.peak_rss_bytes,
            memory.current_phys_footprint_bytes,
            memory.peak_phys_footprint_bytes,
            observed.observed_compute_nodes_by_backend,
        );
        if backend == crate::ggml_runtime::GgmlCpuGraphBackend::Metal {
            assert!(
                !observed.observed_compute_nodes_by_backend.is_empty()
                    && observed
                        .observed_compute_nodes_by_backend
                        .keys()
                        .all(|backend| {
                            let backend = backend.to_ascii_lowercase();
                            backend.starts_with("mtl") || backend.contains("metal")
                        }),
                "explicit Metal FireRedPunc benchmark observed non-Metal compute: {:?}",
                observed.observed_compute_nodes_by_backend
            );
        }
    }

    /// Converter golden gate: the engine's per-token argmax labels for the
    /// converted `.oasr` pack must exactly match the upstream PyTorch forward.
    /// Both env vars are dev-only (the pack is uncommitted; the JSON is emitted
    /// by `tmp/firered-punc-src/reference_forward.py`), so the default suite
    /// skips this -- it is the publish-time parity proof, mirroring the qwen
    /// forced-aligner reference convention. The JSON is a list of
    /// `{content_ids: [u32], ref_labels: [usize]}` entries; the same content
    /// ids are fed to both sides so this isolates the numeric forward from
    /// tokenization.
    #[test]
    #[ignore = "host-local: needs OPENASR_FIRERED_PUNC_REAL_PACK and OPENASR_FIRERED_PUNC_GOLDEN_JSON"]
    fn real_pack_labels_match_pytorch_reference_golden() {
        let pack = crate::testing::external_test_fixture_path(
            "OPENASR_FIRERED_PUNC_REAL_PACK",
            "FireRedPunc runtime pack",
        )
        .expect("OPENASR_FIRERED_PUNC_REAL_PACK");
        let json = crate::testing::external_test_fixture_path(
            "OPENASR_FIRERED_PUNC_GOLDEN_JSON",
            "FireRedPunc PyTorch reference labels",
        )
        .expect("OPENASR_FIRERED_PUNC_GOLDEN_JSON");
        let backend = crate::ggml_runtime::GgmlCpuGraphConfig::runtime_default().backend;
        let execution_placement = crate::GgmlExecutionTelemetryCollector::new();
        let _execution_placement_guard = execution_placement.install();
        let runtime = FireRedPuncRuntime::from_pack(&pack, backend).expect("load real pack");
        let text = std::fs::read_to_string(&json).expect("read golden json");
        let entries: serde_json::Value = serde_json::from_str(&text).expect("parse golden json");
        let entries = entries.as_array().expect("golden json is a list");
        let mut checked = 0usize;
        for entry in entries {
            let content_ids: Vec<u32> = entry["content_ids"]
                .as_array()
                .expect("content_ids array")
                .iter()
                .map(|value| value.as_u64().expect("id is u64") as u32)
                .collect();
            let ref_labels: Vec<usize> = entry["ref_labels"]
                .as_array()
                .expect("ref_labels array")
                .iter()
                .map(|value| value.as_u64().expect("label is u64") as usize)
                .collect();
            let budget = runtime.metadata.max_positions.saturating_sub(1).max(1);
            let engine_labels = content_ids
                .chunks(budget)
                .flat_map(|window| {
                    runtime
                        .predict_window_labels(window)
                        .expect("engine predict")
                })
                .collect::<Vec<_>>();
            assert_eq!(
                engine_labels,
                ref_labels,
                "label mismatch for sentence {:?}",
                entry.get("sentence")
            );
            checked += 1;
        }
        assert!(checked > 0, "golden json had no entries");
        let observed = execution_placement.snapshot();
        let memory = crate::metrics::process_memory_snapshot();
        eprintln!(
            "FIRERED_PUNC_OFFICIAL_PARITY backend={backend:?} sentences={checked} observed_compute_nodes={:?} current_rss_bytes={:?} peak_rss_bytes={:?} current_phys_footprint_bytes={:?} peak_phys_footprint_bytes={:?}",
            observed.observed_compute_nodes_by_backend,
            memory.current_rss_bytes,
            memory.peak_rss_bytes,
            memory.current_phys_footprint_bytes,
            memory.peak_phys_footprint_bytes,
        );
        if backend == crate::ggml_runtime::GgmlCpuGraphBackend::Metal {
            assert!(
                !observed.observed_compute_nodes_by_backend.is_empty()
                    && observed
                        .observed_compute_nodes_by_backend
                        .keys()
                        .all(|backend| {
                            let backend = backend.to_ascii_lowercase();
                            backend.starts_with("mtl") || backend.contains("metal")
                        }),
                "explicit Metal FireRedPunc route observed non-Metal compute: {:?}",
                observed.observed_compute_nodes_by_backend
            );
        }
    }
}
