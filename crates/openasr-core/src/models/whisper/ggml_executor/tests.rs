use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use super::*;
use crate::models::whisper::ggml_tensor_binding::WhisperGgufTensorSlot;
use crate::models::whisper::tokenizer::{
    TOKENIZER_GGML_EOT_TOKEN_ID_KEY, TOKENIZER_GGML_MERGES_KEY, TOKENIZER_GGML_MODEL_KEY,
    TOKENIZER_GGML_MODEL_VALUE_GPT2, TOKENIZER_GGML_NO_TIMESTAMPS_TOKEN_ID_KEY,
    TOKENIZER_GGML_SOT_TOKEN_ID_KEY, TOKENIZER_GGML_SPECIAL_TOKEN_IDS_KEY,
    TOKENIZER_GGML_TOKENS_KEY, TOKENIZER_GGML_TRANSCRIBE_TOKEN_ID_KEY,
};
use crate::testing::{
    TinyGgufFixtureSpec, WhisperExecutionFailureStage,
    assert_tiny_whisper_encoder_output_shape_and_finite, classify_whisper_execution_failure_stage,
    tiny_whisper_encoder_smoke_prepared_audio, tiny_whisper_encoder_smoke_real_mel_input,
    tiny_whisper_encoder_smoke_shape_for_default_fixture, write_tiny_gguf_runtime_source,
};
use crate::{
    GgufMetadata, GgufMetadataValue, read_gguf_metadata_from_runtime_source,
    validate_ggml_runtime_source_path, whisper_runtime_descriptor_v1,
};
use sha2::{Digest, Sha256};

const GOLDEN_DIFF_TINY_WHISPER_ENCODER_PRELUDE_SHA256: &str =
    "1249053500204b8b7e4b08b54e42a7d557e3a1d507a51ff709953ef003c2e826";
// Reference-platform (macOS aarch64) bit-identity golden. ggml CPU float
// compute is not bit-identical across architectures (libm / SIMD / reduction
// order differ), so the decoder-step logit hash is pinned to the capture
// platform; the cross-platform encoder-prelude golden above still gates every
// target (incl. the Linux CI runner).
#[cfg(target_os = "macos")]
const GOLDEN_DIFF_TINY_WHISPER_DECODER_STEP_LOGITS_SHA256: &str =
    "562f79a316fb538274ec4eb03a8ee28407dd5a8f258029d44fe6c1fba5882561";

fn sha256_f32_le(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone)]
enum TestPreludeRunnerOutcome {
    Success,
}

struct TestPreludeRunner {
    called: Arc<AtomicBool>,
    outcome: TestPreludeRunnerOutcome,
}

impl WhisperEncoderPreludeRunner for TestPreludeRunner {
    fn runner_id(&self) -> &'static str {
        "test-whisper-encoder-prelude-runner-v0"
    }

    fn run_encoder_prelude(
        &self,
        _runtime_source: &GgmlRuntimeSource,
        _encoder_weights: &WhisperEncoderWeightBundle,
        plan: &WhisperEncoderPreludePlan,
        mel_input: &WhisperMelFeatureInput,
    ) -> Result<WhisperEncoderPreludeSeamResult, WhisperGgmlExecutorError> {
        self.called.store(true, Ordering::SeqCst);
        match &self.outcome {
            TestPreludeRunnerOutcome::Success => {
                let output_hidden_f32 = vec![0.0; plan.output_frames * plan.output_hidden_size];
                assert_eq!(
                    mel_input.values_f32.len(),
                    plan.input_shape.mel_frames * plan.input_shape.mel_bins
                );
                Ok(WhisperEncoderPreludeSeamResult::GraphExecuted {
                    runner_id: self.runner_id(),
                    output_frames: plan.output_frames,
                    output_hidden_size: plan.output_hidden_size,
                    output_hidden_f32,
                })
            }
        }
    }
}

#[derive(Debug, Clone)]
enum TestEncoderGraphRunnerOutcome {
    Success,
}

struct TestEncoderGraphRunner {
    called: Arc<AtomicBool>,
    outcome: TestEncoderGraphRunnerOutcome,
}

impl WhisperEncoderGraphRunner for TestEncoderGraphRunner {
    fn runner_id(&self) -> &'static str {
        "test-whisper-encoder-graph-runner-v0"
    }

    fn run_encoder_graph(
        &self,
        _runtime_source: &GgmlRuntimeSource,
        _execution: &WhisperGgmlExecutionMetadata,
        _encoder_weights: &WhisperEncoderWeightBundle,
        plan: &WhisperEncoderGraphPlan,
        encoder_hidden_input_f32: &[f32],
    ) -> Result<WhisperEncoderGraphSeamResult, WhisperGgmlExecutorError> {
        self.called.store(true, Ordering::SeqCst);
        assert_eq!(
            encoder_hidden_input_f32.len(),
            plan.output_frames * plan.output_hidden_size
        );
        assert!(
            encoder_hidden_input_f32
                .iter()
                .all(|value| value.is_finite()),
            "encoder graph seam input must stay finite"
        );
        match &self.outcome {
            TestEncoderGraphRunnerOutcome::Success => {
                Ok(WhisperEncoderGraphSeamResult::GraphExecuted {
                    runner_id: self.runner_id(),
                    layer_count: plan.layers.len(),
                    output_frames: plan.output_frames,
                    output_hidden_size: plan.output_hidden_size,
                    output_hidden_f32: vec![0.0; plan.output_frames * plan.output_hidden_size],
                })
            }
        }
    }
}

#[derive(Debug, Clone)]
enum TestMelFeatureInputProviderOutcome {
    RealFrontend,
    ExtractionFailed { reason: String },
}

struct TestMelFeatureInputProvider {
    called: Arc<AtomicBool>,
    outcome: TestMelFeatureInputProviderOutcome,
}

impl WhisperMelFeatureInputProvider for TestMelFeatureInputProvider {
    fn provider_id(&self) -> &'static str {
        "test-whisper-mel-provider-v0"
    }

    fn prepare_mel_feature_input(
        &self,
        execution: &WhisperGgmlExecutionMetadata,
        prepared_audio: &GgmlAsrPreparedAudio,
    ) -> Result<WhisperMelFeatureInput, WhisperGgmlExecutorError> {
        self.called.store(true, Ordering::SeqCst);
        match &self.outcome {
            TestMelFeatureInputProviderOutcome::RealFrontend => {
                let mel_input = tiny_whisper_encoder_smoke_real_mel_input(
                    prepared_audio,
                    execution.encoder_mels_count,
                )
                .map_err(|reason| {
                    WhisperGgmlExecutorError::MelFeatureInputPreparationFailed {
                        reason: format!("provider='{}' {reason}", self.provider_id()),
                    }
                })?;
                Ok(WhisperMelFeatureInput {
                    source_label: mel_input.source_label,
                    shape: WhisperMelFeatureInputShape {
                        mel_bins: mel_input.mel_bins,
                        mel_frames: mel_input.mel_frames,
                    },
                    values_f32: mel_input.values_f32,
                })
            }
            TestMelFeatureInputProviderOutcome::ExtractionFailed { reason } => {
                Err(WhisperGgmlExecutorError::MelFeatureExtractionFailed {
                    reason: reason.clone(),
                })
            }
        }
    }
}

fn default_prepared_audio() -> GgmlAsrPreparedAudio {
    tiny_whisper_encoder_smoke_prepared_audio()
}

fn whisper_execution_and_tokenizer_fixture() -> (WhisperGgmlExecutionMetadata, WhisperTokenizer) {
    let mut values = std::collections::BTreeMap::new();
    values.insert(
        "general.architecture".to_string(),
        GgufMetadataValue::String("whisper".to_string()),
    );
    values.insert(
        "whisper.encoder.block_count".to_string(),
        GgufMetadataValue::U32(1),
    );
    values.insert(
        "whisper.encoder.embedding_length".to_string(),
        GgufMetadataValue::U32(4),
    );
    values.insert(
        "whisper.encoder.attention.head_count".to_string(),
        GgufMetadataValue::U32(2),
    );
    values.insert(
        "whisper.encoder.context_length".to_string(),
        GgufMetadataValue::U32(1500),
    );
    values.insert(
        "whisper.encoder.mels_count".to_string(),
        GgufMetadataValue::U32(80),
    );
    values.insert(
        "whisper.decoder.block_count".to_string(),
        GgufMetadataValue::U32(1),
    );
    values.insert(
        "whisper.decoder.embedding_length".to_string(),
        GgufMetadataValue::U32(4),
    );
    values.insert(
        "whisper.decoder.attention.head_count".to_string(),
        GgufMetadataValue::U32(2),
    );
    values.insert(
        "whisper.decoder.context_length".to_string(),
        GgufMetadataValue::U32(32),
    );
    values.insert("whisper.vocab_size".to_string(), GgufMetadataValue::U32(14));
    values.insert(
        TOKENIZER_GGML_MODEL_KEY.to_string(),
        GgufMetadataValue::String(TOKENIZER_GGML_MODEL_VALUE_GPT2.to_string()),
    );
    values.insert(
        TOKENIZER_GGML_TOKENS_KEY.to_string(),
        GgufMetadataValue::StringArray(vec![
            "\u{0120}".to_string(),
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "w".to_string(),
            "r".to_string(),
            "d".to_string(),
            "<|endoftext|>".to_string(),
            "<|startoftranscript|>".to_string(),
            "<|transcribe|>".to_string(),
            "<|notimestamps|>".to_string(),
            "<|startofprev|>".to_string(),
            "\u{010A}".to_string(),
        ]),
    );
    values.insert(
        TOKENIZER_GGML_MERGES_KEY.to_string(),
        GgufMetadataValue::StringArray(vec!["x y".to_string()]),
    );
    values.insert(
        TOKENIZER_GGML_SPECIAL_TOKEN_IDS_KEY.to_string(),
        GgufMetadataValue::U32Array(vec![8, 9, 10, 11, 12]),
    );
    values.insert(
        TOKENIZER_GGML_SOT_TOKEN_ID_KEY.to_string(),
        GgufMetadataValue::U32(9),
    );
    values.insert(
        TOKENIZER_GGML_EOT_TOKEN_ID_KEY.to_string(),
        GgufMetadataValue::U32(8),
    );
    values.insert(
        TOKENIZER_GGML_TRANSCRIBE_TOKEN_ID_KEY.to_string(),
        GgufMetadataValue::U32(10),
    );
    values.insert(
        TOKENIZER_GGML_NO_TIMESTAMPS_TOKEN_ID_KEY.to_string(),
        GgufMetadataValue::U32(11),
    );
    let metadata = GgufMetadata::from_values_for_test(values);
    let execution =
        validate_whisper_execution_metadata(&metadata).expect("validate whisper metadata");
    let tokenizer = WhisperTokenizer::from_gguf_metadata(&metadata).expect("load tokenizer");
    (execution, tokenizer)
}

// Pinned to the reference platform — see
// GOLDEN_DIFF_TINY_WHISPER_DECODER_STEP_LOGITS_SHA256.
#[cfg(target_os = "macos")]
#[test]
fn golden_diff_tiny_imported_decoder_graph_executes_one_step() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-decoder-step.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture");
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
    let execution =
        validate_whisper_execution_metadata(&metadata).expect("validate whisper metadata");
    let tensor_index = load_whisper_tensor_index(&runtime_source).expect("load tensor index");
    let tensor_binding =
        bind_whisper_required_tensors(&tensor_index, &execution).expect("bind tensors");
    let tensor_reader = GgufTensorDataReader::from_tensor_index_shared(Arc::clone(
        &tensor_binding.weights.tensor_index,
    ))
    .expect("create tensor reader");
    let decoder_weights =
        build_decoder_weight_seam(&tensor_reader, &tensor_binding.weights.bindings)
            .expect("materialize decoder weights");

    let encoder_frames = 2usize;
    let encoder_hidden = execution.decoder_hidden_size;
    let encoder_hidden_f32 = (0..encoder_frames * encoder_hidden)
        .map(|idx| (idx as f32) * 0.001)
        .collect::<Vec<_>>();
    let token_count = 1usize;
    let plan = build_whisper_decoder_graph_plan(
        WhisperDecoderGraphMetadata {
            decoder_layers: execution.decoder_layers,
            decoder_hidden_size: execution.decoder_hidden_size,
            decoder_attention_heads: execution.decoder_attention_heads,
            vocab_size: execution.vocab_size,
            max_target_positions: execution.max_target_positions,
        },
        &decoder_weights.graph_binding,
        &decoder_weights.graph_materialization,
        WhisperDecoderGraphInputShape {
            token_count,
            encoder_frames,
            hidden_size: encoder_hidden,
        },
    )
    .expect("build decoder plan");
    let mut decoder_tensor_cache = WhisperDecoderExecutionTensorCache::default();
    let mut decoder_graph_runner =
        GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default()).expect("decoder graph runner");
    let step_logits = WhisperDecoderGraphRunnerGgmlV0
        .step_logits(
            &runtime_source,
            &execution,
            &decoder_weights,
            &plan,
            &WhisperDecoderGraphExecutionInput {
                decoder_prefix_tokens: vec![0_u32],
                encoder_hidden_state: encoder_hidden_f32,
                encoder_layout: WhisperDecoderHiddenStateLayout::SequenceHidden,
            },
            WhisperDecoderGraphExecutionConfig {
                attention_heads: execution.decoder_attention_heads,
                use_self_flash_attention: false,
                use_cross_flash_attention: false,
                collect_cross_attention: false,
                layer_norm_epsilon: 1.0e-5_f32,
            },
            &mut decoder_graph_runner,
            None,
            None,
            &mut decoder_tensor_cache,
            &WhisperDecoderStepSeamInput {
                encoder_frames,
                encoder_hidden_size: encoder_hidden,
                step_index: 0,
                position_offset: 0,
            },
        )
        .expect("decoder runner should execute one tiny step");

    assert_eq!(step_logits.logits.len(), execution.vocab_size);
    assert!(
        step_logits.logits.iter().all(|value| value.is_finite()),
        "decoder step logits must remain finite: {:?}",
        step_logits.logits
    );
    assert_eq!(
        sha256_f32_le(&step_logits.logits),
        GOLDEN_DIFF_TINY_WHISPER_DECODER_STEP_LOGITS_SHA256
    );
}

#[test]
fn whisper_preflight_fails_on_missing_metadata_before_encoder_prelude() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-metadata-missing.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_non_streaming_cpu("whisper-fixture");
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
    let called = Arc::new(AtomicBool::new(false));
    let runner = Arc::new(TestPreludeRunner {
        called: Arc::clone(&called),
        outcome: TestPreludeRunnerOutcome::Success,
    });
    let graph_runner = WhisperCpuEncoderGraphComputeRunnerV0;
    let mel_called = Arc::new(AtomicBool::new(false));
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::clone(&mel_called),
        outcome: TestMelFeatureInputProviderOutcome::RealFrontend,
    };
    let adapter = whisper_runtime_descriptor_v1();

    let error = execute_whisper_ggml_non_streaming_cpu(
        &adapter,
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &default_prepared_audio(),
        &mel_provider,
        runner.as_ref(),
        &graph_runner,
        &WhisperDecoderGraphRunnerGgmlV0,
        &WhisperTokenizerProviderGgufV0,
    )
    .expect_err("missing whisper metadata must fail preflight");

    match error {
        WhisperGgmlExecutorError::MissingRequiredMetadata { key } => {
            assert_eq!(key, "general.architecture");
        }
        other => panic!("unexpected error: {other}"),
    }
    assert_eq!(
        classify_whisper_execution_failure_stage(&error.to_string()),
        WhisperExecutionFailureStage::MetadataPreflight
    );
    assert!(
        !called.load(Ordering::SeqCst),
        "encoder prelude seam must not run when metadata preflight fails"
    );
    assert!(
        !mel_called.load(Ordering::SeqCst),
        "mel/input seam must not run when metadata preflight fails"
    );
}

#[test]
fn whisper_tensor_shape_mismatch_fails_before_encoder_prelude() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-shape-mismatch.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_shape_mismatch(
        "whisper-fixture",
        "model.encoder.conv1.weight",
        [3_u64, 3, 8],
    );
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
    let called = Arc::new(AtomicBool::new(false));
    let runner = Arc::new(TestPreludeRunner {
        called: Arc::clone(&called),
        outcome: TestPreludeRunnerOutcome::Success,
    });
    let graph_runner = WhisperCpuEncoderGraphComputeRunnerV0;
    let mel_called = Arc::new(AtomicBool::new(false));
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::clone(&mel_called),
        outcome: TestMelFeatureInputProviderOutcome::RealFrontend,
    };
    let adapter = whisper_runtime_descriptor_v1();

    let error = execute_whisper_ggml_non_streaming_cpu(
        &adapter,
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &default_prepared_audio(),
        &mel_provider,
        runner.as_ref(),
        &graph_runner,
        &WhisperDecoderGraphRunnerGgmlV0,
        &WhisperTokenizerProviderGgufV0,
    )
    .expect_err("tensor shape mismatch must fail before prelude seam");

    assert!(matches!(
        error,
        WhisperGgmlExecutorError::InvalidRequiredTensor { .. }
    ));
    assert_eq!(
        classify_whisper_execution_failure_stage(&error.to_string()),
        WhisperExecutionFailureStage::TensorBindingPreflight
    );
    assert!(
        !called.load(Ordering::SeqCst),
        "encoder prelude seam must not run when tensor shape preflight fails"
    );
    assert!(
        !mel_called.load(Ordering::SeqCst),
        "mel/input seam must not run when tensor preflight fails"
    );
}

#[test]
fn whisper_tensor_type_mismatch_fails_before_encoder_prelude() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-type-mismatch.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_type_mismatch(
        "whisper-fixture",
        "model.encoder.conv1.bias",
    );
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
    let called = Arc::new(AtomicBool::new(false));
    let runner = Arc::new(TestPreludeRunner {
        called: Arc::clone(&called),
        outcome: TestPreludeRunnerOutcome::Success,
    });
    let graph_runner = WhisperCpuEncoderGraphComputeRunnerV0;
    let mel_called = Arc::new(AtomicBool::new(false));
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::clone(&mel_called),
        outcome: TestMelFeatureInputProviderOutcome::RealFrontend,
    };
    let adapter = whisper_runtime_descriptor_v1();

    let error = execute_whisper_ggml_non_streaming_cpu(
        &adapter,
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &default_prepared_audio(),
        &mel_provider,
        runner.as_ref(),
        &graph_runner,
        &WhisperDecoderGraphRunnerGgmlV0,
        &WhisperTokenizerProviderGgufV0,
    )
    .expect_err("tensor type mismatch must fail before prelude seam");

    assert!(matches!(
        error,
        WhisperGgmlExecutorError::InvalidRequiredTensor { .. }
    ));
    let message = error.to_string();
    assert_eq!(
        classify_whisper_execution_failure_stage(&message),
        WhisperExecutionFailureStage::TensorBindingPreflight
    );
    assert!(
        message.contains("does not satisfy expected f32/f16/bf16"),
        "unexpected mismatch reason: {message}"
    );
    assert!(
        !called.load(Ordering::SeqCst),
        "encoder prelude seam must not run when tensor type preflight fails"
    );
    assert!(
        !mel_called.load(Ordering::SeqCst),
        "mel/input seam must not run when tensor preflight fails"
    );
}

#[test]
fn mel_feature_extraction_failure_fails_before_encoder_execution() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-mel-seam.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture");
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
    let prelude_called = Arc::new(AtomicBool::new(false));
    let prelude_runner = TestPreludeRunner {
        called: Arc::clone(&prelude_called),
        outcome: TestPreludeRunnerOutcome::Success,
    };
    let graph_called = Arc::new(AtomicBool::new(false));
    let graph_runner = TestEncoderGraphRunner {
        called: Arc::clone(&graph_called),
        outcome: TestEncoderGraphRunnerOutcome::Success,
    };
    let mel_called = Arc::new(AtomicBool::new(false));
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::clone(&mel_called),
        outcome: TestMelFeatureInputProviderOutcome::ExtractionFailed {
            reason: "frontend fft failed".to_string(),
        },
    };
    let adapter = whisper_runtime_descriptor_v1();

    let error = execute_whisper_ggml_non_streaming_cpu(
        &adapter,
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &default_prepared_audio(),
        &mel_provider,
        &prelude_runner,
        &graph_runner,
        &WhisperDecoderGraphRunnerGgmlV0,
        &WhisperTokenizerProviderGgufV0,
    )
    .expect_err("mel seam should fail closed");

    assert!(
        matches!(
            error,
            WhisperGgmlExecutorError::MelFeatureExtractionFailed { .. }
                | WhisperGgmlExecutorError::TokenizerMissing { .. }
                | WhisperGgmlExecutorError::DecoderWeightsMissing { .. }
                | WhisperGgmlExecutorError::DecoderGraphExecutionFailed { .. }
        ),
        "unexpected fail-closed boundary error: {error}"
    );
    if matches!(
        error,
        WhisperGgmlExecutorError::MelFeatureExtractionFailed { .. }
    ) {
        assert_eq!(
            classify_whisper_execution_failure_stage(&error.to_string()),
            WhisperExecutionFailureStage::MelFeature
        );
        assert!(mel_called.load(Ordering::SeqCst), "mel/input seam must run");
        assert!(
            !prelude_called.load(Ordering::SeqCst),
            "encoder prelude must not run when mel seam fails"
        );
        assert!(
            !graph_called.load(Ordering::SeqCst),
            "encoder graph must not run when mel seam fails"
        );
    }
}

#[test]
fn golden_diff_prepared_audio_real_mel_and_real_encoder_compute_reach_decoder_fail_closed() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-real-mel.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture");
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");

    let prepared_audio = default_prepared_audio();
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::new(AtomicBool::new(false)),
        outcome: TestMelFeatureInputProviderOutcome::RealFrontend,
    };
    let execution =
        validate_whisper_execution_metadata(&metadata).expect("metadata should be valid");
    let tensor_index = load_whisper_tensor_index(&runtime_source).expect("load tensor index");
    let tensor_binding =
        bind_whisper_required_tensors(&tensor_index, &execution).expect("bind tensors");
    let encoder_weights =
        materialize_whisper_encoder_weights(&tensor_binding).expect("materialize encoder");
    let mel_input = prepare_mel_feature_input_seam(&mel_provider, &execution, &prepared_audio)
        .expect("real frontend mel preparation");
    assert!(
        mel_input.values_f32.iter().all(|value| value.is_finite()),
        "mel values must stay finite"
    );
    let prelude_plan = build_whisper_encoder_prelude_plan(
        &tensor_binding.weights.bindings,
        infer_encoder_prelude_input_shape_from_mel_input(&mel_input)
            .expect("infer prelude input shape"),
        execution.encoder_hidden_size,
        execution.encoder_mels_count,
    )
    .expect("build prelude plan");
    let prelude_result = run_encoder_prelude_seam(
        &runtime_source,
        &encoder_weights,
        &prelude_plan,
        &mel_input,
        &WhisperCpuEncoderPreludeComputeRunnerV0,
    )
    .expect("run prelude seam");
    let smoke_shape = tiny_whisper_encoder_smoke_shape_for_default_fixture();
    match prelude_result {
        WhisperEncoderPreludeSeamResult::GraphExecuted {
            output_frames,
            output_hidden_size,
            output_hidden_f32,
            ..
        } => {
            assert_eq!(output_frames, smoke_shape.output_frames);
            assert_eq!(output_hidden_size, smoke_shape.hidden_size);
            assert_tiny_whisper_encoder_output_shape_and_finite(&output_hidden_f32, smoke_shape);
            assert_eq!(
                sha256_f32_le(&output_hidden_f32),
                GOLDEN_DIFF_TINY_WHISPER_ENCODER_PRELUDE_SHA256
            );
        }
    }

    let output = execute_whisper_ggml_non_streaming_cpu(
        &whisper_runtime_descriptor_v1(),
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &prepared_audio,
        &mel_provider,
        &WhisperCpuEncoderPreludeComputeRunnerV0,
        &WhisperCpuEncoderGraphComputeRunnerV0,
        &WhisperDecoderGraphRunnerGgmlV0,
        &WhisperTokenizerProviderGgufV0,
    );
    match output {
        Ok(text) => {
            assert!(
                !text.trim().is_empty(),
                "decoder graph + tokenizer path should not emit empty text"
            );
        }
        Err(error) => {
            assert!(
                matches!(
                    error,
                    WhisperGgmlExecutorError::DecoderNoEotBeforeMaxTokens { .. }
                        | WhisperGgmlExecutorError::DecoderInvalidTokenDecode { .. }
                        | WhisperGgmlExecutorError::DecoderGraphExecutionFailed { .. }
                        | WhisperGgmlExecutorError::DecoderGraphUnsupported { .. }
                        | WhisperGgmlExecutorError::DecoderWeightsMissing { .. }
                        | WhisperGgmlExecutorError::TokenizerMissing { .. }
                ),
                "unexpected decoder-stage fail-closed error: {error}"
            );
            assert!(
                matches!(
                    classify_whisper_execution_failure_stage(&error.to_string()),
                    WhisperExecutionFailureStage::MetadataPreflight
                        | WhisperExecutionFailureStage::EncoderExecuted
                        | WhisperExecutionFailureStage::Unknown
                ),
                "unexpected failure stage: {error}"
            );
        }
    }
}

#[test]
fn invalid_sample_rate_fails_closed_before_encoder_execution() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-invalid-sample-rate.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture");
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");

    let prelude_called = Arc::new(AtomicBool::new(false));
    let prelude_runner = TestPreludeRunner {
        called: Arc::clone(&prelude_called),
        outcome: TestPreludeRunnerOutcome::Success,
    };
    let graph_called = Arc::new(AtomicBool::new(false));
    let graph_runner = TestEncoderGraphRunner {
        called: Arc::clone(&graph_called),
        outcome: TestEncoderGraphRunnerOutcome::Success,
    };
    let mel_called = Arc::new(AtomicBool::new(false));
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::clone(&mel_called),
        outcome: TestMelFeatureInputProviderOutcome::RealFrontend,
    };
    let mut invalid_audio = default_prepared_audio();
    invalid_audio.sample_rate_hz = 8_000;

    let error = execute_whisper_ggml_non_streaming_cpu(
        &whisper_runtime_descriptor_v1(),
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &invalid_audio,
        &mel_provider,
        &prelude_runner,
        &graph_runner,
        &WhisperDecoderGraphRunnerGgmlV0,
        &WhisperTokenizerProviderGgufV0,
    )
    .expect_err("invalid sample rate must fail before encoder execution");
    assert!(
        matches!(
            error,
            WhisperGgmlExecutorError::MelFeatureInputPreparationFailed { .. }
                | WhisperGgmlExecutorError::TokenizerMissing { .. }
                | WhisperGgmlExecutorError::DecoderWeightsMissing { .. }
                | WhisperGgmlExecutorError::DecoderGraphExecutionFailed { .. }
        ),
        "unexpected fail-closed boundary error: {error}"
    );
    if matches!(
        error,
        WhisperGgmlExecutorError::MelFeatureInputPreparationFailed { .. }
    ) {
        let message = error.to_string();
        assert!(
            message.contains("sample_rate_hz=8000"),
            "unexpected error: {message}"
        );
        assert_eq!(
            classify_whisper_execution_failure_stage(&message),
            WhisperExecutionFailureStage::MelFeature
        );
        assert!(mel_called.load(Ordering::SeqCst), "mel seam must run");
        assert!(
            !prelude_called.load(Ordering::SeqCst),
            "encoder prelude must not run for invalid sample rate"
        );
        assert!(
            !graph_called.load(Ordering::SeqCst),
            "encoder graph must not run for invalid sample rate"
        );
    }
}

#[test]
fn nan_audio_fails_closed_before_encoder_execution() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-nan-audio.gguf");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture");
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");

    let prelude_called = Arc::new(AtomicBool::new(false));
    let prelude_runner = TestPreludeRunner {
        called: Arc::clone(&prelude_called),
        outcome: TestPreludeRunnerOutcome::Success,
    };
    let graph_called = Arc::new(AtomicBool::new(false));
    let graph_runner = TestEncoderGraphRunner {
        called: Arc::clone(&graph_called),
        outcome: TestEncoderGraphRunnerOutcome::Success,
    };
    let mel_called = Arc::new(AtomicBool::new(false));
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::clone(&mel_called),
        outcome: TestMelFeatureInputProviderOutcome::RealFrontend,
    };
    let mut nan_audio = default_prepared_audio();
    nan_audio.samples_f32[5] = f32::NAN;

    let error = execute_whisper_ggml_non_streaming_cpu(
        &whisper_runtime_descriptor_v1(),
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &nan_audio,
        &mel_provider,
        &prelude_runner,
        &graph_runner,
        &WhisperDecoderGraphRunnerGgmlV0,
        &WhisperTokenizerProviderGgufV0,
    )
    .expect_err("non-finite audio must fail before encoder execution");
    assert!(
        matches!(
            error,
            WhisperGgmlExecutorError::MelFeatureInputPreparationFailed { .. }
                | WhisperGgmlExecutorError::TokenizerMissing { .. }
                | WhisperGgmlExecutorError::DecoderWeightsMissing { .. }
                | WhisperGgmlExecutorError::DecoderGraphExecutionFailed { .. }
        ),
        "unexpected fail-closed boundary error: {error}"
    );
    if matches!(
        error,
        WhisperGgmlExecutorError::MelFeatureInputPreparationFailed { .. }
    ) {
        let message = error.to_string();
        assert!(
            message.contains("samples_f32 contains non-finite values"),
            "unexpected error: {message}"
        );
        assert_eq!(
            classify_whisper_execution_failure_stage(&message),
            WhisperExecutionFailureStage::MelFeature
        );
        assert!(mel_called.load(Ordering::SeqCst), "mel seam must run");
        assert!(
            !prelude_called.load(Ordering::SeqCst),
            "encoder prelude must not run for non-finite audio"
        );
        assert!(
            !graph_called.load(Ordering::SeqCst),
            "encoder graph must not run for non-finite audio"
        );
    }
}

#[test]
fn unsupported_primitive_fixture_fails_closed_with_real_prelude_runner() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-prelude-capacity.gguf");
    let spec =
        TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_unsupported_primitive("whisper-fixture");
    write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write gguf fixture");
    let runtime_source =
        validate_ggml_runtime_source_path(&runtime_path).expect("validate runtime source");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
    let adapter = whisper_runtime_descriptor_v1();
    let runner = WhisperCpuEncoderPreludeComputeRunnerV0;
    let graph_runner = WhisperCpuEncoderGraphComputeRunnerV0;
    let mel_provider = TestMelFeatureInputProvider {
        called: Arc::new(AtomicBool::new(false)),
        outcome: TestMelFeatureInputProviderOutcome::RealFrontend,
    };

    let error = execute_whisper_ggml_non_streaming_cpu(
        &adapter,
        &runtime_source,
        &metadata,
        &load_whisper_tensor_index(&runtime_source).expect("load tensor index"),
        &default_prepared_audio(),
        &mel_provider,
        &runner,
        &graph_runner,
        &WhisperDecoderGraphRunnerGgmlV0,
        &WhisperTokenizerProviderGgufV0,
    )
    .expect_err("fixture should force unsupported prelude primitive");

    assert!(
        matches!(
            error,
            WhisperGgmlExecutorError::EncoderPreludePrimitiveUnsupported { .. }
                | WhisperGgmlExecutorError::TokenizerMissing { .. }
                | WhisperGgmlExecutorError::DecoderWeightsMissing { .. }
                | WhisperGgmlExecutorError::DecoderGraphExecutionFailed { .. }
        ),
        "unexpected fail-closed boundary error: {error}"
    );
    if matches!(
        error,
        WhisperGgmlExecutorError::EncoderPreludePrimitiveUnsupported { .. }
    ) {
        let message = error.to_string();
        assert_eq!(
            classify_whisper_execution_failure_stage(&message),
            WhisperExecutionFailureStage::EncoderPrelude
        );
        assert!(
            message.contains("encoder.positional_embedding.slice"),
            "unexpected error: {message}"
        );
    }
}

#[test]
fn decode_generated_token_step_cap_is_bounded() {
    let cap =
        decode_generated_token_step_cap(448, 4).expect("cap should be derived from ctx budget");
    assert_eq!(cap, WHISPER_DEFAULT_DECODE_MAX_GENERATED_TOKENS_CAP);

    let cap = decode_generated_token_step_cap(64, 4).expect("cap should respect tiny budget");
    assert_eq!(cap, 60);
}

#[test]
fn decode_generated_token_step_cap_fails_when_prompt_exhausts_context() {
    let error = decode_generated_token_step_cap(8, 8).expect_err("zero budget should fail");
    assert!(matches!(
        error,
        WhisperGgmlExecutorError::DecoderGraphUnsupported { .. }
    ));
}

#[test]
fn build_whisper_initial_prompt_tokens_appends_encoded_prompt_text() {
    let (execution, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let request_options = GgmlAsrExecutionOptions::from_transcription_request(
        None,
        Some(" hello world ".to_string()),
        None,
    );
    let decoder_start_token_id = tokenizer
        .start_of_transcript_token_id()
        .unwrap_or(execution.decoder_start_token_id);
    let prefix = tokenizer
        .decoder_prefix(
            decoder_start_token_id,
            &WhisperPrefixSpec::transcribe(false),
        )
        .expect("default prefix");
    let encoded_prompt = tokenizer
        .encode_prompt_text("hello world")
        .expect("encode prompt");

    let initial_prompt_tokens =
        build_whisper_initial_prompt_tokens(&execution, &tokenizer, &request_options, None)
            .expect("build initial prompt");

    assert_eq!(&initial_prompt_tokens[..prefix.len()], prefix.as_slice());
    assert_eq!(
        &initial_prompt_tokens[prefix.len()..],
        encoded_prompt.as_slice()
    );
}

#[test]
fn build_whisper_initial_prompt_tokens_truncates_prompt_to_context_tail() {
    let (execution, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let repeated_prompt = std::iter::repeat_n(" hello world", 128).collect::<String>();
    let request_options = GgmlAsrExecutionOptions::from_transcription_request(
        None,
        Some(repeated_prompt.clone()),
        None,
    );
    let decoder_start_token_id = tokenizer
        .start_of_transcript_token_id()
        .unwrap_or(execution.decoder_start_token_id);
    let prefix = tokenizer
        .decoder_prefix(
            decoder_start_token_id,
            &WhisperPrefixSpec::transcribe(false),
        )
        .expect("default prefix");
    let encoded_prompt = tokenizer
        .encode_prompt_text("hello world hello world hello world")
        .expect("sanity encode prompt");
    assert!(
        !encoded_prompt.is_empty(),
        "fixture prompt tokens must be non-empty"
    );

    let initial_prompt_tokens =
        build_whisper_initial_prompt_tokens(&execution, &tokenizer, &request_options, None)
            .expect("build initial prompt");

    let max_prompt_tokens = execution
        .max_target_positions
        .saturating_sub(prefix.len())
        .saturating_sub(1);
    assert_eq!(
        initial_prompt_tokens.len(),
        prefix.len() + max_prompt_tokens
    );
    let full_prompt_tokens = tokenizer
        .encode_prompt_text(repeated_prompt.trim())
        .expect("encode full repeated prompt");
    assert!(full_prompt_tokens.len() > max_prompt_tokens);
    assert_eq!(
        &initial_prompt_tokens[prefix.len()..],
        &full_prompt_tokens[full_prompt_tokens.len() - max_prompt_tokens..]
    );
}

#[test]
fn build_whisper_initial_prompt_tokens_caps_longform_prompt_tail() {
    let (execution, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let repeated_prompt = std::iter::repeat_n(" hello world", 128).collect::<String>();
    let request_options = GgmlAsrExecutionOptions::from_transcription_request(
        None,
        Some(repeated_prompt.clone()),
        Some(crate::LongFormOptions::default()),
    );
    let decoder_start_token_id = tokenizer
        .start_of_transcript_token_id()
        .unwrap_or(execution.decoder_start_token_id);
    let prefix = tokenizer
        .decoder_prefix(
            decoder_start_token_id,
            &WhisperPrefixSpec::transcribe(false),
        )
        .expect("default prefix");
    let prev_token_id = tokenizer
        .token_id_by_content("<|startofprev|>")
        .expect("fixture prev token");
    let full_prompt_tokens = tokenizer
        .encode_prompt_text(repeated_prompt.trim())
        .expect("encode full repeated prompt");

    let initial_prompt_tokens =
        build_whisper_initial_prompt_tokens(&execution, &tokenizer, &request_options, None)
            .expect("build initial prompt");
    let expected_tail = execution
        .max_target_positions
        .saturating_sub(prefix.len())
        .saturating_sub(1)
        .saturating_sub(1)
        .min(WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT);

    assert_eq!(
        initial_prompt_tokens.len(),
        prefix.len() + expected_tail + 1
    );
    assert_eq!(initial_prompt_tokens[0], prev_token_id);
    assert_eq!(
        &initial_prompt_tokens[1..1 + expected_tail],
        &full_prompt_tokens[full_prompt_tokens.len() - expected_tail..]
    );
    assert_eq!(
        &initial_prompt_tokens[1 + expected_tail..],
        prefix.as_slice()
    );
}

#[test]
fn build_whisper_initial_prompt_tokens_prefers_direct_prompt_token_ids() {
    let (execution, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let direct_prompt_tokens = vec![9, 10, 11, 12];
    let request_options = GgmlAsrExecutionOptions {
        prompt: Some("hello world".to_string()),
        prompt_token_ids: Some(direct_prompt_tokens.clone()),
        longform: Some(crate::LongFormOptions::default()),
        ..GgmlAsrExecutionOptions::default()
    };
    let decoder_start_token_id = tokenizer
        .start_of_transcript_token_id()
        .unwrap_or(execution.decoder_start_token_id);
    let prefix = tokenizer
        .decoder_prefix(
            decoder_start_token_id,
            &WhisperPrefixSpec::transcribe(false),
        )
        .expect("default prefix");
    let prev_token_id = tokenizer
        .token_id_by_content("<|startofprev|>")
        .expect("fixture prev token");

    let initial_prompt_tokens =
        build_whisper_initial_prompt_tokens(&execution, &tokenizer, &request_options, None)
            .expect("build initial prompt");

    assert_eq!(initial_prompt_tokens[0], prev_token_id);
    assert_eq!(
        &initial_prompt_tokens[1..1 + direct_prompt_tokens.len()],
        direct_prompt_tokens.as_slice()
    );
    assert_eq!(
        &initial_prompt_tokens[1 + direct_prompt_tokens.len()..],
        prefix.as_slice()
    );
}

#[test]
fn diarization_forced_word_anchors_keep_whisper_decode_path_identical() {
    // F1 regression: word timestamps forced solely for diarization must not
    // switch the whisper decode path (cross flash attention off +
    // cross-attention collection on), because that perturbs the transcript via
    // FP accumulation differences between diarize on/off.
    let plain = GgmlAsrExecutionOptions::default();
    let diarize_forced = GgmlAsrExecutionOptions {
        word_timestamps: true,
        word_timestamps_forced_for_diarization: true,
        ..GgmlAsrExecutionOptions::default()
    };
    let user_requested = GgmlAsrExecutionOptions {
        word_timestamps: true,
        ..GgmlAsrExecutionOptions::default()
    };

    assert_eq!(
        whisper_word_timestamp_mode(&plain),
        WhisperWordTimestampMode::Off
    );
    assert_eq!(
        whisper_word_timestamp_mode(&diarize_forced),
        WhisperWordTimestampMode::PostHocAnchors
    );
    assert_eq!(
        whisper_word_timestamp_mode(&user_requested),
        WhisperWordTimestampMode::CrossAttention
    );

    for cross_flash_enabled in [false, true] {
        // Diarize-forced anchors: decoder flags byte-identical to a plain run.
        assert_eq!(
            whisper_decoder_cross_attention_flags(cross_flash_enabled, &diarize_forced),
            whisper_decoder_cross_attention_flags(cross_flash_enabled, &plain),
            "diarize-forced word anchors must not alter the decode path (cross_flash_enabled={cross_flash_enabled})"
        );
        // User-requested word timestamps keep the higher-fidelity
        // cross-attention behavior: collection on, cross flash attention off.
        assert_eq!(
            whisper_decoder_cross_attention_flags(cross_flash_enabled, &user_requested),
            (false, true)
        );
    }
}

#[test]
fn build_whisper_carry_prompt_token_ids_keeps_last_longform_tail() {
    let (_, tokenizer) = whisper_execution_and_tokenizer_fixture();
    let request_options = GgmlAsrExecutionOptions {
        language: None,
        task: crate::TranscriptionTask::Transcribe,
        prompt: None,
        prompt_token_ids: Some((1..=40).collect()),
        phrase_bias: None,
        inference_threads: None,
        word_timestamps: false,
        word_timestamps_forced_for_diarization: false,
        diarize: false,
        longform: Some(crate::LongFormOptions::default()),
        longform_chunk_count_hint: None,
        prefer_cpu_decoder_for_multichunk_metal: false,
        adapter_path: None,
    };

    let carry_prompt_token_ids =
        build_whisper_carry_prompt_token_ids(&tokenizer, &request_options, &[41, 42, 43, 44])
            .expect("carry prompt tokens")
            .expect("carry prompt token ids");

    assert_eq!(
        carry_prompt_token_ids.len(),
        WHISPER_LONGFORM_PROMPT_TOKEN_TAIL_LIMIT
    );
    assert_eq!(
        carry_prompt_token_ids.as_slice(),
        &(13..=44).collect::<Vec<_>>()
    );
}

#[test]
fn whisper_serve_batch_allows_longform_on_direct_gpu_lane() {
    let mut direct_gpu = GgmlCpuGraphConfig::conservative_default();
    direct_gpu.backend = GgmlCpuGraphBackend::Gpu;
    direct_gpu.use_scheduler = false;
    let request_options = GgmlAsrExecutionOptions {
        longform: Some(crate::LongFormOptions::default()),
        ..GgmlAsrExecutionOptions::default()
    };

    assert!(whisper_can_use_serve_batch(
        direct_gpu,
        &request_options,
        true
    ));
}

#[test]
fn whisper_serve_batch_rejects_scheduler_and_cpu_lanes() {
    let request_options = GgmlAsrExecutionOptions::default();
    let mut scheduler_gpu = GgmlCpuGraphConfig::conservative_default();
    scheduler_gpu.backend = GgmlCpuGraphBackend::Gpu;
    scheduler_gpu.use_scheduler = true;
    let mut cpu = GgmlCpuGraphConfig::conservative_default();
    cpu.backend = GgmlCpuGraphBackend::Cpu;
    cpu.use_scheduler = false;

    assert!(!whisper_can_use_serve_batch(
        scheduler_gpu,
        &request_options,
        false
    ));
    assert!(!whisper_can_use_serve_batch(cpu, &request_options, false));
}

#[test]
fn decoder_quantized_tensor_is_indexed_in_quantized_source_map() {
    let mut tensors_f32_by_name = HashMap::new();
    let mut tensors_f16_bits_by_name = HashMap::new();
    let mut tensors_quantized_by_name = HashMap::new();
    insert_decoder_tensor_owned(
        &mut tensors_f32_by_name,
        &mut tensors_f16_bits_by_name,
        &mut tensors_quantized_by_name,
        WhisperMaterializedTensor {
            slot: WhisperGgufTensorSlot::DecoderLayerSelfAttnQWeight { layer_idx: 0 },
            tensor_name: "model.decoder.layers.0.self_attn.q_proj.weight".to_string(),
            dims: vec![2, 2],
            num_elements: 4,
            payload: WhisperMaterializedTensorPayload::Quantized {
                ggml_type: 8,
                bytes: vec![1, 2, 3, 4],
            },
        },
    )
    .expect("quantized decoder tensor should be indexed");
    assert!(tensors_f32_by_name.is_empty());
    assert!(tensors_f16_bits_by_name.is_empty());
    let (ggml_type, bytes) = tensors_quantized_by_name
        .get("model.decoder.layers.0.self_attn.q_proj.weight")
        .expect("quantized map must include tensor");
    assert_eq!(*ggml_type, 8);
    assert_eq!(bytes.as_ref(), &[1, 2, 3, 4]);
}

#[test]
fn decoder_quantized_tensor_with_empty_bytes_fails_closed() {
    let mut tensors_f32_by_name = HashMap::new();
    let mut tensors_f16_bits_by_name = HashMap::new();
    let mut tensors_quantized_by_name = HashMap::new();
    let error = insert_decoder_tensor_owned(
        &mut tensors_f32_by_name,
        &mut tensors_f16_bits_by_name,
        &mut tensors_quantized_by_name,
        WhisperMaterializedTensor {
            slot: WhisperGgufTensorSlot::DecoderLayerSelfAttnQWeight { layer_idx: 0 },
            tensor_name: "model.decoder.layers.0.self_attn.q_proj.weight".to_string(),
            dims: vec![2, 2],
            num_elements: 4,
            payload: WhisperMaterializedTensorPayload::Quantized {
                ggml_type: 8,
                bytes: Vec::new(),
            },
        },
    )
    .expect_err("empty quantized bytes must fail closed");
    assert!(matches!(
        error,
        WhisperGgmlExecutorError::DecoderWeightsMissing { .. }
    ));
    assert!(
        error
            .to_string()
            .contains("materialized quantized type 8 with empty bytes"),
        "unexpected error: {error}"
    );
}

#[test]
fn encoder_persistent_session_cache_is_backend_scoped() {
    let temp = tempfile::tempdir().expect("tempdir");
    let runtime_path = temp.path().join("whisper-backend-scope.gguf");
    let cpu_config = GgmlCpuGraphConfig::conservative_default();
    let session = WhisperEncoderPersistentStaticSession {
        runner: GgmlCpuGraphRunner::new(cpu_config).expect("runner"),
        resident_weights: None,
        graph_config: cpu_config,
        encoder_layers: 1,
        encoder_hidden_size: 4,
    };

    store_whisper_encoder_persistent_static_session(&runtime_path, session);

    assert!(
        take_whisper_encoder_persistent_static_session(&runtime_path, GgmlCpuGraphBackend::Gpu)
            .is_none(),
        "a GPU request must not steal a CPU encoder session"
    );
    let session =
        take_whisper_encoder_persistent_static_session(&runtime_path, GgmlCpuGraphBackend::Cpu)
            .expect("CPU session should remain cached under the CPU key");
    assert_eq!(session.graph_config.backend, GgmlCpuGraphBackend::Cpu);
}

#[test]
fn encoder_graph_upload_bytes_after_prepare_outputs_remains_supported() {
    const TEST_GGML_TYPE_F16: i32 = 1;
    let mut runner =
        GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::conservative_default()).expect("runner");
    let mut graph = runner.start_graph();
    let output = graph.new_tensor_1d_f32(1, "output").expect("output tensor");
    let quantized_weight = graph
        .new_tensor_2d_typed(32, 1, TEST_GGML_TYPE_F16, "quantized_weight")
        .expect("quantized tensor");
    graph.set_input(output).expect("output input");
    graph.set_input(quantized_weight).expect("quantized input");
    graph.set_output(output).expect("set output");
    graph
        .prepare_outputs_for_upload(&[output])
        .expect("prepare outputs");

    let uploads = vec![
        WhisperEncoderGraphUpload::f32_owned(output, vec![0.0], "output"),
        WhisperEncoderGraphUpload::bytes(
            quantized_weight,
            vec![0_u8; 32 * std::mem::size_of::<u16>()],
            "quantized_weight",
        ),
    ];
    upload_encoder_graph_inputs(&mut graph, uploads).expect("upload bytes payload");

    let out = graph.compute_output_f32(output, 1).expect("compute output");
    assert_eq!(out, vec![0.0]);
}
