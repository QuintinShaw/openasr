//! Bind a diagnostic short-audio run to formal `evidence.v1`.
//!
//! Generic `bench-receipt short-audio` must leave `evidence` absent. This is
//! the only constructor that may attach immutable matrix, catalog, and
//! artifact identity. It does not select a provider, enable compact output, or
//! write `active.json`.

use crate::short_audio_receipt::ShortAudioTraceSummary;
use crate::{
    GgmlGraphLifecycleEventKind, SHORT_AUDIO_RECEIPT_ARTIFACT_CONTRACT,
    SHORT_AUDIO_RECEIPT_EVIDENCE_SCHEMA, ShortAudioArtifactIdentity, ShortAudioCaptureMode,
    ShortAudioCatalogDigests, ShortAudioEvidenceClass, ShortAudioExecutionMode,
    ShortAudioFamilyOracle, ShortAudioOutputPlan, ShortAudioReceipt, ShortAudioReceiptArtifacts,
    ShortAudioReceiptError, ShortAudioReceiptEvidence, ShortAudioReceiptReuseMode,
    ShortAudioReuseMode, ShortAudioSchedulerMode, ShortAudioTopKSummary,
};

/// Caller-supplied identity that a diagnostic receipt cannot infer.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RealFamilyEvidenceBinding {
    pub matrix_sha256: String,
    pub candidate_release_subject: String,
    pub catalog_digests: ShortAudioCatalogDigests,
    pub family: String,
    pub model_id: String,
    pub quant: String,
    pub topology: String,
    pub provider: String,
    pub device_target: String,
    pub backend_id: String,
    pub driver_version: String,
    pub artifact_fingerprint: String,
    pub device: String,
    pub placement: String,
    pub capture_mode: ShortAudioCaptureMode,
    pub scheduler_mode: ShortAudioSchedulerMode,
    pub graph_mode: ShortAudioReceiptReuseMode,
    pub output_plan: ShortAudioOutputPlan,
    pub artifacts: ShortAudioReceiptArtifacts,
}

/// Hashed token/logits artifacts for one measured request.
#[derive(Debug, Clone, PartialEq)]
pub struct RealFamilyTraceArtifacts {
    pub token_trace: ShortAudioArtifactIdentity,
    pub logits: Option<ShortAudioArtifactIdentity>,
    pub top_k: Vec<ShortAudioTopKSummary>,
    pub top1_top2_margin: Option<f64>,
}

/// One physical run yields two disjoint evidence classes. They share the same
/// native observation and must not be substituted for each other.
#[derive(Debug, Clone, PartialEq)]
pub struct RealFamilyEvidenceSet {
    pub placement: ShortAudioReceipt,
    pub token_transcript: ShortAudioReceipt,
}

/// Attach formal evidence to a native diagnostic receipt.
///
/// The input may carry diagnostic notes; the output always clears them. Mock
/// backends, missing placement, incomplete execution, and capture/scheduler
/// mismatches fail closed.
pub fn bind_real_family_evidence(
    mut diagnostic: ShortAudioReceipt,
    binding: &RealFamilyEvidenceBinding,
    traces: &RealFamilyTraceArtifacts,
) -> Result<RealFamilyEvidenceSet, ShortAudioReceiptError> {
    if diagnostic.evidence.is_some() {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "diagnostic receipt already carries evidence",
        });
    }
    if diagnostic.run.backend != "native" {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "real-family evidence requires a native backend run",
        });
    }
    if diagnostic.observed_placement.is_none() {
        return Err(ShortAudioReceiptError::PlacementEvidenceMissing);
    }
    if diagnostic.execution.is_none() {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "runtime execution projection is missing",
        });
    }
    let Some(diagnostics) = diagnostic.decode_diagnostics.as_ref() else {
        return Err(ShortAudioReceiptError::DecodeDiagnosticsMissing);
    };
    if diagnostics.reuse_mode != binding.graph_mode {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "decode graph mode does not match the evidence binding",
        });
    }
    let execution_mode = match (
        diagnostic.run.warmup.as_str(),
        diagnostic.run.cache_state.as_str(),
    ) {
        ("cold", "empty") => ShortAudioReuseMode::Cold,
        ("warm", "populated") => ShortAudioReuseMode::Reuse,
        _ => {
            return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
                reason: "run warmup/cache_state is not a cold or reuse pair",
            });
        }
    };
    validate_live_identity(&diagnostic, binding)?;
    validate_pack_and_fixture(&diagnostic, binding)?;
    validate_capture_and_scheduler(&diagnostic, binding)?;
    if binding.output_plan.kind == crate::ShortAudioOutputPlanKind::FullLogits
        && traces.logits.is_none()
    {
        return Err(ShortAudioReceiptError::InvalidEvidenceField {
            field: "correctness.trace.logits",
            actual: "missing for a complete-output plan".to_string(),
        });
    }

    diagnostic.notes.clear();
    diagnostic.evidence = None;

    let family_oracle = ShortAudioFamilyOracle {
        family: binding.family.clone(),
        tie_policy: binding.output_plan.tie_policy,
    };
    let core_commit = diagnostic.core_commit.clone();
    let actual_provider = diagnostic.actual_provider.clone();
    let actual_stable_device_id = diagnostic.actual_stable_device_id.clone();
    let actual_device = diagnostic.actual_device.clone();
    let common =
        |class: ShortAudioEvidenceClass, include_token_fields: bool| -> ShortAudioReceiptEvidence {
            ShortAudioReceiptEvidence {
                schema: SHORT_AUDIO_RECEIPT_EVIDENCE_SCHEMA.to_string(),
                contract: SHORT_AUDIO_RECEIPT_ARTIFACT_CONTRACT.to_string(),
                evidence_class: class,
                matrix_sha256: binding.matrix_sha256.clone(),
                candidate_release_subject: binding.candidate_release_subject.clone(),
                core_commit: core_commit.clone(),
                catalog_digests: binding.catalog_digests.clone(),
                family: binding.family.clone(),
                model_id: binding.model_id.clone(),
                quant: binding.quant.clone(),
                topology: binding.topology.clone(),
                provider: binding.provider.clone(),
                device_target: binding.device_target.clone(),
                backend_id: binding.backend_id.clone(),
                driver_version: binding.driver_version.clone(),
                artifact_fingerprint: binding.artifact_fingerprint.clone(),
                device: binding.device.clone(),
                actual_provider: actual_provider.clone(),
                actual_stable_device_id: actual_stable_device_id.clone(),
                actual_device: actual_device.clone(),
                placement: binding.placement.clone(),
                capture_mode: binding.capture_mode,
                scheduler_mode: binding.scheduler_mode,
                result: "pass".to_string(),
                artifacts: binding.artifacts.clone(),
                output_plan: include_token_fields.then(|| binding.output_plan.clone()),
                family_oracle: include_token_fields.then(|| family_oracle.clone()),
                execution: include_token_fields.then_some(ShortAudioExecutionMode {
                    mode: execution_mode,
                    graph_rebuild_reason: None,
                }),
                trace: include_token_fields.then(|| ShortAudioTraceSummary {
                    token_trace: traces.token_trace.clone(),
                    logits: traces.logits.clone(),
                    top_k: traces.top_k.clone(),
                    top1_top2_margin: traces.top1_top2_margin,
                }),
            }
        };

    let mut placement = diagnostic.clone();
    placement.evidence = Some(common(ShortAudioEvidenceClass::PlacementResource, false));
    placement.validate_qualification_eligibility()?;

    let mut token_transcript = diagnostic;
    token_transcript.evidence = Some(common(ShortAudioEvidenceClass::TokenTranscript, true));
    token_transcript.validate_qualification_eligibility()?;

    Ok(RealFamilyEvidenceSet {
        placement,
        token_transcript,
    })
}

fn validate_live_identity(
    diagnostic: &ShortAudioReceipt,
    binding: &RealFamilyEvidenceBinding,
) -> Result<(), ShortAudioReceiptError> {
    let Some(actual_provider) = diagnostic.actual_provider.as_deref() else {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "final live provider is missing",
        });
    };
    let Some(actual_device) = diagnostic.actual_stable_device_id.as_deref() else {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "final live device identity is missing",
        });
    };
    if diagnostic.actual_device.is_none() {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "final live device facts are missing",
        });
    }
    if actual_provider != binding.provider {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "live provider differs from the evidence binding",
        });
    }
    if actual_device != binding.device {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "live device identity differs from the evidence binding",
        });
    }
    Ok(())
}

fn validate_pack_and_fixture(
    diagnostic: &ShortAudioReceipt,
    binding: &RealFamilyEvidenceBinding,
) -> Result<(), ShortAudioReceiptError> {
    let expected_model = format!("{}:{}", binding.model_id, binding.quant);
    if diagnostic.pack.model_id != expected_model || diagnostic.pack.quant != binding.quant {
        return Err(ShortAudioReceiptError::InvalidEvidenceField {
            field: "pack.model_id",
            actual: format!(
                "{} / {} vs binding {expected_model} / {}",
                diagnostic.pack.model_id, diagnostic.pack.quant, binding.quant
            ),
        });
    }
    if binding.artifacts.pack.sha256 != diagnostic.pack.content_sha256 {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "pack artifact hash does not match the receipt pack",
        });
    }
    if binding.artifacts.fixture.sha256 != diagnostic.audio.sha256 {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "fixture artifact hash does not match the receipt audio",
        });
    }
    Ok(())
}

fn validate_capture_and_scheduler(
    diagnostic: &ShortAudioReceipt,
    binding: &RealFamilyEvidenceBinding,
) -> Result<(), ShortAudioReceiptError> {
    let Some(lifecycle) = diagnostic.graph_lifecycle.as_ref() else {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "graph lifecycle observation is missing",
        });
    };
    if lifecycle.overflowed || lifecycle.events.is_empty() {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "graph lifecycle observation is empty or overflowed",
        });
    }
    let mut scheduler_enabled = None;
    let mut capture_supported = false;
    let mut capture_enabled = None;
    let mut tracked = false;
    let mut executable = false;
    let mut executable_generation = false;
    let mut compute_consumed_generation = false;
    let mut compute_in_flight = false;
    let mut created_during_compute = false;
    for event in &lifecycle.events {
        match &event.kind {
            GgmlGraphLifecycleEventKind::Created {
                scheduler_enabled: enabled,
            }
            | GgmlGraphLifecycleEventKind::ExistingGraphObserved {
                scheduler_enabled: enabled,
                ..
            } => {
                scheduler_enabled = Some(*enabled);
            }
            GgmlGraphLifecycleEventKind::CaptureStateObserved {
                capture_supported: supported,
                graph_tracked,
                capture_enabled: enabled,
                executable_present,
                ..
            } => {
                capture_supported |= *supported;
                tracked |= *graph_tracked;
                if let Some(value) = *enabled {
                    capture_enabled = Some(capture_enabled.unwrap_or(false) || value);
                }
                executable |= *executable_present;
            }
            GgmlGraphLifecycleEventKind::CaptureExecutableCreated { .. } => {
                executable_generation = true;
                created_during_compute |= compute_in_flight;
            }
            GgmlGraphLifecycleEventKind::CaptureExecutableObserved { .. } => {
                executable_generation = true;
            }
            GgmlGraphLifecycleEventKind::ComputeStarted {
                capture_executable_generation,
                ..
            } => {
                compute_in_flight = true;
                compute_consumed_generation |= capture_executable_generation.is_some();
            }
            GgmlGraphLifecycleEventKind::ComputeCompleted { .. }
            | GgmlGraphLifecycleEventKind::Dropped => {
                compute_in_flight = false;
            }
            _ => {}
        }
    }
    let observed_scheduler = match scheduler_enabled {
        Some(true) => ShortAudioSchedulerMode::Enabled,
        Some(false) | None => ShortAudioSchedulerMode::Disabled,
    };
    if observed_scheduler != binding.scheduler_mode {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "observed scheduler mode does not match the evidence binding",
        });
    }
    let observed_capture = if capture_supported && capture_enabled == Some(true) && tracked {
        ShortAudioCaptureMode::Enabled
    } else if capture_supported && capture_enabled == Some(false) && tracked {
        ShortAudioCaptureMode::Disabled
    } else {
        ShortAudioCaptureMode::Unsupported
    };
    if observed_capture != binding.capture_mode {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "observed capture mode does not match the evidence binding",
        });
    }
    if binding.capture_mode == ShortAudioCaptureMode::Enabled
        && (!executable
            || !executable_generation
            || !(compute_consumed_generation || created_during_compute))
    {
        return Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete {
            reason: "capture-enabled lane did not observe an executable consumed by compute",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE, SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK,
        SHORT_AUDIO_RECEIPT_SCHEMA, ShortAudioExecutionProjection, ShortAudioLeaseReconciliation,
        ShortAudioOutputPlanKind, ShortAudioReceiptAudio, ShortAudioReceiptDecodeDiagnostics,
        ShortAudioReceiptMetrics, ShortAudioReceiptOutputPlan, ShortAudioReceiptPack,
        ShortAudioReceiptRun, ShortAudioReceiptTranscript, ShortAudioTiePolicy,
        ggml_runtime::{
            GGML_GRAPH_LIFECYCLE_SCHEMA, GgmlActualDeviceFacts, GgmlCaptureExecutableChange,
            GgmlCaptureObservationPhase, GgmlExecutionPlacementSummary, GgmlGraphLifecycleEvent,
            GgmlGraphLifecycleSnapshot,
        },
    };
    use std::collections::BTreeMap;

    fn sha(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn identity(label: &str, digest: char) -> ShortAudioArtifactIdentity {
        ShortAudioArtifactIdentity {
            label: label.to_string(),
            sha256: sha(digest),
            size_bytes: Some(16),
        }
    }

    fn qualifying_execution() -> ShortAudioExecutionProjection {
        ShortAudioExecutionProjection {
            request_attempt_id: Some(
                crate::RequestAttemptId::parse("0123456789abcdef0123456789abcdef").expect("id"),
            ),
            request_attempt_conflicted: false,
            candidate_attempt_ids: Vec::new(),
            lanes: Vec::new(),
            memory_domains: Vec::new(),
            live_lease_reconciliation: ShortAudioLeaseReconciliation::Matched,
            reconciliation_reason: None,
            live_state_complete: true,
            live_state_reason: None,
            event_history_complete: true,
            event_history_reason: None,
            dropped_events: 0,
            phase_duration_micros: BTreeMap::from([
                ("upload-ingest".to_string(), 1),
                ("decode-normalize".to_string(), 1),
                ("admission-wait".to_string(), 1),
                ("compute".to_string(), 1),
            ]),
            timing_complete: true,
            terminal: Some("succeeded".to_string()),
            request_receipt_complete: true,
        }
    }

    fn lifecycle(capture_enabled: bool) -> GgmlGraphLifecycleSnapshot {
        let device = "hip0";
        let provider = "hip";
        GgmlGraphLifecycleSnapshot {
            events: vec![
                GgmlGraphLifecycleEvent {
                    schema: GGML_GRAPH_LIFECYCLE_SCHEMA.into(),
                    sequence: 1,
                    provider: provider.into(),
                    device: device.into(),
                    graph_instance: 1,
                    graph_generation: 1,
                    kind: GgmlGraphLifecycleEventKind::Created {
                        scheduler_enabled: false,
                    },
                },
                GgmlGraphLifecycleEvent {
                    schema: GGML_GRAPH_LIFECYCLE_SCHEMA.into(),
                    sequence: 2,
                    provider: provider.into(),
                    device: device.into(),
                    graph_instance: 1,
                    graph_generation: 1,
                    kind: GgmlGraphLifecycleEventKind::CaptureStateObserved {
                        phase: GgmlCaptureObservationPhase::BeforeCompute,
                        capture_supported: true,
                        graph_tracked: true,
                        capture_enabled: Some(capture_enabled),
                        executable_present: capture_enabled,
                    },
                },
                GgmlGraphLifecycleEvent {
                    schema: GGML_GRAPH_LIFECYCLE_SCHEMA.into(),
                    sequence: 3,
                    provider: provider.into(),
                    device: device.into(),
                    graph_instance: 1,
                    graph_generation: 1,
                    kind: GgmlGraphLifecycleEventKind::ComputeStarted {
                        compute_sequence: 1,
                        prepare_generation: Some(1),
                        input_generation_consumed: Some(1),
                        capture_executable_generation: capture_enabled.then_some(4),
                    },
                },
                GgmlGraphLifecycleEvent {
                    schema: GGML_GRAPH_LIFECYCLE_SCHEMA.into(),
                    sequence: 4,
                    provider: provider.into(),
                    device: device.into(),
                    graph_instance: 1,
                    graph_generation: 1,
                    kind: GgmlGraphLifecycleEventKind::CaptureExecutableObserved {
                        capture_executable_generation: 4,
                        last_change: GgmlCaptureExecutableChange::Instantiated,
                    },
                },
                GgmlGraphLifecycleEvent {
                    schema: GGML_GRAPH_LIFECYCLE_SCHEMA.into(),
                    sequence: 5,
                    provider: provider.into(),
                    device: device.into(),
                    graph_instance: 1,
                    graph_generation: 1,
                    kind: GgmlGraphLifecycleEventKind::Dropped,
                },
            ],
            overflowed: false,
        }
    }

    fn diagnostic_receipt() -> ShortAudioReceipt {
        ShortAudioReceipt {
            schema: SHORT_AUDIO_RECEIPT_SCHEMA.to_string(),
            core_commit: "0123456789abcdef0123456789abcdef01234567".to_string(),
            pack: ShortAudioReceiptPack {
                model_id: "funasr-nano:q4_k".to_string(),
                content_sha256: sha('e'),
                size_bytes: 12,
                quant: "q4_k".to_string(),
            },
            audio: ShortAudioReceiptAudio {
                path_or_label: format!("audio-sha256:{}", sha('f')),
                sha256: sha('f'),
                duration_s: Some(1.5),
            },
            run: ShortAudioReceiptRun {
                backend: "native".to_string(),
                device: "hip".to_string(),
                os: "windows".to_string(),
                command: vec!["openasr".to_string(), "qualify-family".to_string()],
                env_allowlist: BTreeMap::from([
                    ("OPENASR_GGML_BACKEND".to_string(), "hip".to_string()),
                    (
                        "OPENASR_BUILD_COMMIT".to_string(),
                        "0123456789abcdef0123456789abcdef01234567".to_string(),
                    ),
                    ("OPENASR_OFFLINE".to_string(), "true".to_string()),
                ]),
                warmup: "cold".to_string(),
                cache_state: "empty".to_string(),
            },
            metrics: ShortAudioReceiptMetrics {
                wer_or_cer: None,
                rtf_samples: vec![0.4],
                rtf_median: None,
                ttft_s: None,
                peak_rss_bytes: Some(1024),
                peak_rss_before_model_bytes: Some(640),
                rss_before_model_bytes: Some(512),
                rss_after_model_bytes: Some(768),
                phys_footprint_before_model_bytes: None,
                phys_footprint_after_model_bytes: None,
                peak_phys_footprint_before_model_bytes: None,
                peak_phys_footprint_bytes: None,
                peak_vram_bytes: None,
                measurement_method: Some(SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK.to_string()),
            },
            transcript: ShortAudioReceiptTranscript::from_text("hello"),
            placement: "hip".to_string(),
            observed_placement: Some(GgmlExecutionPlacementSummary {
                direct_graph_computes: 1,
                scheduler_graph_computes: 0,
                observed_nodes_by_backend: BTreeMap::from([("HIP0".to_string(), 12)]),
                observed_compute_nodes_by_backend: BTreeMap::from([("HIP0".to_string(), 10)]),
                observed_node_output_bytes_by_backend: BTreeMap::from([("HIP0".to_string(), 4096)]),
                fallback_node_samples_by_backend: BTreeMap::new(),
            }),
            graph_lifecycle: Some(lifecycle(true)),
            actual_provider: Some("hip".to_string()),
            actual_stable_device_id: Some("hip0".to_string()),
            actual_device: Some(GgmlActualDeviceFacts {
                device_type: "gpu".to_string(),
                name: "hip0".to_string(),
                description: "test hip device".to_string(),
                provider_device_id: Some("0000:03:00.0".to_string()),
                pci_vendor_id: Some(0x1002),
            }),
            evidence: None,
            execution: Some(qualifying_execution()),
            scope: SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE.to_string(),
            notes: vec!["decode_stop=stop_token".to_string()],
            decode_diagnostics: Some(ShortAudioReceiptDecodeDiagnostics {
                output_plan: ShortAudioReceiptOutputPlan::FullLogits,
                reuse_mode: ShortAudioReceiptReuseMode::FreshGraph,
                capability_evidence_revision: Some(1),
                steps: Vec::new(),
                first_divergence: None,
                encoder_decoder_splits: Vec::new(),
            }),
        }
    }

    fn binding() -> RealFamilyEvidenceBinding {
        RealFamilyEvidenceBinding {
            matrix_sha256: sha('a'),
            candidate_release_subject: "v0.1.36-test".to_string(),
            catalog_digests: ShortAudioCatalogDigests {
                inventory_sha256: sha('1'),
                model_catalog_sha256: sha('2'),
                backend_catalog_sha256: sha('3'),
            },
            family: "qwen".to_string(),
            model_id: "funasr-nano".to_string(),
            quant: "q4_k".to_string(),
            topology: "causal-self-attention-kv".to_string(),
            provider: "hip".to_string(),
            device_target: "gfx1200".to_string(),
            backend_id: "hip-windows-x86_64-test-gfx1200".to_string(),
            driver_version: "7.1.0".to_string(),
            artifact_fingerprint: sha('9'),
            device: "hip0".to_string(),
            placement: "full_device".to_string(),
            capture_mode: ShortAudioCaptureMode::Enabled,
            scheduler_mode: ShortAudioSchedulerMode::Disabled,
            graph_mode: ShortAudioReceiptReuseMode::FreshGraph,
            output_plan: ShortAudioOutputPlan {
                kind: ShortAudioOutputPlanKind::FullLogits,
                requires_complete_output: true,
                tie_policy: ShortAudioTiePolicy::FirstMaximum,
            },
            artifacts: ShortAudioReceiptArtifacts {
                binary: identity("openasr-test-binary", 'c'),
                plugin: identity("hip-plugin", 'd'),
                pack: identity("fixture-pack", 'e'),
                fixture: identity("jfk-short", 'f'),
            },
        }
    }

    fn traces() -> RealFamilyTraceArtifacts {
        RealFamilyTraceArtifacts {
            token_trace: identity("token-trace.jsonl", '4'),
            logits: Some(identity("logits.jsonl", '5')),
            top_k: vec![ShortAudioTopKSummary {
                token_id: 7,
                value: 1.25,
            }],
            top1_top2_margin: Some(0.5),
        }
    }

    #[test]
    fn binder_emits_disjoint_placement_and_token_receipts() {
        let set = bind_real_family_evidence(diagnostic_receipt(), &binding(), &traces())
            .expect("complete native diagnostic should bind");
        assert!(set.placement.notes.is_empty());
        assert!(set.token_transcript.notes.is_empty());
        assert_eq!(
            set.placement.evidence.as_ref().unwrap().evidence_class,
            ShortAudioEvidenceClass::PlacementResource
        );
        assert_eq!(
            set.token_transcript
                .evidence
                .as_ref()
                .unwrap()
                .evidence_class,
            ShortAudioEvidenceClass::TokenTranscript
        );
        assert!(set.placement.evidence.as_ref().unwrap().trace.is_none());
        assert!(
            set.token_transcript
                .evidence
                .as_ref()
                .unwrap()
                .trace
                .is_some()
        );
        set.placement
            .validate_qualification_eligibility()
            .expect("placement class is qualification-eligible");
        set.token_transcript
            .validate_qualification_eligibility()
            .expect("token class is qualification-eligible");
    }

    fn first_compute_capture_lifecycle() -> GgmlGraphLifecycleSnapshot {
        let device = "ROCm0";
        let provider = "hip";
        GgmlGraphLifecycleSnapshot {
            events: vec![
                GgmlGraphLifecycleEvent {
                    schema: GGML_GRAPH_LIFECYCLE_SCHEMA.into(),
                    sequence: 1,
                    provider: provider.into(),
                    device: device.into(),
                    graph_instance: 78,
                    graph_generation: 79,
                    kind: GgmlGraphLifecycleEventKind::Created {
                        scheduler_enabled: false,
                    },
                },
                GgmlGraphLifecycleEvent {
                    schema: GGML_GRAPH_LIFECYCLE_SCHEMA.into(),
                    sequence: 2,
                    provider: provider.into(),
                    device: device.into(),
                    graph_instance: 78,
                    graph_generation: 79,
                    kind: GgmlGraphLifecycleEventKind::CaptureStateObserved {
                        phase: GgmlCaptureObservationPhase::BeforeCompute,
                        capture_supported: true,
                        graph_tracked: false,
                        capture_enabled: None,
                        executable_present: false,
                    },
                },
                GgmlGraphLifecycleEvent {
                    schema: GGML_GRAPH_LIFECYCLE_SCHEMA.into(),
                    sequence: 3,
                    provider: provider.into(),
                    device: device.into(),
                    graph_instance: 78,
                    graph_generation: 79,
                    kind: GgmlGraphLifecycleEventKind::ComputeStarted {
                        compute_sequence: 1,
                        prepare_generation: Some(1),
                        input_generation_consumed: Some(1),
                        capture_executable_generation: None,
                    },
                },
                GgmlGraphLifecycleEvent {
                    schema: GGML_GRAPH_LIFECYCLE_SCHEMA.into(),
                    sequence: 4,
                    provider: provider.into(),
                    device: device.into(),
                    graph_instance: 78,
                    graph_generation: 79,
                    kind: GgmlGraphLifecycleEventKind::CaptureStateObserved {
                        phase: GgmlCaptureObservationPhase::AfterCompute,
                        capture_supported: true,
                        graph_tracked: true,
                        capture_enabled: Some(true),
                        executable_present: true,
                    },
                },
                GgmlGraphLifecycleEvent {
                    schema: GGML_GRAPH_LIFECYCLE_SCHEMA.into(),
                    sequence: 5,
                    provider: provider.into(),
                    device: device.into(),
                    graph_instance: 78,
                    graph_generation: 79,
                    kind: GgmlGraphLifecycleEventKind::CaptureExecutableCreated {
                        capture_executable_generation: 1,
                        change: GgmlCaptureExecutableChange::Instantiated,
                    },
                },
                GgmlGraphLifecycleEvent {
                    schema: GGML_GRAPH_LIFECYCLE_SCHEMA.into(),
                    sequence: 6,
                    provider: provider.into(),
                    device: device.into(),
                    graph_instance: 78,
                    graph_generation: 79,
                    kind: GgmlGraphLifecycleEventKind::ComputeCompleted {
                        compute_sequence: 1,
                        output_generation: 6,
                    },
                },
                GgmlGraphLifecycleEvent {
                    schema: GGML_GRAPH_LIFECYCLE_SCHEMA.into(),
                    sequence: 7,
                    provider: provider.into(),
                    device: device.into(),
                    graph_instance: 78,
                    graph_generation: 79,
                    kind: GgmlGraphLifecycleEventKind::Dropped,
                },
            ],
            overflowed: false,
        }
    }

    #[test]
    fn binder_accepts_first_compute_capture_launch_on_fresh_graph() {
        let mut diagnostic = diagnostic_receipt();
        diagnostic.actual_stable_device_id = Some("ROCm0".to_string());
        diagnostic.actual_device = Some(GgmlActualDeviceFacts {
            device_type: "gpu".to_string(),
            name: "ROCm0".to_string(),
            description: "test hip device".to_string(),
            provider_device_id: Some("0000:03:00.0".to_string()),
            pci_vendor_id: Some(0x1002),
        });
        diagnostic.graph_lifecycle = Some(first_compute_capture_lifecycle());
        let mut binding = binding();
        binding.device = "ROCm0".to_string();
        bind_real_family_evidence(diagnostic, &binding, &traces())
            .expect("FreshGraph capture-on launches the executable on the capturing compute");
    }

    #[test]
    fn binder_rejects_mock_and_capture_mismatch() {
        let mut mock = diagnostic_receipt();
        mock.run.backend = "mock".to_string();
        assert!(matches!(
            bind_real_family_evidence(mock, &binding(), &traces()),
            Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete { .. })
        ));

        let mut disabled = binding();
        disabled.capture_mode = ShortAudioCaptureMode::Disabled;
        assert!(matches!(
            bind_real_family_evidence(diagnostic_receipt(), &disabled, &traces()),
            Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete { .. })
        ));
    }

    #[test]
    fn binder_rejects_pack_or_provider_drift() {
        let mut wrong_pack = binding();
        wrong_pack.artifacts.pack.sha256 = sha('z');
        assert!(matches!(
            bind_real_family_evidence(diagnostic_receipt(), &wrong_pack, &traces()),
            Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete { .. })
        ));

        let mut wrong_provider = binding();
        wrong_provider.provider = "cuda".to_string();
        assert!(matches!(
            bind_real_family_evidence(diagnostic_receipt(), &wrong_provider, &traces()),
            Err(ShortAudioReceiptError::RealFamilyEvidenceIncomplete { .. })
        ));

        let mut aliased_quant = binding();
        aliased_quant.quant = "q4".to_string();
        let error = bind_real_family_evidence(diagnostic_receipt(), &aliased_quant, &traces())
            .expect_err("alias quant must not bind as canonical pack identity");
        assert!(
            error
                .to_string()
                .contains("funasr-nano:q4_k / q4_k vs binding funasr-nano:q4 / q4")
        );
    }
}
