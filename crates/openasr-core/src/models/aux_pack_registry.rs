//! Single dispatch point for auxiliary (non-ASR) runtime pack contracts.
//!
//! ASR families are looked up through one data-driven table --
//! [`crate::arch::OpenAsrArchitectureRegistry`] -- keyed by `general.architecture`
//! and cross-checked (`openasr.model.family` / audio-frontend / decode-policy /
//! tokenizer) before an adapter is selected. Auxiliary packs (speaker
//! embedder, speaker segmenter, translation, punctuation, forced alignment)
//! are not ASR transcription architectures -- they have no audio frontend or
//! decode policy in that sense -- so forcing them into
//! `OpenAsrArchitectureDescriptor` would model a shape they don't have (see
//! `models::pyannote` module docs, which already say so explicitly). They still
//! deserve **one** table instead of an ad hoc chain of `if let Some(...)` calls
//! in `api::backend::native`, so this module is that table: one
//! `general.architecture` value per aux family, matched by a single lookup,
//! fail-closed (`None` when no aux entry matches, so the caller falls through
//! to ASR adapter selection, which then fails closed on its own if the pack
//! matches nothing at all).
//!
//! [`aux_pack_architecture_ids_are_unique_and_disjoint_from_asr`] is the safety
//! net a hand-rolled chain never had: it fails the test suite if a future aux
//! family ever reuses a `general.architecture` value already claimed by an ASR
//! descriptor (which would otherwise silently shadow one or the other,
//! depending on chain order, instead of raising `Ambiguous`).

use std::path::Path;

use crate::arch::GENERAL_ARCHITECTURE_KEY;
use crate::device::{
    execution_policy::{AcceleratedPlacementCapabilities, ExecutionCapabilities},
    execution_route::ExecutionProvider,
};
use crate::ggml_runtime::AutoGpuPolicy;
use crate::{GgufMetadata, GgufTensorIndex};

/// Runtime placement contract for a non-ASR model stage.
///
/// Auxiliary packs deliberately do not masquerade as ASR architecture
/// descriptors, but their execution placement is still mandatory data. This
/// keeps request-level hardware targets truthful across post-processing and
/// speaker attribution instead of letting each auxiliary caller rediscover a
/// backend from environment defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuxiliaryExecutionPolicy {
    /// The stage follows the request intent using these provider/placement
    /// rows and its own Auto default.
    RequestScoped {
        capabilities: ExecutionCapabilities,
        auto_gpu_policy: AutoGpuPolicy,
    },
}

const AUX_CPU_AND_FULL_DEVICE_EXECUTION: ExecutionCapabilities = ExecutionCapabilities::new(true)
    .with_provider(
        ExecutionProvider::Metal,
        AcceleratedPlacementCapabilities::FULL_DEVICE,
    )
    .with_provider(
        ExecutionProvider::Cuda,
        AcceleratedPlacementCapabilities::FULL_DEVICE,
    )
    .with_provider(
        ExecutionProvider::Hip,
        AcceleratedPlacementCapabilities::FULL_DEVICE,
    )
    .with_provider(
        ExecutionProvider::Vulkan,
        AcceleratedPlacementCapabilities::FULL_DEVICE,
    );

const AUX_CPU_METAL_CUDA_VULKAN_FULL_DEVICE_EXECUTION: ExecutionCapabilities =
    ExecutionCapabilities::new(true)
        .with_provider(
            ExecutionProvider::Metal,
            AcceleratedPlacementCapabilities::FULL_DEVICE,
        )
        .with_provider(
            ExecutionProvider::Cuda,
            AcceleratedPlacementCapabilities::FULL_DEVICE,
        )
        .with_provider(
            ExecutionProvider::Vulkan,
            AcceleratedPlacementCapabilities::FULL_DEVICE,
        );

const AUX_CPU_METAL_FULL_DEVICE_CUDA_VULKAN_HYBRID_EXECUTION: ExecutionCapabilities =
    ExecutionCapabilities::new(true)
        .with_provider(
            ExecutionProvider::Metal,
            AcceleratedPlacementCapabilities::FULL_DEVICE,
        )
        .with_provider(
            ExecutionProvider::Cuda,
            AcceleratedPlacementCapabilities::HYBRID,
        )
        .with_provider(
            ExecutionProvider::Vulkan,
            AcceleratedPlacementCapabilities::HYBRID,
        );

/// Which pull-time error prefix a matched aux family reports, preserving the
/// exact wording `api::backend::native`'s tests assert on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuxPackKind {
    /// Speaker embedder (ReDimNet2-B6) / speaker segmenter (pyannote) diarization
    /// support packs.
    Diarization,
    /// Punctuation-restoration packs (FireRedPunc).
    Punctuation,
    /// Forced-alignment word-timestamp refiner packs (Qwen3-ForcedAligner).
    ForcedAlignment,
}

/// Persistent-state ownership contract for one auxiliary family. Every new
/// descriptor must make this choice explicitly; a validation-only registry
/// entry can no longer silently grow a process singleton later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuxiliaryRuntimeOwnership {
    /// CPU uses a Send-safe host owner while accelerated routes own a `!Send`
    /// backend runtime on a dedicated actor thread.
    AdmittedHostOrPinnedActor,
    /// `!Send` backend runtime held and destroyed on a dedicated actor thread.
    AdmittedPinnedActor,
    /// Fresh per invocation; no state survives the stage boundary.
    InvocationTransient,
}

impl AuxiliaryRuntimeOwnership {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::AdmittedHostOrPinnedActor => "admitted-host-or-pinned-actor",
            Self::AdmittedPinnedActor => "admitted-pinned-actor",
            Self::InvocationTransient => "invocation-transient",
        }
    }
}

impl AuxPackKind {
    /// The `"<label> failed: <error>"` prefix `verify_native_runtime_model_pack_path`
    /// reports for this kind (unchanged from the pre-consolidation call sites).
    pub(crate) fn validation_failure_label(self) -> &'static str {
        match self {
            AuxPackKind::Diarization => "diarization pack validation failed",
            AuxPackKind::Punctuation => "punctuation pack validation failed",
            AuxPackKind::ForcedAlignment => "forced-alignment pack validation failed",
        }
    }
}

struct AuxPackDescriptor {
    /// `general.architecture` value that identifies this aux family's packs.
    architecture_id: &'static str,
    /// Stable join key used by catalog authoring and publish receipts. This is
    /// deliberately data on the canonical Rust route descriptor, not a Python
    /// architecture-name table.
    catalog_family_id: &'static str,
    kind: AuxPackKind,
    execution_policy: AuxiliaryExecutionPolicy,
    ownership: AuxiliaryRuntimeOwnership,
    quantization_classification: crate::models::pack_quant::TensorQuantizationContract,
    /// Cheap pull-time contract probe: constructs/parses just enough of the
    /// pack to prove the runtime loader can build from it, without
    /// materializing full weights for execution.
    validate: fn(&Path, &GgufMetadata, &GgufTensorIndex) -> Result<(), String>,
}

fn validate_pyannote(
    _path: &Path,
    _metadata: &GgufMetadata,
    tensor_index: &GgufTensorIndex,
) -> Result<(), String> {
    crate::diarize::segment::PyannoteSegmenter::quoted_persistent_host_commitment_bytes(
        tensor_index,
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

fn validate_diarizen(
    _path: &Path,
    metadata: &GgufMetadata,
    tensor_index: &GgufTensorIndex,
) -> Result<(), String> {
    crate::diarize::segment::DiariZenSegmenter::probe_preflight_parts(metadata, tensor_index)
        .map_err(|error| error.to_string())
}

fn validate_redimnet2(
    _path: &Path,
    _metadata: &GgufMetadata,
    tensor_index: &GgufTensorIndex,
) -> Result<(), String> {
    crate::diarize::embed::RedimNet2Embedder::quoted_persistent_host_commitment_bytes(tensor_index)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn validate_firered_punc(
    _path: &Path,
    metadata: &GgufMetadata,
    _tensor_index: &GgufTensorIndex,
) -> Result<(), String> {
    crate::models::firered_punc::runtime_contract::parse_and_validate_firered_punc_metadata(
        metadata,
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

fn validate_forced_aligner(
    _path: &Path,
    metadata: &GgufMetadata,
    tensor_index: &GgufTensorIndex,
) -> Result<(), String> {
    crate::models::qwen::validate_forced_aligner_runtime_pack_contract(metadata)
        .and_then(|()| {
            crate::models::qwen::validate_forced_aligner_quantization_contract(
                metadata,
                tensor_index,
            )
        })
        .map_err(|error| error.to_string())
}

/// `general.architecture` value ReDimNet2-B6 speaker-embedder packs carry.
/// No dedicated `models::redimnet2` module owns this constant (the model
/// forward pass lives in `crate::diarize::embed::redimnet`, and packaging is
/// the family-agnostic `models::diarize_pack_import`), so this aux registry
/// -- the only production reader of the string -- is its home. Referenced by
/// `models::pack_quant_audit` too, so both stay in sync.
pub(crate) const REDIMNET2_GGML_ARCHITECTURE_ID: &str = "redimnet2";

const AUX_PACK_DESCRIPTORS: &[AuxPackDescriptor] = &[
    AuxPackDescriptor {
        architecture_id: REDIMNET2_GGML_ARCHITECTURE_ID,
        catalog_family_id: "redimnet2",
        kind: AuxPackKind::Diarization,
        // The resident ggml graph has validated full-device CUDA, Vulkan, and
        // Metal paths. Auto may use CUDA/Vulkan; Metal remains explicit-only
        // until current-pack parity/performance/RSS gates promote it.
        execution_policy: AuxiliaryExecutionPolicy::RequestScoped {
            capabilities: AUX_CPU_METAL_CUDA_VULKAN_FULL_DEVICE_EXECUTION,
            auto_gpu_policy: AutoGpuPolicy::ExceptMetal,
        },
        ownership: AuxiliaryRuntimeOwnership::AdmittedPinnedActor,
        quantization_classification:
            crate::models::pack_quant::TensorQuantizationContract::EntireAcousticPack {
                model_architecture: REDIMNET2_GGML_ARCHITECTURE_ID,
            },
        validate: validate_redimnet2,
    },
    AuxPackDescriptor {
        architecture_id: crate::models::pyannote::PYANNOTE_GGML_ARCHITECTURE_ID,
        catalog_family_id: "pyannote-segmentation",
        kind: AuxPackKind::Diarization,
        // Metal owns the complete device graph. CUDA and Vulkan use the
        // numerically verified host SincNet frontend plus a direct-device
        // recurrent/classifier graph. Auto may use CUDA/Vulkan; Metal remains
        // explicit until its product-level latency evidence is promoted.
        execution_policy: AuxiliaryExecutionPolicy::RequestScoped {
            capabilities: AUX_CPU_METAL_FULL_DEVICE_CUDA_VULKAN_HYBRID_EXECUTION,
            auto_gpu_policy: AutoGpuPolicy::ExceptMetal,
        },
        ownership: AuxiliaryRuntimeOwnership::AdmittedHostOrPinnedActor,
        quantization_classification:
            crate::models::pack_quant::TensorQuantizationContract::NotApplicable {
                model_architecture: crate::models::pyannote::PYANNOTE_GGML_ARCHITECTURE_ID,
                reason: "speaker segmentation has no audio-encoder quantization tier",
            },
        validate: validate_pyannote,
    },
    AuxPackDescriptor {
        architecture_id: crate::diarize::segment::DIARIZEN_GGML_ARCHITECTURE_ID,
        catalog_family_id: "diarizen-segmentation",
        kind: AuxPackKind::Diarization,
        execution_policy: AuxiliaryExecutionPolicy::RequestScoped {
            capabilities: AUX_CPU_AND_FULL_DEVICE_EXECUTION,
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
        },
        ownership: AuxiliaryRuntimeOwnership::AdmittedPinnedActor,
        quantization_classification:
            crate::models::pack_quant::TensorQuantizationContract::NotApplicable {
                model_architecture: crate::diarize::segment::DIARIZEN_GGML_ARCHITECTURE_ID,
                reason: "speaker segmentation has no audio-encoder quantization tier",
            },
        validate: validate_diarizen,
    },
    AuxPackDescriptor {
        architecture_id: crate::models::firered_punc::config::FIRERED_PUNC_ARCHITECTURE_VALUE,
        catalog_family_id: "firered-punc",
        kind: AuxPackKind::Punctuation,
        execution_policy: AuxiliaryExecutionPolicy::RequestScoped {
            capabilities: AUX_CPU_AND_FULL_DEVICE_EXECUTION,
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
        },
        ownership: AuxiliaryRuntimeOwnership::AdmittedPinnedActor,
        quantization_classification:
            crate::models::pack_quant::TensorQuantizationContract::NotApplicable {
                model_architecture:
                    crate::models::firered_punc::config::FIRERED_PUNC_ARCHITECTURE_VALUE,
                reason: "punctuation restoration has no acoustic encoder",
            },
        validate: validate_firered_punc,
    },
    AuxPackDescriptor {
        architecture_id: crate::models::qwen::QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID,
        catalog_family_id: "qwen3-forced-aligner",
        kind: AuxPackKind::ForcedAlignment,
        // Every published pack satisfies the runtime's semantic mixed-precision
        // floor: precision-sensitive matrices stay Q8_0 or higher while the
        // policy-guarded q4_k tier may quantize eligible decoder matrices.
        // Metal retains its validated precise FullDevice graph. CUDA and Vulkan
        // share a Hybrid topology that runs the audio encoder without flash
        // attention on the selected GPU while retaining numerically sensitive
        // decoder/logits state on CPU. HIP remains fail-closed until the same
        // timestamp and performance gates have passed there. Auto stays off
        // Metal because its near-tie timestamp drift still exceeds the envelope.
        execution_policy: AuxiliaryExecutionPolicy::RequestScoped {
            capabilities: AUX_CPU_METAL_FULL_DEVICE_CUDA_VULKAN_HYBRID_EXECUTION,
            auto_gpu_policy: AutoGpuPolicy::ExceptMetal,
        },
        ownership: AuxiliaryRuntimeOwnership::InvocationTransient,
        quantization_classification:
            crate::models::pack_quant::TensorQuantizationContract::SemanticRolesV1 {
                model_architecture: crate::models::qwen::QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID,
                classify: crate::models::qwen::forced_aligner_tensor_role,
                quantized_axis: crate::models::pack_quant::QuantizedAxis::First,
            },
        validate: validate_forced_aligner,
    },
];

/// Execution contract for one known auxiliary architecture.
pub(crate) fn auxiliary_execution_policy(
    architecture_id: &str,
) -> Option<AuxiliaryExecutionPolicy> {
    AUX_PACK_DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.architecture_id == architecture_id)
        .map(|descriptor| descriptor.execution_policy)
}

pub(crate) fn auxiliary_runtime_ownership(
    architecture_id: &str,
) -> Option<AuxiliaryRuntimeOwnership> {
    AUX_PACK_DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.architecture_id == architecture_id)
        .map(|descriptor| descriptor.ownership)
}

pub(crate) fn auxiliary_catalog_family_id(architecture_id: &str) -> Option<&'static str> {
    AUX_PACK_DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.architecture_id == architecture_id)
        .map(|descriptor| descriptor.catalog_family_id)
}

pub(crate) fn auxiliary_quantization_classification(
    architecture_id: &str,
) -> Option<crate::models::pack_quant::TensorQuantizationContract> {
    AUX_PACK_DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.architecture_id == architecture_id)
        .map(|descriptor| descriptor.quantization_classification)
}

/// Every aux family's `general.architecture` id. Lets a caller that needs the
/// full non-ASR family list (e.g. `models::pack_quant_audit`'s quant-floor
/// coverage test) enumerate it without depending on `AuxPackKind` or
/// `validate_aux_runtime_pack_contract`'s metadata-driven dispatch.
/// Test-only today (no non-test caller needs the full list yet).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn aux_pack_architecture_ids() -> impl Iterator<Item = &'static str> {
    AUX_PACK_DESCRIPTORS
        .iter()
        .map(|descriptor| descriptor.architecture_id)
}

/// Pull-time contract dispatch for auxiliary (non-ASR) runtime packs.
///
/// Returns `None` when `metadata` does not declare one of the known aux
/// `general.architecture` values, so the caller (`verify_native_runtime_model_pack_path`)
/// falls through to ASR family-adapter selection -- which then fails closed on
/// its own for a pack that matches neither table. Returns `Some((kind,
/// result))` when an aux family claims the pack, `result` being that family's
/// cheap runtime-loader probe (no weight materialization).
pub(crate) fn validate_aux_runtime_pack_contract(
    path: &Path,
    metadata: &GgufMetadata,
    tensor_index: &GgufTensorIndex,
) -> Option<(AuxPackKind, Result<(), String>)> {
    let architecture = metadata.get_string(GENERAL_ARCHITECTURE_KEY)?.trim();
    let descriptor = AUX_PACK_DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.architecture_id == architecture)?;
    Some((
        descriptor.kind,
        (descriptor.validate)(path, metadata, tensor_index),
    ))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::arch::OpenAsrArchitectureRegistry;

    fn empty_tensor_index() -> GgufTensorIndex {
        GgufTensorIndex::empty_for_test(PathBuf::from("/nonexistent"))
    }

    fn forced_aligner_tensor_index(ggml_type: i32) -> GgufTensorIndex {
        GgufTensorIndex::from_snapshot(crate::ggml_runtime::GgufTensorIndexSnapshot {
            path: PathBuf::from("/nonexistent/forced-aligner-policy.oasr"),
            data_section_offset_bytes: 0,
            tensors: vec![crate::ggml_runtime::GgufTensorMetadata {
                name: "blk.0.ffn_gate.weight".to_string(),
                dims: vec![1024, 3072],
                ggml_type,
                type_name: "synthetic".to_string(),
                size_bytes: 0,
                offset_bytes: 0,
            }],
        })
        .expect("valid forced-aligner tensor index")
    }

    /// Fail-closed safety net the previous hand-rolled `if let Some(...)` chain
    /// in `api::backend::native` never had: every aux `general.architecture`
    /// value must be unique among aux families AND disjoint from every ASR
    /// `OpenAsrArchitectureDescriptor::model_architecture`. A collision would
    /// otherwise be resolved by chain/table iteration order instead of an
    /// explicit `Ambiguous` error -- exactly the silent-shadowing failure mode
    /// The canonical architecture registry refuses to allow within the ASR
    /// table.
    #[test]
    fn aux_pack_architecture_ids_are_unique_and_disjoint_from_asr() {
        let mut seen: Vec<&'static str> = Vec::new();
        for descriptor in AUX_PACK_DESCRIPTORS {
            assert!(
                !seen.contains(&descriptor.architecture_id),
                "duplicate aux architecture id: {}",
                descriptor.architecture_id
            );
            seen.push(descriptor.architecture_id);
        }

        let asr_registry = OpenAsrArchitectureRegistry::with_builtins();
        for descriptor in AUX_PACK_DESCRIPTORS {
            assert!(
                asr_registry
                    .find_by_model_architecture(descriptor.architecture_id)
                    .is_none(),
                "aux architecture id '{}' collides with a registered ASR architecture",
                descriptor.architecture_id
            );
        }
    }

    #[test]
    fn forced_aligner_uses_validated_provider_topologies_while_auto_avoids_metal() {
        let policy = auxiliary_execution_policy(
            crate::models::qwen::QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID,
        );
        let Some(AuxiliaryExecutionPolicy::RequestScoped {
            capabilities,
            auto_gpu_policy,
        }) = policy
        else {
            panic!("forced aligner must remain request-scoped");
        };
        assert_eq!(auto_gpu_policy, AutoGpuPolicy::ExceptMetal);
        assert!(capabilities.supports_cpu());
        assert!(capabilities.supports(
            ExecutionProvider::Metal,
            crate::device::execution_policy::ExecutionPlacement::FullDevice,
        ));
        assert!(!capabilities.supports(
            ExecutionProvider::Metal,
            crate::device::execution_policy::ExecutionPlacement::Hybrid,
        ));
        for provider in [ExecutionProvider::Cuda, ExecutionProvider::Vulkan] {
            assert!(capabilities.supports(
                provider,
                crate::device::execution_policy::ExecutionPlacement::Hybrid,
            ));
            assert!(!capabilities.supports(
                provider,
                crate::device::execution_policy::ExecutionPlacement::FullDevice,
            ));
        }
        assert!(!capabilities.supports(
            ExecutionProvider::Hip,
            crate::device::execution_policy::ExecutionPlacement::FullDevice,
        ));
    }

    #[test]
    fn redimnet_auto_uses_cuda_vulkan_while_metal_remains_explicit() {
        let policy = auxiliary_execution_policy(REDIMNET2_GGML_ARCHITECTURE_ID);
        let Some(AuxiliaryExecutionPolicy::RequestScoped {
            capabilities,
            auto_gpu_policy,
        }) = policy
        else {
            panic!("ReDimNet must be request-scoped");
        };
        assert_eq!(auto_gpu_policy, AutoGpuPolicy::ExceptMetal);
        assert!(capabilities.supports_cpu());
        for provider in [
            ExecutionProvider::Metal,
            ExecutionProvider::Cuda,
            ExecutionProvider::Vulkan,
        ] {
            assert!(capabilities.supports(
                provider,
                crate::device::execution_policy::ExecutionPlacement::FullDevice,
            ));
            assert!(!capabilities.supports(
                provider,
                crate::device::execution_policy::ExecutionPlacement::Hybrid,
            ));
        }
        assert!(!capabilities.supports(
            ExecutionProvider::Hip,
            crate::device::execution_policy::ExecutionPlacement::FullDevice,
        ));
    }

    #[test]
    fn pyannote_auto_uses_cuda_vulkan_while_metal_remains_explicit() {
        let architecture = crate::models::pyannote::PYANNOTE_GGML_ARCHITECTURE_ID;
        let policy = auxiliary_execution_policy(architecture);
        let Some(AuxiliaryExecutionPolicy::RequestScoped {
            capabilities,
            auto_gpu_policy,
        }) = policy
        else {
            panic!("PyanNet must be request-scoped");
        };
        assert_eq!(auto_gpu_policy, AutoGpuPolicy::ExceptMetal);
        assert!(capabilities.supports_cpu());
        assert!(capabilities.supports(
            ExecutionProvider::Metal,
            crate::device::execution_policy::ExecutionPlacement::FullDevice,
        ));
        assert!(!capabilities.supports(
            ExecutionProvider::Metal,
            crate::device::execution_policy::ExecutionPlacement::Hybrid,
        ));
        for provider in [ExecutionProvider::Cuda, ExecutionProvider::Vulkan] {
            assert!(!capabilities.supports(
                provider,
                crate::device::execution_policy::ExecutionPlacement::FullDevice,
            ));
            assert!(capabilities.supports(
                provider,
                crate::device::execution_policy::ExecutionPlacement::Hybrid,
            ));
        }
        assert!(!capabilities.supports(
            ExecutionProvider::Hip,
            crate::device::execution_policy::ExecutionPlacement::FullDevice,
        ));
        assert_eq!(
            auxiliary_runtime_ownership(architecture),
            Some(AuxiliaryRuntimeOwnership::AdmittedHostOrPinnedActor),
        );
    }

    #[test]
    fn only_validated_auxiliary_topologies_accept_cuda_vulkan_hybrid_compute() {
        for descriptor in AUX_PACK_DESCRIPTORS {
            let AuxiliaryExecutionPolicy::RequestScoped { capabilities, .. } =
                descriptor.execution_policy;
            for provider in [
                ExecutionProvider::Metal,
                ExecutionProvider::Cuda,
                ExecutionProvider::Hip,
                ExecutionProvider::Vulkan,
            ] {
                let expected = matches!(
                    descriptor.architecture_id,
                    crate::models::pyannote::PYANNOTE_GGML_ARCHITECTURE_ID
                        | crate::models::qwen::QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID
                ) && matches!(
                    provider,
                    ExecutionProvider::Cuda | ExecutionProvider::Vulkan
                );
                assert_eq!(
                    capabilities.supports(
                        provider,
                        crate::device::execution_policy::ExecutionPlacement::Hybrid,
                    ),
                    expected,
                    "unexpected Hybrid capability for auxiliary architecture '{}' under {provider}",
                    descriptor.architecture_id,
                );
            }
        }
    }

    #[test]
    fn every_auxiliary_family_has_an_explicit_runtime_ownership_contract() {
        for descriptor in AUX_PACK_DESCRIPTORS {
            assert_eq!(
                auxiliary_runtime_ownership(descriptor.architecture_id),
                Some(descriptor.ownership),
                "auxiliary family '{}' lost its ownership contract",
                descriptor.architecture_id,
            );
        }
    }

    #[test]
    fn auxiliary_catalog_family_ids_are_nonempty_and_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for descriptor in AUX_PACK_DESCRIPTORS {
            assert!(
                !descriptor.catalog_family_id.trim().is_empty(),
                "auxiliary architecture '{}' has an empty catalog family id",
                descriptor.architecture_id,
            );
            assert!(
                seen.insert(descriptor.catalog_family_id),
                "duplicate auxiliary catalog family id: {}",
                descriptor.catalog_family_id,
            );
            assert_eq!(
                auxiliary_catalog_family_id(descriptor.architecture_id),
                Some(descriptor.catalog_family_id),
            );
        }
    }

    #[test]
    fn auxiliary_runtime_families_match_published_capability_families() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tooling/publish-model/models-core.toml");
        let source = std::fs::read_to_string(&path).expect("read models-core.toml");
        let catalog: toml::Value = toml::from_str(&source).expect("parse models-core.toml");
        let published = catalog
            .as_table()
            .expect("models-core.toml top-level table")
            .values()
            .filter_map(toml::Value::as_table)
            .filter(|entry| {
                entry.get("kind").and_then(toml::Value::as_str) == Some("capability-pack")
            })
            .map(|entry| {
                entry
                    .get("family")
                    .and_then(toml::Value::as_str)
                    .expect("capability-pack family")
            })
            .collect::<std::collections::BTreeSet<_>>();
        let runtime = AUX_PACK_DESCRIPTORS
            .iter()
            .map(|descriptor| descriptor.catalog_family_id)
            .collect::<std::collections::BTreeSet<_>>();

        assert_eq!(
            runtime, published,
            "aux runtime and publish catalog capability-family IDs drifted apart",
        );
    }

    #[test]
    fn every_auxiliary_quantization_contract_is_owned_by_its_descriptor() {
        for descriptor in AUX_PACK_DESCRIPTORS {
            let quantization = descriptor.quantization_classification;
            assert_eq!(
                quantization.model_architecture(),
                descriptor.architecture_id,
                "auxiliary family '{}' has a quantization contract owned by '{}', not itself",
                descriptor.architecture_id,
                quantization.model_architecture(),
            );
            if let crate::models::pack_quant::TensorQuantizationContract::NotApplicable {
                reason,
                ..
            } = quantization
            {
                assert!(!reason.trim().is_empty());
            }
        }
    }

    #[test]
    fn dispatch_returns_none_for_unknown_architecture() {
        let mut values = std::collections::BTreeMap::new();
        values.insert(
            GENERAL_ARCHITECTURE_KEY.to_string(),
            crate::ggml_runtime::GgufMetadataValue::String("totally-unknown-arch".to_string()),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        assert!(
            validate_aux_runtime_pack_contract(
                Path::new("/nonexistent"),
                &metadata,
                &empty_tensor_index(),
            )
            .is_none()
        );
    }

    /// A complete, minimal set of `qwen3_forced_aligner.*` + tokenizer keys --
    /// mirrors exactly what a real published pack carries (verified against
    /// the rebuilt `qwen3-forced-aligner-0.6b` packs' shared GGUF header:
    /// no `openasr.audio.frontend` / `openasr.decode.policy`, only these).
    fn valid_forced_aligner_metadata() -> GgufMetadata {
        use crate::ggml_runtime::GgufMetadataValue;
        let mut values = std::collections::BTreeMap::new();
        values.insert(
            GENERAL_ARCHITECTURE_KEY.to_string(),
            GgufMetadataValue::String(
                crate::models::qwen::QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID.to_string(),
            ),
        );
        values.insert(
            "openasr.model.id".to_string(),
            GgufMetadataValue::String("qwen3-forced-aligner-0.6b".to_string()),
        );
        for key in [
            "qwen3_forced_aligner.audio.sample_rate_hz",
            "qwen3_forced_aligner.audio.n_mels",
            "qwen3_forced_aligner.audio.n_fft",
            "qwen3_forced_aligner.audio.win_length",
            "qwen3_forced_aligner.audio.hop_length",
            "qwen3_forced_aligner.audio.n_layers",
            "qwen3_forced_aligner.audio.d_model",
            "qwen3_forced_aligner.audio.n_heads",
            "qwen3_forced_aligner.llm.n_layers",
            "qwen3_forced_aligner.llm.d_model",
            "qwen3_forced_aligner.llm.n_heads",
            "qwen3_forced_aligner.llm.n_kv_heads",
            "qwen3_forced_aligner.llm.head_dim",
            "qwen3_forced_aligner.llm.embed_vocab_size",
            "qwen3_forced_aligner.llm.classify_num",
            "qwen3_forced_aligner.llm.max_positions",
            "qwen3_forced_aligner.audio_start_token_id",
            "qwen3_forced_aligner.audio_end_token_id",
            "qwen3_forced_aligner.audio_pad_token_id",
            "qwen3_forced_aligner.timestamp_token_id",
            "qwen3_forced_aligner.timestamp_segment_time_ms",
        ] {
            values.insert(key.to_string(), GgufMetadataValue::U32(1));
        }
        values.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::StringArray(vec!["<pad>".to_string()]),
        );
        values.insert(
            "tokenizer.ggml.merges".to_string(),
            GgufMetadataValue::StringArray(Vec::new()),
        );
        GgufMetadata::from_values_for_test(values)
    }

    /// Positive direction: a forced-aligner pack that carries every metadata
    /// key the runtime loader needs is routed to the aux table and accepted,
    /// never rejected by ASR runtime adapter selection (which would happen if
    /// this architecture were not registered here -- see
    /// `native.rs::pull_contract_validation_routes_diarize_packs_to_their_loader`
    /// for the same shape of proof on the diarization aux kind).
    #[test]
    fn forced_aligner_pack_with_complete_metadata_is_accepted() {
        let metadata = valid_forced_aligner_metadata();
        let (kind, result) = validate_aux_runtime_pack_contract(
            Path::new("/nonexistent"),
            &metadata,
            &empty_tensor_index(),
        )
        .expect("forced-aligner architecture must be claimed by the aux table");
        assert_eq!(kind, AuxPackKind::ForcedAlignment);
        assert!(result.is_ok(), "got: {result:?}");
    }

    #[test]
    fn forced_aligner_pack_accepts_policy_guarded_q4_k_and_rejects_q3_decoder_weights() {
        let metadata = valid_forced_aligner_metadata();
        let q8 = crate::models::pack_quant_audit::GGML_TYPE_Q8_0 as i32;
        let q4 = crate::models::pack_quant_audit::GGML_TYPE_Q4_K as i32;
        let q3 = crate::models::pack_quant_audit::GGML_TYPE_Q3_K as i32;
        let (_, q8_result) = validate_aux_runtime_pack_contract(
            Path::new("/nonexistent"),
            &metadata,
            &forced_aligner_tensor_index(q8),
        )
        .expect("forced-aligner architecture must be claimed");
        q8_result.expect("Q8 decoder weight must satisfy the pack contract");

        let mut q4_values = metadata.values().clone();
        q4_values.insert(
            "openasr.model.id".to_string(),
            crate::ggml_runtime::GgufMetadataValue::String(
                "qwen3-forced-aligner-0.6b:q4_k".to_string(),
            ),
        );
        let q4_metadata = GgufMetadata::from_values_for_test(q4_values);
        let (_, q4_result) = validate_aux_runtime_pack_contract(
            Path::new("/nonexistent"),
            &q4_metadata,
            &forced_aligner_tensor_index(q4),
        )
        .expect("forced-aligner architecture must be claimed");
        q4_result.expect("Q4_K decoder weight must satisfy the q4_k contract");

        let (_, q3_result) = validate_aux_runtime_pack_contract(
            Path::new("/nonexistent"),
            &q4_metadata,
            &forced_aligner_tensor_index(q3),
        )
        .expect("forced-aligner architecture must be claimed");
        let error = q3_result.expect_err("Q3 decoder weight must fail closed");
        assert!(error.contains("require q4_k"), "got: {error}");
    }

    /// Negative direction: a forced-aligner pack missing a required
    /// `qwen3_forced_aligner.*` key must still be claimed by the aux table
    /// (so it is never silently accepted by ASR adapter selection instead)
    /// but must fail validation -- the actual bug this module closes: before
    /// this architecture was registered, a real published pack (which has
    /// never carried `openasr.audio.frontend`) fell through the aux table
    /// entirely and was rejected by unrelated ASR-adapter-selection metadata
    /// requirements instead of this family's own contract.
    #[test]
    fn forced_aligner_pack_missing_required_metadata_is_rejected() {
        let mut values = valid_forced_aligner_metadata().values().clone();
        values.remove("qwen3_forced_aligner.llm.classify_num");
        let metadata = GgufMetadata::from_values_for_test(values);

        let (kind, result) = validate_aux_runtime_pack_contract(
            Path::new("/nonexistent"),
            &metadata,
            &empty_tensor_index(),
        )
        .expect("forced-aligner architecture must still be claimed by the aux table");
        assert_eq!(kind, AuxPackKind::ForcedAlignment);
        let error = result.expect_err("pack missing a required metadata key must be rejected");
        assert!(
            error.contains("qwen3_forced_aligner.llm.classify_num"),
            "got: {error}"
        );

        // Also missing a tokenizer array (present in every real pack but not
        // covered by `parse_forced_aligner_runtime_metadata`'s scalar keys)
        // must independently fail closed.
        let mut values_no_tokens = valid_forced_aligner_metadata().values().clone();
        values_no_tokens.remove("tokenizer.ggml.tokens");
        let metadata_no_tokens = GgufMetadata::from_values_for_test(values_no_tokens);
        let (_, result_no_tokens) = validate_aux_runtime_pack_contract(
            Path::new("/nonexistent"),
            &metadata_no_tokens,
            &empty_tensor_index(),
        )
        .expect("forced-aligner architecture must still be claimed by the aux table");
        let error_no_tokens =
            result_no_tokens.expect_err("pack missing the BPE tokenizer array must be rejected");
        assert!(
            error_no_tokens.contains("tokenizer.ggml.tokens"),
            "got: {error_no_tokens}"
        );
    }
}
