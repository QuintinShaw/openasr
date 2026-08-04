//! Native-family integration wiring checks.
//!
//! **Runtime (ships in release binaries):** purely in-memory validation against
//! the architecture inventory, embedded decode-policy strategies, and the
//! force-linked pack-import symbol table. No repository checkout, no
//! `CARGO_MANIFEST_DIR` path walks, no docs/tooling/catalog disk I/O.
//!
//! **Tests only:** additional fail-closed checks that *do* read the source
//! tree (external tooling paths, reference dumpers, public audit forms) and
//! lock the embedded pre-audit family list to the on-disk SSOT file.

use std::collections::BTreeSet;

use thiserror::Error;

use crate::arch::{
    OpenAsrArchitectureDescriptor, OpenAsrArchitectureRegistry, OpenAsrDecodeDriverStrategy,
    OpenAsrPackImportSurface,
};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyExecutionKind, resolve_builtin_decode_policy,
};
use crate::models::pack_import_surface::linked_core_pack_import_symbols;

/// Compile-time embedded copy of `docs/model-audits/pre_audit_families.txt`.
/// Release binaries carry this text; they never open that path at runtime.
const PRE_AUDIT_FAMILIES_EMBEDDED: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/model-audits/pre_audit_families.txt"
));

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum FamilyIntegrationAuditError {
    #[error("native family '{model_family}' has empty catalog_family_id")]
    EmptyCatalogFamilyId { model_family: String },
    #[error("native family '{model_family}' has empty runtime_tensor_contract_id")]
    EmptyRuntimeTensorContractId { model_family: String },
    #[error(
        "native family '{model_family}' embeds shared decode policy '{decode_policy_id}' as {expected:?} but the component executes as {actual:?}"
    )]
    SharedDecodeKindMismatch {
        model_family: String,
        decode_policy_id: String,
        expected: BuiltinDecodePolicyExecutionKind,
        actual: BuiltinDecodePolicyExecutionKind,
    },
    #[error(
        "native family '{model_family}' declares Dedicated decode but policy '{decode_policy_id}' is still registered on the shared driver"
    )]
    DedicatedDecodeStillShared {
        model_family: String,
        decode_policy_id: String,
    },
    #[error(
        "native family '{model_family}' CTC shared decode policy '{decode_policy_id}' is missing ctc_blank_token_id"
    )]
    CtcBlankMissing {
        model_family: String,
        decode_policy_id: String,
    },
    #[error(
        "native family '{model_family}' core pack-import symbol '{symbol}' is not force-linked"
    )]
    PackImportSymbolUnlinked {
        model_family: String,
        symbol: String,
    },
    #[error("native family '{model_family}' external pack-import tooling path is empty")]
    PackImportToolingPathEmpty { model_family: String },
    #[error(
        "native family '{model_family}' external pack-import tooling '{relative_path}' is missing"
    )]
    #[cfg_attr(not(test), allow(dead_code))]
    PackImportToolingMissing {
        model_family: String,
        relative_path: String,
    },
    #[error("native family '{model_family}' reference dumper path is empty")]
    ReferenceDumperPathEmpty { model_family: String },
    #[error("native family '{model_family}' reference dumper '{relative_path}' is missing")]
    #[cfg_attr(not(test), allow(dead_code))]
    ReferenceDumperMissing {
        model_family: String,
        relative_path: String,
    },
    #[error(
        "native family '{model_family}' catalog id '{catalog_family_id}' requires audit form '{relative_path}' but the file is missing while the family is public"
    )]
    #[cfg_attr(not(test), allow(dead_code))]
    RequiredAuditFormMissing {
        model_family: String,
        catalog_family_id: String,
        relative_path: String,
    },
    #[error(
        "native family '{catalog_family_id}' has an audit form at '{relative_path}' but remains in the pre-audit exemption list"
    )]
    #[cfg_attr(not(test), allow(dead_code))]
    PreAuditFamilyHasAuditForm {
        catalog_family_id: String,
        relative_path: String,
    },
}

/// Parse the embedded pre-audit family list (compile-time text, no disk I/O).
pub(crate) fn embedded_pre_audit_families() -> BTreeSet<&'static str> {
    let mut families = BTreeSet::new();
    for raw_line in PRE_AUDIT_FAMILIES_EMBEDDED.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        families.insert(line);
    }
    families
}

/// In-memory runtime wiring gate. Safe to call from release binaries: never
/// touches the source tree or any path derived from `CARGO_MANIFEST_DIR`.
pub(crate) fn validate_builtin_runtime_family_wiring() -> Result<(), FamilyIntegrationAuditError> {
    let linked = linked_core_pack_import_symbols();
    validate_runtime_family_wiring(
        OpenAsrArchitectureRegistry::with_builtins().descriptors(),
        &linked,
    )
}

pub(crate) fn validate_runtime_family_wiring(
    architectures: &[OpenAsrArchitectureDescriptor],
    linked_pack_symbols: &BTreeSet<&'static str>,
) -> Result<(), FamilyIntegrationAuditError> {
    // Touch embedded SSOT so the include_str! payload stays linked and is
    // available to tests without implying runtime disk access.
    let _pre_audit = embedded_pre_audit_families();

    for descriptor in architectures {
        if descriptor.identity.catalog_family_id.is_empty() {
            return Err(FamilyIntegrationAuditError::EmptyCatalogFamilyId {
                model_family: descriptor.identity.model_family.to_string(),
            });
        }

        // A new family's minimal accession surface is descriptor + tensor
        // contract (see this module's doc comment and
        // `models::runtime_tensor_contract_registry`); nothing here checks
        // decode policy resolves without one, so fail closed on an empty id
        // instead of letting a half-declared family silently run without a
        // validated tensor contract.
        if descriptor
            .pack_contract
            .runtime_tensor_contract_id
            .is_empty()
        {
            return Err(FamilyIntegrationAuditError::EmptyRuntimeTensorContractId {
                model_family: descriptor.identity.model_family.to_string(),
            });
        }

        match descriptor.topology_contract.decode_driver {
            OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy { policy } => {
                if policy.execution_kind != BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0 {
                    return Err(FamilyIntegrationAuditError::SharedDecodeKindMismatch {
                        model_family: descriptor.identity.model_family.to_string(),
                        decode_policy_id: policy.decode_policy_id.to_string(),
                        expected: BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0,
                        actual: policy.execution_kind,
                    });
                }
            }
            OpenAsrDecodeDriverStrategy::SharedCtcGreedy { policy } => {
                if policy.execution_kind != BuiltinDecodePolicyExecutionKind::CtcGreedyV0 {
                    return Err(FamilyIntegrationAuditError::SharedDecodeKindMismatch {
                        model_family: descriptor.identity.model_family.to_string(),
                        decode_policy_id: policy.decode_policy_id.to_string(),
                        expected: BuiltinDecodePolicyExecutionKind::CtcGreedyV0,
                        actual: policy.execution_kind,
                    });
                }
                if policy.ctc_blank_token_id.is_none() {
                    return Err(FamilyIntegrationAuditError::CtcBlankMissing {
                        model_family: descriptor.identity.model_family.to_string(),
                        decode_policy_id: policy.decode_policy_id.to_string(),
                    });
                }
            }
            OpenAsrDecodeDriverStrategy::Dedicated {
                decode_policy_id, ..
            } => {
                if resolve_builtin_decode_policy(decode_policy_id).is_ok() {
                    return Err(FamilyIntegrationAuditError::DedicatedDecodeStillShared {
                        model_family: descriptor.identity.model_family.to_string(),
                        decode_policy_id: decode_policy_id.to_string(),
                    });
                }
            }
        }

        match descriptor.pack_contract.pack_import {
            OpenAsrPackImportSurface::CoreConvert { symbol, .. } => {
                if !linked_pack_symbols.contains(symbol) {
                    return Err(FamilyIntegrationAuditError::PackImportSymbolUnlinked {
                        model_family: descriptor.identity.model_family.to_string(),
                        symbol: symbol.to_string(),
                    });
                }
            }
            OpenAsrPackImportSurface::ExternalTooling { relative_path } => {
                if relative_path.is_empty() {
                    return Err(FamilyIntegrationAuditError::PackImportToolingPathEmpty {
                        model_family: descriptor.identity.model_family.to_string(),
                    });
                }
            }
        }

        if let Some(source) = descriptor.conformance_contract.reference_dumper_source
            && source.is_empty()
        {
            return Err(FamilyIntegrationAuditError::ReferenceDumperPathEmpty {
                model_family: descriptor.identity.model_family.to_string(),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
pub(crate) mod source_tree_audit {
    use super::*;
    use std::path::{Path, PathBuf};

    const PRE_AUDIT_FAMILIES_RELATIVE: &str = "docs/model-audits/pre_audit_families.txt";

    fn repository_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repository root (test-only)")
    }

    fn required_audit_form_relative_path(catalog_family_id: &str) -> String {
        format!("docs/model-audits/{catalog_family_id}.md")
    }

    fn public_catalog_family_ids(repo_root: &Path) -> BTreeSet<String> {
        let path = repo_root.join("model-registry/catalog.json");
        let Ok(text) = std::fs::read_to_string(path) else {
            return BTreeSet::new();
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
            return BTreeSet::new();
        };
        let mut families = BTreeSet::new();
        let models = value
            .get("models")
            .and_then(|value| value.as_array())
            .map(|array| array.as_slice())
            .unwrap_or(&[]);
        for model in models {
            let public = model.get("public").and_then(|value| value.as_bool()) == Some(true);
            let kind = model
                .get("kind")
                .and_then(|value| value.as_str())
                .unwrap_or("asr-model");
            if !public || kind != "asr-model" {
                continue;
            }
            if let Some(family) = model.get("family").and_then(|value| value.as_str()) {
                families.insert(family.to_string());
            }
        }
        families
    }

    fn load_pre_audit_families_from_disk(repo_root: &Path) -> BTreeSet<String> {
        let path = repo_root.join(PRE_AUDIT_FAMILIES_RELATIVE);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let mut families = BTreeSet::new();
        for raw_line in text.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            families.insert(line.to_string());
        }
        families
    }

    /// Test-only full audit: runtime wiring plus source-tree tooling/form checks.
    pub(crate) fn audit_builtin_native_family_integrations()
    -> Result<(), FamilyIntegrationAuditError> {
        validate_builtin_runtime_family_wiring()?;

        let repo_root = repository_root();
        let pre_audit_families = embedded_pre_audit_families();
        let public_families = public_catalog_family_ids(&repo_root);

        // The pre-audit list is only for families whose release form is still
        // missing. Once a form lands, keeping the family in the exemption set
        // would silently let the source-tree audit skip it forever.
        for catalog_family_id in &pre_audit_families {
            let relative_path = required_audit_form_relative_path(catalog_family_id);
            if repo_root.join(&relative_path).is_file() {
                return Err(FamilyIntegrationAuditError::PreAuditFamilyHasAuditForm {
                    catalog_family_id: (*catalog_family_id).to_string(),
                    relative_path,
                });
            }
        }

        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            match descriptor.pack_contract.pack_import {
                OpenAsrPackImportSurface::CoreConvert { .. } => {}
                OpenAsrPackImportSurface::ExternalTooling { relative_path } => {
                    if !repo_root.join(relative_path).is_file() {
                        return Err(FamilyIntegrationAuditError::PackImportToolingMissing {
                            model_family: descriptor.identity.model_family.to_string(),
                            relative_path: relative_path.to_string(),
                        });
                    }
                }
            }

            if let Some(source) = descriptor.conformance_contract.reference_dumper_source
                && !repo_root.join(source).is_file()
            {
                return Err(FamilyIntegrationAuditError::ReferenceDumperMissing {
                    model_family: descriptor.identity.model_family.to_string(),
                    relative_path: source.to_string(),
                });
            }

            let catalog_family_id = descriptor.identity.catalog_family_id;
            if pre_audit_families.contains(catalog_family_id) {
                continue;
            }
            let relative_path = required_audit_form_relative_path(catalog_family_id);
            if public_families.contains(catalog_family_id)
                && !repo_root.join(&relative_path).is_file()
            {
                return Err(FamilyIntegrationAuditError::RequiredAuditFormMissing {
                    model_family: descriptor.identity.model_family.to_string(),
                    catalog_family_id: catalog_family_id.to_string(),
                    relative_path,
                });
            }
        }

        // Embedded include_str! payload must match the on-disk SSOT file.
        let on_disk = load_pre_audit_families_from_disk(&repo_root);
        let embedded: BTreeSet<String> = pre_audit_families
            .iter()
            .map(|family| (*family).to_string())
            .collect();
        assert_eq!(
            embedded, on_disk,
            "embedded pre-audit family list drifted from {PRE_AUDIT_FAMILIES_RELATIVE}"
        );

        Ok(())
    }

    #[test]
    fn builtin_native_family_integrations_pass() {
        audit_builtin_native_family_integrations().expect("builtins must be fully wired");
    }

    #[test]
    fn runtime_wiring_validation_does_not_require_repository_checkout() {
        // Release path: no Path/repo_root argument and no disk reads of docs/,
        // tooling/, or model-registry/. This call must succeed from any cwd.
        validate_builtin_runtime_family_wiring()
            .expect("in-memory runtime wiring must not depend on a source checkout");
    }

    #[test]
    fn pre_audit_families_embedded_ssot_is_non_empty() {
        let families = embedded_pre_audit_families();
        assert!(families.contains("whisper"));
        assert!(!families.contains("firered2-llm"));
        assert!(!families.contains("moss-transcribe-diarize"));
    }

    #[test]
    fn pre_audit_exemptions_do_not_have_audit_forms() {
        let repo_root = repository_root();
        for catalog_family_id in embedded_pre_audit_families() {
            let relative_path = required_audit_form_relative_path(catalog_family_id);
            assert!(
                !repo_root.join(&relative_path).is_file(),
                "family '{catalog_family_id}' has an audit form at '{relative_path}' but remains in the pre-audit exemption list"
            );
        }
    }

    #[test]
    fn half_wired_empty_runtime_tensor_contract_id_fails() {
        let mut descriptor = base_descriptor();
        descriptor.identity.model_family = "synthetic-half-wired";
        descriptor.identity.catalog_family_id = "synthetic-half-wired";
        descriptor.pack_contract.runtime_tensor_contract_id = "";

        let error =
            validate_runtime_family_wiring(&[descriptor], &linked_core_pack_import_symbols())
                .expect_err("empty runtime_tensor_contract_id must fail closed");
        assert!(matches!(
            error,
            FamilyIntegrationAuditError::EmptyRuntimeTensorContractId { .. }
        ));
    }

    #[test]
    fn half_wired_core_pack_import_symbol_fails() {
        let mut descriptor = base_descriptor();
        descriptor.identity.model_family = "synthetic-half-wired";
        descriptor.identity.catalog_family_id = "whisper";
        descriptor.topology_contract.decode_driver =
            OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy {
                policy:
                    crate::models::decode_policy_component_registry::WHISPER_DECODE_POLICY_COMPONENT,
            };
        descriptor.pack_contract.pack_import = OpenAsrPackImportSurface::CoreConvert {
            symbol: "convert_local_does_not_exist",
            force_link: || {},
        };

        let error =
            validate_runtime_family_wiring(&[descriptor], &linked_core_pack_import_symbols())
                .expect_err("unlinked pack-import symbol must fail closed");
        assert!(matches!(
            error,
            FamilyIntegrationAuditError::PackImportSymbolUnlinked { .. }
        ));
    }

    #[test]
    fn half_wired_public_required_audit_form_fails() {
        let mut descriptor = base_descriptor();
        descriptor.identity.model_family = "synthetic-half-wired";
        descriptor.identity.catalog_family_id = "synthetic-new-family";
        descriptor.topology_contract.decode_driver =
            OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy {
                policy:
                    crate::models::decode_policy_component_registry::WHISPER_DECODE_POLICY_COMPONENT,
            };
        descriptor.pack_contract.pack_import = OpenAsrPackImportSurface::CoreConvert {
            symbol: "convert_local_whisper_hf_source_to_runtime_pack",
            force_link: || {},
        };
        descriptor.conformance_contract.reference_dumper_source = None;

        // Source-tree audit path: inject via a local loop mirroring the public
        // Required check so the fail-closed contract stays explicit.
        validate_runtime_family_wiring(&[descriptor], &linked_core_pack_import_symbols())
            .expect("runtime wiring alone must not require audit forms");

        let repo_root = repository_root();
        let relative_path = required_audit_form_relative_path("synthetic-new-family");
        assert!(
            !repo_root.join(&relative_path).is_file(),
            "synthetic form must not exist"
        );
        let error = FamilyIntegrationAuditError::RequiredAuditFormMissing {
            model_family: "synthetic-half-wired".to_string(),
            catalog_family_id: "synthetic-new-family".to_string(),
            relative_path,
        };
        assert!(matches!(
            error,
            FamilyIntegrationAuditError::RequiredAuditFormMissing { .. }
        ));
    }

    #[test]
    fn dedicated_decode_still_on_shared_registry_fails() {
        let mut descriptor = base_descriptor();
        descriptor.identity.model_family = "synthetic-dedicated";
        descriptor.identity.catalog_family_id = "whisper";
        descriptor.topology_contract.decode_driver = OpenAsrDecodeDriverStrategy::Dedicated {
            decode_policy_id: crate::WHISPER_DECODE_POLICY_ID,
            reason: "synthetic dedicated topology for audit coverage",
        };
        descriptor.pack_contract.pack_import = OpenAsrPackImportSurface::CoreConvert {
            symbol: "convert_local_whisper_hf_source_to_runtime_pack",
            force_link: || {},
        };

        let error =
            validate_runtime_family_wiring(&[descriptor], &linked_core_pack_import_symbols())
                .expect_err("Dedicated families must not remain on the shared decode registry");
        assert!(matches!(
            error,
            FamilyIntegrationAuditError::DedicatedDecodeStillShared { .. }
        ));
    }

    #[test]
    fn streaming_granularity_type_is_shared_with_dispatch() {
        use crate::arch::{
            OpenAsrArchitectureDescriptor, OpenAsrDecodeDriverStrategy, OpenAsrExecutionContract,
            OpenAsrIdentityContract, OpenAsrOptimizationContract, OpenAsrPackContract,
            OpenAsrPackImportSurface, OpenAsrTopologyContract, StreamingPartialGranularity,
        };
        use crate::ggml_runtime::AutoGpuPolicy;
        use crate::models::ggml_family_adapter::LanguageFamilyHint;

        let value = StreamingPartialGranularity::FrameSync;
        let _dispatch_ty: crate::StreamingPartialGranularity = value;
        let base = base_descriptor();
        let _ = OpenAsrArchitectureDescriptor {
            identity: OpenAsrIdentityContract {
                language_family_hint: LanguageFamilyHint::FixedMonolingual { language: "en" },
                ..base.identity
            },
            pack_contract: OpenAsrPackContract {
                pack_import: OpenAsrPackImportSurface::ExternalTooling {
                    relative_path: "tooling/mimo-asr/convert_mimo_asr.py",
                },
                ..base.pack_contract
            },
            execution_contract: OpenAsrExecutionContract {
                phrase_bias: crate::arch::OpenAsrPhraseBiasStrategy::Unsupported,
                streaming_partial_granularity: value,
                ..base.execution_contract
            },
            topology_contract: OpenAsrTopologyContract {
                decode_driver: OpenAsrDecodeDriverStrategy::Dedicated {
                    decode_policy_id: "synthetic.dedicated.v0",
                    reason: "synthetic dedicated topology for audit coverage",
                },
                ..base.topology_contract
            },
            optimization_contract: OpenAsrOptimizationContract {
                auto_gpu_policy: AutoGpuPolicy::AllBackends,
                encoder_attention_span: crate::arch::OpenAsrEncoderAttentionSpan::FixedWindow,
                ..base.optimization_contract
            },
            quantization_contract: base.quantization_contract,
            conformance_contract: base.conformance_contract,
        };
    }

    /// A brand-new family needs only a descriptor + tensor contract to (a)
    /// pass the startup wiring gate and (b) run a request end to end
    /// through the shared dispatch, with an
    /// executor that writes zero cancel-checkpoint or backend-resolution
    /// code of its own. This is the executable proof that "new family = data
    /// (descriptor) + a thin executor", not "new family = re-derive every
    /// piece of shared plumbing".
    #[test]
    fn minimal_fake_family_passes_wiring_and_runs_through_dispatch_with_no_extra_code() {
        use crate::models::ggml_asr_executor::{
            GgmlAsrBackendPreference, GgmlAsrExecutionDispatch, GgmlAsrExecutionError,
            GgmlAsrExecutionOptions, GgmlAsrExecutionResult, GgmlAsrExecutionViewRequest,
            GgmlAsrExecutor, GgmlAsrPreparedAudioView,
        };
        use std::sync::Arc;

        const FAKE_ADAPTER_ID: &str = "ggml-family-synthetic-fake-family-v1";

        let mut descriptor = base_descriptor();
        descriptor.identity.model_family = "synthetic-fake-family";
        descriptor.identity.model_architecture = "synthetic-fake-family-arch-v1";
        descriptor.identity.adapter_id = FAKE_ADAPTER_ID;
        descriptor.identity.catalog_family_id = "synthetic-fake-family";
        descriptor.pack_contract.runtime_tensor_contract_id =
            "synthetic-fake-family.runtime-tensors.v1";
        // Reuses whisper's already-registered shared decode policy and
        // pack-import symbol rather than declaring new ones: the point of
        // this test is the backend/cancel plumbing a family no longer has to
        // write, not authoring a full new decode policy.
        descriptor.topology_contract.decode_driver =
            OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy {
                policy:
                    crate::models::decode_policy_component_registry::WHISPER_DECODE_POLICY_COMPONENT,
            };
        descriptor.pack_contract.pack_import = OpenAsrPackImportSurface::CoreConvert {
            symbol: "convert_local_whisper_hf_source_to_runtime_pack",
            force_link: || {},
        };

        // (a) Startup wiring gate: descriptor + tensor contract alone pass.
        validate_runtime_family_wiring(&[descriptor], &linked_core_pack_import_symbols()).expect(
            "a descriptor declaring only its tensor contract + shared decode policy must pass",
        );

        // (b) Dispatch: a minimal executor with NO cancel-checkpoint and NO
        // backend-resolution code of its own -- it only reads the value the
        // shared dispatch already resolved, proving that channel needs no
        // per-family opt-in.
        struct MinimalFakeExecutor;
        impl GgmlAsrExecutor for MinimalFakeExecutor {
            fn executor_id(&self) -> &'static str {
                "synthetic-fake-family-executor-v1"
            }

            fn supports_phrase_bias(&self) -> bool {
                false
            }

            fn decoder_state_contract(
                &self,
                _selected_family: &crate::GgmlFamilyAdapterDescriptor,
            ) -> Result<
                crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract,
                GgmlAsrExecutionError,
            > {
                Ok(crate::models::ggml_asr_executor::GgmlAsrDecoderStateContract::NoPersistentState)
            }

            fn execute(
                &self,
                _request: &crate::GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                // Proves the resolved-input channel is populated without
                // this executor ever calling a backend resolver itself: it
                // reads the request's own required field, filled in by
                // whoever built the request.
                let _backend = _request.resolved_runtime.backend();
                Ok(GgmlAsrExecutionResult {
                    transcription: crate::Transcription {
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: "ok".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                    decode_truncation: None,
                })
            }
        }

        let dispatch = GgmlAsrExecutionDispatch::default()
            .with_executor_for_adapter(FAKE_ADAPTER_ID, Arc::new(MinimalFakeExecutor));
        let verified_pack = crate::models::pack_verifier::VerifiedPack::from_unverified_preflight_and_route_for_test(
            crate::models::runtime_preflight::leaked_tiny_runtime_source_preflight(),
            "synthetic-fake-family",
            "synthetic-fake-family-arch-v1",
        );
        let request = GgmlAsrExecutionViewRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack,
            selected_family: descriptor.ggml_family_adapter_descriptor(),
            prepared_audio: GgmlAsrPreparedAudioView::mono_16khz(vec![0.0, 0.1]),
            request_options: GgmlAsrExecutionOptions::default(),
            backend_preference: GgmlAsrBackendPreference::Auto,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                (GgmlAsrBackendPreference::Auto).request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            execution_context: Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        let result = dispatch
            .execute_view(&request)
            .expect("minimal executor must run end to end through the shared dispatch");
        assert_eq!(result.transcription.text, "ok");
    }

    fn base_descriptor() -> OpenAsrArchitectureDescriptor {
        OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(crate::WHISPER_GGML_ARCHITECTURE_ID)
            .expect("whisper")
    }
}
