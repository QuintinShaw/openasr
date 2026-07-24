//! Fail-closed native-family integration audit.
//!
//! Walks the architecture registry (single authority for runtime integration
//! facts) and checks every applicable contract: shared decode-driver class,
//! pack-import surface linkage, optional reference dumpers, and audit-form
//! requirements derived from the shared pre-audit family list. Pure helpers
//! accept injected inputs so half-wired fixtures can prove the gate turns red.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::arch::{
    OpenAsrArchitectureDescriptor, OpenAsrArchitectureRegistry, OpenAsrPackImportSurface,
    OpenAsrSharedDecodeDriver,
};
use crate::models::decode_policy_component_registry::{
    BuiltinDecodePolicyComponentDescriptor, BuiltinDecodePolicyComponentRegistryError,
    BuiltinDecodePolicyExecutionKind, resolve_builtin_decode_policy,
};
use crate::models::pack_import_surface::linked_core_pack_import_symbols;

const PRE_AUDIT_FAMILIES_RELATIVE: &str = "docs/model-audits/pre_audit_families.txt";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum FamilyIntegrationAuditError {
    #[error("native family '{model_family}' has empty catalog_family_id")]
    EmptyCatalogFamilyId { model_family: String },
    #[error(
        "native family '{model_family}' shared-decode driver {expected:?} is not registered for policy '{decode_policy_id}': {reason}"
    )]
    SharedDecodeMissing {
        model_family: String,
        decode_policy_id: String,
        expected: OpenAsrSharedDecodeDriver,
        reason: String,
    },
    #[error(
        "native family '{model_family}' shared-decode driver is {expected:?} but policy '{decode_policy_id}' resolved as {actual:?}"
    )]
    SharedDecodeKindMismatch {
        model_family: String,
        decode_policy_id: String,
        expected: OpenAsrSharedDecodeDriver,
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
    #[error(
        "native family '{model_family}' external pack-import tooling '{relative_path}' is missing"
    )]
    PackImportToolingMissing {
        model_family: String,
        relative_path: String,
    },
    #[error("native family '{model_family}' reference dumper '{relative_path}' is missing")]
    ReferenceDumperMissing {
        model_family: String,
        relative_path: String,
    },
    #[error(
        "native family '{model_family}' catalog id '{catalog_family_id}' requires audit form '{relative_path}' but the file is missing while the family is public"
    )]
    RequiredAuditFormMissing {
        model_family: String,
        catalog_family_id: String,
        relative_path: String,
    },
    #[error("failed to read {PRE_AUDIT_FAMILIES_RELATIVE}: {reason}")]
    PreAuditListUnreadable { reason: String },
}

pub(crate) fn load_pre_audit_families(
    repo_root: &Path,
) -> Result<BTreeSet<String>, FamilyIntegrationAuditError> {
    let path = repo_root.join(PRE_AUDIT_FAMILIES_RELATIVE);
    let text = std::fs::read_to_string(&path).map_err(|error| {
        FamilyIntegrationAuditError::PreAuditListUnreadable {
            reason: format!("{} ({error})", path.display()),
        }
    })?;
    let mut families = BTreeSet::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        families.insert(line.to_string());
    }
    Ok(families)
}

pub(crate) fn required_audit_form_relative_path(catalog_family_id: &str) -> String {
    format!("docs/model-audits/{catalog_family_id}.md")
}

pub(crate) fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root")
}

pub(crate) fn public_catalog_family_ids(repo_root: &Path) -> BTreeSet<String> {
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

pub(crate) fn audit_architecture_integrations(
    architectures: &[OpenAsrArchitectureDescriptor],
    decode_resolve: &dyn Fn(
        &str,
    ) -> Result<
        BuiltinDecodePolicyComponentDescriptor,
        BuiltinDecodePolicyComponentRegistryError,
    >,
    linked_pack_symbols: &BTreeMap<&'static str, usize>,
    repo_root: &Path,
    pre_audit_families: &BTreeSet<String>,
    public_families: &BTreeSet<String>,
) -> Result<(), FamilyIntegrationAuditError> {
    for descriptor in architectures {
        if descriptor.integration.catalog_family_id.is_empty() {
            return Err(FamilyIntegrationAuditError::EmptyCatalogFamilyId {
                model_family: descriptor.model_family.to_string(),
            });
        }

        match descriptor.integration.shared_decode_driver {
            OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy => {
                let policy = decode_resolve(descriptor.decode_policy_id).map_err(|error| {
                    FamilyIntegrationAuditError::SharedDecodeMissing {
                        model_family: descriptor.model_family.to_string(),
                        decode_policy_id: descriptor.decode_policy_id.to_string(),
                        expected: OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy,
                        reason: error.to_string(),
                    }
                })?;
                if policy.execution_kind != BuiltinDecodePolicyExecutionKind::Seq2SeqGreedyV0 {
                    return Err(FamilyIntegrationAuditError::SharedDecodeKindMismatch {
                        model_family: descriptor.model_family.to_string(),
                        decode_policy_id: descriptor.decode_policy_id.to_string(),
                        expected: OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy,
                        actual: policy.execution_kind,
                    });
                }
            }
            OpenAsrSharedDecodeDriver::SharedCtcGreedy => {
                let policy = decode_resolve(descriptor.decode_policy_id).map_err(|error| {
                    FamilyIntegrationAuditError::SharedDecodeMissing {
                        model_family: descriptor.model_family.to_string(),
                        decode_policy_id: descriptor.decode_policy_id.to_string(),
                        expected: OpenAsrSharedDecodeDriver::SharedCtcGreedy,
                        reason: error.to_string(),
                    }
                })?;
                if policy.execution_kind != BuiltinDecodePolicyExecutionKind::CtcGreedyV0 {
                    return Err(FamilyIntegrationAuditError::SharedDecodeKindMismatch {
                        model_family: descriptor.model_family.to_string(),
                        decode_policy_id: descriptor.decode_policy_id.to_string(),
                        expected: OpenAsrSharedDecodeDriver::SharedCtcGreedy,
                        actual: policy.execution_kind,
                    });
                }
                if policy.ctc_blank_token_id.is_none() {
                    return Err(FamilyIntegrationAuditError::CtcBlankMissing {
                        model_family: descriptor.model_family.to_string(),
                        decode_policy_id: descriptor.decode_policy_id.to_string(),
                    });
                }
            }
            OpenAsrSharedDecodeDriver::Dedicated => {
                if decode_resolve(descriptor.decode_policy_id).is_ok() {
                    return Err(FamilyIntegrationAuditError::DedicatedDecodeStillShared {
                        model_family: descriptor.model_family.to_string(),
                        decode_policy_id: descriptor.decode_policy_id.to_string(),
                    });
                }
            }
        }

        match descriptor.integration.pack_import {
            OpenAsrPackImportSurface::CoreConvert { symbol } => {
                if !linked_pack_symbols.contains_key(symbol) {
                    return Err(FamilyIntegrationAuditError::PackImportSymbolUnlinked {
                        model_family: descriptor.model_family.to_string(),
                        symbol: symbol.to_string(),
                    });
                }
            }
            OpenAsrPackImportSurface::ExternalTooling { relative_path } => {
                if !repo_root.join(relative_path).is_file() {
                    return Err(FamilyIntegrationAuditError::PackImportToolingMissing {
                        model_family: descriptor.model_family.to_string(),
                        relative_path: relative_path.to_string(),
                    });
                }
            }
        }

        if let Some(source) = descriptor.integration.reference_dumper_source
            && !repo_root.join(source).is_file()
        {
            return Err(FamilyIntegrationAuditError::ReferenceDumperMissing {
                model_family: descriptor.model_family.to_string(),
                relative_path: source.to_string(),
            });
        }

        let catalog_family_id = descriptor.integration.catalog_family_id;
        if pre_audit_families.contains(catalog_family_id) {
            // LegacyExempt -- form optional; publish still validates a form that exists.
        } else {
            let relative_path = required_audit_form_relative_path(catalog_family_id);
            if public_families.contains(catalog_family_id)
                && !repo_root.join(&relative_path).is_file()
            {
                return Err(FamilyIntegrationAuditError::RequiredAuditFormMissing {
                    model_family: descriptor.model_family.to_string(),
                    catalog_family_id: catalog_family_id.to_string(),
                    relative_path,
                });
            }
        }
    }

    Ok(())
}

pub(crate) fn audit_builtin_native_family_integrations() -> Result<(), FamilyIntegrationAuditError>
{
    let repo_root = repository_root();
    let pre_audit_families = load_pre_audit_families(&repo_root)?;
    let public_families = public_catalog_family_ids(&repo_root);
    let linked = linked_core_pack_import_symbols();
    let architectures = OpenAsrArchitectureRegistry::with_builtins().descriptors();
    audit_architecture_integrations(
        architectures,
        &resolve_builtin_decode_policy,
        &linked,
        &repo_root,
        &pre_audit_families,
        &public_families,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{
        OpenAsrArchitectureDescriptor, OpenAsrEncoderAttentionSpan, OpenAsrPackImportSurface,
        OpenAsrSharedDecodeDriver, StreamingPartialGranularity,
    };
    use crate::ggml_runtime::AutoGpuPolicy;
    use crate::models::ggml_family_adapter::{GgmlExecutionCapability, LanguageFamilyHint};

    fn base_descriptor() -> OpenAsrArchitectureDescriptor {
        OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(crate::WHISPER_GGML_ARCHITECTURE_ID)
            .expect("whisper")
    }

    #[test]
    fn builtin_native_family_integrations_pass() {
        audit_builtin_native_family_integrations().expect("builtins must be fully wired");
    }

    #[test]
    fn pre_audit_families_ssot_is_non_empty_and_matches_python_gate_path() {
        let families = load_pre_audit_families(&repository_root()).expect("pre-audit list");
        assert!(families.contains("whisper"));
        assert!(families.contains("qwen"));
        assert!(!families.contains("firered2-llm"));
        assert!(!families.contains("moss-transcribe-diarize"));
    }

    #[test]
    fn half_wired_shared_seq2seq_without_decode_policy_fails() {
        let mut descriptor = base_descriptor();
        descriptor.model_family = "synthetic-half-wired";
        descriptor.integration.catalog_family_id = "synthetic-half-wired";
        descriptor.integration.shared_decode_driver =
            OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy;
        descriptor.decode_policy_id = "synthetic.greedy.seq2seq.v0";
        descriptor.integration.pack_import = OpenAsrPackImportSurface::CoreConvert {
            symbol: "convert_local_whisper_hf_source_to_runtime_pack",
        };

        let error = audit_architecture_integrations(
            &[descriptor],
            &resolve_builtin_decode_policy,
            &linked_core_pack_import_symbols(),
            &repository_root(),
            &BTreeSet::new(),
            &BTreeSet::new(),
        )
        .expect_err("missing shared decode policy must fail closed");
        assert!(matches!(
            error,
            FamilyIntegrationAuditError::SharedDecodeMissing { .. }
        ));
    }

    #[test]
    fn half_wired_core_pack_import_symbol_fails() {
        let mut descriptor = base_descriptor();
        descriptor.model_family = "synthetic-half-wired";
        descriptor.integration.catalog_family_id = "whisper"; // legacy exempt
        descriptor.integration.shared_decode_driver =
            OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy;
        descriptor.decode_policy_id = crate::WHISPER_DECODE_POLICY_ID;
        descriptor.integration.pack_import = OpenAsrPackImportSurface::CoreConvert {
            symbol: "convert_local_does_not_exist",
        };

        let error = audit_architecture_integrations(
            &[descriptor],
            &resolve_builtin_decode_policy,
            &linked_core_pack_import_symbols(),
            &repository_root(),
            &load_pre_audit_families(&repository_root()).unwrap(),
            &BTreeSet::new(),
        )
        .expect_err("unlinked pack-import symbol must fail closed");
        assert!(matches!(
            error,
            FamilyIntegrationAuditError::PackImportSymbolUnlinked { .. }
        ));
    }

    #[test]
    fn half_wired_public_required_audit_form_fails() {
        let mut descriptor = base_descriptor();
        descriptor.model_family = "synthetic-half-wired";
        descriptor.integration.catalog_family_id = "synthetic-new-family";
        descriptor.integration.shared_decode_driver =
            OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy;
        descriptor.decode_policy_id = crate::WHISPER_DECODE_POLICY_ID;
        descriptor.integration.pack_import = OpenAsrPackImportSurface::CoreConvert {
            symbol: "convert_local_whisper_hf_source_to_runtime_pack",
        };
        descriptor.integration.reference_dumper_source = None;

        let public = BTreeSet::from(["synthetic-new-family".to_string()]);
        let error = audit_architecture_integrations(
            &[descriptor],
            &resolve_builtin_decode_policy,
            &linked_core_pack_import_symbols(),
            &repository_root(),
            &BTreeSet::new(),
            &public,
        )
        .expect_err("public Required family without audit form must fail closed");
        assert!(matches!(
            error,
            FamilyIntegrationAuditError::RequiredAuditFormMissing { .. }
        ));
    }

    #[test]
    fn dedicated_decode_still_on_shared_registry_fails() {
        let mut descriptor = base_descriptor();
        descriptor.model_family = "synthetic-dedicated";
        descriptor.integration.catalog_family_id = "whisper";
        descriptor.integration.shared_decode_driver = OpenAsrSharedDecodeDriver::Dedicated;
        descriptor.decode_policy_id = crate::WHISPER_DECODE_POLICY_ID;
        descriptor.integration.pack_import = OpenAsrPackImportSurface::CoreConvert {
            symbol: "convert_local_whisper_hf_source_to_runtime_pack",
        };

        let error = audit_architecture_integrations(
            &[descriptor],
            &resolve_builtin_decode_policy,
            &linked_core_pack_import_symbols(),
            &repository_root(),
            &load_pre_audit_families(&repository_root()).unwrap(),
            &BTreeSet::new(),
        )
        .expect_err("Dedicated families must not remain on the shared decode registry");
        assert!(matches!(
            error,
            FamilyIntegrationAuditError::DedicatedDecodeStillShared { .. }
        ));
    }

    #[test]
    fn streaming_granularity_type_is_shared_with_dispatch() {
        // Compile-time/type-level lock: architecture integration and the
        // public streaming dispatch API share one enum.
        let value = StreamingPartialGranularity::FrameSync;
        let _dispatch_ty: crate::StreamingPartialGranularity = value;
        let _ = OpenAsrArchitectureDescriptor {
            integration: crate::arch::OpenAsrFamilyIntegrationDescriptor {
                catalog_family_id: "x",
                supports_phrase_bias: false,
                streaming_partial_granularity: value,
                shared_decode_driver: OpenAsrSharedDecodeDriver::Dedicated,
                pack_import: OpenAsrPackImportSurface::ExternalTooling {
                    relative_path: "tooling/mimo-asr/convert_mimo_asr.py",
                },
                reference_dumper_source: None,
            },
            language_family_hint: LanguageFamilyHint::FixedMonolingual { language: "en" },
            execution_capability: GgmlExecutionCapability::DedicatedRuntimeExecutorV1,
            auto_gpu_policy: AutoGpuPolicy::AllBackends,
            encoder_attention_span: OpenAsrEncoderAttentionSpan::FixedWindow,
            ..base_descriptor()
        };
    }
}
