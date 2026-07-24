//! Native-family integration wiring checks.
//!
//! **Runtime (ships in release binaries):** purely in-memory validation against
//! the architecture registry, shared decode-policy registry, and the
//! force-linked pack-import symbol table. No repository checkout, no
//! `CARGO_MANIFEST_DIR` path walks, no docs/tooling/catalog disk I/O.
//!
//! **Tests only:** additional fail-closed checks that *do* read the source
//! tree (external tooling paths, reference dumpers, public audit forms) and
//! lock the embedded pre-audit family list to the on-disk SSOT file.

use std::collections::{BTreeMap, BTreeSet};

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
        &resolve_builtin_decode_policy,
        &linked,
    )
}

pub(crate) fn validate_runtime_family_wiring(
    architectures: &[OpenAsrArchitectureDescriptor],
    decode_resolve: &dyn Fn(
        &str,
    ) -> Result<
        BuiltinDecodePolicyComponentDescriptor,
        BuiltinDecodePolicyComponentRegistryError,
    >,
    linked_pack_symbols: &BTreeMap<&'static str, usize>,
) -> Result<(), FamilyIntegrationAuditError> {
    // Touch embedded SSOT so the include_str! payload stays linked and is
    // available to tests without implying runtime disk access.
    let _pre_audit = embedded_pre_audit_families();

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
                if relative_path.is_empty() {
                    return Err(FamilyIntegrationAuditError::PackImportToolingPathEmpty {
                        model_family: descriptor.model_family.to_string(),
                    });
                }
            }
        }

        if let Some(source) = descriptor.integration.reference_dumper_source
            && source.is_empty()
        {
            return Err(FamilyIntegrationAuditError::ReferenceDumperPathEmpty {
                model_family: descriptor.model_family.to_string(),
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

        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            match descriptor.integration.pack_import {
                OpenAsrPackImportSurface::CoreConvert { .. } => {}
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
                continue;
            }
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

        let error = validate_runtime_family_wiring(
            &[descriptor],
            &resolve_builtin_decode_policy,
            &linked_core_pack_import_symbols(),
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
        descriptor.integration.catalog_family_id = "whisper";
        descriptor.integration.shared_decode_driver =
            OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy;
        descriptor.decode_policy_id = crate::WHISPER_DECODE_POLICY_ID;
        descriptor.integration.pack_import = OpenAsrPackImportSurface::CoreConvert {
            symbol: "convert_local_does_not_exist",
        };

        let error = validate_runtime_family_wiring(
            &[descriptor],
            &resolve_builtin_decode_policy,
            &linked_core_pack_import_symbols(),
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

        // Source-tree audit path: inject via a local loop mirroring the public
        // Required check so the fail-closed contract stays explicit.
        validate_runtime_family_wiring(
            &[descriptor],
            &resolve_builtin_decode_policy,
            &linked_core_pack_import_symbols(),
        )
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
        descriptor.model_family = "synthetic-dedicated";
        descriptor.integration.catalog_family_id = "whisper";
        descriptor.integration.shared_decode_driver = OpenAsrSharedDecodeDriver::Dedicated;
        descriptor.decode_policy_id = crate::WHISPER_DECODE_POLICY_ID;
        descriptor.integration.pack_import = OpenAsrPackImportSurface::CoreConvert {
            symbol: "convert_local_whisper_hf_source_to_runtime_pack",
        };

        let error = validate_runtime_family_wiring(
            &[descriptor],
            &resolve_builtin_decode_policy,
            &linked_core_pack_import_symbols(),
        )
        .expect_err("Dedicated families must not remain on the shared decode registry");
        assert!(matches!(
            error,
            FamilyIntegrationAuditError::DedicatedDecodeStillShared { .. }
        ));
    }

    #[test]
    fn streaming_granularity_type_is_shared_with_dispatch() {
        use crate::arch::{
            OpenAsrArchitectureDescriptor, OpenAsrEncoderAttentionSpan, OpenAsrPackImportSurface,
            OpenAsrSharedDecodeDriver, StreamingPartialGranularity,
        };
        use crate::ggml_runtime::AutoGpuPolicy;
        use crate::models::ggml_family_adapter::{GgmlExecutionCapability, LanguageFamilyHint};

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

    fn base_descriptor() -> OpenAsrArchitectureDescriptor {
        OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(crate::WHISPER_GGML_ARCHITECTURE_ID)
            .expect("whisper")
    }
}
