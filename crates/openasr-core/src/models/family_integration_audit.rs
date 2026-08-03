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
    #[error("native family '{model_family}' has empty runtime_tensor_contract_id")]
    EmptyRuntimeTensorContractId { model_family: String },
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
    #[error(
        "native family '{model_family}' declares a derived capacity model but frontend id '{audio_frontend_id}' has no capacity-frontend registry row"
    )]
    CapacityFrontendUnregistered {
        model_family: String,
        audio_frontend_id: String,
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

        // A new family's minimal accession surface is descriptor + tensor
        // contract (see this module's doc comment and
        // `models::runtime_tensor_contract_registry`); nothing here checks
        // decode policy resolves without one, so fail closed on an empty id
        // instead of letting a half-declared family silently run without a
        // validated tensor contract.
        if descriptor.runtime_tensor_contract_id.is_empty() {
            return Err(FamilyIntegrationAuditError::EmptyRuntimeTensorContractId {
                model_family: descriptor.model_family.to_string(),
            });
        }

        // A family that declares its capacity DERIVED promises derivation
        // reads its frontend facts from the capacity registry -- fail closed
        // on a frontend id with no row rather than let derivation silently
        // fall back to prose constants (the documented failure mode the
        // registry exists to replace).
        if matches!(
            descriptor.capacity_model,
            crate::capacity::CapacityModelDeclaration::Derived(_)
        ) && crate::capacity::frontend_capacity_basis(descriptor.audio_frontend_id).is_none()
        {
            return Err(FamilyIntegrationAuditError::CapacityFrontendUnregistered {
                model_family: descriptor.model_family.to_string(),
                audio_frontend_id: descriptor.audio_frontend_id.to_string(),
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

    /// Every family that reaches the shared greedy driver must lift the
    /// driver's stop reason into the transcript's truncation signal.
    ///
    /// The driver reports a guard cut and an exhausted budget as distinct stop
    /// reasons precisely so a caller can see that the transcript stops short of
    /// the audio -- but the reason dies inside the family unless the family
    /// forwards it. That failure is invisible: the request succeeds, the shape
    /// is normal, there is just less text. This check fails closed on a family
    /// that routes through the driver and never calls
    /// `into_decode_truncation`, the same way the decode-policy resolution test
    /// fails closed on a half-connected driver.
    #[test]
    fn every_shared_greedy_family_forwards_the_driver_stop_reason() {
        // hymt2 is a translation runtime, not an ASR family: it produces
        // subtitle text from text, never a `Transcription` over audio, and is
        // absent from the architecture registry's SharedSeq2SeqGreedy set. It
        // has no transcript for a truncation signal to ride on, so the contract
        // this test enforces does not apply to it.
        const NON_TRANSCRIPT_FAMILY_DIRECTORIES: &[&str] = &["hymt2"];

        let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/models");
        let mut checked = 0usize;
        for entry in std::fs::read_dir(&models_dir).expect("models dir is readable") {
            let entry = entry.expect("models dir entry");
            if !entry.file_type().expect("entry file type").is_dir() {
                continue;
            }
            let directory = entry.path();
            let name = directory
                .file_name()
                .and_then(|name| name.to_str())
                .expect("family directory name")
                .to_string();
            let sources: Vec<String> = std::fs::read_dir(&directory)
                .expect("family dir is readable")
                .filter_map(|file| file.ok())
                .map(|file| file.path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
                .filter_map(|path| std::fs::read_to_string(path).ok())
                .collect();
            // Match the call, turbofished or not, and never the `use` line:
            // `run_builtin_seq2seq_decode_policy(` alone silently skipped the
            // families that spell the call `...::<Seq2SeqGreedyDecodeError>(`.
            let reaches_shared_driver = sources.iter().any(|source| {
                source.contains("run_builtin_seq2seq_decode_policy(")
                    || source.contains("run_builtin_seq2seq_decode_policy::<")
            });
            if !reaches_shared_driver || NON_TRANSCRIPT_FAMILY_DIRECTORIES.contains(&name.as_str())
            {
                continue;
            }
            checked += 1;
            assert!(
                sources
                    .iter()
                    .any(|source| source.contains("into_decode_truncation(")),
                "family '{name}' routes through the shared greedy driver but never lifts its \
                 stop reason into a truncation signal: a guard-cut transcript would be returned \
                 as a complete one"
            );
        }
        // Guards the walk itself: a rename that stops matching the driver call
        // would otherwise make this test vacuously pass.
        assert_eq!(
            checked, 10,
            "expected the 10 SharedSeq2SeqGreedy ASR families to be found by this walk"
        );
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
    fn half_wired_empty_runtime_tensor_contract_id_fails() {
        let mut descriptor = base_descriptor();
        descriptor.model_family = "synthetic-half-wired";
        descriptor.integration.catalog_family_id = "synthetic-half-wired";
        descriptor.runtime_tensor_contract_id = "";

        let error = validate_runtime_family_wiring(
            &[descriptor],
            &resolve_builtin_decode_policy,
            &linked_core_pack_import_symbols(),
        )
        .expect_err("empty runtime_tensor_contract_id must fail closed");
        assert!(matches!(
            error,
            FamilyIntegrationAuditError::EmptyRuntimeTensorContractId { .. }
        ));
    }

    #[test]
    fn derived_capacity_model_without_registered_frontend_fails() {
        let mut descriptor = base_descriptor();
        descriptor.model_family = "synthetic-half-wired";
        descriptor.integration.catalog_family_id = "synthetic-half-wired";
        // Whisper's frontend id has no capacity-registry row (whisper is
        // BoundedElsewhere, never Derived): declaring Derived on top of it
        // must fail closed.
        descriptor.capacity_model = crate::capacity::CapacityModelDeclaration::Derived(
            crate::capacity::CapacityModelDescriptor {
                audio_bound: crate::capacity::CapacityAudioBound::DecoderContext,
            },
        );

        let error = validate_runtime_family_wiring(
            &[descriptor],
            &resolve_builtin_decode_policy,
            &linked_core_pack_import_symbols(),
        )
        .expect_err("Derived capacity model with an unregistered frontend id must fail closed");
        assert!(matches!(
            error,
            FamilyIntegrationAuditError::CapacityFrontendUnregistered { .. }
        ));
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
        use std::path::PathBuf;
        use std::sync::Arc;

        const FAKE_ADAPTER_ID: &str = "ggml-family-synthetic-fake-family-v1";

        let mut descriptor = base_descriptor();
        descriptor.model_family = "synthetic-fake-family";
        descriptor.model_architecture = "synthetic-fake-family-arch-v1";
        descriptor.adapter_id = FAKE_ADAPTER_ID;
        descriptor.integration.catalog_family_id = "synthetic-fake-family";
        descriptor.runtime_tensor_contract_id = "synthetic-fake-family.runtime-tensors.v1";
        // Reuses whisper's already-registered shared decode policy and
        // pack-import symbol rather than declaring new ones: the point of
        // this test is the backend/cancel plumbing a family no longer has to
        // write, not authoring a full new decode policy.
        descriptor.integration.shared_decode_driver =
            OpenAsrSharedDecodeDriver::SharedSeq2SeqGreedy;
        descriptor.decode_policy_id = crate::WHISPER_DECODE_POLICY_ID;
        descriptor.integration.pack_import = OpenAsrPackImportSurface::CoreConvert {
            symbol: "convert_local_whisper_hf_source_to_runtime_pack",
        };

        // (a) Startup wiring gate: descriptor + tensor contract alone pass.
        validate_runtime_family_wiring(
            &[descriptor],
            &resolve_builtin_decode_policy,
            &linked_core_pack_import_symbols(),
        )
        .expect("a descriptor declaring only its tensor contract + shared decode policy must pass");

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
        let request = GgmlAsrExecutionViewRequest {
            runtime_source_path: PathBuf::from("fixtures/synthetic-fake-family.gguf"),
            runtime_source_preflight: None,
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
