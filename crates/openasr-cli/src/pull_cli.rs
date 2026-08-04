use std::{path::Path, sync::Arc};

use anyhow::{Result, bail};
use openasr_core::{
    CatalogPullRequest, DEFAULT_MODEL_BOOTSTRAP_QUANT, DEFAULT_MODEL_ID, DownloadSourcePref,
    InstalledPack, LaunchPackRequest, LicenseClass, ModelCatalog, ModelInstallLicenseDecision,
    NativeExecutionServices, OpenAsrConfig, PullModelPackRequest, QuantPreference,
    ResolvedCatalogPull, host_quant_recommendation_profile,
    install_model_pack_from_path_with_execution_services, list_installed_packs, load_config,
    load_model_catalog, model_install_license_decision, openasr_home,
    remove_model_pack_with_execution_services, resolve_catalog_pull_with_profile, resolve_chain,
    resolve_launch_pack,
};

use crate::PullCommandOptions;
use crate::consent::{self, CliExit, ExitCode, PullConsent};

pub(crate) fn pull(
    native_execution_services: &Arc<NativeExecutionServices>,
    options: PullCommandOptions<'_>,
) -> Result<()> {
    let home = openasr_home()?;
    let config = load_config(&home)?;
    let catalog = load_model_catalog(options.catalog_url, &home)?;
    let pull_request = CatalogPullRequest {
        reference: options.reference.to_string(),
        quant: options.quant.map(ToOwned::to_owned),
        size: options.size.map(ToOwned::to_owned),
    };
    // §1.2: with no quant pinned, default to this machine's device-recommended
    // quant (largest that fits ~75% of RAM); an explicit :quant / --quant wins.
    let device_profile = host_quant_recommendation_profile();
    let resolved =
        resolve_catalog_pull_with_profile(&catalog, &pull_request, Some(device_profile))?;
    if options.quant.is_none() && !options.reference.contains(':') {
        eprintln!(
            "Selected quant '{}' for this machine; override with <model>:<quant> (e.g. :q4_k or :fp16).",
            resolved.quant
        );
    }

    ensure_explicit_pull_license_acceptance(&resolved, options.accept_license)?;

    let mut reporter = crate::progress::PullReporter::new(&resolved.pull);
    let progress = |event| reporter.on(event);

    let source_pref = match options.source {
        Some(source) => DownloadSourcePref::parse_env_value(source)
            .ok_or_else(|| anyhow::anyhow!("Unsupported download source '{source}'"))?,
        None => config.download_source.clone(),
    };
    let source_chain = resolve_chain(&source_pref);

    let installed = if let Some(path) = options.from {
        install_model_pack_from_path_with_execution_services(
            &resolved,
            path,
            &home,
            Some(native_execution_services.as_ref()),
            progress,
        )?
    } else {
        PullModelPackRequest::new(&resolved, &home)
            .execution_services(native_execution_services.as_ref())
            .sources(&source_chain)
            .execute(progress)?
    };

    let preference = if options.quant.is_some() || options.reference.contains(':') {
        QuantPreference::pinned(&installed.quant)
    } else {
        QuantPreference::Auto
    };
    if should_update_default_asr_model(&catalog, &installed.model_id) {
        openasr_core::default_selection::persist(&home, &installed, preference)?;
    } else {
        eprintln!(
            "{}",
            non_default_asr_install_status(&catalog, &installed.model_id, &installed.pull)
        );
    }
    println!(
        "{}\t{}\t{}\t{}",
        installed.pull,
        installed.size_bytes,
        installed.sha256,
        installed.path.display()
    );
    Ok(())
}

fn non_default_asr_install_status(catalog: &ModelCatalog, model_id: &str, pull: &str) -> String {
    let pack_kind = match catalog_model_kind(catalog, model_id) {
        Some(openasr_core::CatalogModelKind::TranslationModel) => "translation model",
        _ => "capability pack",
    };
    format!("Installed {pack_kind} {pull}; default ASR model was not changed.")
}

fn catalog_model_kind(
    catalog: &ModelCatalog,
    model_id: &str,
) -> Option<openasr_core::CatalogModelKind> {
    catalog
        .models
        .iter()
        .find(|model| model.id == model_id)
        .map(|model| model.kind)
}

fn should_update_default_asr_model(catalog: &ModelCatalog, model_id: &str) -> bool {
    catalog
        .models
        .iter()
        .find(|model| model.id == model_id)
        .is_some_and(|model| model.public && model.kind == openasr_core::CatalogModelKind::AsrModel)
}

fn ensure_explicit_pull_license_acceptance(
    resolved: &ResolvedCatalogPull,
    accepted: bool,
) -> Result<()> {
    match model_install_license_decision(&resolved.license_class, accepted) {
        ModelInstallLicenseDecision::Allowed => Ok(()),
        ModelInstallLicenseDecision::Unsupported => bail!(
            "Model '{}' has an unsupported license class and cannot be installed by this OpenASR version.",
            resolved.model_id
        ),
        ModelInstallLicenseDecision::ExplicitAcceptanceRequired
            if resolved.license_class == LicenseClass::Noncommercial =>
        {
            bail!(
                "Model '{}' is licensed for non-commercial use only ({}).\nReview {} and rerun with --accept-license.",
                resolved.model_id,
                resolved.license,
                resolved.license_url,
            )
        }
        ModelInstallLicenseDecision::ExplicitAcceptanceRequired => bail!(
            "Model '{}' requires vendor license acceptance before installation.\nOpen vendor site: {}\nThen rerun with --accept-license.",
            resolved.model_id,
            resolved.license_url,
        ),
    }
}

/// Automatic CLI convenience pulls must never stand in for accepting a
/// restricted license. Returns a user-facing route to the explicit pull flow.
pub(crate) fn automatic_pull_license_refusal(resolved: &ResolvedCatalogPull) -> Option<String> {
    match model_install_license_decision(&resolved.license_class, false) {
        ModelInstallLicenseDecision::Allowed => None,
        ModelInstallLicenseDecision::ExplicitAcceptanceRequired
            if resolved.license_class == LicenseClass::Noncommercial =>
        {
            Some(format!(
                "Model '{}' is licensed for non-commercial use only ({}) and cannot be downloaded automatically.\nReview {} then run: openasr pull {} --accept-license",
                resolved.model_id, resolved.license, resolved.license_url, resolved.pull
            ))
        }
        ModelInstallLicenseDecision::ExplicitAcceptanceRequired => Some(format!(
            "Model '{}' requires accepting a vendor license and cannot be downloaded automatically.\nReview {} then run: openasr pull {} --accept-license",
            resolved.model_id, resolved.license_url, resolved.pull
        )),
        ModelInstallLicenseDecision::Unsupported => Some(format!(
            "Model '{}' has an unsupported license class and cannot be downloaded automatically.",
            resolved.model_id
        )),
    }
}

pub(crate) fn list_installed() -> Result<()> {
    let home = openasr_home()?;
    let packs = list_installed_packs(home)?;
    if packs.is_empty() {
        println!("No models installed. Pull one with: openasr pull qwen3-asr-0.6b");
        return Ok(());
    }
    for pack in packs {
        println!(
            "{}\t{}\t{}\t{}",
            pack.pull,
            pack.size_bytes,
            pack.sha256,
            pack.path.display()
        );
    }
    Ok(())
}

pub(crate) fn remove_installed(
    native_execution_services: &Arc<NativeExecutionServices>,
    id: &str,
) -> Result<()> {
    let home = openasr_home()?;
    match remove_model_pack_with_execution_services(
        home,
        id,
        Some(native_execution_services.as_ref()),
    )? {
        Some(pack) => {
            println!("Removed {}", pack.pull);
            Ok(())
        }
        None => bail!("Model pack is not installed: {id}"),
    }
}

/// Ensures an ASR model pack is installed for `model` (the resolved default when
/// `None`), pulling it with a visible, confirmed download when it is missing.
///
/// This is a CLI-only affordance and must never be called from the server. A
/// pull only happens here, gated on `--offline`, an interactive terminal, or an
/// explicit `--yes`. Restricted-license models are refused and routed to the
/// explicit `openasr pull --accept-license` path so download consent cannot
/// become license acceptance. When the model is already installed this answers
/// from on-disk packs with no network access.
pub(crate) fn ensure_asr_model_installed(
    native_execution_services: &Arc<NativeExecutionServices>,
    model: Option<&str>,
    config: &OpenAsrConfig,
    consent: &PullConsent,
) -> Result<()> {
    let home = openasr_home()?;
    let model_ref = model
        .map(str::to_string)
        .or_else(|| config.default_model.clone())
        .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string());
    let packs = list_installed_packs(&home)?;

    // Fast path: installed under its canonical id, answerable with zero network.
    let local_probe = LaunchPackRequest {
        model_ref: &model_ref,
        preference: &QuantPreference::Auto,
        catalog: None,
        host_profile: host_quant_recommendation_profile(),
    };
    if resolve_launch_pack(&packs, &local_probe).is_ok() {
        return Ok(());
    }

    if consent.offline {
        return Err(CliExit::new(
            ExitCode::ModelNotInstalled,
            format!(
                "Model '{model_ref}' is not installed and OpenASR is offline.\nRun: openasr pull {model_ref}"
            ),
        )
        .into());
    }

    // Non-interactive callers (CI, pipes, no TTY) without --yes must never touch
    // the network: fail closed HERE, before loading the catalog. This keeps the
    // promise honest (no silent download) and keeps tests/scripts from hanging on
    // a catalog fetch they can never confirm.
    if !consent.assume_yes && !consent::is_interactive() {
        return Err(CliExit::new(
            ExitCode::ModelNotInstalled,
            format!(
                "Model '{model_ref}' is not installed.\nRun: openasr pull {model_ref}   (or pass --yes to pull non-interactively)"
            ),
        )
        .into());
    }

    // Now we need the catalog (cache/embedded-first) -- for alias resolution and
    // to resolve the pull. Loading it only here keeps a declined/installed run
    // from contacting project infrastructure.
    let catalog = load_model_catalog(None, &home)?;
    let catalog_probe = LaunchPackRequest {
        model_ref: &model_ref,
        preference: &QuantPreference::Auto,
        catalog: Some(&catalog),
        host_profile: host_quant_recommendation_profile(),
    };
    if resolve_launch_pack(&packs, &catalog_probe).is_ok() {
        return Ok(());
    }

    // Pin the bootstrap quant for the built-in default so a newcomer's first
    // download is bounded; an explicit `openasr pull` keeps the full ladder.
    let pinned_quant = (model_ref == DEFAULT_MODEL_ID && !model_ref.contains(':'))
        .then(|| DEFAULT_MODEL_BOOTSTRAP_QUANT.to_string());
    let pull_request = CatalogPullRequest {
        reference: model_ref.clone(),
        quant: pinned_quant,
        size: None,
    };
    let resolved = resolve_catalog_pull_with_profile(
        &catalog,
        &pull_request,
        Some(host_quant_recommendation_profile()),
    )
    .map_err(|error| {
        CliExit::new(
            ExitCode::ModelNotInstalled,
            format!("Could not resolve model '{model_ref}': {error}"),
        )
    })?;

    if let Some(message) = automatic_pull_license_refusal(&resolved) {
        return Err(CliExit::new(ExitCode::ModelNotInstalled, message).into());
    }

    let disclosure = format!(
        "Model '{}' ({}) is not installed.\n  download: {:.0} MB from huggingface.co (catalog index from catalog.openasr.org; both observe your IP)\n  license:  {}",
        resolved.pull,
        resolved.model_id,
        resolved.size_bytes as f64 / 1_000_000.0,
        resolved.license,
    );

    if consent.assume_yes {
        eprintln!("{disclosure}\nDownloading (confirmed by --yes).");
    } else {
        // Guaranteed interactive here (the non-interactive case failed closed above).
        eprintln!("{disclosure}");
        if !consent::confirm("Download this model now?") {
            return Err(CliExit::new(
                ExitCode::ModelNotInstalled,
                format!("Declined. Model '{model_ref}' was not downloaded."),
            )
            .into());
        }
    }

    let installed = perform_consent_pull(native_execution_services, &resolved, &home, config)
        .map_err(|error| {
            CliExit::new(
                ExitCode::DownloadFailed,
                format!("Download failed: {error}"),
            )
        })?;
    if should_update_default_asr_model(&catalog, &installed.model_id) {
        let preference = QuantPreference::pinned(&installed.quant);
        openasr_core::default_selection::persist(&home, &installed, preference)?;
    }
    Ok(())
}

fn perform_consent_pull(
    native_execution_services: &Arc<NativeExecutionServices>,
    resolved: &ResolvedCatalogPull,
    home: &Path,
    config: &OpenAsrConfig,
) -> Result<InstalledPack> {
    let mut reporter = crate::progress::PullReporter::new(&resolved.pull);
    let progress = |event| reporter.on(event);
    let source_chain = resolve_chain(&config.download_source);
    Ok(PullModelPackRequest::new(resolved, home)
        .execution_services(native_execution_services.as_ref())
        .sources(&source_chain)
        .execute(progress)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog_model(id: &str, kind: openasr_core::CatalogModelKind) -> openasr_core::CatalogModel {
        openasr_core::CatalogModel {
            id: id.to_string(),
            kind,
            capability: (kind == openasr_core::CatalogModelKind::CapabilityPack).then(|| {
                openasr_core::CatalogCapability {
                    feature: openasr_core::CATALOG_FEATURE_SPEAKER_DIARIZATION.to_string(),
                    role: openasr_core::CatalogCapabilityRole::SpeakerEmbedder,
                }
            }),
            experimental: false,
            display_name: id.to_string(),
            family: id.to_string(),
            aliases: Vec::new(),
            pull_alias: None,
            size: "tiny".to_string(),
            languages: vec!["en".to_string()],
            language_mode: None,
            language_default: None,
            source_langs: Vec::new(),
            target_langs: Vec::new(),
            vendor: None,
            license: "MIT".to_string(),
            license_url: "https://example.invalid/license".to_string(),
            license_class: LicenseClass::Permissive,
            hf_repo: format!("OpenASR/{id}"),
            hf_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            public: true,
            min_cli_version: "0.1.0".to_string(),
            min_core_version: None,
            recommended_quant: "q8_0".to_string(),
            pull_recommended: format!("{id}:q8"),
            sort_weight: 0,
            recommended: false,
            upstream_release_date: None,
            speaker_source: None,
            word_timestamp_source: None,
            emits_punctuation: None,
            prose: None,
            prose_locales: None,
            quants: Vec::new(),
        }
    }

    fn resolved_pull(license_class: LicenseClass) -> ResolvedCatalogPull {
        ResolvedCatalogPull {
            requested: "diarizen-large-s80-v2".to_string(),
            model_id: "diarizen-large-s80-v2".to_string(),
            catalog_family_id: "diarizen-segmentation".to_string(),
            display_name: "DiariZen Large-s80-md-v2".to_string(),
            quant: "fp16".to_string(),
            suffix: "fp16".to_string(),
            pull: "diarizen-large-s80-v2:fp16".to_string(),
            filename: "diarizen-large-s80-v2-fp16.oasr".to_string(),
            url: "https://example.invalid/diarizen.oasr".to_string(),
            mirrors: Vec::new(),
            hf_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            sha256: "a".repeat(64),
            size_bytes: 123,
            license: "CC BY-NC 4.0".to_string(),
            license_url: "https://example.invalid/license".to_string(),
            license_class,
        }
    }

    #[test]
    fn restricted_pull_requires_explicit_acceptance_even_for_local_import() {
        let mut resolved = resolved_pull(LicenseClass::Noncommercial);
        let download_error = ensure_explicit_pull_license_acceptance(&resolved, false)
            .expect_err("download needs explicit license confirmation");
        assert!(
            download_error
                .to_string()
                .contains("non-commercial use only")
        );

        let import_error = ensure_explicit_pull_license_acceptance(&resolved, false)
            .expect_err("local import still needs explicit license confirmation");
        assert!(import_error.to_string().contains("--accept-license"));
        ensure_explicit_pull_license_acceptance(&resolved, true)
            .expect("explicit confirmation permits the pull");

        resolved.license_class = LicenseClass::Gated;
        ensure_explicit_pull_license_acceptance(&resolved, false)
            .expect_err("a local gated pack is not proof of license acceptance");
        ensure_explicit_pull_license_acceptance(&resolved, true)
            .expect("explicit confirmation permits a gated local pack");
    }

    #[test]
    fn automatic_pull_refuses_every_restricted_license_class() {
        let noncommercial =
            automatic_pull_license_refusal(&resolved_pull(LicenseClass::Noncommercial))
                .expect("noncommercial auto-pull refusal");
        assert!(noncommercial.contains("non-commercial use only"));
        assert!(noncommercial.contains("--accept-license"));

        assert!(automatic_pull_license_refusal(&resolved_pull(LicenseClass::Gated)).is_some());
        assert!(automatic_pull_license_refusal(&resolved_pull(LicenseClass::Unknown)).is_some());
        assert!(automatic_pull_license_refusal(&resolved_pull(LicenseClass::Permissive)).is_none());
    }

    #[test]
    fn capability_pack_pull_does_not_update_default_asr_model() {
        let catalog = ModelCatalog {
            schema_version: 1,
            generated_at: "2026-06-11T00:00:00Z".to_string(),
            catalog_url: "fixture".to_string(),
            backends: Vec::new(),
            language_labels: std::collections::BTreeMap::new(),
            models: vec![
                catalog_model("moonshine-tiny", openasr_core::CatalogModelKind::AsrModel),
                catalog_model(
                    "redimnet2-b6-cn",
                    openasr_core::CatalogModelKind::CapabilityPack,
                ),
                catalog_model(
                    "hymt2-1.8b",
                    openasr_core::CatalogModelKind::TranslationModel,
                ),
            ],
        };

        assert!(should_update_default_asr_model(&catalog, "moonshine-tiny"));
        assert!(!should_update_default_asr_model(
            &catalog,
            "redimnet2-b6-cn"
        ));
        assert!(!should_update_default_asr_model(&catalog, "hymt2-1.8b"));
    }

    #[test]
    fn non_default_asr_install_status_names_catalog_kind() {
        let catalog = ModelCatalog {
            schema_version: 1,
            generated_at: "2026-06-11T00:00:00Z".to_string(),
            catalog_url: "fixture".to_string(),
            backends: Vec::new(),
            language_labels: std::collections::BTreeMap::new(),
            models: vec![
                catalog_model(
                    "redimnet2-b6-cn",
                    openasr_core::CatalogModelKind::CapabilityPack,
                ),
                catalog_model(
                    "hymt2-1.8b",
                    openasr_core::CatalogModelKind::TranslationModel,
                ),
            ],
        };

        assert_eq!(
            non_default_asr_install_status(&catalog, "redimnet2-b6-cn", "redimnet2-b6-cn:fp16"),
            "Installed capability pack redimnet2-b6-cn:fp16; default ASR model was not changed."
        );
        assert_eq!(
            non_default_asr_install_status(&catalog, "hymt2-1.8b", "hymt2-1.8b:q4km"),
            "Installed translation model hymt2-1.8b:q4km; default ASR model was not changed."
        );
    }
}
