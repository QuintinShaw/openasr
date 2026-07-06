use std::path::Path;

use anyhow::{Result, bail};
use openasr_core::{
    CatalogPullRequest, DEFAULT_MODEL_BOOTSTRAP_QUANT, DEFAULT_MODEL_ID, DownloadSourcePref,
    InstalledPack, LaunchPackRequest, LicenseClass, ModelCatalog, OpenAsrConfig,
    PullModelPackRequest, QuantPreference, ResolvedCatalogPull, host_quant_recommendation_profile,
    install_model_pack_from_path, list_installed_packs, load_config, load_model_catalog,
    openasr_home, persist_default_pack_pointer, remove_model_pack,
    resolve_catalog_pull_with_profile, resolve_chain, resolve_launch_pack,
    save_default_model_selection,
};

use crate::PullCommandOptions;
use crate::consent::{self, CliExit, ExitCode, PullConsent};

pub(crate) fn pull(options: PullCommandOptions<'_>) -> Result<()> {
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

    if matches!(resolved.license_class, LicenseClass::Gated)
        && !options.accept_license
        && options.from.is_none()
    {
        bail!(
            "Model '{}' requires vendor license acceptance before download.\nOpen vendor site: {}\nThen rerun with --accept-license or --from <local-pack>.",
            resolved.model_id,
            resolved.license_url
        );
    }

    let mut reporter = crate::progress::PullReporter::new(&resolved.pull);
    let progress = |event| reporter.on(event);

    let source_pref = match options.source {
        Some(source) => DownloadSourcePref::parse_env_value(source)
            .ok_or_else(|| anyhow::anyhow!("Unsupported download source '{source}'"))?,
        None => config.download_source.clone(),
    };
    let source_chain = resolve_chain(&source_pref);

    let installed = if let Some(path) = options.from {
        install_model_pack_from_path(&resolved, path, &home, progress)?
    } else {
        PullModelPackRequest::new(&resolved, &home)
            .sources(&source_chain)
            .execute(progress)?
    };

    let preference = if options.quant.is_some() || options.reference.contains(':') {
        QuantPreference::pinned(&installed.quant)
    } else {
        QuantPreference::Auto
    };
    if should_update_default_asr_model(&catalog, &installed.model_id) {
        save_default_model_selection(&home, installed.model_id.clone(), preference)?;
        persist_default_pack_pointer(&home, &installed)?;
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

pub(crate) fn remove_installed(id: &str) -> Result<()> {
    let home = openasr_home()?;
    match remove_model_pack(home, id)? {
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
/// explicit `--yes`. Gated-license models are refused and routed to the explicit
/// `openasr pull --accept-license` path so consent cannot become a license
/// bypass. When the model is already installed this answers from on-disk packs
/// with no network access.
pub(crate) fn ensure_asr_model_installed(
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

    if matches!(resolved.license_class, LicenseClass::Gated) {
        return Err(CliExit::new(
            ExitCode::ModelNotInstalled,
            format!(
                "Model '{}' requires accepting a vendor license before download.\nReview {} then run: openasr pull {} --accept-license",
                resolved.model_id, resolved.license_url, model_ref
            ),
        )
        .into());
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

    let installed = perform_consent_pull(&resolved, &home, config).map_err(|error| {
        CliExit::new(
            ExitCode::DownloadFailed,
            format!("Download failed: {error}"),
        )
    })?;
    if should_update_default_asr_model(&catalog, &installed.model_id) {
        let preference = QuantPreference::pinned(&installed.quant);
        save_default_model_selection(&home, installed.model_id.clone(), preference)?;
        persist_default_pack_pointer(&home, &installed)?;
    }
    Ok(())
}

fn perform_consent_pull(
    resolved: &ResolvedCatalogPull,
    home: &Path,
    config: &OpenAsrConfig,
) -> Result<InstalledPack> {
    let mut reporter = crate::progress::PullReporter::new(&resolved.pull);
    let progress = |event| reporter.on(event);
    let source_chain = resolve_chain(&config.download_source);
    Ok(PullModelPackRequest::new(resolved, home)
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
            emits_punctuation: None,
            prose: None,
            prose_locales: None,
            quants: Vec::new(),
        }
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
                    "wespeaker-voxceleb-resnet34-lm",
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
            "wespeaker-voxceleb-resnet34-lm"
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
                    "wespeaker-voxceleb-resnet34-lm",
                    openasr_core::CatalogModelKind::CapabilityPack,
                ),
                catalog_model(
                    "hymt2-1.8b",
                    openasr_core::CatalogModelKind::TranslationModel,
                ),
            ],
        };

        assert_eq!(
            non_default_asr_install_status(
                &catalog,
                "wespeaker-voxceleb-resnet34-lm",
                "wespeaker-voxceleb-resnet34-lm:f32"
            ),
            "Installed capability pack wespeaker-voxceleb-resnet34-lm:f32; default ASR model was not changed."
        );
        assert_eq!(
            non_default_asr_install_status(&catalog, "hymt2-1.8b", "hymt2-1.8b:q4km"),
            "Installed translation model hymt2-1.8b:q4km; default ASR model was not changed."
        );
    }
}
