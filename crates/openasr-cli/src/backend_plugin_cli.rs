use std::time::Duration;

use anyhow::{Context, Result};
use openasr_core::{
    BackendHostAbi, CatalogBackendVendor, PullProgress, activate_installed_backend_pack_auto,
    backend_artifact_fingerprint, backend_pack_download_plan, backend_plugin_status,
    deactivate_backend_pack, gc_backend_store, install_and_activate_backend_pack,
    install_and_activate_backend_provider, install_backend_pack_from_catalog, load_model_catalog,
    openasr_home, resolve_catalog_backend_pull, resolve_catalog_backend_pull_for_host,
};
use serde_json::json;

use crate::{catalog_cli::load_cli_model_catalog, cli_args::BackendPluginCommand};

pub(crate) fn backend_plugin_command(command: BackendPluginCommand) -> Result<()> {
    let home = openasr_home().context("Could not resolve OpenASR home")?;
    match command {
        BackendPluginCommand::Status => {
            let status = backend_plugin_status(&home)?;
            println!("{}", serde_json::to_string(&status)?);
        }
        BackendPluginCommand::ResolveProvider { provider } => {
            let catalog = load_backend_catalog(&home)?;
            let vendor = provider_vendor(&provider);
            let resolved = resolve_catalog_backend_pull_for_host(
                &catalog,
                vendor,
                &BackendHostAbi::current(),
            )?;
            let plan = backend_pack_download_plan(&home, &resolved)?;
            println!(
                "{}",
                json!({
                    "schema_version": 1,
                    "event": "resolved",
                    "backend_id": resolved.backend_id,
                    "vendor": provider,
                    "artifact_fingerprint": backend_artifact_fingerprint(&resolved),
                    "host_abi_fingerprint": resolved.host_abi.fingerprint,
                    "size_bytes": plan.total_bytes,
                    "plugin_size_bytes": plan.plugin_bytes,
                    "vendor_size_bytes": plan.vendor_bytes,
                    "required_download_size_bytes": plan.required_download_bytes,
                    "required_plugin_download_size_bytes": plan.required_plugin_bytes,
                    "required_vendor_download_size_bytes": plan.required_vendor_bytes,
                })
            );
        }
        BackendPluginCommand::Install { backend_id } => {
            let catalog = load_backend_catalog(&home)?;
            let requested = resolve_catalog_backend_pull(&catalog, &backend_id)?;
            let installed =
                install_backend_pack_from_catalog(&catalog, &backend_id, &home, print_progress)?;
            println!(
                "{}",
                json!({
                    "schema_version": 1,
                    "event": "installed",
                    "backend_id": installed.backend_id,
                    "vendor": requested.vendor,
                    "version": installed.version,
                    "artifact_fingerprint": installed.artifact_fingerprint,
                    "host_abi_fingerprint": requested.host_abi.fingerprint,
                    "size_bytes": requested.files.iter().map(|file| file.size_bytes).sum::<u64>(),
                })
            );
        }
        BackendPluginCommand::Activate { backend_id } => {
            let catalog = load_backend_catalog(&home)?;
            let activated = activate_installed_backend_pack_auto(&catalog, &backend_id, &home)?;
            println!("{}", serde_json::to_string(&activated)?);
        }
        BackendPluginCommand::InstallActivate { backend_id } => {
            let catalog = load_backend_catalog(&home)?;
            let activated =
                install_and_activate_backend_pack(&catalog, &backend_id, &home, print_progress)?;
            println!("{}", serde_json::to_string(&activated)?);
        }
        BackendPluginCommand::InstallActivateProvider { provider } => {
            let catalog = load_backend_catalog(&home)?;
            let vendor = provider_vendor(&provider);
            let activated =
                install_and_activate_backend_provider(&catalog, vendor, &home, print_progress)?;
            println!("{}", serde_json::to_string(&activated)?);
        }
        BackendPluginCommand::Deactivate => {
            deactivate_backend_pack(&home)?;
            println!("{}", json!({"schema_version": 1, "event": "deactivated"}));
        }
        BackendPluginCommand::Gc {
            keep_backend_ids,
            min_age_seconds,
        } => {
            let report = gc_backend_store(
                &home,
                keep_backend_ids,
                Some(Duration::from_secs(min_age_seconds)),
            )?;
            println!("{}", serde_json::to_string(&report)?);
        }
    }
    Ok(())
}

fn load_backend_catalog(home: &std::path::Path) -> Result<openasr_core::ModelCatalog> {
    load_cli_model_catalog(home)?
        .map(Ok)
        .unwrap_or_else(|| load_model_catalog(None, home).map_err(Into::into))
}

fn provider_vendor(provider: &str) -> CatalogBackendVendor {
    match provider {
        "cuda" => CatalogBackendVendor::Cuda,
        "hip" => CatalogBackendVendor::Hip,
        _ => unreachable!("clap validates provider"),
    }
}

fn print_progress(progress: PullProgress) {
    let value = match progress {
        PullProgress::UsingInstalled { .. } => json!({"event": "using_installed"}),
        PullProgress::DownloadStarted {
            bytes_total,
            resume_from,
        } => json!({
            "event": "download_started",
            "bytes_total": bytes_total,
            "resume_from": resume_from,
        }),
        PullProgress::Downloading {
            bytes_done,
            bytes_total,
        } => json!({
            "event": "downloading",
            "bytes_done": bytes_done,
            "bytes_total": bytes_total,
        }),
        PullProgress::Verifying { bytes_done } => {
            json!({"event": "verifying", "bytes_done": bytes_done})
        }
        PullProgress::Installed { .. } => json!({"event": "installed_bytes"}),
    };
    println!("{value}");
}
