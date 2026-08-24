use std::time::Duration;

use anyhow::{Context, Result};
use openasr_core::{
    CatalogBackendVendor, PullProgress, activate_installed_backend_pack_auto,
    backend_plugin_status, clear_backend_qualification, deactivate_backend_pack,
    describe_backend_provider, gc_backend_store, install_and_activate_backend_pack,
    install_and_activate_backend_provider, install_backend_pack_from_catalog,
    installed_backend_protected_bytes, load_model_catalog, openasr_home,
    prepare_backend_pack_for_qualification, prepare_backend_provider_for_live_device,
    resolve_catalog_backend_pull, sha256_file,
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
        BackendPluginCommand::DescribeProvider { provider } => {
            let catalog = load_backend_catalog(&home)?;
            let description =
                describe_backend_provider(&catalog, provider_vendor(&provider), &home)?;
            println!("{}", serde_json::to_string(&description)?);
        }
        BackendPluginCommand::PrepareProvider { provider } => {
            let catalog = load_backend_catalog(&home)?;
            match prepare_backend_provider_for_live_device(
                &catalog,
                provider_vendor(&provider),
                &home,
                print_progress,
            ) {
                Ok(prepared) => println!(
                    "{}",
                    terminal_record("prepared", serde_json::to_value(prepared)?)
                ),
                Err(error) => return provider_failure(error),
            }
        }
        BackendPluginCommand::Install { backend_id } => {
            let catalog = load_backend_catalog(&home)?;
            let requested = resolve_catalog_backend_pull(&catalog, &backend_id)?;
            let device_target = match (requested.vendor, requested.targets.as_slice()) {
                (_, [target]) => Some(target.clone()),
                (CatalogBackendVendor::Vulkan, []) => None,
                _ => anyhow::bail!(
                    "install-only requires one target-scoped pack or one generic Vulkan pack"
                ),
            };
            let installed =
                install_backend_pack_from_catalog(&catalog, &backend_id, &home, print_progress)?;
            let protected_bytes = installed_backend_protected_bytes(&requested, &home)?;
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
                    "device_target": device_target,
                    "size_bytes": requested.files.iter().map(|file| file.size_bytes).sum::<u64>(),
                    "protected_bytes": protected_bytes,
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
            match install_and_activate_backend_provider(&catalog, vendor, &home, print_progress) {
                Ok(activated) => println!(
                    "{}",
                    terminal_record("activated", serde_json::to_value(activated)?)
                ),
                Err(error) => return provider_failure(error),
            }
        }
        BackendPluginCommand::PrepareQualification {
            backend_id,
            device_target,
            scope,
        } => {
            let catalog = load_backend_catalog(&home)?;
            let (_, catalog_sha256) = sha256_file(home.join("catalog.json"))
                .context("Could not hash the verified cached qualification catalog")?;
            let prepared = prepare_backend_pack_for_qualification(
                &catalog,
                &backend_id,
                device_target.as_deref(),
                &catalog_sha256,
                &scope,
                &home,
                print_progress,
            )?;
            println!("{}", serde_json::to_string(&prepared)?);
        }
        BackendPluginCommand::ClearQualification { scope } => {
            clear_backend_qualification(&home, &scope)?;
            println!(
                "{}",
                json!({"schema_version": 1, "event": "qualification_cleared"})
            );
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
        "vulkan" => CatalogBackendVendor::Vulkan,
        _ => unreachable!("clap validates provider"),
    }
}

fn terminal_record(event: &'static str, value: serde_json::Value) -> serde_json::Value {
    let mut object = value.as_object().cloned().unwrap_or_default();
    object.insert("schema_version".to_string(), json!(1));
    object.insert("event".to_string(), json!(event));
    serde_json::Value::Object(object)
}

fn provider_failure(error: openasr_core::BackendActivationError) -> Result<()> {
    println!(
        "{}",
        json!({
            "schema_version": 1,
            "event": "failed",
            "class": error.machine_failure_class(),
            "code": error.machine_failure_code(),
            "message": error.to_string(),
        })
    );
    Err(anyhow::anyhow!("backend provider command failed"))
}

fn print_progress(progress: PullProgress) {
    let mut value = match progress {
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
    if let Some(object) = value.as_object_mut() {
        object.insert("schema_version".to_string(), json!(1));
    }
    println!("{value}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_records_always_carry_the_machine_protocol_envelope() {
        let record = terminal_record(
            "prepared",
            json!({"backend_id":"cuda-windows-sm_86","schema_version":999}),
        );
        assert_eq!(record["schema_version"], 1);
        assert_eq!(record["event"], "prepared");
        assert_eq!(record["backend_id"], "cuda-windows-sm_86");
    }
}
