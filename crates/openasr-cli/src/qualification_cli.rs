use std::{env, fs, path::Path, process::Command};

use anyhow::{Context, Result, bail};
use openasr_core::{
    PullProgress, VerifiedQualificationManifest, execute_backend_qualification,
    prepare_backend_qualification_artifacts, verify_and_parse_qualification_manifest,
};

pub(crate) fn run_parent(
    manifest_path: &Path,
    signature_path: &Path,
    manifest_url: &str,
    qualification_home: &Path,
) -> Result<()> {
    let verified = load_verified_manifest(manifest_path, signature_path, manifest_url)?;
    let preparation = prepare_backend_qualification_artifacts(
        &verified,
        qualification_home,
        render_pull_progress,
    )
    .context("qualification artifact preparation failed")?;
    let executable =
        env::current_exe().context("could not resolve current qualification binary")?;
    let manifest = fs::canonicalize(manifest_path).with_context(|| {
        format!(
            "could not canonicalize qualification manifest '{}'",
            manifest_path.display()
        )
    })?;
    let signature = fs::canonicalize(signature_path).with_context(|| {
        format!(
            "could not canonicalize qualification signature '{}'",
            signature_path.display()
        )
    })?;
    let output = Command::new(executable)
        .arg("__openasr-qualification-child")
        .arg("--manifest")
        .arg(manifest)
        .arg("--signature")
        .arg(signature)
        .arg("--manifest-url")
        .arg(manifest_url)
        .arg("--qualification-home")
        .arg(qualification_home)
        .arg("--expected-manifest-sha256")
        .arg(&preparation.manifest_sha256)
        .env("OPENASR_HOME", qualification_home)
        .env_remove("OPENASR_BACKEND_PLUGIN_ID")
        .env_remove("OPENASR_BACKEND_PLUGIN_TARGET")
        .env_remove("OPENASR_CATALOG_URL")
        .env_remove("OPENASR_CATALOG_FILE")
        .env_remove("OPENASR_CATALOG_IDENTITY")
        .env_remove("OPENASR_GGML_BACKEND")
        .output()
        .context("could not launch qualification child")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr)
            .chars()
            .take(8_192)
            .collect::<String>();
        bail!(
            "qualification child failed with exit {}: {}",
            output.status.code().unwrap_or(-1),
            stderr.trim()
        );
    }
    let stdout =
        std::str::from_utf8(&output.stdout).context("qualification child output was not UTF-8")?;
    let value: serde_json::Value =
        serde_json::from_str(stdout).context("qualification child output was not JSON")?;
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

pub(crate) fn run_child(
    manifest_path: &Path,
    signature_path: &Path,
    manifest_url: &str,
    qualification_home: &Path,
    expected_manifest_sha256: &str,
) -> Result<()> {
    let configured_home = env::var_os("OPENASR_HOME")
        .context("qualification child requires OPENASR_HOME to be set by its parent")?;
    let configured_home = fs::canonicalize(configured_home)
        .context("qualification child OPENASR_HOME could not be canonicalized")?;
    let qualification_home = fs::canonicalize(qualification_home)
        .context("qualification home could not be canonicalized in child")?;
    if configured_home != qualification_home {
        bail!("qualification child OPENASR_HOME differs from --qualification-home");
    }
    let verified = load_verified_manifest(manifest_path, signature_path, manifest_url)?;
    if verified.manifest_sha256() != expected_manifest_sha256 {
        bail!("qualification child manifest differs from the parent-prepared identity");
    }
    let evidence =
        execute_backend_qualification(&verified, &qualification_home, render_pull_progress)
            .context("qualification child failed closed")?;
    println!("{}", serde_json::to_string_pretty(&evidence)?);
    Ok(())
}

fn load_verified_manifest(
    manifest_path: &Path,
    signature_path: &Path,
    manifest_url: &str,
) -> Result<VerifiedQualificationManifest> {
    let manifest = fs::read(manifest_path).with_context(|| {
        format!(
            "could not read qualification manifest '{}'",
            manifest_path.display()
        )
    })?;
    let signature = fs::read(signature_path).with_context(|| {
        format!(
            "could not read qualification signature '{}'",
            signature_path.display()
        )
    })?;
    verify_and_parse_qualification_manifest(&manifest, &signature, manifest_url)
        .context("qualification manifest did not verify against the production trust root")
}

fn render_pull_progress(progress: PullProgress) {
    match progress {
        PullProgress::UsingInstalled { path } | PullProgress::Installed { path } => {
            eprintln!("qualification artifact ready: {}", path.display());
        }
        PullProgress::DownloadStarted {
            bytes_total,
            resume_from,
        } => eprintln!("qualification download: total={bytes_total} resume_from={resume_from}"),
        PullProgress::Downloading {
            bytes_done,
            bytes_total,
        } => eprintln!("qualification download: {bytes_done}/{bytes_total}"),
        PullProgress::Verifying { bytes_done } => {
            eprintln!("qualification verify: {bytes_done} bytes")
        }
    }
}
