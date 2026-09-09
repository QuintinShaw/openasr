//! Explicit real-family qualification producer.
//!
//! Generic `bench-receipt` stays evidence-free. This command runs one native
//! cold request and one same-process reuse request, then binds both to
//! `evidence.v1` using caller-supplied matrix/artifact identity.

use std::{fs, path::Path, sync::Arc};

use anyhow::{Context, Result, bail};
use openasr_core::{
    BackendKind, NativeExecutionServices, RealFamilyEvidenceBinding, RealFamilyTraceArtifacts,
    ShortAudioArtifactIdentity, ShortAudioTopKSummary, bind_real_family_evidence, sha256_hex_bytes,
};

use crate::bench_receipt_cli::{
    CollectedShortAudio, ShortAudioReceiptOptions, bench_receipt_short_audio,
};

pub(crate) fn run(
    native_execution_services: &Arc<NativeExecutionServices>,
    model: Option<&str>,
    audio: &Path,
    device: &str,
    model_pack: Option<&Path>,
    binding_path: &Path,
    out_dir: &Path,
    core_commit: Option<&str>,
    ffmpeg_bin: Option<std::path::PathBuf>,
) -> Result<()> {
    let binding: RealFamilyEvidenceBinding =
        serde_json::from_str(&fs::read_to_string(binding_path).with_context(|| {
            format!(
                "could not read real-family binding {}",
                binding_path.display()
            )
        })?)
        .context("real-family binding JSON is invalid")?;
    fs::create_dir_all(out_dir).with_context(|| {
        format!(
            "could not create real-family output directory {}",
            out_dir.display()
        )
    })?;
    let git_cwd = std::env::current_dir().ok();
    let dummy_out = out_dir.join(".diagnostic-receipt.json");
    let cold = collect(
        native_execution_services,
        model,
        audio,
        device,
        model_pack,
        &dummy_out,
        0,
        core_commit,
        ffmpeg_bin.clone(),
        git_cwd.as_deref(),
    )?;
    let reuse = collect(
        native_execution_services,
        model,
        audio,
        device,
        model_pack,
        &dummy_out,
        1,
        core_commit,
        ffmpeg_bin,
        git_cwd.as_deref(),
    )?;
    write_bound_pair(&binding, &cold, out_dir, "cold")?;
    write_bound_pair(&binding, &reuse, out_dir, "reuse")?;
    let _ = fs::remove_file(dummy_out);
    eprintln!(
        "Wrote real-family evidence.v1 receipts for cold and reuse into {}",
        out_dir.display()
    );
    Ok(())
}

fn collect(
    native_execution_services: &Arc<NativeExecutionServices>,
    model: Option<&str>,
    audio: &Path,
    device: &str,
    model_pack: Option<&Path>,
    dummy_out: &Path,
    warmup_runs: usize,
    core_commit: Option<&str>,
    ffmpeg_bin: Option<std::path::PathBuf>,
    git_cwd: Option<&Path>,
) -> Result<CollectedShortAudio> {
    bench_receipt_short_audio(
        native_execution_services,
        ShortAudioReceiptOptions {
            model,
            audio,
            backend_kind: BackendKind::Native,
            device,
            model_pack,
            out: dummy_out,
            runs: 1,
            warmup_runs,
            core_commit,
            scope: "short-audio-gate",
            ffmpeg_bin,
            git_cwd,
            trace_out: None,
            logits_out: None,
            write_outputs: false,
        },
    )
}

fn write_bound_pair(
    binding: &RealFamilyEvidenceBinding,
    collected: &CollectedShortAudio,
    out_dir: &Path,
    mode: &str,
) -> Result<()> {
    let token_jsonl = collected
        .token_trace_jsonl
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("native real-family run did not produce a token trace"))?;
    let token_label = format!("token-{mode}.jsonl");
    let token_path = out_dir.join(&token_label);
    fs::write(&token_path, token_jsonl)
        .with_context(|| format!("could not write token trace {}", token_path.display()))?;
    if let Ok(json) = collected.receipt.to_pretty_json() {
        let _ = fs::write(
            out_dir.join(format!("diagnostic-{mode}.json")),
            format!("{json}\n"),
        );
    }
    let traces = RealFamilyTraceArtifacts {
        token_trace: hashed_artifact(&token_label, token_jsonl.as_bytes()),
        logits: match collected.logits_jsonl.as_deref() {
            Some(jsonl) => {
                let label = format!("logits-{mode}.jsonl");
                let path = out_dir.join(&label);
                fs::write(&path, jsonl)
                    .with_context(|| format!("could not write logits {}", path.display()))?;
                Some(hashed_artifact(&label, jsonl.as_bytes()))
            }
            None => None,
        },
        top_k: parse_top_k(token_jsonl)?,
        top1_top2_margin: parse_margin(token_jsonl),
    };
    let bound = bind_real_family_evidence(collected.receipt.clone(), binding, &traces).map_err(
        |error| {
            anyhow::anyhow!(
                "diagnostic receipt could not be bound as real-family evidence: {error}"
            )
        },
    )?;
    write_receipt(
        out_dir.join(format!("placement-{mode}.json")),
        &bound.placement,
    )?;
    write_receipt(
        out_dir.join(format!("token-{mode}.json")),
        &bound.token_transcript,
    )?;
    Ok(())
}

fn hashed_artifact(label: &str, bytes: &[u8]) -> ShortAudioArtifactIdentity {
    ShortAudioArtifactIdentity {
        label: label.to_string(),
        sha256: sha256_hex_bytes(bytes),
        size_bytes: Some(bytes.len() as u64),
    }
}

fn parse_top_k(jsonl: &str) -> Result<Vec<ShortAudioTopKSummary>> {
    for line in jsonl.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if value.get("event").and_then(|event| event.as_str()) != Some("top_k") {
            continue;
        }
        let Some(items) = value.get("items").and_then(|items| items.as_array()) else {
            continue;
        };
        let mut top_k = Vec::new();
        for item in items {
            let Some(token_id) = item.get("token_id").and_then(|value| value.as_u64()) else {
                continue;
            };
            let Some(score) = item.get("value").and_then(|value| value.as_f64()) else {
                continue;
            };
            top_k.push(ShortAudioTopKSummary {
                token_id: token_id as u32,
                value: score,
            });
        }
        if !top_k.is_empty() {
            return Ok(top_k);
        }
    }
    bail!("native token trace has no top-k summary");
}

fn parse_margin(jsonl: &str) -> Option<f64> {
    jsonl.lines().find_map(|line| {
        let value: serde_json::Value = serde_json::from_str(line).ok()?;
        if value.get("event").and_then(|event| event.as_str()) != Some("top_k") {
            return None;
        }
        value
            .get("top1_top2_margin")
            .and_then(|margin| margin.as_f64())
    })
}

fn write_receipt(
    path: std::path::PathBuf,
    receipt: &openasr_core::ShortAudioReceipt,
) -> Result<()> {
    let json = receipt
        .to_pretty_json()
        .context("could not serialize bound real-family receipt")?;
    fs::write(&path, format!("{json}\n"))
        .with_context(|| format!("could not write {}", path.display()))
}
