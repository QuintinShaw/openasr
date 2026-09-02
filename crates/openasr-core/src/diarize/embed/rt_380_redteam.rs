//! Red-team falsifiers for PR #380 WeSpeaker promises.
//!
//! Each test encodes a promised external observation. They stay ignored so the
//! default nextest gate stays green; `--run-ignored` must fail on current main.

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repo root")
    }

    fn read_repo_file(relative: &str) -> String {
        std::fs::read_to_string(repo_root().join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
    }

    fn prefix_before(src: &str, marker: &str) -> String {
        let idx = src
            .find(marker)
            .unwrap_or_else(|| panic!("missing marker {marker}"));
        src[idx.saturating_sub(240)..idx].to_string()
    }

    /// If cosine >= 0.999 is a default-nextest proof, the CPU/Metal goldens are
    /// not `#[ignore]`. Otherwise they stay host-local and CI never runs them.
    #[test]
    #[ignore = "redteam: pr-380"]
    fn rt_380_wespeaker_pytorch_goldens_are_not_ignored() {
        let src = read_repo_file("crates/openasr-core/src/diarize/embed/wespeaker/mod.rs");
        for fn_name in [
            "fn wespeaker_resnet_matches_pytorch_on_cpu",
            "fn wespeaker_resnet_matches_pytorch_on_metal",
        ] {
            let prefix = prefix_before(&src, fn_name);
            assert!(
                !prefix.contains("#[ignore"),
                "{fn_name} is #[ignore], so default cargo nextest run --workspace never proves cosine >= 0.999"
            );
        }
    }

    /// If all four depths are golden-gated in CI, committed reference vectors
    /// exist on disk. Otherwise dump_reference.py goldens stay uncommitted.
    #[test]
    #[ignore = "redteam: pr-380"]
    fn rt_380_wespeaker_golden_vectors_are_committed() {
        let root = repo_root();
        let candidates = [
            root.join("crates/openasr-core/tests/fixtures/wespeaker"),
            root.join("tooling/wespeaker/golden"),
            root.join("tooling/wespeaker/golden-152"),
            root.join("tooling/wespeaker/golden-221"),
            root.join("tooling/wespeaker/golden-293"),
        ];
        let mut embeddings = Vec::new();
        for dir in candidates {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let name = entry.file_name();
                if name.to_string_lossy().ends_with(".embedding.npy") {
                    embeddings.push(entry.path());
                }
            }
        }
        assert!(
            !embeddings.is_empty(),
            "promised cosine >= 0.999 for resnet 34/152/221/293 requires committed *.embedding.npy goldens"
        );
    }

    /// If CI executes the host-local goldens, the workspace workflow must export
    /// OPENASR_WESPEAKER_SPIKE_ROOT. Otherwise the default nextest step skips them.
    #[test]
    #[ignore = "redteam: pr-380"]
    fn rt_380_ci_exports_wespeaker_spike_root() {
        let ci = read_repo_file(".github/workflows/ci.yml");
        assert!(
            ci.contains("OPENASR_WESPEAKER_SPIKE_ROOT"),
            "Workspace tests run cargo nextest run --workspace with no WeSpeaker spike env, so cosine >= 0.999 is not a CI gate"
        );
    }

    /// If all four depths are required, the golden runner cannot skip a missing
    /// depth and still pass. Otherwise one pack proves the whole family.
    #[test]
    #[ignore = "redteam: pr-380"]
    fn rt_380_wespeaker_all_four_depths_are_required() {
        let src = read_repo_file("crates/openasr-core/src/diarize/embed/wespeaker/mod.rs");
        assert!(
            !src.contains("skipping depth"),
            "golden runner skips missing depths, so cosine >= 0.999 is not proven for 34/152/221/293 together"
        );
    }

    /// If 0.1.38 binaries cannot run wespeaker-resnet, staged catalog floors
    /// must not claim that release. Equal floors would mark a future public
    /// projection Available on the shipped 0.1.38 binary.
    #[test]
    #[ignore = "redteam: pr-380"]
    fn rt_380_wespeaker_min_core_version_exceeds_shipped_0_1_38() {
        let text = read_repo_file("tooling/publish-model/models-core.toml");
        for model_id in [
            "wespeaker-voxceleb-resnet34-lm",
            "wespeaker-voxceleb-resnet152-lm",
            "wespeaker-voxceleb-resnet221-lm",
            "wespeaker-voxceleb-resnet293-lm",
        ] {
            let header = format!("[\"{model_id}\"]");
            let start = text
                .find(&header)
                .unwrap_or_else(|| panic!("missing {model_id} table"));
            let rest = &text[start..];
            let end = rest[header.len()..]
                .find("\n[")
                .map(|idx| header.len() + idx)
                .unwrap_or(rest.len());
            let table = &rest[..end];
            let floor = table
                .lines()
                .find_map(|line| {
                    line.trim()
                        .strip_prefix("min_core_version")
                        .and_then(|rest| rest.split('"').nth(1))
                })
                .unwrap_or("");
            assert_ne!(
                floor, "0.1.38",
                "{model_id} min_core_version={floor} would tell a 0.1.38 binary the pack is Available"
            );
        }
    }

    /// If file-stream jobs honor the persisted WeSpeaker preference, the stream
    /// handler copies apply_transcription_preferences. Otherwise stream silently
    /// keeps NativeAsrOfflineRequest's ReDimNet2 default.
    #[test]
    #[ignore = "redteam: pr-380"]
    fn rt_380_stream_transcription_applies_voice_id_embedder() {
        let src = read_repo_file("crates/openasr-server/src/realtime/mod.rs");
        let start = src
            .find("pub(crate) async fn stream_transcription")
            .expect("stream_transcription");
        let rest = &src[start..];
        let end = rest[1..]
            .find("\npub(crate) async fn ")
            .map(|idx| idx + 1)
            .unwrap_or(rest.len());
        let body = &rest[..end];
        assert!(
            body.contains("apply_transcription_preferences"),
            "POST /v1/audio/transcriptions?stream=true never copies voice_id_embedder from config.json"
        );
    }

    /// If an explicit WeSpeaker preference fail-closes on its own pack, the
    /// ReDimNet-only capability probe cannot run first.
    #[test]
    #[ignore = "redteam: pr-380"]
    fn rt_380_native_diarize_probe_honors_selected_embedder() {
        let src = read_repo_file("crates/openasr-core/src/api/backend/native_transcribe.rs");
        let marker = "crate::diarize::embed::embedder_pack_installed()";
        let prefix = prefix_before(&src, marker);
        assert!(
            prefix.contains("voice_id_embedder"),
            "native diarize still probes ReDimNet via embedder_pack_installed() before the selected WeSpeaker pack"
        );
    }

    /// Model cards claim CPU/Metal cosine >= 0.999 as packaging evidence. If
    /// that is true in the default suite, the card files and the un-ignored
    /// golden must agree. Cards currently overclaim a host-local ignored test.
    #[test]
    #[ignore = "redteam: pr-380"]
    fn rt_380_wespeaker_cards_do_not_claim_ci_cosine_without_a_default_gate() {
        let card =
            read_repo_file("tooling/publish-model/cards/wespeaker-voxceleb-resnet34-lm.toml");
        let src = read_repo_file("crates/openasr-core/src/diarize/embed/wespeaker/mod.rs");
        let cpu_prefix = prefix_before(&src, "fn wespeaker_resnet_matches_pytorch_on_cpu");
        let claims_parity = card.contains("cosine") && card.contains("0.999");
        if claims_parity {
            assert!(
                !cpu_prefix.contains("#[ignore"),
                "card claims cosine >= 0.999 CPU/Metal while the only golden is #[ignore]"
            );
        }
    }
}
