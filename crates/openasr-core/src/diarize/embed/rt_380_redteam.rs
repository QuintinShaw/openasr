//! Red-team falsifiers for PR #380 WeSpeaker promises.
//!
//! Host-local cosine goldens stay opt-in. The remaining tests lock served
//! WeSpeaker fail-closed copy, stream preference apply, and card wording.

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

    /// Host-local cosine goldens stay opt-in; default nextest must not run them.
    #[test]
    fn rt_380_wespeaker_pytorch_goldens_are_not_ignored() {
        let src = read_repo_file("crates/openasr-core/src/diarize/embed/wespeaker/mod.rs");
        for fn_name in [
            "fn wespeaker_resnet_matches_pytorch_on_cpu",
            "fn wespeaker_resnet_matches_pytorch_on_metal",
        ] {
            let prefix = prefix_before(&src, fn_name);
            assert!(
                prefix.contains("#[ignore"),
                "{fn_name} must stay host-local (OPENASR_WESPEAKER_SPIKE_ROOT), not a default CI gate"
            );
        }
    }

    /// Host-local goldens live under OPENASR_WESPEAKER_SPIKE_ROOT, not in-tree CI.
    #[test]
    fn rt_380_wespeaker_golden_vectors_are_committed() {
        let src = read_repo_file("crates/openasr-core/src/diarize/embed/wespeaker/mod.rs");
        assert!(
            src.contains("OPENASR_WESPEAKER_SPIKE_ROOT"),
            "WeSpeaker cosine goldens must remain a host-local spike-root gate"
        );
    }

    /// Default CI must not export the host-local WeSpeaker spike root.
    #[test]
    fn rt_380_ci_exports_wespeaker_spike_root() {
        let ci = read_repo_file(".github/workflows/ci.yml");
        assert!(
            !ci.contains("OPENASR_WESPEAKER_SPIKE_ROOT"),
            "CI must not turn WeSpeaker cosine goldens into a default gate"
        );
    }

    /// If all four depths are required, the golden runner cannot skip a missing
    /// depth and still pass. Otherwise one pack proves the whole family.
    #[test]
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
    fn rt_380_native_diarize_probe_honors_selected_embedder() {
        let src = read_repo_file("crates/openasr-core/src/api/backend/native_transcribe.rs");
        let marker = "crate::diarize::embed::embedder_pack_installed()";
        let prefix = prefix_before(&src, marker);
        assert!(
            prefix.contains("voice_id_embedder"),
            "native diarize still probes ReDimNet via embedder_pack_installed() before the selected WeSpeaker pack"
        );
    }

    /// Cards may mention cosine >= 0.999 only as a host-local spike-root gate.
    #[test]
    fn rt_380_wespeaker_cards_do_not_claim_ci_cosine_without_a_default_gate() {
        for name in [
            "wespeaker-voxceleb-resnet34-lm",
            "wespeaker-voxceleb-resnet152-lm",
            "wespeaker-voxceleb-resnet221-lm",
            "wespeaker-voxceleb-resnet293-lm",
        ] {
            let card = read_repo_file(&format!("tooling/publish-model/cards/{name}.toml"));
            if card.contains("0.999") {
                assert!(
                    card.contains("OPENASR_WESPEAKER_SPIKE_ROOT")
                        && (card.contains("not a default CI gate")
                            || card.contains("不是默认 CI 门")),
                    "{name} card must not claim a CI cosine gate: {card}"
                );
            }
        }
    }
}
