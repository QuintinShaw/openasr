//! Generic resolution of an installed optional capability-pack file (a
//! `.oasr`/`.safetensors` support model that augments a family's own decode
//! path, e.g. the ReDimNet2-B6 speaker-embedder, the pyannote segmenter, the
//! Qwen3-ForcedAligner word-timestamp refiner, or FireRedPunc) from the resolved
//! model-pack storage root (see `config::models_dir` -- honors an
//! `OPENASR_MODELS_DIR`/`config.models_dir` override, defaulting to
//! `openasr_home()/models/`). Extracted from `diarize::pack` so each
//! capability-pack family does not duplicate the same lookup -- infrastructure
//! that decides where an installed pack lives stays model-agnostic; only the env
//! var name and the model-id hint are per-feature.
//!
//! # Which layout is authoritative
//!
//! Installed packs live as `refs/<model_id>/<quant>.json` naming an object under
//! `objects/sha256/`. That is the only layout an install writes, so it is what
//! this module consults first, through the same `InstalledModelStore` reader the
//! rest of the codebase uses -- there is deliberately no second scanner here.
//!
//! The pre-content-store layout (`<models>/<model_id>/<quant>/<pack>.oasr`) stays
//! recognized as a *fallback*, because capability-pack discovery must not die
//! before `pull::migrate_legacy_model_store` has actually converted a given home.
//! That migration runs at CLI startup, but a server or an embedding host can
//! resolve capability packs without ever having gone through it, and a capability
//! that silently turns itself off is far worse than one extra directory scan:
//! this exact gap is what made Voice ID a no-op on every content-addressed
//! install while reporting no error at all.

use std::path::{Path, PathBuf};

/// Resolve a capability-pack path.
///
/// In priority order: the `env_var` override (if it points at a file), the
/// content-addressed object of an installed pack whose model id matches
/// `model_id_hint`, then the legacy per-quant directory layout.
pub(crate) fn resolve_installed_capability_pack(
    env_var: &str,
    model_id_hint: &str,
) -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var(env_var) {
        let path = PathBuf::from(explicit);
        if path.is_file() {
            return Some(path);
        }
    }
    let home = crate::openasr_home().ok()?;
    resolve_installed_capability_pack_in(&home, model_id_hint)
}

/// The layout half of [`resolve_installed_capability_pack`], against an explicit
/// home so it is testable without touching process environment.
///
/// Content-addressed refs first (what every install writes), legacy per-quant
/// directories second (what an unconverted home still has).
pub(crate) fn resolve_installed_capability_pack_in(
    home: &Path,
    model_id_hint: &str,
) -> Option<PathBuf> {
    if let Some(path) = installed_capability_pack(home, model_id_hint) {
        return Some(path);
    }
    let config = crate::config::load_config(home).unwrap_or_default();
    find_pack(&crate::config::models_dir(home, &config), model_id_hint)
}

/// The object of an installed pack whose model id matches `model_id_hint`.
///
/// The hint stays a substring test, but of the **model id recorded in a
/// validated ref** rather than of an arbitrary directory name, and the bytes
/// returned are that ref's own object. Identity and content are therefore bound
/// together by the ref the store already validated (digest well-formed, object
/// present, size matching, no symlink in the path) instead of being "some pack
/// file found inside some directory whose name looked right".
///
/// It stays a substring rather than an exact id because capability-pack ids carry
/// their model revision -- `redimnet2-b6-cn`, `qwen3-forced-aligner-0.6b` -- so
/// pinning exact ids here would need a code change for every repack, which is
/// what the catalog's own "callers should not hardcode its id" guidance warns
/// against. Ties are broken deterministically by (model id, quant).
fn installed_capability_pack(home: &Path, model_id_hint: &str) -> Option<PathBuf> {
    let store = crate::InstalledModelStore::read(home).ok()?;
    let mut matches: Vec<&crate::InstalledPack> = store
        .packs()
        .iter()
        .filter(|pack| pack.model_id.to_ascii_lowercase().contains(model_id_hint))
        .collect();
    matches
        .sort_by(|left, right| (&left.model_id, &left.quant).cmp(&(&right.model_id, &right.quant)));
    matches.first().map(|pack| pack.path.clone())
}

/// Whether `path` is a GGUF (`.oasr`) pack, by sniffing the 4-byte magic rather
/// than trusting the extension. A capability pack may be delivered as either a
/// pulled GGUF `.oasr` or a raw `.safetensors` (the dev fast path), so loaders
/// branch on this.
pub(crate) fn is_gguf_capability_pack(path: &Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut magic))
        .is_ok()
        && &magic == b"GGUF"
}

/// Legacy fallback: the first pack under a `models/*` directory whose *name*
/// contains the hint. Superseded by [`installed_capability_pack`] and kept only
/// until a home has been through `migrate_legacy_model_store`.
///
/// Retirement condition (not yet met, so do not delete this on a timer alone):
/// safe to remove once every process that resolves a capability pack -- not
/// just the CLI's `migrate_model_store_once` -- unconditionally runs
/// `migrate_legacy_model_store` before the first resolution, so no home can
/// reach here with an unconverted legacy layout. Today only the CLI does
/// that; `openasr serve` and an embedding host do not. Deleting this before
/// that gap closes would silently reintroduce the exact Voice-ID-goes-quiet
/// failure mode this module's header doc describes, for those callers.
fn find_pack(root: &Path, dir_substr: &str) -> Option<PathBuf> {
    let mut model_dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.to_ascii_lowercase().contains(dir_substr))
                    .unwrap_or(false)
        })
        .collect();
    model_dirs.sort();
    model_dirs.iter().find_map(|dir| first_pack_file(dir))
}

/// Find a pack file directly in `dir` or one quant subdirectory, preferring the
/// `.oasr` catalog/pull format over a raw `.safetensors` (the dev fast path) when
/// both are present -- so a pulled pack wins over a leftover dev safetensors.
fn first_pack_file(dir: &Path) -> Option<PathBuf> {
    if let Some(path) = best_pack_in_dir(dir) {
        return Some(path);
    }
    let mut subdirs: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    subdirs.sort();
    subdirs.iter().find_map(|sub| best_pack_in_dir(sub))
}

/// The highest-priority pack file directly in `dir`: `.oasr` (priority 0) beats
/// `.safetensors` (priority 1); ties broken by name for determinism.
fn best_pack_in_dir(dir: &Path) -> Option<PathBuf> {
    let priority = |path: &Path| match path.extension().and_then(|ext| ext.to_str()) {
        Some("oasr") => Some(0u8),
        Some("safetensors") => Some(1u8),
        _ => None,
    };
    let mut best: Option<(u8, PathBuf)> = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(rank) = priority(&path) else {
            continue;
        };
        let better = match &best {
            None => true,
            Some((best_rank, best_path)) => {
                rank < *best_rank || (rank == *best_rank && path < *best_path)
            }
        };
        if better {
            best = Some((rank, path));
        }
    }
    best.map(|(_, path)| path)
}

#[cfg(test)]
mod tests {
    use super::{
        best_pack_in_dir, first_pack_file, is_gguf_capability_pack,
        resolve_installed_capability_pack, resolve_installed_capability_pack_in,
    };
    use crate::InstalledPack;
    use sha2::Digest;
    use std::fs;
    use std::path::{Path, PathBuf};

    /// Install a capability pack the way a real pull does: an object under
    /// `objects/sha256/<digest>/content` plus the ref that names it.
    fn install_content_addressed(
        home: &Path,
        model_id: &str,
        quant: &str,
        bytes: &[u8],
    ) -> PathBuf {
        let models = home.join("models");
        let digest = format!("{:x}", sha2::Sha256::digest(bytes));
        let object = models.join("objects/sha256").join(&digest).join("content");
        fs::create_dir_all(object.parent().unwrap()).unwrap();
        fs::write(&object, bytes).unwrap();
        let pack = InstalledPack {
            model_id: model_id.to_string(),
            display_name: model_id.to_string(),
            quant: quant.to_string(),
            suffix: quant.to_string(),
            pull: format!("{model_id}:{quant}"),
            filename: format!("{model_id}-{quant}.oasr"),
            path: object.clone(),
            url: "https://example.invalid/pack.oasr".to_string(),
            hf_revision: "test".to_string(),
            sha256: digest,
            size_bytes: bytes.len() as u64,
            installed_at_unix_seconds: 1,
            source: None,
        };
        let ref_path = models
            .join("refs")
            .join(model_id)
            .join(format!("{quant}.json"));
        fs::create_dir_all(ref_path.parent().unwrap()).unwrap();
        fs::write(&ref_path, serde_json::to_string(&pack).unwrap()).unwrap();
        object
    }

    /// Install a capability pack in the pre-content-store layout.
    fn install_legacy(home: &Path, model_id: &str, quant: &str, bytes: &[u8]) -> PathBuf {
        let dir = home.join("models").join(model_id).join(quant);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{model_id}-{quant}.oasr"));
        fs::write(&path, bytes).unwrap();
        path
    }

    /// Every capability pack that ships today, with the hint its feature passes.
    /// A new capability pack must be added here: the whole class regressed at
    /// once when discovery only understood the legacy layout, so the coverage
    /// has to be per-feature rather than "redimnet works".
    const SHIPPED_CAPABILITY_PACKS: &[(&str, &str)] = &[
        ("redimnet", "redimnet2-b6-cn"),
        ("pyannote", "pyannote-segmentation-3.0"),
        ("forced-aligner", "qwen3-forced-aligner-0.6b"),
        ("firered-punc", "firered-punc"),
    ];

    #[test]
    fn every_capability_pack_resolves_from_the_content_addressed_layout() {
        // The regression: a content-addressed install produces no
        // `models/<name>/` directory at all, so a directory-name scan found
        // nothing and the feature silently turned itself off.
        for (hint, model_id) in SHIPPED_CAPABILITY_PACKS {
            let home = tempfile::tempdir().unwrap();
            let bytes = format!("GGUF{model_id}").into_bytes();
            let object = install_content_addressed(home.path(), model_id, "fp16", &bytes);

            assert!(
                !home.path().join("models").join(model_id).exists(),
                "a content-addressed install must not create a per-model directory"
            );
            assert_eq!(
                resolve_installed_capability_pack_in(home.path(), hint).as_deref(),
                Some(object.as_path()),
                "capability pack '{model_id}' (hint '{hint}') must resolve"
            );
        }
    }

    #[test]
    fn every_capability_pack_still_resolves_from_the_legacy_layout() {
        // Discovery must not die before `migrate_legacy_model_store` has run:
        // a server or embedding host can resolve capability packs without ever
        // having gone through CLI startup.
        for (hint, model_id) in SHIPPED_CAPABILITY_PACKS {
            let home = tempfile::tempdir().unwrap();
            let legacy = install_legacy(home.path(), model_id, "fp16", b"GGUFlegacy");
            assert_eq!(
                resolve_installed_capability_pack_in(home.path(), hint).as_deref(),
                Some(legacy.as_path()),
                "legacy capability pack '{model_id}' must stay discoverable"
            );
        }
    }

    #[test]
    fn content_addressed_pack_wins_over_a_leftover_legacy_copy() {
        let home = tempfile::tempdir().unwrap();
        let object = install_content_addressed(
            home.path(),
            "redimnet2-b6-cn",
            "fp16",
            b"GGUFcontent-addressed",
        );
        let legacy = install_legacy(home.path(), "redimnet2-b6-cn", "fp16", b"GGUFstale-legacy");

        let resolved = resolve_installed_capability_pack_in(home.path(), "redimnet").unwrap();
        assert_eq!(resolved, object);
        assert_ne!(resolved, legacy);
        assert_eq!(fs::read(&resolved).unwrap(), b"GGUFcontent-addressed");
    }

    #[test]
    fn capability_pack_resolution_follows_a_custom_models_dir() {
        let home = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        fs::write(
            home.path().join("config.json"),
            serde_json::json!({ "models_dir": elsewhere.path().join("models") }).to_string(),
        )
        .unwrap();
        let object =
            install_content_addressed(elsewhere.path(), "redimnet2-b6-cn", "fp16", b"GGUFx");

        assert_eq!(
            resolve_installed_capability_pack_in(home.path(), "redimnet").as_deref(),
            Some(object.as_path())
        );
    }

    #[test]
    fn missing_capability_pack_resolves_to_none() {
        let home = tempfile::tempdir().unwrap();
        install_content_addressed(home.path(), "whisper-small", "q8_0", b"GGUFasr");
        assert_eq!(
            resolve_installed_capability_pack_in(home.path(), "redimnet"),
            None,
            "an unrelated installed ASR model must not satisfy a capability probe"
        );
    }

    /// Direct A/B of the two resolution strategies against one real-layout home.
    ///
    /// `find_pack` is the unchanged pre-fix implementation (it survives as the
    /// legacy fallback), so running both over the same fixture is a faithful
    /// before/after rather than a reconstruction. Printed with `--nocapture` as
    /// the evidence that a content-addressed install was invisible.
    #[test]
    fn before_after_content_addressed_capability_pack_discovery() {
        let home = tempfile::tempdir().unwrap();
        let object =
            install_content_addressed(home.path(), "redimnet2-b6-cn", "fp16", b"GGUFredimnet");
        let models = home.path().join("models");

        let legacy_only = super::find_pack(&models, "redimnet");
        let fixed = resolve_installed_capability_pack_in(home.path(), "redimnet");

        println!("layout on disk:");
        println!(
            "  models/*redimnet* dirs : {}",
            fs::read_dir(&models)
                .unwrap()
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().contains("redimnet"))
                .count()
        );
        println!(
            "  models/refs/ entries   : {:?}",
            fs::read_dir(models.join("refs"))
                .unwrap()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
        );
        println!("BEFORE (directory-name scan): {legacy_only:?}");
        println!("AFTER  (installed-ref lookup): {fixed:?}");
        println!(
            "embedder_pack_installed() would be: before={} after={}",
            legacy_only.is_some(),
            fixed.is_some()
        );

        assert_eq!(
            legacy_only, None,
            "pre-fix behaviour: a content-addressed install is invisible to a \
             directory-name scan, which is why Voice ID silently no-opped"
        );
        assert_eq!(fixed.as_deref(), Some(object.as_path()));
    }

    #[test]
    fn env_override_wins_over_an_installed_pack() {
        // A distinct env var name per test keeps this parallel-safe: the
        // override path returns before any home lookup, so no OPENASR_HOME
        // manipulation is needed. The value itself is still restored through
        // the shared RAII guard (rather than a manual set/remove pair) so a
        // panic mid-test cannot leak the override into a sibling test.
        const ENV: &str = "OPENASR_TEST_CAPABILITY_PACK_OVERRIDE";
        let dir = tempfile::tempdir().unwrap();
        let explicit = dir.path().join("explicit.oasr");
        fs::write(&explicit, b"GGUFexplicit").unwrap();

        let resolved = crate::test_process_env::with_test_process_env(
            [(ENV, Some(explicit.clone().into_os_string()))],
            || resolve_installed_capability_pack(ENV, "redimnet"),
        );

        assert_eq!(resolved.as_deref(), Some(explicit.as_path()));
    }

    #[test]
    fn is_gguf_sniffs_magic_not_extension() {
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("pack.oasr");
        fs::write(&gguf, b"GGUF\x00\x00\x00\x00rest").unwrap();
        assert!(is_gguf_capability_pack(&gguf));

        let safetensors = dir.path().join("pack.safetensors");
        fs::write(&safetensors, b"\x10\x00\x00\x00\x00\x00\x00\x00{}").unwrap();
        assert!(!is_gguf_capability_pack(&safetensors));

        assert!(!is_gguf_capability_pack(&dir.path().join("missing")));
    }

    #[test]
    fn first_pack_file_prefers_oasr_over_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), b"st").unwrap();
        fs::write(dir.path().join("model.oasr"), b"GGUF").unwrap();
        let found = first_pack_file(dir.path()).unwrap();
        assert_eq!(found.extension().unwrap(), "oasr");
    }

    #[test]
    fn first_pack_file_falls_back_to_safetensors_and_subdirs() {
        let only_st = tempfile::tempdir().unwrap();
        fs::write(only_st.path().join("model.safetensors"), b"st").unwrap();
        assert_eq!(
            first_pack_file(only_st.path())
                .unwrap()
                .extension()
                .unwrap(),
            "safetensors"
        );

        let nested = tempfile::tempdir().unwrap();
        let quant = nested.path().join("q8_0");
        fs::create_dir(&quant).unwrap();
        fs::write(quant.join("model.oasr"), b"GGUF").unwrap();
        assert_eq!(
            first_pack_file(nested.path()).unwrap().extension().unwrap(),
            "oasr"
        );
    }

    #[test]
    fn best_pack_in_dir_ignores_non_pack_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("readme.txt"), b"x").unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        assert!(best_pack_in_dir(dir.path()).is_none());
    }
}
