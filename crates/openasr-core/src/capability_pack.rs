//! Generic resolution of an installed optional capability-pack file (a
//! `.oasr`/`.safetensors` support model that augments a family's own decode
//! path, e.g. the WeSpeaker speaker-embedder or the Qwen3-ForcedAligner
//! word-timestamp refiner) from `openasr_home()/models/<dir>/`. Extracted from
//! `diarize::pack` so a second capability-pack family (forced alignment) does
//! not duplicate the same env-override + directory-scan logic -- infrastructure
//! that decides where an installed pack lives stays model-agnostic; only the
//! env var name and directory substring are per-feature.

use std::path::{Path, PathBuf};

/// Resolve a pack path: the `env_var` override (if it points at a file), else the
/// first `.oasr`/`.safetensors` under a `models/*` directory whose name contains
/// `dir_substr`.
pub(crate) fn resolve_installed_capability_pack(
    env_var: &str,
    dir_substr: &str,
) -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var(env_var) {
        let path = PathBuf::from(explicit);
        if path.is_file() {
            return Some(path);
        }
    }
    let home = crate::openasr_home().ok()?;
    find_pack(&home.join("models"), dir_substr)
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
    use super::{best_pack_in_dir, first_pack_file, is_gguf_capability_pack};
    use std::fs;

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
