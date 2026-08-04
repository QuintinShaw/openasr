//! Read-only discovery of the on-disk model store.
//!
//! An installed pack is a ref at `<models>/refs/<model>/<quant>.json` naming an
//! immutable object at `<models>/objects/sha256/<digest>/content`. This module
//! is the single authority that makes them visible to CLI, server, and runtime
//! selection, and it never writes.
//!
//! Packs used to be recorded instead as `<models>/<model>/<quant>/installed.json`
//! beside their `.oasr`. That layout is no longer *read*: it is converted once,
//! at process start, by `pull::migrate_legacy_model_store`, which still uses
//! [`validate_legacy_record`] here to decide what is eligible. Keeping two
//! readable layouts would mean two code paths, doubled test surface, and a
//! precedence rule whose only job is to be wrong eventually -- and, as shipped,
//! it let a converted store keep a full duplicate of every pack forever.

use std::{
    collections::HashSet,
    fs, io,
    path::{Path, PathBuf},
};

use crate::{
    InstalledPack, PullError, canonical_quant_tag,
    safety::{validate_safe_relative_path, validate_sha256},
};

/// A non-fatal problem found while scanning an installed-model store.
///
/// A malformed or stale record must not make unrelated valid packs disappear.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstalledModelDiagnostic {
    pub path: PathBuf,
    pub reason: String,
}

/// Immutable snapshot of model packs visible from an OpenASR home.
///
/// The snapshot is intentionally rebuilt per request: model pull/import/delete
/// operations may happen in a different process, and caching would otherwise
/// return stale availability.
#[derive(Debug, Default)]
pub struct InstalledModelStore {
    packs: Vec<InstalledPack>,
    diagnostics: Vec<InstalledModelDiagnostic>,
}

impl InstalledModelStore {
    /// Discovers installed packs without mutating the model store.
    pub fn read(home: &Path) -> Result<Self, PullError> {
        let root = models_root(home);
        let mut store = Self::default();
        let mut variants = HashSet::new();

        store.read_content_addressed(&root, &mut variants)?;
        store.packs.sort_by(|left, right| {
            (&left.model_id, canonical_quant_tag(&left.quant), &left.pull).cmp(&(
                &right.model_id,
                canonical_quant_tag(&right.quant),
                &right.pull,
            ))
        });
        Ok(store)
    }

    pub fn packs(&self) -> &[InstalledPack] {
        &self.packs
    }

    pub fn into_packs(self) -> Vec<InstalledPack> {
        self.packs
    }

    pub fn diagnostics(&self) -> &[InstalledModelDiagnostic] {
        &self.diagnostics
    }

    fn read_content_addressed(
        &mut self,
        root: &Path,
        variants: &mut HashSet<(String, String)>,
    ) -> Result<(), PullError> {
        let refs_root = root.join("refs");
        for model_dir in read_dir_entries_or_empty(&refs_root)? {
            let model_path = model_dir.path();
            if !is_real_directory(&model_path) {
                self.diagnose(model_path, "model ref directory is not a real directory");
                continue;
            }
            let Some(model_id) = file_name(&model_path) else {
                self.diagnose(model_path, "model ref directory has no UTF-8 name");
                continue;
            };
            if let Err(reason) = validate_safe_relative_path("model id", model_id) {
                self.diagnose(model_path, reason);
                continue;
            }

            for ref_file in read_dir_entries_or_empty(&model_path)? {
                let ref_path = ref_file.path();
                if ref_path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    != Some("json")
                    || !is_real_file(&ref_path)
                {
                    self.diagnose(ref_path, "model ref is not a regular JSON file");
                    continue;
                }
                let Some(quant) = ref_path.file_stem().and_then(|stem| stem.to_str()) else {
                    self.diagnose(ref_path, "model ref has no UTF-8 quant name");
                    continue;
                };
                if let Err(reason) = validate_safe_relative_path("quant", quant) {
                    self.diagnose(ref_path, reason);
                    continue;
                }

                let contents = match fs::read_to_string(&ref_path) {
                    Ok(contents) => contents,
                    Err(error) => {
                        self.diagnose(ref_path, error.to_string());
                        continue;
                    }
                };
                let mut pack = match serde_json::from_str::<InstalledPack>(&contents) {
                    Ok(pack) => pack,
                    Err(error) => {
                        self.diagnose(ref_path, format!("invalid model ref JSON: {error}"));
                        continue;
                    }
                };
                if let Err(reason) =
                    validate_content_addressed_ref(root, model_id, quant, &mut pack)
                {
                    self.diagnose(ref_path, reason);
                    continue;
                }

                let key = variant_key(&pack);
                if variants.insert(key) {
                    self.packs.push(pack);
                }
            }
        }
        Ok(())
    }

    fn diagnose(&mut self, path: PathBuf, reason: impl Into<String>) {
        self.diagnostics.push(InstalledModelDiagnostic {
            path,
            reason: reason.into(),
        });
    }
}

fn models_root(home: &Path) -> PathBuf {
    let config = crate::config::load_config(home).unwrap_or_default();
    crate::config::models_dir(home, &config)
}

fn read_dir_entries_or_empty(path: &Path) -> Result<Vec<fs::DirEntry>, PullError> {
    let entries = match fs::read_dir(path) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => {
            return Err(PullError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    let mut entries = entries
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| PullError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    entries.sort_by_key(fs::DirEntry::path);
    Ok(entries)
}

fn file_name(path: &Path) -> Option<&str> {
    path.file_name()?.to_str()
}

fn is_real_directory(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .is_ok_and(|metadata| !metadata.file_type().is_symlink() && metadata.file_type().is_dir())
}

fn is_real_file(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .is_ok_and(|metadata| !metadata.file_type().is_symlink() && metadata.file_type().is_file())
}

fn variant_key(pack: &InstalledPack) -> (String, String) {
    (
        pack.model_id.clone(),
        canonical_quant_tag(&pack.quant).to_string(),
    )
}

/// Validate one ref and bind it to the object its own digest names.
///
/// `pack.path` as stored is **not** an input to this decision. An object's
/// location is fully determined by the current models root plus the digest, so
/// the recorded absolute path is redundant -- and actively wrong the moment the
/// user moves their model storage somewhere else, which the desktop's
/// "change storage location" does by relocating files without rewriting refs.
/// Recomputing rather than comparing is also strictly safer than the equality
/// check it replaces: a ref that names some other file cannot mislead a reader
/// that never reads the field.
fn validate_content_addressed_ref(
    root: &Path,
    expected_model_id: &str,
    expected_quant: &str,
    pack: &mut InstalledPack,
) -> Result<(), String> {
    validate_safe_relative_path("model id", &pack.model_id)?;
    validate_safe_relative_path("quant", &pack.quant)?;
    validate_safe_relative_path("filename", &pack.filename)?;
    validate_sha256("sha256", &pack.sha256)?;
    if pack.filename.contains('/') || pack.filename.contains('\\') {
        return Err("filename must not contain a path separator".to_string());
    }
    if pack.model_id != expected_model_id {
        return Err("model id does not match its ref directory".to_string());
    }
    if canonical_quant_tag(&pack.quant) != canonical_quant_tag(expected_quant) {
        return Err("quant does not match its ref filename".to_string());
    }

    if !real_path_under(
        root,
        Path::new("objects/sha256")
            .join(&pack.sha256)
            .join("content"),
    ) {
        return Err(
            "content-addressed object is missing, not a regular file, or traverses a symlink"
                .to_string(),
        );
    }
    let object_path = root
        .join("objects")
        .join("sha256")
        .join(&pack.sha256)
        .join("content");
    let metadata = fs::symlink_metadata(&object_path)
        .map_err(|error| format!("could not stat content-addressed object: {error}"))?;
    if metadata.len() != pack.size_bytes {
        return Err("content-addressed object size does not match ref".to_string());
    }
    pack.path = object_path;
    Ok(())
}

fn real_path_under(root: &Path, relative: PathBuf) -> bool {
    let mut current = root.to_path_buf();
    if !is_real_directory(&current) {
        return false;
    }
    let components = relative.components().collect::<Vec<_>>();
    for (index, component) in components.iter().enumerate() {
        let std::path::Component::Normal(component) = component else {
            return false;
        };
        current.push(component);
        if index + 1 == components.len() {
            return is_real_file(&current);
        }
        if !is_real_directory(&current) {
            return false;
        }
    }
    false
}

/// Shared with the pull path's legacy-record migration: only a record that this
/// reader would accept is eligible to be re-admitted into the content store.
pub(crate) fn validate_legacy_record(pack: &InstalledPack, quant_dir: &Path) -> Result<(), String> {
    validate_safe_relative_path("model id", &pack.model_id)?;
    validate_safe_relative_path("quant", &pack.quant)?;
    validate_safe_relative_path("filename", &pack.filename)?;
    if pack.filename.contains('/')
        || pack.filename.contains('\\')
        || !crate::has_openasr_runtime_pack_extension(&pack.filename)
    {
        return Err("legacy filename is not an .oasr basename".to_string());
    }
    let Some(model_dir) = quant_dir.parent() else {
        return Err("legacy quant directory has no model parent".to_string());
    };
    if file_name(model_dir) != Some(pack.model_id.as_str())
        || file_name(quant_dir) != Some(pack.quant.as_str())
        || pack.path != quant_dir.join(&pack.filename)
    {
        return Err("legacy record does not match its directory".to_string());
    }
    let metadata =
        fs::symlink_metadata(&pack.path).map_err(|_| "legacy pack file is missing".to_string())?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err("legacy pack is not a regular file".to_string());
    }
    if metadata.len() != pack.size_bytes {
        return Err("legacy pack size does not match record".to_string());
    }
    crate::verify_native_runtime_model_pack_path(&pack.path)
        .map_err(|error| format!("legacy pack fails runtime validation: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
    use tempfile::TempDir;

    #[cfg(unix)]
    use std::os::unix::fs::symlink;

    fn write_ref(home: &Path, model_id: &str, quant: &str, digest: &str, size: u64) -> PathBuf {
        let object = home
            .join("models/objects/sha256")
            .join(digest)
            .join("content");
        fs::create_dir_all(object.parent().unwrap()).unwrap();
        fs::write(&object, vec![7; size as usize]).unwrap();
        let ref_path = home
            .join("models/refs")
            .join(model_id)
            .join(format!("{quant}.json"));
        fs::create_dir_all(ref_path.parent().unwrap()).unwrap();
        let pack = InstalledPack {
            model_id: model_id.to_string(),
            display_name: model_id.to_string(),
            quant: quant.to_string(),
            suffix: if canonical_quant_tag(quant) == "q4_k" {
                "q4"
            } else {
                "q8"
            }
            .to_string(),
            pull: format!(
                "{model_id}:{}",
                if canonical_quant_tag(quant) == "q4_k" {
                    "q4"
                } else {
                    "q8"
                }
            ),
            filename: format!("{model_id}-{quant}.oasr"),
            path: object.clone(),
            url: "https://example.invalid/model.oasr".to_string(),
            hf_revision: "test".to_string(),
            sha256: digest.to_string(),
            size_bytes: size,
            installed_at_unix_seconds: 1,
            source: None,
        };
        fs::write(&ref_path, serde_json::to_string(&pack).unwrap()).unwrap();
        ref_path
    }

    fn write_legacy_pack(home: &Path, model_id: &str, quant: &str) -> PathBuf {
        let dir = home.join("models").join(model_id).join(quant);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{model_id}-{quant}.oasr"));
        let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer(model_id);
        write_tiny_gguf_runtime_source(&path, &spec).unwrap();
        let pack = InstalledPack {
            model_id: model_id.to_string(),
            display_name: model_id.to_string(),
            quant: quant.to_string(),
            suffix: "q4".to_string(),
            pull: format!("{model_id}:q4"),
            filename: path.file_name().unwrap().to_str().unwrap().to_string(),
            path: path.clone(),
            url: "https://example.invalid/model.oasr".to_string(),
            hf_revision: "test".to_string(),
            sha256: "f".repeat(64),
            size_bytes: fs::metadata(&path).unwrap().len(),
            installed_at_unix_seconds: 1,
            source: None,
        };
        fs::write(
            dir.join("installed.json"),
            serde_json::to_string(&pack).unwrap(),
        )
        .unwrap();
        path
    }

    #[test]
    fn discovers_content_addressed_refs_and_canonical_quant_aliases() {
        let home = TempDir::new().unwrap();
        write_ref(
            home.path(),
            "xasr-zh-en",
            "q4_k",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            7,
        );
        let store = InstalledModelStore::read(home.path()).unwrap();
        assert_eq!(store.packs().len(), 1);
        assert_eq!(canonical_quant_tag(&store.packs()[0].quant), "q4_k");
        assert_eq!(store.packs()[0].pull, "xasr-zh-en:q4");
    }

    #[test]
    fn content_addressed_ref_wins_over_duplicate_legacy_variant() {
        let home = TempDir::new().unwrap();
        write_ref(
            home.path(),
            "xasr-zh-en",
            "q4_k",
            "9999999999999999999999999999999999999999999999999999999999999999",
            7,
        );
        let legacy_path = write_legacy_pack(home.path(), "xasr-zh-en", "q4_k");

        let store = InstalledModelStore::read(home.path()).unwrap();
        assert_eq!(store.packs().len(), 1);
        assert_ne!(store.packs()[0].path, legacy_path);
        assert_eq!(store.packs()[0].pull, "xasr-zh-en:q4");
    }

    #[test]
    fn legacy_layout_is_not_a_discovery_source() {
        // The reader knows exactly one layout. An unconverted legacy tree is
        // invisible here by design; `migrate_legacy_model_store` is what makes
        // it visible, and it runs at startup.
        let home = TempDir::new().unwrap();
        let legacy_path = write_legacy_pack(home.path(), "xasr-zh-en", "q4_k");
        assert!(legacy_path.is_file());

        let store = InstalledModelStore::read(home.path()).unwrap();
        assert!(store.packs().is_empty());
        // Not an error either: an unconverted tree is a pending migration, not a
        // corrupt store.
        assert!(store.diagnostics().is_empty());
    }

    #[test]
    fn default_resolution_uses_content_addressed_q4_alias() {
        let home = TempDir::new().unwrap();
        write_ref(
            home.path(),
            "xasr-zh-en",
            "q4_k",
            "abababababababababababababababababababababababababababababababab",
            7,
        );
        fs::write(
            home.path().join("config.json"),
            r#"{"default_model":"xasr-zh-en"}"#,
        )
        .unwrap();

        let resolved = crate::default_selection::resolve_with_catalog(home.path(), None).unwrap();
        assert!(matches!(
            resolved,
            crate::default_selection::DefaultModelResolution::Installed(pack)
                if pack.pull == "xasr-zh-en:q4"
        ));
    }

    #[test]
    fn malformed_ref_does_not_hide_other_content_addressed_models() {
        let home = TempDir::new().unwrap();
        let bad = write_ref(
            home.path(),
            "bad-model",
            "q4_k",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            7,
        );
        // A digest that is not a digest: the ref can never name an object, so it
        // is unusable no matter what else it says.
        let mut record: serde_json::Value =
            serde_json::from_slice(&fs::read(&bad).unwrap()).unwrap();
        record["sha256"] = serde_json::Value::String("not-a-sha256".to_string());
        fs::write(&bad, serde_json::to_string(&record).unwrap()).unwrap();
        write_ref(
            home.path(),
            "good-model",
            "q8_0",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            7,
        );

        let store = InstalledModelStore::read(home.path()).unwrap();
        assert_eq!(store.packs().len(), 1);
        assert_eq!(store.packs()[0].model_id, "good-model");
        assert_eq!(store.diagnostics().len(), 1);
    }

    #[test]
    fn a_refs_recorded_path_is_never_an_authority() {
        // The object's location is derived from the models root and the digest.
        // A stale path (the desktop relocates storage without rewriting refs) or
        // a crafted one must both resolve to the same real object.
        let home = TempDir::new().unwrap();
        let digest = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        let ref_path = write_ref(home.path(), "good-model", "q8_0", digest, 7);
        let escaped = home.path().join("outside");
        fs::write(&escaped, b"outside").unwrap();
        let mut record: serde_json::Value =
            serde_json::from_slice(&fs::read(&ref_path).unwrap()).unwrap();
        record["path"] = serde_json::Value::String(escaped.display().to_string());
        fs::write(&ref_path, serde_json::to_string(&record).unwrap()).unwrap();

        let store = InstalledModelStore::read(home.path()).unwrap();
        assert_eq!(store.packs().len(), 1);
        assert_eq!(
            store.packs()[0].path,
            home.path()
                .join("models/objects/sha256")
                .join(digest)
                .join("content")
        );
        assert!(store.diagnostics().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_content_object_is_rejected() {
        let home = TempDir::new().unwrap();
        write_ref(
            home.path(),
            "symlink-model",
            "q4_k",
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            7,
        );
        let object = home
            .path()
            .join("models/objects/sha256/ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff/content");
        fs::remove_file(&object).unwrap();
        let outside = home.path().join("outside-content");
        fs::write(&outside, vec![7; 7]).unwrap();
        symlink(&outside, &object).unwrap();

        let store = InstalledModelStore::read(home.path()).unwrap();
        assert!(store.packs().is_empty());
        assert_eq!(store.diagnostics().len(), 1);
        assert!(store.diagnostics()[0].reason.contains("symlink"));
    }

    #[test]
    fn missing_object_is_diagnostic_and_does_not_hide_other_refs() {
        let home = TempDir::new().unwrap();
        let missing = write_ref(
            home.path(),
            "missing-model",
            "q4_k",
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            7,
        );
        let missing_object = home
            .path()
            .join("models/objects/sha256/dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd/content");
        fs::remove_file(missing_object).unwrap();
        write_ref(
            home.path(),
            "good-model",
            "q4_k",
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            7,
        );

        let store = InstalledModelStore::read(home.path()).unwrap();
        assert_eq!(store.packs().len(), 1);
        assert_eq!(store.diagnostics()[0].path, missing);
    }
}
