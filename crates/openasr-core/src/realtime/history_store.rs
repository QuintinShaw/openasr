use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ResponseFormat, atomic_file};

pub const DAEMON_HISTORY_INDEX_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonHistoryKind {
    File,
    Live,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonHistoryProvenance {
    AutoSaved,
    UserInitiated,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonHistoryEntry {
    pub id: String,
    pub kind: DaemonHistoryKind,
    pub model: String,
    pub created_at_unix_seconds: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diarization_active: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<DaemonHistoryProvenance>,
    pub formats: Vec<String>,
    pub preview: String,
    // Internal sidecar location: derivable from `id`, never read back from a
    // loaded entry, and must not leak into the on-disk index or the loopback
    // HTTP response. Kept on the in-memory record for convenience only.
    #[serde(skip)]
    pub text_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DaemonHistoryDetail {
    #[serde(flatten)]
    pub entry: DaemonHistoryEntry,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DaemonHistoryRecord {
    pub kind: DaemonHistoryKind,
    pub model: String,
    pub source_name: Option<String>,
    pub duration_seconds: Option<f32>,
    pub output_format: Option<ResponseFormat>,
    pub diarization_active: Option<bool>,
    pub provenance: Option<DaemonHistoryProvenance>,
    pub formats: Vec<String>,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct DaemonHistoryStore {
    root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct DaemonHistoryIndex {
    version: u32,
    entries: Vec<DaemonHistoryEntry>,
}

#[derive(Debug, Error)]
pub enum DaemonHistoryStoreError {
    #[error("Invalid history id '{id}': {reason}")]
    InvalidId { id: String, reason: &'static str },
    #[error("Unsupported history index version {found}. Expected version {expected}.")]
    UnsupportedIndexVersion { found: u32, expected: u32 },
    #[error("Invalid history record field '{field}': {reason}")]
    InvalidRecord { field: &'static str, reason: String },
    #[error("History store lock for '{path}' was poisoned.")]
    LockPoisoned { path: PathBuf },
    #[error("Could not create history directory '{path}': {source}")]
    CreateDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Could not read history index '{path}': {source}")]
    ReadIndex {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Could not parse history index '{path}': {source}")]
    ParseIndex {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("Could not serialize history index: {0}")]
    SerializeIndex(serde_json::Error),
    #[error("Could not write history file '{path}': {source}")]
    WriteFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Could not read history text sidecar '{path}': {source}")]
    ReadText {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Could not remove history text sidecar '{path}': {source}")]
    RemoveText {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl DaemonHistoryStore {
    pub fn open(openasr_home: impl AsRef<Path>) -> Self {
        Self {
            root: history_dir(openasr_home),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn list(&self) -> Result<Vec<DaemonHistoryEntry>, DaemonHistoryStoreError> {
        let lock = history_store_lock(&self.root);
        let _guard = lock
            .lock()
            .map_err(|_| DaemonHistoryStoreError::LockPoisoned {
                path: self.root.clone(),
            })?;
        let mut entries = self.load_index()?.entries;
        entries.sort_by(|left, right| {
            right
                .created_at_unix_seconds
                .cmp(&left.created_at_unix_seconds)
                .then_with(|| right.id.cmp(&left.id))
        });
        Ok(entries)
    }

    pub fn get(&self, id: &str) -> Result<Option<DaemonHistoryDetail>, DaemonHistoryStoreError> {
        let lock = history_store_lock(&self.root);
        let _guard = lock
            .lock()
            .map_err(|_| DaemonHistoryStoreError::LockPoisoned {
                path: self.root.clone(),
            })?;
        validate_history_id(id)?;
        let index = self.load_index()?;
        let Some(entry) = index.entries.into_iter().find(|entry| entry.id == id) else {
            return Ok(None);
        };
        let text_path = self.text_path(id);
        let text =
            fs::read_to_string(&text_path).map_err(|source| DaemonHistoryStoreError::ReadText {
                path: text_path,
                source,
            })?;
        Ok(Some(DaemonHistoryDetail { entry, text }))
    }

    pub fn delete(&self, id: &str) -> Result<bool, DaemonHistoryStoreError> {
        let lock = history_store_lock(&self.root);
        let _guard = lock
            .lock()
            .map_err(|_| DaemonHistoryStoreError::LockPoisoned {
                path: self.root.clone(),
            })?;
        validate_history_id(id)?;
        let mut index = self.load_index()?;
        let Some(position) = index.entries.iter().position(|entry| entry.id == id) else {
            return Ok(false);
        };
        index.entries.remove(position);
        self.remove_text_sidecar(id)?;
        self.save_index(&index)?;
        Ok(true)
    }

    pub fn delete_older_than(
        &self,
        cutoff_unix_seconds: u64,
    ) -> Result<usize, DaemonHistoryStoreError> {
        self.prune_entries(|entry| entry.created_at_unix_seconds < cutoff_unix_seconds)
    }

    pub fn retain_most_recent(&self, max_entries: usize) -> Result<usize, DaemonHistoryStoreError> {
        let lock = history_store_lock(&self.root);
        let _guard = lock
            .lock()
            .map_err(|_| DaemonHistoryStoreError::LockPoisoned {
                path: self.root.clone(),
            })?;
        let mut index = self.load_index()?;
        if index.entries.len() <= max_entries {
            return Ok(0);
        }

        let mut ordered = index.entries.iter().collect::<Vec<_>>();
        ordered.sort_by(|left, right| {
            right
                .created_at_unix_seconds
                .cmp(&left.created_at_unix_seconds)
                .then_with(|| right.id.cmp(&left.id))
        });
        let keep = ordered
            .into_iter()
            .take(max_entries)
            .map(|entry| entry.id.clone())
            .collect::<HashSet<_>>();

        let removed = remove_index_entries(&mut index, |entry| !keep.contains(&entry.id));
        if removed.is_empty() {
            return Ok(0);
        }
        for id in &removed {
            self.remove_text_sidecar(id)?;
        }
        self.save_index(&index)?;
        Ok(removed.len())
    }

    pub fn record(
        &self,
        record: DaemonHistoryRecord,
    ) -> Result<DaemonHistoryEntry, DaemonHistoryStoreError> {
        let lock = history_store_lock(&self.root);
        let _guard = lock
            .lock()
            .map_err(|_| DaemonHistoryStoreError::LockPoisoned {
                path: self.root.clone(),
            })?;
        validate_record(&record)?;
        fs::create_dir_all(self.texts_dir()).map_err(|source| {
            DaemonHistoryStoreError::CreateDir {
                path: self.texts_dir(),
                source,
            }
        })?;

        let id = self.next_id();
        let text_path = self.text_path(&id);
        write_file_atomically(&text_path, record.text.as_bytes())?;

        let entry = DaemonHistoryEntry {
            id,
            kind: record.kind,
            model: record.model.trim().to_string(),
            created_at_unix_seconds: unix_seconds_now(),
            source_name: record
                .source_name
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned),
            duration_seconds: record.duration_seconds,
            output_format: record.output_format,
            diarization_active: record.diarization_active,
            provenance: record.provenance,
            formats: normalized_formats(record.formats),
            preview: preview_text(&record.text),
            text_path,
        };
        let mut index = self.load_index()?;
        index.entries.retain(|existing| existing.id != entry.id);
        index.entries.push(entry.clone());
        self.save_index(&index)?;
        Ok(entry)
    }

    fn load_index(&self) -> Result<DaemonHistoryIndex, DaemonHistoryStoreError> {
        let path = self.index_path();
        match fs::read_to_string(&path) {
            Ok(contents) => {
                let index: DaemonHistoryIndex =
                    serde_json::from_str(&contents).map_err(|source| {
                        DaemonHistoryStoreError::ParseIndex {
                            path: path.clone(),
                            source,
                        }
                    })?;
                if index.version != DAEMON_HISTORY_INDEX_VERSION {
                    return Err(DaemonHistoryStoreError::UnsupportedIndexVersion {
                        found: index.version,
                        expected: DAEMON_HISTORY_INDEX_VERSION,
                    });
                }
                Ok(index)
            }
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                Ok(DaemonHistoryIndex {
                    version: DAEMON_HISTORY_INDEX_VERSION,
                    entries: Vec::new(),
                })
            }
            Err(source) => Err(DaemonHistoryStoreError::ReadIndex { path, source }),
        }
    }

    fn save_index(&self, index: &DaemonHistoryIndex) -> Result<(), DaemonHistoryStoreError> {
        fs::create_dir_all(&self.root).map_err(|source| DaemonHistoryStoreError::CreateDir {
            path: self.root.clone(),
            source,
        })?;
        let path = self.index_path();
        let contents =
            serde_json::to_vec_pretty(index).map_err(DaemonHistoryStoreError::SerializeIndex)?;
        write_file_atomically(&path, &contents)
    }

    fn prune_entries(
        &self,
        should_remove: impl FnMut(&DaemonHistoryEntry) -> bool,
    ) -> Result<usize, DaemonHistoryStoreError> {
        let lock = history_store_lock(&self.root);
        let _guard = lock
            .lock()
            .map_err(|_| DaemonHistoryStoreError::LockPoisoned {
                path: self.root.clone(),
            })?;
        let mut index = self.load_index()?;
        let removed = remove_index_entries(&mut index, should_remove);
        if removed.is_empty() {
            return Ok(0);
        }
        for id in &removed {
            self.remove_text_sidecar(id)?;
        }
        self.save_index(&index)?;
        Ok(removed.len())
    }

    fn remove_text_sidecar(&self, id: &str) -> Result<(), DaemonHistoryStoreError> {
        let text_path = self.text_path(id);
        match fs::remove_file(&text_path) {
            Ok(()) => Ok(()),
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(source) => Err(DaemonHistoryStoreError::RemoveText {
                path: text_path,
                source,
            }),
        }
    }

    fn index_path(&self) -> PathBuf {
        self.root.join("index.json")
    }

    fn texts_dir(&self) -> PathBuf {
        self.root.join("texts")
    }

    fn text_path(&self, id: &str) -> PathBuf {
        self.texts_dir().join(format!("{id}.txt"))
    }

    fn next_id(&self) -> String {
        let base = format!("hist-{}", unix_millis_now());
        for attempt in 0..1000_u16 {
            let id = if attempt == 0 {
                base.clone()
            } else {
                format!("{base}-{attempt}")
            };
            if !self.text_path(&id).exists() {
                return id;
            }
        }
        format!("{base}-{}", std::process::id())
    }
}

fn remove_index_entries(
    index: &mut DaemonHistoryIndex,
    mut should_remove: impl FnMut(&DaemonHistoryEntry) -> bool,
) -> Vec<String> {
    let mut removed = Vec::new();
    index.entries.retain(|entry| {
        if should_remove(entry) {
            removed.push(entry.id.clone());
            false
        } else {
            true
        }
    });
    removed
}

pub fn history_dir(openasr_home: impl AsRef<Path>) -> PathBuf {
    openasr_home.as_ref().join("history")
}

fn history_store_lock(root: &Path) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
    let mut locks = LOCKS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("history lock registry mutex poisoned");
    locks
        .entry(root.to_path_buf())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

fn validate_history_id(id: &str) -> Result<(), DaemonHistoryStoreError> {
    if id.is_empty() {
        return Err(DaemonHistoryStoreError::InvalidId {
            id: id.to_string(),
            reason: "must be non-empty",
        });
    }
    if id.len() > 160 {
        return Err(DaemonHistoryStoreError::InvalidId {
            id: id.to_string(),
            reason: "must be at most 160 bytes",
        });
    }
    if !id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(DaemonHistoryStoreError::InvalidId {
            id: id.to_string(),
            reason: "must contain only ASCII letters, digits, '-' or '_'",
        });
    }
    Ok(())
}

fn validate_record(record: &DaemonHistoryRecord) -> Result<(), DaemonHistoryStoreError> {
    if record.model.trim().is_empty() {
        return invalid_record("model", "must be non-empty");
    }
    if let Some(duration) = record.duration_seconds
        && (!duration.is_finite() || duration < 0.0)
    {
        return invalid_record("duration_seconds", "must be finite and non-negative");
    }
    for format in &record.formats {
        if format.trim().is_empty() {
            return invalid_record("formats", "must not contain empty entries");
        }
    }
    Ok(())
}

fn invalid_record(
    field: &'static str,
    reason: impl Into<String>,
) -> Result<(), DaemonHistoryStoreError> {
    Err(DaemonHistoryStoreError::InvalidRecord {
        field,
        reason: reason.into(),
    })
}

fn normalized_formats(formats: Vec<String>) -> Vec<String> {
    let mut normalized = formats
        .into_iter()
        .map(|format| format.trim().to_string())
        .filter(|format| !format.is_empty())
        .collect::<Vec<_>>();
    if normalized.is_empty() {
        normalized.push("text".to_string());
    }
    normalized.sort();
    normalized.dedup();
    normalized
}

fn preview_text(text: &str) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    normalized.chars().take(160).collect()
}

fn write_file_atomically(path: &Path, contents: &[u8]) -> Result<(), DaemonHistoryStoreError> {
    atomic_file::write_file_atomically(path, contents).map_err(|source| {
        DaemonHistoryStoreError::WriteFile {
            path: path.to_path_buf(),
            source,
        }
    })
}

fn unix_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn unix_millis_now() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daemon_history_store_records_lists_gets_and_deletes_sidecar() {
        let temp = tempfile::tempdir().unwrap();
        let store = DaemonHistoryStore::open(temp.path());
        let entry = store
            .record(DaemonHistoryRecord {
                kind: DaemonHistoryKind::File,
                model: "whisper-large-v3-turbo".to_string(),
                source_name: Some("sample.wav".to_string()),
                duration_seconds: Some(2.5),
                output_format: Some(ResponseFormat::Srt),
                diarization_active: Some(true),
                provenance: Some(DaemonHistoryProvenance::AutoSaved),
                formats: vec!["json".to_string()],
                text: "hello OpenASR history".to_string(),
            })
            .unwrap();

        assert!(entry.text_path.exists());
        let entries = store.list().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].output_format, Some(ResponseFormat::Srt));
        assert_eq!(entries[0].diarization_active, Some(true));
        assert_eq!(
            entries[0].provenance,
            Some(DaemonHistoryProvenance::AutoSaved)
        );
        let detail = store.get(&entry.id).unwrap().unwrap();
        assert_eq!(detail.text, "hello OpenASR history");
        assert_eq!(detail.entry.output_format, Some(ResponseFormat::Srt));
        assert_eq!(detail.entry.diarization_active, Some(true));
        assert_eq!(
            detail.entry.provenance,
            Some(DaemonHistoryProvenance::AutoSaved)
        );

        assert!(store.delete(&entry.id).unwrap());
        assert!(!entry.text_path.exists());
        assert!(store.list().unwrap().is_empty());
    }

    #[test]
    fn daemon_history_store_serializes_concurrent_writes_for_one_daemon() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().to_path_buf();
        let mut workers = Vec::new();

        for index in 0..24 {
            let home = home.clone();
            workers.push(std::thread::spawn(move || {
                let store = DaemonHistoryStore::open(home);
                store
                    .record(DaemonHistoryRecord {
                        kind: DaemonHistoryKind::Live,
                        model: "whisper-large-v3-turbo".to_string(),
                        source_name: Some(format!("session-{index}")),
                        duration_seconds: Some(index as f32),
                        output_format: Some(ResponseFormat::Text),
                        diarization_active: Some(false),
                        provenance: Some(DaemonHistoryProvenance::AutoSaved),
                        formats: vec!["text".to_string()],
                        text: format!("hello from session {index}"),
                    })
                    .unwrap();
            }));
        }

        for worker in workers {
            worker.join().unwrap();
        }

        let store = DaemonHistoryStore::open(temp.path());
        let entries = store.list().unwrap();
        assert_eq!(entries.len(), 24);
        for entry in entries {
            assert!(store.get(&entry.id).unwrap().is_some());
        }
    }

    #[test]
    fn daemon_history_store_deletes_entries_older_than_cutoff() {
        let temp = tempfile::tempdir().unwrap();
        let store = DaemonHistoryStore::open(temp.path());
        let old = record_history_for_test(&store, "old transcript");
        let fresh = record_history_for_test(&store, "fresh transcript");
        set_history_created_at_for_test(&store, &old.id, 10);
        set_history_created_at_for_test(&store, &fresh.id, 20);

        assert_eq!(store.delete_older_than(15).unwrap(), 1);

        assert!(store.get(&old.id).unwrap().is_none());
        assert!(!old.text_path.exists());
        assert!(store.get(&fresh.id).unwrap().is_some());
    }

    #[test]
    fn daemon_history_store_retains_most_recent_entries_and_sidecars() {
        let temp = tempfile::tempdir().unwrap();
        let store = DaemonHistoryStore::open(temp.path());
        let oldest = record_history_for_test(&store, "oldest transcript");
        let middle = record_history_for_test(&store, "middle transcript");
        let newest = record_history_for_test(&store, "newest transcript");
        set_history_created_at_for_test(&store, &oldest.id, 10);
        set_history_created_at_for_test(&store, &middle.id, 20);
        set_history_created_at_for_test(&store, &newest.id, 30);

        assert_eq!(store.retain_most_recent(2).unwrap(), 1);

        let remaining = store
            .list()
            .unwrap()
            .into_iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>();
        assert_eq!(remaining, vec![newest.id.clone(), middle.id.clone()]);
        assert!(!oldest.text_path.exists());
        assert!(middle.text_path.exists());
        assert!(newest.text_path.exists());
    }

    #[test]
    fn daemon_history_detail_json_contract_matches_desktop_client() {
        let entry = DaemonHistoryEntry {
            id: "hist-1".to_string(),
            kind: DaemonHistoryKind::Live,
            model: "qwen".to_string(),
            created_at_unix_seconds: 1_780_290_000,
            source_name: None,
            duration_seconds: Some(2.5),
            output_format: Some(ResponseFormat::Text),
            diarization_active: Some(false),
            provenance: Some(DaemonHistoryProvenance::AutoSaved),
            formats: vec!["text".to_string()],
            preview: "hello".to_string(),
            text_path: PathBuf::from("/tmp/openasr/history/texts/hist-1.txt"),
        };
        let detail = DaemonHistoryDetail {
            entry,
            text: "hello world".to_string(),
        };

        let value = serde_json::to_value(&detail).unwrap();
        assert_eq!(value["id"], "hist-1");
        assert_eq!(value["kind"], "live");
        assert_eq!(value["created_at_unix_seconds"], 1_780_290_000);
        assert!(value.get("created_at").is_none());
        assert!(value.get("source_name").is_none());
        assert_eq!(value["duration_seconds"], 2.5);
        assert_eq!(value["output_format"], "text");
        assert_eq!(value["diarization_active"], false);
        assert_eq!(value["provenance"], "auto_saved");
        assert_eq!(value["text"], "hello world");
        assert!(value.get("response_format").is_none());
        assert!(value.get("text_path").is_none());
    }

    #[test]
    fn daemon_history_store_reads_entries_without_optional_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let history_root = temp.path().join("history");
        let texts = history_root.join("texts");
        fs::create_dir_all(&texts).unwrap();
        fs::write(texts.join("hist-old.txt"), "legacy text").unwrap();
        fs::write(
            history_root.join("index.json"),
            r#"{
  "version": 1,
  "entries": [
    {
      "id": "hist-old",
      "kind": "file",
      "model": "whisper-large-v3-turbo",
      "created_at_unix_seconds": 1780290000,
      "duration_seconds": 1.25,
      "formats": ["text"],
      "preview": "legacy"
    }
  ]
}"#,
        )
        .unwrap();

        let store = DaemonHistoryStore::open(temp.path());
        let entries = store.list().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].output_format, None);
        assert_eq!(entries[0].diarization_active, None);
        assert_eq!(entries[0].provenance, None);
        let detail = store.get("hist-old").unwrap().unwrap();
        assert_eq!(detail.text, "legacy text");
        assert_eq!(detail.entry.output_format, None);
        assert_eq!(detail.entry.diarization_active, None);
        assert_eq!(detail.entry.provenance, None);
    }

    fn record_history_for_test(store: &DaemonHistoryStore, text: &str) -> DaemonHistoryEntry {
        store
            .record(DaemonHistoryRecord {
                kind: DaemonHistoryKind::Live,
                model: "whisper-large-v3-turbo".to_string(),
                source_name: None,
                duration_seconds: None,
                output_format: Some(ResponseFormat::Text),
                diarization_active: Some(false),
                provenance: Some(DaemonHistoryProvenance::AutoSaved),
                formats: vec!["text".to_string()],
                text: text.to_string(),
            })
            .unwrap()
    }

    fn set_history_created_at_for_test(store: &DaemonHistoryStore, id: &str, created_at: u64) {
        let mut index = store.load_index().unwrap();
        let entry = index
            .entries
            .iter_mut()
            .find(|entry| entry.id == id)
            .expect("history entry exists");
        entry.created_at_unix_seconds = created_at;
        store.save_index(&index).unwrap();
    }
}
