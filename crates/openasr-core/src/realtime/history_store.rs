//! Daemon transcription history: a local SQLite store under
//! `~/.openasr/history/history.db`.
//!
//! Design notes (this feature has never shipped to a user -- zero stored data
//! to migrate -- so the previous JSON-index + text-sidecar implementation was
//! replaced outright rather than migrated):
//!
//! - **Schema**: one `history_entries` row per `DaemonHistoryEntry`, full
//!   transcript text included as a column (no `texts/*.txt` sidecar). The
//!   entries here are transcripts (bytes to low KB), not audio -- there is no
//!   size pressure that justifies a separate file per row, and folding the
//!   text into the row removes the entire class of "index says the entry
//!   exists but the sidecar write didn't land" partial-failure this file used
//!   to guard against with a temp-file-based atomic writer.
//! - **Full-text search**: a standalone (non "external content") FTS5 virtual
//!   table `history_entries_fts(id UNINDEXED, search_text)` using the
//!   `trigram` tokenizer, kept in sync by hand inside the same transaction as
//!   every write/delete (chosen over `content=`/triggers because our primary
//!   key is a `TEXT` id, not an integer `rowid`, which is what SQLite's
//!   external-content-table sync machinery is built around; with a single
//!   read-modify-write per call there is nothing an external-content trigger
//!   would buy us). The trigram tokenizer indexes overlapping 3-character
//!   windows over Unicode codepoints, so it does not depend on word
//!   boundaries -- critical for Chinese/Japanese text, which unicode61's
//!   whitespace/punctuation based tokenizer segments badly. Substring lookups
//!   run as `search_text LIKE '%needle%'` against the FTS5 table rather than
//!   `MATCH`: FTS5 `MATCH` queries shorter than 3 characters tokenize to
//!   nothing and silently match zero rows, but a great many meaningful CJK
//!   search terms *are* 1-2 characters (e.g. "历史"), so `MATCH` would break
//!   the exact use case this feature exists for. `LIKE` against a
//!   trigram-tokenized table is index-accelerated for patterns of 3+
//!   characters and gracefully falls back to a full scan below that (see
//!   <https://sqlite.org/fts5.html#the_trigram_tokenizer>) -- both are
//!   correct at the local, single-user history sizes this store handles.
//! - **Concurrency**: every call opens its own short-lived `Connection`
//!   (matches the existing call pattern -- `DaemonHistoryStore::open` is
//!   already called fresh per HTTP request / per realtime session) with WAL
//!   journaling and a busy timeout, so SQLite's own file locking serializes
//!   concurrent writers instead of an in-process `Mutex` registry. This is
//!   simpler than the previous per-root `Mutex<()>` registry and remains
//!   correct across processes, which the old in-process lock never was.
//! - **Corruption isolation**: `open()` stays infallible (it only resolves a
//!   path); connecting and touching the schema happens lazily on first use of
//!   any method, so a corrupt `history.db` surfaces as a typed
//!   `DaemonHistoryStoreError` from that call, not a panic. Callers already
//!   treat history recording as best-effort (see
//!   `record_file_transcription_history` in openasr-server), so this keeps
//!   transcription itself unaffected by a broken history store.

use std::{
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ResponseFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonHistoryKind {
    File,
    Live,
}

impl DaemonHistoryKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Live => "live",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "file" => Some(Self::File),
            "live" => Some(Self::Live),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonHistoryProvenance {
    AutoSaved,
    UserInitiated,
}

impl DaemonHistoryProvenance {
    const fn as_str(self) -> &'static str {
        match self {
            Self::AutoSaved => "auto_saved",
            Self::UserInitiated => "user_initiated",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "auto_saved" => Some(Self::AutoSaved),
            "user_initiated" => Some(Self::UserInitiated),
            _ => None,
        }
    }
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

/// Filter/pagination request for [`DaemonHistoryStore::query`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DaemonHistoryQuery {
    /// Substring search over model, source name, and transcript text.
    pub search: Option<String>,
    pub kind: Option<DaemonHistoryKind>,
    /// `None` means "no limit" (used by the back-compat `list()` helper).
    pub limit: Option<usize>,
    pub offset: usize,
}

/// A page of results plus the total row count matching the filter (ignoring
/// `limit`/`offset`), so callers can render pagination controls.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DaemonHistoryPage {
    pub entries: Vec<DaemonHistoryEntry>,
    pub total: usize,
}

#[derive(Debug, Clone)]
pub struct DaemonHistoryStore {
    root: PathBuf,
}

#[derive(Debug, Error)]
pub enum DaemonHistoryStoreError {
    #[error("Invalid history id '{id}': {reason}")]
    InvalidId { id: String, reason: &'static str },
    #[error("Invalid history record field '{field}': {reason}")]
    InvalidRecord { field: &'static str, reason: String },
    #[error("Could not create history directory '{path}': {source}")]
    CreateDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Could not open history database '{path}': {source}")]
    OpenDatabase {
        path: PathBuf,
        #[source]
        source: rusqlite::Error,
    },
    #[error("History database query failed: {0}")]
    Query(#[source] rusqlite::Error),
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

    /// All entries, most recent first. Back-compat convenience over
    /// [`Self::query`] for callers that do not need filtering/pagination.
    pub fn list(&self) -> Result<Vec<DaemonHistoryEntry>, DaemonHistoryStoreError> {
        Ok(self.query(&DaemonHistoryQuery::default())?.entries)
    }

    /// Filtered, paginated listing, most recent first (ties broken by id
    /// descending, matching the id's embedded-millis ordering).
    pub fn query(
        &self,
        query: &DaemonHistoryQuery,
    ) -> Result<DaemonHistoryPage, DaemonHistoryStoreError> {
        let conn = self.connection()?;

        let mut where_clauses: Vec<String> = Vec::new();
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        let joined_search_table = query.search.is_some();

        if let Some(kind) = query.kind {
            where_clauses.push("e.kind = ?".to_string());
            params.push(Box::new(kind.as_str().to_string()));
        }
        let like_pattern = query.search.as_deref().map(like_substring_pattern);
        if let Some(pattern) = &like_pattern {
            where_clauses.push("f.search_text LIKE ? ESCAPE '\\'".to_string());
            params.push(Box::new(pattern.clone()));
        }

        let where_sql = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };
        let from_sql = if joined_search_table {
            "history_entries e JOIN history_entries_fts f ON f.id = e.id"
        } else {
            "history_entries e"
        };

        let total: usize = {
            let sql = format!("SELECT COUNT(*) FROM {from_sql} {where_sql}");
            let mut statement = conn.prepare(&sql).map_err(DaemonHistoryStoreError::Query)?;
            let param_refs: Vec<&dyn rusqlite::ToSql> =
                params.iter().map(|value| value.as_ref()).collect();
            let count: i64 = statement
                .query_row(param_refs.as_slice(), |row| row.get(0))
                .map_err(DaemonHistoryStoreError::Query)?;
            count.max(0) as usize
        };

        let limit_sql = match query.limit {
            Some(limit) => format!(" LIMIT {} OFFSET {}", limit, query.offset),
            None => {
                if query.offset > 0 {
                    // SQLite requires a LIMIT for OFFSET to take effect; -1 means
                    // "no limit".
                    format!(" LIMIT -1 OFFSET {}", query.offset)
                } else {
                    String::new()
                }
            }
        };
        let sql = format!(
            "SELECT e.id, e.kind, e.model, e.created_at_unix_seconds, e.source_name, \
             e.duration_seconds, e.output_format, e.diarization_active, e.provenance, \
             e.formats, e.preview \
             FROM {from_sql} {where_sql} \
             ORDER BY e.created_at_unix_seconds DESC, e.id DESC{limit_sql}"
        );
        let mut statement = conn.prepare(&sql).map_err(DaemonHistoryStoreError::Query)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> =
            params.iter().map(|value| value.as_ref()).collect();
        let entries = statement
            .query_map(param_refs.as_slice(), row_to_entry)
            .map_err(DaemonHistoryStoreError::Query)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(DaemonHistoryStoreError::Query)?;

        Ok(DaemonHistoryPage { entries, total })
    }

    pub fn get(&self, id: &str) -> Result<Option<DaemonHistoryDetail>, DaemonHistoryStoreError> {
        validate_history_id(id)?;
        let conn = self.connection()?;
        conn.query_row(
            "SELECT e.id, e.kind, e.model, e.created_at_unix_seconds, e.source_name, \
             e.duration_seconds, e.output_format, e.diarization_active, e.provenance, \
             e.formats, e.preview, e.text \
             FROM history_entries e WHERE e.id = ?1",
            params![id],
            |row| {
                let entry = row_to_entry(row)?;
                let text: String = row.get(11)?;
                Ok(DaemonHistoryDetail { entry, text })
            },
        )
        .optional()
        .map_err(DaemonHistoryStoreError::Query)
    }

    pub fn delete(&self, id: &str) -> Result<bool, DaemonHistoryStoreError> {
        validate_history_id(id)?;
        let mut conn = self.connection()?;
        let tx = conn.transaction().map_err(DaemonHistoryStoreError::Query)?;
        tx.execute("DELETE FROM history_entries_fts WHERE id = ?1", params![id])
            .map_err(DaemonHistoryStoreError::Query)?;
        let removed = tx
            .execute("DELETE FROM history_entries WHERE id = ?1", params![id])
            .map_err(DaemonHistoryStoreError::Query)?;
        tx.commit().map_err(DaemonHistoryStoreError::Query)?;
        Ok(removed > 0)
    }

    pub fn delete_older_than(
        &self,
        cutoff_unix_seconds: u64,
    ) -> Result<usize, DaemonHistoryStoreError> {
        let mut conn = self.connection()?;
        let tx = conn.transaction().map_err(DaemonHistoryStoreError::Query)?;
        tx.execute(
            "DELETE FROM history_entries_fts WHERE id IN \
             (SELECT id FROM history_entries WHERE created_at_unix_seconds < ?1)",
            params![cutoff_unix_seconds as i64],
        )
        .map_err(DaemonHistoryStoreError::Query)?;
        let removed = tx
            .execute(
                "DELETE FROM history_entries WHERE created_at_unix_seconds < ?1",
                params![cutoff_unix_seconds as i64],
            )
            .map_err(DaemonHistoryStoreError::Query)?;
        tx.commit().map_err(DaemonHistoryStoreError::Query)?;
        Ok(removed)
    }

    pub fn retain_most_recent(&self, max_entries: usize) -> Result<usize, DaemonHistoryStoreError> {
        let mut conn = self.connection()?;
        let tx = conn.transaction().map_err(DaemonHistoryStoreError::Query)?;
        let keep_cte = "WITH keep(id) AS (SELECT id FROM history_entries \
             ORDER BY created_at_unix_seconds DESC, id DESC LIMIT ?1)";
        tx.execute(
            &format!(
                "{keep_cte} DELETE FROM history_entries_fts WHERE id NOT IN (SELECT id FROM keep)"
            ),
            params![max_entries as i64],
        )
        .map_err(DaemonHistoryStoreError::Query)?;
        let removed = tx
            .execute(
                &format!(
                    "{keep_cte} DELETE FROM history_entries WHERE id NOT IN (SELECT id FROM keep)"
                ),
                params![max_entries as i64],
            )
            .map_err(DaemonHistoryStoreError::Query)?;
        tx.commit().map_err(DaemonHistoryStoreError::Query)?;
        Ok(removed)
    }

    pub fn record(
        &self,
        record: DaemonHistoryRecord,
    ) -> Result<DaemonHistoryEntry, DaemonHistoryStoreError> {
        validate_record(&record)?;
        let mut conn = self.connection()?;
        let tx = conn.transaction().map_err(DaemonHistoryStoreError::Query)?;

        let model = record.model.trim().to_string();
        let source_name = record
            .source_name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let formats = normalized_formats(record.formats);
        let preview = preview_text(&record.text);
        let created_at_unix_seconds = unix_seconds_now();
        let formats_json = serde_json::to_string(&formats).expect("Vec<String> always serializes");
        let search_text = format!(
            "{model} {} {}",
            source_name.as_deref().unwrap_or(""),
            record.text
        );

        let base = format!("hist-{}", unix_millis_now());
        let mut id = base.clone();
        let mut attempt = 0u16;
        loop {
            let insert_result = tx.execute(
                "INSERT INTO history_entries (\
                    id, kind, model, created_at_unix_seconds, source_name, duration_seconds, \
                    output_format, diarization_active, provenance, formats, preview, text\
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![
                    id,
                    record.kind.as_str(),
                    model,
                    created_at_unix_seconds as i64,
                    source_name,
                    record.duration_seconds,
                    record.output_format.map(ResponseFormat::as_str),
                    record.diarization_active,
                    record.provenance.map(DaemonHistoryProvenance::as_str),
                    formats_json,
                    preview,
                    record.text,
                ],
            );
            match insert_result {
                Ok(_) => break,
                Err(rusqlite::Error::SqliteFailure(error, _))
                    if error.code == rusqlite::ErrorCode::ConstraintViolation && attempt < 1000 =>
                {
                    attempt += 1;
                    id = format!("{base}-{attempt}");
                    continue;
                }
                Err(other) => return Err(DaemonHistoryStoreError::Query(other)),
            }
        }
        tx.execute(
            "INSERT INTO history_entries_fts (id, search_text) VALUES (?1, ?2)",
            params![id, search_text],
        )
        .map_err(DaemonHistoryStoreError::Query)?;
        tx.commit().map_err(DaemonHistoryStoreError::Query)?;

        Ok(DaemonHistoryEntry {
            id,
            kind: record.kind,
            model,
            created_at_unix_seconds,
            source_name,
            duration_seconds: record.duration_seconds,
            output_format: record.output_format,
            diarization_active: record.diarization_active,
            provenance: record.provenance,
            formats,
            preview,
        })
    }

    fn connection(&self) -> Result<Connection, DaemonHistoryStoreError> {
        std::fs::create_dir_all(&self.root).map_err(|source| {
            DaemonHistoryStoreError::CreateDir {
                path: self.root.clone(),
                source,
            }
        })?;
        let path = self.db_path();
        let conn =
            Connection::open(&path).map_err(|source| DaemonHistoryStoreError::OpenDatabase {
                path: path.clone(),
                source,
            })?;
        conn.busy_timeout(Duration::from_secs(5))
            .map_err(|source| DaemonHistoryStoreError::OpenDatabase {
                path: path.clone(),
                source,
            })?;
        // WAL lets concurrent readers proceed alongside a writer; the daemon
        // opens a fresh connection per call, so file-level locking (not an
        // in-process mutex) is what serializes writers across requests.
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|source| DaemonHistoryStoreError::OpenDatabase {
                path: path.clone(),
                source,
            })?;
        ensure_schema(&conn).map_err(|source| DaemonHistoryStoreError::OpenDatabase {
            path: path.clone(),
            source,
        })?;
        Ok(conn)
    }

    fn db_path(&self) -> PathBuf {
        self.root.join("history.db")
    }
}

fn ensure_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS history_entries (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at_unix_seconds INTEGER NOT NULL,
            source_name TEXT,
            duration_seconds REAL,
            output_format TEXT,
            diarization_active INTEGER,
            provenance TEXT,
            formats TEXT NOT NULL,
            preview TEXT NOT NULL,
            text TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS history_entries_created_at_idx
            ON history_entries (created_at_unix_seconds DESC, id DESC);
        CREATE INDEX IF NOT EXISTS history_entries_kind_idx ON history_entries (kind);
        CREATE VIRTUAL TABLE IF NOT EXISTS history_entries_fts USING fts5(
            id UNINDEXED,
            search_text,
            tokenize = 'trigram'
        );",
    )
}

fn row_to_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<DaemonHistoryEntry> {
    let kind: String = row.get(1)?;
    let created_at_unix_seconds: i64 = row.get(3)?;
    let output_format: Option<String> = row.get(6)?;
    let diarization_active: Option<bool> = row.get(7)?;
    let provenance: Option<String> = row.get(8)?;
    let formats_json: String = row.get(9)?;

    Ok(DaemonHistoryEntry {
        id: row.get(0)?,
        kind: DaemonHistoryKind::parse(&kind).unwrap_or(DaemonHistoryKind::File),
        model: row.get(2)?,
        created_at_unix_seconds: created_at_unix_seconds.max(0) as u64,
        source_name: row.get(4)?,
        duration_seconds: row.get(5)?,
        output_format: output_format
            .as_deref()
            .and_then(|value| <ResponseFormat as std::str::FromStr>::from_str(value).ok()),
        diarization_active,
        provenance: provenance
            .as_deref()
            .and_then(DaemonHistoryProvenance::parse),
        formats: serde_json::from_str(&formats_json).unwrap_or_else(|_| vec!["text".to_string()]),
        preview: row.get(10)?,
    })
}

/// Escapes `%`, `_`, and `\` for a `LIKE ... ESCAPE '\'` substring pattern,
/// then wraps the needle in `%...%`.
fn like_substring_pattern(needle: &str) -> String {
    let mut escaped = String::with_capacity(needle.len() + 2);
    escaped.push('%');
    for ch in needle.chars() {
        if matches!(ch, '%' | '_' | '\\') {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped.push('%');
    escaped
}

pub fn history_dir(openasr_home: impl AsRef<Path>) -> PathBuf {
    openasr_home.as_ref().join("history")
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
    fn daemon_history_store_records_lists_gets_and_deletes() {
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
        assert!(store.list().unwrap().is_empty());
        assert!(store.get(&entry.id).unwrap().is_none());
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
        assert!(store.get(&fresh.id).unwrap().is_some());
    }

    #[test]
    fn daemon_history_store_quarter_retention_prunes_at_the_90_day_boundary() {
        let temp = tempfile::tempdir().unwrap();
        let store = DaemonHistoryStore::open(temp.path());
        let now = unix_seconds_now();
        let max_age = crate::config::HistoryRetentionPolicy::Quarter
            .max_age_seconds()
            .expect("quarter is an age-based policy");
        assert_eq!(max_age, 90 * 24 * 60 * 60);
        let cutoff = now - max_age;

        let beyond = record_history_for_test(&store, "older than 90 days");
        let exactly = record_history_for_test(&store, "exactly 90 days old");
        let within = record_history_for_test(&store, "within 90 days");
        set_history_created_at_for_test(&store, &beyond.id, cutoff - 1);
        set_history_created_at_for_test(&store, &exactly.id, cutoff);
        set_history_created_at_for_test(&store, &within.id, cutoff + 1);

        // Strictly-older-than semantics: an entry created exactly at the
        // cutoff instant survives.
        assert_eq!(store.delete_older_than(cutoff).unwrap(), 1);
        assert!(store.get(&beyond.id).unwrap().is_none());
        assert!(store.get(&exactly.id).unwrap().is_some());
        assert!(store.get(&within.id).unwrap().is_some());
    }

    #[test]
    fn daemon_history_store_retains_most_recent_entries() {
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
        assert!(store.get(&oldest.id).unwrap().is_none());
        assert!(store.get(&middle.id).unwrap().is_some());
        assert!(store.get(&newest.id).unwrap().is_some());
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
        let store = DaemonHistoryStore::open(temp.path());
        store
            .record(DaemonHistoryRecord {
                kind: DaemonHistoryKind::File,
                model: "whisper-large-v3-turbo".to_string(),
                source_name: None,
                duration_seconds: None,
                output_format: None,
                diarization_active: None,
                provenance: None,
                formats: vec![],
                text: "legacy text".to_string(),
            })
            .unwrap();

        let entries = store.list().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].output_format, None);
        assert_eq!(entries[0].diarization_active, None);
        assert_eq!(entries[0].provenance, None);
        assert_eq!(entries[0].source_name, None);
        assert_eq!(entries[0].duration_seconds, None);
    }

    #[test]
    fn daemon_history_store_search_finds_chinese_substrings() {
        let temp = tempfile::tempdir().unwrap();
        let store = DaemonHistoryStore::open(temp.path());
        record_history_for_test(&store, "我们讨论了历史记录的设计方案");
        record_history_for_test(&store, "今天天气很好，适合散步");

        let page = store
            .query(&DaemonHistoryQuery {
                search: Some("历史".to_string()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(page.total, 1);
        assert_eq!(page.entries.len(), 1);
        assert!(page.entries[0].preview.contains("历史"));

        let miss = store
            .query(&DaemonHistoryQuery {
                search: Some("天气预报".to_string()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(miss.total, 0);
        assert!(miss.entries.is_empty());
    }

    #[test]
    fn daemon_history_store_query_filters_by_kind() {
        let temp = tempfile::tempdir().unwrap();
        let store = DaemonHistoryStore::open(temp.path());
        store
            .record(DaemonHistoryRecord {
                kind: DaemonHistoryKind::File,
                model: "whisper".to_string(),
                source_name: None,
                duration_seconds: None,
                output_format: None,
                diarization_active: None,
                provenance: None,
                formats: vec![],
                text: "file transcript".to_string(),
            })
            .unwrap();
        record_history_for_test(&store, "live transcript");

        let page = store
            .query(&DaemonHistoryQuery {
                kind: Some(DaemonHistoryKind::Live),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(page.total, 1);
        assert_eq!(page.entries[0].kind, DaemonHistoryKind::Live);
    }

    #[test]
    fn daemon_history_store_query_paginates_with_stable_order() {
        let temp = tempfile::tempdir().unwrap();
        let store = DaemonHistoryStore::open(temp.path());
        let first = record_history_for_test(&store, "entry one");
        let second = record_history_for_test(&store, "entry two");
        let third = record_history_for_test(&store, "entry three");
        set_history_created_at_for_test(&store, &first.id, 10);
        set_history_created_at_for_test(&store, &second.id, 20);
        set_history_created_at_for_test(&store, &third.id, 30);

        let page_one = store
            .query(&DaemonHistoryQuery {
                limit: Some(2),
                offset: 0,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(page_one.total, 3);
        assert_eq!(
            page_one
                .entries
                .iter()
                .map(|e| e.id.clone())
                .collect::<Vec<_>>(),
            vec![third.id.clone(), second.id.clone()]
        );

        let page_two = store
            .query(&DaemonHistoryQuery {
                limit: Some(2),
                offset: 2,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(page_two.total, 3);
        assert_eq!(
            page_two
                .entries
                .iter()
                .map(|e| e.id.clone())
                .collect::<Vec<_>>(),
            vec![first.id.clone()]
        );
    }

    #[test]
    fn daemon_history_store_reports_corrupt_database_without_panicking() {
        let temp = tempfile::tempdir().unwrap();
        let history_root = temp.path().join("history");
        std::fs::create_dir_all(&history_root).unwrap();
        // Not a valid SQLite file: any query against it must surface a typed
        // error, not panic the caller (e.g. an in-flight transcription).
        std::fs::write(history_root.join("history.db"), b"not a sqlite database").unwrap();

        let store = DaemonHistoryStore::open(temp.path());
        assert!(store.list().is_err());
        assert!(store.get("hist-anything").is_err());
        assert!(
            store
                .record(DaemonHistoryRecord {
                    kind: DaemonHistoryKind::File,
                    model: "whisper".to_string(),
                    source_name: None,
                    duration_seconds: None,
                    output_format: None,
                    diarization_active: None,
                    provenance: None,
                    formats: vec![],
                    text: "should not persist".to_string(),
                })
                .is_err()
        );
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
        let conn = store.connection().unwrap();
        conn.execute(
            "UPDATE history_entries SET created_at_unix_seconds = ?1 WHERE id = ?2",
            params![created_at as i64, id],
        )
        .unwrap();
    }
}
