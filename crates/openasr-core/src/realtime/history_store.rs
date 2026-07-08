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
use crate::api::backend::Segment;

/// Guards the one-time-per-database WAL switch and schema creation in
/// [`DaemonHistoryStore::connection`]. See that method for why this can't
/// rely on `busy_timeout` alone.
static CONNECTION_SETUP_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Columns shared by every query that needs to build a [`DaemonHistoryEntry`]
/// via [`row_to_entry`]. Deliberately selects `segments_json` instead of the
/// persisted (and no longer trusted -- see that column's schema comment)
/// `formats` column: `row_to_entry` derives the advertised formats from
/// whether this row actually has segments, at read time, rather than
/// replaying whatever a past write recorded.
const ENTRY_COLUMNS: &str = "e.id, e.kind, e.model, e.created_at_unix_seconds, e.source_name, \
     e.duration_seconds, e.output_format, e.diarization_active, e.provenance, e.segments_json, \
     e.preview";

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
    Recorded,
    UserInitiated,
}

impl DaemonHistoryProvenance {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Recorded => "recorded",
            Self::UserInitiated => "user_initiated",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "recorded" => Some(Self::Recorded),
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
    /// Per-segment transcript with word/speaker timing, in the same JSON shape
    /// as the transcription API's segments (`format/json.rs`). Empty for rows
    /// written before segments were persisted (and for live entries, which
    /// store only aggregated text). The desktop export UI reconstructs
    /// SRT/VTT/JSON from these; see [`DaemonHistoryDetail::to_transcription`].
    #[serde(default)]
    pub segments: Vec<Segment>,
}

impl DaemonHistoryDetail {
    /// Reconstructs a [`Transcription`](crate::api::backend::Transcription) from
    /// the stored transcript text and segments so the authoritative
    /// [`render_transcription`](crate::render_transcription) renderer can emit
    /// any export format. Language and long-form metadata are not persisted in
    /// daemon history, so they come back `None` rather than being fabricated.
    pub fn to_transcription(&self) -> crate::api::backend::Transcription {
        crate::api::backend::Transcription {
            text: self.text.clone(),
            segments: self.segments.clone(),
            longform: None,
            language: None,
        }
    }
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
    pub text: String,
    /// Per-segment transcript (word/speaker timing). File transcriptions pass
    /// `transcription.segments`; live entries pass an empty vec (they persist
    /// only aggregated text). The available export `formats` are derived from
    /// whether this is non-empty -- callers cannot claim a format the stored
    /// data cannot render.
    pub segments: Vec<Segment>,
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
            "SELECT {ENTRY_COLUMNS} \
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
            &format!("SELECT {ENTRY_COLUMNS}, e.text FROM history_entries e WHERE e.id = ?1"),
            params![id],
            |row| {
                let entry = row_to_entry(row)?;
                let text: String = row.get(11)?;
                // segments_json lives at the same ordinal `row_to_entry` already
                // read to derive `entry.formats`; re-read it here for the full
                // segment payload. A NULL column (rows written before segments
                // were persisted, or Live entries, which persist only aggregated
                // text) or an unparseable blob degrades to no segments: the
                // transcript text still comes back and detail fetches never 500
                // on a corrupt segment payload.
                let segments_json: Option<String> = row.get(9)?;
                let segments = parse_segments_json(segments_json);
                Ok(DaemonHistoryDetail {
                    entry,
                    text,
                    segments,
                })
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
        // Serialize segments (word/speaker timing) alongside the text so exports
        // can rebuild SRT/VTT/JSON later; `formats` is derived from their
        // presence so we never advertise a format the stored data can't render.
        // Serialization only fails on non-finite floats (NaN/Inf timestamps), a
        // narrow case that must not cost the whole row: degrade to a
        // segments-less (text-only) row instead of erroring out of the INSERT,
        // symmetric with the read path treating a corrupt/legacy blob as "no
        // segments" rather than failing the whole fetch.
        let segments_json = if record.segments.is_empty() {
            None
        } else {
            match serde_json::to_string(&record.segments) {
                Ok(json) => Some(json),
                Err(error) => {
                    eprintln!(
                        "history: could not serialize segments, recording \
                         text-only for this row: {error}"
                    );
                    None
                }
            }
        };
        let formats = formats_for_content(segments_json.is_some());
        let preview = preview_text(&record.text);
        let created_at_unix_seconds = unix_seconds_now();
        // `formats` is written for backward-compatible schema reasons only
        // (the column is `NOT NULL`); no read path trusts it -- see the
        // `formats` column comment in `ensure_schema` and `ENTRY_COLUMNS`.
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
                    output_format, diarization_active, provenance, formats, preview, text, \
                    segments_json\
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
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
                    segments_json,
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

    /// Serializes the one-time-per-`history.db` WAL switch and schema
    /// creation performed in [`Self::connection`] across all connections
    /// opened by this process. See the comment at the lock's call site for
    /// why this needs its own guard instead of relying on `busy_timeout`.
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
        // in-process mutex) is what serializes writers across requests once
        // the database is already in WAL mode.
        //
        // The *first* switch from the default rollback-journal into WAL mode
        // needs a brief exclusive lock to rewrite the file header, and that
        // specific lock acquisition does not go through the normal
        // busy-handler retry loop `busy_timeout` installs (that loop covers
        // ordinary step-time lock contention, not this one-time schema-level
        // transition) -- so concurrent first-time callers can observe an
        // immediate `SQLITE_BUSY` here instead of a patient wait. The schema
        // DDL right below has the same shape (CREATE TABLE/VIRTUAL TABLE IF
        // NOT EXISTS still touches the schema even when it is a no-op). Both
        // only matter once, on first use of a given `history.db`; serialize
        // them with an in-process mutex so at most one thread performs the
        // WAL switch/schema creation at a time instead of racing SQLite's
        // exclusive lock for it.
        let _setup_guard = CONNECTION_SETUP_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|source| DaemonHistoryStoreError::OpenDatabase {
                path: path.clone(),
                source,
            })?;
        ensure_schema(&conn).map_err(|source| DaemonHistoryStoreError::OpenDatabase {
            path: path.clone(),
            source,
        })?;
        drop(_setup_guard);
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
            -- Legacy: written for schema/back-compat continuity (NOT NULL) but
            -- no longer trusted by any read path. Rows written under 0.1.10 and
            -- earlier persisted a build-time `ResponseFormat::ALL` here that
            -- overclaimed srt/vtt for text-only rows; every read now derives
            -- `formats` from `segments_json` instead (see `ENTRY_COLUMNS` /
            -- `row_to_entry`), so old and new rows report equally honest
            -- formats regardless of what this column says.
            formats TEXT NOT NULL,
            preview TEXT NOT NULL,
            text TEXT NOT NULL,
            segments_json TEXT
        );
        CREATE INDEX IF NOT EXISTS history_entries_created_at_idx
            ON history_entries (created_at_unix_seconds DESC, id DESC);
        CREATE INDEX IF NOT EXISTS history_entries_kind_idx ON history_entries (kind);
        CREATE VIRTUAL TABLE IF NOT EXISTS history_entries_fts USING fts5(
            id UNINDEXED,
            search_text,
            tokenize = 'trigram'
        );",
    )?;
    ensure_segments_json_column(conn)
}

/// Additive migration for databases created before `segments_json` existed:
/// `CREATE TABLE IF NOT EXISTS` never alters an existing table, so an older
/// `history.db` keeps its columnless-of-`segments_json` shape until this adds
/// the nullable column. Existing rows read back as `NULL` (no segments), which
/// the query paths already treat as "text only". Idempotent: a fresh database
/// already has the column from the `CREATE TABLE` above and skips the `ALTER`.
fn ensure_segments_json_column(conn: &Connection) -> rusqlite::Result<()> {
    let has_column = conn
        .prepare("SELECT 1 FROM pragma_table_info('history_entries') WHERE name = 'segments_json'")?
        .exists([])?;
    if !has_column {
        conn.execute(
            "ALTER TABLE history_entries ADD COLUMN segments_json TEXT",
            [],
        )?;
    }
    Ok(())
}

/// Parses a persisted `segments_json` column into segments. A `NULL` column
/// (rows written before segments were persisted, or Live entries, which
/// persist only aggregated text) or an unparseable blob degrades to "no
/// segments" rather than surfacing an error -- callers must not 500 (or, for
/// `row_to_entry`, misreport `formats`) on a corrupt/legacy payload.
fn parse_segments_json(segments_json: Option<String>) -> Vec<Segment> {
    segments_json
        .and_then(|json| serde_json::from_str::<Vec<Segment>>(&json).ok())
        .unwrap_or_default()
}

/// Builds a [`DaemonHistoryEntry`] from a row selected via [`ENTRY_COLUMNS`].
/// `formats` is derived here from whether `segments_json` actually holds
/// segments, not read back from the persisted `formats` column: that column
/// can lie (rows written under 0.1.10 and earlier claimed srt/vtt for every
/// row regardless of content), so trusting it would let a stale write-time
/// snapshot re-surface a bug already fixed at read time. Deriving at read time
/// keeps old and new rows equally honest and needs no data migration.
fn row_to_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<DaemonHistoryEntry> {
    let kind: String = row.get(1)?;
    let created_at_unix_seconds: i64 = row.get(3)?;
    let output_format: Option<String> = row.get(6)?;
    let diarization_active: Option<bool> = row.get(7)?;
    let provenance: Option<String> = row.get(8)?;
    let segments_json: Option<String> = row.get(9)?;
    let has_segments = !parse_segments_json(segments_json).is_empty();

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
        formats: formats_for_content(has_segments),
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

/// Export formats the stored transcript can actually render, derived from
/// whether per-segment timing was persisted. Text/JSON/Markdown need only the
/// flat text; SRT/VTT additionally need segment timestamps. Deriving this here
/// (rather than trusting a caller-supplied list) is what keeps the advertised
/// `formats` honest -- a transcript with no segments never claims SRT/VTT.
/// Returned sorted to match the previous column ordering.
fn formats_for_content(has_segments: bool) -> Vec<String> {
    let mut formats = vec![
        ResponseFormat::Json.as_str().to_string(),
        ResponseFormat::Markdown.as_str().to_string(),
        ResponseFormat::Text.as_str().to_string(),
    ];
    if has_segments {
        formats.push(ResponseFormat::Srt.as_str().to_string());
        formats.push(ResponseFormat::Vtt.as_str().to_string());
    }
    formats.sort();
    formats
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
    use crate::api::backend::WordTimestamp;

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
                provenance: Some(DaemonHistoryProvenance::Recorded),
                text: "hello OpenASR history".to_string(),
                segments: vec![Segment {
                    start: 0.0,
                    end: 1.5,
                    text: "hello OpenASR history".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_profile_id: None,
                    words: Vec::new(),
                }],
            })
            .unwrap();

        let entries = store.list().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].output_format, Some(ResponseFormat::Srt));
        assert_eq!(entries[0].diarization_active, Some(true));
        assert_eq!(
            entries[0].provenance,
            Some(DaemonHistoryProvenance::Recorded)
        );
        // A row with segments advertises the timing-dependent formats too.
        assert_eq!(
            entries[0].formats,
            vec![
                "json".to_string(),
                "markdown".to_string(),
                "srt".to_string(),
                "text".to_string(),
                "vtt".to_string(),
            ]
        );
        let detail = store.get(&entry.id).unwrap().unwrap();
        assert_eq!(detail.text, "hello OpenASR history");
        assert_eq!(detail.entry.output_format, Some(ResponseFormat::Srt));
        assert_eq!(detail.entry.diarization_active, Some(true));
        assert_eq!(
            detail.entry.provenance,
            Some(DaemonHistoryProvenance::Recorded)
        );
        // Segments round-trip through the store and rebuild a Transcription that
        // the authoritative renderer turns into SRT with the persisted timing.
        assert_eq!(detail.segments.len(), 1);
        assert_eq!(detail.segments[0].end, 1.5);
        let srt =
            crate::render_transcription(&detail.to_transcription(), ResponseFormat::Srt).unwrap();
        assert!(srt.contains("00:00:00,000 --> 00:00:01,500"), "{srt}");
        assert!(srt.contains("hello OpenASR history"), "{srt}");

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
                        provenance: Some(DaemonHistoryProvenance::Recorded),
                        segments: Vec::new(),
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
            provenance: Some(DaemonHistoryProvenance::Recorded),
            formats: vec!["text".to_string()],
            preview: "hello".to_string(),
        };
        let detail = DaemonHistoryDetail {
            entry,
            text: "hello world".to_string(),
            segments: vec![Segment {
                start: 0.0,
                end: 1.0,
                text: "hello world".to_string(),
                speaker: Some("Speaker 1".to_string()),
                speaker_label: None,
                speaker_profile_id: None,
                words: vec![WordTimestamp {
                    word: "hello".to_string(),
                    start: 0.0,
                    end: 0.5,
                    confidence: None,
                }],
            }],
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
        assert_eq!(value["provenance"], "recorded");
        assert_eq!(value["text"], "hello world");
        assert!(value.get("response_format").is_none());
        assert!(value.get("text_path").is_none());
        // Segments serialize in the transcription API's JsonSegment shape so the
        // desktop export UI can reuse its existing segment deserializer: fields
        // in the same order, empty/None fields skipped.
        assert_eq!(value["segments"][0]["start"], 0.0);
        assert_eq!(value["segments"][0]["end"], 1.0);
        assert_eq!(value["segments"][0]["text"], "hello world");
        assert_eq!(value["segments"][0]["speaker"], "Speaker 1");
        assert!(value["segments"][0].get("speaker_label").is_none());
        assert_eq!(value["segments"][0]["words"][0]["word"], "hello");
        assert!(value["segments"][0]["words"][0].get("confidence").is_none());
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
                segments: Vec::new(),
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
    fn daemon_history_store_search_treats_like_wildcards_as_literals() {
        let temp = tempfile::tempdir().unwrap();
        let store = DaemonHistoryStore::open(temp.path());
        let percent = record_history_for_test(&store, "progress hit 100% today");
        let underscore = record_history_for_test(&store, "field a_b equals one");
        // Decoy: matches an unescaped '%a_b%' pattern ('_' = any one char), so
        // it only stays out of the results while escaping works.
        let decoy = record_history_for_test(&store, "field axb equals one");
        let backslash = record_history_for_test(&store, "path back\\slash here");

        let search = |needle: &str| {
            store
                .query(&DaemonHistoryQuery {
                    search: Some(needle.to_string()),
                    ..Default::default()
                })
                .unwrap()
        };

        // '%' unescaped matches every row; escaped it is a literal.
        let page = search("%");
        assert_eq!(page.total, 1);
        assert_eq!(page.entries[0].id, percent.id);

        // '_' unescaped is a single-character wildcard that would also match
        // the "axb" decoy; escaped it only hits the literal "a_b".
        let page = search("a_b");
        assert_eq!(page.total, 1);
        assert_eq!(page.entries[0].id, underscore.id);

        // '\' is the ESCAPE character itself and must match literally.
        let page = search("\\");
        assert_eq!(page.total, 1);
        assert_eq!(page.entries[0].id, backslash.id);

        // Ordinary substrings keep working alongside the escaping.
        let page = search("field");
        assert_eq!(page.total, 2);
        let ids: Vec<&str> = page.entries.iter().map(|entry| entry.id.as_str()).collect();
        assert!(ids.contains(&underscore.id.as_str()));
        assert!(ids.contains(&decoy.id.as_str()));
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
                segments: Vec::new(),
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
                    segments: Vec::new(),
                    text: "should not persist".to_string(),
                })
                .is_err()
        );
    }

    #[test]
    fn daemon_history_store_derives_formats_from_segment_presence() {
        let temp = tempfile::tempdir().unwrap();
        let store = DaemonHistoryStore::open(temp.path());

        // No segments: only the text-shaped formats are advertised.
        let text_only = store
            .record(DaemonHistoryRecord {
                kind: DaemonHistoryKind::File,
                model: "whisper".to_string(),
                source_name: None,
                duration_seconds: None,
                output_format: Some(ResponseFormat::Text),
                diarization_active: None,
                provenance: None,
                segments: Vec::new(),
                text: "plain body".to_string(),
            })
            .unwrap();
        assert_eq!(
            text_only.formats,
            vec![
                "json".to_string(),
                "markdown".to_string(),
                "text".to_string()
            ]
        );
        // NULL segments_json round-trips to an empty segment list.
        let detail = store.get(&text_only.id).unwrap().unwrap();
        assert!(detail.segments.is_empty());

        // With segments: SRT/VTT join the set because timing is now available.
        let with_segments = store
            .record(DaemonHistoryRecord {
                kind: DaemonHistoryKind::File,
                model: "whisper".to_string(),
                source_name: None,
                duration_seconds: Some(2.0),
                output_format: Some(ResponseFormat::Srt),
                diarization_active: None,
                provenance: None,
                segments: vec![Segment {
                    start: 0.0,
                    end: 2.0,
                    text: "timed body".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_profile_id: None,
                    words: Vec::new(),
                }],
                text: "timed body".to_string(),
            })
            .unwrap();
        assert_eq!(
            with_segments.formats,
            vec![
                "json".to_string(),
                "markdown".to_string(),
                "srt".to_string(),
                "text".to_string(),
                "vtt".to_string(),
            ]
        );
        // Segments rebuild a Transcription that the shared renderer turns into VTT.
        let detail = store.get(&with_segments.id).unwrap().unwrap();
        let vtt =
            crate::render_transcription(&detail.to_transcription(), ResponseFormat::Vtt).unwrap();
        assert!(vtt.starts_with("WEBVTT"), "{vtt}");
        assert!(vtt.contains("00:00:00.000 --> 00:00:02.000"), "{vtt}");
        assert!(vtt.contains("timed body"), "{vtt}");
    }

    #[test]
    fn daemon_history_store_adds_segments_column_to_a_legacy_database() {
        let temp = tempfile::tempdir().unwrap();
        let history_root = history_dir(temp.path());
        std::fs::create_dir_all(&history_root).unwrap();
        let db_path = history_root.join("history.db");

        // Recreate the pre-`segments_json` schema and a row written under it, so
        // the additive `ALTER TABLE` migration has something to upgrade.
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE history_entries (
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
                CREATE VIRTUAL TABLE history_entries_fts USING fts5(
                    id UNINDEXED, search_text, tokenize = 'trigram'
                );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO history_entries (\
                    id, kind, model, created_at_unix_seconds, source_name, duration_seconds, \
                    output_format, diarization_active, provenance, formats, preview, text\
                ) VALUES ('hist-legacy', 'file', 'whisper', 100, NULL, NULL, 'text', NULL, \
                    NULL, '[\"text\"]', 'legacy', 'legacy body')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO history_entries_fts (id, search_text) VALUES ('hist-legacy', 'legacy body')",
                [],
            )
            .unwrap();
        }

        // Opening through the store runs the migration; the legacy row survives
        // and reads back with no segments (not a crash).
        let store = DaemonHistoryStore::open(temp.path());
        let detail = store.get("hist-legacy").unwrap().unwrap();
        assert_eq!(detail.text, "legacy body");
        assert!(detail.segments.is_empty());

        // New writes populate the freshly added column end to end.
        let fresh = store
            .record(DaemonHistoryRecord {
                kind: DaemonHistoryKind::File,
                model: "whisper".to_string(),
                source_name: None,
                duration_seconds: Some(1.0),
                output_format: Some(ResponseFormat::Srt),
                diarization_active: None,
                provenance: None,
                segments: vec![Segment {
                    start: 0.0,
                    end: 1.0,
                    text: "new body".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_profile_id: None,
                    words: Vec::new(),
                }],
                text: "new body".to_string(),
            })
            .unwrap();
        let fresh_detail = store.get(&fresh.id).unwrap().unwrap();
        assert_eq!(fresh_detail.segments.len(), 1);
        assert_eq!(fresh_detail.segments[0].text, "new body");
    }

    /// Regression for the read-path fix: rows written under 0.1.10 and earlier
    /// persisted a `formats` column that unconditionally claimed every format
    /// (including srt/vtt) regardless of whether segments were ever stored.
    /// `segments_json` is `NULL` for these rows (segments did not exist yet),
    /// so the old "trust the persisted column" read path rendered an empty
    /// SRT/VTT for them. `row_to_entry` must ignore that lying column and
    /// derive `formats` from `segments_json` instead, so both `list()`/`query()`
    /// and `get()` report the honest (text-shaped only) set.
    #[test]
    fn daemon_history_store_read_path_corrects_legacy_lying_formats() {
        let temp = tempfile::tempdir().unwrap();
        let store = DaemonHistoryStore::open(temp.path());
        // Force schema creation (including the segments_json column) before
        // hand-inserting a row that predates it in spirit: formats overclaims
        // srt/vtt/verbose_json, segments_json is NULL.
        let conn = store.connection().unwrap();
        conn.execute(
            "INSERT INTO history_entries (\
                id, kind, model, created_at_unix_seconds, source_name, duration_seconds, \
                output_format, diarization_active, provenance, formats, preview, text, \
                segments_json\
            ) VALUES ('hist-legacy-lie', 'file', 'whisper', 100, NULL, NULL, 'text', NULL, \
                NULL, '[\"text\",\"srt\",\"vtt\",\"verbose_json\",\"json\",\"markdown\"]', \
                'legacy lie', 'legacy lie body', NULL)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO history_entries_fts (id, search_text) VALUES ('hist-legacy-lie', 'legacy lie body')",
            [],
        )
        .unwrap();
        drop(conn);

        let honest_formats = vec![
            "json".to_string(),
            "markdown".to_string(),
            "text".to_string(),
        ];

        let entries = store.list().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].formats, honest_formats,
            "list() must not echo the persisted formats column's srt/vtt overclaim"
        );

        let detail = store.get("hist-legacy-lie").unwrap().unwrap();
        assert_eq!(detail.entry.formats, honest_formats);
        assert!(detail.segments.is_empty());
    }

    // No test exercises `record()`'s "segments fail to serialize -> degrade to
    // text-only" branch (the fix B change in `record`) with a real trigger:
    // `serde_json::to_string` cannot actually fail for `Vec<Segment>`. Its only
    // fields are `String`/`f32`/`Option`/`Vec` with derived `Serialize` impls,
    // and non-finite floats do not error -- confirmed empirically
    // (`serde_json::to_string(&f32::NAN)` and `&f64::INFINITY` both return
    // `Ok("null")` on serde_json 1.0.149, the version pinned here). There is no
    // reachable input that produces `Err` from that call today, so the branch
    // is unreachable in practice; it is kept as defense-in-depth (symmetric
    // with the read path's "corrupt/legacy blob -> no segments" degrade) in
    // case that ever changes, e.g. a future field with a fallible custom
    // `Serialize` impl. Manufacturing a fake failure would require injecting a
    // test-only non-`Segment` payload into `record()`, which isn't a real
    // regression guard and isn't worth the extra surface.

    fn record_history_for_test(store: &DaemonHistoryStore, text: &str) -> DaemonHistoryEntry {
        store
            .record(DaemonHistoryRecord {
                kind: DaemonHistoryKind::Live,
                model: "whisper-large-v3-turbo".to_string(),
                source_name: None,
                duration_seconds: None,
                output_format: Some(ResponseFormat::Text),
                diarization_active: Some(false),
                provenance: Some(DaemonHistoryProvenance::Recorded),
                segments: Vec::new(),
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
