//! SQLite-backed Voice ID store.
//!
//! Path: `$OPENASR_HOME/diarize/voice-id.db` (override with `OPENASR_VOICE_ID_DB`).
//! Mutations run under `BEGIN IMMEDIATE`. Raw audio is never stored.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension, params};
use thiserror::Error;

use super::domain::{
    CaptureContext, ConsentRecord, EnrollmentSample, Person, PersonPrototype, PersonStatus,
    PersonView, PrototypeMember, SampleEmbedding, SampleQuality, SampleView,
};
use super::ids::{IdError, PersonId, PrototypeId, SampleId};
use super::matcher::{MatcherPerson, PersonMatcher};
use super::prototypes::{
    DEFAULT_CLUSTER_COSINE_DISTANCE, PrototypeSample, build_person_prototypes,
};
use super::space::EmbeddingSpace;
use crate::diarize::contract::SpeakerEmbedding;

pub const VOICE_ID_DB_ENV: &str = "OPENASR_VOICE_ID_DB";
pub const VOICE_ID_SCHEMA_VERSION: i32 = 1;

static CONNECTION_SETUP_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[derive(Debug, Error)]
pub enum VoiceIdStoreError {
    #[error("could not determine OpenASR home for voice-id store")]
    HomeUnavailable,
    #[error("could not create voice-id directory {path}: {source}")]
    CreateDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("could not open voice-id database {path}: {source}")]
    OpenDatabase {
        path: PathBuf,
        source: rusqlite::Error,
    },
    #[error("voice-id database error: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("voice-id person not found: {0}")]
    NotFound(String),
    #[error("voice-id sample not found: {0}")]
    SampleNotFound(String),
    #[error("voice-id display name must not be empty")]
    EmptyName,
    #[error("voice-id revision conflict for {id}: expected {expected}, found {found}")]
    RevisionConflict {
        id: String,
        expected: u64,
        found: u64,
    },
    #[error("voice-id person {0} is not active")]
    NotActive(String),
    #[error("{0}")]
    InvalidId(#[from] IdError),
    #[error("voice-id serialization error: {0}")]
    Serialize(String),
    #[error("voice-id migration failed: {0}")]
    Migration(String),
}

#[derive(Debug, Clone)]
pub struct VoiceIdStore {
    root: PathBuf,
    db_path: PathBuf,
}

impl VoiceIdStore {
    pub fn open(openasr_home: impl AsRef<Path>) -> Self {
        let root = openasr_home.as_ref().join("diarize");
        let db_path = std::env::var(VOICE_ID_DB_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("voice-id.db"));
        Self { root, db_path }
    }

    pub fn open_default() -> Result<Self, VoiceIdStoreError> {
        let home = crate::openasr_home().map_err(|_| VoiceIdStoreError::HomeUnavailable)?;
        Ok(Self::open(home))
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn list_persons(
        &self,
        active_space: Option<&EmbeddingSpace>,
    ) -> Result<Vec<PersonView>, VoiceIdStoreError> {
        let conn = self.connection()?;
        let mut stmt = conn.prepare(
            "SELECT person_id, display_name, status, created_at, updated_at, revision, color_preference
             FROM persons
             WHERE status != 'deleted'
             ORDER BY created_at ASC, person_id ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)? as u64,
                row.get::<_, Option<String>>(6)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (person_id, display_name, status, created_at, updated_at, revision, color) = row?;
            let status = PersonStatus::parse(&status).unwrap_or(PersonStatus::Deleted);
            let samples = load_sample_views(&conn, &person_id, active_space)?;
            let needs_reenrollment = samples.iter().all(|s| s.needs_reenrollment)
                || samples.is_empty()
                || !status.allows_matching();
            out.push(PersonView {
                person_id,
                display_name,
                status,
                created_at,
                updated_at,
                revision,
                sample_count: samples.len(),
                needs_reenrollment,
                color_preference: color,
                samples,
            });
        }
        Ok(out)
    }

    pub fn get_person(
        &self,
        person_id: &PersonId,
        active_space: Option<&EmbeddingSpace>,
    ) -> Result<PersonView, VoiceIdStoreError> {
        let conn = self.connection()?;
        let row = conn
            .query_row(
                "SELECT person_id, display_name, status, created_at, updated_at, revision, color_preference
                 FROM persons WHERE person_id = ?1",
                params![person_id.as_str()],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, i64>(5)? as u64,
                        row.get::<_, Option<String>>(6)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| VoiceIdStoreError::NotFound(person_id.as_str().to_string()))?;
        let status = PersonStatus::parse(&row.2).unwrap_or(PersonStatus::Deleted);
        if status == PersonStatus::Deleted {
            return Err(VoiceIdStoreError::NotFound(person_id.as_str().to_string()));
        }
        let samples = load_sample_views(&conn, &row.0, active_space)?;
        let needs_reenrollment = samples.iter().all(|s| s.needs_reenrollment) || samples.is_empty();
        Ok(PersonView {
            person_id: row.0,
            display_name: row.1,
            status,
            created_at: row.3,
            updated_at: row.4,
            revision: row.5,
            sample_count: samples.len(),
            needs_reenrollment,
            color_preference: row.6,
            samples,
        })
    }

    pub fn enroll_person(
        &self,
        display_name: impl Into<String>,
        consent: ConsentRecord,
        samples: Vec<NewSampleInput>,
        color_preference: Option<String>,
    ) -> Result<PersonView, VoiceIdStoreError> {
        let display_name = normalize_name(display_name.into())?;
        if samples.is_empty() {
            return Err(VoiceIdStoreError::Migration(
                "enrollment requires at least one accepted sample".into(),
            ));
        }
        let person_id = PersonId::generate();
        let now = timestamp_now();
        let conn = self.connection()?;
        immediate_transaction(&conn, || {
            conn.execute(
                "INSERT INTO persons(person_id, display_name, status, created_at, updated_at, revision, color_preference)
                 VALUES (?1, ?2, 'active', ?3, ?3, 1, ?4)",
                params![
                    person_id.as_str(),
                    display_name,
                    now,
                    color_preference
                ],
            )?;
            for sample in &samples {
                insert_sample(&conn, &person_id, sample, &consent, &now)?;
            }
            rebuild_prototypes_for_person(&conn, &person_id)?;
            bump_global_revision(&conn)?;
            Ok(())
        })?;
        self.get_person(&person_id, samples.first().map(|s| &s.space))
    }

    pub fn add_sample(
        &self,
        person_id: &PersonId,
        expected_revision: Option<u64>,
        consent: ConsentRecord,
        sample: NewSampleInput,
    ) -> Result<PersonView, VoiceIdStoreError> {
        let conn = self.connection()?;
        let space = sample.space.clone();
        immediate_transaction(&conn, || {
            let (status, revision) = person_status_revision(&conn, person_id)?;
            if !status.allows_matching() {
                return Err(VoiceIdStoreError::NotActive(person_id.as_str().to_string()));
            }
            if let Some(expected) = expected_revision
                && expected != revision
            {
                return Err(VoiceIdStoreError::RevisionConflict {
                    id: person_id.as_str().to_string(),
                    expected,
                    found: revision,
                });
            }
            let now = timestamp_now();
            insert_sample(&conn, person_id, &sample, &consent, &now)?;
            rebuild_prototypes_for_person(&conn, person_id)?;
            touch_person(&conn, person_id, &now)?;
            bump_global_revision(&conn)?;
            Ok(())
        })?;
        self.get_person(person_id, Some(&space))
    }

    pub fn rename_person(
        &self,
        person_id: &PersonId,
        display_name: impl Into<String>,
        expected_revision: Option<u64>,
    ) -> Result<PersonView, VoiceIdStoreError> {
        let display_name = normalize_name(display_name.into())?;
        let conn = self.connection()?;
        immediate_transaction(&conn, || {
            let (status, revision) = person_status_revision(&conn, person_id)?;
            if status == PersonStatus::Deleted {
                return Err(VoiceIdStoreError::NotFound(person_id.as_str().to_string()));
            }
            if let Some(expected) = expected_revision
                && expected != revision
            {
                return Err(VoiceIdStoreError::RevisionConflict {
                    id: person_id.as_str().to_string(),
                    expected,
                    found: revision,
                });
            }
            let now = timestamp_now();
            conn.execute(
                "UPDATE persons SET display_name = ?1, updated_at = ?2, revision = revision + 1 WHERE person_id = ?3",
                params![display_name, now, person_id.as_str()],
            )?;
            bump_global_revision(&conn)?;
            Ok(())
        })?;
        self.get_person(person_id, None)
    }

    pub fn delete_sample(
        &self,
        sample_id: &SampleId,
        expected_person_revision: Option<u64>,
    ) -> Result<PersonView, VoiceIdStoreError> {
        let conn = self.connection()?;
        let person_id = immediate_transaction(&conn, || {
            let person_id = conn
                .query_row(
                    "SELECT person_id FROM enrollment_samples WHERE sample_id = ?1",
                    params![sample_id.as_str()],
                    |row| row.get::<_, String>(0),
                )
                .optional()?
                .ok_or_else(|| VoiceIdStoreError::SampleNotFound(sample_id.as_str().to_string()))?;
            let person_id = PersonId::parse(person_id)?;
            let (status, revision) = person_status_revision(&conn, &person_id)?;
            if status == PersonStatus::Deleted {
                return Err(VoiceIdStoreError::NotFound(person_id.as_str().to_string()));
            }
            if let Some(expected) = expected_person_revision
                && expected != revision
            {
                return Err(VoiceIdStoreError::RevisionConflict {
                    id: person_id.as_str().to_string(),
                    expected,
                    found: revision,
                });
            }
            purge_sample(&conn, sample_id)?;
            rebuild_prototypes_for_person(&conn, &person_id)?;
            touch_person(&conn, &person_id, &timestamp_now())?;
            bump_global_revision(&conn)?;
            Ok(person_id)
        })?;
        self.get_person(&person_id, None)
    }

    pub fn delete_person(
        &self,
        person_id: &PersonId,
        expected_revision: Option<u64>,
        reason: &str,
    ) -> Result<(), VoiceIdStoreError> {
        self.purge_person(person_id, expected_revision, PersonStatus::Deleted, reason)
    }

    pub fn revoke_consent(
        &self,
        person_id: &PersonId,
        expected_revision: Option<u64>,
        reason: &str,
    ) -> Result<(), VoiceIdStoreError> {
        self.purge_person(
            person_id,
            expected_revision,
            PersonStatus::ConsentRevoked,
            reason,
        )
    }

    fn purge_person(
        &self,
        person_id: &PersonId,
        expected_revision: Option<u64>,
        final_status: PersonStatus,
        reason: &str,
    ) -> Result<(), VoiceIdStoreError> {
        let conn = self.connection()?;
        immediate_transaction(&conn, || {
            let (status, revision) = person_status_revision(&conn, person_id)?;
            if status == PersonStatus::Deleted && final_status == PersonStatus::Deleted {
                return Err(VoiceIdStoreError::NotFound(person_id.as_str().to_string()));
            }
            if let Some(expected) = expected_revision
                && expected != revision
            {
                return Err(VoiceIdStoreError::RevisionConflict {
                    id: person_id.as_str().to_string(),
                    expected,
                    found: revision,
                });
            }
            let sample_ids = conn
                .prepare("SELECT sample_id FROM enrollment_samples WHERE person_id = ?1")?
                .query_map(params![person_id.as_str()], |row| row.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?;
            for sample_id in sample_ids {
                purge_sample(&conn, &SampleId::parse(sample_id)?)?;
            }
            conn.execute(
                "DELETE FROM prototypes WHERE person_id = ?1",
                params![person_id.as_str()],
            )?;
            let now = timestamp_now();
            conn.execute(
                "UPDATE persons SET status = ?1, updated_at = ?2, revision = revision + 1,
                 display_name = CASE WHEN ?1 = 'deleted' THEN display_name ELSE display_name END
                 WHERE person_id = ?3",
                params![final_status.as_str(), now, person_id.as_str()],
            )?;
            conn.execute(
                "INSERT OR REPLACE INTO person_tombstones(person_id, revoked_or_deleted_at, reason)
                 VALUES (?1, ?2, ?3)",
                params![person_id.as_str(), now, reason],
            )?;
            bump_global_revision(&conn)?;
            Ok(())
        })
    }

    pub fn resolve_legacy_profile_id(
        &self,
        legacy_profile_id: &str,
    ) -> Result<Option<PersonId>, VoiceIdStoreError> {
        let conn = self.connection()?;
        let person_id = conn
            .query_row(
                "SELECT person_id FROM legacy_profile_aliases WHERE legacy_profile_id = ?1",
                params![legacy_profile_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        Ok(person_id.map(PersonId::parse).transpose()?)
    }

    pub fn insert_legacy_alias(
        &self,
        legacy_profile_id: &str,
        person_id: &PersonId,
    ) -> Result<(), VoiceIdStoreError> {
        let conn = self.connection()?;
        conn.execute(
            "INSERT OR REPLACE INTO legacy_profile_aliases(legacy_profile_id, person_id)
             VALUES (?1, ?2)",
            params![legacy_profile_id, person_id.as_str()],
        )?;
        Ok(())
    }

    /// Build a matcher for the active embedding space and optional candidate scope.
    pub fn matcher_for_space(
        &self,
        space: &EmbeddingSpace,
        accept_threshold: f32,
        margin: f32,
    ) -> Result<PersonMatcher, VoiceIdStoreError> {
        if !space.is_matchable() {
            return Ok(PersonMatcher::new(
                space.clone(),
                Vec::new(),
                accept_threshold,
                margin,
            ));
        }
        let conn = self.connection()?;
        let mut persons_stmt = conn.prepare(
            "SELECT person_id, display_name, status FROM persons WHERE status = 'active'",
        )?;
        let person_rows = persons_stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        let mut persons = Vec::new();
        for row in person_rows {
            let (person_id_raw, display_name, status_raw) = row?;
            let person_id = PersonId::parse(person_id_raw)?;
            let status = PersonStatus::parse(&status_raw).unwrap_or(PersonStatus::Deleted);
            let prototypes = load_prototypes(&conn, &person_id, space)?;
            if prototypes.is_empty() {
                continue;
            }
            let member_embeddings = load_member_embeddings(&conn, &person_id, space)?;
            persons.push(MatcherPerson {
                person_id,
                display_name,
                status,
                prototypes,
                member_embeddings,
            });
        }
        Ok(PersonMatcher::new(
            space.clone(),
            persons,
            accept_threshold,
            margin,
        ))
    }

    pub fn export_metadata_json(&self) -> Result<String, VoiceIdStoreError> {
        let persons = self.list_persons(None)?;
        serde_json::to_string_pretty(&serde_json::json!({
            "version": 1,
            "exported_at": timestamp_now(),
            "includes_embeddings": false,
            "persons": persons,
        }))
        .map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))
    }

    /// Mark migration ledger state. Used by the v1 JSON importer.
    pub fn set_migration_state(&self, key: &str, value: &str) -> Result<(), VoiceIdStoreError> {
        let conn = self.connection()?;
        conn.execute(
            "INSERT INTO voice_id_meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn migration_state(&self, key: &str) -> Result<Option<String>, VoiceIdStoreError> {
        let conn = self.connection()?;
        Ok(conn
            .query_row(
                "SELECT value FROM voice_id_meta WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?)
    }

    /// Resolve a caller-supplied id that may be a v2 `person_*` id or a legacy
    /// `vp_*` alias from the v1 JSON store.
    pub fn resolve_person_ref(&self, raw: &str) -> Result<PersonId, VoiceIdStoreError> {
        if let Ok(person_id) = PersonId::parse(raw) {
            // Confirm the person still exists and is not deleted.
            let _ = self.get_person(&person_id, None)?;
            return Ok(person_id);
        }
        if let Some(person_id) = self.resolve_legacy_profile_id(raw)? {
            return Ok(person_id);
        }
        Err(VoiceIdStoreError::NotFound(raw.to_string()))
    }

    /// Prefer a legacy alias when one exists so older clients keep seeing `vp_*`.
    pub fn preferred_public_id(&self, person_id: &PersonId) -> Result<String, VoiceIdStoreError> {
        let conn = self.connection()?;
        let legacy = conn
            .query_row(
                "SELECT legacy_profile_id FROM legacy_profile_aliases
                 WHERE person_id = ?1
                 ORDER BY legacy_profile_id ASC
                 LIMIT 1",
                params![person_id.as_str()],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        Ok(legacy.unwrap_or_else(|| person_id.as_str().to_string()))
    }

    /// Low-level import helper used by migration: insert a fully built person
    /// graph in one IMMEDIATE transaction. Skips when `legacy_profile_id` is
    /// already present so partial prior runs stay idempotent.
    pub fn import_person_graph(
        &self,
        person: &Person,
        samples: &[(EnrollmentSample, SampleEmbedding)],
        legacy_profile_id: Option<&str>,
    ) -> Result<bool, VoiceIdStoreError> {
        let conn = self.connection()?;
        immediate_transaction(&conn, || {
            import_person_graph_on_conn(&conn, person, samples, legacy_profile_id)
        })
    }

    /// Import many migrated persons and advance the migration ledger in the
    /// same IMMEDIATE transaction. Each entry is skipped when its legacy id is
    /// already aliased, so a crash mid-import followed by retry cannot create
    /// duplicate Alice persons.
    pub fn import_migrated_profiles_atomic(
        &self,
        imports: &[(
            Person,
            Vec<(EnrollmentSample, SampleEmbedding)>,
            String, /* legacy_profile_id */
        )],
        ledger_key: &str,
        ledger_value: &str,
    ) -> Result<usize, VoiceIdStoreError> {
        let conn = self.connection()?;
        immediate_transaction(&conn, || {
            let mut imported = 0usize;
            for (person, samples, legacy_id) in imports {
                if import_person_graph_on_conn(&conn, person, samples, Some(legacy_id.as_str()))? {
                    imported += 1;
                }
            }
            conn.execute(
                "INSERT INTO voice_id_meta(key, value) VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![ledger_key, ledger_value],
            )?;
            Ok(imported)
        })
    }

    fn connection(&self) -> Result<Connection, VoiceIdStoreError> {
        if let Some(parent) = self.db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| VoiceIdStoreError::CreateDir {
                path: parent.to_path_buf(),
                source,
            })?;
            set_owner_only_dir_permissions(parent);
        } else {
            std::fs::create_dir_all(&self.root).map_err(|source| VoiceIdStoreError::CreateDir {
                path: self.root.clone(),
                source,
            })?;
            set_owner_only_dir_permissions(&self.root);
        }
        let conn =
            Connection::open(&self.db_path).map_err(|source| VoiceIdStoreError::OpenDatabase {
                path: self.db_path.clone(),
                source,
            })?;
        conn.busy_timeout(Duration::from_secs(5))
            .map_err(|source| VoiceIdStoreError::OpenDatabase {
                path: self.db_path.clone(),
                source,
            })?;
        let _guard = CONNECTION_SETUP_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|source| VoiceIdStoreError::OpenDatabase {
                path: self.db_path.clone(),
                source,
            })?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(|source| VoiceIdStoreError::OpenDatabase {
                path: self.db_path.clone(),
                source,
            })?;
        ensure_schema(&conn).map_err(|source| VoiceIdStoreError::OpenDatabase {
            path: self.db_path.clone(),
            source,
        })?;
        set_owner_only_file_permissions(&self.db_path);
        // Best-effort protect WAL/SHM siblings.
        let wal = PathBuf::from(format!("{}-wal", self.db_path.display()));
        let shm = PathBuf::from(format!("{}-shm", self.db_path.display()));
        set_owner_only_file_permissions(&wal);
        set_owner_only_file_permissions(&shm);
        Ok(conn)
    }
}

#[derive(Debug, Clone)]
pub struct NewSampleInput {
    pub sample_id: SampleId,
    pub capture_context: CaptureContext,
    pub quality: SampleQuality,
    pub space: EmbeddingSpace,
    pub embedding: SpeakerEmbedding,
}

fn ensure_schema(conn: &Connection) -> rusqlite::Result<()> {
    let user_version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if user_version == 0 {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS voice_id_meta (
                key TEXT PRIMARY KEY NOT NULL,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS persons (
                person_id TEXT PRIMARY KEY NOT NULL,
                display_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                revision INTEGER NOT NULL,
                color_preference TEXT
            );
            CREATE TABLE IF NOT EXISTS enrollment_samples (
                sample_id TEXT PRIMARY KEY NOT NULL,
                person_id TEXT NOT NULL REFERENCES persons(person_id),
                created_at TEXT NOT NULL,
                consent_json TEXT NOT NULL,
                quality_json TEXT NOT NULL,
                context_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embedding_spaces (
                space_id TEXT PRIMARY KEY NOT NULL,
                canonical_json TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                pack_fingerprint TEXT NOT NULL,
                legacy_unverifiable INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS sample_embeddings (
                sample_id TEXT NOT NULL REFERENCES enrollment_samples(sample_id) ON DELETE CASCADE,
                space_id TEXT NOT NULL REFERENCES embedding_spaces(space_id),
                embedding_blob BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                PRIMARY KEY (sample_id, space_id)
            );
            CREATE TABLE IF NOT EXISTS prototypes (
                prototype_id TEXT PRIMARY KEY NOT NULL,
                person_id TEXT NOT NULL REFERENCES persons(person_id),
                space_id TEXT NOT NULL REFERENCES embedding_spaces(space_id),
                medoid_sample_id TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                medoid_blob BLOB NOT NULL,
                medoid_dim INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prototype_members (
                prototype_id TEXT NOT NULL REFERENCES prototypes(prototype_id) ON DELETE CASCADE,
                sample_id TEXT NOT NULL,
                quality_weight REAL NOT NULL,
                PRIMARY KEY (prototype_id, sample_id)
            );
            CREATE TABLE IF NOT EXISTS legacy_profile_aliases (
                legacy_profile_id TEXT PRIMARY KEY NOT NULL,
                person_id TEXT NOT NULL REFERENCES persons(person_id)
            );
            CREATE TABLE IF NOT EXISTS person_tombstones (
                person_id TEXT PRIMARY KEY NOT NULL,
                revoked_or_deleted_at TEXT NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS enrollment_samples_person_idx
                ON enrollment_samples(person_id);
            CREATE INDEX IF NOT EXISTS prototypes_person_space_idx
                ON prototypes(person_id, space_id);
            ",
        )?;
        conn.pragma_update(None, "user_version", VOICE_ID_SCHEMA_VERSION)?;
        conn.execute(
            "INSERT OR IGNORE INTO voice_id_meta(key, value) VALUES ('schema_version', ?1)",
            params![VOICE_ID_SCHEMA_VERSION.to_string()],
        )?;
    } else if user_version != VOICE_ID_SCHEMA_VERSION {
        // Future migrations land here. Unknown newer/older versions fail closed.
        return Err(rusqlite::Error::InvalidQuery);
    }
    Ok(())
}

fn insert_sample(
    conn: &Connection,
    person_id: &PersonId,
    sample: &NewSampleInput,
    consent: &ConsentRecord,
    now: &str,
) -> Result<(), VoiceIdStoreError> {
    let consent_json =
        serde_json::to_string(consent).map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
    let quality_json = serde_json::to_string(&sample.quality)
        .map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
    let context_json = serde_json::to_string(&sample.capture_context)
        .map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
    conn.execute(
        "INSERT INTO enrollment_samples(sample_id, person_id, created_at, consent_json, quality_json, context_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            sample.sample_id.as_str(),
            person_id.as_str(),
            now,
            consent_json,
            quality_json,
            context_json
        ],
    )?;
    upsert_space(conn, &sample.space)?;
    let blob = embedding_to_blob(&sample.embedding)?;
    conn.execute(
        "INSERT INTO sample_embeddings(sample_id, space_id, embedding_blob, embedding_dim)
         VALUES (?1, ?2, ?3, ?4)",
        params![
            sample.sample_id.as_str(),
            sample.space.space_id,
            blob,
            sample.embedding.dim() as i64
        ],
    )?;
    Ok(())
}

fn purge_sample(conn: &Connection, sample_id: &SampleId) -> Result<(), VoiceIdStoreError> {
    conn.execute(
        "DELETE FROM prototype_members WHERE sample_id = ?1",
        params![sample_id.as_str()],
    )?;
    conn.execute(
        "DELETE FROM sample_embeddings WHERE sample_id = ?1",
        params![sample_id.as_str()],
    )?;
    conn.execute(
        "DELETE FROM enrollment_samples WHERE sample_id = ?1",
        params![sample_id.as_str()],
    )?;
    Ok(())
}

fn rebuild_prototypes_for_person(
    conn: &Connection,
    person_id: &PersonId,
) -> Result<(), VoiceIdStoreError> {
    conn.execute(
        "DELETE FROM prototype_members WHERE prototype_id IN (
            SELECT prototype_id FROM prototypes WHERE person_id = ?1
         )",
        params![person_id.as_str()],
    )?;
    conn.execute(
        "DELETE FROM prototypes WHERE person_id = ?1",
        params![person_id.as_str()],
    )?;

    // Group sample embeddings by space_id.
    let mut stmt = conn.prepare(
        "SELECT se.sample_id, se.space_id, se.embedding_blob, se.embedding_dim, es.canonical_json, s.quality_json
         FROM sample_embeddings se
         JOIN enrollment_samples s ON s.sample_id = se.sample_id
         JOIN embedding_spaces es ON es.space_id = se.space_id
         WHERE s.person_id = ?1",
    )?;
    let rows = stmt.query_map(params![person_id.as_str()], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Vec<u8>>(2)?,
            row.get::<_, i64>(3)? as usize,
            row.get::<_, String>(4)?,
            row.get::<_, String>(5)?,
        ))
    })?;

    let mut by_space: std::collections::BTreeMap<String, (EmbeddingSpace, Vec<PrototypeSample>)> =
        std::collections::BTreeMap::new();
    for row in rows {
        let (sample_id, _space_id, blob, dim, space_json, quality_json) = row?;
        let space: EmbeddingSpace = serde_json::from_str(&space_json)
            .map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
        if !space.is_matchable() {
            continue;
        }
        let embedding = blob_to_embedding(&blob, dim)?;
        let quality: SampleQuality = serde_json::from_str(&quality_json)
            .map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
        let entry = by_space
            .entry(space.space_id.clone())
            .or_insert_with(|| (space, Vec::new()));
        entry.1.push(PrototypeSample {
            sample_id: SampleId::parse(sample_id)?,
            embedding,
            quality,
        });
    }

    for (space, samples) in by_space.values() {
        let prototypes =
            build_person_prototypes(person_id, space, samples, DEFAULT_CLUSTER_COSINE_DISTANCE);
        for prototype in prototypes {
            persist_prototype(conn, &prototype)?;
        }
    }
    Ok(())
}

fn persist_prototype(
    conn: &Connection,
    prototype: &PersonPrototype,
) -> Result<(), VoiceIdStoreError> {
    upsert_space(conn, &prototype.space)?;
    let blob = embedding_to_blob(&prototype.medoid_embedding)?;
    conn.execute(
        "INSERT INTO prototypes(
            prototype_id, person_id, space_id, medoid_sample_id, policy_version, medoid_blob, medoid_dim
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            prototype.prototype_id.as_str(),
            prototype.person_id.as_str(),
            prototype.space.space_id,
            prototype.medoid_sample_id.as_str(),
            prototype.policy_version,
            blob,
            prototype.medoid_embedding.dim() as i64
        ],
    )?;
    for member in &prototype.members {
        conn.execute(
            "INSERT INTO prototype_members(prototype_id, sample_id, quality_weight)
             VALUES (?1, ?2, ?3)",
            params![
                prototype.prototype_id.as_str(),
                member.sample_id.as_str(),
                member.quality_weight as f64
            ],
        )?;
    }
    Ok(())
}

fn load_prototypes(
    conn: &Connection,
    person_id: &PersonId,
    space: &EmbeddingSpace,
) -> Result<Vec<PersonPrototype>, VoiceIdStoreError> {
    let mut stmt = conn.prepare(
        "SELECT prototype_id, medoid_sample_id, policy_version, medoid_blob, medoid_dim
         FROM prototypes WHERE person_id = ?1 AND space_id = ?2",
    )?;
    let rows = stmt.query_map(params![person_id.as_str(), space.space_id], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Vec<u8>>(3)?,
            row.get::<_, i64>(4)? as usize,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (prototype_id, medoid_sample_id, policy_version, blob, dim) = row?;
        let members = conn
            .prepare(
                "SELECT sample_id, quality_weight FROM prototype_members WHERE prototype_id = ?1",
            )?
            .query_map(params![prototype_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? as f32))
            })?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|(sample_id, quality_weight)| {
                Ok(PrototypeMember {
                    sample_id: SampleId::parse(sample_id)?,
                    quality_weight,
                })
            })
            .collect::<Result<Vec<_>, VoiceIdStoreError>>()?;
        out.push(PersonPrototype {
            prototype_id: PrototypeId::parse(prototype_id)?,
            person_id: person_id.clone(),
            space: space.clone(),
            medoid_sample_id: SampleId::parse(medoid_sample_id)?,
            medoid_embedding: blob_to_embedding(&blob, dim)?,
            policy_version,
            members,
        });
    }
    Ok(out)
}

fn load_member_embeddings(
    conn: &Connection,
    person_id: &PersonId,
    space: &EmbeddingSpace,
) -> Result<Vec<(SampleId, SpeakerEmbedding, f32)>, VoiceIdStoreError> {
    let mut stmt = conn.prepare(
        "SELECT se.sample_id, se.embedding_blob, se.embedding_dim, s.quality_json
         FROM sample_embeddings se
         JOIN enrollment_samples s ON s.sample_id = se.sample_id
         WHERE s.person_id = ?1 AND se.space_id = ?2",
    )?;
    let rows = stmt.query_map(params![person_id.as_str(), space.space_id], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Vec<u8>>(1)?,
            row.get::<_, i64>(2)? as usize,
            row.get::<_, String>(3)?,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (sample_id, blob, dim, quality_json) = row?;
        let quality: SampleQuality = serde_json::from_str(&quality_json)
            .map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
        out.push((
            SampleId::parse(sample_id)?,
            blob_to_embedding(&blob, dim)?,
            quality.weight(),
        ));
    }
    Ok(out)
}

fn load_sample_views(
    conn: &Connection,
    person_id: &str,
    active_space: Option<&EmbeddingSpace>,
) -> Result<Vec<SampleView>, VoiceIdStoreError> {
    let mut stmt = conn.prepare(
        "SELECT sample_id, created_at, quality_json, context_json FROM enrollment_samples
         WHERE person_id = ?1 ORDER BY created_at ASC, sample_id ASC",
    )?;
    let rows = stmt.query_map(params![person_id], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (sample_id, created_at, quality_json, context_json) = row?;
        let quality: SampleQuality = serde_json::from_str(&quality_json)
            .map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
        let capture_context: CaptureContext = serde_json::from_str(&context_json)
            .map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
        let spaces = conn
            .prepare(
                "SELECT es.canonical_json FROM sample_embeddings se
                 JOIN embedding_spaces es ON es.space_id = se.space_id
                 WHERE se.sample_id = ?1",
            )?
            .query_map(params![sample_id], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        // needs_reenrollment is relative to the active matching space when one
        // is supplied: a sample that is matchable in some other pack/space still
        // needs reenrollment for the currently loaded embedder.
        let mut space_compatible = false;
        let mut needs_reenrollment = true;
        for space_json in spaces {
            let space: EmbeddingSpace = serde_json::from_str(&space_json)
                .map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
            match active_space {
                Some(active) => {
                    if space.is_comparable_to(active) {
                        space_compatible = true;
                        needs_reenrollment = false;
                    }
                }
                None => {
                    if space.is_matchable() {
                        space_compatible = true;
                        needs_reenrollment = false;
                    }
                }
            }
        }
        out.push(SampleView {
            sample_id,
            created_at,
            sample_label: capture_context.sample_label.clone(),
            quality,
            capture_context,
            space_compatible,
            needs_reenrollment,
        });
    }
    Ok(out)
}

/// Insert one migrated person graph on an open connection. Returns `false` when
/// `legacy_profile_id` is already aliased (idempotent skip). Callers must hold
/// an open IMMEDIATE transaction when batching with the migration ledger.
fn import_person_graph_on_conn(
    conn: &Connection,
    person: &Person,
    samples: &[(EnrollmentSample, SampleEmbedding)],
    legacy_profile_id: Option<&str>,
) -> Result<bool, VoiceIdStoreError> {
    if let Some(legacy) = legacy_profile_id {
        let exists = conn
            .query_row(
                "SELECT 1 FROM legacy_profile_aliases WHERE legacy_profile_id = ?1",
                params![legacy],
                |_row| Ok(1i32),
            )
            .optional()?
            .is_some();
        if exists {
            return Ok(false);
        }
    }

    conn.execute(
        "INSERT INTO persons(person_id, display_name, status, created_at, updated_at, revision, color_preference)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            person.person_id.as_str(),
            person.display_name,
            person.status.as_str(),
            person.created_at,
            person.updated_at,
            person.revision as i64,
            person.color_preference,
        ],
    )?;
    for (sample, embedding) in samples {
        let consent_json = serde_json::to_string(&sample.consent)
            .map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
        let quality_json = serde_json::to_string(&sample.quality)
            .map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
        let context_json = serde_json::to_string(&sample.capture_context)
            .map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
        conn.execute(
            "INSERT INTO enrollment_samples(
                sample_id, person_id, created_at, consent_json, quality_json, context_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                sample.sample_id.as_str(),
                sample.person_id.as_str(),
                sample.created_at,
                consent_json,
                quality_json,
                context_json
            ],
        )?;
        upsert_space(conn, &embedding.space)?;
        let blob = embedding_to_blob(&embedding.embedding)?;
        conn.execute(
            "INSERT INTO sample_embeddings(sample_id, space_id, embedding_blob, embedding_dim)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                sample.sample_id.as_str(),
                embedding.space.space_id,
                blob,
                embedding.embedding.dim() as i64
            ],
        )?;
    }
    rebuild_prototypes_for_person(conn, &person.person_id)?;
    if let Some(legacy) = legacy_profile_id {
        conn.execute(
            "INSERT INTO legacy_profile_aliases(legacy_profile_id, person_id)
             VALUES (?1, ?2)",
            params![legacy, person.person_id.as_str()],
        )?;
    }
    bump_global_revision(conn)?;
    Ok(true)
}

fn upsert_space(conn: &Connection, space: &EmbeddingSpace) -> Result<(), VoiceIdStoreError> {
    let json =
        serde_json::to_string(space).map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
    conn.execute(
        "INSERT INTO embedding_spaces(space_id, canonical_json, dimension, pack_fingerprint, legacy_unverifiable)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(space_id) DO UPDATE SET canonical_json = excluded.canonical_json",
        params![
            space.space_id,
            json,
            space.dimension as i64,
            space.pack_fingerprint,
            space.legacy_unverifiable as i64
        ],
    )?;
    Ok(())
}

fn person_status_revision(
    conn: &Connection,
    person_id: &PersonId,
) -> Result<(PersonStatus, u64), VoiceIdStoreError> {
    let row = conn
        .query_row(
            "SELECT status, revision FROM persons WHERE person_id = ?1",
            params![person_id.as_str()],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64)),
        )
        .optional()?
        .ok_or_else(|| VoiceIdStoreError::NotFound(person_id.as_str().to_string()))?;
    let status = PersonStatus::parse(&row.0).unwrap_or(PersonStatus::Deleted);
    Ok((status, row.1))
}

fn touch_person(
    conn: &Connection,
    person_id: &PersonId,
    now: &str,
) -> Result<(), VoiceIdStoreError> {
    conn.execute(
        "UPDATE persons SET updated_at = ?1, revision = revision + 1 WHERE person_id = ?2",
        params![now, person_id.as_str()],
    )?;
    Ok(())
}

fn bump_global_revision(conn: &Connection) -> Result<(), VoiceIdStoreError> {
    conn.execute(
        "INSERT INTO voice_id_meta(key, value) VALUES ('global_revision', '1')
         ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)",
        [],
    )?;
    Ok(())
}

fn embedding_to_blob(embedding: &SpeakerEmbedding) -> Result<Vec<u8>, VoiceIdStoreError> {
    let mut bytes = Vec::with_capacity(embedding.dim() * 4);
    for value in &embedding.0 {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn blob_to_embedding(blob: &[u8], dim: usize) -> Result<SpeakerEmbedding, VoiceIdStoreError> {
    if blob.len() != dim * 4 {
        return Err(VoiceIdStoreError::Serialize(format!(
            "embedding blob length {} does not match dim {dim}",
            blob.len()
        )));
    }
    let mut values = Vec::with_capacity(dim);
    for chunk in blob.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(SpeakerEmbedding(values))
}

fn normalize_name(name: String) -> Result<String, VoiceIdStoreError> {
    let trimmed = name.trim().to_string();
    if trimmed.is_empty() {
        Err(VoiceIdStoreError::EmptyName)
    } else {
        Ok(trimmed)
    }
}

pub fn timestamp_now() -> String {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => format_unix_millis(duration.as_secs(), duration.subsec_millis()),
        Err(_) => "1970-01-01T00:00:00.000Z".to_string(),
    }
}

fn format_unix_millis(seconds: u64, millis: u32) -> String {
    let days = (seconds / 86_400) as i64;
    let seconds_of_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    (year, month as u32, day as u32)
}

fn set_owner_only_dir_permissions(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700));
    }
    #[cfg(windows)]
    {
        let _ = apply_windows_owner_only_dacl(path, true);
    }
    #[cfg(all(not(unix), not(windows)))]
    let _ = path;
}

fn set_owner_only_file_permissions(path: &Path) {
    if !path.exists() {
        return;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    #[cfg(windows)]
    {
        let _ = apply_windows_owner_only_dacl(path, false);
    }
    #[cfg(all(not(unix), not(windows)))]
    let _ = path;
}

// TODO(windows): apply an owner-only DACL to voice-id.db / -wal / -shm. The
// unix path uses 0600/0700; matching that on Windows needs CreateWellKnownSid +
// SetEntriesInAclW and is tracked with the same gap as atomic_file.rs.
#[cfg(windows)]
fn apply_windows_owner_only_dacl(_path: &Path, _is_dir: bool) -> std::io::Result<()> {
    Ok(())
}

fn immediate_transaction<T>(
    conn: &Connection,
    body: impl FnOnce() -> Result<T, VoiceIdStoreError>,
) -> Result<T, VoiceIdStoreError> {
    conn.execute_batch("BEGIN IMMEDIATE")?;
    match body() {
        Ok(value) => {
            conn.execute_batch("COMMIT")?;
            Ok(value)
        }
        Err(err) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarize::calibration::REDIMNET_CALIBRATION_VERSION;
    use crate::diarize::voice_id::domain::CandidateScope;
    use crate::diarize::voice_id::space::{MATCHER_POLICY_VERSION, REDIMNET_FRONTEND_VERSION};
    use tempfile::tempdir;

    fn test_space(fp: &str) -> EmbeddingSpace {
        EmbeddingSpace::from_parts(
            2,
            fp,
            "test",
            "test",
            "v1",
            REDIMNET_FRONTEND_VERSION,
            REDIMNET_CALIBRATION_VERSION,
            MATCHER_POLICY_VERSION,
        )
    }

    fn sample_input(space: &EmbeddingSpace, values: Vec<f32>) -> NewSampleInput {
        NewSampleInput {
            sample_id: SampleId::generate(),
            capture_context: CaptureContext {
                device_class: "test".into(),
                input_route: "mic".into(),
                environment_hint: None,
                sample_label: Some("clip".into()),
            },
            quality: SampleQuality {
                speech_seconds: 10.0,
                snr_estimate: 20.0,
                clipping_ratio: 0.0,
                vad_coverage: 0.8,
                accepted_reason: "test".into(),
            },
            space: space.clone(),
            embedding: SpeakerEmbedding::l2_normalized(values),
        }
    }

    fn consent() -> ConsentRecord {
        ConsentRecord {
            granted_at: timestamp_now(),
            notice_version: "voice-id-notice-v1".into(),
            capture_method: "test".into(),
        }
    }

    #[test]
    fn enroll_match_rename_and_revision_conflict() {
        let dir = tempdir().unwrap();
        let store = VoiceIdStore::open(dir.path());
        let space = test_space("sha256:a");
        let person = store
            .enroll_person(
                "Alice",
                consent(),
                vec![sample_input(&space, vec![1.0, 0.0])],
                None,
            )
            .unwrap();
        assert_eq!(person.display_name, "Alice");
        assert_eq!(person.revision, 1);
        assert!(!person.needs_reenrollment);

        let matcher = store.matcher_for_space(&space, 0.5, 0.0).unwrap();
        let hit = matcher
            .best_match(&SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]))
            .unwrap();
        assert_eq!(hit.display_name, "Alice");
        assert_eq!(hit.person_id.as_str(), person.person_id);

        let renamed = store
            .rename_person(
                &PersonId::parse(&person.person_id).unwrap(),
                "Alicia",
                Some(person.revision),
            )
            .unwrap();
        assert_eq!(renamed.display_name, "Alicia");
        assert_eq!(renamed.revision, 2);
        // Identity unchanged.
        assert_eq!(renamed.person_id, person.person_id);

        let conflict = store.rename_person(
            &PersonId::parse(&person.person_id).unwrap(),
            "Nope",
            Some(1),
        );
        assert!(matches!(
            conflict,
            Err(VoiceIdStoreError::RevisionConflict { .. })
        ));
    }

    #[test]
    fn two_same_name_persons_remain_distinct_for_margin() {
        let dir = tempdir().unwrap();
        let store = VoiceIdStore::open(dir.path());
        let space = test_space("sha256:b");
        let a = store
            .enroll_person(
                "Zhang Wei",
                consent(),
                vec![sample_input(&space, vec![1.0, 0.0])],
                None,
            )
            .unwrap();
        let b = store
            .enroll_person(
                "Zhang Wei",
                consent(),
                vec![sample_input(&space, vec![0.96, 0.05])],
                None,
            )
            .unwrap();
        assert_ne!(a.person_id, b.person_id);
        let matcher = store.matcher_for_space(&space, 0.5, 0.15).unwrap();
        // Ambiguous between the two same-name persons -> Unknown.
        assert!(
            matcher
                .best_match(&SpeakerEmbedding::l2_normalized(vec![0.98, 0.02]))
                .is_none()
        );
    }

    #[test]
    fn consent_revoke_purges_and_blocks_match() {
        let dir = tempdir().unwrap();
        let store = VoiceIdStore::open(dir.path());
        let space = test_space("sha256:c");
        let person = store
            .enroll_person(
                "Bob",
                consent(),
                vec![sample_input(&space, vec![0.0, 1.0])],
                None,
            )
            .unwrap();
        store
            .revoke_consent(
                &PersonId::parse(&person.person_id).unwrap(),
                Some(person.revision),
                "user_request",
            )
            .unwrap();
        let matcher = store.matcher_for_space(&space, 0.5, 0.0).unwrap();
        assert!(matcher.is_empty());
        let revoked = store
            .get_person(&PersonId::parse(&person.person_id).unwrap(), None)
            .unwrap();
        assert_eq!(revoked.status, PersonStatus::ConsentRevoked);
        assert_eq!(revoked.sample_count, 0);
    }

    #[test]
    fn empty_scope_disables_matcher() {
        let dir = tempdir().unwrap();
        let store = VoiceIdStore::open(dir.path());
        let space = test_space("sha256:d");
        store
            .enroll_person(
                "Eve",
                consent(),
                vec![sample_input(&space, vec![1.0, 0.0])],
                None,
            )
            .unwrap();
        let matcher = store
            .matcher_for_space(&space, 0.5, 0.0)
            .unwrap()
            .with_scope(&CandidateScope::Explicit(vec![]));
        assert!(matcher.is_empty());
    }

    #[test]
    fn needs_reenrollment_is_relative_to_active_space() {
        let dir = tempdir().unwrap();
        let store = VoiceIdStore::open(dir.path());
        let space_a = test_space("sha256:space-a");
        let space_b = test_space("sha256:space-b");
        let person = store
            .enroll_person(
                "SpaceBound",
                consent(),
                vec![sample_input(&space_a, vec![1.0, 0.0])],
                None,
            )
            .unwrap();

        let with_a = store
            .get_person(&PersonId::parse(&person.person_id).unwrap(), Some(&space_a))
            .unwrap();
        assert!(
            !with_a.needs_reenrollment,
            "matching active space must not require reenrollment"
        );
        assert!(with_a.samples.iter().all(|s| s.space_compatible));

        let with_b = store
            .get_person(&PersonId::parse(&person.person_id).unwrap(), Some(&space_b))
            .unwrap();
        assert!(
            with_b.needs_reenrollment,
            "incompatible active space must require reenrollment"
        );
        assert!(with_b.samples.iter().all(|s| !s.space_compatible));

        let listed = store.list_persons(Some(&space_b)).unwrap();
        assert_eq!(listed.len(), 1);
        assert!(listed[0].needs_reenrollment);
    }

    #[test]
    fn no_raw_audio_columns_exist() {
        let dir = tempdir().unwrap();
        let store = VoiceIdStore::open(dir.path());
        let conn = store.connection().unwrap();
        let mut stmt = conn
            .prepare("SELECT name, sql FROM sqlite_master WHERE type = 'table'")
            .unwrap();
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .unwrap();
        for row in rows {
            let (name, sql) = row.unwrap();
            let lower = sql.to_ascii_lowercase();
            assert!(
                !lower.contains("audio") && !lower.contains("wav") && !lower.contains("pcm"),
                "table {name} must not store audio: {sql}"
            );
        }
    }
}
