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
    CaptureContext, ConsentRecord, PersonPrototype, PersonStatus, PersonView, PrototypeMember,
    SampleQuality, SampleView, VOICE_ID_LABEL_MAX_CHARS, VoiceIdColor,
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
const IDEMPOTENCY_TTL_SECS: i64 = 24 * 60 * 60;
const IDEMPOTENCY_MAX_RECORDS: i64 = 1024;

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
    #[error("voice-id sample label must not be empty")]
    EmptySampleLabel,
    #[error("voice-id {field} must not exceed {max} characters (got {got})")]
    LabelTooLong {
        field: &'static str,
        max: usize,
        got: usize,
    },
    #[error("voice-id color preference is invalid: {0}")]
    InvalidColorPreference(String),
    #[error("voice-id person PATCH requires display_name or color_preference")]
    EmptyPersonMetadataUpdate,
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
    #[error("voice-id idempotency key was reused with a different request")]
    IdempotencyConflict,
    #[error("voice-id idempotency record is invalid: {0}")]
    IdempotencyRecord(String),
    #[error("voice-id database schema {found} is unreleased schema; reset required")]
    UnreleasedSchema { found: i32 },
    #[error("voice-id enrollment failed: {0}")]
    InvalidEnrollment(String),
}

#[derive(Debug, Clone)]
pub struct VoiceIdStore {
    root: PathBuf,
    db_path: PathBuf,
}

/// The editable person fields. `Some(None)` clears the color preference;
/// `None` leaves that field unchanged.
#[derive(Debug, Clone, Default)]
pub struct PersonMetadataUpdate {
    pub display_name: Option<String>,
    pub color_preference: Option<Option<String>>,
}

/// A privacy-preserving representation of an HTTP idempotency request. Both
/// values are SHA-256 digests; neither the client key nor audio bytes are kept.
#[derive(Debug, Clone)]
pub struct IdempotencyRequest {
    pub key_hash: String,
    pub request_hash: String,
}

#[derive(Debug, Clone)]
pub struct IdempotentPersonResult {
    pub person: PersonView,
    pub etag: String,
    pub replayed: bool,
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
        Self::open_checked(home)
    }

    /// Opens a fresh v1 schema or verifies that an existing database is v1.
    /// Development schemas were never released, so they are rejected rather
    /// than migrated or deleted.
    pub fn open_checked(openasr_home: impl AsRef<Path>) -> Result<Self, VoiceIdStoreError> {
        let store = Self::open(openasr_home);
        let _ = store.connection()?;
        Ok(store)
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
            let color = parse_color_preference(color)?;
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
        let color_preference = parse_color_preference(row.6)?;
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
            color_preference,
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
        let color_preference = normalize_color_preference(color_preference)?;
        let samples = samples
            .into_iter()
            .map(normalize_new_sample_input)
            .collect::<Result<Vec<_>, _>>()?;
        if samples.is_empty() {
            return Err(VoiceIdStoreError::InvalidEnrollment(
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
                    color_preference.map(VoiceIdColor::as_str)
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

    /// Enroll once for an idempotency key. The person graph and replay record
    /// commit together, so a response lost after commit is safe to retry.
    pub fn enroll_person_idempotent(
        &self,
        display_name: impl Into<String>,
        consent: ConsentRecord,
        samples: Vec<NewSampleInput>,
        color_preference: Option<String>,
        idempotency: IdempotencyRequest,
    ) -> Result<IdempotentPersonResult, VoiceIdStoreError> {
        let display_name = normalize_name(display_name.into())?;
        let color_preference = normalize_color_preference(color_preference)?;
        let samples = samples
            .into_iter()
            .map(normalize_new_sample_input)
            .collect::<Result<Vec<_>, _>>()?;
        if samples.is_empty() {
            return Err(VoiceIdStoreError::InvalidEnrollment(
                "enrollment requires at least one accepted sample".into(),
            ));
        }
        let space = samples[0].space.clone();
        let conn = self.connection()?;
        immediate_transaction(&conn, || {
            if let Some(replay) = lookup_idempotency(&conn, "enroll_person", &idempotency)? {
                return Ok(replay);
            }
            let person_id = PersonId::generate();
            let now = timestamp_now();
            conn.execute(
                "INSERT INTO persons(person_id, display_name, status, created_at, updated_at, revision, color_preference)
                 VALUES (?1, ?2, 'active', ?3, ?3, 1, ?4)",
                params![
                    person_id.as_str(),
                    display_name,
                    now,
                    color_preference.map(VoiceIdColor::as_str)
                ],
            )?;
            for sample in &samples {
                insert_sample(&conn, &person_id, sample, &consent, &now)?;
            }
            rebuild_prototypes_for_person(&conn, &person_id)?;
            bump_global_revision(&conn)?;
            let person = get_person_on_conn(&conn, &person_id, Some(&space))?;
            persist_idempotency(&conn, "enroll_person", &idempotency, &person)?;
            Ok(IdempotentPersonResult {
                etag: format!("\"{}\"", person.revision),
                person,
                replayed: false,
            })
        })
    }

    pub fn add_sample(
        &self,
        person_id: &PersonId,
        expected_revision: Option<u64>,
        consent: ConsentRecord,
        sample: NewSampleInput,
    ) -> Result<PersonView, VoiceIdStoreError> {
        let sample = normalize_new_sample_input(sample)?;
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

    /// Add one sample once for an idempotency key. The expected person revision
    /// remains part of the request fingerprint, preventing stale replays from
    /// being silently applied to a later version of a person.
    pub fn add_sample_idempotent(
        &self,
        person_id: &PersonId,
        expected_revision: Option<u64>,
        consent: ConsentRecord,
        sample: NewSampleInput,
        idempotency: IdempotencyRequest,
    ) -> Result<IdempotentPersonResult, VoiceIdStoreError> {
        let sample = normalize_new_sample_input(sample)?;
        let space = sample.space.clone();
        let conn = self.connection()?;
        immediate_transaction(&conn, || {
            if let Some(replay) = lookup_idempotency(&conn, "add_sample", &idempotency)? {
                return Ok(replay);
            }
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
            let person = get_person_on_conn(&conn, person_id, Some(&space))?;
            persist_idempotency(&conn, "add_sample", &idempotency, &person)?;
            Ok(IdempotentPersonResult {
                etag: format!("\"{}\"", person.revision),
                person,
                replayed: false,
            })
        })
    }

    pub fn rename_person(
        &self,
        person_id: &PersonId,
        display_name: impl Into<String>,
        expected_revision: Option<u64>,
    ) -> Result<PersonView, VoiceIdStoreError> {
        self.update_person_metadata(
            person_id,
            expected_revision,
            PersonMetadataUpdate {
                display_name: Some(display_name.into()),
                color_preference: None,
            },
        )
    }

    /// Atomically update editable person metadata under the owning person's
    /// revision. Every successful call advances both revisions exactly once.
    pub fn update_person_metadata(
        &self,
        person_id: &PersonId,
        expected_revision: Option<u64>,
        update: PersonMetadataUpdate,
    ) -> Result<PersonView, VoiceIdStoreError> {
        if update.display_name.is_none() && update.color_preference.is_none() {
            return Err(VoiceIdStoreError::EmptyPersonMetadataUpdate);
        }
        let display_name = update.display_name.map(normalize_name).transpose()?;
        let color_preference = update
            .color_preference
            .map(normalize_color_preference)
            .transpose()?;
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
            match (display_name.as_deref(), color_preference.as_ref()) {
                (Some(display_name), Some(color_preference)) => {
                    conn.execute(
                        "UPDATE persons SET display_name = ?1, color_preference = ?2, updated_at = ?3, revision = revision + 1 WHERE person_id = ?4",
                        params![display_name, color_preference.map(VoiceIdColor::as_str), now, person_id.as_str()],
                    )?;
                }
                (Some(display_name), None) => {
                    conn.execute(
                        "UPDATE persons SET display_name = ?1, updated_at = ?2, revision = revision + 1 WHERE person_id = ?3",
                        params![display_name, now, person_id.as_str()],
                    )?;
                }
                (None, Some(color_preference)) => {
                    conn.execute(
                        "UPDATE persons SET color_preference = ?1, updated_at = ?2, revision = revision + 1 WHERE person_id = ?3",
                        params![color_preference.map(VoiceIdColor::as_str), now, person_id.as_str()],
                    )?;
                }
                (None, None) => unreachable!("empty updates are rejected before the transaction"),
            }
            bump_global_revision(&conn)?;
            Ok(())
        })?;
        self.get_person(person_id, None)
    }

    /// Update only the presentation label stored in the sample's capture
    /// context. Embeddings, quality records, and prototypes are immutable.
    pub fn rename_sample(
        &self,
        sample_id: &SampleId,
        sample_label: impl Into<String>,
        expected_person_revision: Option<u64>,
    ) -> Result<PersonView, VoiceIdStoreError> {
        let sample_label = normalize_sample_label(sample_label.into())?;
        let conn = self.connection()?;
        let person_id = immediate_transaction(&conn, || {
            let (person_id_raw, context_json) = conn
                .query_row(
                    "SELECT person_id, context_json FROM enrollment_samples WHERE sample_id = ?1",
                    params![sample_id.as_str()],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                )
                .optional()?
                .ok_or_else(|| VoiceIdStoreError::SampleNotFound(sample_id.as_str().to_string()))?;
            let person_id = PersonId::parse(person_id_raw)?;
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
            let mut context: serde_json::Value = serde_json::from_str(&context_json)
                .map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
            let context = context.as_object_mut().ok_or_else(|| {
                VoiceIdStoreError::Serialize("sample capture context must be a JSON object".into())
            })?;
            context.insert(
                "sample_label".into(),
                serde_json::Value::String(sample_label.clone()),
            );
            let context_json = serde_json::to_string(&context)
                .map_err(|e| VoiceIdStoreError::Serialize(e.to_string()))?;
            conn.execute(
                "UPDATE enrollment_samples SET context_json = ?1 WHERE sample_id = ?2",
                params![context_json, sample_id.as_str()],
            )?;
            touch_person(&conn, &person_id, &timestamp_now())?;
            bump_global_revision(&conn)?;
            Ok(person_id)
        })?;
        self.get_person(&person_id, None)
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
        .map_err(|error| VoiceIdStoreError::Serialize(error.to_string()))
    }

    /// Returns internal metadata used for the global mutation revision.
    pub fn metadata_value(&self, key: &str) -> Result<Option<String>, VoiceIdStoreError> {
        let conn = self.connection()?;
        Ok(conn
            .query_row(
                "SELECT value FROM voice_id_meta WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?)
    }

    pub fn resolve_person_ref(&self, raw: &str) -> Result<PersonId, VoiceIdStoreError> {
        let person_id = PersonId::parse(raw)?;
        let _ = self.get_person(&person_id, None)?;
        Ok(person_id)
    }

    pub fn preferred_public_id(&self, person_id: &PersonId) -> Result<String, VoiceIdStoreError> {
        let _ = self.get_person(person_id, None)?;
        Ok(person_id.as_str().to_string())
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
        ensure_schema(&conn)?;
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

fn get_person_on_conn(
    conn: &Connection,
    person_id: &PersonId,
    active_space: Option<&EmbeddingSpace>,
) -> Result<PersonView, VoiceIdStoreError> {
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
    let samples = load_sample_views(conn, &row.0, active_space)?;
    Ok(PersonView {
        person_id: row.0,
        display_name: row.1,
        status,
        created_at: row.3,
        updated_at: row.4,
        revision: row.5,
        sample_count: samples.len(),
        needs_reenrollment: samples.iter().all(|sample| sample.needs_reenrollment)
            || samples.is_empty(),
        color_preference: parse_color_preference(row.6)?,
        samples,
    })
}

fn lookup_idempotency(
    conn: &Connection,
    scope: &str,
    request: &IdempotencyRequest,
) -> Result<Option<IdempotentPersonResult>, VoiceIdStoreError> {
    let now = unix_timestamp_secs();
    conn.execute(
        "DELETE FROM voice_id_idempotency WHERE expires_at <= ?1",
        params![now],
    )?;
    let record = conn
        .query_row(
            "SELECT request_hash, response_json, etag FROM voice_id_idempotency
             WHERE scope = ?1 AND key_hash = ?2",
            params![scope, request.key_hash],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()?;
    let Some((request_hash, response_json, etag)) = record else {
        return Ok(None);
    };
    if request_hash != request.request_hash {
        return Err(VoiceIdStoreError::IdempotencyConflict);
    }
    let person = serde_json::from_str(&response_json)
        .map_err(|error| VoiceIdStoreError::IdempotencyRecord(error.to_string()))?;
    Ok(Some(IdempotentPersonResult {
        person,
        etag,
        replayed: true,
    }))
}

fn persist_idempotency(
    conn: &Connection,
    scope: &str,
    request: &IdempotencyRequest,
    person: &PersonView,
) -> Result<(), VoiceIdStoreError> {
    let now = unix_timestamp_secs();
    let response_json = serde_json::to_string(person)
        .map_err(|error| VoiceIdStoreError::Serialize(error.to_string()))?;
    conn.execute(
        "INSERT INTO voice_id_idempotency(
             scope, key_hash, request_hash, response_json, etag, created_at, expires_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            scope,
            request.key_hash,
            request.request_hash,
            response_json,
            format!("\"{}\"", person.revision),
            now,
            now + IDEMPOTENCY_TTL_SECS,
        ],
    )?;
    conn.execute(
        "DELETE FROM voice_id_idempotency WHERE rowid IN (
             SELECT rowid FROM voice_id_idempotency
             ORDER BY created_at DESC, rowid DESC
             LIMIT -1 OFFSET ?1
         )",
        params![IDEMPOTENCY_MAX_RECORDS],
    )?;
    Ok(())
}

fn unix_timestamp_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn ensure_schema(conn: &Connection) -> Result<(), VoiceIdStoreError> {
    let user_version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if user_version == 0 {
        conn.execute_batch(
            "
            CREATE TABLE voice_id_meta (
                key TEXT PRIMARY KEY NOT NULL,
                value TEXT NOT NULL
            );
            CREATE TABLE persons (
                person_id TEXT PRIMARY KEY NOT NULL,
                display_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                revision INTEGER NOT NULL,
                color_preference TEXT
            );
            CREATE TABLE enrollment_samples (
                sample_id TEXT PRIMARY KEY NOT NULL,
                person_id TEXT NOT NULL REFERENCES persons(person_id),
                created_at TEXT NOT NULL,
                consent_json TEXT NOT NULL,
                quality_json TEXT NOT NULL,
                context_json TEXT NOT NULL,
                sample_ordinal INTEGER NOT NULL
            );
            CREATE TABLE embedding_spaces (
                space_id TEXT PRIMARY KEY NOT NULL,
                canonical_json TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                pack_fingerprint TEXT NOT NULL,
                legacy_unverifiable INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE sample_embeddings (
                sample_id TEXT NOT NULL REFERENCES enrollment_samples(sample_id) ON DELETE CASCADE,
                space_id TEXT NOT NULL REFERENCES embedding_spaces(space_id),
                embedding_blob BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                PRIMARY KEY (sample_id, space_id)
            );
            CREATE TABLE prototypes (
                prototype_id TEXT PRIMARY KEY NOT NULL,
                person_id TEXT NOT NULL REFERENCES persons(person_id),
                space_id TEXT NOT NULL REFERENCES embedding_spaces(space_id),
                medoid_sample_id TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                medoid_blob BLOB NOT NULL,
                medoid_dim INTEGER NOT NULL
            );
            CREATE TABLE prototype_members (
                prototype_id TEXT NOT NULL REFERENCES prototypes(prototype_id) ON DELETE CASCADE,
                sample_id TEXT NOT NULL,
                quality_weight REAL NOT NULL,
                PRIMARY KEY (prototype_id, sample_id)
            );
            CREATE TABLE person_tombstones (
                person_id TEXT PRIMARY KEY NOT NULL,
                revoked_or_deleted_at TEXT NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE TABLE voice_id_idempotency (
                scope TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                response_json TEXT NOT NULL,
                etag TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                PRIMARY KEY (scope, key_hash)
            );
            CREATE INDEX voice_id_idempotency_expires_idx
                ON voice_id_idempotency(expires_at);
            CREATE INDEX enrollment_samples_person_idx
                ON enrollment_samples(person_id);
            CREATE UNIQUE INDEX enrollment_samples_person_ordinal_idx
                ON enrollment_samples(person_id, sample_ordinal);
            CREATE INDEX prototypes_person_space_idx
                ON prototypes(person_id, space_id);
            ",
        )?;
        conn.pragma_update(None, "user_version", VOICE_ID_SCHEMA_VERSION)?;
        conn.execute(
            "INSERT INTO voice_id_meta(key, value) VALUES ('schema_version', ?1)",
            params![VOICE_ID_SCHEMA_VERSION.to_string()],
        )?;
        return Ok(());
    }
    if user_version != VOICE_ID_SCHEMA_VERSION || !schema_is_current(conn)? {
        return Err(VoiceIdStoreError::UnreleasedSchema {
            found: user_version,
        });
    }
    Ok(())
}

fn schema_is_current(conn: &Connection) -> Result<bool, VoiceIdStoreError> {
    const TABLES: &[&str] = &[
        "voice_id_meta",
        "persons",
        "enrollment_samples",
        "embedding_spaces",
        "sample_embeddings",
        "prototypes",
        "prototype_members",
        "person_tombstones",
        "voice_id_idempotency",
    ];
    for table in TABLES {
        let exists = conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
                params![table],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if !exists {
            return Ok(false);
        }
    }
    let sample_ordinal_exists = conn
        .prepare("PRAGMA table_info(enrollment_samples)")?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .any(|column| column == "sample_ordinal");
    if !sample_ordinal_exists {
        return Ok(false);
    }
    let schema_version = conn
        .query_row(
            "SELECT value FROM voice_id_meta WHERE key = 'schema_version'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    Ok(schema_version.as_deref() == Some("1"))
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
    let sample_ordinal = next_sample_ordinal(conn, person_id)?;
    conn.execute(
        "INSERT INTO enrollment_samples(
            sample_id, person_id, created_at, consent_json, quality_json, context_json, sample_ordinal
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            sample.sample_id.as_str(),
            person_id.as_str(),
            now,
            consent_json,
            quality_json,
            context_json,
            sample_ordinal,
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

fn next_sample_ordinal(conn: &Connection, person_id: &PersonId) -> Result<i64, VoiceIdStoreError> {
    Ok(conn.query_row(
        "SELECT COALESCE(MAX(sample_ordinal) + 1, 0) FROM enrollment_samples WHERE person_id = ?1",
        params![person_id.as_str()],
        |row| row.get(0),
    )?)
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
         WHERE person_id = ?1 ORDER BY sample_ordinal ASC, sample_id ASC",
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
    normalize_label(name, "display name", VoiceIdStoreError::EmptyName)
}

fn normalize_sample_label(label: String) -> Result<String, VoiceIdStoreError> {
    normalize_label(label, "sample label", VoiceIdStoreError::EmptySampleLabel)
}

fn normalize_label(
    value: String,
    field: &'static str,
    empty_error: VoiceIdStoreError,
) -> Result<String, VoiceIdStoreError> {
    let trimmed = value.trim().to_string();
    if trimmed.is_empty() {
        return Err(empty_error);
    }
    let got = trimmed.chars().count();
    if got > VOICE_ID_LABEL_MAX_CHARS {
        return Err(VoiceIdStoreError::LabelTooLong {
            field,
            max: VOICE_ID_LABEL_MAX_CHARS,
            got,
        });
    }
    Ok(trimmed)
}

fn normalize_color_preference(
    color_preference: Option<String>,
) -> Result<Option<VoiceIdColor>, VoiceIdStoreError> {
    color_preference
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            VoiceIdColor::parse(&normalized).ok_or(VoiceIdStoreError::InvalidColorPreference(value))
        })
        .transpose()
}

fn parse_color_preference(
    color_preference: Option<String>,
) -> Result<Option<VoiceIdColor>, VoiceIdStoreError> {
    color_preference
        .map(|value| {
            VoiceIdColor::parse(&value).ok_or(VoiceIdStoreError::InvalidColorPreference(value))
        })
        .transpose()
}

fn normalize_new_sample_input(
    mut sample: NewSampleInput,
) -> Result<NewSampleInput, VoiceIdStoreError> {
    if let Some(sample_label) = sample.capture_context.sample_label.take() {
        sample.capture_context.sample_label = Some(normalize_sample_label(sample_label)?);
    }
    Ok(sample)
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

    fn idempotency(key: &str, request: &str) -> IdempotencyRequest {
        IdempotencyRequest {
            key_hash: key.into(),
            request_hash: request.into(),
        }
    }

    #[test]
    fn fresh_database_is_complete_v1_schema() {
        let dir = tempdir().unwrap();
        let store = VoiceIdStore::open_checked(dir.path()).unwrap();
        let conn = Connection::open(store.db_path()).unwrap();
        let user_version: i32 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(user_version, VOICE_ID_SCHEMA_VERSION);
        assert_eq!(
            conn.query_row(
                "SELECT value FROM voice_id_meta WHERE key = 'schema_version'",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
            "1"
        );
        for table in [
            "persons",
            "enrollment_samples",
            "embedding_spaces",
            "sample_embeddings",
            "prototypes",
            "prototype_members",
            "person_tombstones",
            "voice_id_idempotency",
        ] {
            assert!(
                conn.query_row(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
                    [table],
                    |_| Ok(()),
                )
                .optional()
                .unwrap()
                .is_some()
            );
        }
        let sample_columns = conn
            .prepare("PRAGMA table_info(enrollment_samples)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            sample_columns
                .iter()
                .any(|column| column == "sample_ordinal")
        );
        assert!(conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'voice_id_idempotency_expires_idx'",
                [],
                |_| Ok(()),
            )
            .optional()
            .unwrap()
            .is_some());
    }

    #[test]
    fn unreleased_schema_is_rejected_without_resetting_database() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("diarize/voice-id.db");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE development_data(value TEXT); PRAGMA user_version = 4;")
            .unwrap();
        drop(conn);

        let error = VoiceIdStore::open_checked(dir.path()).unwrap_err();
        assert!(matches!(
            error,
            VoiceIdStoreError::UnreleasedSchema { found: 4 }
        ));
        let conn = Connection::open(path).unwrap();
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM development_data", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn idempotent_enrollment_replays_across_reopen_without_extra_revisions() {
        let dir = tempdir().unwrap();
        let space = test_space("sha256:idempotency");
        let request = idempotency("key-hash", "request-hash");
        let first = VoiceIdStore::open(dir.path())
            .enroll_person_idempotent(
                "Alice",
                consent(),
                vec![sample_input(&space, vec![1.0, 0.0])],
                None,
                request.clone(),
            )
            .unwrap();
        let replay = VoiceIdStore::open(dir.path())
            .enroll_person_idempotent(
                "Alice",
                consent(),
                vec![sample_input(&space, vec![1.0, 0.0])],
                None,
                request,
            )
            .unwrap();

        assert!(!first.replayed);
        assert!(replay.replayed);
        assert_eq!(first.person, replay.person);
        assert_eq!(first.etag, replay.etag);
        let store = VoiceIdStore::open(dir.path());
        assert_eq!(store.list_persons(Some(&space)).unwrap().len(), 1);
        assert_eq!(
            store.metadata_value("global_revision").unwrap().as_deref(),
            Some("1")
        );
    }

    #[test]
    fn idempotency_conflict_and_expiry_are_handled_in_the_mutation_transaction() {
        let dir = tempdir().unwrap();
        let store = VoiceIdStore::open(dir.path());
        let space = test_space("sha256:idempotency-expiry");
        store
            .enroll_person_idempotent(
                "Alice",
                consent(),
                vec![sample_input(&space, vec![1.0, 0.0])],
                None,
                idempotency("key-hash", "first-request"),
            )
            .unwrap();
        assert!(matches!(
            store.enroll_person_idempotent(
                "Bob",
                consent(),
                vec![sample_input(&space, vec![1.0, 0.0])],
                None,
                idempotency("key-hash", "different-request"),
            ),
            Err(VoiceIdStoreError::IdempotencyConflict)
        ));

        let conn = Connection::open(store.db_path()).unwrap();
        conn.execute("UPDATE voice_id_idempotency SET expires_at = 0", [])
            .unwrap();
        store
            .enroll_person_idempotent(
                "Bob",
                consent(),
                vec![sample_input(&space, vec![1.0, 0.0])],
                None,
                idempotency("key-hash", "different-request"),
            )
            .unwrap();
        assert_eq!(store.list_persons(Some(&space)).unwrap().len(), 2);
    }

    #[test]
    fn idempotent_add_sample_replays_person_view_without_a_second_write() {
        let dir = tempdir().unwrap();
        let store = VoiceIdStore::open(dir.path());
        let space = test_space("sha256:idempotency-add-sample");
        let enrolled = store
            .enroll_person(
                "Alice",
                consent(),
                vec![sample_input(&space, vec![1.0, 0.0])],
                None,
            )
            .unwrap();
        let person_id = PersonId::parse(enrolled.person_id).unwrap();
        let first = store
            .add_sample_idempotent(
                &person_id,
                Some(enrolled.revision),
                consent(),
                sample_input(&space, vec![0.9, 0.1]),
                idempotency("add-key", "add-request"),
            )
            .unwrap();
        let replay = VoiceIdStore::open(dir.path())
            .add_sample_idempotent(
                &person_id,
                Some(enrolled.revision),
                consent(),
                sample_input(&space, vec![0.9, 0.1]),
                idempotency("add-key", "add-request"),
            )
            .unwrap();
        assert_eq!(first.person.sample_count, 2);
        assert_eq!(first.person, replay.person);
        assert!(replay.replayed);
        assert_eq!(
            store.metadata_value("global_revision").unwrap().as_deref(),
            Some("2")
        );
    }

    #[test]
    fn concurrent_idempotent_enrollment_advances_global_revision_once() {
        let dir = tempdir().unwrap();
        let store = VoiceIdStore::open(dir.path());
        let space = test_space("sha256:idempotency-concurrent");
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let store = store.clone();
                let space = space.clone();
                scope.spawn(move || {
                    store
                        .enroll_person_idempotent(
                            "Alice",
                            consent(),
                            vec![sample_input(&space, vec![1.0, 0.0])],
                            None,
                            idempotency("concurrent-key", "concurrent-request"),
                        )
                        .unwrap();
                });
            }
        });
        assert_eq!(store.list_persons(Some(&space)).unwrap().len(), 1);
        assert_eq!(
            store.metadata_value("global_revision").unwrap().as_deref(),
            Some("1")
        );
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
    fn person_metadata_update_is_atomic_validated_and_persistent() {
        let dir = tempdir().unwrap();
        let store = VoiceIdStore::open(dir.path());
        let space = test_space("sha256:person-metadata");
        let person = store
            .enroll_person(
                "Alice",
                consent(),
                vec![sample_input(&space, vec![1.0, 0.0])],
                Some("blue".into()),
            )
            .unwrap();
        assert_eq!(person.revision, 1);
        assert_eq!(person.color_preference, Some(VoiceIdColor::Blue));

        let color_only = store
            .update_person_metadata(
                &PersonId::parse(&person.person_id).unwrap(),
                Some(person.revision),
                PersonMetadataUpdate {
                    display_name: None,
                    color_preference: Some(Some(" purple ".into())),
                },
            )
            .unwrap();
        assert_eq!(color_only.display_name, "Alice");
        assert_eq!(color_only.color_preference, Some(VoiceIdColor::Purple));
        assert_eq!(color_only.revision, 2);
        assert_eq!(
            store.metadata_value("global_revision").unwrap().as_deref(),
            Some("2")
        );

        let name_only = store
            .update_person_metadata(
                &PersonId::parse(&person.person_id).unwrap(),
                Some(color_only.revision),
                PersonMetadataUpdate {
                    display_name: Some(" Alicia ".into()),
                    color_preference: None,
                },
            )
            .unwrap();
        assert_eq!(name_only.display_name, "Alicia");
        assert_eq!(name_only.color_preference, Some(VoiceIdColor::Purple));
        assert_eq!(name_only.revision, 3);

        let both = store
            .update_person_metadata(
                &PersonId::parse(&person.person_id).unwrap(),
                Some(name_only.revision),
                PersonMetadataUpdate {
                    display_name: Some("Alice Example".into()),
                    color_preference: Some(Some("green".into())),
                },
            )
            .unwrap();
        assert_eq!(both.display_name, "Alice Example");
        assert_eq!(both.color_preference, Some(VoiceIdColor::Green));
        assert_eq!(both.revision, 4);
        assert_eq!(
            store.metadata_value("global_revision").unwrap().as_deref(),
            Some("4")
        );

        let reopened = VoiceIdStore::open(dir.path());
        let persisted = reopened
            .get_person(&PersonId::parse(&person.person_id).unwrap(), None)
            .unwrap();
        assert_eq!(persisted.display_name, "Alice Example");
        assert_eq!(persisted.color_preference, Some(VoiceIdColor::Green));
        assert_eq!(persisted.revision, 4);

        assert!(matches!(
            store.update_person_metadata(
                &PersonId::parse(&person.person_id).unwrap(),
                Some(4),
                PersonMetadataUpdate::default(),
            ),
            Err(VoiceIdStoreError::EmptyPersonMetadataUpdate)
        ));
        assert!(matches!(
            store.update_person_metadata(
                &PersonId::parse(&person.person_id).unwrap(),
                Some(4),
                PersonMetadataUpdate {
                    display_name: None,
                    color_preference: Some(Some("#123456".into())),
                },
            ),
            Err(VoiceIdStoreError::InvalidColorPreference(_))
        ));
        assert!(matches!(
            store.update_person_metadata(
                &PersonId::parse(&person.person_id).unwrap(),
                Some(4),
                PersonMetadataUpdate {
                    display_name: Some("   ".into()),
                    color_preference: None,
                },
            ),
            Err(VoiceIdStoreError::EmptyName)
        ));
        assert!(matches!(
            store.update_person_metadata(
                &PersonId::parse(&person.person_id).unwrap(),
                Some(3),
                PersonMetadataUpdate {
                    display_name: Some("Stale".into()),
                    color_preference: None,
                },
            ),
            Err(VoiceIdStoreError::RevisionConflict { .. })
        ));
    }

    #[test]
    fn sample_label_update_preserves_biometric_data_and_persists() {
        let dir = tempdir().unwrap();
        let store = VoiceIdStore::open(dir.path());
        let space = test_space("sha256:sample-label");
        let person = store
            .enroll_person(
                "Alice",
                consent(),
                vec![sample_input(&space, vec![1.0, 0.0])],
                None,
            )
            .unwrap();
        let sample = &person.samples[0];
        let sample_id = SampleId::parse(&sample.sample_id).unwrap();
        let before_quality = sample.quality.clone();
        let conn = store.connection().unwrap();
        let before_embedding: String = conn
            .query_row(
                "SELECT hex(embedding_blob) FROM sample_embeddings WHERE sample_id = ?1",
                params![sample_id.as_str()],
                |row| row.get(0),
            )
            .unwrap();
        let before_prototype: String = conn
            .query_row(
                "SELECT hex(medoid_blob) FROM prototypes WHERE person_id = ?1",
                params![person.person_id],
                |row| row.get(0),
            )
            .unwrap();

        let renamed = store
            .rename_sample(&sample_id, "  Meeting room  ", Some(person.revision))
            .unwrap();
        assert_eq!(renamed.revision, 2);
        assert_eq!(
            renamed.samples[0].sample_label.as_deref(),
            Some("Meeting room")
        );
        assert_eq!(
            renamed.samples[0].capture_context.sample_label.as_deref(),
            Some("Meeting room")
        );
        assert_eq!(renamed.samples[0].quality, before_quality);
        assert_eq!(
            store.metadata_value("global_revision").unwrap().as_deref(),
            Some("2")
        );

        let after_embedding: String = conn
            .query_row(
                "SELECT hex(embedding_blob) FROM sample_embeddings WHERE sample_id = ?1",
                params![sample_id.as_str()],
                |row| row.get(0),
            )
            .unwrap();
        let after_prototype: String = conn
            .query_row(
                "SELECT hex(medoid_blob) FROM prototypes WHERE person_id = ?1",
                params![person.person_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(after_embedding, before_embedding);
        assert_eq!(after_prototype, before_prototype);

        let reopened = VoiceIdStore::open(dir.path());
        let persisted = reopened
            .get_person(&PersonId::parse(&person.person_id).unwrap(), None)
            .unwrap();
        assert_eq!(
            persisted.samples[0].sample_label.as_deref(),
            Some("Meeting room")
        );
        assert!(matches!(
            store.rename_sample(&sample_id, "   ", Some(2)),
            Err(VoiceIdStoreError::EmptySampleLabel)
        ));
        assert!(matches!(
            store.rename_sample(&sample_id, "Stale", Some(1)),
            Err(VoiceIdStoreError::RevisionConflict { .. })
        ));
    }

    #[test]
    fn enrollment_preserves_client_sample_labels_and_validates_shared_limit() {
        let dir = tempdir().unwrap();
        let store = VoiceIdStore::open(dir.path());
        let space = test_space("sha256:initial-labels");
        let mut first = sample_input(&space, vec![1.0, 0.0]);
        first.capture_context.sample_label = Some("  First take  ".into());
        let mut second = sample_input(&space, vec![0.0, 1.0]);
        second.capture_context.sample_label = Some("Second take".into());
        let person = store
            .enroll_person("Alice", consent(), vec![first, second], None)
            .unwrap();
        assert_eq!(
            person.samples[0].sample_label.as_deref(),
            Some("First take")
        );
        assert_eq!(
            person.samples[1].sample_label.as_deref(),
            Some("Second take")
        );

        let mut too_long = sample_input(&space, vec![1.0, 0.0]);
        too_long.capture_context.sample_label = Some("x".repeat(VOICE_ID_LABEL_MAX_CHARS + 1));
        assert!(matches!(
            store.enroll_person("Alice", consent(), vec![too_long], None),
            Err(VoiceIdStoreError::LabelTooLong {
                field: "sample label",
                ..
            })
        ));
        assert!(matches!(
            store.enroll_person(
                "x".repeat(VOICE_ID_LABEL_MAX_CHARS + 1),
                consent(),
                vec![sample_input(&space, vec![1.0, 0.0])],
                None
            ),
            Err(VoiceIdStoreError::LabelTooLong {
                field: "display name",
                ..
            })
        ));
    }

    #[test]
    fn sample_label_update_preserves_unknown_capture_context_fields() {
        let dir = tempdir().unwrap();
        let store = VoiceIdStore::open(dir.path());
        let space = test_space("sha256:future-context");
        let person = store
            .enroll_person(
                "Alice",
                consent(),
                vec![sample_input(&space, vec![1.0, 0.0])],
                None,
            )
            .unwrap();
        let sample_id = SampleId::parse(&person.samples[0].sample_id).unwrap();
        let conn = store.connection().unwrap();
        conn.execute(
            "UPDATE enrollment_samples SET context_json = ?1 WHERE sample_id = ?2",
            params![
                r#"{"device_class":"test","input_route":"mic","sample_label":"old","future_context":{"transport":"new-client"}}"#,
                sample_id.as_str(),
            ],
        )
        .unwrap();
        drop(conn);

        store
            .rename_sample(&sample_id, "Updated", Some(person.revision))
            .unwrap();
        let conn = store.connection().unwrap();
        let context: serde_json::Value = conn
            .query_row(
                "SELECT context_json FROM enrollment_samples WHERE sample_id = ?1",
                params![sample_id.as_str()],
                |row| row.get(0),
            )
            .map(|raw: String| serde_json::from_str(&raw).unwrap())
            .unwrap();
        assert_eq!(context["sample_label"], "Updated");
        assert_eq!(context["future_context"]["transport"], "new-client");
    }

    #[test]
    fn enrollment_sample_order_follows_request_order_across_repeated_runs() {
        let space = test_space("sha256:sample-order");
        for _ in 0..20 {
            let dir = tempdir().unwrap();
            let store = VoiceIdStore::open(dir.path());
            let mut samples = vec![
                sample_input(&space, vec![1.0, 0.0]),
                sample_input(&space, vec![0.0, 1.0]),
                sample_input(&space, vec![0.7, 0.7]),
            ];
            for (index, sample) in samples.iter_mut().enumerate() {
                sample.capture_context.sample_label = Some(format!("request-{}", index + 1));
            }
            let person = store
                .enroll_person("Alice", consent(), samples, None)
                .unwrap();
            let labels = person
                .samples
                .iter()
                .map(|sample| sample.sample_label.as_deref())
                .collect::<Vec<_>>();
            assert_eq!(
                labels,
                vec![Some("request-1"), Some("request-2"), Some("request-3")]
            );
        }
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
