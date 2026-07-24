//! Conservative v1 JSON (`voiceprints.json`) -> v2 SQLite migration.
//!
//! Rules:
//! - every v1 profile becomes its own Person (never merge by name)
//! - legacy profile ids are retained only as aliases
//! - migrated embeddings land in a non-matchable `legacy-unverifiable-v1` space
//! - fail closed on corrupt / unknown schema; never delete the JSON until the
//!   DB transaction has committed and the ledger records success

use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;

use super::domain::{
    CaptureContext, ConsentRecord, EnrollmentSample, Person, PersonStatus, SampleEmbedding,
    SampleQuality,
};
use super::ids::{PersonId, SampleId};
use super::space::EmbeddingSpace;
use super::store::{VoiceIdStore, VoiceIdStoreError, timestamp_now};
use crate::diarize::contract::SpeakerEmbedding;
use crate::diarize::enrollment::{VOICEPRINT_STORE_VERSION, VoiceprintStore};

pub const MIGRATION_LEDGER_KEY: &str = "v1_json_migration";
pub const MIGRATION_STATE_DONE: &str = "done";
pub const MIGRATION_STATE_JSON_PENDING_DELETE: &str = "json_pending_delete";

#[derive(Debug, Error)]
pub enum VoiceIdMigrationError {
    #[error("{0}")]
    Store(#[from] VoiceIdStoreError),
    #[error("v1 voiceprint migration failed: {0}")]
    Failed(String),
    #[error("v1 voiceprint store is corrupt or unsupported; left untouched at {path}: {detail}")]
    FailClosed { path: PathBuf, detail: String },
}

/// Open the v2 store under `openasr_home` and migrate `voiceprints.json` when
/// present. Safe to call on every startup: ledger state makes it idempotent.
pub fn open_store_with_v1_migration(
    openasr_home: impl AsRef<Path>,
) -> Result<VoiceIdStore, VoiceIdMigrationError> {
    let home = openasr_home.as_ref();
    let store = VoiceIdStore::open(home);
    let json_path = voiceprints_json_path(home);
    migrate_v1_json_if_needed(&store, &json_path)?;
    Ok(store)
}

pub fn migrate_v1_json_if_needed(
    store: &VoiceIdStore,
    json_path: &Path,
) -> Result<(), VoiceIdMigrationError> {
    let state = store.migration_state(MIGRATION_LEDGER_KEY)?;
    match state.as_deref() {
        Some(MIGRATION_STATE_DONE) => {
            // Migration already applied. If a previous run crashed after DB
            // commit but before JSON delete, finish the cleanup once.
            if json_path.is_file() {
                let _ = fs::remove_file(json_path);
            }
            return Ok(());
        }
        Some(MIGRATION_STATE_JSON_PENDING_DELETE) => {
            if json_path.is_file() {
                fs::remove_file(json_path).map_err(|e| {
                    VoiceIdMigrationError::Failed(format!(
                        "could not remove migrated v1 json {}: {e}",
                        json_path.display()
                    ))
                })?;
            }
            store.set_migration_state(MIGRATION_LEDGER_KEY, MIGRATION_STATE_DONE)?;
            return Ok(());
        }
        Some(other) => {
            return Err(VoiceIdMigrationError::Failed(format!(
                "unknown migration ledger state '{other}'"
            )));
        }
        None => {}
    }

    if !json_path.is_file() {
        store.set_migration_state(MIGRATION_LEDGER_KEY, MIGRATION_STATE_DONE)?;
        return Ok(());
    }

    let v1 = load_v1_store(json_path)?;
    for profile in &v1.profiles {
        validate_v1_profile(profile, json_path)?;
    }

    // Import each profile as an independent Person. Same display names stay
    // separate; nothing is merged.
    for profile in &v1.profiles {
        let person_id = PersonId::generate();
        let sample_id = SampleId::generate();
        let now = timestamp_now();
        let person = Person {
            person_id: person_id.clone(),
            display_name: profile.name.clone(),
            status: PersonStatus::Active,
            created_at: profile.created_at.clone(),
            updated_at: profile.updated_at.clone(),
            revision: 1,
            color_preference: None,
        };
        let sample = EnrollmentSample {
            sample_id: sample_id.clone(),
            person_id: person_id.clone(),
            created_at: profile.created_at.clone(),
            consent: ConsentRecord {
                granted_at: profile.created_at.clone(),
                notice_version: "legacy-v1-import".into(),
                capture_method: "migrated".into(),
            },
            capture_context: CaptureContext {
                device_class: "unknown".into(),
                input_route: "unknown".into(),
                environment_hint: None,
                sample_label: Some("Migrated enrollment".into()),
            },
            quality: SampleQuality {
                speech_seconds: profile.sample_seconds,
                snr_estimate: 0.0,
                clipping_ratio: 0.0,
                vad_coverage: 1.0,
                accepted_reason: "migrated_v1".into(),
            },
        };
        let space = EmbeddingSpace::legacy_unverifiable_v1(
            profile.embedding_dim,
            profile.pack_fingerprint.clone(),
        );
        let embedding = SampleEmbedding {
            sample_id,
            space,
            embedding: SpeakerEmbedding(profile.embedding.clone()),
        };
        store
            .import_person_graph(&person, &[(sample, embedding)], Some(&profile.id))
            .map_err(|err| match err {
                VoiceIdStoreError::Database(db) => VoiceIdMigrationError::FailClosed {
                    path: json_path.to_path_buf(),
                    detail: db.to_string(),
                },
                other => VoiceIdMigrationError::Failed(other.to_string()),
            })?;
        let _ = now;
    }

    store.set_migration_state(MIGRATION_LEDGER_KEY, MIGRATION_STATE_JSON_PENDING_DELETE)?;
    if json_path.is_file() {
        fs::remove_file(json_path).map_err(|e| {
            VoiceIdMigrationError::Failed(format!(
                "v1 profiles imported but could not remove {}: {e}",
                json_path.display()
            ))
        })?;
    }
    store.set_migration_state(MIGRATION_LEDGER_KEY, MIGRATION_STATE_DONE)?;
    Ok(())
}

fn voiceprints_json_path(openasr_home: &Path) -> PathBuf {
    if let Ok(path) = std::env::var(crate::diarize::enrollment::VOICEPRINT_STORE_ENV) {
        return PathBuf::from(path);
    }
    openasr_home.join("diarize").join("voiceprints.json")
}

fn load_v1_store(path: &Path) -> Result<VoiceprintStore, VoiceIdMigrationError> {
    let bytes = fs::read(path).map_err(|e| VoiceIdMigrationError::FailClosed {
        path: path.to_path_buf(),
        detail: format!("read error: {e}"),
    })?;
    let store: VoiceprintStore =
        serde_json::from_slice(&bytes).map_err(|e| VoiceIdMigrationError::FailClosed {
            path: path.to_path_buf(),
            detail: format!("parse error: {e}"),
        })?;
    if store.version != VOICEPRINT_STORE_VERSION {
        return Err(VoiceIdMigrationError::FailClosed {
            path: path.to_path_buf(),
            detail: format!(
                "unsupported version {}; expected {VOICEPRINT_STORE_VERSION}",
                store.version
            ),
        });
    }
    Ok(store)
}

fn validate_v1_profile(
    profile: &crate::diarize::enrollment::SpeakerProfile,
    path: &Path,
) -> Result<(), VoiceIdMigrationError> {
    if profile.id.trim().is_empty() {
        return Err(VoiceIdMigrationError::FailClosed {
            path: path.to_path_buf(),
            detail: "profile id is empty".into(),
        });
    }
    if profile.name.trim().is_empty() {
        return Err(VoiceIdMigrationError::FailClosed {
            path: path.to_path_buf(),
            detail: format!("profile {} has empty name", profile.id),
        });
    }
    if profile.embedding_dim == 0 || profile.embedding.len() != profile.embedding_dim {
        return Err(VoiceIdMigrationError::FailClosed {
            path: path.to_path_buf(),
            detail: format!(
                "profile {} has illegal embedding dim {}/len {}",
                profile.id,
                profile.embedding_dim,
                profile.embedding.len()
            ),
        });
    }
    if profile.pack_fingerprint.trim().is_empty() {
        return Err(VoiceIdMigrationError::FailClosed {
            path: path.to_path_buf(),
            detail: format!("profile {} missing pack fingerprint", profile.id),
        });
    }
    if !profile.embedding.iter().all(|v| v.is_finite()) {
        return Err(VoiceIdMigrationError::FailClosed {
            path: path.to_path_buf(),
            detail: format!("profile {} has non-finite embedding values", profile.id),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarize::enrollment::SpeakerProfile;
    use tempfile::tempdir;

    fn v1_profile(id: &str, name: &str, embedding: Vec<f32>) -> SpeakerProfile {
        SpeakerProfile {
            id: id.into(),
            name: name.into(),
            created_at: "2026-01-01T00:00:00.000Z".into(),
            updated_at: "2026-01-01T00:00:00.000Z".into(),
            sample_seconds: 8.0,
            embedding_dim: embedding.len(),
            pack_fingerprint: "sha256:legacy".into(),
            match_similarity: 0.5,
            embedding,
        }
    }

    #[test]
    fn migrates_same_name_profiles_to_distinct_persons() {
        let dir = tempdir().unwrap();
        let home = dir.path();
        let diarize = home.join("diarize");
        fs::create_dir_all(&diarize).unwrap();
        let json_path = diarize.join("voiceprints.json");
        let store_json = VoiceprintStore {
            version: VOICEPRINT_STORE_VERSION,
            profiles: vec![
                v1_profile("vp_aaaaaaaaaaaaaaaa", "Alice", vec![1.0, 0.0, 0.0, 0.0]),
                v1_profile("vp_bbbbbbbbbbbbbbbb", "Alice", vec![0.0, 1.0, 0.0, 0.0]),
            ],
        };
        fs::write(&json_path, serde_json::to_vec_pretty(&store_json).unwrap()).unwrap();

        let store = open_store_with_v1_migration(home).unwrap();
        let persons = store.list_persons(None).unwrap();
        assert_eq!(persons.len(), 2);
        assert_eq!(persons[0].display_name, "Alice");
        assert_eq!(persons[1].display_name, "Alice");
        assert_ne!(persons[0].person_id, persons[1].person_id);
        // Both need reenrollment: legacy space is non-matchable.
        assert!(persons.iter().all(|p| p.needs_reenrollment));
        // Aliases resolve.
        let a = store
            .resolve_legacy_profile_id("vp_aaaaaaaaaaaaaaaa")
            .unwrap()
            .unwrap();
        let b = store
            .resolve_legacy_profile_id("vp_bbbbbbbbbbbbbbbb")
            .unwrap()
            .unwrap();
        assert_ne!(a, b);
        assert!(
            !json_path.exists(),
            "v1 json should be removed after success"
        );

        // Idempotent second open.
        let store2 = open_store_with_v1_migration(home).unwrap();
        assert_eq!(store2.list_persons(None).unwrap().len(), 2);
    }

    #[test]
    fn corrupt_json_fails_closed_and_preserves_file() {
        let dir = tempdir().unwrap();
        let home = dir.path();
        let diarize = home.join("diarize");
        fs::create_dir_all(&diarize).unwrap();
        let json_path = diarize.join("voiceprints.json");
        fs::write(&json_path, b"{not-json").unwrap();
        let err = open_store_with_v1_migration(home).unwrap_err();
        assert!(matches!(err, VoiceIdMigrationError::FailClosed { .. }));
        assert!(json_path.exists());
    }

    #[test]
    fn illegal_embedding_fails_closed() {
        let dir = tempdir().unwrap();
        let home = dir.path();
        let diarize = home.join("diarize");
        fs::create_dir_all(&diarize).unwrap();
        let json_path = diarize.join("voiceprints.json");
        let mut profile = v1_profile("vp_cccccccccccccccc", "Bob", vec![1.0, 2.0]);
        profile.embedding_dim = 4; // mismatch
        let store_json = VoiceprintStore {
            version: VOICEPRINT_STORE_VERSION,
            profiles: vec![profile],
        };
        fs::write(&json_path, serde_json::to_vec_pretty(&store_json).unwrap()).unwrap();
        let err = open_store_with_v1_migration(home).unwrap_err();
        assert!(matches!(err, VoiceIdMigrationError::FailClosed { .. }));
        assert!(json_path.exists());
    }
}
