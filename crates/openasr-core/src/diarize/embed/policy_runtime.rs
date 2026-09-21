//! Policy-resolved ownership for the speaker-embedding stage.
//!
//! Family-specific admitted-host / actor / batch protocol lives in
//! `policy_family`. This module is the public load and diarizer facade and
//! type-erases at `Arc<dyn SpeakerEmbedder>`.

use std::{num::NonZeroU16, sync::Arc};

use crate::{NativeExecutionServices, device::execution_policy::ExecutionIntent};

use super::{
    EmbedError, SpeakerEmbedder, SpeakerEmbedderFamily, SpeakerEmbedderIdentity,
    pack::prepare_embedder,
    policy_family::{self, RedimNetPolicy, WeSpeakerPolicy},
};
use crate::config::VoiceIdEmbedderPreference;
use crate::diarize::{
    streaming::{StreamingDiarizer, StreamingSpeakerChangeDetector},
    voice_id::{EmbeddingSpace, PersonMatcher, load_person_matcher_for_embedder},
};

#[derive(Clone)]
pub struct PolicyResolvedSpeakerRuntime {
    embedder: Arc<dyn SpeakerEmbedder>,
    identity: SpeakerEmbedderIdentity,
}

impl PolicyResolvedSpeakerRuntime {
    /// Resolve the selected embedding space through the verified pack boundary.
    /// Metadata queries must not admit, construct or warm an inference runtime.
    pub fn resolve_selected_identity() -> Result<Option<SpeakerEmbedderIdentity>, EmbedError> {
        Ok(prepare_embedder(persisted_embedder_preference())?.map(|prepared| prepared.identity()))
    }

    pub fn load(
        execution_services: Arc<NativeExecutionServices>,
    ) -> Result<Option<Self>, EmbedError> {
        Self::load_with_intent(execution_services, ExecutionIntent::Auto)
    }

    /// Carry the admitted session's CPU budget into speaker inference too.
    /// Keep it on this runtime, not in process-global or thread-local state:
    /// embedding work moves between the connection, blocking and actor threads.
    pub fn load_with_inference_threads(
        execution_services: Arc<NativeExecutionServices>,
        inference_threads: Option<NonZeroU16>,
    ) -> Result<Option<Self>, EmbedError> {
        Self::load_with_settings(
            execution_services,
            ExecutionIntent::Auto,
            persisted_embedder_preference(),
            inference_threads,
        )
    }

    pub(crate) fn load_with_intent(
        execution_services: Arc<NativeExecutionServices>,
        execution_intent: ExecutionIntent,
    ) -> Result<Option<Self>, EmbedError> {
        Self::load_with_preference(
            execution_services,
            execution_intent,
            persisted_embedder_preference(),
        )
    }

    pub(crate) fn load_with_preference(
        execution_services: Arc<NativeExecutionServices>,
        execution_intent: ExecutionIntent,
        preference: VoiceIdEmbedderPreference,
    ) -> Result<Option<Self>, EmbedError> {
        Self::load_with_settings(execution_services, execution_intent, preference, None)
    }

    fn load_with_settings(
        execution_services: Arc<NativeExecutionServices>,
        execution_intent: ExecutionIntent,
        preference: VoiceIdEmbedderPreference,
        inference_threads: Option<NonZeroU16>,
    ) -> Result<Option<Self>, EmbedError> {
        let Some(prepared) = prepare_embedder(preference)? else {
            return Ok(None);
        };
        let loaded = match prepared.family {
            SpeakerEmbedderFamily::ReDimNet2 => policy_family::load_family::<RedimNetPolicy>(
                execution_services,
                execution_intent,
                prepared,
                inference_threads,
            )?,
            SpeakerEmbedderFamily::WeSpeakerResNet => {
                policy_family::load_family::<WeSpeakerPolicy>(
                    execution_services,
                    execution_intent,
                    prepared,
                    inference_threads,
                )?
            }
        };
        Ok(loaded.map(|(embedder, identity)| Self { embedder, identity }))
    }

    pub fn diarizer(
        &self,
        sample_rate_hz: u32,
    ) -> Result<StreamingDiarizer, crate::diarize::voice_id::VoiceIdLibraryError> {
        let persons = load_person_matcher_for_embedder(&self.identity, self.embedder.as_ref())?;
        Ok(StreamingDiarizer::with_shared_embedder_and_persons(
            Arc::clone(&self.embedder),
            sample_rate_hz,
            persons,
        ))
    }

    /// Anonymous SPEAKER_00 clustering. Enrolled Voice ID names stay on the
    /// originating client; this matcher is intentionally empty.
    pub fn anonymous_diarizer(&self, sample_rate_hz: u32) -> StreamingDiarizer {
        let space = EmbeddingSpace::for_active_embedder(&self.identity);
        StreamingDiarizer::with_shared_embedder_and_persons(
            Arc::clone(&self.embedder),
            sample_rate_hz,
            PersonMatcher::new(space, Vec::new(), 1.0, 0.0),
        )
    }

    pub fn speaker_change_detector(&self, sample_rate_hz: u32) -> StreamingSpeakerChangeDetector {
        StreamingSpeakerChangeDetector::with_shared_embedder(
            Arc::clone(&self.embedder),
            sample_rate_hz,
        )
    }

    pub fn identity(&self) -> &SpeakerEmbedderIdentity {
        &self.identity
    }

    pub fn embedder(&self) -> &dyn SpeakerEmbedder {
        self.embedder.as_ref()
    }

    pub(crate) fn shared_embedder(&self) -> Arc<dyn SpeakerEmbedder> {
        Arc::clone(&self.embedder)
    }
}

fn persisted_embedder_preference() -> VoiceIdEmbedderPreference {
    crate::openasr_home()
        .ok()
        .and_then(|home| crate::config::load_config_document(home).ok())
        .map(|document| document.preferences.voice_id_embedder)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_identity_uses_verified_metadata_without_runtime_weights() {
        use crate::models::pack_verifier::{PackCandidate, PackVerifier};
        use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

        for (architecture, variable, expected) in [
            (
                "redimnet2",
                "OPENASR_REDIMNET_PACK",
                SpeakerEmbedderIdentity::redimnet2("", "fixture-speaker"),
            ),
            (
                "wespeaker-resnet",
                "OPENASR_WESPEAKER_PACK",
                SpeakerEmbedderIdentity::wespeaker_resnet("", "fixture-speaker"),
            ),
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            let pack = dir.path().join("speaker.oasr");
            let spec = TinyGgufFixtureSpec::new(
                [
                    ("openasr.package.version", "1"),
                    ("general.architecture", architecture),
                    ("openasr.model.id", "fixture-speaker"),
                ]
                .into_iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect(),
            );
            write_tiny_gguf_runtime_source(&pack, &spec).expect("write metadata fixture");
            let verified = PackVerifier
                .verify_candidate(PackCandidate::new(&pack))
                .expect("metadata fixture must cross production verification");
            let expected = SpeakerEmbedderIdentity {
                pack_fingerprint: verified.content_id().to_string(),
                ..expected
            };
            crate::test_process_env::with_test_process_env(
                [
                    ("OPENASR_HOME", Some(dir.path().as_os_str().to_os_string())),
                    ("OPENASR_MODELS_DIR", None),
                    (
                        "OPENASR_REDIMNET_PACK",
                        (variable == "OPENASR_REDIMNET_PACK")
                            .then(|| pack.as_os_str().to_os_string()),
                    ),
                    (
                        "OPENASR_WESPEAKER_PACK",
                        (variable == "OPENASR_WESPEAKER_PACK")
                            .then(|| pack.as_os_str().to_os_string()),
                    ),
                ],
                || {
                    // The single scalar tensor cannot construct either model.
                    // A runtime load here would fail rather than yield identity.
                    assert_eq!(
                        PolicyResolvedSpeakerRuntime::resolve_selected_identity()
                            .expect("resolve verified identity")
                            .expect("selected pack"),
                        expected
                    );
                },
            );
        }
    }

    #[test]
    fn selected_identity_preserves_missing_pack_semantics() {
        let dir = tempfile::tempdir().expect("tempdir");
        crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_HOME", Some(dir.path().as_os_str().to_os_string())),
                ("OPENASR_MODELS_DIR", None),
                ("OPENASR_REDIMNET_PACK", None),
                ("OPENASR_WESPEAKER_PACK", None),
            ],
            || {
                assert!(
                    PolicyResolvedSpeakerRuntime::resolve_selected_identity()
                        .expect("default missing pack is optional")
                        .is_none()
                );
                std::fs::write(
                    dir.path().join("config.json"),
                    r#"{"preferences":{"voice_id_embedder":"wespeaker"}}"#,
                )
                .expect("write preference");
                let error = PolicyResolvedSpeakerRuntime::resolve_selected_identity()
                    .expect_err("explicit missing pack must fail closed");
                assert!(error.to_string().contains("WeSpeaker"));
            },
        );
    }

    #[test]
    fn selected_identity_rejects_invalid_pack() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = dir.path().join("invalid.oasr");
        std::fs::write(&pack, b"not a verified speaker pack").expect("write invalid pack");
        crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_HOME", Some(dir.path().as_os_str().to_os_string())),
                ("OPENASR_MODELS_DIR", None),
                (
                    "OPENASR_REDIMNET_PACK",
                    Some(pack.as_os_str().to_os_string()),
                ),
                ("OPENASR_WESPEAKER_PACK", None),
            ],
            || {
                assert!(PolicyResolvedSpeakerRuntime::resolve_selected_identity().is_err());
            },
        );
    }

    #[test]
    fn load_with_intent_reads_persisted_wespeaker_preference() {
        assert_eq!(
            persisted_embedder_preference(),
            VoiceIdEmbedderPreference::ReDimNet2
        );
        let dir = tempfile::tempdir().expect("tempdir");
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{"preferences":{"voice_id_embedder":"wespeaker"}}"#,
        )
        .expect("write config");
        crate::test_process_env::with_test_process_env(
            [("OPENASR_HOME", Some(dir.path().as_os_str().to_os_string()))],
            || {
                assert_eq!(
                    persisted_embedder_preference(),
                    VoiceIdEmbedderPreference::WeSpeaker
                );
            },
        );
    }
}
