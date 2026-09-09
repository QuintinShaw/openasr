//! Policy-resolved ownership for the speaker-embedding stage.
//!
//! Family-specific admitted-host / actor / batch protocol lives in
//! `policy_family`. This module is the public load and diarizer facade and
//! type-erases at `Arc<dyn SpeakerEmbedder>`.

use std::sync::Arc;

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
    pub fn load(
        execution_services: Arc<NativeExecutionServices>,
    ) -> Result<Option<Self>, EmbedError> {
        Self::load_with_intent(execution_services, ExecutionIntent::Auto)
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
        let Some(prepared) = prepare_embedder(preference)? else {
            return Ok(None);
        };
        let loaded = match prepared.family {
            SpeakerEmbedderFamily::ReDimNet2 => policy_family::load_family::<RedimNetPolicy>(
                execution_services,
                execution_intent,
                prepared,
            )?,
            SpeakerEmbedderFamily::WeSpeakerResNet => {
                policy_family::load_family::<WeSpeakerPolicy>(
                    execution_services,
                    execution_intent,
                    prepared,
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
