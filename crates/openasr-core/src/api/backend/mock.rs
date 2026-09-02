use super::{BackendError, Segment, Transcription, TranscriptionBackend, TranscriptionRequest};

#[derive(Debug, Default, Clone, Copy)]
pub struct MockBackend;

impl TranscriptionBackend for MockBackend {
    fn transcribe(&self, request: TranscriptionRequest) -> Result<Transcription, BackendError> {
        super::reject_unsupported_diarization(&request, "mock")?;
        super::reject_unsupported_phrase_bias(&request, "mock")?;
        if request.return_speaker_embeddings && !request.anonymous_diarize && !request.voice_id {
            return Err(BackendError::SpeakerEmbeddingsRequireDiarization);
        }
        #[cfg(any(test, feature = "testing"))]
        if crate::testing::apply_mock_transcribe_delay() {
            return Err(BackendError::TranscriptionCanceled);
        }
        // OADP Phase 0: fail closed instead of pretending the adapter applied.
        if request.adapter_path.is_some() {
            return Err(BackendError::AdapterNotSupported { backend: "mock" });
        }

        let file_name = request
            .display_file_name
            .as_deref()
            .or_else(|| {
                request
                    .input_path
                    .file_name()
                    .and_then(|name| name.to_str())
            })
            .unwrap_or("audio");
        let text = format!(
            "OpenASR mock transcription for {file_name} using {}.",
            request.model_id
        );
        let (speaker, speaker_label) = if request.anonymous_diarize {
            (
                Some("SPEAKER_00".to_string()),
                Some("SPEAKER_00".to_string()),
            )
        } else {
            (None, None)
        };

        Ok(Transcription {
            truncated_decodes: Vec::new(),
            unnamed_speakers: Vec::new(),
            text: text.clone(),
            segments: vec![Segment {
                start: 0.0,
                end: 2.5,
                text,
                speaker,
                speaker_label,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            }],
            longform: None,
            language: None,
            ..Default::default()
        })
    }
}

pub fn transcribe_with_mock_backend(
    request: TranscriptionRequest,
) -> Result<Transcription, BackendError> {
    MockBackend.transcribe(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_backend_returns_deterministic_transcript() {
        let backend = MockBackend;
        let request = TranscriptionRequest::new("fixtures/jfk.wav", "whisper-tiny");

        let transcription = backend.transcribe(request).unwrap();

        assert_eq!(
            transcription.text,
            "OpenASR mock transcription for jfk.wav using whisper-tiny."
        );
        assert_eq!(transcription.segments.len(), 1);
        assert_eq!(transcription.segments[0].start, 0.0);
        assert_eq!(transcription.segments[0].end, 2.5);
        assert_eq!(transcription.segments[0].text, transcription.text);
        assert_eq!(transcription.segments[0].speaker, None);
    }

    #[test]
    fn mock_backend_rejects_diarization_without_fake_speakers() {
        let backend = MockBackend;
        let request =
            TranscriptionRequest::new("fixtures/jfk.wav", "whisper-tiny").with_voice_id(true);

        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(error.contains("speaker-embedder pack"));
        assert!(error.contains("redimnet2-b6-cn"));
        assert!(error.contains("mock backend"));
        assert!(error.contains("omit --diarize / diarize=true"));
    }

    #[test]
    fn mock_backend_anonymous_diarize_labels_speaker_without_person_id() {
        let backend = MockBackend;
        let request = TranscriptionRequest::new("fixtures/jfk.wav", "whisper-tiny")
            .with_anonymous_diarize(true);

        let transcription = backend.transcribe(request).unwrap();
        assert_eq!(
            transcription.segments[0].speaker_label.as_deref(),
            Some("SPEAKER_00")
        );
        assert_eq!(
            transcription.segments[0].speaker.as_deref(),
            Some("SPEAKER_00")
        );
        assert!(transcription.segments[0].speaker_person_id.is_none());
        assert!(transcription.speaker_embeddings.is_none());
    }

    #[test]
    fn mock_backend_does_not_fabricate_speaker_embeddings() {
        let backend = MockBackend;
        let request = TranscriptionRequest::new("fixtures/jfk.wav", "whisper-tiny")
            .with_anonymous_diarize(true)
            .with_return_speaker_embeddings(true);

        let transcription = backend.transcribe(request).unwrap();
        assert_eq!(
            transcription.segments[0].speaker_label.as_deref(),
            Some("SPEAKER_00")
        );
        assert!(transcription.speaker_embeddings.is_none());
    }

    #[test]
    fn mock_backend_rejects_speaker_embeddings_without_diarization() {
        let backend = MockBackend;
        let request = TranscriptionRequest::new("fixtures/jfk.wav", "whisper-tiny")
            .with_return_speaker_embeddings(true);

        let error = backend.transcribe(request).unwrap_err().to_string();
        assert!(error.contains("return_speaker_embeddings requires diarize=true"));
    }

    #[test]
    fn mock_backend_rejects_adapter_instead_of_ignoring_it() {
        let backend = MockBackend;
        let request = TranscriptionRequest::new("fixtures/jfk.wav", "whisper-tiny")
            .with_adapter_path(Some(std::path::PathBuf::from("adapter.oadp")));

        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(error.contains("Adapter packs"));
        assert!(error.contains("mock backend"));
        assert!(error.contains("rejected instead of silently ignoring"));
    }

    #[test]
    fn mock_backend_rejects_phrase_bias_instead_of_ignoring_it() {
        let backend = MockBackend;
        let phrase_bias = crate::PhraseBiasConfig::from_phrases([("OpenASR", 2.0)]).unwrap();
        let request = TranscriptionRequest::new("fixtures/jfk.wav", "whisper-tiny")
            .with_phrase_bias(Some(phrase_bias));

        let error = backend.transcribe(request).unwrap_err().to_string();

        assert!(error.contains("Phrase bias / hotword boosting is not supported"));
        assert!(error.contains("mock backend"));
        assert!(error.contains("silently ignoring phrase_bias"));
    }
}
