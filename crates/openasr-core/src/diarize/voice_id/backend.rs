//! Diarizer backend boundary.
//!
//! Separates anonymous turn production from optional embedding evidence used by
//! Voice ID. MOSS decoder tags (`S01`, ...) are session-local anonymous labels
//! only -- never Person identities and never embedding evidence.

use crate::diarize::contract::{SpeakerEmbedding, SpeakerTurn, TimeRange};

/// Which diarization engine produced the turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiarizerBackendKind {
    /// VAD + speaker-embedder clustering path.
    VadEmbedder,
    /// MOSS-Transcribe-Diarize joint ASR+diarization path.
    MossTranscribeDiarize,
    /// Other / test backends.
    Other,
}

/// Optional per-turn embedding evidence that Voice ID may consume.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingEvidence {
    pub range: TimeRange,
    pub embedding: SpeakerEmbedding,
}

/// Normalized diarization output. Voice ID resolvers consume this instead of
/// reaching into backend-specific tag vocabularies.
#[derive(Debug, Clone, PartialEq)]
pub struct DiarizationOutput {
    pub backend_kind: DiarizerBackendKind,
    pub anonymous_turns: Vec<SpeakerTurn>,
    /// Present only when the backend actually produced speaker embeddings.
    /// MOSS must leave this `None`.
    pub optional_embedding_evidence: Option<Vec<EmbeddingEvidence>>,
}

impl DiarizationOutput {
    pub fn moss_anonymous(turns: Vec<SpeakerTurn>) -> Self {
        Self {
            backend_kind: DiarizerBackendKind::MossTranscribeDiarize,
            anonymous_turns: turns,
            optional_embedding_evidence: None,
        }
    }

    pub fn vad_embedder(turns: Vec<SpeakerTurn>, evidence: Vec<EmbeddingEvidence>) -> Self {
        Self {
            backend_kind: DiarizerBackendKind::VadEmbedder,
            anonymous_turns: turns,
            optional_embedding_evidence: Some(evidence),
        }
    }

    /// Whether this output may feed the Person matcher.
    pub fn supports_voice_id_matching(&self) -> bool {
        match self.backend_kind {
            DiarizerBackendKind::VadEmbedder => self
                .optional_embedding_evidence
                .as_ref()
                .is_some_and(|e| !e.is_empty()),
            DiarizerBackendKind::MossTranscribeDiarize => false,
            DiarizerBackendKind::Other => self
                .optional_embedding_evidence
                .as_ref()
                .is_some_and(|e| !e.is_empty()),
        }
    }
}

/// Resolve Voice ID only when embedding evidence exists. MOSS anonymous turns
/// always stay anonymous.
pub fn voice_id_evidence_from_output(output: &DiarizationOutput) -> Option<&[EmbeddingEvidence]> {
    if !output.supports_voice_id_matching() {
        return None;
    }
    output.optional_embedding_evidence.as_deref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarize::contract::SpeakerId;

    #[test]
    fn moss_output_never_supports_person_matching() {
        let turns = vec![SpeakerTurn {
            range: TimeRange::new(0.0, 1.0),
            speaker: SpeakerId(0),
            overlap: false,
        }];
        let out = DiarizationOutput::moss_anonymous(turns);
        assert!(!out.supports_voice_id_matching());
        assert!(voice_id_evidence_from_output(&out).is_none());
        // Even if a buggy caller stuffed evidence, backend kind still wins.
        let mut bad = out;
        bad.optional_embedding_evidence = Some(vec![EmbeddingEvidence {
            range: TimeRange::new(0.0, 1.0),
            embedding: SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]),
        }]);
        assert!(!bad.supports_voice_id_matching());
    }

    #[test]
    fn vad_embedder_output_exposes_evidence() {
        let turns = vec![SpeakerTurn {
            range: TimeRange::new(0.0, 1.0),
            speaker: SpeakerId(0),
            overlap: false,
        }];
        let evidence = vec![EmbeddingEvidence {
            range: TimeRange::new(0.0, 1.0),
            embedding: SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]),
        }];
        let out = DiarizationOutput::vad_embedder(turns, evidence);
        assert!(out.supports_voice_id_matching());
        assert_eq!(voice_id_evidence_from_output(&out).unwrap().len(), 1);
    }
}
