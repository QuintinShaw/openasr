//! The source-independent identity stage: turn scope-local speaker labels into
//! known people.
//!
//! # Scope is the load-bearing idea
//!
//! Segmentation answers "these turns are by the same speaker". That answer is
//! only valid inside the **scope** it was computed in -- one decode unit. A
//! source numbers speakers in arrival order within its own scope, so scope A's
//! `SPEAKER_01` and scope B's `SPEAKER_01` are two unrelated labels that happen
//! to collide. Today one transcription is one scope; serve-batch will cut one
//! transcription into several slices decoded independently, and each slice will
//! be its own scope. Nothing else about this stage changes when that lands,
//! which is the point of making scope explicit ([`SpeakerScope`]) instead of
//! assuming a single continuous decode.
//!
//! Stitching scopes back together -- deciding that A's `SPEAKER_01` and B's
//! `SPEAKER_02` are one person -- is therefore not an optional nicety layered on
//! top of transcription; it is what makes the speaker labels of a multi-scope
//! transcript mean anything at all, and it can only be done from voice
//! evidence. That is this module's job, and it is why it works from embeddings
//! and never from labels: a label is a counter, not an identity.
//!
//! # Erring toward anonymous
//!
//! A wrong name is worse than no name. A user who sees "Ada" believes it; being
//! told "Speaker 2" merely fails to help. So every gate here is one-sided: when
//! the evidence is thin or the match is borderline, the label stays anonymous
//! and the person is left for the user to identify. Concretely this stage
//! - matches through [`PersonMatcher::best_match`], the strict default gate
//!   (accept threshold **and** top1-vs-top2 margin, both from the embedder's
//!   calibration profile). It deliberately does not use
//!   `best_match_with_gates`, whose `threshold_tolerance` lowers the accept
//!   floor for the latency-bound streaming path -- a batch transcript has no
//!   such excuse;
//! - refuses to name a label backed by less audio than the diarization pipeline
//!   considers embeddable at all
//!   ([`crate::diarize::pipeline::MIN_EMBEDDING_EVIDENCE_SECONDS`]);
//! - emits a name or nothing, never a hedge. "Probably Ada" is not a state a
//!   non-technical user can act on, and it invites exactly the misplaced trust
//!   the strict gates exist to prevent.
//!
//! Anyone loosening these should read this paragraph first: the recall they
//! gain (a familiar voice occasionally not recognized, one manual tap to fix)
//! is being traded for transcripts that confidently attribute words to the
//! wrong person.

use std::collections::BTreeMap;

use crate::Segment;
use crate::diarize::contract::SpeakerEmbedding;
use crate::diarize::pipeline::MIN_EMBEDDING_EVIDENCE_SECONDS;

/// Audio sample rate the speaker embedder is fed at; the transcription
/// pipeline resamples to this before decode, so segment times index directly
/// into a scope's samples at this rate.
const EMBEDDER_SAMPLE_RATE_HZ: usize = 16_000;

/// One decode unit's labeled segments together with the audio they refer to.
///
/// Speaker labels are meaningful only within one of these. A caller hands the
/// identity stage every scope of a transcription at once precisely so the stage
/// can relate them; handing them over one at a time would silently reintroduce
/// the "same number means same person" assumption this type exists to deny.
pub struct SpeakerScope<'a> {
    /// Segments carrying this scope's own `SPEAKER_NN` labels, rewritten in
    /// place with whatever identity could be established.
    pub segments: &'a mut [Segment],
    /// 16 kHz mono audio the segment times index into. May be empty, in which
    /// case nothing in this scope can be named.
    pub samples: &'a [f32],
}

/// Establish identities across every scope of one transcription.
///
/// Labels that match the same enrolled person -- in the same scope or in
/// different ones -- end up with the same display name and `person_id`.
/// Labels that match nobody keep an anonymous label, made globally distinct
/// when there is more than one scope (see [`SpeakerScope`]: two scopes'
/// identical numbering must never read as one speaker just because no evidence
/// was available to tell them apart).
pub fn name_speakers_across_scopes(scopes: &mut [SpeakerScope<'_>]) {
    for scope in scopes.iter_mut() {
        normalize_local_labels(scope.segments);
    }
    if scopes.len() > 1 {
        disambiguate_labels_across_scopes(scopes);
    }
    let Some(embedder) = crate::diarize::embed::shared_embedder() else {
        // No embedder: separation stands, naming does not. Labels stay
        // anonymous rather than guessed at.
        return;
    };

    // Keyed by the label as it now reads, which is scope-unique after the
    // disambiguation pass above -- so evidence from two scopes is never pooled
    // into one centroid unless a caller genuinely produced one scope.
    let mut evidence: BTreeMap<String, LabelEvidence> = BTreeMap::new();
    for scope in scopes.iter() {
        for segment in scope.segments.iter() {
            let Some(label) = segment.speaker_label.as_ref() else {
                continue;
            };
            let Some(clip) = segment_clip(segment, scope.samples) else {
                continue;
            };
            let Ok(embedding) = embedder.embed(clip, EMBEDDER_SAMPLE_RATE_HZ as u32) else {
                continue;
            };
            let entry = evidence
                .entry(label.clone())
                .or_insert_with(|| LabelEvidence::new(embedding.dim()));
            entry.accumulate(
                &embedding,
                clip.len() as f64 / EMBEDDER_SAMPLE_RATE_HZ as f64,
            );
        }
    }

    let matcher = super::load_person_matcher_for_active_embedder();
    let matches: BTreeMap<String, super::PersonMatch> = evidence
        .into_iter()
        .filter_map(|(label, evidence)| {
            let centroid = evidence.centroid()?;
            matcher
                .best_match(&centroid)
                .map(|matched| (label, matched))
        })
        .collect();

    for scope in scopes.iter_mut() {
        for segment in scope.segments.iter_mut() {
            let Some(label) = segment.speaker_label.as_deref() else {
                continue;
            };
            let Some(person) = matches.get(label) else {
                continue;
            };
            segment.speaker = Some(person.display_name.clone());
            segment.speaker_person_id = Some(person.person_id.as_str().to_string());
            segment.speaker_snapshot_label = Some(person.display_name.clone());
        }
    }
}

/// Single-scope convenience for the one caller that has a single decode unit
/// (every offline transcription today). Same semantics as
/// [`name_speakers_across_scopes`] with one scope.
pub fn name_speakers_from_labeled_segments(segments: &mut [Segment], samples: &[f32]) {
    name_speakers_across_scopes(&mut [SpeakerScope { segments, samples }]);
}

/// Summed embeddings plus how much audio backed them, per label.
struct LabelEvidence {
    sum: Vec<f32>,
    seconds: f64,
}

impl LabelEvidence {
    fn new(dim: usize) -> Self {
        Self {
            sum: vec![0.0; dim],
            seconds: 0.0,
        }
    }

    fn accumulate(&mut self, embedding: &SpeakerEmbedding, seconds: f64) {
        if self.sum.len() != embedding.dim() {
            return;
        }
        for (sum, value) in self.sum.iter_mut().zip(&embedding.0) {
            *sum += value;
        }
        self.seconds += seconds;
    }

    /// The label's mean embedding, or `None` when too little audio backs it to
    /// risk putting a name to it (see the module doc's "Erring toward
    /// anonymous").
    fn centroid(self) -> Option<SpeakerEmbedding> {
        (self.seconds >= MIN_EMBEDDING_EVIDENCE_SECONDS)
            .then(|| SpeakerEmbedding::l2_normalized(self.sum))
    }
}

/// Keep the displayed speaker and the stable scope-local label in sync before
/// matching: `speaker` is what a caller renders, `speaker_label` is the label
/// identity resolution keys on, and a segmentation source may have set only one
/// of them.
fn normalize_local_labels(segments: &mut [Segment]) {
    for segment in segments.iter_mut() {
        if segment.speaker_label.is_none() {
            segment.speaker_label = segment.speaker.clone();
        }
        if segment.speaker.is_none()
            && let Some(label) = &segment.speaker_label
        {
            segment.speaker = Some(label.clone());
        }
    }
}

/// Renumber every scope's labels into one globally distinct series, in order of
/// first appearance.
///
/// Two scopes both numbering their speakers from one is the normal case, not an
/// error, so the collision has to be resolved before anything reads the labels.
/// It is resolved by splitting (each scope's label becomes its own speaker),
/// never by merging: without voice evidence there is nothing to justify calling
/// two scopes' speakers the same person, and over-counting speakers is the
/// recoverable mistake. Matching then re-merges whatever the embeddings support.
fn disambiguate_labels_across_scopes(scopes: &mut [SpeakerScope<'_>]) {
    let mut next_index = 0_u32;
    for scope in scopes.iter_mut() {
        let mut renamed: BTreeMap<String, String> = BTreeMap::new();
        for segment in scope.segments.iter_mut() {
            let Some(label) = segment.speaker_label.as_deref() else {
                continue;
            };
            let global = renamed.entry(label.to_string()).or_insert_with(|| {
                let global = crate::diarize::contract::SpeakerId(next_index).label();
                next_index += 1;
                global
            });
            if segment.speaker.as_deref() == segment.speaker_label.as_deref() {
                segment.speaker = Some(global.clone());
            }
            segment.speaker_label = Some(global.clone());
        }
    }
}

fn segment_clip<'a>(segment: &Segment, samples: &'a [f32]) -> Option<&'a [f32]> {
    let rate = EMBEDDER_SAMPLE_RATE_HZ as f32;
    let start = (segment.start.max(0.0) * rate).floor() as usize;
    let end = (segment.end.max(segment.start).max(0.0) * rate).ceil() as usize;
    samples.get(start.min(samples.len())..end.min(samples.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labeled(start: f32, end: f32, speaker: Option<&str>) -> Segment {
        Segment {
            start,
            end,
            text: "hello".to_string(),
            speaker: speaker.map(str::to_string),
            speaker_label: None,
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words: Vec::new(),
        }
    }

    /// Without an embedder the stage must still leave usable scope-local labels
    /// behind (the "can separate, cannot name" degrade), and must never invent
    /// a person id out of a label.
    #[test]
    fn local_labels_survive_and_never_become_person_identities() {
        let mut segments = vec![
            labeled(0.0, 1.0, Some("SPEAKER_01")),
            labeled(1.0, 2.0, None),
        ];
        name_speakers_from_labeled_segments(&mut segments, &[]);

        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segments[0].speaker_label.as_deref(), Some("SPEAKER_01"));
        assert!(segments[0].speaker_person_id.is_none());
        assert!(segments[1].speaker.is_none());
        assert!(segments[1].speaker_person_id.is_none());
    }

    #[test]
    fn a_label_only_segment_gets_its_display_speaker_filled_in() {
        let mut segments = vec![labeled(0.0, 1.0, None)];
        segments[0].speaker_label = Some("SPEAKER_03".to_string());
        name_speakers_from_labeled_segments(&mut segments, &[]);
        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_03"));
    }

    /// A single scope keeps its source's own numbering verbatim -- including a
    /// gap, which a family may legitimately assert (see
    /// `models::moss_transcribe_diarize::speaker_segments`).
    #[test]
    fn a_single_scope_keeps_the_source_numbering_verbatim() {
        let mut segments = vec![
            labeled(0.0, 1.0, Some("SPEAKER_01")),
            labeled(1.0, 2.0, Some("SPEAKER_05")),
        ];
        name_speakers_from_labeled_segments(&mut segments, &[]);
        assert_eq!(segments[0].speaker_label.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segments[1].speaker_label.as_deref(), Some("SPEAKER_05"));
    }

    /// The serve-batch contract: two independently decoded scopes both start
    /// numbering at one, and those two `SPEAKER_01`s are unrelated. With no
    /// voice evidence to relate them they must come out as distinct speakers,
    /// never silently merged into one.
    #[test]
    fn identical_labels_in_two_scopes_are_never_merged_without_evidence() {
        let mut first = vec![
            labeled(0.0, 1.0, Some("SPEAKER_01")),
            labeled(1.0, 2.0, Some("SPEAKER_02")),
        ];
        let mut second = vec![
            labeled(0.0, 1.0, Some("SPEAKER_01")),
            labeled(1.0, 2.0, Some("SPEAKER_01")),
        ];
        name_speakers_across_scopes(&mut [
            SpeakerScope {
                segments: &mut first,
                samples: &[],
            },
            SpeakerScope {
                segments: &mut second,
                samples: &[],
            },
        ]);

        let label = |segment: &Segment| segment.speaker_label.clone().unwrap();
        // Within a scope, one label stays one speaker.
        assert_eq!(label(&second[0]), label(&second[1]));
        // Across scopes, colliding labels are split apart.
        assert_ne!(label(&first[0]), label(&second[0]));
        assert_ne!(label(&first[1]), label(&second[0]));
        // The rendered speaker follows the label, and no identity was invented.
        for segment in first.iter().chain(second.iter()) {
            assert_eq!(segment.speaker, segment.speaker_label);
            assert!(segment.speaker_person_id.is_none());
        }
    }

    /// Evidence too short to embed reliably yields no centroid at all, so such
    /// a label can never be handed to the matcher.
    #[test]
    fn thin_evidence_is_not_worth_a_name() {
        let mut evidence = LabelEvidence::new(2);
        evidence.accumulate(
            &SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]),
            MIN_EMBEDDING_EVIDENCE_SECONDS / 2.0,
        );
        assert!(evidence.centroid().is_none());

        let mut evidence = LabelEvidence::new(2);
        evidence.accumulate(
            &SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]),
            MIN_EMBEDDING_EVIDENCE_SECONDS,
        );
        assert!(evidence.centroid().is_some());
    }
}
