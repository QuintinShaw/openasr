//! The source-independent identity stage: turn scope-local speaker labels into
//! known people.
//!
//! # Scope is the load-bearing idea
//!
//! Segmentation answers "these turns are by the same speaker". That answer is
//! only valid inside the **scope** it was computed in -- one decode unit. A
//! source numbers speakers in arrival order within its own scope, so scope A's
//! `SPEAKER_01` and scope B's `SPEAKER_01` are two unrelated labels that happen
//! to collide. A whole-recording decode is one scope; a recording cut into
//! longform slices that an in-decoder-diarizing family decoded independently
//! (`arch::OpenAsrLongformSliceShape::ScopedSlices`) is one scope per slice.
//! Nothing about this stage depends on which of those produced its input, which
//! is the point of making scope explicit ([`SpeakerScope`]).
//!
//! Stitching scopes back together -- deciding that A's `SPEAKER_01` and B's
//! `SPEAKER_02` are one person -- is therefore not an optional nicety layered on
//! top of transcription; it is what makes the speaker labels of a multi-scope
//! transcript mean anything at all, and it can only be done from voice
//! evidence. That is this module's job, and it is why it works from embeddings
//! and never from labels: a label is a counter, not an identity. It happens in
//! two steps that must stay in this order: every scope's labels are first split
//! apart unconditionally ([`disambiguate_labels_across_scopes`]), then only
//! acoustic agreement puts any of them back together
//! ([`stitch_labels_across_scopes`], and enrolled-person matching further
//! down). Nothing in between is allowed to treat a shared number as a shared
//! person.
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
//! - refuses to name a label backed by too little usable voice
//!   ([`MIN_NAMING_EVIDENCE_SECONDS`], measured independently of whatever
//!   segmenter produced the labels -- see that constant);
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

/// Audio sample rate the speaker embedder is fed at; the transcription
/// pipeline resamples to this before decode, so segment times index directly
/// into a scope's samples at this rate.
const EMBEDDER_SAMPLE_RATE_HZ: usize = 16_000;

/// # The invariant these two constants exist to enforce
///
/// This stage *validates* someone else's output: a segmenter decided which
/// turns exist and how finely to cut them, and this stage then decides whether
/// the voice behind a label is worth risking a person's name on. Those are two
/// different judgements and they must never share a number.
///
/// If the naming gate is stated in terms of the segmenter's own minimum
/// segment length, the check is a closed loop. That is not hypothetical -- it
/// shipped. The gate was `accumulated_seconds >= pipeline::MIN_SEGMENT_S`,
/// and the segmenter guarantees every segment it emits is at least
/// `MIN_SEGMENT_S` long, so any label with even one segment cleared the gate
/// by construction. On the six real meeting sessions this repo evaluates
/// against (AliMeeting far-field + AISHELL-4, 6 x 600s, 45 labelled speakers)
/// it accepted **45 of 45** -- including four people who say a single word all
/// meeting. No value of that constant fixes it: raising it to make naming
/// stricter also makes the segmenter throw away more short turns, which costs
/// diarization recall for a reason that has nothing to do with naming.
///
/// **INVARIANT: nothing here may read a segmenter's minimum segment length,
/// and no segmenter may read these.** "Unifying the constants" reinstates the
/// bug. `pipeline::MIN_SEGMENT_S` is private for that reason.
///
/// # What replaces it
///
/// How much voice backs a label, counted in a way that does not assume any
/// particular segment granularity: only segments individually long enough to
/// produce a stable embedding count, and enough of them must accumulate.
///
/// Total duration alone is the wrong measure, and the corpus shows why: a
/// label made of twenty sub-second "mm"s and a label with one continuous turn
/// can carry the same number of seconds while being worlds apart as evidence.
/// AISHELL-4 `L_R003S01C02` speaker `003-F` is exactly that -- 5.18s spread
/// over eight fragments of 0.39-0.93s, every one of them too short for a
/// trustworthy embedding, and a plain 3s total would have named her.
///
/// Shortest single segment whose embedding is stable enough to be evidence for
/// naming. Corroborated by the streaming path, which independently arrived at
/// the same figure for the same question
/// (`diarize::streaming::MIN_CENTROID_UPDATE_DURATION_S`: "centroids are
/// updated only from embeddings with enough speech context to be stable").
/// The corpus surface is flat here: anything in 0.75-1.5s rejects exactly the
/// same ten labels.
const MIN_RELIABLE_EMBEDDING_SECONDS: f64 = 1.0;

/// How much of that reliable voice a label needs before this stage will match
/// it against enrolled people at all.
///
/// Naming a real person cannot be cheaper than inventing an anonymous one, and
/// the streaming path already requires 2.5s before it will create even an
/// anonymous session speaker (`streaming::MIN_NEW_SPEAKER_DURATION_S`), so 3s
/// is the floor consistency demands. It is also where the corpus is flat and
/// the two populations are cleanly apart: over those 45 speakers the reliable
/// seconds are 0, 0, 1.08, 1.29, 1.29, 1.32, 1.43, 1.70, 1.84, 1.98 | 4.28,
/// 4.84, 5.93, 9.40, ... -- **nothing at all lands between 2.0 and 4.3**, so
/// any threshold in 2.0-4.0 rejects the same ten walk-on speakers and keeps
/// the same thirty-five real participants. Every genuine meeting participant
/// clears it by 1.4x or more, most by two orders of magnitude.
///
/// This is deliberately one-sided, per this module's "erring toward
/// anonymous": the cost of rejecting is one familiar voice left as
/// "Speaker 2", the cost of accepting is a transcript that confidently
/// attributes a stranger's words to someone the user knows.
const MIN_NAMING_EVIDENCE_SECONDS: f64 = 3.0;

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
    for (scope_index, scope) in scopes.iter().enumerate() {
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
                .or_insert_with(|| LabelEvidence::new(embedding.dim(), scope_index));
            entry.accumulate(
                &embedding,
                clip.len() as f64 / EMBEDDER_SAMPLE_RATE_HZ as f64,
            );
        }
    }

    // Put the scopes back together: labels from different scopes whose voices
    // match become one speaker again. Without this, disambiguation's deliberate
    // over-split is the final answer and every scope seam reads as a fresh cast
    // of speakers.
    if scopes.len() > 1 {
        let stitched = stitch_labels_across_scopes(
            &evidence,
            &crate::diarize::clustering::AgglomerativeClusterer::for_embedder(embedder),
        );
        if !stitched.is_empty() {
            evidence = pool_evidence_by_canonical_label(evidence, &stitched);
            for scope in scopes.iter_mut() {
                for segment in scope.segments.iter_mut() {
                    let Some(label) = segment.speaker_label.as_deref() else {
                        continue;
                    };
                    let Some(canonical) = stitched.get(label) else {
                        continue;
                    };
                    if segment.speaker.as_deref() == segment.speaker_label.as_deref() {
                        segment.speaker = Some(canonical.clone());
                    }
                    segment.speaker_label = Some(canonical.clone());
                }
            }
        }
    }

    let matcher = super::load_person_matcher_for_active_embedder();
    let matches: BTreeMap<String, super::PersonMatch> = evidence
        .into_iter()
        .filter_map(|(label, evidence)| {
            let centroid = evidence.centroid_for_naming()?;
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
    /// Of `seconds`, the part contributed by segments that were individually
    /// at least [`MIN_RELIABLE_EMBEDDING_SECONDS`] long. This is the quantity
    /// both gates below judge by, and the reason they do not care how finely
    /// the segmenter cut: a run of sub-second fragments adds to `seconds` and
    /// contributes nothing here.
    reliable_seconds: f64,
    /// Which scope this label belongs to. Two labels sharing a scope were told
    /// apart by that scope's own segmenter and must never be stitched back
    /// together (see [`stitch_labels_across_scopes`]).
    scope_index: usize,
}

impl LabelEvidence {
    fn new(dim: usize, scope_index: usize) -> Self {
        Self {
            sum: vec![0.0; dim],
            seconds: 0.0,
            reliable_seconds: 0.0,
            scope_index,
        }
    }

    fn absorb(&mut self, other: &LabelEvidence) {
        if self.sum.len() != other.sum.len() {
            return;
        }
        for (sum, value) in self.sum.iter_mut().zip(&other.sum) {
            *sum += value;
        }
        self.seconds += other.seconds;
        self.reliable_seconds += other.reliable_seconds;
    }

    fn accumulate(&mut self, embedding: &SpeakerEmbedding, seconds: f64) {
        if self.sum.len() != embedding.dim() {
            return;
        }
        for (sum, value) in self.sum.iter_mut().zip(&embedding.0) {
            *sum += value;
        }
        self.seconds += seconds;
        if seconds >= MIN_RELIABLE_EMBEDDING_SECONDS {
            self.reliable_seconds += seconds;
        }
    }

    /// The label's mean embedding for *stitching* -- deciding that two scopes'
    /// labels are the same voice.
    ///
    /// The bar is only that some single segment was long enough to embed
    /// stably at all: stitching's failure mode is fusing two people, and the
    /// alternative to stitching (leaving the label as its own speaker) is the
    /// recoverable direction, so this gate does not need naming's margin.
    fn centroid_for_stitching(&self) -> Option<SpeakerEmbedding> {
        (self.reliable_seconds >= MIN_RELIABLE_EMBEDDING_SECONDS)
            .then(|| SpeakerEmbedding::l2_normalized(self.sum.clone()))
    }

    /// The label's mean embedding for *naming* -- attaching an enrolled
    /// person's display name.
    ///
    /// Strictly stricter than [`Self::centroid_for_stitching`], because the
    /// failure mode is strictly worse: a user who reads a name believes it.
    /// See [`MIN_NAMING_EVIDENCE_SECONDS`] for why this is measured in
    /// reliable seconds rather than in segments or in raw duration.
    fn centroid_for_naming(&self) -> Option<SpeakerEmbedding> {
        (self.reliable_seconds >= MIN_NAMING_EVIDENCE_SECONDS)
            .then(|| SpeakerEmbedding::l2_normalized(self.sum.clone()))
    }
}

/// Decide which scope-local labels are the same voice, and return the rename
/// map that collapses each such group onto one canonical label.
///
/// This is the "only from voice evidence" half of the scope contract. The
/// numbering collision is resolved by splitting before this runs
/// ([`disambiguate_labels_across_scopes`]); this is the only thing allowed to
/// put labels back together, and it has exactly two rules on top of the
/// clustering itself:
///
/// - **A label with too little audio to embed reliably is never stitched**
///   ([`LabelEvidence::centroid_for_stitching`]): a speaker who says two words
///   in one slice stays their own speaker rather than being attached to
///   whoever they happened to sound closest to. Over-counting is the
///   recoverable mistake; fusing two people is not. This is a lower bar than
///   naming, deliberately -- see [`MIN_NAMING_EVIDENCE_SECONDS`].
/// - **Two labels from the same scope are never merged.** That scope's own
///   segmenter already asserted they are different speakers, and it had the
///   full turn structure of that decode unit to say so; a centroid comparison
///   is not better evidence than that. Encoded as a cannot-link by giving every
///   label in a scope the same synthetic time range, which is precisely the
///   constraint `AgglomerativeClusterer::cluster_with_context` already applies
///   for simultaneous speech (two labels that overlap in time cannot be one
///   voice) -- same rule, reused rather than re-implemented.
///
/// The merge stop itself is the embedder's own calibrated plain AHC threshold,
/// the same one the external diarization path clusters segments with, so
/// stitching is no looser than the clustering that produced the labels.
fn stitch_labels_across_scopes(
    evidence: &BTreeMap<String, LabelEvidence>,
    clusterer: &crate::diarize::clustering::AgglomerativeClusterer,
) -> BTreeMap<String, String> {
    use crate::diarize::clustering::{ClusterContext, SpeakerClusterer};
    use crate::diarize::contract::{DiarizeHint, TimeRange};

    let mut labels: Vec<&str> = Vec::new();
    let mut centroids: Vec<SpeakerEmbedding> = Vec::new();
    let mut context: Vec<ClusterContext> = Vec::new();
    for (label, entry) in evidence {
        let Some(centroid) = entry.centroid_for_stitching() else {
            continue;
        };
        labels.push(label.as_str());
        centroids.push(centroid);
        // One unit-wide range per scope: same scope -> ranges overlap ->
        // cannot-link; different scopes -> disjoint -> free to merge.
        let start = entry.scope_index as f64;
        context.push(ClusterContext {
            range: TimeRange::new(start, start + 1.0),
            local_speaker: None,
            overlap: false,
        });
    }
    if labels.len() < 2 {
        return BTreeMap::new();
    }
    let assignments = clusterer.cluster_with_context(&centroids, &context, DiarizeHint::Auto);
    if assignments.len() != labels.len() {
        return BTreeMap::new();
    }
    // Canonical label per cluster: the first member in label order, so the
    // rename is deterministic and never invents a label that no scope produced.
    let mut canonical: BTreeMap<u32, &str> = BTreeMap::new();
    for (label, speaker) in labels.iter().zip(&assignments) {
        canonical.entry(speaker.0).or_insert(label);
    }
    labels
        .iter()
        .zip(&assignments)
        .filter_map(|(label, speaker)| {
            let target = canonical.get(&speaker.0)?;
            (target != label).then(|| ((*label).to_string(), (*target).to_string()))
        })
        .collect()
}

/// Re-pool per-label evidence onto the canonical labels [`stitch_labels_across_scopes`]
/// chose, so person matching below sees one centroid per stitched speaker
/// rather than matching each scope's fragment on its own.
fn pool_evidence_by_canonical_label(
    evidence: BTreeMap<String, LabelEvidence>,
    stitched: &BTreeMap<String, String>,
) -> BTreeMap<String, LabelEvidence> {
    let mut pooled: BTreeMap<String, LabelEvidence> = BTreeMap::new();
    for (label, entry) in evidence {
        let canonical = stitched.get(&label).cloned().unwrap_or(label);
        match pooled.get_mut(&canonical) {
            Some(existing) => existing.absorb(&entry),
            None => {
                pooled.insert(canonical, entry);
            }
        }
    }
    pooled
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

    /// One turn's worth of continuous speech, comfortably over every gate.
    const PLENTY_OF_EVIDENCE_SECONDS: f64 = 8.0;
    /// One short utterance: enough to embed, nowhere near enough to name.
    const A_SINGLE_FRAGMENT_SECONDS: f64 = 0.25;

    fn evidence_entry(scope_index: usize, values: Vec<f32>, seconds: f64) -> LabelEvidence {
        let mut entry = LabelEvidence::new(values.len(), scope_index);
        entry.accumulate(&SpeakerEmbedding::l2_normalized(values), seconds);
        entry
    }

    fn stitch(entries: &[(&str, LabelEvidence)]) -> BTreeMap<String, String> {
        let evidence: BTreeMap<String, LabelEvidence> = entries
            .iter()
            .map(|(label, entry)| {
                (
                    (*label).to_string(),
                    LabelEvidence {
                        sum: entry.sum.clone(),
                        seconds: entry.seconds,
                        reliable_seconds: entry.reliable_seconds,
                        scope_index: entry.scope_index,
                    },
                )
            })
            .collect();
        stitch_labels_across_scopes(
            &evidence,
            &crate::diarize::clustering::AgglomerativeClusterer::default(),
        )
    }

    /// The stitch side of the serve-batch contract: two scopes decoded the same
    /// voice under their own numbering, and the acoustic evidence is what puts
    /// them back together.
    #[test]
    fn scopes_are_stitched_back_together_by_matching_voices() {
        let seconds = PLENTY_OF_EVIDENCE_SECONDS;
        let stitched = stitch(&[
            (
                "SPEAKER_00",
                evidence_entry(0, vec![1.0, 0.0, 0.0], seconds),
            ),
            (
                "SPEAKER_01",
                evidence_entry(0, vec![0.0, 1.0, 0.0], seconds),
            ),
            (
                "SPEAKER_02",
                evidence_entry(1, vec![0.99, 0.1, 0.0], seconds),
            ),
        ]);
        // The second scope's speaker is the first scope's SPEAKER_00 voice.
        assert_eq!(
            stitched.get("SPEAKER_02").map(String::as_str),
            Some("SPEAKER_00")
        );
        // The two genuinely different voices are left alone.
        assert!(!stitched.contains_key("SPEAKER_00"));
        assert!(!stitched.contains_key("SPEAKER_01"));
    }

    /// Two speakers the *same* scope's segmenter told apart are never fused,
    /// even when their centroids are close enough that a plain threshold would
    /// merge them: that scope saw the whole turn structure and its verdict
    /// outranks a centroid comparison.
    #[test]
    fn labels_from_one_scope_are_never_stitched_to_each_other() {
        let seconds = PLENTY_OF_EVIDENCE_SECONDS;
        let stitched = stitch(&[
            (
                "SPEAKER_00",
                evidence_entry(0, vec![1.0, 0.0, 0.0], seconds),
            ),
            (
                "SPEAKER_01",
                evidence_entry(0, vec![1.0, 0.01, 0.0], seconds),
            ),
        ]);
        assert!(stitched.is_empty(), "{stitched:?}");
    }

    /// Voices that do not match stay separate speakers rather than being
    /// collapsed onto whichever label they were nearest to.
    #[test]
    fn different_voices_in_different_scopes_stay_separate() {
        let seconds = PLENTY_OF_EVIDENCE_SECONDS;
        let stitched = stitch(&[
            (
                "SPEAKER_00",
                evidence_entry(0, vec![1.0, 0.0, 0.0], seconds),
            ),
            (
                "SPEAKER_01",
                evidence_entry(1, vec![0.0, 1.0, 0.0], seconds),
            ),
        ]);
        assert!(stitched.is_empty(), "{stitched:?}");
    }

    /// Too little audio to embed reliably is too little audio to stitch on:
    /// such a label keeps its own scope-local identity (over-counting) rather
    /// than being attached to the nearest centroid (fusing two people).
    #[test]
    fn a_label_with_thin_evidence_is_not_stitched() {
        let stitched = stitch(&[
            (
                "SPEAKER_00",
                evidence_entry(0, vec![1.0, 0.0, 0.0], PLENTY_OF_EVIDENCE_SECONDS),
            ),
            (
                "SPEAKER_01",
                evidence_entry(1, vec![1.0, 0.0, 0.0], A_SINGLE_FRAGMENT_SECONDS),
            ),
        ]);
        assert!(stitched.is_empty(), "{stitched:?}");
    }

    /// Pooling follows the stitch so person matching sees one centroid per
    /// stitched speaker, with the audio evidence of every scope it spans.
    #[test]
    fn stitched_labels_pool_their_evidence() {
        let seconds = PLENTY_OF_EVIDENCE_SECONDS;
        let evidence: BTreeMap<String, LabelEvidence> = [
            (
                "SPEAKER_00".to_string(),
                evidence_entry(0, vec![1.0, 0.0], seconds),
            ),
            (
                "SPEAKER_01".to_string(),
                evidence_entry(1, vec![1.0, 0.0], seconds),
            ),
        ]
        .into_iter()
        .collect();
        let stitched: BTreeMap<String, String> =
            [("SPEAKER_01".to_string(), "SPEAKER_00".to_string())]
                .into_iter()
                .collect();
        let pooled = pool_evidence_by_canonical_label(evidence, &stitched);
        assert_eq!(pooled.len(), 1);
        assert_eq!(pooled["SPEAKER_00"].seconds, seconds * 2.0);
    }

    /// Evidence too short to embed reliably yields no centroid at all, so such
    /// a label can never be stitched onto another scope's label nor handed to
    /// the matcher.
    #[test]
    fn thin_evidence_is_not_worth_a_name() {
        let mut evidence = LabelEvidence::new(2, 0);
        evidence.accumulate(
            &SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]),
            A_SINGLE_FRAGMENT_SECONDS,
        );
        assert!(evidence.centroid_for_stitching().is_none());
        assert!(evidence.centroid_for_naming().is_none());

        let mut evidence = LabelEvidence::new(2, 0);
        evidence.accumulate(
            &SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]),
            PLENTY_OF_EVIDENCE_SECONDS,
        );
        assert!(evidence.centroid_for_stitching().is_some());
        assert!(evidence.centroid_for_naming().is_some());
    }

    /// The naming gate has to be able to say no, and the only way to prove
    /// that is to show it disagreeing with the segmenter that fed it.
    ///
    /// The old gate could not disagree: it asked whether a label's accumulated
    /// seconds reached the segmenter's own minimum segment length, and the
    /// segmenter guarantees every segment it emits is at least that long, so
    /// one segment always sufficed. The old rule is restated here as a local
    /// literal rather than imported, on purpose -- if it read a production
    /// constant, moving that constant would silently turn this half of the
    /// test into a tautology and the proof would evaporate.
    #[test]
    fn the_naming_gate_is_independent_of_the_segmenters_minimum_segment_length() {
        /// `pipeline::MIN_SEGMENT_S` as it stood when this test was written.
        /// Deliberately a copy: this test pins the *shape* of the old rule,
        /// not today's value of it.
        const SEGMENTER_MIN_SEGMENT_SECONDS: f64 = 0.5;
        let voice = SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]);

        // AISHELL-4 L_R003S01C02 speaker 003-F: eight fragments across ten
        // minutes, 5.18s in total, longest 0.93s. A real person, but not one
        // this recording gives enough voice to put a name to.
        let fragments = [0.88, 0.93, 0.86, 0.60, 0.60, 0.52, 0.39, 0.40];
        let mut walk_on = LabelEvidence::new(2, 0);
        for seconds in fragments {
            walk_on.accumulate(&voice, seconds);
        }
        assert!(
            walk_on.seconds >= SEGMENTER_MIN_SEGMENT_SECONDS,
            "the old gate named this speaker; if it no longer would, this test \
             has stopped proving anything"
        );
        assert!(
            walk_on.centroid_for_naming().is_none(),
            "eight sub-second fragments are not evidence for a person's name"
        );

        // The structural half: the smallest label the segmenter can possibly
        // emit already cleared the old gate, so it was incapable of rejecting
        // anything at all -- no value of that constant would have fixed it.
        let mut smallest_possible = LabelEvidence::new(2, 0);
        smallest_possible.accumulate(&voice, SEGMENTER_MIN_SEGMENT_SECONDS);
        assert!(
            smallest_possible.seconds >= SEGMENTER_MIN_SEGMENT_SECONDS,
            "the old gate read its own input back"
        );
        assert!(smallest_possible.centroid_for_naming().is_none());

        // And the gate is not simply closed: a real participant clears it.
        // AISHELL-4 L_R003S02C02 speaker 007-M, the thinnest genuine
        // participant in the evaluation corpus (37.78s over seven turns).
        let mut participant = LabelEvidence::new(2, 0);
        for seconds in [1.31, 6.04, 4.66, 1.86, 6.65, 13.85, 3.42] {
            participant.accumulate(&voice, seconds);
        }
        assert!(participant.centroid_for_naming().is_some());
    }
}
