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

use thiserror::Error;

use super::evidence::{self, JudgedWindows, MIN_PURITY_VERDICT_WINDOWS, WINDOW_SECONDS};
use crate::Segment;
use crate::diarize::contract::{SpeakerEmbedding, TimeRange};

/// This stage's one fail-closed case.
///
/// Every other gate in this module (naming evidence, purity verdict) is
/// one-sided toward anonymous per the module docs above: refusing costs a
/// name, never a wrong one, so a refusal is silent. A missing embedder is
/// different in kind. It is not a judgement this stage made about the
/// evidence; it is the evidence-gathering machinery itself being unavailable,
/// and whether that is safe to paper over depends on what the caller stood to
/// lose -- see [`name_speakers_across_scopes`] for exactly when it fires.
#[derive(Debug, Error)]
pub enum SpeakerIdentityError {
    #[error("{}", crate::diarize::embed::VOICE_ID_NAMING_EMBEDDER_MISSING_REASON)]
    EmbedderPackMissing,
}

/// Audio sample rate the speaker embedder is fed at; the transcription
/// pipeline resamples to this before decode, so segment times index directly
/// into a scope's samples at this rate.
const EMBEDDER_SAMPLE_RATE_HZ: usize = 16_000;

/// # The invariant this constant exists to enforce
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
/// particular segment granularity. Half of that is now structural rather than
/// a constant: the embedding unit is a fixed
/// [`WINDOW_SECONDS`](evidence::WINDOW_SECONDS) window, so "was this unit long
/// enough to embed stably" cannot be false, and a segment too short to hold one
/// contributes nothing by construction (see [`evidence`]). A previous
/// `MIN_RELIABLE_EMBEDDING_SECONDS = 1.0` said the same thing about segments
/// and is gone with them -- the window is twice as long and, unlike a segment,
/// is guaranteed to be.
///
/// Total duration alone is the wrong measure, and the corpus shows why: a
/// label made of twenty sub-second "mm"s and a label with one continuous turn
/// can carry the same number of seconds while being worlds apart as evidence.
/// AISHELL-4 `L_R003S01C02` speaker `003-F` is exactly that -- 5.18s spread
/// over eight fragments of 0.39-0.93s, every one of them too short for a
/// trustworthy embedding, and a plain 3s total would have named her. Under
/// windowing she yields no window at all.
///
/// # This constant
///
/// How much voice a label needs before this stage will match it against
/// enrolled people at all, measured over the audio the surviving main-cluster
/// windows actually cover -- distinct audio, since consecutive windows overlap.
///
/// Naming a real person cannot be cheaper than inventing an anonymous one, and
/// the streaming path already requires 2.5s before it will create even an
/// anonymous session speaker (`streaming::MIN_NEW_SPEAKER_DURATION_S`), so 3s
/// is the floor consistency demands. It is also where the corpus is flat and
/// the two populations are cleanly apart: over those 45 speakers the reliable
/// seconds are 0, 0, 1.08, 1.29, 1.29, 1.32, 1.43, 1.70, 1.84, 1.98 | 4.28,
/// 4.84, 5.93, 9.40, ... -- **nothing at all lands between 2.0 and 4.3**, so
/// any threshold in 2.0-4.0 rejects the same ten walk-on speakers and keeps
/// the same thirty-five real participants.
///
/// This is deliberately one-sided, per this module's "erring toward
/// anonymous": the cost of rejecting is one familiar voice left as
/// "Speaker 2", the cost of accepting is a transcript that confidently
/// attributes a stranger's words to someone the user knows.
///
/// # Its relationship to the window count gate
///
/// Naming also requires [`MIN_PURITY_VERDICT_WINDOWS`] surviving windows, and
/// the two are **nested, not competing**: both are one-sided toward anonymous,
/// so they can only ever refuse together. At today's geometry the window count
/// is the binding one -- five windows span at least 6.0s of distinct audio
/// however they are arranged, twice this floor -- and it is a statement about
/// the *purity verdict* being computable at all, not about quantity of voice.
/// This constant is the quantity statement, in the unit that stays meaningful
/// if the window geometry is ever retuned. Neither may be restated in terms of
/// the other, for the same reason the section above gives.
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
///
/// # When a missing embedder is an error, and when it is not
///
/// Without an embedder this stage cannot do either of its two jobs: it cannot
/// stitch scopes back together, and it cannot match a label to an enrolled
/// person. Whether that is safe to swallow depends on whether either job had
/// anything to do:
///
/// - **Single scope, nobody enrolled.** Neither job exists -- there is nothing
///   to stitch (one scope) and nothing to match against (empty library). This
///   is the ordinary "Voice ID is on but unused" state, and it must keep
///   succeeding: refusing it would turn plain anonymous multi-speaker
///   transcription into a hard failure over a pack that has nothing to do.
/// - **Multiple scopes, or somebody enrolled.** At least one job would have
///   run and silently did not: a longform recording's later slices would stay
///   artificially separated from its earlier ones, or an enrolled person's
///   segments would stay anonymous with no signal to the caller that
///   anything went wrong. That is the exact silent-degrade this stage exists
///   to not do, so it fails closed with [`SpeakerIdentityError::EmbedderPackMissing`]
///   instead.
pub fn name_speakers_across_scopes(
    scopes: &mut [SpeakerScope<'_>],
) -> Result<(), SpeakerIdentityError> {
    name_speakers_across_scopes_with(crate::diarize::embed::shared_embedder(), scopes)
}

/// [`name_speakers_across_scopes`] with the embedder passed explicitly.
///
/// The public entry point resolves the process-wide shared embedder; this
/// seam exists so tests exercise both sides of the missing-embedder contract
/// deterministically instead of inheriting whatever packs the host machine
/// happens to have installed.
fn name_speakers_across_scopes_with(
    embedder: Option<&'static dyn crate::diarize::embed::SpeakerEmbedder>,
    scopes: &mut [SpeakerScope<'_>],
) -> Result<(), SpeakerIdentityError> {
    for scope in scopes.iter_mut() {
        normalize_local_labels(scope.segments);
    }
    if scopes.len() > 1 {
        disambiguate_labels_across_scopes(scopes);
    }
    let Some(embedder) = embedder else {
        if scopes.len() > 1 || super::person_library_is_non_empty() {
            return Err(SpeakerIdentityError::EmbedderPackMissing);
        }
        // No embedder, nobody enrolled, one scope: separation stands, naming
        // was never going to do anything here regardless of the embedder.
        return Ok(());
    };

    // Keyed by the label as it now reads, which is scope-unique after the
    // disambiguation pass above -- so evidence from two scopes is never pooled
    // into one centroid unless a caller genuinely produced one scope.
    let mut evidence: BTreeMap<String, LabelEvidence> = BTreeMap::new();
    for (scope_index, scope) in scopes.iter().enumerate() {
        for (label, windows) in evidence::plan_label_windows(scope.segments) {
            let mut embeddings = Vec::with_capacity(windows.len());
            let mut spans = Vec::with_capacity(windows.len());
            for window in windows {
                let Some(clip) = window_clip(&window, scope.samples) else {
                    continue;
                };
                let Ok(embedding) = embedder.embed(clip, EMBEDDER_SAMPLE_RATE_HZ as u32) else {
                    continue;
                };
                embeddings.push(embedding);
                spans.push(window);
            }
            if embeddings.is_empty() {
                continue;
            }
            evidence.insert(
                label,
                LabelEvidence::from_windows(
                    evidence::judge_windows(&embeddings, &spans),
                    scope_index,
                ),
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
    Ok(())
}

/// Single-scope convenience for the one caller that has a single decode unit
/// (every offline transcription today). Same semantics as
/// [`name_speakers_across_scopes`] with one scope.
pub fn name_speakers_from_labeled_segments(
    segments: &mut [Segment],
    samples: &[f32],
) -> Result<(), SpeakerIdentityError> {
    name_speakers_across_scopes(&mut [SpeakerScope { segments, samples }])
}

/// What a label's voice amounts to: the windows that survived main-cluster
/// filtering, plus the verdict on whether they came from one person.
struct LabelEvidence {
    /// Main-cluster window embeddings. Every centroid below is their mean, and
    /// nothing else is ever averaged in -- the filtering happens once, here,
    /// and is not conditional on `single_voice`.
    kept: Vec<SpeakerEmbedding>,
    /// Distinct audio the kept windows cover.
    kept_seconds: f64,
    /// Whether the windows split into two clusters far enough apart to be two
    /// people. Naming requires this; stitching deliberately does not (see
    /// [`stitch_labels_across_scopes`]).
    single_voice: bool,
    /// Which scope this label belongs to. Two labels sharing a scope were told
    /// apart by that scope's own segmenter and must never be stitched back
    /// together (see [`stitch_labels_across_scopes`]).
    scope_index: usize,
}

impl LabelEvidence {
    fn from_windows(judged: JudgedWindows, scope_index: usize) -> Self {
        Self {
            kept: judged.kept,
            kept_seconds: judged.kept_seconds,
            single_voice: judged.single_voice,
            scope_index,
        }
    }

    /// Merge another label's evidence after stitching decided they are one
    /// person.
    ///
    /// The already-filtered windows are concatenated rather than re-judged.
    /// Re-running the purity split across scopes would be asking a different
    /// question than the one it can answer: the same person recorded in two
    /// scopes can legitimately split into two clusters by channel or distance,
    /// and stitching has already ruled on identity using the whole-label
    /// centroids. Each part keeps the verdict its own scope earned, and the
    /// merged label is single-voice only if every part was.
    fn absorb(&mut self, other: &LabelEvidence) {
        self.kept.extend(other.kept.iter().cloned());
        self.kept_seconds += other.kept_seconds;
        self.single_voice &= other.single_voice;
    }

    /// The label's mean embedding for *stitching* -- deciding that two scopes'
    /// labels are the same voice.
    ///
    /// Any surviving window is enough: stitching's failure mode is fusing two
    /// people, and the alternative to stitching (leaving the label as its own
    /// speaker) is the recoverable direction, so this gate does not need
    /// naming's margin. It notably does **not** require `single_voice` -- a
    /// label the verdict rejected still has a main-cluster centroid that is
    /// clean enough to place, and refusing to stitch it would fragment one
    /// person across scopes for a reason that only concerns naming.
    fn centroid_for_stitching(&self) -> Option<SpeakerEmbedding> {
        evidence::centroid(self.kept.iter())
    }

    /// The label's mean embedding for *naming* -- attaching an enrolled
    /// person's display name.
    ///
    /// Strictly stricter than [`Self::centroid_for_stitching`], because the
    /// failure mode is strictly worse: a user who reads a name believes it. All
    /// three conditions point the same way (see [`MIN_NAMING_EVIDENCE_SECONDS`]
    /// on why the last two are not one gate stated twice).
    fn centroid_for_naming(&self) -> Option<SpeakerEmbedding> {
        (self.single_voice
            && self.kept.len() >= MIN_PURITY_VERDICT_WINDOWS
            && self.kept_seconds >= MIN_NAMING_EVIDENCE_SECONDS)
            .then(|| evidence::centroid(self.kept.iter()))
            .flatten()
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
///   in one slice produces no window at all, so stays their own speaker rather
///   than being attached to whoever they happened to sound closest to.
///   Over-counting is the recoverable mistake; fusing two people is not. This
///   is a lower bar than naming, deliberately -- see
///   [`MIN_NAMING_EVIDENCE_SECONDS`].
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

/// The samples one window names, or `None` if the scope's audio does not
/// contain the whole window.
///
/// Whole or nothing, on purpose: the point of a fixed unit is that every
/// embedding is backed by the same amount of audio, and a truncated tail window
/// would quietly reintroduce the variable-length unit this stage exists to get
/// rid of.
fn window_clip<'a>(window: &TimeRange, samples: &'a [f32]) -> Option<&'a [f32]> {
    let rate = EMBEDDER_SAMPLE_RATE_HZ as f64;
    let start = (window.start_s.max(0.0) * rate).round() as usize;
    let length = (WINDOW_SECONDS * rate).round() as usize;
    samples.get(start..start.checked_add(length)?)
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
        name_speakers_from_labeled_segments(&mut segments, &[]).unwrap();

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
        name_speakers_from_labeled_segments(&mut segments, &[]).unwrap();
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
        name_speakers_from_labeled_segments(&mut segments, &[]).unwrap();
        assert_eq!(segments[0].speaker_label.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segments[1].speaker_label.as_deref(), Some("SPEAKER_05"));
    }

    /// The serve-batch contract: two independently decoded scopes both start
    /// numbering at one, and those two `SPEAKER_01`s are unrelated. Splitting
    /// them apart is disambiguation's job and happens unconditionally, before
    /// this stage even looks for an embedder -- so the split survives even on
    /// the fail-closed path this test runs on (no embedder pack in this test
    /// process, and multiple scopes is exactly the condition that now errors
    /// rather than silently skipping stitching; see
    /// [`SpeakerIdentityError::EmbedderPackMissing`]).
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
        let result = name_speakers_across_scopes_with(
            None,
            &mut [
                SpeakerScope {
                    segments: &mut first,
                    samples: &[],
                },
                SpeakerScope {
                    segments: &mut second,
                    samples: &[],
                },
            ],
        );
        assert!(
            matches!(result, Err(SpeakerIdentityError::EmbedderPackMissing)),
            "multi-scope naming without an embedder must fail closed, not silently skip stitching"
        );

        let label = |segment: &Segment| segment.speaker_label.clone().unwrap();
        // Within a scope, one label stays one speaker.
        assert_eq!(label(&second[0]), label(&second[1]));
        // Across scopes, colliding labels are split apart. Disambiguation ran
        // before the embedder check and its mutation is not rolled back by
        // the later error.
        assert_ne!(label(&first[0]), label(&second[0]));
        assert_ne!(label(&first[1]), label(&second[0]));
        // The rendered speaker follows the label, and no identity was invented.
        for segment in first.iter().chain(second.iter()) {
            assert_eq!(segment.speaker, segment.speaker_label);
            assert!(segment.speaker_person_id.is_none());
        }
    }

    /// The other half of the fail-closed gate: a single scope with nobody
    /// enrolled is a legitimate no-op even with no embedder, because neither
    /// of this stage's jobs (stitching, naming) had anything to do. This is
    /// the common "Voice ID on, unused" state and must keep succeeding.
    #[test]
    fn single_scope_empty_library_without_embedder_is_not_an_error() {
        let mut segments = vec![labeled(0.0, 1.0, Some("SPEAKER_01"))];
        let result = name_speakers_across_scopes_with(
            None,
            &mut [SpeakerScope {
                segments: &mut segments,
                samples: &[],
            }],
        );
        assert!(result.is_ok());
    }

    /// A talkative label: comfortably over every gate.
    const PLENTY_OF_WINDOWS: usize = 8;
    /// One short utterance: too short to hold a single window.
    const NOT_EVEN_ONE_WINDOW: usize = 0;

    fn evidence_entry(scope_index: usize, values: Vec<f32>, windows: usize) -> LabelEvidence {
        let voice = SpeakerEmbedding::l2_normalized(values);
        LabelEvidence {
            kept: vec![voice; windows],
            kept_seconds: if windows == 0 {
                0.0
            } else {
                WINDOW_SECONDS + (windows - 1) as f64 * evidence::WINDOW_STEP_SECONDS
            },
            single_voice: true,
            scope_index,
        }
    }

    fn stitch(entries: &[(&str, LabelEvidence)]) -> BTreeMap<String, String> {
        let evidence: BTreeMap<String, LabelEvidence> = entries
            .iter()
            .map(|(label, entry)| {
                (
                    (*label).to_string(),
                    LabelEvidence {
                        kept: entry.kept.clone(),
                        kept_seconds: entry.kept_seconds,
                        single_voice: entry.single_voice,
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
        let windows = PLENTY_OF_WINDOWS;
        let stitched = stitch(&[
            (
                "SPEAKER_00",
                evidence_entry(0, vec![1.0, 0.0, 0.0], windows),
            ),
            (
                "SPEAKER_01",
                evidence_entry(0, vec![0.0, 1.0, 0.0], windows),
            ),
            (
                "SPEAKER_02",
                evidence_entry(1, vec![0.99, 0.1, 0.0], windows),
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
        let windows = PLENTY_OF_WINDOWS;
        let stitched = stitch(&[
            (
                "SPEAKER_00",
                evidence_entry(0, vec![1.0, 0.0, 0.0], windows),
            ),
            (
                "SPEAKER_01",
                evidence_entry(0, vec![1.0, 0.01, 0.0], windows),
            ),
        ]);
        assert!(stitched.is_empty(), "{stitched:?}");
    }

    /// Voices that do not match stay separate speakers rather than being
    /// collapsed onto whichever label they were nearest to.
    #[test]
    fn different_voices_in_different_scopes_stay_separate() {
        let windows = PLENTY_OF_WINDOWS;
        let stitched = stitch(&[
            (
                "SPEAKER_00",
                evidence_entry(0, vec![1.0, 0.0, 0.0], windows),
            ),
            (
                "SPEAKER_01",
                evidence_entry(1, vec![0.0, 1.0, 0.0], windows),
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
                evidence_entry(0, vec![1.0, 0.0, 0.0], PLENTY_OF_WINDOWS),
            ),
            (
                "SPEAKER_01",
                evidence_entry(1, vec![1.0, 0.0, 0.0], NOT_EVEN_ONE_WINDOW),
            ),
        ]);
        assert!(stitched.is_empty(), "{stitched:?}");
    }

    /// Pooling follows the stitch so person matching sees one centroid per
    /// stitched speaker, with the audio evidence of every scope it spans.
    #[test]
    fn stitched_labels_pool_their_evidence() {
        let windows = PLENTY_OF_WINDOWS;
        let evidence: BTreeMap<String, LabelEvidence> = [
            (
                "SPEAKER_00".to_string(),
                evidence_entry(0, vec![1.0, 0.0], windows),
            ),
            (
                "SPEAKER_01".to_string(),
                evidence_entry(1, vec![1.0, 0.0], windows),
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
        assert_eq!(pooled["SPEAKER_00"].kept.len(), windows * 2);
    }

    /// A label too short to hold a single window yields no centroid at all, so
    /// it can never be stitched onto another scope's label nor handed to the
    /// matcher.
    #[test]
    fn thin_evidence_is_not_worth_a_name() {
        let thin = evidence_entry(0, vec![1.0, 0.0], NOT_EVEN_ONE_WINDOW);
        assert!(thin.centroid_for_stitching().is_none());
        assert!(thin.centroid_for_naming().is_none());

        let plenty = evidence_entry(0, vec![1.0, 0.0], PLENTY_OF_WINDOWS);
        assert!(plenty.centroid_for_stitching().is_some());
        assert!(plenty.centroid_for_naming().is_some());
    }

    /// Stitching is the recoverable direction, so it does not wait on the
    /// purity verdict: a label the verdict rejected still has a main-cluster
    /// centroid worth placing, and refusing to place it would scatter one
    /// person across scope seams for a reason that only concerns naming.
    #[test]
    fn a_label_the_purity_verdict_rejected_can_still_be_stitched() {
        let mut mixed = evidence_entry(0, vec![1.0, 0.0], PLENTY_OF_WINDOWS);
        mixed.single_voice = false;
        assert!(mixed.centroid_for_stitching().is_some());
        assert!(mixed.centroid_for_naming().is_none());
    }

    /// Both naming gates have to be able to say no on their own.
    #[test]
    fn naming_needs_enough_windows_and_enough_distinct_audio() {
        let mut short_of_windows =
            evidence_entry(0, vec![1.0, 0.0], MIN_PURITY_VERDICT_WINDOWS - 1);
        assert!(short_of_windows.centroid_for_naming().is_none());
        // Even with the seconds gate satisfied outright.
        short_of_windows.kept_seconds = 60.0;
        assert!(short_of_windows.centroid_for_naming().is_none());

        let mut short_of_audio = evidence_entry(0, vec![1.0, 0.0], MIN_PURITY_VERDICT_WINDOWS);
        assert!(short_of_audio.centroid_for_naming().is_some());
        short_of_audio.kept_seconds = MIN_NAMING_EVIDENCE_SECONDS - 0.1;
        assert!(short_of_audio.centroid_for_naming().is_none());
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
    ///
    /// Under windowing the disagreement is structural: the smallest segment the
    /// segmenter can emit is shorter than one window plus its trim, so it
    /// contributes no evidence at all no matter how many of them a label has.
    #[test]
    fn the_naming_gate_is_independent_of_the_segmenters_minimum_segment_length() {
        /// `pipeline::MIN_SEGMENT_S` as it stood when this test was written.
        /// Deliberately a copy: this test pins the *shape* of the old rule,
        /// not today's value of it.
        const SEGMENTER_MIN_SEGMENT_SECONDS: f64 = 0.5;

        fn label_windows(durations: &[f64]) -> usize {
            // Spread the turns far apart so nothing overlaps and the count is
            // purely a function of each turn's own length.
            let mut segments = Vec::new();
            let mut cursor = 0.0f32;
            for duration in durations {
                segments.push(labeled(
                    cursor,
                    cursor + *duration as f32,
                    Some("SPEAKER_00"),
                ));
                cursor += *duration as f32 + 100.0;
            }
            for segment in &mut segments {
                segment.speaker_label = segment.speaker.clone();
            }
            evidence::plan_label_windows(&segments)
                .get("SPEAKER_00")
                .map_or(0, Vec::len)
        }

        // AISHELL-4 L_R003S01C02 speaker 003-F: eight fragments across ten
        // minutes, 5.18s in total, longest 0.93s. A real person, but not one
        // this recording gives enough voice to put a name to.
        let fragments = [0.88, 0.93, 0.86, 0.60, 0.60, 0.52, 0.39, 0.40];
        assert!(
            fragments.iter().sum::<f64>() >= SEGMENTER_MIN_SEGMENT_SECONDS,
            "the old gate named this speaker; if it no longer would, this test \
             has stopped proving anything"
        );
        assert_eq!(
            label_windows(&fragments),
            0,
            "eight sub-second fragments are not evidence for a person's name"
        );

        // The structural half: the smallest label the segmenter can possibly
        // emit already cleared the old gate, so it was incapable of rejecting
        // anything at all -- no value of that constant would have fixed it.
        // It cannot clear this one, at any repetition count.
        assert_eq!(label_windows(&[SEGMENTER_MIN_SEGMENT_SECONDS]), 0);
        assert_eq!(label_windows(&[SEGMENTER_MIN_SEGMENT_SECONDS; 40]), 0);

        // And the gate is not simply closed: a real participant clears it.
        // AISHELL-4 L_R003S02C02 speaker 007-M, the thinnest genuine
        // participant in the evaluation corpus (37.78s over seven turns).
        let participant = [1.31, 6.04, 4.66, 1.86, 6.65, 13.85, 3.42];
        assert!(label_windows(&participant) >= MIN_PURITY_VERDICT_WINDOWS);
    }
}
