//! The unit naming judges from: fixed sub-windows, filtered to one voice.
//!
//! # Why a label's segments are not the unit
//!
//! A segmenter's job is to cut turns; its boundaries carry the error of that
//! job. Handing a whole segment to the embedder makes the embedding inherit
//! that error, and there is no segment-level signal that can undo it. The
//! evaluation corpus has the canonical example: in `R8003_M8001` one 30.69s
//! "turn" (312.22-342.91) is really 8002 speaking until 330.0 and 8001 after
//! it. That segment is long, fluent, and has a perfectly normal character rate
//! -- every segment-level quality signal calls it excellent, and its embedding
//! is a blend of two people. Picking better segments cannot fix a dirty unit;
//! only making the unit smaller than the error can.
//!
//! So the unit is a fixed [`WINDOW_SECONDS`] window, which is what every
//! published diarization system uses (Sell & Garcia-Romero 2014 at 1.5s;
//! Kaldi/JHU at 1.5s/0.75s; VBx at 1.44s/0.24s). Two consequences fall out for
//! free: every embedding is backed by exactly the same amount of audio, so
//! averaging them needs no weighting; and the number of embeddings per speaker
//! is capped ([`MAX_WINDOWS_PER_LABEL`]) instead of growing with the length of
//! the meeting.
//!
//! # Why the split into two clusters, and why it is used twice
//!
//! Small windows do not by themselves make a label clean -- they just move the
//! contamination from *inside* one embedding to *some of* the embeddings. An
//! offline ablation on that session measured the whole effect: the closest
//! different-speaker centroid pair goes 0.297 (whole segments) -> 0.308
//! (windows, all averaged) -> 0.554 (windows, minority discarded). Switching
//! the unit alone buys 0.011 and still leaves a wrong merge under the live 0.45
//! accept threshold; discarding the minority is what does the work, and it
//! reaches 97% of the 0.570 a ground-truth window picker achieves.
//!
//! Over the three AliMeeting sessions this stage was measured on, that is the
//! difference between wrongly merging two people and not:
//!
//! | session       | segments | windows + main cluster |
//! |---------------|----------|------------------------|
//! | `R8003_M8001` | 0.235    | 0.554                  |
//! | `R8001_M8004` | 0.271    | 0.465                  |
//! | `R8007_M8010` | 0.324    | 0.462                  |
//!
//! Every segment-unit figure is below the accept threshold (so two different
//! people read as one); every window figure is above it.
//!
//! The same cut answers a second, *different* question -- is this label one
//! person or two -- and it is essential that the two answers stay separate:
//!
//! - **The main cluster is the only source of centroid quality.**
//! - **The verdict only decides whether a name may be attached at all.**
//!
//! Collapsing these into one branch ("if mixed, do nothing") would make
//! centroid quality depend on the verdict being right, and the verdict is the
//! part that can be wrong. Both directions were measured. A label the verdict
//! *rejects* still yields a clean centroid: `R8003_M8001` `SPEAKER_01` is 66%
//! one speaker across its windows, is correctly called mixed, and its main
//! cluster is 98% one speaker -- good enough to stitch on, which is why
//! stitching does not consult the verdict. And a label the verdict *wrongly
//! accepts* is still not poisoned: `R8001_M8004` `SPEAKER_03` slice 1 is 88%
//! one speaker and passes as pure, and its main cluster is 92%. Two independent
//! defences; the weaker one failing costs a name, not a wrong name.
//!
//! # When the split itself proves there is nothing to discard
//!
//! Filtering to the main cluster is not free: on a genuinely single-voice
//! label it still throws away whatever the AHC cut calls the minority, purely
//! because the cut is unconditional. For a short label that minority can be
//! the difference between clearing [`MIN_PURITY_VERDICT_WINDOWS`] and not, so
//! discarding it when nothing was gained is a pure loss, not a caution.
//!
//! [`MIXED_MIN_SPLIT_DISTANCE`] already answers "are these two clusters far
//! enough apart to be two people" for the verdict; [`main_cluster`] asks the
//! same geometric question and, when the answer is no, keeps every window
//! instead of only the larger cluster. This does **not** collapse main-cluster
//! filtering into the verdict: it consults only the distance, never
//! [`MIXED_MIN_SECOND_CLUSTER_FRACTION`], so a label the verdict calls mixed on
//! fraction grounds alone can still have its minority reclaimed if the two
//! clusters are not actually far apart -- and conversely a label kept whole
//! here can still fail the verdict's fraction gate and go unnamed. The two
//! stay two decisions; they just now share the one measurement that is safe
//! to share.
//!
//! Whether reusing the distance this way is safe rests entirely on the two
//! populations not overlapping at 0.30, and that is measured, not assumed.
//! Across the AliMeeting acceptance sessions and the ladder recordings this
//! stage was validated on: labels confirmed single-voice by ground truth split
//! at 0.055-0.253 whenever the split is what the *verdict* actually turns on
//! (second cluster at or above [`MIXED_MIN_SECOND_CLUSTER_FRACTION`] **and**
//! at or above [`MIN_PURITY_VERDICT_WINDOWS`], since [`is_single_voice`]
//! short-circuits to single-voice below that window count regardless of
//! distance); every such label confirmed to genuinely contain two people
//! splits no closer than 0.400 (`R8007_M8010` `SPEAKER_02#0`: 3 windows,
//! 60.5% ground-truth purity, split 0.400). 0.30 sits in the gap with room on
//! both sides -- about 0.05 below the highest confirmed-single split seen and
//! 0.10 above the lowest confirmed-mixed one -- so reclaiming below it cannot
//! let a real second speaker's windows back into a centroid. That margin is
//! narrower than it looks from the verdict population alone: [`main_cluster`]
//! (unlike the verdict) has no window-count floor, so it applies the same
//! reclaim decision to small groups the verdict population above excludes --
//! it is only the specific 0.400 split landing above [`MIXED_MIN_SPLIT_DISTANCE`]
//! that keeps this one from being wrongly reclaimed today, not a structural
//! exclusion. Reclaiming buys back exactly the windows a genuinely single
//! voice was losing to an over-eager cut, at zero measured cost to the
//! stitching distances the whole scheme exists to protect: three acceptance
//! sessions' minimum different-speaker centroid distance held at 0.562 /
//! 0.440 unchanged and rose from 0.471 to 0.481 on the third.
//!
//! A single-voice label whose split happens to land *above* 0.30 -- which
//! happens; `R8001_M8004` `SPEAKER_00#0` is 95% one speaker and still splits
//! at 0.80 -- gets no reclaim and keeps losing its minority exactly as before.
//! That is intentional: the fraction gate ([`MIXED_MIN_SECOND_CLUSTER_FRACTION`])
//! is what protects a label like that from being called mixed, and reclaiming
//! on distance alone here would be reusing a signal past where it was shown
//! safe. This is why `identity::MIN_CONTINUOUS_SPEECH_SECONDS_FOR_NAMING`
//! still budgets for a minority being discarded rather than assuming reclaim
//! always fires.

use std::collections::BTreeMap;

use crate::Segment;
use crate::diarize::contract::{DiarizeHint, SpeakerEmbedding, SpeakerId, SpeakerTurn, TimeRange};

/// Length of one embedding window.
///
/// Measured flat across the plausible range on the acceptance session -- the
/// closest different-speaker pair lands at 0.549 (1.5s/0.75s), 0.554 (2.0s/1.0s),
/// 0.551 (2.0s/0.5s), 0.533 (3.0s/1.5s). 2.0s is picked for being comfortably
/// above the ~1.5s where speaker embeddings stop improving while still being
/// short enough to sit inside a single turn.
pub(super) const WINDOW_SECONDS: f64 = 2.0;

/// Hop between consecutive windows. Half the window: adjacent windows share
/// half their audio, which is what keeps a short turn from yielding a single
/// window.
pub(super) const WINDOW_STEP_SECONDS: f64 = 1.0;

/// Cut off each end of a turn before windowing.
///
/// Segment boundaries are exactly where a segmenter's error lives, and the
/// boundary that carries that error is a *speaker change*: the seconds right
/// after a turn starts and right before it ends are the most likely to
/// actually belong to whoever spoke before or after. So this is charged once
/// per turn, at the two edges [`label_turn_runs`] identifies as real, and not
/// at every sentence boundary inside a turn -- see that function for what
/// happened when it was.
///
/// 0.5s replaces a prior 0.25s. Splitting three real recordings' windows by
/// distance to the nearest turn boundary showed a quarter second was not
/// enough: turn-first windows scored 61% worse and turn-last windows 52% /
/// 23% worse (by minimum cross-speaker distance) than interior windows of the
/// same label, consistently across all three sessions and both speakers,
/// while RMS-vs-distance correlation across those same windows was near
/// zero. The damage is positional, not a matter of quiet or unusual audio
/// near a boundary -- which a purity/RMS filter cannot fix and only a wider
/// trim can.
pub(super) const SEGMENT_EDGE_TRIM_SECONDS: f64 = 0.5;

/// Hard cap on how many windows one label ever contributes.
///
/// This is a **ceiling, not a target**, and raising it makes results worse, not
/// merely slower: measured on the acceptance session the closest
/// different-speaker pair is 0.561 at 5 windows, 0.587 at 8, 0.565 at 10, 0.566
/// at 15, 0.554 at 20, then 0.540 at 30 and 0.497 at 40. Past ~20 the budget
/// starts reaching into the parts of a mislabelled group that belong to the
/// other speaker, which dilutes the majority the main cluster is selected by.
/// It is also what makes the cost of naming independent of meeting length.
pub(super) const MAX_WINDOWS_PER_LABEL: usize = 20;

/// Smallest set of windows this machinery can say anything about, used at both
/// ends of it.
///
/// Before the split, on the whole group: cutting three points into two clusters
/// always "succeeds" and says nothing about whether they came from one voice,
/// so below this size no verdict is claimed at all. After the split, on the
/// survivors: naming requires this many windows to have come through
/// main-cluster filtering (`identity::LabelEvidence::centroid_for_naming`).
/// Since the split always removes a minority, that second use is the binding
/// one and it implies a group of six or seven -- deliberately, because those
/// are the labels where a majority vote of windows is thin enough that the
/// minority could just as well have been the majority.
///
/// This is not an evidence-quantity gate; that is
/// `identity::MIN_NAMING_EVIDENCE_SECONDS`, and the two are related there.
/// Five is the smallest budget the acceptance experiment measured end to end.
pub(super) const MIN_PURITY_VERDICT_WINDOWS: usize = 5;

/// A second cluster smaller than this fraction of the group is treated as
/// stray windows rather than a second person, whatever its distance.
///
/// This is the **false-positive** guard: it exists to stop a handful of outlying
/// windows -- one distant utterance, a cough, a channel change -- from
/// condemning a label that really is one person. It is not the discriminator;
/// the distance below is. Measured over the three AliMeeting sessions this pair
/// was calibrated on (19 scope-local labels, 18 with ground-truth coverage), it
/// is exactly where the false positives stop: 0.15 and above gives none, 0.10
/// gives three and 0.05 gives four, all of them genuinely single-speaker labels
/// whose windows happen to split by acoustic condition
/// (`R8001_M8004` `SPEAKER_00`, 93% one voice, splits at distance 0.795).
const MIXED_MIN_SECOND_CLUSTER_FRACTION: f64 = 0.15;

/// How far apart the two clusters must be before the label is called two people.
///
/// The actual discriminator, and the one place where being *too* strict is the
/// expensive mistake: over the same three sessions this threshold costs one
/// missed mixed label at 0.25-0.35 and four at 0.40, because a label built from
/// many short fragments of two people splits at only 0.38-0.39 (`R8007_M8010`
/// `SPEAKER_02`, 60% one voice, split 0.391). 0.30 sits mid-plateau of the flat
/// 0.25-0.35 region and reproduces the offline calibration.
///
/// The single remaining miss is `R8001_M8004` `SPEAKER_03` slice 1 (88% one
/// voice, split 0.224) -- which has four windows and so is refused a name by
/// [`MIN_PURITY_VERDICT_WINDOWS`] anyway, and whose main cluster is 92% one
/// speaker regardless. That is the two-defence structure working as intended.
///
/// [`main_cluster`] also reads this constant, for a related but distinct
/// question: not "is this label mixed" (that also needs
/// [`MIXED_MIN_SECOND_CLUSTER_FRACTION`]) but "does the split itself prove the
/// second cluster is not a different person". Both questions are safe to
/// answer off the same number because the same measurement backs both: see
/// the module docs for why a 0.30 cut leaves room on both sides of the
/// confirmed-single and confirmed-mixed populations. Any retuning of this
/// constant moves both consumers at once, which is intended -- a value that
/// stops being a safe mixed/single cutoff also stops being a safe reclaim
/// cutoff, for the same reason.
const MIXED_MIN_SPLIT_DISTANCE: f32 = 0.30;

/// A label's windows, judged.
pub(super) struct JudgedWindows {
    /// Main-cluster windows, in window order. The only embeddings any centroid
    /// is ever built from -- see the module docs on the two defences.
    pub(super) kept: Vec<SpeakerEmbedding>,
    /// Distinct audio the kept windows cover, in seconds. Windows overlap, so
    /// this is the union of their spans and not `kept.len() * WINDOW_SECONDS`.
    pub(super) kept_seconds: f64,
    /// Whether the group passed as a single voice. `false` means "do not put a
    /// name on this label"; it does **not** mean the kept windows are unusable.
    pub(super) single_voice: bool,
}

/// Windows to embed for each label of one scope, in scan order.
///
/// Windows are taken only where the scope's segmentation says exactly one
/// speaker is active: a window overlapping another label's segment is
/// contaminated by construction and no downstream filter should have to deal
/// with it. Each label stops at [`MAX_WINDOWS_PER_LABEL`], so this also bounds
/// the embedding work per scope.
///
/// They are planned over *turns* ([`label_turn_runs`]), not over the segments
/// as handed in, so how finely the transcript happens to be cut cannot change
/// how much evidence a voice is credited with.
pub(super) fn plan_label_windows(segments: &[Segment]) -> BTreeMap<String, Vec<TimeRange>> {
    let spans = segments
        .iter()
        .filter_map(|segment| {
            let label = segment.speaker_label.clone()?;
            let range = TimeRange::new(
                f64::from(segment.start).max(0.0),
                f64::from(segment.end)
                    .max(f64::from(segment.start))
                    .max(0.0),
            );
            Some(LabeledSpan {
                label,
                range,
                evidence_eligible: true,
            })
        })
        .collect();
    plan_windows(spans)
}

/// Fixed-window Voice ID evidence planned directly from the canonical speaker
/// timeline. Overlap-marked turns remain in the contamination mask but never
/// seed windows of their own.
pub(super) fn plan_timeline_windows(turns: &[SpeakerTurn]) -> BTreeMap<SpeakerId, Vec<TimeRange>> {
    plan_windows(
        turns
            .iter()
            .map(|turn| LabeledSpan {
                label: turn.speaker,
                range: turn.range,
                evidence_eligible: !turn.overlap,
            })
            .collect(),
    )
}

#[derive(Clone)]
struct LabeledSpan<K> {
    label: K,
    range: TimeRange,
    evidence_eligible: bool,
}

fn plan_windows<K: Clone + Ord>(mut ordered: Vec<LabeledSpan<K>>) -> BTreeMap<K, Vec<TimeRange>> {
    ordered.sort_by(|a, b| a.range.start_s.total_cmp(&b.range.start_s));

    let mut planned: BTreeMap<K, Vec<TimeRange>> = BTreeMap::new();
    for (label, range) in label_turn_runs(&ordered) {
        let windows = planned.entry(label.clone()).or_default();
        if windows.len() >= MAX_WINDOWS_PER_LABEL {
            continue;
        }
        let first = range.start_s + SEGMENT_EDGE_TRIM_SECONDS;
        let last = range.end_s - SEGMENT_EDGE_TRIM_SECONDS;
        let mut start = first;
        while start + WINDOW_SECONDS <= last + f64::EPSILON {
            let window = TimeRange::new(start, start + WINDOW_SECONDS);
            if !overlaps_another_label(&window, &label, &ordered) {
                windows.push(window);
                if windows.len() >= MAX_WINDOWS_PER_LABEL {
                    break;
                }
            }
            start += WINDOW_STEP_SECONDS;
        }
    }
    planned.retain(|_, windows| !windows.is_empty());
    planned
}

/// Each label's segments coalesced into the turns they came from, in start
/// order.
///
/// # Why the segments cannot be windowed as given
///
/// [`SEGMENT_EDGE_TRIM_SECONDS`] is a *turn*-edge trim: it exists because the
/// seconds right after a speaker starts and right before they stop are the
/// ones most likely to belong to whoever spoke around them. It used to be
/// applied per segment on the assumption -- stated in that constant's own docs
/// -- that a segment already is one turn. That assumption does not survive the
/// pipeline: subtitle-grade cue re-segmentation
/// (`api::backend::cue_segmentation`) runs before this stage and cuts one turn
/// into several sentence-sized segments, none of whose interior boundaries is
/// a speaker change.
///
/// Charging the trim at every one of those boundaries is not a small loss. It
/// costs a full second per cue *and* silently discards every cue shorter than
/// `2 * SEGMENT_EDGE_TRIM_SECONDS + WINDOW_SECONDS` = 3.0s, which is most of
/// them. Measured on a 14s single-speaker recording whose 12.9s turn arrived
/// as four cues, the label yielded 3 windows instead of 10 -- under
/// [`MIN_PURITY_VERDICT_WINDOWS`] -- so an enrolled speaker that scores 0.81
/// against their own prototype was never even compared to it. The evidence
/// gates are supposed to measure how much voice backs a label; per-segment
/// windowing made them measure how finely the transcript was cut instead.
///
/// # What counts as one turn
///
/// Consecutive same-label segments separated by no more than
/// [`SEGMENT_EDGE_TRIM_SECONDS`]. A pause that short is within-turn breathing
/// -- shorter than the audio the trim was already prepared to throw away at
/// each side of the boundary -- so bridging it admits no more untrusted audio
/// than the previous behaviour did. Anything longer ends the run and both
/// sides get trimmed as the genuine turn edges they are.
///
/// Bridging a gap cannot pull in another speaker: a window that reaches across
/// a segment where a different label is active is still rejected by
/// [`overlaps_another_label`] below, exactly as before.
fn label_turn_runs<K: Clone + Ord>(ordered: &[LabeledSpan<K>]) -> Vec<(K, TimeRange)> {
    let mut open: BTreeMap<K, usize> = BTreeMap::new();
    let mut runs: Vec<(K, TimeRange)> = Vec::new();
    for span in ordered.iter().filter(|span| span.evidence_eligible) {
        match open.get(&span.label).copied() {
            Some(index)
                if span.range.start_s - runs[index].1.end_s <= SEGMENT_EDGE_TRIM_SECONDS =>
            {
                let end = runs[index].1.end_s.max(span.range.end_s);
                runs[index].1 = TimeRange::new(runs[index].1.start_s, end);
            }
            _ => {
                open.insert(span.label.clone(), runs.len());
                runs.push((span.label.clone(), span.range));
            }
        }
    }
    runs
}

fn overlaps_another_label<K: Eq>(
    window: &TimeRange,
    label: &K,
    ordered: &[LabeledSpan<K>],
) -> bool {
    ordered
        .iter()
        .any(|span| span.label != *label && span.range.overlaps(window))
}

/// Split a label's window embeddings into the majority voice and a verdict on
/// whether a second voice is present.
///
/// The split is average-linkage AHC cut at exactly two clusters -- the same
/// clusterer the rest of diarization uses, asked for a fixed count so the
/// answer does not depend on any threshold. Being a clustering rather than a
/// running mean is the point: a label whose windows alternate between two
/// speakers converges to a perfectly stable blend under averaging, and only
/// looking at the pairwise structure can see that it is a blend.
pub(super) fn judge_windows(embeddings: &[SpeakerEmbedding], spans: &[TimeRange]) -> JudgedWindows {
    // Computed once and handed to both consumers below: main-cluster filtering
    // and the mixed-voice verdict ask two different questions of the same cut,
    // and there is no reason to make the clusterer answer them separately.
    let split = split_in_two(embeddings);
    let members = main_cluster(embeddings, split.as_ref());
    let single_voice = is_single_voice(embeddings, split.as_ref());
    let kept: Vec<SpeakerEmbedding> = members.iter().map(|&i| embeddings[i].clone()).collect();
    let kept_seconds = union_seconds(members.iter().filter_map(|&i| spans.get(i).copied()));
    JudgedWindows {
        kept,
        kept_seconds,
        single_voice,
    }
}

/// Indices of the windows main-cluster filtering keeps, in window order.
///
/// Ordinarily the larger of the two clusters the group splits into. Two
/// situations keep every window instead: a group too small to split at all
/// (`split` is `None`), and a group whose two clusters sit at or under
/// [`MIXED_MIN_SPLIT_DISTANCE`] apart -- the clustering has itself proven the
/// second cluster is not a different person, so discarding it would be a pure
/// loss (see the module docs for why that distance-only check is safe to
/// reuse here, and why it deliberately does not also consult
/// [`MIXED_MIN_SECOND_CLUSTER_FRACTION`]).
///
/// `split` is the label's [`split_in_two`] result, computed once by
/// [`judge_windows`] and shared with [`is_single_voice`].
fn main_cluster(embeddings: &[SpeakerEmbedding], split: Option<&TwoClusterSplit>) -> Vec<usize> {
    let Some(split) = split else {
        return (0..embeddings.len()).collect();
    };
    if split
        .distance
        .is_some_and(|distance| distance <= MIXED_MIN_SPLIT_DISTANCE)
    {
        return (0..embeddings.len()).collect();
    }
    split.main.clone()
}

/// Whether a group of windows passed as a single voice. See [`main_cluster`]
/// for the `split` parameter.
fn is_single_voice(embeddings: &[SpeakerEmbedding], split: Option<&TwoClusterSplit>) -> bool {
    if embeddings.len() < MIN_PURITY_VERDICT_WINDOWS {
        // Too few windows for the split to carry information. Reporting "single
        // voice" here is safe only because naming separately requires more
        // evidence than this; the verdict on its own never grants a name.
        return true;
    }
    let Some(split) = split else {
        return true;
    };
    let second_fraction = split.second.len() as f64 / embeddings.len() as f64;
    if second_fraction < MIXED_MIN_SECOND_CLUSTER_FRACTION {
        return true;
    }
    let Some(distance) = split.distance else {
        return true;
    };
    distance <= MIXED_MIN_SPLIT_DISTANCE
}

/// A label's windows cut into exactly two clusters, plus the one distance
/// [`main_cluster`] and [`is_single_voice`] both read.
struct TwoClusterSplit {
    /// Larger cluster's member indices, in window order.
    main: Vec<usize>,
    /// Smaller cluster's member indices, in window order.
    second: Vec<usize>,
    /// Cosine distance between the two clusters' centroids. `None` only when
    /// [`centroid`] returns `None` for one of the two clusters, which in
    /// practice means that cluster's *first* embedding has dimension 0:
    /// `centroid` seeds its running sum from the first embedding's dimension
    /// and only `continue`s (still returning `Some`) on a later embedding
    /// whose dimension does not match, so an ordinary dimension mismatch
    /// never produces `None` here. Not expected in practice. The two
    /// consumers do not treat it uniformly as "not proven safe": in
    /// [`main_cluster`], `distance.is_some_and(...)` is `false` on `None`, so
    /// it falls through to discarding the minority cluster -- the
    /// conservative reading. In [`is_single_voice`], `let Some(distance) =
    /// split.distance else { return true; }` reports single-voice on `None`
    /// -- the permissive reading, same as a distance under the gate.
    distance: Option<f32>,
}

/// Cut a label's windows into exactly two clusters and measure how far apart
/// they are. `None` when the group cannot be cut in two at all (fewer than
/// two windows, or the clusterer could not separate them).
fn split_in_two(embeddings: &[SpeakerEmbedding]) -> Option<TwoClusterSplit> {
    use crate::diarize::clustering::{AgglomerativeClusterer, SpeakerClusterer};

    if embeddings.len() < 2 {
        return None;
    }
    let assignments =
        AgglomerativeClusterer::default().cluster(embeddings, DiarizeHint::NumSpeakers(2));
    if assignments.len() != embeddings.len() {
        return None;
    }
    let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (index, speaker) in assignments.iter().enumerate() {
        groups.entry(speaker.0).or_default().push(index);
    }
    let mut groups: Vec<Vec<usize>> = groups.into_values().collect();
    if groups.len() < 2 {
        return None;
    }
    // Largest first; ties broken by the earlier window, so the answer never
    // depends on map iteration order.
    groups.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a[0].cmp(&b[0])));
    let mut groups = groups.into_iter();
    let main = groups.next()?;
    let second = groups.next()?;
    let distance = match (
        centroid(main.iter().map(|&i| &embeddings[i])),
        centroid(second.iter().map(|&i| &embeddings[i])),
    ) {
        (Some(main_centroid), Some(second_centroid)) => {
            Some(1.0 - main_centroid.cosine(&second_centroid))
        }
        _ => None,
    };
    Some(TwoClusterSplit {
        main,
        second,
        distance,
    })
}

/// Equal-weight mean of L2-normalized embeddings, re-normalized.
///
/// Every input is already unit length (the embedder normalizes), so this is a
/// plain mean of directions -- the standard recipe, and the reason no window
/// gets implicitly up-weighted for having a larger raw norm.
pub(super) fn centroid<'a>(
    embeddings: impl IntoIterator<Item = &'a SpeakerEmbedding>,
) -> Option<SpeakerEmbedding> {
    let mut sum: Vec<f32> = Vec::new();
    for embedding in embeddings {
        if sum.is_empty() {
            sum = vec![0.0; embedding.dim()];
        }
        if sum.len() != embedding.dim() {
            continue;
        }
        for (slot, value) in sum.iter_mut().zip(&embedding.0) {
            *slot += value;
        }
    }
    (!sum.is_empty()).then(|| SpeakerEmbedding::l2_normalized(sum))
}

/// Total length of the union of `spans`. Adjacent windows overlap by design, so
/// summing their durations would count the shared audio twice.
fn union_seconds(spans: impl IntoIterator<Item = TimeRange>) -> f64 {
    let mut spans: Vec<TimeRange> = spans.into_iter().collect();
    spans.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
    let mut total = 0.0;
    let mut covered_to = f64::NEG_INFINITY;
    for span in spans {
        let start = span.start_s.max(covered_to);
        if span.end_s > start {
            total += span.end_s - start;
            covered_to = span.end_s;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(start: f32, end: f32, label: &str) -> Segment {
        Segment {
            start,
            end,
            text: String::new(),
            speaker: Some(label.to_string()),
            speaker_label: Some(label.to_string()),
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words: Vec::new(),
        }
    }

    fn spans(count: usize) -> Vec<TimeRange> {
        (0..count)
            .map(|i| {
                let start = i as f64 * WINDOW_STEP_SECONDS;
                TimeRange::new(start, start + WINDOW_SECONDS)
            })
            .collect()
    }

    /// A voice, as a unit vector a small angle off the first axis.
    fn voice(angle: f32) -> SpeakerEmbedding {
        SpeakerEmbedding::l2_normalized(vec![angle.cos(), angle.sin(), 0.0])
    }

    #[test]
    fn windows_are_fixed_length_and_step_by_half() {
        let planned = plan_label_windows(&[segment(0.0, 5.5, "SPEAKER_00")]);
        let windows = &planned["SPEAKER_00"];
        // A lone segment is a turn of one, trimmed on both ends: 0.5..5.0,
        // which fits windows at 0.5, 1.5, 2.5.
        assert_eq!(windows.len(), 3);
        assert!((windows[0].start_s - 0.5).abs() < 1e-9);
        for window in windows {
            assert!((window.duration_s() - WINDOW_SECONDS).abs() < 1e-9);
        }
        assert!((windows[1].start_s - windows[0].start_s - WINDOW_STEP_SECONDS).abs() < 1e-9);
    }

    /// The trim is not decoration: a segment only 2.9s long has no window at
    /// all once its edges -- where the segmenter's error lives -- are dropped.
    #[test]
    fn a_segment_shorter_than_a_trimmed_window_yields_nothing() {
        assert!(plan_label_windows(&[segment(0.0, 2.9, "SPEAKER_00")]).is_empty());
        assert_eq!(
            plan_label_windows(&[segment(0.0, 3.0, "SPEAKER_00")])["SPEAKER_00"].len(),
            1
        );
    }

    /// The regression this whole turn-run business exists for: how finely a
    /// transcript is cut must not change how much evidence a voice has.
    ///
    /// One person talking continuously for 12s arrives as four sentence cues
    /// once `cue_segmentation` has run. Windowed per cue that label yields one
    /// window and is refused a name; windowed per turn it yields the ten
    /// windows the same audio always deserved.
    #[test]
    fn a_turn_delivered_as_cues_keeps_the_whole_turns_windows() {
        let whole_turn = plan_label_windows(&[segment(0.0, 12.0, "SPEAKER_00")]);
        let cut_into_cues = plan_label_windows(&[
            segment(0.0, 2.7, "SPEAKER_00"),
            segment(3.0, 5.7, "SPEAKER_00"),
            segment(6.0, 8.7, "SPEAKER_00"),
            segment(9.0, 12.0, "SPEAKER_00"),
        ]);
        assert_eq!(whole_turn["SPEAKER_00"].len(), 10);
        assert_eq!(cut_into_cues["SPEAKER_00"], whole_turn["SPEAKER_00"]);
        assert!(cut_into_cues["SPEAKER_00"].len() >= MIN_PURITY_VERDICT_WINDOWS);
    }

    /// The other side of that rule: a real pause ends the turn, so both of its
    /// edges are trimmed and no window is ever laid over the silence.
    #[test]
    fn a_pause_longer_than_the_trim_ends_the_turn() {
        let planned = plan_label_windows(&[
            segment(0.0, 3.0, "SPEAKER_00"),
            segment(3.6, 6.6, "SPEAKER_00"),
        ]);
        let windows = &planned["SPEAKER_00"];
        assert_eq!(windows.len(), 2);
        for window in windows {
            assert!(
                window.end_s <= 3.0 || window.start_s >= 3.6,
                "window {window:?} was laid over the pause"
            );
        }
    }

    /// Bridging a short gap must not be a way around the overlap rule: whoever
    /// speaks inside the gap still blocks every window that reaches across it.
    #[test]
    fn another_label_inside_a_bridged_gap_still_blocks_the_crossing_window() {
        let planned = plan_label_windows(&[
            segment(0.0, 6.0, "SPEAKER_00"),
            segment(6.1, 6.4, "SPEAKER_01"),
            segment(6.5, 12.0, "SPEAKER_00"),
        ]);
        for window in &planned["SPEAKER_00"] {
            assert!(
                window.end_s <= 6.1 || window.start_s >= 6.4,
                "window {window:?} crosses the interjection"
            );
        }
    }

    /// Where two labels are simultaneously active neither one gets a window:
    /// the audio there is a mixture no embedding can be attributed to either.
    #[test]
    fn no_window_is_taken_where_two_labels_overlap() {
        let planned = plan_label_windows(&[
            segment(0.0, 10.0, "SPEAKER_00"),
            segment(3.0, 6.0, "SPEAKER_01"),
        ]);
        for window in &planned["SPEAKER_00"] {
            assert!(
                window.end_s <= 3.0 || window.start_s >= 6.0,
                "window {window:?} crosses the overlapped region"
            );
        }
        // And the shorter overlapping label gets nothing at all here.
        assert!(!planned.contains_key("SPEAKER_01"));
    }

    /// The budget is a hard ceiling per label, so embedding cost does not grow
    /// with how long somebody talks.
    #[test]
    fn the_window_budget_caps_a_talkative_label() {
        let planned = plan_label_windows(&[segment(0.0, 600.0, "SPEAKER_00")]);
        assert_eq!(planned["SPEAKER_00"].len(), MAX_WINDOWS_PER_LABEL);
    }

    /// Two voices in even alternation: the very case a running-mean convergence
    /// test calls "converged". The split has to see it.
    #[test]
    fn an_alternating_two_voice_group_is_not_single_voice() {
        let embeddings: Vec<SpeakerEmbedding> = (0..10)
            .map(|i| if i % 2 == 0 { voice(0.0) } else { voice(1.2) })
            .collect();
        let judged = judge_windows(&embeddings, &spans(10));
        assert!(!judged.single_voice);
    }

    /// One voice with ordinary within-speaker variation stays one voice.
    ///
    /// The split still runs and still drops a minority -- it is asked for two
    /// clusters unconditionally, so it always produces two -- but on one voice
    /// the two are near-identical, which is exactly what the verdict reads.
    /// Discarding part of a genuinely pure group costs nothing: the survivors
    /// are the same voice, and the budget is a ceiling rather than a target.
    #[test]
    fn a_single_voice_group_passes() {
        let embeddings: Vec<SpeakerEmbedding> = (0..10).map(|i| voice(i as f32 * 0.02)).collect();
        let judged = judge_windows(&embeddings, &spans(10));
        assert!(judged.single_voice);
        let centroid = centroid(judged.kept.iter()).unwrap();
        assert!(centroid.cosine(&voice(0.09)) > 0.99);
    }

    /// The other gate, on its own: a group that splits evenly but into two
    /// clusters that are close together is one speaker moving around, not two
    /// speakers. Only the distance can tell those apart -- the size of the
    /// second cluster says nothing here.
    #[test]
    fn a_large_but_close_second_cluster_is_still_one_voice() {
        // 1 - cos(0.64) = 0.198, comfortably inside one speaker's own spread.
        let embeddings: Vec<SpeakerEmbedding> = (0..10)
            .map(|i| if i % 2 == 0 { voice(0.0) } else { voice(0.64) })
            .collect();
        let judged = judge_windows(&embeddings, &spans(10));
        assert!(judged.single_voice);
    }

    /// A single stray window must not condemn a label; that is what the second
    /// gate is for, and it is deliberately the weaker of the two.
    #[test]
    fn one_outlying_window_does_not_make_a_label_mixed() {
        let mut embeddings: Vec<SpeakerEmbedding> = (0..10).map(|_| voice(0.0)).collect();
        embeddings[7] = voice(1.2);
        let judged = judge_windows(&embeddings, &spans(10));
        assert!(
            judged.single_voice,
            "1 of 10 is below the second-cluster floor"
        );
        // ...but it is still dropped from the centroid, because main-cluster
        // filtering does not consult the verdict.
        assert_eq!(judged.kept.len(), 9);
    }

    /// The load-bearing invariant of the two defences: when the verdict is
    /// wrong, the centroid is still clean. Four windows of a second speaker out
    /// of ten is under the fraction floor, so this group is called single-voice
    /// -- and every one of those four is still excluded from the centroid.
    #[test]
    fn a_wrong_purity_verdict_still_leaves_a_clean_centroid() {
        let intruder = voice(1.2);
        let mut embeddings: Vec<SpeakerEmbedding> = (0..30).map(|_| voice(0.0)).collect();
        for slot in embeddings.iter_mut().take(4) {
            *slot = intruder.clone();
        }
        let judged = judge_windows(&embeddings, &spans(30));
        assert!(
            judged.single_voice,
            "4 of 30 is below the second-cluster floor, so the verdict misses it"
        );
        assert_eq!(judged.kept.len(), 26);
        let centroid = centroid(judged.kept.iter()).unwrap();
        assert!(
            centroid.cosine(&intruder) < 0.5,
            "the intruder leaked into the centroid"
        );
        assert!(centroid.cosine(&voice(0.0)) > 0.99);
    }

    /// A group sitting exactly on the verdict floor, all of it one tight voice:
    /// the split still runs, but its own distance proves the minority is not a
    /// second person, so nothing is thrown away. This is the reclaim this
    /// stage exists for -- before it existed, a genuinely single voice this
    /// short always lost a window here and could not clear
    /// [`MIN_PURITY_VERDICT_WINDOWS`] regardless of how much more it spoke.
    #[test]
    fn a_group_at_the_verdict_floor_keeps_every_window_when_the_split_proves_it_single() {
        let embeddings: Vec<SpeakerEmbedding> = (0..MIN_PURITY_VERDICT_WINDOWS)
            .map(|i| voice(i as f32 * 0.02))
            .collect();
        let judged = judge_windows(&embeddings, &spans(MIN_PURITY_VERDICT_WINDOWS));
        assert!(judged.single_voice);
        assert_eq!(
            judged.kept.len(),
            MIN_PURITY_VERDICT_WINDOWS,
            "the split distance here is far under the reclaim gate, so nothing should be discarded"
        );
    }

    /// The other side of the same gate: a split whose distance sits above
    /// [`MIXED_MIN_SPLIT_DISTANCE`] still loses its minority exactly as before
    /// V3, even though the fraction floor calls the group single-voice. This
    /// pins the reclaim as conditional, not unconditional -- V1's mistake.
    #[test]
    fn main_cluster_still_discards_the_minority_when_the_split_is_far_apart() {
        let mut embeddings: Vec<SpeakerEmbedding> = (0..10).map(|_| voice(0.0)).collect();
        embeddings[9] = voice(1.2); // 1 - cos(1.2) = 0.638, comfortably over 0.30
        let judged = judge_windows(&embeddings, &spans(10));
        assert!(
            judged.single_voice,
            "1 of 10 is below the second-cluster fraction floor"
        );
        assert_eq!(
            judged.kept.len(),
            9,
            "the split lands well above the reclaim gate, so the outlier must still be dropped"
        );
    }

    /// The reclaim gate is inclusive: exactly at [`MIXED_MIN_SPLIT_DISTANCE`]
    /// the minority comes back, same as strictly under it. `1 - cos(angle) =
    /// 0.30` at `angle = acos(0.70)`.
    #[test]
    fn main_cluster_reclaims_a_split_exactly_at_the_gate() {
        let angle = 0.7_f32.acos();
        let mut embeddings: Vec<SpeakerEmbedding> = (0..8).map(|_| voice(0.0)).collect();
        embeddings[6] = voice(angle);
        embeddings[7] = voice(angle);
        let judged = judge_windows(&embeddings, &spans(8));
        assert_eq!(
            judged.kept.len(),
            8,
            "a split distance of exactly 0.30 must still be reclaimed (<=, not <)"
        );
    }

    /// The efficiency change this stage also makes: `main_cluster` and
    /// `is_single_voice` used to each cut the same windows into two clusters
    /// independently. They now share one [`split_in_two`] call. Because the
    /// clusterer is deterministic, calling it a second time on the same
    /// embeddings must reproduce the exact same groups and distance --
    /// otherwise sharing one call instead of two could have silently changed
    /// what either consumer saw.
    #[test]
    fn split_in_two_is_deterministic_so_sharing_one_call_changes_nothing() {
        let embeddings: Vec<SpeakerEmbedding> = (0..10)
            .map(|i| if i % 2 == 0 { voice(0.0) } else { voice(1.2) })
            .collect();
        let first = split_in_two(&embeddings).expect("splits into two clusters");
        let second = split_in_two(&embeddings).expect("splits into two clusters");
        assert_eq!(first.main, second.main);
        assert_eq!(first.second, second.second);
        assert_eq!(first.distance, second.distance);

        // And the two independent consumers, each fed that one shared split,
        // must land on the answers the old two-call version was measured to
        // give for this exact fixture (see
        // `an_alternating_two_voice_group_is_not_single_voice`): mixed, main
        // cluster only.
        let judged = judge_windows(&embeddings, &spans(10));
        assert!(!judged.single_voice);
        assert_eq!(judged.kept.len(), first.main.len());
    }

    /// Below the group size where a two-way split means anything, no verdict is
    /// claimed -- naming is refused by evidence quantity instead.
    #[test]
    fn a_group_too_small_to_split_is_not_judged() {
        let embeddings = vec![voice(0.0), voice(1.2)];
        let judged = judge_windows(&embeddings, &spans(2));
        assert!(judged.single_voice);
    }

    /// The averaging order that matters: normalize, then mean, then normalize.
    /// Two equally-trusted windows must land exactly halfway apart no matter
    /// what raw magnitude the embedder happened to produce -- averaging raw
    /// vectors would silently weight windows by their norm.
    #[test]
    fn centroids_average_directions_not_magnitudes() {
        let first = SpeakerEmbedding::l2_normalized(vec![1.0, 0.0, 0.0]);
        let second = SpeakerEmbedding::l2_normalized(vec![0.0, 1.0, 0.0]);
        let mean = centroid([&first, &second]).unwrap();
        assert!((mean.cosine(&first) - mean.cosine(&second)).abs() < 1e-6);
        assert!(
            (mean.cosine(&mean) - 1.0).abs() < 1e-6,
            "result is renormalized"
        );

        // The same two directions, one of them handed over with a large raw
        // magnitude: because the embedder normalizes before this code sees it,
        // the halfway answer is unchanged.
        let loud = SpeakerEmbedding::l2_normalized(vec![100.0, 0.0, 0.0]);
        let mean_from_loud = centroid([&loud, &second]).unwrap();
        assert!((mean_from_loud.cosine(&first) - mean.cosine(&first)).abs() < 1e-6);
    }

    /// Overlapping windows cover less distinct audio than their durations sum
    /// to, and the evidence gate must not be fooled by the double counting.
    #[test]
    fn kept_seconds_counts_distinct_audio_once() {
        // Five 2s windows at a 1s hop span 0..6, not 10 seconds.
        assert!((union_seconds(spans(5)) - 6.0).abs() < 1e-9);
        // Disjoint windows do add up, and order does not matter.
        let disjoint = vec![
            TimeRange::new(10.0, 12.0),
            TimeRange::new(0.0, 2.0),
            TimeRange::new(1.0, 3.0),
        ];
        assert!((union_seconds(disjoint) - 5.0).abs() < 1e-9);
    }
}
