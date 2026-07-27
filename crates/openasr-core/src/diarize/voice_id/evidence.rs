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
//! - **The main cluster is the only source of centroid quality.** It is applied
//!   unconditionally, whatever the verdict says.
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

use std::collections::BTreeMap;

use crate::Segment;
use crate::diarize::contract::{DiarizeHint, SpeakerEmbedding, TimeRange};

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

/// Cut off each end of a segment before windowing.
///
/// Segment boundaries are exactly where a segmenter's error lives; in this
/// pipeline a segment already *is* one speaker turn (`plan_label_windows`
/// takes each label's segments as given, one per turn), so this is a
/// turn-edge trim in effect and the seconds right after a turn starts and
/// right before it ends are the most likely to actually belong to whoever
/// spoke before or after.
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
pub(super) fn plan_label_windows(segments: &[Segment]) -> BTreeMap<String, Vec<TimeRange>> {
    let mut ordered: Vec<(&str, TimeRange)> = segments
        .iter()
        .filter_map(|segment| {
            let label = segment.speaker_label.as_deref()?;
            let range = TimeRange::new(
                f64::from(segment.start).max(0.0),
                f64::from(segment.end)
                    .max(f64::from(segment.start))
                    .max(0.0),
            );
            Some((label, range))
        })
        .collect();
    ordered.sort_by(|a, b| a.1.start_s.total_cmp(&b.1.start_s));

    let mut planned: BTreeMap<String, Vec<TimeRange>> = BTreeMap::new();
    for (label, range) in &ordered {
        let windows = planned.entry((*label).to_string()).or_default();
        if windows.len() >= MAX_WINDOWS_PER_LABEL {
            continue;
        }
        let first = range.start_s + SEGMENT_EDGE_TRIM_SECONDS;
        let last = range.end_s - SEGMENT_EDGE_TRIM_SECONDS;
        let mut start = first;
        while start + WINDOW_SECONDS <= last + f64::EPSILON {
            let window = TimeRange::new(start, start + WINDOW_SECONDS);
            if !overlaps_another_label(&window, label, &ordered) {
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

fn overlaps_another_label(window: &TimeRange, label: &str, ordered: &[(&str, TimeRange)]) -> bool {
    ordered
        .iter()
        .any(|(other, range)| *other != label && range.overlaps(window))
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
    let members = main_cluster(embeddings);
    let single_voice = is_single_voice(embeddings);
    let kept: Vec<SpeakerEmbedding> = members.iter().map(|&i| embeddings[i].clone()).collect();
    let kept_seconds = union_seconds(members.iter().filter_map(|&i| spans.get(i).copied()));
    JudgedWindows {
        kept,
        kept_seconds,
        single_voice,
    }
}

/// Indices of the larger of the two clusters the group splits into, in window
/// order. Groups too small to split keep every window.
fn main_cluster(embeddings: &[SpeakerEmbedding]) -> Vec<usize> {
    let Some((main, _)) = split_in_two(embeddings) else {
        return (0..embeddings.len()).collect();
    };
    main
}

fn is_single_voice(embeddings: &[SpeakerEmbedding]) -> bool {
    if embeddings.len() < MIN_PURITY_VERDICT_WINDOWS {
        // Too few windows for the split to carry information. Reporting "single
        // voice" here is safe only because naming separately requires more
        // evidence than this; the verdict on its own never grants a name.
        return true;
    }
    let Some((main, second)) = split_in_two(embeddings) else {
        return true;
    };
    let second_fraction = second.len() as f64 / embeddings.len() as f64;
    if second_fraction < MIXED_MIN_SECOND_CLUSTER_FRACTION {
        return true;
    }
    let (Some(main_centroid), Some(second_centroid)) = (
        centroid(main.iter().map(|&i| &embeddings[i])),
        centroid(second.iter().map(|&i| &embeddings[i])),
    ) else {
        return true;
    };
    let split_distance = 1.0 - main_centroid.cosine(&second_centroid);
    split_distance <= MIXED_MIN_SPLIT_DISTANCE
}

/// `(larger, smaller)` member indices, each in window order. `None` when the
/// group cannot be cut in two at all (fewer than two windows, or the clusterer
/// could not separate them).
fn split_in_two(embeddings: &[SpeakerEmbedding]) -> Option<(Vec<usize>, Vec<usize>)> {
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
    Some((main, second))
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

    /// The split always takes something away, so a group sitting exactly on the
    /// verdict floor cannot supply that many survivors. Naming therefore needs a
    /// group with room to spare, which is the intended reading of the constant
    /// and the reason it is documented as binding after the split.
    #[test]
    fn a_group_at_the_verdict_floor_cannot_supply_that_many_pure_windows() {
        let embeddings: Vec<SpeakerEmbedding> = (0..MIN_PURITY_VERDICT_WINDOWS)
            .map(|i| voice(i as f32 * 0.02))
            .collect();
        let judged = judge_windows(&embeddings, &spans(MIN_PURITY_VERDICT_WINDOWS));
        assert!(judged.single_voice);
        assert!(judged.kept.len() < MIN_PURITY_VERDICT_WINDOWS);
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
