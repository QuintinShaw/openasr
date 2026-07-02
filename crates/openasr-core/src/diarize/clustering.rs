//! Agglomerative speaker clustering (pure Rust, no weights).
//!
//! Average-linkage agglomerative hierarchical clustering on cosine
//! dissimilarity (`1 - cos`), the sherpa-onnx-style default. Embedding counts
//! are small (one per speech segment — tens to low hundreds), so the naive
//! O(n^3) merge loop is comfortably fast. When pyannote segmentation context is
//! available, clustering also honors overlap cannot-link constraints.

use super::calibration::{
    ClusteringCalibrationProfile, ContextGapCalibrationProfile, WESPEAKER_CALIBRATION,
};
use super::contract::{DiarizeHint, SpeakerEmbedding, SpeakerId, TimeRange};
use super::embed::SpeakerEmbedder;

/// Default threshold on cosine **dissimilarity** (`1 - cos`): clusters merge
/// while their average-linkage distance is below this.
///
/// This is the WeSpeaker plain AHC threshold. Valid range is `[0, 2]`.
pub const DEFAULT_MERGE_THRESHOLD: f32 = WESPEAKER_CALIBRATION.clustering.plain_merge_threshold;
/// Context-aware auto clustering can safely use a looser stop because overlap
/// constraints prevent merging simultaneous speakers.
pub const CONTEXT_AUTO_MERGE_THRESHOLD: f32 = WESPEAKER_CALIBRATION
    .clustering
    .context_auto_merge_threshold;

#[derive(Debug, Clone, Copy)]
pub struct ClusterContext {
    pub range: TimeRange,
    pub local_speaker: Option<SpeakerId>,
    pub overlap: bool,
}

/// Assigns each embedding to a session-relative [`SpeakerId`].
pub trait SpeakerClusterer: Send + Sync {
    fn cluster(&self, embeddings: &[SpeakerEmbedding], hint: DiarizeHint) -> Vec<SpeakerId>;

    fn cluster_with_context(
        &self,
        embeddings: &[SpeakerEmbedding],
        context: &[ClusterContext],
        hint: DiarizeHint,
    ) -> Vec<SpeakerId> {
        let _ = context;
        self.cluster(embeddings, hint)
    }
}

/// Average-linkage agglomerative clusterer over cosine dissimilarity.
#[derive(Debug, Clone, Copy)]
pub struct AgglomerativeClusterer {
    /// Merge stop threshold used for `Auto` / `Threshold` hints.
    pub threshold: f32,
    profile: ClusteringCalibrationProfile,
}

impl Default for AgglomerativeClusterer {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_MERGE_THRESHOLD,
            profile: WESPEAKER_CALIBRATION.clustering,
        }
    }
}

impl AgglomerativeClusterer {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            profile: ClusteringCalibrationProfile {
                plain_merge_threshold: threshold,
                context_auto_merge_threshold: threshold,
                dense_context_min_embeddings: usize::MAX,
                dense_context_merge_threshold: threshold,
                context_gap: None,
            },
        }
    }

    pub fn for_embedder(embedder: &dyn SpeakerEmbedder) -> Self {
        Self::for_profile(embedder.calibration_profile().clustering)
    }

    pub(crate) fn for_profile(profile: ClusteringCalibrationProfile) -> Self {
        Self {
            threshold: profile.plain_merge_threshold,
            profile,
        }
    }
}

impl SpeakerClusterer for AgglomerativeClusterer {
    fn cluster(&self, embeddings: &[SpeakerEmbedding], hint: DiarizeHint) -> Vec<SpeakerId> {
        self.cluster_inner(embeddings, None, hint)
    }

    fn cluster_with_context(
        &self,
        embeddings: &[SpeakerEmbedding],
        context: &[ClusterContext],
        hint: DiarizeHint,
    ) -> Vec<SpeakerId> {
        self.cluster_inner(embeddings, Some(context), hint)
    }
}

impl AgglomerativeClusterer {
    fn cluster_inner(
        &self,
        embeddings: &[SpeakerEmbedding],
        context: Option<&[ClusterContext]>,
        hint: DiarizeHint,
    ) -> Vec<SpeakerId> {
        let n = embeddings.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![SpeakerId(0)];
        }

        // Pairwise cosine similarity (symmetric; diagonal unused).
        let mut sim = vec![0.0f32; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let s = embeddings[i].cosine(&embeddings[j]);
                sim[i * n + j] = s;
                sim[j * n + i] = s;
            }
        }

        // Active clusters as member-index lists.
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let target = match hint {
            DiarizeHint::NumSpeakers(k) => (k as usize).max(1),
            _ => 1,
        };
        // Cosine dissimilarity over L2-normalized vectors is in [0, 2]; clamp so
        // an out-of-range client knob can't silently over-/under-split.
        let context = context.filter(|context| context.len() == n);
        let has_context_signal = context.is_some_and(context_has_real_signal);
        if matches!(hint, DiarizeHint::Auto)
            && has_context_signal
            && let Some(context) = context
            && let Some(labels) = self.cluster_by_context_gap(&sim, n, context)
        {
            return labels;
        }

        let stop_threshold = self.stop_threshold(n, has_context_signal, hint);

        while clusters.len() > target {
            // Closest pair by average-linkage cosine distance (1 - mean sim).
            let mut best = (0usize, 1usize);
            let mut best_dist = f32::INFINITY;
            for a in 0..clusters.len() {
                for b in (a + 1)..clusters.len() {
                    if context.is_some_and(|context| {
                        clusters_overlap(&clusters[a], &clusters[b], context)
                    }) {
                        continue;
                    }
                    let dist = 1.0 - average_similarity(&clusters[a], &clusters[b], &sim, n);
                    if dist < best_dist {
                        best_dist = dist;
                        best = (a, b);
                    }
                }
            }
            // For Auto/Threshold, stop once the closest clusters are too far.
            if !matches!(hint, DiarizeHint::NumSpeakers(_)) && best_dist > stop_threshold {
                break;
            }
            if !best_dist.is_finite() {
                break;
            }
            let (a, b) = best;
            let merged_b = clusters.remove(b);
            clusters[a].extend(merged_b);
        }

        if let Some(context) = context {
            assign_time_order_labels(&clusters, context, n)
        } else {
            assign_arrival_order_labels(&clusters, n)
        }
    }

    fn stop_threshold(&self, n: usize, has_context_signal: bool, hint: DiarizeHint) -> f32 {
        match hint {
            DiarizeHint::Threshold(t) => t.clamp(0.0, 2.0),
            DiarizeHint::Auto
                if has_context_signal && n >= self.profile.dense_context_min_embeddings =>
            {
                self.profile.dense_context_merge_threshold
            }
            DiarizeHint::Auto if has_context_signal => self.profile.context_auto_merge_threshold,
            _ => self.profile.plain_merge_threshold,
        }
    }

    fn cluster_by_context_gap(
        &self,
        sim: &[f32],
        n: usize,
        context: &[ClusterContext],
    ) -> Option<Vec<SpeakerId>> {
        let gap = self.profile.context_gap?;
        if n >= self.profile.dense_context_min_embeddings {
            return None;
        }
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut states: Vec<Option<Vec<Vec<usize>>>> = vec![None; n + 1];
        let mut merge_dist_by_k = vec![f32::NAN; n + 1];
        states[n] = Some(clusters.clone());

        while clusters.len() > 1 {
            let mut best = (0usize, 1usize);
            let mut best_dist = f32::INFINITY;
            for a in 0..clusters.len() {
                for b in (a + 1)..clusters.len() {
                    if clusters_overlap(&clusters[a], &clusters[b], context) {
                        continue;
                    }
                    let dist = 1.0 - average_similarity(&clusters[a], &clusters[b], sim, n);
                    if dist < best_dist {
                        best_dist = dist;
                        best = (a, b);
                    }
                }
            }
            if !best_dist.is_finite() {
                break;
            }
            let k = clusters.len();
            merge_dist_by_k[k] = best_dist;
            let (a, b) = best;
            let merged_b = clusters.remove(b);
            clusters[a].extend(merged_b);
            states[clusters.len()] = Some(clusters.clone());
        }

        let chosen_k = choose_context_gap_speaker_count(&merge_dist_by_k, &states, n, gap)?;
        let clusters = states[chosen_k].as_ref()?;
        Some(assign_time_order_labels(clusters, context, n))
    }
}

fn choose_context_gap_speaker_count(
    merge_dist_by_k: &[f32],
    states: &[Option<Vec<Vec<usize>>>],
    n: usize,
    gap: ContextGapCalibrationProfile,
) -> Option<usize> {
    let max_k = gap.max_speakers.min(n).max(1);
    let mut best_k = None;
    let mut best_gap = f32::NEG_INFINITY;
    for (k, state) in states.iter().enumerate().take(max_k + 1).skip(2) {
        let this_dist = merge_dist_by_k.get(k).copied().unwrap_or(f32::NAN);
        let prev_dist = merge_dist_by_k.get(k + 1).copied().unwrap_or(f32::NAN);
        if !this_dist.is_finite() || !prev_dist.is_finite() || state.is_none() {
            continue;
        }
        let candidate_gap = this_dist - prev_dist;
        if candidate_gap > best_gap {
            best_gap = candidate_gap;
            best_k = Some(k);
        }
    }

    if best_gap >= gap.min_gap {
        return best_k;
    }

    let fallback = gap.fallback_speakers.clamp(1, max_k);
    if states[fallback].is_some() {
        return Some(fallback);
    }
    (1..=max_k).rev().find(|&k| states[k].is_some())
}

fn clusters_overlap(a: &[usize], b: &[usize], context: &[ClusterContext]) -> bool {
    a.iter().any(|&i| {
        b.iter()
            .any(|&j| context[i].range.overlaps(&context[j].range))
    })
}

fn average_similarity(a: &[usize], b: &[usize], sim: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    for &i in a {
        for &j in b {
            total += sim[i * n + j];
        }
    }
    total / (a.len() * b.len()) as f32
}

fn context_has_real_signal(context: &[ClusterContext]) -> bool {
    context
        .iter()
        .any(|item| item.local_speaker.is_some() || item.overlap)
}

fn assign_time_order_labels(
    clusters: &[Vec<usize>],
    context: &[ClusterContext],
    n: usize,
) -> Vec<SpeakerId> {
    let mut order: Vec<usize> = (0..clusters.len()).collect();
    order.sort_by(|&left, &right| {
        let left_start = clusters[left]
            .iter()
            .map(|&member| context[member].range.start_s)
            .fold(f64::INFINITY, f64::min);
        let right_start = clusters[right]
            .iter()
            .map(|&member| context[member].range.start_s)
            .fold(f64::INFINITY, f64::min);
        left_start
            .partial_cmp(&right_start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut labels = vec![SpeakerId(0); n];
    for (speaker_idx, &cluster_idx) in order.iter().enumerate() {
        for &member in &clusters[cluster_idx] {
            labels[member] = SpeakerId(speaker_idx as u32);
        }
    }
    labels
}

/// Label clusters by arrival order: the cluster whose earliest member index is
/// smallest gets `SPEAKER_00`, and so on, so labels are deterministic and
/// reflect when each speaker first appears.
fn assign_arrival_order_labels(clusters: &[Vec<usize>], n: usize) -> Vec<SpeakerId> {
    let mut order: Vec<usize> = (0..clusters.len()).collect();
    order.sort_by_key(|&c| clusters[c].iter().copied().min().unwrap_or(usize::MAX));

    let mut labels = vec![SpeakerId(0); n];
    for (speaker_idx, &cluster_idx) in order.iter().enumerate() {
        for &member in &clusters[cluster_idx] {
            labels[member] = SpeakerId(speaker_idx as u32);
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emb(v: Vec<f32>) -> SpeakerEmbedding {
        SpeakerEmbedding::l2_normalized(v)
    }

    fn ctx(
        start_s: f64,
        end_s: f64,
        local_speaker: Option<SpeakerId>,
        overlap: bool,
    ) -> ClusterContext {
        ClusterContext {
            range: TimeRange::new(start_s, end_s),
            local_speaker,
            overlap,
        }
    }

    #[test]
    fn empty_and_single() {
        let clusterer = AgglomerativeClusterer::default();
        assert!(clusterer.cluster(&[], DiarizeHint::Auto).is_empty());
        assert_eq!(
            clusterer.cluster(&[emb(vec![1.0, 0.0])], DiarizeHint::Auto),
            vec![SpeakerId(0)]
        );
    }

    #[test]
    fn separates_two_clear_speakers_by_threshold() {
        // Two tight groups around orthogonal directions.
        let embeddings = vec![
            emb(vec![1.0, 0.05]),
            emb(vec![1.0, 0.0]),
            emb(vec![0.0, 1.0]),
            emb(vec![0.05, 1.0]),
        ];
        let labels = AgglomerativeClusterer::default().cluster(&embeddings, DiarizeHint::Auto);
        assert_eq!(labels[0], labels[1], "group A together");
        assert_eq!(labels[2], labels[3], "group B together");
        assert_ne!(labels[0], labels[2], "groups distinct");
        // Arrival order: first group is SPEAKER_00.
        assert_eq!(labels[0], SpeakerId(0));
        assert_eq!(labels[2], SpeakerId(1));
    }

    #[test]
    fn num_speakers_hint_forces_exact_count() {
        let embeddings = vec![
            emb(vec![1.0, 0.0]),
            emb(vec![0.9, 0.1]),
            emb(vec![0.0, 1.0]),
        ];
        let labels =
            AgglomerativeClusterer::default().cluster(&embeddings, DiarizeHint::NumSpeakers(1));
        assert!(labels.iter().all(|l| *l == SpeakerId(0)), "all one speaker");
    }

    #[test]
    fn high_threshold_merges_everything() {
        let embeddings = vec![emb(vec![1.0, 0.0]), emb(vec![0.0, 1.0])];
        let labels =
            AgglomerativeClusterer::default().cluster(&embeddings, DiarizeHint::Threshold(2.0));
        assert_eq!(labels[0], labels[1]);
    }

    #[test]
    fn context_without_pyannote_signal_uses_plain_threshold() {
        let embeddings = vec![emb(vec![1.0, 0.0]), emb(vec![0.5, 0.866_025_4])];
        let context = vec![ctx(0.0, 1.0, None, false), ctx(1.0, 2.0, None, false)];

        let labels = AgglomerativeClusterer::default().cluster_with_context(
            &embeddings,
            &context,
            DiarizeHint::Auto,
        );

        assert_ne!(
            labels[0], labels[1],
            "plain TimeRange context must not unlock the loose context threshold"
        );
    }

    #[test]
    fn local_slot_aba_is_not_repaired_after_clustering() {
        let embeddings = vec![
            emb(vec![1.0, 0.0]),
            emb(vec![0.4, 0.916_515_1]),
            emb(vec![1.0, 0.0]),
        ];
        let context = vec![
            ctx(0.0, 5.0, Some(SpeakerId(7)), false),
            ctx(5.0, 7.0, Some(SpeakerId(7)), false),
            ctx(7.0, 12.0, Some(SpeakerId(7)), false),
        ];
        let labels = AgglomerativeClusterer::default().cluster_with_context(
            &embeddings,
            &context,
            DiarizeHint::Auto,
        );

        assert_eq!(
            labels,
            vec![SpeakerId(0), SpeakerId(1), SpeakerId(0)],
            "A-B-A local-slot islands are left to the clustering evidence"
        );
    }
}
