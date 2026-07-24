//! ReDimNet2-B6 diarization calibration.
//!
//! Batch clustering and streaming registry gates stay in one profile so runtime
//! code consumes calibrated thresholds without embedding model conditionals.

/// Stable calibration identity for ReDimNet2-B6's cosine space.
/// Bump when any threshold that participates in Voice ID matching changes.
pub const REDIMNET_CALIBRATION_VERSION: &str = "redimnet2-b6-cal-v2";

#[derive(Debug, Clone, Copy)]
pub struct SpeakerCalibrationProfile {
    pub(crate) clustering: ClusteringCalibrationProfile,
    pub(crate) streaming: StreamingCalibrationProfile,
    /// Default cosine-similarity floor for a newly enrolled voice-match
    /// profile (`SpeakerProfile::match_similarity`) in this embedder's cosine
    /// space, used when the caller does not supply an explicit override. See
    /// `enrollment::DEFAULT_MATCH_SIMILARITY` for how this is consumed.
    pub(crate) enrollment_default_match_similarity: f32,
    /// Second-stage confidence gate for batch voice-match: the minimum
    /// top1-vs-top2 cosine-similarity margin required, on top of clearing
    /// `enrollment_default_match_similarity` (or an explicit
    /// `match_similarity` override), before a display name is attached to a
    /// diarized speaker. See
    /// `SpeakerProfileMatcher::best_match_with_margin` for how this is
    /// consumed; a library with a single compatible profile has no runner-up
    /// to measure a margin against and always clears this gate (see that
    /// method's doc comment).
    pub(crate) enrollment_match_margin: f32,
}

impl SpeakerCalibrationProfile {
    /// Default accept threshold for Voice ID person matching in this space.
    pub fn voice_id_accept_threshold(&self) -> f32 {
        self.enrollment_default_match_similarity
    }

    /// Person-level top1-vs-top2 margin for Voice ID matching.
    pub fn voice_id_margin(&self) -> f32 {
        self.enrollment_match_margin
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ClusteringCalibrationProfile {
    /// Merge stop threshold on cosine dissimilarity (`1 - cosine`) when no
    /// segmentation context is available.
    pub plain_merge_threshold: f32,
    /// Merge stop threshold for context-aware AHC when no denser-session or
    /// gap profile takes over.
    pub context_auto_merge_threshold: f32,
    /// Embeddable-region count where the dense meeting distribution is safer
    /// with a tight similarity floor than with gap-based speaker count.
    pub dense_context_min_embeddings: usize,
    /// Context-aware threshold used at or above `dense_context_min_embeddings`.
    pub dense_context_merge_threshold: f32,
    /// Optional constrained AHC merge-gap speaker-count profile for short
    /// context-rich files.
    pub context_gap: Option<ContextGapCalibrationProfile>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ContextGapCalibrationProfile {
    pub min_gap: f32,
    pub max_speakers: usize,
    pub fallback_speakers: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StreamingCalibrationProfile {
    pub match_similarity: f32,
    pub strong_existing_match_similarity: f32,
    /// Relaxed same-speaker reuse floor for long, audible utterances whose best
    /// anonymous centroid clearly outscores every other registry centroid (see
    /// `relaxed_match_margin`).
    pub relaxed_match_similarity: f32,
    /// Required lead of the best anonymous centroid over every other registry
    /// centroid (anonymous or profile-owned) before the relaxed floor applies.
    pub relaxed_match_margin: f32,
    /// Maximum centroid-update weight accepted for a relaxed-reuse hit. The
    /// normal weight cap (`MAX_CENTROID_UPDATE_WEIGHT_S = 10 s`) is never
    /// triggered in a single turn, so without a lower cap each relaxed-reuse
    /// miss-absorption would compound: the absorbed turn's embedding (possibly
    /// from a different-but-similar voice) pulls the centroid proportionally to
    /// its duration, and repeated misses drift it away from the true speaker.
    /// Capping at 3 s limits the pull to one short but reliable segment, letting
    /// the centroid self-correct on the next strong-match turn.
    pub relaxed_reuse_max_weight: f32,
    pub new_speaker_max_existing_similarity: f32,
    pub profile_anchor_similarity: f32,
    pub native_profile_anchor_similarity: f32,
    pub speaker_change_max_cosine: f32,
}

/// ReDimNet2-B6 calibration (192-dim cosine space).
///
/// Product Voice ID enrollment-match defaults below come from the phase-1
/// quality eval report (`tmp/voice-id-b6-eval/outputs/REPORT.md`, not checked
/// into this repo). That pass used official PyTorch ReDimNet2-B6 on LibriSpeech
/// `test-clean` (40 speakers, gallery sizes 5/10/20/40) plus AISHELL-1
/// out-of-gallery strangers. Headline product gates:
/// - Prior core default thr 0.55 + margin 0.15 still hit ~99.5%, but produced
///   stranger false-name events on a hard pair. Phase-1 therefore recommends
///   **threshold 0.60 / margin 0.15** as the conservative v1 product bar
///   (misnaming an enrolled person is worse than Unknown). Margin is unchanged
///   from the earlier LibriSpeech margin-distribution cut (enrolled p10 > 0.3,
///   impostor p90 < 0.17).
/// - Clustering merge floors stay on the older acoustic 0.55 cosine reference
///   (`plain_merge_threshold` / `dense_context_merge_threshold` = 0.45 as
///   `1 - 0.55`). They are **not** auto-rewritten to `1 - 0.60` here: no fresh
///   multi-speaker clustering eval backs a 0.40 dissimilarity cut, and
///   product Voice ID matching is a separate gate from AHC merge stop.
///
/// Caveat: phase-1 is clean, single-speaker-per-file read speech plus
/// cross-corpus strangers. It does not cover AISHELL-4 meeting leakage,
/// cross-device capture, or streaming incremental decisions; those remain
/// out of scope for this profile bump.
///
/// This version ships batch / Voice ID enrollment-match defaults for ReDimNet2
/// only; the streaming (realtime) fields further down have no equivalent
/// streaming trial data yet and stay conservative TODO placeholders (batch
/// enrollment default was updated independently of those placeholders).
pub(crate) const REDIMNET_CALIBRATION: SpeakerCalibrationProfile = SpeakerCalibrationProfile {
    clustering: ClusteringCalibrationProfile {
        // Held at the older acoustic 0.55 cosine reference as `1 - cosine`
        // dissimilarity (0.45). Not retuned with the product Voice ID thr
        // bump to 0.60; needs a dedicated clustering eval before changing.
        plain_merge_threshold: 0.45,
        // Not independently measured -- no multi-speaker segmentation corpus
        // in the LibriSpeech eval to calibrate the context-assisted merge
        // gate. Extrapolated as ~1.70x the plain threshold
        // (0.45 * 1.70 ~= 0.76). Needs a real multi-speaker meeting-style
        // corpus (e.g. AISHELL-4) before this can be called calibrated.
        context_auto_merge_threshold: 0.76,
        dense_context_min_embeddings: 30,
        // Same hold as `plain_merge_threshold`: keep dense-context merge on
        // the acoustic 0.55 reference until clustering is re-eval'd.
        dense_context_merge_threshold: 0.45,
        context_gap: Some(ContextGapCalibrationProfile {
            min_gap: 0.05,
            max_speakers: 4,
            fallback_speakers: 3,
        }),
    },
    // Streaming (realtime registry consolidation) is out of scope for this
    // calibration pass: phase-1 only exercised batch-style enrollment/matching
    // trials, not the incremental same-turn-vs-new-turn decisions streaming
    // makes. Values below stay conservative, fail-toward-"no match"
    // placeholders until a dedicated streaming/Challenge Set pass calibrates
    // them for real. Do not treat the coincidental 0.60 streaming
    // `match_similarity` placeholder as the same decision as the product batch
    // enrollment default below.
    streaming: StreamingCalibrationProfile {
        // TODO(voice-id-eval): placeholder, needs streaming/Challenge Set calibration.
        match_similarity: 0.60,
        // TODO(voice-id-eval): placeholder, needs streaming/Challenge Set calibration.
        strong_existing_match_similarity: 0.70,
        // TODO(voice-id-eval): placeholder, needs streaming/Challenge Set calibration.
        relaxed_match_similarity: 0.40,
        // TODO(voice-id-eval): placeholder, needs streaming/Challenge Set calibration.
        relaxed_match_margin: 0.20,
        // TODO(voice-id-eval): placeholder, needs streaming/Challenge Set calibration.
        relaxed_reuse_max_weight: 3.0,
        // TODO(voice-id-eval): placeholder, needs streaming/Challenge Set calibration.
        new_speaker_max_existing_similarity: 0.50,
        // TODO(voice-id-eval): placeholder, needs streaming/Challenge Set calibration.
        profile_anchor_similarity: 0.80,
        // TODO(voice-id-eval): placeholder, needs streaming/Challenge Set calibration.
        native_profile_anchor_similarity: 0.55,
        // TODO(voice-id-eval): placeholder, needs streaming/Challenge Set calibration.
        speaker_change_max_cosine: 0.45,
    },
    // Product Voice ID / batch enrollment default from phase-1 eval
    // (`tmp/voice-id-b6-eval/outputs/REPORT.md`): thr 0.60 lowers stranger
    // false-name risk vs the prior 0.55 default while keeping hit rate high
    // on clean single-speaker audio.
    enrollment_default_match_similarity: 0.60,
    // Unchanged from the earlier margin-distribution cut and phase-1
    // recommendation: top1-vs-top2 margin 0.15 stays the "confident enough to
    // display a name" bar (`tmp/voice-id-b6-eval/outputs/REPORT.md`).
    enrollment_match_margin: 0.15,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redimnet_calibration_profile_is_pinned() {
        assert_eq!(REDIMNET_CALIBRATION.clustering.plain_merge_threshold, 0.45);
        assert_eq!(
            REDIMNET_CALIBRATION.clustering.context_auto_merge_threshold,
            0.76
        );
        assert_eq!(
            REDIMNET_CALIBRATION
                .clustering
                .dense_context_merge_threshold,
            0.45
        );
        assert_eq!(REDIMNET_CALIBRATION.streaming.match_similarity, 0.60);
        assert_eq!(
            REDIMNET_CALIBRATION
                .streaming
                .strong_existing_match_similarity,
            0.70
        );
        assert_eq!(
            REDIMNET_CALIBRATION.enrollment_default_match_similarity,
            0.60
        );
        assert_eq!(REDIMNET_CALIBRATION.enrollment_match_margin, 0.15);
    }

    /// Clustering thresholds stay on the older acoustic 0.55 cosine reference
    /// (`plain_merge_threshold` / `dense_context_merge_threshold` = `1 - 0.55`)
    /// and must not be silently retied to the product Voice ID enrollment
    /// default (0.60). Context-assisted merge must stay looser than plain.
    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn redimnet_clustering_thresholds_are_self_consistent_with_acoustic_reference() {
        const ACOUSTIC_CLUSTERING_COSINE_REFERENCE: f32 = 0.55;
        let expected_dissimilarity = 1.0 - ACOUSTIC_CLUSTERING_COSINE_REFERENCE;
        assert!(
            (REDIMNET_CALIBRATION.clustering.plain_merge_threshold - expected_dissimilarity).abs()
                < 1e-6
        );
        assert_eq!(
            REDIMNET_CALIBRATION.clustering.plain_merge_threshold,
            REDIMNET_CALIBRATION
                .clustering
                .dense_context_merge_threshold,
            "dense-context merge threshold should match the plain threshold"
        );
        assert!(
            REDIMNET_CALIBRATION.clustering.context_auto_merge_threshold
                > REDIMNET_CALIBRATION.clustering.plain_merge_threshold,
            "context-assisted merging must stay looser than the acoustic-only floor"
        );
        assert!(
            (REDIMNET_CALIBRATION.enrollment_default_match_similarity
                - ACOUSTIC_CLUSTERING_COSINE_REFERENCE)
                .abs()
                > 1e-6,
            "product Voice ID enrollment default is intentionally independent of the clustering acoustic reference"
        );
    }

    #[test]
    fn redimnet_enrollment_default_match_similarity_is_product_floor() {
        assert_eq!(
            REDIMNET_CALIBRATION.enrollment_default_match_similarity,
            0.60
        );
        assert_eq!(REDIMNET_CALIBRATION.enrollment_match_margin, 0.15);
    }
}
