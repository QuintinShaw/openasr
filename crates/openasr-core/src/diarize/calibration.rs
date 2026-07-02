//! WeSpeaker diarization calibration.
//!
//! Batch clustering and streaming registry gates stay in one profile so runtime
//! code consumes calibrated thresholds without embedding model conditionals.

#[derive(Debug, Clone, Copy)]
pub struct SpeakerCalibrationProfile {
    pub(crate) clustering: ClusteringCalibrationProfile,
    pub(crate) streaming: StreamingCalibrationProfile,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ClusteringCalibrationProfile {
    /// Merge stop threshold on cosine dissimilarity (`1 - cosine`) when no
    /// segmentation context is available.
    pub plain_merge_threshold: f32,
    /// Merge stop threshold for context-aware AHC when no denser-session or
    /// gap profile takes over.
    pub context_auto_merge_threshold: f32,
    /// Embeddable-region count where WeSpeaker's dense meeting distribution is
    /// safer with a tight similarity floor than with gap-based speaker count.
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
    /// `relaxed_match_margin`). Calibrated against real speaker-playback
    /// sessions where WeSpeaker same-speaker cosines sag to ~0.37..0.54 while
    /// cross-speaker cosines stay at or below ~0.24.
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

pub(crate) const WESPEAKER_CALIBRATION: SpeakerCalibrationProfile = SpeakerCalibrationProfile {
    clustering: ClusteringCalibrationProfile {
        plain_merge_threshold: 0.43,
        context_auto_merge_threshold: 0.73,
        dense_context_min_embeddings: 30,
        dense_context_merge_threshold: 0.43,
        context_gap: Some(ContextGapCalibrationProfile {
            min_gap: 0.05,
            max_speakers: 4,
            fallback_speakers: 3,
        }),
    },
    streaming: StreamingCalibrationProfile {
        match_similarity: 0.57,
        strong_existing_match_similarity: 0.65,
        relaxed_match_similarity: 0.33,
        relaxed_match_margin: 0.20,
        relaxed_reuse_max_weight: 3.0,
        new_speaker_max_existing_similarity: 0.44,
        profile_anchor_similarity: 0.80,
        native_profile_anchor_similarity: 0.50,
        speaker_change_max_cosine: 0.42,
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wespeaker_calibration_profile_is_pinned() {
        assert_eq!(WESPEAKER_CALIBRATION.clustering.plain_merge_threshold, 0.43);
        assert_eq!(
            WESPEAKER_CALIBRATION
                .clustering
                .context_auto_merge_threshold,
            0.73
        );
        assert_eq!(
            WESPEAKER_CALIBRATION
                .clustering
                .dense_context_min_embeddings,
            30
        );
        assert_eq!(
            WESPEAKER_CALIBRATION
                .clustering
                .dense_context_merge_threshold,
            0.43
        );
        let gap = WESPEAKER_CALIBRATION
            .clustering
            .context_gap
            .expect("WeSpeaker short-context gap profile");
        assert_eq!(gap.min_gap, 0.05);
        assert_eq!(gap.max_speakers, 4);
        assert_eq!(gap.fallback_speakers, 3);
        assert_eq!(WESPEAKER_CALIBRATION.streaming.match_similarity, 0.57);
        assert_eq!(
            WESPEAKER_CALIBRATION.streaming.relaxed_match_similarity,
            0.33
        );
        assert_eq!(WESPEAKER_CALIBRATION.streaming.relaxed_match_margin, 0.20);
        assert_eq!(
            WESPEAKER_CALIBRATION.streaming.relaxed_reuse_max_weight,
            3.0
        );
        assert_eq!(
            WESPEAKER_CALIBRATION
                .streaming
                .strong_existing_match_similarity,
            0.65
        );
        assert_eq!(
            WESPEAKER_CALIBRATION
                .streaming
                .new_speaker_max_existing_similarity,
            0.44
        );
        assert_eq!(
            WESPEAKER_CALIBRATION.streaming.profile_anchor_similarity,
            0.80
        );
        assert_eq!(
            WESPEAKER_CALIBRATION
                .streaming
                .native_profile_anchor_similarity,
            0.50
        );
        assert_eq!(
            WESPEAKER_CALIBRATION.streaming.speaker_change_max_cosine,
            0.42
        );
    }
}
