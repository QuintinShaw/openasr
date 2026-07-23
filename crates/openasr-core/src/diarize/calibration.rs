//! WeSpeaker diarization calibration.
//!
//! Batch clustering and streaming registry gates stay in one profile so runtime
//! code consumes calibrated thresholds without embedding model conditionals.

#[derive(Debug, Clone, Copy)]
pub struct SpeakerCalibrationProfile {
    pub(crate) clustering: ClusteringCalibrationProfile,
    pub(crate) streaming: StreamingCalibrationProfile,
    /// Default cosine-similarity floor for a newly enrolled voice-match
    /// profile (`SpeakerProfile::match_similarity`) in this embedder's cosine
    /// space, used when the caller does not supply an explicit override. See
    /// `enrollment::DEFAULT_MATCH_SIMILARITY` for how this is consumed.
    pub(crate) enrollment_default_match_similarity: f32,
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
    enrollment_default_match_similarity: 0.5,
};

/// ReDimNet2-B6 calibration (192-dim cosine space), distinct from
/// `WESPEAKER_CALIBRATION` -- the two embedders' cosine distributions are not
/// comparable, so nothing here is copied from the WeSpeaker profile above.
///
/// The batch-pipeline (clustering) thresholds below are derived from a real
/// voice-id evaluation: LibriSpeech `test-clean`, 40 speakers, 42,960 trials
/// across enrolled-library sizes of 5/10/20/40 speakers and 5-20 s enrollment
/// clips (`tmp/voiceid-eval/results/{summary_metrics,threshold_curves,
/// margin_distributions}.csv`, not checked into this repo). Headline findings:
/// - A cosine match threshold of 0.55 keeps FAR at 0-1.5% with 99.4-100% hit
///   rate across 5-20 speaker libraries (`threshold_curves.csv`, worst case
///   is the 20-speaker `multi3_12s` config at FAR 1.53% / hit 99.76%); a
///   20-speaker library shows roughly an order of magnitude more false
///   accepts than a 5-speaker library at the same threshold, so 0.55 is the
///   floor, not a number to relax as libraries grow. Per-scenario thresholds
///   that hit the tighter FAR < 1% bar range 0.43-0.57
///   (`summary_metrics.csv` `rec_threshold_far_lt_1pct`); 0.55 is a
///   reasonable single cutover value but is not FAR < 1% for every scenario.
/// - The top1-vs-top2 similarity margin cleanly separates registered speakers
///   from strangers: enrolled-speaker margins have p10 > 0.3, impostor margins
///   have p90 < 0.17, so >= 0.15 margin is a reasonable "safe to display a
///   name with high confidence" bar (`margin_distributions.csv`). Nothing in
///   this crate currently gates the batch match on a runner-up margin (see
///   `SpeakerProfileMatcher::best_match`, margin = 0.0) -- that is a
///   deliberately separate change, not folded into this calibration pass.
///
/// Caveat: the eval is a clean, single-speaker-per-file, native-microphone
/// read-speech corpus (LibriSpeech). It calibrates the acoustic same/
/// different-speaker cosine floor well but says nothing about cross-device
/// recording conditions, noisy/far-field capture, or multi-speaker
/// segmentation quality; those need a dedicated Challenge Set pass before the
/// thresholds below can be trusted outside this scenario.
///
/// This version ships batch file-transcription support for ReDimNet2 only;
/// the streaming (realtime) fields further down have no equivalent streaming
/// trial data yet and stay conservative TODO placeholders.
pub(crate) const REDIMNET_CALIBRATION: SpeakerCalibrationProfile = SpeakerCalibrationProfile {
    clustering: ClusteringCalibrationProfile {
        // Measured: LibriSpeech eval's 0.55 cosine main match threshold,
        // expressed as `1 - cosine` dissimilarity (0.45). Clean read-speech
        // only; see module-level caveat above.
        plain_merge_threshold: 0.45,
        // Not independently measured -- no multi-speaker segmentation corpus
        // in the LibriSpeech eval to calibrate the context-assisted merge
        // gate. Extrapolated by applying WeSpeaker's own
        // context_auto_merge_threshold / plain_merge_threshold ratio
        // (0.73 / 0.43 ~= 1.70) to the measured plain threshold above
        // (0.45 * 1.70 ~= 0.76). Needs a real multi-speaker meeting-style
        // corpus (e.g. AISHELL-4) before this can be called calibrated.
        context_auto_merge_threshold: 0.76,
        dense_context_min_embeddings: 30,
        // Same reasoning as `plain_merge_threshold`: WeSpeaker keeps this
        // equal to its plain threshold, and the measured 0.55 main match
        // threshold gives no separate signal for dense-meeting behavior.
        dense_context_merge_threshold: 0.45,
        context_gap: Some(ContextGapCalibrationProfile {
            min_gap: 0.05,
            max_speakers: 4,
            fallback_speakers: 3,
        }),
    },
    // Streaming (realtime registry consolidation) is out of scope for this
    // calibration pass: the LibriSpeech eval only exercised batch-style
    // enrollment/matching trials, not the incremental same-turn-vs-new-turn
    // decisions streaming makes. Values below stay the original conservative,
    // fail-toward-"no match" placeholders (distinct from WeSpeaker's tuned
    // numbers) until a dedicated streaming/Challenge Set pass calibrates them
    // for real; this crate does not ship streaming support for ReDimNet2 yet.
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
    // Measured: LibriSpeech eval's 0.55 cosine main match threshold. This is
    // the default floor for a newly enrolled `SpeakerProfile` in ReDimNet2's
    // cosine space (batch file-transcription enrollment/matching), up from
    // WeSpeaker's unrelated 0.5 default -- the two embedders' cosine spaces
    // are not comparable, see module doc.
    enrollment_default_match_similarity: 0.55,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redimnet_calibration_profile_is_pinned_and_distinct_from_wespeaker() {
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
            0.55
        );
        assert_ne!(
            REDIMNET_CALIBRATION.clustering.plain_merge_threshold,
            WESPEAKER_CALIBRATION.clustering.plain_merge_threshold,
            "redimnet calibration must not be copied verbatim from wespeaker's tuned values"
        );
        assert_ne!(
            REDIMNET_CALIBRATION.streaming.match_similarity,
            WESPEAKER_CALIBRATION.streaming.match_similarity,
            "redimnet calibration must not be copied verbatim from wespeaker's tuned values"
        );
        assert_ne!(
            REDIMNET_CALIBRATION.enrollment_default_match_similarity,
            WESPEAKER_CALIBRATION.enrollment_default_match_similarity,
            "redimnet calibration must not be copied verbatim from wespeaker's tuned values"
        );
    }

    /// The clustering thresholds must stay internally consistent with the
    /// measured 0.55 cosine main match threshold: `plain_merge_threshold` and
    /// `dense_context_merge_threshold` are both `1 - 0.55` dissimilarity, and
    /// the context-assisted threshold must loosen (not tighten) relative to
    /// the plain one, matching WeSpeaker's own merge-threshold ordering.
    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn redimnet_clustering_thresholds_are_self_consistent_with_measured_main_threshold() {
        const MEASURED_MAIN_MATCH_COSINE: f32 = 0.55;
        let expected_dissimilarity = 1.0 - MEASURED_MAIN_MATCH_COSINE;
        assert!(
            (REDIMNET_CALIBRATION.clustering.plain_merge_threshold - expected_dissimilarity).abs()
                < 1e-6
        );
        assert_eq!(
            REDIMNET_CALIBRATION.clustering.plain_merge_threshold,
            REDIMNET_CALIBRATION
                .clustering
                .dense_context_merge_threshold,
            "dense-context merge threshold should match the plain threshold, as in WeSpeaker's profile"
        );
        assert!(
            REDIMNET_CALIBRATION.clustering.context_auto_merge_threshold
                > REDIMNET_CALIBRATION.clustering.plain_merge_threshold,
            "context-assisted merging must stay looser than the acoustic-only floor"
        );
        assert!(
            (REDIMNET_CALIBRATION.enrollment_default_match_similarity - MEASURED_MAIN_MATCH_COSINE)
                .abs()
                < 1e-6,
            "enrollment default should track the measured main match threshold"
        );
    }

    /// The per-embedder enrollment default must actually diverge: WeSpeaker
    /// keeps its historical 0.5 default (`enrollment::DEFAULT_MATCH_SIMILARITY`)
    /// while ReDimNet2 uses the measured 0.55 floor.
    #[test]
    fn enrollment_default_match_similarity_is_embedder_specific() {
        assert_eq!(
            WESPEAKER_CALIBRATION.enrollment_default_match_similarity,
            0.5
        );
        assert_eq!(
            REDIMNET_CALIBRATION.enrollment_default_match_similarity,
            0.55
        );
        assert_ne!(
            WESPEAKER_CALIBRATION.enrollment_default_match_similarity,
            REDIMNET_CALIBRATION.enrollment_default_match_similarity
        );
    }

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
        assert_eq!(
            WESPEAKER_CALIBRATION.enrollment_default_match_similarity,
            0.5
        );
    }
}
