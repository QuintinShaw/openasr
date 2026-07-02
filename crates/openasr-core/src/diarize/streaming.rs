//! Streaming diarization (P5): incremental speaker assignment with a persistent,
//! arrival-order speaker registry.
//!
//! Unlike batch diarization (which clusters all segments at once), streaming has
//! no future context, so labels must stay consistent as utterances arrive. Each
//! finalized utterance is embedded and matched against a growing set of speaker
//! centroids. Conservative gates keep short/noisy turns from creating new
//! identities, while high-confidence enrollment matches anchor profile-owned
//! speakers before anonymous assignment.

use std::time::Instant;

use super::calibration::StreamingCalibrationProfile;
use super::contract::{SpeakerEmbedding, SpeakerId};
use super::debug::diarize_debug_enabled as debug_enabled;
use super::embed::{SpeakerEmbedder, shared_embedder};
use super::enrollment::{SpeakerDisplayAssignment, SpeakerProfileMatcher};

/// Default cosine-similarity floor to reuse an existing speaker. Mirrors the
/// legacy reuse floor. The active WeSpeaker calibration can override this in
/// runtime constructors; this constant remains the conservative fallback.
pub const DEFAULT_MATCH_SIMILARITY: f32 = 0.65;
/// Shortest utterance (seconds at 16 kHz) worth embedding for diarization.
const MIN_UTTERANCE_S: f32 = 0.5;
/// New anonymous speakers need substantially more speech than a single short
/// embedding window. 2.5 s is the shortest fixed-window duration used in the project
/// diagnostics and matches the common "a few seconds" speaker-verification
/// rule of thumb; shorter turns can only attach to a strong existing centroid.
pub const MIN_NEW_SPEAKER_DURATION_S: f32 = 2.5;
/// Centroids are updated only from embeddings with enough speech context to be
/// stable. Shorter turns may get a label, but they do not pull the centroid.
pub const MIN_CENTROID_UPDATE_DURATION_S: f32 = 1.0;
/// Profile-owned speakers may be created from less speech than anonymous
/// speakers, but only after a very high, unambiguous profile match.
pub const MIN_PROFILE_ANCHOR_DURATION_S: f32 = 1.0;
/// Short/low-confidence turns may only attach to an existing anonymous speaker
/// when the match is clearly above the normal reuse floor.
pub const STRONG_EXISTING_MATCH_SIMILARITY: f32 = 0.72;
/// A new speaker is created only when it is separated from every existing
/// session centroid by at least 0.15 below the normal reuse floor. The band
/// between this ceiling and the reuse floor is left unlabelled instead of
/// over-splitting, unless the calibrated relaxed-margin reuse gate
/// (`relaxed_match_similarity` / `relaxed_match_margin`) attaches the turn to
/// a clearly dominant existing anonymous centroid first.
pub const NEW_SPEAKER_MAX_EXISTING_SIMILARITY: f32 = 0.50;
/// Realtime enrollment anchoring is stricter than the stored profile's normal
/// match floor because it creates a stable named speaker for the session.
pub const PROFILE_ANCHOR_SIMILARITY: f32 = 0.85;
/// Native streaming utterance buffers are speech-gated and profile-owned labels
/// must stay stable across short terminal chunks. Use the stored profile's
/// match floor as the native floor instead of the stricter anonymous-session
/// creation floor.
pub const NATIVE_PROFILE_ANCHOR_SIMILARITY: f32 = 0.50;
/// Native true-streaming short turns can lose some edge speech to VAD debounce.
/// This tolerance is only allowed to reuse a profile already anchored strictly
/// in the same session; it must never create the first named anchor.
pub const PROFILE_ANCHOR_THRESHOLD_TOLERANCE: f32 = 0.10;
/// Required lead over the second-best enrolled profile before anchoring.
pub const PROFILE_ANCHOR_MARGIN: f32 = 0.08;
/// Required lead over an established anonymous centroid before a profile match
/// can capture the turn.
pub const PROFILE_ANCHOR_ANONYMOUS_MARGIN: f32 = 0.08;
/// Audibility floor for learning new centroids. The input is already VAD-gated,
/// so these are intentionally much lower than speech-start VAD thresholds and
/// are only meant to reject silence/hangover fragments.
pub const MIN_DIARIZE_RMS: f32 = 0.001;
pub const MIN_DIARIZE_PEAK: f32 = 0.006;
/// Window compared against recent speech for streaming speaker-change splits.
pub const SPEAKER_CHANGE_REFERENCE_WINDOW_S: f32 = 2.5;
/// Recent speech window used to detect that the in-flight utterance has changed
/// speakers. WeSpeaker needs a few seconds to produce stable speaker-separation
/// scores on this real mixed mic/video case.
pub const SPEAKER_CHANGE_RECENT_WINDOW_S: f32 = 2.5;
/// Re-check cadence for in-flight speaker changes.
pub const SPEAKER_CHANGE_HOP_S: f32 = 1.0;
const MAX_CENTROID_UPDATE_WEIGHT_S: f32 = 10.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingDiarizePath {
    Direct,
    Fallback,
    Native,
}

impl StreamingDiarizePath {
    fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Fallback => "fallback",
            Self::Native => "native",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SpeakerOwner {
    Anonymous,
    Profile(String),
}

#[derive(Debug, Clone)]
struct SpeakerCentroid {
    id: SpeakerId,
    sum: Vec<f32>,
    weight: f32,
    owner: SpeakerOwner,
}

#[derive(Debug, Clone, Copy)]
struct RegistryMatch {
    index: usize,
    speaker_id: SpeakerId,
    similarity: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AssignmentDecision {
    AnonymousNew,
    AnonymousReuse,
    AnonymousReuseRelaxedMargin,
    AnonymousReuseStrong,
    EmbedFailed,
    ProfileAnchor,
    ProfileAnchorBlocked,
    TooShort,
    UnlabelledAmbiguousNewSpeaker,
    UnlabelledShortOrLowConfidence,
}

impl AssignmentDecision {
    fn as_str(self) -> &'static str {
        match self {
            Self::AnonymousNew => "anonymous_new",
            Self::AnonymousReuse => "anonymous_reuse",
            Self::AnonymousReuseRelaxedMargin => "anonymous_reuse_relaxed_margin",
            Self::AnonymousReuseStrong => "anonymous_reuse_strong",
            Self::EmbedFailed => "embed_failed",
            Self::ProfileAnchor => "profile_anchor",
            Self::ProfileAnchorBlocked => "profile_anchor_blocked",
            Self::TooShort => "too_short",
            Self::UnlabelledAmbiguousNewSpeaker => "unlabelled_ambiguous_new_speaker",
            Self::UnlabelledShortOrLowConfidence => "unlabelled_short_or_low_confidence",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct UtteranceQuality {
    duration_s: f32,
    rms: f32,
    peak: f32,
}

impl UtteranceQuality {
    fn from_samples(samples: &[f32], sample_rate_hz: u32) -> Self {
        let duration_s = samples.len() as f32 / sample_rate_hz as f32;
        if samples.is_empty() {
            return Self {
                duration_s,
                rms: 0.0,
                peak: 0.0,
            };
        }
        let mut sum_square = 0.0_f64;
        let mut peak = 0.0_f32;
        for sample in samples {
            peak = peak.max(sample.abs());
            sum_square += f64::from(*sample * *sample);
        }
        let rms = (sum_square / samples.len() as f64).sqrt() as f32;
        Self {
            duration_s,
            rms,
            peak,
        }
    }

    fn has_audible_speech(self) -> bool {
        self.rms >= MIN_DIARIZE_RMS && self.peak >= MIN_DIARIZE_PEAK
    }

    fn can_create_anonymous(self) -> bool {
        self.duration_s >= MIN_NEW_SPEAKER_DURATION_S && self.has_audible_speech()
    }

    fn can_anchor_profile(self) -> bool {
        self.duration_s >= MIN_PROFILE_ANCHOR_DURATION_S && self.has_audible_speech()
    }

    fn can_update_centroid(self) -> bool {
        self.duration_s >= MIN_CENTROID_UPDATE_DURATION_S && self.has_audible_speech()
    }

    fn can_use_default_reuse_floor(self) -> bool {
        self.can_update_centroid()
    }

    fn centroid_weight(self) -> Option<f32> {
        self.can_update_centroid()
            .then_some(self.duration_s.min(MAX_CENTROID_UPDATE_WEIGHT_S))
    }
}

struct AnonymousAssignment {
    speaker_id: Option<SpeakerId>,
    best_registry_similarity: Option<f32>,
    decision: AssignmentDecision,
}

/// A growing set of session-relative speaker centroids, keyed by arrival order.
#[derive(Debug, Default)]
pub struct SpeakerRegistry {
    speakers: Vec<SpeakerCentroid>,
    next_id: u32,
}

impl SpeakerRegistry {
    fn best_match(
        &self,
        embedding: &SpeakerEmbedding,
        include_profile_owned: bool,
    ) -> Option<RegistryMatch> {
        let mut best: Option<RegistryMatch> = None;
        for (index, speaker) in self.speakers.iter().enumerate() {
            if !include_profile_owned && speaker.owner != SpeakerOwner::Anonymous {
                continue;
            }
            let centroid = SpeakerEmbedding::l2_normalized(speaker.sum.clone());
            if centroid.dim() != embedding.dim() {
                continue;
            }
            let similarity = centroid.cosine(embedding);
            if best
                .map(|candidate| similarity > candidate.similarity)
                .unwrap_or(true)
            {
                best = Some(RegistryMatch {
                    index,
                    speaker_id: speaker.id,
                    similarity,
                });
            }
        }
        best
    }

    fn best_anonymous_match(&self, embedding: &SpeakerEmbedding) -> Option<RegistryMatch> {
        self.best_match(embedding, false)
    }

    /// Highest similarity to any registry centroid (anonymous or
    /// profile-owned) other than `exclude_index`; the relaxed reuse gate uses
    /// it as the runner-up that must trail the best anonymous candidate.
    fn best_similarity_excluding(
        &self,
        embedding: &SpeakerEmbedding,
        exclude_index: usize,
    ) -> Option<f32> {
        let mut best: Option<f32> = None;
        for (index, speaker) in self.speakers.iter().enumerate() {
            if index == exclude_index {
                continue;
            }
            let centroid = SpeakerEmbedding::l2_normalized(speaker.sum.clone());
            if centroid.dim() != embedding.dim() {
                continue;
            }
            let similarity = centroid.cosine(embedding);
            if best.map(|value| similarity > value).unwrap_or(true) {
                best = Some(similarity);
            }
        }
        best
    }

    fn profile_index(&self, profile_id: &str) -> Option<usize> {
        self.speakers.iter().position(|speaker| {
            matches!(&speaker.owner, SpeakerOwner::Profile(existing) if existing == profile_id)
        })
    }

    fn has_profile_anchor(&self, profile_id: &str) -> bool {
        self.profile_index(profile_id).is_some()
    }

    fn allocate(
        &mut self,
        owner: SpeakerOwner,
        embedding: &SpeakerEmbedding,
        weight: f32,
    ) -> SpeakerId {
        let id = SpeakerId(self.next_id);
        self.next_id += 1;
        self.speakers.push(SpeakerCentroid {
            id,
            sum: embedding.0.iter().map(|value| value * weight).collect(),
            weight,
            owner,
        });
        id
    }

    fn update(&mut self, index: usize, embedding: &SpeakerEmbedding, weight: f32) {
        let Some(speaker) = self.speakers.get_mut(index) else {
            return;
        };
        for (acc, value) in speaker.sum.iter_mut().zip(&embedding.0) {
            *acc += value * weight;
        }
        speaker.weight += weight;
    }

    fn assign_profile(
        &mut self,
        profile_id: &str,
        embedding: &SpeakerEmbedding,
        quality: UtteranceQuality,
    ) -> Option<SpeakerId> {
        if let Some(index) = self.profile_index(profile_id) {
            let speaker_id = self.speakers[index].id;
            if let Some(weight) = quality.centroid_weight() {
                self.update(index, embedding, weight);
            }
            return Some(speaker_id);
        }
        if !quality.can_anchor_profile() {
            return None;
        }
        Some(
            self.allocate(
                SpeakerOwner::Profile(profile_id.to_string()),
                embedding,
                quality
                    .centroid_weight()
                    .expect("profile anchor can update"),
            ),
        )
    }

    fn assign_anonymous(
        &mut self,
        embedding: &SpeakerEmbedding,
        quality: UtteranceQuality,
        match_similarity: f32,
        profile: StreamingCalibrationProfile,
    ) -> AnonymousAssignment {
        let best_registry_similarity = self
            .best_match(embedding, true)
            .map(|candidate| candidate.similarity);
        let best_anonymous = self.best_anonymous_match(embedding);
        if let Some(candidate) = best_anonymous {
            let required_similarity = if quality.can_use_default_reuse_floor() {
                match_similarity
            } else {
                profile.strong_existing_match_similarity
            };
            if candidate.similarity >= required_similarity {
                if let Some(weight) = quality.centroid_weight() {
                    self.update(candidate.index, embedding, weight);
                }
                return AnonymousAssignment {
                    speaker_id: Some(candidate.speaker_id),
                    best_registry_similarity,
                    decision: if required_similarity > match_similarity {
                        AssignmentDecision::AnonymousReuseStrong
                    } else {
                        AssignmentDecision::AnonymousReuse
                    },
                };
            }
            // Relaxed margin-based reuse for long, audible turns. Real
            // speaker-playback sessions (video narration over loudspeakers)
            // push WeSpeaker same-speaker cosines well below the normal reuse
            // floor, into the old spawn band, so the same voice fragmented
            // into several anonymous speakers. When the best anonymous
            // centroid still clearly outscores every other registry centroid,
            // reusing it is much safer than spawning or leaving the turn
            // unlabelled; cross-speaker cosines measured on the same sessions
            // stay far below this floor (cross-speaker ≤ 0.24 measured).
            //
            // The `.unwrap_or(true)` below is the lone-centroid case: when the
            // registry holds only one anonymous centroid there is no runner-up,
            // so the margin gate vacuously passes. This is the core fix for
            // users without an enrolled voiceprint (no profile centroid exists
            // to serve as runner-up). The accepted residual risk: a new voice
            // whose cosine against the lone centroid falls in [0.33, 0.56]
            // (the relaxed floor to the normal reuse floor) will be absorbed
            // into the existing speaker rather than spawned as a new one. The
            // calibration evidence (cross-speaker ≤ 0.24 measured) justifies
            // this tradeoff: legitimate cross-speaker scores land well below
            // 0.33, so the residual window is not reachable in practice.
            //
            // To limit drift compounding when a similar new voice is absorbed
            // on a relaxed-reuse hit, the centroid-update weight is capped
            // separately from the normal `MAX_CENTROID_UPDATE_WEIGHT_S` cap.
            if quality.can_create_anonymous()
                && candidate.similarity >= profile.relaxed_match_similarity
                && self
                    .best_similarity_excluding(embedding, candidate.index)
                    .map(|runner_up| {
                        candidate.similarity - runner_up >= profile.relaxed_match_margin
                    })
                    .unwrap_or(true)
            // lone-centroid: no runner-up → gate passes
            {
                if let Some(weight) = quality.centroid_weight() {
                    self.update(
                        candidate.index,
                        embedding,
                        weight.min(profile.relaxed_reuse_max_weight),
                    );
                }
                return AnonymousAssignment {
                    speaker_id: Some(candidate.speaker_id),
                    best_registry_similarity,
                    decision: AssignmentDecision::AnonymousReuseRelaxedMargin,
                };
            }
        }

        if !quality.can_create_anonymous() {
            return AnonymousAssignment {
                speaker_id: None,
                best_registry_similarity,
                decision: AssignmentDecision::UnlabelledShortOrLowConfidence,
            };
        }

        let clearly_new = best_registry_similarity
            .map(|similarity| similarity <= profile.new_speaker_max_existing_similarity)
            .unwrap_or(true);
        if clearly_new {
            let speaker_id = self.allocate(
                SpeakerOwner::Anonymous,
                embedding,
                quality
                    .centroid_weight()
                    .expect("anonymous spawn can update"),
            );
            AnonymousAssignment {
                speaker_id: Some(speaker_id),
                best_registry_similarity,
                decision: AssignmentDecision::AnonymousNew,
            }
        } else {
            AnonymousAssignment {
                speaker_id: None,
                best_registry_similarity,
                decision: AssignmentDecision::UnlabelledAmbiguousNewSpeaker,
            }
        }
    }

    pub fn speaker_count(&self) -> usize {
        self.speakers.len()
    }
}

/// Per-session streaming diarizer over the shared WeSpeaker embedder.
pub struct StreamingDiarizer {
    embedder: &'static dyn SpeakerEmbedder,
    registry: SpeakerRegistry,
    profiles: SpeakerProfileMatcher,
    profile: StreamingCalibrationProfile,
    match_similarity: f32,
    min_samples: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingSpeakerChange {
    pub split_sample: usize,
    pub reference_duration_s: f32,
    pub recent_duration_s: f32,
    pub cosine_similarity: f32,
    pub elapsed_ms: u128,
}

/// Lightweight in-flight speaker-change detector for true-streaming sessions.
///
/// It compares a recent speaker embedding against the segment's first stable
/// speech window. The detector never creates labels itself; callers use a
/// positive decision to split the ASR segment and then run the normal streaming
/// diarizer on each side.
pub struct StreamingSpeakerChangeDetector {
    embedder: &'static dyn SpeakerEmbedder,
    sample_rate_hz: u32,
    reference_window_samples: usize,
    recent_window_samples: usize,
    hop_samples: usize,
    min_total_samples: usize,
    last_analyzed_samples: usize,
    change_max_cosine: f32,
    anchor_embedding: Option<SpeakerEmbedding>,
    anchor_quality: Option<UtteranceQuality>,
}

impl StreamingSpeakerChangeDetector {
    pub fn shared(sample_rate_hz: u32) -> Option<Self> {
        shared_change_detector_embedder()
            .map(|embedder| Self::with_embedder(embedder, sample_rate_hz))
    }

    pub fn with_embedder(embedder: &'static dyn SpeakerEmbedder, sample_rate_hz: u32) -> Self {
        let reference_window_samples =
            (SPEAKER_CHANGE_REFERENCE_WINDOW_S * sample_rate_hz as f32) as usize;
        let recent_window_samples =
            (SPEAKER_CHANGE_RECENT_WINDOW_S * sample_rate_hz as f32) as usize;
        let hop_samples = (SPEAKER_CHANGE_HOP_S * sample_rate_hz as f32) as usize;
        let profile = embedder.calibration_profile().streaming;
        Self {
            embedder,
            sample_rate_hz,
            reference_window_samples,
            recent_window_samples,
            hop_samples,
            min_total_samples: reference_window_samples + recent_window_samples,
            last_analyzed_samples: 0,
            change_max_cosine: profile.speaker_change_max_cosine,
            anchor_embedding: None,
            anchor_quality: None,
        }
    }

    pub fn should_analyze(&self, sample_count: usize) -> bool {
        sample_count >= self.min_total_samples
            && sample_count.saturating_sub(self.last_analyzed_samples) >= self.hop_samples
    }

    pub fn reset(&mut self) {
        self.last_analyzed_samples = 0;
        self.anchor_embedding = None;
        self.anchor_quality = None;
    }

    pub fn analyze(&mut self, samples: &[f32]) -> Option<StreamingSpeakerChange> {
        if !self.should_analyze(samples.len()) {
            return None;
        }
        self.last_analyzed_samples = samples.len();

        let recent_start = samples.len().saturating_sub(self.recent_window_samples);
        if recent_start < self.reference_window_samples {
            return None;
        }
        let reference = &samples[..self.reference_window_samples];
        let recent = &samples[recent_start..];
        let reference_quality = UtteranceQuality::from_samples(reference, self.sample_rate_hz);
        let recent_quality = UtteranceQuality::from_samples(recent, self.sample_rate_hz);
        if !reference_quality.has_audible_speech() || !recent_quality.has_audible_speech() {
            log_change_debug(
                samples.len(),
                self.sample_rate_hz,
                reference_quality,
                recent_quality,
                recent_start,
                None,
                0,
                "skipped_low_quality",
            );
            return None;
        }

        let started_at = Instant::now();
        let reference_embedding = match self.anchor_embedding.clone() {
            Some(embedding) => embedding,
            None => {
                let embedding = match self.embedder.embed(reference, self.sample_rate_hz) {
                    Ok(embedding) => embedding,
                    Err(_) => {
                        log_change_debug(
                            samples.len(),
                            self.sample_rate_hz,
                            reference_quality,
                            recent_quality,
                            recent_start,
                            None,
                            started_at.elapsed().as_millis(),
                            "embed_failed",
                        );
                        return None;
                    }
                };
                self.anchor_embedding = Some(embedding.clone());
                self.anchor_quality = Some(reference_quality);
                embedding
            }
        };
        let reference_quality = self.anchor_quality.unwrap_or(reference_quality);
        let recent_embedding = match self.embedder.embed(recent, self.sample_rate_hz) {
            Ok(embedding) => embedding,
            Err(_) => {
                log_change_debug(
                    samples.len(),
                    self.sample_rate_hz,
                    reference_quality,
                    recent_quality,
                    recent_start,
                    None,
                    started_at.elapsed().as_millis(),
                    "embed_failed",
                );
                return None;
            }
        };
        let elapsed_ms = started_at.elapsed().as_millis();
        let cosine_similarity = reference_embedding.cosine(&recent_embedding);
        let decision = if cosine_similarity <= self.change_max_cosine {
            "change"
        } else {
            "same"
        };
        log_change_debug(
            samples.len(),
            self.sample_rate_hz,
            reference_quality,
            recent_quality,
            recent_start,
            Some(cosine_similarity),
            elapsed_ms,
            decision,
        );
        (cosine_similarity <= self.change_max_cosine).then_some(StreamingSpeakerChange {
            split_sample: recent_start,
            reference_duration_s: reference_quality.duration_s,
            recent_duration_s: recent_quality.duration_s,
            cosine_similarity,
            elapsed_ms,
        })
    }
}

fn shared_change_detector_embedder() -> Option<&'static dyn SpeakerEmbedder> {
    shared_embedder()
}

impl StreamingDiarizer {
    /// Build over the shared embedder, or `None` if the pack is unavailable.
    pub fn shared(sample_rate_hz: u32) -> Option<Self> {
        shared_embedder().map(|embedder| {
            Self::with_embedder_and_profiles(
                embedder,
                sample_rate_hz,
                super::enrollment::load_compatible_profile_matcher_for_active_embedder(),
            )
        })
    }

    /// Build over a caller-supplied embedder (tests, alternative backends).
    pub fn with_embedder(embedder: &'static dyn SpeakerEmbedder, sample_rate_hz: u32) -> Self {
        Self::with_embedder_and_profiles(embedder, sample_rate_hz, SpeakerProfileMatcher::default())
    }

    pub fn with_embedder_and_profiles(
        embedder: &'static dyn SpeakerEmbedder,
        sample_rate_hz: u32,
        profiles: SpeakerProfileMatcher,
    ) -> Self {
        let profile = embedder.calibration_profile().streaming;
        Self {
            embedder,
            registry: SpeakerRegistry::default(),
            profiles,
            profile,
            match_similarity: profile.match_similarity,
            min_samples: (MIN_UTTERANCE_S * sample_rate_hz as f32) as usize,
        }
    }

    /// Embed a finalized utterance's speech-gated audio and return its display
    /// speaker assignment. Raw fallback buffers should be trimmed to VAD speech
    /// before calling this; native streaming already passes speech-gated audio.
    pub fn assign(
        &mut self,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Option<SpeakerDisplayAssignment> {
        self.assign_with_path(samples, sample_rate_hz, StreamingDiarizePath::Direct)
    }

    pub fn assign_with_path(
        &mut self,
        samples: &[f32],
        sample_rate_hz: u32,
        path: StreamingDiarizePath,
    ) -> Option<SpeakerDisplayAssignment> {
        let quality = UtteranceQuality::from_samples(samples, sample_rate_hz);
        if samples.len() < self.min_samples {
            log_debug(
                path,
                quality,
                None,
                None,
                AssignmentDecision::TooShort,
                None,
            );
            return None;
        }

        let embedding = match self.embedder.embed(samples, sample_rate_hz) {
            Ok(embedding) => embedding,
            Err(_) => {
                log_debug(
                    path,
                    quality,
                    None,
                    None,
                    AssignmentDecision::EmbedFailed,
                    None,
                );
                return None;
            }
        };

        let profile_anchor_similarity = match path {
            StreamingDiarizePath::Native => self.profile.native_profile_anchor_similarity,
            StreamingDiarizePath::Direct | StreamingDiarizePath::Fallback => {
                self.profile.profile_anchor_similarity
            }
        };
        let profile_for_log = debug_enabled()
            .then(|| {
                self.profiles
                    .best_similarity_and_threshold(&embedding, profile_anchor_similarity)
            })
            .flatten();
        let strict_profile_match = self.profiles.strong_unambiguous_match(
            &embedding,
            profile_anchor_similarity,
            PROFILE_ANCHOR_MARGIN,
        );
        let profile_match = if strict_profile_match.is_some() {
            strict_profile_match
        } else if matches!(path, StreamingDiarizePath::Native) {
            self.profiles
                .strong_unambiguous_match_with_tolerance(
                    &embedding,
                    profile_anchor_similarity,
                    PROFILE_ANCHOR_MARGIN,
                    PROFILE_ANCHOR_THRESHOLD_TOLERANCE,
                )
                .filter(|profile_match| self.registry.has_profile_anchor(&profile_match.profile_id))
        } else {
            None
        };
        let best_anonymous_similarity = profile_match
            .as_ref()
            .and_then(|_| self.registry.best_anonymous_match(&embedding))
            .map(|candidate| candidate.similarity);
        let profile_beats_anonymous = profile_match
            .as_ref()
            .map(|profile_match| {
                best_anonymous_similarity
                    .map(|anonymous_similarity| {
                        profile_match.similarity
                            >= anonymous_similarity + PROFILE_ANCHOR_ANONYMOUS_MARGIN
                    })
                    .unwrap_or(true)
            })
            .unwrap_or(false);
        if let Some(profile_match) = profile_match.filter(|_| profile_beats_anonymous) {
            let best_registry_similarity = self
                .registry
                .best_match(&embedding, true)
                .map(|candidate| candidate.similarity);
            let Some(speaker_id) =
                self.registry
                    .assign_profile(&profile_match.profile_id, &embedding, quality)
            else {
                log_debug(
                    path,
                    quality,
                    best_registry_similarity,
                    profile_for_log,
                    AssignmentDecision::ProfileAnchorBlocked,
                    None,
                );
                return None;
            };
            let assignment = SpeakerDisplayAssignment::from_match(speaker_id, profile_match);
            log_debug(
                path,
                quality,
                best_registry_similarity,
                profile_for_log,
                AssignmentDecision::ProfileAnchor,
                Some(&assignment),
            );
            return Some(assignment);
        }

        let anonymous = self.registry.assign_anonymous(
            &embedding,
            quality,
            self.match_similarity,
            self.profile,
        );
        let assignment = anonymous
            .speaker_id
            .map(SpeakerDisplayAssignment::anonymous);
        log_debug(
            path,
            quality,
            anonymous.best_registry_similarity,
            profile_for_log,
            anonymous.decision,
            assignment.as_ref(),
        );
        assignment
    }

    pub fn registry(&self) -> &SpeakerRegistry {
        &self.registry
    }
}

fn log_debug(
    path: StreamingDiarizePath,
    quality: UtteranceQuality,
    best_registry_similarity: Option<f32>,
    profile_similarity_and_threshold: Option<(f32, f32)>,
    decision: AssignmentDecision,
    assignment: Option<&SpeakerDisplayAssignment>,
) {
    if !debug_enabled() {
        return;
    }
    let (profile_similarity, profile_threshold) = profile_similarity_and_threshold
        .map(|(similarity, threshold)| (format!("{similarity:.4}"), format!("{threshold:.4}")))
        .unwrap_or_else(|| ("none".to_string(), "none".to_string()));
    let registry_similarity = best_registry_similarity
        .map(|similarity| format!("{similarity:.4}"))
        .unwrap_or_else(|| "none".to_string());
    let speaker_label = assignment
        .map(|assignment| assignment.speaker_label.as_str())
        .unwrap_or("None");
    let speaker_profile_id = assignment
        .and_then(|assignment| assignment.speaker_profile_id.as_deref())
        .unwrap_or("None");
    eprintln!(
        "openasr_diarize_debug path={} duration_s={:.3} rms={:.6} peak={:.6} best_registry_cosine={} profile_cosine={} profile_threshold={} decision={} speaker_label={} speaker_profile_id={}",
        path.as_str(),
        quality.duration_s,
        quality.rms,
        quality.peak,
        registry_similarity,
        profile_similarity,
        profile_threshold,
        decision.as_str(),
        speaker_label,
        speaker_profile_id
    );
}

fn log_change_debug(
    total_samples: usize,
    sample_rate_hz: u32,
    reference_quality: UtteranceQuality,
    recent_quality: UtteranceQuality,
    split_sample: usize,
    cosine_similarity: Option<f32>,
    elapsed_ms: u128,
    decision: &str,
) {
    if !debug_enabled() {
        return;
    }
    let total_s = total_samples as f32 / sample_rate_hz as f32;
    let split_s = split_sample as f32 / sample_rate_hz as f32;
    let cosine = cosine_similarity
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "none".to_string());
    eprintln!(
        "openasr_diarize_change_debug total_s={total_s:.3} split_s={split_s:.3} reference_s={:.3} reference_rms={:.6} recent_s={:.3} recent_rms={:.6} cosine={} elapsed_ms={} decision={}",
        reference_quality.duration_s,
        reference_quality.rms,
        recent_quality.duration_s,
        recent_quality.rms,
        cosine,
        elapsed_ms,
        decision
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarize::embed::EmbedError;
    use crate::diarize::enrollment::SpeakerProfile;

    fn emb(v: Vec<f32>) -> SpeakerEmbedding {
        SpeakerEmbedding::l2_normalized(v)
    }

    fn quality(duration_s: f32) -> UtteranceQuality {
        UtteranceQuality {
            duration_s,
            rms: 0.02,
            peak: 0.1,
        }
    }

    fn test_streaming_profile() -> StreamingCalibrationProfile {
        super::super::calibration::WESPEAKER_CALIBRATION.streaming
    }

    #[test]
    fn arrival_order_labels_and_consistency() {
        let mut reg = SpeakerRegistry::default();
        let a1 = reg
            .assign_anonymous(
                &emb(vec![1.0, 0.0]),
                quality(3.0),
                0.65,
                test_streaming_profile(),
            )
            .speaker_id
            .unwrap();
        let b1 = reg
            .assign_anonymous(
                &emb(vec![0.0, 1.0]),
                quality(3.0),
                0.65,
                test_streaming_profile(),
            )
            .speaker_id
            .unwrap();
        let a2 = reg
            .assign_anonymous(
                &emb(vec![0.95, 0.05]),
                quality(3.0),
                0.65,
                test_streaming_profile(),
            )
            .speaker_id
            .unwrap();
        let b2 = reg
            .assign_anonymous(
                &emb(vec![0.05, 1.0]),
                quality(3.0),
                0.65,
                test_streaming_profile(),
            )
            .speaker_id
            .unwrap();
        assert_eq!(a1, SpeakerId(0));
        assert_eq!(b1, SpeakerId(1));
        assert_eq!(a2, a1, "speaker A stays A");
        assert_eq!(b2, b1, "speaker B stays B");
        assert_eq!(reg.speaker_count(), 2);
    }

    #[test]
    fn short_or_low_confidence_turns_do_not_spawn_new_speakers() {
        let mut reg = SpeakerRegistry::default();
        let profile = test_streaming_profile();
        assert!(
            reg.assign_anonymous(
                &emb(vec![1.0, 0.0]),
                quality(1.2),
                profile.match_similarity,
                profile,
            )
            .speaker_id
            .is_none(),
            "short first turns are left unlabelled instead of creating SPEAKER_00"
        );

        let first = reg
            .assign_anonymous(
                &emb(vec![1.0, 0.0]),
                quality(3.0),
                profile.match_similarity,
                profile,
            )
            .speaker_id
            .unwrap();
        let weak_short = reg.assign_anonymous(
            &emb(vec![0.4, 0.9]),
            quality(0.8),
            profile.match_similarity,
            profile,
        );
        assert_eq!(weak_short.speaker_id, None);

        let strong_short = reg.assign_anonymous(
            &emb(vec![0.99, 0.01]),
            quality(0.8),
            profile.match_similarity,
            profile,
        );
        assert_eq!(strong_short.speaker_id, Some(first));
        assert_eq!(
            strong_short.decision,
            AssignmentDecision::AnonymousReuseStrong
        );
        assert_eq!(reg.speaker_count(), 1);
    }

    #[test]
    fn ambiguous_nonmatch_does_not_spawn_new_speaker() {
        // Two established speakers, and a turn that lands in the dead band
        // close to BOTH of them: no relaxed-margin winner, so the turn stays
        // unlabelled instead of spawning a third speaker.
        let mut reg = SpeakerRegistry::default();
        let profile = test_streaming_profile();
        reg.assign_anonymous(&emb(vec![1.0, 0.0, 0.0]), quality(3.0), 0.65, profile);
        reg.assign_anonymous(&emb(vec![0.0, 1.0, 0.0]), quality(3.0), 0.65, profile);
        let ambiguous =
            reg.assign_anonymous(&emb(vec![0.50, 0.48, 0.7218]), quality(3.0), 0.65, profile);
        assert_eq!(ambiguous.speaker_id, None);
        assert_eq!(
            ambiguous.decision,
            AssignmentDecision::UnlabelledAmbiguousNewSpeaker
        );
        assert_eq!(reg.speaker_count(), 2);
    }

    #[test]
    fn relaxed_margin_reuses_low_band_same_speaker_instead_of_spawning() {
        // Mirrors the real speaker-playback session
        // (tmp/diar-anon-consistency-1781250865.wav): the video narrator's
        // turns score 0.37..0.54 against their own centroid (below the 0.57
        // reuse floor, around the 0.44 spawn ceiling) while the enrolled
        // speaker's centroid trails near zero. The clear margin lets the turn
        // reuse the narrator instead of fragmenting into new speakers.
        let mut reg = SpeakerRegistry::default();
        let profile = test_streaming_profile();
        let narrator = reg
            .assign_anonymous(&emb(vec![1.0, 0.0, 0.0]), quality(4.0), 0.57, profile)
            .speaker_id
            .unwrap();
        reg.assign_profile(
            "vp_aaaaaaaaaaaaaaaa",
            &emb(vec![0.0, 1.0, 0.0]),
            quality(3.0),
        );

        // cosine ≈ 0.40 to the narrator, ≈ 0.05 to the profile centroid.
        let low_band =
            reg.assign_anonymous(&emb(vec![0.40, 0.05, 0.9152]), quality(4.0), 0.57, profile);
        assert_eq!(low_band.speaker_id, Some(narrator));
        assert_eq!(
            low_band.decision,
            AssignmentDecision::AnonymousReuseRelaxedMargin
        );
        assert_eq!(reg.speaker_count(), 2);
    }

    #[test]
    fn relaxed_margin_requires_long_turn() {
        // Short turns keep the strong floor; the relaxed floor never applies.
        let mut reg = SpeakerRegistry::default();
        let profile = test_streaming_profile();
        reg.assign_anonymous(&emb(vec![1.0, 0.0]), quality(4.0), 0.57, profile);
        let short = reg.assign_anonymous(&emb(vec![0.50, 0.8660]), quality(1.5), 0.57, profile);
        assert_eq!(short.speaker_id, None);
        assert_eq!(
            short.decision,
            AssignmentDecision::UnlabelledShortOrLowConfidence
        );
    }

    #[test]
    fn clearly_separated_new_voice_still_spawns_under_relaxed_reuse() {
        // Cross-speaker cosines measured on the real sessions stay at or
        // below ~0.24; a genuinely new voice below the relaxed floor must
        // still get its own speaker.
        let mut reg = SpeakerRegistry::default();
        let profile = test_streaming_profile();
        let first = reg
            .assign_anonymous(&emb(vec![1.0, 0.0]), quality(4.0), 0.57, profile)
            .speaker_id
            .unwrap();
        let second = reg.assign_anonymous(&emb(vec![0.24, 0.9708]), quality(4.0), 0.57, profile);
        assert_ne!(second.speaker_id, Some(first));
        assert_eq!(second.decision, AssignmentDecision::AnonymousNew);
        assert_eq!(reg.speaker_count(), 2);
    }

    #[test]
    fn lone_centroid_relaxed_reuse_absorbs_new_voice_in_residual_window() {
        // Adversarial probe of the documented residual false-merge window:
        // a new synthetic voice at cosine ~0.40 against the lone existing
        // centroid IS absorbed via relaxed-reuse rather than spawned as a
        // new speaker.
        //
        // This is the ACCEPTED tradeoff (not an oversight). When the registry
        // holds only one anonymous centroid there is no runner-up, so the
        // margin gate passes vacuously (`.unwrap_or(true)`). The lone-centroid
        // case is the core fix for users without an enrolled voiceprint.
        // Cross-speaker cosines measured on the same real sessions stay at or
        // below ~0.24, so the residual window [0.33, 0.56] is not reachable
        // by a genuinely different speaker in practice.
        let mut reg = SpeakerRegistry::default();
        let profile = test_streaming_profile();

        // Establish the lone centroid at [1.0, 0.0].
        let first = reg
            .assign_anonymous(&emb(vec![1.0, 0.0]), quality(4.0), 0.57, profile)
            .speaker_id
            .unwrap();

        // A synthetic "new" voice at cosine ≈ 0.40 to the lone centroid:
        // above the relaxed floor (0.33) and below the normal floor (0.57).
        // vec![0.40, ~0.9165] → cosine(lone=[1,0]) ≈ 0.40.
        let residual = reg.assign_anonymous(&emb(vec![0.40, 0.9165]), quality(4.0), 0.57, profile);

        // The turn IS absorbed — pin this behavior so future readers see it
        // was a decision, not a bug.
        assert_eq!(
            residual.speaker_id,
            Some(first),
            "lone-centroid residual window: cosine ~0.40 is absorbed (accepted tradeoff)"
        );
        assert_eq!(
            residual.decision,
            AssignmentDecision::AnonymousReuseRelaxedMargin
        );
        assert_eq!(reg.speaker_count(), 1, "no new speaker is spawned");
    }

    #[test]
    fn speaker_change_detector_splits_anchor_from_recent_speaker() {
        static EMBEDDER: SequenceEmbedder = SequenceEmbedder;
        let mut detector = StreamingSpeakerChangeDetector::with_embedder(&EMBEDDER, 16_000);
        let mut samples = vec![0.1; detector.reference_window_samples];
        samples.extend(vec![-0.1; detector.recent_window_samples]);

        let change = detector
            .analyze(&samples)
            .expect("recent window changed away from the segment anchor");
        assert_eq!(change.split_sample, detector.reference_window_samples);
        assert!(change.cosine_similarity <= detector.change_max_cosine);
    }

    #[test]
    fn speaker_change_detector_does_not_split_same_speaker() {
        static EMBEDDER: SequenceEmbedder = SequenceEmbedder;
        let mut detector = StreamingSpeakerChangeDetector::with_embedder(&EMBEDDER, 16_000);
        let samples = vec![0.1; detector.reference_window_samples + detector.recent_window_samples];

        assert!(detector.analyze(&samples).is_none());
    }

    #[test]
    fn change_detector_uses_wespeaker_calibration() {
        static WESPEAKER: WeSpeakerProfileEmbedder = WeSpeakerProfileEmbedder;

        let detector = StreamingSpeakerChangeDetector::with_embedder(&WESPEAKER, 16_000);
        assert_eq!(
            detector.change_max_cosine,
            super::super::calibration::WESPEAKER_CALIBRATION
                .streaming
                .speaker_change_max_cosine
        );
    }

    #[test]
    fn profile_anchor_gets_stable_session_speaker_before_anonymous_registry() {
        static EMBEDDER: SequenceEmbedder = SequenceEmbedder;
        let matcher = SpeakerProfileMatcher::from_profiles(vec![profile(
            "vp_aaaaaaaaaaaaaaaa",
            "Alice",
            vec![1.0, 0.0],
        )]);
        let mut diarizer =
            StreamingDiarizer::with_embedder_and_profiles(&EMBEDDER, 16_000, matcher);

        let first = diarizer
            .assign(&vec![0.1; 16_000 * 2], 16_000)
            .expect("profile anchor");
        assert_eq!(first.speaker, "Alice");
        assert_eq!(first.speaker_label, "SPEAKER_00");
        assert_eq!(
            first.speaker_profile_id,
            Some("vp_aaaaaaaaaaaaaaaa".to_string())
        );

        let second = diarizer
            .assign(&vec![0.1; 16_000 * 2], 16_000)
            .expect("same profile anchor");
        assert_eq!(second.speaker, "Alice");
        assert_eq!(second.speaker_label, "SPEAKER_00");
        assert_eq!(diarizer.registry().speaker_count(), 1);
    }

    #[test]
    fn native_profile_anchor_requires_strict_session_prior_for_tolerance() {
        static EMBEDDER: NativeProfileSequenceEmbedder = NativeProfileSequenceEmbedder;
        let matcher = SpeakerProfileMatcher::from_profiles(vec![profile(
            "vp_aaaaaaaaaaaaaaaa",
            "Alice",
            vec![1.0, 0.0],
        )]);

        let mut no_prior =
            StreamingDiarizer::with_embedder_and_profiles(&EMBEDDER, 16_000, matcher.clone());
        let anonymous = no_prior
            .assign_with_path(
                &vec![-0.1; 16_000 * 3],
                16_000,
                StreamingDiarizePath::Native,
            )
            .expect("weak first chunk falls back to anonymous assignment");
        assert_eq!(anonymous.speaker, "SPEAKER_00");
        assert_eq!(anonymous.speaker_profile_id, None);

        let mut native =
            StreamingDiarizer::with_embedder_and_profiles(&EMBEDDER, 16_000, matcher.clone());
        let strict = native
            .assign_with_path(&vec![0.1; 16_000 * 2], 16_000, StreamingDiarizePath::Native)
            .expect("strict native match creates the profile anchor");
        assert_eq!(strict.speaker, "Alice");
        assert_eq!(strict.speaker_label, "SPEAKER_00");
        assert_eq!(
            strict.speaker_profile_id,
            Some("vp_aaaaaaaaaaaaaaaa".to_string())
        );

        let tolerated = native
            .assign_with_path(
                &vec![-0.1; 16_000 * 2],
                16_000,
                StreamingDiarizePath::Native,
            )
            .expect("session-prior tolerance reuses the profile anchor");
        assert_eq!(tolerated.speaker, "Alice");
        assert_eq!(tolerated.speaker_label, "SPEAKER_00");
        assert_eq!(
            tolerated.speaker_profile_id,
            Some("vp_aaaaaaaaaaaaaaaa".to_string())
        );
    }

    #[test]
    fn profile_anchor_does_not_capture_stronger_anonymous_centroid() {
        static EMBEDDER: UnitXEmbedder = UnitXEmbedder;
        let matcher = SpeakerProfileMatcher::from_profiles(vec![profile(
            "vp_aaaaaaaaaaaaaaaa",
            "Alice",
            vec![0.86, 0.51029414],
        )]);
        let mut diarizer =
            StreamingDiarizer::with_embedder_and_profiles(&EMBEDDER, 16_000, matcher);
        let anonymous = diarizer
            .registry
            .assign_anonymous(
                &emb(vec![1.0, 0.0]),
                quality(3.0),
                0.65,
                test_streaming_profile(),
            )
            .speaker_id
            .expect("anonymous centroid");

        let assignment = diarizer
            .assign(&vec![0.1; 16_000 * 2], 16_000)
            .expect("anonymous reuse");

        assert_eq!(assignment.speaker_id, anonymous);
        assert_eq!(assignment.speaker, "SPEAKER_00");
        assert_eq!(assignment.speaker_label, "SPEAKER_00");
        assert_eq!(assignment.speaker_profile_id, None);
        assert_eq!(diarizer.registry().speaker_count(), 1);
    }

    #[test]
    fn profile_owned_centroid_blocks_ambiguous_anonymous_spawn() {
        let mut reg = SpeakerRegistry::default();
        reg.assign_profile("vp_aaaaaaaaaaaaaaaa", &emb(vec![1.0, 0.0]), quality(3.0));
        let ambiguous = reg.assign_anonymous(
            &emb(vec![0.55, 0.84]),
            quality(3.0),
            0.65,
            test_streaming_profile(),
        );
        assert_eq!(ambiguous.speaker_id, None);
        assert_eq!(reg.speaker_count(), 1);
    }

    fn profile(id: &str, name: &str, embedding: Vec<f32>) -> SpeakerProfile {
        SpeakerProfile {
            id: id.to_string(),
            name: name.to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
            sample_seconds: 5.0,
            embedding_dim: embedding.len(),
            pack_fingerprint: "sha256:test".to_string(),
            match_similarity: 0.5,
            embedding: SpeakerEmbedding::l2_normalized(embedding).0,
        }
    }

    struct SequenceEmbedder;

    impl SpeakerEmbedder for SequenceEmbedder {
        fn embed(&self, samples: &[f32], _sr: u32) -> Result<SpeakerEmbedding, EmbedError> {
            if samples.first().copied().unwrap_or_default() >= 0.0 {
                Ok(emb(vec![1.0, 0.02]))
            } else {
                Ok(emb(vec![0.0, 1.0]))
            }
        }

        fn embedding_dim(&self) -> usize {
            2
        }
    }

    struct UnitXEmbedder;

    impl SpeakerEmbedder for UnitXEmbedder {
        fn embed(&self, _samples: &[f32], _sr: u32) -> Result<SpeakerEmbedding, EmbedError> {
            Ok(emb(vec![1.0, 0.0]))
        }

        fn embedding_dim(&self) -> usize {
            2
        }
    }

    struct WeSpeakerProfileEmbedder;

    impl SpeakerEmbedder for WeSpeakerProfileEmbedder {
        fn embed(&self, _samples: &[f32], _sr: u32) -> Result<SpeakerEmbedding, EmbedError> {
            Ok(emb(vec![1.0, 0.0]))
        }

        fn embedding_dim(&self) -> usize {
            2
        }

        fn calibration_profile(&self) -> super::super::calibration::SpeakerCalibrationProfile {
            super::super::calibration::WESPEAKER_CALIBRATION
        }
    }

    struct NativeProfileSequenceEmbedder;

    impl SpeakerEmbedder for NativeProfileSequenceEmbedder {
        fn embed(&self, samples: &[f32], _sr: u32) -> Result<SpeakerEmbedding, EmbedError> {
            if samples.first().copied().unwrap_or_default() >= 0.0 {
                Ok(emb(vec![0.784, 0.620_760_8]))
            } else {
                Ok(emb(vec![0.407, 0.913_428_2]))
            }
        }

        fn embedding_dim(&self) -> usize {
            2
        }
    }
}
