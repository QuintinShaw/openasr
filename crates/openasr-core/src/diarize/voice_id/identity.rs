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
//!
//! # Refusing silently is a separate mistake
//!
//! Erring toward anonymous is about the *verdict*; it says nothing about
//! whether the verdict may be invisible. Every gate above therefore reports
//! why it refused, as a [`UnnamedSpeaker`] return value rather than only as an
//! `OPENASR_DIARIZE_DEBUG` line -- see [`naming`](super::naming) for why the
//! three refusal reasons are three different things for a user to do, and why
//! collapsing them into a bare `SPEAKER_01` reads as a broken feature.
//! Reporting the reason is explicitly **not** a licence to lower a gate so the
//! name appears instead.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use super::evidence::{self, JudgedWindows, MIN_PURITY_VERDICT_WINDOWS, WINDOW_SECONDS};
use super::naming::{SpeakerNamingRefusal, UnnamedSpeaker};
use crate::Segment;
use crate::diarize::contract::{SpeakerEmbedding, TimeRange};

/// Fail-closed failures of the identity evidence machinery. Evidence-quality
/// refusals remain anonymous results; model/runtime failures and cancellation
/// are not evidence judgements and must never be silently converted into one.
#[derive(Debug, Error)]
pub enum SpeakerIdentityError {
    #[error("{}", crate::diarize::embed::VOICE_ID_NAMING_EMBEDDER_MISSING_REASON)]
    EmbedderPackMissing,
    #[error("Voice ID speaker embedding failed: {reason}")]
    EmbeddingFailed { reason: String },
    #[error("Voice ID person library is unavailable: {0}")]
    LibraryUnavailable(#[from] super::VoiceIdLibraryError),
    #[error("Voice ID decode-scope provenance is invalid: {reason}")]
    InvalidScopeProvenance { reason: String },
    #[error("Voice ID speaker embedding was canceled")]
    Canceled,
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
///
/// # Return value
///
/// Every label that came out anonymous, with the reason (see
/// [`naming`](super::naming)). Named labels are absent -- they carry their
/// person on every segment already. An empty vector therefore means "every
/// speaker in this transcript has a name", and is the only shape a
/// single-speaker recording of an enrolled person produces.
pub fn name_speakers_across_scopes(
    embedder: Option<&dyn crate::diarize::embed::SpeakerEmbedder>,
    scopes: &mut [SpeakerScope<'_>],
) -> Result<Vec<UnnamedSpeaker>, SpeakerIdentityError> {
    name_speakers_across_scopes_with(embedder, scopes)
}

/// Request-scoped variant used after transcription preflight has frozen an
/// exact embedding-space snapshot.
pub(crate) fn name_speakers_across_scopes_with_embedder(
    embedder: &dyn crate::diarize::embed::SpeakerEmbedder,
    scopes: &mut [SpeakerScope<'_>],
) -> Result<Vec<UnnamedSpeaker>, SpeakerIdentityError> {
    name_speakers_across_scopes_with(Some(embedder), scopes)
}

/// [`name_speakers_across_scopes`] with the embedder passed explicitly.
///
/// The public boundary passes its policy-resolved, admitted embedder here.
/// This seam resolves the persistent library state once and then hands every
/// external availability input to the pure core below, so tests can exercise
/// both sides of the missing-embedder contract without inheriting a user's
/// installed packs or Voice ID database.
fn name_speakers_across_scopes_with(
    embedder: Option<&dyn crate::diarize::embed::SpeakerEmbedder>,
    scopes: &mut [SpeakerScope<'_>],
) -> Result<Vec<UnnamedSpeaker>, SpeakerIdentityError> {
    name_speakers_across_scopes_with_library_state(
        embedder,
        super::person_library_is_non_empty()?,
        scopes,
    )
}

/// Pure identity-resolution core with every external availability input made
/// explicit. Production resolves the library state once at the boundary;
/// tests inject it so their result cannot depend on a user's real Voice ID
/// database.
fn name_speakers_across_scopes_with_library_state(
    embedder: Option<&dyn crate::diarize::embed::SpeakerEmbedder>,
    person_library_non_empty: bool,
    scopes: &mut [SpeakerScope<'_>],
) -> Result<Vec<UnnamedSpeaker>, SpeakerIdentityError> {
    for scope in scopes.iter_mut() {
        normalize_local_labels(scope.segments);
    }
    if scopes.len() > 1 {
        disambiguate_labels_across_scopes(scopes);
    }
    let Some(embedder) = embedder else {
        if scopes.len() > 1 || person_library_non_empty {
            return Err(SpeakerIdentityError::EmbedderPackMissing);
        }
        // No embedder, nobody enrolled, one scope: separation stands, naming
        // was never going to do anything here regardless of the embedder.
        // Nothing was measured, so every label is unnamed for that reason and
        // not for any judgement about its voice -- the one refusal that
        // describes a missing pack rather than thin evidence.
        return Ok(unnamed_speakers(scopes, &BTreeMap::new(), |_| {
            SpeakerNamingRefusal::EmbedderUnavailable
        }));
    };

    // Why each label stayed anonymous, keyed by the label it was judged under.
    // Filled as each gate refuses; a label that clears every gate and matches a
    // person is removed again below.
    let mut refusals: BTreeMap<String, SpeakerNamingRefusal> = BTreeMap::new();
    // Keyed by the label as it now reads, which is scope-unique after the
    // disambiguation pass above -- so evidence from two scopes is never pooled
    // into one centroid unless a caller genuinely produced one scope.
    let mut evidence: BTreeMap<String, LabelEvidence> = BTreeMap::new();
    for (scope_index, scope) in scopes.iter().enumerate() {
        for (label, windows) in evidence::plan_label_windows(scope.segments) {
            let planned_windows = windows.len();
            let mut clips = Vec::with_capacity(windows.len());
            let mut valid_windows = Vec::with_capacity(windows.len());
            for window in windows {
                let Some(clip) = window_clip(&window, scope.samples) else {
                    continue;
                };
                clips.push(clip);
                valid_windows.push(window);
            }
            let results = embedder.embed_batch(&clips, EMBEDDER_SAMPLE_RATE_HZ as u32);
            if results.len() != valid_windows.len() {
                return Err(SpeakerIdentityError::EmbeddingFailed {
                    reason: format!(
                        "embedder returned {} results for {} evidence windows",
                        results.len(),
                        valid_windows.len()
                    ),
                });
            }
            let mut embeddings = Vec::with_capacity(valid_windows.len());
            let mut spans = Vec::with_capacity(valid_windows.len());
            for (window, result) in valid_windows.into_iter().zip(results) {
                match result {
                    Ok(embedding) => {
                        embeddings.push(embedding);
                        spans.push(window);
                    }
                    Err(crate::diarize::embed::EmbedError::TooShort) => {}
                    Err(crate::diarize::embed::EmbedError::Canceled) => {
                        return Err(SpeakerIdentityError::Canceled);
                    }
                    Err(error) => {
                        return Err(SpeakerIdentityError::EmbeddingFailed {
                            reason: error.to_string(),
                        });
                    }
                }
            }
            if embeddings.is_empty() {
                log_naming_debug(format_args!(
                    "stage=voice-id-evidence label={label} scope={scope_index} planned_windows={planned_windows} embedded_windows=0 decision=no-usable-window"
                ));
                // Same user-facing answer as failing the window/second gates
                // below, reached one step earlier: the label produced windows
                // the audio could not back. Reported rather than skipped so a
                // label with no evidence at all is not the one case that
                // silently vanishes from the report.
                refusals.insert(label, not_enough_speech(0, 0.0));
                continue;
            }
            let entry = LabelEvidence::from_windows(
                evidence::judge_windows(&embeddings, &spans),
                scope_index,
            );
            log_naming_debug(format_args!(
                "stage=voice-id-evidence label={label} scope={scope_index} planned_windows={planned_windows} embedded_windows={} kept_windows={} kept_seconds={:.2} single_voice={} min_windows={MIN_PURITY_VERDICT_WINDOWS} min_seconds={MIN_NAMING_EVIDENCE_SECONDS:.2} naming_evidence={}",
                embeddings.len(),
                entry.kept.len(),
                entry.kept_seconds,
                entry.single_voice,
                if entry.centroid_for_naming().is_some() {
                    "accepted"
                } else {
                    "refused"
                },
            ));
            evidence.insert(label, entry);
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

    let identity = embedder
        .identity()
        .ok_or(super::VoiceIdLibraryError::EmbedderIdentityUnavailable)?;
    let matcher = super::load_person_matcher_for_embedder(&identity, embedder)?;
    let mut matches: BTreeMap<String, super::PersonMatch> = BTreeMap::new();
    for (label, evidence) in evidence {
        let Some(centroid) = evidence.centroid_for_naming() else {
            refusals.insert(label, evidence.refusal());
            continue;
        };
        let matched = matcher.best_match(&centroid);
        let scored = matcher.best_score_and_threshold(&centroid);
        if crate::diarize::debug::diarize_debug_enabled() {
            let (best, threshold) = scored
                .map(|(score, threshold)| (format!("{score:.4}"), format!("{threshold:.4}")))
                .unwrap_or_else(|| ("n/a".to_string(), "n/a".to_string()));
            log_naming_debug(format_args!(
                "stage=voice-id-match label={label} library_empty={} best_score={best} accept_threshold={threshold} decision={}",
                matcher.is_empty(),
                match &matched {
                    Some(person) => format!("named:{}", person.person_id.as_str()),
                    None => "anonymous".to_string(),
                },
            ));
        }
        match matched {
            Some(matched) => {
                matches.insert(label, matched);
            }
            None => {
                refusals.insert(
                    label,
                    SpeakerNamingRefusal::NoMatchInLibrary {
                        library_empty: matcher.is_empty(),
                        best_score: scored.map(|(score, _)| score),
                        accept_threshold: scored.map(|(_, threshold)| threshold),
                    },
                );
            }
        }
    }

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
    Ok(unnamed_speakers(scopes, &refusals, |_| {
        // A label the evidence pass never saw: it carries segments but
        // produced no turn long enough to plan a window over, which is the
        // shortest-speech end of the same gate.
        not_enough_speech(0, 0.0)
    }))
}

/// The labels a finished transcript still spells anonymously, in label order,
/// each with the reason recorded for it (or `fallback` when no gate recorded
/// one).
///
/// Read from the segments rather than from the evidence map on purpose: the
/// segments are what a caller renders, so a label only appears here if the user
/// can actually see it, and it appears under exactly the spelling the user
/// sees -- after scope disambiguation and stitching have had their say.
fn unnamed_speakers(
    scopes: &[SpeakerScope<'_>],
    refusals: &BTreeMap<String, SpeakerNamingRefusal>,
    fallback: impl Fn(&str) -> SpeakerNamingRefusal,
) -> Vec<UnnamedSpeaker> {
    let mut labels: BTreeSet<&str> = BTreeSet::new();
    for scope in scopes {
        for segment in scope.segments.iter() {
            if segment.speaker_person_id.is_some() {
                continue;
            }
            if let Some(label) = segment.speaker_label.as_deref() {
                labels.insert(label);
            }
        }
    }
    labels
        .into_iter()
        .map(|label| UnnamedSpeaker {
            label: label.to_string(),
            reason: refusals
                .get(label)
                .cloned()
                .unwrap_or_else(|| fallback(label)),
        })
        .collect()
}

/// The "this voice did not talk for long enough" refusal, stated against the
/// two gates [`LabelEvidence::centroid_for_naming`] applies.
fn not_enough_speech(windows: usize, seconds: f64) -> SpeakerNamingRefusal {
    SpeakerNamingRefusal::NotEnoughSpeech {
        windows,
        required_windows: MIN_PURITY_VERDICT_WINDOWS,
        seconds,
        required_seconds: MIN_NAMING_EVIDENCE_SECONDS,
        required_continuous_seconds: MIN_CONTINUOUS_SPEECH_SECONDS_FOR_NAMING,
    }
}

/// The shortest single uninterrupted turn that can clear naming's two gates.
///
/// Derived here, once, from the geometry that actually decides it, because
/// every consumer that tried to state the requirement from a single threshold
/// got it wrong: [`MIN_NAMING_EVIDENCE_SECONDS`] is 3.0 but is *not* the
/// binding gate, so "speak for 3 seconds" is advice that fails on the retry.
///
/// Three things stack up, and leaving any of them out understates the answer:
///
/// - naming counts the windows that *survive* main-cluster filtering, and a
///   turn has to plan one window more than the gate asks for to cover the
///   case where that filtering still discards one (see below -- this is no
///   longer the *typical* case, but it remains the one this figure has to
///   survive);
/// - `n` overlapping windows span
///   `WINDOW_SECONDS + (n - 1) * WINDOW_STEP_SECONDS` of audio;
/// - [`evidence::SEGMENT_EDGE_TRIM_SECONDS`] is charged at both ends of a turn
///   before windowing starts, so the turn must be that much longer again.
///
/// # Why the "+1" margin is still charged, even though reclaim exists
///
/// Main-cluster filtering used to discard a minority unconditionally, so a
/// planned-windows-minus-one margin was simply correct. It now reclaims that
/// minority whenever the split lands at or under `evidence`'s
/// `MIXED_MIN_SPLIT_DISTANCE` -- which real single-voice turns almost always
/// do (measured 0.055-0.253 there) -- so *most* single-voice labels no longer
/// lose anything here. The margin stays anyway, because reclaim is
/// conditional on distance alone, not on label size, and a real single voice
/// can still land above that distance by acoustic condition rather than
/// identity (`evidence`'s module docs cite `R8001_M8004` `SPEAKER_00#0`: 95%
/// one voice, split 0.80, never reclaimed). At the exact planned-window count
/// this constant assumes (six), that particular escape does not exist yet --
/// a six-window split can only be ruled single-voice through the same
/// distance check that reclaims it, since one window out of six is already
/// above the fraction floor that would otherwise excuse a large split
/// distance -- so the bare minimum turn this constant describes clears with
/// reclaim doing the work, not the margin. The margin is what keeps the
/// *promise* correct for longer single-voice labels, where a large-enough
/// window budget lets the fraction floor protect a label whose split distance
/// reclaim does not reach. Removing it would make this figure a claim that
/// happens to hold at the minimum length tested and silently stops holding
/// past it -- exactly the kind of promise this constant exists to rule out.
///
/// The seconds gate is folded in with a `max` rather than assumed smaller, so
/// retuning either constant keeps this honest, and
/// `the_advertised_speech_length_is_long_enough_to_actually_clear_both_gates`
/// runs the resulting figure through the real windowing rather than trusting
/// this arithmetic --
/// `naming_still_clears_a_longer_single_voice_label_whose_minority_goes_unreclaimed`
/// does the same for the case the margin exists for, where main-cluster
/// filtering still discards a window despite a true single voice.
const MIN_CONTINUOUS_SPEECH_SECONDS_FOR_NAMING: f64 = {
    let planned_windows = MIN_PURITY_VERDICT_WINDOWS as f64 + 1.0;
    let windowed_span = WINDOW_SECONDS + (planned_windows - 1.0) * evidence::WINDOW_STEP_SECONDS;
    let evidence_span = if windowed_span > MIN_NAMING_EVIDENCE_SECONDS {
        windowed_span
    } else {
        MIN_NAMING_EVIDENCE_SECONDS
    };
    evidence_span + 2.0 * evidence::SEGMENT_EDGE_TRIM_SECONDS
};

/// Single-scope convenience for the one caller that has a single decode unit
/// (every offline transcription today). Same semantics as
/// [`name_speakers_across_scopes`] with one scope.
pub fn name_speakers_from_labeled_segments(
    embedder: Option<&dyn crate::diarize::embed::SpeakerEmbedder>,
    segments: &mut [Segment],
    samples: &[f32],
) -> Result<Vec<UnnamedSpeaker>, SpeakerIdentityError> {
    name_speakers_across_scopes(embedder, &mut [SpeakerScope { segments, samples }])
}

pub(crate) fn name_speakers_from_labeled_segments_with_embedder(
    embedder: &dyn crate::diarize::embed::SpeakerEmbedder,
    segments: &mut [Segment],
    samples: &[f32],
) -> Result<Vec<UnnamedSpeaker>, SpeakerIdentityError> {
    name_speakers_across_scopes_with_embedder(embedder, &mut [SpeakerScope { segments, samples }])
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

    /// Which gate of [`Self::centroid_for_naming`] refused, for a label that
    /// has evidence but no naming centroid.
    ///
    /// The quantity gates are reported ahead of the purity verdict because
    /// below [`MIN_PURITY_VERDICT_WINDOWS`] no verdict is claimed at all (see
    /// [`evidence`]), so `single_voice` there is an absence of evidence rather
    /// than a finding of two voices -- reporting "we heard two people" off it
    /// would be inventing a fact. Above the quantity gates the verdict is real
    /// and is the honest answer.
    fn refusal(&self) -> SpeakerNamingRefusal {
        if self.kept.len() < MIN_PURITY_VERDICT_WINDOWS
            || self.kept_seconds < MIN_NAMING_EVIDENCE_SECONDS
        {
            return not_enough_speech(self.kept.len(), self.kept_seconds);
        }
        if !self.single_voice {
            return SpeakerNamingRefusal::MixedVoices {
                windows: self.kept.len(),
                seconds: self.kept_seconds,
            };
        }
        // Every gate passed and the centroid still came out empty: the kept
        // embeddings could not be averaged (a zero-length or degenerate
        // embedding space). Nothing was measurable, so say that rather than
        // blaming the speaker's audio.
        SpeakerNamingRefusal::EmbedderUnavailable
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

/// Trace one naming decision on stderr under the shared `OPENASR_DIARIZE_DEBUG`
/// gate.
///
/// Every gate in this module is deliberately silent toward the transcript (see
/// the module docs), which leaves "a familiar voice was not named" and "the
/// voice was never even measured" indistinguishable from the outside. These
/// lines are the only place that difference is observable, so they report the
/// evidence a decision was made on -- window counts, seconds, the similarity
/// and the threshold it was compared against -- not just the verdict.
fn log_naming_debug(fields: std::fmt::Arguments<'_>) {
    if !crate::diarize::debug::diarize_debug_enabled() {
        return;
    }
    eprintln!("openasr_diarize_debug {fields}");
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

    fn name_without_embedder_or_enrollment(
        segments: &mut [Segment],
        samples: &[f32],
    ) -> Result<Vec<UnnamedSpeaker>, SpeakerIdentityError> {
        name_speakers_across_scopes_with_library_state(
            None,
            false,
            &mut [SpeakerScope { segments, samples }],
        )
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
        name_without_embedder_or_enrollment(&mut segments, &[]).unwrap();

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
        name_without_embedder_or_enrollment(&mut segments, &[]).unwrap();
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
        name_without_embedder_or_enrollment(&mut segments, &[]).unwrap();
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
        let result = name_speakers_across_scopes_with_library_state(
            None,
            false,
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
        let result = name_speakers_across_scopes_with_library_state(
            None,
            false,
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

    /// A constant, deterministic embedder so the naming path can be exercised
    /// end to end without a model pack. Every window embeds to the same voice,
    /// which is the point: the refusals under test are about *how much* voice
    /// there is, never about which one.
    struct OneVoiceEmbedder;

    fn deterministic_test_embedder_identity() -> crate::diarize::embed::SpeakerEmbedderIdentity {
        crate::diarize::embed::SpeakerEmbedderIdentity {
            embedding_dim: 2,
            pack_fingerprint: "voice-id-identity-tests-v1".to_string(),
        }
    }

    impl crate::diarize::embed::SpeakerEmbedder for OneVoiceEmbedder {
        fn embed(
            &self,
            _samples: &[f32],
            _sample_rate_hz: u32,
        ) -> Result<SpeakerEmbedding, crate::diarize::embed::EmbedError> {
            Ok(SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]))
        }

        fn embedding_dim(&self) -> usize {
            2
        }

        fn identity(&self) -> Option<crate::diarize::embed::SpeakerEmbedderIdentity> {
            Some(deterministic_test_embedder_identity())
        }
    }

    fn one_voice_embedder() -> &'static dyn crate::diarize::embed::SpeakerEmbedder {
        &OneVoiceEmbedder
    }

    fn one_voice_or_marked_too_short(
        samples: &[f32],
    ) -> Result<SpeakerEmbedding, crate::diarize::embed::EmbedError> {
        if samples.first().copied().unwrap_or(0.0) == -1.0 {
            Err(crate::diarize::embed::EmbedError::TooShort)
        } else {
            Ok(SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]))
        }
    }

    struct DefaultMarkedTooShortEmbedder;

    impl crate::diarize::embed::SpeakerEmbedder for DefaultMarkedTooShortEmbedder {
        fn embed(
            &self,
            samples: &[f32],
            _sample_rate_hz: u32,
        ) -> Result<SpeakerEmbedding, crate::diarize::embed::EmbedError> {
            one_voice_or_marked_too_short(samples)
        }

        fn embedding_dim(&self) -> usize {
            2
        }

        fn identity(&self) -> Option<crate::diarize::embed::SpeakerEmbedderIdentity> {
            Some(deterministic_test_embedder_identity())
        }
    }

    struct BatchProbeEmbedder {
        batch_calls: std::sync::atomic::AtomicUsize,
        single_calls: std::sync::atomic::AtomicUsize,
        failures: std::sync::atomic::AtomicUsize,
        observed_starts: std::sync::Mutex<Vec<f32>>,
    }

    impl BatchProbeEmbedder {
        fn new() -> Self {
            Self {
                batch_calls: std::sync::atomic::AtomicUsize::new(0),
                single_calls: std::sync::atomic::AtomicUsize::new(0),
                failures: std::sync::atomic::AtomicUsize::new(0),
                observed_starts: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl crate::diarize::embed::SpeakerEmbedder for BatchProbeEmbedder {
        fn embed(
            &self,
            samples: &[f32],
            _sample_rate_hz: u32,
        ) -> Result<SpeakerEmbedding, crate::diarize::embed::EmbedError> {
            self.single_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            one_voice_or_marked_too_short(samples)
        }

        fn embed_batch(
            &self,
            clips: &[&[f32]],
            _sample_rate_hz: u32,
        ) -> Vec<Result<SpeakerEmbedding, crate::diarize::embed::EmbedError>> {
            self.batch_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.observed_starts
                .lock()
                .expect("observed starts")
                .extend(clips.iter().map(|clip| clip[0]));
            clips
                .iter()
                .map(|clip| {
                    let result = one_voice_or_marked_too_short(clip);
                    if result.is_err() {
                        self.failures
                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                    result
                })
                .collect()
        }

        fn embedding_dim(&self) -> usize {
            2
        }

        fn identity(&self) -> Option<crate::diarize::embed::SpeakerEmbedderIdentity> {
            Some(deterministic_test_embedder_identity())
        }
    }

    #[test]
    fn voice_id_evidence_batch_matches_single_path_and_preserves_window_order() {
        let seconds = 12.0_f32;
        let sample_count = (seconds * EMBEDDER_SAMPLE_RATE_HZ as f32) as usize;
        let samples: Vec<f32> = (0..sample_count).map(|index| index as f32).collect();
        let mut single_segments = vec![labeled(0.0, seconds, Some("SPEAKER_01"))];
        let single = name_speakers_across_scopes_with(
            Some(one_voice_embedder()),
            &mut [SpeakerScope {
                segments: &mut single_segments,
                samples: &samples,
            }],
        )
        .expect("single path");

        let batch_embedder = BatchProbeEmbedder::new();
        let mut batch_segments = vec![labeled(0.0, seconds, Some("SPEAKER_01"))];
        let batch = name_speakers_across_scopes_with(
            Some(&batch_embedder),
            &mut [SpeakerScope {
                segments: &mut batch_segments,
                samples: &samples,
            }],
        )
        .expect("batch path");

        assert_eq!(batch, single);
        assert_eq!(batch_segments[0].speaker, single_segments[0].speaker);
        assert_eq!(
            batch_segments[0].speaker_label,
            single_segments[0].speaker_label
        );
        assert_eq!(
            batch_embedder
                .batch_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert_eq!(
            batch_embedder
                .single_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "identity evidence bypassed the batch seam"
        );
        let starts = batch_embedder
            .observed_starts
            .lock()
            .expect("observed starts");
        assert!(starts.len() > 1);
        assert!(starts.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn voice_id_evidence_batch_skips_only_one_too_short_window_like_single_path() {
        let seconds = 30.0_f32;
        let mut samples = vec![0.0_f32; (seconds * EMBEDDER_SAMPLE_RATE_HZ as f32) as usize];
        let first_window_start =
            (evidence::SEGMENT_EDGE_TRIM_SECONDS * EMBEDDER_SAMPLE_RATE_HZ as f64) as usize;
        samples[first_window_start] = -1.0;

        let mut single_segments = vec![labeled(0.0, seconds, Some("SPEAKER_01"))];
        let single = name_speakers_across_scopes_with(
            Some(&DefaultMarkedTooShortEmbedder),
            &mut [SpeakerScope {
                segments: &mut single_segments,
                samples: &samples,
            }],
        )
        .expect("single too-short path");

        let batch_embedder = BatchProbeEmbedder::new();
        let mut batch_segments = vec![labeled(0.0, seconds, Some("SPEAKER_01"))];
        let batch = name_speakers_across_scopes_with(
            Some(&batch_embedder),
            &mut [SpeakerScope {
                segments: &mut batch_segments,
                samples: &samples,
            }],
        )
        .expect("batch too-short path");

        assert_eq!(batch, single);
        assert_eq!(batch_segments[0].speaker, single_segments[0].speaker);
        assert_eq!(
            batch_segments[0].speaker_label,
            single_segments[0].speaker_label
        );
        assert_eq!(
            batch_embedder
                .failures
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "exactly the marked window must fail"
        );
        assert!(matches!(
            batch[0].reason,
            SpeakerNamingRefusal::NoMatchInLibrary {
                library_empty: true,
                ..
            }
        ));
    }

    struct RuntimeFailureEmbedder {
        canceled: bool,
    }

    impl crate::diarize::embed::SpeakerEmbedder for RuntimeFailureEmbedder {
        fn embed(
            &self,
            _samples: &[f32],
            _sample_rate_hz: u32,
        ) -> Result<SpeakerEmbedding, crate::diarize::embed::EmbedError> {
            if self.canceled {
                Err(crate::diarize::embed::EmbedError::Canceled)
            } else {
                Err(crate::diarize::embed::EmbedError::Unavailable(
                    "test runtime unavailable".to_string(),
                ))
            }
        }

        fn embedding_dim(&self) -> usize {
            2
        }
    }

    #[test]
    fn voice_id_evidence_runtime_failure_is_not_misreported_as_thin_evidence() {
        let (mut segments, samples) = one_speaker_scope(12.0);
        let error = name_speakers_across_scopes_with(
            Some(&RuntimeFailureEmbedder { canceled: false }),
            &mut [SpeakerScope {
                segments: &mut segments,
                samples: &samples,
            }],
        )
        .expect_err("a model/runtime failure must fail closed");
        assert!(matches!(
            error,
            SpeakerIdentityError::EmbeddingFailed { reason }
                if reason.contains("test runtime unavailable")
        ));
    }

    #[test]
    fn voice_id_evidence_cancellation_stays_typed() {
        let (mut segments, samples) = one_speaker_scope(12.0);
        let error = name_speakers_across_scopes_with(
            Some(&RuntimeFailureEmbedder { canceled: true }),
            &mut [SpeakerScope {
                segments: &mut segments,
                samples: &samples,
            }],
        )
        .expect_err("cancellation must stop identity finalization");
        assert!(matches!(error, SpeakerIdentityError::Canceled));
    }

    /// One voice everywhere, except a window whose clip happens to start on a
    /// marked sample, which embeds as a different voice instead.
    ///
    /// Models the real worst case `MIN_CONTINUOUS_SPEECH_SECONDS_FOR_NAMING`'s
    /// margin defends: a genuine single voice whose windows still split, by
    /// acoustic condition rather than identity, into a minority large-cluster
    /// filtering keeps discarding. A single marked sample (rather than a
    /// marked range) is enough and stays unambiguous even though windows
    /// overlap: only the one window whose clip *starts* exactly on the mark
    /// sees it as its first sample, because no other window starts there.
    struct OneVoiceWithOneUnreclaimedOutlierWindow;

    impl crate::diarize::embed::SpeakerEmbedder for OneVoiceWithOneUnreclaimedOutlierWindow {
        fn embed(
            &self,
            samples: &[f32],
            _sample_rate_hz: u32,
        ) -> Result<SpeakerEmbedding, crate::diarize::embed::EmbedError> {
            Ok(if samples.first().copied().unwrap_or(0.0) > 0.5 {
                SpeakerEmbedding::l2_normalized(vec![0.0, 1.0])
            } else {
                SpeakerEmbedding::l2_normalized(vec![1.0, 0.0])
            })
        }

        fn embedding_dim(&self) -> usize {
            2
        }

        fn identity(&self) -> Option<crate::diarize::embed::SpeakerEmbedderIdentity> {
            Some(deterministic_test_embedder_identity())
        }
    }

    /// One label spanning `seconds` of continuous speech, with matching audio.
    fn one_speaker_scope(seconds: f32) -> (Vec<Segment>, Vec<f32>) {
        let segments = vec![labeled(0.0, seconds, Some("SPEAKER_01"))];
        let samples = vec![0.0_f32; (seconds * EMBEDDER_SAMPLE_RATE_HZ as f32) as usize];
        (segments, samples)
    }

    /// The reported case: a clip too short to produce the windows a name needs.
    /// The refusal itself is correct and stays -- what must not happen is the
    /// caller being told only "SPEAKER_01" with no way to distinguish this from
    /// a broken feature.
    ///
    /// 3.6s plans exactly one window (a second window needs 4.5s here), which
    /// keeps this fixture meaningful regardless of main-cluster reclaim: with
    /// only one window there is nothing for the AHC cut to split, so this is
    /// the "never even reached two windows" shape of the refusal, not the
    /// "lost one to the split" shape -- that one is exercised in
    /// `naming_still_clears_a_longer_single_voice_label_whose_minority_goes_unreclaimed`.
    #[test]
    fn a_clip_too_short_to_judge_reports_how_short_it_was() {
        let (mut segments, samples) = one_speaker_scope(3.6);
        let unnamed = name_speakers_across_scopes_with(
            Some(one_voice_embedder()),
            &mut [SpeakerScope {
                segments: &mut segments,
                samples: &samples,
            }],
        )
        .unwrap();

        assert_eq!(unnamed.len(), 1, "{unnamed:?}");
        assert_eq!(unnamed[0].label, "SPEAKER_01");
        let SpeakerNamingRefusal::NotEnoughSpeech {
            windows,
            required_windows,
            seconds,
            required_seconds,
            required_continuous_seconds,
        } = unnamed[0].reason
        else {
            panic!("expected a not-enough-speech refusal, got {:?}", unnamed[0]);
        };
        assert!(
            windows < required_windows,
            "{windows} vs {required_windows}"
        );
        assert_eq!(required_windows, MIN_PURITY_VERDICT_WINDOWS);
        assert_eq!(required_seconds, MIN_NAMING_EVIDENCE_SECONDS);
        assert!(
            seconds < required_seconds,
            "{seconds} vs {required_seconds}"
        );
        // The number a user is told to act on has to be one that works: the
        // clip they were just refused for is 3.6s, so advice built from the
        // 3.0s gate would send them back to fail again.
        assert!(
            required_continuous_seconds > 3.6,
            "advice of {required_continuous_seconds}s would not have saved this clip"
        );
        // The gate is untouched: still anonymous, still no invented person.
        assert_eq!(segments[0].speaker_label.as_deref(), Some("SPEAKER_01"));
        assert!(segments[0].speaker_person_id.is_none());
    }

    /// Plenty of speech, nobody enrolled: a different situation with a
    /// different remedy, and it has to read differently.
    #[test]
    fn a_long_clip_with_an_empty_library_reports_that_nobody_matched() {
        let (mut segments, samples) = one_speaker_scope(30.0);
        let unnamed = name_speakers_across_scopes_with(
            Some(one_voice_embedder()),
            &mut [SpeakerScope {
                segments: &mut segments,
                samples: &samples,
            }],
        )
        .unwrap();

        assert_eq!(unnamed.len(), 1, "{unnamed:?}");
        assert!(
            matches!(
                unnamed[0].reason,
                SpeakerNamingRefusal::NoMatchInLibrary {
                    library_empty: true,
                    ..
                }
            ),
            "{:?}",
            unnamed[0]
        );
    }

    /// No embedder at all is the one refusal that describes a malfunction, and
    /// it must not be dressed up as a judgement about the speaker's audio.
    #[test]
    fn a_missing_embedder_is_reported_as_a_missing_embedder() {
        let mut segments = vec![labeled(0.0, 30.0, Some("SPEAKER_01"))];
        let unnamed = name_speakers_across_scopes_with_library_state(
            None,
            false,
            &mut [SpeakerScope {
                segments: &mut segments,
                samples: &[],
            }],
        )
        .unwrap();

        assert_eq!(
            unnamed,
            vec![UnnamedSpeaker {
                label: "SPEAKER_01".to_string(),
                reason: SpeakerNamingRefusal::EmbedderUnavailable,
            }]
        );
    }

    /// The user-facing "speak for this long" figure has to be the one that
    /// actually clears the gates, not the smaller threshold that does not.
    ///
    /// A speaker who follows advice derived from `MIN_NAMING_EVIDENCE_SECONDS`
    /// alone produces too few windows and is refused a second time, so this
    /// pins the derivation against the geometry that decides it.
    #[test]
    fn the_advertised_speech_length_is_long_enough_to_actually_clear_both_gates() {
        let advertised = MIN_CONTINUOUS_SPEECH_SECONDS_FOR_NAMING;
        assert!(advertised > MIN_NAMING_EVIDENCE_SECONDS);

        // One continuous turn of exactly the advertised length must survive
        // every gate, through the real windowing.
        let (mut segments, samples) = one_speaker_scope(advertised as f32);
        let unnamed = name_speakers_across_scopes_with(
            Some(one_voice_embedder()),
            &mut [SpeakerScope {
                segments: &mut segments,
                samples: &samples,
            }],
        )
        .unwrap();
        assert_eq!(unnamed.len(), 1, "{unnamed:?}");
        assert!(
            !matches!(
                unnamed[0].reason,
                SpeakerNamingRefusal::NotEnoughSpeech { .. }
            ),
            "a turn of the advertised length is still refused for being short: {:?}",
            unnamed[0]
        );
    }

    /// The case `MIN_CONTINUOUS_SPEECH_SECONDS_FOR_NAMING`'s "+1" margin
    /// exists for: a genuinely single voice whose windows still split far
    /// enough apart that main-cluster filtering does not reclaim the
    /// minority, only surviving because the fraction floor
    /// (`evidence::MIXED_MIN_SECOND_CLUSTER_FRACTION`) keeps the verdict at
    /// single-voice regardless of that distance. This needs more than the bare
    /// minimum window budget -- one window out of seven is the fewest that
    /// clears the 0.15 fraction floor -- so it is a longer turn than the
    /// advertised minimum, not the minimum itself; at the minimum's six
    /// planned windows the fraction floor cannot excuse a distant split
    /// (1/6 is already over 0.15), so reclaim is what clears the gate there
    /// instead (see the previous test). Both paths have to work.
    #[test]
    fn naming_still_clears_a_longer_single_voice_label_whose_minority_goes_unreclaimed() {
        // 7 planned windows: first=0.5, last=8.5, spanning 9.0s of turn.
        let (mut segments, mut samples) = one_speaker_scope(9.0);
        let marker_sample =
            (evidence::SEGMENT_EDGE_TRIM_SECONDS * EMBEDDER_SAMPLE_RATE_HZ as f64) as usize;
        samples[marker_sample] = 1.0;

        let unnamed = name_speakers_across_scopes_with(
            Some(&OneVoiceWithOneUnreclaimedOutlierWindow),
            &mut [SpeakerScope {
                segments: &mut segments,
                samples: &samples,
            }],
        )
        .unwrap();

        assert_eq!(unnamed.len(), 1, "{unnamed:?}");
        assert!(
            !matches!(
                unnamed[0].reason,
                SpeakerNamingRefusal::NotEnoughSpeech { .. }
                    | SpeakerNamingRefusal::MixedVoices { .. }
            ),
            "a longer single-voice turn must still clear naming even when one \
             window is never reclaimed: {:?}",
            unnamed[0]
        );
    }

    /// Enough windows but two voices in them: reported as its own thing, since
    /// "record for longer" is not the remedy.
    #[test]
    fn a_mixed_label_is_reported_as_mixed_rather_than_as_too_short() {
        let mut mixed = evidence_entry(0, vec![1.0, 0.0], PLENTY_OF_WINDOWS);
        mixed.single_voice = false;
        assert!(matches!(
            mixed.refusal(),
            SpeakerNamingRefusal::MixedVoices { .. }
        ));
    }

    /// Below the purity-verdict window count no verdict is claimed at all (see
    /// [`evidence`]), so a thin label must never be reported as "we heard two
    /// people" -- that would be inventing a finding out of an absence of one.
    #[test]
    fn a_thin_label_is_never_reported_as_mixed_voices() {
        let mut thin = evidence_entry(0, vec![1.0, 0.0], MIN_PURITY_VERDICT_WINDOWS - 1);
        thin.single_voice = false;
        assert!(matches!(
            thin.refusal(),
            SpeakerNamingRefusal::NotEnoughSpeech { .. }
        ));
    }

    /// The report describes what the reader sees: it is keyed on the labels the
    /// finished segments carry, and a named speaker is not in it at all.
    #[test]
    fn the_report_covers_exactly_the_anonymous_labels_on_the_segments() {
        let mut segments = vec![
            labeled(0.0, 1.0, Some("SPEAKER_01")),
            labeled(1.0, 2.0, Some("SPEAKER_02")),
            labeled(2.0, 3.0, Some("SPEAKER_01")),
        ];
        normalize_local_labels(&mut segments);
        segments[1].speaker_person_id = Some("person_abc".to_string());
        let scopes = [SpeakerScope {
            segments: &mut segments,
            samples: &[],
        }];
        let reported = unnamed_speakers(&scopes, &BTreeMap::new(), |_| {
            SpeakerNamingRefusal::EmbedderUnavailable
        });
        let labels: Vec<&str> = reported.iter().map(|entry| entry.label.as_str()).collect();
        assert_eq!(labels, vec!["SPEAKER_01"]);
    }
}
