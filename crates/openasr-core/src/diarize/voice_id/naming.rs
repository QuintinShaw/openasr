//! Why a speaker in a finished transcript is still anonymous.
//!
//! # Refusing is normal; refusing invisibly is not
//!
//! Every naming gate in this module's siblings is one-sided toward anonymous
//! (see [`identity`](super::identity)'s module docs): a wrong name is worse
//! than no name, so thin evidence and borderline matches both end as
//! `SPEAKER_01`. That policy is right and nothing here loosens it.
//!
//! What it left behind was a reporting hole. A bare `SPEAKER_01` is the same
//! pixel for three situations a user has to act on differently:
//!
//! 1. **too little speech** -- the person may well be enrolled, but four
//!    seconds of audio is not enough to risk their name on; record longer and
//!    the same voice is recognized;
//! 2. **nobody matched** -- the evidence was good and this voice is simply not
//!    in the library; enroll them;
//! 3. **nothing ran** -- the speaker embedder is not installed, so no matching
//!    was even attempted; install the pack.
//!
//! Only the third is a malfunction, and it is the one users assume in all
//! three cases, because a number with no explanation reads as a broken
//! feature. The refusal reason is therefore a **return value**, not a
//! `OPENASR_DIARIZE_DEBUG` trace: the numbers behind the verdict already
//! existed, they were merely unreachable from anywhere a user could see.
//!
//! # Contract
//!
//! A refusal is reported for a label **as the finished transcript spells it**,
//! so a consumer can join it against `Segment::speaker_label` with no
//! knowledge of the scope disambiguation and stitching that produced the name.
//! Labels that *were* named are absent: they already carry their person on
//! every segment.

/// One anonymous speaker in a finished transcript, with the reason it stayed
/// that way.
#[derive(Debug, Clone, PartialEq)]
pub struct UnnamedSpeaker {
    /// The label the transcript's segments carry (`SPEAKER_01`).
    pub label: String,
    pub reason: SpeakerNamingRefusal,
}

/// Why one label was not matched to an enrolled person.
///
/// The variants are ordered by how early the pipeline gives up, and each
/// carries the numbers its own gate compared -- so a caller can say "2.0s of
/// 3.0s" rather than only "not enough". They are deliberately not collapsed
/// into a single "unavailable" case: the action a user can take differs per
/// variant, which is the entire point of reporting them.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeakerNamingRefusal {
    /// The speaker embedder pack is not installed, so no evidence was gathered
    /// and no matching ran. The only variant that describes a malfunction
    /// rather than a judgement.
    EmbedderUnavailable,
    /// Too little usable voice behind the label to risk a name on it.
    ///
    /// Both gates of [`identity`](super::identity) are reported together
    /// because they are nested rather than competing (see
    /// `MIN_NAMING_EVIDENCE_SECONDS`): whichever bound bit, the user-facing
    /// answer is the same one, "this speaker needs to talk for longer".
    NotEnoughSpeech {
        /// Embedding windows that survived main-cluster filtering.
        windows: usize,
        required_windows: usize,
        /// Distinct audio those windows cover.
        seconds: f64,
        required_seconds: f64,
        /// How long one uninterrupted turn has to be before it can clear both
        /// gates, derived from the window geometry rather than restated.
        ///
        /// This exists because neither raw threshold is the number to put in
        /// front of a user. `required_seconds` is the *smaller*, non-binding
        /// one: a speaker who talks for exactly that long still yields too few
        /// windows and is refused again. Telling them "3 seconds" would send
        /// them back to fail a second time, which is worse than saying
        /// nothing. Only the engine knows the real figure, so only the engine
        /// may state it -- a UI must never derive it.
        required_continuous_seconds: f64,
    },
    /// There was enough voice, but it did not come from one person, so naming
    /// it would attribute two people's words to whichever one matched.
    MixedVoices { windows: usize, seconds: f64 },
    /// The evidence cleared every gate and still matched nobody: this voice is
    /// not in the Voice ID library (or is, but not closely enough).
    NoMatchInLibrary {
        /// No one is enrolled at all, which is a different thing to say than
        /// "we looked and none of them were you".
        library_empty: bool,
        /// Best similarity any enrolled person scored, and the floor it had to
        /// clear. `None` when scoring could not run at all (an embedding space
        /// the library was not built in), which is not the same as scoring
        /// zero.
        best_score: Option<f32>,
        accept_threshold: Option<f32>,
    },
}

impl SpeakerNamingRefusal {
    /// Stable wire/log token for this reason.
    ///
    /// Kept next to the variants so a new reason cannot reach an API without
    /// picking one, and spelled in kebab-case because that is what every other
    /// machine-readable reason string in the transcript JSON uses (see
    /// `DecodeTruncation::reason`).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::EmbedderUnavailable => "embedder-unavailable",
            Self::NotEnoughSpeech { .. } => "not-enough-speech",
            Self::MixedVoices { .. } => "mixed-voices",
            Self::NoMatchInLibrary { .. } => "no-match-in-library",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_refusal_reason_has_its_own_wire_token() {
        let reasons = [
            SpeakerNamingRefusal::EmbedderUnavailable,
            SpeakerNamingRefusal::NotEnoughSpeech {
                windows: 1,
                required_windows: 5,
                seconds: 2.0,
                required_seconds: 3.0,
                required_continuous_seconds: 7.0,
            },
            SpeakerNamingRefusal::MixedVoices {
                windows: 6,
                seconds: 7.0,
            },
            SpeakerNamingRefusal::NoMatchInLibrary {
                library_empty: false,
                best_score: Some(0.21),
                accept_threshold: Some(0.45),
            },
        ];
        let mut kinds: Vec<&str> = reasons.iter().map(SpeakerNamingRefusal::kind).collect();
        kinds.sort_unstable();
        let count = kinds.len();
        kinds.dedup();
        assert_eq!(kinds.len(), count, "refusal reasons share a wire token");
    }
}
