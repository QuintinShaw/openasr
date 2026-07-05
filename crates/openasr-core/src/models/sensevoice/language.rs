//! SenseVoiceSmall language / prompt-token selection.
//!
//! SenseVoice conditions recognition on a fixed 4-token prompt that is embedded
//! (via the model's input embedding table) and concatenated in front of the LFR
//! speech features:
//!
//! ```text
//!   [ <lang-id>, <event-query>, <emotion-query>, <textnorm> ]  ++  speech
//! ```
//!
//! `<event-query>` and `<emotion-query>` are the two fixed query embeddings
//! (indices 1 and 2) that make the model emit its `<|EVENT|>` / `<|EMOTION|>`
//! prefix tags. The language slot and the text-normalization (ITN) slot are the
//! request-selectable ones. This module owns the single code -> embedding-index
//! map and the fail-closed prompt builder: an unknown language code is rejected
//! with a typed error rather than silently decoding under a fabricated prompt,
//! mirroring `dolphin::language` and the whisper missing-language path.

// The prompt builder + tag-shadow flag are consumed by the SenseVoice executor
// (later stage); until then they are exercised only by the unit tests here.
#![allow(dead_code)]

use crate::models::language::normalize_language;

/// Fixed query-embedding indices SenseVoice prepends between the language slot
/// and the textnorm slot (the `<event>` and `<emotion>` query tokens). These are
/// not request-selectable; they drive the emotion/event tag prefix the decoder
/// emits (kept shadowed from the public capability surface -- see
/// [`SenseVoiceTagShadow`]).
pub(crate) const EVENT_QUERY_EMBED_INDEX: usize = 1;
pub(crate) const EMOTION_QUERY_EMBED_INDEX: usize = 2;

/// Text-normalization / inverse-text-normalization prompt indices (SenseVoice
/// `textnorm_dict`). `withitn` applies ITN (digits, punctuation); `woitn` leaves
/// the raw spoken form.
const TEXTNORM_WITHITN_EMBED_INDEX: usize = 14;
const TEXTNORM_WOITN_EMBED_INDEX: usize = 15;

/// Default recognition language when the request leaves `language` unset: `auto`
/// (SenseVoice performs its own LID over the six advertised languages).
pub(crate) const SENSEVOICE_DEFAULT_LANGUAGE_CODE: &str = "auto";

/// Map a normalized recognition code to its SenseVoice language-id embedding
/// index (`lid_dict`). `auto` asks the model to detect the language itself.
/// Returns `None` for any code SenseVoiceSmall does not advertise, which the
/// prompt builder turns into a fail-closed `UnsupportedLanguageCode`.
pub(crate) fn sensevoice_language_embed_index(code: &str) -> Option<usize> {
    Some(match code {
        "auto" => 0,
        "zh" => 3,
        "en" => 4,
        "yue" => 7,
        "ja" => 11,
        "ko" => 12,
        _ => return None,
    })
}

/// Every language code SenseVoiceSmall advertises (excluding `auto`), in the
/// order the catalog lists them. Pinned against the embed map by a test so the
/// picker never advertises a code the prompt builder cannot honor.
pub(crate) const SENSEVOICE_LANGUAGE_CODES: &[&str] = &["zh", "yue", "en", "ja", "ko"];

/// Fail-closed reasons the prompt builder rejects a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SenseVoicePromptError {
    /// The requested code is not a language SenseVoiceSmall recognizes.
    UnsupportedLanguageCode { code: String },
}

impl std::fmt::Display for SenseVoicePromptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedLanguageCode { code } => write!(
                f,
                "language {code:?} is not a language this SenseVoice model supports"
            ),
        }
    }
}

/// A resolved SenseVoice prompt: the 4 embedding indices to prepend, plus the
/// normalized language code the prompt encodes (for an honest read-back).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SenseVoicePrompt {
    /// `[lang, event, emotion, textnorm]` embedding-table indices, in order.
    pub embed_indices: [usize; 4],
    /// The normalized recognition code the prompt encodes (`auto`, `zh`, ...).
    pub resolved_language: String,
    /// Whether inverse text normalization (`withitn`) was requested.
    pub use_itn: bool,
}

/// Build the 4-token SenseVoice prompt for `requested` (default `auto` when unset
/// or blank) and the ITN preference. Fails closed with a typed error on an
/// unknown language code -- it never falls back to a different language.
pub(crate) fn build_sensevoice_prompt(
    requested: Option<&str>,
    use_itn: bool,
) -> Result<SenseVoicePrompt, SenseVoicePromptError> {
    let resolved_language = requested
        .map(str::trim)
        .filter(|code| !code.is_empty())
        .map(normalize_language)
        .unwrap_or_else(|| SENSEVOICE_DEFAULT_LANGUAGE_CODE.to_string());
    let lang_index = sensevoice_language_embed_index(&resolved_language).ok_or_else(|| {
        SenseVoicePromptError::UnsupportedLanguageCode {
            code: resolved_language.clone(),
        }
    })?;
    let textnorm_index = if use_itn {
        TEXTNORM_WITHITN_EMBED_INDEX
    } else {
        TEXTNORM_WOITN_EMBED_INDEX
    };
    Ok(SenseVoicePrompt {
        embed_indices: [
            lang_index,
            EVENT_QUERY_EMBED_INDEX,
            EMOTION_QUERY_EMBED_INDEX,
            textnorm_index,
        ],
        resolved_language,
        use_itn,
    })
}

/// Controls whether SenseVoice's emotion/event prefix tags are surfaced on the
/// public capability surface. They are parsed into structured fields internally
/// but kept **shadowed** (unexposed) for now -- mirroring the repo's other
/// "capability shadow" flags -- until the emotion/event product surface is
/// designed. Default is `Shadowed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum SenseVoiceTagShadow {
    /// Emotion/event tags parsed but not exposed on the result (current default).
    #[default]
    Shadowed,
    /// Emotion/event tags surfaced (reserved for a future opt-in product path).
    Exposed,
}

impl SenseVoiceTagShadow {
    pub(crate) fn exposes_emotion_event(self) -> bool {
        matches!(self, Self::Exposed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_language_defaults_to_auto_woitn() {
        let prompt = build_sensevoice_prompt(None, false).expect("prompt");
        // [auto=0, event=1, emotion=2, woitn=15].
        assert_eq!(prompt.embed_indices, [0, 1, 2, 15]);
        assert_eq!(prompt.resolved_language, "auto");
        assert!(!prompt.use_itn);
        // Blank/whitespace is treated as unset, not an unknown code.
        let blank = build_sensevoice_prompt(Some("   "), false).expect("prompt");
        assert_eq!(blank.embed_indices, [0, 1, 2, 15]);
    }

    #[test]
    fn each_advertised_language_selects_its_embed_index() {
        for (code, lang_index) in [("zh", 3), ("en", 4), ("yue", 7), ("ja", 11), ("ko", 12)] {
            let prompt = build_sensevoice_prompt(Some(code), false).expect("prompt");
            assert_eq!(prompt.embed_indices[0], lang_index, "lang index for {code}");
            assert_eq!(prompt.embed_indices[1], EVENT_QUERY_EMBED_INDEX);
            assert_eq!(prompt.embed_indices[2], EMOTION_QUERY_EMBED_INDEX);
            assert_eq!(prompt.resolved_language, code);
        }
    }

    #[test]
    fn itn_flag_selects_withitn_slot() {
        let withitn = build_sensevoice_prompt(Some("zh"), true).expect("prompt");
        assert_eq!(withitn.embed_indices[3], 14);
        assert!(withitn.use_itn);
        let woitn = build_sensevoice_prompt(Some("zh"), false).expect("prompt");
        assert_eq!(woitn.embed_indices[3], 15);
    }

    #[test]
    fn request_code_is_normalized_before_lookup() {
        let prompt = build_sensevoice_prompt(Some("  ZH "), false).expect("prompt");
        assert_eq!(prompt.embed_indices[0], 3);
        assert_eq!(prompt.resolved_language, "zh");
    }

    #[test]
    fn unknown_code_fails_closed() {
        assert_eq!(
            build_sensevoice_prompt(Some("fr"), false),
            Err(SenseVoicePromptError::UnsupportedLanguageCode {
                code: "fr".to_string()
            })
        );
        // A language SenseVoice does not advertise (e.g. Spanish) also rejects.
        assert_eq!(
            build_sensevoice_prompt(Some("es"), false),
            Err(SenseVoicePromptError::UnsupportedLanguageCode {
                code: "es".to_string()
            })
        );
    }

    #[test]
    fn advertised_codes_all_resolve_to_an_embed_index() {
        for &code in SENSEVOICE_LANGUAGE_CODES {
            assert!(
                sensevoice_language_embed_index(code).is_some(),
                "advertised code '{code}' must map to an embed index"
            );
            build_sensevoice_prompt(Some(code), false)
                .unwrap_or_else(|error| panic!("prompt build failed for '{code}': {error}"));
        }
        // `auto` default maps too (not part of the advertised recognition set).
        assert_eq!(
            sensevoice_language_embed_index(SENSEVOICE_DEFAULT_LANGUAGE_CODE),
            Some(0)
        );
    }

    #[test]
    fn emotion_event_tags_are_shadowed_by_default() {
        assert!(!SenseVoiceTagShadow::default().exposes_emotion_event());
        assert!(SenseVoiceTagShadow::Exposed.exposes_emotion_event());
    }
}
