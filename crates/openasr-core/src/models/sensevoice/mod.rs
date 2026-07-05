//! SenseVoiceSmall (`FunAudioLLM/SenseVoiceSmall`) model family.
//!
//! Encoder-only SAN-M / DFSMN acoustic model with a CTC head. Multilingual
//! (zh, yue, en, ja, ko) with model-side language id. Licensed under the **FunASR
//! Model License v1.1** (<https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE>),
//! not Apache-2.0 -- see `ACKNOWLEDGMENTS` / catalog license fields.
//!
//! Stage status:
//! - Frontend (fbank + LFR + CMVN) and prompt/language selection are implemented
//!   and unit-tested here ([`frontend`], [`language`]).
//! - The SAN-M/FSMN encoder graph, CTC executor, weight loader, and the
//!   FunASR-checkpoint-to-GGUF converter land in later stages.

// The tag parser + re-exports are consumed by the SenseVoice executor (later
// stage); until then they are exercised only by the unit tests here.
#![allow(dead_code)]

pub(crate) mod frontend;
pub(crate) mod language;

#[allow(unused_imports)]
pub(crate) use frontend::{
    SenseVoiceFbankFeatures, SenseVoiceFbankFrontend, SenseVoiceFrontendError,
    SenseVoiceLfrFeatures, apply_cmvn, apply_lfr,
};
#[allow(unused_imports)]
pub(crate) use language::{
    SenseVoicePrompt, SenseVoicePromptError, SenseVoiceTagShadow, build_sensevoice_prompt,
};

/// Structured tags SenseVoice emits as a `<|...|>` prefix before the transcript,
/// e.g. `<|zh|><|EMO_UNKNOWN|><|Speech|><|woitn|>大家好`. Emotion and event are
/// kept **shadowed** (parsed but unexposed) per [`SenseVoiceTagShadow`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SenseVoiceStructuredTags {
    /// Detected language tag content (e.g. `zh`), if a leading language tag was
    /// present.
    pub language: Option<String>,
    /// Emotion tag content (e.g. `EMO_UNKNOWN`, `HAPPY`) -- shadowed.
    pub emotion: Option<String>,
    /// Acoustic-event tag content (e.g. `Speech`, `BGM`, `Applause`) -- shadowed.
    pub event: Option<String>,
    /// ITN tag content (`withitn` / `woitn`), if present.
    pub itn: Option<String>,
    /// Any additional `<|...|>` tags seen before the first plain-text token.
    pub other: Vec<String>,
}

/// Strip SenseVoice's leading `<|...|>` tag prefix from a raw decoded string,
/// returning the structured tags and the remaining transcript text. The first
/// four tags are mapped positionally to `[language, emotion, event, itn]`
/// (SenseVoice always emits them in that order); any further leading tags are
/// collected in `other`. Text after the tag run is returned verbatim (leading
/// whitespace trimmed once).
pub(crate) fn strip_sensevoice_tag_prefix(raw: &str) -> (SenseVoiceStructuredTags, String) {
    let mut tags = SenseVoiceStructuredTags::default();
    let mut rest = raw;
    let mut position = 0usize;
    while let Some(inner) = rest.strip_prefix("<|") {
        let Some(end) = inner.find("|>") else { break };
        let content = &inner[..end];
        match position {
            0 => tags.language = Some(content.to_string()),
            1 => tags.emotion = Some(content.to_string()),
            2 => tags.event = Some(content.to_string()),
            3 => tags.itn = Some(content.to_string()),
            _ => tags.other.push(content.to_string()),
        }
        position += 1;
        rest = &inner[end + 2..];
    }
    (tags, rest.trim_start().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_full_tag_prefix_into_structured_fields() {
        let (tags, text) =
            strip_sensevoice_tag_prefix("<|zh|><|EMO_UNKNOWN|><|Speech|><|woitn|>大家好");
        assert_eq!(tags.language.as_deref(), Some("zh"));
        assert_eq!(tags.emotion.as_deref(), Some("EMO_UNKNOWN"));
        assert_eq!(tags.event.as_deref(), Some("Speech"));
        assert_eq!(tags.itn.as_deref(), Some("woitn"));
        assert!(tags.other.is_empty());
        assert_eq!(text, "大家好");
    }

    #[test]
    fn english_prefix_and_leading_space_trimmed() {
        let (tags, text) =
            strip_sensevoice_tag_prefix("<|en|><|HAPPY|><|Speech|><|withitn|> Hello world");
        assert_eq!(tags.language.as_deref(), Some("en"));
        assert_eq!(tags.emotion.as_deref(), Some("HAPPY"));
        assert_eq!(tags.itn.as_deref(), Some("withitn"));
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn plain_text_without_tags_is_returned_unchanged() {
        let (tags, text) = strip_sensevoice_tag_prefix("no tags here");
        assert_eq!(tags, SenseVoiceStructuredTags::default());
        assert_eq!(text, "no tags here");
    }

    #[test]
    fn extra_leading_tags_collected_in_other() {
        let (tags, text) =
            strip_sensevoice_tag_prefix("<|zh|><|EMO_UNKNOWN|><|Speech|><|woitn|><|extra|>hi");
        assert_eq!(tags.other, vec!["extra".to_string()]);
        assert_eq!(text, "hi");
    }
}
