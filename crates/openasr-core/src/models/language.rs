//! Shared helpers for turning a request language hint into a Whisper-style
//! language control token, so every family that consumes `language` agrees on
//! normalization (trim + lowercase) and token spelling (`<|xx|>`).

use crate::GgufMetadata;
use crate::models::ggml_family_adapter::LanguageFamilyHint;

/// Normalize a request language code: trim surrounding whitespace and lowercase.
/// Whisper/Cohere language control tokens are lowercase BCP-47-ish codes
/// (`en`, `fr`, `zh`, ...), so this is the single canonical form.
pub(crate) fn normalize_language(code: &str) -> String {
    code.trim().to_lowercase()
}

/// The Whisper-family language control token for an already-normalized code,
/// e.g. `"fr"` -> `"<|fr|>"`. Resolution against a specific pack's vocab is the
/// caller's job (and is fail-closed when the token is absent).
pub(crate) fn language_control_token(normalized_code: &str) -> String {
    format!("<|{normalized_code}|>")
}

/// Resolved, per-pack source-language behavior: the single axis the fail-closed
/// gate (and, later, the capability surface) dispatch on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageMode {
    /// Decode-time auto-detect plus explicit selection. Multilingual Whisper.
    DetectAndSpecify,
    /// The model self-detects internally; an explicit hint is rejected with this
    /// reason (no per-request selection, no readout). Qwen3-ASR.
    DetectImplicit { reject_reason: &'static str },
    /// Explicit selection via a prompt language token; `default_language` is used
    /// when the request leaves it unset. Cohere transcribe.
    SpecifyOnly { default_language: &'static str },
    /// Intrinsically a single language; only that language is accepted.
    FixedMonolingual { language: &'static str },
    /// Intrinsically a fixed set with no per-request selection. XASR zh-en.
    FixedMultilingual { languages: &'static [&'static str] },
}

/// Resolve a pack's [`LanguageMode`] from its family hint plus, where the family
/// is vocab-gated (Whisper), the pack's own vocab. Tying the resolved behavior to
/// what the pack can actually honor (rather than to a catalog claim) means code
/// and catalog cannot drift.
pub(crate) fn resolve_language_mode(
    hint: LanguageFamilyHint,
    metadata: &GgufMetadata,
) -> LanguageMode {
    match hint {
        LanguageFamilyHint::WhisperVocabGated => {
            if crate::models::whisper::whisper_metadata_is_multilingual(metadata) {
                LanguageMode::DetectAndSpecify
            } else {
                LanguageMode::FixedMonolingual { language: "en" }
            }
        }
        LanguageFamilyHint::SelfDetectsRejectsHint { reject_reason } => {
            LanguageMode::DetectImplicit { reject_reason }
        }
        LanguageFamilyHint::SelectsViaPrompt { default_language } => {
            LanguageMode::SpecifyOnly { default_language }
        }
        LanguageFamilyHint::FixedMonolingual { language } => {
            LanguageMode::FixedMonolingual { language }
        }
        LanguageFamilyHint::FixedMultilingual { languages } => {
            LanguageMode::FixedMultilingual { languages }
        }
    }
}

/// The source language to report for a finished transcription, given the resolved
/// mode and the request hint. Honest by construction: a language is reported only
/// when it was genuinely determined for this audio - explicitly selected, the
/// model's conditioned default, or an intrinsically fixed language. Families that
/// self-detect without exposing the result, or that have no decode-time detection
/// yet, report `None` rather than a fabricated guess.
pub(crate) fn effective_reported_language(
    mode: LanguageMode,
    requested: Option<&str>,
) -> Option<String> {
    let explicit = requested
        .map(str::trim)
        .filter(|code| !code.is_empty())
        .map(normalize_language);
    match mode {
        // Auto stays None until decode-time detection (whisper SOT pass) fills
        // it; an explicit selection is reported as the language used.
        LanguageMode::DetectAndSpecify => explicit,
        // Self-detects internally but does not expose the detected language.
        LanguageMode::DetectImplicit { .. } => None,
        // Conditioned on the explicit code or the family default - either way a
        // genuine input the model was steered with.
        LanguageMode::SpecifyOnly { default_language } => {
            Some(explicit.unwrap_or_else(|| default_language.to_string()))
        }
        // Intrinsically this single language.
        LanguageMode::FixedMonolingual { language } => Some(language.to_string()),
        // Fixed set with no readout of which language was emitted.
        LanguageMode::FixedMultilingual { .. } => None,
    }
}

/// Map an ISO-639-1 (or Whisper-style) language code to its lowercase English
/// name for OpenAI-compatible `verbose_json` output (OpenAI returns the name, not
/// the code). Fail-open: an unrecognized code is returned normalized and
/// unchanged rather than erroring, so a future/unknown code never breaks output.
/// Table sourced verbatim from Whisper's `LANGUAGES` dict
/// (openai/whisper tokenizer.py).
pub(crate) fn code_to_english_name(code: &str) -> String {
    let normalized = normalize_language(code);
    english_name_for_code(&normalized)
        .map(str::to_string)
        .unwrap_or(normalized)
}

fn english_name_for_code(code: &str) -> Option<&'static str> {
    Some(match code {
        "en" => "english",
        "zh" => "chinese",
        "de" => "german",
        "es" => "spanish",
        "ru" => "russian",
        "ko" => "korean",
        "fr" => "french",
        "ja" => "japanese",
        "pt" => "portuguese",
        "tr" => "turkish",
        "pl" => "polish",
        "ca" => "catalan",
        "nl" => "dutch",
        "ar" => "arabic",
        "sv" => "swedish",
        "it" => "italian",
        "id" => "indonesian",
        "hi" => "hindi",
        "fi" => "finnish",
        "vi" => "vietnamese",
        "he" => "hebrew",
        "uk" => "ukrainian",
        "el" => "greek",
        "ms" => "malay",
        "cs" => "czech",
        "ro" => "romanian",
        "da" => "danish",
        "hu" => "hungarian",
        "ta" => "tamil",
        "no" => "norwegian",
        "th" => "thai",
        "ur" => "urdu",
        "hr" => "croatian",
        "bg" => "bulgarian",
        "lt" => "lithuanian",
        "la" => "latin",
        "mi" => "maori",
        "ml" => "malayalam",
        "cy" => "welsh",
        "sk" => "slovak",
        "te" => "telugu",
        "fa" => "persian",
        "lv" => "latvian",
        "bn" => "bengali",
        "sr" => "serbian",
        "az" => "azerbaijani",
        "sl" => "slovenian",
        "kn" => "kannada",
        "et" => "estonian",
        "mk" => "macedonian",
        "br" => "breton",
        "eu" => "basque",
        "is" => "icelandic",
        "hy" => "armenian",
        "ne" => "nepali",
        "mn" => "mongolian",
        "bs" => "bosnian",
        "kk" => "kazakh",
        "sq" => "albanian",
        "sw" => "swahili",
        "gl" => "galician",
        "mr" => "marathi",
        "pa" => "punjabi",
        "si" => "sinhala",
        "km" => "khmer",
        "sn" => "shona",
        "yo" => "yoruba",
        "so" => "somali",
        "af" => "afrikaans",
        "oc" => "occitan",
        "ka" => "georgian",
        "be" => "belarusian",
        "tg" => "tajik",
        "sd" => "sindhi",
        "gu" => "gujarati",
        "am" => "amharic",
        "yi" => "yiddish",
        "lo" => "lao",
        "uz" => "uzbek",
        "fo" => "faroese",
        "ht" => "haitian creole",
        "ps" => "pashto",
        "tk" => "turkmen",
        "nn" => "nynorsk",
        "mt" => "maltese",
        "sa" => "sanskrit",
        "lb" => "luxembourgish",
        "my" => "myanmar",
        "bo" => "tibetan",
        "tl" => "tagalog",
        "mg" => "malagasy",
        "as" => "assamese",
        "tt" => "tatar",
        "haw" => "hawaiian",
        "ln" => "lingala",
        "ha" => "hausa",
        "ba" => "bashkir",
        "jw" => "javanese",
        "su" => "sundanese",
        "yue" => "cantonese",
        _ => return None,
    })
}

/// The Whisper language-code set (the codes whose `<|code|>` token can appear at
/// the SOT position). Used by whisper LID to build the id->code mask. Kept in sync
/// with `english_name_for_code` by a test below. Order matches Whisper's LANGUAGES.
pub(crate) const WHISPER_LANGUAGE_CODES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_trims_and_lowercases() {
        assert_eq!(normalize_language("  FR "), "fr");
        assert_eq!(normalize_language("en"), "en");
        assert_eq!(normalize_language("ZH"), "zh");
    }

    #[test]
    fn control_token_wraps_normalized_code() {
        assert_eq!(language_control_token("fr"), "<|fr|>");
        assert_eq!(
            language_control_token(&normalize_language(" DE ")),
            "<|de|>"
        );
    }

    fn metadata_with_token_count(count: usize) -> GgufMetadata {
        use crate::ggml_runtime::GgufMetadataValue;
        let mut values = std::collections::BTreeMap::new();
        values.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::StringArray((0..count).map(|i| i.to_string()).collect()),
        );
        GgufMetadata::from_values_for_test(values)
    }

    #[test]
    fn resolve_maps_non_whisper_hints_directly() {
        let empty = GgufMetadata::from_values_for_test(std::collections::BTreeMap::new());
        assert_eq!(
            resolve_language_mode(
                LanguageFamilyHint::SelfDetectsRejectsHint { reject_reason: "x" },
                &empty
            ),
            LanguageMode::DetectImplicit { reject_reason: "x" }
        );
        assert_eq!(
            resolve_language_mode(
                LanguageFamilyHint::SelectsViaPrompt {
                    default_language: "en"
                },
                &empty
            ),
            LanguageMode::SpecifyOnly {
                default_language: "en"
            }
        );
        assert_eq!(
            resolve_language_mode(
                LanguageFamilyHint::FixedMonolingual { language: "en" },
                &empty
            ),
            LanguageMode::FixedMonolingual { language: "en" }
        );
        assert_eq!(
            resolve_language_mode(
                LanguageFamilyHint::FixedMultilingual {
                    languages: &["en", "zh"]
                },
                &empty
            ),
            LanguageMode::FixedMultilingual {
                languages: &["en", "zh"]
            }
        );
    }

    #[test]
    fn effective_reported_language_is_honest_per_mode() {
        // DetectAndSpecify (whisper): explicit reported; auto stays None until detection.
        assert_eq!(
            effective_reported_language(LanguageMode::DetectAndSpecify, Some("FR ")),
            Some("fr".to_string())
        );
        assert_eq!(
            effective_reported_language(LanguageMode::DetectAndSpecify, None),
            None
        );
        assert_eq!(
            effective_reported_language(LanguageMode::DetectAndSpecify, Some("  ")),
            None
        );
        // DetectImplicit (qwen): never fabricated.
        assert_eq!(
            effective_reported_language(LanguageMode::DetectImplicit { reject_reason: "x" }, None),
            None
        );
        // SpecifyOnly (cohere): explicit, else the conditioned default.
        assert_eq!(
            effective_reported_language(
                LanguageMode::SpecifyOnly {
                    default_language: "en"
                },
                None
            ),
            Some("en".to_string())
        );
        assert_eq!(
            effective_reported_language(
                LanguageMode::SpecifyOnly {
                    default_language: "en"
                },
                Some("de")
            ),
            Some("de".to_string())
        );
        // FixedMonolingual: always the fixed language.
        assert_eq!(
            effective_reported_language(LanguageMode::FixedMonolingual { language: "en" }, None),
            Some("en".to_string())
        );
        // FixedMultilingual (xasr): no readout.
        assert_eq!(
            effective_reported_language(
                LanguageMode::FixedMultilingual {
                    languages: &["en", "zh"]
                },
                None
            ),
            None
        );
    }

    #[test]
    fn resolve_whisper_uses_vocab_to_pick_multilingual_vs_english_only() {
        // A short token list (below the English-only ceiling) -> fixed English.
        let english_only = metadata_with_token_count(8);
        assert_eq!(
            resolve_language_mode(LanguageFamilyHint::WhisperVocabGated, &english_only),
            LanguageMode::FixedMonolingual { language: "en" }
        );
        // An unreadable token list fails toward multilingual: the decoder still
        // validates the explicit `<|lang|>` token before use.
        let empty = GgufMetadata::from_values_for_test(std::collections::BTreeMap::new());
        assert_eq!(
            resolve_language_mode(LanguageFamilyHint::WhisperVocabGated, &empty),
            LanguageMode::DetectAndSpecify
        );
    }

    #[test]
    fn code_to_english_name_matches_openai_convention() {
        assert_eq!(code_to_english_name("en"), "english");
        assert_eq!(code_to_english_name("ZH"), "chinese");
        // Whisper's spelling, not the geographic adjective.
        assert_eq!(code_to_english_name("nl"), "dutch");
        assert_eq!(code_to_english_name(" fr "), "french");
        assert_eq!(code_to_english_name("yue"), "cantonese");
        // Fail-open: an unknown code returns normalized and unchanged.
        assert_eq!(code_to_english_name("xx"), "xx");
        assert_eq!(code_to_english_name("ZZ"), "zz");
    }

    #[test]
    fn whisper_language_codes_all_have_a_name() {
        // Guards against WHISPER_LANGUAGE_CODES drifting from english_name_for_code:
        // every code must resolve to a real name (not fail-open to the code itself).
        for &code in WHISPER_LANGUAGE_CODES {
            assert_ne!(
                code_to_english_name(code),
                code,
                "code '{code}' has no English name; WHISPER_LANGUAGE_CODES drifted"
            );
        }
    }
}
