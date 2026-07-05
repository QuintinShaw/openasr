//! Dolphin dialect language/region selection: turn a request language code into
//! the OWSM-style decode prefix `<sos> <lang> <region> <asr> <notimestamp>` the
//! Transformer decoder is conditioned on.
//!
//! Dolphin `small.cn` recognizes Mandarin plus a set of Chinese regional dialects,
//! each selected by a `<REGION>` prompt token (the same mechanism OWSM/Whisper use
//! for `<|lang|>`). This module owns the single code -> region-token map and the
//! fail-closed prefix builder: an unknown code, or a region/control token absent
//! from the pack vocab, is rejected with a typed error rather than silently
//! decoding under the wrong (or a fabricated) prefix -- mirroring the whisper
//! missing-language path (`WhisperPrefixError::LanguageTokenMissing`).

use crate::models::language::normalize_language;

/// Language token shared by every Dolphin recognition code: `small.cn`'s advertised
/// dialects are all Sinitic, so the OWSM `<lang>` slot is always `<zh>` and the
/// per-code selection rides the `<region>` slot instead.
const DOLPHIN_LANGUAGE_TOKEN: &str = "<zh>";
/// Task token: Dolphin runs ASR (not the `<vad>`/`<lid>` OWSM tasks).
const DOLPHIN_TASK_TOKEN: &str = "<asr>";
/// Start-of-transcript and no-timestamp control tokens bracketing the prefix.
const DOLPHIN_SOS_TOKEN: &str = "<sos>";
const DOLPHIN_NOTIMESTAMP_TOKEN: &str = "<notimestamp>";

/// The recognition code used when the request leaves `language` unset: general
/// Mandarin (`<CN>`), matching the family's `SelectsViaPrompt { default: "zh" }`.
pub(crate) const DOLPHIN_DEFAULT_LANGUAGE_CODE: &str = "zh";

/// Map a normalized recognition code to its Dolphin `<REGION>` vocab token. The
/// keys are exactly Phase 1's `REGISTERED_DIALECT_CODES` plus the bare `zh`
/// default; a test below pins the two lists together so the executor honors
/// precisely the codes the signed catalog advertises (no lies in the picker).
/// Returns `None` for any other code, which the builder turns into a fail-closed
/// `UnsupportedLanguageCode`.
pub(crate) fn dolphin_region_token_for_code(code: &str) -> Option<&'static str> {
    Some(match code {
        "zh" => "<CN>",
        "zh-tw" => "<TW>",
        "zh-sichuan" => "<SICHUAN>",
        "zh-shanxi" => "<SHANXI>",
        "zh-anhui" => "<ANHUI>",
        "zh-tianjin" => "<TIANJIN>",
        "zh-ningxia" => "<NINGXIA>",
        "zh-shaanxi" => "<SHAANXI>",
        "zh-hebei" => "<HEBEI>",
        "zh-shandong" => "<SHANDONG>",
        "zh-guangdong" => "<GUANGDONG>",
        "zh-shanghai" => "<SHANGHAI>",
        "zh-hubei" => "<HUBEI>",
        "zh-jiangsu" => "<JIANGSU>",
        _ => return None,
    })
}

/// Fail-closed reasons the decode-prefix builder rejects a request. Kept typed so
/// the executor can surface a precise message and tests can assert the exact case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DolphinPrefixError {
    /// The requested code is not a recognition code this family supports (no
    /// region mapping) -- e.g. a typo or a language Dolphin does not recognize.
    UnsupportedLanguageCode { code: String },
    /// A required control token (`<sos>`/`<zh>`/`<asr>`/`<notimestamp>`) is absent
    /// from this pack's vocab.
    ControlTokenMissing { token: &'static str },
    /// The `<region>` token this code selects is absent from this pack's vocab.
    RegionTokenMissing {
        code: String,
        region_token: &'static str,
    },
}

impl std::fmt::Display for DolphinPrefixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedLanguageCode { code } => write!(
                f,
                "language {code:?} is not a recognition code this Dolphin model supports"
            ),
            Self::ControlTokenMissing { token } => {
                write!(f, "this dolphin pack vocab has no '{token}' control token")
            }
            Self::RegionTokenMissing { code, region_token } => write!(
                f,
                "this dolphin pack vocab has no '{region_token}' region token for language {code:?}"
            ),
        }
    }
}

/// A resolved decode prefix plus the code it encodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DolphinDecodePrefix {
    /// `<sos> <zh> <region> <asr> <notimestamp>` token ids, in decode order.
    pub token_ids: Vec<u32>,
    /// The normalized recognition code the prefix encodes (`zh`, `zh-sichuan`,
    /// ...), for an honest read-back on the finished transcription.
    pub resolved_language: String,
}

/// Build the OWSM-style decode prefix for `requested` (default `zh`/`<CN>` when
/// unset or blank), resolving every token against this pack's `vocab`. Fails
/// closed with a typed error on an unknown code or a missing region/control
/// token -- it never falls back to a different prefix.
pub(crate) fn build_dolphin_decode_prefix(
    vocab: &[String],
    requested: Option<&str>,
) -> Result<DolphinDecodePrefix, DolphinPrefixError> {
    let resolved_language = requested
        .map(str::trim)
        .filter(|code| !code.is_empty())
        .map(normalize_language)
        .unwrap_or_else(|| DOLPHIN_DEFAULT_LANGUAGE_CODE.to_string());
    let region_token = dolphin_region_token_for_code(&resolved_language).ok_or_else(|| {
        DolphinPrefixError::UnsupportedLanguageCode {
            code: resolved_language.clone(),
        }
    })?;
    let control_id = |token: &'static str| -> Result<u32, DolphinPrefixError> {
        token_id_for_content(vocab, token).ok_or(DolphinPrefixError::ControlTokenMissing { token })
    };
    // Assembled in decode order; each slot validates left-to-right against vocab.
    let token_ids = vec![
        control_id(DOLPHIN_SOS_TOKEN)?,
        control_id(DOLPHIN_LANGUAGE_TOKEN)?,
        token_id_for_content(vocab, region_token).ok_or_else(|| {
            DolphinPrefixError::RegionTokenMissing {
                code: resolved_language.clone(),
                region_token,
            }
        })?,
        control_id(DOLPHIN_TASK_TOKEN)?,
        control_id(DOLPHIN_NOTIMESTAMP_TOKEN)?,
    ];
    Ok(DolphinDecodePrefix {
        token_ids,
        resolved_language,
    })
}

/// First vocab id whose token content is exactly `content`. Dolphin's special
/// tokens are unique and front-loaded (ids < 110), so a linear scan is trivial.
fn token_id_for_content(vocab: &[String], content: &str) -> Option<u32> {
    vocab
        .iter()
        .position(|token| token == content)
        .map(|index| index as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::language::REGISTERED_DIALECT_CODES;

    /// Synthetic vocab mirroring the real `units.txt` special-token block: ids
    /// 0..=35 are the exact control/language/region tokens, 36..=108 are reserved
    /// filler, and `<notimestamp>` sits at id 109 -- so an asserted prefix like
    /// `[2, 5, 10, 4, 109]` matches the committed Sichuan golden.
    fn dolphin_special_vocab() -> Vec<String> {
        let mut vocab: Vec<String> = [
            "<blank>",
            "<unk>",
            "<sos>",
            "<eos>",
            "<asr>",
            "<zh>",
            "<en>",
            "<CN>",
            "<TW>",
            "<WU>",
            "<SICHUAN>",
            "<SHANXI>",
            "<ANHUI>",
            "<TIANJIN>",
            "<NINGXIA>",
            "<SHAANXI>",
            "<HEBEI>",
            "<SHANDONG>",
            "<GUANGDONG>",
            "<SHANGHAI>",
            "<HUBEI>",
            "<LIAONING>",
            "<GANSU>",
            "<FUJIAN>",
            "<HUNAN>",
            "<HENAN>",
            "<YUNNAN>",
            "<MINNAN>",
            "<WENZHOU>",
            "<BEIJING>",
            "<JILIN>",
            "<NEIMENGGU>",
            "<GUANGXI>",
            "<GUIZHOU>",
            "<HEILONGJIANG>",
            "<JIANGSU>",
        ]
        .iter()
        .map(|token| token.to_string())
        .collect();
        while vocab.len() < 109 {
            vocab.push(format!("<reserved_{}>", vocab.len()));
        }
        vocab.push("<notimestamp>".to_string());
        vocab
    }

    #[test]
    fn sichuan_prefix_matches_the_committed_golden_ids() {
        let vocab = dolphin_special_vocab();
        let prefix = build_dolphin_decode_prefix(&vocab, Some("zh-sichuan")).expect("build");
        // `<sos> <zh> <SICHUAN> <asr> <notimestamp>` == the baked SICHUAN_PREFIX.
        assert_eq!(prefix.token_ids, vec![2, 5, 10, 4, 109]);
        assert_eq!(prefix.resolved_language, "zh-sichuan");
    }

    #[test]
    fn unset_language_defaults_to_mandarin_cn_region() {
        let vocab = dolphin_special_vocab();
        // Default (None) => `<sos> <zh> <CN> <asr> <notimestamp>`.
        let prefix = build_dolphin_decode_prefix(&vocab, None).expect("build");
        assert_eq!(prefix.token_ids, vec![2, 5, 7, 4, 109]);
        assert_eq!(prefix.resolved_language, "zh");
        // Blank/whitespace is treated as unset, not as an unknown code.
        let blank = build_dolphin_decode_prefix(&vocab, Some("   ")).expect("build");
        assert_eq!(blank.token_ids, vec![2, 5, 7, 4, 109]);
        assert_eq!(blank.resolved_language, "zh");
    }

    #[test]
    fn per_region_codes_select_the_right_region_token() {
        let vocab = dolphin_special_vocab();
        // A couple of non-Sichuan regions build the expected `<region>` id.
        let guangdong = build_dolphin_decode_prefix(&vocab, Some("zh-guangdong")).expect("build");
        assert_eq!(guangdong.token_ids, vec![2, 5, 18, 4, 109]);
        let jiangsu = build_dolphin_decode_prefix(&vocab, Some("zh-jiangsu")).expect("build");
        assert_eq!(jiangsu.token_ids, vec![2, 5, 35, 4, 109]);
        let tw = build_dolphin_decode_prefix(&vocab, Some("zh-tw")).expect("build");
        assert_eq!(tw.token_ids, vec![2, 5, 8, 4, 109]);
    }

    #[test]
    fn request_code_is_normalized_before_lookup() {
        let vocab = dolphin_special_vocab();
        // Trim + lowercase, region subtag preserved (mirrors normalize_language).
        let prefix = build_dolphin_decode_prefix(&vocab, Some("  ZH-Sichuan ")).expect("build");
        assert_eq!(prefix.token_ids, vec![2, 5, 10, 4, 109]);
        assert_eq!(prefix.resolved_language, "zh-sichuan");
    }

    #[test]
    fn unknown_code_fails_closed() {
        let vocab = dolphin_special_vocab();
        // A typo'd region and a language Dolphin does not recognize both reject.
        assert_eq!(
            build_dolphin_decode_prefix(&vocab, Some("zh-sichaun")),
            Err(DolphinPrefixError::UnsupportedLanguageCode {
                code: "zh-sichaun".to_string()
            })
        );
        assert_eq!(
            build_dolphin_decode_prefix(&vocab, Some("fr")),
            Err(DolphinPrefixError::UnsupportedLanguageCode {
                code: "fr".to_string()
            })
        );
    }

    #[test]
    fn missing_region_token_in_vocab_fails_closed() {
        // Truncate the vocab so `<GUANGDONG>` (id 18) is absent but the control
        // tokens (`<sos>`/`<zh>`/`<asr>`) survive: a registered code whose region
        // token this pack lacks must fail closed, not decode under a wrong prefix.
        let mut vocab = dolphin_special_vocab();
        vocab.truncate(18);
        vocab.push("<notimestamp>".to_string()); // keep the timestamp control token
        assert_eq!(
            build_dolphin_decode_prefix(&vocab, Some("zh-guangdong")),
            Err(DolphinPrefixError::RegionTokenMissing {
                code: "zh-guangdong".to_string(),
                region_token: "<GUANGDONG>",
            })
        );
    }

    #[test]
    fn missing_control_token_in_vocab_fails_closed() {
        // A vocab with the region tokens but no `<notimestamp>` control token.
        let mut vocab = dolphin_special_vocab();
        vocab.truncate(36); // drops the reserved filler and `<notimestamp>` (id 109)
        assert_eq!(
            build_dolphin_decode_prefix(&vocab, Some("zh-sichuan")),
            Err(DolphinPrefixError::ControlTokenMissing {
                token: "<notimestamp>"
            })
        );
    }

    #[test]
    fn region_map_covers_exactly_the_registered_dialect_codes() {
        // Every advertised dialect code must resolve to a region token, and that
        // token must exist in the real special-token block -- otherwise the picker
        // would advertise a code the executor cannot honor.
        let vocab = dolphin_special_vocab();
        for &code in REGISTERED_DIALECT_CODES {
            let region = dolphin_region_token_for_code(code)
                .unwrap_or_else(|| panic!("registered dialect code '{code}' has no region token"));
            assert!(
                token_id_for_content(&vocab, region).is_some(),
                "region token '{region}' for '{code}' is not in the special-token block"
            );
            // The full prefix must build for every advertised code.
            build_dolphin_decode_prefix(&vocab, Some(code))
                .unwrap_or_else(|error| panic!("prefix build failed for '{code}': {error}"));
        }
        // The bare `zh` default is mapped too (not part of the dialect-code set).
        assert_eq!(
            dolphin_region_token_for_code(DOLPHIN_DEFAULT_LANGUAGE_CODE),
            Some("<CN>")
        );
    }
}
