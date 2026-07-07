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
    /// Multilingual-only: the `<lang>` token this code selects is absent from
    /// this pack's vocab. Kept separate from `ControlTokenMissing` (which uses
    /// a `&'static str` token, since the cn-dialect family's `<lang>` slot is
    /// the fixed `<zh>`) because the multilingual builder computes the token
    /// content per code from [`DOLPHIN_MULTILINGUAL_LANGUAGE_TABLE`].
    LanguageTokenMissing {
        code: String,
        language_token: String,
    },
    /// Multilingual-only: the `<region>` token this code's default region
    /// selects is absent from this pack's vocab (see `LanguageTokenMissing`).
    MultilingualRegionTokenMissing { code: String, region_token: String },
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
            Self::LanguageTokenMissing {
                code,
                language_token,
            } => write!(
                f,
                "this dolphin pack vocab has no '{language_token}' language token for language {code:?}"
            ),
            Self::MultilingualRegionTokenMissing { code, region_token } => write!(
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

// --- multilingual (dolphin-small / dolphin-base) ----------------------------
//
// The multilingual Dolphin checkpoints recognize 40 Eastern languages (plus
// Mandarin's 22 dialects, which these checkpoints do NOT advertise as
// separate recognition codes -- unlike the dedicated cn-dialect models, they
// fold every Chinese dialect into `zh`). Their vocab uses the SAME OWSM-style
// two-slot `<lang><region>` prompt as the cn-dialect family, but the `<lang>`
// slot actually varies per code instead of being fixed at `<zh>`, and the
/// `<region>` slot is a per-language country/region tag rather than a Chinese
// province.
//
/// code -> (Dolphin's own `<lang>` token content, default `<region>` token
/// content), both without angle brackets. Source: DataoceanAI's
/// `languages.md` "Language Code" + "Language Region Code" tables. Dolphin's
/// own language code matches this crate's normalized ISO code for every entry
/// except Cantonese (Dolphin: `ct`, this table's key: `yue`, matching
/// `LANG_BY_FAMILY`/`LANGUAGE_DISPLAY_LABELS`'s existing `yue` convention).
/// The default region is the card's unqualified/most-common region where it
/// lists several (e.g. Arabic's `ar-GLA` is the only entry labeled plain
/// "Arabic", the rest carry a country qualifier), or `NULL` where the card
/// only ever lists a `NULL` region (Yue, Uighur, Kabyle, Bashkir) -- these are
/// defaults only; a request can always override the resolved language, though
/// this builder does not yet expose a separate region parameter (see
/// `build_dolphin_multilingual_decode_prefix`'s doc comment).
const DOLPHIN_MULTILINGUAL_LANGUAGE_TABLE: &[(&str, &str, &str)] = &[
    ("ar", "ar", "GLA"),
    ("az", "az", "AZ"),
    ("ba", "ba", "NULL"),
    ("bn", "bn", "BD"),
    ("fa", "fa", "IR"),
    ("fil", "fil", "PH"),
    ("gu", "gu", "IN"),
    ("hi", "hi", "IN"),
    ("id", "id", "ID"),
    ("ja", "ja", "JP"),
    ("jv", "jv", "ID"),
    ("kab", "kab", "NULL"),
    ("kk", "kk", "KZ"),
    ("km", "km", "KH"),
    ("ko", "ko", "KR"),
    ("ks", "ks", "IN"),
    ("ky", "ky", "KG"),
    ("lo", "lo", "LA"),
    ("mn", "mn", "MN"),
    ("mr", "mr", "IN"),
    ("ms", "ms", "MY"),
    ("my", "my", "MM"),
    ("ne", "ne", "NP"),
    ("or", "or", "IN"),
    ("pa", "pa", "IN"),
    ("ps", "ps", "AF"),
    ("ru", "ru", "RU"),
    ("si", "si", "LK"),
    ("su", "su", "ID"),
    ("ta", "ta", "IN"),
    ("te", "te", "IN"),
    ("tg", "tg", "TJ"),
    ("th", "th", "TH"),
    ("tl", "tl", "PH"),
    ("ug", "ug", "NULL"),
    ("ur", "ur", "IN"),
    ("uz", "uz", "UZ"),
    ("vi", "vi", "VN"),
    ("yue", "ct", "NULL"),
    ("zh", "zh", "CN"),
];

/// The recognition code the multilingual builder defaults to when the request
/// leaves `language` unset: Mandarin (`<zh><CN>`), matching the family's
/// `SelectsViaPrompt { default: "zh" }` (same default as the cn-dialect
/// builder's `DOLPHIN_DEFAULT_LANGUAGE_CODE`, kept as its own constant so the
/// two builders stay independently readable).
pub(crate) const DOLPHIN_MULTILINGUAL_DEFAULT_LANGUAGE_CODE: &str = "zh";

/// code -> (`<lang>` token content, default `<region>` token content), or
/// `None` for a language this multilingual table does not carry.
pub(crate) fn dolphin_multilingual_lang_region_for_code(
    code: &str,
) -> Option<(&'static str, &'static str)> {
    DOLPHIN_MULTILINGUAL_LANGUAGE_TABLE
        .iter()
        .find(|(table_code, _, _)| *table_code == code)
        .map(|(_, lang, region)| (*lang, *region))
}

/// Build the OWSM-style decode prefix for the multilingual Dolphin
/// checkpoints (`dolphin-small`/`dolphin-base`): `<sos> <lang> <region> <asr>
/// <notimestamp>`, with BOTH `<lang>` and `<region>` varying per code (unlike
/// the cn-dialect builder, whose `<lang>` is fixed at `<zh>`). `requested`
/// selects the language only; the region always resolves to that language's
/// table default -- there is no separate region parameter on the recognition
/// request today, so a caller cannot yet pick e.g. `ar-EG` over the `ar`
/// default `ar-GLA`. Fails closed (typed) on an unknown code or a missing
/// language/region/control token, exactly like
/// [`build_dolphin_decode_prefix`].
pub(crate) fn build_dolphin_multilingual_decode_prefix(
    vocab: &[String],
    requested: Option<&str>,
) -> Result<DolphinDecodePrefix, DolphinPrefixError> {
    let resolved_language = requested
        .map(str::trim)
        .filter(|code| !code.is_empty())
        .map(normalize_language)
        .unwrap_or_else(|| DOLPHIN_MULTILINGUAL_DEFAULT_LANGUAGE_CODE.to_string());
    let (language_token_content, region_token_content) =
        dolphin_multilingual_lang_region_for_code(&resolved_language).ok_or_else(|| {
            DolphinPrefixError::UnsupportedLanguageCode {
                code: resolved_language.clone(),
            }
        })?;
    let language_token = format!("<{language_token_content}>");
    let region_token = format!("<{region_token_content}>");
    let control_id = |token: &'static str| -> Result<u32, DolphinPrefixError> {
        token_id_for_content(vocab, token).ok_or(DolphinPrefixError::ControlTokenMissing { token })
    };
    let token_ids = vec![
        control_id(DOLPHIN_SOS_TOKEN)?,
        token_id_for_content(vocab, &language_token).ok_or_else(|| {
            DolphinPrefixError::LanguageTokenMissing {
                code: resolved_language.clone(),
                language_token: language_token.clone(),
            }
        })?,
        token_id_for_content(vocab, &region_token).ok_or_else(|| {
            DolphinPrefixError::MultilingualRegionTokenMissing {
                code: resolved_language.clone(),
                region_token: region_token.clone(),
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

/// The 40 codes `dolphin-small`/`dolphin-base` advertise as their catalog
/// `languages` override (see `tooling/publish-model/models-core.toml`), kept
/// in sync by hand with that TOML list -- there is no single Rust source of
/// truth for a per-model catalog `languages` override, unlike the family-wide
/// `REGISTERED_DIALECT_CODES`. Used both by the importer's producer-side
/// guard (`assert_every_advertised_multilingual_code_resolves`) and this
/// module's own tests.
pub(crate) const DOLPHIN_MULTILINGUAL_CATALOG_LANGUAGES: &[&str] = &[
    "ar", "az", "ba", "bn", "fa", "fil", "gu", "hi", "id", "ja", "jv", "kab", "kk", "km", "ko",
    "ks", "ky", "lo", "mn", "mr", "ms", "my", "ne", "or", "pa", "ps", "ru", "si", "su", "ta", "te",
    "tg", "th", "tl", "ug", "ur", "uz", "vi", "yue", "zh",
];

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

    // --- multilingual (dolphin-small / dolphin-base) ------------------------

    /// Synthetic vocab covering the multilingual control tokens plus every
    /// `<lang>`/`<region>` token the catalog's 40-language list and their
    /// table defaults need, at arbitrary (non-contiguous) ids -- unlike the
    /// cn-dialect fixture, the real multilingual vocab does NOT front-load
    /// `<sos>`/`<eos>` (they sit after ~40k BPE pieces), so this fixture
    /// intentionally puts them at high ids too.
    fn dolphin_multilingual_vocab() -> Vec<String> {
        let mut tokens: Vec<String> = vec!["<blank>".to_string(), "<unk>".to_string()];
        for &code in DOLPHIN_MULTILINGUAL_CATALOG_LANGUAGES {
            let (lang, region) = dolphin_multilingual_lang_region_for_code(code)
                .unwrap_or_else(|| panic!("catalog language '{code}' has no multilingual mapping"));
            tokens.push(format!("<{lang}>"));
            tokens.push(format!("<{region}>"));
        }
        tokens.push("<asr>".to_string());
        tokens.push("<notimestamp>".to_string());
        tokens.push("<sos>".to_string());
        tokens.push("<eos>".to_string());
        tokens
    }

    #[test]
    fn multilingual_unset_language_defaults_to_mandarin_cn_region() {
        let vocab = dolphin_multilingual_vocab();
        let prefix = build_dolphin_multilingual_decode_prefix(&vocab, None).expect("build");
        assert_eq!(prefix.resolved_language, "zh");
        let sos = token_id_for_content(&vocab, "<sos>").unwrap();
        let zh = token_id_for_content(&vocab, "<zh>").unwrap();
        let cn = token_id_for_content(&vocab, "<CN>").unwrap();
        let asr = token_id_for_content(&vocab, "<asr>").unwrap();
        let notimestamp = token_id_for_content(&vocab, "<notimestamp>").unwrap();
        assert_eq!(prefix.token_ids, vec![sos, zh, cn, asr, notimestamp]);
    }

    #[test]
    fn multilingual_yue_selects_dolphins_own_ct_language_token() {
        // The catalog's `yue` (ISO) recognition code maps to Dolphin's own
        // `<ct>` language token, not a fabricated `<yue>` one.
        let vocab = dolphin_multilingual_vocab();
        let prefix = build_dolphin_multilingual_decode_prefix(&vocab, Some("yue")).expect("build");
        assert_eq!(prefix.resolved_language, "yue");
        let ct = token_id_for_content(&vocab, "<ct>").unwrap();
        let null_region = token_id_for_content(&vocab, "<NULL>").unwrap();
        assert!(prefix.token_ids.contains(&ct));
        assert!(prefix.token_ids.contains(&null_region));
    }

    #[test]
    fn multilingual_table_covers_exactly_the_catalog_languages() {
        // Every language the catalog advertises for dolphin-small/dolphin-base
        // must resolve to a `<lang><region>` pair, and the full prefix must
        // build against a vocab carrying every one of those tokens -- the
        // producer-side guard mirroring `region_map_covers_exactly_the_registered_dialect_codes`.
        let vocab = dolphin_multilingual_vocab();
        for &code in DOLPHIN_MULTILINGUAL_CATALOG_LANGUAGES {
            dolphin_multilingual_lang_region_for_code(code)
                .unwrap_or_else(|| panic!("catalog language '{code}' has no multilingual mapping"));
            build_dolphin_multilingual_decode_prefix(&vocab, Some(code)).unwrap_or_else(|error| {
                panic!("multilingual prefix build failed for '{code}': {error}")
            });
        }
    }

    #[test]
    fn multilingual_unknown_code_fails_closed() {
        let vocab = dolphin_multilingual_vocab();
        assert_eq!(
            build_dolphin_multilingual_decode_prefix(&vocab, Some("zh-sichuan")),
            Err(DolphinPrefixError::UnsupportedLanguageCode {
                code: "zh-sichuan".to_string()
            })
        );
        assert_eq!(
            build_dolphin_multilingual_decode_prefix(&vocab, Some("xx")),
            Err(DolphinPrefixError::UnsupportedLanguageCode {
                code: "xx".to_string()
            })
        );
    }

    #[test]
    fn multilingual_missing_language_token_fails_closed() {
        // A vocab with every region token but no `<ja>` language token for a
        // registered code must reject, not silently decode under `<zh>`.
        let mut vocab = dolphin_multilingual_vocab();
        vocab.retain(|token| token != "<ja>");
        assert_eq!(
            build_dolphin_multilingual_decode_prefix(&vocab, Some("ja")),
            Err(DolphinPrefixError::LanguageTokenMissing {
                code: "ja".to_string(),
                language_token: "<ja>".to_string(),
            })
        );
    }

    #[test]
    fn multilingual_missing_region_token_fails_closed() {
        let mut vocab = dolphin_multilingual_vocab();
        vocab.retain(|token| token != "<JP>");
        assert_eq!(
            build_dolphin_multilingual_decode_prefix(&vocab, Some("ja")),
            Err(DolphinPrefixError::MultilingualRegionTokenMissing {
                code: "ja".to_string(),
                region_token: "<JP>".to_string(),
            })
        );
    }
}
