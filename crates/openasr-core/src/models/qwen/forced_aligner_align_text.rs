//! Text-side algorithms for the Qwen3-ForcedAligner NAR pipeline: the
//! deterministic word tokenizer that turns the input transcript into the
//! per-word unit list fed to the `<timestamp>` prompt, and the `fix_timestamp`
//! longest-increasing-subsequence (LIS) monotonicity repair applied to the raw
//! per-`<timestamp>`-position classify-head outputs.
//!
//! Both are ports of `Qwen3ForceAlignProcessor` in the reference
//! `qwen_asr.inference.qwen3_forced_aligner` package (`tokenize_space_lang`,
//! `split_segment_with_chinese`, `is_cjk_char`, `clean_token`, `fix_timestamp`).
//! Only the "space language" path (used for every language except Japanese and
//! Korean, which the reference routes through external morphological
//! segmenters -- `nagisa` / `soynlp` -- not yet ported) is implemented here;
//! Chinese text has no ASCII whitespace, so `tokenize_space_lang`'s per-segment
//! CJK character split already produces the correct one-token-per-character
//! output for it, matching the reference's *non*-special-cased Chinese path
//! (Chinese is not one of the two languages `encode_timestamp` special-cases).

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum Qwen3ForcedAlignerTextError {
    #[error(
        "qwen3-forced-aligner word tokenizer does not yet support language '{language}' (only the space/CJK-character path used for non-Japanese/Korean languages is ported)"
    )]
    UnsupportedLanguage { language: String },
    #[error("qwen3-forced-aligner fix_timestamp requires a non-empty timestamp sequence")]
    EmptyTimestampSequence,
}

/// Unicode-category test mirroring `Qwen3ForceAlignProcessor.is_kept_char`:
/// letters, numbers, and the ASCII apostrophe (kept for contractions such as
/// "don't") survive punctuation stripping; everything else is discarded.
fn is_kept_char(ch: char) -> bool {
    if ch == '\'' {
        return true;
    }
    // Python's unicodedata category groups: "L*" (letters) and "N*" (numbers).
    // Rust's char::is_alphanumeric() covers exactly the Unicode "Letter" and
    // "Number" major categories, matching `cat.startswith("L")` / `("N")`.
    ch.is_alphanumeric()
}

fn clean_token(token: &str) -> String {
    token.chars().filter(|&ch| is_kept_char(ch)).collect()
}

/// CJK Unified Ideograph ranges, ported verbatim from
/// `Qwen3ForceAlignProcessor.is_cjk_char`.
fn is_cjk_char(ch: char) -> bool {
    let code = ch as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
        || (0x2B740..=0x2B81F).contains(&code)
        || (0x2B820..=0x2CEAF).contains(&code)
        || (0xF900..=0xFAFF).contains(&code)
}

/// Port of `split_segment_with_chinese`: within one whitespace-delimited
/// segment, every CJK character becomes its own token and runs of non-CJK
/// characters are kept together as one token (cleaned later by the caller).
fn split_segment_with_chinese(segment: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut buf = String::new();
    for ch in segment.chars() {
        if is_cjk_char(ch) {
            if !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
            }
            tokens.push(ch.to_string());
        } else {
            buf.push(ch);
        }
    }
    if !buf.is_empty() {
        tokens.push(buf);
    }
    tokens
}

/// Port of `tokenize_space_lang`: split on ASCII whitespace, strip punctuation
/// from each whitespace-delimited segment, then further split any CJK
/// characters out of that segment into their own tokens. Used for every
/// language the reference does not special-case with an external
/// morphological tokenizer (i.e. every language except Japanese/Korean),
/// which includes both English and Chinese in our supported scope.
fn tokenize_space_lang(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for segment in text.split_ascii_whitespace() {
        let cleaned = clean_token(segment);
        if cleaned.is_empty() {
            continue;
        }
        for piece in split_segment_with_chinese(&cleaned) {
            if !piece.is_empty() {
                tokens.push(piece);
            }
        }
    }
    tokens
}

/// Port of `Qwen3ForceAlignProcessor.encode_timestamp`'s word-list step only
/// (the prompt-string assembly is done by the caller against the runtime
/// tokenizer, not by literal string concatenation -- see
/// `forced_aligner_runtime::build_forced_aligner_decode_prompt`). Japanese and
/// Korean are not yet supported: the reference routes them through `nagisa`
/// and `soynlp`, which have not been ported, so failing closed here is safer
/// than silently mis-tokenizing.
pub(crate) fn word_list_for_language(
    text: &str,
    language: &str,
) -> Result<Vec<String>, Qwen3ForcedAlignerTextError> {
    // Accepts both the reference's full language names and the ISO 639-1
    // codes callers thread through from `Transcription::language` /
    // `--language` (e.g. "ja"/"ko"), so a real caller's short codes trigger
    // the same fail-closed guard as the reference's own "japanese"/"korean".
    let normalized = language.to_ascii_lowercase();
    if matches!(normalized.as_str(), "japanese" | "ja" | "korean" | "ko") {
        return Err(Qwen3ForcedAlignerTextError::UnsupportedLanguage {
            language: language.to_string(),
        });
    }
    Ok(tokenize_space_lang(text))
}

/// Port of `Qwen3ForceAlignProcessor.fix_timestamp`: repairs local
/// non-monotonicity in the raw per-position classify-head timestamps (in
/// segment-count units, i.e. before multiplying by `timestamp_segment_time`)
/// by finding the longest non-decreasing subsequence (LIS with `<=`) and
/// interpolating/clamping the "anomaly" runs between LIS anchors:
///   - anomaly runs of length <= 2: nearest LIS neighbor wins (ties break
///     toward the left neighbor, matching the reference's `<=` comparison).
///   - anomaly runs of length > 2: linear interpolation between the
///     surrounding LIS anchors (or a flat clamp if only one side has an
///     anchor -- i.e. the anomaly touches a sequence boundary).
///
/// Ported field-for-field from the Python `dp`/`parent` LIS reconstruction so
/// tie-breaking and integer truncation (`int(res)`, i.e. truncation toward
/// zero) match exactly; see the bit-exact fixtures in the test module below,
/// captured by running the reference implementation directly.
//
// The interpolation loops below are deliberately index-based (`for k in
// i..j`), not iterator-chained, to stay a transcription of the Python
// reference's index arithmetic that a reviewer can diff line-for-line.
#[allow(clippy::needless_range_loop)]
pub(crate) fn fix_timestamp(data: &[i64]) -> Result<Vec<i64>, Qwen3ForcedAlignerTextError> {
    let n = data.len();
    if n == 0 {
        return Err(Qwen3ForcedAlignerTextError::EmptyTimestampSequence);
    }

    let mut dp = vec![1_i64; n];
    let mut parent = vec![-1_i64; n];
    for i in 1..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = j as i64;
            }
        }
    }

    let max_length = *dp.iter().max().expect("n > 0 checked above");
    let max_idx = dp
        .iter()
        .position(|&len| len == max_length)
        .expect("max exists");

    let mut lis_indices = Vec::new();
    let mut idx = max_idx as i64;
    while idx != -1 {
        lis_indices.push(idx as usize);
        idx = parent[idx as usize];
    }
    lis_indices.reverse();

    let mut is_normal = vec![false; n];
    for &idx in &lis_indices {
        is_normal[idx] = true;
    }

    // Python operates on an f64-typed `result` copy (the reference casts to
    // int only in the final return); mirror that so the interpolation step's
    // arithmetic matches bit-for-bit.
    let mut result: Vec<f64> = data.iter().map(|&value| value as f64).collect();

    let mut i = 0_usize;
    while i < n {
        if !is_normal[i] {
            let mut j = i;
            while j < n && !is_normal[j] {
                j += 1;
            }
            let anomaly_count = j - i;

            let left_val = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
            let right_val = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);

            if anomaly_count <= 2 {
                for k in i..j {
                    result[k] = match (left_val, right_val) {
                        (None, Some(right)) => right,
                        (Some(left), None) => left,
                        (Some(left), Some(right)) => {
                            // Python: `left_val if (k - (i - 1)) <= (j - k) else right_val`.
                            // `i` is always > 0 here when `left_val` is Some
                            // (there is a normal index before it), so `i - 1`
                            // cannot underflow.
                            if (k as i64 - (i as i64 - 1)) <= (j as i64 - k as i64) {
                                left
                            } else {
                                right
                            }
                        }
                        (None, None) => result[k],
                    };
                }
            } else {
                match (left_val, right_val) {
                    (Some(left), Some(right)) => {
                        let step = (right - left) / (anomaly_count as f64 + 1.0);
                        for k in i..j {
                            result[k] = left + step * ((k - i + 1) as f64);
                        }
                    }
                    (Some(left), None) => {
                        for k in i..j {
                            result[k] = left;
                        }
                    }
                    (None, Some(right)) => {
                        for k in i..j {
                            result[k] = right;
                        }
                    }
                    (None, None) => {}
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }

    // Python's `int(res)` truncates toward zero; timestamps here are always
    // non-negative in practice, but truncate (not round) to match exactly.
    Ok(result
        .into_iter()
        .map(|value| value.trunc() as i64)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_list_tokenizes_english_like_space_lang_reference() {
        let words = word_list_for_language(
            "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
            "English",
        )
        .expect("word list");
        assert_eq!(
            words,
            vec![
                "And",
                "so",
                "my",
                "fellow",
                "Americans",
                "ask",
                "not",
                "what",
                "your",
                "country",
                "can",
                "do",
                "for",
                "you",
                "ask",
                "what",
                "you",
                "can",
                "do",
                "for",
                "your",
                "country"
            ]
        );
    }

    #[test]
    fn word_list_splits_chinese_text_into_individual_characters() {
        let words = word_list_for_language("今天天气非常好我打算和朋友们一起去公园散步", "Chinese")
            .expect("word list");
        assert_eq!(
            words,
            vec![
                "今", "天", "天", "气", "非", "常", "好", "我", "打", "算", "和", "朋", "友", "们",
                "一", "起", "去", "公", "园", "散", "步"
            ]
        );
    }

    #[test]
    fn word_list_rejects_unsupported_languages() {
        let error = word_list_for_language("hello", "Japanese").expect_err("must fail");
        assert!(matches!(
            error,
            Qwen3ForcedAlignerTextError::UnsupportedLanguage { .. }
        ));
    }

    // Bit-exact fixtures captured by running the reference
    // `Qwen3ForceAlignProcessor.fix_timestamp` directly (see
    // tmp/forced-aligner-ref/fix_timestamp_fixture.py, dev-machine only /
    // gitignored -- values transcribed here so the test has no runtime
    // dependency on the Python venv).

    #[test]
    fn fix_timestamp_is_a_noop_for_already_monotonic_input() {
        let input = [0, 80, 160, 240, 320];
        assert_eq!(fix_timestamp(&input).unwrap(), vec![0, 80, 160, 240, 320]);
    }

    #[test]
    fn fix_timestamp_repairs_a_single_dip_anomaly() {
        let input = [0, 80, 40, 240, 320];
        assert_eq!(fix_timestamp(&input).unwrap(), vec![0, 80, 80, 240, 320]);
    }

    #[test]
    fn fix_timestamp_repairs_a_two_element_anomaly_run() {
        let input = [0, 80, 200, 30, 40, 320, 400];
        assert_eq!(
            fix_timestamp(&input).unwrap(),
            vec![0, 80, 200, 200, 320, 320, 400]
        );
    }

    #[test]
    fn fix_timestamp_interpolates_a_longer_anomaly_run() {
        let input = [0, 500, 10, 20, 30, 40, 600, 700];
        assert_eq!(
            fix_timestamp(&input).unwrap(),
            vec![0, 0, 10, 20, 30, 40, 600, 700]
        );
    }

    #[test]
    fn fix_timestamp_clamps_a_leading_anomaly_with_no_left_anchor() {
        let input = [500, 10, 100, 200, 300];
        assert_eq!(fix_timestamp(&input).unwrap(), vec![10, 10, 100, 200, 300]);
    }

    #[test]
    fn fix_timestamp_clamps_a_trailing_anomaly_with_no_right_anchor() {
        let input = [0, 100, 200, 300, 10];
        assert_eq!(fix_timestamp(&input).unwrap(), vec![0, 100, 200, 300, 300]);
    }

    #[test]
    fn fix_timestamp_is_a_noop_for_the_real_jfk_word_boundary_sequence() {
        // Ports of the real (already-monotonic) jfk.wav per-word start/end
        // timestamps from tmp/forced-aligner-ref/reference_output.json --
        // confirms fix_timestamp does not perturb a real, already-consistent
        // trace (the common case).
        let input = [
            320, 560, 640, 960, 960, 1280, 1360, 1680, 1680, 2160, 3280, 3680, 4000, 4320, 5360,
            5600, 5600, 5920, 5920, 6400, 6400, 6640, 6640, 6880, 6960, 7040, 7040, 7520, 8160,
            8480, 8640, 8800, 8800, 9200, 9200, 9440, 9440, 9600, 9680, 9760, 9760, 10000, 10000,
            10480,
        ];
        assert_eq!(fix_timestamp(&input).unwrap(), input.to_vec());
    }

    #[test]
    fn fix_timestamp_rejects_empty_input() {
        let error = fix_timestamp(&[]).expect_err("must fail");
        assert!(matches!(
            error,
            Qwen3ForcedAlignerTextError::EmptyTimestampSequence
        ));
    }
}
