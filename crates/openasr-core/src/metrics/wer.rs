//! Word- and character-error-rate metrics for the performance harness.
//!
//! Ported from `scripts/quant_eval_common.py` so the Rust regression gate and
//! the offline Python quant pipeline agree on numbers. One documented
//! divergence: the Python normalizer applies Unicode NFKC + category-based
//! `L`/`N` filtering via `unicodedata`; std Rust has neither without a crate
//! dependency, so we approximate with `char::to_lowercase` (casefold) and
//! `char::is_alphanumeric` (Unicode L/N). For ASCII-range references (the
//! LibriSpeech evalset) the two are identical; non-Latin scripts may differ
//! slightly. Kept dependency-free on purpose.

/// Normalize transcription text for comparison: casefold, replace every
/// non-alphanumeric character with a space, and collapse runs of whitespace.
///
/// Mirrors `quant_eval_common.normalize_text` (minus NFKC; see module docs).
pub fn normalize_text(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_whitespace() {
            out.push(' ');
        } else {
            // casefold approximation
            for lowered in ch.to_lowercase() {
                if lowered.is_alphanumeric() {
                    out.push(lowered);
                } else {
                    out.push(' ');
                }
            }
        }
    }
    // collapse whitespace + trim
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn tokens(value: &str) -> Vec<String> {
    normalize_text(value)
        .split(' ')
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect()
}

/// Classic two-row dynamic-programming Levenshtein edit distance.
///
/// Mirrors `quant_eval_common.levenshtein_distance`.
pub fn levenshtein<T: PartialEq>(left: &[T], right: &[T]) -> usize {
    if left == right {
        return 0;
    }
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    for (row, left_value) in left.iter().enumerate() {
        let mut current = vec![row + 1];
        for (column, right_value) in right.iter().enumerate() {
            let cost = usize::from(left_value != right_value);
            let candidate = (previous[column + 1] + 1)
                .min(current[column] + 1)
                .min(previous[column] + cost);
            current.push(candidate);
        }
        previous = current;
    }
    previous[right.len()]
}

/// Edit count + reference length for a hypothesis/reference pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WerCounts {
    pub errors: usize,
    pub ref_units: usize,
}

/// Word-level error count + reference word count.
///
/// Empty-reference rule matches the Python helper: zero errors when the
/// hypothesis is also empty, otherwise the hypothesis word count is the error.
pub fn wer_counts(hyp: &str, reference: &str) -> WerCounts {
    let hyp_words = tokens(hyp);
    let ref_words = tokens(reference);
    if ref_words.is_empty() {
        return WerCounts {
            errors: hyp_words.len(),
            ref_units: 0,
        };
    }
    WerCounts {
        errors: levenshtein(&hyp_words, &ref_words),
        ref_units: ref_words.len(),
    }
}

/// Word error rate in `[0, ∞)`. Returns `0.0` for an empty reference matched by
/// an empty hypothesis, else `1.0` when only the reference is empty.
pub fn wer(hyp: &str, reference: &str) -> f64 {
    let counts = wer_counts(hyp, reference);
    if counts.ref_units == 0 {
        return if counts.errors == 0 { 0.0 } else { 1.0 };
    }
    counts.errors as f64 / counts.ref_units as f64
}

/// Word error rate for a live partial against the equally long prefix of a full
/// reference/final transcript.
///
/// This is the metric a realtime partial regression gate wants: a first partial
/// like "Answer." should fail against the final prefix "And", while "And so."
/// should pass against "And so, my fellow Americans...". Returns `None` when the
/// partial or reference has no normalized words.
pub fn word_prefix_error_rate(partial: &str, reference: &str) -> Option<f64> {
    let partial_words = tokens(partial);
    if partial_words.is_empty() {
        return None;
    }
    let reference_words = tokens(reference);
    let prefix_len = partial_words.len().min(reference_words.len());
    if prefix_len == 0 {
        return None;
    }
    let errors = levenshtein(&partial_words, &reference_words[..prefix_len]);
    Some(errors as f64 / prefix_len as f64)
}

/// Character-level error count + reference character count (normalized).
pub fn cer_counts(hyp: &str, reference: &str) -> WerCounts {
    let hyp_chars: Vec<char> = normalize_text(hyp).chars().collect();
    let ref_chars: Vec<char> = normalize_text(reference).chars().collect();
    if ref_chars.is_empty() {
        return WerCounts {
            errors: hyp_chars.len(),
            ref_units: 0,
        };
    }
    WerCounts {
        errors: levenshtein(&hyp_chars, &ref_chars),
        ref_units: ref_chars.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_text_has_zero_wer() {
        assert_eq!(wer("the cat sat", "the cat sat"), 0.0);
    }

    #[test]
    fn normalization_ignores_case_and_punctuation() {
        assert_eq!(wer("The CAT, sat!", "the cat sat"), 0.0);
    }

    #[test]
    fn single_substitution_counts_one_error() {
        // 3 reference words, one wrong -> 1/3
        let value = wer("the dog sat", "the cat sat");
        assert!((value - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn empty_reference_with_empty_hyp_is_zero() {
        assert_eq!(wer("", ""), 0.0);
        assert_eq!(wer("   ", ""), 0.0);
    }

    #[test]
    fn empty_reference_with_nonempty_hyp_is_one() {
        assert_eq!(wer("hello", ""), 1.0);
    }

    #[test]
    fn counts_track_errors_and_ref_words() {
        let counts = wer_counts("the dog sat down", "the cat sat");
        assert_eq!(counts.ref_units, 3);
        // one substitution (cat->dog) + one insertion (down) = 2
        assert_eq!(counts.errors, 2);
    }

    #[test]
    fn levenshtein_basic() {
        let a: Vec<char> = "kitten".chars().collect();
        let b: Vec<char> = "sitting".chars().collect();
        assert_eq!(levenshtein(&a, &b), 3);
    }

    #[test]
    fn word_prefix_error_rate_compares_against_reference_prefix() {
        assert_eq!(
            word_prefix_error_rate("And so.", "And so, my fellow Americans, ask not.").unwrap(),
            0.0
        );
        assert_eq!(
            word_prefix_error_rate("Answer.", "And so, my fellow Americans, ask not.").unwrap(),
            1.0
        );
        assert_eq!(word_prefix_error_rate("", "And so"), None);
        assert_eq!(word_prefix_error_rate("And", ""), None);
    }

    #[test]
    fn cer_counts_normalized() {
        let counts = cer_counts("abc", "abd");
        assert_eq!(counts.ref_units, 3);
        assert_eq!(counts.errors, 1);
    }
}
