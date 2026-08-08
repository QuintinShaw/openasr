//! Optional punctuation-restoration post-processing stage.
//!
//! FireRedPunc (see `models::firered_punc`) is a BERT token classifier that
//! predicts, for each subword of an unpunctuated transcript, which Chinese
//! punctuation mark (if any) follows it. This stage turns those per-token
//! labels into a punctuated string, and decides *when* the stage runs.
//!
//! Design (kept decoupled from ASR, high-cohesion / low-coupling):
//!
//! - **Opt-in and auto-gated.** The stage runs only for models whose transcript
//!   is unpunctuated, keyed off the catalog `emits_punctuation` capability
//!   (`Some(false)` -> run; `Some(true)`/`None` -> leave the text untouched, so
//!   punctuating families like whisper/qwen are never double-punctuated). See
//!   [`should_apply_punctuation`].
//! - **Chinese-only by construction, gated at the runtime entry.** FireRedPunc's
//!   label space is five full-width Chinese marks and its training data is
//!   Chinese-only, but `emits_punctuation` is a model-family-wide capability
//!   flag -- it cannot see that a *given segment* has no Chinese in it (e.g. a
//!   `LanguageFamilyHint::FixedMultilingual` family like FireRed, whose
//!   `Transcription.language` is always `None`, can still emit an all-English
//!   segment). That per-segment check lives in
//!   `FireRedPuncRuntime::punctuate` (the single entry both the batch stage
//!   and the streaming session call): a segment with zero Han ideographs (per
//!   [`crate::models::firered_punc::tokenizer::is_cjk_char`], the same
//!   definition the tokenizer was built against) is returned verbatim. The
//!   [`restore_punctuation`] mechanism below stays deliberately
//!   language-agnostic -- policy belongs to the caller, not the walk.
//!
//!   Known boundaries of that gate (accepted, documented, not bugs):
//!   - A mixed zh/en segment that *does* contain Han ideographs is punctuated
//!     with full-width marks throughout, which is the correct GB/T 15834
//!     treatment of Chinese text with embedded Latin; no half-width mapping is
//!     attempted on the Latin spans.
//!   - For multilingual families like dolphin, a Japanese segment containing
//!     kanji passes the Han gate and gets Chinese punctuation (pre-existing
//!     limitation). Honest en/mixed punctuation needs an upstream zh+en punc
//!     checkpoint (long-term item), not more gating here.
//! - **Finalize-only.** It is meant to run once on a finalized segment, not per
//!   streaming partial -- re-punctuating every partial with a bidirectional
//!   encoder is expensive and reintroduces caption flicker.
//! - **Offset-preserving.** Labels attach to the original text via the
//!   tokenizer's char spans; the stage inserts marks into the original
//!   characters and never re-emits normalised (lowercased) text.
//!
//! The classifier itself is abstracted behind [`PunctuationClassifier`] so this
//! orchestration is fully unit-testable without model weights; the ggml
//! FireRedPunc runtime implements the trait.

use crate::models::firered_punc::config::punctuation_for_label;
use crate::models::firered_punc::tokenizer::{FireRedPuncTokenizer, WordPiecePiece};

#[derive(Debug, thiserror::Error)]
pub(crate) enum PunctuationError {
    // Constructed by the ggml FireRedPunc runtime (lands with the runtime
    // stage) when a forward pass fails; the stage orchestration only forwards
    // it.
    #[error("punctuation classifier failed: {0}")]
    Classifier(String),
    #[error(
        "punctuation classifier returned {got} labels for a {expected}-token window (must match)"
    )]
    LabelCountMismatch { got: usize, expected: usize },
}

/// A model that scores punctuation for one window of content tokens.
///
/// Implementations receive the content token ids for a single window (the
/// tokenizer's model-specific special-token wrapping is the implementation's
/// concern) and
/// return the argmax label id per content token, in the same order and length.
/// Label ids index [`crate::models::firered_punc::config::PUNC_LABELS`].
pub(crate) trait PunctuationClassifier {
    fn predict_window_labels(
        &self,
        content_token_ids: &[u32],
    ) -> Result<Vec<usize>, PunctuationError>;
}

/// Runtime knobs for the restoration walk.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PunctuationRestoreConfig {
    /// Maximum content tokens per classifier window (BERT max positions minus
    /// the leading `[CLS]`). Windows follow the upstream fixed-width token
    /// split; punctuation is emitted only from word-final pieces.
    pub max_content_tokens: usize,
}

impl Default for PunctuationRestoreConfig {
    fn default() -> Self {
        // chinese-lert-base max_position_embeddings (512) - leading [CLS].
        Self {
            max_content_tokens: 511,
        }
    }
}

/// Whether the punctuation stage should run for a model with the given
/// `emits_punctuation` capability. Only unpunctuated families (`Some(false)`)
/// opt in; `None` ("unknown", treated as already punctuated) and `Some(true)`
/// are left untouched.
pub(crate) fn should_apply_punctuation(emits_punctuation: Option<bool>) -> bool {
    emits_punctuation == Some(false)
}

/// Restore punctuation on `text` using `classifier`, returning the punctuated
/// string. Insertion points come from the tokenizer's char spans, so the
/// original characters (casing included) are preserved verbatim; only
/// punctuation marks are added.
pub(crate) fn restore_punctuation(
    text: &str,
    tokenizer: &FireRedPuncTokenizer,
    classifier: &dyn PunctuationClassifier,
    config: PunctuationRestoreConfig,
) -> Result<String, PunctuationError> {
    let pieces = tokenizer.encode(text);
    if pieces.is_empty() {
        return Ok(text.to_string());
    }

    // (char_end offset in the original text, punctuation mark) insertions.
    let mut insertions: Vec<(usize, char)> = Vec::new();
    for window in fixed_token_windows(&pieces, config.max_content_tokens) {
        let ids: Vec<u32> = window.iter().map(|piece| piece.token_id).collect();
        let labels = classifier.predict_window_labels(&ids)?;
        if labels.len() != window.len() {
            return Err(PunctuationError::LabelCountMismatch {
                got: labels.len(),
                expected: window.len(),
            });
        }
        for (piece, label) in window.iter().zip(labels) {
            // Only attach at word-final subwords so a mark never lands inside a
            // word; ignore the "no punctuation" class (label 0).
            if !piece.word_final {
                continue;
            }
            if let Some(mark) = punctuation_for_label(label) {
                insertions.push((piece.char_end, mark));
            }
        }
    }

    Ok(apply_insertions(text, &mut insertions))
}

/// Split content pieces into the upstream model's fixed-width windows.
///
/// FireRedPunc tokenizes the whole input first and then applies
/// `Tensor::split(max_position_embeddings - 1)`. A window boundary may
/// therefore fall between two WordPiece subwords. Punctuation is still
/// attached only to `word_final` pieces below, so matching that encoder
/// context does not insert a mark inside a word.
fn fixed_token_windows(
    pieces: &[WordPiecePiece],
    max_content_tokens: usize,
) -> Vec<&[WordPiecePiece]> {
    let budget = max_content_tokens.max(1);
    pieces.chunks(budget).collect()
}

/// Insert punctuation marks into `text` at the given original char offsets.
/// Skips an insertion whose slot is already occupied by punctuation (avoids
/// doubling when the transcript already carries a mark).
fn apply_insertions(text: &str, insertions: &mut [(usize, char)]) -> String {
    let chars: Vec<char> = text.chars().collect();
    // Apply right-to-left so earlier offsets stay valid; dedup by offset.
    insertions.sort_by_key(|&(offset, _)| std::cmp::Reverse(offset));
    let mut result = chars.clone();
    let mut last_offset: Option<usize> = None;
    for &(offset, mark) in insertions.iter() {
        if offset > result.len() {
            continue;
        }
        if last_offset == Some(offset) {
            continue;
        }
        last_offset = Some(offset);
        // Skip when the transcript already has a punctuation mark at this slot.
        if offset < chars.len() && is_existing_punctuation(chars[offset]) {
            continue;
        }
        if offset > 0 && is_existing_punctuation(chars[offset - 1]) {
            continue;
        }
        result.insert(offset, mark);
    }
    result.into_iter().collect()
}

fn is_existing_punctuation(c: char) -> bool {
    matches!(c, '，' | '。' | '？' | '！' | '、' | '；' | '：') || c.is_ascii_punctuation()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tokenizer() -> FireRedPuncTokenizer {
        let vocab = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "你", "好", "世", "界", "hello", "wor", "##ld",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        FireRedPuncTokenizer::new(vocab).expect("vocab has special tokens")
    }

    /// Returns caller-scripted labels per content token, in order across all
    /// windows (the harness concatenates window predictions).
    struct ScriptedClassifier {
        labels: std::cell::RefCell<std::collections::VecDeque<usize>>,
    }

    impl ScriptedClassifier {
        fn new(labels: Vec<usize>) -> Self {
            Self {
                labels: std::cell::RefCell::new(labels.into()),
            }
        }
    }

    impl PunctuationClassifier for ScriptedClassifier {
        fn predict_window_labels(
            &self,
            content_token_ids: &[u32],
        ) -> Result<Vec<usize>, PunctuationError> {
            let mut queue = self.labels.borrow_mut();
            let out: Vec<usize> = content_token_ids
                .iter()
                .map(|_| queue.pop_front().unwrap_or(0))
                .collect();
            Ok(out)
        }
    }

    #[test]
    fn gating_only_runs_for_unpunctuated_models() {
        assert!(should_apply_punctuation(Some(false)));
        assert!(!should_apply_punctuation(Some(true)));
        assert!(!should_apply_punctuation(None));
    }

    #[test]
    fn readme_golden_period_after_final_char() {
        // "你好世界" -> label 2 ('。') on the last char, none elsewhere.
        let tok = test_tokenizer();
        let classifier = ScriptedClassifier::new(vec![0, 0, 0, 2]);
        let out = restore_punctuation(
            "你好世界",
            &tok,
            &classifier,
            PunctuationRestoreConfig::default(),
        )
        .expect("restore");
        assert_eq!(out, "你好世界。");
    }

    #[test]
    fn comma_and_period_are_inserted_at_char_boundaries() {
        // "你好世界" -> comma after '好' (idx1), period after '界' (idx3).
        let tok = test_tokenizer();
        let classifier = ScriptedClassifier::new(vec![0, 1, 0, 2]);
        let out = restore_punctuation(
            "你好世界",
            &tok,
            &classifier,
            PunctuationRestoreConfig::default(),
        )
        .expect("restore");
        assert_eq!(out, "你好，世界。");
    }

    #[test]
    fn mid_word_labels_are_ignored_only_word_final_attaches() {
        // "world" -> pieces [wor(not final), ##ld(final)]. A label on the
        // non-final piece must be dropped; the mark attaches after the word.
        let tok = test_tokenizer();
        let classifier = ScriptedClassifier::new(vec![2, 1]);
        let out = restore_punctuation(
            "world",
            &tok,
            &classifier,
            PunctuationRestoreConfig::default(),
        )
        .expect("restore");
        assert_eq!(out, "world，", "period on 'wor' dropped; comma after word");
    }

    #[test]
    fn original_casing_and_spacing_preserved() {
        let tok = test_tokenizer();
        // "HELLO" is one piece; label period after it.
        let classifier = ScriptedClassifier::new(vec![2]);
        let out = restore_punctuation(
            "HELLO",
            &tok,
            &classifier,
            PunctuationRestoreConfig::default(),
        )
        .expect("restore");
        assert_eq!(out, "HELLO。", "uppercase preserved, mark appended");
    }

    #[test]
    fn existing_punctuation_is_not_doubled() {
        let tok = test_tokenizer();
        // Text already ends with a period char at index 4; predicting another
        // period after '界' (char_end 4) must be skipped.
        let classifier = ScriptedClassifier::new(vec![0, 0, 0, 2]);
        let out = restore_punctuation(
            "你好世界。",
            &tok,
            &classifier,
            PunctuationRestoreConfig::default(),
        )
        .expect("restore");
        assert_eq!(out, "你好世界。");
    }

    #[test]
    fn empty_text_is_returned_unchanged() {
        let tok = test_tokenizer();
        let classifier = ScriptedClassifier::new(vec![]);
        let out = restore_punctuation("", &tok, &classifier, PunctuationRestoreConfig::default())
            .expect("restore");
        assert_eq!(out, "");
    }

    #[test]
    fn windows_match_the_upstream_fixed_token_split() {
        // Force a tiny budget so "你好世界" splits into multiple windows; every
        // char must still be classified and its label applied.
        let tok = test_tokenizer();
        let pieces = tok.encode("你好世界");
        let windows = fixed_token_windows(&pieces, 2);
        assert!(windows.len() >= 2, "tiny budget splits the sequence");
        let total: usize = windows.iter().map(|w| w.len()).sum();
        assert_eq!(total, pieces.len(), "windows partition all pieces");

        let classifier = ScriptedClassifier::new(vec![0, 1, 0, 2]);
        let cfg = PunctuationRestoreConfig {
            max_content_tokens: 2,
        };
        let out = restore_punctuation("你好世界", &tok, &classifier, cfg).expect("restore");
        assert_eq!(out, "你好，世界。");
    }

    #[test]
    fn fixed_windows_may_split_wordpieces_without_inserting_inside_a_word() {
        let tok = test_tokenizer();
        let pieces = tok.encode("hello world");
        let windows = fixed_token_windows(&pieces, 2);
        assert_eq!(
            windows
                .iter()
                .map(|window| window.len())
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert!(!windows[0][1].word_final, "first window ends inside world");
        assert!(
            windows[1][0].word_final,
            "second window owns word-final label"
        );

        let classifier = ScriptedClassifier::new(vec![2, 1, 2]);
        let out = restore_punctuation(
            "hello world",
            &tok,
            &classifier,
            PunctuationRestoreConfig {
                max_content_tokens: 2,
            },
        )
        .expect("restore");
        assert_eq!(out, "hello。 world。");
    }
}
