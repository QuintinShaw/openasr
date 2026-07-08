//! Offset-preserving BERT WordPiece tokenizer for FireRedPunc.
//!
//! Unlike the ASR families (which only ever *detokenize*), punctuation
//! restoration must *encode* transcript text into subword ids to feed the BERT
//! classifier, then map each predicted label back onto the original characters.
//! So this tokenizer tracks, for every content subword, the char span it came
//! from in the *original* (un-normalised) text -- the classifier stage inserts
//! punctuation at those offsets and never re-emits normalised text, preserving
//! the transcript's exact casing and characters.
//!
//! Normalisation matches BERT uncased basic tokenisation minus accent
//! stripping: control chars dropped, whitespace splits words, each CJK char and
//! each punctuation char is its own word, ASCII letters lowercased. Accent
//! stripping is intentionally omitted (ASR transcript text is Chinese/English
//! with rare combining marks; those fall back to `[UNK]`), keeping this free of
//! a unicode-normalisation dependency.

const CLS_TOKEN: &str = "[CLS]";
const SEP_TOKEN: &str = "[SEP]";
const UNK_TOKEN: &str = "[UNK]";
const PAD_TOKEN: &str = "[PAD]";
const WORDPIECE_CONTINUATION: &str = "##";
/// BERT's `max_input_chars_per_word`: longer words map straight to `[UNK]`.
const MAX_INPUT_CHARS_PER_WORD: usize = 100;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum FireRedPuncTokenizerError {
    #[error("firered-punc vocab is missing required special token '{0}'")]
    MissingSpecialToken(&'static str),
    #[error("firered-punc vocab is empty")]
    EmptyVocab,
}

/// One content subword and the char span it occupies in the original text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WordPiecePiece {
    pub token_id: u32,
    /// Char index (into the original string) of the first source char.
    pub char_start: usize,
    /// Char index one past the last source char (exclusive).
    pub char_end: usize,
    /// True when this is the last subword of its word, i.e. a legal place to
    /// attach a following punctuation mark. Mid-word continuation pieces
    /// (`##...`) are never word-final, so punctuation can never land inside a
    /// word.
    pub word_final: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct FireRedPuncTokenizer {
    vocab: Vec<String>,
    token_to_id: std::collections::HashMap<String, u32>,
    cls_id: u32,
    sep_id: u32,
    unk_id: u32,
    pad_id: u32,
}

impl FireRedPuncTokenizer {
    pub(crate) fn new(vocab: Vec<String>) -> Result<Self, FireRedPuncTokenizerError> {
        if vocab.is_empty() {
            return Err(FireRedPuncTokenizerError::EmptyVocab);
        }
        let mut token_to_id = std::collections::HashMap::with_capacity(vocab.len());
        for (idx, token) in vocab.iter().enumerate() {
            token_to_id.entry(token.clone()).or_insert(idx as u32);
        }
        let lookup = |tok: &'static str| {
            token_to_id
                .get(tok)
                .copied()
                .ok_or(FireRedPuncTokenizerError::MissingSpecialToken(tok))
        };
        let cls_id = lookup(CLS_TOKEN)?;
        let sep_id = lookup(SEP_TOKEN)?;
        let unk_id = lookup(UNK_TOKEN)?;
        let pad_id = lookup(PAD_TOKEN)?;
        Ok(Self {
            vocab,
            token_to_id,
            cls_id,
            sep_id,
            unk_id,
            pad_id,
        })
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub(crate) fn cls_id(&self) -> u32 {
        self.cls_id
    }

    pub(crate) fn sep_id(&self) -> u32 {
        self.sep_id
    }

    pub(crate) fn pad_id(&self) -> u32 {
        self.pad_id
    }

    /// Tokenise `text` into content subwords with original-text char spans.
    /// The result excludes `[CLS]`/`[SEP]`; call [`Self::window_input_ids`] to
    /// wrap a slice of pieces into model input ids.
    pub(crate) fn encode(&self, text: &str) -> Vec<WordPiecePiece> {
        let chars: Vec<char> = text.chars().collect();
        let mut pieces = Vec::new();
        let mut idx = 0usize;
        while idx < chars.len() {
            let c = chars[idx];
            if is_whitespace(c) || is_control(c) {
                idx += 1;
                continue;
            }
            if is_cjk_char(c) || is_split_punctuation(c) {
                // CJK chars and punctuation each form a standalone word.
                self.push_word(&chars[idx..idx + 1], idx, &mut pieces);
                idx += 1;
                continue;
            }
            // Accumulate a maximal run of non-whitespace, non-CJK,
            // non-standalone-punctuation chars into one word.
            let start = idx;
            while idx < chars.len() {
                let cc = chars[idx];
                if is_whitespace(cc)
                    || is_control(cc)
                    || is_cjk_char(cc)
                    || is_split_punctuation(cc)
                {
                    break;
                }
                idx += 1;
            }
            self.push_word(&chars[start..idx], start, &mut pieces);
        }
        pieces
    }

    /// WordPiece-tokenise a single word (`word_chars`, whose first char is at
    /// original char index `word_start`) and append the resulting pieces.
    fn push_word(&self, word_chars: &[char], word_start: usize, out: &mut Vec<WordPiecePiece>) {
        // Normalised (lowercased) form used only for vocab lookup; spans are
        // always expressed against the original indices.
        let normalised: String = word_chars.iter().flat_map(|c| c.to_lowercase()).collect();
        let norm_chars: Vec<char> = normalised.chars().collect();

        // Lowercasing is 1:1 for the scripts ASR emits; when it is not (rare),
        // fall back to whole-word [UNK] so spans stay trustworthy.
        let spans_aligned = norm_chars.len() == word_chars.len();
        if !spans_aligned || norm_chars.len() > MAX_INPUT_CHARS_PER_WORD {
            out.push(WordPiecePiece {
                token_id: self.unk_id,
                char_start: word_start,
                char_end: word_start + word_chars.len(),
                word_final: true,
            });
            return;
        }

        // Greedy longest-match-first WordPiece over the normalised chars.
        let mut sub_pieces: Vec<WordPiecePiece> = Vec::new();
        let mut cursor = 0usize;
        let mut is_bad = false;
        while cursor < norm_chars.len() {
            let mut end = norm_chars.len();
            let mut matched: Option<u32> = None;
            let mut matched_end = cursor;
            while end > cursor {
                let mut candidate: String = norm_chars[cursor..end].iter().collect();
                if cursor > 0 {
                    candidate.insert_str(0, WORDPIECE_CONTINUATION);
                }
                if let Some(&id) = self.token_to_id.get(&candidate) {
                    matched = Some(id);
                    matched_end = end;
                    break;
                }
                end -= 1;
            }
            match matched {
                Some(id) => {
                    sub_pieces.push(WordPiecePiece {
                        token_id: id,
                        char_start: word_start + cursor,
                        char_end: word_start + matched_end,
                        word_final: false,
                    });
                    cursor = matched_end;
                }
                None => {
                    is_bad = true;
                    break;
                }
            }
        }

        if is_bad || sub_pieces.is_empty() {
            out.push(WordPiecePiece {
                token_id: self.unk_id,
                char_start: word_start,
                char_end: word_start + word_chars.len(),
                word_final: true,
            });
            return;
        }

        if let Some(last) = sub_pieces.last_mut() {
            last.word_final = true;
        }
        out.extend(sub_pieces);
    }

    /// Wrap a window of content pieces into `[CLS] ... [SEP]` model input ids.
    pub(crate) fn window_input_ids(&self, window: &[WordPiecePiece]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(window.len() + 2);
        ids.push(self.cls_id);
        ids.extend(window.iter().map(|piece| piece.token_id));
        ids.push(self.sep_id);
        ids
    }
}

fn is_whitespace(c: char) -> bool {
    c == ' ' || c == '\t' || c == '\n' || c == '\r' || c.is_whitespace()
}

fn is_control(c: char) -> bool {
    if c == '\t' || c == '\n' || c == '\r' {
        return false;
    }
    c.is_control()
}

/// Punctuation that BERT basic tokenisation splits into its own token: all
/// ASCII punctuation plus any Unicode punctuation category char. Splitting them
/// out means transcript punctuation the model *does* see stays isolated.
fn is_split_punctuation(c: char) -> bool {
    let cp = c as u32;
    let ascii_punct = (0x21..=0x2F).contains(&cp)
        || (0x3A..=0x40).contains(&cp)
        || (0x5B..=0x60).contains(&cp)
        || (0x7B..=0x7E).contains(&cp);
    ascii_punct || c.is_ascii_punctuation() || is_unicode_punctuation(c)
}

fn is_unicode_punctuation(c: char) -> bool {
    matches!(
        c,
        '\u{3001}'..='\u{3003}'
            | '\u{3008}'..='\u{3011}'
            | '\u{FF01}'..='\u{FF0F}'
            | '\u{FF1A}'..='\u{FF20}'
            | '\u{FF3B}'..='\u{FF40}'
            | '\u{FF5B}'..='\u{FF65}'
            | '\u{2018}'..='\u{201F}'
    )
}

/// BERT `_is_chinese_char`: the CJK Unified Ideograph blocks. Note this is the
/// upstream definition -- it deliberately excludes CJK punctuation/kana, which
/// are handled as punctuation/other words.
fn is_cjk_char(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal synthetic WordPiece vocab: the special tokens, a few CJK chars,
    /// and English pieces for "hello"/"world" exercising `##` continuation.
    fn test_tokenizer() -> FireRedPuncTokenizer {
        let vocab = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "你", "好", "世", "界", "hello", "wor", "##ld",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        FireRedPuncTokenizer::new(vocab).expect("vocab has special tokens")
    }

    #[test]
    fn missing_special_token_is_rejected() {
        let err = FireRedPuncTokenizer::new(vec!["你".to_string()]).unwrap_err();
        assert_eq!(err, FireRedPuncTokenizerError::MissingSpecialToken("[CLS]"));
    }

    #[test]
    fn cjk_is_one_piece_per_char_with_char_spans() {
        let tok = test_tokenizer();
        let pieces = tok.encode("你好世界");
        assert_eq!(pieces.len(), 4);
        for (i, piece) in pieces.iter().enumerate() {
            assert_eq!(piece.char_start, i);
            assert_eq!(piece.char_end, i + 1);
            assert!(piece.word_final, "every CJK char is its own word");
        }
        let ids: Vec<u32> = pieces.iter().map(|p| p.token_id).collect();
        assert_eq!(ids, vec![4, 5, 6, 7]);
    }

    #[test]
    fn window_ids_wrap_with_cls_and_sep() {
        let tok = test_tokenizer();
        let pieces = tok.encode("你好");
        let ids = tok.window_input_ids(&pieces);
        assert_eq!(ids, vec![tok.cls_id(), 4, 5, tok.sep_id()]);
    }

    #[test]
    fn english_word_splits_into_wordpieces_only_last_is_word_final() {
        let tok = test_tokenizer();
        let pieces = tok.encode("world");
        assert_eq!(pieces.len(), 2, "wor + ##ld");
        assert_eq!(pieces[0].token_id, 9);
        assert!(!pieces[0].word_final);
        assert_eq!(pieces[0].char_start, 0);
        assert_eq!(pieces[0].char_end, 3);
        assert_eq!(pieces[1].token_id, 10);
        assert!(pieces[1].word_final);
        assert_eq!(pieces[1].char_start, 3);
        assert_eq!(pieces[1].char_end, 5);
    }

    #[test]
    fn lowercasing_preserves_original_spans() {
        let tok = test_tokenizer();
        let pieces = tok.encode("HELLO");
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].token_id, 8, "lowercased to the 'hello' piece");
        assert_eq!(pieces[0].char_start, 0);
        assert_eq!(pieces[0].char_end, 5, "span is against the original text");
        assert!(pieces[0].word_final);
    }

    #[test]
    fn unknown_word_falls_back_to_unk_spanning_the_word() {
        let tok = test_tokenizer();
        let pieces = tok.encode("zzz");
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].token_id, tok.unk_id);
        assert_eq!(pieces[0].char_start, 0);
        assert_eq!(pieces[0].char_end, 3);
        assert!(pieces[0].word_final);
    }

    #[test]
    fn whitespace_between_words_is_skipped_but_offsets_track_original() {
        let tok = test_tokenizer();
        let pieces = tok.encode("你 好");
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0].char_start, 0);
        assert_eq!(pieces[0].char_end, 1);
        // The second CJK char sits at original index 2 (after the space).
        assert_eq!(pieces[1].char_start, 2);
        assert_eq!(pieces[1].char_end, 3);
    }
}
