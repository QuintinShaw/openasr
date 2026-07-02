//! X-ASR SentencePiece-style BPE detokenizer.

use crate::api::backend::WordTimestamp;
use crate::ggml_runtime::GgufMetadata;
use crate::models::sentencepiece_word_timestamps::{
    TimedSentencePieceToken, WORD_START_MARKER, assemble_sentencepiece_word_timestamps,
    frame_to_seconds,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XasrZipformerTokenizer {
    tokens: Vec<String>,
    blank_id: u32,
}

impl XasrZipformerTokenizer {
    pub(crate) fn from_metadata(metadata: &GgufMetadata, blank_id: u32) -> Result<Self, String> {
        let tokens = metadata
            .get_string_array("tokenizer.ggml.tokens")
            .ok_or_else(|| "xasr-zipformer pack missing tokenizer.ggml.tokens".to_string())?
            .to_vec();
        Self::new(tokens, blank_id)
    }

    pub(crate) fn new(tokens: Vec<String>, blank_id: u32) -> Result<Self, String> {
        if tokens.is_empty() {
            return Err("xasr-zipformer tokenizer vocab is empty".to_string());
        }
        if blank_id as usize >= tokens.len() {
            return Err(format!(
                "xasr-zipformer blank id {blank_id} out of range for vocab {}",
                tokens.len()
            ));
        }
        Ok(Self { tokens, blank_id })
    }

    pub(crate) fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let mut out = String::new();
        for &id in ids {
            if id == self.blank_id {
                continue;
            }
            let token = self.tokens.get(id as usize).ok_or_else(|| {
                format!(
                    "xasr-zipformer token id {id} out of range (vocab {})",
                    self.tokens.len()
                )
            })?;
            if token.starts_with('<') && token.ends_with('>') {
                continue;
            }
            out.push_str(token);
        }
        Ok(normalize_sentencepiece_spacing(
            &out.replace(WORD_START_MARKER, " "),
        ))
    }

    pub(crate) fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// Word timestamps from RNN-T emission alignment: each token's span runs
    /// from its emission frame to the next frame, mapped proportionally onto
    /// the clip duration. Emission instants are the real alignment a
    /// transducer provides — several tokens may legitimately share a frame.
    /// Blank and `<...>` special tokens are skipped, mirroring `decode`.
    /// `emit_probabilities` (joiner softmax per emission, parallel to `ids`)
    /// feeds word confidence; empty means "not captured" and yields `None`.
    pub(crate) fn word_timestamps_from_emission_frames(
        &self,
        ids: &[u32],
        emit_frames: &[usize],
        emit_probabilities: &[f32],
        encoder_frames: usize,
        duration_seconds: f32,
    ) -> Result<Vec<WordTimestamp>, String> {
        if ids.len() != emit_frames.len() {
            return Err(format!(
                "xasr-zipformer emission alignment mismatch: {} tokens vs {} frames",
                ids.len(),
                emit_frames.len()
            ));
        }
        if !emit_probabilities.is_empty() && emit_probabilities.len() != ids.len() {
            return Err(format!(
                "xasr-zipformer emission probability mismatch: {} tokens vs {} probabilities",
                ids.len(),
                emit_probabilities.len()
            ));
        }
        if ids.is_empty() || encoder_frames == 0 || !duration_seconds.is_finite() {
            return Ok(Vec::new());
        }
        let mut timed = Vec::with_capacity(ids.len());
        for (index, (&id, &frame)) in ids.iter().zip(emit_frames).enumerate() {
            if id == self.blank_id {
                continue;
            }
            let token = self.tokens.get(id as usize).ok_or_else(|| {
                format!(
                    "xasr-zipformer token id {id} out of range (vocab {})",
                    self.tokens.len()
                )
            })?;
            if token.starts_with('<') && token.ends_with('>') {
                continue;
            }
            let start_seconds = frame_to_seconds(frame, encoder_frames, duration_seconds);
            let end_seconds =
                frame_to_seconds(frame.saturating_add(1), encoder_frames, duration_seconds);
            timed.push(TimedSentencePieceToken {
                token,
                start_seconds,
                end_seconds: end_seconds.max(start_seconds),
                probability: emit_probabilities.get(index).copied(),
            });
        }
        Ok(assemble_sentencepiece_word_timestamps(timed))
    }
}

/// Exact streaming equivalent of [`XasrZipformerTokenizer::decode`]: feeding
/// tokens one at a time yields, at every step, byte-for-byte the same string
/// `decode` would produce for the tokens seen so far — and the output is
/// append-only across steps, so callers can take deltas without re-decoding
/// the whole history. This works because the spacing rule only needs the
/// nearest non-space neighbors of each space, and the final trim is exactly
/// "drop leading spaces / withhold trailing spaces until a visible char
/// arrives".
#[derive(Debug, Clone, Default)]
pub(crate) struct XasrStreamingDetokenizer {
    out: String,
    pending_spaces: usize,
    last_visible: Option<char>,
}

impl XasrStreamingDetokenizer {
    pub(crate) fn push_token(
        &mut self,
        tokenizer: &XasrZipformerTokenizer,
        id: u32,
    ) -> Result<(), String> {
        if id == tokenizer.blank_id {
            return Ok(());
        }
        let token = tokenizer.tokens.get(id as usize).ok_or_else(|| {
            format!(
                "xasr-zipformer token id {id} out of range (vocab {})",
                tokenizer.tokens.len()
            )
        })?;
        if token.starts_with('<') && token.ends_with('>') {
            return Ok(());
        }
        for ch in token.chars() {
            let ch = if ch == WORD_START_MARKER { ' ' } else { ch };
            if ch == ' ' {
                self.pending_spaces += 1;
                continue;
            }
            if self.pending_spaces > 0 {
                if let Some(left) = self.last_visible
                    && !should_suppress_sentencepiece_space(left, ch)
                {
                    for _ in 0..self.pending_spaces {
                        self.out.push(' ');
                    }
                }
                self.pending_spaces = 0;
            }
            self.out.push(ch);
            self.last_visible = Some(ch);
        }
        Ok(())
    }

    pub(crate) fn text(&self) -> &str {
        &self.out
    }

    /// Drops already-returned text while keeping token-derived spacing state
    /// for the next segment.
    pub(crate) fn rebase_preserving_boundary_context(&mut self) {
        self.out.clear();
    }

    pub(crate) fn reset(&mut self) {
        self.out.clear();
        self.pending_spaces = 0;
        self.last_visible = None;
    }
}

fn normalize_sentencepiece_spacing(text: &str) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let mut out = String::with_capacity(text.len());
    for (index, ch) in chars.iter().copied().enumerate() {
        if ch == ' ' {
            let left = chars[..index]
                .iter()
                .rev()
                .copied()
                .find(|candidate| *candidate != ' ');
            let right = chars[index + 1..]
                .iter()
                .copied()
                .find(|candidate| *candidate != ' ');
            if let (Some(left), Some(right)) = (left, right)
                && should_suppress_sentencepiece_space(left, right)
            {
                continue;
            }
        }
        out.push(ch);
    }
    out.trim().to_string()
}

fn should_suppress_sentencepiece_space(left: char, right: char) -> bool {
    is_cjk_text_or_punctuation(left) && is_cjk_text_or_punctuation(right)
}

fn is_cjk_text_or_punctuation(ch: char) -> bool {
    matches!(
        ch as u32,
        0x2E80..=0x2EFF
            | 0x3000..=0x303F
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0xFE10..=0xFE1F
            | 0xFF01..=0xFF0F
            | 0xFF1A..=0xFF20
            | 0xFF3B..=0xFF40
            | 0xFF5B..=0xFF65
            | 0x20000..=0x2EBEF
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_sentencepiece_word_marker_and_skips_specials() {
        let tokenizer = XasrZipformerTokenizer::new(
            vec![
                "<blk>".to_string(),
                "<sos/eos>".to_string(),
                "\u{2581}你好".to_string(),
                "\u{2581}world".to_string(),
                "!".to_string(),
            ],
            0,
        )
        .unwrap();
        assert_eq!(tokenizer.decode(&[0, 1, 2, 3, 4]).unwrap(), "你好 world!");
    }

    #[test]
    fn removes_sentencepiece_spaces_between_cjk_pieces() {
        let tokenizer = XasrZipformerTokenizer::new(
            vec![
                "<blk>".to_string(),
                "\u{2581}二".to_string(),
                "\u{2581}零".to_string(),
                "\u{2581}年".to_string(),
                "\u{2581}，".to_string(),
                "\u{2581}美国".to_string(),
                "\u{2581}OpenASR".to_string(),
            ],
            0,
        )
        .unwrap();
        assert_eq!(
            tokenizer.decode(&[1, 2, 3, 4, 5, 6]).unwrap(),
            "二零年，美国 OpenASR"
        );
    }

    #[test]
    fn streaming_detokenizer_matches_batch_decode_at_every_prefix() {
        let tokenizer = XasrZipformerTokenizer::new(
            vec![
                "<blk>".to_string(),
                "<sos/eos>".to_string(),
                "\u{2581}二".to_string(),
                "\u{2581}零".to_string(),
                "\u{2581}，".to_string(),
                "\u{2581}美国".to_string(),
                "\u{2581}OpenASR".to_string(),
                "\u{2581}is".to_string(),
                "!".to_string(),
                "\u{2581}\u{2581}gap".to_string(),
            ],
            0,
        )
        .unwrap();
        // Mixed CJK/Latin, specials, blanks, double markers, punctuation.
        let ids = [0u32, 2, 3, 1, 4, 5, 6, 0, 7, 8, 9, 2];
        let mut streaming = XasrStreamingDetokenizer::default();
        for (count, &id) in ids.iter().enumerate() {
            streaming.push_token(&tokenizer, id).unwrap();
            let batch = tokenizer.decode(&ids[..=count]).unwrap();
            assert_eq!(streaming.text(), batch, "prefix of {} tokens", count + 1);
        }
    }

    #[test]
    fn streaming_detokenizer_output_is_append_only() {
        let tokenizer = XasrZipformerTokenizer::new(
            vec![
                "<blk>".to_string(),
                "\u{2581}你好".to_string(),
                "\u{2581}world".to_string(),
            ],
            0,
        )
        .unwrap();
        let mut streaming = XasrStreamingDetokenizer::default();
        let mut previous = String::new();
        for id in [1u32, 2, 1] {
            streaming.push_token(&tokenizer, id).unwrap();
            assert!(streaming.text().starts_with(&previous));
            previous = streaming.text().to_string();
        }
    }

    #[test]
    fn streaming_detokenizer_rebase_preserves_sentencepiece_boundary_spacing() {
        let tokenizer = XasrZipformerTokenizer::new(
            vec![
                "<blk>".to_string(),
                "\u{2581}hello".to_string(),
                "\u{2581}world".to_string(),
                "\u{2581}二".to_string(),
                "\u{2581}零".to_string(),
                "\u{2581}".to_string(),
                "tail".to_string(),
            ],
            0,
        )
        .unwrap();

        let mut latin = XasrStreamingDetokenizer::default();
        latin.push_token(&tokenizer, 1).unwrap();
        latin.rebase_preserving_boundary_context();
        latin.push_token(&tokenizer, 2).unwrap();
        assert_eq!(latin.text(), " world");

        let mut cjk = XasrStreamingDetokenizer::default();
        cjk.push_token(&tokenizer, 3).unwrap();
        cjk.rebase_preserving_boundary_context();
        cjk.push_token(&tokenizer, 4).unwrap();
        assert_eq!(cjk.text(), "零");

        let mut pending_space = XasrStreamingDetokenizer::default();
        pending_space.push_token(&tokenizer, 1).unwrap();
        pending_space.push_token(&tokenizer, 5).unwrap();
        pending_space.rebase_preserving_boundary_context();
        pending_space.push_token(&tokenizer, 6).unwrap();
        assert_eq!(pending_space.text(), " tail");
    }

    #[test]
    fn word_timestamps_skip_specials_and_map_emission_frames_proportionally() {
        let tokenizer = XasrZipformerTokenizer::new(
            vec![
                "<blk>".to_string(),
                "<sos/eos>".to_string(),
                "\u{2581}你好".to_string(),
                "\u{2581}wor".to_string(),
                "ld".to_string(),
            ],
            0,
        )
        .unwrap();
        // 100 encoder frames over 10s; specials and blank carry frames too but
        // must be dropped from the word stream.
        let words = tokenizer
            .word_timestamps_from_emission_frames(
                &[1, 2, 0, 3, 4],
                &[0, 10, 11, 50, 80],
                &[0.9, 0.8, 1.0, 0.6, 0.4],
                100,
                10.0,
            )
            .unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "你好");
        assert!((words[0].start - 1.0).abs() < 1e-6);
        assert!((words[0].end - 1.1).abs() < 1e-6);
        assert_eq!(words[1].word, "world");
        assert!((words[1].start - 5.0).abs() < 1e-6);
        assert!((words[1].end - 8.1).abs() < 1e-6);
    }

    #[test]
    fn word_timestamps_reject_misaligned_inputs_and_handle_empty() {
        let tokenizer =
            XasrZipformerTokenizer::new(vec!["<blk>".to_string(), "\u{2581}a".to_string()], 0)
                .unwrap();
        assert!(
            tokenizer
                .word_timestamps_from_emission_frames(&[1], &[], &[], 10, 1.0)
                .is_err()
        );
        assert!(
            tokenizer
                .word_timestamps_from_emission_frames(&[], &[], &[], 10, 1.0)
                .unwrap()
                .is_empty()
        );
        assert!(
            tokenizer
                .word_timestamps_from_emission_frames(&[1], &[0], &[1.0], 0, 1.0)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn rejects_out_of_range_ids() {
        let tokenizer =
            XasrZipformerTokenizer::new(vec!["<blk>".to_string(), "a".to_string()], 0).unwrap();
        assert!(tokenizer.decode(&[2]).is_err());
    }
}
