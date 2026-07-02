//! Shared SentencePiece word-timestamp assembly: fold a stream of timed vocab
//! pieces (with the `▁` U+2581 word-start marker) into [`WordTimestamp`]
//! words. Family tokenizers resolve ids to piece strings and map their own
//! alignment space (CTC frame spans, RNN-T emission frames, ...) to seconds;
//! the word-boundary folding here is alignment-source agnostic.

use crate::api::backend::WordTimestamp;

pub(crate) const WORD_START_MARKER: char = '\u{2581}';

/// One decoded vocab piece with its acoustic span in seconds.
pub(crate) struct TimedSentencePieceToken<'a> {
    pub token: &'a str,
    pub start_seconds: f32,
    pub end_seconds: f32,
    /// Softmax probability of this piece at its decode step, when the family
    /// decoder captured one. A word's confidence is the mean over its pieces.
    pub probability: Option<f32>,
}

/// Folds timed pieces into words: a `▁` marker closes the current word and
/// starts the next one; unmarked pieces extend the current word. Pieces that
/// bundle several markers are split, interpolating time linearly by character
/// count within the piece's span.
pub(crate) fn assemble_sentencepiece_word_timestamps<'a>(
    tokens: impl IntoIterator<Item = TimedSentencePieceToken<'a>>,
) -> Vec<WordTimestamp> {
    let mut words = Vec::new();
    let mut current = CurrentWord::default();

    for timed in tokens {
        let token_start = timed.start_seconds;
        let token_end = timed.end_seconds.max(token_start);
        if timed.token.contains(WORD_START_MARKER) {
            append_marked_sentencepiece_token(
                timed.token,
                token_start,
                token_end,
                timed.probability,
                &mut words,
                &mut current,
            );
        } else {
            current.append_piece(timed.token, token_start, token_end, timed.probability);
        }
    }
    current.push_word_timestamp(&mut words);
    words
}

/// Proportional alignment-frame to wall-clock mapping over the full clip.
pub(crate) fn frame_to_seconds(frame: usize, frame_count: usize, duration_seconds: f32) -> f32 {
    duration_seconds.max(0.0) * frame.min(frame_count) as f32 / frame_count as f32
}

/// Word-in-progress accumulator: text + span + mean-probability state.
#[derive(Default)]
struct CurrentWord {
    text: String,
    start: f32,
    end: f32,
    probability_sum: f32,
    probability_count: usize,
}

impl CurrentWord {
    fn append_piece(&mut self, piece: &str, start: f32, end: f32, probability: Option<f32>) {
        if piece.is_empty() {
            return;
        }
        if self.text.is_empty() {
            self.start = start;
        }
        self.text.push_str(piece);
        self.end = end.max(start);
        if let Some(probability) = probability {
            self.probability_sum += probability;
            self.probability_count += 1;
        }
    }

    fn push_word_timestamp(&mut self, words: &mut Vec<WordTimestamp>) {
        if self.text.trim().is_empty() {
            self.reset();
            return;
        }
        let confidence = (self.probability_count > 0)
            .then(|| (self.probability_sum / self.probability_count as f32).clamp(0.0, 1.0));
        words.push(WordTimestamp {
            word: self.text.trim().to_string(),
            start: self.start,
            end: self.end.max(self.start),
            confidence,
        });
        self.reset();
    }

    fn reset(&mut self) {
        self.text.clear();
        self.start = 0.0;
        self.end = 0.0;
        self.probability_sum = 0.0;
        self.probability_count = 0;
    }
}

fn append_marked_sentencepiece_token(
    token: &str,
    token_start: f32,
    token_end: f32,
    probability: Option<f32>,
    words: &mut Vec<WordTimestamp>,
    current: &mut CurrentWord,
) {
    let pieces = token.split(WORD_START_MARKER).collect::<Vec<_>>();
    let total_chars = pieces
        .iter()
        .map(|piece| piece.chars().count())
        .sum::<usize>();
    let token_duration = (token_end - token_start).max(0.0);
    let mut cursor = token_start;
    let mut consumed_chars = 0usize;

    for (index, piece) in pieces.iter().enumerate() {
        if index > 0 {
            current.push_word_timestamp(words);
        }
        if piece.is_empty() {
            continue;
        }
        let piece_chars = piece.chars().count().max(1);
        consumed_chars = consumed_chars.saturating_add(piece_chars);
        let piece_end = if total_chars == 0 || consumed_chars >= total_chars {
            token_end
        } else {
            token_start + token_duration * (consumed_chars as f32 / total_chars as f32)
        };
        current.append_piece(piece, cursor, piece_end, probability);
        cursor = piece_end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timed(token: &str, start: f32, end: f32) -> TimedSentencePieceToken<'_> {
        TimedSentencePieceToken {
            token,
            start_seconds: start,
            end_seconds: end,
            probability: None,
        }
    }

    fn timed_p(token: &str, start: f32, end: f32, probability: f32) -> TimedSentencePieceToken<'_> {
        TimedSentencePieceToken {
            token,
            start_seconds: start,
            end_seconds: end,
            probability: Some(probability),
        }
    }

    #[test]
    fn folds_marked_and_continuation_pieces_into_words() {
        let words = assemble_sentencepiece_word_timestamps([
            timed("\u{2581}hello", 0.0, 1.0),
            timed("\u{2581}wor", 1.0, 1.5),
            timed("ld", 1.5, 2.0),
        ]);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!((words[0].start, words[0].end), (0.0, 1.0));
        assert_eq!(words[0].confidence, None);
        assert_eq!(words[1].word, "world");
        assert_eq!((words[1].start, words[1].end), (1.0, 2.0));
    }

    #[test]
    fn splits_multi_marker_token_with_char_proportional_times() {
        let words =
            assemble_sentencepiece_word_timestamps([timed("\u{2581}ab\u{2581}cd", 0.0, 1.0)]);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "ab");
        assert!((words[0].end - 0.5).abs() < 1e-6);
        assert_eq!(words[1].word, "cd");
        assert!((words[1].start - 0.5).abs() < 1e-6);
        assert!((words[1].end - 1.0).abs() < 1e-6);
    }

    #[test]
    fn instantaneous_spans_yield_monotonic_words() {
        // RNN-T emissions can share a frame: spans may collapse to points.
        let words = assemble_sentencepiece_word_timestamps([
            timed("\u{2581}你好", 0.4, 0.4),
            timed("\u{2581}world", 0.4, 0.44),
        ]);
        assert_eq!(words.len(), 2);
        assert!(words[0].start <= words[1].start);
        assert!(words.iter().all(|word| word.end >= word.start));
    }

    #[test]
    fn word_confidence_is_the_mean_of_its_piece_probabilities() {
        let words = assemble_sentencepiece_word_timestamps([
            timed_p("\u{2581}wor", 0.0, 0.5, 0.9),
            timed_p("ld", 0.5, 1.0, 0.5),
            timed_p("\u{2581}ok", 1.0, 1.5, 1.0),
        ]);
        assert_eq!(words.len(), 2);
        assert!((words[0].confidence.unwrap() - 0.7).abs() < 1e-6);
        assert!((words[1].confidence.unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn missing_piece_probabilities_yield_no_confidence() {
        // Mixed availability still averages only the observed pieces; fully
        // unobserved words must report None, never an invented value.
        let words = assemble_sentencepiece_word_timestamps([
            timed("\u{2581}plain", 0.0, 0.5),
            timed_p("\u{2581}sco", 0.5, 0.8, 0.6),
            timed("red", 0.8, 1.0),
        ]);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].confidence, None);
        assert!((words[1].confidence.unwrap() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn frame_mapping_is_proportional_and_clamped() {
        assert_eq!(frame_to_seconds(0, 100, 10.0), 0.0);
        assert_eq!(frame_to_seconds(50, 100, 10.0), 5.0);
        assert_eq!(frame_to_seconds(150, 100, 10.0), 10.0);
        assert_eq!(frame_to_seconds(1, 100, -3.0), 0.0);
    }
}
