//! Compatibility shim: subtitle cue packing lives in [`crate::subtitle::cues`].
//!
//! The historical module path is kept so existing `cargo test --lib
//! cue_segmentation` filters and `super::cue_segmentation` call sites keep
//! resolving. New code should import from `crate::subtitle`.

#[cfg(test)]
mod tests {
    // Re-export the subtitle cue tests under this module path so the historical
    // filter `cue_segmentation` still exercises the packer.
    use crate::api::backend::{Segment, Transcription, WordTimestamp};
    use crate::subtitle::cues::{resegment_transcription_cues, segment_into_cues};

    fn word(text: &str, start: f32, end: f32) -> WordTimestamp {
        WordTimestamp {
            word: text.to_string(),
            start,
            end,
            confidence: None,
        }
    }

    fn segment(text: &str, words: Vec<WordTimestamp>) -> Segment {
        let start = words.first().map_or(0.0, |w| w.start);
        let end = words.last().map_or(0.0, |w| w.end);
        Segment {
            start,
            end,
            text: text.to_string(),
            speaker: None,
            speaker_label: None,
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words,
        }
    }

    fn transcription(segments: Vec<Segment>) -> Transcription {
        Transcription {
            text: segments
                .iter()
                .map(|s| s.text.trim())
                .collect::<Vec<_>>()
                .join(" "),
            segments,
            ..Default::default()
        }
    }

    const MAX_CUE_SECONDS: f32 = 8.0;
    const LATIN_MAX_CHARS: usize = 84;
    const ORPHAN_MAX_WORDS: usize = 2;

    #[test]
    fn splits_latin_monolithic_segment_at_sentence_punctuation() {
        let text = "And so my fellow americans ask not what your country can do for you. Ask what you can do for your country";
        let words = vec![
            word("And", 0.96, 1.00),
            word("so", 1.43, 1.47),
            word("my", 1.55, 1.59),
            word("fellow", 1.71, 1.91),
            word("americans", 2.19, 3.19),
            word("ask", 4.11, 4.14),
            word("not", 4.90, 4.94),
            word("what", 5.74, 5.78),
            word("your", 6.22, 6.26),
            word("country", 6.50, 6.54),
            word("can", 6.86, 6.89),
            word("do", 7.13, 7.17),
            word("for", 7.49, 7.53),
            word("you.", 7.93, 7.97),
            word("Ask", 8.77, 9.01),
            word("what", 9.21, 9.25),
            word("you", 9.41, 9.45),
            word("can", 9.61, 9.64),
            word("do", 9.84, 9.88),
            word("for", 10.08, 10.12),
            word("your", 10.28, 10.32),
            word("country", 10.80, 10.84),
        ];
        let cues = segment_into_cues(segment(text, words));
        assert!(cues.len() >= 2, "cues: {cues:?}");
        for cue in &cues {
            assert!(
                cue.end - cue.start <= MAX_CUE_SECONDS + 1e-3,
                "cue too long: {cue:?}"
            );
            assert!(cue.text.chars().count() <= LATIN_MAX_CHARS);
        }
        let joined: Vec<&str> = cues
            .iter()
            .flat_map(|c| c.words.iter().map(|w| w.word.as_str()))
            .collect();
        assert_eq!(joined.len(), 22);
        assert_eq!(joined[0], "And");
        assert_eq!(joined[21], "country");
        assert!(cues.iter().any(|c| c.text.ends_with("you.")));
    }

    #[test]
    fn keeps_short_single_sentence_whole() {
        let text = "hello world this is short";
        let words = vec![
            word("hello", 0.0, 0.3),
            word("world", 0.4, 0.7),
            word("this", 0.8, 1.0),
            word("is", 1.1, 1.2),
            word("short", 1.3, 1.6),
        ];
        let cues = segment_into_cues(segment(text, words));
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "hello world this is short");
    }

    #[test]
    fn splits_cjk_segment_at_ideographic_period() {
        let text = "\u{4f60}\u{597d}\u{4e16}\u{754c}\u{3002}\u{4eca}\u{5929}\u{5929}\u{6c14}\u{5f88}\u{597d}";
        let words = vec![
            word("\u{4f60}", 0.0, 0.3),
            word("\u{597d}", 0.3, 0.6),
            word("\u{4e16}", 0.6, 0.9),
            word("\u{754c}\u{3002}", 0.9, 1.2),
            word("\u{4eca}", 1.3, 1.6),
            word("\u{5929}", 1.6, 1.9),
            word("\u{5929}", 1.9, 2.2),
            word("\u{6c14}", 2.2, 2.5),
            word("\u{5f88}", 2.5, 2.8),
            word("\u{597d}", 2.8, 3.1),
        ];
        let cues = segment_into_cues(segment(text, words));
        assert_eq!(cues.len(), 2, "cues: {cues:?}");
        assert_eq!(cues[0].text, "\u{4f60}\u{597d}\u{4e16}\u{754c}\u{3002}");
        assert_eq!(
            cues[1].text,
            "\u{4eca}\u{5929}\u{5929}\u{6c14}\u{5f88}\u{597d}"
        );
    }

    #[test]
    fn preserves_forced_aligner_words_across_latin_punctuation() {
        let text = "hello, world; this is a deliberately long sentence. another clause follows";
        let tokens = [
            "hello",
            "world",
            "this",
            "is",
            "a",
            "deliberately",
            "long",
            "sentence",
            "another",
            "clause",
            "follows",
        ];
        let words = tokens
            .iter()
            .enumerate()
            .map(|(index, token)| word(token, index as f32, index as f32 + 0.5))
            .collect::<Vec<_>>();

        let cues = segment_into_cues(segment(text, words));

        assert!(
            cues.len() >= 2,
            "sentence punctuation should remain visible: {cues:?}"
        );
        let preserved = cues
            .iter()
            .flat_map(|cue| cue.words.iter().map(|word| word.word.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(preserved, tokens);
        assert!(cues.iter().any(|cue| cue.text.ends_with("sentence.")));
    }

    #[test]
    fn preserves_forced_aligner_words_across_cjk_punctuation() {
        let text = "你好世界。今天天气很好";
        let tokens = ["你", "好", "世", "界", "今", "天", "天", "气", "很", "好"];
        let words = tokens
            .iter()
            .enumerate()
            .map(|(index, token)| word(token, index as f32 * 0.3, index as f32 * 0.3 + 0.2))
            .collect::<Vec<_>>();

        let cues = segment_into_cues(segment(text, words));

        assert_eq!(
            cues.len(),
            2,
            "ideographic period should remain visible: {cues:?}"
        );
        let preserved = cues
            .iter()
            .flat_map(|cue| cue.words.iter().map(|word| word.word.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(preserved, tokens);
        assert_eq!(cues[0].text, "你好世界。");
        assert_eq!(cues[1].text, "今天天气很好");
    }

    #[test]
    fn merges_zero_duration_forced_aligner_tail_across_sentence_boundary() {
        let text = "嗯。嗯。";
        let words = vec![word("嗯", 55.60, 55.92), word("嗯", 55.92, 55.92)];

        let cues = segment_into_cues(segment(text, words));

        assert_eq!(cues.len(), 1, "zero-duration tail must not stand alone");
        assert_eq!(cues[0].text, text);
        assert_eq!(cues[0].words.len(), 2);
        assert_eq!(cues[0].start, 55.60);
        assert_eq!(cues[0].end, 55.92);
    }

    #[test]
    fn splits_long_unpunctuated_segment_by_duration() {
        let words: Vec<WordTimestamp> = (0..10)
            .map(|i| {
                let start = i as f32 * 1.0;
                word("word", start, start + 0.5)
            })
            .collect();
        let text = "word word word word word word word word word word";
        let cues = segment_into_cues(segment(text, words));
        assert!(cues.len() >= 2, "a 9.5s run must split: {cues:?}");
        for cue in &cues {
            assert!(cue.end - cue.start <= MAX_CUE_SECONDS + 1e-3);
        }
    }

    #[test]
    fn never_crosses_speaker_turns() {
        let mut a = segment(
            "alpha bravo charlie. delta echo foxtrot.",
            vec![
                word("alpha", 0.0, 0.3),
                word("bravo", 0.4, 0.7),
                word("charlie.", 0.8, 1.1),
                word("delta", 1.2, 1.5),
                word("echo", 1.6, 1.9),
                word("foxtrot.", 2.0, 2.3),
            ],
        );
        a.speaker = Some("SPEAKER_00".to_string());
        let mut b = segment(
            "golf hotel.",
            vec![word("golf", 2.5, 2.8), word("hotel.", 2.9, 3.2)],
        );
        b.speaker = Some("SPEAKER_01".to_string());
        let out = resegment_transcription_cues(transcription(vec![a, b]));
        for cue in &out.segments {
            let speaker = cue.speaker.as_deref().unwrap();
            if cue.text.contains("golf") || cue.text.contains("hotel") {
                assert_eq!(speaker, "SPEAKER_01");
            } else {
                assert_eq!(speaker, "SPEAKER_00");
            }
        }
        assert!(
            out.segments
                .iter()
                .any(|c| c.speaker.as_deref() == Some("SPEAKER_01"))
        );
    }

    #[test]
    fn merges_trailing_orphan_into_previous_cue() {
        // Duration forces a cut that leaves a dangling two-word tail of the
        // same sentence. Inter-word gaps stay below MIN_PAUSE_GAP_S so the
        // orphan merge is allowed (pause-split cues must not be re-glued).
        // (Keep fixture in lockstep with subtitle::cues::tests.)
        let words = vec![
            word("alpha", 0.0, 0.5),
            word("bravo", 0.6, 1.1),
            word("charlie", 1.2, 1.7),
            word("delta", 1.8, 2.3),
            word("echo", 2.4, 2.9),
            word("foxtrot", 3.0, 3.5),
            word("golf", 3.6, 4.1),
            word("hotel", 4.2, 4.7),
            word("india", 4.8, 5.3),
            word("juliet", 5.4, 5.9),
            // 0.30s gap (< MIN_PAUSE_GAP_S) is the widest, so choose_cut lands
            // here; the remaining two words are an orphan tail.
            word("kilo", 6.2, 6.5),
            word("lima", 6.55, 6.8),
        ];
        let text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima";
        let cues = segment_into_cues(segment(text, words));
        assert!(
            cues.last().map(|c| c.words.len()).unwrap_or(0) > ORPHAN_MAX_WORDS || cues.len() == 1,
            "orphan tail was not merged: {cues:?}"
        );
    }

    #[test]
    fn preserves_transcription_text_verbatim() {
        let text = "one two three. four five six. seven eight nine.";
        let words = vec![
            word("one", 0.0, 0.3),
            word("two", 0.4, 0.7),
            word("three.", 0.8, 1.1),
            word("four", 1.2, 1.5),
            word("five", 1.6, 1.9),
            word("six.", 2.0, 2.3),
            word("seven", 2.4, 2.7),
            word("eight", 2.8, 3.1),
            word("nine.", 3.2, 3.5),
        ];
        let original = transcription(vec![segment(text, words)]);
        let text_before = original.text.clone();
        let out = resegment_transcription_cues(original);
        assert_eq!(out.text, text_before, "joined text must be untouched");
        assert!(out.segments.len() >= 3);
    }

    #[test]
    fn mixed_cjk_latin_splits_on_sentence_boundary() {
        let text = "Hello 世界。Next line starts here";
        let words = vec![
            word("Hello", 0.0, 0.4),
            word("世", 0.5, 0.7),
            word("界。", 0.7, 1.0),
            word("Next", 1.2, 1.5),
            word("line", 1.6, 1.9),
            word("starts", 2.0, 2.4),
            word("here", 2.5, 2.9),
        ];
        let cues = segment_into_cues(segment(text, words));
        assert!(cues.len() >= 2, "mixed CJK/Latin must split: {cues:?}");
        assert!(cues.iter().any(|c| c.text.contains("世界")));
        assert!(cues.iter().any(|c| c.text.contains("Next")));
    }
}
