use super::*;
use crate::api::backend::{Segment, Transcription, TranscriptionLongFormMetadata, WordTimestamp};

fn sample() -> Transcription {
    Transcription {
        text: "hello world".to_string(),
        segments: vec![Segment {
            start: 0.0,
            end: 2.5,
            text: "hello world".to_string(),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words: Vec::new(),
        }],
        longform: None,
        language: None,
    }
}

fn speaker_sample() -> Transcription {
    Transcription {
        text: "hello world\nnext line".to_string(),
        segments: vec![
            Segment {
                start: 0.0,
                end: 2.5,
                text: "hello world".to_string(),
                speaker: Some("SPEAKER_00".to_string()),
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Segment {
                start: 2.5,
                end: 4.0,
                text: "next line".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
        ],
        longform: None,
        language: None,
    }
}

fn matched_profile_sample() -> Transcription {
    Transcription {
        text: "hello world\nnext line".to_string(),
        segments: vec![
            Segment {
                start: 0.0,
                end: 2.5,
                text: "hello world".to_string(),
                speaker: Some("Alice".to_string()),
                speaker_label: Some("SPEAKER_00".to_string()),
                speaker_profile_id: Some("vp_aaaaaaaaaaaaaaaa".to_string()),
                words: Vec::new(),
            },
            Segment {
                start: 2.5,
                end: 4.0,
                text: "next line".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
        ],
        longform: None,
        language: None,
    }
}

fn word_sample() -> Transcription {
    Transcription {
        text: "hello world".to_string(),
        segments: vec![Segment {
            start: 0.0,
            end: 1.0,
            text: "hello world".to_string(),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words: vec![
                WordTimestamp {
                    word: "hello".to_string(),
                    start: 0.0,
                    end: 0.4,
                    confidence: None,
                },
                WordTimestamp {
                    word: "world".to_string(),
                    start: 0.4,
                    end: 1.0,
                    confidence: None,
                },
            ],
        }],
        longform: None,
        language: None,
    }
}

#[test]
fn parses_supported_formats() {
    assert_eq!("text".parse(), Ok(ResponseFormat::Text));
    assert_eq!("json".parse(), Ok(ResponseFormat::Json));
    assert_eq!("srt".parse(), Ok(ResponseFormat::Srt));
    assert_eq!("vtt".parse(), Ok(ResponseFormat::Vtt));
    assert_eq!("verbose_json".parse(), Ok(ResponseFormat::VerboseJson));
    assert_eq!("markdown".parse(), Ok(ResponseFormat::Markdown));
}

#[test]
fn rejects_unknown_format_with_friendly_message() {
    let error = "xml".parse::<ResponseFormat>().unwrap_err();
    assert!(error.contains("Unsupported response format 'xml'"));
    assert!(error.contains("verbose_json"));
}

#[test]
fn displays_verbose_json() {
    assert_eq!(ResponseFormat::VerboseJson.to_string(), "verbose_json");
}

#[test]
fn renders_text() {
    assert_eq!(
        render_transcription(&sample(), ResponseFormat::Text).unwrap(),
        "hello world\n"
    );
}

#[test]
fn renders_json() {
    let rendered = render_transcription(&sample(), ResponseFormat::Json).unwrap();
    assert!(rendered.contains("\"text\": \"hello world\""));
    assert!(rendered.contains("\"start\": 0.0"));
    assert!(!rendered.contains("\"speaker\""));
    // The plain `json` format stays free of the verbose_json-only fields.
    assert!(!rendered.contains("\"id\""));
    assert!(!rendered.contains("\"duration\""));
}

#[test]
fn renders_json_speaker_identity_only_when_present() {
    let rendered = render_transcription(&matched_profile_sample(), ResponseFormat::Json).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();

    assert_eq!(parsed["segments"][0]["speaker"], "Alice");
    assert_eq!(parsed["segments"][0]["speaker_label"], "SPEAKER_00");
    assert_eq!(
        parsed["segments"][0]["speaker_profile_id"],
        "vp_aaaaaaaaaaaaaaaa"
    );
    assert!(parsed["segments"][1].get("speaker").is_none());
    assert!(parsed["segments"][1].get("speaker_label").is_none());
    assert!(parsed["segments"][1].get("speaker_profile_id").is_none());
}

#[test]
fn renders_verbose_json() {
    let rendered = render_transcription(&sample(), ResponseFormat::VerboseJson).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();
    assert_eq!(parsed["text"], "hello world");
    assert_eq!(parsed["segments"][0]["start"], 0.0);
    // OpenAI verbose_json compatibility surface: `duration` (last segment end)
    // and zero-based segment `id`s; `words` only appears with word timestamps.
    assert_eq!(parsed["duration"], 2.5);
    assert_eq!(parsed["segments"][0]["id"], 0);
    assert!(parsed.get("words").is_none());
    assert!(parsed.get("language").is_none());
}

#[test]
fn renders_verbose_json_with_top_level_words() {
    let rendered = render_transcription(&word_sample(), ResponseFormat::VerboseJson).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();
    assert_eq!(parsed["words"][0]["word"], "hello");
    assert_eq!(parsed["words"][1]["word"], "world");
    assert_eq!(parsed["words"][1]["end"], 1.0);
    // The per-segment words stay for existing clients.
    assert_eq!(parsed["segments"][0]["words"][0]["word"], "hello");
}

#[test]
fn renders_json_with_word_timestamps_when_present() {
    let rendered = render_transcription(&word_sample(), ResponseFormat::Json).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();

    assert_eq!(parsed["segments"][0]["words"][0]["word"], "hello");
    assert_eq!(parsed["segments"][0]["words"][0]["start"], 0.0);
    assert_eq!(parsed["segments"][0]["words"][1]["end"], 1.0);
}

#[test]
fn renders_verbose_json_with_longform_metadata() {
    let transcription = Transcription {
        text: "hello world".to_string(),
        segments: vec![Segment {
            start: 0.0,
            end: 2.0,
            text: "hello world".to_string(),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words: Vec::new(),
        }],
        longform: Some(TranscriptionLongFormMetadata {
            chunk_count: 4,
            skipped_silent_chunks: 1,
            duplicate_merge_count: 2,
            provenance: vec!["core.longform.plan:auto".to_string()],
        }),
        language: None,
    };
    let rendered = render_transcription(&transcription, ResponseFormat::VerboseJson).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();
    assert_eq!(parsed["longform"]["chunk_count"], 4);
    assert_eq!(parsed["longform"]["skipped_silent_chunks"], 1);
    assert_eq!(parsed["longform"]["duplicate_merge_count"], 2);
}

#[test]
fn renders_srt() {
    assert_eq!(
        render_transcription(&sample(), ResponseFormat::Srt).unwrap(),
        "1\n00:00:00,000 --> 00:00:02,500\nhello world\n"
    );
}

#[test]
fn renders_srt_speaker_prefix_only_when_present() {
    assert_eq!(
        render_transcription(&speaker_sample(), ResponseFormat::Srt).unwrap(),
        "1\n00:00:00,000 --> 00:00:02,500\nSPEAKER_00: hello world\n\n2\n00:00:02,500 --> 00:00:04,000\nnext line\n"
    );
}

#[test]
fn renders_vtt() {
    assert_eq!(
        render_transcription(&sample(), ResponseFormat::Vtt).unwrap(),
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nhello world\n"
    );
}

#[test]
fn renders_vtt_speaker_prefix_only_when_present() {
    assert_eq!(
        render_transcription(&speaker_sample(), ResponseFormat::Vtt).unwrap(),
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nSPEAKER_00: hello world\n\n00:00:02.500 --> 00:00:04.000\nnext line\n"
    );
}

#[test]
fn renders_word_level_vtt_when_word_timestamps_are_present() {
    assert_eq!(
        render_transcription(&word_sample(), ResponseFormat::Vtt).unwrap(),
        "WEBVTT\n\n00:00:00.000 --> 00:00:00.400\nhello\n\n00:00:00.400 --> 00:00:01.000\nworld\n"
    );
}

#[test]
fn renders_markdown() {
    assert_eq!(
        render_transcription(&sample(), ResponseFormat::Markdown).unwrap(),
        "# Transcript\n\nhello world\n"
    );
}

#[test]
fn renders_markdown_speaker_prefix_only_when_present() {
    assert_eq!(
        render_transcription(&speaker_sample(), ResponseFormat::Markdown).unwrap(),
        "# Transcript\n\nSPEAKER_00: hello world\n\nnext line\n"
    );
}

#[test]
fn renders_markdown_coalesces_consecutive_same_speaker_cues() {
    // The cue re-segmentation pass emits many short cues per speaker turn.
    // Markdown groups consecutive same-speaker cues into one paragraph while a
    // speaker change still starts a new one.
    let transcription = Transcription {
        text: "one two three four".to_string(),
        segments: vec![
            Segment {
                start: 0.0,
                end: 1.0,
                text: "one two".to_string(),
                speaker: Some("SPEAKER_00".to_string()),
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Segment {
                start: 1.0,
                end: 2.0,
                text: "three".to_string(),
                speaker: Some("SPEAKER_00".to_string()),
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Segment {
                start: 2.0,
                end: 3.0,
                text: "four".to_string(),
                speaker: Some("SPEAKER_01".to_string()),
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
        ],
        longform: None,
        language: None,
    };
    assert_eq!(
        render_transcription(&transcription, ResponseFormat::Markdown).unwrap(),
        "# Transcript\n\nSPEAKER_00: one two three\n\nSPEAKER_01: four\n"
    );
}
