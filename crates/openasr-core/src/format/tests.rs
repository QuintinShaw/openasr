use super::*;
use crate::api::backend::{
    DecodeTruncation, DecodeTruncationReason, Segment, Transcription,
    TranscriptionLongFormMetadata, TruncatedDecode, WordTimestamp,
};

fn sample() -> Transcription {
    Transcription {
        truncated_decodes: Vec::new(),
        unnamed_speakers: Vec::new(),
        text: "hello world".to_string(),
        segments: vec![Segment {
            start: 0.0,
            end: 2.5,
            text: "hello world".to_string(),
            speaker: None,
            speaker_label: None,
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words: Vec::new(),
        }],
        longform: None,
        language: None,
    }
}

fn speaker_sample() -> Transcription {
    Transcription {
        truncated_decodes: Vec::new(),
        unnamed_speakers: Vec::new(),
        text: "hello world\nnext line".to_string(),
        segments: vec![
            Segment {
                start: 0.0,
                end: 2.5,
                text: "hello world".to_string(),
                speaker: Some("SPEAKER_00".to_string()),
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            },
            Segment {
                start: 2.5,
                end: 4.0,
                text: "next line".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            },
        ],
        longform: None,
        language: None,
    }
}

fn matched_profile_sample() -> Transcription {
    Transcription {
        truncated_decodes: Vec::new(),
        unnamed_speakers: Vec::new(),
        text: "hello world\nnext line".to_string(),
        segments: vec![
            Segment {
                start: 0.0,
                end: 2.5,
                text: "hello world".to_string(),
                speaker: Some("Alice".to_string()),
                speaker_label: Some("SPEAKER_00".to_string()),
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            },
            Segment {
                start: 2.5,
                end: 4.0,
                text: "next line".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            },
        ],
        longform: None,
        language: None,
    }
}

fn word_sample() -> Transcription {
    Transcription {
        truncated_decodes: Vec::new(),
        unnamed_speakers: Vec::new(),
        text: "hello world".to_string(),
        segments: vec![Segment {
            start: 0.0,
            end: 1.0,
            text: "hello world".to_string(),
            speaker: None,
            speaker_label: None,
            speaker_person_id: None,
            speaker_snapshot_label: None,
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
    assert!(parsed["segments"][1].get("speaker").is_none());
    assert!(parsed["segments"][1].get("speaker_label").is_none());
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
        truncated_decodes: Vec::new(),
        unnamed_speakers: Vec::new(),
        text: "hello world".to_string(),
        segments: vec![Segment {
            start: 0.0,
            end: 2.0,
            text: "hello world".to_string(),
            speaker: None,
            speaker_label: None,
            speaker_person_id: None,
            speaker_snapshot_label: None,
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
        truncated_decodes: Vec::new(),
        unnamed_speakers: Vec::new(),
        text: "one two three four".to_string(),
        segments: vec![
            Segment {
                start: 0.0,
                end: 1.0,
                text: "one two".to_string(),
                speaker: Some("SPEAKER_00".to_string()),
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            },
            Segment {
                start: 1.0,
                end: 2.0,
                text: "three".to_string(),
                speaker: Some("SPEAKER_00".to_string()),
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            },
            Segment {
                start: 2.0,
                end: 3.0,
                text: "four".to_string(),
                speaker: Some("SPEAKER_01".to_string()),
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
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

/// The json and verbose_json renderers must both surface a truncated decode.
///
/// This is the last hop of the truncation signal: everything upstream can be
/// correct and the user still gets a silently short transcript if the
/// serializer drops the field. Asserted on the plain `json` format too, not
/// only `verbose_json` -- "this is not all of your audio" is not an opt-in
/// diagnostic.
#[test]
fn json_formats_report_a_truncated_decode() {
    let mut transcription = sample();
    transcription.truncated_decodes = vec![
        TruncatedDecode {
            slice_index: Some(3),
            truncation: DecodeTruncation {
                reason: DecodeTruncationReason::DegenerateRepeatGuard,
                transcript_covers_up_to_seconds: Some(12.5),
            },
        },
        TruncatedDecode {
            slice_index: None,
            truncation: DecodeTruncation {
                reason: DecodeTruncationReason::BudgetExhausted,
                transcript_covers_up_to_seconds: None,
            },
        },
    ];

    for format in [ResponseFormat::Json, ResponseFormat::VerboseJson] {
        let rendered = render_transcription(&transcription, format).expect("render");
        let parsed: serde_json::Value = serde_json::from_str(&rendered).expect("valid json");
        let truncated = parsed
            .get("truncated")
            .and_then(|value| value.as_array())
            .unwrap_or_else(|| panic!("{format} must carry a truncated array"));
        assert_eq!(truncated.len(), 2, "{format}");
        assert_eq!(truncated[0]["slice"], 3, "{format}");
        assert_eq!(
            truncated[0]["reason"], "degenerate-repeat-guard",
            "{format}"
        );
        assert_eq!(truncated[0]["covers_up_to_seconds"], 12.5, "{format}");
        // A single-pass decode has no slice, and a family without intra-decode
        // timestamps has no anchor: both are omitted rather than filled with a
        // placeholder a consumer would read as real.
        assert!(truncated[1].get("slice").is_none(), "{format}");
        assert_eq!(truncated[1]["reason"], "budget-exhausted", "{format}");
        assert!(
            truncated[1].get("covers_up_to_seconds").is_none(),
            "{format}"
        );
    }
}

/// Both json renderers must surface why a speaker is still a number.
///
/// Same last-hop argument as truncation: the identity stage can judge
/// correctly and the user still sees a bare `SPEAKER_01` if the serializer
/// drops the reason -- which is the state that reads as a broken feature. The
/// numbers behind each verdict travel with it so a client can say "2.0s of the
/// 3.0s needed" rather than only "not enough".
#[test]
fn json_formats_report_why_a_speaker_is_unnamed() {
    use crate::diarize::voice_id::{SpeakerNamingRefusal, UnnamedSpeaker};

    let mut transcription = sample();
    transcription.unnamed_speakers = vec![
        UnnamedSpeaker {
            label: "SPEAKER_00".to_string(),
            reason: SpeakerNamingRefusal::NotEnoughSpeech {
                windows: 1,
                required_windows: 5,
                seconds: 2.0,
                required_seconds: 3.0,
                required_continuous_seconds: 7.0,
            },
        },
        UnnamedSpeaker {
            label: "SPEAKER_01".to_string(),
            reason: SpeakerNamingRefusal::NoMatchInLibrary {
                library_empty: false,
                best_score: Some(0.25),
                accept_threshold: Some(0.45),
            },
        },
        UnnamedSpeaker {
            label: "SPEAKER_02".to_string(),
            reason: SpeakerNamingRefusal::EmbedderUnavailable,
        },
    ];

    for format in [ResponseFormat::Json, ResponseFormat::VerboseJson] {
        let rendered = render_transcription(&transcription, format).expect("render");
        let parsed: serde_json::Value = serde_json::from_str(&rendered).expect("valid json");
        let unnamed = parsed
            .get("unnamed_speakers")
            .and_then(|value| value.as_array())
            .unwrap_or_else(|| panic!("{format} must carry an unnamed_speakers array"));
        assert_eq!(unnamed.len(), 3, "{format}");

        assert_eq!(unnamed[0]["label"], "SPEAKER_00", "{format}");
        assert_eq!(unnamed[0]["reason"], "not-enough-speech", "{format}");
        assert_eq!(unnamed[0]["windows"], 1, "{format}");
        assert_eq!(unnamed[0]["required_windows"], 5, "{format}");
        assert_eq!(unnamed[0]["seconds"], 2.0, "{format}");
        assert_eq!(unnamed[0]["required_seconds"], 3.0, "{format}");
        assert_eq!(unnamed[0]["required_continuous_seconds"], 7.0, "{format}");

        assert_eq!(unnamed[1]["reason"], "no-match-in-library", "{format}");
        assert_eq!(unnamed[1]["library_empty"], false, "{format}");
        assert_eq!(unnamed[1]["best_score"], 0.25, "{format}");
        assert_eq!(unnamed[1]["accept_threshold"], 0.45, "{format}");
        // Evidence numbers belong to the evidence reasons; a match refusal
        // must not carry a fabricated window count.
        assert!(unnamed[1].get("windows").is_none(), "{format}");

        assert_eq!(unnamed[2]["reason"], "embedder-unavailable", "{format}");
        assert!(unnamed[2].get("library_empty").is_none(), "{format}");
    }
}

/// A transcript whose speakers all have names (or that has no speakers at all)
/// must serialize byte-identically to before this field existed.
#[test]
fn json_formats_omit_unnamed_speakers_when_there_is_nothing_to_explain() {
    let transcription = sample();
    assert!(transcription.unnamed_speakers.is_empty());
    for format in [ResponseFormat::Json, ResponseFormat::VerboseJson] {
        let rendered = render_transcription(&transcription, format).expect("render");
        assert!(
            !rendered.contains("unnamed_speakers"),
            "{format} must not mention unnamed speakers when every speaker is named: {rendered}"
        );
    }
}

/// A transcript that ended on its own stop token must serialize byte-identically
/// to before this field existed: the healthy path stays untouched, so no
/// existing consumer sees a new key until something actually went wrong.
#[test]
fn json_formats_omit_truncation_on_a_complete_transcript() {
    let transcription = sample();
    assert!(!transcription.is_truncated());
    for format in [ResponseFormat::Json, ResponseFormat::VerboseJson] {
        let rendered = render_transcription(&transcription, format).expect("render");
        assert!(
            !rendered.contains("truncated"),
            "{format} must not mention truncation on a complete transcript: {rendered}"
        );
    }
}
