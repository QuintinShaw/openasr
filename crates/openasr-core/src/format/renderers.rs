use crate::api::backend::Transcription;

use super::{
    ResponseFormat,
    json::{JsonTranscription, VerboseJsonTranscription},
};

pub(super) fn render(
    transcription: &Transcription,
    format: ResponseFormat,
) -> Result<String, serde_json::Error> {
    match format {
        ResponseFormat::Text => Ok(render_text_body(transcription.text.as_str())),
        ResponseFormat::Json => {
            serde_json::to_string_pretty(&JsonTranscription::from(transcription))
        }
        ResponseFormat::VerboseJson => {
            serde_json::to_string_pretty(&VerboseJsonTranscription::from(transcription))
        }
        ResponseFormat::Srt => Ok(render_timed_cues(transcription, TimedCueFormat::srt())),
        ResponseFormat::Vtt => Ok(render_vtt(transcription)),
        ResponseFormat::Markdown => Ok(render_markdown(transcription)),
    }
}

#[derive(Clone, Copy)]
struct TimedCueFormat {
    separator: &'static str,
    include_index: bool,
    time_format: fn(f32) -> String,
    prefix: &'static str,
    suffix: &'static str,
}

impl TimedCueFormat {
    const fn srt() -> Self {
        Self {
            separator: "\n",
            include_index: true,
            time_format: format_srt_time,
            prefix: "",
            suffix: "",
        }
    }

    const fn vtt() -> Self {
        Self {
            separator: "\n\n",
            include_index: false,
            time_format: format_vtt_time,
            prefix: "WEBVTT\n\n",
            suffix: "\n",
        }
    }
}

fn render_timed_cues(transcription: &Transcription, spec: TimedCueFormat) -> String {
    let cues = render_segments(transcription, spec.separator, |index, start, end, text| {
        render_timed_cue_row(index, start, end, text, spec)
    });
    format!("{}{cues}{}", spec.prefix, spec.suffix)
}

fn render_vtt(transcription: &Transcription) -> String {
    if transcription
        .segments
        .iter()
        .any(|segment| !segment.words.is_empty())
    {
        return render_word_timed_vtt(transcription);
    }
    render_timed_cues(transcription, TimedCueFormat::vtt())
}

fn render_word_timed_vtt(transcription: &Transcription) -> String {
    let cues = transcription
        .segments
        .iter()
        .flat_map(|segment| {
            segment.words.iter().map(|word| {
                render_timed_cue_row(
                    0,
                    word.start,
                    word.end,
                    render_segment_text(&word.word, segment.speaker.as_deref()),
                    TimedCueFormat::vtt(),
                )
            })
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    format!("WEBVTT\n\n{cues}\n")
}

fn render_markdown(transcription: &Transcription) -> String {
    let body = if transcription
        .segments
        .iter()
        .any(|segment| segment.speaker.is_some())
    {
        render_segments_body(transcription, "\n\n")
    } else {
        transcription.text.clone()
    };
    format!("# Transcript\n\n{body}\n")
}

fn render_text_body(text: &str) -> String {
    format!("{text}\n")
}

fn render_timed_cue_row(
    index: usize,
    start: f32,
    end: f32,
    text: String,
    spec: TimedCueFormat,
) -> String {
    let timing = format!(
        "{} --> {}",
        (spec.time_format)(start),
        (spec.time_format)(end)
    );
    if spec.include_index {
        format!("{index}\n{timing}\n{text}\n")
    } else {
        format!("{timing}\n{text}")
    }
}

fn render_segments(
    transcription: &Transcription,
    separator: &str,
    mut render: impl FnMut(usize, f32, f32, String) -> String,
) -> String {
    transcription
        .segments
        .iter()
        .enumerate()
        .map(|(index, segment)| {
            let text = render_segment_text(&segment.text, segment.speaker.as_deref());
            render(index + 1, segment.start, segment.end, text)
        })
        .collect::<Vec<_>>()
        .join(separator)
}

fn render_segments_body(transcription: &Transcription, separator: &str) -> String {
    transcription
        .segments
        .iter()
        .map(|segment| render_segment_text(&segment.text, segment.speaker.as_deref()))
        .collect::<Vec<_>>()
        .join(separator)
}

fn render_segment_text(text: &str, speaker: Option<&str>) -> String {
    match speaker {
        Some(speaker) if !speaker.trim().is_empty() => format!("{speaker}: {text}"),
        _ => text.to_string(),
    }
}

fn format_srt_time(seconds: f32) -> String {
    format_timestamp(seconds, ',')
}

fn format_vtt_time(seconds: f32) -> String {
    format_timestamp(seconds, '.')
}

fn format_timestamp(seconds: f32, separator: char) -> String {
    let millis = (seconds * 1000.0).round() as u64;
    let hours = millis / 3_600_000;
    let minutes = (millis % 3_600_000) / 60_000;
    let seconds = (millis % 60_000) / 1000;
    let millis = millis % 1000;

    format!("{hours:02}:{minutes:02}:{seconds:02}{separator}{millis:03}")
}
