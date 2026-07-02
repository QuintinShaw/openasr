use serde::Serialize;

use super::{
    RealtimeExportFormat, RealtimeHistoryEntry, RealtimeHistoryExportError,
    RealtimeHistoryRevision, RealtimePostProcessOutput, RealtimePostProcessor, history_sort,
    history_subtitle, history_text,
};

#[derive(Debug, Clone, Serialize)]
pub(super) struct RealtimeHistoryExportJson {
    pub(super) text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) title: Option<String>,
    pub(super) entries: Vec<RealtimeHistoryEntry>,
    pub(super) revisions: Vec<RealtimeHistoryRevision>,
}

pub(super) fn export_rendered(
    format: RealtimeExportFormat,
    entries: &[RealtimeHistoryEntry],
    revisions: &[RealtimeHistoryRevision],
    processed: RealtimePostProcessOutput,
    options: &RealtimePostProcessor,
) -> Result<String, RealtimeHistoryExportError> {
    match format {
        RealtimeExportFormat::Text => Ok(render_trailing_newline_or_empty(&processed.joined_text)),
        RealtimeExportFormat::Json => export_json(entries, revisions, processed, options),
        RealtimeExportFormat::Markdown => export_markdown(processed, options),
        RealtimeExportFormat::Srt => export_subtitles(entries, options, SubtitleKind::Srt),
        RealtimeExportFormat::Vtt => export_subtitles(entries, options, SubtitleKind::Vtt),
    }
}

fn export_json(
    entries: &[RealtimeHistoryEntry],
    revisions: &[RealtimeHistoryRevision],
    processed: RealtimePostProcessOutput,
    options: &RealtimePostProcessor,
) -> Result<String, RealtimeHistoryExportError> {
    let payload = RealtimeHistoryExportJson {
        text: processed.joined_text,
        title: processed.title,
        entries: history_sort::post_processed_entries(entries, options),
        revisions: revisions.to_vec(),
    };
    serde_json::to_string_pretty(&payload).map_err(Into::into)
}

fn export_markdown(
    processed: RealtimePostProcessOutput,
    options: &RealtimePostProcessor,
) -> Result<String, RealtimeHistoryExportError> {
    let heading = processed.title.unwrap_or_else(|| "Transcript".to_string());
    let body = if options.join_segments {
        processed.joined_text
    } else {
        processed.lines.join("\n\n")
    };
    Ok(render_markdown_doc(&heading, &body))
}

enum SubtitleKind {
    Srt,
    Vtt,
}

impl SubtitleKind {
    fn layout(self) -> SubtitleLayout {
        match self {
            Self::Srt => SubtitleLayout {
                header: None,
                include_index: true,
                time_format: history_text::format_srt_ms,
            },
            Self::Vtt => SubtitleLayout {
                header: Some("WEBVTT\n"),
                include_index: false,
                time_format: history_text::format_vtt_ms,
            },
        }
    }
}

struct SubtitleLayout {
    header: Option<&'static str>,
    include_index: bool,
    time_format: fn(u64) -> String,
}

fn export_subtitles(
    entries: &[RealtimeHistoryEntry],
    options: &RealtimePostProcessor,
    kind: SubtitleKind,
) -> Result<String, RealtimeHistoryExportError> {
    let layout = kind.layout();
    let rows = history_subtitle::export_subtitle_rows(
        entries,
        options,
        layout.time_format,
        layout.include_index,
    )?;
    if rows.is_empty() {
        return Ok(layout.header.unwrap_or_default().to_string());
    }

    Ok(render_subtitle_doc(layout.header, rows.join("\n\n")))
}

fn render_trailing_newline_or_empty(text: &str) -> String {
    if text.is_empty() {
        String::new()
    } else {
        format!("{text}\n")
    }
}

fn render_markdown_doc(heading: &str, body: &str) -> String {
    if body.is_empty() {
        format!("# {heading}\n")
    } else {
        format!("# {heading}\n\n{body}\n")
    }
}

fn render_subtitle_doc(header: Option<&'static str>, body: String) -> String {
    match header {
        Some(header) => format!("{header}\n{body}\n"),
        None => format!("{body}\n"),
    }
}
