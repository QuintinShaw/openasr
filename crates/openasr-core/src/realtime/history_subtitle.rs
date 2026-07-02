use super::{
    RealtimeHistoryEntry, RealtimeHistoryExportError, RealtimePostProcessor,
    history_sort::sorted_entries, history_text::normalize_whitespace,
};

pub(super) fn export_subtitle_rows(
    entries: &[RealtimeHistoryEntry],
    options: &RealtimePostProcessor,
    format_ms: fn(u64) -> String,
    include_index: bool,
) -> Result<Vec<String>, RealtimeHistoryExportError> {
    let mut rows = Vec::new();
    for (index, entry) in sorted_entries(entries).iter().enumerate() {
        if entry.end_ms <= entry.start_ms {
            return Err(RealtimeHistoryExportError::InvalidSubtitleTiming {
                index: index + 1,
                start_ms: entry.start_ms,
                end_ms: entry.end_ms,
            });
        }
        let text = normalize_whitespace(&entry.text, options);
        if text.is_empty() {
            continue;
        }
        let cue = format!(
            "{} --> {}\n{}",
            format_ms(entry.start_ms),
            format_ms(entry.end_ms),
            text
        );
        if include_index {
            rows.push(format!("{}\n{cue}", rows.len() + 1));
        } else {
            rows.push(cue);
        }
    }
    Ok(rows)
}
