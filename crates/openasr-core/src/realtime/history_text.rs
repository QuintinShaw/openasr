use super::RealtimePostProcessor;

pub(super) fn normalize_whitespace(input: &str, options: &RealtimePostProcessor) -> String {
    let value = if options.trim_whitespace {
        input.trim()
    } else {
        input
    };
    if !options.collapse_internal_whitespace {
        return value.to_string();
    }
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(super) fn suggest_title_from_text(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let compact = trimmed.trim_end_matches(|ch: char| ['.', ',', ';', ':', '!', '?'].contains(&ch));
    let words = compact.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() {
        return None;
    }
    let take_count = words.len().min(8);
    let mut title = words[..take_count].join(" ");
    if words.len() > take_count {
        title.push_str("...");
    }
    Some(title)
}

pub(super) fn format_srt_ms(ms: u64) -> String {
    format_timestamp_ms(ms, ',')
}

pub(super) fn format_vtt_ms(ms: u64) -> String {
    format_timestamp_ms(ms, '.')
}

fn format_timestamp_ms(ms: u64, separator: char) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1000;
    let millis = ms % 1000;
    format!("{hours:02}:{minutes:02}:{seconds:02}{separator}{millis:03}")
}
