use super::{RealtimeHistoryEntry, RealtimePostProcessor, history_text::normalize_whitespace};

pub(super) fn sorted_entries(entries: &[RealtimeHistoryEntry]) -> Vec<RealtimeHistoryEntry> {
    let mut entries = entries.to_vec();
    entries.sort_by(|left, right| {
        left.start_ms
            .cmp(&right.start_ms)
            .then_with(|| left.end_ms.cmp(&right.end_ms))
            .then_with(|| {
                left.source_seq
                    .unwrap_or(u64::MAX)
                    .cmp(&right.source_seq.unwrap_or(u64::MAX))
            })
            .then_with(|| left.utterance_id.cmp(&right.utterance_id))
            .then_with(|| left.segment_id.cmp(&right.segment_id))
    });
    entries
}

pub(super) fn post_processed_entries(
    entries: &[RealtimeHistoryEntry],
    options: &RealtimePostProcessor,
) -> Vec<RealtimeHistoryEntry> {
    sorted_entries(entries)
        .into_iter()
        .filter_map(|entry| normalize_entry(entry, options))
        .collect()
}

fn normalize_entry(
    mut entry: RealtimeHistoryEntry,
    options: &RealtimePostProcessor,
) -> Option<RealtimeHistoryEntry> {
    let normalized = normalize_whitespace(&entry.text, options);
    if normalized.is_empty() {
        return None;
    }
    entry.text = normalized;
    Some(entry)
}
