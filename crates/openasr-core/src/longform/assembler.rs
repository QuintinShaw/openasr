use crate::{Segment, Transcription, WordTimestamp};

use super::slicing::{AudioSlice, LongFormBenchmarkMetadata};
use super::timeline::TimelineMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentTimeDomain {
    RelativeToSliceContent,
    AbsoluteOriginal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SliceTranscript {
    pub slice: AudioSlice,
    pub text: String,
    pub segments: Vec<Segment>,
    pub time_domain: SegmentTimeDomain,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentMergePolicy {
    pub max_gap_seconds: f32,
    pub redundant_overlap_ratio: f32,
    pub redundant_min_words: usize,
}

impl Default for SegmentMergePolicy {
    fn default() -> Self {
        Self {
            max_gap_seconds: 1.2,
            redundant_overlap_ratio: 0.8,
            redundant_min_words: 4,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LongFormAssembleStats {
    pub skipped_silent_chunks: usize,
    pub duplicate_merge_count: usize,
}

#[derive(Debug)]
pub struct TranscriptAssembler {
    timeline: TimelineMap,
    merge_policy: SegmentMergePolicy,
    segments: Vec<Segment>,
    /// Decode-scope provenance aligned one-for-one with `segments`. This is
    /// intentionally carried beside the public transcript contract: local
    /// speaker counters are meaningful only inside the slice that produced
    /// them, but that implementation detail must not leak into serialized
    /// transcript segments.
    speaker_scope_by_segment: Vec<Option<usize>>,
    stats: LongFormAssembleStats,
    /// End of the audio region (original-timeline seconds) already committed by
    /// prior slices. Consecutive slices overlap at forced/energy cuts, so the
    /// next slice re-decodes that region; anything it emits before this point is
    /// a redundant re-read (or a weak-model hallucination of partial audio) and
    /// is trimmed by time before it can survive into the transcript.
    committed_end_original: Option<f32>,
    approximate_word_timestamps: bool,
}

impl TranscriptAssembler {
    pub fn new(timeline: TimelineMap, merge_policy: SegmentMergePolicy) -> Self {
        Self {
            timeline,
            merge_policy,
            segments: Vec::new(),
            speaker_scope_by_segment: Vec::new(),
            stats: LongFormAssembleStats::default(),
            committed_end_original: None,
            approximate_word_timestamps: false,
        }
    }

    /// Only decoders without native word alignment may regenerate estimates
    /// after stitching. Acoustic/native anchors retain their original times.
    pub(crate) fn with_approximate_word_timestamps(mut self, approximate: bool) -> Self {
        self.approximate_word_timestamps = approximate;
        self
    }

    pub fn push_slice_result(&mut self, transcript: SliceTranscript) {
        self.push_slice_result_with_scope(transcript, None);
    }

    /// Assemble one independently decoded slice while retaining the exact
    /// scope that owns its local speaker labels. The scope travels through the
    /// same overlap trimming and duplicate suppression as the segment itself,
    /// so callers never need to guess provenance back from final timestamps.
    pub(crate) fn push_slice_result_with_speaker_scope(
        &mut self,
        transcript: SliceTranscript,
        speaker_scope: usize,
    ) {
        self.push_slice_result_with_scope(transcript, Some(speaker_scope));
    }

    fn push_slice_result_with_scope(
        &mut self,
        mut transcript: SliceTranscript,
        speaker_scope: Option<usize>,
    ) {
        // The trim boundary is the region committed by *prior* slices; this
        // slice's own span is folded into the boundary afterwards so the next
        // slice trims against it (even if this slice is silent / emits nothing).
        let trim_boundary = self.committed_end_original;
        let slice_committed_end = self.slice_content_end_original(&transcript.slice);
        self.committed_end_original = Some(match self.committed_end_original {
            Some(previous) => previous.max(slice_committed_end),
            None => slice_committed_end,
        });
        transcript.text = transcript.text.trim().to_string();
        if transcript.text.is_empty()
            && transcript
                .segments
                .iter()
                .all(|segment| segment.text.trim().is_empty())
        {
            self.stats.skipped_silent_chunks += 1;
            return;
        }
        if transcript.segments.is_empty() {
            let sample_rate = 16_000.0_f32;
            transcript.segments.push(Segment {
                start: 0.0,
                end: transcript.slice.content_duration_samples() as f32 / sample_rate,
                text: transcript.text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            });
            transcript.time_domain = SegmentTimeDomain::RelativeToSliceContent;
        }
        let time_domain = transcript.time_domain;
        let slice = transcript.slice.clone();
        // Text stitch is a cross-slice seam repair for wordless seq2seq
        // families. Same-slice adjacent cues (Whisper word timestamps, cue
        // splitter output) stay distinct even when their texts share a tail.
        let mut cross_slice_seam = true;
        for mut segment in transcript.segments {
            segment.text = segment.text.trim().to_string();
            if segment.text.is_empty() {
                continue;
            }
            let mut mapped = self.map_segment_time(&segment, &slice, time_domain);
            // Cross-slice stitch first. DecodeInvariant families (qwen/moss/
            // firered/mimo) force interpolated words for cue splitting; those
            // words make a midpoint trim eat the textual re-read prefix and
            // would skip stitch if we required `words.is_empty()`. Same-slice
            // adjacent cues still skip stitch via `cross_slice_seam`.
            if cross_slice_seam && self.try_stitch_seam_overlap(&mut mapped) {
                self.stats.duplicate_merge_count += 1;
                if mapped.text.trim().is_empty() {
                    continue;
                }
                self.segments.push(mapped);
                self.speaker_scope_by_segment.push(speaker_scope);
                cross_slice_seam = false;
                continue;
            }
            // Time-domain overlap trim: drop any leading words / whole segments
            // whose audio lies in the region a prior slice already committed.
            // Skip when stitch already consumed the re-read; interpolated
            // seq2seq words would otherwise delete the matching prefix.
            if let Some(boundary) = trim_boundary
                && trim_committed_overlap(&mut mapped, boundary)
            {
                self.stats.duplicate_merge_count += 1;
                continue;
            }
            if self.try_drop_redundant_segment(&mapped) {
                self.stats.duplicate_merge_count += 1;
                continue;
            }
            cross_slice_seam = false;
            // Distinct segments are kept distinct: the post-ASR cue
            // re-segmentation pass owns subtitle granularity, so the assembler
            // no longer coalesces adjacent same-speaker segments into paragraph
            // blobs. Only exact / high-overlap duplicates from slice overlap are
            // dropped above.
            self.segments.push(mapped);
            self.speaker_scope_by_segment.push(speaker_scope);
        }
    }

    pub fn into_transcription(self) -> Transcription {
        self.into_parts().0
    }

    pub fn into_parts(self) -> (Transcription, LongFormAssembleStats) {
        let (transcription, stats, _) = self.into_parts_with_speaker_scopes();
        (transcription, stats)
    }

    /// Return the assembled transcript together with exact per-segment decode
    /// scope provenance. The vector is aligned with `transcription.segments`;
    /// `None` denotes a caller that did not opt into local-speaker scopes.
    pub(crate) fn into_parts_with_speaker_scopes(
        self,
    ) -> (Transcription, LongFormAssembleStats, Vec<Option<usize>>) {
        let text = crate::transcript_text::join_segment_texts(
            self.segments.iter().map(|segment| segment.text.as_str()),
        );
        let transcription = Transcription {
            truncated_decodes: Vec::new(),
            unnamed_speakers: Vec::new(),
            text,
            segments: self.segments,
            longform: None,
            language: None,
            ..Default::default()
        };
        debug_assert_eq!(
            transcription.segments.len(),
            self.speaker_scope_by_segment.len(),
            "speaker scope provenance must stay aligned with assembled segments"
        );
        (transcription, self.stats, self.speaker_scope_by_segment)
    }

    pub fn benchmark_metadata(&self) -> LongFormBenchmarkMetadata {
        LongFormBenchmarkMetadata {
            chunk_count: self.segments.len(),
            skipped_silent_chunks: self.stats.skipped_silent_chunks,
            duplicate_merge_count: self.stats.duplicate_merge_count,
            provenance: vec!["core.longform.assembler".to_string()],
        }
    }

    fn map_segment_time(
        &self,
        segment: &Segment,
        slice: &AudioSlice,
        time_domain: SegmentTimeDomain,
    ) -> Segment {
        let sample_rate = 16_000.0_f32;
        let mut start = segment.start.max(0.0);
        let mut end = segment.end.max(start);
        let mut content_offset = 0.0_f32;
        if time_domain == SegmentTimeDomain::RelativeToSliceContent {
            content_offset = slice.content_start_sample as f32 / sample_rate;
            start += content_offset;
            end += content_offset;
        }
        let original_start = self.timeline.map_processed_to_original_seconds(start);
        let original_end = self
            .timeline
            .map_processed_to_original_seconds(end)
            .max(original_start);
        let words = segment
            .words
            .iter()
            .filter_map(|word| {
                map_word_time_to_original(
                    word,
                    content_offset,
                    time_domain,
                    &self.timeline,
                    original_start,
                    original_end,
                )
            })
            .collect();
        Segment {
            start: original_start,
            end: original_end,
            text: segment.text.clone(),
            speaker: segment.speaker.clone(),
            speaker_label: segment.speaker_label.clone(),
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words,
        }
    }

    /// End of this slice's content span in original-timeline seconds. The
    /// `content_end_sample` indexes the same processed/plan audio domain that
    /// [`Self::map_segment_time`] maps from, so mapping it through the timeline
    /// yields the original-time cut point this slice commits up to.
    fn slice_content_end_original(&self, slice: &AudioSlice) -> f32 {
        let sample_rate = 16_000.0_f32;
        let processed_end = slice.content_end_sample as f32 / sample_rate;
        self.timeline
            .map_processed_to_original_seconds(processed_end)
    }

    fn try_drop_redundant_segment(&self, current: &Segment) -> bool {
        let Some(previous) = self.segments.last() else {
            return false;
        };
        if current.start < previous.end {
            return false;
        }
        let gap_seconds = current.start - previous.end;
        if gap_seconds > self.merge_policy.max_gap_seconds {
            return false;
        }
        let previous_words = normalize_words(&previous.text);
        let current_words = normalize_words(&current.text);
        if previous_words.is_empty() || current_words.is_empty() {
            return false;
        }
        if previous_words == current_words {
            return true;
        }
        if current_words.len() >= self.merge_policy.redundant_min_words
            && contains_window(&previous_words, &current_words)
        {
            return true;
        }
        // Fuzzy LCW drop stays ASCII-only. Per-character CJK units make the
        // longform_en_zh fixture (repeated Mandarin blocks plus a new tail)
        // look like an 80% re-read and would delete the continuation.
        if !mostly_ascii_units(&previous_words) || !mostly_ascii_units(&current_words) {
            return false;
        }
        let overlap = longest_common_window_len(&previous_words, &current_words);
        let min_len = previous_words.len().min(current_words.len());
        min_len >= self.merge_policy.redundant_min_words
            && overlap as f32 / min_len as f32 >= self.merge_policy.redundant_overlap_ratio
    }

    /// Stitch a wordless cross-slice re-read by a tail-anchored suffix/prefix
    /// match. Seq2seq families such as Qwen emit one segment spanning the
    /// whole slice and no word timestamps, so the time-domain trim above cannot
    /// see the 0.5–2s energy overlap.
    ///
    /// Thresholds: n ≥ 2 when the windows time-overlap, n ≥ 3 on a mere
    /// abutment. A one-unit match is allowed only for a single non-numeral
    /// CJK char that is not a doubled sentence-initial (「谢谢」), and only
    /// when the segment time-overlap covers that char's estimated duration
    /// -- so 「我 我」 / 「吃饭」+「饭，」 still stitch, while 「八」/「谢」
    /// and English head-words do not. Matches are always a suffix of
    /// `previous`; an interior window is never searched.
    ///
    /// The stitch rewrites text only. `previous.end` is left alone so a
    /// later slice cannot swallow the earlier segment's boundary.
    fn try_stitch_seam_overlap(&mut self, current: &mut Segment) -> bool {
        let Some(previous) = self.segments.last_mut() else {
            return false;
        };
        let time_overlap = previous.end - current.start;
        let time_overlaps = time_overlap > 1.0e-3;
        let gap_seconds = current.start - previous.end;
        if !time_overlaps && gap_seconds > self.merge_policy.max_gap_seconds {
            return false;
        }
        let min_units = if time_overlaps { 2 } else { 3 };
        let stitched = apply_suffix_prefix_stitch(previous, current, min_units, time_overlap);
        if stitched && self.approximate_word_timestamps {
            // A prior seam may already have advanced these estimates beyond
            // the original audio window. Preserve that committed boundary
            // when this same segment is stitched to yet another slice.
            let previous_start = previous
                .words
                .first()
                .map_or(previous.start, |word| word.start.max(previous.start))
                .min(previous.end);
            previous.words = crate::subtitle::cues::interpolate_word_timestamps(
                &previous.text,
                previous_start,
                previous.end,
            );
            // Keep both original segment windows for subsequent alignment.
            // Only the approximate remainder starts after committed speech.
            let remainder_start = current.start.max(previous.end).min(current.end);
            current.words = crate::subtitle::cues::interpolate_word_timestamps(
                &current.text,
                remainder_start,
                current.end,
            );
        }
        stitched
    }
}

fn speakers_conflict(previous: &Segment, current: &Segment) -> bool {
    match (&previous.speaker, &current.speaker) {
        (Some(left), Some(right)) => left != right,
        _ => false,
    }
}

/// Rewrite seam text when a tail-anchored suffix/prefix overlap exists.
/// Returns `true` when `current` was trimmed (caller drops it if empty).
/// Never extends `previous.end`.
fn apply_suffix_prefix_stitch(
    previous: &mut Segment,
    current: &mut Segment,
    min_units: usize,
    time_overlap_seconds: f32,
) -> bool {
    let Some(overlap) = suffix_prefix_overlap(
        &previous.text,
        &current.text,
        min_units,
        time_overlap_seconds,
    ) else {
        return false;
    };
    let consume_end = extend_consumed_current_end(&current.text, overlap.curr_end);
    let prev_prefix = char_prefix(&previous.text, overlap.prev_start);
    let consumed = current
        .text
        .chars()
        .take(consume_end)
        .skip(overlap.curr_start)
        .collect::<String>();
    let consumed = consumed.trim();
    let remainder = char_suffix(&current.text, consume_end).trim().to_string();
    let previous_words = split_words_at_char(&previous.text, &previous.words, overlap.prev_start);
    let current_words = split_words_at_char(&current.text, &current.words, consume_end);
    let leftover_words = current_words
        .as_ref()
        .map(|(_, leftover)| leftover.clone())
        .unwrap_or_default();
    if speakers_conflict(previous, current) {
        current.text = remainder;
        current.words = leftover_words;
        return true;
    }
    if let Some((keep_prefix, _)) = previous_words {
        previous.words = keep_prefix;
    }
    let mut completed = format!("{prev_prefix}{consumed}");
    // A period sitting on `previous` after the overlap is a truncated-slice
    // hallucination when current continues with content (no punct right after
    // the overlap). Keep it only when the remainder is empty or current
    // already carried immediately-following seam punct.
    let strip_truncated_period = consume_end == overlap.curr_end && !remainder.is_empty();
    if !strip_truncated_period {
        let trailing = previous_trailing_punct(&previous.text, overlap.prev_end);
        if !trailing.is_empty() && !completed.ends_with(&trailing) {
            completed.push_str(&trailing);
        }
    }
    previous.text = completed;
    current.text = remainder;
    current.words = leftover_words;
    true
}

fn map_word_time_to_original(
    word: &WordTimestamp,
    content_offset: f32,
    time_domain: SegmentTimeDomain,
    timeline: &TimelineMap,
    segment_start: f32,
    segment_end: f32,
) -> Option<WordTimestamp> {
    let text = word.word.trim();
    if text.is_empty() || !word.start.is_finite() || !word.end.is_finite() {
        return None;
    }
    let mut start = word.start.max(0.0);
    let mut end = word.end.max(start);
    if time_domain == SegmentTimeDomain::RelativeToSliceContent {
        start += content_offset;
        end += content_offset;
    }
    let original_start = timeline
        .map_processed_to_original_seconds(start)
        .clamp(segment_start, segment_end);
    let original_end = timeline
        .map_processed_to_original_seconds(end)
        .max(original_start)
        .clamp(original_start, segment_end);
    Some(WordTimestamp {
        word: text.to_string(),
        start: original_start,
        end: original_end,
        confidence: word.confidence,
    })
}

/// Trim the part of a mapped segment that lies in the audio region a prior slice
/// already committed (`[.., boundary)` in original-timeline seconds). Returns
/// `true` when the whole segment falls inside that region and should be dropped.
///
/// A word is assigned to whichever side of `boundary` holds the majority of it
/// (midpoint rule), so a word straddling the cut is kept in exactly one slice.
/// Leading committed words are dropped and the segment text is reconstructed
/// from the surviving word span (exact substring of the original text, so CJK
/// and glued punctuation stay intact); a segment left empty is dropped.
fn trim_committed_overlap(segment: &mut Segment, boundary: f32) -> bool {
    // Whole segment already behind the committed frontier: drop it outright.
    // (This is the standalone-orphan shape, e.g. a hallucinated 1-word cue.)
    if segment.end <= boundary {
        return true;
    }
    if segment.words.is_empty() {
        // A wordless slice-spanning segment's midpoint almost never sits inside
        // the 0.5–2s overlap, so a midpoint rule would keep the whole re-read.
        // Leave the text for suffix-prefix stitch instead of dropping or keeping
        // the segment as a unit.
        return false;
    }
    let first_keep = segment
        .words
        .iter()
        .position(|word| 0.5 * (word.start + word.end) >= boundary);
    let Some(first_keep) = first_keep else {
        // Every word's majority sits in the committed region.
        return true;
    };
    if first_keep == 0 {
        return false;
    }
    let chars: Vec<char> = segment.text.chars().collect();
    let new_text = match leading_word_char_offset(&chars, &segment.words, first_keep) {
        Some(offset) => chars[offset..]
            .iter()
            .collect::<String>()
            .trim()
            .to_string(),
        // Words did not align to the text (unexpected): rebuild from the kept
        // word tokens rather than mis-slice the string.
        None => crate::transcript_text::join_segment_texts(
            segment.words[first_keep..]
                .iter()
                .map(|word| word.word.as_str()),
        ),
    };
    segment.words.drain(0..first_keep);
    if let Some(first) = segment.words.first() {
        segment.start = first.start;
    }
    segment.text = new_text;
    segment.text.trim().is_empty()
}

/// Char offset at which `words[first_keep]` begins within `chars`, found by the
/// same greedy whitespace-delimited match the cue splitter uses. Returns `None`
/// if a leading word does not align to the text.
fn leading_word_char_offset(
    chars: &[char],
    words: &[WordTimestamp],
    first_keep: usize,
) -> Option<usize> {
    let mut idx = 0usize;
    for word in &words[..first_keep] {
        while idx < chars.len() && chars[idx].is_whitespace() {
            idx += 1;
        }
        let token: Vec<char> = word.word.trim().chars().collect();
        if token.is_empty() {
            continue;
        }
        if idx + token.len() > chars.len() || chars[idx..idx + token.len()] != token[..] {
            return None;
        }
        idx += token.len();
    }
    while idx < chars.len() && chars[idx].is_whitespace() {
        idx += 1;
    }
    Some(idx)
}

/// Split `words` at the first token whose aligned start is `>= char_index`.
/// `None` when the greedy word/text walk does not align (leave words alone).
fn split_words_at_char(
    text: &str,
    words: &[WordTimestamp],
    char_index: usize,
) -> Option<(Vec<WordTimestamp>, Vec<WordTimestamp>)> {
    if words.is_empty() {
        return Some((Vec::new(), Vec::new()));
    }
    let chars: Vec<char> = text.chars().collect();
    let mut split_at = words.len();
    for index in 0..=words.len() {
        let offset = leading_word_char_offset(&chars, words, index)?;
        if offset >= char_index {
            split_at = index;
            break;
        }
    }
    Some((words[..split_at].to_vec(), words[split_at..].to_vec()))
}

/// Word list for [`TranscriptAssembler::try_drop_redundant_segment`].
/// ASCII keeps the historical fold (`don't` → `dont`, `twenty-one` →
/// `twentyone`, `café` → `caf`). CJK has no ASCII letters, so it falls
/// through to one unit per non-punctuation character.
fn normalize_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .flat_map(|word| {
            let ascii: String = word
                .chars()
                .filter(|ch| ch.is_ascii_alphanumeric())
                .collect::<String>()
                .to_ascii_lowercase();
            if !ascii.is_empty() {
                return vec![ascii];
            }
            word.chars()
                .filter(|ch| !is_seam_punctuation(*ch))
                .map(|ch| ch.to_string())
                .collect::<Vec<_>>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OverlapUnit {
    token: String,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SuffixPrefixOverlap {
    prev_start: usize,
    prev_end: usize,
    curr_start: usize,
    curr_end: usize,
}

fn is_seam_punctuation(ch: char) -> bool {
    matches!(
        ch,
        ',' | '.' | '!' | '?' | ';' | ':' | '，' | '。' | '！' | '？' | '；' | '：' | '、'
    )
}

fn is_ascii_word_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '\'' || ch == '-'
}

fn is_cjk_numeral(ch: char) -> bool {
    matches!(
        ch,
        '零' | '一'
            | '二'
            | '三'
            | '四'
            | '五'
            | '六'
            | '七'
            | '八'
            | '九'
            | '十'
            | '百'
            | '千'
            | '万'
    )
}

fn is_numeric_unit(token: &str) -> bool {
    !token.is_empty()
        && token
            .chars()
            .all(|ch| ch.is_ascii_digit() || is_cjk_numeral(ch))
}

fn is_single_cjk_char(token: &str) -> bool {
    let mut chars = token.chars();
    match (chars.next(), chars.next()) {
        (Some(ch), None) => !ch.is_ascii() && !ch.is_ascii_whitespace(),
        _ => false,
    }
}

fn overlap_units(text: &str) -> Vec<OverlapUnit> {
    let chars: Vec<char> = text.chars().collect();
    let mut units = Vec::new();
    let mut index = 0usize;
    while index < chars.len() {
        let ch = chars[index];
        if ch.is_whitespace() || is_seam_punctuation(ch) {
            index += 1;
            continue;
        }
        if is_ascii_word_char(ch) {
            let start = index;
            let mut token = String::new();
            while index < chars.len() && is_ascii_word_char(chars[index]) {
                token.push(chars[index].to_ascii_lowercase());
                index += 1;
            }
            units.push(OverlapUnit {
                token,
                start,
                end: index,
            });
            continue;
        }
        units.push(OverlapUnit {
            token: ch.to_string(),
            start: index,
            end: index + 1,
        });
        index += 1;
    }
    units
}

fn char_prefix(text: &str, char_count: usize) -> String {
    text.chars().take(char_count).collect()
}

fn char_suffix(text: &str, char_skip: usize) -> String {
    text.chars().skip(char_skip).collect()
}

const CJK_CHAR_SECONDS: f32 = 0.18;

/// Longest *suffix* of `previous` that is a prefix of `current`. Interior
/// matches are not considered: a re-read can only replay the previous tail.
fn suffix_prefix_overlap(
    previous: &str,
    current: &str,
    min_units: usize,
    time_overlap_seconds: f32,
) -> Option<SuffixPrefixOverlap> {
    let previous_units = overlap_units(previous);
    let current_units = overlap_units(current);
    if previous_units.is_empty() || current_units.is_empty() {
        return None;
    }
    let max_n = previous_units.len().min(current_units.len());
    if max_n == 0 {
        return None;
    }
    for n in (1..=max_n).rev() {
        let previous_suffix = &previous_units[previous_units.len() - n..];
        let current_prefix = &current_units[..n];
        if !previous_suffix
            .iter()
            .zip(current_prefix)
            .all(|(left, right)| left.token == right.token)
        {
            continue;
        }
        if !accept_overlap_n(
            n,
            min_units,
            previous_suffix,
            &current_units,
            time_overlap_seconds,
        ) {
            continue;
        }
        return Some(SuffixPrefixOverlap {
            prev_start: previous_suffix[0].start,
            prev_end: previous_suffix[n - 1].end,
            curr_start: current_prefix[0].start,
            curr_end: current_prefix[n - 1].end,
        });
    }
    None
}

fn accept_overlap_n(
    n: usize,
    min_units: usize,
    overlap: &[OverlapUnit],
    current_units: &[OverlapUnit],
    time_overlap_seconds: f32,
) -> bool {
    if overlap.iter().all(|unit| is_numeric_unit(&unit.token)) {
        return false;
    }
    if n >= min_units {
        return true;
    }
    // Single-char CJK re-read (「我 我」, 「吃饭」/「饭，」): require the
    // windows to actually overlap by at least that char's spoken duration.
    // A doubled sentence-initial (「谢谢」) is a new phrase, not a re-read.
    n == 1
        && time_overlap_seconds > 1.0e-3
        && time_overlap_seconds + 1.0e-3 >= CJK_CHAR_SECONDS
        && overlap.len() == 1
        && is_single_cjk_char(&overlap[0].token)
        && !(current_units.len() >= 2 && current_units[0].token == current_units[1].token)
}

fn extend_consumed_current_end(current: &str, overlap_end: usize) -> usize {
    let chars: Vec<char> = current.chars().collect();
    if overlap_end >= chars.len() {
        return chars.len();
    }
    let mut end = overlap_end;
    while end < chars.len() && chars[end].is_whitespace() {
        end += 1;
    }
    while end < chars.len() && is_seam_punctuation(chars[end]) {
        end += 1;
    }
    end
}

/// Punctuation that already sat on `previous` after the overlap units.
fn previous_trailing_punct(previous: &str, prev_end: usize) -> String {
    let chars: Vec<char> = previous.chars().collect();
    let mut index = prev_end;
    while index < chars.len() && chars[index].is_whitespace() {
        index += 1;
    }
    let mut trailing = String::new();
    while index < chars.len() && is_seam_punctuation(chars[index]) {
        trailing.push(chars[index]);
        index += 1;
    }
    trailing
}

fn mostly_ascii_units(words: &[String]) -> bool {
    if words.is_empty() {
        return false;
    }
    let ascii = words.iter().filter(|word| word.is_ascii()).count();
    ascii * 2 >= words.len()
}

fn contains_window(haystack: &[String], needle: &[String]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// Longest common *contiguous* window. DP is O(n·m) / O(min(n, m)) space.
fn longest_common_window_len(left: &[String], right: &[String]) -> usize {
    if left.is_empty() || right.is_empty() {
        return 0;
    }
    let (short, long) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    let mut prev = vec![0usize; short.len() + 1];
    let mut best = 0usize;
    for long_token in long {
        let mut curr = vec![0usize; short.len() + 1];
        for (index, short_token) in short.iter().enumerate() {
            if short_token == long_token {
                curr[index + 1] = prev[index] + 1;
                best = best.max(curr[index + 1]);
            }
        }
        prev = curr;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::longform::{AudioSlice, AudioSliceKind, TimelineAnchor};

    fn slice(start: usize, end: usize) -> AudioSlice {
        AudioSlice {
            index: 0,
            kind: AudioSliceKind::Fixed,
            start_sample: start,
            end_sample: end,
            content_start_sample: start,
            content_end_sample: end,
        }
    }

    #[test]
    fn assembler_maps_relative_segment_times() {
        let timeline = TimelineMap::from_anchors(vec![
            TimelineAnchor {
                processed_seconds: 0.0,
                original_seconds: 0.0,
            },
            TimelineAnchor {
                processed_seconds: 10.0,
                original_seconds: 10.0,
            },
        ]);
        let mut assembler = TranscriptAssembler::new(timeline, SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000, 32_000),
            text: "hello".to_string(),
            segments: vec![Segment {
                start: 0.0,
                end: 0.5,
                text: "hello".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: vec![WordTimestamp {
                    word: "hello".to_string(),
                    start: 0.1,
                    end: 0.4,
                    confidence: None,
                }],
            }],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 1);
        assert!(transcription.segments[0].start >= 1.0);
        assert_eq!(transcription.segments[0].words.len(), 1);
        assert_eq!(transcription.segments[0].words[0].word, "hello");
        assert!(transcription.segments[0].words[0].start >= 1.1);
        assert!(transcription.segments[0].words[0].end <= 1.4);
    }

    #[test]
    fn assembler_keeps_exact_scope_provenance_across_slice_overlap() {
        fn labeled_segment(start: f32, end: f32, text: &str) -> Segment {
            Segment {
                start,
                end,
                text: text.to_string(),
                speaker: Some("SPEAKER_01".to_string()),
                speaker_label: Some("SPEAKER_01".to_string()),
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            }
        }

        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result_with_speaker_scope(
            SliceTranscript {
                slice: slice(0, 480_000),
                text: "first owner".to_string(),
                segments: vec![labeled_segment(29.6, 29.9, "first owner")],
                time_domain: SegmentTimeDomain::RelativeToSliceContent,
            },
            0,
        );
        assembler.push_slice_result_with_speaker_scope(
            SliceTranscript {
                slice: slice(472_000, 496_000),
                text: "second owner".to_string(),
                segments: vec![labeled_segment(0.6, 0.9, "second owner")],
                time_domain: SegmentTimeDomain::RelativeToSliceContent,
            },
            1,
        );

        let (transcription, _, scopes) = assembler.into_parts_with_speaker_scopes();
        assert_eq!(transcription.segments.len(), 2);
        assert!((transcription.segments[0].start - 29.6).abs() < 1e-4);
        assert!((transcription.segments[1].start - 30.1).abs() < 1e-4);
        assert_eq!(scopes, vec![Some(0), Some(1)]);
    }

    #[test]
    fn assembler_drops_redundant_overlap() {
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world from openasr".to_string(),
            segments: vec![Segment {
                start: 0.0,
                end: 1.0,
                text: "hello world from openasr".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            }],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(15_000, 31_000),
            text: "hello world from openasr".to_string(),
            segments: vec![Segment {
                start: 1.05,
                end: 2.0,
                text: "hello world from openasr".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            }],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 1);
    }

    #[test]
    fn assembler_keeps_adjacent_segments_distinct() {
        // Adjacent, non-overlapping segments are no longer coalesced into a
        // paragraph blob: the post-ASR cue re-segmentation pass owns subtitle
        // granularity. Each segment survives with its own words and timing, and
        // the joined transcript text still reads as one paragraph.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 32_000),
            text: "hello world".to_string(),
            segments: vec![
                Segment {
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
                            start: 0.1,
                            end: 0.4,
                            confidence: None,
                        },
                        WordTimestamp {
                            word: "world".to_string(),
                            start: 0.5,
                            end: 0.9,
                            confidence: None,
                        },
                    ],
                },
                Segment {
                    start: 1.0,
                    end: 2.0,
                    text: "from openasr".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_person_id: None,
                    speaker_snapshot_label: None,
                    words: vec![
                        WordTimestamp {
                            word: "from".to_string(),
                            start: 1.1,
                            end: 1.4,
                            confidence: None,
                        },
                        WordTimestamp {
                            word: "openasr".to_string(),
                            start: 1.5,
                            end: 1.9,
                            confidence: None,
                        },
                    ],
                },
            ],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(
            transcription.segments.len(),
            2,
            "adjacent segments must stay distinct"
        );
        assert_eq!(transcription.segments[0].text, "hello world");
        assert_eq!(transcription.segments[1].text, "from openasr");
        assert_eq!(transcription.text, "hello world from openasr");
    }

    #[test]
    fn assembler_preserves_slice_boundaries_for_synthesized_slice_segments() {
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "first chunk".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000, 32_000),
            text: "second chunk".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[0].start, 0.0);
        assert_eq!(transcription.segments[0].end, 1.0);
        assert_eq!(transcription.segments[1].start, 1.0);
        assert_eq!(transcription.segments[1].end, 2.0);
    }

    fn word(text: &str, start: f32, end: f32) -> WordTimestamp {
        WordTimestamp {
            word: text.to_string(),
            start,
            end,
            confidence: None,
        }
    }

    fn absolute_segment(text: &str, start: f32, end: f32, words: Vec<WordTimestamp>) -> Segment {
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

    /// DecodeInvariant interpolation: Han+trailing punct (「我。」) and ASCII
    /// runs with attached punct, linearly spaced — same grouping as
    /// `seq2seq_word_timestamps_from_generated_tokens`.
    fn interpolated_words(text: &str, start: f32, end: f32) -> Vec<WordTimestamp> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                continue;
            }
            let starts_han = !ch.is_ascii() && !is_seam_punctuation(ch);
            let last_is_han = current
                .chars()
                .next_back()
                .is_some_and(|last| !last.is_ascii() && !is_seam_punctuation(last));
            if (starts_han && !current.is_empty()) || (ch.is_ascii_alphanumeric() && last_is_han) {
                tokens.push(std::mem::take(&mut current));
            }
            current.push(ch);
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        if tokens.is_empty() {
            return Vec::new();
        }
        let step = (end - start).max(1.0e-3) / tokens.len() as f32;
        tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                let word_start = start + step * index as f32;
                word(token, word_start, word_start + step)
            })
            .collect()
    }

    #[test]
    fn assembler_time_trims_hallucinated_leading_overlap_word() {
        // Field defect shape: a forced cut at 1.0s widens the overlap so the next
        // slice re-reads the straddling audio. A weak model hallucinates the head
        // of its monolithic segment ("If,") from that already-committed region;
        // its text does not match anything in slice 1, so text-equality dedup is
        // blind to it. The time trim drops it by timestamp and keeps the rest.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(12_000, 32_000),
            text: "If, mad indeed would".to_string(),
            segments: vec![absolute_segment(
                "If, mad indeed would",
                0.80,
                1.90,
                vec![
                    word("If,", 0.80, 0.95),
                    word("mad", 1.10, 1.30),
                    word("indeed", 1.35, 1.60),
                    word("would", 1.65, 1.90),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[1].text, "mad indeed would");
        assert_eq!(transcription.segments[1].words.len(), 3);
        assert_eq!(transcription.segments[1].words[0].word, "mad");
        assert!((transcription.segments[1].start - 1.10).abs() < 1e-4);
        assert_eq!(transcription.text, "hello world mad indeed would");
        // The trimmed word is not a dropped segment, so no whole-segment drop.
        assert_eq!(stats.duplicate_merge_count, 0);
    }

    #[test]
    fn assembler_drops_standalone_orphan_inside_committed_span() {
        // The "If," rendered as its own leading cue: the whole segment sits behind
        // the committed frontier and is dropped outright.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(12_000, 32_000),
            text: "If,".to_string(),
            segments: vec![
                absolute_segment("If,", 0.80, 0.95, vec![word("If,", 0.80, 0.95)]),
                absolute_segment(
                    "mad indeed",
                    1.10,
                    1.60,
                    vec![word("mad", 1.10, 1.30), word("indeed", 1.35, 1.60)],
                ),
            ],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[0].text, "hello world");
        assert_eq!(transcription.segments[1].text, "mad indeed");
        assert_eq!(transcription.text, "hello world mad indeed");
        assert_eq!(stats.duplicate_merge_count, 1);
    }

    #[test]
    fn assembler_keeps_straddling_word_with_majority_after_boundary() {
        // A word straddling the 1.0s cut whose midpoint (1.025s) is past the
        // boundary belongs to the new slice and is kept whole.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(12_000, 32_000),
            text: "straddle tail".to_string(),
            segments: vec![absolute_segment(
                "straddle tail",
                0.85,
                1.60,
                vec![word("straddle", 0.85, 1.20), word("tail", 1.30, 1.60)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[1].text, "straddle tail");
        assert_eq!(transcription.segments[1].words.len(), 2);
    }

    #[test]
    fn assembler_trims_straddling_word_with_majority_before_boundary() {
        // Same cut, but the leading word's midpoint (0.925s) is before the
        // boundary, so the word belongs to the prior slice and is trimmed.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(12_000, 32_000),
            text: "straddle tail".to_string(),
            segments: vec![absolute_segment(
                "straddle tail",
                0.75,
                1.60,
                vec![word("straddle", 0.75, 1.10), word("tail", 1.30, 1.60)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[1].text, "tail");
        assert_eq!(transcription.segments[1].words.len(), 1);
        assert_eq!(transcription.segments[1].words[0].word, "tail");
    }

    #[test]
    fn assembler_does_not_trim_without_overlap() {
        // Abutting, non-overlapping slices: the second slice's words all sit past
        // the committed frontier, so nothing is trimmed.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000, 32_000),
            text: "next words here".to_string(),
            segments: vec![absolute_segment(
                "next words here",
                1.20,
                1.90,
                vec![
                    word("next", 1.20, 1.40),
                    word("words", 1.50, 1.70),
                    word("here", 1.75, 1.90),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[1].text, "next words here");
        assert_eq!(transcription.segments[1].words.len(), 3);
        assert_eq!(stats.duplicate_merge_count, 0);
    }

    #[test]
    fn assembler_maps_packed_slice_segments_back_to_original_timeline() {
        let timeline = TimelineMap::from_anchors(vec![
            TimelineAnchor {
                processed_seconds: 0.0,
                original_seconds: 0.0,
            },
            TimelineAnchor {
                processed_seconds: 1.0,
                original_seconds: 1.0,
            },
            TimelineAnchor {
                processed_seconds: 1.2,
                original_seconds: 12.0,
            },
            TimelineAnchor {
                processed_seconds: 2.2,
                original_seconds: 13.0,
            },
        ]);
        let mut assembler = TranscriptAssembler::new(timeline, SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: AudioSlice {
                index: 0,
                kind: AudioSliceKind::Energy,
                start_sample: 0,
                end_sample: 35_200,
                content_start_sample: 0,
                content_end_sample: 35_200,
            },
            text: "first second".to_string(),
            segments: vec![
                Segment {
                    start: 0.1,
                    end: 0.9,
                    text: "first".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_person_id: None,
                    speaker_snapshot_label: None,
                    words: Vec::new(),
                },
                Segment {
                    start: 1.3,
                    end: 2.0,
                    text: "second".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_person_id: None,
                    speaker_snapshot_label: None,
                    words: Vec::new(),
                },
            ],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert!(transcription.segments[0].end <= 1.0);
        assert!(transcription.segments[1].start >= 12.0);
        assert!(transcription.segments[1].end <= 13.0);
    }

    fn energy_slice(index: usize, start: usize, end: usize) -> AudioSlice {
        AudioSlice {
            index,
            kind: AudioSliceKind::Energy,
            start_sample: start,
            end_sample: end,
            content_start_sample: start,
            content_end_sample: end,
        }
    }

    fn assemble_three_wordless(slices: [&str; 3]) -> String {
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        let bounds = [
            (0, 16_000 * 25 + 8_000),
            (16_000 * 25, 16_000 * 50 + 8_000),
            (16_000 * 50, 16_000 * 69 + 9_600),
        ];
        for (index, (text, (start, end))) in slices.iter().zip(bounds).enumerate() {
            assembler.push_slice_result(SliceTranscript {
                slice: energy_slice(index, start, end),
                text: (*text).to_string(),
                segments: Vec::new(),
                time_domain: SegmentTimeDomain::RelativeToSliceContent,
            });
        }
        assembler.into_transcription().text
    }

    fn assemble_wordless_overlap(previous: &str, current: &str) -> Transcription {
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 25 + 8_000),
            text: previous.to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 25, 16_000 * 50 + 8_000),
            text: current.to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.into_transcription()
    }

    #[test]
    fn assembler_stitches_qwen_e2e_abutting_wordless_slice_segments() {
        let previous = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country. 今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭。听说那里的麻婆豆腐特别正宗。周末的时候，我。";
        let current = "的时候，我通常会读书或者看一部电影，放松一下。今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭。听说那里的麻婆豆腐特别正宗。吃饭。听说那里的麻婆豆腐特别正宗。周末的时候，我通常会读书或者看一部电影，放松一下。And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
        let (transcription, stats) = {
            let mut assembler =
                TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
            assembler.push_slice_result(SliceTranscript {
                slice: energy_slice(0, 0, 16_000 * 25 + 8_000),
                text: previous.to_string(),
                segments: Vec::new(),
                time_domain: SegmentTimeDomain::RelativeToSliceContent,
            });
            assembler.push_slice_result(SliceTranscript {
                slice: energy_slice(1, 16_000 * 25, 16_000 * 69 + 9_600),
                text: current.to_string(),
                segments: Vec::new(),
                time_domain: SegmentTimeDomain::RelativeToSliceContent,
            });
            assembler.into_parts()
        };
        assert!(
            stats.duplicate_merge_count >= 1,
            "abutting 的时候我 (n=4) must stitch, got {stats:?} text={:?}",
            transcription.text
        );
        assert!(
            transcription.segments[0].text.ends_with("周末的时候，我"),
            "previous must keep its own window text, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            !transcription.segments[0].text.contains("通常会"),
            "current body must not move onto previous, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            transcription.segments[1].text.starts_with("通常会"),
            "remainder must start after the overlap units, got {:#?}",
            transcription.segments[1]
        );
        assert!(
            !transcription.segments[1].text.starts_with("的时候"),
            "re-read prefix must not survive on the later slice, got {:#?}",
            transcription.segments
        );
        assert!(
            (transcription.segments[0].end - 25.5).abs() < 1e-3,
            "previous.end must stay on the first slice window, got {:#?}",
            transcription.segments
        );
        assert!(
            (transcription.segments[1].start - 25.0).abs() < 1e-3,
            "remainder must keep the later slice window, got {:#?}",
            transcription.segments
        );
    }

    #[test]
    fn stitched_estimates_fill_the_original_windows_without_retiming_native_words() {
        let previous = "周末的时候，我。";
        let current = "的时候，我通常会读书。";
        for approximate in [false, true] {
            let mut assembler =
                TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default())
                    .with_approximate_word_timestamps(approximate);
            assembler.push_slice_result(SliceTranscript {
                slice: energy_slice(0, 0, 16_000 * 25 + 8_000),
                text: previous.into(),
                segments: vec![absolute_segment(
                    previous,
                    0.0,
                    25.5,
                    interpolated_words(previous, 0.0, 25.5),
                )],
                time_domain: SegmentTimeDomain::RelativeToSliceContent,
            });
            let original_words = interpolated_words(current, 0.0, 20.0);
            assembler.push_slice_result(SliceTranscript {
                slice: energy_slice(1, 16_000 * 25, 16_000 * 45),
                text: current.into(),
                segments: vec![absolute_segment(current, 0.0, 20.0, original_words.clone())],
                time_domain: SegmentTimeDomain::RelativeToSliceContent,
            });
            let out = assembler.into_transcription();
            assert_eq!(out.segments.len(), 2);
            let left = &out.segments[0];
            let right = &out.segments[1];
            assert_eq!(left.end, 25.5);
            assert_eq!(right.start, 25.0);
            assert_eq!(right.end, 45.0);
            assert_eq!(right.text, "通常会读书。");
            if approximate {
                assert_eq!(left.words.last().unwrap().end, left.end);
                assert_eq!(right.words.first().unwrap().start, left.end);
                assert_eq!(right.words.last().unwrap().end, right.end);
                assert_eq!(
                    left.words
                        .iter()
                        .map(|word| word.word.as_str())
                        .collect::<String>(),
                    left.text
                );
                assert!(right.words.iter().all(|word| word.confidence.is_none()));
                // A later stitch must not rewind a start already repaired at
                // the preceding seam, even when the original windows overlap.
                let mut next = absolute_segment("会读书。然后散步。", 44.5, 60.0, Vec::new());
                let mut resumed = TranscriptAssembler::new(
                    TimelineMap::identity(),
                    SegmentMergePolicy::default(),
                )
                .with_approximate_word_timestamps(true);
                resumed.segments = out.segments.clone();
                assert!(resumed.try_stitch_seam_overlap(&mut next));
                assert_eq!(resumed.segments[1].words[0].start, 25.5);
            } else {
                let native_first = original_words
                    .iter()
                    .find(|word| word.word == "通")
                    .unwrap();
                assert_eq!(
                    right.words.first().unwrap().start,
                    25.0 + native_first.start
                );
            }
        }
    }

    #[test]
    fn assembler_stitches_decode_invariant_worded_qwen_slice_seam() {
        // Production CLI path: DecodeInvariant words are forced on for cue
        // splitting, so qwen emits one slice-spanning segment with interpolated
        // timestamps. Midpoint trim would eat "的时候" and skip stitch.
        let previous = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country. 今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭。听说那里的麻婆豆腐特别正宗。周末的时候，我。";
        let current = "的时候，我通常会读书或者看一部电影，放松一下。今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭。听说那里的麻婆豆腐特别正宗。吃饭。听说那里的麻婆豆腐特别正宗。周末的时候，我通常会读书或者看一部电影，放松一下。And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 25 + 8_000),
            text: previous.to_string(),
            segments: vec![absolute_segment(
                previous,
                0.0,
                25.5,
                interpolated_words(previous, 0.0, 25.5),
            )],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 25, 16_000 * 69 + 9_600),
            text: current.to_string(),
            segments: vec![absolute_segment(
                current,
                0.0,
                44.1,
                interpolated_words(current, 0.0, 44.1),
            )],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let (transcription, stats) = assembler.into_parts();
        assert!(
            stats.duplicate_merge_count >= 1,
            "worded DecodeInvariant seam must stitch, got {stats:?} text={:?}",
            transcription.text
        );
        assert!(
            transcription.segments[0].text.ends_with("周末的时候，我"),
            "previous must keep its own window text, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            !transcription.segments[0].text.contains("通常会"),
            "current body must not move onto previous, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            transcription.segments.len() >= 2,
            "remainder must stay after the overlap units, got {:#?}",
            transcription.segments
        );
        assert!(
            transcription.segments[1].text.starts_with("通常会"),
            "remainder must start after the overlap units, got {:#?}",
            transcription.segments[1]
        );
        assert!(
            (transcription.segments[0].end - 25.5).abs() < 1e-3,
            "previous.end must stay on the first slice window, got {:#?}",
            transcription.segments
        );
        assert!(
            (transcription.segments[1].start - 25.0).abs() < 1e-3,
            "remainder must keep the later slice window, got {:#?}",
            transcription.segments
        );
    }

    #[test]
    fn assembler_stitches_wordless_cjk_overlap_re_read() {
        // Field defect from qwen3-asr-0.6b on fixtures/longform_en_zh.wav: the
        // energy cut at 25.5s re-reads ~0.5s of audio, the model emits one
        // wordless segment per slice, and the overlap decoded as "的时候，"
        // after a truncated "周末的时候，我。".
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 25 + 8_000),
            text: "周末的时候，我。".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 25, 16_000 * 50 + 8_000),
            text: "的时候，我通常会读书或者看一部电影，放松一下。".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(
            transcription.text.replace(' ', ""),
            "周末的时候，我通常会读书或者看一部电影，放松一下。"
        );
        assert_eq!(transcription.segments[0].text, "周末的时候，我");
        assert!(
            transcription.segments[1].text.starts_with("通常会"),
            "remainder must start after the overlap units, got {:#?}",
            transcription.segments
        );
        assert!(
            !transcription
                .segments
                .iter()
                .any(|segment| segment.text.contains("的时候，") && segment.start + 1.0e-3 >= 25.0),
            "re-read prefix must not survive on the later slice, got {:#?}",
            transcription.segments
        );
        assert!(
            (transcription.segments[0].end - 25.5).abs() < 1e-3,
            "stitch must not swallow previous.end, got {:#?}",
            transcription.segments
        );
        assert!(
            stats.duplicate_merge_count >= 1,
            "overlap re-read must count as a seam stitch, got {stats:?}"
        );
    }

    #[test]
    fn assembler_strips_truncated_period_when_remainder_continues() {
        // Qwen seam: the model plants `。` at the slice cut. Current continues
        // with content after the overlap, so that period is not real.
        let previous = "周末的时候，我。";
        let current = "的时候，我通常会读书或者看一部电影，放松一下。";
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 25 + 8_000),
            text: previous.to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 25, 16_000 * 50 + 8_000),
            text: current.to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert!(
            transcription.segments[0].text.ends_with('我'),
            "truncated period must be stripped, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            !transcription.segments[0].text.ends_with('。'),
            "previous must not keep the fake period, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            (transcription.segments[0].end - 25.5).abs() < 1e-3,
            "stripping the period must not move previous.end, got {:#?}",
            transcription.segments
        );
    }

    #[test]
    fn assembler_does_not_move_current_sentence_body_onto_previous() {
        // Text may cross the slice window; timestamps must stay with the
        // audio that produced them. Consumed = overlap units + following
        // punct only.
        let previous = "周末的时候，我。";
        let current = "的时候，我通常会读书或者看一部电影，放松一下。今天天气非常好，";
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 25 + 8_000),
            text: previous.to_string(),
            segments: vec![absolute_segment(previous, 0.0, 25.5, Vec::new())],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 25, 16_000 * 50 + 8_000),
            text: current.to_string(),
            segments: vec![absolute_segment(current, 0.0, 25.5, Vec::new())],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments[0].text, "周末的时候，我");
        assert!(
            !transcription.segments[0].text.contains("通常会"),
            "previous must not absorb speech after the slice boundary, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            transcription.segments[1]
                .text
                .contains("通常会读书或者看一部电影，放松一下。"),
            "sentence body must stay on the later slice, got {:#?}",
            transcription.segments[1]
        );
        assert!(
            (transcription.segments[0].end - 25.5).abs() < 1e-3,
            "previous.end must stay 25.5, got {:#?}",
            transcription.segments
        );
        assert!(
            (transcription.segments[1].start - 25.0).abs() < 1e-3,
            "remainder must keep the later slice start, got {:#?}",
            transcription.segments
        );
    }

    #[test]
    fn assembler_joins_origin_longform_golden_slice_texts() {
        let mimo = assemble_three_wordless([
            "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country. 今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭，听说那里的麻婆豆腐特别正宗。周末的时候，我",
            "我通常会读书或者看一部电影放松一下。And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭，听说那里的麻婆豆腐特别正宗。",
            "周末的时候，我通常会读书或者看一部电影放松一下。And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.",
        ]);
        let firered_aed = assemble_three_wordless([
            "AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我",
            "我通常会读书或者看一部电影放松一下 AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗",
            "周末的时候我通常会读书或者看一部电影放松一下 ANDSO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY",
        ]);
        let firered_llm = assemble_three_wordless([
            "and so my fellow americans ask not what your country can do for you ask what you can do for your country 今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我",
            "我通常会读书或者看一部电影放松一下 and so my fellow americans ask not what your country can do for you ask what you can do for your country 今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗中",
            "周末的时候我通常会读书或者看一部电影放松一下 and so my fellow americans ask not what your country can do for you ask what you can do for your country",
        ]);
        assert_eq!(
            mimo,
            "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country. 今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭，听说那里的麻婆豆腐特别正宗。周末的时候，我通常会读书或者看一部电影放松一下。And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭，听说那里的麻婆豆腐特别正宗。周末的时候，我通常会读书或者看一部电影放松一下。And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country."
        );
        assert_eq!(
            firered_aed,
            "AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我通常会读书或者看一部电影放松一下 AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我通常会读书或者看一部电影放松一下 ANDSO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY"
        );
        assert_eq!(
            firered_llm,
            "and so my fellow americans ask not what your country can do for you ask what you can do for your country 今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我通常会读书或者看一部电影放松一下 and so my fellow americans ask not what your country can do for you ask what you can do for your country 今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗中周末的时候我通常会读书或者看一部电影放松一下 and so my fellow americans ask not what your country can do for you ask what you can do for your country"
        );
    }

    #[test]
    fn assembler_stitches_single_cjk_char_when_windows_time_overlap() {
        // firered / mimo goldens on the same fixture: "...我" + "我通常会..."
        let transcription =
            assemble_wordless_overlap("周末的时候，我", "我通常会读书或者看一部电影放松一下。");
        assert_eq!(
            transcription.text,
            "周末的时候，我通常会读书或者看一部电影放松一下。"
        );
    }

    #[test]
    fn assembler_keeps_short_abutting_repetition_without_time_overlap() {
        // Two-unit prefix shared by abutting complete slices must not stitch.
        // Exact duplicates still drop via text-equality; this pair differs.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000),
            text: "今天好".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000, 32_000),
            text: "今天坏".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.text, "今天好今天坏");
    }

    #[test]
    fn assembler_drops_leading_orphan_contained_in_previous_suffix() {
        // 50.5s qwen shape: next slice re-reads a tail-anchored prefix
        // "吃饭。听说…正宗。" (15 units ≥ 2), then continues.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 50 + 8_000),
            text: "晚上我们还计划去一家新开的川菜馆吃饭。听说那里的麻婆豆腐特别正宗。".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 50, 16_000 * 69),
            text: "吃饭。听说那里的麻婆豆腐特别正宗。周末的时候，我通常会读书或者看一部电影，放松一下。".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let (transcription, stats) = assembler.into_parts();
        assert!(
            !transcription.text.contains("吃饭。吃饭")
                && transcription
                    .text
                    .matches("听说那里的麻婆豆腐特别正宗")
                    .count()
                    == 1,
            "leading 吃饭。 plus the re-read sentence must be dropped, got {:?}",
            transcription.text
        );
        assert!(
            transcription
                .text
                .contains("周末的时候，我通常会读书或者看一部电影，放松一下。"),
            "continuation after the re-read must stay, got {:?}",
            transcription.text
        );
        assert!(
            !transcription
                .segments
                .iter()
                .any(|segment| segment.text.trim() == "吃饭。"),
            "isolated 吃饭。 cue must not survive, got {:#?}",
            transcription.segments
        );
        assert!(
            stats.duplicate_merge_count >= 1,
            "leading re-read trim must count, got {stats:?}"
        );
    }

    #[test]
    fn assembler_stitches_partial_suffix_char_re_read() {
        // moss shape at the same seam: "...吃饭" + "饭，听说..."
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 50 + 8_000),
            text: "晚上我们还计划去一家新开的川菜馆吃饭".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 50, 16_000 * 69),
            text: "饭，听说那里的麻婆豆腐特别正宗。".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(
            transcription.text.replace(' ', ""),
            "晚上我们还计划去一家新开的川菜馆吃饭，听说那里的麻婆豆腐特别正宗。"
        );
    }

    #[test]
    fn assembler_keeps_wo_when_new_sentence_starts_with_wo() {
        let transcription = assemble_wordless_overlap(
            "今天天气非常好，我打算和朋友们一起去公园散步。",
            "我觉得晚上会下雨。",
        );
        assert!(
            transcription.text.contains("我觉得晚上会下雨"),
            "new-sentence 我 must stay, got {:?}",
            transcription.text
        );
        assert!(!transcription.text.contains("散步。觉得"));
    }

    #[test]
    fn assembler_keeps_jintian_when_it_is_not_the_previous_tail() {
        let transcription = assemble_wordless_overlap("我们今天去公园散步。", "今天晚上吃什么？");
        assert!(
            transcription.text.contains("今天晚上吃什么"),
            "new-sentence 今天 must stay, got {:?}",
            transcription.text
        );
    }

    #[test]
    fn assembler_keeps_english_head_word_that_is_not_the_previous_tail() {
        let transcription = assemble_wordless_overlap(
            "and so my fellow americans ask not what your country can do for you",
            "and then we all went home",
        );
        assert!(
            transcription.text.contains("and then we all went home"),
            "English head-word must stay, got {:?}",
            transcription.text
        );
        assert!(!transcription.text.contains("for you then"));
    }

    #[test]
    fn assembler_does_not_glue_digit_strings_at_a_shared_numeral() {
        let transcription = assemble_wordless_overlap("电话是一三八", "八零零一二三四");
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.text, "电话是一三八八零零一二三四");
    }

    #[test]
    fn assembler_does_not_eat_reduplicated_thanks() {
        let transcription = assemble_wordless_overlap("非常感谢", "谢谢大家");
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.text, "非常感谢谢谢大家");
    }

    #[test]
    fn assembler_keeps_worded_adjacent_segments_in_the_same_slice() {
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 6),
            text: "that is how it ended in the end in the end we all agreed".to_string(),
            segments: vec![
                absolute_segment(
                    "that is how it ended in the end",
                    0.0,
                    3.0,
                    vec![
                        word("that", 0.0, 0.3),
                        word("is", 0.3, 0.5),
                        word("how", 0.5, 0.8),
                        word("it", 0.8, 1.0),
                        word("ended", 1.0, 1.5),
                        word("in", 1.6, 1.8),
                        word("the", 1.8, 2.1),
                        word("end", 2.1, 3.0),
                    ],
                ),
                absolute_segment(
                    "in the end we all agreed",
                    3.0,
                    6.0,
                    vec![
                        word("in", 3.0, 3.2),
                        word("the", 3.2, 3.4),
                        word("end", 3.4, 3.8),
                        word("we", 3.8, 4.1),
                        word("all", 4.1, 4.5),
                        word("agreed", 4.5, 6.0),
                    ],
                ),
            ],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(
            transcription.segments[0].text,
            "that is how it ended in the end"
        );
        assert_eq!(transcription.segments[1].text, "in the end we all agreed");
        assert_eq!(transcription.segments[0].words.len(), 8);
        assert_eq!(transcription.segments[1].words.len(), 6);
    }

    #[test]
    fn normalize_words_keeps_ascii_fold_and_splits_cjk() {
        assert_eq!(normalize_words("don't"), vec!["dont".to_string()]);
        assert_eq!(normalize_words("twenty-one"), vec!["twentyone".to_string()]);
        assert_eq!(normalize_words("café"), vec!["caf".to_string()]);
        assert_eq!(
            normalize_words("今天好"),
            vec!["今".to_string(), "天".to_string(), "好".to_string()]
        );
    }

    #[test]
    fn longest_common_window_is_quadratic_for_200_units() {
        let left: Vec<String> = (0..200).map(|index| format!("t{index}")).collect();
        let mut right = left.clone();
        right.rotate_left(40);
        let started = std::time::Instant::now();
        let overlap = longest_common_window_len(&left, &right);
        let elapsed = started.elapsed();
        assert_eq!(overlap, 160);
        assert!(
            elapsed.as_millis() < 200,
            "LCW of 200 units must stay well under 200ms, took {elapsed:?}"
        );
    }
}
