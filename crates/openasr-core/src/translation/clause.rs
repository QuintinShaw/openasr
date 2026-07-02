use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClauseId(u64);

impl ClauseId {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ClauseId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "c-{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClauseStatus {
    Active,
    Finalized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClauseBoundaryReason {
    StrongPunctuation,
    CommaPunctuation,
    SafeLength,
    AsrFinal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClauseSegment {
    pub clause_id: ClauseId,
    pub replaces_clause_id: Option<ClauseId>,
    pub source_version: u64,
    pub text: String,
    pub status: ClauseStatus,
    pub boundary_reason: Option<ClauseBoundaryReason>,
    pub revised: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ClauseSegmentationUpdate {
    pub segments: Vec<ClauseSegment>,
    pub retired_clause_ids: Vec<ClauseId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClauseSegmentationConfig {
    pub min_clause_chars: usize,
    pub soft_char_threshold: usize,
    pub hard_char_threshold: usize,
}

impl Default for ClauseSegmentationConfig {
    fn default() -> Self {
        Self {
            min_clause_chars: 4,
            soft_char_threshold: 24,
            hard_char_threshold: 32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClauseSegmenter {
    config: ClauseSegmentationConfig,
    next_clause_id: u64,
    active_clause_id: ClauseId,
    active_source_version: u64,
    committed_clauses: Vec<CommittedClause>,
    active_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommittedClause {
    clause_id: ClauseId,
    /// The exact source bytes this clause covers, including any surrounding
    /// whitespace. The concatenation of all `raw_text` MUST stay a byte-exact
    /// prefix of the caller's full text: `push_text` detects revisions with
    /// `strip_prefix`, so dropping even one whitespace byte here makes every
    /// later push look like a revision (retire + re-emit loop) forever.
    raw_text: String,
}

impl Default for ClauseSegmenter {
    fn default() -> Self {
        Self::new(ClauseSegmentationConfig::default())
    }
}

impl ClauseSegmenter {
    pub fn new(config: ClauseSegmentationConfig) -> Self {
        Self {
            config,
            next_clause_id: 2,
            active_clause_id: ClauseId::new(1),
            active_source_version: 0,
            committed_clauses: Vec::new(),
            active_text: String::new(),
        }
    }

    pub fn push_partial(&mut self, text: &str) -> ClauseSegmentationUpdate {
        self.push_text(text, false)
    }

    pub fn push_final(&mut self, text: &str) -> ClauseSegmentationUpdate {
        self.push_text(text, true)
    }

    fn push_text(&mut self, text: &str, force_final: bool) -> ClauseSegmentationUpdate {
        let mut update = ClauseSegmentationUpdate::default();
        let committed_text = self.committed_text();
        let (mut working, revised) = if let Some(uncommitted) = text.strip_prefix(&committed_text) {
            (uncommitted.to_string(), false)
        } else {
            let (working, retired_clause_ids) = self.reset_committed_region_for_revision(text);
            update.retired_clause_ids = retired_clause_ids;
            (working, true)
        };

        // Whitespace-only clause bytes waiting to be attached to the next
        // emitted clause's raw text, so no source byte ever falls out of the
        // committed region.
        let mut blank_prefix = String::new();
        while let Some(boundary) = find_clause_boundary(&working, &self.config) {
            let mut raw: String = working[..boundary.byte_end].to_string();
            working.replace_range(..boundary.byte_end, "");
            let clause = raw.trim().to_string();
            if clause.is_empty() {
                blank_prefix.push_str(&raw);
                continue;
            }
            if !blank_prefix.is_empty() {
                raw.insert_str(0, &blank_prefix);
                blank_prefix.clear();
            }
            update.segments.push(self.emit_clause(
                clause,
                raw,
                ClauseStatus::Finalized,
                Some(boundary.reason),
                revised,
            ));
        }

        if force_final {
            let mut raw = std::mem::take(&mut working);
            if !blank_prefix.is_empty() {
                raw.insert_str(0, &blank_prefix);
            }
            let clause = raw.trim().to_string();
            if !clause.is_empty() {
                update.segments.push(self.emit_clause(
                    clause,
                    raw,
                    ClauseStatus::Finalized,
                    Some(ClauseBoundaryReason::AsrFinal),
                    revised,
                ));
            } else if let Some(last) = self.committed_clauses.last_mut() {
                // Trailing whitespace after the last clause must stay in the
                // committed region or the next utterance's strip_prefix breaks.
                last.raw_text.push_str(&raw);
            }
            assign_replacement_clause_ids(&mut update);
            self.active_text.clear();
            return update;
        }

        let active = working.trim().to_string();
        if active.is_empty() {
            self.active_text.clear();
        } else if active != self.active_text || revised {
            self.active_text = active.clone();
            self.active_source_version = self.active_source_version.saturating_add(1);
            update.segments.push(ClauseSegment {
                clause_id: self.active_clause_id,
                replaces_clause_id: None,
                source_version: self.active_source_version,
                text: active,
                status: ClauseStatus::Active,
                boundary_reason: None,
                revised,
            });
        } else {
            update.segments.push(ClauseSegment {
                clause_id: self.active_clause_id,
                replaces_clause_id: None,
                source_version: self.active_source_version,
                text: active,
                status: ClauseStatus::Active,
                boundary_reason: None,
                revised: false,
            });
        }
        assign_replacement_clause_ids(&mut update);
        update
    }

    fn committed_text(&self) -> String {
        self.committed_clauses
            .iter()
            .map(|clause| clause.raw_text.as_str())
            .collect()
    }

    fn reset_committed_region_for_revision(&mut self, text: &str) -> (String, Vec<ClauseId>) {
        let committed_text = self.committed_text();
        let common_prefix = longest_common_prefix_byte_len(&committed_text, text);
        let mut preserved_byte_len = 0usize;
        let mut keep_count = 0usize;
        for clause in &self.committed_clauses {
            let next_len = preserved_byte_len.saturating_add(clause.raw_text.len());
            if next_len <= common_prefix {
                preserved_byte_len = next_len;
                keep_count = keep_count.saturating_add(1);
            } else {
                break;
            }
        }
        let mut retired_clause_ids = self
            .committed_clauses
            .iter()
            .skip(keep_count)
            .map(|clause| clause.clause_id)
            .collect::<Vec<_>>();
        if !self.active_text.is_empty() {
            retired_clause_ids.push(self.active_clause_id);
        }
        self.committed_clauses.truncate(keep_count);
        self.active_text.clear();
        self.active_source_version = 0;
        // No trim here: the working text must cover every byte after the
        // preserved committed region, or the prefix invariant breaks again.
        (
            text.get(preserved_byte_len..)
                .unwrap_or_default()
                .to_string(),
            retired_clause_ids,
        )
    }

    fn emit_clause(
        &mut self,
        text: String,
        raw_text: String,
        status: ClauseStatus,
        boundary_reason: Option<ClauseBoundaryReason>,
        revised: bool,
    ) -> ClauseSegment {
        debug_assert_eq!(raw_text.trim(), text);
        self.active_source_version = self.active_source_version.saturating_add(1);
        let segment = ClauseSegment {
            clause_id: self.active_clause_id,
            replaces_clause_id: None,
            source_version: self.active_source_version,
            text,
            status,
            boundary_reason,
            revised,
        };
        self.committed_clauses.push(CommittedClause {
            clause_id: segment.clause_id,
            raw_text,
        });
        self.active_text.clear();
        self.active_clause_id = self.allocate_clause_id();
        self.active_source_version = 0;
        segment
    }

    fn allocate_clause_id(&mut self) -> ClauseId {
        let clause_id = ClauseId::new(self.next_clause_id);
        self.next_clause_id = self.next_clause_id.saturating_add(1);
        clause_id
    }
}

fn assign_replacement_clause_ids(update: &mut ClauseSegmentationUpdate) {
    if update.retired_clause_ids.is_empty() {
        return;
    }
    for (segment, retired_clause_id) in update
        .segments
        .iter_mut()
        .zip(update.retired_clause_ids.iter().copied())
    {
        segment.replaces_clause_id = Some(retired_clause_id);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Boundary {
    byte_end: usize,
    reason: ClauseBoundaryReason,
}

fn find_clause_boundary(text: &str, config: &ClauseSegmentationConfig) -> Option<Boundary> {
    let strong = first_punctuation_boundary(text, true, 1).map(|byte_end| Boundary {
        byte_end,
        reason: ClauseBoundaryReason::StrongPunctuation,
    });
    let secondary =
        first_punctuation_boundary(text, false, config.min_clause_chars).map(|byte_end| Boundary {
            byte_end,
            reason: ClauseBoundaryReason::CommaPunctuation,
        });
    match (strong, secondary) {
        (Some(strong), Some(secondary)) => {
            return Some(if secondary.byte_end < strong.byte_end {
                secondary
            } else {
                strong
            });
        }
        (Some(strong), None) => return Some(strong),
        (None, Some(secondary)) => return Some(secondary),
        (None, None) => {}
    }
    safe_length_boundary(text, config).map(|byte_end| Boundary {
        byte_end,
        reason: ClauseBoundaryReason::SafeLength,
    })
}

fn first_punctuation_boundary(text: &str, strong: bool, min_chars: usize) -> Option<usize> {
    let mut count = 0usize;
    for (byte_start, ch) in text.char_indices() {
        if ch.is_whitespace() {
            continue;
        }
        count = count.saturating_add(1);
        let matches = if strong {
            is_strong_clause_punctuation(ch)
        } else {
            is_secondary_clause_punctuation(ch)
        };
        if matches && count >= min_chars {
            return Some(byte_start + ch.len_utf8());
        }
    }
    None
}

fn safe_length_boundary(text: &str, config: &ClauseSegmentationConfig) -> Option<usize> {
    let chars = text.char_indices().collect::<Vec<_>>();
    if chars.len() < config.soft_char_threshold {
        return None;
    }
    let soft = config.soft_char_threshold.min(chars.len());
    let hard = config.hard_char_threshold.min(chars.len());

    for index in (config.min_clause_chars..=soft).rev() {
        let byte_end = char_end_at(&chars, index)?;
        if is_preferred_boundary(text, byte_end) && is_safe_boundary(text, byte_end) {
            return Some(byte_end);
        }
    }
    for index in soft.saturating_add(1)..=hard {
        let byte_end = char_end_at(&chars, index)?;
        if is_preferred_boundary(text, byte_end) && is_safe_boundary(text, byte_end) {
            return Some(byte_end);
        }
    }
    for index in soft..=hard {
        let byte_end = char_end_at(&chars, index)?;
        if is_safe_boundary(text, byte_end) {
            return Some(byte_end);
        }
    }
    let extended_hard = hard.saturating_add(8).min(chars.len());
    for index in hard.saturating_add(1)..=extended_hard {
        let byte_end = char_end_at(&chars, index)?;
        if is_safe_boundary(text, byte_end) {
            return Some(byte_end);
        }
    }
    None
}

fn longest_common_prefix_byte_len(left: &str, right: &str) -> usize {
    let mut common = 0usize;
    for ((left_index, left_ch), (right_index, right_ch)) in
        left.char_indices().zip(right.char_indices())
    {
        if left_ch != right_ch {
            break;
        }
        common = left_index
            .max(right_index)
            .saturating_add(left_ch.len_utf8());
    }
    common
}

fn char_end_at(chars: &[(usize, char)], char_count: usize) -> Option<usize> {
    if char_count == 0 {
        return Some(0);
    }
    chars
        .get(char_count.saturating_sub(1))
        .map(|(byte_start, ch)| byte_start + ch.len_utf8())
}

fn is_preferred_boundary(text: &str, byte_end: usize) -> bool {
    let Some(prev) = prev_char(text, byte_end) else {
        return false;
    };
    prev.is_whitespace()
        || is_strong_clause_punctuation(prev)
        || is_secondary_clause_punctuation(prev)
        || matches!(
            prev,
            '的' | '了'
                | '着'
                | '过'
                | '吧'
                | '吗'
                | '呢'
                | '啊'
                | '呀'
                | '和'
                | '与'
                | '及'
                | '或'
                | '但'
                | '而'
                | '并'
        )
}

fn is_safe_boundary(text: &str, byte_end: usize) -> bool {
    if byte_end == 0 || byte_end >= text.len() {
        return false;
    }
    if boundary_inside_protected_cjk_term(text, byte_end) {
        return false;
    }
    let Some(prev) = prev_char(text, byte_end) else {
        return false;
    };
    let Some(next) = next_char(text, byte_end) else {
        return false;
    };
    if prev.is_ascii_alphanumeric() && next.is_ascii_alphanumeric() {
        return false;
    }
    if prev.is_ascii_digit() && (is_unit_char(next) || next == '.') {
        return false;
    }
    if (is_unit_char(prev) || prev == '.') && next.is_ascii_digit() {
        return false;
    }
    true
}

fn prev_char(text: &str, byte_end: usize) -> Option<char> {
    text.get(..byte_end)?.chars().next_back()
}

fn next_char(text: &str, byte_end: usize) -> Option<char> {
    text.get(byte_end..)?.chars().next()
}

fn boundary_inside_protected_cjk_term(text: &str, byte_end: usize) -> bool {
    const PROTECTED_TERMS: &[&str] = &["语音识别模型", "语音识别", "大语言模型", "流式识别"];
    PROTECTED_TERMS.iter().any(|term| {
        text.match_indices(term)
            .any(|(start, matched)| start < byte_end && byte_end < start + matched.len())
    })
}

fn is_strong_clause_punctuation(ch: char) -> bool {
    matches!(ch, '。' | '！' | '？' | '；' | '!' | '?' | ';')
}

fn is_secondary_clause_punctuation(ch: char) -> bool {
    matches!(ch, '，' | '、' | '：' | ',')
}

/// Subtitle punctuation follow-through: a clause cut at a secondary boundary
/// (comma/enumeration/colon) is a continuing fragment, but the model —
/// translating it as standalone text — terminates it with a full stop, which
/// renders as a false sentence end in live captions. Mirror the source's
/// continuation punctuation onto the translation: swap a trailing `.`/`。`
/// for `,` (ellipses and other terminals are left alone). Returns `None`
/// when no adjustment is needed.
pub(crate) fn align_translation_terminal_punctuation(
    source: &str,
    translation: &str,
) -> Option<String> {
    let source_last = source.trim_end().chars().next_back()?;
    if !is_secondary_clause_punctuation(source_last) {
        return None;
    }
    let trimmed = translation.trim_end();
    if trimmed.ends_with("...") || trimmed.ends_with('…') {
        return None;
    }
    let last = trimmed.chars().next_back()?;
    if last != '.' && last != '。' {
        return None;
    }
    // Dotted abbreviations ("U.S.", "e.g.") keep their final dot: detect the
    // `.X.` shape — a single letter between two dots — before swapping.
    let mut rev = trimmed.chars().rev().skip(1);
    if let (Some(prev), Some(prev2)) = (rev.next(), rev.next())
        && prev.is_ascii_alphanumeric()
        && prev2 == '.'
    {
        return None;
    }
    let mut aligned = trimmed.to_string();
    aligned.truncate(aligned.len() - last.len_utf8());
    aligned.push(',');
    Some(aligned)
}

fn is_unit_char(ch: char) -> bool {
    matches!(
        ch,
        '%' | '％'
            | '°'
            | '℃'
            | '年'
            | '月'
            | '日'
            | '号'
            | '点'
            | '分'
            | '秒'
            | '米'
            | '克'
            | '千'
            | '万'
            | '亿'
            | 'm'
            | 'M'
            | 'g'
            | 'G'
            | 'k'
            | 'K'
            | 'l'
            | 'L'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_punctuation_follows_comma_bounded_source_clauses() {
        // Comma-bounded fragment: trailing full stop becomes a comma.
        assert_eq!(
            align_translation_terminal_punctuation(
                "几乎把所有群体都拉踩个遍，",
                "It mocks every group."
            ),
            Some("It mocks every group,".to_string())
        );
        assert_eq!(
            align_translation_terminal_punctuation("开头都是，", "It all starts like this。"),
            Some("It all starts like this,".to_string())
        );
        // Sentence-final source keeps the translation's terminator.
        assert_eq!(
            align_translation_terminal_punctuation("包括你我。", "Including you and me."),
            None
        );
        // No trailing punctuation on the source: leave the translation alone.
        assert_eq!(
            align_translation_terminal_punctuation("后面继续", "It continues."),
            None
        );
        // Ellipses and existing commas stay untouched.
        assert_eq!(
            align_translation_terminal_punctuation("先这样，", "Like this..."),
            None
        );
        assert_eq!(
            align_translation_terminal_punctuation("先这样，", "Like this,"),
            None
        );
        // Dotted abbreviations keep their dot.
        assert_eq!(
            align_translation_terminal_punctuation("在美国，", "In the U.S."),
            None
        );
    }

    #[test]
    fn punctuation_first_segments_zh_sentence_and_comma_boundaries() {
        let mut segmenter = ClauseSegmenter::default();
        let update = segmenter.push_partial("我们先保持中文实时字幕，然后翻译。后面继续");

        assert_eq!(update.segments.len(), 3);
        assert_eq!(update.segments[0].text, "我们先保持中文实时字幕，");
        assert_eq!(
            update.segments[0].boundary_reason,
            Some(ClauseBoundaryReason::CommaPunctuation)
        );
        assert_eq!(update.segments[1].text, "然后翻译。");
        assert_eq!(
            update.segments[1].boundary_reason,
            Some(ClauseBoundaryReason::StrongPunctuation)
        );
        assert_eq!(update.segments[2].text, "后面继续");
        assert_eq!(update.segments[2].status, ClauseStatus::Active);
    }

    #[test]
    fn length_segmentation_uses_24_char_safe_threshold_not_16_hard_cut() {
        let mut segmenter = ClauseSegmenter::default();
        let update = segmenter.push_partial("这个语音识别模型可以在本地快速运行并保持低延迟显示");

        assert!(update.segments.len() >= 2);
        assert_ne!(update.segments[0].text.chars().count(), 16);
        assert!(
            !update.segments[0].text.ends_with("语音识别模"),
            "must not split the protected CJK term at the 16-char style boundary"
        );
        assert!(
            update
                .segments
                .iter()
                .any(|segment| segment.text.contains("语音识别模型"))
        );
    }

    #[test]
    fn safe_length_boundary_does_not_split_latin_or_number_unit_runs() {
        let mut segmenter = ClauseSegmenter::new(ClauseSegmentationConfig {
            min_clause_chars: 4,
            soft_char_threshold: 8,
            hard_char_threshold: 12,
        });
        let update = segmenter.push_partial("我们测试OpenASR2026版本在30ms内显示");

        assert!(update.segments.len() >= 2);
        for pair in update.segments.windows(2) {
            let prev = pair[0].text.chars().next_back();
            let next = pair[1].text.chars().next();
            assert!(
                !matches!((prev, next), (Some(left), Some(right)) if left.is_ascii_alphanumeric() && right.is_ascii_alphanumeric())
            );
            assert!(
                !matches!((prev, next), (Some(left), Some(right)) if left.is_ascii_digit() && is_unit_char(right))
            );
        }
    }

    #[test]
    fn safe_length_boundary_defers_unbroken_ascii_runs() {
        let mut segmenter = ClauseSegmenter::new(ClauseSegmentationConfig {
            min_clause_chars: 4,
            soft_char_threshold: 8,
            hard_char_threshold: 12,
        });
        let update = segmenter.push_partial("Supercalifragilistic2026BuildPipeline");

        assert_eq!(update.segments.len(), 1);
        assert_eq!(update.segments[0].status, ClauseStatus::Active);
        assert_eq!(
            update.segments[0].text,
            "Supercalifragilistic2026BuildPipeline"
        );
    }

    #[test]
    fn active_only_revision_shrinks_without_reissuing_clause_id() {
        let mut segmenter = ClauseSegmenter::default();
        let first = segmenter.push_partial("我们正在调试实时字幕显示");
        assert_eq!(first.segments.len(), 1);
        let active_clause_id = first.segments[0].clause_id;

        let shrink = segmenter.push_partial("我们正在调试实时");

        assert_eq!(shrink.segments.len(), 1);
        assert_eq!(shrink.segments[0].clause_id, active_clause_id);
        assert_eq!(shrink.segments[0].text, "我们正在调试实时");
        assert_eq!(shrink.segments[0].status, ClauseStatus::Active);
        assert!(!shrink.segments[0].revised);
    }

    #[test]
    fn committed_revision_reissues_affected_clause_ids_with_revised_flag() {
        let mut segmenter = ClauseSegmenter::default();
        let first = segmenter.push_partial("我们先保持中文实时字幕，然后继续");
        assert_eq!(first.segments.len(), 2);
        assert_eq!(first.segments[0].text, "我们先保持中文实时字幕，");
        assert_eq!(first.segments[0].status, ClauseStatus::Finalized);
        assert_eq!(first.segments[1].text, "然后继续");
        let original_final_id = first.segments[0].clause_id;
        let original_active_id = first.segments[1].clause_id;

        let revision = segmenter.push_partial("我们先保持实时字幕，然后继续");

        assert_eq!(revision.segments.len(), 2);
        assert_eq!(revision.segments[0].text, "我们先保持实时字幕，");
        assert_eq!(revision.segments[0].status, ClauseStatus::Finalized);
        assert_ne!(revision.segments[0].clause_id, original_final_id);
        assert!(revision.segments[0].revised);
        assert_eq!(revision.segments[1].text, "然后继续");
        assert_eq!(revision.segments[1].status, ClauseStatus::Active);
        assert_ne!(revision.segments[1].clause_id, original_active_id);
        assert!(revision.segments[1].revised);
    }

    /// Regression for the live zh-en stall (recording 1781267241): a clause
    /// boundary landing next to an ASCII-run space must not break the
    /// committed-prefix match, otherwise every later partial is misread as a
    /// revision and the lane livelocks in a retire/re-emit loop.
    #[test]
    fn whitespace_at_clause_boundary_does_not_trigger_revision_loop() {
        let mut segmenter = ClauseSegmenter::default();
        // The comma boundary leaves a remainder that STARTS with a space
        // (" codex ..."): the committed region must keep that byte or every
        // later push fails the prefix match.
        let prefix = "我们现在来看一下这个东西好吗， codex 是一个工具";
        let first = segmenter.push_partial(prefix);
        assert!(
            first
                .segments
                .iter()
                .any(|segment| segment.status == ClauseStatus::Finalized
                    && segment.text == "我们现在来看一下这个东西好吗，"),
            "the comma must finalize the first clause; got {:?}",
            first.segments
        );

        let mut grown = String::from(prefix);
        for tail in ["确实不错。", "再看第二个，", "然后是 chats 部分，"] {
            grown.push_str(tail);
            let update = segmenter.push_partial(&grown);
            assert!(
                update.retired_clause_ids.is_empty(),
                "pure append must never be treated as a revision; retired {:?} on {:?}",
                update.retired_clause_ids,
                grown
            );
            assert!(update.segments.iter().all(|segment| !segment.revised));
        }
    }

    /// The committed region must stay a byte-exact prefix of the source text
    /// across partials, finals, and revisions; `push_text` relies on it.
    #[test]
    fn committed_text_stays_byte_prefix_across_final_and_next_utterance() {
        let mut segmenter = ClauseSegmenter::default();
        segmenter.push_partial("我们看这两种类型好吗， projects 先看一下");
        // The ASR final commits a second clause whose raw text starts with the
        // space left after the comma; both clauses must keep their bytes.
        segmenter.push_final("我们看这两种类型好吗， projects 先看一下");
        // The next utterance's text is appended onto the finalized text.
        let update =
            segmenter.push_partial("我们看这两种类型好吗， projects 先看一下然后是第二个部分。");
        assert!(
            update.retired_clause_ids.is_empty(),
            "appending a new utterance after a final must not retire clauses; got {:?}",
            update.retired_clause_ids
        );
        assert!(
            update
                .segments
                .iter()
                .any(|segment| segment.status == ClauseStatus::Finalized
                    && segment.text == "然后是第二个部分。"),
            "got {:?}",
            update.segments
        );
    }

    #[test]
    fn asr_final_forces_current_clause_finalize() {
        let mut segmenter = ClauseSegmenter::default();
        let update = segmenter.push_final("还没有标点也要提交");

        assert_eq!(update.segments.len(), 1);
        assert_eq!(update.segments[0].status, ClauseStatus::Finalized);
        assert_eq!(
            update.segments[0].boundary_reason,
            Some(ClauseBoundaryReason::AsrFinal)
        );
    }
}
