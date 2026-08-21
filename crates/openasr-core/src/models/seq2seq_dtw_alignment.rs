//! DTW-based alignment of decoded tokens to audio frames.
//!
//! Ports the core refinement strategy from the whisper-timestamped project.
//! Coarse word timestamps are produced by placing each token at the center of
//! mass of its cross-attention distribution, then splitting words at
//! midpoints between word centers. As soon as two tokens' attention
//! distributions overlap, each token's single center smears the other's
//! timing and the midpoint splits land in the middle of syllables.
//!
//! Instead, this module runs a monotone dynamic-time-warping pass over the
//! `token x frame` cross-attention matrix. The DTW path assigns every token
//! an ordered, non-overlapping span of frames: token `k` owns the frames from
//! where the path first enters token `k` until the path first enters token
//! `k + 1` (the last token runs to the end of the window). Word spans are
//! then derived from their member tokens' spans (see
//! `seq2seq_word_timestamps_from_token_spans`), which keeps the timeline
//! monotone by construction while letting each span follow where its
//! attention actually sits.

/// Width of the per-frame median filter applied to each token's attention row
/// before alignment. Matches whisper-timestamped's default of 9.
const DTW_MEDIAN_FILTER_WIDTH: usize = 9;

/// Whether a token's decoded text carries real speech and so may bracket the
/// DTW alignment window.
///
/// Punctuation and whitespace tokens ("` .`", "` ,`", "` .,`") attach to a
/// neighbouring content word but carry no audible span of their own, and their
/// cross-attention is precisely the diffuse mass that bleeds into surrounding
/// silence or non-speech. whisper-timestamped excludes a trailing final
/// punctuation from segment timing for the same reason
/// (`include_punctuation_in_timing = False`). Letting such a token set the
/// earliest/latest bounds would stretch the band past the spoken audio -- e.g.
/// the "." after a word attending into the ignored trailing music stretches the
/// final word's end far past where the speech actually stops. Whichever tokens
/// survive this filter bracket the band on each side; combined
/// word+punctuation pieces (e.g. "Okay,") contain a letter and so stay.
pub(crate) fn token_text_carries_speech(text: &str) -> bool {
    text.chars().any(|ch| ch.is_alphanumeric())
}

/// Padding, in frames, left around the detected speech band so a token whose
/// single strongest frame sits just inside an edge still owns its full onset
/// and offset. 10 frames is 0.2s at Whisper's 0.02s/frame.
pub(crate) const DTW_SPEECH_BAND_MARGIN_FRAMES: usize = 10;

/// Frame index band the DTW path should align onto, derived from where each
/// token's cross-attention actually peaks. Returns `[start, end)` (end
/// exclusive) within `frame_count`, or `None` when the rows are empty or
/// non-finite and no band can be computed (the caller then falls back to the
/// clip-duration window).
///
/// whisper-timestamped bounds its DTW to the decoded `<|start|>`/`<|end|>`
/// timestamp tokens (`weights[..., start: end]`). OpenASR decodes
/// `<|notimestamps|>`, so it has no such tokens. The closest signal remains the
/// cross-attention: each decoded token is a real spoken syllable, and its
/// head-averaged attention concentrates on the frames where that syllable
/// sounds, so the band from the earliest to the latest per-token peak brackets
/// the spoken text. This is robust to the softmax diffuse floor (each row still
/// spreads a small mass over every frame, so a summed envelope never fully
/// drops at the tail) and to trailing non-speech or leading silence the model
/// ignored, because neither hosts a per-token peak.
///
/// `is_content` holds one flag per row; only rows whose flag is `true` may
/// bracket the band. Passing `token_text_carries_speech` per token keeps
/// punctuation/subword tokens (whose attention is the diffuse mass that bleeds
/// into nearby silence) from stretching the band, as whisper-timestamped does
/// for a trailing final punctuation. Without this the monotone DTW path runs
/// the whole padded window, stretching the first word's start to frame 0 and
/// the last word's end to the window tail.
pub(crate) fn speech_frame_bounds(
    attention: &[Vec<f32>],
    is_content: &[bool],
) -> Option<(usize, usize)> {
    let frame_count = attention.first()?.len();
    if frame_count == 0 {
        return None;
    }
    let mut earliest: Option<usize> = None;
    let mut latest: Option<usize> = None;
    for (index, row) in attention.iter().enumerate() {
        if row.len() != frame_count || !is_content.get(index).copied().unwrap_or(false) {
            continue;
        }
        let (dominant_frame, &dominant_value) = row
            .iter()
            .enumerate()
            .filter(|&(_, &value)| value.is_finite())
            .max_by(|(_, a), (_, b)| a.total_cmp(b))?;
        if dominant_value > 0.0 {
            earliest =
                Some(earliest.map_or(dominant_frame, |existing| existing.min(dominant_frame)));
            latest = Some(latest.map_or(dominant_frame, |existing| existing.max(dominant_frame)));
        }
    }
    let earliest = earliest?;
    let latest = latest?;
    let margin = DTW_SPEECH_BAND_MARGIN_FRAMES;
    let start = earliest.saturating_sub(margin);
    let end = (latest + 1).saturating_add(margin).min(frame_count);
    Some((start, end.max(start + 1)))
}

/// Per-token frame span within one audio window. `frame_end` is exclusive so
/// ordered tokens always satisfy `span[k].frame_end == span[k + 1].frame_start`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TokenFrameSpan {
    pub frame_start: usize,
    pub frame_end: usize,
}

/// Align tokens to audio frames with a monotone DTW pass.
///
/// `attention` holds one row per token; row `t` is the (non-negative)
/// cross-attention weight of token `t` over each audio frame. Returns one
/// frame span per token, or `None` when the matrix is empty, ragged, or
/// non-finite and no alignment can be computed. The caller decides how to
/// degrade (the shared fallback is a per-token center of mass).
pub(crate) fn dtw_align_token_frames(attention: &[Vec<f32>]) -> Option<Vec<TokenFrameSpan>> {
    let token_count = attention.len();
    let frame_count = attention.first()?.len();
    if token_count == 0 || frame_count == 0 {
        return None;
    }
    for row in attention {
        if row.len() != frame_count || row.iter().any(|weight| !weight.is_finite()) {
            return None;
        }
    }
    let mut cost: Vec<Vec<f64>> = attention
        .iter()
        .map(|row| row_alignment_cost(row))
        .collect();
    // Bias the first cell to the global minimum so the path prefers spending
    // its budget early. Mirrors whisper-timestamped's `weights[0, 0] =
    // weights.min()` "encourage to start early" step.
    let earliest = cost.iter().flatten().copied().fold(f64::INFINITY, f64::min);
    if earliest.is_finite() {
        cost[0][0] = earliest;
    }
    let path = dtw_min_cost_path(&cost);
    Some(token_spans_from_path(&path, token_count, frame_count))
}

/// Per-token DTW cost row: median-smooth the attention, sharpen it with a
/// softmax, L2-normalize across frames so every token carries comparable
/// energy, then negate so the DTW minimizes cost (maximizes attention).
/// The reference pipeline head-averages before these steps; the caller's
/// frame rows are already head-averaged and normalized, so the remaining
/// stages are applied directly.
fn row_alignment_cost(row: &[f32]) -> Vec<f64> {
    let smoothed = median_filter_row(row, DTW_MEDIAN_FILTER_WIDTH);
    let sharpened = softmax_row(&smoothed);
    let mut norm = 0.0_f64;
    for &weight in &sharpened {
        let value = weight as f64;
        norm += value * value;
    }
    if norm > 0.0 && norm.is_finite() {
        let inv_norm = 1.0 / norm.sqrt();
        return sharpened
            .iter()
            .map(|&weight| -((weight as f64) * inv_norm))
            .collect();
    }
    vec![0.0_f64; row.len()]
}

/// Centered sliding median over the frame axis, clamped at the row edges.
fn median_filter_row(row: &[f32], width: usize) -> Vec<f32> {
    let len = row.len();
    let mut window_width = width.min(len);
    if window_width.is_multiple_of(2) {
        window_width -= 1;
    }
    if window_width <= 1 {
        return row.to_vec();
    }
    let half = window_width / 2;
    let mut window = Vec::with_capacity(window_width);
    (0..len)
        .map(|index| {
            let start = index.saturating_sub(half);
            let end = (index + half + 1).min(len);
            window.clear();
            window.extend_from_slice(&row[start..end]);
            window.sort_by(f32::total_cmp);
            window[(end - start) / 2]
        })
        .collect()
}

/// Row-wise softmax. Rows are already normalized attention, so this only
/// sharpens the peaks; the fallback uniform row keeps degenerate input from
/// producing NaN costs.
fn softmax_row(row: &[f32]) -> Vec<f32> {
    if row.is_empty() {
        return Vec::new();
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![1.0 / row.len() as f32; row.len()];
    }
    let sum: f32 = row.iter().map(|&weight| (weight - max).exp()).sum();
    if sum > 0.0 && sum.is_finite() {
        return row
            .iter()
            .map(|&weight| (weight - max).exp() / sum)
            .collect();
    }
    vec![1.0 / row.len() as f32; row.len()]
}

/// DP + backtrack over the `token x frame` cost matrix with symmetric-1
/// steps (advance token, advance frame, or both). Mirrors
/// whisper-timestamped's step pattern, which allows two tokens on one frame
/// (subwords can be empty) but never decreases a frame.
fn dtw_min_cost_path(cost: &[Vec<f64>]) -> Vec<(usize, usize)> {
    const FROM_TOP: u8 = 1;
    const FROM_LEFT: u8 = 2;
    const FROM_DIAG: u8 = 3;
    let token_count = cost.len();
    let frame_count = cost[0].len();
    let mut came_from = vec![FROM_DIAG; token_count * frame_count];
    let mut previous = vec![f64::INFINITY; frame_count];
    let mut current = vec![f64::INFINITY; frame_count];
    previous[0] = cost[0][0];
    came_from[0] = 0;
    for frame in 1..frame_count {
        previous[frame] = cost[0][frame] + previous[frame - 1];
        came_from[frame] = FROM_LEFT;
    }
    for token in 1..token_count {
        current[0] = cost[token][0] + previous[0];
        came_from[token * frame_count] = FROM_TOP;
        for frame in 1..frame_count {
            let top = previous[frame];
            let left = current[frame - 1];
            let diag = previous[frame - 1];
            let (best, from) = if diag <= top && diag <= left {
                (diag, FROM_DIAG)
            } else if left <= top {
                (left, FROM_LEFT)
            } else {
                (top, FROM_TOP)
            };
            current[frame] = cost[token][frame] + best;
            came_from[token * frame_count + frame] = from;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    let mut path = Vec::with_capacity(token_count + frame_count - 1);
    let (mut token, mut frame) = (token_count - 1, frame_count - 1);
    path.push((token, frame));
    while token > 0 || frame > 0 {
        match came_from[token * frame_count + frame] {
            FROM_LEFT => frame -= 1,
            FROM_TOP => token -= 1,
            _ => {
                token -= 1;
                frame -= 1;
            }
        }
        path.push((token, frame));
    }
    path.reverse();
    path
}

/// Convert the DTW path into per-token frame spans. A token's span starts
/// where the path first enters the token and ends where the path first
/// enters the next token (exclusive); the last token runs to the final frame.
fn token_spans_from_path(
    path: &[(usize, usize)],
    token_count: usize,
    frame_count: usize,
) -> Vec<TokenFrameSpan> {
    let mut first_frame = vec![0usize; token_count];
    let mut seen = vec![false; token_count];
    for &(token, frame) in path {
        if !seen[token] {
            seen[token] = true;
            first_frame[token] = frame;
        }
    }
    (0..token_count)
        .map(|token| {
            let frame_start = first_frame[token];
            let frame_end = first_frame.get(token + 1).copied().unwrap_or(frame_count);
            TokenFrameSpan {
                frame_start: frame_start.min(frame_end),
                frame_end,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_detection_keeps_letters_drops_punctuation_and_space() {
        assert!(token_text_carries_speech(" Okay"));
        assert!(token_text_carries_speech("got"));
        assert!(token_text_carries_speech("dog."));
        assert!(token_text_carries_speech("a1"));
        assert!(!token_text_carries_speech(" ."));
        assert!(!token_text_carries_speech(" ,"));
        assert!(!token_text_carries_speech(" "));
        assert!(!token_text_carries_speech("!?"))
    }

    /// A smooth, normalized bump centered on `center` within a `frames`-long
    /// window. Real cross-attention rows are soft over several frames, unlike
    /// a single-frame spike (which the median filter would legitimately
    /// suppress).
    fn bump(frames: usize, center: usize) -> Vec<f32> {
        let values: Vec<f32> = (0..frames)
            .map(|frame| {
                let delta = frame as f32 - center as f32;
                (-delta * delta / 4.0).exp()
            })
            .collect();
        let total: f32 = values.iter().sum();
        values.iter().map(|value| value / total).collect()
    }

    fn assert_tiled(spans: &[TokenFrameSpan], frame_count: usize) {
        assert!(spans.iter().all(|span| span.frame_start <= span.frame_end));
        assert_eq!(spans[0].frame_start, 0);
        assert_eq!(spans.last().unwrap().frame_end, frame_count);
        for window in spans.windows(2) {
            assert_eq!(window[0].frame_end, window[1].frame_start);
        }
    }

    #[test]
    fn empty_or_ragged_matrices_align_to_none() {
        assert!(dtw_align_token_frames(&[]).is_none());
        assert!(dtw_align_token_frames(&[vec![]]).is_none());
        assert!(dtw_align_token_frames(&[vec![1.0], vec![1.0, 1.0]]).is_none());
        assert!(dtw_align_token_frames(&[vec![f32::NAN]]).is_none());
        assert!(dtw_align_token_frames(&[vec![1.0, f32::INFINITY]]).is_none());
    }

    #[test]
    fn single_token_owns_the_whole_window() {
        let spans = dtw_align_token_frames(&[vec![0.25, 0.25, 0.25, 0.25]]).unwrap();
        assert_eq!(
            spans,
            vec![TokenFrameSpan {
                frame_start: 0,
                frame_end: 4
            }]
        );
    }

    #[test]
    fn tokens_trace_their_attention_monotonically() {
        let frames = 48;
        let attention: Vec<Vec<f32>> = (0..8).map(|token| bump(frames, token * 6)).collect();
        let spans = dtw_align_token_frames(&attention).unwrap();
        assert_tiled(&spans, frames);
        for (token, span) in spans.iter().enumerate() {
            let center = token * 6;
            assert!(
                span.frame_start <= center + 4,
                "token {token} starts too late: {span:?}"
            );
            assert!(
                span.frame_end >= center.saturating_sub(4),
                "token {token} ends too early: {span:?}"
            );
        }
    }

    #[test]
    fn overlapping_attention_keeps_order_and_tiles_the_window() {
        // All tokens pile their attention on the same region; the spans must
        // stay ordered and tile the window. The pile's center frame must land
        // inside at least one token's span so the timing tracks the signal.
        let frames = 12;
        let center = 6;
        let attention = vec![bump(frames, center); 5];
        let spans = dtw_align_token_frames(&attention).unwrap();
        assert_tiled(&spans, frames);
        assert!(
            spans
                .iter()
                .any(|span| span.frame_start <= center && center < span.frame_end),
            "no span covers the attention pile center {center}: {spans:?}"
        );
    }

    #[test]
    fn more_tokens_than_frames_still_tiles() {
        let frames = 3;
        let attention: Vec<Vec<f32>> = (0..8).map(|_| vec![0.2, 0.6, 0.2]).collect();
        let spans = dtw_align_token_frames(&attention).unwrap();
        assert_tiled(&spans, frames);
        assert_eq!(spans.len(), 8);
    }

    #[test]
    fn flat_attention_tiles_proportionally() {
        let frames = 8;
        let attention = vec![vec![0.125; frames]; 4];
        let spans = dtw_align_token_frames(&attention).unwrap();
        assert_tiled(&spans, frames);
    }

    /// A row that concentrates its attention on a single dominant `peak` frame
    /// (its argmax) with a diffuse floor elsewhere, like a real cross-attention
    /// row. `peak` must be strictly the argmax so the band tracks it exactly.
    fn attention_peaking_at(frames: usize, peak: usize) -> Vec<f32> {
        let mut row = vec![0.001_f32; frames];
        row[peak] = 1.0;
        let total: f32 = row.iter().sum();
        row.iter().map(|v| v / total).collect()
    }

    #[test]
    fn speech_bounds_bracket_the_token_peaks() {
        let frames = 120;
        // Two tokens peaking at frames 30 and 90: the band must cover both
        // peaks plus a small margin, and drop the leading/trailing padding.
        let attention = vec![
            attention_peaking_at(frames, 30),
            attention_peaking_at(frames, 90),
        ];
        let all_content = vec![true; attention.len()];
        let (start, end) = speech_frame_bounds(&attention, &all_content).unwrap();
        // Band = [first_peak - margin, last_peak + 1 + margin) = [20, 101).
        assert_eq!((start, end), (30 - 10, 90 + 1 + 10), "band {start}..{end}");
        assert!(start > 0, "band should drop the leading padding");
        assert!(end < frames, "band should drop the trailing padding");
    }

    #[test]
    fn speech_bounds_stop_before_the_ignored_tail() {
        // Mirrors the real clip: tokens attend only in the first third, then a
        // loud non-speech tail the model ignored (no per-token peak there). The
        // band must stop well before the tail rather than run to frame_count.
        let frames = 300;
        let attention = vec![
            attention_peaking_at(frames, 40),
            attention_peaking_at(frames, 90),
            attention_peaking_at(frames, 140),
        ];
        let all_content = vec![true; attention.len()];
        let (start, end) = speech_frame_bounds(&attention, &all_content).unwrap();
        assert!(
            end <= 160,
            "band bled toward the ignored tail: end={end} of {frames}"
        );
        assert_eq!(start, 40 - 10);
    }

    #[test]
    fn speech_bounds_ignores_non_content_tokens() {
        // A content token peaking mid-window plus a punctuation token whose only
        // peak sits deep in the ignored trailing region. The punctuation peak
        // must not stretch the band; the band brackets the content token alone.
        let frames = 300;
        let attention = vec![
            attention_peaking_at(frames, 40),  // " dog"
            attention_peaking_at(frames, 270), // " ." (attends into ignored music)
        ];
        let content = vec![true, false];
        let (_, end) = speech_frame_bounds(&attention, &content).unwrap();
        assert!(
            end < 300,
            "punctuation token stretched the band into the ignored tail: end={end}"
        );
        // Content-only band = [30, 51); the punctuation peak at 270 is ignored.
        assert_eq!(
            end,
            40 + 1 + 10,
            "band should bracket the content peak only"
        );
    }

    #[test]
    fn speech_bounds_all_zero_attention_is_none() {
        assert!(speech_frame_bounds(&[vec![0.0; 16], vec![0.0; 16]], &[true, true]).is_none());
        assert!(speech_frame_bounds(&[], &[]).is_none());
        assert!(speech_frame_bounds(&[Vec::new()], &[]).is_none());
    }

    #[test]
    fn speech_bounds_all_non_content_is_none() {
        // Every token filtered out (pure punctuation) -> no content to bracket
        // the band -> None, so the caller falls back to the duration window.
        let frames = 30;
        let attention = vec![
            attention_peaking_at(frames, 10),
            attention_peaking_at(frames, 20),
        ];
        let content = vec![false, false];
        assert!(speech_frame_bounds(&attention, &content).is_none());
    }

    #[test]
    fn speech_bounds_single_token_uses_margin() {
        // One token mid-window: the band is its peak plus the margin on either
        // side, clamped to the window.
        let frames = 200;
        let attention = vec![attention_peaking_at(frames, 100)];
        let content = vec![true];
        let (start, end) = speech_frame_bounds(&attention, &content).unwrap();
        assert_eq!((start, end), (90, 111));
    }
}
