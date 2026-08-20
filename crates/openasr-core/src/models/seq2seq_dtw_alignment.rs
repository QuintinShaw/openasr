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
}
