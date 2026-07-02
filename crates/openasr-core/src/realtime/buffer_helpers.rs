use std::collections::VecDeque;

use super::{
    ActiveUtterance, BufferedUtterance, RealtimeAudioFrame, RealtimeBufferConfig,
    RealtimeBufferError, RealtimeUtteranceEndReason, TranscriptUtteranceId,
};

pub(super) fn frame_duration_ms(frame: &RealtimeAudioFrame) -> u64 {
    u64::from(frame.duration_ms().expect("validated realtime frame"))
}

pub(super) fn buffered_samples(frames: &[RealtimeAudioFrame]) -> usize {
    frames.iter().map(RealtimeAudioFrame::sample_count).sum()
}

pub(super) fn ensure_capacity(
    config: RealtimeBufferConfig,
    frames: &[RealtimeAudioFrame],
) -> Result<(), RealtimeBufferError> {
    if frames.len() > config.max_buffered_frames {
        return Err(RealtimeBufferError::AudioBufferOverflow {
            buffered_frames: frames.len(),
            max_buffered_frames: config.max_buffered_frames,
            buffered_samples: buffered_samples(frames),
            max_buffered_samples: config.max_buffered_samples,
        });
    }

    let samples = buffered_samples(frames);
    if samples > config.max_buffered_samples {
        return Err(RealtimeBufferError::AudioBufferOverflow {
            buffered_frames: frames.len(),
            max_buffered_frames: config.max_buffered_frames,
            buffered_samples: samples,
            max_buffered_samples: config.max_buffered_samples,
        });
    }

    Ok(())
}

pub(super) fn trim_pre_roll(
    mut candidate: VecDeque<RealtimeAudioFrame>,
    pre_roll_ms: u32,
) -> VecDeque<RealtimeAudioFrame> {
    let target_ms = u64::from(pre_roll_ms);
    if target_ms == 0 {
        candidate.clear();
        return candidate;
    }

    let mut total_ms: u64 = candidate.iter().map(frame_duration_ms).sum();
    while let Some(front_duration_ms) = candidate.front().map(frame_duration_ms) {
        if total_ms.saturating_sub(front_duration_ms) < target_ms {
            break;
        }
        total_ms = total_ms.saturating_sub(front_duration_ms);
        candidate.pop_front();
    }
    candidate
}

pub(super) fn limited_active_frames(
    active: &ActiveUtterance,
    max_duration_ms: Option<u32>,
) -> Option<Vec<RealtimeAudioFrame>> {
    let Some(limit_ms) = max_duration_ms else {
        return Some(active.frames.clone());
    };
    if limit_ms == 0 {
        return Some(active.frames.clone());
    }

    let mut total = 0u64;
    let mut trimmed = VecDeque::new();
    for frame in active.frames.iter().rev() {
        let duration = frame.duration_ms().ok()? as u64;
        if total + duration > u64::from(limit_ms) && !trimmed.is_empty() {
            break;
        }
        total += duration;
        trimmed.push_front(frame.clone());
    }
    Some(trimmed.into_iter().collect())
}

pub(super) fn finish_utterance(
    active: ActiveUtterance,
    expected_utterance_id: &TranscriptUtteranceId,
    end_ms: u64,
    reason: RealtimeUtteranceEndReason,
) -> Result<BufferedUtterance, ActiveUtterance> {
    if &active.utterance_id != expected_utterance_id {
        return Err(active);
    }

    let (start_ms, _) = frame_time_range(&active.frames, active.start_ms);
    Ok(BufferedUtterance {
        utterance_id: active.utterance_id,
        start_ms,
        end_ms,
        frames: active.frames,
        reason,
    })
}

pub(super) fn frame_time_range(
    frames: &[RealtimeAudioFrame],
    fallback_start_ms: u64,
) -> (u64, u64) {
    let start_ms = frames
        .first()
        .map(|frame| frame.start_ms)
        .unwrap_or(fallback_start_ms);
    let end_ms = frames
        .last()
        .map(RealtimeAudioFrame::end_ms)
        .unwrap_or(start_ms);
    (start_ms, end_ms)
}
