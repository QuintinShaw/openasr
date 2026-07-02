use std::collections::VecDeque;

pub(super) const TARGET_SAMPLE_RATE_HZ: usize = 16_000;
#[allow(dead_code)]
pub(super) const TARGET_CHANNELS: usize = 1;
pub(super) const TARGET_FRAME_DURATION_MS: usize = 20;
pub(super) const TARGET_FRAME_SAMPLES: usize =
    TARGET_SAMPLE_RATE_HZ * TARGET_FRAME_DURATION_MS / 1_000;

pub(super) struct Pcm16FrameChunker {
    pending: VecDeque<u8>,
}

impl Pcm16FrameChunker {
    pub(super) fn new() -> Self {
        Self {
            pending: VecDeque::with_capacity(TARGET_FRAME_SAMPLES * 2 * 4),
        }
    }

    #[allow(dead_code)]
    pub(super) fn push_bytes(
        &mut self,
        bytes: &[u8],
        mut on_frame: impl FnMut(Vec<i16>) -> Result<(), String>,
    ) -> Result<(), String> {
        self.pending.extend(bytes.iter().copied());
        self.emit_complete_frames(&mut on_frame)
    }

    // Used by the macOS CoreAudio backend (macos.rs) and the unit test; not part of
    // the Linux/Windows lib build graph, so allow dead_code only there.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(super) fn push_samples(
        &mut self,
        samples: &[i16],
        mut on_frame: impl FnMut(Vec<i16>) -> Result<(), String>,
    ) -> Result<(), String> {
        for sample in samples {
            self.pending.extend(sample.to_le_bytes());
        }
        self.emit_complete_frames(&mut on_frame)
    }

    fn emit_complete_frames(
        &mut self,
        on_frame: &mut impl FnMut(Vec<i16>) -> Result<(), String>,
    ) -> Result<(), String> {
        while self.pending.len() >= TARGET_FRAME_SAMPLES * 2 {
            on_frame(self.pop_frame(false))?;
        }
        Ok(())
    }

    pub(super) fn flush_padded(
        &mut self,
        mut on_frame: impl FnMut(Vec<i16>) -> Result<(), String>,
    ) -> Result<(), String> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let frame = self.pop_frame(true);
        on_frame(frame)
    }

    fn pop_frame(&mut self, pad: bool) -> Vec<i16> {
        let mut frame = Vec::with_capacity(TARGET_FRAME_SAMPLES);
        while frame.len() < TARGET_FRAME_SAMPLES {
            let Some(lo) = self.pending.pop_front() else {
                if pad {
                    frame.push(0);
                    continue;
                }
                break;
            };
            let hi = self.pending.pop_front().unwrap_or(0);
            frame.push(i16::from_le_bytes([lo, hi]));
        }
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_complete_twenty_ms_frames() {
        let mut chunker = Pcm16FrameChunker::new();
        let mut bytes = Vec::new();
        for value in 0..TARGET_FRAME_SAMPLES {
            bytes.extend_from_slice(&(value as i16).to_le_bytes());
        }
        let mut frames = Vec::new();

        chunker
            .push_bytes(&bytes, |frame| {
                frames.push(frame);
                Ok(())
            })
            .expect("push bytes");

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].len(), TARGET_FRAME_SAMPLES);
        assert_eq!(frames[0][0], 0);
        assert_eq!(
            frames[0][TARGET_FRAME_SAMPLES - 1],
            (TARGET_FRAME_SAMPLES - 1) as i16
        );
    }

    #[test]
    fn flush_pads_partial_frame() {
        let mut chunker = Pcm16FrameChunker::new();
        let mut frames = Vec::new();

        chunker
            .push_bytes(&42_i16.to_le_bytes(), |_| {
                panic!("partial frame should not emit before flush")
            })
            .expect("push partial bytes");
        chunker
            .flush_padded(|frame| {
                frames.push(frame);
                Ok(())
            })
            .expect("flush padded");

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].len(), TARGET_FRAME_SAMPLES);
        assert_eq!(frames[0][0], 42);
        assert!(frames[0][1..].iter().all(|sample| *sample == 0));
    }

    #[test]
    fn push_samples_uses_same_frame_boundaries() {
        let mut chunker = Pcm16FrameChunker::new();
        let samples = (0..TARGET_FRAME_SAMPLES)
            .map(|value| value as i16)
            .collect::<Vec<_>>();
        let mut frames = Vec::new();

        chunker
            .push_samples(&samples, |frame| {
                frames.push(frame);
                Ok(())
            })
            .expect("push samples");

        assert_eq!(frames, vec![samples]);
    }
}
