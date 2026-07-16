// Platform-agnostic realtime microphone **capture engine**: turns whatever
// raw interleaved PCM a platform's audio API hands back (variable sample
// rate, channel count, sample depth) into the canonical 16 kHz mono
// `RealtimeAudioFrame` stream the rest of `realtime` (VAD, buffering,
// transcription) already speaks.
//
// This module intentionally contains **zero OS audio APIs** (no cpal, no
// AVAudioSession, no `AudioRecord`). Platform glue -- the desktop CLI's cpal
// stream, and the closed mobile shell's `AVAudioSession`/`AudioRecord`
// bindings -- owns permission prompts, device selection, and the actual
// hardware callback; it hands raw sample chunks to `CaptureEngine` (via
// `CaptureBackpressureQueue` if the callback runs on a real-time audio
// thread that cannot block) and receives back frames it can feed directly
// into `crate::realtime::VadStateMachine` / `crate::realtime::RealtimeBuffer`
// (or the higher-level `crate::realtime::RealtimeSessionController`) with
// no further adaptation. See `capture/tests.rs` for a worked end-to-end
// example of that hand-off.
//
// Desktop's `openasr-cli` live-mic path (`crates/openasr-cli/src/live.rs`)
// is a thin cpal-specific wrapper around this same engine, so the resample /
// downmix / framing logic has exactly one implementation.

use std::collections::VecDeque;
use std::sync::Mutex;

use thiserror::Error;

use super::audio::{
    DEFAULT_REALTIME_SAMPLE_RATE_HZ, RealtimeAudioFormat, RealtimeAudioFrame, RealtimeFrameError,
};

/// Describes the raw audio a platform's capture API is actually delivering.
/// Unlike [`RealtimeAudioFormat`] (which is always normalized 16 kHz mono),
/// this is whatever the hardware/OS negotiated -- e.g. 48 kHz stereo f32 on
/// macOS, or 44.1 kHz mono i16 on an Android device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureInputFormat {
    pub sample_rate_hz: u32,
    pub channels: u16,
}

impl CaptureInputFormat {
    pub fn new(sample_rate_hz: u32, channels: u16) -> Result<Self, CaptureEngineError> {
        if sample_rate_hz == 0 {
            return Err(CaptureEngineError::ZeroSampleRate);
        }
        if channels == 0 {
            return Err(CaptureEngineError::ZeroChannels);
        }
        Ok(Self {
            sample_rate_hz,
            channels,
        })
    }
}

/// One raw interleaved chunk handed to the engine by platform glue, in
/// whatever sample representation the platform's audio callback natively
/// produces. Covers the sample formats OpenASR's own capture glue (desktop
/// cpal today; iOS/Android AVAudioSession/AudioRecord shells later) actually
/// hits in practice.
#[derive(Debug, Clone, PartialEq)]
pub enum CaptureSample {
    F32(Vec<f32>),
    I16(Vec<i16>),
    U16(Vec<u16>),
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CaptureEngineError {
    #[error("Capture input sample rate must be greater than 0.")]
    ZeroSampleRate,
    #[error("Capture input channel count must be greater than 0.")]
    ZeroChannels,
    #[error(
        "Captured interleaved audio chunk had {sample_count} samples, which is not divisible by {channels} channel(s)."
    )]
    NonMultipleOfChannels { sample_count: usize, channels: u16 },
    #[error(transparent)]
    Frame(#[from] RealtimeFrameError),
    #[error("Capture backpressure queue capacity must be greater than 0.")]
    ZeroQueueCapacity,
}

/// Resamples (linear interpolation, streaming-friendly -- no lookahead
/// buffering beyond what a single output sample needs), downmixes, and
/// frames raw platform PCM into normalized [`RealtimeAudioFrame`]s.
///
/// Linear interpolation (rather than the FFT resampler `crate::audio` uses
/// for one-shot file decoding) is deliberate here: it processes samples as
/// they arrive with no chunking latency, which matters for a live mic path.
#[derive(Debug)]
pub struct CaptureEngine {
    input: CaptureInputFormat,
    frame_duration_ms: u32,
    frame_sample_count: usize,
    pending_output: Vec<i16>,
    resample_input: Vec<f32>,
    resample_pos: f64,
    resample_step: f64,
    next_seq: u64,
    next_start_ms: u64,
}

impl CaptureEngine {
    pub fn new(
        input: CaptureInputFormat,
        frame_duration_ms: u32,
    ) -> Result<Self, CaptureEngineError> {
        // Re-validate even though `CaptureInputFormat::new` already checks
        // this: the struct's fields are public (platform glue often builds
        // it from a device-reported config struct directly), so a caller
        // bypassing the constructor must still fail closed here.
        if input.sample_rate_hz == 0 {
            return Err(CaptureEngineError::ZeroSampleRate);
        }
        if input.channels == 0 {
            return Err(CaptureEngineError::ZeroChannels);
        }
        let frame_sample_count =
            RealtimeAudioFormat::pcm16_mono_16khz().sample_count_for_duration_ms(frame_duration_ms)?;
        Ok(Self {
            input,
            frame_duration_ms,
            frame_sample_count,
            pending_output: Vec::new(),
            resample_input: Vec::new(),
            resample_pos: 0.0,
            resample_step: input.sample_rate_hz as f64 / DEFAULT_REALTIME_SAMPLE_RATE_HZ as f64,
            next_seq: 1,
            next_start_ms: 0,
        })
    }

    pub fn input_format(&self) -> CaptureInputFormat {
        self.input
    }

    pub fn frame_duration_ms(&self) -> u32 {
        self.frame_duration_ms
    }

    /// The `start_ms` timestamp the next emitted frame will carry -- useful
    /// for computing a flush/shutdown timestamp once capture stops.
    pub fn next_frame_start_ms(&self) -> u64 {
        self.next_start_ms
    }

    /// Push one raw interleaved chunk, dispatching on its sample
    /// representation. Convenience entry point for glue that already has a
    /// [`CaptureSample`] (e.g. drained from a [`CaptureBackpressureQueue`]).
    pub fn push(&mut self, sample: CaptureSample) -> Result<Vec<RealtimeAudioFrame>, CaptureEngineError> {
        match sample {
            CaptureSample::F32(samples) => self.push_f32_interleaved(&samples),
            CaptureSample::I16(samples) => self.push_i16_interleaved(&samples),
            CaptureSample::U16(samples) => self.push_u16_interleaved(&samples),
        }
    }

    pub fn push_f32_interleaved(
        &mut self,
        samples: &[f32],
    ) -> Result<Vec<RealtimeAudioFrame>, CaptureEngineError> {
        let mono = downmix_interleaved(samples, self.input.channels, |sample| {
            sample.clamp(-1.0, 1.0)
        })?;
        self.push_mono_f32(&mono)
    }

    pub fn push_i16_interleaved(
        &mut self,
        samples: &[i16],
    ) -> Result<Vec<RealtimeAudioFrame>, CaptureEngineError> {
        let mono = downmix_interleaved(samples, self.input.channels, |sample| {
            sample as f32 / 32768.0
        })?;
        self.push_mono_f32(&mono)
    }

    pub fn push_u16_interleaved(
        &mut self,
        samples: &[u16],
    ) -> Result<Vec<RealtimeAudioFrame>, CaptureEngineError> {
        let mono = downmix_interleaved(samples, self.input.channels, |sample| {
            (sample as f32 - 32768.0) / 32768.0
        })?;
        self.push_mono_f32(&mono)
    }

    fn push_mono_f32(&mut self, mono: &[f32]) -> Result<Vec<RealtimeAudioFrame>, CaptureEngineError> {
        let mut pcm16 = if self.input.sample_rate_hz == DEFAULT_REALTIME_SAMPLE_RATE_HZ {
            mono.iter().map(|sample| f32_to_i16(*sample)).collect()
        } else {
            self.resample_to_pcm16(mono)
        };
        self.pending_output.append(&mut pcm16);
        self.drain_frames()
    }

    fn resample_to_pcm16(&mut self, mono: &[f32]) -> Vec<i16> {
        self.resample_input.extend_from_slice(mono);
        let mut output = Vec::new();
        while self.resample_pos + 1.0 < self.resample_input.len() as f64 {
            let index = self.resample_pos.floor() as usize;
            let fraction = (self.resample_pos - index as f64) as f32;
            let a = self.resample_input[index];
            let b = self.resample_input[index + 1];
            output.push(f32_to_i16(a + (b - a) * fraction));
            self.resample_pos += self.resample_step;
        }
        let consumed = self.resample_pos.floor() as usize;
        if consumed > 0 {
            self.resample_input.drain(0..consumed);
            self.resample_pos -= consumed as f64;
        }
        output
    }

    fn drain_frames(&mut self) -> Result<Vec<RealtimeAudioFrame>, CaptureEngineError> {
        let mut frames = Vec::new();
        while self.pending_output.len() >= self.frame_sample_count {
            let samples = self
                .pending_output
                .drain(0..self.frame_sample_count)
                .collect::<Vec<_>>();
            let frame = RealtimeAudioFrame::new(
                self.next_seq,
                self.next_start_ms,
                RealtimeAudioFormat::pcm16_mono_16khz(),
                samples,
            )?;
            self.next_seq += 1;
            self.next_start_ms += u64::from(self.frame_duration_ms);
            frames.push(frame);
        }
        Ok(frames)
    }
}

fn downmix_interleaved<T: Copy>(
    samples: &[T],
    channels: u16,
    convert: impl Fn(T) -> f32,
) -> Result<Vec<f32>, CaptureEngineError> {
    let channel_count = channels as usize;
    if channel_count == 0 || !samples.len().is_multiple_of(channel_count) {
        return Err(CaptureEngineError::NonMultipleOfChannels {
            sample_count: samples.len(),
            channels,
        });
    }
    Ok(samples
        .chunks_exact(channel_count)
        .map(|frame| frame.iter().copied().map(&convert).sum::<f32>() / channel_count as f32)
        .collect())
}

fn f32_to_i16(sample: f32) -> i16 {
    let clamped = sample.clamp(-1.0, 1.0);
    if clamped >= 0.0 {
        (clamped * i16::MAX as f32).round() as i16
    } else {
        (clamped * 32768.0).round() as i16
    }
}

/// Outcome of a single [`CaptureBackpressureQueue::try_push`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapturePushOutcome {
    Accepted,
    /// The queue was already at capacity; the incoming item was **dropped**,
    /// not the oldest queued item -- already-buffered audio is never
    /// silently evicted to make room for new audio. The queue's sticky
    /// overflow flag is set; callers (mirroring the desktop CLI live path)
    /// are expected to treat overflow as fail-closed -- stop the session
    /// instead of silently losing audio -- rather than ignore it.
    Overflowed,
}

struct CaptureQueueState<T> {
    items: VecDeque<T>,
    overflowed: bool,
}

/// Bounded, thread-safe, non-blocking hand-off queue for the boundary
/// between an OS audio callback (which must never block or allocate
/// unpredictably) and whatever thread/task drains it into a
/// [`CaptureEngine`]. `try_push` never blocks and never evicts existing
/// data: once full, new pushes are dropped and flagged via a sticky
/// overflow bit, mirroring the drop-and-flag backpressure policy the
/// desktop CLI already implements by hand with `mpsc::sync_channel` +
/// `AtomicBool` (see `crates/openasr-cli/src/live.rs`).
pub struct CaptureBackpressureQueue<T> {
    capacity: usize,
    state: Mutex<CaptureQueueState<T>>,
}

// Manual impl (rather than `#[derive(Debug)]`) so callers aren't forced to
// make their queued item type `T: Debug` just to debug-print the queue
// itself; only the capacity/length are ever interesting here.
impl<T> std::fmt::Debug for CaptureBackpressureQueue<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CaptureBackpressureQueue")
            .field("capacity", &self.capacity)
            .field("len", &self.len())
            .finish()
    }
}

impl<T> CaptureBackpressureQueue<T> {
    pub fn new(capacity: usize) -> Result<Self, CaptureEngineError> {
        if capacity == 0 {
            return Err(CaptureEngineError::ZeroQueueCapacity);
        }
        Ok(Self {
            capacity,
            state: Mutex::new(CaptureQueueState {
                items: VecDeque::with_capacity(capacity),
                overflowed: false,
            }),
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Non-blocking push. Never panics on a poisoned lock caller path
    /// beyond what any other `Mutex` use in this codebase already assumes;
    /// a panicking producer/consumer is an existing-invariant violation
    /// elsewhere, not something this queue needs to newly guard against.
    pub fn try_push(&self, item: T) -> CapturePushOutcome {
        let mut state = self.state.lock().unwrap_or_else(|poison| poison.into_inner());
        if state.items.len() >= self.capacity {
            state.overflowed = true;
            return CapturePushOutcome::Overflowed;
        }
        state.items.push_back(item);
        CapturePushOutcome::Accepted
    }

    /// Drain every currently-queued item, in FIFO order.
    pub fn drain(&self) -> Vec<T> {
        let mut state = self.state.lock().unwrap_or_else(|poison| poison.into_inner());
        state.items.drain(..).collect()
    }

    /// Read-and-clear the sticky overflow flag (matches
    /// `AtomicBool::swap(false, Ordering::SeqCst)` semantics).
    pub fn take_overflowed(&self) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|poison| poison.into_inner());
        std::mem::take(&mut state.overflowed)
    }

    pub fn len(&self) -> usize {
        let state = self.state.lock().unwrap_or_else(|poison| poison.into_inner());
        state.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
