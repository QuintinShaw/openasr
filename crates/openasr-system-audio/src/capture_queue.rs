use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError},
};
use std::thread;
use std::time::{Duration, Instant};

use crate::pcm::{Pcm16FrameChunker, TARGET_FRAME_DURATION_MS};

/// 16 kHz mono 20 ms frames: 512 slots absorb ~10.24 s of consumer stall.
/// Nested caller queues (CLI live is 64) are independent and may still overflow.
pub(crate) const CAPTURE_QUEUE_FRAMES: usize = 512;

pub(crate) const TRAILING_CAPTURE_WINDOW: Duration = Duration::from_millis(200);
pub(crate) const CONSUMER_POLL: Duration = Duration::from_millis(100);

const DROP_DIAGNOSTIC_INTERVAL: Duration = Duration::from_secs(1);

pub(crate) struct CaptureQueueProducer {
    tx: Option<SyncSender<Vec<i16>>>,
    dropped: Arc<AtomicU64>,
    chunker: Pcm16FrameChunker,
}

pub(crate) struct CaptureQueueConsumer {
    rx: Receiver<Vec<i16>>,
    dropped: Arc<AtomicU64>,
    reported_drops: u64,
    last_drop_report: Option<Instant>,
}

pub(crate) fn capture_queue() -> (CaptureQueueProducer, CaptureQueueConsumer) {
    capture_queue_with_capacity(CAPTURE_QUEUE_FRAMES)
}

pub(crate) fn capture_queue_with_capacity(
    capacity: usize,
) -> (CaptureQueueProducer, CaptureQueueConsumer) {
    let (tx, rx) = mpsc::sync_channel(capacity.max(1));
    let dropped = Arc::new(AtomicU64::new(0));
    (
        CaptureQueueProducer {
            tx: Some(tx),
            dropped: Arc::clone(&dropped),
            chunker: Pcm16FrameChunker::new(),
        },
        CaptureQueueConsumer {
            rx,
            dropped,
            reported_drops: 0,
            last_drop_report: None,
        },
    )
}

impl CaptureQueueProducer {
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(crate) fn push_bytes(&mut self, bytes: &[u8]) {
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        let dropped = &self.dropped;
        let _ = self.chunker.push_bytes(bytes, |frame| {
            try_push_frame(tx, dropped, frame);
            Ok(())
        });
    }

    #[cfg_attr(not(any(windows, test)), allow(dead_code))]
    pub(crate) fn push_deque(&mut self, bytes: &mut std::collections::VecDeque<u8>) {
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        let dropped = &self.dropped;
        let _ = self.chunker.push_deque(bytes, |frame| {
            try_push_frame(tx, dropped, frame);
            Ok(())
        });
    }

    #[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
    pub(crate) fn push_samples(&mut self, samples: &[i16]) {
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        let dropped = &self.dropped;
        let _ = self.chunker.push_samples(samples, |frame| {
            try_push_frame(tx, dropped, frame);
            Ok(())
        });
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn try_push_frame(&self, frame: Vec<i16>) {
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        try_push_frame(tx, &self.dropped, frame);
    }

    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) fn drop_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.dropped)
    }

    /// After the device has stopped, `try_send` the padded tail. A blocking
    /// send would hang stop/join if `on_frame` is stuck and the queue is full.
    pub(crate) fn flush_padded(&mut self) {
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        let dropped = &self.dropped;
        let _ = self.chunker.flush_padded(|frame| {
            try_push_frame(tx, dropped, frame);
            Ok(())
        });
    }

    #[cfg(test)]
    pub(crate) fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl CaptureQueueConsumer {
    fn recv_timeout(&mut self, timeout: Duration) -> Result<Vec<i16>, RecvTimeoutError> {
        self.rx.recv_timeout(timeout)
    }

    pub(crate) fn drain(&mut self) -> Vec<Vec<i16>> {
        let mut frames = Vec::new();
        while let Ok(frame) = self.rx.try_recv() {
            frames.push(frame);
        }
        frames
    }

    fn take_drop_diagnostic(&mut self, force: bool) -> Option<String> {
        let total = self.dropped.load(Ordering::Relaxed);
        let unreported = total.saturating_sub(self.reported_drops);
        if unreported == 0 {
            return None;
        }
        let due = force
            || self.last_drop_report.is_none_or(|previous| {
                Instant::now().saturating_duration_since(previous) >= DROP_DIAGNOSTIC_INTERVAL
            });
        if !due {
            return None;
        }
        self.reported_drops = total;
        self.last_drop_report = Some(Instant::now());
        Some(format_drop_diagnostic(unreported))
    }
}

pub(crate) enum CaptureConsumerEvent<'a> {
    Frame(Vec<i16>),
    Diagnostic(&'a str),
}

pub(crate) fn run_capture_consumer(
    mut consumer: CaptureQueueConsumer,
    mut on_event: impl FnMut(CaptureConsumerEvent<'_>) -> Result<(), String>,
) -> Result<(), String> {
    loop {
        match consumer.recv_timeout(CONSUMER_POLL) {
            Ok(frame) => {
                emit_pending_drops(&mut consumer, false, &mut on_event)?;
                on_event(CaptureConsumerEvent::Frame(frame))?;
            }
            Err(RecvTimeoutError::Timeout) => {
                emit_pending_drops(&mut consumer, false, &mut on_event)?;
            }
            Err(RecvTimeoutError::Disconnected) => {
                for frame in consumer.drain() {
                    on_event(CaptureConsumerEvent::Frame(frame))?;
                }
                emit_pending_drops(&mut consumer, true, &mut on_event)?;
                return Ok(());
            }
        }
    }
}

#[cfg_attr(windows, allow(dead_code))]
pub(crate) fn forward_capture_events(
    mut on_frame: impl FnMut(Vec<i16>) -> Result<(), String>,
    mut on_diagnostic: impl FnMut(&str) -> Result<(), String>,
) -> impl FnMut(CaptureConsumerEvent<'_>) -> Result<(), String> {
    move |event| match event {
        CaptureConsumerEvent::Frame(frame) => on_frame(frame),
        CaptureConsumerEvent::Diagnostic(message) => on_diagnostic(message),
    }
}

fn emit_pending_drops(
    consumer: &mut CaptureQueueConsumer,
    force: bool,
    on_event: &mut impl FnMut(CaptureConsumerEvent<'_>) -> Result<(), String>,
) -> Result<(), String> {
    let Some(message) = consumer.take_drop_diagnostic(force) else {
        return Ok(());
    };
    on_event(CaptureConsumerEvent::Diagnostic(&message))
}

#[cfg_attr(windows, allow(dead_code))]
pub(crate) fn wait_for_stop_with_trailing(stop: &AtomicBool, mut abort: impl FnMut() -> bool) {
    while !stop.load(Ordering::SeqCst) && !abort() {
        thread::sleep(CONSUMER_POLL);
    }
    if stop.load(Ordering::SeqCst) && !abort() {
        thread::sleep(TRAILING_CAPTURE_WINDOW);
    }
}

pub(crate) fn join_capture_consumer(
    joined: thread::Result<Result<(), String>>,
) -> Result<(), String> {
    match joined {
        Ok(result) => result,
        Err(_) => Err("The capture consumer thread panicked.".to_string()),
    }
}

pub(crate) fn try_push_frame(tx: &SyncSender<Vec<i16>>, dropped: &AtomicU64, frame: Vec<i16>) {
    match tx.try_send(frame) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            dropped.fetch_add(1, Ordering::Relaxed);
        }
        Err(TrySendError::Disconnected(_)) => {}
    }
}

fn format_drop_diagnostic(dropped: u64) -> String {
    let dropped_ms = dropped.saturating_mul(TARGET_FRAME_DURATION_MS as u64);
    format!(
        "Dropped {dropped} system-audio frame(s) (~{dropped_ms} ms) because the consumer lagged; audio was discarded."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcm::TARGET_FRAME_SAMPLES;

    fn test_frame(fill: i16) -> Vec<i16> {
        vec![fill; TARGET_FRAME_SAMPLES]
    }

    fn collect_capture_events<'a>(
        frames: &'a mut Vec<Vec<i16>>,
        diagnostics: &'a mut Vec<String>,
    ) -> impl FnMut(CaptureConsumerEvent<'_>) -> Result<(), String> + 'a {
        move |event| {
            match event {
                CaptureConsumerEvent::Frame(frame) => frames.push(frame),
                CaptureConsumerEvent::Diagnostic(message) => diagnostics.push(message.to_string()),
            }
            Ok(())
        }
    }

    #[test]
    fn queue_holds_ten_seconds_of_twenty_ms_frames() {
        assert_eq!(CAPTURE_QUEUE_FRAMES, 512);
        assert_eq!(CAPTURE_QUEUE_FRAMES * TARGET_FRAME_DURATION_MS, 10_240);
    }

    #[test]
    fn deque_capture_preserves_split_samples_frames_and_tail() {
        let (mut producer, mut consumer) = capture_queue_with_capacity(4);
        let samples: Vec<i16> = (0..TARGET_FRAME_SAMPLES * 2 + 1)
            .map(|index| index as i16 - 400)
            .collect();
        let mut bytes: std::collections::VecDeque<u8> = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        let mut first_byte = std::collections::VecDeque::from([bytes.pop_front().unwrap()]);
        producer.push_deque(&mut first_byte);
        assert!(first_byte.is_empty());
        assert!(consumer.drain().is_empty());

        producer.push_deque(&mut bytes);
        assert!(bytes.is_empty());
        let frames = consumer.drain();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], samples[..TARGET_FRAME_SAMPLES]);
        assert_eq!(
            frames[1],
            samples[TARGET_FRAME_SAMPLES..TARGET_FRAME_SAMPLES * 2]
        );

        producer.flush_padded();
        let tail = consumer.drain();
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].len(), TARGET_FRAME_SAMPLES);
        assert_eq!(tail[0][0], samples[TARGET_FRAME_SAMPLES * 2]);
        assert!(tail[0][1..].iter().all(|sample| *sample == 0));
        assert_eq!(producer.dropped_count(), 0);
    }

    #[test]
    fn full_queue_increments_drop_count_without_evicting() {
        let (producer, mut consumer) = capture_queue_with_capacity(2);
        producer.try_push_frame(test_frame(1));
        producer.try_push_frame(test_frame(2));
        producer.try_push_frame(test_frame(3));

        assert_eq!(producer.dropped_count(), 1);
        let frames = consumer.drain();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0][0], 1);
        assert_eq!(frames[1][0], 2);
    }

    #[test]
    fn drain_delivers_remaining_frames() {
        let (producer, mut consumer) = capture_queue_with_capacity(4);
        producer.try_push_frame(test_frame(4));
        producer.try_push_frame(test_frame(5));
        producer.try_push_frame(test_frame(6));

        let frames = consumer.drain();
        assert_eq!(
            frames.iter().map(|frame| frame[0]).collect::<Vec<_>>(),
            vec![4, 5, 6]
        );
        assert!(consumer.drain().is_empty());
    }

    #[test]
    fn stop_flushes_chunker_and_drains_tail() {
        let (mut producer, consumer) = capture_queue_with_capacity(8);
        let mut samples = vec![7_i16; TARGET_FRAME_SAMPLES];
        samples.extend(std::iter::repeat_n(9_i16, 40));
        producer.push_samples(&samples);
        producer.flush_padded();
        drop(producer);

        let mut frames = Vec::new();
        let mut diagnostics = Vec::new();
        run_capture_consumer(
            consumer,
            collect_capture_events(&mut frames, &mut diagnostics),
        )
        .expect("consumer should drain the tail");

        assert_eq!(frames.len(), 2);
        assert!(
            frames
                .iter()
                .all(|frame| frame.len() == TARGET_FRAME_SAMPLES)
        );
        assert_eq!(frames[0][0], 7);
        assert_eq!(frames[1][0], 9);
        assert!(frames[1][40..].iter().all(|sample| *sample == 0));
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn flush_on_full_queue_counts_drop_and_keeps_queued_frames() {
        let (mut producer, mut consumer) = capture_queue_with_capacity(1);
        producer.try_push_frame(test_frame(1));
        producer.push_samples(&[9]);
        producer.flush_padded();

        assert_eq!(producer.dropped_count(), 1);
        let frames = consumer.drain();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0][0], 1);
    }

    #[test]
    fn consumer_reports_drops_instead_of_staying_silent() {
        let (producer, consumer) = capture_queue_with_capacity(1);
        producer.try_push_frame(test_frame(1));
        producer.try_push_frame(test_frame(2));
        drop(producer);

        let mut frames = Vec::new();
        let mut diagnostics = Vec::new();
        run_capture_consumer(
            consumer,
            collect_capture_events(&mut frames, &mut diagnostics),
        )
        .expect("consumer should finish after disconnect");

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0][0], 1);
        assert_eq!(diagnostics.len(), 1);
        assert!(
            diagnostics[0].contains("Dropped 1 system-audio frame(s)"),
            "diagnostic was {}",
            diagnostics[0]
        );
    }
}
