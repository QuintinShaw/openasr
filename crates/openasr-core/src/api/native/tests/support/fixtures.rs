use super::super::NativeAsrModelPackRef;
use crate::RealtimeAudioFrame;

pub(in super::super) fn test_model_pack() -> NativeAsrModelPackRef {
    NativeAsrModelPackRef::new(
        super::TEST_ONLY_STREAMING_FIXTURE_ID,
        "test-only-native-streaming",
        "/tmp/openasr/test-only-native-streaming",
    )
}

pub(in super::super) fn test_frame(seq: u64, start_ms: u64) -> RealtimeAudioFrame {
    RealtimeAudioFrame::new(
        seq,
        start_ms,
        crate::RealtimeAudioFormat::pcm16_mono_16khz(),
        vec![0; 320],
    )
    .unwrap()
}

pub(super) fn test_time(index: u64) -> String {
    format!("2026-05-12T00:00:{index:02}Z")
}
