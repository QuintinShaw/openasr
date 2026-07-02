use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use wasapi::{
    AudioClient, DeviceEnumerator, Direction, SampleType, StreamMode, WasapiError, WaveFormat,
};

use crate::{
    CaptureBackendError, SystemAudioSupport,
    pcm::{Pcm16FrameChunker, TARGET_CHANNELS, TARGET_FRAME_SAMPLES, TARGET_SAMPLE_RATE_HZ},
};

const SILENT_STREAK_DIAGNOSTIC_FRAMES: u32 = 250;

pub fn support_status() -> SystemAudioSupport {
    SystemAudioSupport {
        supported: true,
        label: "System audio (Windows smoke)".to_string(),
        detail: "Windows WASAPI all-system loopback smoke path is available. Per-process loopback remains deferred."
            .to_string(),
        platform: "windows".to_string(),
    }
}

pub fn run_loopback_capture(
    stop: Arc<AtomicBool>,
    mut on_frame: impl FnMut(Vec<i16>) -> Result<(), String>,
    mut on_diagnostic: impl FnMut(&str) -> Result<(), String>,
) -> Result<String, CaptureBackendError> {
    wasapi::initialize_mta()
        .ok()
        .map_err(|error| CaptureBackendError {
            code: "capture_backend_failed",
            message: "Could not initialize COM for WASAPI".to_string(),
            diagnostic: error.to_string(),
        })?;

    // Pair the CoInitializeEx above with CoUninitialize on EVERY exit path. Each
    // `?` below would otherwise skip teardown and leave COM initialized with an
    // unbalanced refcount; the guard also runs last (after the IAudioClient /
    // capture-client COM pointers drop), fixing the prior ordering where
    // deinitialize() ran while those interfaces were still alive. Mirrors the
    // RAII the macOS arm uses.
    let _com = ComUninitGuard;

    let enumerator = DeviceEnumerator::new().map_err(map_wasapi_error(
        "capture_backend_failed",
        "Could not create WASAPI device enumerator",
    ))?;
    let device = enumerator
        .get_default_device(&Direction::Render)
        .map_err(map_wasapi_error(
            "no_default_render_endpoint",
            "No default render endpoint is available",
        ))?;

    let mut client: AudioClient = device.get_iaudioclient().map_err(map_wasapi_error(
        "capture_backend_failed",
        "Could not acquire IAudioClient",
    ))?;

    let desired = WaveFormat::new(
        16,
        16,
        &SampleType::Int,
        TARGET_SAMPLE_RATE_HZ,
        TARGET_CHANNELS,
        None,
    );

    let (_default_period_hns, min_period_hns) =
        client.get_device_period().map_err(map_wasapi_error(
            "capture_backend_failed",
            "Could not read WASAPI device period",
        ))?;

    let stream_mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: if min_period_hns > 0 {
            min_period_hns
        } else {
            200_000
        },
    };

    client
        .initialize_client(&desired, &Direction::Capture, &stream_mode)
        .map_err(map_wasapi_error(
            "format_unsupported",
            "Could not initialize WASAPI loopback capture in PCM16 mono 16k mode",
        ))?;

    let capture = client.get_audiocaptureclient().map_err(map_wasapi_error(
        "capture_backend_failed",
        "Could not get WASAPI capture client",
    ))?;
    let event = client.set_get_eventhandle().map_err(map_wasapi_error(
        "capture_backend_failed",
        "Could not create WASAPI event handle",
    ))?;

    // The WASAPI capture is forced to PCM16 mono 16k (autoconvert), so each
    // frame is exactly two bytes. The shared Pcm16FrameChunker assumes that
    // layout, so require it explicitly rather than silently mis-framing.
    let block_align = desired.get_blockalign() as usize;
    if block_align != 2 {
        return Err(CaptureBackendError {
            code: "format_unsupported",
            message: "WASAPI did not provide a PCM16 mono frame layout.".to_string(),
            diagnostic: format!(
                "Expected a 2-byte (16-bit mono) frame block alignment, got {block_align}."
            ),
        });
    }

    let mut queue = VecDeque::with_capacity(block_align * TARGET_FRAME_SAMPLES * 64);
    let mut chunker = Pcm16FrameChunker::new();
    let mut silent_frame_streak: u32 = 0;
    let mut waiting_for_playback_noted = false;

    client.start_stream().map_err(map_wasapi_error(
        "capture_backend_failed",
        "Could not start WASAPI loopback stream",
    ))?;

    while !stop.load(Ordering::SeqCst) {
        let info = capture
            .read_from_device_to_deque(&mut queue)
            .map_err(map_wasapi_error(
                "capture_backend_failed",
                "Could not read WASAPI loopback buffer",
            ))?;

        if info.flags.data_discontinuity {
            // A shared-mode discontinuity (glitch / format renegotiation) is
            // routine and recoverable: note it and keep capturing instead of
            // tearing down the whole session.
            silent_frame_streak = 0;
            waiting_for_playback_noted = false;
            on_diagnostic(
                "Render endpoint reported an audio buffer discontinuity; continuing capture.",
            )
            .map_err(|error| CaptureBackendError {
                code: "capture_backend_failed",
                message: "Could not emit system-audio diagnostic to desktop frontend.".to_string(),
                diagnostic: error,
            })?;
        }

        let mut pending: Vec<u8> = queue.drain(..).collect();
        if info.flags.silent {
            // AUDCLNT_BUFFERFLAGS_SILENT: the packet's buffer contents are
            // undefined and must be treated as silence (the memory is not
            // guaranteed to be zeroed). Zero the drained bytes so undefined data
            // never reaches the ASR pipeline and so is_silent_frame() correctly
            // advances the silence streak instead of seeing garbage.
            pending.iter_mut().for_each(|byte| *byte = 0);
        }
        chunker
            .push_bytes(&pending, |frame| {
                if is_silent_frame(&frame) {
                    silent_frame_streak = silent_frame_streak.saturating_add(1);
                    if silent_frame_streak >= SILENT_STREAK_DIAGNOSTIC_FRAMES
                        && !waiting_for_playback_noted
                    {
                        on_diagnostic(
                            "No active render stream detected yet. Capture remains armed; start local playback to stream system audio.",
                        )?;
                        waiting_for_playback_noted = true;
                    }
                } else {
                    silent_frame_streak = 0;
                    waiting_for_playback_noted = false;
                }
                on_frame(frame)
            })
            .map_err(|error| CaptureBackendError {
                code: "capture_backend_failed",
                message: "Could not emit system-audio frame to desktop frontend.".to_string(),
                diagnostic: error,
            })?;

        if let Err(wait_error) = event.wait_for_event(200) {
            if stop.load(Ordering::SeqCst) {
                break;
            }
            if !matches!(wait_error, WasapiError::EventTimeout) {
                return Err(CaptureBackendError {
                    code: "capture_backend_failed",
                    message: "WASAPI event wait failed during loopback capture.".to_string(),
                    diagnostic: wait_error.to_string(),
                });
            }
        }
    }

    chunker
        .flush_padded(on_frame)
        .map_err(|error| CaptureBackendError {
            code: "capture_backend_failed",
            message: "Could not emit final padded system-audio frame.".to_string(),
            diagnostic: error,
        })?;

    let _ = client.stop_stream();

    Ok("Capture stopped".to_string())
}

/// RAII guard that calls `wasapi::deinitialize()` (CoUninitialize) on drop, so
/// COM is always balanced however `run_loopback_capture` returns.
struct ComUninitGuard;

impl Drop for ComUninitGuard {
    fn drop(&mut self) {
        wasapi::deinitialize();
    }
}

fn is_silent_frame(samples: &[i16]) -> bool {
    samples.iter().all(|sample| sample.unsigned_abs() <= 64)
}

fn map_wasapi_error(
    code: &'static str,
    message: &'static str,
) -> impl FnOnce(WasapiError) -> CaptureBackendError {
    move |error| CaptureBackendError {
        code,
        message: message.to_string(),
        diagnostic: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_silence_with_threshold() {
        assert!(is_silent_frame(&[0, 1, -2, 63]));
        assert!(!is_silent_frame(&[0, 70, 0]));
    }

    #[test]
    #[ignore = "requires a Windows render endpoint and local playback"]
    fn windows_wasapi_system_audio_smoke_emits_non_silent_frames() {
        use super::super::smoke_test_support::{
            MIN_SMOKE_FRAMES, NON_SILENT_PEAK_THRESHOLD, frame_peak, write_smoke_wav,
        };
        use std::path::Path;
        use std::process::{Child, Command};
        use std::thread;
        use std::time::Duration;

        let support = support_status();
        assert!(
            support.supported,
            "system audio support should be available for smoke: {}",
            support.detail
        );

        let playback_path = write_smoke_wav("openasr-windows-system-audio-smoke");
        let mut playback = spawn_windows_playback(&playback_path);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_after_timeout = Arc::clone(&stop);
        let stopper = thread::spawn(move || {
            thread::sleep(Duration::from_secs(5));
            stop_after_timeout.store(true, Ordering::SeqCst);
        });

        let mut frames = 0_usize;
        let mut peak = 0_i32;
        let stop_after_signal = Arc::clone(&stop);
        let result = run_loopback_capture(
            Arc::clone(&stop),
            |samples| {
                frames += 1;
                peak = peak.max(frame_peak(&samples));
                if frames >= MIN_SMOKE_FRAMES && peak > NON_SILENT_PEAK_THRESHOLD {
                    stop_after_signal.store(true, Ordering::SeqCst);
                }
                Ok(())
            },
            |message| {
                eprintln!("{message}");
                Ok(())
            },
        );

        stop.store(true, Ordering::SeqCst);
        let _ = playback.kill();
        let _ = playback.wait();
        let _ = stopper.join();
        let _ = std::fs::remove_file(&playback_path);

        result.expect("Windows WASAPI system-audio capture should run");
        eprintln!("Windows system-audio smoke captured {frames} frames, peak {peak}");
        assert!(
            frames >= MIN_SMOKE_FRAMES,
            "expected at least {MIN_SMOKE_FRAMES} frames, got {frames}"
        );
        assert!(
            peak > NON_SILENT_PEAK_THRESHOLD,
            "expected non-silent system audio, peak={peak}"
        );

        fn spawn_windows_playback(path: &Path) -> Child {
            let escaped_path = path.to_string_lossy().replace('\'', "''");
            let command = format!(
                "$player = New-Object System.Media.SoundPlayer '{}'; $player.PlaySync()",
                escaped_path
            );
            Command::new("powershell.exe")
                .args([
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    &command,
                ])
                .spawn()
                .expect("start PowerShell SoundPlayer")
        }
    }
}
