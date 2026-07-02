use std::io::Read;
use std::process::{ChildStderr, Command, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, RecvTimeoutError},
};
use std::time::Duration;

use crate::{
    CaptureBackendError, SystemAudioSupport,
    pcm::{Pcm16FrameChunker, TARGET_CHANNELS, TARGET_SAMPLE_RATE_HZ},
};

const READ_TIMEOUT_MS: u64 = 100;

pub fn support_status() -> SystemAudioSupport {
    let has_pactl = command_available("pactl");
    let has_parec = command_available("parec");
    let supported = has_pactl && has_parec;
    SystemAudioSupport {
        supported,
        label: "System audio (Linux Pulse/PipeWire monitor)".to_string(),
        detail: if supported {
            "Linux system-audio capture uses the default sink monitor via pactl + parec. This works with PulseAudio and PipeWire's Pulse compatibility service."
                .to_string()
        } else {
            "Linux system-audio capture requires PulseAudio/PipeWire Pulse tools: pactl and parec. Install pulseaudio-utils or the distro equivalent."
                .to_string()
        },
        platform: "linux".to_string(),
    }
}

pub fn run_loopback_capture(
    stop: Arc<AtomicBool>,
    mut on_frame: impl FnMut(Vec<i16>) -> Result<(), String>,
    mut on_diagnostic: impl FnMut(&str) -> Result<(), String>,
) -> Result<String, CaptureBackendError> {
    ensure_linux_tools()?;
    let monitor_source = default_monitor_source()?;
    emit_diagnostic(
        &mut on_diagnostic,
        &format!("Capturing Linux system audio from monitor source '{monitor_source}'."),
    )?;

    let mut child = Command::new("parec")
        .arg("--format=s16le")
        .arg(format!("--rate={TARGET_SAMPLE_RATE_HZ}"))
        .arg(format!("--channels={TARGET_CHANNELS}"))
        .arg(format!("--device={monitor_source}"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| CaptureBackendError {
            code: "capture_backend_failed",
            message: "Could not start Linux system-audio monitor capture.".to_string(),
            diagnostic: error.to_string(),
        })?;

    let mut stderr = child.stderr.take();
    let stdout = child.stdout.take().ok_or_else(|| CaptureBackendError {
        code: "capture_backend_failed",
        message: "Could not open stdout for Linux system-audio capture.".to_string(),
        diagnostic: "parec did not provide a stdout pipe.".to_string(),
    })?;

    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut stdout = stdout;
        let mut buffer = [0_u8; 4096];
        loop {
            match stdout.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => {
                    if tx.send(Ok(buffer[..count].to_vec())).is_err() {
                        break;
                    }
                }
                Err(error) => {
                    let _ = tx.send(Err(error));
                    break;
                }
            }
        }
    });

    let mut chunker = Pcm16FrameChunker::new();
    let mut stopped_by_request = false;
    loop {
        if stop.load(Ordering::SeqCst) {
            stopped_by_request = true;
            let _ = child.kill();
            break;
        }

        match rx.recv_timeout(Duration::from_millis(READ_TIMEOUT_MS)) {
            Ok(Ok(bytes)) => chunker
                .push_bytes(&bytes, &mut on_frame)
                .map_err(callback_error("Could not emit Linux system-audio frame."))?,
            Ok(Err(error)) => {
                let _ = child.kill();
                return Err(CaptureBackendError {
                    code: "capture_backend_failed",
                    message: "Linux system-audio capture stream failed.".to_string(),
                    diagnostic: error.to_string(),
                });
            }
            Err(RecvTimeoutError::Timeout) => {
                if let Some(status) = child.try_wait().map_err(|error| CaptureBackendError {
                    code: "capture_backend_failed",
                    message: "Could not inspect Linux system-audio capture process.".to_string(),
                    diagnostic: error.to_string(),
                })? {
                    if status.success() {
                        break;
                    }
                    return Err(CaptureBackendError {
                        code: "capture_backend_failed",
                        message: "Linux system-audio capture process exited unexpectedly."
                            .to_string(),
                        diagnostic: child_stderr(&mut stderr),
                    });
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }

    if stopped_by_request {
        let _ = child.kill();
    }
    let _ = child.wait();
    let _ = reader.join();
    chunker.flush_padded(&mut on_frame).map_err(callback_error(
        "Could not emit final padded Linux system-audio frame.",
    ))?;

    Ok("Capture stopped".to_string())
}

fn ensure_linux_tools() -> Result<(), CaptureBackendError> {
    let missing = ["pactl", "parec"]
        .into_iter()
        .filter(|command| !command_available(command))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }
    Err(CaptureBackendError {
        code: "unsupported",
        message: "Linux system-audio capture tools are missing.".to_string(),
        diagnostic: format!(
            "Missing command(s): {}. Install pulseaudio-utils or the distro equivalent.",
            missing.join(", ")
        ),
    })
}

fn default_monitor_source() -> Result<String, CaptureBackendError> {
    let sink = command_stdout("pactl", &["get-default-sink"])
        .or_else(|_| default_sink_from_pactl_info())?;
    monitor_source_for_sink(&sink)
}

fn monitor_source_for_sink(sink: &str) -> Result<String, CaptureBackendError> {
    let sink = sink.trim();
    if sink.is_empty() {
        return Err(CaptureBackendError {
            code: "no_default_render_endpoint",
            message: "No default Linux audio sink is available.".to_string(),
            diagnostic: "pactl returned an empty default sink.".to_string(),
        });
    }
    if sink.ends_with(".monitor") {
        Ok(sink.to_string())
    } else {
        Ok(format!("{sink}.monitor"))
    }
}

fn default_sink_from_pactl_info() -> Result<String, CaptureBackendError> {
    let info = command_stdout("pactl", &["info"])?;
    parse_default_sink_from_pactl_info(&info).ok_or_else(|| CaptureBackendError {
        code: "no_default_render_endpoint",
        message: "Could not determine the default Linux audio sink.".to_string(),
        diagnostic: "pactl info did not contain a Default Sink entry.".to_string(),
    })
}

fn parse_default_sink_from_pactl_info(info: &str) -> Option<String> {
    info.lines()
        .find_map(|line| line.trim().strip_prefix("Default Sink:"))
        .map(str::trim)
        .filter(|sink| !sink.is_empty())
        .map(ToOwned::to_owned)
}

fn command_stdout(command: &str, args: &[&str]) -> Result<String, CaptureBackendError> {
    let output =
        Command::new(command)
            .args(args)
            .output()
            .map_err(|error| CaptureBackendError {
                code: "capture_backend_failed",
                message: format!("Could not run {command}."),
                diagnostic: error.to_string(),
            })?;
    if !output.status.success() {
        return Err(CaptureBackendError {
            code: "capture_backend_failed",
            message: format!("{command} failed."),
            diagnostic: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn command_available(command: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {command} >/dev/null 2>&1"))
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn emit_diagnostic(
    on_diagnostic: &mut impl FnMut(&str) -> Result<(), String>,
    message: &str,
) -> Result<(), CaptureBackendError> {
    on_diagnostic(message).map_err(|error| CaptureBackendError {
        code: "capture_backend_failed",
        message: "Could not emit Linux system-audio diagnostic to desktop frontend.".to_string(),
        diagnostic: error,
    })
}

fn callback_error(message: &'static str) -> impl FnOnce(String) -> CaptureBackendError {
    move |diagnostic| CaptureBackendError {
        code: "capture_backend_failed",
        message: message.to_string(),
        diagnostic,
    }
}

fn child_stderr(stderr: &mut Option<ChildStderr>) -> String {
    let Some(stderr) = stderr else {
        return "No stderr was captured from parec.".to_string();
    };
    let mut output = Vec::new();
    let _ = stderr.read_to_end(&mut output);
    let text = String::from_utf8_lossy(&output).trim().to_string();
    if text.is_empty() {
        "parec exited without stderr output.".to_string()
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_monitor_suffix_to_default_sink() {
        assert_eq!(
            monitor_source_for_sink("alsa_output.pci-0000_00_1f.3.analog-stereo")
                .expect("monitor source"),
            "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
        );
    }

    #[test]
    fn preserves_monitor_source_when_default_is_already_monitor() {
        assert_eq!(
            monitor_source_for_sink("bluez_output.headset.monitor").expect("monitor source"),
            "bluez_output.headset.monitor"
        );
    }

    #[test]
    fn rejects_empty_default_sink() {
        let error = monitor_source_for_sink(" \n").expect_err("empty sink should fail");
        assert_eq!(error.code, "no_default_render_endpoint");
    }

    #[test]
    fn parses_default_sink_from_pactl_info() {
        let info = "\
Server String: /run/user/1000/pulse/native
Library Protocol Version: 35
Default Sink: alsa_output.usb.Focusrite.monitorless
Default Source: alsa_input.usb.Focusrite";

        assert_eq!(
            parse_default_sink_from_pactl_info(info).as_deref(),
            Some("alsa_output.usb.Focusrite.monitorless")
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires a Linux PulseAudio/PipeWire session, an output device, and local playback"]
    fn linux_pulse_monitor_system_audio_smoke_emits_non_silent_frames() {
        use super::super::smoke_test_support::{
            MIN_SMOKE_FRAMES, NON_SILENT_PEAK_THRESHOLD, frame_peak, write_smoke_wav,
        };
        use std::path::Path;
        use std::thread;
        use std::time::Duration;

        let support = support_status();
        assert!(
            support.supported,
            "system audio support should be available for smoke: {}",
            support.detail
        );

        let playback_path = write_smoke_wav("openasr-linux-system-audio-smoke");
        let mut playback = spawn_linux_playback(&playback_path);
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

        result.expect("Linux system-audio capture should run");
        eprintln!("Linux system-audio smoke captured {frames} frames, peak {peak}");
        assert!(
            frames >= MIN_SMOKE_FRAMES,
            "expected at least {MIN_SMOKE_FRAMES} frames, got {frames}"
        );
        assert!(
            peak > NON_SILENT_PEAK_THRESHOLD,
            "expected non-silent system audio, peak={peak}"
        );

        fn spawn_linux_playback(path: &Path) -> std::process::Child {
            for command in ["paplay", "pw-play", "aplay"] {
                if command_available(command) {
                    return Command::new(command)
                        .arg(path)
                        .spawn()
                        .unwrap_or_else(|error| panic!("start {command}: {error}"));
                }
            }
            panic!("Linux smoke requires one playback command: paplay, pw-play, or aplay");
        }
    }
}
