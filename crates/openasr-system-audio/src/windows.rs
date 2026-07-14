use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use wasapi::{
    AudioClient, DeviceEnumerator, Direction, SampleType, StreamMode, WasapiError, WaveFormat,
};
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW, TH32CS_SNAPPROCESS,
};
use windows::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

use crate::{
    CandidateProcess, CaptureBackendError, ProcessLoopbackMode, ProcessLoopbackSupport,
    SystemAudioSupport,
    pcm::{Pcm16FrameChunker, TARGET_CHANNELS, TARGET_FRAME_SAMPLES, TARGET_SAMPLE_RATE_HZ},
};

const SILENT_STREAK_DIAGNOSTIC_FRAMES: u32 = 250;
// Fallback WASAPI buffer duration (100ns units = 20ms), used when the
// default render endpoint does not report a usable minimum period, and
// always for process-loopback clients: AudioClient::get_device_period() is
// documented by the wasapi crate to return "Not implemented" for those, so
// that path cannot query a device-reported minimum at all.
const DEFAULT_BUFFER_DURATION_HNS: i64 = 200_000;

pub fn support_status() -> SystemAudioSupport {
    SystemAudioSupport {
        supported: true,
        label: "System audio (Windows smoke)".to_string(),
        detail: "Windows WASAPI all-system loopback smoke path is available. Call process_loopback_support() to probe per-process capture separately."
            .to_string(),
        platform: "windows".to_string(),
    }
}

pub fn run_loopback_capture(
    stop: Arc<AtomicBool>,
    on_frame: impl FnMut(Vec<i16>) -> Result<(), String>,
    on_diagnostic: impl FnMut(&str) -> Result<(), String>,
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

    let client: AudioClient = device.get_iaudioclient().map_err(map_wasapi_error(
        "capture_backend_failed",
        "Could not acquire IAudioClient",
    ))?;

    // get_device_period() only works on a device-backed client (it is
    // documented not to work at all on process-loopback clients), so this
    // probe stays in the all-system path rather than the shared session
    // helper below.
    let (_default_period_hns, min_period_hns) =
        client.get_device_period().map_err(map_wasapi_error(
            "capture_backend_failed",
            "Could not read WASAPI device period",
        ))?;
    let buffer_duration_hns = if min_period_hns > 0 {
        min_period_hns
    } else {
        DEFAULT_BUFFER_DURATION_HNS
    };

    run_wasapi_loopback_session(client, buffer_duration_hns, stop, on_frame, on_diagnostic)
}

/// Per-process loopback capture: captures only audio rendered by
/// `process_id` (and, per `mode`, its child processes), using the Windows
/// 10 2004+ `AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK` API. The wasapi
/// crate's `AudioClient::new_application_loopback_client` wraps the
/// `ActivateAudioInterfaceAsync` COM activation dance for this.
pub fn run_process_loopback_capture(
    process_id: u32,
    mode: ProcessLoopbackMode,
    stop: Arc<AtomicBool>,
    on_frame: impl FnMut(Vec<i16>) -> Result<(), String>,
    on_diagnostic: impl FnMut(&str) -> Result<(), String>,
) -> Result<String, CaptureBackendError> {
    ensure_process_exists(process_id)?;

    wasapi::initialize_mta()
        .ok()
        .map_err(|error| CaptureBackendError {
            code: "capture_backend_failed",
            message: "Could not initialize COM for WASAPI".to_string(),
            diagnostic: error.to_string(),
        })?;

    // See ComUninitGuard's doc comment on run_loopback_capture: pairs this
    // initialize_mta() with deinitialize() on every exit path, including the
    // early returns inside run_wasapi_loopback_session.
    let _com = ComUninitGuard;

    let client = AudioClient::new_application_loopback_client(
        process_id,
        include_tree_for_mode(mode),
    )
    .map_err(
        |error| CaptureBackendError {
            // ensure_process_exists() above already confirmed process_id is a
            // real, currently-running process, so an activation failure here
            // is far more likely a missing OS feature (pre-Windows 10 2004,
            // which lacks AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK) than a
            // transient per-call error. Fail closed as "unsupported" so
            // callers degrade cleanly instead of surfacing a bare backend
            // error for what is really a capability gap. Callers that want to
            // distinguish this ahead of time should call
            // process_loopback_support() first.
            code: "unsupported",
            message: "Could not activate a Windows process-loopback audio client. This requires Windows 10 2004 (build 19041) or later.".to_string(),
            diagnostic: error.to_string(),
        },
    )?;

    run_wasapi_loopback_session(
        client,
        DEFAULT_BUFFER_DURATION_HNS,
        stop,
        on_frame,
        on_diagnostic,
    )
}

/// Capability probe for `run_process_loopback_capture`. Performs a genuine
/// runtime activation probe (targeting this process's own PID, dropped
/// immediately without starting a stream) rather than an OS-version
/// heuristic: `GetVersionEx`-style checks can be lied to by application
/// compatibility shims, whereas attempting the real activation the capture
/// path uses cannot. This mirrors the CoreAudio symbol-load probe the macOS
/// backend uses for the same "does this API actually exist here" question.
pub fn process_loopback_support() -> ProcessLoopbackSupport {
    match probe_process_loopback_activation() {
        Ok(()) => ProcessLoopbackSupport {
            supported: true,
            detail: "Windows process-loopback capture (AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK) is available."
                .to_string(),
            platform: "windows".to_string(),
        },
        Err(error) => ProcessLoopbackSupport {
            supported: false,
            detail: format!(
                "Windows process-loopback capture is not available on this system (requires Windows 10 2004 / build 19041 or later). {}: {}",
                error.message, error.diagnostic
            ),
            platform: "windows".to_string(),
        },
    }
}

fn probe_process_loopback_activation() -> Result<(), CaptureBackendError> {
    wasapi::initialize_mta()
        .ok()
        .map_err(|error| CaptureBackendError {
            code: "capture_backend_failed",
            message: "Could not initialize COM for WASAPI".to_string(),
            diagnostic: error.to_string(),
        })?;
    let _com = ComUninitGuard;

    AudioClient::new_application_loopback_client(std::process::id(), false)
        .map(|_client| ())
        .map_err(map_wasapi_error(
            "unsupported",
            "Could not activate a Windows process-loopback audio client",
        ))
}

fn include_tree_for_mode(mode: ProcessLoopbackMode) -> bool {
    matches!(mode, ProcessLoopbackMode::IncludeProcessTree)
}

fn ensure_process_exists(process_id: u32) -> Result<(), CaptureBackendError> {
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, process_id) };
    match handle {
        Ok(handle) => {
            unsafe {
                let _ = CloseHandle(handle);
            }
            Ok(())
        }
        // Covers both "no such process" and "exists but access denied"
        // (e.g. a protected process): either way this pid cannot be used as
        // a per-process loopback target, so fail closed with one typed code
        // and let the OS error text in `diagnostic` distinguish the reason.
        Err(error) => Err(CaptureBackendError {
            code: "process_not_found",
            message: format!(
                "Process {process_id} could not be opened for per-process loopback capture."
            ),
            diagnostic: error.to_string(),
        }),
    }
}

/// Lists running processes as candidates for `run_process_loopback_capture`.
/// Uses a Toolhelp32 snapshot (`CreateToolhelp32Snapshot` /
/// `Process32FirstW` / `Process32NextW`), which does not require per-process
/// open rights to read pid + executable name, unlike `OpenProcess`.
pub fn list_candidate_processes() -> Result<Vec<CandidateProcess>, CaptureBackendError> {
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) }.map_err(|error| {
        CaptureBackendError {
            code: "capture_backend_failed",
            message: "Could not snapshot running processes for per-process loopback capture."
                .to_string(),
            diagnostic: error.to_string(),
        }
    })?;
    if snapshot.is_invalid() {
        return Err(CaptureBackendError {
            code: "capture_backend_failed",
            message: "Could not snapshot running processes for per-process loopback capture."
                .to_string(),
            diagnostic: "CreateToolhelp32Snapshot returned an invalid handle.".to_string(),
        });
    }
    let _snapshot_guard = ToolhelpSnapshotGuard(snapshot);

    let mut entry = PROCESSENTRY32W {
        dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
        ..Default::default()
    };

    let mut processes = Vec::new();
    // If even the first entry can't be read, treat it as an empty list
    // rather than an error: a snapshot that activated successfully but
    // yields no readable entries is a degenerate-but-not-fatal case for a
    // "candidate processes" listing (nothing to capture, not a backend
    // failure), and callers should not have to special-case it.
    let mut step = unsafe { Process32FirstW(snapshot, &mut entry) };
    while step.is_ok() {
        processes.push(CandidateProcess {
            pid: entry.th32ProcessID,
            name: process_name_from_entry(&entry),
        });
        entry.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
        step = unsafe { Process32NextW(snapshot, &mut entry) };
    }

    Ok(processes)
}

struct ToolhelpSnapshotGuard(HANDLE);

impl Drop for ToolhelpSnapshotGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

fn process_name_from_entry(entry: &PROCESSENTRY32W) -> String {
    let end = entry
        .szExeFile
        .iter()
        .position(|&unit| unit == 0)
        .unwrap_or(entry.szExeFile.len());
    String::from_utf16_lossy(&entry.szExeFile[..end])
}

/// Runs the shared post-activation WASAPI capture loop: format negotiation,
/// PCM16 mono 16k framing, the discontinuity/silence diagnostics, and clean
/// stream teardown. Both the all-system loopback path (`run_loopback_capture`)
/// and the per-process loopback path (`run_process_loopback_capture`) reach
/// this after obtaining an `AudioClient` in whatever way is specific to that
/// source; everything past that point is identical.
fn run_wasapi_loopback_session(
    mut client: AudioClient,
    buffer_duration_hns: i64,
    stop: Arc<AtomicBool>,
    mut on_frame: impl FnMut(Vec<i16>) -> Result<(), String>,
    mut on_diagnostic: impl FnMut(&str) -> Result<(), String>,
) -> Result<String, CaptureBackendError> {
    let desired = WaveFormat::new(
        16,
        16,
        &SampleType::Int,
        TARGET_SAMPLE_RATE_HZ,
        TARGET_CHANNELS,
        None,
    );

    let stream_mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns,
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
    fn maps_process_loopback_mode_to_include_tree() {
        assert!(include_tree_for_mode(
            ProcessLoopbackMode::IncludeProcessTree
        ));
        assert!(!include_tree_for_mode(
            ProcessLoopbackMode::ExcludeProcessTree
        ));
    }

    #[test]
    fn process_name_from_entry_trims_at_null_terminator() {
        let mut entry = PROCESSENTRY32W {
            dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        let name: Vec<u16> = "notepad.exe\0garbage-after-terminator"
            .encode_utf16()
            .collect();
        entry.szExeFile[..name.len()].copy_from_slice(&name);

        assert_eq!(process_name_from_entry(&entry), "notepad.exe");
    }

    #[test]
    fn process_name_from_entry_handles_missing_terminator() {
        let mut entry = PROCESSENTRY32W {
            dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        let name: Vec<u16> = "a".repeat(entry.szExeFile.len()).encode_utf16().collect();
        entry.szExeFile.copy_from_slice(&name);

        assert_eq!(
            process_name_from_entry(&entry),
            "a".repeat(entry.szExeFile.len())
        );
    }

    #[test]
    fn ensure_process_exists_rejects_an_unlikely_pid() {
        // Windows process IDs are allocated in multiples of 4; u32::MAX is
        // never a live pid, so OpenProcess must fail closed here rather than
        // panicking or silently treating a missing process as present.
        let error =
            ensure_process_exists(u32::MAX).expect_err("u32::MAX should not be a live process");
        assert_eq!(error.code, "process_not_found");
    }

    #[test]
    fn list_candidate_processes_includes_this_process() {
        let processes = list_candidate_processes().expect("list candidate processes");
        let own_pid = std::process::id();
        assert!(
            processes.iter().any(|process| process.pid == own_pid),
            "expected the current process ({own_pid}) in the candidate list of {} processes",
            processes.len()
        );
        assert!(
            processes
                .iter()
                .all(|process| !process.name.trim().is_empty()),
            "every candidate process should have a non-empty name"
        );
    }

    #[test]
    #[ignore = "requires a Windows 10 2004+ (build 19041) process-loopback API"]
    fn windows_process_loopback_smoke_emits_non_silent_frames() {
        use super::super::smoke_test_support::{
            MIN_SMOKE_FRAMES, NON_SILENT_PEAK_THRESHOLD, frame_peak, write_smoke_wav,
        };
        use std::path::Path;
        use std::process::{Child, Command};
        use std::thread;
        use std::time::Duration;

        let probe = process_loopback_support();
        assert!(
            probe.supported,
            "process-loopback support should be available for smoke: {}",
            probe.detail
        );

        let playback_path = write_smoke_wav("openasr-windows-process-loopback-smoke");
        let mut playback = spawn_windows_playback(&playback_path);
        let process_id = playback.id();

        let stop = Arc::new(AtomicBool::new(false));
        let stop_after_timeout = Arc::clone(&stop);
        let stopper = thread::spawn(move || {
            thread::sleep(Duration::from_secs(5));
            stop_after_timeout.store(true, Ordering::SeqCst);
        });

        let mut frames = 0_usize;
        let mut peak = 0_i32;
        let stop_after_signal = Arc::clone(&stop);
        let result = run_process_loopback_capture(
            process_id,
            ProcessLoopbackMode::IncludeProcessTree,
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

        result.expect("Windows process-loopback capture should run");
        eprintln!("Windows process-loopback smoke captured {frames} frames, peak {peak}");
        assert!(
            frames >= MIN_SMOKE_FRAMES,
            "expected at least {MIN_SMOKE_FRAMES} frames, got {frames}"
        );
        assert!(
            peak > NON_SILENT_PEAK_THRESHOLD,
            "expected non-silent process-loopback audio, peak={peak}"
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
