use std::ffi::{CStr, c_void};
use std::mem::{self, MaybeUninit};
use std::process::Command;
use std::ptr::{self, NonNull};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, RecvTimeoutError, SyncSender, TrySendError},
};
use std::time::Duration;

use libloading::Library;
use objc2::AnyThread;
use objc2_core_audio::{
    AudioDeviceCreateIOProcID, AudioDeviceDestroyIOProcID, AudioDeviceIOProcID, AudioDeviceStart,
    AudioDeviceStop, AudioHardwareCreateAggregateDevice, AudioHardwareDestroyAggregateDevice,
    AudioObjectGetPropertyData, AudioObjectID, AudioObjectPropertyAddress,
    AudioObjectPropertySelector, AudioObjectSetPropertyData, CATapDescription, CATapMuteBehavior,
    kAudioAggregateDeviceIsPrivateKey, kAudioAggregateDeviceNameKey,
    kAudioAggregateDevicePropertyTapList, kAudioAggregateDeviceTapAutoStartKey,
    kAudioAggregateDeviceTapListKey, kAudioAggregateDeviceUIDKey, kAudioHardwareNoError,
    kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal, kAudioObjectUnknown,
    kAudioSubTapUIDKey, kAudioTapPropertyFormat,
};
use objc2_core_audio_types::{
    AudioBuffer, AudioBufferList, AudioStreamBasicDescription, AudioTimeStamp,
    kAudioFormatFlagIsBigEndian, kAudioFormatFlagIsFloat, kAudioFormatFlagIsNonInterleaved,
    kAudioFormatFlagIsSignedInteger, kAudioFormatLinearPCM,
};
use objc2_core_foundation::{CFArray, CFBoolean, CFDictionary, CFRetained, CFString, CFType};
use objc2_foundation::{NSArray, NSNumber, NSString, NSUUID};

use crate::{
    CandidateProcess, CaptureBackendError, ProcessLoopbackMode, ProcessLoopbackSupport,
    SystemAudioSupport,
    pcm::{Pcm16FrameChunker, TARGET_SAMPLE_RATE_HZ},
};

type OSStatus = i32;

const CORE_AUDIO_FRAMEWORK: &str = "/System/Library/Frameworks/CoreAudio.framework/CoreAudio";
const READ_TIMEOUT_MS: u64 = 100;
const CAPTURE_QUEUE_FRAMES: usize = 128;
const MIN_TAP_MACOS_VERSION: MacOsVersion = MacOsVersion {
    major: 14,
    minor: 2,
    patch: 0,
};

type AudioHardwareCreateProcessTapFn =
    unsafe extern "C-unwind" fn(Option<&CATapDescription>, *mut AudioObjectID) -> OSStatus;
type AudioHardwareDestroyProcessTapFn = unsafe extern "C-unwind" fn(AudioObjectID) -> OSStatus;

pub fn support_status() -> SystemAudioSupport {
    let version = current_macos_version();
    let supported = version
        .map(|version| {
            macos_version_supports_process_taps(version) && process_tap_symbols_available()
        })
        .unwrap_or(false);

    SystemAudioSupport {
        supported,
        label: "System audio (macOS Core Audio tap)".to_string(),
        detail: macos_support_detail(version, supported),
        platform: "macos".to_string(),
    }
}

pub fn run_loopback_capture(
    stop: Arc<AtomicBool>,
    mut on_frame: impl FnMut(Vec<i16>) -> Result<(), String>,
    mut on_diagnostic: impl FnMut(&str) -> Result<(), String>,
) -> Result<String, CaptureBackendError> {
    ensure_macos_process_taps_supported()?;

    let (tx, rx) = mpsc::sync_channel(CAPTURE_QUEUE_FRAMES);
    let mut session = MacOsCaptureSession::create(tx)?;
    emit_diagnostic(
        &mut on_diagnostic,
        &format!(
            "Capturing macOS system audio through Core Audio process tap '{}'.",
            session.tap_uid
        ),
    )?;
    emit_diagnostic(
        &mut on_diagnostic,
        &format!("Core Audio tap format: {}.", session.format.describe()),
    )?;

    session.start()?;

    let mut chunker = Pcm16FrameChunker::new();
    loop {
        if stop.load(Ordering::SeqCst) {
            break;
        }

        match rx.recv_timeout(Duration::from_millis(READ_TIMEOUT_MS)) {
            Ok(samples) => {
                chunker
                    .push_samples(&samples, &mut on_frame)
                    .map_err(callback_error("Could not emit macOS system-audio frame."))?;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                return Err(CaptureBackendError {
                    code: "device_disconnected",
                    message: "macOS system-audio capture stream disconnected.".to_string(),
                    diagnostic: "The Core Audio callback queue closed before capture stopped."
                        .to_string(),
                });
            }
        }
    }

    session.stop();
    chunker.flush_padded(&mut on_frame).map_err(callback_error(
        "Could not emit final padded macOS system-audio frame.",
    ))?;

    Ok("Capture stopped".to_string())
}

/// Per-process loopback capture is not implemented on macOS: the Core Audio
/// process tap this backend uses (`AudioHardwareCreateProcessTap`) already
/// captures a single process's audio when constructed with a specific PID
/// instead of `initMonoGlobalTapButExcludeProcesses`, but wiring that up is
/// separate scope from the Windows per-process work this module was written
/// for. Fail closed rather than pretending support exists.
pub fn process_loopback_support() -> ProcessLoopbackSupport {
    ProcessLoopbackSupport {
        supported: false,
        detail: "macOS per-process loopback capture is not implemented; use the all-system Core Audio process tap via run_loopback_capture instead."
            .to_string(),
        platform: "macos".to_string(),
    }
}

pub fn list_candidate_processes() -> Result<Vec<CandidateProcess>, CaptureBackendError> {
    Err(CaptureBackendError {
        code: "unsupported",
        message:
            "Process enumeration for per-process loopback capture is not implemented on macOS."
                .to_string(),
        diagnostic: "list_candidate_processes has no macOS backend yet.".to_string(),
    })
}

pub fn run_process_loopback_capture(
    _process_id: u32,
    _mode: ProcessLoopbackMode,
    _stop: Arc<AtomicBool>,
    _on_frame: impl FnMut(Vec<i16>) -> Result<(), String>,
    _on_diagnostic: impl FnMut(&str) -> Result<(), String>,
) -> Result<String, CaptureBackendError> {
    Err(CaptureBackendError {
        code: "unsupported",
        message: "Per-process loopback capture is not implemented on macOS.".to_string(),
        diagnostic: "run_process_loopback_capture has no macOS backend yet; use run_loopback_capture for all-system Core Audio process-tap capture."
            .to_string(),
    })
}

struct MacOsCaptureSession {
    symbols: ProcessTapSymbols,
    tap_id: AudioObjectID,
    aggregate_device_id: AudioObjectID,
    io_proc_id: Option<AudioDeviceIOProcID>,
    callback_state: Box<MacOsAudioCallbackState>,
    started: bool,
    tap_uid: String,
    format: CoreAudioPcmFormat,
}

impl MacOsCaptureSession {
    fn create(samples_tx: SyncSender<Vec<i16>>) -> Result<Self, CaptureBackendError> {
        let symbols = ProcessTapSymbols::load()?;
        let tap_uuid = NSUUID::UUID();
        let tap_uid = tap_uuid.UUIDString().to_string();
        let tap_id = create_process_tap(&symbols, &tap_uuid)?;
        let aggregate_device_id = match create_aggregate_device(&tap_uid) {
            Ok(device_id) => device_id,
            Err(error) => {
                unsafe {
                    (symbols.destroy)(tap_id);
                }
                return Err(error);
            }
        };

        let format = match read_tap_format(tap_id).and_then(CoreAudioPcmFormat::from_asbd) {
            Ok(format) => format,
            Err(error) => {
                unsafe {
                    AudioHardwareDestroyAggregateDevice(aggregate_device_id);
                    (symbols.destroy)(tap_id);
                }
                return Err(error);
            }
        };

        let callback_state = Box::new(MacOsAudioCallbackState {
            samples_tx,
            converter: Mutex::new(CoreAudioPcmConverter::new(format)),
        });

        let mut session = Self {
            symbols,
            tap_id,
            aggregate_device_id,
            io_proc_id: None,
            callback_state,
            started: false,
            tap_uid,
            format,
        };

        session.set_tap_list()?;
        session.create_io_proc()?;
        Ok(session)
    }

    fn set_tap_list(&mut self) -> Result<(), CaptureBackendError> {
        let tap_uid = CFString::from_str(&self.tap_uid);
        let tap_list = CFArray::<CFString>::from_objects(&[tap_uid.as_ref()]);
        let mut tap_list_ref = CFRetained::as_ptr(&tap_list).as_ptr().cast::<c_void>();
        let mut address = property_address(kAudioAggregateDevicePropertyTapList);
        let status = unsafe {
            AudioObjectSetPropertyData(
                self.aggregate_device_id,
                NonNull::from(&mut address),
                0,
                ptr::null(),
                mem::size_of::<*const c_void>() as u32,
                NonNull::from(&mut tap_list_ref).cast(),
            )
        };
        check_os_status(
            status,
            "capture_backend_failed",
            "Could not attach the Core Audio tap to the aggregate device",
        )
    }

    fn create_io_proc(&mut self) -> Result<(), CaptureBackendError> {
        let mut io_proc_id: AudioDeviceIOProcID = None;
        let status = unsafe {
            AudioDeviceCreateIOProcID(
                self.aggregate_device_id,
                Some(core_audio_io_proc),
                (&mut *self.callback_state as *mut MacOsAudioCallbackState).cast(),
                NonNull::from(&mut io_proc_id),
            )
        };
        check_os_status(
            status,
            "capture_backend_failed",
            "Could not register the Core Audio aggregate-device input callback",
        )?;
        self.io_proc_id = Some(io_proc_id);
        Ok(())
    }

    fn start(&mut self) -> Result<(), CaptureBackendError> {
        let Some(io_proc_id) = self.io_proc_id else {
            return Err(CaptureBackendError {
                code: "capture_backend_failed",
                message: "Core Audio input callback was not registered.".to_string(),
                diagnostic: "AudioDeviceCreateIOProcID did not return an IOProcID.".to_string(),
            });
        };
        let status = unsafe { AudioDeviceStart(self.aggregate_device_id, io_proc_id) };
        check_os_status(
            status,
            "capture_backend_failed",
            "Could not start macOS Core Audio system-audio capture",
        )?;
        self.started = true;
        Ok(())
    }

    fn stop(&mut self) {
        if let Some(io_proc_id) = self.io_proc_id
            && self.started
        {
            unsafe {
                AudioDeviceStop(self.aggregate_device_id, io_proc_id);
            }
            self.started = false;
        }
    }
}

impl Drop for MacOsCaptureSession {
    fn drop(&mut self) {
        self.stop();
        if let Some(io_proc_id) = self.io_proc_id.take() {
            unsafe {
                AudioDeviceDestroyIOProcID(self.aggregate_device_id, io_proc_id);
            }
        }
        unsafe {
            AudioHardwareDestroyAggregateDevice(self.aggregate_device_id);
            (self.symbols.destroy)(self.tap_id);
        }
    }
}

struct MacOsAudioCallbackState {
    samples_tx: SyncSender<Vec<i16>>,
    converter: Mutex<CoreAudioPcmConverter>,
}

unsafe extern "C-unwind" fn core_audio_io_proc(
    _device: AudioObjectID,
    _now: NonNull<AudioTimeStamp>,
    input_data: NonNull<AudioBufferList>,
    _input_time: NonNull<AudioTimeStamp>,
    _output_data: NonNull<AudioBufferList>,
    _output_time: NonNull<AudioTimeStamp>,
    client_data: *mut c_void,
) -> OSStatus {
    let Some(state) = (unsafe { (client_data as *mut MacOsAudioCallbackState).as_ref() }) else {
        return kAudioHardwareNoError;
    };

    let Ok(mut converter) = state.converter.try_lock() else {
        return kAudioHardwareNoError;
    };

    let input_data = unsafe { input_data.as_ref() };
    if let Ok(samples) = converter.convert_buffer_list(input_data)
        && !samples.is_empty()
    {
        match state.samples_tx.try_send(samples) {
            Ok(()) | Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => {}
        }
    }

    kAudioHardwareNoError
}

struct ProcessTapSymbols {
    _library: Library,
    create: AudioHardwareCreateProcessTapFn,
    destroy: AudioHardwareDestroyProcessTapFn,
}

impl ProcessTapSymbols {
    fn load() -> Result<Self, CaptureBackendError> {
        let library =
            unsafe { Library::new(CORE_AUDIO_FRAMEWORK) }.map_err(|error| CaptureBackendError {
                code: "unsupported",
                message: "Could not load the Core Audio framework.".to_string(),
                diagnostic: error.to_string(),
            })?;

        let create = {
            let symbol = unsafe {
                library.get::<AudioHardwareCreateProcessTapFn>(b"AudioHardwareCreateProcessTap\0")
            }
            .map_err(|error| CaptureBackendError {
                code: "unsupported",
                message: "macOS Core Audio process taps are not available on this system."
                    .to_string(),
                diagnostic: error.to_string(),
            })?;
            *symbol
        };
        let destroy = {
            let symbol = unsafe {
                library.get::<AudioHardwareDestroyProcessTapFn>(b"AudioHardwareDestroyProcessTap\0")
            }
            .map_err(|error| CaptureBackendError {
                code: "unsupported",
                message: "macOS Core Audio process taps are not available on this system."
                    .to_string(),
                diagnostic: error.to_string(),
            })?;
            *symbol
        };

        Ok(Self {
            _library: library,
            create,
            destroy,
        })
    }
}

fn process_tap_symbols_available() -> bool {
    ProcessTapSymbols::load().is_ok()
}

fn create_process_tap(
    symbols: &ProcessTapSymbols,
    tap_uuid: &NSUUID,
) -> Result<AudioObjectID, CaptureBackendError> {
    let excluded_processes = NSArray::<NSNumber>::from_slice(&[]);
    let description = unsafe {
        CATapDescription::initMonoGlobalTapButExcludeProcesses(
            CATapDescription::alloc(),
            &excluded_processes,
        )
    };
    let tap_name = NSString::from_str("OpenASR System Audio");
    unsafe {
        description.setName(&tap_name);
        description.setUUID(tap_uuid);
        description.setPrivate(true);
        description.setProcessRestoreEnabled(false);
        description.setMuteBehavior(CATapMuteBehavior::Unmuted);
    }

    let mut tap_id = kAudioObjectUnknown;
    let status = unsafe { (symbols.create)(Some(&description), &mut tap_id) };
    check_os_status(
        status,
        "capture_backend_failed",
        "Could not create a macOS Core Audio process tap",
    )?;
    if tap_id == kAudioObjectUnknown {
        return Err(CaptureBackendError {
            code: "capture_backend_failed",
            message: "Core Audio returned an unknown tap object.".to_string(),
            diagnostic: "AudioHardwareCreateProcessTap succeeded with kAudioObjectUnknown."
                .to_string(),
        });
    }
    Ok(tap_id)
}

fn create_aggregate_device(tap_uid: &str) -> Result<AudioObjectID, CaptureBackendError> {
    let aggregate_name_key = cf_key(kAudioAggregateDeviceNameKey)?;
    let aggregate_uid_key = cf_key(kAudioAggregateDeviceUIDKey)?;
    let private_key = cf_key(kAudioAggregateDeviceIsPrivateKey)?;
    let tap_list_key = cf_key(kAudioAggregateDeviceTapListKey)?;
    let tap_auto_start_key = cf_key(kAudioAggregateDeviceTapAutoStartKey)?;
    let sub_tap_uid_key = cf_key(kAudioSubTapUIDKey)?;

    let aggregate_name = CFString::from_str("OpenASR System Audio Aggregate");
    let aggregate_uid = CFString::from_str(&format!("dev.openasr.desktop.system-audio.{tap_uid}"));
    let is_private = CFBoolean::new(true);
    let tap_auto_start = CFBoolean::new(true);
    let tap_uid_value = CFString::from_str(tap_uid);
    let sub_tap = CFDictionary::<CFType, CFType>::from_slices(
        &[sub_tap_uid_key.as_ref()],
        &[tap_uid_value.as_ref()],
    );
    let tap_list = CFArray::<CFType>::from_objects(&[sub_tap.as_ref()]);

    let description = CFDictionary::<CFType, CFType>::from_slices(
        &[
            aggregate_name_key.as_ref(),
            aggregate_uid_key.as_ref(),
            private_key.as_ref(),
            tap_list_key.as_ref(),
            tap_auto_start_key.as_ref(),
        ],
        &[
            aggregate_name.as_ref(),
            aggregate_uid.as_ref(),
            is_private.as_ref(),
            tap_list.as_ref(),
            tap_auto_start.as_ref(),
        ],
    );

    let mut device_id = kAudioObjectUnknown;
    let status = unsafe {
        AudioHardwareCreateAggregateDevice(description.as_opaque(), NonNull::from(&mut device_id))
    };
    check_os_status(
        status,
        "capture_backend_failed",
        "Could not create a private Core Audio aggregate device for system-audio capture",
    )?;
    if device_id == kAudioObjectUnknown {
        return Err(CaptureBackendError {
            code: "capture_backend_failed",
            message: "Core Audio returned an unknown aggregate device.".to_string(),
            diagnostic: "AudioHardwareCreateAggregateDevice succeeded with kAudioObjectUnknown."
                .to_string(),
        });
    }
    Ok(device_id)
}

fn read_tap_format(
    tap_id: AudioObjectID,
) -> Result<AudioStreamBasicDescription, CaptureBackendError> {
    let mut address = property_address(kAudioTapPropertyFormat);
    let mut size = mem::size_of::<AudioStreamBasicDescription>() as u32;
    let mut format = MaybeUninit::<AudioStreamBasicDescription>::zeroed();
    let status = unsafe {
        AudioObjectGetPropertyData(
            tap_id,
            NonNull::from(&mut address),
            0,
            ptr::null(),
            NonNull::from(&mut size),
            NonNull::new(format.as_mut_ptr().cast()).expect("format pointer"),
        )
    };
    check_os_status(
        status,
        "format_unsupported",
        "Could not read the Core Audio tap stream format",
    )?;
    if size as usize != mem::size_of::<AudioStreamBasicDescription>() {
        return Err(CaptureBackendError {
            code: "format_unsupported",
            message: "Core Audio returned an unexpected tap stream-format size.".to_string(),
            diagnostic: format!(
                "Expected {} bytes, received {size} bytes.",
                mem::size_of::<AudioStreamBasicDescription>()
            ),
        });
    }
    Ok(unsafe { format.assume_init() })
}

#[derive(Debug, Clone, Copy)]
struct CoreAudioPcmFormat {
    sample_rate_hz: f64,
    channels: usize,
    bytes_per_frame: usize,
    bytes_per_sample: usize,
    non_interleaved: bool,
    big_endian: bool,
    encoding: SampleEncoding,
}

impl CoreAudioPcmFormat {
    fn from_asbd(format: AudioStreamBasicDescription) -> Result<Self, CaptureBackendError> {
        if format.mFormatID != kAudioFormatLinearPCM {
            return Err(CaptureBackendError {
                code: "format_unsupported",
                message: "Core Audio tap did not expose linear PCM audio.".to_string(),
                diagnostic: format!(
                    "Unsupported Core Audio format id 0x{:08x}.",
                    format.mFormatID
                ),
            });
        }
        if !format.mSampleRate.is_finite() || format.mSampleRate <= 0.0 {
            return Err(CaptureBackendError {
                code: "format_unsupported",
                message: "Core Audio tap returned an invalid sample rate.".to_string(),
                diagnostic: format!("Sample rate was {}.", format.mSampleRate),
            });
        }
        if format.mChannelsPerFrame == 0 {
            return Err(CaptureBackendError {
                code: "format_unsupported",
                message: "Core Audio tap returned no audio channels.".to_string(),
                diagnostic: "mChannelsPerFrame was zero.".to_string(),
            });
        }

        let flags = format.mFormatFlags;
        let is_float = flags & kAudioFormatFlagIsFloat != 0;
        let is_signed_int = flags & kAudioFormatFlagIsSignedInteger != 0;
        let encoding = match (is_float, is_signed_int, format.mBitsPerChannel) {
            (true, _, 32) => SampleEncoding::Float32,
            (true, _, 64) => SampleEncoding::Float64,
            (false, true, 16) => SampleEncoding::Int16,
            (false, true, 32) => SampleEncoding::Int32,
            _ => {
                return Err(CaptureBackendError {
                    code: "format_unsupported",
                    message: "Core Audio tap returned an unsupported PCM sample encoding."
                        .to_string(),
                    diagnostic: format!(
                        "flags=0x{:08x}, bits_per_channel={}.",
                        flags, format.mBitsPerChannel
                    ),
                });
            }
        };

        let bytes_per_sample = (format.mBitsPerChannel / 8) as usize;
        let channels = format.mChannelsPerFrame as usize;
        let minimum_bytes_per_frame = bytes_per_sample.saturating_mul(channels);
        let bytes_per_frame = if format.mBytesPerFrame == 0 {
            minimum_bytes_per_frame
        } else {
            format.mBytesPerFrame as usize
        };
        if bytes_per_sample == 0 || bytes_per_frame < bytes_per_sample {
            return Err(CaptureBackendError {
                code: "format_unsupported",
                message: "Core Audio tap returned an invalid PCM frame layout.".to_string(),
                diagnostic: format!(
                    "bytes_per_sample={bytes_per_sample}, bytes_per_frame={bytes_per_frame}."
                ),
            });
        }

        Ok(Self {
            sample_rate_hz: format.mSampleRate,
            channels,
            bytes_per_frame,
            bytes_per_sample,
            non_interleaved: flags & kAudioFormatFlagIsNonInterleaved != 0,
            big_endian: flags & kAudioFormatFlagIsBigEndian != 0,
            encoding,
        })
    }

    fn describe(&self) -> String {
        let layout = if self.non_interleaved {
            "non-interleaved"
        } else {
            "interleaved"
        };
        format!(
            "{:.0} Hz, {} channel(s), {:?}, {layout}",
            self.sample_rate_hz, self.channels, self.encoding
        )
    }
}

#[derive(Debug, Clone, Copy)]
enum SampleEncoding {
    Float32,
    Float64,
    Int16,
    Int32,
}

struct CoreAudioPcmConverter {
    format: CoreAudioPcmFormat,
    resampler: LinearPcm16Resampler,
}

impl CoreAudioPcmConverter {
    fn new(format: CoreAudioPcmFormat) -> Self {
        Self {
            format,
            resampler: LinearPcm16Resampler::new(format.sample_rate_hz),
        }
    }

    fn convert_buffer_list(
        &mut self,
        input_data: &AudioBufferList,
    ) -> Result<Vec<i16>, CaptureBackendError> {
        let buffers = audio_buffers(input_data);
        if buffers.is_empty() {
            return Ok(Vec::new());
        }
        let mono = if self.format.non_interleaved {
            self.decode_non_interleaved(buffers)
        } else {
            self.decode_interleaved(&buffers[0])
        }?;
        let mut output = Vec::with_capacity(
            (mono.len() as f64 * TARGET_SAMPLE_RATE_HZ as f64 / self.format.sample_rate_hz).ceil()
                as usize,
        );
        self.resampler.push(&mono, &mut output);
        Ok(output)
    }

    fn decode_interleaved(&self, buffer: &AudioBuffer) -> Result<Vec<f32>, CaptureBackendError> {
        let data = audio_buffer_bytes(buffer)?;
        let channels = if buffer.mNumberChannels > 0 {
            buffer.mNumberChannels as usize
        } else {
            self.format.channels
        };
        let bytes_per_frame = self
            .format
            .bytes_per_frame
            .max(self.format.bytes_per_sample.saturating_mul(channels));
        if channels == 0 || bytes_per_frame == 0 {
            return Ok(Vec::new());
        }

        let frames = data.len() / bytes_per_frame;
        let mut mono = Vec::with_capacity(frames);
        for frame_index in 0..frames {
            let frame_offset = frame_index * bytes_per_frame;
            let mut sum = 0.0_f32;
            let mut count = 0_usize;
            for channel in 0..channels {
                let sample_offset = frame_offset + channel * self.format.bytes_per_sample;
                if sample_offset + self.format.bytes_per_sample <= data.len() {
                    sum += self.decode_sample(
                        &data[sample_offset..sample_offset + self.format.bytes_per_sample],
                    );
                    count += 1;
                }
            }
            if count > 0 {
                mono.push(sum / count as f32);
            }
        }
        Ok(mono)
    }

    fn decode_non_interleaved(
        &self,
        buffers: &[AudioBuffer],
    ) -> Result<Vec<f32>, CaptureBackendError> {
        let channel_count = buffers.len().min(self.format.channels).max(1);
        let mut channel_bytes = Vec::with_capacity(channel_count);
        for buffer in buffers.iter().take(channel_count) {
            channel_bytes.push(audio_buffer_bytes(buffer)?);
        }
        let frames = channel_bytes
            .iter()
            .map(|bytes| bytes.len() / self.format.bytes_per_sample)
            .min()
            .unwrap_or(0);

        let mut mono = Vec::with_capacity(frames);
        for frame_index in 0..frames {
            let mut sum = 0.0_f32;
            for bytes in &channel_bytes {
                let sample_offset = frame_index * self.format.bytes_per_sample;
                sum += self.decode_sample(
                    &bytes[sample_offset..sample_offset + self.format.bytes_per_sample],
                );
            }
            mono.push(sum / channel_bytes.len() as f32);
        }
        Ok(mono)
    }

    fn decode_sample(&self, bytes: &[u8]) -> f32 {
        match self.format.encoding {
            SampleEncoding::Float32 => {
                let raw = [bytes[0], bytes[1], bytes[2], bytes[3]];
                let value = if self.format.big_endian {
                    f32::from_be_bytes(raw)
                } else {
                    f32::from_le_bytes(raw)
                };
                value.clamp(-1.0, 1.0)
            }
            SampleEncoding::Float64 => {
                let raw = [
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ];
                let value = if self.format.big_endian {
                    f64::from_be_bytes(raw)
                } else {
                    f64::from_le_bytes(raw)
                };
                (value as f32).clamp(-1.0, 1.0)
            }
            SampleEncoding::Int16 => {
                let raw = [bytes[0], bytes[1]];
                let value = if self.format.big_endian {
                    i16::from_be_bytes(raw)
                } else {
                    i16::from_le_bytes(raw)
                };
                value as f32 / i16::MAX as f32
            }
            SampleEncoding::Int32 => {
                let raw = [bytes[0], bytes[1], bytes[2], bytes[3]];
                let value = if self.format.big_endian {
                    i32::from_be_bytes(raw)
                } else {
                    i32::from_le_bytes(raw)
                };
                value as f32 / i32::MAX as f32
            }
        }
    }
}

struct LinearPcm16Resampler {
    source_rate_hz: f64,
    next_output_source_index: f64,
    absolute_source_index: u64,
    last_source_sample: Option<f32>,
}

impl LinearPcm16Resampler {
    fn new(source_rate_hz: f64) -> Self {
        Self {
            source_rate_hz,
            next_output_source_index: 0.0,
            absolute_source_index: 0,
            last_source_sample: None,
        }
    }

    fn push(&mut self, input: &[f32], output: &mut Vec<i16>) {
        if input.is_empty() {
            return;
        }

        if (self.source_rate_hz - TARGET_SAMPLE_RATE_HZ as f64).abs() < f64::EPSILON {
            output.extend(input.iter().copied().map(float_to_i16));
            self.absolute_source_index += input.len() as u64;
            self.last_source_sample = input.last().copied();
            self.next_output_source_index = self.absolute_source_index as f64;
            return;
        }

        let base = self.absolute_source_index;
        let end = base + input.len() as u64;
        let step = self.source_rate_hz / TARGET_SAMPLE_RATE_HZ as f64;
        if self.next_output_source_index < base as f64 && self.last_source_sample.is_none() {
            self.next_output_source_index = base as f64;
        }

        while self.next_output_source_index + 1.0 < end as f64 {
            let left_index = self.next_output_source_index.floor() as u64;
            let right_index = left_index + 1;
            let Some(left) = self.sample_at(base, input, left_index) else {
                self.next_output_source_index = base as f64;
                continue;
            };
            let Some(right) = self.sample_at(base, input, right_index) else {
                break;
            };
            let fraction = (self.next_output_source_index - left_index as f64) as f32;
            output.push(float_to_i16(left + (right - left) * fraction));
            self.next_output_source_index += step;
        }

        self.absolute_source_index = end;
        self.last_source_sample = input.last().copied();
    }

    fn sample_at(&self, base: u64, input: &[f32], index: u64) -> Option<f32> {
        if index + 1 == base {
            return self.last_source_sample;
        }
        if index < base {
            return None;
        }
        input.get((index - base) as usize).copied()
    }
}

fn audio_buffers(input_data: &AudioBufferList) -> &[AudioBuffer] {
    if input_data.mNumberBuffers == 0 {
        return &[];
    }
    unsafe {
        std::slice::from_raw_parts(
            input_data.mBuffers.as_ptr(),
            input_data.mNumberBuffers as usize,
        )
    }
}

fn audio_buffer_bytes(buffer: &AudioBuffer) -> Result<&[u8], CaptureBackendError> {
    if buffer.mDataByteSize == 0 || buffer.mData.is_null() {
        return Ok(&[]);
    }
    Ok(unsafe {
        std::slice::from_raw_parts(buffer.mData.cast::<u8>(), buffer.mDataByteSize as usize)
    })
}

fn float_to_i16(value: f32) -> i16 {
    let value = value.clamp(-1.0, 1.0);
    if value < 0.0 {
        (value * 32768.0).round() as i16
    } else {
        (value * 32767.0).round() as i16
    }
}

fn ensure_macos_process_taps_supported() -> Result<(), CaptureBackendError> {
    let version = current_macos_version().ok_or_else(|| CaptureBackendError {
        code: "unsupported",
        message: "Could not determine the macOS version for system-audio capture.".to_string(),
        diagnostic: "sw_vers -productVersion did not return a parseable version.".to_string(),
    })?;

    if !macos_version_supports_process_taps(version) {
        return Err(CaptureBackendError {
            code: "unsupported",
            message: "macOS system-audio capture requires Core Audio process taps.".to_string(),
            diagnostic: format!(
                "Detected macOS {version}; Core Audio process taps require macOS {MIN_TAP_MACOS_VERSION} or later."
            ),
        });
    }

    Ok(())
}

fn current_macos_version() -> Option<MacOsVersion> {
    let output = Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_macos_version(String::from_utf8_lossy(&output.stdout).trim())
}

fn macos_support_detail(version: Option<MacOsVersion>, supported: bool) -> String {
    if supported {
        return "macOS Core Audio process-tap capture is available. The first capture may prompt for System Audio Recording permission."
            .to_string();
    }
    match version {
        Some(version) if !macos_version_supports_process_taps(version) => format!(
            "macOS system-audio capture requires macOS {MIN_TAP_MACOS_VERSION} or later; detected macOS {version}."
        ),
        Some(version) => format!(
            "macOS {version} is new enough for Core Audio process taps, but the process-tap symbols were not available from CoreAudio."
        ),
        None => "Could not determine macOS version for Core Audio process-tap capture.".to_string(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MacOsVersion {
    major: u32,
    minor: u32,
    patch: u32,
}

impl std::fmt::Display for MacOsVersion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

fn parse_macos_version(value: &str) -> Option<MacOsVersion> {
    let mut parts = value.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    Some(MacOsVersion {
        major,
        minor,
        patch,
    })
}

fn macos_version_supports_process_taps(version: MacOsVersion) -> bool {
    version >= MIN_TAP_MACOS_VERSION
}

fn cf_key(key: &CStr) -> Result<CFRetained<CFString>, CaptureBackendError> {
    key.to_str()
        .map(CFString::from_str)
        .map_err(|error| CaptureBackendError {
            code: "capture_backend_failed",
            message: "Core Audio exposed a non-UTF8 aggregate-device dictionary key.".to_string(),
            diagnostic: error.to_string(),
        })
}

fn property_address(selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}

fn emit_diagnostic(
    on_diagnostic: &mut impl FnMut(&str) -> Result<(), String>,
    message: &str,
) -> Result<(), CaptureBackendError> {
    on_diagnostic(message).map_err(|error| CaptureBackendError {
        code: "capture_backend_failed",
        message: "Could not emit macOS system-audio diagnostic to desktop frontend.".to_string(),
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

fn check_os_status(
    status: OSStatus,
    code: &'static str,
    message: &'static str,
) -> Result<(), CaptureBackendError> {
    if status == kAudioHardwareNoError {
        return Ok(());
    }
    Err(CaptureBackendError {
        code,
        message: message.to_string(),
        diagnostic: format!("Core Audio returned {}.", os_status_diagnostic(status)),
    })
}

fn os_status_diagnostic(status: OSStatus) -> String {
    let bytes = (status as u32).to_be_bytes();
    if bytes
        .iter()
        .all(|byte| byte.is_ascii_graphic() || *byte == b' ')
    {
        format!("'{}' ({status})", String::from_utf8_lossy(&bytes))
    } else {
        status.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::super::smoke_test_support::{
        MIN_SMOKE_FRAMES, NON_SILENT_PEAK_THRESHOLD, frame_peak, write_smoke_wav,
    };
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn parses_macos_versions_with_optional_patch() {
        assert_eq!(
            parse_macos_version("14.2"),
            Some(MacOsVersion {
                major: 14,
                minor: 2,
                patch: 0
            })
        );
        assert_eq!(
            parse_macos_version("15.5.1"),
            Some(MacOsVersion {
                major: 15,
                minor: 5,
                patch: 1
            })
        );
        assert_eq!(parse_macos_version("not-a-version"), None);
    }

    #[test]
    fn gates_process_taps_to_macos_14_2_or_later() {
        assert!(!macos_version_supports_process_taps(MacOsVersion {
            major: 14,
            minor: 1,
            patch: 9
        }));
        assert!(macos_version_supports_process_taps(MacOsVersion {
            major: 14,
            minor: 2,
            patch: 0
        }));
        assert!(macos_version_supports_process_taps(MacOsVersion {
            major: 15,
            minor: 0,
            patch: 0
        }));
    }

    #[test]
    fn resamples_48k_float_to_16k_pcm() {
        let mut resampler = LinearPcm16Resampler::new(48_000.0);
        let input = (0..960)
            .map(|index| if index % 2 == 0 { 0.5 } else { -0.5 })
            .collect::<Vec<_>>();
        let mut output = Vec::new();

        resampler.push(&input, &mut output);

        assert_eq!(output.len(), 320);
        assert!(output.iter().all(|sample| sample.unsigned_abs() > 0));
    }

    #[test]
    fn mixes_interleaved_stereo_float_samples() {
        let format = CoreAudioPcmFormat {
            sample_rate_hz: 16_000.0,
            channels: 2,
            bytes_per_frame: 8,
            bytes_per_sample: 4,
            non_interleaved: false,
            big_endian: false,
            encoding: SampleEncoding::Float32,
        };
        let mut converter = CoreAudioPcmConverter::new(format);
        let mut raw = Vec::new();
        for sample in [1.0_f32, -1.0, 0.5, 0.5] {
            raw.extend_from_slice(&sample.to_le_bytes());
        }
        let buffer = AudioBuffer {
            mNumberChannels: 2,
            mDataByteSize: raw.len() as u32,
            mData: raw.as_mut_ptr().cast(),
        };
        let list = AudioBufferList {
            mNumberBuffers: 1,
            mBuffers: [buffer],
        };

        let output = converter.convert_buffer_list(&list).expect("convert");

        assert_eq!(output, vec![0, 16384]);
    }

    #[test]
    fn formats_four_char_osstatus() {
        assert_eq!(os_status_diagnostic(0x7768_6174), "'what' (2003329396)");
    }

    #[test]
    #[ignore = "requires macOS system-audio permission, an output device, and local playback"]
    fn macos_core_audio_system_audio_smoke_emits_non_silent_frames() {
        let support = support_status();
        assert!(
            support.supported,
            "system audio support should be available for smoke: {}",
            support.detail
        );

        let playback_path = write_smoke_wav("openasr-macos-system-audio-smoke");
        let mut playback = Command::new("afplay")
            .arg(&playback_path)
            .spawn()
            .expect("start afplay");
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

        result.expect("macOS system-audio capture should run");
        eprintln!("macOS system-audio smoke captured {frames} frames, peak {peak}");
        assert!(
            frames >= MIN_SMOKE_FRAMES,
            "expected at least {MIN_SMOKE_FRAMES} frames, got {frames}"
        );
        assert!(
            peak > NON_SILENT_PEAK_THRESHOLD,
            "expected non-silent system audio, peak={peak}"
        );
    }
}
