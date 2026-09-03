//! Windows Media Foundation converter for formats the in-process decoder
//! refuses (HE-AAC SBR/PS, and other containers the OS can decode).
//!
//! This is the Windows peer of macOS `/usr/bin/afconvert`: system APIs only,
//! never Fraunhofer FDK, never a spawned ffmpeg.
//!
//! `mfplat.dll` / `mfreadwrite.dll` are resolved at run time, not imported.
//! Windows N/KN editions and Windows Server without the Media Foundation
//! feature do not ship them; a load-time import would make every binary that
//! links this crate fail to start with `STATUS_DLL_NOT_FOUND` and no message.
//! Loading lazily keeps the process alive and turns the gap into an ordinary
//! conversion error for the one input that needs the system decoder.

use std::ffi::c_void;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use windows::Win32::Media::MediaFoundation::{
    IMFByteStream, IMFMediaBuffer, IMFMediaType, IMFSample, IMFSourceReader, MF_ACCESSMODE_READ,
    MF_FILE_ACCESSMODE, MF_FILE_FLAGS, MF_FILE_OPENMODE, MF_FILEFLAGS_NONE,
    MF_MT_AUDIO_AVG_BYTES_PER_SECOND, MF_MT_AUDIO_BITS_PER_SAMPLE, MF_MT_AUDIO_BLOCK_ALIGNMENT,
    MF_MT_AUDIO_NUM_CHANNELS, MF_MT_AUDIO_SAMPLES_PER_SECOND, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE,
    MF_OPENMODE_FAIL_IF_NOT_EXIST, MF_SOURCE_READER_ALL_STREAMS,
    MF_SOURCE_READER_FIRST_AUDIO_STREAM, MF_SOURCE_READERF_ENDOFSTREAM, MF_SOURCE_READERF_ERROR,
    MF_VERSION, MFAudioFormat_PCM, MFMediaType_Audio, MFSTARTUP_NOSOCKET,
};
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoUninitialize};
use windows::core::{HRESULT, HSTRING, Interface, PCWSTR};
use windows_sys::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};

/// `RPC_E_CHANGED_MODE`: this thread already has a different COM apartment.
const RPC_E_CHANGED_MODE: HRESULT = HRESULT(0x8001_0106_u32 as i32);

const TARGET_RATE_HZ: u32 = 16_000;
const TARGET_CHANNELS: u32 = 1;
const TARGET_BITS: u32 = 16;

pub(super) fn convert_to_wav16k_mono(input: &Path, output: &Path) -> Result<(), String> {
    let api = MediaFoundationApi::get()?;
    let _com = ComApartment::enter()?;
    let _mf = MediaFoundation::startup(api)?;
    unsafe { convert_with_source_reader(api, input, output) }
}

type MfStartupFn = unsafe extern "system" fn(version: u32, flags: u32) -> HRESULT;
type MfShutdownFn = unsafe extern "system" fn() -> HRESULT;
type MfCreateFileFn = unsafe extern "system" fn(
    access_mode: MF_FILE_ACCESSMODE,
    open_mode: MF_FILE_OPENMODE,
    flags: MF_FILE_FLAGS,
    url: PCWSTR,
    byte_stream: *mut *mut c_void,
) -> HRESULT;
type MfCreateMediaTypeFn = unsafe extern "system" fn(media_type: *mut *mut c_void) -> HRESULT;
type MfCreateSourceReaderFromByteStreamFn = unsafe extern "system" fn(
    byte_stream: *mut c_void,
    attributes: *mut c_void,
    source_reader: *mut *mut c_void,
) -> HRESULT;

/// The free functions this converter needs from the two Media Foundation
/// system DLLs, resolved once per process from `%SystemRoot%\System32` only.
/// COM interface methods dispatch through vtables and need no import.
struct MediaFoundationApi {
    startup: MfStartupFn,
    shutdown: MfShutdownFn,
    create_file: MfCreateFileFn,
    create_media_type: MfCreateMediaTypeFn,
    create_source_reader_from_byte_stream: MfCreateSourceReaderFromByteStreamFn,
}

impl MediaFoundationApi {
    fn get() -> Result<&'static Self, String> {
        static API: OnceLock<Result<MediaFoundationApi, String>> = OnceLock::new();
        API.get_or_init(Self::load).as_ref().map_err(Clone::clone)
    }

    fn load() -> Result<Self, String> {
        let mfplat = load_system_library("mfplat.dll")?;
        let mfreadwrite = load_system_library("mfreadwrite.dll")?;
        // SAFETY: each symbol is resolved from the DLL that documents it and
        // transmuted to that function's documented `extern "system"` signature.
        unsafe {
            Ok(Self {
                startup: std::mem::transmute::<*const c_void, MfStartupFn>(system_symbol(
                    mfplat,
                    "mfplat.dll",
                    b"MFStartup\0",
                )?),
                shutdown: std::mem::transmute::<*const c_void, MfShutdownFn>(system_symbol(
                    mfplat,
                    "mfplat.dll",
                    b"MFShutdown\0",
                )?),
                create_file: std::mem::transmute::<*const c_void, MfCreateFileFn>(system_symbol(
                    mfplat,
                    "mfplat.dll",
                    b"MFCreateFile\0",
                )?),
                create_media_type: std::mem::transmute::<*const c_void, MfCreateMediaTypeFn>(
                    system_symbol(mfplat, "mfplat.dll", b"MFCreateMediaType\0")?,
                ),
                create_source_reader_from_byte_stream: std::mem::transmute::<
                    *const c_void,
                    MfCreateSourceReaderFromByteStreamFn,
                >(system_symbol(
                    mfreadwrite,
                    "mfreadwrite.dll",
                    b"MFCreateSourceReaderFromByteStream\0",
                )?),
            })
        }
    }
}

fn load_system_library(name: &str) -> Result<*mut c_void, String> {
    let wide = name
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect::<Vec<u16>>();
    // SAFETY: `wide` is a live NUL-terminated UTF-16 name. The handle is kept
    // for the life of the process (the API table is a process-wide singleton).
    let handle = unsafe {
        LoadLibraryExW(
            wide.as_ptr(),
            std::ptr::null_mut(),
            LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
    };
    if handle.is_null() {
        return Err(format!(
            "Windows Media Foundation is not available on this system ({name}: {}). \
             Windows N/KN editions need the Media Feature Pack; Windows Server needs the \
             Media Foundation feature. Convert the file to WAV/FLAC/MP3 or install it.",
            std::io::Error::last_os_error()
        ));
    }
    Ok(handle.cast())
}

fn system_symbol(
    handle: *mut c_void,
    library: &str,
    name: &'static [u8],
) -> Result<*const c_void, String> {
    debug_assert_eq!(name.last(), Some(&0));
    // SAFETY: `handle` is a live library handle and `name` is NUL-terminated.
    let function = unsafe { GetProcAddress(handle.cast(), name.as_ptr()) }.ok_or_else(|| {
        format!(
            "Windows Media Foundation ({library}) is missing {}",
            String::from_utf8_lossy(&name[..name.len() - 1])
        )
    })?;
    Ok(function as *const () as *const c_void)
}

struct ComApartment {
    owned: bool,
}

impl ComApartment {
    fn enter() -> Result<Self, String> {
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if hr.is_ok() {
            return Ok(Self { owned: hr.0 == 0 });
        }
        if hr == RPC_E_CHANGED_MODE {
            return Ok(Self { owned: false });
        }
        Err(format!("COM init failed: {hr}"))
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        if self.owned {
            unsafe { CoUninitialize() };
        }
    }
}

struct MediaFoundation {
    api: &'static MediaFoundationApi,
}

impl MediaFoundation {
    fn startup(api: &'static MediaFoundationApi) -> Result<Self, String> {
        unsafe { (api.startup)(MF_VERSION, MFSTARTUP_NOSOCKET) }
            .ok()
            .map_err(|error| format!("Media Foundation startup failed: {error}"))?;
        Ok(Self { api })
    }
}

impl Drop for MediaFoundation {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.api.shutdown)();
        }
    }
}

/// Take ownership of a COM interface written through an out-pointer by one of
/// the runtime-resolved factory functions.
unsafe fn com_out<T: Interface>(hr: HRESULT, raw: *mut c_void, context: &str) -> Result<T, String> {
    hr.ok().map_err(|error| format!("{context}: {error}"))?;
    if raw.is_null() {
        return Err(format!("{context}: no interface returned"));
    }
    // SAFETY: the callee wrote one owned reference to a `T` into `raw`.
    Ok(unsafe { T::from_raw(raw) })
}

unsafe fn convert_with_source_reader(
    api: &'static MediaFoundationApi,
    input: &Path,
    output: &Path,
) -> Result<(), String> {
    let path = HSTRING::from(input.to_string_lossy().as_ref());
    let mut raw_stream: *mut c_void = std::ptr::null_mut();
    let hr = unsafe {
        (api.create_file)(
            MF_ACCESSMODE_READ,
            MF_OPENMODE_FAIL_IF_NOT_EXIST,
            MF_FILEFLAGS_NONE,
            PCWSTR(path.as_ptr()),
            &mut raw_stream,
        )
    };
    let stream: IMFByteStream =
        unsafe { com_out(hr, raw_stream, "could not open input for Media Foundation") }?;

    let mut raw_reader: *mut c_void = std::ptr::null_mut();
    let hr = unsafe {
        (api.create_source_reader_from_byte_stream)(
            stream.as_raw(),
            std::ptr::null_mut(),
            &mut raw_reader,
        )
    };
    let reader: IMFSourceReader =
        unsafe { com_out(hr, raw_reader, "Media Foundation could not open this file") }?;
    let all_streams = MF_SOURCE_READER_ALL_STREAMS.0 as u32;
    let audio_stream = MF_SOURCE_READER_FIRST_AUDIO_STREAM.0 as u32;

    unsafe {
        reader
            .SetStreamSelection(all_streams, false)
            .map_err(|error| format!("could not deselect Media Foundation streams: {error}"))?;
        reader
            .SetStreamSelection(audio_stream, true)
            .map_err(|error| format!("no audio stream for Media Foundation: {error}"))?;
    }

    let mut raw_media_type: *mut c_void = std::ptr::null_mut();
    let hr = unsafe { (api.create_media_type)(&mut raw_media_type) };
    let media_type: IMFMediaType =
        unsafe { com_out(hr, raw_media_type, "could not create PCM media type") }?;
    unsafe {
        media_type
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)
            .map_err(|error| format!("could not set audio major type: {error}"))?;
        media_type
            .SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_PCM)
            .map_err(|error| format!("could not set PCM subtype: {error}"))?;
        media_type
            .SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, TARGET_CHANNELS)
            .map_err(|error| format!("could not set channel count: {error}"))?;
        media_type
            .SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, TARGET_RATE_HZ)
            .map_err(|error| format!("could not set sample rate: {error}"))?;
        media_type
            .SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, TARGET_BITS)
            .map_err(|error| format!("could not set bit depth: {error}"))?;
        media_type
            .SetUINT32(&MF_MT_AUDIO_BLOCK_ALIGNMENT, 2)
            .map_err(|error| format!("could not set block alignment: {error}"))?;
        media_type
            .SetUINT32(&MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 32_000)
            .map_err(|error| format!("could not set byte rate: {error}"))?;
        reader
            .SetCurrentMediaType(audio_stream, None, &media_type)
            .map_err(|error| {
                format!(
                    "Windows Media Foundation cannot produce 16 kHz mono PCM16 from this file: {error}"
                )
            })?;
    }

    let mut pcm = Vec::new();
    loop {
        let mut flags = 0_u32;
        let mut sample: Option<IMFSample> = None;
        unsafe {
            reader
                .ReadSample(
                    audio_stream,
                    0,
                    None,
                    Some(&mut flags),
                    None,
                    Some(&mut sample),
                )
                .map_err(|error| format!("Media Foundation read failed: {error}"))?;
        }
        if flags & MF_SOURCE_READERF_ERROR.0 as u32 != 0 {
            return Err("Media Foundation reported a decode error".to_string());
        }
        if flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 {
            break;
        }
        let Some(sample) = sample else {
            continue;
        };
        let buffer = unsafe { sample.ConvertToContiguousBuffer() }
            .map_err(|error| format!("Media Foundation sample buffer failed: {error}"))?;
        let locked = LockedBuffer::lock(&buffer)?;
        let chunk = locked.bytes().to_vec();
        drop(locked);
        if !chunk.is_empty() {
            pcm.extend_from_slice(&chunk);
        }
    }

    if pcm.is_empty() {
        return Err("Media Foundation produced no PCM audio".to_string());
    }
    write_pcm16_wav(output, &pcm).map_err(|error| format!("could not write prepared WAV: {error}"))
}

struct LockedBuffer<'a> {
    buffer: &'a IMFMediaBuffer,
    data: *mut u8,
    length: u32,
}

impl<'a> LockedBuffer<'a> {
    fn lock(buffer: &'a IMFMediaBuffer) -> Result<Self, String> {
        let mut data: *mut u8 = std::ptr::null_mut();
        let mut length = 0_u32;
        unsafe {
            buffer
                .Lock(&mut data, None, Some(&mut length))
                .map_err(|error| format!("Media Foundation buffer lock failed: {error}"))?;
        }
        Ok(Self {
            buffer,
            data,
            length,
        })
    }

    fn bytes(&self) -> &[u8] {
        if self.data.is_null() || self.length == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.data, self.length as usize) }
        }
    }
}

impl Drop for LockedBuffer<'_> {
    fn drop(&mut self) {
        let _ = unsafe { self.buffer.Unlock() };
    }
}

fn write_pcm16_wav(path: &Path, pcm: &[u8]) -> std::io::Result<()> {
    let data_len = u32::try_from(pcm.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "decoded PCM exceeded WAV size",
        )
    })?;
    let mut bytes = Vec::with_capacity(44 + pcm.len());
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&TARGET_RATE_HZ.to_le_bytes());
    bytes.extend_from_slice(&32_000_u32.to_le_bytes());
    bytes.extend_from_slice(&2_u16.to_le_bytes());
    bytes.extend_from_slice(&16_u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    bytes.extend_from_slice(pcm);
    fs::write(path, bytes)
}
