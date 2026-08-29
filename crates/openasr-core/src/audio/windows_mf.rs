//! Windows Media Foundation converter for formats the in-process decoder
//! refuses (HE-AAC SBR/PS, and other containers the OS can decode).
//!
//! This is the Windows peer of macOS `/usr/bin/afconvert`: system APIs only,
//! never Fraunhofer FDK, never a spawned ffmpeg.

use std::fs;
use std::path::Path;

use windows::Win32::Media::MediaFoundation::{
    IMFMediaBuffer, IMFMediaType, IMFSample, IMFSourceReader, MF_ACCESSMODE_READ,
    MF_FILEFLAGS_NONE, MF_MT_AUDIO_AVG_BYTES_PER_SECOND, MF_MT_AUDIO_BITS_PER_SAMPLE,
    MF_MT_AUDIO_BLOCK_ALIGNMENT, MF_MT_AUDIO_NUM_CHANNELS, MF_MT_AUDIO_SAMPLES_PER_SECOND,
    MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE, MF_OPENMODE_FAIL_IF_NOT_EXIST, MF_SOURCE_READER_ALL_STREAMS,
    MF_SOURCE_READER_FIRST_AUDIO_STREAM, MF_SOURCE_READERF_ENDOFSTREAM, MF_SOURCE_READERF_ERROR,
    MF_VERSION, MFAudioFormat_PCM, MFCreateFile, MFCreateMediaType,
    MFCreateSourceReaderFromByteStream, MFMediaType_Audio, MFSTARTUP_NOSOCKET, MFShutdown,
    MFStartup,
};
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoUninitialize};
use windows::core::{HRESULT, HSTRING};

/// `RPC_E_CHANGED_MODE`: this thread already has a different COM apartment.
const RPC_E_CHANGED_MODE: HRESULT = HRESULT(0x8001_0106_u32 as i32);

const TARGET_RATE_HZ: u32 = 16_000;
const TARGET_CHANNELS: u32 = 1;
const TARGET_BITS: u32 = 16;

pub(super) fn convert_to_wav16k_mono(input: &Path, output: &Path) -> Result<(), String> {
    let _com = ComApartment::enter()?;
    let _mf = MediaFoundation::startup()?;
    unsafe { convert_with_source_reader(input, output) }
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

struct MediaFoundation;

impl MediaFoundation {
    fn startup() -> Result<Self, String> {
        unsafe { MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET) }
            .map_err(|error| format!("Media Foundation startup failed: {error}"))?;
        Ok(Self)
    }
}

impl Drop for MediaFoundation {
    fn drop(&mut self) {
        unsafe {
            let _ = MFShutdown();
        }
    }
}

unsafe fn convert_with_source_reader(input: &Path, output: &Path) -> Result<(), String> {
    let path = HSTRING::from(input.to_string_lossy().as_ref());
    let stream = unsafe {
        MFCreateFile(
            MF_ACCESSMODE_READ,
            MF_OPENMODE_FAIL_IF_NOT_EXIST,
            MF_FILEFLAGS_NONE,
            &path,
        )
    }
    .map_err(|error| format!("could not open input for Media Foundation: {error}"))?;

    let reader: IMFSourceReader = unsafe { MFCreateSourceReaderFromByteStream(&stream, None) }
        .map_err(|error| format!("Media Foundation could not open this file: {error}"))?;
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

    let media_type: IMFMediaType = unsafe { MFCreateMediaType() }
        .map_err(|error| format!("could not create PCM media type: {error}"))?;
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
