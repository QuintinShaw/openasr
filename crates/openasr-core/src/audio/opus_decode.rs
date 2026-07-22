//! Opus packet decoding via the bundled libopus C library.
//!
//! Symphonia's ogg and mkv/webm demuxers surface Opus packets but symphonia
//! has never shipped an Opus decoder (still true as of 0.6.0), so this module
//! is the decode half of the in-process Opus path: `symphonia_decode` demuxes
//! the container and feeds the packets to [`OpusStream`], which decodes them
//! with libopus (BSD-3-Clause, compiled from the source vendored inside the
//! `libopus_sys` crates.io package -- not vendored into this repository, see
//! the workspace `Cargo.toml`).
//!
//! Header handling follows RFC 7845 (Ogg Opus) and the WebM/Matroska Opus
//! conventions:
//!
//! - The `OpusHead` identification header (carried in the track's
//!   `extra_data` by both symphonia demuxers -- the full Ogg identification
//!   packet, or the Matroska `CodecPrivate`, which is the same structure)
//!   supplies the channel count, pre-skip, and output gain.
//! - Pre-skip samples (48 kHz) are discarded from the start of the decoded
//!   stream, exactly as the RFC requires of playback.
//! - Output gain (Q7.8 dB) is applied inside libopus via `OPUS_SET_GAIN`.
//! - Ogg end-trimming: the final packet's granule position (symphonia exposes
//!   it as the packet timestamp when the track's time base is 1/48000) bounds
//!   the total output length, so the encoder's final-packet padding is
//!   removed. Containers whose timestamps are not 48 kHz sample counts (e.g.
//!   mkv/webm's millisecond timecodes) skip this trim; at most one packet's
//!   worth (~120 ms) of trailing padding remains, which the ASR pipeline
//!   downstream does not care about.
//! - libopus always outputs 48 kHz; the 48k->16k resample happens in the
//!   caller like any other decoded source rate.
//!
//! Mapping families beyond plain mono/stereo (Opus multistream, >2 channels)
//! are intentionally not decoded here -- they fall back to the external
//! converter chain the same way HE-AAC does (see
//! `symphonia_decode::is_unsupported_aac_extension`), instead of risking a
//! wrong channel interpretation.
//!
//! # Untrusted input
//!
//! Packets come from third-party demuxers and may be malformed. libopus
//! validates packets and reports errors through return codes rather than
//! aborting: a corrupt packet is skipped (matching `symphonia_decode`'s
//! recoverable-`DecodeError` policy) while a hard decoder error fails the
//! whole stream closed (`None` -> `Unsupported` -> external fallback, never a
//! fabricated decode). Everything here also runs inside the caller's existing
//! `catch_unwind` guard, so even an unexpected Rust panic surfaces as
//! `ParserPanicked` instead of crashing the process.

use std::os::raw::c_int;

/// The sample rate libopus decodes at. Opus is defined to store audio at
/// 48 kHz internally regardless of the encoder's input rate, so every decoded
/// stream leaves this module at this rate (the caller resamples to 16 kHz).
pub(crate) const OPUS_DECODE_RATE_HZ: u32 = 48_000;

/// Maximum samples *per channel* a single Opus packet can decode to at
/// 48 kHz: the codec's 120 ms packet duration limit.
const MAX_FRAME_SAMPLES_PER_CHANNEL: usize = 5760;

/// The `OpusHead` magic signature, common to the Ogg identification packet
/// and the Matroska/WebM `CodecPrivate`.
const OPUS_HEAD_MAGIC: &[u8; 8] = b"OpusHead";

/// Parsed `OpusHead` identification header fields this module needs (RFC 7845
/// section 5.1; the Matroska/WebM `CodecPrivate` for `A_OPUS` is the same
/// structure).
struct OpusHead {
    /// 1 (mono) or 2 (stereo). Anything else is rejected up front -- see the
    /// multistream note in the module docs.
    channels: u16,
    /// Samples at 48 kHz to discard from the start of the decoded stream.
    pre_skip: u16,
    /// Output gain in Q7.8 dB to apply to the decoded audio.
    output_gain_q78: i16,
}

fn parse_opus_head(extra_data: Option<&[u8]>) -> Option<OpusHead> {
    let extra_data = extra_data?;
    // 8 magic + 1 version + 1 channels + 2 pre-skip + 4 input rate + 2 gain
    // + 1 mapping family = 19 bytes minimum.
    if extra_data.len() < 19 || &extra_data[..8] != OPUS_HEAD_MAGIC {
        return None;
    }
    // The version byte splits into major (high 4 bits) / minor (low 4 bits);
    // RFC 7845 says to treat any major version other than 0 as incompatible.
    if extra_data[8] >> 4 != 0 {
        return None;
    }
    let channels = u16::from(extra_data[9]);
    // 0 is invalid; >2 would need the libopus multistream API, which this
    // module deliberately does not drive -- fall back to an external
    // converter rather than guess at a channel mapping.
    if !(1..=2).contains(&channels) {
        return None;
    }
    Some(OpusHead {
        channels,
        pre_skip: u16::from_le_bytes([extra_data[10], extra_data[11]]),
        output_gain_q78: i16::from_le_bytes([extra_data[16], extra_data[17]]),
    })
}

/// RAII handle for a libopus decoder. Owns the C state exclusively and frees
/// it on drop; only ever used single-threaded inside one decode attempt.
struct OpusDecoderHandle {
    inner: *mut libopus_sys::OpusDecoder,
}

impl OpusDecoderHandle {
    /// Creates a decoder for `channels` at [`OPUS_DECODE_RATE_HZ`], applying
    /// the RFC 7845 output gain inside libopus so every decoded sample is
    /// already gain-adjusted.
    fn new(channels: u16, output_gain_q78: i16) -> Option<Self> {
        let mut error: c_int = 0;
        // SAFETY: opus_decoder_create allocates and fully initializes (or
        // returns null and reports the failure through `error`) before
        // returning; `error` is a local we hand it a valid pointer to.
        let inner = unsafe {
            libopus_sys::opus_decoder_create(
                OPUS_DECODE_RATE_HZ as c_int,
                c_int::from(channels),
                &mut error,
            )
        };
        if inner.is_null() || error != libopus_sys::OPUS_OK as c_int {
            return None;
        }
        let handle = Self { inner };
        if !handle.set_output_gain(output_gain_q78) {
            return None;
        }
        Some(handle)
    }

    fn set_output_gain(&self, output_gain_q78: i16) -> bool {
        if output_gain_q78 == 0 {
            return true;
        }
        // SAFETY: `inner` is a live decoder created above; OPUS_SET_GAIN
        // takes a single opus_int32 argument, passed here as the matching
        // C int.
        let result = unsafe {
            libopus_sys::opus_decoder_ctl(
                self.inner,
                libopus_sys::OPUS_SET_GAIN_REQUEST as c_int,
                c_int::from(output_gain_q78),
            )
        };
        result == libopus_sys::OPUS_OK as c_int
    }

    /// Decodes one packet, appending its 48 kHz f32 output (interleaved if
    /// stereo) to `out`. Returns the number of decoded *frames* (per
    /// channel), or the kind of failure seen.
    fn decode_packet(&mut self, packet: &[u8], channels: u16, out: &mut Vec<f32>) -> PacketDecode {
        // A valid Opus packet is far smaller than i32::MAX (the codec caps a
        // packet at ~61 KB); anything larger is demuxer garbage and cannot be
        // a real packet.
        let Ok(len) = c_int::try_from(packet.len()) else {
            return PacketDecode::Corrupt;
        };
        let mut scratch = [0.0_f32; MAX_FRAME_SAMPLES_PER_CHANNEL * 2];
        // SAFETY: `inner` is a live decoder created for `channels` channels;
        // `packet` is a valid slice and `len` its exact byte length; `scratch`
        // holds MAX_FRAME_SAMPLES_PER_CHANNEL frames for up to 2 channels,
        // the maximum a single 48 kHz packet can decode to, so the frame_size
        // argument can never overflow the buffer. libopus writes interleaved
        // f32 output and returns either the decoded frame count or a negative
        // error code.
        let result = unsafe {
            libopus_sys::opus_decode_float(
                self.inner,
                packet.as_ptr(),
                len,
                scratch.as_mut_ptr(),
                MAX_FRAME_SAMPLES_PER_CHANNEL as c_int,
                0,
            )
        };
        if result < 0 {
            return if result == libopus_sys::OPUS_INVALID_PACKET {
                PacketDecode::Corrupt
            } else {
                // OPUS_BAD_ARG / OPUS_INTERNAL_ERROR / OPUS_INVALID_STATE /
                // OPUS_ALLOC_FAIL (or anything else negative): the decoder
                // state can no longer be trusted to produce correct audio.
                PacketDecode::Fatal
            };
        }
        let frames = result as usize;
        out.extend_from_slice(&scratch[..frames * channels as usize]);
        PacketDecode::Frames(frames)
    }
}

impl Drop for OpusDecoderHandle {
    fn drop(&mut self) {
        // SAFETY: `inner` came from opus_decoder_create and is destroyed
        // exactly once here; the handle is the sole owner.
        unsafe { libopus_sys::opus_decoder_destroy(self.inner) };
    }
}

enum PacketDecode {
    /// Decoded this many frames (per channel) of audio.
    Frames(usize),
    /// Corrupt packet: skipped, the stream continues.
    Corrupt,
    /// Unrecoverable decoder error: the stream fails closed.
    Fatal,
}

/// The decoded result: 48 kHz mono f32 samples with the RFC 7845 pre-skip
/// removed, output gain applied, and (when the container provided
/// sample-accurate timestamps) end-trimmed -- plus the source channel count
/// before the mono downmix, for diagnostics.
pub(crate) struct OpusDecodedMono {
    pub(crate) samples: Vec<f32>,
    pub(crate) channels: u16,
}

/// Streaming Opus decoder: feed demuxed packets via [`OpusStream::push_packet`]
/// (with each packet's end timestamp, when the container tracks one), then
/// take the finished 48 kHz mono samples with [`OpusStream::finish`].
pub(crate) struct OpusStream {
    decoder: OpusDecoderHandle,
    channels: u16,
    /// Pre-skip samples still to discard from the decoded output.
    pre_skip_remaining: u64,
    /// The original pre-skip, kept to convert the Ogg granule position (which
    /// counts samples *including* the pre-skip padding) into an output length.
    pre_skip_total: u64,
    /// Whether the demuxer's packet timestamps are 48 kHz sample positions
    /// (Ogg, whose track time base is 1/48000); only then is the RFC 7845
    /// end-trim applied in [`OpusStream::finish`].
    timestamps_are_samples: bool,
    /// The latest packet end timestamp seen, when `timestamps_are_samples`.
    end_position: Option<u64>,
    /// Decoded mono samples so far (pre-skip already removed).
    samples: Vec<f32>,
    /// Scratch for one packet's decoded frames, interleaved if stereo, sized
    /// for the codec's maximum 120 ms packet; allocated once per stream.
    scratch: Vec<f32>,
}

impl OpusStream {
    /// Builds a decoder from the track's `OpusHead` (`extra_data`). Returns
    /// `None` for a missing/malformed header or a channel count this module
    /// does not decode (see the multistream note in the module docs) -- the
    /// caller falls back to the external converter chain in that case.
    ///
    /// `timestamps_are_samples` says whether the demuxer's packet timestamps
    /// are 48 kHz sample positions (true for Ogg, whose time base is
    /// 1/48000); only then is the RFC 7845 end-trim applied.
    pub(crate) fn new(extra_data: Option<&[u8]>, timestamps_are_samples: bool) -> Option<Self> {
        let head = parse_opus_head(extra_data)?;
        let decoder = OpusDecoderHandle::new(head.channels, head.output_gain_q78)?;
        Some(Self {
            decoder,
            channels: head.channels,
            pre_skip_remaining: u64::from(head.pre_skip),
            pre_skip_total: u64::from(head.pre_skip),
            timestamps_are_samples,
            end_position: None,
            samples: Vec::new(),
            scratch: Vec::with_capacity(MAX_FRAME_SAMPLES_PER_CHANNEL * head.channels as usize),
        })
    }

    /// Decodes one packet, applying the pre-skip discard and the mono
    /// downmix on the fly. `end_ts` is the packet's end timestamp from the
    /// demuxer (symphonia's `Packet::ts`); `u64::MAX` is symphonia's
    /// unknown-timestamp sentinel and is ignored. Returns `false` only on a
    /// fatal decoder error, in which case the stream must fail closed;
    /// corrupt individual packets are skipped.
    pub(crate) fn push_packet(&mut self, packet: &[u8], end_ts: u64) -> bool {
        self.scratch.clear();
        match self
            .decoder
            .decode_packet(packet, self.channels, &mut self.scratch)
        {
            PacketDecode::Fatal => return false,
            PacketDecode::Corrupt => return true,
            PacketDecode::Frames(frames) => {
                let dropped = self.pre_skip_remaining.min(frames as u64) as usize;
                self.pre_skip_remaining -= dropped as u64;
                self.push_downmixed(dropped, frames);
            }
        }
        // Granule positions accumulate across the stream, so the latest valid
        // one seen is the final end position.
        if self.timestamps_are_samples && end_ts != u64::MAX {
            self.end_position = Some(end_ts);
        }
        true
    }

    /// Downmixes `scratch[dropped..frames]` (interleaved at the stream's
    /// channel count, with the first `dropped` frames consumed by pre-skip)
    /// onto `samples` as mono.
    fn push_downmixed(&mut self, dropped: usize, frames: usize) {
        match self.channels {
            1 => self
                .samples
                .extend_from_slice(&self.scratch[dropped..frames]),
            _ => {
                self.samples.reserve(frames - dropped);
                for frame in dropped..frames {
                    let left = self.scratch[frame * 2];
                    let right = self.scratch[frame * 2 + 1];
                    self.samples.push((left + right) * 0.5);
                }
            }
        }
    }

    /// Consumes the stream and returns the finished mono samples, applying
    /// the Ogg end-trim (RFC 7845 section 4.5): when the container reported
    /// sample-accurate timestamps, the final granule position bounds the
    /// output length and the encoder's final-packet padding is cut. `None`
    /// when the stream decoded to nothing usable.
    pub(crate) fn finish(mut self) -> Option<OpusDecodedMono> {
        if let Some(granule) = self.end_position
            && let Some(expected) = granule.checked_sub(self.pre_skip_total)
            && self.samples.len() as u64 > expected
        {
            self.samples.truncate(expected as usize);
        }
        if self.samples.is_empty() {
            return None;
        }
        Some(OpusDecodedMono {
            samples: self.samples,
            channels: self.channels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal well-formed `OpusHead`: mono, 312 samples pre-skip (the
    /// typical ffmpeg/libopus encoder delay), zero output gain, mapping
    /// family 0.
    fn opus_head_mono() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(OPUS_HEAD_MAGIC);
        bytes.push(0x01); // version: major 0, minor 1
        bytes.push(1); // channels
        bytes.extend_from_slice(&312_u16.to_le_bytes()); // pre-skip
        bytes.extend_from_slice(&16_000_u32.to_le_bytes()); // input rate (informational)
        bytes.extend_from_slice(&0_i16.to_le_bytes()); // output gain
        bytes.push(0); // mapping family
        bytes
    }

    #[test]
    fn parses_a_minimal_mono_header() {
        let head = parse_opus_head(Some(&opus_head_mono())).expect("well-formed OpusHead");
        assert_eq!(head.channels, 1);
        assert_eq!(head.pre_skip, 312);
        assert_eq!(head.output_gain_q78, 0);
    }

    #[test]
    fn rejects_malformed_headers() {
        assert!(parse_opus_head(None).is_none());
        assert!(parse_opus_head(Some(b"")).is_none());
        assert!(parse_opus_head(Some(&opus_head_mono()[..18])).is_none());

        let mut bad_magic = opus_head_mono();
        bad_magic[0] = b'X';
        assert!(parse_opus_head(Some(&bad_magic)).is_none());

        let mut bad_major_version = opus_head_mono();
        bad_major_version[8] = 0x10; // major version 1: incompatible per RFC 7845
        assert!(parse_opus_head(Some(&bad_major_version)).is_none());

        let mut zero_channels = opus_head_mono();
        zero_channels[9] = 0;
        assert!(parse_opus_head(Some(&zero_channels)).is_none());

        // 3 channels means Opus multistream, which this module deliberately
        // leaves to the external converter chain.
        let mut multistream = opus_head_mono();
        multistream[9] = 3;
        assert!(parse_opus_head(Some(&multistream)).is_none());
    }

    #[test]
    fn garbage_packets_are_skipped_and_yield_no_audio() {
        let mut stream = OpusStream::new(Some(&opus_head_mono()), true)
            .expect("decoder creation must succeed for a well-formed header");

        // Corrupt packets are skipped (recoverable), not fatal -- the stream
        // simply decodes to nothing and `finish` reports no usable audio
        // instead of panicking or fabricating samples.
        assert!(stream.push_packet(b"not an opus packet at all", u64::MAX));
        assert!(stream.push_packet(&[0xFF; 64], u64::MAX));
        assert!(stream.finish().is_none());
    }

    #[test]
    fn empty_stream_finishes_to_nothing() {
        let stream = OpusStream::new(Some(&opus_head_mono()), false).expect("decoder creation");
        assert!(stream.finish().is_none());
    }
}
