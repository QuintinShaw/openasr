#!/usr/bin/env python3
"""Generate a deterministic longform pause probe from a local speech WAV."""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_WAV = REPO_ROOT / "tmp/audio/clips/black_cat_poe_ty_5min.wav"
DEFAULT_OUTPUT_WAV = REPO_ROOT / "tmp/audio/generated/black_cat_pause_probe_126s.wav"

# Keep spans within a relatively local narrative region so the probe exercises
# pause handling without introducing extreme semantic jumps between islands.
SOURCE_SPANS = [
    (0.0, 12.0),
    (14.0, 27.0),
    (30.0, 44.0),
    (48.0, 60.0),
    (63.0, 81.0),
    (86.0, 99.0),
    (103.0, 118.0),
    (122.0, 137.0),
]

PAUSE_PLAN = [
    {"kind": "room_tone", "seconds": 1.0, "amplitude": 0.010},
    {"kind": "noisy_edge_silence", "seconds": 3.5, "edge_seconds": 0.35, "edge_amplitude": 0.040},
    {"kind": "silence", "seconds": 6.0},
    {"kind": "room_tone", "seconds": 1.8, "amplitude": 0.008},
    {"kind": "room_tone", "seconds": 4.0, "amplitude": 0.006},
    {"kind": "noisy_edge_silence", "seconds": 4.5, "edge_seconds": 0.45, "edge_amplitude": 0.035},
    {"kind": "room_tone", "seconds": 2.2, "amplitude": 0.012},
]


def read_wav_pcm16(path: Path) -> tuple[list[int], int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        if channels != 1 or sample_width != 2:
            raise ValueError(
                f"expected mono PCM16 WAV, got channels={channels} sample_width={sample_width}"
            )
        frames = handle.readframes(handle.getnframes())
    samples = list(struct.unpack("<" + "h" * (len(frames) // 2), frames))
    return samples, sample_rate


def write_wav_pcm16(path: Path, sample_rate: int, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = struct.pack("<" + "h" * len(samples), *samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return max(0, int(round(seconds * sample_rate)))


def clamp_pcm16(value: float) -> int:
    return max(-32768, min(32767, int(round(value))))


def extract_span(samples: list[int], sample_rate: int, start_s: float, end_s: float) -> list[int]:
    start = seconds_to_samples(start_s, sample_rate)
    end = seconds_to_samples(end_s, sample_rate)
    if not (0 <= start < end <= len(samples)):
        raise ValueError(f"invalid source span {start_s}-{end_s}")
    return samples[start:end]


def synth_room_tone(length: int, amplitude: float, rng: random.Random) -> list[int]:
    out: list[int] = []
    prev = 0.0
    for index in range(length):
        white = rng.uniform(-1.0, 1.0)
        low = 0.985 * prev + 0.015 * white
        prev = low
        flutter = 0.30 * math.sin((index / 97.0) * math.pi * 2.0)
        out.append(clamp_pcm16((low + flutter) * amplitude * 32767.0))
    return out


def synth_pause(spec: dict[str, float | str], sample_rate: int, rng: random.Random) -> list[int]:
    kind = str(spec["kind"])
    total = seconds_to_samples(float(spec["seconds"]), sample_rate)
    if kind == "silence":
        return [0] * total
    if kind == "room_tone":
        return synth_room_tone(total, float(spec["amplitude"]), rng)
    if kind == "noisy_edge_silence":
        edge = min(total // 2, seconds_to_samples(float(spec["edge_seconds"]), sample_rate))
        center = max(0, total - edge * 2)
        edge_amp = float(spec["edge_amplitude"])
        return (
            synth_room_tone(edge, edge_amp, rng)
            + [0] * center
            + synth_room_tone(edge, edge_amp, rng)
        )
    raise ValueError(f"unsupported pause kind '{kind}'")


def build_probe(source_samples: list[int], sample_rate: int) -> tuple[list[int], dict[str, object]]:
    rng = random.Random(20260528)
    output: list[int] = []
    provenance_segments: list[dict[str, object]] = []
    for index, span in enumerate(SOURCE_SPANS):
        chunk = extract_span(source_samples, sample_rate, span[0], span[1])
        output.extend(chunk)
        provenance_segments.append(
            {
                "kind": "speech",
                "source_start_seconds": span[0],
                "source_end_seconds": span[1],
                "duration_seconds": round(len(chunk) / sample_rate, 3),
            }
        )
        if index < len(PAUSE_PLAN):
            pause = synth_pause(PAUSE_PLAN[index], sample_rate, rng)
            output.extend(pause)
            provenance_segments.append(
                {
                    "kind": str(PAUSE_PLAN[index]["kind"]),
                    "duration_seconds": round(len(pause) / sample_rate, 3),
                    **{
                        key: value
                        for key, value in PAUSE_PLAN[index].items()
                        if key not in {"kind", "seconds"}
                    },
                }
            )
    metadata = {
        "source_wav": str(DEFAULT_SOURCE_WAV),
        "sample_rate": sample_rate,
        "total_duration_seconds": round(len(output) / sample_rate, 3),
        "source_spans": SOURCE_SPANS,
        "pause_plan": PAUSE_PLAN,
        "timeline": provenance_segments,
    }
    return output, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-wav", type=Path, default=DEFAULT_SOURCE_WAV)
    parser.add_argument("--output-wav", type=Path, default=DEFAULT_OUTPUT_WAV)
    parser.add_argument("--metadata-json", type=Path, default=None)
    args = parser.parse_args()

    source_samples, sample_rate = read_wav_pcm16(args.source_wav)
    output_samples, metadata = build_probe(source_samples, sample_rate)
    metadata["source_wav"] = str(args.source_wav)
    metadata["output_wav"] = str(args.output_wav)

    write_wav_pcm16(args.output_wav, sample_rate, output_samples)
    if args.metadata_json is not None:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
