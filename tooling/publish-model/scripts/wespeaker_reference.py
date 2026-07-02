#!/usr/bin/env python3
"""Generate WeSpeaker ResNet34 reference artifacts for the Rust embedder.

Inputs are the Hugging Face `pyannote/wespeaker-voxceleb-resnet34-LM`
`pytorch_model.bin` checkpoint and optional WAV files. The script reconstructs
the pyannote.audio 3.1.1 WeSpeakerResNet34 forward path:

  waveform * 32768
  torchaudio.compliance.kaldi.fbank(..., window_type="hamming", dither=0)
  per-utterance CMN
  ResNet34 [3,4,6,3] + TSTP + seg_1

It writes:
  * a clean F32 `.safetensors` source file for `openasr model-pack
    import wespeaker`
  * a compact Rust-readable golden (`WSR1`) with waveform, fbank, and embedding
  * an optional `.npz` stage dump for debugging conv/pool parity

The checkpoint contains pyannote metadata classes. To avoid requiring
pyannote.audio just to read the state dict, this script registers small dummy
classes with the same module names before `torch.load`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi


SOURCE_NAME = "pyannote/wespeaker-voxceleb-resnet34-LM"
SOURCE_REVISION = "837717ddb9ff5507820346191109dc79c958d614"
LICENSE_NAME = "CC-BY-4.0"


def install_pyannote_checkpoint_stubs() -> None:
    for name in [
        "pyannote",
        "pyannote.audio",
        "pyannote.audio.core",
        "pyannote.audio.core.task",
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))
    module = sys.modules["pyannote.audio.core.task"]

    class Specifications:
        def __new__(cls, *args, **kwargs):
            obj = object.__new__(cls)
            obj.args = args
            obj.kwargs = kwargs
            return obj

    class Problem:
        def __new__(cls, *args, **kwargs):
            obj = object.__new__(cls)
            obj.args = args
            obj.kwargs = kwargs
            return obj

    class Resolution:
        def __new__(cls, *args, **kwargs):
            obj = object.__new__(cls)
            obj.args = args
            obj.kwargs = kwargs
            return obj

    for cls in (Specifications, Problem, Resolution):
        cls.__module__ = "pyannote.audio.core.task"
    module.Specifications = Specifications
    module.Problem = Problem
    module.Resolution = Resolution


class StatsPool(nn.Module):
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        mean = sequences.mean(dim=-1)
        std = sequences.std(dim=-1, correction=1)
        return torch.cat([mean, std], dim=-1)


class TSTP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.stats_pool = StatsPool()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, dim, channel, frames = features.shape
        features = features.reshape(batch, dim * channel, frames)
        return self.stats_pool(features)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet34(nn.Module):
    def __init__(self, feat_dim: int = 80, embed_dim: int = 256, m_channels: int = 32):
        super().__init__()
        self.in_planes = m_channels
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(m_channels, 3, stride=1)
        self.layer2 = self._make_layer(m_channels * 2, 4, stride=2)
        self.layer3 = self._make_layer(m_channels * 4, 6, stride=2)
        self.layer4 = self._make_layer(m_channels * 8, 3, stride=2)
        self.pool = TSTP(self.stats_dim)
        self.seg_1 = nn.Linear(self.stats_dim * 2, embed_dim)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_with_stages(self, fbank: torch.Tensor) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
        stages: OrderedDict[str, torch.Tensor] = OrderedDict()
        x = fbank.permute(0, 2, 1).unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        stages["conv1"] = out.detach().cpu()
        out = self.layer1(out)
        stages["layer1"] = out.detach().cpu()
        out = self.layer2(out)
        stages["layer2"] = out.detach().cpu()
        out = self.layer3(out)
        stages["layer3"] = out.detach().cpu()
        out = self.layer4(out)
        stages["layer4"] = out.detach().cpu()
        stats = self.pool(out)
        stages["pool"] = stats.detach().cpu()
        embedding = self.seg_1(stats)
        stages["embedding"] = embedding.detach().cpu()
        return embedding, stages

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        return self.forward_with_stages(fbank)[0]


def load_state_dict(path: Path) -> OrderedDict[str, torch.Tensor]:
    install_pyannote_checkpoint_stubs()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    return OrderedDict(
        (name, value.detach().cpu().contiguous())
        for name, value in state.items()
        if isinstance(value, torch.Tensor) and value.dtype == torch.float32
    )


def compute_fbank(waveform: np.ndarray) -> torch.Tensor:
    wav = torch.from_numpy(waveform.astype(np.float32))[None, :] * (1 << 15)
    features = kaldi.fbank(
        wav,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=16000,
        window_type="hamming",
        use_energy=False,
    )
    return features - torch.mean(features, dim=0, keepdim=True)


def read_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 16000:
        raise SystemExit(f"{path}: expected 16 kHz, got {sample_rate}")
    return data.mean(axis=1).astype(np.float32)


def synthetic_cases() -> list[tuple[str, np.ndarray]]:
    sr = 16000
    t1 = np.arange(int(2.2 * sr), dtype=np.float32) / sr
    sine_mix = 0.12 * np.sin(2 * math.pi * 220 * t1)
    sine_mix += 0.07 * np.sin(2 * math.pi * 440 * t1 + 0.2)

    t2 = np.arange(int(3.1 * sr), dtype=np.float32) / sr
    chirp_phase = 2 * math.pi * (130 * t2 + 0.5 * 170 * t2 * t2 / t2[-1])
    chirp = 0.10 * np.sin(chirp_phase)
    chirp *= np.linspace(0.35, 1.0, len(chirp), dtype=np.float32)

    rng = np.random.default_rng(20260611)
    noise = rng.normal(0.0, 0.015, int(2.4 * sr)).astype(np.float32)
    noise += 0.04 * np.sin(2 * math.pi * 165 * np.arange(len(noise), dtype=np.float32) / sr)

    return [
        ("synthetic_sine_mix", sine_mix.astype(np.float32)),
        ("synthetic_chirp", chirp.astype(np.float32)),
        ("synthetic_voiced_noise", noise.astype(np.float32)),
    ]


def write_safetensors(
    path: Path, tensors: OrderedDict[str, torch.Tensor], checkpoint_sha256: str
) -> None:
    header: dict[str, object] = {
        "__metadata__": {
            "source_name": SOURCE_NAME,
            "source_revision": SOURCE_REVISION,
            "license": LICENSE_NAME,
            "checkpoint_sha256": checkpoint_sha256,
        }
    }
    blob = bytearray()
    for name in sorted(tensors):
        arr = tensors[name].numpy().astype(np.float32, copy=False)
        data = arr.tobytes(order="C")
        start = len(blob)
        blob.extend(data)
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [start, len(blob)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(blob)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_string(handle, value: str) -> None:
    encoded = value.encode("utf-8")
    handle.write(struct.pack("<I", len(encoded)))
    handle.write(encoded)


def write_golden(path: Path, cases: list[dict[str, np.ndarray]], checkpoint_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"WSR1")
        handle.write(struct.pack("<I", len(cases)))
        write_string(handle, SOURCE_NAME)
        write_string(handle, SOURCE_REVISION)
        write_string(handle, checkpoint_sha256)
        for case in cases:
            name = case["name"].encode("utf-8")
            wav = np.ascontiguousarray(case["wav"], dtype=np.float32)
            fbank = np.ascontiguousarray(case["fbank"], dtype=np.float32)
            embedding = np.ascontiguousarray(case["embedding"], dtype=np.float32)
            frames = fbank.shape[0]
            handle.write(struct.pack("<I", len(name)))
            handle.write(name)
            handle.write(struct.pack("<III", len(wav), frames, len(embedding)))
            handle.write(wav.tobytes(order="C"))
            handle.write(fbank.tobytes(order="C"))
            handle.write(embedding.tobytes(order="C"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--safetensors-out", type=Path)
    parser.add_argument("--golden-out", required=True, type=Path)
    parser.add_argument("--stages-out", type=Path)
    parser.add_argument("--wav", action="append", default=[], type=Path)
    args = parser.parse_args(argv)

    checkpoint_sha256 = file_sha256(args.checkpoint)
    state = load_state_dict(args.checkpoint)
    if args.safetensors_out is not None:
        write_safetensors(args.safetensors_out, state, checkpoint_sha256)

    model = ResNet34()
    model_state = OrderedDict(
        (name.removeprefix("resnet."), value) for name, value in state.items()
    )
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()

    named_waveforms = synthetic_cases()
    for wav_path in args.wav:
        named_waveforms.append((wav_path.stem, read_wav(wav_path)))

    cases: list[dict[str, np.ndarray]] = []
    stage_dump: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, waveform in named_waveforms:
            fbank = compute_fbank(waveform)
            embedding, stages = model.forward_with_stages(fbank.unsqueeze(0))
            emb = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
            cases.append(
                {
                    "name": name,
                    "wav": waveform.astype(np.float32),
                    "fbank": fbank.detach().cpu().numpy().astype(np.float32),
                    "embedding": emb,
                }
            )
            for stage_name, value in stages.items():
                stage_dump[f"{name}/{stage_name}"] = value.numpy().astype(np.float32)
            norm = float(np.linalg.norm(emb))
            print(f"{name}: samples={len(waveform)} frames={fbank.shape[0]} dim={len(emb)} norm={norm:.6f}")

    write_golden(args.golden_out, cases, checkpoint_sha256)
    if args.stages_out is not None:
        args.stages_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.stages_out, **stage_dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
