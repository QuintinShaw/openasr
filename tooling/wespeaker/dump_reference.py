#!/usr/bin/env python3
"""Dump WeSpeaker ResNet34 fbank + embedding goldens aligned with official
WeSpeaker forward (TSTP = sqrt(var(unbiased) + 1e-7)).

Writes numpy arrays under ``--out`` (not committed). Typical layout::

    golden/jfk.wav.npy
    golden/jfk.fbank.npy
    golden/jfk.embedding.npy
    golden/synthetic_sine_mix.{wav,fbank,embedding}.npy
    manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi


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


class TSTP(nn.Module):
    """Official WeSpeaker temporal statistics pooling."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, dim, channel, frames = features.shape
        sequences = features.reshape(batch, dim * channel, frames)
        mean = sequences.mean(dim=-1)
        std = torch.sqrt(torch.var(sequences, dim=-1, unbiased=True) + 1e-7)
        return torch.cat([mean, std], dim=-1)


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
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
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
        self.pool = TSTP()
        self.seg_1 = nn.Linear(self.stats_dim * 2, embed_dim)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        x = fbank.permute(0, 2, 1).unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        stats = self.pool(out)
        return self.seg_1(stats)


def load_state_dict(path: Path) -> OrderedDict[str, torch.Tensor]:
    install_pyannote_checkpoint_stubs()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = name.removeprefix("resnet.")
        if key.endswith("num_batches_tracked") or key.startswith("projection"):
            continue
        if key.startswith("seg_2.") or key.startswith("seg_bn_1."):
            continue
        out[key] = value.detach().cpu().contiguous()
    return out


def compute_fbank(waveform: np.ndarray) -> torch.Tensor:
    wav = torch.from_numpy(waveform.astype(np.float32))[None, :] * 32768.0
    features = kaldi.fbank(
        wav,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=16000,
        window_type="hamming",
        use_energy=False,
        snip_edges=True,
        preemphasis_coefficient=0.97,
        low_freq=20.0,
        high_freq=8000.0,
        energy_floor=torch.finfo(torch.float32).eps,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--wav", action="append", default=[], type=Path)
    args = parser.parse_args(argv)

    state = load_state_dict(args.checkpoint)
    model = ResNet34()
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [name for name in unexpected if not name.startswith("projection")]
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()

    named_waveforms = synthetic_cases()
    for wav_path in args.wav:
        named_waveforms.append((wav_path.stem, read_wav(wav_path)))

    out_dir = args.out / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    with torch.no_grad():
        for name, waveform in named_waveforms:
            fbank = compute_fbank(waveform)
            embedding = model(fbank.unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float32)
            fbank_np = fbank.cpu().numpy().astype(np.float32)
            wav_np = waveform.astype(np.float32)
            np.save(out_dir / f"{name}.wav.npy", wav_np)
            np.save(out_dir / f"{name}.fbank.npy", fbank_np)
            np.save(out_dir / f"{name}.embedding.npy", embedding)
            cases.append(
                {
                    "name": name,
                    "samples": int(len(wav_np)),
                    "frames": int(fbank_np.shape[0]),
                    "dim": int(embedding.shape[0]),
                    "norm": float(np.linalg.norm(embedding)),
                }
            )
            print(
                f"{name}: samples={len(wav_np)} frames={fbank_np.shape[0]} "
                f"dim={embedding.shape[0]} norm={float(np.linalg.norm(embedding)):.6f}"
            )

    manifest = {
        "architecture": "wespeaker-resnet",
        "depth": 34,
        "pooling": "TSTP",
        "tstp_eps": 1e-7,
        "unbiased_var": True,
        "window": "hamming",
        "cases": cases,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
