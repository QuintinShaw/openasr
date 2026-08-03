#!/usr/bin/env python3
"""Dump deterministic DiariZen Large-s80-v2 stage goldens for the native port.

This tool intentionally depends on the pinned upstream DiariZen source and a
research Python environment.  Runtime code does not.  Goldens are generated
from a synthetic waveform, so no meeting/customer audio enters the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tomllib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_waveform(samples: int, sample_rate: int = 16_000) -> torch.Tensor:
    t = torch.arange(samples, dtype=torch.float64) / sample_rate
    # Deterministic, bounded, non-periodically-trivial signal. Float64 is used
    # only to construct it; the model input is the exact float32 array dumped.
    wave = (
        0.31 * torch.sin(2.0 * torch.pi * 173.0 * t)
        + 0.17 * torch.sin(2.0 * torch.pi * 421.0 * t + 0.23)
        + 0.09 * torch.cos(2.0 * torch.pi * 37.0 * t)
        + 0.03 * torch.sin(2.0 * torch.pi * (83.0 + 7.0 * t) * t)
    )
    envelope = torch.clamp(t / 0.08, max=1.0) * torch.clamp((samples / sample_rate - t) / 0.08, max=1.0)
    return (wave * envelope).to(torch.float32).unsqueeze(0).unsqueeze(0)


def build_model(diarizen_source: Path, config_path: Path) -> nn.Module:
    sys.path.insert(0, str(diarizen_source))
    from diarizen.models.module.conformer import ConformerEncoder
    from diarizen.models.module.wav2vec2.model import wav2vec2_model
    from diarizen.models.module.wavlm_config import get_config as get_wavlm_config

    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    args = dict(config["model"]["args"])

    class GoldenModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wavlm_model = wav2vec2_model(**get_wavlm_config(args["wavlm_src"]))
            self.weight_sum = nn.Linear(args["wavlm_layer_num"], 1, bias=False)
            self.proj = nn.Linear(args["wavlm_feat_dim"], args.get("attention_in", 256))
            self.lnorm = nn.LayerNorm(args.get("attention_in", 256))
            self.conformer = ConformerEncoder(
                attention_in=args.get("attention_in", 256),
                ffn_hidden=args.get("ffn_hidden", 1024),
                num_head=args.get("num_head", 4),
                num_layer=args.get("num_layer", 4),
                kernel_size=args.get("kernel_size", 31),
                dropout=args.get("dropout", 0.1),
                use_posi=args.get("use_posi", False),
                output_activate_function=args.get("output_activate_function", False),
            )
            powerset_classes = sum(
                math.comb(args["max_speakers_per_chunk"], active)
                for active in range(args["max_speakers_per_frame"] + 1)
            )
            self.classifier = nn.Linear(args.get("attention_in", 256), powerset_classes)

        def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
            representations, _ = self.wavlm_model.extract_features(waveforms[:, 0, :])
            mixed = self.weight_sum(torch.stack(representations, dim=-1)).squeeze(-1)
            hidden = self.lnorm(self.proj(mixed))
            return self.classifier(self.conformer(hidden))

    return GoldenModel()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diarizen-source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.seconds <= 0.1 or args.seconds > 16.0:
        parser.error("--seconds must be in (0.1, 16]")

    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = build_model(args.diarizen_source, args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [
        name
        for name in incompatible.unexpected_keys
        if not name.startswith(("loss_func.", "validation_metric."))
    ]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={unexpected}"
        )
    model.eval()

    captured: dict[str, np.ndarray] = {}

    def capture(name: str):
        def hook(_module, _inputs, output) -> None:
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value.detach().cpu().to(torch.float32).numpy()

        return hook

    def capture_input(name: str):
        def hook(_module, inputs) -> None:
            captured[name] = inputs[0].detach().cpu().to(torch.float32).numpy()

        return hook

    handles = [
        model.wavlm_model.feature_extractor.register_forward_hook(capture("wavlm_feature_extractor")),
        model.wavlm_model.encoder.feature_projection.register_forward_hook(capture("wavlm_feature_projection")),
        model.wavlm_model.encoder.transformer.pos_conv_embed.register_forward_hook(capture("wavlm_positional_conv")),
        model.weight_sum.register_forward_hook(capture("weighted_layer_sum_raw")),
        model.proj.register_forward_hook(capture("projection_raw")),
        model.lnorm.register_forward_hook(capture("projection_norm")),
        model.classifier.register_forward_hook(capture("logits")),
    ]
    for index, layer in enumerate(model.wavlm_model.encoder.transformer.layers):
        if index == 0:
            handles.append(
                layer.register_forward_pre_hook(capture_input("wavlm_transformer_preprocessed"))
            )
        handles.append(layer.register_forward_hook(capture(f"wavlm_layer_{index:02d}")))
    for index, layer in enumerate(model.conformer.conformer_layer):
        handles.append(layer.register_forward_hook(capture(f"conformer_layer_{index:02d}")))

    waveform = synthetic_waveform(round(args.seconds * 16_000))
    with torch.inference_mode():
        logits = model(waveform)
        powerset_class = logits.argmax(dim=-1)
    for handle in handles:
        handle.remove()

    captured["waveform"] = waveform.numpy()
    captured["powerset_class"] = powerset_class.cpu().numpy().astype(np.int64)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "diarizen_large_s80_v2_golden.npz", **captured)
    metadata = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "config_sha256": sha256(args.config),
        "synthetic_seconds": args.seconds,
        "sample_rate": 16_000,
        "torch_version": torch.__version__,
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(captured.items())
        },
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.out_dir / "diarizen_large_s80_v2_golden.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
