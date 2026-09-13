#!/usr/bin/env python3
"""Dump the official Moonshine conv-stem normalization for an offline checkpoint.

Requires torch, transformers and soundfile. Outputs model-derived tensors: keep
the JSON outside version control. The ignored moonshine_conv_stem_real_reference
test consumes it via OPENASR_MOONSHINE_STEM_REFERENCE (CPU or Metal).
"""

import argparse
import hashlib
import json
from pathlib import Path

import soundfile
import torch
import transformers
from transformers import MoonshineForConditionalGeneration


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audio, rate = soundfile.read(args.audio, dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        parser.error("audio must be 16 kHz mono")
    torch.set_num_threads(1)
    model = MoonshineForConditionalGeneration.from_pretrained(
        args.checkpoint, local_files_only=True
    ).eval()
    encoder = model.model.encoder
    with torch.inference_mode():
        state = torch.tanh(encoder.conv1(torch.from_numpy(audio)[None, None]))
        expected = encoder.groupnorm(state)
    result = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "checkpoint_sha256": sha256(args.checkpoint / "model.safetensors"),
        "audio_sha256": sha256(args.audio),
        "frames": state.shape[2],
        "channels": state.shape[1],
        "input": state.flatten().tolist(),
        "weight": encoder.groupnorm.weight.tolist(),
        "bias": encoder.groupnorm.bias.tolist(),
        "expected": expected.flatten().tolist(),
    }
    args.output.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {result['channels']} channels x {result['frames']} frames")


if __name__ == "__main__":
    main()
