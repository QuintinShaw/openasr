#!/usr/bin/env python3
"""Reference-implementation golden-transcript dumper for moss-transcribe-diarize
(OpenMOSS/MOSS-Transcribe-Diarize, 0.9B).

Runs the *official* python reference stack (not a reimplementation) --
`AutoModelForCausalLM.generate()` plus the checkpoint's own `AutoProcessor`
and `parse_transcript` helper -- on real fixture wavs, greedy/deterministic,
CPU, fp32, and dumps one JSON record per sample: full generated token ids,
decoded text, prompt token ids, and parsed speaker/timestamp segments. This
is the process that produced the committed
`crates/openasr-core/src/models/moss_transcribe_diarize/executor.rs`
`GOLDEN_JFK_TEXT` / `GOLDEN_EN_ZH_MIXED_TEXT` fixtures (and the
`tmp/moss-td/golden/*.json` records those constants were transcribed from) --
this script makes that generation process reproducible in-tree rather than
leaving it as an untracked one-off. Satisfies the "Reference dumper exists
for this family" row in `docs/model-audits/TEMPLATE.md` section 10.

Official reference source (do not vendor -- clone locally)
------------------------------------------------------------
Code:    https://github.com/OpenMOSS/MOSS-Transcribe-Diarize
         pinned commit 40cf8549c6b5634ba36b7e817cf523b5ad400c2e (2026-07-20)
Weights: https://huggingface.co/OpenMOSS/MOSS-Transcribe-Diarize (gated;
         request access, then download the standard HF layout: `config.json`,
         `configuration_moss_transcribe_diarize.py`,
         `modeling_moss_transcribe_diarize.py`,
         `processing_moss_transcribe_diarize.py`, `model-*.safetensors`,
         `model.safetensors.index.json`, `vocab.json`, `merges.txt`,
         `tokenizer.json`, `added_tokens.json`, `special_tokens_map.json`,
         `tokenizer_config.json`, `preprocessor_config.json`,
         `processor_config.json`, `generation_config.json`).

This script does NOT vendor the official repo or any weights into the repo
(both are third-party; the repo ships its own `modeling_*.py`/
`processing_*.py` alongside the checkpoint via `trust_remote_code=True`, so
no separate code clone is even required at *load* time -- `--moss-repo` only
needs to point at a checkout for its `parse_transcript` helper, which is not
part of the HF `trust_remote_code` module set).

Setup (host python, matches `tooling/publish-model` / `firered2-reference-
dumper` convention -- no dedicated venv in this repo):

    python3 -m pip install --user "transformers>=5.0,<6.0" "torch>=2.8" numpy

Usage
-----
    cd tooling/moss-reference-dumper
    python3 dump_golden.py \\
      --moss-repo /path/to/MOSS-Transcribe-Diarize \\
      --weights-dir /path/to/moss-transcribe-diarize-weights \\
      --samples-dir ../../fixtures \\
      --sample jfk=jfk.wav \\
      --out-dir /path/to/scratch/moss-td-golden

`--sample NAME=RELATIVE_WAV_PATH` may be repeated; each is resolved against
`--samples-dir`. Every sample writes `<out-dir>/<name>.json` in the exact
shape `executor.rs`'s golden tests were transcribed from (see
`GOLDEN_JFK_TEXT`'s doc comment there for the byte-level nuance: the
committed fp16+flash ggml decode matches this fp32 reference's text
byte-for-byte on `jfk`/`aishell4_multispeaker_3min` and up to a 0.02s time-
anchor shift on `en_zh_mixed`, consistent with an fp16-vs-fp32 numeric
delta rather than a bug).

Memory
------
The whole checkpoint (Whisper-Medium encoder + VQAdaptor + Qwen3-0.6B
decoder, ~1.6B params combined) fits comfortably in fp32 on a 16GB dev
machine (well under 4GB resident) -- unlike firered2-llm's 7B-parameter Qwen2
decoder, this family needs none of that dumper's meta-device layer-streaming
trick. `AutoModelForCausalLM.from_pretrained(..., dtype=torch.float32)` loads
the whole model directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import transformers


def parse_sample_arg(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--sample expects NAME=RELATIVE_WAV_PATH, got '{raw}'")
    name, _, relative_wav = raw.partition("=")
    name = name.strip()
    relative_wav = relative_wav.strip()
    if not name or not relative_wav:
        raise argparse.ArgumentTypeError(f"--sample expects NAME=RELATIVE_WAV_PATH, got '{raw}'")
    return name, relative_wav


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--moss-repo",
        type=Path,
        required=True,
        help="local clone of OpenMOSS/MOSS-Transcribe-Diarize (for parse_transcript)",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="local HF checkpoint directory (config.json, model-*.safetensors, ...)",
    )
    parser.add_argument("--samples-dir", type=Path, required=True, help="base dir for --sample wav paths")
    parser.add_argument(
        "--sample",
        dest="samples",
        action="append",
        type=parse_sample_arg,
        required=True,
        help="NAME=RELATIVE_WAV_PATH, repeatable",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    sys.path.insert(0, str(args.moss_repo))
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        prepare_inputs,
    )
    from transformers import AutoModelForCausalLM, AutoProcessor

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    dtype = torch.float32

    print(f"transformers={transformers.__version__} torch={torch.__version__}")
    print(f"loading {args.weights_dir} on CPU (fp32, deterministic)...")
    t0 = time.time()
    torch.manual_seed(args.seed)
    model = (
        AutoModelForCausalLM.from_pretrained(args.weights_dir, trust_remote_code=True, dtype=dtype)
        .to(device=device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(args.weights_dir, trust_remote_code=True)
    print(f"model loaded in {time.time() - t0:.1f}s")

    for name, relative_wav in args.samples:
        audio_path = args.samples_dir / relative_wav
        if not audio_path.exists():
            print(f"SKIP {name}: {audio_path} not found")
            continue
        print(f"\n=== {name} ({audio_path}) ===")
        torch.manual_seed(args.seed)
        messages = build_transcription_messages(str(audio_path))
        inputs = prepare_inputs(processor, messages, device=device)
        prompt_len = int(inputs["attention_mask"][0].sum().item())
        prompt_ids = inputs["input_ids"][0][:prompt_len].tolist()

        generation_config = model.generation_config
        generation_config.do_sample = False
        generation_config.max_new_tokens = args.max_new_tokens

        t0 = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"],
                audio_feature_lengths=inputs["audio_feature_lengths"],
                audio_chunk_mapping=inputs["audio_chunk_mapping"],
                generation_config=generation_config,
            )
        elapsed = time.time() - t0

        full_ids = outputs[0].tolist()
        generated_ids = full_ids[prompt_len:]
        text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        text_with_special = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)

        print(f"text:\n{text}")
        print(f"prompt_len={prompt_len} generated_tokens={len(generated_ids)} elapsed={elapsed:.1f}s")

        segments = parse_transcript(text)
        for seg in segments:
            print(f"  [{seg.start}][{seg.speaker}] {seg.text} [{seg.end}]")

        record = {
            "sample_name": name,
            "audio_path": str(audio_path),
            "device": "cpu",
            "dtype": "float32",
            "seed": args.seed,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "prompt_len": prompt_len,
            "generated_tokens": len(generated_ids),
            "elapsed_seconds": elapsed,
            "prompt_input_ids": prompt_ids,
            "generated_token_ids": generated_ids,
            "full_token_ids": full_ids,
            "text": text,
            "text_with_special_tokens": text_with_special,
            "segments": [
                {"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text}
                for s in segments
            ],
        }
        out_path = args.out_dir / f"{name}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"saved -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
