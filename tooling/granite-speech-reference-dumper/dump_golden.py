#!/usr/bin/env python3
"""Reference-implementation golden-transcript dumper for granite-speech
(ibm-granite/granite-speech-4.1-2b).

Runs the *official* python reference stack (not a reimplementation) --
`transformers`' native `GraniteSpeechForConditionalGeneration.generate()` plus
its own `AutoProcessor` / chat template -- on real fixture wavs, greedy /
deterministic, CPU, and dumps one JSON record per sample: full generated token
ids, decoded text (with and without special tokens), and prompt token ids.
This is the script intended to produce the records any future
`crates/openasr-core/src/models/granite_speech/executor.rs` `GOLDEN_*_TEXT`
constants are transcribed from (same role `../moss-reference-dumper/dump_golden.py`
and `../qwen-reference-dumper/dump_golden.py` play for their families).
Satisfies the "Reference dumper exists for this family" row in
`docs/model-audits/TEMPLATE.md` section 10.

Default question matches the ggml executor
(`GRANITE_SPEECH_DEFAULT_QUESTION` in `executor.rs`):

    can you transcribe the speech into a written format?

which the chat template expands to the same
`USER: <|audio|>...\\n ASSISTANT:` shape the executor assembles by hand. Override
with `--question` for punctuation / KWB prompt variants.

Official reference source (do not vendor -- download locally)
-------------------------------------------------------------
Modeling: ships in `transformers` natively (no separate code clone, no
          `trust_remote_code`). Supported since `transformers>=4.52.1`; the
          checkpoint records `transformers_version: "4.57.6"`. Verified against
          `transformers==5.13.0`.
Weights:  https://huggingface.co/ibm-granite/granite-speech-4.1-2b
          (open, Apache-2.0; standard HF layout, 3 shards, ~4.6 GB total --
          see README.md for the expected local file list).

This script does NOT vendor the official repo or any weights into the repo
(both are third-party). Dump outputs stay under `--out-dir` (outside the
tracked tree).

Setup (host python, matches `tooling/publish-model` / the other reference
dumpers -- no dedicated venv in this repo):

    python3 -m pip install --user \\
      "transformers>=4.52.1,<6.0" "torch>=2.8" torchaudio soundfile numpy

Usage
-----
    cd tooling/granite-speech-reference-dumper
    python3 dump_golden.py \\
      --weights-dir /path/to/granite-speech-4.1-2b-weights \\
      --samples-dir ../../fixtures \\
      --sample jfk=jfk.wav \\
      --out-dir /path/to/scratch/granite-speech-golden

`--sample NAME=RELATIVE_WAV_PATH` may be repeated; each is resolved against
`--samples-dir`. Every sample writes `<out-dir>/<name>.json`.

Greedy (`do_sample=False`, `num_beams=1`), CPU, seeded (`--seed`, default `0`)
for determinism. Default load dtype is bf16 (checkpoint native); pass
`--dtype float32` only on a machine with enough RAM (~8 GB weights alone).

Memory
------
~2B params. bf16 load fits a 16 GB dev machine; full-fp32 generate does not
comfortably. The JSON records the dtype used so a future bf16-vs-fp32
numeric nuance is auditable against any ggml golden pin.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers


# Mirrors `GRANITE_SPEECH_DEFAULT_QUESTION` in
# `crates/openasr-core/src/models/granite_speech/executor.rs`.
DEFAULT_QUESTION = "can you transcribe the speech into a written format?"
AUDIO_TOKEN = "<|audio|>"


def parse_sample_arg(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--sample expects NAME=RELATIVE_WAV_PATH, got '{raw}'"
        )
    name, _, relative_wav = raw.partition("=")
    name = name.strip()
    relative_wav = relative_wav.strip()
    if not name or not relative_wav:
        raise argparse.ArgumentTypeError(
            f"--sample expects NAME=RELATIVE_WAV_PATH, got '{raw}'"
        )
    return name, relative_wav


def parse_dtype(raw: str) -> torch.dtype:
    key = raw.strip().lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp32", "float32", "f32"):
        return torch.float32
    if key in ("fp16", "float16", "f16", "half"):
        return torch.float16
    raise argparse.ArgumentTypeError(
        f"--dtype expects bfloat16|float32|float16, got '{raw}'"
    )


def dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    return str(dtype)


def build_user_prompt(question: str) -> str:
    """Build the single user-turn content string the chat template expects.

    The HF model card and the ggml executor both put the literal `<|audio|>`
    token at the start of the user content, immediately followed by the
    free-text question (no intervening space). The chat template then wraps
    that as `USER: ...\\n ASSISTANT:`.
    """
    if AUDIO_TOKEN in question:
        # Caller already included the placeholder; do not double it.
        return question
    return f"{AUDIO_TOKEN}{question}"


def load_mono_16k_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        raise SystemExit(
            f"{path}: expected 16 kHz mono wav, got sr={sr}. "
            "Resample offline (the dumper does not resample)."
        )
    return np.asarray(wav, dtype=np.float32)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="local HF checkpoint directory (config.json, model-*.safetensors, ...)",
    )
    parser.add_argument(
        "--samples-dir", type=Path, required=True, help="base dir for --sample wav paths"
    )
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
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help=(
            "free-text question after <|audio|> (default matches the ggml "
            "executor's GRANITE_SPEECH_DEFAULT_QUESTION)"
        ),
    )
    parser.add_argument(
        "--dtype",
        type=parse_dtype,
        default=torch.bfloat16,
        help="load dtype (default bfloat16; float32 needs ~8GB weights alone)",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    if not args.weights_dir.is_dir():
        print(f"error: --weights-dir does not exist: {args.weights_dir}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    dtype = args.dtype

    print(f"transformers={transformers.__version__} torch={torch.__version__}")
    print(
        f"loading {args.weights_dir} on CPU ({dtype_name(dtype)}, deterministic)..."
    )
    t0 = time.time()
    torch.manual_seed(args.seed)

    # fix_mistral_regex: the granite tokenizer inherits a Mistral-style regex
    # that recent transformers warn about; same fix the qwen dumper uses.
    processor = AutoProcessor.from_pretrained(
        args.weights_dir, fix_mistral_regex=True
    )
    model = (
        AutoModelForSpeechSeq2Seq.from_pretrained(args.weights_dir, torch_dtype=dtype)
        .to(device=device)
        .eval()
    )
    tokenizer = processor.tokenizer
    print(f"model loaded in {time.time() - t0:.1f}s")

    user_prompt = build_user_prompt(args.question)
    chat = [{"role": "user", "content": user_prompt}]
    prompt_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    print(f"prompt_text={prompt_text!r}")

    for name, relative_wav in args.samples:
        audio_path = args.samples_dir / relative_wav
        if not audio_path.exists():
            print(f"SKIP {name}: {audio_path} not found")
            continue
        print(f"\n=== {name} ({audio_path}) ===")
        torch.manual_seed(args.seed)
        wav = load_mono_16k_wav(audio_path)
        # processor(text, audio, ...) matches the HF model-card call path.
        # Audio is passed as a 1-D float32 numpy array at 16 kHz.
        model_inputs = processor(
            prompt_text, wav, device=str(device), return_tensors="pt"
        )
        # Move every tensor to device; cast only floating-point tensors to the
        # load dtype (input_features). Id / mask tensors must stay integral.
        inputs = {}
        for key, tensor in model_inputs.items():
            if not hasattr(tensor, "to"):
                inputs[key] = tensor
                continue
            tensor = tensor.to(device)
            if torch.is_floating_point(tensor):
                tensor = tensor.to(dtype)
            inputs[key] = tensor

        prompt_len = int(inputs["input_ids"].shape[-1])
        prompt_ids = inputs["input_ids"][0].tolist()

        t0 = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
            )
        elapsed = time.time() - t0

        full_ids = outputs[0].tolist()
        generated_ids = full_ids[prompt_len:]
        text = tokenizer.batch_decode(
            [generated_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        text_with_special = tokenizer.batch_decode(
            [generated_ids], skip_special_tokens=False
        )[0]

        print(f"text:\n{text}")
        print(
            f"prompt_len={prompt_len} generated_tokens={len(generated_ids)} "
            f"elapsed={elapsed:.1f}s"
        )

        record = {
            "sample_name": name,
            "audio_path": str(audio_path),
            "device": "cpu",
            "dtype": dtype_name(dtype),
            "seed": args.seed,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "question": args.question,
            "user_prompt": user_prompt,
            "prompt_text": prompt_text,
            "prompt_len": prompt_len,
            "generated_tokens": len(generated_ids),
            "elapsed_seconds": elapsed,
            "prompt_input_ids": prompt_ids,
            "generated_token_ids": generated_ids,
            "full_token_ids": full_ids,
            "text": text,
            "text_with_special_tokens": text_with_special,
        }
        out_path = args.out_dir / f"{name}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"saved -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
