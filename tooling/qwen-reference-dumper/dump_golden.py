#!/usr/bin/env python3
"""Reference-implementation golden-transcript dumper for qwen3-asr
(Qwen/Qwen3-ASR-0.6B).

Runs the *official* python reference stack (not a reimplementation) -- the
`qwen_asr` package's transformers backend, i.e. its registered
`Qwen3ASRForConditionalGeneration.generate()` plus its own `Qwen3ASRProcessor`
chat template and `parse_asr_output` helper -- on real fixture wavs, greedy /
deterministic, CPU, fp32, and dumps one JSON record per sample: full generated
token ids, decoded text (with and without special tokens), prompt token ids,
and the parsed `language` + transcript `asr_text`. This is the script intended
to produce the records the committed
`crates/openasr-core/src/models/qwen/ggml_executor.rs` `GOLDEN_*_TEXT`
constants are transcribed from (the qwen family does not yet pin golden
transcripts in-tree -- this dumper makes that generation reproducible rather
than leaving it as an untracked one-off, exactly the role
`../moss-reference-dumper/dump_golden.py` plays for moss-transcribe-diarize).
Satisfies the "Reference dumper exists for this family" row in
`docs/model-audits/TEMPLATE.md` section 10.

Which field is the future golden constant? The ggml executor strips the
`language <name><asr_text>` control prefix from the decoded token stream (see
`greedy_decode_strips_qwen_asr_control_prefix` in `greedy_decode.rs`) and emits
just the transcript. That transcript is byte-for-byte what `parse_asr_output`
returns as its second element, so the `asr_text` JSON field below -- not the raw
`text` field, which still carries the control prefix -- is the one a future
`GOLDEN_*_TEXT` constant should match. Both are recorded so the relationship is
auditable.

Official reference source (do not vendor -- clone locally)
-----------------------------------------------------------
Code:    https://github.com/QwenLM/Qwen3-ASR
         pinned commit 7c6daf77a2421100f5fb066495372c00129d39ff (2026-06-26).
         `import qwen_asr` from that clone registers the family's config /
         model / processor with transformers' Auto classes (the checkpoint
         ships no `modeling_*.py` of its own, so there is nothing to
         `trust_remote_code` -- the modeling lives in the pip-installable
         `qwen_asr` package, unlike moss whose modeling ships in the checkpoint
         via `trust_remote_code=True`).
Weights: https://huggingface.co/Qwen/Qwen3-ASR-0.6B (open, Apache-2.0; download
         the standard HF layout: `config.json`, `model.safetensors` (single
         shard, ~1.2GB fp16 on disk, loaded fp32 here), `generation_config.json`,
         `chat_template.json`, `preprocessor_config.json`,
         `tokenizer_config.json`, `special_tokens_map.json` (if present),
         `vocab.json`, `merges.txt`).

This script does NOT vendor the official repo or any weights into the repo
(both are third-party).

Setup (host python, matches `tooling/publish-model` /
`firered2-reference-dumper` / `moss-reference-dumper` convention -- no dedicated
venv in this repo). NOTE the transformers pin differs from moss: `qwen_asr`
requires `transformers==4.57.6` (its `AutoConfig.register("qwen3_asr", ...)`
would collide with the model type transformers later merged natively), so pin
that exact version rather than moss's `transformers>=5.0`:

    python3 -m pip install --user "transformers==4.57.6" "torch>=2.8" numpy \\
        librosa soundfile nagisa

(`librosa`/`soundfile` come from `qwen_asr.inference.utils`' audio loading,
`nagisa` from its forced-aligner module import -- both are pulled in by the
plain `import qwen_asr` even though this dumper never runs the aligner.
`soynlp`, `gradio`, `flask`, `vllm` are NOT needed here. Alternatively,
`python3 -m pip install --user -e /path/to/Qwen3-ASR` installs the official
full dependency set, including the gradio/flask the dumper does not use.)

Usage
-----
    cd tooling/qwen-reference-dumper
    python3 dump_golden.py \\
      --qwen-repo /path/to/Qwen3-ASR \\
      --weights-dir /path/to/qwen3-asr-0.6b-weights \\
      --samples-dir ../../fixtures \\
      --sample jfk=jfk.wav \\
      --out-dir /path/to/scratch/qwen-golden

`--sample NAME=RELATIVE_WAV_PATH` may be repeated; each is resolved against
`--samples-dir`. Every sample writes `<out-dir>/<name>.json` (prompt token ids,
full/generated token ids, decoded text with and without special tokens, parsed
language + transcript, elapsed time, environment versions).

Greedy (`do_sample=False`), CPU, fp32, seeded (`--seed`, default `0`) for
determinism -- matches the golden-diff convention every other builtin family's
dev-only reference dump uses (see `../moss-reference-dumper/dump_golden.py` and
`../firered2-reference-dumper/dump_reference.py`'s `llm` stage).

Memory
------
The whole checkpoint (~0.6B params: 18-layer Whisper-style audio encoder +
audio projector + 28-layer Qwen3 text decoder) fits comfortably in fp32 on a
16GB dev machine (well under 4GB resident) -- unlike firered2-llm's 7B-parameter
Qwen2 decoder, this family needs none of that dumper's meta-device
layer-streaming trick. `AutoModel.from_pretrained(..., dtype=torch.float32)`
loads the whole model directly.
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
        "--qwen-repo",
        type=Path,
        required=True,
        help="local clone of QwenLM/Qwen3-ASR (provides the `qwen_asr` package)",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="local HF checkpoint directory (config.json, model.safetensors, ...)",
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
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser


def build_transcription_messages() -> list[dict]:
    """Mirror `Qwen3ASRModel._build_messages` (empty system context, one audio)."""
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]


def main() -> int:
    args = build_arg_parser().parse_args()
    sys.path.insert(0, str(args.qwen_repo))
    # `import qwen_asr` registers Qwen3ASRConfig / Qwen3ASRForConditionalGeneration
    # / Qwen3ASRProcessor with transformers' Auto classes, so AutoModel /
    # AutoProcessor below resolve the family off the plain checkpoint dir.
    import qwen_asr  # noqa: F401
    from qwen_asr.inference.utils import normalize_audios, parse_asr_output
    from transformers import AutoModel, AutoProcessor

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    dtype = torch.float32

    print(f"transformers={transformers.__version__} torch={torch.__version__}")
    print(f"loading {args.weights_dir} on CPU (fp32, deterministic)...")
    t0 = time.time()
    torch.manual_seed(args.seed)
    model = (
        AutoModel.from_pretrained(args.weights_dir, dtype=dtype).to(device=device).eval()
    )
    processor = AutoProcessor.from_pretrained(args.weights_dir, fix_mistral_regex=True)
    print(f"model loaded in {time.time() - t0:.1f}s")

    prompt_text = processor.apply_chat_template(
        build_transcription_messages(), add_generation_prompt=True, tokenize=False
    )

    for name, relative_wav in args.samples:
        audio_path = args.samples_dir / relative_wav
        if not audio_path.exists():
            print(f"SKIP {name}: {audio_path} not found")
            continue
        print(f"\n=== {name} ({audio_path}) ===")
        torch.manual_seed(args.seed)
        wav = normalize_audios(str(audio_path))[0]
        batch = processor(text=[prompt_text], audio=[wav], return_tensors="pt", padding=True)
        # Cast only floating-point tensors (input_features) to fp32; the id /
        # mask tensors must stay integral. (BatchFeature.to(dtype) already skips
        # ints on recent transformers, but be explicit rather than rely on it.)
        inputs = {
            key: (tensor.to(device).to(dtype) if torch.is_floating_point(tensor) else tensor.to(device))
            for key, tensor in batch.items()
        }
        prompt_len = int(inputs["input_ids"].shape[1])
        prompt_ids = inputs["input_ids"][0].tolist()

        t0 = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
        elapsed = time.time() - t0

        full_ids = outputs.sequences[0].tolist()
        generated_ids = full_ids[prompt_len:]
        text = processor.batch_decode(
            [generated_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        text_with_special = processor.batch_decode([generated_ids], skip_special_tokens=False)[0]

        # `parse_asr_output` splits the raw "language <name><asr_text>..." decode
        # into (language, transcript); its transcript is exactly what the ggml
        # executor emits after stripping the control prefix, so it is the future
        # golden-constant field.
        language, asr_text = parse_asr_output(text)

        print(f"language: {language}")
        print(f"asr_text:\n{asr_text}")
        print(f"prompt_len={prompt_len} generated_tokens={len(generated_ids)} elapsed={elapsed:.1f}s")

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
            "language": language,
            "asr_text": asr_text,
        }
        out_path = args.out_dir / f"{name}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"saved -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
