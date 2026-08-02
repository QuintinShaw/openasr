#!/usr/bin/env python3
"""Reference-implementation intermediate-activation dumper for granite-speech
(ibm-granite/granite-speech-4.1-2b).

Runs the *official* python reference stack (`transformers`' native
`GraniteSpeechForConditionalGeneration`) on one fixture wav and dumps every
stage the ggml side diffs against:

  mel frontend -> Conformer encoder (mid-layer tap + final) -> Q-Former
  projector -> (optional) full audio-spliced prefill last logits

plus a pure text-only decoder prefill on a fixed 12-token id sequence so the
decoder graph can be parity-checked independent of the audio splice. Output
file names match what
`crates/openasr-core/src/models/granite_speech/parity.rs` loads today
(`{sample}_input_features.npy`, `{sample}_encoder_mid_block_out.npy`,
`{sample}_encoder_out.npy`, `{sample}_projector_out.npy`,
`decoder_input_ids.npy`, `decoder_embed_scaled.npy`, `decoder_hidden_out.npy`,
`decoder_logits.npy`), so this script is the in-tree replacement for the
out-of-tree `tmp/granite-work/dump_{encoder,decoder}_golden.py` one-offs that
originally produced those fixtures.

The greedy-decode stage of the full pipeline is NOT dumped here; it is
`dump_golden.py`'s full `model.generate()` call (mirrors moss/qwen, whose
`dump_intermediate.py` likewise stops at the prefill logits).

Official reference source (do not vendor -- download locally)
-------------------------------------------------------------
Modeling: ships in `transformers` natively (see `dump_golden.py` module doc).
Weights:  https://huggingface.co/ibm-granite/granite-speech-4.1-2b
          (see README.md for the expected local layout).

Setup: same as `dump_golden.py`:

    python3 -m pip install --user \\
      "transformers>=4.52.1,<6.0" "torch>=2.8" torchaudio soundfile numpy

Usage
-----
    cd tooling/granite-speech-reference-dumper
    python3 dump_intermediate.py \\
      --weights-dir /path/to/granite-speech-4.1-2b-weights \\
      --wav /path/to/en_short.wav \\
      --sample-name en_short \\
      --out-dir /path/to/scratch/granite-speech-dump

Dumps, into `--out-dir` (see README.md for the full shape table). Pass
`--sample-name en_short` to regenerate the exact fixture names `parity.rs`
loads. Pass `--skip-decoder` / `--skip-audio-prefill` to drop the pure-decoder
or full-prompt prefill stages when only the encoder/projector fixtures are
needed (saves the LM fp32 upcast).

Memory
------
Loads the checkpoint in bf16, then upcasts only the stage under test to fp32
(encoder + projector for the audio path; `language_model` for the pure
decoder path) -- same convention the original out-of-tree dump scripts and
the ggml parity harness use ("diff against an f32 PyTorch reference").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Reuse the golden dumper's prompt / wav helpers so the two scripts cannot
# drift on the chat-template shape or the default question.
from dump_golden import (
    AUDIO_TOKEN,
    DEFAULT_QUESTION,
    build_user_prompt,
    dtype_name,
    load_mono_16k_wav,
)


# Fixed pure-decoder probe: 12 arbitrary in-vocab token ids, well below
# vocab_size=100353 and avoiding special tokens. Identical to the sequence
# the original `tmp/granite-work/dump_decoder_golden.py` used, so regenerating
# fixtures stays bit-comparable to the existing parity goldens.
DEFAULT_DECODER_INPUT_IDS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]


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
    parser.add_argument("--wav", type=Path, required=True, help="16 kHz mono wav")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory for .npy dumps (created if missing; not committed)",
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        default=None,
        help="prefix for audio-path fixtures (default: wav stem). "
        "Use 'en_short' to match parity.rs fixture names.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="free-text question after <|audio|> for the audio-spliced prefill",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="torch manual seed (default 0)",
    )
    parser.add_argument(
        "--skip-decoder",
        action="store_true",
        help="skip the pure text-only decoder prefill fixtures",
    )
    parser.add_argument(
        "--skip-audio-prefill",
        action="store_true",
        help="skip the full audio-spliced prompt forward (prefill_last_logits)",
    )
    return parser


def save_f32(path: Path, tensor: torch.Tensor) -> None:
    arr = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    np.save(path, arr)
    print(
        f"  saved {path.name}: shape={arr.shape} dtype={arr.dtype} "
        f"mean={arr.mean():.5f} std={arr.std():.5f}"
    )


def save_i64(path: Path, tensor: torch.Tensor) -> None:
    arr = tensor.detach().to(dtype=torch.int64, device="cpu").contiguous().numpy()
    np.save(path, arr)
    print(f"  saved {path.name}: shape={arr.shape} dtype={arr.dtype}")


def dump_audio_stages(
    model: torch.nn.Module,
    processor,
    wav: np.ndarray,
    out_dir: Path,
    sample_name: str,
) -> torch.Tensor:
    """Dump mel / encoder mid / encoder out / projector out. Returns input_features."""
    # Feature extractor path used by the original dump_encoder_golden.py and by
    # the processor's audio leg: `processor.audio_processor(waveform)`.
    audio_t = torch.from_numpy(wav).unsqueeze(0)  # [1, T]
    feats = processor.audio_processor(audio_t)
    input_features = feats["input_features"].float()  # [1, T', 160]
    print("input_features", tuple(input_features.shape), input_features.dtype)
    save_f32(out_dir / f"{sample_name}_input_features.npy", input_features)

    # Upcast only the audio numeric core to fp32 for a precise reference.
    model.encoder = model.encoder.float()
    model.projector = model.projector.float()

    mid_layer_idx = int(model.encoder.num_layers) // 2
    mid_capture: dict[str, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_module, _inp, out):
            # Layers are 1-indexed in the original dump script's enumeration
            # (`enumerate(..., start=1)`), matching `num_layers // 2` == 8 for
            # the 16-layer encoder (the self-conditioned CTC tap point).
            if idx == mid_layer_idx:
                mid_capture["mid_block_out"] = out.detach().clone()

        return hook

    handles = []
    for i, layer in enumerate(model.encoder.layers, start=1):
        handles.append(layer.register_forward_hook(make_hook(i)))

    with torch.inference_mode():
        last_hidden = model.encoder(input_features)  # [1, T', 1024]
        proj_out = model.projector(last_hidden)  # [1, N, 2048]

    for handle in handles:
        handle.remove()

    print("encoder last_hidden_state", tuple(last_hidden.shape))
    print("projector output", tuple(proj_out.shape))
    if "mid_block_out" not in mid_capture:
        raise RuntimeError(
            f"mid-layer hook did not fire (looked for layer index {mid_layer_idx})"
        )
    print(
        f"mid_block_out (post layer {mid_layer_idx})",
        tuple(mid_capture["mid_block_out"].shape),
    )

    save_f32(out_dir / f"{sample_name}_encoder_mid_block_out.npy", mid_capture["mid_block_out"])
    save_f32(out_dir / f"{sample_name}_encoder_out.npy", last_hidden)
    save_f32(out_dir / f"{sample_name}_projector_out.npy", proj_out)
    return input_features


def dump_pure_decoder(model: torch.nn.Module, out_dir: Path) -> None:
    """Pure text-only decoder prefill on a fixed id sequence (no audio splice)."""
    model.language_model = model.language_model.float()
    input_ids = torch.tensor([DEFAULT_DECODER_INPUT_IDS], dtype=torch.long)
    save_i64(out_dir / "decoder_input_ids.npy", input_ids)

    with torch.inference_mode():
        inputs_embeds = model.language_model.model.embed_tokens(input_ids)
        inputs_embeds_scaled = (
            inputs_embeds * model.language_model.model.embedding_multiplier
        )
        save_f32(out_dir / "decoder_embed_scaled.npy", inputs_embeds_scaled)

        outputs = model.language_model(
            input_ids=input_ids, use_cache=False, output_hidden_states=True
        )
        hidden = outputs.hidden_states[-1]
        logits = outputs.logits

    print("decoder input_ids", tuple(input_ids.shape))
    print("decoder hidden", tuple(hidden.shape))
    print("decoder logits", tuple(logits.shape))
    save_f32(out_dir / "decoder_hidden_out.npy", hidden)
    save_f32(out_dir / "decoder_logits.npy", logits)

    topk = torch.topk(logits[0, -1], 10)
    print("last-position top10 token ids:", topk.indices.tolist())
    print(
        "last-position top10 logits:",
        [float(v) for v in topk.values.tolist()],
    )


def dump_audio_prefill(
    model: torch.nn.Module,
    processor,
    wav: np.ndarray,
    question: str,
    out_dir: Path,
    device: torch.device,
) -> None:
    """Full audio-spliced prompt forward; dump last-position logits + prompt ids."""
    tokenizer = processor.tokenizer
    user_prompt = build_user_prompt(question)
    chat = [{"role": "user", "content": user_prompt}]
    prompt_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    print(f"audio-prefill prompt_text={prompt_text!r}")

    model_inputs = processor(prompt_text, wav, device=str(device), return_tensors="pt")
    inputs = {}
    for key, tensor in model_inputs.items():
        if not hasattr(tensor, "to"):
            inputs[key] = tensor
            continue
        tensor = tensor.to(device)
        # Prefill runs in whatever dtype the model currently holds. After the
        # audio-stage dump the encoder/projector are fp32; the LM may still be
        # bf16 unless --skip-decoder was false (which upcasts it). Cast only
        # floating inputs to the LM's parameter dtype to avoid a matmul mismatch.
        if torch.is_floating_point(tensor):
            lm_dtype = next(model.language_model.parameters()).dtype
            tensor = tensor.to(lm_dtype)
        inputs[key] = tensor

    save_i64(out_dir / "prompt_input_ids.npy", inputs["input_ids"])

    with torch.inference_mode():
        out = model(**inputs)
    logits = out.logits[0, -1].to(torch.float32).cpu().numpy()
    np.save(out_dir / "prefill_last_logits.npy", logits)
    topk = np.argsort(logits)[::-1][:10]
    print(
        "audio-prefill top10 tokens:",
        [(int(t), float(logits[t])) for t in topk],
    )
    print(
        f"  saved prefill_last_logits.npy: shape={logits.shape} dtype={logits.dtype}"
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    if not args.weights_dir.is_dir():
        print(f"error: --weights-dir does not exist: {args.weights_dir}", file=sys.stderr)
        return 2
    if not args.wav.is_file():
        print(f"error: --wav does not exist: {args.wav}", file=sys.stderr)
        return 2

    sample_name = args.sample_name if args.sample_name else args.wav.stem
    # Refuse path separators in the sample name so a hostile value cannot write
    # outside --out-dir.
    if Path(sample_name).name != sample_name or sample_name in ("", ".", ".."):
        print(f"error: invalid --sample-name {sample_name!r}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    print(f"loading {args.weights_dir} on CPU (bf16 weights, stage-wise fp32)...")
    processor = AutoProcessor.from_pretrained(
        args.weights_dir, fix_mistral_regex=True
    )
    model = (
        AutoModelForSpeechSeq2Seq.from_pretrained(
            args.weights_dir, torch_dtype=torch.bfloat16
        )
        .to(device=device)
        .eval()
    )
    print(f"load dtype=bfloat16; audio stages will upcast to {dtype_name(torch.float32)}")

    wav = load_mono_16k_wav(args.wav)
    print(f"wav {args.wav} samples={wav.shape[0]} ({wav.shape[0] / 16000.0:.2f}s)")

    print("\n== audio stages (mel / encoder / projector) ==")
    dump_audio_stages(model, processor, wav, args.out_dir, sample_name)

    if not args.skip_decoder:
        print("\n== pure decoder prefill ==")
        dump_pure_decoder(model, args.out_dir)

    if not args.skip_audio_prefill:
        print("\n== audio-spliced prefill ==")
        dump_audio_prefill(
            model, processor, wav, args.question, args.out_dir, device
        )

    # Silence an unused-import lint on AUDIO_TOKEN when only helpers are pulled;
    # the constant is part of the public helper surface dump_dumper_test covers.
    _ = AUDIO_TOKEN
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
