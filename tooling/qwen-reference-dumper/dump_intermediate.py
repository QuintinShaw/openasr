#!/usr/bin/env python3
"""Reference-implementation intermediate-activation dumper for qwen3-asr
(Qwen/Qwen3-ASR-0.6B).

Runs the *official* python reference stack (the `qwen_asr` package's
transformers backend) on one fixture wav and dumps the mel frontend output,
the audio-encoder output (post `ln_post`, pre-projector), the audio-adaptor
output (the projector's `proj1 -> act -> proj2` result spliced into the LLM
prompt at the audio-pad token positions), and the first prefill step's logits
as `.npy` files -- so the ggml side
(`crates/openasr-core/src/models/qwen/frontend.rs`, `audio_encoder.rs`,
`llm_prefill.rs`) can be diffed against ground truth stage by stage, the same
role `dump_reference.py`'s `fbank` / `encoder` / `adapter` / `llm` stages play
for firered2-llm (see `../firered2-reference-dumper/README.md`) and
`dump_intermediate.py` plays for moss-transcribe-diarize (see
`../moss-reference-dumper/README.md`). Like the moss dumper this is a single
fixed pipeline run rather than a `--stage`-selectable one: this family's whole
checkpoint (~0.6B params) is small enough that dumping every stage costs
nothing extra, so there is no `--stage`-driven memory trade-off to make (see
`dump_golden.py`'s module doc "Memory" section for why the checkpoint size
makes firered2-llm's meta-device layer-streaming machinery unnecessary here).

The greedy-decode stage of the fbank -> encoder -> adaptor -> LLM-prefill ->
greedy-decode pipeline is NOT dumped here; it is `dump_golden.py`'s full
`model.generate()` call (mirrors moss, whose `dump_intermediate.py` likewise
stops at the prefill logits and leaves the full decode to `dump_golden.py`).

Encoder/adaptor boundary: in the official `Qwen3ASRAudioEncoder.forward` the
adaptor is fused at the tail of the encoder module --
`... -> encoder layers -> ln_post -> proj1 -> act -> proj2 -> output`. So the
stage boundary the ggml side diffs against is the `ln_post` output (the encoder
proper, 896-dim), and the projector (`proj1/act/proj2`) is the adaptor
(896 -> 1024-dim, matching the LLM `d_model`). We tap that boundary with a
forward hook on `audio_tower.ln_post` rather than reimplementing the encoder.

Official reference source (do not vendor -- clone locally)
-----------------------------------------------------------
Code:    https://github.com/QwenLM/Qwen3-ASR
         pinned commit 7c6daf77a2421100f5fb066495372c00129d39ff (2026-06-26)
Weights: https://huggingface.co/Qwen/Qwen3-ASR-0.6B (see `dump_golden.py`'s
         module doc for the expected local layout)

Setup: same as `dump_golden.py` (note the `transformers==4.57.6` pin):

    python3 -m pip install --user "transformers==4.57.6" "torch>=2.8" numpy \\
        librosa soundfile nagisa

Usage
-----
    cd tooling/qwen-reference-dumper
    python3 dump_intermediate.py \\
      --qwen-repo /path/to/Qwen3-ASR \\
      --weights-dir /path/to/qwen3-asr-0.6b-weights \\
      --wav ../../fixtures/jfk.wav \\
      --out-dir /path/to/scratch/qwen-dump

Dumps, into `--out-dir`:

| file | contents | shape |
| --- | --- | --- |
| `input_features.npy` | mel frontend output fed to the audio encoder (128 mel bins) | `[1, 128, mel_frames]` |
| `feature_attention_mask.npy` | valid-frame mask for `input_features` | `[1, mel_frames]` |
| `audio_encoder.npy` | audio encoder's `ln_post` output (pre-projector) | `[n_audio_tokens, 896]` |
| `audio_adaptor.npy` | projector (`proj1 -> act -> proj2`) output spliced into the LLM prompt at the audio-pad token positions | `[n_audio_tokens, 1024]` |
| `prefill_last_logits.npy` | full-prompt forward's last-position logits (pre-generation, one token's worth) | `[vocab_size]` |

Unlike the moss dumper, there is no single-chunk (<=30s) restriction here:
`Qwen3ASRAudioEncoder` does its own windowed-attention chunking (`n_window=50`,
i.e. 100-frame / 1s chunks), so `get_audio_features` handles arbitrarily long
clips (up to the reference stack's 1200s cap) and the dumped tensors are the
real packed valid-frame sequence, not a single-chunk trim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--qwen-repo", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    sys.path.insert(0, str(args.qwen_repo))
    import qwen_asr  # noqa: F401  (registers the family with transformers' Auto classes)
    from qwen_asr.inference.utils import normalize_audios
    from transformers import AutoModel, AutoProcessor

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    dtype = torch.float32

    torch.manual_seed(0)
    model = AutoModel.from_pretrained(args.weights_dir, dtype=dtype).to(device=device).eval()
    processor = AutoProcessor.from_pretrained(args.weights_dir, fix_mistral_regex=True)
    thinker = model.thinker

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    wav = normalize_audios(str(args.wav))[0]
    batch = processor(text=[prompt_text], audio=[wav], return_tensors="pt", padding=True)
    # Cast only floating-point tensors (input_features) to fp32; id / mask
    # tensors must stay integral.
    inputs = {
        key: (tensor.to(device).to(dtype) if torch.is_floating_point(tensor) else tensor.to(device))
        for key, tensor in batch.items()
    }

    input_features = inputs["input_features"]
    feature_attention_mask = inputs["feature_attention_mask"]
    print("input_features", tuple(input_features.shape), input_features.dtype)
    print("feature_attention_mask", tuple(feature_attention_mask.shape), "valid_frames", int(feature_attention_mask.sum()))
    np.save(args.out_dir / "input_features.npy", input_features.to(torch.float32).numpy())
    np.save(args.out_dir / "feature_attention_mask.npy", feature_attention_mask.numpy())

    captured: dict[str, torch.Tensor] = {}

    def capture_ln_post(_module, _inputs, output):
        captured["encoder"] = output.detach()

    def capture_adaptor(_module, _inputs, output):
        captured["adaptor"] = output.last_hidden_state.detach()

    ln_post_handle = thinker.audio_tower.ln_post.register_forward_hook(capture_ln_post)
    tower_handle = thinker.audio_tower.register_forward_hook(capture_adaptor)
    with torch.inference_mode():
        audio_features = thinker.get_audio_features(
            input_features, feature_attention_mask=feature_attention_mask
        )
    ln_post_handle.remove()
    tower_handle.remove()

    n_audio_pad = int((inputs["input_ids"] == processor.audio_token_id).sum())
    print("encoder (ln_post out)", tuple(captured["encoder"].shape))
    print("adaptor (projector out)", tuple(captured["adaptor"].shape))
    print("audio_features", tuple(audio_features.shape), "n_audio_pad_tokens", n_audio_pad)
    np.save(args.out_dir / "audio_encoder.npy", captured["encoder"].to(torch.float32).numpy())
    np.save(args.out_dir / "audio_adaptor.npy", captured["adaptor"].to(torch.float32).numpy())

    with torch.inference_mode():
        out = thinker(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
    logits = out.logits[0, -1].to(torch.float32).numpy()
    np.save(args.out_dir / "prefill_last_logits.npy", logits)
    topk = np.argsort(logits)[::-1][:10]
    print("prefill top10 tokens:", [(int(t), float(logits[t])) for t in topk])

    for name in ["audio_encoder", "audio_adaptor"]:
        a = np.load(args.out_dir / f"{name}.npy")
        print(f"{name}: shape={a.shape} mean={a.mean():.5f} std={a.std():.5f} min={a.min():.4f} max={a.max():.4f}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
