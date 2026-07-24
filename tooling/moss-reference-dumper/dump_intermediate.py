#!/usr/bin/env python3
"""Reference-implementation intermediate-activation dumper for
moss-transcribe-diarize (OpenMOSS/MOSS-Transcribe-Diarize, 0.9B).

Runs the *official* python reference stack on one fixture wav and dumps the
Whisper-Medium encoder output, the post-time-merge/pre-adaptor features, the
VQAdaptor output, and the first prefill step's logits as `.npy` files -- so
the ggml side (`crates/openasr-core/src/models/moss_transcribe_diarize/
encoder_graph.rs`, `adaptor_graph.rs`, `llm_decoder.rs`) can be diffed against
ground truth stage by stage, the same role `dump_reference.py`'s `fbank`/
`encoder`/`adapter`/`llm` stages play for firered2-llm (see
`../firered2-reference-dumper/README.md`). Unlike that dumper this is a
single fixed pipeline run rather than a `--stage`-selectable one: this
family's whole checkpoint (~1.6B params) is small enough that dumping every
stage costs nothing extra, so there is no `--stage`-driven memory trade-off
to make in the first place (see `dump_golden.py`'s module doc's "Memory"
section for why the checkpoint size makes firered2-llm's meta-device
layer-streaming machinery unnecessary here).

Official reference source (do not vendor -- clone locally)
------------------------------------------------------------
Code:    https://github.com/OpenMOSS/MOSS-Transcribe-Diarize
         pinned commit 40cf8549c6b5634ba36b7e817cf523b5ad400c2e (2026-07-20)
Weights: https://huggingface.co/OpenMOSS/MOSS-Transcribe-Diarize (gated; see
         `dump_golden.py`'s module doc for the expected local layout)

Setup: same as `dump_golden.py`:

    python3 -m pip install --user "transformers>=5.0,<6.0" "torch>=2.8" numpy

Usage
-----
    cd tooling/moss-reference-dumper
    python3 dump_intermediate.py \\
      --moss-repo /path/to/MOSS-Transcribe-Diarize \\
      --weights-dir /path/to/moss-transcribe-diarize-weights \\
      --wav ../../fixtures/jfk.wav \\
      --out-dir /path/to/scratch/moss-td-dump

Dumps, into `--out-dir`:

| file | contents | shape |
| --- | --- | --- |
| `input_features.npy` | mel frontend output fed to the Whisper encoder | `[1, n_mels, mel_frames]` |
| `whisper_encoder.npy` | Whisper-Medium encoder's `last_hidden_state` | `[1, max_source_positions, d_model]` |
| `merged.npy` | post-`time_merge` (4x reshape, no learned weights), trimmed to this clip's valid token length before merging (mirrors `get_audio_features`'s `whisper_features[chunk_idx:chunk_idx+1, :token_len*4]` truncation -- see `executor.rs`'s module doc) | `[1, token_len, 4*d_model]` |
| `adaptor.npy` | `VQAdaptor` output spliced into the LLM prompt at the audio-pad token's positions | `[1, token_len, llm_d_model]` |
| `prefill_last_logits.npy` | full-prompt forward's last-position logits (pre-generation, one token's worth) | `[vocab_size]` |

Only single-chunk (<=30s) wavs are supported by this dumper's trimming logic
-- it mirrors the executor's own single-chunk trim math, not its multi-chunk
concatenation loop; a wav longer than 30s needs the multi-chunk path
`dump_golden.py`'s full `model.generate()` call already exercises correctly
via the checkpoint's own `get_audio_features`, just without exposing these
intermediate stage-by-stage tensors.
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
    parser.add_argument("--moss-repo", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    sys.path.insert(0, str(args.moss_repo))
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs
    from transformers import AutoModelForCausalLM, AutoProcessor

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    torch.manual_seed(0)
    model = (
        AutoModelForCausalLM.from_pretrained(args.weights_dir, trust_remote_code=True, dtype=torch.float32)
        .to(device=device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(args.weights_dir, trust_remote_code=True)

    messages = build_transcription_messages(str(args.wav))
    inputs = prepare_inputs(processor, messages, device=device)
    print("input_features", inputs["input_features"].shape, inputs["input_features"].dtype)
    print("audio_feature_lengths", inputs["audio_feature_lengths"])
    print("audio_chunk_mapping", inputs["audio_chunk_mapping"])

    mm = model.model
    input_features = inputs["input_features"].to(torch.float32)
    np.save(args.out_dir / "input_features.npy", input_features.numpy())

    with torch.inference_mode():
        whisper_out = mm.whisper_encoder(input_features, return_dict=True).last_hidden_state
        print("whisper last_hidden_state", whisper_out.shape)
        np.save(args.out_dir / "whisper_encoder.npy", whisper_out.to(torch.float32).numpy())

        # Mirrors `get_audio_features`'s single-chunk trim (see
        # `executor.rs`'s module doc): only valid for a <=30s clip, where the
        # whole input is one chunk.
        token_len = int(inputs["audio_feature_lengths"][0].item())
        feat = whisper_out[0:1, : token_len * 4]
        print("trimmed encoder feat", feat.shape, "token_len", token_len)
        merged = mm.time_merge(feat)
        print("merged", merged.shape)
        np.save(args.out_dir / "merged.npy", merged.to(torch.float32).numpy())
        adapted = mm.vq_adaptor(merged)
        print("adaptor out (audio_features)", adapted.shape)
        np.save(args.out_dir / "adaptor.npy", adapted.to(torch.float32).numpy())

        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            audio_feature_lengths=inputs["audio_feature_lengths"],
            audio_chunk_mapping=inputs["audio_chunk_mapping"],
        )
        logits = out.logits[0, -1].to(torch.float32).numpy()
        np.save(args.out_dir / "prefill_last_logits.npy", logits)
        topk = np.argsort(logits)[::-1][:10]
        print("prefill top10 tokens:", [(int(t), float(logits[t])) for t in topk])

    for name in ["whisper_encoder", "merged", "adaptor"]:
        a = np.load(args.out_dir / f"{name}.npy")
        print(f"{name}: shape={a.shape} mean={a.mean():.5f} std={a.std():.5f} min={a.min():.4f} max={a.max():.4f}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
