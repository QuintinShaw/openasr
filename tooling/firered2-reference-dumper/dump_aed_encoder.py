"""Reference dumper for FireRedTeam/FireRedASR2-AED's Conformer encoder.

Runs the official FireRedASR2S python reference on a real fixture wav and
prints/dumps the encoder's output, backing the numeric parity pin in
crates/openasr-core/src/models/firered_aed/encoder_graph.rs
(`encoder_matches_reference_pytorch_output_on_jfk_wav`) and the layer-tap
bisection harness (`dump_encoder_layer_taps_for_v2_bisection`).

This is separate from dump_reference.py (firered2-llm's dumper): FireRedASR2-AED
ships its own model.pth.tar with "args" + "model_state_dict" bundled together
(see fireredasr2s/fireredasr2/asr.py load_fireredasr_aed_model), rather than
firered2-llm's split asr_encoder.pth.tar + derived Qwen2 checkpoint layout.
See the README in this directory for checkpoint provenance and flag usage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_refcode_to_path(repo_dir: Path) -> None:
    pkg_root = repo_dir / "fireredasr2s"
    if not pkg_root.is_dir():
        raise FileNotFoundError(f"{pkg_root} not found")
    sys.path.insert(0, str(pkg_root))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fireredasr2s-repo", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, required=True, help="dir with model.pth.tar + cmvn.ark")
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument(
        "--dump-layers-dir",
        type=Path,
        default=None,
        help=(
            "if set, dump the subsample-stem output and each of the 16 "
            "Conformer blocks' final output as row-major float32 "
            "'subsample_out.f32' / 'block_XX.f32' files in this directory, "
            "for the Rust-side bisection test to diff against."
        ),
    )
    ap.add_argument(
        "--tap-layer-idx",
        type=int,
        default=None,
        help=(
            "if set (0-indexed), additionally manually recompute that one "
            "block's forward pass step-by-step (not via a forward hook -- "
            "see the NOTE below) and dump its four intra-block tap points "
            "(ffn1_out/attn_out/conv_out/ffn2_out) plus its final "
            "post-LayerNorm output into --dump-layers-dir as "
            "'tap_ffn1_out.f32' etc, mirroring Rust's FireRedEncoderLayerTaps."
        ),
    )
    ap.add_argument(
        "--fp16-weights",
        action="store_true",
        help=(
            "round-trip every loaded encoder weight tensor through "
            "w.half().float() before running the forward pass (activations "
            "stay fp32 throughout -- only storage precision is perturbed). "
            "Discriminates 'residual is fp16 weight-storage rounding' from "
            "'residual is a structural/graph bug': our .oasr pack stores "
            "weights fp16 and computes in f32, so this reference should be "
            "the correct like-for-like comparison, not the pure-fp32 one."
        ),
    )
    args = ap.parse_args()

    add_refcode_to_path(args.fireredasr2s_repo)

    import numpy as np
    import torch

    from fireredasr2.data.asr_feat import CMVN, KaldifeatFbank
    from fireredasr2.models.module.conformer_encoder import ConformerEncoder
    import kaldiio

    model_path = args.weights_dir / "model.pth.tar"
    cmvn_path = args.weights_dir / "cmvn.ark"

    print(f"loading {model_path} ...", flush=True)
    package = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model_args = package["args"]
    print("encoder args:", {
        k: getattr(model_args, k)
        for k in ("idim", "n_layers_enc", "n_head", "d_model", "residual_dropout", "dropout_rate", "kernel_size", "pe_maxlen")
        if hasattr(model_args, k)
    }, flush=True)

    encoder = ConformerEncoder(
        model_args.idim,
        model_args.n_layers_enc,
        model_args.n_head,
        model_args.d_model,
        model_args.residual_dropout,
        model_args.dropout_rate,
        model_args.kernel_size,
        model_args.pe_maxlen,
    )
    state = {
        k[len("encoder."):]: v
        for k, v in package["model_state_dict"].items()
        if k.startswith("encoder.")
    }
    if args.fp16_weights:
        state = {
            k: (v.half().float() if v.is_floating_point() else v)
            for k, v in state.items()
        }
        print("fp16-weights: round-tripped all encoder tensors through half().float()", flush=True)

    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"encoder state_dict mismatch: missing={missing} unexpected={unexpected}")
    encoder.eval()

    sample_rate, wav_np = kaldiio.load_mat(str(args.wav))
    fbank_extractor = KaldifeatFbank(num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0)
    raw = fbank_extractor((sample_rate, wav_np))
    cmvn = CMVN(str(cmvn_path))
    normed = cmvn(raw).astype(np.float32)

    feat = torch.from_numpy(normed).float().unsqueeze(0)
    length = torch.tensor([feat.shape[1]], dtype=torch.long)

    layer_outputs: list = []
    subsample_output: list = []
    intra_taps: dict = {}

    if args.dump_layers_dir is not None:
        # NOTE: this manually re-runs `ConformerEncoder.forward` /
        # `RelPosEmbConformerBlock.forward` instead of using forward hooks on
        # `block.ffn1` / `block.ffn2` -- the block does NOT use those
        # submodules' raw output directly, it macaron-reweights them
        # (`out = 0.5*x + 0.5*self.ffn1(x)`, and `self.ffn1(x)` itself already
        # folds in its own inner residual, `net(x)+x`). A hook on `ffn1`
        # alone would capture `net(x)+x`, not the `x + 0.5*net(x)` the block
        # actually carries forward -- silently wrong taps. Reusing the same
        # submodules (already loaded with the real weights) via direct calls
        # keeps this exact rather than approximate.
        args.dump_layers_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            padded_input = torch.nn.functional.pad(
                feat, (0, 0, 0, encoder.input_preprocessor.context - 1), "constant", 0.0
            )
            src_mask = encoder.padding_position_is_0(padded_input, length)
            embed_output, _input_lengths, src_mask = encoder.input_preprocessor(
                padded_input, src_mask
            )
            subsample_output.append(embed_output.detach().numpy().astype(np.float32)[0].copy())
            enc_output = encoder.dropout(embed_output)  # no-op at eval
            pos_emb = encoder.dropout(encoder.positional_encoding(embed_output))  # no-op at eval

            for idx, block in enumerate(encoder.layer_stack):
                if idx == args.tap_layer_idx:
                    x = enc_output
                    out = 0.5 * x + 0.5 * block.ffn1(x)
                    intra_taps["ffn1_out"] = out.detach().numpy().astype(np.float32)[0].copy()
                    out = block.mhsa(out, out, out, pos_emb, mask=src_mask)[0]
                    intra_taps["attn_out"] = out.detach().numpy().astype(np.float32)[0].copy()
                    out = block.conv(out, src_mask)
                    intra_taps["conv_out"] = out.detach().numpy().astype(np.float32)[0].copy()
                    out = 0.5 * out + 0.5 * block.ffn2(out)
                    intra_taps["ffn2_out"] = out.detach().numpy().astype(np.float32)[0].copy()
                    out = block.layer_norm(out)
                    intra_taps["block_out"] = out.detach().numpy().astype(np.float32)[0].copy()
                    enc_output = out
                else:
                    enc_output = block(
                        enc_output, pos_emb, slf_attn_mask=src_mask, pad_mask=src_mask
                    )
                layer_outputs.append(enc_output.detach().numpy().astype(np.float32)[0].copy())

    with torch.no_grad():
        enc_out, enc_len, _mask = encoder(feat, length)

    if args.dump_layers_dir is not None:
        (args.dump_layers_dir / "subsample_out.f32").write_bytes(
            subsample_output[0].tobytes()
        )
        for i, layer_out in enumerate(layer_outputs):
            (args.dump_layers_dir / f"block_{i:02d}.f32").write_bytes(layer_out.tobytes())
        for name, value in intra_taps.items():
            (args.dump_layers_dir / f"tap_{name}.f32").write_bytes(value.tobytes())
        print(
            f"dumped subsample_out + {len(layer_outputs)} block outputs"
            + (f" + {len(intra_taps)} intra-block taps" if intra_taps else "")
            + f" to {args.dump_layers_dir}",
            flush=True,
        )

    out = enc_out.detach().numpy().astype(np.float32)[0]
    mode = "fp16-weights (half().float() round-trip)" if args.fp16_weights else "fp32 (pure)"
    print(f"reference mode: {mode}", flush=True)
    print(f"raw_fbank_frames={raw.shape[0]}", flush=True)
    print(f"enc_outputs.shape=[1, {out.shape[0]}, {out.shape[1]}]", flush=True)
    print(f"lengths=[{int(enc_len[0].item())}]", flush=True)
    print("frame0_first8=", [float(x) for x in out[0][:8]], flush=True)


if __name__ == "__main__":
    main()
