#!/usr/bin/env python3
"""Convert a Fun-ASR-Nano-2512 ``model.pt`` checkpoint into an f32 safetensors
file keyed by the openasr ``.oasr`` tensor-name convention, plus a JSON
metadata sidecar, for the Rust ``funasr_nano::package_import`` importer to
quantize (fp16 / q8_0 / q4_k) and pack.

Fun-ASR-Nano = SenseVoice SAN-M encoder (70 layers) + 2-layer transformer
adaptor + Qwen3-0.6B LLM. The published ``model.pt`` state_dict has exactly
three top-level groups -- ``audio_encoder.*`` (914), ``audio_adaptor.*`` (36),
``llm.*`` (311) -- and, crucially, NO ``ctc_decoder.*`` tensors: the CTC branch
(a training-only auxiliary head) is already excluded from the release
checkpoint, so there is nothing to strip. The LLM half is a stock Qwen3-0.6B
(no ASR customization; tied ``lm_head``/``embed_tokens`` both present and
identical), so its tensors map onto the crate's existing ``qwen`` LLM branch
naming (``blk.N.*`` / ``token_embd`` / ``output`` / ``output_norm``); the
encoder reuses the ``sensevoice`` SAN-M naming (``enc.blk`` / ``tp.blk`` /
``enc.after_norm`` / ``tp.norm``).

Emitted tensor names (openasr convention):
  enc.blk.{0..49}.{attn.norm,attn.qkv,attn.out,attn.fsmn,ffn.norm,ffn.up,ffn.down}.*
  tp.blk.{0..19}.<same>   enc.after_norm.*   tp.norm.*
  adaptor.linear1/2.{weight,bias}
  adaptor.blk.{0,1}.{attn.norm,attn.q,attn.k,attn.v,attn.out,ffn.norm,ffn.up,ffn.down}.*
  token_embd.weight   output.weight   output_norm.weight
  blk.{0..27}.{attn_norm,attn_q,attn_k,attn_v,attn_output,attn_q_norm,attn_k_norm,
               ffn_norm,ffn_gate,ffn_up,ffn_down}.weight

The FSMN depthwise kernel is emitted as the raw torch ``(D,1,K)`` C-contiguous
tensor (index ``d*K + k``): the crate's ``nn::encoder::sanm_fsmn_encoder_layer``
does ``reshape_4d(fsmn, K, 1, 1, D)`` which reads exactly that layout as the
per-channel kernel, matching how ``sensevoice::package_import`` stores it.
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file

# Architecture constants read from config.yaml + verified against the actual
# state_dict tensor shapes (adaptor FFN intermediate is 256, taken from the
# w_1 weight shape, not config.yaml's stale ffn_dim=2048).
ENC_BLOCKS = 50        # encoders0.0 (1) + encoders.0..48 (49)
TP_BLOCKS = 20
ADP_BLOCKS = 2
LLM_LAYERS = 28


def emit_sanm(out, sd, dst, src):
    out[f"{dst}.attn.norm.weight"] = sd[f"{src}.norm1.weight"]
    out[f"{dst}.attn.norm.bias"] = sd[f"{src}.norm1.bias"]
    out[f"{dst}.attn.qkv.weight"] = sd[f"{src}.self_attn.linear_q_k_v.weight"]
    out[f"{dst}.attn.qkv.bias"] = sd[f"{src}.self_attn.linear_q_k_v.bias"]
    out[f"{dst}.attn.out.weight"] = sd[f"{src}.self_attn.linear_out.weight"]
    out[f"{dst}.attn.out.bias"] = sd[f"{src}.self_attn.linear_out.bias"]
    out[f"{dst}.attn.fsmn.weight"] = sd[f"{src}.self_attn.fsmn_block.weight"]
    out[f"{dst}.ffn.norm.weight"] = sd[f"{src}.norm2.weight"]
    out[f"{dst}.ffn.norm.bias"] = sd[f"{src}.norm2.bias"]
    out[f"{dst}.ffn.up.weight"] = sd[f"{src}.feed_forward.w_1.weight"]
    out[f"{dst}.ffn.up.bias"] = sd[f"{src}.feed_forward.w_1.bias"]
    out[f"{dst}.ffn.down.weight"] = sd[f"{src}.feed_forward.w_2.weight"]
    out[f"{dst}.ffn.down.bias"] = sd[f"{src}.feed_forward.w_2.bias"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pt", required=True)
    ap.add_argument("--qwen_dir", required=True, help="Qwen3-0.6B dir (for token ids)")
    ap.add_argument("--out_safetensors", required=True)
    ap.add_argument("--out_meta", required=True)
    args = ap.parse_args()

    sd = torch.load(args.model_pt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = {k: v.to(torch.float32).contiguous() for k, v in sd.items()}

    for k in sd:
        if k.startswith("ctc_decoder"):
            raise SystemExit(f"unexpected CTC tensor {k}: release checkpoint should have none")

    out = {}
    ae = "audio_encoder"
    emit_sanm(out, sd, "enc.blk.0", f"{ae}.encoders0.0")
    for i in range(ENC_BLOCKS - 1):
        emit_sanm(out, sd, f"enc.blk.{i + 1}", f"{ae}.encoders.{i}")
    out["enc.after_norm.weight"] = sd[f"{ae}.after_norm.weight"]
    out["enc.after_norm.bias"] = sd[f"{ae}.after_norm.bias"]
    for i in range(TP_BLOCKS):
        emit_sanm(out, sd, f"tp.blk.{i}", f"{ae}.tp_encoders.{i}")
    out["tp.norm.weight"] = sd[f"{ae}.tp_norm.weight"]
    out["tp.norm.bias"] = sd[f"{ae}.tp_norm.bias"]

    ad = "audio_adaptor"
    out["adaptor.linear1.weight"] = sd[f"{ad}.linear1.weight"]
    out["adaptor.linear1.bias"] = sd[f"{ad}.linear1.bias"]
    out["adaptor.linear2.weight"] = sd[f"{ad}.linear2.weight"]
    out["adaptor.linear2.bias"] = sd[f"{ad}.linear2.bias"]
    for i in range(ADP_BLOCKS):
        p, d = f"{ad}.blocks.{i}", f"adaptor.blk.{i}"
        out[f"{d}.attn.norm.weight"] = sd[f"{p}.norm1.weight"]
        out[f"{d}.attn.norm.bias"] = sd[f"{p}.norm1.bias"]
        for proj in ("q", "k", "v"):
            out[f"{d}.attn.{proj}.weight"] = sd[f"{p}.self_attn.linear_{proj}.weight"]
            out[f"{d}.attn.{proj}.bias"] = sd[f"{p}.self_attn.linear_{proj}.bias"]
        out[f"{d}.attn.out.weight"] = sd[f"{p}.self_attn.linear_out.weight"]
        out[f"{d}.attn.out.bias"] = sd[f"{p}.self_attn.linear_out.bias"]
        out[f"{d}.ffn.norm.weight"] = sd[f"{p}.norm2.weight"]
        out[f"{d}.ffn.norm.bias"] = sd[f"{p}.norm2.bias"]
        out[f"{d}.ffn.up.weight"] = sd[f"{p}.feed_forward.w_1.weight"]
        out[f"{d}.ffn.up.bias"] = sd[f"{p}.feed_forward.w_1.bias"]
        out[f"{d}.ffn.down.weight"] = sd[f"{p}.feed_forward.w_2.weight"]
        out[f"{d}.ffn.down.bias"] = sd[f"{p}.feed_forward.w_2.bias"]

    lm = "llm.model"
    out["token_embd.weight"] = sd[f"{lm}.embed_tokens.weight"]
    out["output.weight"] = sd["llm.lm_head.weight"]
    out["output_norm.weight"] = sd[f"{lm}.norm.weight"]
    for i in range(LLM_LAYERS):
        p, d = f"{lm}.layers.{i}", f"blk.{i}"
        out[f"{d}.attn_norm.weight"] = sd[f"{p}.input_layernorm.weight"]
        out[f"{d}.attn_q.weight"] = sd[f"{p}.self_attn.q_proj.weight"]
        out[f"{d}.attn_k.weight"] = sd[f"{p}.self_attn.k_proj.weight"]
        out[f"{d}.attn_v.weight"] = sd[f"{p}.self_attn.v_proj.weight"]
        out[f"{d}.attn_output.weight"] = sd[f"{p}.self_attn.o_proj.weight"]
        out[f"{d}.attn_q_norm.weight"] = sd[f"{p}.self_attn.q_norm.weight"]
        out[f"{d}.attn_k_norm.weight"] = sd[f"{p}.self_attn.k_norm.weight"]
        out[f"{d}.ffn_norm.weight"] = sd[f"{p}.post_attention_layernorm.weight"]
        out[f"{d}.ffn_gate.weight"] = sd[f"{p}.mlp.gate_proj.weight"]
        out[f"{d}.ffn_up.weight"] = sd[f"{p}.mlp.up_proj.weight"]
        out[f"{d}.ffn_down.weight"] = sd[f"{p}.mlp.down_proj.weight"]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.qwen_dir)
    meta = {
        "enc": {
            "n_layers": ENC_BLOCKS, "tp_blocks": TP_BLOCKS, "d_model": 512,
            "n_heads": 4, "head_dim": 128, "ffn_dim": 2048, "fsmn_kernel": 11,
            "feature_dim": 560, "layer_norm_eps": 1e-5,
        },
        "adp": {"n_layers": ADP_BLOCKS, "n_heads": 8, "llm_dim": 1024, "encoder_dim": 512},
        "llm": {
            "n_layers": LLM_LAYERS, "d_model": 1024, "n_heads": 16, "n_kv_heads": 8,
            "head_dim": 128, "ffn_dim": 3072, "vocab_size": 151936, "max_positions": 40960,
            "rope_theta": 1_000_000.0, "rms_norm_eps": 1e-6,
            "chatml_im_start_token_id": int(tok.convert_tokens_to_ids("<|im_start|>")),
            "chatml_im_end_token_id": int(tok.convert_tokens_to_ids("<|im_end|>")),
            "endoftext_token_id": int(tok.convert_tokens_to_ids("<|endoftext|>")),
        },
    }

    save_file(out, args.out_safetensors)
    json.dump(meta, open(args.out_meta, "w"), indent=2)
    print(f"wrote {len(out)} tensors -> {args.out_safetensors} "
          f"({os.path.getsize(args.out_safetensors) / 1e6:.1f} MB)")
    print(f"wrote metadata -> {args.out_meta}")


if __name__ == "__main__":
    main()
