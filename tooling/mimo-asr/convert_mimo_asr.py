#!/usr/bin/env python3
"""Convert MiMo-V2.5-ASR (main model + MiMo-Audio-Tokenizer encoder) into a single
OpenASR ``.oasr`` GGUF pack.

This is the stage-2 P2.1 conversion pipeline. It produces a *usable local pack*
only; runtime family registration, the decode-policy descriptor, and the ggml
executor land later in P2.2. Nothing here touches the catalog or the model
registry.

Layout follows the design's "single .oasr = single GGUF, tensor-prefix
namespaces" decision (mirrors qwen3-asr's ``package_import.rs``):

  backbone (36L Qwen2)  : token_embd / blk.{i}.* / output_norm / output
  input-local (6L)      : inlocal.blk.{i}.* / inlocal.norm / speech_embd.{0..7} / speech_group_proj
  tokenizer encoder     : audiotok.conv1/conv2 / audiotok.blk.{0..31}.* / audiotok.norm
                          audiotok.down_sample / audiotok.down_sample_norm
                          audiotok.quant.{0..7}.codebook (first 8 RVQ levels only)
                          audiotok.mel_filters / audiotok.mel_window

Three P2.0 "blood-lesson" corrections are encoded as GGUF metadata so the P2.2
runtime forward pass reproduces them exactly (they are graph behaviours, not
weights -- the converter's job is to preserve the enabling weights and record
the hparams):

  * ``mimo.tok.encoder.skip_layer_id = 3`` -- layer-3 (idx 2) output is added to
    the layer-32 (idx 31) output *before* the final LayerNorm. Dropping it
    mis-codes every RVQ frame.
  * ``mimo.tok.conv1.stride = 1`` / ``mimo.tok.conv2.stride = 2`` -- conv1 does
    NOT downsample (only lifts 128->1280); only conv2 does the 2x time stride.
  * 8-codebook *summation* path: the 8 ``speech_embd.{i}`` tables are looked up
    per RVQ channel and summed (not concatenated); ``mimo.audio.channels = 8``.

Discarded on purpose (ASR-only): ``local_transformer.*`` (16L speech-gen),
``local_transformer_lm_heads.*``, ``hidden_states_downcast.*``, tokenizer
``decoder.*``/vocoder, RVQ levels 8..19, and the ``embed_avg``/``cluster_size``/
``inited`` EMA buffers.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# ---------------------------------------------------------------------------
# hparams
# ---------------------------------------------------------------------------

AUDIO_CHANNELS = 8            # RVQ channels the LLM consumes (config.audio_channels)
RVQ_PACKED = 8               # RVQ codebook levels packed into the pack
ARCH = "mimo-asr"
PACKAGE_VERSION = "1"
MODEL_FAMILY = "mimo-asr"
AUDIO_FRONTEND = "mimo-tokenizer-rvq-v0"
DECODE_POLICY = "mimo-asr.greedy.seq2seq.v0"

# Special-token ids (from HF added_tokens / special_tokens_map, pinned in P2.0).
SPECIAL_TOKENS = {
    "eos": 151643,
    "im_start": 151644,
    "im_end": 151645,
    "sosp": 151665,
    "eosp": 151666,
    "empty": 151667,
    "sostm": 151670,
    "eostm": 151671,
    "eot": 151672,
}


@dataclass
class MainHParams:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    attention_bias: bool
    audio_channels: int
    group_size: int
    input_local_layers: int
    input_local_dim: int
    input_local_attn_heads: int
    input_local_head_dim: int
    input_local_intermediate_size: int
    input_full_attention: bool
    input_local_rope_theta: float
    speech_vocab_size: list[int]
    speech_zeroemb_idx: list[int]

    @staticmethod
    def from_config(cfg: dict) -> "MainHParams":
        audio = cfg.get("audio_config", {})

        def _parse_dashed(value) -> list[int]:
            if isinstance(value, list):
                return [int(v) for v in value]
            return [int(v) for v in str(value).split("-")]

        return MainHParams(
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            head_dim=int(cfg["head_dim"]),
            intermediate_size=int(cfg["intermediate_size"]),
            rope_theta=float(cfg["rope_theta"]),
            rms_norm_eps=float(cfg["rms_norm_eps"]),
            vocab_size=int(cfg["vocab_size"]),
            max_position_embeddings=int(cfg["max_position_embeddings"]),
            attention_bias=bool(cfg.get("attention_bias", True)),
            audio_channels=int(cfg.get("audio_channels", AUDIO_CHANNELS)),
            group_size=int(cfg.get("group_size", 4)),
            input_local_layers=int(cfg.get("input_local_layers", audio.get("input_local_layers", 6))),
            input_local_dim=int(cfg.get("input_local_dim", audio.get("input_local_dim", 1024))),
            input_local_attn_heads=int(audio.get("input_local_attn_heads", 64)),
            input_local_head_dim=int(audio.get("input_local_head_dim", 16)),
            input_local_intermediate_size=int(audio.get("input_local_intermediate_size", 4096)),
            input_full_attention=bool(cfg.get("input_full_attention", True)),
            input_local_rope_theta=float(audio.get("rope_theta", cfg["rope_theta"])),
            speech_vocab_size=_parse_dashed(cfg["speech_vocab_size"]),
            speech_zeroemb_idx=_parse_dashed(cfg["speech_zeroemb_idx"]),
        )


@dataclass
class TokHParams:
    n_mels: int
    d_model: int
    encoder_layers: int
    encoder_attention_heads: int
    encoder_ffn_dim: int
    encoder_skip_layer_id: int
    kernel_size: int
    conv1_stride: int
    conv2_stride: int
    avg_pooler: int
    rope_theta: float
    sampling_rate: int
    nfft: int
    hop_length: int
    window_size: int
    fmin: float
    fmax: Optional[float]
    num_quantizers: int
    codebook_size: list[int]

    @staticmethod
    def from_config(cfg: dict) -> "TokHParams":
        return TokHParams(
            n_mels=int(cfg["n_mels"]),
            d_model=int(cfg["d_model"]),
            encoder_layers=int(cfg["encoder_layers"]),
            encoder_attention_heads=int(cfg["encoder_attention_heads"]),
            encoder_ffn_dim=int(cfg["encoder_ffn_dim"]),
            encoder_skip_layer_id=int(cfg["encoder_skip_layer_id"]),
            kernel_size=int(cfg["kernel_size"]),
            # conv1 uses the nn.Conv1d default stride of 1 (P2.0 correction:
            # only conv2 carries config.stride_size).
            conv1_stride=1,
            conv2_stride=int(cfg["stride_size"]),
            avg_pooler=int(cfg["avg_pooler"]),
            rope_theta=float(cfg["rope_theta"]),
            sampling_rate=int(cfg["sampling_rate"]),
            nfft=int(cfg["nfft"]),
            hop_length=int(cfg["hop_length"]),
            window_size=int(cfg["window_size"]),
            fmin=float(cfg["fmin"]),
            fmax=(None if cfg.get("fmax") is None else float(cfg["fmax"])),
            num_quantizers=int(cfg["num_quantizers"]),
            codebook_size=[int(v) for v in cfg["codebook_size"]],
        )


# ---------------------------------------------------------------------------
# tensor-name remapping (pure -- unit tested)
# ---------------------------------------------------------------------------

# Qwen2-style backbone / input-local sublayer map.
_QWEN2_SUB = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.o_proj.weight": "attn_output.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}

# Tokenizer encoder sublayer map (plain-GELU FFN, LayerNorm, asymmetric qkv bias:
# q/v have bias, k does not).
_TOK_SUB = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.out_proj.weight": "attn_out.weight",
    "self_attn.out_proj.bias": "attn_out.bias",
    "self_attn_layer_norm.weight": "attn_norm.weight",
    "self_attn_layer_norm.bias": "attn_norm.bias",
    "final_layer_norm.weight": "ffn_norm.weight",
    "final_layer_norm.bias": "ffn_norm.bias",
    "fc1.weight": "ffn_up.weight",
    "fc1.bias": "ffn_up.bias",
    "fc2.weight": "ffn_down.weight",
    "fc2.bias": "ffn_down.bias",
}


class ConversionError(RuntimeError):
    pass


def _split_layer(rest: str, prefix: str) -> tuple[int, str]:
    """``layers.7.self_attn.q_proj.weight`` -> ``(7, "self_attn.q_proj.weight")``."""
    assert rest.startswith(prefix), rest
    tail = rest[len(prefix):]
    idx_str, _, sub = tail.partition(".")
    return int(idx_str), sub


def remap_main_tensor(name: str) -> Optional[str]:
    """Map a MiMo-V2.5-ASR safetensors key to a GGUF name, or ``None`` to drop.

    Dropped: the 16L speech-gen ``local_transformer`` stack, its 8 codebook
    heads, and ``hidden_states_downcast`` (all ASR-irrelevant per P2.0).
    """
    if (
        name.startswith("local_transformer.")
        or name.startswith("local_transformer_lm_heads.")
        or name == "hidden_states_downcast.weight"
    ):
        return None

    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "model.norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight":
        return "output.weight"

    if name.startswith("model.layers."):
        idx, sub = _split_layer(name[len("model."):], "layers.")
        if sub not in _QWEN2_SUB:
            raise ConversionError(f"unmapped backbone sublayer: {name}")
        return f"blk.{idx}.{_QWEN2_SUB[sub]}"

    if name.startswith("input_local_transformer.layers."):
        idx, sub = _split_layer(name[len("input_local_transformer."):], "layers.")
        if sub not in _QWEN2_SUB:
            raise ConversionError(f"unmapped input-local sublayer: {name}")
        return f"inlocal.blk.{idx}.{_QWEN2_SUB[sub]}"
    if name == "input_local_transformer.norm.weight":
        return "inlocal.norm.weight"

    if name.startswith("speech_embeddings."):
        idx = int(name[len("speech_embeddings."):].split(".")[0])
        return f"speech_embd.{idx}.weight"
    if name == "speech_group_downcast.weight":
        return "speech_group_proj.weight"

    raise ConversionError(f"unexpected main-model tensor: {name}")


def remap_tok_tensor(name: str, rvq_packed: int = RVQ_PACKED) -> Optional[str]:
    """Map a MiMo-Audio-Tokenizer safetensors key to a GGUF name, or ``None`` to drop.

    Encode side only. Keeps ``_codebook.embed`` for the first ``rvq_packed``
    RVQ levels; drops decoder/vocoder, EMA buffers, and RVQ levels >= rvq_packed.
    """
    if not name.startswith("encoder."):
        return None  # decoder.* / vocoder.* -> synthesis path, dropped
    rest = name[len("encoder."):]

    if rest in ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"):
        return f"audiotok.{rest}"
    if rest in ("layer_norm.weight", "layer_norm.bias"):
        return f"audiotok.norm.{rest.split('.')[1]}"
    if rest == "down_sample_layer.0.weight":
        return "audiotok.down_sample.weight"
    if rest in ("down_sample_norm.weight", "down_sample_norm.bias"):
        return f"audiotok.down_sample_norm.{rest.split('.')[1]}"

    if rest.startswith("layers."):
        idx, sub = _split_layer(rest, "layers.")
        if sub not in _TOK_SUB:
            raise ConversionError(f"unmapped tokenizer sublayer: {name}")
        return f"audiotok.blk.{idx}.{_TOK_SUB[sub]}"

    if rest.startswith("quantizer.vq.layers."):
        idx, sub = _split_layer(rest[len("quantizer.vq."):], "layers.")
        if sub != "_codebook.embed":
            return None  # embed_avg / cluster_size / inited -> EMA buffers, dropped
        if idx >= rvq_packed:
            return None  # RVQ levels 8..19 unused by ASR (residual causality)
        return f"audiotok.quant.{idx}.codebook"

    raise ConversionError(f"unexpected tokenizer tensor: {name}")


# ---------------------------------------------------------------------------
# reference 8-codebook summation semantics (documented + unit tested)
# ---------------------------------------------------------------------------

def sum_speech_embeddings(
    tables: list[np.ndarray],
    codes: np.ndarray,
    zeroemb_idx: list[int],
) -> np.ndarray:
    """Reference for the LLM's audio-embedding path (modeling_mimo_audio.py
    ``_prepare_input_embeds``): look up each of the 8 RVQ channels in its own
    ``speech_embd`` table and *sum* (not concatenate), zeroing rows equal to the
    channel's zeroemb id.

    tables: 8 arrays, table i has shape [speech_vocab_size_i, dim].
    codes:  int array [T, 8] of per-frame per-channel codebook ids.
    returns [T, dim].
    """
    if len(tables) != len(zeroemb_idx):
        raise ValueError("tables/zeroemb length mismatch")
    T = codes.shape[0]
    dim = tables[0].shape[1]
    out = np.zeros((T, dim), dtype=np.float64)
    for ch, table in enumerate(tables):
        ids = codes[:, ch]
        emb = table[ids].astype(np.float64)
        emb[ids == zeroemb_idx[ch]] = 0.0
        out += emb
    return out


# ---------------------------------------------------------------------------
# quant / dtype policy (pure -- unit tested)
# ---------------------------------------------------------------------------

def _force_f32(gguf_name: str, rank: int) -> bool:
    return (
        rank <= 1
        or gguf_name.endswith(".bias")
        or gguf_name.endswith("norm.weight")
        or gguf_name.endswith(".codebook")   # RVQ distances run in f32 upstream
        or gguf_name.startswith("audiotok.mel_")
    )


def _is_backbone_weight(gguf_name: str) -> bool:
    return (
        gguf_name.startswith("blk.")
        or gguf_name in ("token_embd.weight", "output.weight")
    )


def choose_tensor_type(gguf_name: str, shape: tuple[int, ...], quant_label: str) -> str:
    """Return one of ``"q8_0"`` / ``"f16"`` / ``"f32"``.

    * fp16 pack: every eligible weight -> f16, forced tensors -> f32.
    * q8_0 pack: only the *backbone* rank-2 ``.weight`` matrices with a
      32-aligned inner dim are quantized; the whole audio side (audiotok.*,
      inlocal.*, speech_embd.*, codebooks) stays f16/f32 for encode fidelity.
    """
    rank = len(shape)
    if _force_f32(gguf_name, rank):
        return "f32"
    if (
        quant_label == "q8_0"
        and _is_backbone_weight(gguf_name)
        and gguf_name.endswith(".weight")
        and rank == 2
        and shape[-1] % 32 == 0
    ):
        return "q8_0"
    return "f16"


# ---------------------------------------------------------------------------
# GGUF metadata (pure builder -- unit tested)
# ---------------------------------------------------------------------------

@dataclass
class MetaItem:
    key: str
    kind: str   # "str" | "u32" | "f32" | "bool" | "u32_array" | "str_array"
    value: object


def build_metadata(
    main: MainHParams,
    tok: TokHParams,
    model_id: str,
    quant_label: str,
) -> list[MetaItem]:
    m: list[MetaItem] = []

    def s(k, v): m.append(MetaItem(k, "str", str(v)))
    def u(k, v): m.append(MetaItem(k, "u32", int(v)))
    def f(k, v): m.append(MetaItem(k, "f32", float(v)))
    def b(k, v): m.append(MetaItem(k, "bool", bool(v)))
    def ua(k, v): m.append(MetaItem(k, "u32_array", [int(x) for x in v]))

    # openasr envelope
    s("openasr.package.version", PACKAGE_VERSION)
    s("openasr.model.family", MODEL_FAMILY)
    s("openasr.model.architecture", ARCH)
    s("openasr.model.id", model_id)
    s("openasr.audio.frontend", AUDIO_FRONTEND)
    s("openasr.decode.policy", DECODE_POLICY)
    s("openasr.pack.quant", quant_label)

    # backbone (36L Qwen2, GQA, qkv-bias, no qk-norm)
    u("mimo.llm.block_count", main.num_hidden_layers)
    u("mimo.llm.context_length", main.max_position_embeddings)
    u("mimo.llm.embedding_length", main.hidden_size)
    u("mimo.llm.feed_forward_length", main.intermediate_size)
    u("mimo.llm.attention.head_count", main.num_attention_heads)
    u("mimo.llm.attention.head_count_kv", main.num_key_value_heads)
    u("mimo.llm.attention.key_length", main.head_dim)
    u("mimo.llm.attention.value_length", main.head_dim)
    f("mimo.llm.attention.layer_norm_rms_epsilon", main.rms_norm_eps)
    f("mimo.llm.rope.freq_base", main.rope_theta)
    u("mimo.llm.vocab_size", main.vocab_size)
    b("mimo.llm.attention.qkv_bias", main.attention_bias)
    b("mimo.llm.attention.qk_norm", False)

    # input-local (6L) + speech embedding sum path
    u("mimo.audio.channels", main.audio_channels)
    u("mimo.audio.group_size", main.group_size)
    u("mimo.inlocal.block_count", main.input_local_layers)
    u("mimo.inlocal.embedding_length", main.input_local_dim)
    u("mimo.inlocal.attention.head_count", main.input_local_attn_heads)
    u("mimo.inlocal.attention.head_dim", main.input_local_head_dim)
    u("mimo.inlocal.feed_forward_length", main.input_local_intermediate_size)
    b("mimo.inlocal.full_attention", main.input_full_attention)
    f("mimo.inlocal.rope.freq_base", main.input_local_rope_theta)
    ua("mimo.speech.vocab_size", main.speech_vocab_size)
    ua("mimo.speech.zeroemb_idx", main.speech_zeroemb_idx)

    # tokenizer encoder (32L) + the three P2.0 blood-lesson hparams
    u("mimo.tok.block_count", tok.encoder_layers)
    u("mimo.tok.embedding_length", tok.d_model)
    u("mimo.tok.attention.head_count", tok.encoder_attention_heads)
    u("mimo.tok.feed_forward_length", tok.encoder_ffn_dim)
    u("mimo.tok.encoder.skip_layer_id", tok.encoder_skip_layer_id)   # blood lesson #1
    u("mimo.tok.conv.kernel_size", tok.kernel_size)
    u("mimo.tok.conv1.stride", tok.conv1_stride)                     # blood lesson #2 (=1)
    u("mimo.tok.conv2.stride", tok.conv2_stride)                     # blood lesson #2 (=2)
    u("mimo.tok.down_sample.stride", tok.avg_pooler)
    f("mimo.tok.rope.freq_base", tok.rope_theta)
    s("mimo.tok.ln_type", "layernorm")
    b("mimo.tok.attention.qk_bias_asymmetric", True)  # q/v bias, k none
    packed = min(RVQ_PACKED, tok.num_quantizers)
    u("mimo.tok.rvq.num_quantizers_total", tok.num_quantizers)
    u("mimo.tok.rvq.num_quantizers_packed", packed)
    ua("mimo.tok.rvq.codebook_sizes", tok.codebook_size[:packed])

    # mel front-end spec (torchaudio.MelSpectrogram, htk / norm=None / power=1 /
    # ln(clip 1e-7) / center=True) -- baked filters+window shipped as tensors.
    fmax = tok.fmax if tok.fmax is not None else tok.sampling_rate / 2.0
    u("mimo.mel.sample_rate", tok.sampling_rate)
    u("mimo.mel.n_fft", tok.nfft)
    u("mimo.mel.hop_length", tok.hop_length)
    u("mimo.mel.win_length", tok.window_size)
    u("mimo.mel.n_mels", tok.n_mels)
    f("mimo.mel.f_min", tok.fmin)
    f("mimo.mel.f_max", fmax)
    s("mimo.mel.mel_scale", "htk")
    s("mimo.mel.norm", "none")
    f("mimo.mel.power", 1.0)
    s("mimo.mel.log_type", "ln")
    f("mimo.mel.log_clip", 1e-7)
    b("mimo.mel.center", True)

    # special tokens
    for name, tid in SPECIAL_TOKENS.items():
        u(f"mimo.special.{name}_id", tid)

    return m


def apply_metadata(writer, items: list[MetaItem]) -> None:
    for it in items:
        if it.kind == "str":
            writer.add_string(it.key, it.value)
        elif it.kind == "u32":
            writer.add_uint32(it.key, it.value)
        elif it.kind == "f32":
            writer.add_float32(it.key, it.value)
        elif it.kind == "bool":
            writer.add_bool(it.key, it.value)
        elif it.kind == "u32_array":
            writer.add_array(it.key, it.value)
        elif it.kind == "str_array":
            writer.add_array(it.key, it.value)
        else:
            raise ConversionError(f"unknown metadata kind {it.kind}")


# ---------------------------------------------------------------------------
# mel filterbank / window (baked to match torchaudio exactly)
# ---------------------------------------------------------------------------

def mel_filters_and_window(tok: TokHParams) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torchaudio

    fmax = tok.fmax if tok.fmax is not None else tok.sampling_rate / 2.0
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=tok.nfft // 2 + 1,
        f_min=float(tok.fmin),
        f_max=float(fmax),
        n_mels=tok.n_mels,
        sample_rate=tok.sampling_rate,
        norm=None,
        mel_scale="htk",
    )  # [n_freqs, n_mels]
    window = torch.hann_window(tok.window_size, periodic=True)
    return fb.numpy().astype(np.float32), window.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# source iteration (streaming, one tensor at a time)
# ---------------------------------------------------------------------------

@dataclass
class SourceTensor:
    gguf_name: str
    array: np.ndarray            # float32
    source_name: str


def _st_to_f32(t) -> np.ndarray:
    import torch
    return t.to(torch.float32).contiguous().numpy()


def iter_main_tensors(main_dir: Path) -> Iterator[SourceTensor]:
    from safetensors import safe_open

    shards = sorted(main_dir.glob("model-*-of-*.safetensors"))
    if not shards:
        shards = sorted(main_dir.glob("*.safetensors"))
    if not shards:
        raise ConversionError(f"no safetensors under {main_dir}")
    for shard in shards:
        with safe_open(str(shard), framework="pt") as fh:
            for key in fh.keys():
                gname = remap_main_tensor(key)
                if gname is None:
                    continue
                yield SourceTensor(gname, _st_to_f32(fh.get_tensor(key)), key)


def iter_tok_tensors(tok_path: Path) -> Iterator[SourceTensor]:
    from safetensors import safe_open

    with safe_open(str(tok_path), framework="pt") as fh:
        for key in fh.keys():
            gname = remap_tok_tensor(key)
            if gname is None:
                continue
            yield SourceTensor(gname, _st_to_f32(fh.get_tensor(key)), key)


# ---------------------------------------------------------------------------
# pack writer
# ---------------------------------------------------------------------------

def write_pack(
    main_dir: Path,
    tok_path: Path,
    out_path: Path,
    quant_label: str,
    model_id: str,
    *,
    bake_mel: bool = True,
    verbose: bool = True,
) -> dict:
    import gguf

    main_cfg = json.loads((main_dir / "config.json").read_text())
    tok_cfg = json.loads((tok_path.parent / "config.json").read_text())
    main = MainHParams.from_config(main_cfg)
    tok = TokHParams.from_config(tok_cfg)

    writer = gguf.GGUFWriter(str(out_path), ARCH, use_temp_file=True)
    apply_metadata(writer, build_metadata(main, tok, model_id, quant_label))

    type_counts = {"q8_0": 0, "f16": 0, "f32": 0}
    seen: set[str] = set()

    def emit(src: SourceTensor) -> None:
        if src.gguf_name in seen:
            raise ConversionError(f"duplicate GGUF tensor {src.gguf_name}")
        seen.add(src.gguf_name)
        arr = src.array
        ttype = choose_tensor_type(src.gguf_name, arr.shape, quant_label)
        if ttype == "q8_0":
            data = gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(src.gguf_name, data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        elif ttype == "f16":
            writer.add_tensor(
                src.gguf_name, arr.astype(np.float16),
                raw_dtype=gguf.GGMLQuantizationType.F16,
            )
        else:
            writer.add_tensor(
                src.gguf_name, arr.astype(np.float32),
                raw_dtype=gguf.GGMLQuantizationType.F32,
            )
        type_counts[ttype] += 1

    n = 0
    for src in iter_main_tensors(main_dir):
        emit(src)
        n += 1
        if verbose and n % 100 == 0:
            print(f"  .. {n} tensors", flush=True)
        del src
        gc.collect()
    for src in iter_tok_tensors(tok_path):
        emit(src)
        n += 1
        del src
        gc.collect()

    if bake_mel:
        fb, window = mel_filters_and_window(tok)
        writer.add_tensor("audiotok.mel_filters", fb, raw_dtype=gguf.GGMLQuantizationType.F32)
        writer.add_tensor("audiotok.mel_window", window, raw_dtype=gguf.GGMLQuantizationType.F32)
        type_counts["f32"] += 2
        n += 2

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return {
        "output": str(out_path),
        "tensor_count": n,
        "type_counts": type_counts,
        "size_bytes": out_path.stat().st_size,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Convert MiMo-V2.5-ASR to a .oasr GGUF pack")
    ap.add_argument("--main-dir", required=True, type=Path, help="MiMo-V2.5-ASR dir (config.json + shards)")
    ap.add_argument("--tokenizer", required=True, type=Path, help="MiMo-Audio-Tokenizer model.safetensors (config.json alongside)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--package-id", default="mimo-v2.5-asr")
    ap.add_argument("--quant", action="append", choices=["q8_0", "fp16"], help="repeatable; default q8_0 + fp16")
    ap.add_argument("--no-mel", action="store_true", help="skip baking mel filters/window")
    args = ap.parse_args(argv)

    quants = args.quant or ["q8_0", "fp16"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for q in quants:
        # canonical quant label for tensor policy is q8_0/f16 style; pack name uses q8_0/fp16
        policy_label = "q8_0" if q == "q8_0" else "fp16"
        out_path = args.out_dir / f"{args.package_id}-{q}.oasr"
        model_id = f"{args.package_id}-{q}"
        print(f"[convert] {q} -> {out_path}", flush=True)
        res = write_pack(args.main_dir, args.tokenizer, out_path, policy_label, model_id)
        print(f"[done] {q}: {res['tensor_count']} tensors, "
              f"{res['size_bytes']/1e9:.2f} GB, types={res['type_counts']}", flush=True)
        results.append(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
