#!/usr/bin/env python3
"""Reference-implementation dumper for the firered2-llm family
(FireRedTeam/FireRedASR2-LLM, Encoder-Adapter-LLM).

Runs the *official* FireRedASR2S python modules (not a reimplementation)
stage-by-stage on a real fixture wav and dumps every intermediate activation
to an npz file, so the ggml side (`crates/openasr-core/src/models/firered_llm`)
can be diffed against it. Satisfies the "Reference dumper exists for this
family" row in `docs/model-audits/TEMPLATE.md` section 10.

Stages (see `--stage`):
  fbank      80-dim kaldi fbank + global CMVN (ASRFeatExtractor)
  encoder    16-layer Conformer encoder (per-layer + final output)
  adapter    2x frame-stacking Linear-ReLU-Linear projector into LLM space
  llm        Qwen2-7B-Instruct (LoRA pre-merged) prefill last-hidden-state +
             first N greedy decode steps' logits
  all        every stage above, feeding forward (fbank -> encoder -> adapter
             -> llm), default

Official reference source
--------------------------
Code:    https://github.com/FireRedTeam/FireRedASR2S
         pinned commit 4e7d9aaf4482a47cec1724807026b9b151926eb5 (2026-06-02)
Weights: https://huggingface.co/FireRedTeam/FireRedASR2-LLM
         (encoder + adapter + LoRA-merged Qwen2-7B-Instruct; see
         `tooling/publish-model/scripts/firered_llm_merge_lora.py` for how
         the LoRA increment is folded into the base Qwen2 weights offline --
         this dumper loads the already-merged checkpoint, not the LoRA
         checkpoint + `peft`, since the merge itself is a fixed, previously
         audited linear-algebra step (see that script's docstring) and
         skipping it here avoids a `peft` dependency for a reference tool).

This script does NOT vendor the FireRedASR2S code or any weights into the
repo (both are third-party and, in the weights' case, multi-GB). Point
`--fireredasr2s-repo` at your own clone of the pinned commit above, and
`--weights-dir` at a local copy of the upstream weight layout:

    <weights-dir>/
      asr_encoder.pth.tar     # AED encoder args (architecture only)
      model.pth.tar           # encoder + adapter + LoRA state dict (or
                               # derived/model.safetensors, the same content
                               # produced by pt_to_safetensors.py)
      cmvn.ark                # kaldi global CMVN stats
      Qwen2-7B-Instruct/      # tokenizer files (config.json etc.)
      derived/qwen2-merged.safetensors  # LoRA-merged Qwen2 base weights,
                               # produced by firered_llm_merge_lora.py

Setup (host python, matches `tooling/publish-model` convention -- no
dedicated venv in this repo; installs are small pure/wheel packages):

    python3 -m pip install --user torch transformers safetensors numpy \\
        kaldi_native_fbank kaldiio gguf accelerate

Memory discipline (16GB dev machines)
--------------------------------------
fbank/encoder/adapter fit comfortably (<4GB total, fp32). The LLM stage is
the expensive one: Qwen2-7B-Instruct is ~7.6B params, ~15GB at fp16 -- too
much to keep resident on a 16GB machine alongside anything else. Instead of
loading the whole model, `StreamingQwen2Layers` builds the HF module graph on
the meta device (zero host memory) and materializes ONE decoder layer's
weights at a time, straight off the safetensors mmap with an fp16 downcast,
runs that layer's forward, then releases it back to a meta tensor before
moving to the next layer. Peak extra RSS for the LLM stage is therefore
bounded by one layer's weights (~0.5GB fp16) + embed/lm_head/norm (~2.2GB
fp16) + KV cache + activations, not the full 15GB. This trades wall-clock
(weights are re-read from disk on every decode step) for memory headroom,
which is the right trade for a one-time correctness dump (not a perf probe).

Use `--check-mem-before-llm` (default on) to poll `vm_stat` and sleep-wait
until enough headroom exists before the LLM stage starts, since this is
meant to run alongside other work on a shared dev machine.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Memory discipline helpers
# ---------------------------------------------------------------------------


def available_memory_gb() -> float:
    """Rough macOS-only 'available' estimate: free + inactive + purgeable +
    speculative pages, mirroring the vm_stat accounting used elsewhere in
    this repo's dev workflow (see AGENTS.md memory discipline notes)."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return float("inf")  # non-macOS or vm_stat unavailable: don't block
    page_size = 16384
    counts: dict[str, int] = {}
    for line in out.splitlines():
        if "page size of" in line:
            try:
                page_size = int(line.split("page size of")[1].split()[0])
            except (IndexError, ValueError):
                pass
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip().rstrip(".")
        if val.isdigit():
            counts[key.strip()] = int(val)
    free = counts.get("Pages free", 0)
    inactive = counts.get("Pages inactive", 0)
    purgeable = counts.get("Pages purgeable", 0)
    speculative = counts.get("Pages speculative", 0)
    return (free + inactive + purgeable + speculative) * page_size / (1024**3)


def wait_for_memory(min_gb: float, poll_seconds: float = 20.0, max_wait_seconds: float = 1800.0) -> None:
    waited = 0.0
    while True:
        avail = available_memory_gb()
        if avail >= min_gb:
            print(f"[mem] {avail:.2f} GB available (>= {min_gb} GB threshold), proceeding", file=sys.stderr)
            return
        if waited >= max_wait_seconds:
            print(
                f"[mem] WARNING: still only {avail:.2f} GB available after {waited:.0f}s wait; "
                f"proceeding anyway (raise --min-mem-gb wait budget if this OOMs)",
                file=sys.stderr,
            )
            return
        print(f"[mem] only {avail:.2f} GB available (< {min_gb} GB), sleeping {poll_seconds:.0f}s...", file=sys.stderr)
        time.sleep(poll_seconds)
        waited += poll_seconds


# ---------------------------------------------------------------------------
# Official reference code loading
# ---------------------------------------------------------------------------


def add_refcode_to_path(repo_dir: Path) -> None:
    pkg_root = repo_dir / "fireredasr2s"
    if not pkg_root.is_dir():
        raise FileNotFoundError(
            f"{pkg_root} not found -- point --fireredasr2s-repo at a clone of "
            "https://github.com/FireRedTeam/FireRedASR2S.git (pin 4e7d9aaf4482a47cec1724807026b9b151926eb5)"
        )
    sys.path.insert(0, str(pkg_root))


# ---------------------------------------------------------------------------
# Stage 1: fbank frontend
# ---------------------------------------------------------------------------


@dataclass
class FbankDump:
    raw_fbank: np.ndarray  # (T, 80) before CMVN
    cmvn_fbank: np.ndarray  # (T, 80) after CMVN, this is the encoder input
    sample_rate: int
    duration_s: float


def run_fbank(wav_path: Path, cmvn_ark: Path) -> FbankDump:
    from fireredasr2.data.asr_feat import CMVN, KaldifeatFbank
    import kaldiio

    sample_rate, wav_np = kaldiio.load_mat(str(wav_path))
    fbank_extractor = KaldifeatFbank(num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0)
    raw = fbank_extractor((sample_rate, wav_np))
    cmvn = CMVN(str(cmvn_ark))
    normed = cmvn(raw)
    return FbankDump(
        raw_fbank=raw.astype(np.float32),
        cmvn_fbank=normed.astype(np.float32),
        sample_rate=int(sample_rate),
        duration_s=float(wav_np.shape[0]) / float(sample_rate),
    )


# ---------------------------------------------------------------------------
# Stage 2+3: Conformer encoder + Adapter (official modules, real weights)
# ---------------------------------------------------------------------------


@dataclass
class EncoderAdapterDump:
    encoder_layer_outputs: list[np.ndarray]  # 16x (T', 1280)
    encoder_output: np.ndarray  # (T', 1280), == encoder_layer_outputs[-1]
    encoder_length: int
    adapter_output: np.ndarray  # (T'', 3584)
    adapter_length: int


def load_encoder_args(asr_encoder_pth_tar: Path):
    import torch

    package = torch.load(str(asr_encoder_pth_tar), map_location="cpu", weights_only=False)
    return package["args"]


def load_encoder_and_adapter_weights(model_safetensors_or_pth: Path) -> dict[str, "torch.Tensor"]:
    """Load the `encoder.*` + `encoder_projector.*` prefixed tensors (real
    converted weights; the small `asr_encoder.pth.tar` file only carries
    architecture args, see module docstring / README) from either the
    safetensors export or the original .pth.tar, without materializing the
    161M LoRA tensors that live alongside them in the same checkpoint."""
    import torch

    path = model_safetensors_or_pth
    state: dict[str, torch.Tensor] = {}
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("encoder.") or key.startswith("encoder_projector."):
                    state[key] = f.get_tensor(key)
    else:
        package = torch.load(str(path), map_location="cpu", weights_only=False)
        for key, tensor in package["model_state_dict"].items():
            if key.startswith("encoder.") or key.startswith("encoder_projector."):
                state[key] = tensor
    if not state:
        raise RuntimeError(f"no encoder./encoder_projector. tensors found in {path}")
    return state


def build_encoder_and_adapter(args, state: dict[str, "torch.Tensor"]):
    from fireredasr2.models.module.adapter import Adapter
    from fireredasr2.models.module.conformer_encoder import ConformerEncoder

    encoder = ConformerEncoder(
        args.idim,
        args.n_layers_enc,
        args.n_head,
        args.d_model,
        args.residual_dropout,
        args.dropout_rate,
        args.kernel_size,
        args.pe_maxlen,
    )
    encoder_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"encoder state_dict mismatch: missing={missing} unexpected={unexpected}")
    encoder.eval()

    adapter = Adapter(args.d_model, 3584, downsample_rate=2)
    adapter_state = {
        k[len("encoder_projector."):]: v for k, v in state.items() if k.startswith("encoder_projector.")
    }
    missing, unexpected = adapter.load_state_dict(adapter_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"adapter state_dict mismatch: missing={missing} unexpected={unexpected}")
    adapter.eval()
    return encoder, adapter


def run_encoder_and_adapter(encoder, adapter, cmvn_fbank: np.ndarray) -> EncoderAdapterDump:
    import torch

    feat = torch.from_numpy(cmvn_fbank).float().unsqueeze(0)  # (1, T, 80)
    length = torch.tensor([feat.shape[1]], dtype=torch.long)

    layer_outputs: list[np.ndarray] = []
    hooks = []
    for block in encoder.layer_stack:
        def _hook(_module, _inp, out, _store=layer_outputs):
            _store.append(out.detach().numpy().astype(np.float32)[0].copy())

        hooks.append(block.register_forward_hook(_hook))

    with torch.no_grad():
        enc_out, enc_len, _enc_mask = encoder(feat, length)
        adapter_out, adapter_len = adapter(enc_out, enc_len)

    for h in hooks:
        h.remove()

    return EncoderAdapterDump(
        encoder_layer_outputs=layer_outputs,
        encoder_output=enc_out.detach().numpy().astype(np.float32)[0],
        encoder_length=int(enc_len[0].item()),
        adapter_output=adapter_out.detach().numpy().astype(np.float32)[0],
        adapter_length=int(adapter_len[0].item()),
    )


def adapter_ggml_pack_crosscheck(oasr_pack: Path, encoder_output: np.ndarray, encoder_length: int) -> dict[str, Any]:
    """Cross-check: run the SAME adapter math (2x frame-stack -> linear1 ->
    relu -> linear2) using the adapter weights actually shipped in a built
    `.oasr` pack (read via `gguf`, dequantized to f32) against the reference
    encoder output computed above, and diff against the official torch
    Adapter's own output on the same input. This isolates the adapter
    weight-conversion + packaging path from the ggml execution engine
    (running the real Rust/ggml executor is out of scope for this python-
    only dumper) but still exercises real packaged tensors end-to-end
    through the adapter's arithmetic, not just a per-tensor weight diff."""
    from gguf import GGUFReader

    reader = GGUFReader(str(oasr_pack))
    by_name = {t.name: t for t in reader.tensors}

    def pack_f32(name: str) -> np.ndarray:
        t = by_name[name]
        return np.asarray(t.data).astype(np.float32)

    w1 = pack_f32("adapter.linear1.weight")  # (3584, 2560)
    b1 = pack_f32("adapter.linear1.bias")  # (3584,)
    w2 = pack_f32("adapter.linear2.weight")  # (3584, 3584)
    b2 = pack_f32("adapter.linear2.bias")  # (3584,)

    x = encoder_output[:encoder_length]
    t_len = x.shape[0]
    ds = 2
    discard = t_len % ds
    if discard:
        x = x[: t_len - discard]
    stacked = x.reshape(x.shape[0] // ds, x.shape[1] * ds)
    hidden = stacked @ w1.T + b1
    hidden = np.maximum(hidden, 0.0)
    pack_out = hidden @ w2.T + b2
    return {"pack_adapter_output": pack_out.astype(np.float32)}


# ---------------------------------------------------------------------------
# Stage 4: LLM (Qwen2-7B-Instruct, LoRA pre-merged) -- memory-streamed
# ---------------------------------------------------------------------------


class StreamingQwen2:
    """Builds a `Qwen2ForCausalLM` on the meta device (no host memory used
    for parameters) and materializes exactly one decoder layer's weights at
    a time straight from the merged safetensors file, casting fp32 -> fp16
    on the fly, running that layer, then releasing it back to a meta tensor.
    embed_tokens / lm_head / final norm stay resident for the whole run
    (~2.2GB fp16); each decoder layer costs ~0.5GB fp16 while it executes.
    This bounds peak RSS to roughly one layer + the always-resident tensors,
    not the full ~15GB fp16 model, at the cost of re-reading each layer's
    weights from disk on every forward call (acceptable for a one-time
    correctness dump, not a perf-critical path)."""

    def __init__(self, config_path: Path, merged_safetensors: Path, dtype):
        import torch
        from safetensors import safe_open
        from transformers import Qwen2Config, Qwen2ForCausalLM

        self.torch = torch
        self.dtype = dtype
        with open(config_path) as f:
            config_dict = json.load(f)
        self.config = Qwen2Config(**{k: v for k, v in config_dict.items() if k != "architectures"})
        self.num_layers = self.config.num_hidden_layers

        with torch.device("meta"):
            self.model = Qwen2ForCausalLM(self.config)
        self.model.eval()

        # rotary_emb's `inv_freq` buffer is derived purely from config (not a
        # learned/checkpointed tensor), but building the whole module tree
        # under `torch.device("meta")` leaves it as a meta tensor too --
        # rebuild it for real on cpu (cheap: a few KB, config-only).
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

        self.model.model.rotary_emb = Qwen2RotaryEmbedding(config=self.config, device="cpu")

        self._f = safe_open(str(merged_safetensors), framework="pt", device="cpu")

        self._materialize_always_resident()
        self._install_layer_streaming_hooks()

    def _load(self, name: str):
        return self._f.get_tensor(name).to(self.dtype)

    def _materialize_always_resident(self) -> None:
        torch = self.torch
        m = self.model.model
        m.embed_tokens.to_empty(device="cpu")
        m.embed_tokens.load_state_dict({"weight": self._load("model.embed_tokens.weight")}, assign=True)
        m.norm.to_empty(device="cpu")
        m.norm.load_state_dict({"weight": self._load("model.norm.weight")}, assign=True)
        self.model.lm_head.to_empty(device="cpu")
        self.model.lm_head.load_state_dict({"weight": self._load("lm_head.weight")}, assign=True)

    def _layer_state_dict(self, idx: int) -> dict[str, "torch.Tensor"]:
        prefix = f"model.layers.{idx}."
        sd = {}
        for key in self._f.keys():
            if key.startswith(prefix):
                sd[key[len(prefix):]] = self._load(key)
        return sd

    def _install_layer_streaming_hooks(self) -> None:
        for idx, layer in enumerate(self.model.model.layers):
            def _pre(module, _args, _kwargs, _idx=idx):
                module.to_empty(device="cpu")
                module.load_state_dict(self._layer_state_dict(_idx), assign=True)
                return None

            def _post(module, _args, _output, _idx=idx):
                module.to(device="meta")
                gc.collect()

            layer.register_forward_pre_hook(_pre, with_kwargs=True)
            layer.register_forward_hook(_post)

    def close(self) -> None:
        self._f.__exit__(None, None, None) if hasattr(self._f, "__exit__") else None


@dataclass
class LlmDump:
    prompt_ids: list[int]
    speech_token_index: int
    prefill_last_hidden: np.ndarray  # (3584,) post-final-norm hidden of the last prefill position
    prefill_first_logits: np.ndarray  # (vocab,) logits for the first generated token
    decode_steps: list[dict[str, Any]] = field(default_factory=list)
    decoded_text: str = ""


def build_prompt_embeds(streaming_model: StreamingQwen2, tokenizer, speech_features_np: np.ndarray):
    """Batch=1, no-padding specialization of `FireRedAsrLlm.
    _merge_input_ids_with_speech_features`: splice the adapter's speech
    features in at the single `<speech>` placeholder token's position."""
    torch = streaming_model.torch
    from fireredasr2.tokenizer.llm_tokenizer import DEFAULT_SPEECH_TOKEN

    # Mirrors `LlmTokenizerWrapper.preprocess_texts(..., decode=True)` exactly:
    # a user turn (speech placeholder + instruction) followed by an EMPTY
    # assistant turn with no closing `<|im_end|>` -- that open assistant turn
    # is what primes the model to generate the transcript next. Omitting it
    # (an earlier version of this function did) leaves the chat looking
    # unfinished and the model does not know it is its turn to speak yet --
    # confirmed against the same open-`<|im_start|>assistant\n`-turn ChatML
    # priming pattern used by every other speech-LLM arch in
    # references/transcribe.cpp (e.g. src/arch/funasr_nano/model.cpp:253,
    # src/arch/canary_qwen/model.cpp:221) and references/CrispASR
    # (src/moss_audio.cpp:1736, src/crispasr_c_api.cpp:5617).
    TEMPLATE = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}"
        "{% if loop.last %}{{''}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
    )
    messages = [
        {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}请转写音频为文字"},
        {"role": "assistant", "content": ""},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, chat_template=TEMPLATE, add_generation_prompt=False, return_dict=False
    )
    speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    speech_idx = prompt_ids.index(speech_token_id)

    ids_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    embeds = streaming_model.model.model.embed_tokens(ids_tensor)[0]  # (L, 3584)
    speech_embeds = torch.from_numpy(speech_features_np).to(embeds.dtype)  # (S, 3584)

    final_embeds = torch.cat([embeds[:speech_idx], speech_embeds, embeds[speech_idx + 1 :]], dim=0)
    return prompt_ids, speech_idx, final_embeds.unsqueeze(0)  # (1, L', 3584)


def run_llm(streaming_model: StreamingQwen2, tokenizer, speech_features_np: np.ndarray, decode_steps: int) -> LlmDump:
    torch = streaming_model.torch
    prompt_ids, speech_idx, inputs_embeds = build_prompt_embeds(streaming_model, tokenizer, speech_features_np)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long)

    captured: dict[str, Any] = {}

    def _capture_pre_norm(_module, _inp, _store=captured):
        # forward_pre_hook receives (module, args); args[0] is the hidden
        # state about to be fed into the final norm -- exactly the
        # pre-final-norm last-position hidden state we want to dump.
        hidden = _inp[0] if isinstance(_inp, tuple) else _inp
        _store["pre_final_norm_last"] = hidden[0, -1].detach().to(torch.float32).numpy().copy()

    hook = streaming_model.model.model.norm.register_forward_pre_hook(_capture_pre_norm)

    with torch.no_grad():
        out = streaming_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    hook.remove()

    prefill_last_hidden = captured["pre_final_norm_last"]
    first_logits = out.logits[0, -1].detach().to(torch.float32).numpy().copy()

    past_key_values = out.past_key_values
    steps: list[dict[str, Any]] = []
    next_logits = first_logits
    next_token_embed = None
    generated_ids: list[int] = []
    cur_attention_mask = attention_mask
    for step in range(decode_steps):
        token_id = int(np.argmax(next_logits))
        generated_ids.append(token_id)
        steps.append(
            {
                "step": step,
                "token_id": token_id,
                "top1_logit": float(next_logits[token_id]),
                "logits_top5": [int(i) for i in np.argsort(next_logits)[-5:][::-1]],
            }
        )
        token_tensor = torch.tensor([[token_id]], dtype=torch.long)
        token_embed = streaming_model.model.model.embed_tokens(token_tensor)
        cur_attention_mask = torch.cat([cur_attention_mask, torch.ones((1, 1), dtype=torch.long)], dim=1)
        with torch.no_grad():
            out = streaming_model.model(
                inputs_embeds=token_embed,
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = out.past_key_values
        next_logits = out.logits[0, -1].detach().to(torch.float32).numpy().copy()

    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else ""

    return LlmDump(
        prompt_ids=list(prompt_ids),
        speech_token_index=speech_idx,
        prefill_last_hidden=prefill_last_hidden,
        prefill_first_logits=first_logits,
        decode_steps=steps,
        decoded_text=decoded_text,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["fbank", "encoder", "adapter", "llm", "all"], default="all")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output .npz path")
    parser.add_argument(
        "--fireredasr2s-repo",
        type=Path,
        required=True,
        help="path to a clone of https://github.com/FireRedTeam/FireRedASR2S.git "
        "(pin 4e7d9aaf4482a47cec1724807026b9b151926eb5)",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="directory with asr_encoder.pth.tar, cmvn.ark, derived/model.safetensors "
        "(or model.pth.tar), Qwen2-7B-Instruct/config.json, derived/qwen2-merged.safetensors",
    )
    parser.add_argument("--oasr-pack", type=Path, default=None, help="optional built .oasr pack for the adapter cross-check")
    parser.add_argument("--llm-decode-steps", type=int, default=4)
    parser.add_argument("--min-mem-gb-for-llm", type=float, default=6.0)
    parser.add_argument("--skip-mem-check", action="store_true")
    args = parser.parse_args()

    add_refcode_to_path(args.fireredasr2s_repo)

    manifest: dict[str, Any] = {"stage_shapes": {}}
    arrays: dict[str, np.ndarray] = {}

    run_fbank_stage = args.stage in ("fbank", "encoder", "adapter", "llm", "all")
    if not run_fbank_stage:
        raise SystemExit(f"unknown stage {args.stage}")

    print(f"[stage] fbank: {args.wav}", file=sys.stderr)
    fbank = run_fbank(args.wav, args.weights_dir / "cmvn.ark")
    arrays["fbank.raw"] = fbank.raw_fbank
    arrays["fbank.cmvn"] = fbank.cmvn_fbank
    manifest["stage_shapes"]["fbank.raw"] = list(fbank.raw_fbank.shape)
    manifest["stage_shapes"]["fbank.cmvn"] = list(fbank.cmvn_fbank.shape)
    manifest["audio_duration_s"] = fbank.duration_s
    manifest["sample_rate"] = fbank.sample_rate

    enc_adapter_dump = None
    if args.stage in ("encoder", "adapter", "llm", "all"):
        print("[stage] encoder+adapter: loading official ConformerEncoder + Adapter", file=sys.stderr)
        enc_args = load_encoder_args(args.weights_dir / "asr_encoder.pth.tar")
        model_weights_path = args.weights_dir / "derived" / "model.safetensors"
        if not model_weights_path.exists():
            model_weights_path = args.weights_dir / "model.pth.tar"
        state = load_encoder_and_adapter_weights(model_weights_path)
        encoder, adapter = build_encoder_and_adapter(enc_args, state)
        enc_adapter_dump = run_encoder_and_adapter(encoder, adapter, fbank.cmvn_fbank)

        for i, layer_out in enumerate(enc_adapter_dump.encoder_layer_outputs):
            arrays[f"encoder.layer_{i}"] = layer_out
            manifest["stage_shapes"][f"encoder.layer_{i}"] = list(layer_out.shape)
        arrays["encoder.output"] = enc_adapter_dump.encoder_output
        manifest["stage_shapes"]["encoder.output"] = list(enc_adapter_dump.encoder_output.shape)
        manifest["encoder.length"] = enc_adapter_dump.encoder_length

        arrays["adapter.output"] = enc_adapter_dump.adapter_output
        manifest["stage_shapes"]["adapter.output"] = list(enc_adapter_dump.adapter_output.shape)
        manifest["adapter.length"] = enc_adapter_dump.adapter_length

        if args.oasr_pack is not None:
            print(f"[stage] adapter ggml-pack crosscheck: {args.oasr_pack}", file=sys.stderr)
            cross = adapter_ggml_pack_crosscheck(
                args.oasr_pack, enc_adapter_dump.encoder_output, enc_adapter_dump.encoder_length
            )
            pack_out = cross["pack_adapter_output"]
            ref_out = enc_adapter_dump.adapter_output[: pack_out.shape[0]]
            max_abs_diff = float(np.abs(pack_out - ref_out).max())
            manifest["adapter_pack_crosscheck_max_abs_diff"] = max_abs_diff
            arrays["adapter.pack_crosscheck_output"] = pack_out
            print(f"[crosscheck] adapter activation max_abs_diff (official fp32 vs .oasr pack weights) = {max_abs_diff:.6g}", file=sys.stderr)

    if args.stage in ("llm", "all"):
        if not args.skip_mem_check:
            wait_for_memory(args.min_mem_gb_for_llm)
        print("[stage] llm: streaming Qwen2-7B-Instruct (fp16, one layer resident at a time)", file=sys.stderr)
        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.weights_dir / "Qwen2-7B-Instruct"))
        tokenizer.add_special_tokens({"additional_special_tokens": ["<speech>"]})

        streaming_model = StreamingQwen2(
            config_path=args.weights_dir / "Qwen2-7B-Instruct" / "config.json",
            merged_safetensors=args.weights_dir / "derived" / "qwen2-merged.safetensors",
            dtype=torch.float16,
        )
        llm_dump = run_llm(streaming_model, tokenizer, enc_adapter_dump.adapter_output, args.llm_decode_steps)

        arrays["llm.prefill_last_hidden"] = llm_dump.prefill_last_hidden
        arrays["llm.prefill_first_logits"] = llm_dump.prefill_first_logits
        manifest["stage_shapes"]["llm.prefill_last_hidden"] = list(llm_dump.prefill_last_hidden.shape)
        manifest["stage_shapes"]["llm.prefill_first_logits"] = list(llm_dump.prefill_first_logits.shape)
        manifest["llm.prompt_ids"] = llm_dump.prompt_ids
        manifest["llm.speech_token_index"] = llm_dump.speech_token_index
        manifest["llm.decode_steps"] = llm_dump.decode_steps
        manifest["llm.decoded_text"] = llm_dump.decoded_text
        print(f"[llm] decoded (first {args.llm_decode_steps} greedy tokens): {llm_dump.decoded_text!r}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    manifest_path = args.out.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"wrote {args.out} ({sum(a.nbytes for a in arrays.values()) / 1e6:.1f} MB arrays)", file=sys.stderr)
    print(f"wrote {manifest_path}", file=sys.stderr)
    print(json.dumps(manifest["stage_shapes"], indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
