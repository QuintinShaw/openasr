# qwen3-asr reference dumper

Runs the **official** `qwen_asr` python reference implementation (the
transformers backend of `QwenLM/Qwen3-ASR`) on real fixture wavs and dumps its
output, so the ggml side (`crates/openasr-core/src/models/qwen/`) can be diffed
against ground truth. Two scripts:

- `dump_golden.py`: full greedy `model.generate()` -> golden transcript JSON
  (prompt / generated / full token ids, decoded text with and without special
  tokens, parsed `language` + transcript `asr_text`). This is the script that
  produces the records the `GOLDEN_*_TEXT` constants pinned in
  `ggml_executor.rs`'s golden-diff tests are transcribed from -- the qwen
  family does not yet pin golden transcripts in-tree, so this script makes that
  generation reproducible rather than leaving it as an untracked one-off (same
  role `../moss-reference-dumper/dump_golden.py` plays for
  moss-transcribe-diarize).
- `dump_intermediate.py`: single-clip stage-by-stage activation dump (mel
  frontend output, audio-encoder output post-`ln_post`, projector/adaptor
  output, first prefill-step logits) as `.npy` files, for diffing individual
  ggml execution stages against the reference forward pass.

Both are this family's "reference dumper" required by
[`docs/model-audits/TEMPLATE.md`](../../docs/model-audits/TEMPLATE.md)
section 10 ("Reference dumper exists for this family").

Nothing here is vendored into the repo: no third-party code, no weights, no
dump output. All of that lives outside the tracked tree (see below).

## Official reference source (do not vendor -- clone locally)

- Code: <https://github.com/QwenLM/Qwen3-ASR>, pinned commit
  `7c6daf77a2421100f5fb066495372c00129d39ff` (2026-06-26). Clone it yourself:

  ```bash
  git clone https://github.com/QwenLM/Qwen3-ASR.git /path/to/Qwen3-ASR
  cd /path/to/Qwen3-ASR && git checkout 7c6daf77a2421100f5fb066495372c00129d39ff
  ```

  `import qwen_asr` from that clone registers the family's `Qwen3ASRConfig` /
  `Qwen3ASRForConditionalGeneration` / `Qwen3ASRProcessor` with transformers'
  Auto classes, after which `AutoModel.from_pretrained` /
  `AutoProcessor.from_pretrained` resolve the model straight off the checkpoint
  directory. Unlike moss, the checkpoint ships **no** `modeling_*.py` of its own
  (its `config.json` has `architectures: ["Qwen3ASRForConditionalGeneration"]`
  and `model_type: "qwen3_asr"` but no `trust_remote_code` module set) -- the
  modeling lives in the pip-installable `qwen_asr` package, so that package is
  the single source of the forward pass both scripts run.

- Weights: <https://huggingface.co/Qwen/Qwen3-ASR-0.6B> (open, Apache-2.0).
  Point `--weights-dir` at a local copy laid out exactly like the upstream HF
  repo:

  ```text
  <weights-dir>/
    config.json
    generation_config.json
    chat_template.json
    preprocessor_config.json
    tokenizer_config.json
    special_tokens_map.json          # if present
    vocab.json
    merges.txt
    model.safetensors                # single shard, ~1.2GB fp16 on disk
  ```

## Setup

Host python (this repo has no dedicated venv for tooling scripts, matching
`firered2-reference-dumper` / `moss-reference-dumper` convention). **Note the
transformers pin differs from moss**: `qwen_asr` requires `transformers==4.57.6`
(its `AutoConfig.register("qwen3_asr", ...)` would collide with the model type
transformers later merged natively), so pin that exact version rather than
moss's `transformers>=5.0`:

```bash
python3 -m pip install --user "transformers==4.57.6" "torch>=2.8" numpy \
    librosa soundfile nagisa
```

`librosa`/`soundfile` come from `qwen_asr.inference.utils`' audio loading and
`nagisa` from its forced-aligner module import -- both are pulled in by a plain
`import qwen_asr` even though neither script runs the aligner. `soynlp`,
`gradio`, `flask`, and `vllm` are NOT needed here. Alternatively,
`python3 -m pip install --user -e /path/to/Qwen3-ASR` installs the official full
dependency set (including the gradio/flask the dumper does not use).

## Memory

The whole checkpoint (~0.6B params: 18-layer Whisper-style audio encoder +
audio projector + 28-layer Qwen3 text decoder) fits comfortably in fp32 on a
16GB dev machine -- well under 4GB resident. Unlike
`firered2-reference-dumper`'s 7B-parameter Qwen2 decoder, this family needs none
of that dumper's meta-device layer-streaming trick or `vm_stat`-gated wait loop:
`AutoModel.from_pretrained(..., dtype=torch.float32)` loads the whole model
directly, so neither script here has a `--stage` flag or memory gate to make
that trade-off with.

## `dump_golden.py` usage

```bash
cd tooling/qwen-reference-dumper
python3 dump_golden.py \
  --qwen-repo /path/to/Qwen3-ASR \
  --weights-dir /path/to/qwen3-asr-0.6b-weights \
  --samples-dir ../../fixtures \
  --sample jfk=jfk.wav \
  --out-dir /path/to/scratch/qwen-golden
```

`--sample NAME=RELATIVE_WAV_PATH` is repeatable; each path resolves against
`--samples-dir`. Writes `<out-dir>/<name>.json`, one record per sample: prompt
token ids, full/generated token ids, decoded text (with and without special
tokens), parsed `language` + transcript `asr_text`, elapsed time, and
environment versions -- the same field-for-field shape as moss's golden records,
with qwen's `language` / `asr_text` in place of moss's speaker/timestamp
`segments`.

**Which field is the future golden constant?** The ggml executor strips the
`language <name><asr_text>` control prefix and emits just the transcript (see
`greedy_decode_strips_qwen_asr_control_prefix` in `greedy_decode.rs`). That
transcript is byte-for-byte what `parse_asr_output` returns as its second
element, so the `asr_text` field -- not the raw `text` field, which still carries
the control prefix -- is the one a future `GOLDEN_*_TEXT` constant in
`ggml_executor.rs` should match. Both are recorded so the relationship is
auditable.

Greedy (`do_sample=False`), CPU, fp32, seeded (`--seed`, default `0`) for
determinism -- matches the golden-diff convention every other builtin family's
dev-only reference dump uses (see `../moss-reference-dumper/dump_golden.py` and
`../firered2-reference-dumper/dump_reference.py`'s `llm` stage).

## `dump_intermediate.py` usage

```bash
cd tooling/qwen-reference-dumper
python3 dump_intermediate.py \
  --qwen-repo /path/to/Qwen3-ASR \
  --weights-dir /path/to/qwen3-asr-0.6b-weights \
  --wav ../../fixtures/jfk.wav \
  --out-dir /path/to/scratch/qwen-dump
```

Dumps `input_features.npy`, `feature_attention_mask.npy`, `audio_encoder.npy`,
`audio_adaptor.npy`, and `prefill_last_logits.npy` -- see the script's module doc
for exact shapes and semantics.

Encoder/adaptor boundary: in the official `Qwen3ASRAudioEncoder.forward` the
adaptor is fused at the tail of the encoder module --
`... -> encoder layers -> ln_post -> proj1 -> act -> proj2 -> output`. So
`audio_encoder.npy` is the `ln_post` output (the encoder proper, 896-dim, tapped
with a forward hook) and `audio_adaptor.npy` is the projector output
(896 -> 1024-dim, matching the LLM `d_model`) spliced into the LLM prompt at the
audio-pad token positions.

The greedy-decode stage of the fbank -> encoder -> adaptor -> LLM-prefill ->
greedy-decode pipeline is `dump_golden.py`'s full `model.generate()` call, not
this script (mirrors moss, whose `dump_intermediate.py` likewise stops at the
prefill logits and leaves the full decode to `dump_golden.py`).

Unlike the moss dumper, there is no single-chunk (<=30s) restriction here:
`Qwen3ASRAudioEncoder` does its own windowed-attention chunking (`n_window=50`,
i.e. 100-frame / 1s chunks), so `get_audio_features` handles arbitrarily long
clips (up to the reference stack's 1200s cap) and the dumped tensors are the real
packed valid-frame sequence, not a single-chunk trim.

## Verification status

Neither script has been run against the real checkpoint yet: doing so requires
downloading the ~1.2GB `Qwen/Qwen3-ASR-0.6B` weights, which is deferred to a
dedicated measurement window. Both byte-compile (`python3 -m py_compile`) and
follow the official `qwen_asr` transformers-backend call path exactly
(`AutoModel.from_pretrained` + `processor.apply_chat_template` +
`processor(text=..., audio=...)` + `model.generate`, and
`thinker.get_audio_features` / `thinker(...)` for the per-stage taps), but the
numeric cross-check against the ggml runtime's `golden_diff_*` tests is still
outstanding. Do not treat the dump shapes as verified until a real run lands.

## Scope

Neither script here has a self-test: both are thin, mostly-I/O wrappers around
the official reference stack's own `generate()` / module forward calls (unlike
`firered2-reference-dumper/dump_reference.py`, which has enough standalone
arithmetic to make a weight-free unit test worthwhile). The actual correctness
check for this dumper is a real run against the real checkpoint,
cross-referenced against the ggml runtime's own committed `golden_diff_*` tests
in `ggml_executor.rs` (once those pin `GOLDEN_*_TEXT` constants transcribed from
`dump_golden.py`'s output).
