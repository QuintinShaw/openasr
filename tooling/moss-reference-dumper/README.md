# moss-transcribe-diarize reference dumper

Runs the **official** OpenMOSS/MOSS-Transcribe-Diarize python reference
implementation on real fixture wavs and dumps its output, so the ggml side
(`crates/openasr-core/src/models/moss_transcribe_diarize/`) can be diffed
against ground truth. Two scripts:

- `dump_golden.py`: full greedy `model.generate()` -> golden transcript JSON
  (token ids, decoded text, parsed speaker/timestamp segments). This is the
  process that produced `tmp/moss-td/golden/*.json` and, transcribed from
  those, the `GOLDEN_JFK_TEXT` / `GOLDEN_EN_ZH_MIXED_TEXT` constants pinned
  in `executor.rs`'s `#[ignore]`d `golden_diff_end_to_end_transcribe_*` tests
  -- this script makes that generation reproducible in-tree instead of
  leaving it as an untracked one-off.
- `dump_intermediate.py`: single-clip stage-by-stage activation dump
  (Whisper-Medium encoder output, post-time-merge features, VQAdaptor
  output, first prefill-step logits) as `.npy` files, for diffing individual
  ggml execution stages against the reference forward pass.

Both are this family's "reference dumper" required by
[`docs/model-audits/TEMPLATE.md`](../../docs/model-audits/TEMPLATE.md)
section 10 ("Reference dumper exists for this family").

Nothing here is vendored into the repo: no third-party code, no weights, no
dump output. All of that lives outside the tracked tree (see below).

## Official reference source (do not vendor -- clone locally)

- Code: <https://github.com/OpenMOSS/MOSS-Transcribe-Diarize>, pinned commit
  `40cf8549c6b5634ba36b7e817cf523b5ad400c2e` (2026-07-20). Clone it yourself:

  ```bash
  git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git /path/to/moss-td-refcode
  cd /path/to/moss-td-refcode && git checkout 40cf8549c6b5634ba36b7e817cf523b5ad400c2e
  ```

  Both scripts only import from this clone for `parse_transcript` /
  `inference_utils` (`build_transcription_messages`, `prepare_inputs`) --
  the model/processor code itself loads straight off the checkpoint via
  `trust_remote_code=True` (the checkpoint ships its own
  `modeling_moss_transcribe_diarize.py` / `processing_moss_transcribe_diarize.py`,
  same convention `AutoModelForCausalLM.from_pretrained(...,
  trust_remote_code=True)` uses for any custom-code HF repo), so no code
  clone is strictly required just to run the model -- only to reach
  `parse_transcript`, which lives in the repo's installable package rather
  than the checkpoint's `trust_remote_code` module set.

- Weights: <https://huggingface.co/OpenMOSS/MOSS-Transcribe-Diarize> (gated;
  request access on the HF page). Point `--weights-dir` at a local copy laid
  out exactly like the upstream HF repo:

  ```text
  <weights-dir>/
    config.json
    configuration_moss_transcribe_diarize.py
    modeling_moss_transcribe_diarize.py
    processing_moss_transcribe_diarize.py
    processor_config.json
    preprocessor_config.json
    generation_config.json
    tokenizer_config.json
    special_tokens_map.json
    added_tokens.json
    vocab.json
    merges.txt
    tokenizer.json
    model-00000-of-00001.safetensors   # single shard, ~1.8GB fp32
    model.safetensors.index.json
  ```

## Setup

Host python (this repo has no dedicated venv for tooling scripts; these are
all small pure/wheel packages, matching `firered2-reference-dumper`'s
convention):

```bash
python3 -m pip install --user "transformers>=5.0,<6.0" "torch>=2.8" numpy
```

Verified against `transformers==5.13.0` / `torch==2.12.0`.

## Memory

The whole checkpoint (Whisper-Medium encoder + `VQAdaptor` + Qwen3-0.6B
decoder, ~1.6B params combined) fits comfortably in fp32 on a 16GB dev
machine -- well under 4GB resident. Unlike `firered2-reference-dumper`'s
7B-parameter Qwen2 decoder, this family needs none of that dumper's
meta-device layer-streaming trick or `vm_stat`-gated wait loop:
`AutoModelForCausalLM.from_pretrained(..., dtype=torch.float32)` loads the
whole model directly, so both scripts here have no `--stage` flag or memory
gate to make that trade-off with.

## `dump_golden.py` usage

```bash
cd tooling/moss-reference-dumper
python3 dump_golden.py \
  --moss-repo /path/to/moss-td-refcode \
  --weights-dir /path/to/moss-transcribe-diarize-weights \
  --samples-dir ../../fixtures \
  --sample jfk=jfk.wav \
  --out-dir /path/to/scratch/moss-td-golden
```

`--sample NAME=RELATIVE_WAV_PATH` is repeatable; each path resolves against
`--samples-dir`. Writes `<out-dir>/<name>.json`, one record per sample, in
the exact shape the committed golden fixtures were transcribed from: prompt
token ids, full/generated token ids, decoded text (with and without special
tokens), elapsed time, environment versions, and `parse_transcript`'s parsed
`[start, end, speaker, text]` segments.

Greedy (`do_sample=False`), CPU, fp32, seeded (`--seed`, default `0`) for
determinism -- matches the golden-diff convention every other builtin
family's dev-only reference dump uses (see e.g.
`firered2-reference-dumper/dump_reference.py`'s `llm` stage).

## `dump_intermediate.py` usage

```bash
cd tooling/moss-reference-dumper
python3 dump_intermediate.py \
  --moss-repo /path/to/moss-td-refcode \
  --weights-dir /path/to/moss-transcribe-diarize-weights \
  --wav ../../fixtures/jfk.wav \
  --out-dir /path/to/scratch/moss-td-dump
```

Dumps `input_features.npy`, `whisper_encoder.npy`, `merged.npy`,
`adaptor.npy`, and `prefill_last_logits.npy` -- see the script's module doc
for exact shapes and semantics. Only single-chunk (<=30s) wavs are supported
by this dumper's trim math (it mirrors the executor's single-chunk trim, not
its multi-chunk concatenation loop); `dump_golden.py`'s full `generate()`
call already exercises the multi-chunk path correctly via the checkpoint's
own `get_audio_features`, it just does not expose these intermediate
per-stage tensors.

## Verified results (2026-07-21, Apple M1 16GB)

- `jfk.wav`, `en_zh_mixed.wav`, and a 3-minute multi-speaker `aishell4`
  clip all ran end to end through `dump_golden.py`'s predecessor; their
  outputs are what `GOLDEN_JFK_TEXT` / `GOLDEN_EN_ZH_MIXED_TEXT` in
  `executor.rs` were transcribed from (see those constants' own doc comments
  for the exact fp16+flash-vs-fp32 numeric nuance: byte-identical text on
  `jfk`/`aishell4`, two 0.02s time-anchor shifts on `en_zh_mixed`).
- Peak RSS during the full pipeline (model load + encoder + adaptor +
  greedy generate to `<|im_end|>`): well under 4GB, consistent with the
  "Memory" section above.

## Scope

Neither script here has a self-test: both are thin, mostly-I/O wrappers
around the official reference stack's own `generate()` / module forward
calls (unlike `firered2-reference-dumper/dump_reference.py`, which has
enough standalone arithmetic -- the memory-check parser, the adapter's
frame-stacking math -- to make a weight-free unit test worthwhile). The
actual correctness check for this dumper is a real run against the real
checkpoint, cross-referenced against the ggml runtime's own committed
`golden_diff_*` tests in `executor.rs`.
