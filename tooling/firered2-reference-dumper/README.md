# firered2-llm reference dumper

Runs the **official** FireRedASR2S python reference implementation
(FireRedTeam/FireRedASR2-LLM, Encoder-Adapter-LLM) stage-by-stage on a real
fixture wav and dumps every intermediate activation to an `.npz` file, so the
ggml side (`crates/openasr-core/src/models/firered_llm`) can be diffed
against ground truth. This is the family's "reference dumper" required by
[`docs/model-audits/TEMPLATE.md`](../../docs/model-audits/TEMPLATE.md)
section 10 ("Reference dumper exists for this family").

Nothing here is vendored into the repo: no third-party code, no weights, no
dump output. All of that lives outside the tracked tree (see below).

## What it dumps

| stage     | contents |
| --- | --- |
| `fbank`   | 80-dim kaldi fbank pre- and post-CMVN (`fbank.raw`, `fbank.cmvn`) |
| `encoder` | every one of the 16 Conformer layers' output (`encoder.layer_0`..`encoder.layer_15`) plus the final `encoder.output` |
| `adapter` | 2x frame-stacked Linear-ReLU-Linear projector output (`adapter.output`); optionally an **activation-level cross-check** against a built `.oasr` pack's adapter weights (`adapter.pack_crosscheck_output` + `adapter_pack_crosscheck_max_abs_diff` in the manifest) |
| `llm`     | Qwen2-7B-Instruct (LoRA pre-merged) prefill last-hidden-state (pre-final-norm) and first-token logits, plus the first N greedy decode steps (token id, top-1 logit, top-5 candidates, decoded text) |

Every stage writes float32 numpy arrays into one `.npz`; a sibling
`<out>.manifest.json` records shapes, lengths, prompt token ids, and the
decode-step summary.

## Official reference source (do not vendor -- clone locally)

- Code: <https://github.com/FireRedTeam/FireRedASR2S>, pinned commit
  `4e7d9aaf4482a47cec1724807026b9b151926eb5` (2026-06-02). Clone it yourself:

  ```bash
  git clone https://github.com/FireRedTeam/FireRedASR2S.git /path/to/fr2-refcode
  cd /path/to/fr2-refcode && git checkout 4e7d9aaf4482a47cec1724807026b9b151926eb5
  ```

- Weights: <https://huggingface.co/FireRedTeam/FireRedASR2-LLM>. Point
  `--weights-dir` at a local copy laid out like the upstream repo:

  ```text
  <weights-dir>/
    asr_encoder.pth.tar               # AED encoder architecture args only
    model.pth.tar                     # encoder + adapter + LoRA state dict
                                       # (or derived/model.safetensors, same
                                       # content -- see pt_to_safetensors.py)
    cmvn.ark                          # kaldi global CMVN stats
    Qwen2-7B-Instruct/                # tokenizer + config.json
    derived/qwen2-merged.safetensors  # LoRA-merged Qwen2 base weights,
                                       # produced by
                                       # tooling/publish-model/scripts/firered_llm_merge_lora.py
  ```

  This dumper loads the **already LoRA-merged** Qwen2 checkpoint rather than
  re-running `peft` at load time: the merge is a fixed linear-algebra step
  (`scaling * lora_B @ lora_A` folded into the base weight, see
  `firered_llm_merge_lora.py`'s docstring) that was already audited when the
  `.oasr` importer was built, so re-deriving it here would just add a `peft`
  dependency for zero additional coverage.

## Setup

Host python (this repo has no dedicated venv for tooling scripts; these are
all small pure/wheel packages):

```bash
python3 -m pip install --user torch transformers safetensors numpy \
    kaldi_native_fbank kaldiio gguf accelerate
```

## Usage

```bash
cd tooling/firered2-reference-dumper
python3 dump_reference.py \
  --stage all \
  --wav ../../fixtures/jfk.wav \
  --out /path/to/scratch/fr2-dump-jfk.npz \
  --fireredasr2s-repo /path/to/fr2-refcode \
  --weights-dir /path/to/firered2-llm-weights \
  --oasr-pack /path/to/firered2-llm-fp16.oasr \
  --llm-decode-steps 4
```

`--stage` accepts `fbank`, `encoder`, `adapter`, `llm`, or `all` (default);
each earlier stage's output feeds the next, so requesting a later stage
still runs everything before it. `--oasr-pack` is optional -- only needed
for the adapter cross-check.

## Memory discipline (16GB dev machines)

`fbank`/`encoder`/`adapter` fit comfortably (well under 4GB total, fp32
throughout -- the encoder is 723M params / ~2.8GB fp32, the adapter 22M
params / ~84MB).

The `llm` stage is the expensive one: Qwen2-7B-Instruct is ~7.6B params,
~15GB at fp16 -- too much to hold resident on a 16GB machine alongside
anything else. Instead of `from_pretrained`-ing the whole model,
`StreamingQwen2` builds the HF module graph on the **meta device** (zero
host memory for parameters), then materializes **one decoder layer's**
weights at a time straight off the merged safetensors file via forward
pre/post hooks -- casting fp32 -> fp16 on the fly -- runs that layer, and
releases it back to a meta tensor (`layer.to(device="meta")`) before the
next layer loads. `embed_tokens` / `lm_head` / final `norm` stay resident
for the whole run (~2.2GB fp16 combined, since they're not tied and both
sized `vocab_size x hidden_size`); each decoder layer costs ~0.5GB fp16
while it executes. Peak extra RSS for the LLM stage is therefore bounded by
roughly one layer + the always-resident tensors, not the full ~15GB model --
at the cost of re-reading each layer's weights from disk on every forward
call (prefill + every decode step), which is the right trade for a one-time
correctness dump, not a performance probe (do not use this path or its
numbers for any RTF/latency claim).

`--min-mem-gb-for-llm` (default 6.0) and `--skip-mem-check` control a
`vm_stat`-based wait loop before the `llm` stage starts, since this is meant
to run alongside other work (compiles, other tooling) on a shared dev
machine rather than claim it exclusively.

## Verified results (2026-07-22, jfk.wav, Apple M1 16GB, `--stage all`)

- `fbank`: 1098 raw frames x 80 mel bins (11.0s @ 16kHz).
- `encoder`: 275 frames x 1280 (16 layers, all dumped).
- `adapter`: 137 frames x 3584.
- Adapter activation cross-check (official fp32 forward vs the shipped
  `firered2-llm-fp16.oasr` pack's adapter weights, same real encoder output
  as input): **max_abs_diff = 5.76e-4** -- consistent with fp16 rounding
  through two chained linear layers, not a correctness bug.
- `llm`: prefill + 6 greedy decode steps on the streamed Qwen2-7B-Instruct
  decoded to **`"and so my fellow americans"`**, an exact prefix match of
  the ggml runtime's own committed golden transcript for this fixture
  (`GOLDEN_JFK_TEXT` in
  `crates/openasr-core/src/models/firered_llm/executor.rs`: `"and so my
  fellow americans ask not what your country can do for you ask what you
  can do for your country"`). This is the strongest available end-to-end
  cross-check between the official python reference and the ggml runtime
  without instrumenting the Rust executor to dump its own intermediate
  tensors (out of scope for this python-only dumper).
- Peak RSS observed during the `llm` stage: ~3.5-5.0GB (oscillating with
  each layer's materialize/free cycle), on a machine with only ~5-9GB free
  at the time -- confirms the streaming design holds to its budget.

## Self-tests

```bash
cd tooling/firered2-reference-dumper
python3 -m unittest dump_reference_test -v
```

These cover the memory-check parsing/wait-loop logic, the adapter's
frame-stacking + linear-relu-linear arithmetic, and the batch=1
speech-token-splicing logic (`build_prompt_embeds`) against hand-computed
references and fakes -- no official repo clone, weights, or GPU/network
access required. They do **not** exercise the real encoder/adapter/LLM
forward passes end-to-end (that needs the real weights and repo clone above)
-- treat a real `--stage all` run against `fixtures/jfk.wav` as the
end-to-end check.
