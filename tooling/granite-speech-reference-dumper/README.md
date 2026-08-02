# granite-speech reference dumper

Runs the **official** Hugging Face `transformers` reference implementation of
IBM Granite Speech 4.1 (`ibm-granite/granite-speech-4.1-2b`) on real fixture
wavs and dumps its output, so the ggml side
(`crates/openasr-core/src/models/granite_speech/`) can be diffed against ground
truth. Two scripts:

- `dump_golden.py`: full greedy `model.generate()` -> golden transcript JSON
  (prompt / generated / full token ids, decoded text with and without special
  tokens). This is the process intended to produce the records any future
  `GOLDEN_*_TEXT` constants in `executor.rs` are transcribed from -- the same
  role `../moss-reference-dumper/dump_golden.py` plays for moss-transcribe-diarize
  and `../qwen-reference-dumper/dump_golden.py` plays for qwen3-asr.
- `dump_intermediate.py`: single-clip stage-by-stage activation dump matching
  the names `parity.rs`'s ignored harnesses load
  (`{sample}_input_features.npy`, `{sample}_encoder_mid_block_out.npy`,
  `{sample}_encoder_out.npy`, `{sample}_projector_out.npy`, plus the pure-decoder
  fixtures `decoder_input_ids.npy` / `decoder_hidden_out.npy` /
  `decoder_logits.npy` / `decoder_embed_scaled.npy`, and the full
  audio-spliced prefill last-position logits). Used to regenerate the
  `tmp/granite-work/golden/` fixtures the parity harness already consumes.

Both are this family's "reference dumper" required by
[`docs/model-audits/TEMPLATE.md`](../../docs/model-audits/TEMPLATE.md)
section 10 ("Reference dumper exists for this family").

Nothing here is vendored into the repo: no third-party code, no weights, no
dump output. All of that lives outside the tracked tree (see below).

## Architecture stages (what gets dumped)

Granite Speech 4.1 is:

```text
16 kHz mono PCM
  -> GraniteSpeechFeatureExtractor (torchaudio MelSpectrogram n_fft=512 /
     win=400 / hop=160 / n_mels=80, then log10 / floor-clip / /4+1 /
     drop-odd-frame / 2x frame-stack)            -> input_features [1, T, 160]
  -> 16-layer Conformer CTC encoder
       (Shaw rel-pos SA, 4s block-local windows,
        GLU + depthwise-conv, self-conditioned
        CTC tap after layer 8)                   -> encoder_out [1, T, 1024]
                                                   mid_block_out [1, T, 1024]
                                                   (post layer 8, pre CTC tap)
  -> BLIP-2 Q-Former projector
       (window_size=15, downsample_rate=5,
        2 cross-attention layers)                -> projector_out [1, N, 2048]
  -> Granite dense decoder-only LLM
       (GQA + RoPE + SwiGLU + the four Granite
        scaling scalars) with <|audio|> splice   -> prefill logits / generate
```

`dump_intermediate.py` taps every stage above (plus a pure text-only decoder
prefill on a fixed 12-token id sequence, independent of the audio splice, so
the decoder graph can be parity-checked in isolation). `dump_golden.py` runs
the full audio-spliced greedy generate end to end.

## Official reference source (do not vendor -- clone / download locally)

- Modeling code: ships **in** `transformers` natively (no separate code clone,
  no `trust_remote_code`). Supported since `transformers>=4.52.1`; the
  checkpoint's own `config.json` records `transformers_version: "4.57.6"`.
  Verified against `transformers==5.13.0` / `torch==2.12.0` (the host python
  currently used by the other dumpers) -- both load
  `GraniteSpeechForConditionalGeneration` /
  `AutoModelForSpeechSeq2Seq` cleanly.
- Weights: <https://huggingface.co/ibm-granite/granite-speech-4.1-2b>
  (open, Apache-2.0, ~4.6 GB across 3 shards). Point `--weights-dir` at a
  local copy laid out exactly like the upstream HF repo:

  ```text
  <weights-dir>/
    config.json
    generation_config.json          # if present
    preprocessor_config.json
    processor_config.json
    tokenizer_config.json
    special_tokens_map.json
    added_tokens.json
    chat_template.jinja             # or chat_template in tokenizer_config
    vocab.json
    merges.txt
    tokenizer.json
    model-00001-of-00003.safetensors
    model-00002-of-00003.safetensors
    model-00003-of-00003.safetensors
    model.safetensors.index.json
  ```

  Example:

  ```bash
  huggingface-cli download ibm-granite/granite-speech-4.1-2b \
    --local-dir /path/to/granite-speech-4.1-2b-weights
  ```

## Setup

Host python (this repo has no dedicated venv for tooling scripts, matching
`firered2-reference-dumper` / `moss-reference-dumper` / `qwen-reference-dumper`
convention):

```bash
python3 -m pip install --user \
  "transformers>=4.52.1,<6.0" "torch>=2.8" torchaudio soundfile numpy
```

`torchaudio` is required by `GraniteSpeechFeatureExtractor` (its mel frontend
is a `torchaudio.transforms.MelSpectrogram`); `soundfile` is what the scripts
use to load fixture wavs without pulling the full torchaudio I/O stack. The
checkpoint was authored against `transformers==4.57.6`; anything in
`[4.52.1, 6)` that still exposes
`transformers.models.granite_speech.modeling_granite_speech` is expected to
work -- pin tighter only if a future major renames the class.

The tokenizer ships a Mistral-style regex that recent transformers warn about;
both scripts pass `fix_mistral_regex=True` to `AutoProcessor.from_pretrained`
(same fix the qwen dumper uses).

## Memory

The whole checkpoint is ~2B params. Loading it end-to-end in fp32 is ~8 GB of
weights alone and is tight on a 16 GB dev machine once activations land, so
both scripts default to:

- load the checkpoint in **bf16** (the checkpoint's native `config.json`
  `"dtype": "bfloat16"`);
- for intermediate parity dumps, **upcast only the stage under test to fp32**
  (encoder + projector for the audio path; `language_model` for the pure
  decoder path) -- this is the same convention the original out-of-tree
  `tmp/granite-work/dump_{encoder,decoder}_golden.py` scripts used, and
  matches the ggml parity harness's "diff against an f32 PyTorch reference"
  rule;
- for golden generate, keep the full model in bf16 (CPU) and record the dtype
  in the JSON so a future fp16/bf16-vs-fp32 nuance is auditable.

There is no `--stage` / meta-device layer-streaming machinery here (unlike
`firered2-reference-dumper`'s 7B path): 2B bf16 fits, and the fp32 stage
upcast only ever holds one segment at a time.

## `dump_golden.py` usage

```bash
cd tooling/granite-speech-reference-dumper
python3 dump_golden.py \
  --weights-dir /path/to/granite-speech-4.1-2b-weights \
  --samples-dir ../../fixtures \
  --sample jfk=jfk.wav \
  --out-dir /path/to/scratch/granite-speech-golden
```

`--sample NAME=RELATIVE_WAV_PATH` is repeatable; each path resolves against
`--samples-dir`. Writes `<out-dir>/<name>.json`, one record per sample: prompt
token ids, full/generated token ids, decoded text (with and without special
tokens), elapsed time, environment versions, and the exact question / chat
template used.

**Default question** matches the ggml executor
(`crates/openasr-core/src/models/granite_speech/executor.rs`'s
`GRANITE_SPEECH_DEFAULT_QUESTION`):

```text
can you transcribe the speech into a written format?
```

which the chat template expands to the same
`USER: <|audio|>...\\n ASSISTANT:` shape the executor assembles by hand. Override
with `--question` (e.g. the model-card punctuation prompt, or a
`transcribe the speech to text. Keywords: ...` KWB prompt) when dumping a
specific prompt variant.

Greedy (`do_sample=False`, `num_beams=1`), CPU, seeded (`--seed`, default `0`)
for determinism -- matches the golden-diff convention every other builtin
family's dev-only reference dump uses.

## `dump_intermediate.py` usage

```bash
cd tooling/granite-speech-reference-dumper
python3 dump_intermediate.py \
  --weights-dir /path/to/granite-speech-4.1-2b-weights \
  --wav /path/to/en_short.wav \
  --sample-name en_short \
  --out-dir /path/to/scratch/granite-speech-dump
```

Dumps, into `--out-dir` (names match `parity.rs`):

| file | contents | shape |
| --- | --- | --- |
| `{sample}_input_features.npy` | mel frontend output fed to the Conformer encoder | `[1, T, 160]` |
| `{sample}_encoder_mid_block_out.npy` | encoder hidden state after layer 8 (pre CTC self-conditioning tap) | `[1, T, 1024]` |
| `{sample}_encoder_out.npy` | encoder final hidden state | `[1, T, 1024]` |
| `{sample}_projector_out.npy` | Q-Former projector output (spliced into the LLM at `<|audio|>` positions) | `[1, N, 2048]` |
| `decoder_input_ids.npy` | fixed 12-token pure-decoder probe ids (int64) | `[1, 12]` |
| `decoder_embed_scaled.npy` | `embed_tokens(ids) * embedding_multiplier` | `[1, 12, 2048]` |
| `decoder_hidden_out.npy` | pure-decoder last hidden state | `[1, 12, 2048]` |
| `decoder_logits.npy` | pure-decoder logits | `[1, 12, 100353]` |
| `prefill_last_logits.npy` | full audio-spliced prompt forward's last-position logits | `[vocab_size]` |
| `prompt_input_ids.npy` | full audio-spliced prompt token ids (int64; `<|audio|>` already expanded) | `[1, prompt_len]` |

`--sample-name` defaults to the wav stem; set it to `en_short` to regenerate
the exact fixture names `parity.rs` loads today.

The pure-decoder probe is intentionally independent of the wav (a fixed
in-vocab id sequence, no special tokens) so decoder-graph parity does not
depend on the audio splice / chat template -- that path is covered by
`prefill_last_logits.npy` + `dump_golden.py` instead.

## Weight-free smoke test

```bash
cd tooling/granite-speech-reference-dumper
python3 -m unittest dump_dumper_test.py -v
```

Covers argparse / sample parsing / prompt assembly / npy round-trip helpers
only -- no weights, no network. The actual correctness check for this dumper
is a real run against the real checkpoint, cross-referenced against the ggml
runtime's ignored `granite_speech_*_parity` tests in `parity.rs` and any
future `golden_diff_*` pins in `executor.rs`.

## Verification status

The stage names and call path match the out-of-tree scripts that originally
produced `tmp/granite-work/golden/` (the fixtures `parity.rs` already diffs
against). Re-running against a fresh weights checkout is the intended way to
refresh those fixtures; do not commit the `.npy` / `.json` outputs.
