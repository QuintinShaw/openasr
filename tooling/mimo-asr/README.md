# MiMo-V2.5-ASR -> `.oasr` converter (ExternalTooling import surface)

Converts the two upstream MiMo weight repos into a single OpenASR `.oasr`
(GGUF-backed) pack:

- **main model** `XiaomiMiMo/MiMo-V2.5-ASR` (36L Qwen2 backbone + 6L input-local
  transformer + 8 speech-embedding tables), and
- **audio tokenizer** `XiaomiMiMo/MiMo-Audio-Tokenizer` (32L rope encoder +
  conv stem + first 8 RVQ codebooks).

This file is the mimo-asr family's `OpenAsrPackImportSurface::ExternalTooling`
entry: the architecture inventory points at it instead of a Rust `CoreConvert`
importer. The split stops at tensor production -- every pack this script
emits still enters install/run through the SAME production `PackVerifier`
plus the mimo-asr runtime contract (publish staging via
`openasr model-pack preflight --stage`, content-store admission on pull, and
the direct `--model-pack` ingress). To keep that gate honest the converter
writes the full public envelope (`openasr.*` routing keys, the tokenizer id,
and build provenance when the pipeline claims it), always bakes the mel
filterbank/window and the gpt2 tokenizer (both are runtime-contract
requirements, not options), and fails closed instead of emitting a pack the
runtime contract would reject. Nothing here touches the catalog or
`model-registry/` directly; those are generated outputs of the publish lane.

## Usage

```bash
python3 convert_mimo_asr.py \
  --main-dir   /path/to/MiMo-V2.5-ASR \
  --tokenizer  /path/to/MiMo-Audio-Tokenizer/model.safetensors \
  --out        /path/to/out/mimo-v2.5-asr-q8_0.oasr \
  --package-id mimo-v2.5-asr \
  --quant      q8-0                 # CLI canonical token: fp16 | q8-0 | q4-k
```

One invocation produces exactly one pack at `--out` (the publish lane owns
the naming via its `external_converter` template in
`tooling/publish-model/models-publish.toml`). `--quant` takes the lane's CLI
canonical tokens; `openasr.pack.quant` records the canonical label
(`fp16` / `q8_0` / `q4_k`).

**`q4-k` currently fails closed.** The gguf Python library implements only
legacy-quant block quantization (Q4_0..Q8_0); K-quant math belongs to ggml's
single source of truth, so this converter refuses to fork it in Python. The
public catalog's q4_k tier needs a Rust-side repack/requant seam; until that
exists, produce fp16/q8_0 here.

Set `OPENASR_BUILD_COMMIT=<40-hex sha>` to bake `openasr.build.commit`
provenance (the publish pipeline's `convert.sh` always exports it); a set but
malformed value fails the conversion.

The source MUST contain `vocab.json` + `merges.txt` (+ optionally
`tokenizer_config.json`'s `added_tokens_decoder`): the converter bakes
`tokenizer.ggml.{model,tokens,merges}` (gpt2-style byte-level BPE), which the
runtime pack contract requires. A source without them fails closed.

## Tensor layout & metadata

See `GGUF_MANIFEST.md` for the full tensor-name list, ggml shapes, per-tensor
dtype, and every metadata key. GGUF stores tensor dims reversed vs PyTorch
(`ne0` = innermost), e.g. a `[out, in]` Linear becomes `[in, out]`.

Quant policy:

- **q8_0 pack**: only the backbone rank-2 weight matrices (`blk.*`,
  `token_embd`, `output`) are Q8_0; the whole audio side (`audiotok.*`,
  `inlocal.*`, `speech_embd.*`, RVQ codebooks) stays F16, norms/biases/codebooks
  F32. This keeps RVQ encode + audio-prefix fidelity.
- **fp16 pack**: every eligible weight F16, norms/biases/codebooks/mel F32.

Mel filters (`audiotok.mel_filters`) are baked with `torchaudio.melscale_fbanks`
(htk scale, `norm=None`), stored freq-major so the ggml tensor is
`[n_mels=128, n_freqs=481]`; the front-end spec (power=1, `ln(clip 1e-7)`,
`center=True`) is fully described by the `mimo.mel.*` metadata keys. The mel
tables and window are baked into EVERY pack -- the executor reads them from
the pack and has no synthesis fallback.

## The three P2.0 blood-lesson corrections

These are forward-pass behaviours, not weights; the converter preserves the
enabling weights and records the hparams so the P2.2 runtime reproduces them:

1. **skip@L3** (`mimo.tok.encoder.skip_layer_id = 3`): the layer-3 (idx 2)
   encoder output is added to the final layer-32 (idx 31) output *before* the
   encoder's final LayerNorm. All 32 `audiotok.blk.*` layers are preserved.
2. **conv strides** (`mimo.tok.conv1.stride = 1`, `mimo.tok.conv2.stride = 2`):
   conv1 does not downsample (only 128->1280); only conv2 does the 2x time
   stride.
3. **8-codebook summation** (`mimo.audio.channels = 8`): the 8 `speech_embd.{i}`
   tables are looked up per RVQ channel and *summed* (not concatenated), with
   rows equal to each channel's `mimo.speech.zeroemb_idx` masked to zero.

## Tests

```bash
python3 -m unittest convert_mimo_asr_test
```

Covers the pure remap/type/metadata logic, the public envelope keys, the
build-provenance env semantics, and a full tiny synthetic safetensors -> GGUF
round-trip. Each blood-lesson correction has an explicit assertion (skip
layer id + preserved layers, conv strides, 8-table summation semantics).
