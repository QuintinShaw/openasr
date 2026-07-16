# MiMo-V2.5-ASR -> `.oasr` converter (stage 2, P2.1)

Converts the two upstream MiMo weight repos into a single OpenASR `.oasr`
(GGUF-backed) pack:

- **main model** `XiaomiMiMo/MiMo-V2.5-ASR` (36L Qwen2 backbone + 6L input-local
  transformer + 8 speech-embedding tables), and
- **audio tokenizer** `XiaomiMiMo/MiMo-Audio-Tokenizer` (32L rope encoder +
  conv stem + first 8 RVQ codebooks).

This is the conversion pipeline only. Runtime family registration, the
ggml executor, and the decode-policy descriptor land in P2.2. Nothing here
touches the catalog or `model-registry/`.

## Usage

```bash
python3 convert_mimo_asr.py \
  --main-dir   /path/to/MiMo-V2.5-ASR \
  --tokenizer  /path/to/MiMo-Audio-Tokenizer/model.safetensors \
  --out-dir    /path/to/out \
  --package-id mimo-v2.5-asr          # -> mimo-v2.5-asr-{q8_0,fp16}.oasr
```

`--quant q8_0` / `--quant fp16` selects a subset (default: both).

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
`center=True`) is fully described by the `mimo.mel.*` metadata keys.

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

Covers the pure remap/type/metadata logic and a full tiny synthetic
safetensors -> GGUF round-trip. Each blood-lesson correction has an explicit
assertion (skip layer id + preserved layers, conv strides, 8-table summation
semantics).
