# WeSpeaker ResNet speaker embedder (ggml)

Stage-1 architecture for a **parallel** speaker-embedder family beside
ReDimNet2-B6. Default Voice ID / diarization remains ReDimNet2-B6. WeSpeaker
loads only on an explicit `voice_id_embedder=wespeaker` preference or
`OPENASR_WESPEAKER_PACK`. There is no Auto "use whatever is installed".

## Decisions

- **Delivery** = `.oasr` GGUF in ggml `ne` order (torch shape reversed), same
  convention as ReDimNet2. The retired pure-Rust WeSpeaker path is not revived.
- **`general.architecture`** = `wespeaker-resnet` (family, not a size such as
  `wespeaker-resnet34`). Depth and block kind live in metadata
  (`wespeaker.depth`, `wespeaker.block_kind`, `wespeaker.num_blocks`).
- **Frontend** = shared `KaldiFbankFrontend` with `KaldiWindowKind::Hamming`
  (not Povey), 80 mel, 25/10 ms, 16 kHz, preemph 0.97, `input_scale=32768`,
  `log_energy_floor=f32::EPSILON`, fmin 20 / fmax 8000, snip_edges, then
  utterance CMN (mean only).
- **TSTP** = official `sqrt(torch.var(x, dim=-1, unbiased=True) + 1e-7)`.
  Post-stride time length `< 2` fails closed.
- **Inference** does not L2-normalize; callers use `SpeakerEmbedding::l2_normalized`.
- **VBx** community-1 PLDA is gated on `family == WeSpeakerResNet && dim == 256`,
  not dimension alone. ReDimNet2 stays skipped.
- **Calibration** is the restored WeSpeaker 256-d profile (`wespeaker-cal-v1`),
  not a copy of ReDimNet thresholds.

ResNet34 is the first shipped size. 152/221/293 share the same parameterized
ggml builder and config table; they do not get copied graphs.

Converter: `tooling/wespeaker/convert_wespeaker.py`. Reference dump:
`tooling/wespeaker/dump_reference.py`.
