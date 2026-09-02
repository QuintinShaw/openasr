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

All four VoxCeleb LM sizes share the same parameterized ggml builder and
config table; they do not get copied graphs.

| Depth | Block | `num_blocks` | Linear in-dim (TSTP) |
| ---: | --- | --- | ---: |
| 34 | BasicBlock | `[3, 4, 6, 3]` | 5120 |
| 152 | Bottleneck | `[3, 8, 36, 3]` | 20480 |
| 221 | Bottleneck | `[6, 16, 48, 3]` | 20480 |
| 293 | Bottleneck | `[10, 20, 64, 3]` | 20480 |

`m_channels=32`, `feat_dim=80`, `embed_dim=256`, TSTP, `two_emb_layer=false`.
Catalog ids: `wespeaker-voxceleb-resnet{34,152,221,293}-lm`. Languages are
honest `["en"]` (VoxCeleb English); do not copy ReDimNet's `en`/`zh` claim.
Persistent-graph metadata is sized from the live topology
(`metadata_context_bytes_exact`); the shared 1 MiB 4096-slot bump is not
enough for ResNet221/293.

Converter: `tooling/wespeaker/convert_wespeaker.py` (`fp16` is an alias of
`f16`). Reference dump: `tooling/wespeaker/dump_reference.py` (depth 34 writes
`golden/`; 152/221/293 write `golden-{depth}/`). Host-local cosine tests skip
missing depths under `OPENASR_WESPEAKER_SPIKE_ROOT`.
