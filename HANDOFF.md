# ReDimNet2-B6 embedder -- stage-2 handoff

Branch `feat/redimnet2-b6-embedder`. Stage-1 assets (specs, golden `.npy`,
weights) live under `/Volumes/QuintinDocument/openasr-dev/tmp/redimnet2-spike/`.
This handoff records exactly what is done and the precise remaining plan.
**Delete this file before the PR is marked ready** (it is a working note).

## Done this session (each step verified)

- **Architecture defined**: `docs/design/redimnet2-b6-embedder.md` (5 decisions
  + full upstream forward reference + backbone plan + risks).
- **Converter**: `tooling/redimnet2/convert_redimnet2.py` (torch `.pt` -> GGUF
  `.oasr`, standard ggml `ne` order, drops `spec.*` + `num_batches_tracked`,
  per-tensor f32/f16/q8_0). Unit test `convert_redimnet2_test.py` -- **9 tests
  green**. Real run: 739 tensors kept / 82 dropped -> 50 MB f32 pack at
  `tmp/redimnet2-spike/redimnet2-b6-f32.oasr` (also usable as the backbone
  parity fixture).
- **Front end**: `crates/openasr-core/src/diarize/embed/redimnet/frontend.rs`
  (TFMelBanks port, pure Rust). Parity vs `frontend_dump/*.npy` -- **green**:
  kernels/mel-matrix ~1e-13 / exactly 0; preemph exactly 0; mel_linear mean
  ~7e-6; CMN mean ~7e-7 (gates: preemph max<1e-4, mel mean<1e-3, cmn mean<1e-4).
- **Structural constants**: `redimnet/config.rs` (per-stage dims, self-checked
  against checkpoint shapes by a unit test -- green).
- **Module wired**: `redimnet` added to `embed/mod.rs` (compiles; `#![allow(
  dead_code)]` until the backbone consumes it). `cargo fmt` + `cargo clippy
  --all-targets -D warnings` **clean**.

Not started: the backbone ggml graph, ASTP pool/BN/linear, the `SpeakerEmbedder`
impl, runtime pack resolution, calibration profile, catalog/registry entry,
audit form. WeSpeaker is untouched and remains the sole runtime embedder.

## Remaining plan (staged, golden-pinned)

Golden anchors in `tmp/redimnet2-spike/stage_dump_b6_jfk/` (jfk.wav, 176000
samples). Reference source in `tmp/redimnet2-spike/repo/redimnet2/`:
`redimnet2.py` (build/forward, read), `layers/blocks.py` (ConvBlock2d,
TimeContextBlock1d -- **read next**), `layers/resblocks.py`, `layers/attention.py`
(wav2vec2 attention), `layers/convnext.py`, `layers/poolings.py` (ASTP),
`layers/redim_structural.py` (to1d/to2d/weigth1d -- already read; `to1d` =
`permute(0,2,1,3).reshape(bs,c*f,t)`, `to2d` = inverse).

Build the backbone as `redimnet/backbone.rs` mirroring
`models/dolphin/encoder_graph.rs` (arena weight upload, `start_graph`,
`set_input`, `compute_output_f32`). Load weights with
`diarize::embed::weights::Weights::from_oasr` on the f32 pack (flat f32 in ggml
memory order -> upload verbatim into graph tensors with matching `ne` dims). Add
a `#[ignore]` parity test per step. Tolerance 1e-3 max abs (cumulative f32).

1. **Stem** -> `01_outputs_1d.npy (4608,1096)`. Ops: `conv_2d` (1->64, 3x3, pad
   'same'=1) + bias; LayerNorm channels-first over the 64-channel axis
   (`stem.1`, eps 1e-6); `to1d` (permute `(c,f,t)->(c*f,t)`, frequency-major
   flatten -- derive the ggml permute/cont carefully, risk #1); GroupNorm(64,
   4608) (`stem_gnorm`). Input = `00_spec_output` truncated to T multiple of 4.
2. **Stage 0** (sf1,st1,3 blocks,exp3,red64) -> `02_outputs_1d`. `weigth1d`
   agg(N=1) -> `to2d(f=72,c=64)` -> down-conv `stage0.2 (192,1,1,1)`... note
   `stage0.2.weight (192,1,1,1)` = grouped 1x1 (groups=gcd(64,192)=64) up to 192;
   3x `ConvBlock2d` (depthwise conv1 3x3 + conv1pw 1x1 + bn1 + ReLU, ditto conv2,
   residual+ReLU) at 192ch; 1x1+BN `stage0.6` (192->64); `to1d`;
   `TimeContextBlock1d` `stage0.8` (red_dim_conv 4608->72+BN, 4 dwconvs
   k=7/19/31/59 +BN+pwconv1, attention block hidden72, exp_dim_conv 72->4608);
   Upsample(st=1)=noop; GroupNorm `stage0.10`.
3. **Stages 1..5** -> `03..07_outputs_1d`. Same recipe at each running `(c,f)`
   from `config::STAGES`; strided down-conv `(sf,st)` with kernel `(sf,stt)`
   where `stt` is the cumulative time stride at that stage (see `redimnet2.py`
   build: `kernel_size=(sf,stt)`, `stride=(sf,stt)`), `groups=gcd(c, sf*c*exp)`.
   Watch the Upsample(stt) that restores T after time-strided stages (2 and 4).
4. **fin_wght1d** -> `08_outputs_1d`. `weigth1d(N=7)` softmax-weighted sum of the
   7 collected `(4608,1096)` maps.
5. **head + fin_to2d** -> `99_backbone_2d_output (224,9,1096)` and
   `a0_pre_pool_flattened (2016,1096)`. `fin_to2d` reshapes `(4608,T)` at final
   `(c=512? no: c*f=4608 with c=512,f=9 -> 512*9=4608)` to `(512,9,T)`; `head`
   Conv2d(512->224,1x1); flatten `(224*9=2016,T)`.
   NOTE: after stage5 running `c=512,f=9`; `to2d` uses those, so `fin_to2d`
   is `to2d(f=9,c=512)`. Verify the 4608 flatten is frequency-major.
6. **ASTP pool -> BN -> linear** -> `a1_post_pool (4032,)`, `a2_post_bn`,
   `a3_final_embedding (192,)`. ASTP with `global_context_att`: concat
   `[x, mean_T, std_T]` (2016->6048), `pool.linear1 (6048->128)` -> tanh ->
   `pool.linear2 (128->2016)` -> softmax over T -> attentive mean+std -> 4032;
   `bn (4032)`; `linear (4032->192)`. Then L2-normalize.
7. **End-to-end cosine gate**: jfk/zh/en_zh embedding vs `embeddings_b6/*.npy`
   cosine > 0.9999. B6 reference cosines (sanity): jfk-split 0.8288, zh vs
   en_zh (same speaker) 0.9725, cross-speaker ~ -0.02.

## Then (post-backbone, separate steps)

- `SpeakerEmbedder` impl on a `RedimNet2Embedder` (embedding_dim 192), a
  `REDIMNET_CALIBRATION` profile (192-d cosine space -- must be measured, not
  copied from WeSpeaker), and runtime pack resolution
  (`OPENASR_REDIMNET_PACK` + installed-dir hint) mirroring `pack.rs`.
- Catalog/registry entry + model card + `docs/model-audits/redimnet2-b6.md`
  (new-family release gate).
- Convert final shipping pack at the chosen quant; benchmark RTF/RAM (separate,
  measurement-isolated).

## Gotchas captured

- `to1d`/`to2d` are frequency-major (`permute(0,2,1,3)`); the 4608 flatten
  order matters -- pin against `01_outputs_1d` first.
- Down-conv is **grouped** (`groups=gcd`); ConvBlock2d conv1/conv2 are
  **depthwise** (weight `(C,1,3,3)`) followed by pointwise `conv?pw (C,C,1,1)`.
- BatchNorm is applied explicitly (not folded in the pack).
- `conv_exp` can be < 1 (stages 4/5): `block_channels = round(c*conv_exp)`.
- Frontend constants are recomputed in Rust; do not expect `spec.*` in the pack.
