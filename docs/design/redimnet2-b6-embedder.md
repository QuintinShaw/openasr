# ReDimNet2-B6 speaker embedder (ggml)

Design and bring-up plan for adding the **ReDimNet2-B6** speaker embedder
(PalabraAI/redimnet2, MIT) as a ggml-graph model under `diarize/embed/`.
B6 is a 12.46 M-param UNet-style "dimension reshaping" net that outputs a
192-dim speaker embedding, with a Chinese-enhanced training mix (vb2+vox2+cnc2).
It is a candidate replacement for the legacy pure-Rust WeSpeaker ResNet34
(256-dim). Removing WeSpeaker is out of scope for this work and gated behind
full B6 validation.

Reference source is the upstream `redimnet2/` Python package (checkpoint
`b6-vb2+vox2+cnc2_v0-lm.pt`, SHA256 `287365f6...`, byte-verified against the
GitHub release digest). Spike anchors (specs, golden `.npy` dumps) live under
`tmp/redimnet2-spike/` and are referenced by the `#[ignore]` parity tests.

## Decisions

### 1. Delivery = GGUF `.oasr` pack + ggml graph (not pure-Rust forward)

The existing WeSpeaker embedder runs a hand-written pure-Rust forward pass -- a
historical exception predating the ggml-only invariant. ReDimNet2 does **not**
repeat that: it executes through a ggml `GgmlCpuGraphBuilder` graph, like every
ASR family, fed from a `.oasr` GGUF pack.

- **Converter**: `tooling/redimnet2/convert_redimnet2.py` (torch `.pt` -> GGUF),
  mirroring the `tooling/mimo-asr/convert_mimo_asr.py` convention (python `gguf`
  `GGUFWriter`, per-tensor type selection, hparams as GGUF metadata). Unit test:
  `convert_redimnet2_test.py` (remap, type selection, synthetic round-trip).
- **Runtime**: a new `diarize/embed/redimnet` module builds the graph on the
  shared `GgmlCpuGraphRunner` (`start_graph` / `set_input` / `compute_output_f32`),
  exactly like `models/dolphin/encoder_graph.rs`. No new ggml infrastructure is
  invented for the embedder.

### 2. Tensor convention: standard ggml `ne` order (reversed), flat f32 reuse

ReDimNet2 is a ggml-graph model, so its pack uses the **standard** ggml tensor
convention: `gguf.GGUFWriter` stores dims in `ne` order (torch shape reversed)
and the payload in ggml memory order (ne0 innermost). The Rust side reads each
tensor's flat f32 via the existing `diarize::embed::weights::Weights::from_oasr`
(which already dequantizes to a logical f32 buffer by name) and uploads it
verbatim into a graph tensor created with the same `ne` dims -- both sides agree
on ggml memory order, so **no transpose is needed** (same as dolphin's conv
weights: torch `(C_out,C_in,KH,KW)` -> ggml kernel `[KW,KH,C_in,C_out]`).

This differs from the WeSpeaker/`import ln` diarize pack, which keeps *logical*
(non-reversed) dims because its pure-Rust reader indexes logically. That path is
untouched; ReDimNet2's pack is a distinct, ggml-native artifact.

### 3. Front end: separate `RedimNetFrontend`, not the WeSpeaker `Fbank`

The B6 front end (`TFMelBanks`) is fundamentally different from the WeSpeaker
Kaldi fbank and shares nothing:

| aspect            | WeSpeaker `Fbank`        | ReDimNet2 `TFMelBanks`               |
|-------------------|--------------------------|--------------------------------------|
| mel bins          | 80                       | **72**                               |
| mel formula       | Kaldi `1127*ln(1+hz/700)`| **Slaney `2595*log10(1+hz/700)`**    |
| f_max             | 8000 Hz                  | **7600 Hz**                          |
| signal scaling    | int16 full-scale (*32768)| **per-utterance zero-mean/unit-std** |
| window            | Povey                    | **Hamming**                          |
| STFT              | rustfft                  | **explicit cos/sin conv1d, pad=hop/2**|
| CMN               | (impl-specific)          | **subtract time-mean only**          |

It is implemented as a new pure-Rust module (`redimnet::frontend`), matching the
existing `Fbank` pattern (front-end feature extraction is CPU preprocessing; the
ggml-only invariant governs the neural backbone). The Hamming window, cos/sin
DFT kernels, and Slaney mel matrix are deterministic constants recomputed in
Rust -- not baked into the pack -- so `spec.*` checkpoint buffers are dropped by
the converter. Numeric parity is pinned by `frontend::tests::frontend_parity`
against `frontend_dump/*.npy` (kernels/matrix bit-tight; staged tensors under
the cross-impl fp32 convention: wide max / tight mean, per the firered-aed
precedent).

### 4. Backbone: per-stage hard-coded dims, staged golden pinning

B6 is **not** a uniform scale-up of B3; every per-stage constant (block count,
`conv_exp`, `att_block_red`, running `c`/`f`, TCM hidden dim) is listed
explicitly in `redimnet::config::STAGES` and cross-checked against checkpoint
tensor shapes by a unit test. The 2D<->1D reshape layout (`to1d`/`to2d`, risk
item #1) is documented per op with its ggml dim derivation when implemented.

Bring-up is staged, each step golden-pinned against `stage_dump_b6_jfk/` before
the next: stem -> stage0..5 -> `fin_wght1d` -> head -> ASTP pool -> BN ->
linear. See "Backbone plan" below.

### 5. `SpeakerEmbedder` wiring deferred until the backbone is golden-clean

The `SpeakerEmbedder` trait impl, runtime pack resolution, and a ReDimNet
calibration profile (192-dim cosine space, distinct from `WESPEAKER_CALIBRATION`)
land **after** the backbone reproduces the golden embeddings. Wiring a
half-built forward into the fail-closed diarize dispatch would risk fabricating
embeddings. WeSpeaker remains the sole runtime embedder until then. The
`embedding_dim` (256 -> 192) and `pack_fingerprint` / embedding-space-version go
through the existing `SpeakerEmbedderIdentity` mechanism at that point.

## Architecture reference (upstream forward)

`ReDimNet2Wrap.forward`: `spec -> backbone -> reshape (C*F,T) -> ASTP pool ->
BatchNorm1d -> Linear(4032->192)`.

`ReDimNet2.forward` (backbone): the input `(1,1,72,T)` is time-truncated to a
multiple of `TIME_STRIDE=4`, then:

```
x0 = stem(inp)                    # Conv2d(1,64,3,'same') -> LayerNorm(64, CF) -> to1d -> stem_gnorm
outputs_1d = [x0]                 # each (C*F=4608, T)
for s in 0..6:
    outputs_1d.append(stage_s(outputs_1d))   # agg1d(weigth1d) -> to2d -> down-conv -> N x ConvBlock2d
                                              #   -> (1x1 conv+BN if conv_exp!=1) -> to1d
                                              #   -> TimeContextBlock1d(TCM) -> Upsample(st) -> gnorm
x = fin_wght1d(outputs_1d)        # softmax-weighted sum of 7 feature maps -> (4608, T)
x = fin_to2d(x)                   # (4608,T) -> (64,72,T)... -> reshaped to (512,9,T) at final c,f
x = head(x)                       # Conv2d(512,224,1) -> (224,9,T)
```

`weigth1d(N,C=4608)`: `w = softmax(param[1,N,4608,1], dim=1)`, output =
`sum_i w[:,i] * xs[i]`. `stage_s` begins by aggregating **all** prior 1D outputs
(count grows each stage), reshapes to 2D at the stage's running `(c,f)`, applies
a strided down-conv `(sf,st)` with `groups=gcd(c, sf*c*conv_exp)`, `num_blocks`
basic ResNet `ConvBlock2d`s at `block_channels`, an optional 1x1 conv+BN back to
`c` when `conv_exp!=1`, `to1d`, a `TimeContextBlock1d` (4 depthwise temporal
convs at kernels 7/19/31/59 + a wav2vec2-style self-attention block at
`tcm_hidden`), nearest `Upsample(st)`, and `GroupNorm(64, 4608)`.

`ConvBlock2d` (basic_resnet) tensor layout, per checkpoint:
`conv1 (C,1,3,3) depthwise -> conv1pw (C,C,1,1) -> bn1 -> ReLU ->
conv2 (C,1,3,3) depthwise -> conv2pw (C,C,1,1) -> bn2`, residual add + ReLU.

`TimeContextBlock1d.tcm`: `red_dim_conv (Conv1d CF->hidden + BN)` ->
`{dwconvs.k (depthwise Conv1d, kernel in [7,19,31,59]) + BN + pwconv1 (1x1)}`
(4 parallel/stacked branches) -> attention block (`q/k/v/out_proj` + 2
LayerNorms + FFN `intermediate_dense`/`output_dense`) -> `exp_dim_conv
(Conv1d hidden->CF)`.

`pool` (ASTP, `global_context_att=True`): concat `[x, mean_T(x), std_T(x)]`
along channels (2016 -> 6048) -> `linear1 (6048->128,1)` -> tanh ->
`linear2 (128->2016,1)` -> softmax over T -> attentive mean+std -> 4032. Then
`bn (4032)` -> `linear (4032->192)`.

## Backbone plan (remaining work, each step golden-pinned)

Golden anchors: `tmp/redimnet2-spike/stage_dump_b6_jfk/` (jfk.wav, 176000
samples). `00_spec_output (72,1099)`, `01..08_outputs_1d (4608,1096)` (01=stem,
02..07=stage0..5, 08=fin_wght1d), `99_backbone_2d_output (224,9,1096)`,
`a0_pre_pool_flattened (2016,1096)`, `a1_post_pool (4032,)`, `a2_post_bn (4032,)`,
`a3_final_embedding (192,)`. End-to-end cosine target vs `embeddings_b6/*.npy`
> 0.9999 for jfk/zh/en_zh.

1. **Stem** -> pin `01_outputs_1d`. Exercises conv_2d (same pad), LayerNorm
   channels-first over C, `to1d` (permute `(bs,c,f,t)->(bs,c*f,t)`), GroupNorm.
2. **Stage 0** (sf1,st1) -> pin `02_outputs_1d`. `weigth1d` agg, `to2d`,
   down-conv (1x1 stride here since sf=st=1 but conv exists), 3 ConvBlock2d,
   1x1+BN (conv_exp=3), to1d, TCM, Upsample(1), gnorm.
3. **Stages 1..5** -> pin `03..07_outputs_1d`. Same structure; watch the
   `to2d`/`to1d` layout at each running `(c,f)` and the strided down-conv.
4. **fin_wght1d** -> pin `08_outputs_1d`. 7-way softmax-weighted sum.
5. **head + fin_to2d** -> pin `99_backbone_2d_output` and `a0_pre_pool_flattened`.
6. **ASTP pool -> BN -> linear** -> pin `a1/a2/a3`, then the cosine gate.

Each stage lands with a `#[ignore]` parity test (loads the f32 `.oasr` pack via
`Weights::from_oasr`, or the reference safetensors export like dolphin, runs the
partial graph, compares against the stage's `.npy`). Tolerance: `1e-3` max abs
per tap (cumulative f32 bound), tightened where a tap is bit-stable.

## Risks

- **2D<->1D layout (#1)**: `to1d` is `permute(0,2,1,3).reshape(bs,c*f,t)` --
  frequency-major flatten. In ggml (ne0 innermost) the equivalent permute/cont
  must be derived carefully per op; document the ne mapping in each reshape.
- **Grouped/depthwise convs**: the down-conv uses `groups=gcd(c, sf*c*conv_exp)`;
  ConvBlock2d and TCM use depthwise convs. Map to `conv_2d_dw_direct` /
  `depthwise_conv_2d` / grouped `conv_1d`; verify group semantics against golden.
- **conv_exp < 1** (stages 4/5): `block_channels < c_out_2d`; the `round()` is
  captured in `config` and unit-tested.
