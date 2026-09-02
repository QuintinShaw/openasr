# Diarization and Voice ID capability packs

This document is the publishing and runtime contract for the auxiliary models
behind local file Voice ID. These packs are not ASR models and have no
transcription path of their own.

## Runtime topology

The signed ASR catalog exposes `speaker_source = "native" | "external"`, derived
from the architecture registry:

| Speaker source | Families | Recording-local speaker path | Identity path |
| --- | --- | --- | --- |
| `native` | `moss-transcribe-diarize` | Decoder `[Sxx]` turns | Selected speaker embedder + shared Voice ID evidence/matching |
| `external` | Every other built-in ASR family | FireRed Stream-VAD + selected segmenter + selected speaker embedder + automatic AHC/spectral clustering + overlap reconstruction | Shared Voice ID evidence/matching |

Exactly one recording-local source runs for a request. Native turns do not run
the external segmenter/clusterer, but they do not bypass the speaker embedder:
speaker labels answer "who spoke when"; acoustic embeddings are what reconcile
scopes and match an enrolled person. Both sources normalize into the same
transcript attribution contract, and an unknown or under-evidenced voice remains
a session-relative `SPEAKER_NN` label.

The default embedder is ReDimNet2-B6. Default diarization capability still
probes only that pack (`embedder_pack_installed()`). An explicit
`voice_id_embedder=wespeaker` preference or `OPENASR_WESPEAKER_PACK` loads the
optional WeSpeaker ResNet family instead. There is no Auto "use whatever is
installed": a selected WeSpeaker pack that is missing or broken fails closed
rather than falling back to ReDimNet. ReDimNet and WeSpeaker occupy different identity
spaces (192-d vs 256-d); enrollments and resident runtimes do not transfer.

This document qualifies the universal path for local file transcription only.
Realtime and remote-compute diarization have separate output/privacy contracts;
do not infer cross-recording identity guarantees for those surfaces from this
file-pipeline design.

## Pack matrix

| Pack | Role | Quantization | License/distribution state |
| --- | --- | --- | --- |
| `redimnet2-b6-cn` | Default speaker embedder and identity space; required for the default capability probe | fp16 | MIT, published |
| `wespeaker-voxceleb-resnet{34,152,221,293}-lm` | Optional speaker embedder and identity space (explicit `voice_id_embedder=wespeaker`) | fp16 | CC BY 4.0; not the default capability pack |
| `pyannote-segmentation-3.0` | Default external local-activity segmenter | f32 | MIT, published |
| `diarizen-large-s80-v2` | Optional external local-activity segmenter | fp16 | CC BY-NC 4.0, published; explicit non-commercial acceptance required |

FireRed Stream-VAD is a vendored Apache-2.0 runtime asset rather than a separate
user-installed capability pack.

`auto` uses segmentation-3.0 in the default installation. An explicitly
consented DiariZen installation may take precedence; removing or disabling it
returns to segmentation-3.0. Request preflight freezes the chosen segmenter and
the exact selected embedder pack content for the whole job. A provider that is
present but broken fails closed instead of silently changing algorithms
mid-request.

## Embedder families

| | ReDimNet2-B6 | WeSpeaker ResNet |
| --- | --- | --- |
| Selection | Default | Explicit `voice_id_embedder=wespeaker` or `OPENASR_WESPEAKER_PACK` |
| Default capability probe | Required | Not a substitute; missing ReDimNet still fails the default probe |
| Dimension | 192 | 256 |
| Frontend | TFMelBanks | Hamming Kaldi fbank + utterance CMN |
| Training claim | VoxBlink2 + VoxCeleb2 + CN-Celeb2 (`en`, `zh`) | VoxCeleb English LM (`en` only) |
| License | MIT | CC BY 4.0 |
| VBx community-1 PLDA | Not eligible | Gated on WeSpeaker family **and** 256-d |
| Calibration | ReDimNet profile | `wespeaker-cal-v1` |
| Sizes | B6 only | ResNet 34 / 152 / 221 / 293, one parameterized ggml graph |

See [WeSpeaker ResNet embedder](design/wespeaker-resnet-embedder.md) and
[ReDimNet2-B6 embedder](design/redimnet2-b6-embedder.md).

DiariZen is a public but non-commercial capability pack. Catalog discovery does
not grant permission to use it: pull surfaces must show the checkpoint license
and require explicit acceptance, and runtime activation is a separate user
choice. Enabling Voice ID alone continues to prepare segmentation-3.0 and never
implicitly downloads or activates DiariZen.

## Qualification evidence

The 2026-08-02 locked comparison used six 10-minute AISHELL-4/AliMeeting excerpts,
duration-weighted DER, a 0.25 s collar, and overlap scoring:

| Path | DER | Scope |
| --- | ---: | --- |
| OpenASR native + DiariZen Large-s80-md-v2 fp16 + ReDimNet2-B6 fp16 | **7.9491%** | Production runtime path on the locked fixtures; fixed-corpus qualification, not a cross-domain guarantee |
| FireRed + DiariZen Large-s80-md-v2 fp16 + ReDimNet2-B6 research adapter | 8.1232% | Qualified pack reconstructed in the locked Python adapter |
| FireRed + DiariZen Large-s80-md-v2 F32 + ReDimNet2-B6 research adapter | 8.1274% | Upstream-checkpoint precision reference |
| FireRed + DiariZen Base-s80 + ReDimNet2-B6 research adapter | 9.0481% | Historical Base-s80 F32 reference configuration; not a current product model or native release claim |
| FireRed + segmentation-3.0 + ReDimNet2-B6 research adapter | 12.4466% | Research reference pipeline; not a native release claim |
| MOSS in-decoder diarization | 18.6787% | Native ASR speaker source baseline |

The native Large-s80-md-v2 result recorded 3.8806% miss, 1.1348% false alarm,
and 2.9336% speaker error. Its collar-zero DER was 12.1879%. The fp16 adapter
result differs from the F32 reference by -0.0042 percentage point, establishing
no material precision loss; it does not establish that fp16 improves quality.

These results qualify the architecture on that fixed Mandarin meeting slice;
they are not a cross-language, cross-domain, or cross-recording Voice ID
guarantee. The native aggregate combines A1-M2 from the original full-manifest
process with M3 from an independently completed supplement after the external
test harness output pipe disappeared before M3 started. Both sources used the
same core revision, segmenter/embedder content IDs and Metal backend, and the
composition preserves per-source provenance. It is valid as a stateless
per-recording micro aggregate, but must not be described as one uninterrupted
six-file process. The AISHELL excerpts also remain a speaker-count weakness
(4/6, 4/6, and 3/6 hypotheses), so DER alone is not an enrollment or
unknown-rejection acceptance gate.

## Build and publish overview

### ReDimNet2-B6

Use the external converter and the normal capability-pack publish lane:

```bash
# Convert upstream checkpoint -> .oasr (see tooling/redimnet2/convert_redimnet2.py)
python3 tooling/redimnet2/convert_redimnet2.py ...

python3 tooling/publish-model/scripts/materialize_result_sidecars.py redimnet2-b6-cn --quant fp16
tooling/publish-model/scripts/regenerate_all.sh --public redimnet2-b6-cn
```

Runtime override for a local development pack:

```bash
export OPENASR_REDIMNET_PACK=/path/to/redimnet2-b6-cn-fp16.oasr
```

### WeSpeaker ResNet (optional)

One converter and one ggml builder cover ResNet 34/152/221/293. Depth and block
kind are pack metadata, not copied graphs. Catalog `{quant}` is `fp16`; the
converter accepts `fp16` as an alias of `f16`.

```bash
python3 tooling/wespeaker/convert_wespeaker.py \
    --in /path/to/checkpoint \
    --out /path/to/wespeaker-voxceleb-resnet34-lm-fp16.oasr \
    --quant fp16 \
    --model-id wespeaker-voxceleb-resnet34-lm \
    --depth 34
```

Official Hugging Face checkpoint filenames are not uniform:

| Size | Upstream repo | Checkpoint file |
| --- | --- | --- |
| 34 | `Wespeaker/wespeaker-voxceleb-resnet34-LM` | `avg_model` |
| 152 | `Wespeaker/wespeaker-voxceleb-resnet152-LM` | `voxceleb_resnet152_LM.pt` |
| 221 | `Wespeaker/wespeaker-voxceleb-resnet221-LM` | `voxceleb_resnet221_LM.pt` |
| 293 | `Wespeaker/wespeaker-voxceleb-resnet293-LM` | `avg_model.pt` |

Runtime override for a local development pack:

```bash
export OPENASR_WESPEAKER_PACK=/path/to/wespeaker-voxceleb-resnet34-lm-fp16.oasr
```

Host-local PyTorch cosine goldens (`#[ignore]`, CPU then Metal) live under
`OPENASR_WESPEAKER_SPIKE_ROOT` and are not committed.

### pyannote segmentation-3.0

```bash
openasr model-pack import pyannote \
    tmp/pyannote/pyannote_seg.safetensors \
    tmp/publish/pyannote-segmentation-3.0/packs/pyannote-segmentation-3.0-f32.oasr \
    --package-id pyannote-segmentation-3.0

python3 tooling/publish-model/scripts/materialize_result_sidecars.py pyannote-segmentation-3.0 --quant f32
tooling/publish-model/scripts/regenerate_all.sh --public pyannote-segmentation-3.0
```

Runtime override for a local development pack:

```bash
export OPENASR_PYANNOTE_PACK=/path/to/pyannote-segmentation-3.0-f32.oasr
```

### DiariZen Large-s80-md-v2

The converter reproduces the published fp16 artifact from the pinned upstream
checkpoint:

```bash
python3 tooling/diarizen/convert_diarizen.py \
    --checkpoint /path/to/pytorch_model.bin \
    --config /path/to/config.toml \
    --out /path/to/diarizen-large-s80-v2-fp16.oasr \
    --model-id diarizen-large-s80-v2 \
    --quant fp16
```

The local override remains available for controlled qualification:

```bash
export OPENASR_DIARIZEN_PACK=/path/to/diarizen-large-s80-v2-fp16.oasr
```

## Catalog signing

For an already approved public pack, bump `model-registry/catalog.epoch`, then
re-sign with the production seed (`OPENASR_CATALOG_SIGNING_KEY_SEED_HEX` in env):

```bash
tooling/publish-model/scripts/publish_catalog.sh
```

That refreshes the committed full and public catalog signatures. Deploying the
public projection to Cloudflare is a separate release action. DiariZen may enter
that projection only with `license_class = "noncommercial"` and verified
CC BY-NC 4.0 provenance embedded in the pack.

## Operator pull

The currently published capability packs are pullable explicitly:

```bash
openasr pull redimnet2-b6-cn
openasr pull pyannote-segmentation-3.0
openasr pull diarizen-large-s80-v2 --accept-license
```

Omitting `--accept-license` for DiariZen fails closed. Installed packs are
resolved through the content-addressed model store or the development overrides
above; runtime selection never authorizes a download.
