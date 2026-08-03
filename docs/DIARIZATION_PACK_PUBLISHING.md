# Diarization and Voice ID capability packs

This document is the publishing and runtime contract for the auxiliary models
behind local file Voice ID. These packs are not ASR models and have no
transcription path of their own.

## Runtime topology

The signed ASR catalog exposes `speaker_source = "native" | "external"`, derived
from the architecture registry:

| Speaker source | Families | Recording-local speaker path | Identity path |
| --- | --- | --- | --- |
| `native` | `moss-transcribe-diarize` | Decoder `[Sxx]` turns | ReDimNet2-B6 + shared Voice ID evidence/matching |
| `external` | Every other built-in ASR family | FireRed Stream-VAD + selected segmenter + ReDimNet2-B6 + automatic AHC/spectral clustering + overlap reconstruction | Shared Voice ID evidence/matching |

Exactly one recording-local source runs for a request. Native turns do not run
the external segmenter/clusterer, but they do not bypass ReDimNet2-B6: speaker
labels answer "who spoke when"; acoustic embeddings are what reconcile scopes
and match an enrolled person. Both sources normalize into the same transcript
attribution contract, and an unknown or under-evidenced voice remains a
session-relative `SPEAKER_NN` label.

This document qualifies the universal path for local file transcription only.
Realtime and remote-compute diarization have separate output/privacy contracts;
do not infer cross-recording identity guarantees for those surfaces from this
file-pipeline design.

## Pack matrix

| Pack | Role | Quantization | License/distribution state |
| --- | --- | --- | --- |
| `redimnet2-b6-cn` | Sole speaker embedder and identity space | fp16 | MIT, published |
| `pyannote-segmentation-3.0` | Default external local-activity segmenter | f32 | MIT, published |
| `diarizen-large-s80-v2` | Optional external local-activity segmenter | fp16 | CC BY-NC 4.0, source-only staged, not published or pullable |

FireRed Stream-VAD is a vendored Apache-2.0 runtime asset rather than a separate
user-installed capability pack.

`auto` uses segmentation-3.0 in the default installation. A future explicitly
consented DiariZen installation may take precedence; removing or disabling it
returns to segmentation-3.0. Request preflight freezes the chosen segmenter and
the exact ReDimNet pack content for the whole job. A provider that is present but
broken fails closed instead of silently changing algorithms mid-request.

DiariZen's `release_public = false` publishing source deliberately produces no
registry card, signed catalog entry, URL, size, or sha256. Do not manufacture
those fields. Publishing requires a separately approved native quality/resource
audit and product consent flow for its non-commercial license.

## Qualification evidence

The 2026-08-02 locked comparison used six 10-minute AISHELL-4/AliMeeting excerpts,
duration-weighted DER, a 0.25 s collar, and overlap scoring:

| Path | DER | Scope |
| --- | ---: | --- |
| OpenASR native + DiariZen Large-s80-md-v2 fp16 + ReDimNet2-B6 fp16 | **7.9491%** | Production runtime path on the locked fixtures; source-only qualification, not a release/distribution claim |
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

### DiariZen Large-s80-md-v2 qualification only

The converter is available for local qualification, but the output is not a
published artifact:

```bash
python3 tooling/diarizen/convert_diarizen.py \
    --checkpoint /path/to/pytorch_model.bin \
    --config /path/to/config.toml \
    --out /path/to/diarizen-large-s80-v2-fp16.oasr \
    --model-id diarizen-large-s80-v2 \
    --quant fp16
```

Do not run public regeneration for this source while `release_public = false`.
The local override is for controlled qualification, not a distribution promise:

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
public projection to Cloudflare is a separate release action. A source-only
DiariZen row must remain absent from both generated catalogs.

## Operator pull

The currently published capability packs are pullable explicitly:

```bash
openasr pull redimnet2-b6-cn
openasr pull pyannote-segmentation-3.0
```

`openasr pull diarizen-large-s80-v2` is intentionally unavailable until a separate
publication approval creates a real signed entry. Installed packs are resolved
through the content-addressed model store or the development overrides above;
runtime selection never authorizes a download.
