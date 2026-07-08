# Diarization pack publishing (WeSpeaker + pyannote)

How the speaker-diarization models become first-class `openasr pull` artifacts.
The diarization models are **auxiliary** packs — WeSpeaker emits speaker
embeddings, pyannote-seg emits per-frame speaker-activity probabilities
— so they are **not** ASR transcription models, are **not** registered in
`BUILTIN_ARCHITECTURE_DESCRIPTORS`, and their pure-Rust forward passes live in
`crate::diarize::{embed,segment}`.

Everything except the actual HF upload + catalog re-sign is committed in-repo:

- `models-core.toml` / `models-publish.toml` carry the
  `wespeaker-voxceleb-resnet34-lm` and `pyannote-segmentation-3.0` entries.
  The published catalog keeps the WeSpeaker `f32` variant.
  In the generated catalog they are `kind: "capability-pack"` and carry
  `capability.feature = "speaker-diarization"` with roles
  `speaker-embedder` / `speaker-segmenter`. They stay `public:true` because
  public means downloadable/importable; ASR market visibility is derived by
  `CatalogModel::is_market_listed()` (`public && kind == asr-model`).
- `publish_model_targets.py` clears both ids for the release lane and validates
  the quant set against the catalog-declared list per model.
- `render_card.py` renders the diarize HF README from
  `template/DIARIZE_CARD.md.tmpl` + `cards/<id>.toml` prose (no ASR pipeline
  tags / bench table).
- `openasr pull` validates these packs by **constructing the diarize model from
  the pack** (dispatched through `crate::models::aux_pack_registry`, which
  `validate_native_runtime_model_pack_contract` calls before ASR runtime
  adapter selection) instead of ASR runtime adapter selection — fail-closed
  either way.

The published packs are raw **f32** GGUF-v0 so embeddings and the 7-class
powerset logits stay bit-exact.

## Step 1 — obtain sources

```bash
# WeSpeaker ResNet34 (pyannote/wespeaker-voxceleb-resnet34-LM, CC-BY-4.0):
PINNED_WESPEAKER_REV=837717ddb9ff5507820346191109dc79c958d614
huggingface-cli download pyannote/wespeaker-voxceleb-resnet34-LM \
    pytorch_model.bin --revision "$PINNED_WESPEAKER_REV" \
    --local-dir tmp/wespeaker
python3 tooling/publish-model/scripts/wespeaker_reference.py \
    --checkpoint tmp/wespeaker/pytorch_model.bin \
    --safetensors-out tmp/wespeaker/wespeaker_voxceleb_resnet34_lm.ref.safetensors \
    --golden-out tmp/wespeaker/wespeaker_resnet34_golden.bin \
    --stages-out tmp/wespeaker/wespeaker_resnet34_stages.npz \
    --wav fixtures/jfk.wav

# The reference script writes the safetensors provenance metadata enforced by
# import wespeaker and writes source/revision/checkpoint SHA-256 into the
# WSR1 parity golden header.

# pyannote segmentation-3.0 (MIT) via the un-gated onnx-community ONNX mirror,
# pinned at the revision the loader tensor names + parity goldens were
# validated against (also the default in pyannote_extract.py):
PINNED_REV=733a93b6473d019a773298e08cefa686894b1854
huggingface-cli download onnx-community/pyannote-segmentation-3.0 \
    onnx/model.onnx --revision "$PINNED_REV" --local-dir tmp/models/pyannote
python3 tooling/publish-model/scripts/pyannote_extract.py \
    --onnx tmp/models/pyannote/onnx/model.onnx \
    --out  tmp/models/pyannote/pyannote_seg.safetensors --revision "$PINNED_REV"
```

> The pinned revision matters: earlier mirror revisions exported three weights
> under ORT-folded names (`ortshared_*`); the pinned revision uses the semantic
> names (`sincnet.wav_norm1d.{weight,bias}`, `classifier.bias`) that
> `diarize/segment/pyannet.rs` loads. Values are identical apart from float
> noise (~1e-6) in the materialized sinc filter.

## Step 2 — build the `.oasr` packs into the publish workdir

```bash
openasr model-pack import wespeaker \
    tmp/wespeaker/wespeaker_voxceleb_resnet34_lm.ref.safetensors \
    tmp/publish/wespeaker-voxceleb-resnet34-lm/packs/wespeaker-voxceleb-resnet34-lm-f32.oasr \
    --package-id wespeaker-voxceleb-resnet34-lm \
    --source-revision 837717ddb9ff5507820346191109dc79c958d614 \
    --license-name CC-BY-4.0
openasr model-pack import pyannote \
    tmp/models/pyannote/pyannote_seg.safetensors \
    tmp/publish/pyannote-segmentation-3.0/packs/pyannote-segmentation-3.0-f32.oasr \
    --package-id pyannote-segmentation-3.0
```

Then write `tmp/publish/<id>/metrics.json` with a single `f32` quant entry
(`size_bytes` = pack size; the ASR bench fields stay `null` — they do not apply
to support packs) and materialize the result sidecars:

```bash
python3 tooling/publish-model/scripts/materialize_result_sidecars.py wespeaker-voxceleb-resnet34-lm --quant f32
python3 tooling/publish-model/scripts/materialize_result_sidecars.py pyannote-segmentation-3.0 --quant f32
```

## Step 3 — validate (no seed needed)

```bash
OPENASR_WESPEAKER_PACK=tmp/wespeaker/wespeaker_voxceleb_resnet34_lm.ref.safetensors \
OPENASR_PYANNOTE_PACK=tmp/models/pyannote/pyannote_seg.safetensors \
  cargo nextest run -p openasr-core -E 'test(oasr_roundtrip_matches_safetensors)' --run-ignored all
# End-to-end with the built packs (capability report, --diarize, enrollment):
OPENASR_WESPEAKER_PACK=tmp/publish/wespeaker-voxceleb-resnet34-lm/packs/wespeaker-voxceleb-resnet34-lm-f32.oasr \
OPENASR_PYANNOTE_PACK=tmp/publish/pyannote-segmentation-3.0/packs/pyannote-segmentation-3.0-f32.oasr \
  openasr transcribe fixtures/jfk.wav --model <asr-model> --backend native --diarize --format srt
```

## Step 4 — upload + catalog (SEED-GATED — release owner only)

Adding a model to `catalog.json` changes the signed payload, so it must be
regenerated **and re-signed** in one pass or the
`bundled_catalog_signature_verifies_committed_catalog_and_epoch` gate fails.
Changing the public set requires updating the pinned public-id gate in
`registry_tests_schema_validation.rs`; capability packs remain in the public
projection while staying out of ASR model-market listings.

```bash
# HF_TOKEN in env; uploads the staged packs + README and writes
# tmp/publish/<id>/hf_{repo,revision}.txt:
python3 tooling/publish-model/scripts/publish_model_targets.py \
    --model wespeaker-voxceleb-resnet34-lm --quant f32 --target hf
python3 tooling/publish-model/scripts/publish_model_targets.py \
    --model pyannote-segmentation-3.0 --quant f32 --target hf

# Registry cards + catalog entries (public gate probes the live HF repos):
tooling/publish-model/scripts/regenerate_all.sh --public wespeaker-voxceleb-resnet34-lm pyannote-segmentation-3.0

# Bump model-registry/catalog.epoch, then re-sign (OPENASR_CATALOG_SIGNING_KEY_SEED_HEX in env):
tooling/publish-model/scripts/publish_catalog.sh
```

Once published, `openasr pull wespeaker-voxceleb-resnet34-lm` and
`openasr pull pyannote-segmentation-3.0` install packs under
`openasr_home()/models/<id>/<quant>/`, where the diarization resolvers pick them
up automatically (matching `wespeaker`/`pyannote` directory-name substrings,
preferring `.oasr` over any dev safetensors). CLI `--diarize` on
transcribe/batch/live may install the missing required speaker-embedder pack as
the consent moment; server/session.start remains fail-closed with no download.
