# Diarization pack publishing (ReDimNet2-B6 + pyannote)

The diarization models are **auxiliary** packs — ReDimNet2-B6 emits speaker
embeddings (192-d), and pyannote segmentation-3.0 emits speaker-change /
overlap-aware speech regions. They are not ASR models: they have no
transcription path of their own and are only pulled when a user opts into
`--diarize` / Voice ID.

## Catalog entries

- Capability packs: `redimnet2-b6-cn` and `pyannote-segmentation-3.0`.
- The published catalog keeps the ReDimNet2-B6 `fp16` / `q8_0` / `f32` variants
  and the pyannote `f32` variant.
- ReDimNet2-B6 is the **only** supported speaker embedder. When it is missing,
  diarization and Voice ID fail closed with a clear install error (no fallback
  embedder).

## Build / publish overview

### ReDimNet2-B6

Use the external converter and the normal capability-pack publish lane:

```bash
# Convert upstream checkpoint -> .oasr (see tooling/redimnet2/convert_redimnet2.py)
python3 tooling/redimnet2/convert_redimnet2.py ...

# Materialize result sidecars + metrics, then publish HF + regenerate catalog
python3 tooling/publish-model/scripts/materialize_result_sidecars.py redimnet2-b6-cn --quant fp16
# ... repeat for q8_0 / f32 as needed ...
tooling/publish-model/scripts/regenerate_all.sh --public redimnet2-b6-cn
```

Runtime override for local packs:

```bash
export OPENASR_REDIMNET_PACK=/path/to/redimnet2-b6-cn-fp16.oasr
```

### pyannote segmentation-3.0

```bash
openasr model-pack import pyannote \
    tmp/pyannote/pytorch_model.bin \
    tmp/publish/pyannote-segmentation-3.0/packs/pyannote-segmentation-3.0-f32.oasr \
    --package-id pyannote-segmentation-3.0

python3 tooling/publish-model/scripts/materialize_result_sidecars.py pyannote-segmentation-3.0 --quant f32
tooling/publish-model/scripts/regenerate_all.sh --public pyannote-segmentation-3.0
```

## Catalog signing

Bump `model-registry/catalog.epoch`, then re-sign with the production seed
(`OPENASR_CATALOG_SIGNING_KEY_SEED_HEX` in env):

```bash
tooling/publish-model/scripts/publish_catalog.sh
```

That refreshes the committed full + public catalog signatures. Deploying the
public projection to Cloudflare is a separate step (see `publish_catalog.sh`
notes / deploy-catalog workflow).

## Operator pull

Once published:

```bash
openasr pull redimnet2-b6-cn
openasr pull pyannote-segmentation-3.0
```

Installed packs are discovered automatically (matching `redimnet` / `pyannote`
directory-name substrings under the models home, or via the env overrides
above).
