# Native Streaming Smoke

This helper regenerates temporary true-streaming `.oasr` packs from local model
source trees and runs the ignored real-runtime smoke:

```bash
tooling/native-streaming-smoke/run.sh
```

It covers the seven native streaming families:

- `qwen`
- `whisper`
- `cohere`
- `moonshine`
- `parakeet`
- `wav2vec2`
- `xasr`

For each family, it:

1. imports a fresh `.oasr` pack into `tmp/native-streaming-smoke/`;
2. runs `show` and requires
   `openasr.features.streaming=ggml-true-streaming-v1`;
3. requires derived realtime capability to be `true_streaming` with partials;
4. runs `verify`;
5. runs `cargo test -p openasr-core native_streaming_real_runtime_smoke_from_env
   -- --ignored --nocapture` against `fixtures/jfk.wav`.

Useful focused run:

```bash
tooling/native-streaming-smoke/run.sh --families whisper,moonshine --max-ms 4000
```

Write a redacted evidence artifact:

```bash
tooling/native-streaming-smoke/run.sh \
  --families all \
  --workdir tmp/desktop-streaming-smoke \
  --skip-import \
  --summary-json tmp/native-streaming-smoke/summary.json \
  --summary-md tmp/native-streaming-smoke/validation.md
```

Smoke an existing release or candidate pack directly:

```bash
tooling/native-streaming-smoke/run.sh \
  --families qwen \
  --pack qwen=tmp/publish/qwen3-asr-0.6b/packs/qwen3-asr-0.6b-q8_0.oasr \
  --build-id "$(git rev-parse HEAD)" \
  --strict-release-evidence \
  --summary-json tmp/native-streaming-smoke/qwen-release-pack.json \
  --summary-md tmp/native-streaming-smoke/qwen-release-pack.md
```

`--pack FAMILY=PATH` bypasses local source import for that family and runs the
same inspect, validate, and ignored real-runtime smoke against the provided
`.oasr` file. Repeat `--pack` for multiple selected families. Summaries record
only the pack filename, SHA256, byte size, inspect-derived model identity,
inspect-derived runtime family, and `pack_origin=provided`; they do not copy
absolute release-pack paths.

Use `--strict-release-evidence` for final release-pack validation. It fails
unless every selected family is supplied through `--pack FAMILY=PATH`, a
runner build identifier is supplied through `--build-id`, and `--summary-json`
is set. `--summary-md` remains optional and useful for
a local validation evidence log, but JSON is required so the summary can be
machine-checked.
Strict mode also rejects pack filenames ending in `.streaming.oasr`, which are
reserved for local source-import/preflight smoke artifacts.
Leave strict mode off for local source-import or `--skip-import --workdir`
preflights.

The JSON summary records only the runner build id, family names, pack filenames,
pack SHA256 hashes, pack byte sizes, pack origin, inspect-derived model identity
and runtime family, the audio fixture filename,
the requested max duration, capability-gate booleans, and final transcript text.
It intentionally avoids absolute local source/audio paths, model weights, and
generated pack contents.

The Markdown summary is a redacted validation evidence block for a local
validation evidence log. It keeps `YYYY-MM-DD` as a placeholder so the operator
can paste it into the final release-pack validation record with the actual run
date and signed release
artifact references after publication.

Environment overrides:

```bash
OPENASR_STREAMING_SMOKE_BIN=target/debug/openasr \
OPENASR_STREAMING_SMOKE_AUDIO=fixtures/jfk.wav \
OPENASR_STREAMING_SMOKE_WORKDIR=tmp/native-streaming-smoke \
OPENASR_GGML_BACKEND=cpu \
tooling/native-streaming-smoke/run.sh
```

Generated packs are validation artifacts only. They stay under ignored `tmp/`
and must not be committed.

Local helper checks:

```bash
python3 -m unittest discover -s tooling/native-streaming-smoke -p 'test_*.py'
```
