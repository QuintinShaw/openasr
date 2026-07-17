# iOS / macOS SDK (xcframework)

`crates/openasr-ffi` exposes the OpenASR engine (`openasr-core`) through a C
ABI, packaged as `OpenASR.xcframework` for embedding in iOS and macOS apps.
This mirrors how comparable ggml-based ASR engines (whisper.cpp,
transcribe.cpp, CrispASR) ship a distributable SDK: a small, stable C surface
plus a generated header, not a Rust-shaped API.

## Scope (v1, deliberately minimal)

- Version query.
- Load a local `.oasr` model pack path -> opaque handle.
- **Batch**: transcribe one whole in-memory 16 kHz mono PCM buffer (`f32` or
  `i16`) -> text, optional per-segment timestamps, detected/requested language.
- **Streaming**: open a live session, feed 16 kHz mono `f32` PCM chunks, receive
  incremental partial/committed transcript events, then finish for the assembled
  final transcript (`openasr_streaming_*`). This is the in-process,
  transport-free path an iOS app uses for live captioning -- iOS cannot spawn the
  desktop realtime server, so it links the same
  `openasr_core::StreamingSession` engine directly.
- **Model market**: fetch and verify the signed model catalog, then (under
  explicit consent) download + sha256-verify + install a model pack, list
  installed packs, and remove one (`openasr_catalog_*` / `openasr_pull_model` /
  `openasr_install_local_pack` / `openasr_list_installed_json` /
  `openasr_remove_model`). This is the on-device equivalent of `openasr pull`,
  for a native app that has no CLI or local server to lean on.
- Error codes + last-error text; every call is panic-safe (no unwind crosses
  the FFI boundary; see `crates/openasr-ffi/src/lib.rs` module docs).

**Not in v1**: the local HTTP server. SDK consumers can still bring and manage
their own `.oasr` packs directly (`openasr_model_open`), but the market API above
also lets an app fetch the signed catalog and pull packs itself.

**"No silent download" still holds.** The market API never touches the network
on its own: fetching the catalog and pulling a model are each an *explicit* call
the app makes -- the catalog fetch to render its market UI, the pull only after
showing the user the model/quant/size/host/license and getting consent. There is
no auto-install path and no transcription path that can trigger a download, and
the download's URL + sha256 come from the in-core verified catalog (the app only
passes a `model:quant` reference), so an app cannot redirect the fetch or bypass
the digest check. All catalog/pack signature verification stays in `openasr-core`
(see `AGENTS.md`), exactly as the CLI does it.

**CPU-only v1, and iOS Metal is not wired up at all yet.** `crates/openasr-core/build.rs`
gates the `GGML_METAL`/`GGML_METAL_EMBED_LIBRARY` cmake flags on
`target.contains("apple-darwin")`, i.e. **macOS only** -- `aarch64-apple-ios`
and `aarch64-apple-ios-sim` do not match that check, so the `ios-arm64` and
`ios-arm64-simulator` xcframework slices are built **CPU/NEON-only**, with no
Metal backend compiled in at all. The `macos-arm64` slice does get the same
Metal build the CLI/desktop build already produces, but it is unevaluated on
real SDK integration -- Metal-on-M1 was already found to regress vs. CPU for
this engine's decode shape in desktop benchmarking (see
`perf/PERFORMANCE.md`), and mobile GPU acceleration needs its own on-device
measurement before being recommended here. Treat the whole SDK as CPU-only
until a real-device benchmark says otherwise; wiring up an iOS Metal build is
unstarted, follow-up work.

**License**: Apache-2.0, same as the rest of this open-core repository (see
`LICENSE`, `NOTICE`). No additional restriction applies to the xcframework
artifact or the generated header.

## Building

```bash
# One-time: install cbindgen if you don't have it (only needed after
# changing crates/openasr-ffi/src/lib.rs's public surface).
cargo install cbindgen --locked
scripts/generate-ffi-header.sh          # regenerate crates/openasr-ffi/include/openasr.h
scripts/generate-ffi-header.sh --check  # CI gate: fails if the committed header is stale

# Build the xcframework (all three slices, if the host can produce them):
scripts/build-xcframework.sh
# -> target/xcframework/OpenASR.xcframework
```

`scripts/build-xcframework.sh` builds three architecture slices:

| Slice | Rust target | Requires |
| --- | --- | --- |
| `ios-arm64` (device) | `aarch64-apple-ios` | full Xcode (`iphoneos` SDK) |
| `ios-arm64-simulator` | `aarch64-apple-ios-sim` | full Xcode (`iphonesimulator` SDK) |
| `macos-arm64` (host) | `aarch64-apple-darwin` | Xcode Command Line Tools |

**Command Line Tools alone (no full Xcode) cannot build the device/simulator
slices or run `xcodebuild -create-xcframework` at all** -- this is the same
constraint the Phase 1 `ios-compile` CI gate already documented (no
`iphoneos`/`iphonesimulator` SDK under CLT). The script detects this, skips
the slices it cannot build, and -- if `xcodebuild` itself cannot run --
degrades to just staging the buildable static lib(s) and header under
`target/xcframework/slices/` instead of failing outright. Install Xcode
(`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`) to build
locally, or use CI: the `xcframework` job in `.github/workflows/ci.yml`
(`workflow_dispatch`, `macos-latest`, full Xcode) always produces all three
slices and uploads `OpenASR.xcframework` as a build artifact. It is
intentionally not part of the per-PR gate -- `ios-compile` already covers
`openasr-ffi`'s aarch64-apple-ios compile health on every push/PR at a
fraction of the cost.

## C API

See the generated header, `crates/openasr-ffi/include/openasr.h`, for the
authoritative, fully-documented signatures. Shape:

```c
const char *openasr_version(void);

// Load / free a validated .oasr pack handle.
OpenAsrStatus openasr_model_open(const char *path, OpenAsrModel **out_model);
void openasr_model_close(OpenAsrModel *model);

// Transcribe one in-memory 16 kHz mono PCM buffer.
OpenAsrStatus openasr_transcribe_pcm(OpenAsrModel *model,
                                     const void *pcm, uintptr_t pcm_len_samples,
                                     OpenAsrPcmFormat format, uint32_t sample_rate_hz,
                                     bool with_segments,
                                     OpenAsrResult **out_result);
void openasr_result_free(OpenAsrResult *result);

// Read the result (whisper.cpp-style index accessors, not a raw struct --
// the result owns Rust-side strings that are not a valid C value type).
const char *openasr_result_text(const OpenAsrResult *result);
const char *openasr_result_language(const OpenAsrResult *result);       // nullable
uintptr_t   openasr_result_segment_count(const OpenAsrResult *result);
float       openasr_result_segment_start(const OpenAsrResult *result, uintptr_t index);
float       openasr_result_segment_end(const OpenAsrResult *result, uintptr_t index);
const char *openasr_result_segment_text(const OpenAsrResult *result, uintptr_t index);

const char *openasr_last_error_message(void);
```

Example (C), transcribing 16-bit PCM already in memory:

```c
#include "openasr.h"

OpenAsrModel *model = NULL;
if (openasr_model_open("/path/to/model.oasr", &model) != OPEN_ASR_STATUS_OK) {
    fprintf(stderr, "load failed: %s\n", openasr_last_error_message());
    return 1;
}

OpenAsrResult *result = NULL;
OpenAsrStatus status = openasr_transcribe_pcm(
    model, pcm_i16, pcm_sample_count, OPEN_ASR_PCM_FORMAT_S16, 16000,
    /*with_segments=*/true, &result);
if (status != OPEN_ASR_STATUS_OK) {
    fprintf(stderr, "transcribe failed: %s\n", openasr_last_error_message());
    openasr_model_close(model);
    return 1;
}

printf("%s\n", openasr_result_text(result));
for (uintptr_t i = 0; i < openasr_result_segment_count(result); i++) {
    printf("[%.2f-%.2f] %s\n", openasr_result_segment_start(result, i),
           openasr_result_segment_end(result, i), openasr_result_segment_text(result, i));
}

openasr_result_free(result);
openasr_model_close(model);
```

### Streaming (live captioning)

The streaming surface is a separate session type -- not the batch
`OpenAsrModel` -- because it holds live decoder state across chunks. Shape (see
the header for the full accessor set and ownership rules):

```c
// Open / drive / finish a live session. finish() consumes the session and
// returns the final transcript as an OpenAsrResult (read with the same
// openasr_result_* accessors as the batch path).
OpenAsrStatus openasr_streaming_session_open(const char *path,
                                             const OpenAsrStreamingConfig *config,  // NULL = defaults
                                             OpenAsrStreamingSession **out_session);
OpenAsrStatus openasr_streaming_feed(OpenAsrStreamingSession *session,
                                     const float *pcm, uintptr_t pcm_len_samples,  // 16 kHz mono f32
                                     OpenAsrStreamingEvents **out_events);
OpenAsrStatus openasr_streaming_finish(OpenAsrStreamingSession *session, OpenAsrResult **out_result);
void openasr_streaming_free(OpenAsrStreamingSession *session);   // abort without finishing

// Read one feed()'s events by index (whisper.cpp-style accessors again).
uintptr_t                 openasr_streaming_events_count(const OpenAsrStreamingEvents *events);
OpenAsrStreamingEventKind openasr_streaming_event_kind(const OpenAsrStreamingEvents *events, uintptr_t i);
const char               *openasr_streaming_event_text(const OpenAsrStreamingEvents *events, uintptr_t i);
const char               *openasr_streaming_event_segment_id(const OpenAsrStreamingEvents *events, uintptr_t i);
uint64_t                  openasr_streaming_event_start_ms(const OpenAsrStreamingEvents *events, uintptr_t i);
void                      openasr_streaming_events_free(OpenAsrStreamingEvents *events);
```

```c
OpenAsrStreamingSession *session = NULL;
if (openasr_streaming_session_open("/path/to/model.oasr", NULL, &session) != OPEN_ASR_STATUS_OK) {
    fprintf(stderr, "open failed: %s\n", openasr_last_error_message());
    return 1;
}

// Feed captured audio as it arrives (any chunk length):
while (capturing) {
    OpenAsrStreamingEvents *events = NULL;
    if (openasr_streaming_feed(session, chunk_f32, chunk_len, &events) != OPEN_ASR_STATUS_OK) break;
    for (uintptr_t i = 0; i < openasr_streaming_events_count(events); i++) {
        printf("%s: %s\n",
               openasr_streaming_event_kind(events, i) == OPEN_ASR_STREAMING_EVENT_KIND_COMMITTED
                   ? "final" : "partial",
               openasr_streaming_event_text(events, i));
    }
    openasr_streaming_events_free(events);
}

OpenAsrResult *final = NULL;
if (openasr_streaming_finish(session, &final) == OPEN_ASR_STATUS_OK) {   // consumes session
    printf("%s\n", openasr_result_text(final));
    openasr_result_free(final);
}
// session is already freed by finish(); on an error path use openasr_streaming_free(session) instead.
```

### Model market (catalog + pull / install / remove)

The market surface lets a native app show a catalog of downloadable models and
install them on device, with the same trust boundary the CLI enforces. The app
supplies its own sandbox directory as the OpenASR home (the iOS equivalent of
`OPENASR_HOME`); packs install under `<home>/models/...`. Shape (see the header
for full ownership rules):

```c
// Fetch + verify the signed catalog (NULL url = the built-in production endpoint).
OpenAsrStatus openasr_catalog_fetch(const char *catalog_url, const char *home_dir,
                                    OpenAsrCatalog **out_catalog);
const char   *openasr_catalog_json(const OpenAsrCatalog *catalog);  // verified catalog, borrowed
void          openasr_catalog_free(OpenAsrCatalog *catalog);

// Download + sha256-verify + install a model (after showing consent). NULL quant
// = device-recommended; NULL source = automatic chain; callbacks may be NULL.
OpenAsrStatus openasr_pull_model(const OpenAsrCatalog *catalog,
                                 const char *reference, const char *quant, const char *source,
                                 bool accept_license, const char *home_dir,
                                 OpenAsrPullProgressCallback progress_cb, void *progress_user_data,
                                 OpenAsrPullCancelCallback cancel_cb, void *cancel_user_data,
                                 char **out_installed_json);   // free with openasr_string_free

// Verify + install a .oasr the app already has on disk (sha256/size must match catalog).
OpenAsrStatus openasr_install_local_pack(const OpenAsrCatalog *catalog, const char *oasr_path,
                                         const char *home_dir,
                                         OpenAsrPullProgressCallback progress_cb, void *progress_user_data,
                                         char **out_installed_json);

OpenAsrStatus openasr_list_installed_json(const char *home_dir, char **out_json);  // JSON array
OpenAsrStatus openasr_remove_model(const char *home_dir, const char *reference, bool *out_removed);
void          openasr_string_free(char *string);   // free out_*_json strings only
```

Typical flow: `openasr_catalog_fetch` -> parse `openasr_catalog_json` to render
the market and the per-model download disclosure (size, host, license) -> on the
user's tap, `openasr_pull_model(catalog, "moonshine-tiny", NULL, ...)` with a
progress callback and a cancel flag -> the installed pack's `path` (from
`out_installed_json` or `openasr_list_installed_json`) is what you hand to
`openasr_model_open` / `openasr_streaming_session_open`. Run the pull off the main
thread: it blocks for the whole download, invoking the progress/cancel callbacks
synchronously on that same thread.

### Swift bridging (minimal sketch)

Add `OpenASR.xcframework` to your Xcode target, then bridge it with a module
map (`module.modulemap`) so Swift sees the C API directly -- no
Objective-C++ wrapper required for this small a surface:

```
// OpenASR.xcframework/.../Headers/module.modulemap
module OpenASRFFI {
    header "openasr.h"
    export *
}
```

```swift
import OpenASRFFI

func transcribe(modelPath: String, pcm: [Int16]) throws -> String {
    var model: OpaquePointer?
    guard openasr_model_open(modelPath, &model) == OPEN_ASR_STATUS_OK else {
        throw NSError(domain: "OpenASR", code: 1, userInfo: [
            NSLocalizedDescriptionKey: String(cString: openasr_last_error_message())
        ])
    }
    defer { openasr_model_close(model) }

    var result: OpaquePointer?
    let status = pcm.withUnsafeBufferPointer { buffer in
        openasr_transcribe_pcm(model, buffer.baseAddress, buffer.count,
                                OPEN_ASR_PCM_FORMAT_S16, 16000, true, &result)
    }
    guard status == OPEN_ASR_STATUS_OK else {
        throw NSError(domain: "OpenASR", code: Int(status.rawValue), userInfo: [
            NSLocalizedDescriptionKey: String(cString: openasr_last_error_message())
        ])
    }
    defer { openasr_result_free(result) }

    return String(cString: openasr_result_text(result))
}
```

No Xcode project is checked into this repo for the bridging example --
build one against `OpenASR.xcframework` directly if you want a runnable app
target.

## Verifying a build

The macOS slice is directly runnable with a small C smoke test that links
the static lib and transcribes `fixtures/jfk.wav`'s PCM through a real
installed `.oasr` pack (e.g. anything under `~/.openasr/models/*/*/*.oasr`):

```bash
clang smoke.c -I target/xcframework/slices/macos-arm64/include \
  target/xcframework/slices/macos-arm64/lib/libopenasr_ffi.a \
  -framework Accelerate -framework Foundation -framework Metal -framework MetalKit \
  -framework Security -framework CoreFoundation -framework SystemConfiguration -lc++ \
  -o smoke
./smoke ~/.openasr/models/whisper-tiny.en/q8_0/whisper-tiny.en-q8_0.oasr fixtures/jfk.wav
```

The iOS device/simulator slices cannot run on a Mac host directly; verify
those with `lipo -info` (architecture) and `nm -gU` (the `openasr_*` symbols
are exported) against the staged `.a` for that slice, and treat an on-device
build (a real iOS app target linking the xcframework) as the canonical
end-to-end check, the same posture `docs/ANDROID_BUILD.md` takes for Android.
