# iOS / macOS SDK (xcframework)

`crates/openasr-ffi` exposes the OpenASR engine (`openasr-core`) through a C
ABI, packaged as `OpenASR.xcframework` for embedding in iOS and macOS apps.
This mirrors how comparable ggml-based ASR engines (whisper.cpp,
transcribe.cpp, CrispASR) ship a distributable SDK: a small, stable C surface
plus a generated header, not a Rust-shaped API.

## Scope (v1, deliberately minimal)

- Version query.
- Load a local `.oasr` model pack path -> opaque handle.
- Transcribe one whole in-memory 16 kHz mono PCM buffer (`f32` or `i16`) ->
  text, optional per-segment timestamps, detected/requested language.
- Error codes + last-error text; every call is panic-safe (no unwind crosses
  the FFI boundary; see `crates/openasr-ffi/src/lib.rs` module docs).

**Not in v1**: streaming/live dictation, the local HTTP server, or model
download. SDK consumers bring and manage their own `.oasr` packs -- OpenASR's
"no silent download" product boundary (see `AGENTS.md`) is a CLI/server
concern; an embedded SDK has no business reaching the network on its own.

**CPU-only v1.** `crates/openasr-core/build.rs` already builds ggml Metal
sources into the iOS/macOS static libs (inherited from the CLI/desktop build),
but the GPU backend is unevaluated on real iOS/macOS SDK integration --
Metal-on-M1 was already found to regress vs. CPU for this engine's decode
shape in desktop benchmarking (see `perf/PERFORMANCE.md`), and mobile GPU
acceleration needs its own on-device measurement before being recommended
here. Treat the SDK as CPU-only until a real-device benchmark says otherwise.

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
