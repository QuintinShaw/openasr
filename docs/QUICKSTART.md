# Quickstart

Get a real transcript on your own machine in three commands. OpenASR is
local-first: native is the default backend, audio never leaves your machine, and
there is no telemetry.

> Building from source compiles the ggml backend, so clone recursively and have a
> C/C++ toolchain available (`cmake`, a compiler, and on Linux `libasound2-dev`).
> The first build takes several minutes while ggml compiles; later builds are
> incremental. See [README](../README.md#quick-start) for the full build notes.

## 1. Build

```bash
git clone --recurse-submodules https://github.com/QuintinShaw/openasr.git && cd openasr
cargo build --release -p openasr-cli
# the binary is target/release/openasr -- put it on your PATH, or prefix the
# commands below with `cargo run -p openasr-cli --`.
```

## 2. Transcribe

```bash
openasr transcribe audio.wav
```

The first run offers to download the default model (`qwen3-asr-0.6b`) with a
visible confirmation showing the model, quantization, size, host, and license,
then everything runs offline. Add `--yes` to confirm non-interactively, or
`--offline` to fail closed instead of downloading.

## 3. Pick a model, a format, an output

```bash
openasr search                                  # browse the catalog
openasr pull whisper-small                      # install another model
openasr transcribe audio.wav -m whisper-small -f srt -o audio.srt
openasr transcribe ./recordings -o ./transcripts   # a whole folder
```

Short flags: `-m/--model` (also `OPENASR_MODEL`), `-f/--format` (`text`, `json`,
`srt`, `vtt`, `verbose_json`, `markdown` -- repeat `-f` to write several at
once), `-o/--output`, `-l/--language` (`auto` or a hint like `en`). `openasr t`
is an alias for `transcribe`. Run `openasr --help` for the full command set, and
see [FAQ](FAQ.md) and [Known Limitations](KNOWN_LIMITATIONS.md) for current
scope.
