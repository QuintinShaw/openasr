# Contributing to OpenASR

OpenASR is the public, **Apache-2.0** open core of a local-first speech-to-text
platform, under active native ggml-backed ASR development. Contributions are
welcome — issues, docs, and focused PRs alike.

Before writing code, skim [AGENTS.md](AGENTS.md): it lists the load-bearing
invariants (fail-closed `native`, deterministic `mock` default, `.oasr`-only pack
format, no auto-download, open-core trust boundary) that a PR must not regress.

Keep contributions aligned with active behavior and the [roadmap](docs/ROADMAP.md):

- the default executable backend is `native` (local ggml, fail-closed); `mock` is a deterministic opt-in stub (`--backend mock`) used in tests;
- `native` runs local `.oasr` packs and is fail-closed by staged boundaries;
- the only user-facing pack format is `.oasr` (GGUF-backed internally);
- there is no model/runtime auto-download command surface.

## Building from source

The native backend (ggml) is a **git submodule compiled from source**, so a plain
`git clone` will not build until you fetch it:

```bash
git clone --recurse-submodules https://github.com/QuintinShaw/openasr.git
# or, in an existing clone:
git submodule update --init --recursive
```

`crates/openasr-core/build.rs` invokes **CMake** to compile ggml (C/C++, plus Metal
on macOS), so you also need:

- a C/C++ toolchain (clang/gcc; Xcode Command Line Tools on macOS),
- `cmake`,
- on Linux, `libasound2-dev` (ALSA headers, required by `cpal`).

Rust is pinned by `rust-toolchain.toml` (edition 2024). The toolchain installs
automatically with rustup.

## Local setup

From the repository root:

```bash
cargo run -p openasr-cli -- --help
cargo run -p openasr-cli -- list
cargo run -p openasr-cli -- transcribe fixtures/jfk.wav --model whisper-small --backend mock --format json
```

## Testing

Default tests stay deterministic, local, and network-free.

```bash
cargo nextest run --workspace   # or: cargo test
```

Do not require model weights, runtime binaries, private data, or network access in
default test paths. Tests default to the CPU ggml backend.

## Useful local smoke checks

```bash
cargo run -p openasr-cli -- --help
cargo run -p openasr-cli -- list
cargo run -p openasr-cli -- verify /path/to/model.oasr
cargo run -p openasr-cli -- transcribe fixtures/jfk.wav --model whisper-small --backend mock --format text
cargo run -p openasr-cli -- transcribe fixtures/jfk.wav --benchmark --backend mock --model whisper-small --format text
cargo run -p openasr-cli -- doctor
```

## Optional Docker smoke

```bash
docker build -t openasr:local .
docker run --rm -d --name openasr-docker-smoke -p 18080:8080 openasr:local
curl -fsS http://127.0.0.1:18080/health
docker rm -f openasr-docker-smoke
```

## Formatting and linting

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo nextest run --workspace
```

## Docs expectations

Update docs whenever behavior changes.

User-facing docs must stay explicit about:

- `mock` default behavior;
- `native` local-pack execution and fail-closed boundaries;
- `.oasr`-only user-facing pack format;
- no model/runtime auto-download surface;
- local-first, no-cloud-by-default posture.

## Safety and artifact policy

Do not add or commit:

- model weights;
- runtime binaries;
- downloaded caches/temporary download files;
- fake/unverified URL/hash metadata;
- secrets or signing seeds, or private audio.

## Branch naming

Use short descriptive branch names:

```text
feat/...
fix/...
docs/...
chore/...
test/...
```

## Commit and PR conventions

Commit and PR titles follow **Conventional Commits with a scope**, matching the
repository history:

```text
<type>(<scope>): <summary> (#<pr-or-issue>)
```

- `<type>`: one of `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `ci`, `chore`.
- `<scope>`: the affected module — e.g. `core`, `qwen`, `server`, `pull`, `catalog`,
  `ggml`, `oadp`, `windows`. Omit only for truly cross-cutting changes.
- Reference the PR or issue number, e.g. `fix(qwen): default native-GQA off on the
  discrete-GPU lane (#220)`.
- Use **ASCII punctuation** in titles (`-` not `—`, `->` not `→`, `...` not `…`).
- Keep one feature or fix per PR; open a follow-up rather than bundling unrelated work.

## AI usage

OpenASR is built with heavy AI assistance, and that is welcome — but the bar is
**human ownership, not human keystrokes**: whoever submits a change must understand
it fully, be able to explain any line without AI help, and own its maintenance.
**Disclose** meaningful AI involvement in the PR (the template has a field), and
write PR descriptions, issue reports, and reviewer responses yourself — not with AI.
The full policy and the rules for AI coding agents live in [AGENTS.md](AGENTS.md).

## Developer Certificate of Origin (sign-off)

OpenASR uses **inbound = outbound**: contributions are accepted under the same
[Apache-2.0](LICENSE) license that covers the project. We require a
[Developer Certificate of Origin](https://developercertificate.org/) sign-off on
each commit — it certifies you wrote the patch or otherwise have the right to submit
it under that license. Add it with `git commit -s`, which appends:

```text
Signed-off-by: Your Name <you@example.com>
```

## PR checklist

Before opening a PR:

- scope is focused;
- behavior changes include tests;
- docs are updated;
- `cargo fmt --check` passes;
- `cargo clippy --all-targets -- -D warnings` passes;
- relevant tests pass;
- no forbidden artifacts were committed;
- meaningful AI use is disclosed (see [AGENTS.md](AGENTS.md));
- commits are signed off (`git commit -s`).
