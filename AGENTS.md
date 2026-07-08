# AGENTS.md — OpenASR

Guidance for AI coding agents (and humans skimming for the rules) working in this
repository. For step-by-step contributor setup see [CONTRIBUTING.md](CONTRIBUTING.md);
this file is the short, load-bearing list of what must stay true.

OpenASR is a **local-first speech-to-text platform**, maintained as the public
**Apache-2.0 open core**. It ships a Rust CLI, a local OpenAI-compatible HTTP API
subset, a model-metadata registry, and native ggml-backed ASR execution.

## Repository map

```text
crates/
  openasr-core/          # engine: model families, ggml runtime, .oasr packs, catalog, trust boundaries
  openasr-cli/           # the `openasr` binary
  openasr-server/        # local HTTP API + pairing/remote-compute server
  openasr-client/        # client trust primitives (TOFU pinning, pairing safety codes)
  openasr-system-audio/  # system-audio capture backends (macOS/Windows/Linux)
docs/                    # ROADMAP, KNOWN_LIMITATIONS, FAQ, format contracts, design docs
tooling/publish-model/   # model-pack publishing pipeline + public model-card templates
model-registry/          # bundled catalog + signed manifest (public verification key only)
perf/                    # performance harness, suite.toml, committed baselines
```

## Building from source (the part agents forget)

The ggml backend is a **git submodule** compiled from source, so a plain clone will
not build. The very first step is always:

```bash
git submodule update --init --recursive        # pulls crates/openasr-core/third_party/openasr-ggml
```

`crates/openasr-core/build.rs` shells out to **CMake** and compiles ggml C/C++/Metal,
so the host also needs a C/C++ toolchain, `cmake`, and on Linux `libasound2-dev`
(ALSA, for `cpal`). Rust is pinned by `rust-toolchain.toml` (edition 2024). Then:

```bash
cargo run -p openasr-cli -- --help
cargo run -p openasr-cli -- transcribe fixtures/jfk.wav --model whisper-small --backend mock --format json
```

## Invariants — do not regress these

These encode product promises ("no telemetry / fail-closed / no silent download")
and the open-core trust boundary. Treat them as hard constraints:

- **`native` is the default backend** and is fail-closed by stage. It runs local
  ggml `.oasr` packs; new code must fail closed with typed errors, never fabricate
  output, and never reach for the network silently. **`mock` is an opt-in
  deterministic stub** (`--backend mock`, hidden in `--help`) for plumbing/CI;
  default tests pass `--backend mock` explicitly and stay local, network-free, and
  weight-free.
- **No silent downloads.** Auto-install of a missing model happens **only in the
  CLI** `transcribe`/`live` handlers, **only through a visible consent prompt**
  (model, quant, size, host, license), and fails closed before any network access
  in non-interactive or `--offline` runs. The shared resolve paths and the
  **server transcription path never pull** -- serving runs only an explicit local
  `.oasr` pack, and the only server-side install path is the
  **operator-authenticated** pull API (`POST /v1/models/{id}/pull`, operator-only
  per `is_operator_only_path`). Keep consent-pull CLI-only and server pulls
  operator-gated so no transcription request can ever trigger a download.
- **`.oasr` is the only user-facing pack format** (GGUF-backed internally). Bare
  `.gguf` is not accepted as run input or importer output.
- **Trust-boundary code stays in the open core.** Anything that decides audio flow,
  catalog/model signature verification, fail-closed dispatch, TLS/pairing, local-path
  validation, or sandbox/`.oasr`-zip parsing belongs in these crates and must remain
  panic-free on untrusted input (zip-slip safe, checked arithmetic on parsed sizes).
- **Cross-language contracts** (e.g. the canonical quant tag) have a single source of
  truth here; keep fixtures like `crates/openasr-core/tests/fixtures/quant_tag_cases.json` authoritative.
- **Keep infrastructure model-agnostic.** Generic capabilities sink into the base
  layers; model-family semantics stay under `models/` and the `arch/` descriptors.
  Don't push family-specific tensor logic into shared infrastructure.
- **One greedy decode driver.** Every AED / autoregressive seq2seq family reaches
  greedy decode through the single shared driver
  `run_seq2seq_greedy_decode_loop_v0` (via `run_builtin_seq2seq_decode_policy`). A
  new such family MUST provide a `Seq2SeqGreedyDecodeStepExecutor` and declare a
  decode-policy descriptor in `models/decode_policy_component_registry.rs`; it MUST
  NOT hand-write its own argmax step loop or build a decode config that bypasses the
  registry. Hand-rolled loops miss the shared degenerate-loop guard and drift the
  argmax / suppression / stop-token semantics the driver centralizes (a hand-rolled
  firered loop is what caused the long-audio repetition, issue #60). The
  `*.greedy.seq2seq.*` registry-resolution test fails closed on a half-connect.

## Validation before you finish

Run the minimal sufficient set for the change; raise the bar when you touch a trust
boundary, the catalog, runtime dispatch, the `.oasr` format, server auth/pairing,
system audio, or model pull.

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo nextest run --workspace          # or: cargo test
cargo test --workspace --doc
# Trust-boundary / format changes also:
cargo test -p openasr-core golden_diff -- --nocapture
cargo test -p openasr-core bundled_catalog_signature_verifies_committed_catalog_and_epoch -- --nocapture
```

Keep claims tied to executed checks — do not assert performance/quality wins the
harness has not produced.

## Artifact & safety policy

Never add or commit: model weights, runtime binaries, downloaded caches/temp
download files, fake or unverified URL/hash metadata, secrets or signing seeds, or
private/customer audio. The only key material in-tree is the catalog **public**
verification key.

## PRs and commits

Focused scope (one feature or fix per PR); behavior changes carry tests; docs
updated alongside behavior; no forbidden artifacts. Commit and PR titles follow
**Conventional Commits with a scope**, matching the existing history:

```text
<type>(<scope>): <summary> (#<pr-or-issue>)
```

`<scope>` is the affected module (`core`, `qwen`, `server`, `pull`, `catalog`,
`ggml`, ...). Use **ASCII punctuation** in titles and code: `-` not the em-dash,
`->` not the arrow glyph, `...` not the ellipsis glyph. See
[CONTRIBUTING.md](CONTRIBUTING.md) for branch naming, the full `<type>` list, the
PR checklist, and the DCO sign-off.

## AI usage policy

OpenASR is built with heavy AI assistance, and that is fine — the bar is **human
ownership, not human keystrokes**. For every change, whoever submits it:

- **Understands it fully** and can explain any line to a reviewer *without* AI help.
- **Owns its maintenance** — fixes the bugs, answers the review.
- **Discloses** meaningful AI involvement in the PR (one line; trivial autocomplete
  needs none). Never use AI to write PR descriptions, issue reports, or reviewer
  responses — write those yourself.
- Treats AI as **accelerant, not author**: a change you cannot review in full is too
  large to submit.

## Guidelines for AI coding agents

When you (an AI agent) work in this repo:

- **Read before you write.** Read the relevant files and this `AGENTS.md`;
  your change must blend into the surrounding patterns. If a change is large or
  introduces a new pattern or subsystem, **PAUSE and confirm with the user** first.
- **Reuse over invent.** Prefer existing infrastructure; avoid new subsystems or
  invasive rewrites that risk existing behavior.
- **Comments earn their place.** Explain non-obvious invariants, not what the code
  already says; never leave task-specific "this fixes the bug you mentioned" notes.
- **ASCII in code and commit titles** (`-`, `->`, `x`, `...`) — no emdash, unicode
  arrows, or decorative symbols. (Prose docs may stay non-ASCII where intentional.)
- **Never push or open PRs on a contributor's behalf without explicit, per-action
  approval.** Do not run `git push`, `gh pr create`, or `gh pr comment` unprompted;
  read-only context gathering (`gh search issues/prs`, `grep`) is fine.
- When a contributor explicitly asks you to commit for them, mark it with an
  **`Assisted-by: <assistant name>`** trailer -- *not* `Co-authored-by:`. Prefer
  leaving the commit message to the contributor.
- When uncertain, err toward minimal assistance and ask.
