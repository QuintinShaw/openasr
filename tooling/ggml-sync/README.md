# ggml submodule sync

The ggml backend is vendored as a git submodule at
`crates/openasr-core/third_party/openasr-ggml` (a submodule of the
`QuintinShaw/openasr-ggml` fork). `crates/openasr-core/build.rs` compiles it from
source with CMake, so the vendored tree is the real backend, not a mirror.

## Pin shape

Every pin is **one clean upstream `ggml-org/ggml` commit plus a small stack of
OpenASR-local patches** on top. Keeping the pin in that shape (clean base +
reviewed delta, no merge commits) is what lets a sync be a mechanical replay
instead of a history untangle.

The current local patch stack (oldest first):

1. backend-dl: open plugins with `LOAD_WITH_ALTERED_SEARCH_PATH` on Windows.
2. Filter `GGML_LOG_LEVEL_DEBUG` spam unless `GGML_DEBUG` / `OPENASR_GGML_DEBUG`.
3. Persist the Metal pipeline cache (MTLBinaryArchive serialised to disk for
   faster cold start).
4. Isolate that Metal pipeline cache under an `openasr/` cache subdir.
5. Default Metal residency sets off unless the device exposes the tensor API
   (`GGML_METAL_RESIDENCY_ENABLE` / `GGML_METAL_NO_RESIDENCY` override).

GGUF tensor shapes use upstream `gguf_get_tensor_ne`; do not restore local
per-dimension accessors.

Deliberately **not** carried on the pin:

- `metal-bin-f16` (F16 ADD/SUB/MUL/DIV) lives on the fork branch
  `oasr/metal-bin-f16`; F16 activation was ruled out and it duplicates upstream
  PR staging. Left off on purpose.
- `ggml_norm_back` / CONT-backward training ops -- OpenASR is inference-only and
  never references them; dropped to shrink the delta.

## Syncing to a newer upstream

```bash
git submodule update --init crates/openasr-core/third_party/openasr-ggml
tooling/ggml-sync/sync.sh --dry-run                 # fetch + print the plan
tooling/ggml-sync/sync.sh                            # replay onto upstream/master
# or pin it: tooling/ggml-sync/sync.sh --target <sha>
```

`sync.sh` derives the patch stack from the current pin
(`merge-base(target, pin)..pin`, non-merge, oldest first), checks out the target
upstream commit on a new branch in the submodule, and cherry-picks the stack
back. On a conflict it stops with the offending patch named so you can resolve,
`--continue`, or `--skip` (when upstream has subsumed a patch). It **does not**
commit, bump the superproject gitlink, or push -- those stay manual.

After a clean replay:

1. Build + run the full regression from the superproject
   (`cargo build`, `cargo nextest run --workspace`, `cargo test --workspace --doc`,
   `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`), plus the
   Metal depthwise smoke and, where model packs are available, the family
   golden/parity suites.
2. Only then record the new pin:
   `git add crates/openasr-core/third_party/openasr-ggml`.

## When CONV_2D_DW-style upstream ops land

Upstream gaining a native op (e.g. the Metal `GGML_OP_CONV_2D_DW` kernel) can let
a downstream detour retire. After a sync, check whether any backend-conditional
op routing in `crates/openasr-core/src/ggml_runtime/cpu_graph.rs` can collapse to
the now-native op, and drop the fallback rather than leaving both paths.
