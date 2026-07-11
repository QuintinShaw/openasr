//! Golden "regenerate == committed" guard for the HTTP daemon response wire
//! contract's generated TypeScript bindings (`/health`, `/v1/models`,
//! `/v1/capabilities`, `/v1/devices`).
//!
//! # Why ts-rs, and why this shape
//!
//! This mirrors `crates/openasr-core/src/realtime/wire_bindings_test.rs`
//! (the realtime WebSocket wire contract, added first): before this test
//! existed, `apps/desktop/src/lib/openasrClient.ts`,
//! `apps/desktop/src/lib/preferencesClient.ts`, and
//! `apps/desktop/src/lib/daemonComputeDevices.ts` each hand-duplicated a
//! subset of these response shapes as TypeScript types with no compiler link
//! back to the Rust structs -- the exact same drift risk the realtime PR
//! documented (see its module doc for the drift already found there).
//!
//! `HealthResponse`, `ModelsResponse`, `ModelResponse`, `CapabilitiesResponse`,
//! and `DevicesResponse` (in `crate::lib`) now carry a dev-only
//! `#[cfg_attr(test, derive(ts_rs::TS))]`, same as the realtime types. `ts-rs`
//! is a `[dev-dependencies]`-only crate here too (see this crate's
//! `Cargo.toml` comment) -- the derive never becomes part of the shipped
//! rlib.
//!
//! `CapabilitiesResponse` and `DevicesResponse` embed leaf types
//! (`TranscriptionBackendCapabilities`, `BackendKind`, `LanguageCapability`,
//! `ComputeDevice`, and transitively `RealtimeBackendCapabilities`) that are
//! *defined in openasr-core*, not here. Plain `cfg(test)` inside
//! openasr-core only turns on when openasr-core itself is compiled as its
//! own test target -- it stays off when openasr-core is linked as an
//! ordinary library dependency of this crate's test binary, which is what
//! happens here. openasr-core therefore exposes a `ts-export` feature
//! (`derive(ts_rs::TS)` gated on `cfg(any(test, feature = "ts-export"))`)
//! that this crate's `[dev-dependencies]` edge turns on for exactly this
//! test target (see both crates' `Cargo.toml` comments); openasr-core's own
//! real build and every other consumer never enable it, so ts-rs stays
//! fully out of the shipped rlib either way. This is a deliberate departure
//! from the realtime wire PR's plain `cfg(test)` gate, which sufficed there
//! because every exported type in that PR is defined in openasr-core and
//! consumed by openasr-core's own golden test -- no crate boundary to cross.
//!
//! # Directory layout: why some files land under `generated/realtime-wire/`
//! here too, duplicating openasr-core's copy
//!
//! `CapabilitiesResponse.realtime` is `openasr_core::RealtimeBackendCapabilities`
//! and `DevicesResponse.devices` is `Vec<openasr_core::ComputeDevice>` --
//! types ts-rs must recursively export as real dependencies of the response
//! structs being exported here. ts-rs resolves every type's committed file
//! location as `<this test's Config::out_dir>` joined with that type's own
//! `#[ts(export_to = "...")]` attribute; `RealtimeBackendCapabilities` (and
//! its own dependents `RealtimeTranslationCapability`, `RealtimeBackendMode`,
//! `BackendFeatureCapability`, `BackendCapabilityBehavior`) already carry
//! `export_to = "generated/realtime-wire/"` from the realtime wire PR. Since
//! this test's `Config::out_dir` is *this crate's own* `CARGO_MANIFEST_DIR`
//! (matching the established per-crate convention -- see the realtime test),
//! not openasr-core's, those dependency files land a second time under
//! `crates/openasr-server/generated/realtime-wire/`, byte-for-byte identical
//! to `crates/openasr-core/generated/realtime-wire/`.
//!
//! This is intentional, not accidental drift: pointing this crate's `Config`
//! at openasr-core's `CARGO_MANIFEST_DIR` instead (to physically de-duplicate)
//! would mean this crate's regenerate step writes into a sibling crate's
//! source tree from a relative `../openasr-core` path -- a new,
//! more surprising mechanism than "each crate's golden test owns its own
//! `generated/` tree", which is the mechanism the realtime wire PR already
//! established. The small duplicated leaf-type files stay safe because both
//! copies are independently golden-guarded against the same Rust source of
//! truth (this test, and openasr-core's own): a future field change fails
//! whichever copy's test hasn't been regenerated, rather than silently
//! diverging.
//!
//! # Regenerating
//!
//! ```bash
//! REGENERATE_HTTP_WIRE_BINDINGS=1 cargo test -p openasr-server --lib http_wire_bindings_test
//! ```
//!
//! writes straight into `crates/openasr-server/generated/` -- both the new
//! `http-wire/` subtree and, per the note above, the crate-local
//! `realtime-wire/` copy of the shared capability leaf types (commit the
//! diff; both subtrees are compared below, so the duplicated copy is
//! genuinely golden-guarded here too, not just incidentally regenerated).
//! Without that env var, the test below regenerates into a temp directory
//! and diffs the whole `generated/` tree against the committed one, byte for
//! byte; any drift (a changed field, a renamed type, a type gaining/losing
//! the derive) fails the test with the file(s) that differ.
//!
//! # Scope: what is *not* generated
//!
//! `ErrorResponse`/`ErrorBody` (the OpenAI-compatible error envelope) and
//! every request-body struct (`ImportLocalModelRequest`, `StartPullRequest`,
//! `SetDefaultRequest`, ...) are out of scope for this PR: it targets the
//! read-only identity/discovery responses named in the tracking issue
//! (health, models, capabilities, devices). `LocalModelsResponse`,
//! `HistoryListResponse`, and the other `/v1/models/*` and `/v1/history`
//! response families are a natural follow-up with the same mechanism, not
//! bundled here to keep this change reviewable.
//!
//! Like the realtime wire test, every `#[serde(skip_serializing_if = ...)]`
//! field here without an accompanying `#[serde(default)]` renders as
//! `T | null` (or, for `LanguageCapability::fixed_languages`, a required
//! `string[]` even though the key can be entirely absent when empty) rather
//! than an optional TS key -- these types are `Serialize`-only and never
//! constructed from a TS object literal, so this is a conservative
//! over-approximation, not a functional gap.

use std::fs;
use std::path::{Path, PathBuf};

use ts_rs::{Config, TS};

use crate::{CapabilitiesResponse, DevicesResponse, HealthResponse, ModelResponse, ModelsResponse};

const COMMITTED_RELATIVE_DIR: &str = "generated";
const REGENERATE_ENV_VAR: &str = "REGENERATE_HTTP_WIRE_BINDINGS";

macro_rules! export_all_or_panic {
    ($cfg:expr, $($ty:ty),+ $(,)?) => {
        $(
            <$ty as TS>::export_all($cfg).unwrap_or_else(|error| {
                panic!(
                    "ts-rs export_all failed for {}: {error}",
                    stringify!($ty)
                )
            });
        )+
    };
}

fn export_http_wire_bindings(cfg: &Config) {
    export_all_or_panic!(
        cfg,
        HealthResponse,
        ModelsResponse,
        ModelResponse,
        CapabilitiesResponse,
        DevicesResponse,
    );
}

#[test]
fn http_wire_bindings_regenerate_to_match_committed() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    if std::env::var_os(REGENERATE_ENV_VAR).is_some() {
        let cfg = Config::new()
            .with_out_dir(manifest_dir)
            .with_large_int("number");
        export_http_wire_bindings(&cfg);
        return;
    }

    let committed_dir = manifest_dir.join(COMMITTED_RELATIVE_DIR);
    let scratch = tempfile::tempdir().expect("failed to create scratch dir for ts-rs regen");
    let cfg = Config::new()
        .with_out_dir(scratch.path())
        .with_large_int("number");
    export_http_wire_bindings(&cfg);

    let regenerated_dir = scratch.path().join(COMMITTED_RELATIVE_DIR);
    assert_directory_trees_match(&regenerated_dir, &committed_dir);
}

/// Compares two directory trees of generated `.ts` files byte-for-byte,
/// failing with the specific files that differ (missing, extra, or changed)
/// instead of a generic "not equal".
fn assert_directory_trees_match(regenerated_dir: &Path, committed_dir: &Path) {
    let regenerated_files = list_files_relative(regenerated_dir);
    let committed_files = list_files_relative(committed_dir);

    if !committed_dir.is_dir() {
        panic!(
            "Committed HTTP wire bindings directory is missing: {}\n\
             Run `REGENERATE_HTTP_WIRE_BINDINGS=1 cargo test -p openasr-server --lib \
             http_wire_bindings_test` and commit the result.",
            committed_dir.display()
        );
    }

    let missing_from_committed: Vec<_> = regenerated_files
        .iter()
        .filter(|file| !committed_files.contains(*file))
        .collect();
    let stale_in_committed: Vec<_> = committed_files
        .iter()
        .filter(|file| !regenerated_files.contains(*file))
        .collect();

    assert!(
        missing_from_committed.is_empty() && stale_in_committed.is_empty(),
        "HTTP wire bindings drifted from committed golden files.\n\
         Regenerated-but-not-committed: {missing_from_committed:?}\n\
         Committed-but-no-longer-generated (stale): {stale_in_committed:?}\n\
         Run `REGENERATE_HTTP_WIRE_BINDINGS=1 cargo test -p openasr-server --lib \
         http_wire_bindings_test` and commit the diff under {}.",
        committed_dir.display()
    );

    let mut mismatched = Vec::new();
    for relative_path in &regenerated_files {
        let regenerated_content = fs::read_to_string(regenerated_dir.join(relative_path))
            .unwrap_or_else(|error| panic!("failed to read {relative_path}: {error}"));
        let committed_content = fs::read_to_string(committed_dir.join(relative_path))
            .unwrap_or_else(|error| panic!("failed to read {relative_path}: {error}"));
        if regenerated_content != committed_content {
            mismatched.push(relative_path.clone());
        }
    }

    assert!(
        mismatched.is_empty(),
        "HTTP wire bindings drifted from committed golden files (content mismatch): \
         {mismatched:?}\n\
         Run `REGENERATE_HTTP_WIRE_BINDINGS=1 cargo test -p openasr-server --lib \
         http_wire_bindings_test` and commit the diff under {}.",
        committed_dir.display()
    );
}

fn list_files_relative(dir: &Path) -> std::collections::BTreeSet<String> {
    let mut files = std::collections::BTreeSet::new();
    collect_files_relative(dir, dir, &mut files);
    files
}

fn collect_files_relative(root: &Path, dir: &Path, out: &mut std::collections::BTreeSet<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let entry = entry.expect("failed to read directory entry");
        let path = entry.path();
        if path.is_dir() {
            collect_files_relative(root, &path, out);
        } else {
            let relative: PathBuf = path
                .strip_prefix(root)
                .expect("walked path must be under root")
                .to_path_buf();
            out.insert(relative.to_string_lossy().replace('\\', "/"));
        }
    }
}
