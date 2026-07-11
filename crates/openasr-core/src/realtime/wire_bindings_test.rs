//! Golden "regenerate == committed" guard for the realtime WebSocket wire
//! contract's generated TypeScript bindings.
//!
//! # Why ts-rs, and why this shape
//!
//! `crate::realtime::events` / `crate::realtime::backend` / `crate::realtime::audio`
//! (plus the shared `crate::api::backend::BackendFeatureCapability` /
//! `BackendCapabilityBehavior`) are the single source of truth for the
//! realtime `/v1/audio/realtime` WebSocket wire format. Before this test
//! existed, `apps/desktop/src/lib/realtimeClient.ts` hand-duplicated every one
//! of these shapes as TypeScript types with no compiler link back to the Rust
//! structs, so a field rename/add/remove on the Rust side was a silent
//! drift risk caught only by manual review (or not at all -- see the PR
//! description for the drift already found this way, e.g.
//! `RealtimeTranslationTombstone` growing no `translation_version` field
//! while the hand-written TS type carried one that could never be populated).
//!
//! ts-rs was chosen over schemars for this because the deliverable is
//! TypeScript *types*, not a JSON Schema a separate tool would still have to
//! turn into types; ts-rs derives directly from the same struct + serde
//! attributes already on these types (rename, skip_serializing_if, transparent,
//! rename_all), so it does not require re-describing the wire shape a second
//! time, and it stays a `[dev-dependencies]`-only derive (see the
//! `#[cfg_attr(test, derive(ts_rs::TS))]` gate on every exported type in
//! `events.rs` / `backend.rs` / `audio.rs` / `api/backend/mod.rs`) that never
//! becomes part of the shipped rlib.
//!
//! # Regenerating
//!
//! ```bash
//! REGENERATE_REALTIME_WIRE_BINDINGS=1 cargo test -p openasr-core --lib realtime::wire_bindings_test
//! ```
//!
//! writes straight into `crates/openasr-core/generated/realtime-wire/` (commit
//! the diff). Without that env var, the test below regenerates into a temp
//! directory and diffs it against the committed directory, byte for byte;
//! any drift (a changed field, a renamed type, a type gaining/losing the
//! derive) fails the test with the file(s) that differ.
//!
//! # Scope: what is *not* generated, and why
//!
//! `RealtimeEventEnvelope`, `RealtimeEvent`, and the per-family enums
//! (`RealtimeLifecycleEvent`, `RealtimeAudioInputEvent`, `RealtimeVadEvent`,
//! `RealtimeTranscriptEvent`, `RealtimeTranslationEvent`) are deliberately
//! **not** derived here. All of them are `#[serde(untagged)]`, and the
//! envelope additionally `#[serde(flatten)]`s the chosen variant next to a
//! `type` field that is a plain `&'static str` computed by the hand-written
//! `RealtimeEvent::event_type()` method -- it is not a serde tag. ts-rs (like
//! schemars) can only describe what serde attributes actually declare; a
//! quick spike confirmed it renders this shape as
//! `{ type: string, session_id: string, ... } & (A | B | C | ...)`, i.e. an
//! *untagged* union with `type` widened to plain `string`, not narrowed per
//! variant. That is a faithful description of the wire bytes, but it throws
//! away the one thing a discriminated union is for: narrowing `event.type ===
//! "transcript.final"` to the matching payload shape. Forcing ts-rs to
//! produce that would mean generating something that *looks* like a
//! discriminated union but silently isn't one -- worse than the hand-written
//! status quo. So the envelope/dispatch shape stays hand-modeled in
//! `apps/desktop/src/lib/realtimeClient.ts` (a real discriminated union keyed
//! on the literal `type` strings this module's `event_type()` methods
//! produce), while every *leaf payload* type it references is one of the
//! generated bindings below -- narrowing the drift surface to "did the
//! payload shape change", which is where the actual drift risk lives, and
//! leaving "does a new leaf type need wiring into the envelope union" as an
//! explicit, reviewable follow-up rather than something codegen could get
//! wrong silently.
//!
//! `RealtimeAudioFrame` (binary PCM frame, not JSON) and `RealtimeErrorCode`'s
//! consumers already have their own contract tests and are out of scope here
//! by construction (the former never serializes as text/JSON at all).
//!
//! # A modeling nuance worth knowing: `Option<T>` renders as `T | null`, not `T?`
//!
//! Every `#[serde(skip_serializing_if = "Option::is_none")] pub field:
//! Option<T>` here (e.g. `language`, `speaker`, `RealtimeTranslationCapability::mode`)
//! generates `field: T | null` rather than an optional `field?: T`. ts-rs only
//! widens a maybe-omitted field to an optional TS key when it is *also*
//! `#[serde(default)]` (its serde-compat mirrors what makes an omitted key
//! round-trip safely through `Deserialize`); these wire types only derive
//! `Serialize` and are never constructed from a TS object literal, so this is
//! a conservative over-approximation, not a functional gap: the field really
//! is omitted from the JSON on the wire (never sent as literal `null`), and
//! every existing runtime reader (`readOptionalString` et al. in
//! `realtimeClient.ts`) already checks for `undefined`, which the `| null`
//! union does not exclude at the value level, so behavior is unaffected. It
//! only means a hand-constructed fixture literal of one of these types must
//! spell out `field: null` instead of omitting the key.

use std::fs;
use std::path::{Path, PathBuf};

use ts_rs::{Config, TS};

use crate::api::backend::{BackendCapabilityBehavior, BackendFeatureCapability};
use crate::realtime::audio::{RealtimeAudioEncoding, RealtimeAudioFormat};
use crate::realtime::backend::{
    RealtimeBackendCapabilities, RealtimeBackendMode, RealtimeTranslationCapability,
};
use crate::realtime::events::{
    AudioInputStartedEvent, AudioInputStoppedEvent, RealtimeErrorCode, RealtimeErrorEvent,
    RealtimeEventId, RealtimeSessionId, RealtimeTranscriptFinal, RealtimeTranscriptPartial,
    RealtimeTranscriptRevision, RealtimeTranscriptWord, RealtimeTranslationFinal,
    RealtimeTranslationPartial, RealtimeTranslationStatus, RealtimeTranslationTombstone,
    SessionCapabilitiesEvent, SessionClosedEvent, SessionConfiguredEvent, SessionCreatedEvent,
    SessionTranslationSummary, SessionVadSummary, TranscriptSegmentId, TranscriptUtteranceId,
    VadSpeechStartedEvent, VadSpeechStoppedEvent,
};

const COMMITTED_RELATIVE_DIR: &str = "generated/realtime-wire";
const REGENERATE_ENV_VAR: &str = "REGENERATE_REALTIME_WIRE_BINDINGS";

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

fn export_realtime_wire_bindings(cfg: &Config) {
    export_all_or_panic!(
        cfg,
        BackendCapabilityBehavior,
        BackendFeatureCapability,
        RealtimeAudioEncoding,
        RealtimeAudioFormat,
        RealtimeBackendMode,
        RealtimeBackendCapabilities,
        RealtimeTranslationCapability,
        RealtimeSessionId,
        RealtimeEventId,
        TranscriptUtteranceId,
        TranscriptSegmentId,
        SessionCreatedEvent,
        SessionCapabilitiesEvent,
        SessionConfiguredEvent,
        SessionTranslationSummary,
        SessionVadSummary,
        SessionClosedEvent,
        AudioInputStartedEvent,
        AudioInputStoppedEvent,
        VadSpeechStartedEvent,
        VadSpeechStoppedEvent,
        RealtimeTranscriptWord,
        RealtimeTranscriptPartial,
        RealtimeTranscriptFinal,
        RealtimeTranscriptRevision,
        RealtimeTranslationStatus,
        RealtimeTranslationPartial,
        RealtimeTranslationFinal,
        RealtimeTranslationTombstone,
        RealtimeErrorEvent,
        RealtimeErrorCode,
    );
}

#[test]
fn realtime_wire_bindings_regenerate_to_match_committed() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    if std::env::var_os(REGENERATE_ENV_VAR).is_some() {
        let cfg = Config::new()
            .with_out_dir(manifest_dir)
            .with_large_int("number");
        export_realtime_wire_bindings(&cfg);
        return;
    }

    let committed_dir = manifest_dir.join(COMMITTED_RELATIVE_DIR);
    let scratch = tempfile::tempdir().expect("failed to create scratch dir for ts-rs regen");
    let cfg = Config::new()
        .with_out_dir(scratch.path())
        .with_large_int("number");
    export_realtime_wire_bindings(&cfg);

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
            "Committed realtime wire bindings directory is missing: {}\n\
             Run `REGENERATE_REALTIME_WIRE_BINDINGS=1 cargo test -p openasr-core --lib \
             realtime::wire_bindings_test` and commit the result.",
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
        "Realtime wire bindings drifted from committed golden files.\n\
         Regenerated-but-not-committed: {missing_from_committed:?}\n\
         Committed-but-no-longer-generated (stale): {stale_in_committed:?}\n\
         Run `REGENERATE_REALTIME_WIRE_BINDINGS=1 cargo test -p openasr-core --lib \
         realtime::wire_bindings_test` and commit the diff under {}.",
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
        "Realtime wire bindings drifted from committed golden files (content mismatch): \
         {mismatched:?}\n\
         Run `REGENERATE_REALTIME_WIRE_BINDINGS=1 cargo test -p openasr-core --lib \
         realtime::wire_bindings_test` and commit the diff under {}.",
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
