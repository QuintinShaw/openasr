use std::{fs, path::Path, time::Duration};

use sha2::{Digest, Sha256};
use tempfile::TempDir;

use super::*;
use crate::InstalledPack;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

fn digest_of(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn models_dir_of(home: &Path) -> PathBuf {
    models_root(home).unwrap()
}

fn set_writable(path: &Path) {
    let mut permissions = fs::metadata(path).unwrap().permissions();
    #[cfg(unix)]
    permissions.set_mode(0o644);
    #[cfg(not(unix))]
    permissions.set_readonly(false);
    fs::set_permissions(path, permissions).unwrap();
}

fn seal(path: &Path) {
    let mut permissions = fs::metadata(path).unwrap().permissions();
    #[cfg(unix)]
    permissions.set_mode(0o444);
    #[cfg(not(unix))]
    permissions.set_readonly(true);
    fs::set_permissions(path, permissions).unwrap();
}

fn object_content_path(home: &Path, digest: &str) -> PathBuf {
    models_dir_of(home)
        .join("objects/sha256")
        .join(digest)
        .join("content")
}

/// Write an object exactly as `content_store` leaves one: sealed read-only,
/// under its own digest directory.
fn write_object(home: &Path, bytes: &[u8]) -> String {
    let digest = digest_of(bytes);
    let path = object_content_path(home, &digest);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, bytes).unwrap();
    seal(&path);
    digest
}

/// Backdate an object out of the orphan grace period, unsealing and resealing so
/// the fixture still matches production state.
fn age_object(home: &Path, digest: &str, age: Duration) {
    let path = object_content_path(home, digest);
    set_writable(&path);
    let when = SystemTime::now() - age;
    let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_times(fs::FileTimes::new().set_modified(when))
        .unwrap();
    drop(file);
    seal(&path);
}

fn write_ref(home: &Path, model_id: &str, quant: &str, digest: &str, size: u64) -> PathBuf {
    let root = models_dir_of(home);
    let ref_path = root
        .join("refs")
        .join(model_id)
        .join(format!("{quant}.json"));
    fs::create_dir_all(ref_path.parent().unwrap()).unwrap();
    let pack = InstalledPack {
        model_id: model_id.to_string(),
        display_name: model_id.to_string(),
        quant: quant.to_string(),
        suffix: "q8".to_string(),
        pull: format!("{model_id}:q8"),
        filename: format!("{model_id}-{quant}.oasr"),
        path: object_content_path(home, digest),
        url: "https://example.invalid/model.oasr".to_string(),
        hf_revision: "test".to_string(),
        sha256: digest.to_string(),
        size_bytes: size,
        installed_at_unix_seconds: 1,
        source: None,
    };
    fs::write(&ref_path, serde_json::to_string(&pack).unwrap()).unwrap();
    ref_path
}

fn staging_dir(home: &Path) -> PathBuf {
    let path = models_dir_of(home).join(STAGING_DIR_NAME);
    fs::create_dir_all(&path).unwrap();
    path
}

fn write_admit_staging(home: &Path, pid: u32, bytes: &[u8]) -> PathBuf {
    let path = staging_dir(home).join(format!("admit-{pid}-123456789.tmp"));
    fs::write(&path, bytes).unwrap();
    path
}

fn write_lock(home: &Path, name: &str, pid: u32) -> PathBuf {
    let path = staging_dir(home).join(format!("{name}.lock"));
    fs::write(&path, format!("pid={pid}\n")).unwrap();
    path
}

/// A pid that has certainly exited, found by probing rather than guessing.
fn dead_pid() -> u32 {
    (900_000..999_999)
        .find(|pid| crate::pull::process_is_gone(*pid))
        .expect("some high pid must be unused")
}

#[test]
fn unreferenced_object_past_grace_is_collected() {
    let home = TempDir::new().unwrap();
    let digest = write_object(home.path(), b"orphaned-pack-bytes");
    age_object(
        home.path(),
        &digest,
        ORPHAN_OBJECT_GRACE + Duration::from_secs(60),
    );

    let report = collect_model_store_garbage(home.path()).unwrap();
    assert_eq!(report.removed_objects, vec![digest.clone()]);
    assert_eq!(report.freed_bytes, b"orphaned-pack-bytes".len() as u64);
    assert!(report.collection_withheld.is_none());
    assert!(
        !models_dir_of(home.path())
            .join("objects/sha256")
            .join(&digest)
            .exists(),
        "the per-digest directory must go with the object"
    );
}

#[test]
fn unreferenced_object_inside_grace_is_kept() {
    let home = TempDir::new().unwrap();
    // Exactly the "content landed, ref not written yet" window that the
    // content-before-ref ordering guarantees will exist.
    let digest = write_object(home.path(), b"just-landed-bytes");

    let report = collect_model_store_garbage(home.path()).unwrap();
    assert!(report.removed_objects.is_empty());
    assert_eq!(report.retained_young_orphans, 1);
    assert!(object_content_path(home.path(), &digest).is_file());
}

#[test]
fn referenced_object_is_never_collected() {
    let home = TempDir::new().unwrap();
    let bytes = b"referenced-pack-bytes";
    let digest = write_object(home.path(), bytes);
    write_ref(
        home.path(),
        "xasr-zh-en",
        "q8_0",
        &digest,
        bytes.len() as u64,
    );
    age_object(home.path(), &digest, ORPHAN_OBJECT_GRACE * 10);

    let report = collect_model_store_garbage(home.path()).unwrap();
    assert!(report.is_empty());
    assert!(object_content_path(home.path(), &digest).is_file());
}

#[test]
fn object_named_only_by_the_default_pointer_is_kept() {
    let home = TempDir::new().unwrap();
    let bytes = b"default-pointer-bytes";
    let digest = write_object(home.path(), bytes);
    age_object(home.path(), &digest, ORPHAN_OBJECT_GRACE * 10);
    let pointer = serde_json::json!({
        "model_id": "xasr-zh-en",
        "quant": "q8_0",
        "suffix": "q8",
        "pull": "xasr-zh-en:q8",
        "path": object_content_path(home.path(), &digest),
        "sha256": digest,
        "size_bytes": bytes.len(),
        "updated_at_unix_seconds": 1,
    });
    fs::write(
        home.path().join("default.json"),
        serde_json::to_string(&pointer).unwrap(),
    )
    .unwrap();

    let report = collect_model_store_garbage(home.path()).unwrap();
    assert!(report.removed_objects.is_empty());
    assert!(object_content_path(home.path(), &digest).is_file());
}

#[test]
fn unreadable_ref_withholds_object_collection() {
    let home = TempDir::new().unwrap();
    let digest = write_object(home.path(), b"orphan-with-broken-neighbour");
    age_object(home.path(), &digest, ORPHAN_OBJECT_GRACE * 10);
    let broken = models_dir_of(home.path()).join("refs/broken-model/q8_0.json");
    fs::create_dir_all(broken.parent().unwrap()).unwrap();
    fs::write(&broken, b"{ this is not json").unwrap();

    let report = collect_model_store_garbage(home.path()).unwrap();
    assert!(
        report.removed_objects.is_empty(),
        "an incomplete root set must never authorize deletion"
    );
    let reason = report.collection_withheld.expect("withheld with a reason");
    assert!(reason.contains("root set"), "{reason}");
    assert!(object_content_path(home.path(), &digest).is_file());
}

#[test]
#[cfg(unix)]
fn unreadable_refs_root_withholds_object_collection() {
    let home = TempDir::new().unwrap();
    let live_digest = write_object(home.path(), b"a-pack-a-live-ref-points-at");
    age_object(home.path(), &live_digest, ORPHAN_OBJECT_GRACE * 10);
    write_ref(home.path(), "live-model", "q8_0", &live_digest, 27);

    // Make refs/ itself unreadable, leaving the ref file perfectly intact.
    // Listing the directory now fails outright, which used to be
    // indistinguishable from "no refs/ directory exists" and collected the
    // live object out from under an intact ref.
    let refs_root = models_dir_of(home.path()).join("refs");
    let mut permissions = fs::metadata(&refs_root).unwrap().permissions();
    permissions.set_mode(0o000);
    fs::set_permissions(&refs_root, permissions).unwrap();

    let report = collect_model_store_garbage(home.path());

    // Restore before asserting so the TempDir can clean up regardless of the
    // assertion outcome.
    let mut permissions = fs::metadata(&refs_root).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&refs_root, permissions).unwrap();

    let report = report.unwrap();
    assert!(
        report.collection_withheld.is_some(),
        "an unreadable refs root must withhold collection"
    );
    assert!(
        report.removed_objects.is_empty(),
        "collection deleted a referenced object because refs/ could not be listed"
    );
    assert!(object_content_path(home.path(), &live_digest).is_file());
}

#[test]
fn live_pull_lock_withholds_object_collection() {
    let home = TempDir::new().unwrap();
    let digest = write_object(home.path(), b"maybe-still-being-installed");
    age_object(home.path(), &digest, ORPHAN_OBJECT_GRACE * 10);
    write_lock(home.path(), "xasr-zh-en-q8_0", std::process::id());

    let report = collect_model_store_garbage(home.path()).unwrap();
    assert!(report.removed_objects.is_empty());
    let reason = report.collection_withheld.expect("withheld with a reason");
    assert!(reason.contains("pull lock"), "{reason}");
    assert!(object_content_path(home.path(), &digest).is_file());
}

#[test]
fn lock_from_an_exited_process_does_not_withhold_collection() {
    let home = TempDir::new().unwrap();
    let digest = write_object(home.path(), b"abandoned-install-bytes");
    age_object(home.path(), &digest, ORPHAN_OBJECT_GRACE * 10);
    write_lock(home.path(), "xasr-zh-en-q8_0", dead_pid());

    let report = collect_model_store_garbage(home.path()).unwrap();
    assert_eq!(report.removed_objects, vec![digest]);
    assert!(report.collection_withheld.is_none());
}

#[test]
fn staging_owned_by_a_dead_process_is_collected_without_grace() {
    let home = TempDir::new().unwrap();
    // The real leak: 600MB-per-entry transaction files from one retry loop whose
    // process exited minutes later. Freshness is irrelevant here -- nothing can
    // ever finish them.
    let fresh = write_admit_staging(home.path(), dead_pid(), &vec![7; 4096]);

    let report = collect_model_store_garbage(home.path()).unwrap();
    assert_eq!(report.removed_staging, vec![fresh.clone()]);
    assert_eq!(report.freed_bytes, 4096);
    assert!(!fresh.exists());
}

#[test]
fn staging_owned_by_a_live_process_is_kept() {
    let home = TempDir::new().unwrap();
    let mine = write_admit_staging(home.path(), std::process::id(), b"in-flight");

    let report = collect_model_store_garbage(home.path()).unwrap();
    assert!(report.removed_staging.is_empty());
    assert!(mine.is_file());
}

#[test]
fn resumable_download_partials_are_never_collected() {
    let home = TempDir::new().unwrap();
    // Download partials are shared, resumable state with no owning process.
    // Collecting them would silently discard a nearly-complete multi-GB
    // download, so they are out of scope by name.
    let stem = "a".repeat(64);
    let partial = staging_dir(home.path()).join(format!("{stem}-model.oasr.partial"));
    fs::write(&partial, vec![1; 2048]).unwrap();
    let meta = staging_dir(home.path()).join(format!("{stem}-model.oasr.partial.meta.json"));
    fs::write(&meta, b"{}").unwrap();

    let report = collect_model_store_garbage(home.path()).unwrap();
    assert!(report.removed_staging.is_empty());
    assert!(partial.is_file());
    assert!(meta.is_file());
}

#[test]
fn usage_accounts_for_models_orphans_dead_staging_and_reclaimable() {
    let home = TempDir::new().unwrap();
    let live_bytes = b"live-model-bytes-1234567890";
    let live = write_object(home.path(), live_bytes);
    write_ref(
        home.path(),
        "xasr-zh-en",
        "q8_0",
        &live,
        live_bytes.len() as u64,
    );
    let orphan_bytes = b"orphan-bytes";
    let orphan = write_object(home.path(), orphan_bytes);
    age_object(home.path(), &orphan, ORPHAN_OBJECT_GRACE * 2);
    let young = write_object(home.path(), b"young-orphan-bytes");
    write_admit_staging(home.path(), dead_pid(), &vec![3; 512]);

    let usage = model_store_usage(home.path()).unwrap();
    assert_eq!(usage.models_dir, models_dir_of(home.path()));
    assert_eq!(usage.entries.len(), 1);
    assert_eq!(usage.entries[0].pull, "xasr-zh-en:q8");
    assert_eq!(usage.entries[0].size_bytes, live_bytes.len() as u64);
    assert_eq!(usage.objects_count, 3);
    assert_eq!(usage.orphan_object_count, 2);
    assert_eq!(usage.dead_staging_count, 1);
    assert_eq!(usage.dead_staging_bytes, 512);
    // The young orphan is inside its grace window: visible as an orphan, but not
    // advertised as space the user can get back right now.
    assert_eq!(
        usage.reclaimable_bytes,
        orphan_bytes.len() as u64 + 512,
        "young orphan {young} must not count as reclaimable"
    );

    let report = collect_model_store_garbage(home.path()).unwrap();
    assert_eq!(
        report.freed_bytes, usage.reclaimable_bytes,
        "the advertised reclaimable total must be what collection actually frees"
    );
}

#[test]
fn usage_and_collection_follow_a_custom_models_dir() {
    let home = TempDir::new().unwrap();
    let elsewhere = TempDir::new().unwrap();
    // Exercises the persisted `config.models_dir` level of the env/config/default
    // priority. Everything must happen inside the directory the user chose, and
    // nothing may appear in the default `<home>/models`.
    fs::write(
        home.path().join("config.json"),
        serde_json::json!({ "models_dir": elsewhere.path() }).to_string(),
    )
    .unwrap();
    assert_eq!(models_dir_of(home.path()), elsewhere.path());

    let digest = write_object(home.path(), b"custom-dir-orphan");
    assert!(
        elsewhere
            .path()
            .join("objects/sha256")
            .join(&digest)
            .join("content")
            .is_file()
    );
    age_object(home.path(), &digest, ORPHAN_OBJECT_GRACE * 2);

    let usage = model_store_usage(home.path()).unwrap();
    assert_eq!(usage.models_dir, elsewhere.path());
    assert_eq!(usage.orphan_object_count, 1);

    let report = collect_model_store_garbage(home.path()).unwrap();
    assert_eq!(report.removed_objects, vec![digest]);
    assert!(
        !home.path().join("models").exists(),
        "the default root must never be created when storage is redirected"
    );
}

#[test]
fn verify_passes_on_an_intact_store_and_names_a_corrupt_object() {
    let home = TempDir::new().unwrap();
    let bytes = b"verified-pack-bytes";
    let digest = write_object(home.path(), bytes);
    write_ref(
        home.path(),
        "xasr-zh-en",
        "q8_0",
        &digest,
        bytes.len() as u64,
    );

    let verification = verify_model_store(home.path()).unwrap();
    assert!(verification.is_ok());
    assert_eq!(verification.checked.len(), 1);

    // Defeat the seal the way a hardware fault or an errant tool would have to.
    let object = object_content_path(home.path(), &digest);
    set_writable(&object);
    fs::write(&object, b"corrupted-pack-byte").unwrap();

    let verification = verify_model_store(home.path()).unwrap();
    assert!(!verification.is_ok());
    assert_eq!(verification.failures().count(), 1);
}

/// Verify is how a store that lost its seals earns the fast path back: an
/// intact object whose permission bits a backup restore stripped must be
/// re-sealed once it passes, while an object that fails verification is
/// left exactly as found -- never sealed on top of corrupt bytes.
#[test]
fn verify_reseals_intact_objects_but_not_failed_ones() {
    let home = TempDir::new().unwrap();
    let bytes = b"reseal-me-pack-bytes";
    let digest = write_object(home.path(), bytes);
    write_ref(
        home.path(),
        "xasr-zh-en",
        "q8_0",
        &digest,
        bytes.len() as u64,
    );

    // Bytes intact, seal lost.
    let object = object_content_path(home.path(), &digest);
    set_writable(&object);
    assert!(!fs::metadata(&object).unwrap().permissions().readonly());

    assert!(verify_model_store(home.path()).unwrap().is_ok());
    assert!(
        fs::metadata(&object).unwrap().permissions().readonly(),
        "an object that passes verification must come back sealed"
    );

    // Corrupt it and verify again: the failure must be reported and the
    // object must not be re-sealed.
    set_writable(&object);
    fs::write(&object, b"reseal-me-pack-byteX").unwrap();
    let verification = verify_model_store(home.path()).unwrap();
    assert_eq!(verification.failures().count(), 1);
    assert!(
        !fs::metadata(&object).unwrap().permissions().readonly(),
        "a failed object must not be sealed on top of corrupt bytes"
    );
}

#[test]
fn verify_reports_a_ref_whose_object_is_missing() {
    let home = TempDir::new().unwrap();
    let bytes = b"about-to-vanish";
    let digest = write_object(home.path(), bytes);
    write_ref(
        home.path(),
        "xasr-zh-en",
        "q8_0",
        &digest,
        bytes.len() as u64,
    );
    fs::remove_file(object_content_path(home.path(), &digest)).unwrap();

    let verification = verify_model_store(home.path()).unwrap();
    assert_eq!(verification.failures().count(), 1);
}

#[test]
fn collection_is_idempotent_on_a_steady_store() {
    let home = TempDir::new().unwrap();
    let bytes = b"steady-state-bytes";
    let digest = write_object(home.path(), bytes);
    write_ref(
        home.path(),
        "xasr-zh-en",
        "q8_0",
        &digest,
        bytes.len() as u64,
    );
    age_object(home.path(), &digest, ORPHAN_OBJECT_GRACE * 5);

    let first = collect_model_store_garbage(home.path()).unwrap();
    let second = collect_model_store_garbage(home.path()).unwrap();
    assert!(first.is_empty() && second.is_empty());
    assert_eq!(first.freed_bytes, 0);
    assert!(verify_model_store(home.path()).unwrap().is_ok());
}

// ---------------------------------------------------------------------------
// Relocating a store (desktop "change model storage location")
//
// The desktop moves every file under `<models>/` to a new root, rename-first
// with a copy+delete fallback, then persists the new `config.models_dir`. These
// reproduce that byte-for-byte against a sealed content store.
// ---------------------------------------------------------------------------

/// The desktop's per-file move, reproduced: rename first, else copy, verify the
/// length, and delete the original (`sidecar.rs::move_one_file`).
fn move_one_file_like_desktop(src: &Path, dst: &Path, force_copy: bool) -> Result<(), String> {
    if !force_copy && fs::rename(src, dst).is_ok() {
        return Ok(());
    }
    let src_len = fs::metadata(src).map_err(|e| e.to_string())?.len();
    fs::copy(src, dst).map_err(|e| format!("copy failed: {e}"))?;
    let dst_len = fs::metadata(dst).map_err(|e| e.to_string())?.len();
    if dst_len != src_len {
        return Err("truncated".to_string());
    }
    fs::remove_file(src).map_err(|e| format!("could not remove original: {e}"))?;
    Ok(())
}

fn relocate_store_like_desktop(
    source: &Path,
    destination: &Path,
    force_copy: bool,
) -> Result<(), String> {
    fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect(&path, out);
            } else {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    collect(source, &mut files);
    for src in files {
        let relative = src.strip_prefix(source).expect("under source");
        let dst = destination.join(relative);
        fs::create_dir_all(dst.parent().expect("dst parent")).map_err(|e| e.to_string())?;
        move_one_file_like_desktop(&src, &dst, force_copy)?;
    }
    Ok(())
}

/// Point the home at a relocated store, exactly as `daemon_set_models_dir` does
/// once every file has moved.
fn repoint_models_dir(home: &Path, new_root: &Path) {
    fs::write(
        home.join("config.json"),
        serde_json::json!({ "models_dir": new_root }).to_string(),
    )
    .unwrap();
}

#[test]
fn relocating_a_store_by_rename_keeps_every_pack_usable() {
    let home = TempDir::new().unwrap();
    let bytes = b"relocated-pack-bytes";
    let digest = write_object(home.path(), bytes);
    write_ref(
        home.path(),
        "xasr-zh-en",
        "q8_0",
        &digest,
        bytes.len() as u64,
    );
    assert_eq!(verify_model_store(home.path()).unwrap().checked.len(), 1);

    let destination = TempDir::new().unwrap();
    relocate_store_like_desktop(&models_dir_of(home.path()), destination.path(), false).unwrap();
    repoint_models_dir(home.path(), destination.path());

    let moved_object = destination
        .path()
        .join("objects/sha256")
        .join(&digest)
        .join("content");
    assert!(moved_object.is_file(), "the object must arrive");
    assert!(
        fs::metadata(&moved_object)
            .unwrap()
            .permissions()
            .readonly(),
        "a rename preserves the seal"
    );

    let verification = verify_model_store(home.path()).unwrap();
    assert!(
        verification.is_ok() && verification.checked.len() == 1,
        "packs must remain verifiable after relocation: {:?}",
        verification.checked
    );
    let usage = model_store_usage(home.path()).unwrap();
    assert_eq!(usage.entries.len(), 1, "relocated pack must stay listed");
    assert_eq!(
        usage.orphan_object_count, 0,
        "a relocated pack's object must still be seen as referenced, or GC would collect it"
    );
}

#[test]
fn relocating_a_store_by_copy_keeps_the_seal_and_every_pack_usable() {
    let home = TempDir::new().unwrap();
    let bytes = b"copied-across-volumes";
    let digest = write_object(home.path(), bytes);
    write_ref(
        home.path(),
        "xasr-zh-en",
        "q8_0",
        &digest,
        bytes.len() as u64,
    );

    let destination = TempDir::new().unwrap();
    // `force_copy` reproduces the cross-volume fallback on a single test volume:
    // copy, verify length, unlink the sealed original.
    relocate_store_like_desktop(&models_dir_of(home.path()), destination.path(), true).unwrap();
    repoint_models_dir(home.path(), destination.path());

    let moved_object = destination
        .path()
        .join("objects/sha256")
        .join(&digest)
        .join("content");
    assert!(moved_object.is_file());
    assert_eq!(fs::read(&moved_object).unwrap(), bytes);
    assert!(
        fs::metadata(&moved_object)
            .unwrap()
            .permissions()
            .readonly(),
        "the copy fallback must not silently drop the read-only seal"
    );

    let verification = verify_model_store(home.path()).unwrap();
    assert!(
        verification.is_ok() && verification.checked.len() == 1,
        "packs must remain verifiable after a copying relocation: {:?}",
        verification.checked
    );
    assert_eq!(model_store_usage(home.path()).unwrap().entries.len(), 1);
}
#[test]
fn relocating_a_store_keeps_every_pack_listed() {
    // The desktop's "change model storage location" moves files verbatim and
    // then repoints `config.models_dir`; it never rewrites refs. A reader that
    // trusted a ref's recorded absolute path would report zero installed models
    // afterwards -- the packs are all still there, just described by their old
    // location.
    let home = TempDir::new().unwrap();
    let bytes = b"relocated-pack-bytes";
    let digest = write_object(home.path(), bytes);
    write_ref(
        home.path(),
        "xasr-zh-en",
        "q8_0",
        &digest,
        bytes.len() as u64,
    );
    assert_eq!(crate::list_installed_packs(home.path()).unwrap().len(), 1);

    let destination = TempDir::new().unwrap();
    relocate_store_like_desktop(&models_dir_of(home.path()), destination.path(), false).unwrap();
    repoint_models_dir(home.path(), destination.path());

    let store = crate::InstalledModelStore::read(home.path()).unwrap();
    assert!(
        store.diagnostics().is_empty(),
        "relocation is not a fault: {:?}",
        store.diagnostics()
    );
    assert_eq!(store.packs().len(), 1, "the pack must survive relocation");
    // The pack is reported at its real location, not the one baked into the ref.
    assert_eq!(
        store.packs()[0].path,
        destination
            .path()
            .join("objects/sha256")
            .join(&digest)
            .join("content")
    );
    assert!(fs::read(&store.packs()[0].path).unwrap() == bytes);

    // And the pack is still resolvable and removable by reference, which is what
    // the desktop model list and its delete button need.
    assert!(
        crate::pull::resolve_installed_pack_path(home.path(), "xasr-zh-en:q8")
            .unwrap()
            .is_some()
    );
}

#[test]
fn a_ref_naming_a_foreign_file_is_never_followed() {
    // The recorded path is not an input to resolution, so a crafted ref cannot
    // point a reader at bytes outside the object store. It resolves to its own
    // digest's object or it does not resolve at all.
    let home = TempDir::new().unwrap();
    let bytes = b"honest-object-bytes";
    let digest = write_object(home.path(), bytes);
    let ref_path = write_ref(
        home.path(),
        "xasr-zh-en",
        "q8_0",
        &digest,
        bytes.len() as u64,
    );
    let outside = home.path().join("outside.oasr");
    fs::write(&outside, b"attacker-controlled").unwrap();
    let mut record: serde_json::Value =
        serde_json::from_slice(&fs::read(&ref_path).unwrap()).unwrap();
    record["path"] = serde_json::Value::String(outside.display().to_string());
    fs::write(&ref_path, serde_json::to_string(&record).unwrap()).unwrap();

    let store = crate::InstalledModelStore::read(home.path()).unwrap();
    assert_eq!(store.packs().len(), 1);
    assert_eq!(
        store.packs()[0].path,
        object_content_path(home.path(), &digest)
    );
    assert_ne!(store.packs()[0].path, outside);
    assert_eq!(fs::read(&store.packs()[0].path).unwrap(), bytes);
}

/// A `config.json` that fails to parse must not be treated as "no config,
/// use the default `<home>/models`" -- these commands are destructive (GC)
/// or security-relevant (verify), and silently acting against the wrong
/// directory because the real one could not be resolved is worse than
/// refusing to run.
#[test]
fn corrupt_config_fails_gc_closed_instead_of_falling_back_to_default_dir() {
    let home = TempDir::new().unwrap();
    fs::write(crate::config::config_path(home.path()), b"{ not json").unwrap();

    let error = collect_model_store_garbage(home.path()).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("config.json") || message.contains("config"),
        "{message}"
    );

    let error = model_store_usage(home.path()).unwrap_err();
    assert!(!error.to_string().is_empty());

    let error = verify_model_store(home.path()).unwrap_err();
    assert!(!error.to_string().is_empty());
}
