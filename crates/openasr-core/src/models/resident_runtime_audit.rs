//! Model-agnostic integration gates for two runtime-cost invariants that used
//! to be enforced only by prose in `docs/MODEL_ONBOARDING.md` plus a human's
//! eyes on the published model card -- which is exactly how families slipped
//! through rebuilding their whole runtime per request (mimo-asr) or (in earlier
//! incidents) dequantizing weights to host f32.
//!
//! These are **source-tree audits** (tests only; nothing here ships in a
//! release binary), the same lever `family_integration_audit`'s test-only
//! checks use to lock an SSOT list against the on-disk tree. They read the
//! repository source under `CARGO_MANIFEST_DIR`, so they run in CI with no
//! model packs and no inference, and they fail closed the moment a NEW family's
//! source is added without meeting the invariant -- catching the next family at
//! integration time instead of after publishing.
//!
//! - **K1 (keep quantized):** [`k1_host_f32_loader_sites_match_inventory`] locks
//!   the set of source files that materialize a tensor to a host `Vec<f32>`
//!   (via the reader's `host_tensor_f32_copy*` helpers) against the committed
//!   inventory `docs/model-audits/host_f32_loader_sites.txt`. A new host-f32
//!   loader site turns CI red until it is added to the inventory -- which is the
//!   point of human review: the reviewer certifies it loads only tensors that
//!   legitimately stay f32/f16 (norms, biases, conv kernels, get_rows
//!   embeddings, positional tables), NOT a rank-2 `mul_mat` weight, which must
//!   bind natively (`weight_tensor_payload_by_name` + `new_matmul_weight_2d_typed`;
//!   see `dolphin::executor::insert_pool_tensor` classifying per tensor). This
//!   is the structural complement to the pack-header quant floor
//!   (`pack_quant_audit`) and the model card's RAM-ordering self-check.
//!
//! - **K2 (resident reuse):** [`k2_every_ggml_executor_family_is_classified`]
//!   and [`k2_resident_families_reference_a_resident_cache`] require every
//!   dedicated ggml-executor family (a `models/<family>/executor.rs` or
//!   `ggml_executor.rs`) to be classified in [`GGML_EXECUTOR_FAMILY_GATES`] and,
//!   unless explicitly exempt, to reference a resident runtime-cache primitive
//!   in its own module (so a per-request `Runtime::new()` rebuild has somewhere
//!   to be cached). A new family's executor file added without a table entry
//!   fails the completeness lock; classified as resident without a cache
//!   primitive fails the structural check. The byte-identity of a cache HIT vs a
//!   fresh build is proved per family by that family's own dev-pack e2e test
//!   (e.g. mimo-asr's / firered-llm's
//!   `resident_*_cache_reuse_across_consecutive_calls_stays_byte_identical`),
//!   which this static gate backstops.

#![cfg(test)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The committed K1 inventory (see the file's own header for the contract).
const HOST_F32_LOADER_SITES_INVENTORY: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/model-audits/host_f32_loader_sites.txt"
));

/// Reader helpers that materialize a tensor into a host `Vec<f32>`. Any source
/// file calling one of these is a "host-f32 loader site" for K1.
const HOST_F32_LOADER_CALLS: &[&str] = &[
    "host_tensor_f32_copy_dequantized_by_name",
    "host_tensor_f32_copy_by_name",
    "host_tensor_f32_copy_by_id",
];

/// Source-token signatures of a resident runtime-cache primitive. A family that
/// reuses a prepared runtime across requests references at least one of these
/// (a thread-local generation-tagged cache, the shared prepared-runtime cache,
/// or a content-id-keyed process weights pool such as dolphin's). This is the
/// K2 "there is somewhere the per-request runtime is kept resident" signal.
const RESIDENT_CACHE_PRIMITIVES: &[&str] = &[
    "thread_local!",
    "take_generation_tagged",
    "with_thread_local_cached_mut_by_key",
    "PreparedRuntimeCache",
    "take_cached_",
    "store_cached_",
    "DolphinRuntimeWeights",
    "weights_pool",
    "runtime_prepared_registry",
    "UnloadGenerationGated",
    "BoundedRuntimeCache",
];

/// Classification of one dedicated ggml-executor family for the K2 gate.
#[derive(Debug, Clone, Copy)]
enum ResidentClassification {
    /// The family reuses a resident runtime across requests; the structural
    /// check asserts it references a [`RESIDENT_CACHE_PRIMITIVES`] token.
    Resident,
    /// The family legitimately does not keep a resident runtime. The reason is
    /// recorded and must be non-empty; no family currently needs this, but the
    /// slot exists so a genuinely one-shot family can be admitted with review
    /// rather than by silently weakening the gate.
    #[allow(dead_code)]
    Exempt(&'static str),
}

/// Every dedicated ggml-executor family and its K2 classification. The key is
/// the `models/<family>/` directory name. [`k2_every_ggml_executor_family_is_classified`]
/// locks this set against the on-disk executor files, so a newly onboarded
/// family (e.g. a future granite / funasr executor merged on another branch)
/// cannot land without an explicit entry here.
const GGML_EXECUTOR_FAMILY_GATES: &[(&str, ResidentClassification)] = &[
    ("cohere", ResidentClassification::Resident),
    ("dolphin", ResidentClassification::Resident),
    ("firered_aed", ResidentClassification::Resident),
    ("firered_llm", ResidentClassification::Resident),
    ("funasr_nano", ResidentClassification::Resident),
    ("granite_speech", ResidentClassification::Resident),
    ("mimo_asr", ResidentClassification::Resident),
    ("moonshine", ResidentClassification::Resident),
    ("moss_transcribe_diarize", ResidentClassification::Resident),
    ("parakeet_ctc", ResidentClassification::Resident),
    ("parakeet_tdt", ResidentClassification::Resident),
    ("qwen", ResidentClassification::Resident),
    ("sensevoice", ResidentClassification::Resident),
    ("wav2vec2_ctc", ResidentClassification::Resident),
    ("whisper", ResidentClassification::Resident),
    ("xasr_zipformer", ResidentClassification::Resident),
];

/// The dedicated-executor file names that mark a `models/<family>/` directory
/// as a ggml-executor family.
const GGML_EXECUTOR_FILE_NAMES: &[&str] = &["executor.rs", "ggml_executor.rs"];

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models")
}

/// Recursively collects every `.rs` file under `dir`.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("read_dir {}: {error}", dir.display()));
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Parses the inventory file into a set of `models/`-relative paths, ignoring
/// blank lines and `#` comments.
fn parse_inventory(inventory: &str) -> BTreeSet<String> {
    inventory
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// Path of `file` relative to the `models/` directory, using `/` separators.
fn models_relative(models_dir: &Path, file: &Path) -> String {
    file.strip_prefix(models_dir)
        .expect("file under models dir")
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

/// K1: the on-disk set of host-f32 loader sites must equal the committed
/// inventory. A new file that materializes a tensor to host f32 (the load-time
/// dequant pitfall for a bulk weight) turns this red until reviewed and added.
#[test]
fn k1_host_f32_loader_sites_match_inventory() {
    let models_dir = models_dir();
    let mut rs_files = Vec::new();
    collect_rs_files(&models_dir, &mut rs_files);

    let mut on_disk = BTreeSet::new();
    for file in &rs_files {
        let relative = models_relative(&models_dir, file);
        // This audit module names the loader helpers in string constants; it is
        // not itself a loader site.
        if relative == "resident_runtime_audit.rs" {
            continue;
        }
        let source = std::fs::read_to_string(file)
            .unwrap_or_else(|error| panic!("read {}: {error}", file.display()));
        if HOST_F32_LOADER_CALLS
            .iter()
            .any(|call| source.contains(call))
        {
            on_disk.insert(relative);
        }
    }

    let inventory = parse_inventory(HOST_F32_LOADER_SITES_INVENTORY);

    let unlisted: Vec<_> = on_disk.difference(&inventory).cloned().collect();
    let stale: Vec<_> = inventory.difference(&on_disk).cloned().collect();

    assert!(
        unlisted.is_empty(),
        "K1 keep-quantized gate: these source files materialize a tensor to host \
         f32 but are NOT in docs/model-audits/host_f32_loader_sites.txt. Add each \
         after certifying it loads only sanctioned f32/f16 tensors (norms, biases, \
         conv kernels, get_rows embeddings, positional tables) -- NEVER a rank-2 \
         mul_mat weight (bind those natively; see MODEL_ONBOARDING.md): {unlisted:?}"
    );
    assert!(
        stale.is_empty(),
        "K1 keep-quantized gate: these inventory entries no longer call a host-f32 \
         loader; remove the stale lines from docs/model-audits/host_f32_loader_sites.txt: \
         {stale:?}"
    );
}

/// Discovers the `models/<family>/` directories that carry a dedicated ggml
/// executor file.
fn on_disk_ggml_executor_families(models_dir: &Path) -> BTreeSet<String> {
    let mut families = BTreeSet::new();
    let entries = std::fs::read_dir(models_dir).expect("read models dir");
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let has_executor = GGML_EXECUTOR_FILE_NAMES
            .iter()
            .any(|name| path.join(name).is_file());
        if has_executor {
            families.insert(
                path.file_name()
                    .expect("family dir name")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    families
}

/// K2 completeness lock: the classified family set must equal the on-disk set
/// of dedicated ggml-executor families. A new executor file with no entry (or a
/// removed family with a stale entry) fails here, forcing a conscious
/// classification at integration time.
#[test]
fn k2_every_ggml_executor_family_is_classified() {
    let models_dir = models_dir();
    let on_disk = on_disk_ggml_executor_families(&models_dir);
    let classified: BTreeSet<String> = GGML_EXECUTOR_FAMILY_GATES
        .iter()
        .map(|(family, _)| family.to_string())
        .collect();

    let unclassified: Vec<_> = on_disk.difference(&classified).cloned().collect();
    let stale: Vec<_> = classified.difference(&on_disk).cloned().collect();

    assert!(
        unclassified.is_empty(),
        "K2 resident-runtime gate: these families have a dedicated ggml executor but \
         no entry in GGML_EXECUTOR_FAMILY_GATES. Add each as Resident (and wire a \
         resident runtime cache so it does not rebuild per request) or Exempt with a \
         reason: {unclassified:?}"
    );
    assert!(
        stale.is_empty(),
        "K2 resident-runtime gate: these classified families no longer have a \
         dedicated ggml executor on disk; remove the stale entries: {stale:?}"
    );
}

/// K2 structural check: every family classified `Resident` must reference a
/// resident runtime-cache primitive somewhere in its module directory, so the
/// per-request runtime it builds is actually kept resident and reused. `Exempt`
/// families must carry a non-empty reason.
#[test]
fn k2_resident_families_reference_a_resident_cache() {
    let models_dir = models_dir();
    for (family, classification) in GGML_EXECUTOR_FAMILY_GATES {
        match classification {
            ResidentClassification::Exempt(reason) => {
                assert!(
                    !reason.trim().is_empty(),
                    "K2 resident-runtime gate: family '{family}' is Exempt but its reason is empty"
                );
            }
            ResidentClassification::Resident => {
                let family_dir = models_dir.join(family);
                let mut rs_files = Vec::new();
                collect_rs_files(&family_dir, &mut rs_files);
                let references_cache = rs_files.iter().any(|file| {
                    let source = std::fs::read_to_string(file).unwrap_or_default();
                    RESIDENT_CACHE_PRIMITIVES
                        .iter()
                        .any(|primitive| source.contains(primitive))
                });
                assert!(
                    references_cache,
                    "K2 resident-runtime gate: family '{family}' is classified Resident but no \
                     file under models/{family}/ references a resident runtime-cache primitive \
                     ({RESIDENT_CACHE_PRIMITIVES:?}). Either wire a resident cache (the runtime \
                     is otherwise rebuilt on every request -- see firered_llm / mimo_asr) or \
                     reclassify it Exempt with a reason."
                );
            }
        }
    }
}
