//! Model-agnostic integration gates for runtime-cost invariants that used
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
//! - **K2 (resident reuse):** [`k2_every_ggml_executor_family_is_registered`]
//!   and [`k2_registered_families_reference_a_resident_cache`] derive the
//!   expected family set from required architecture facets. Prepared-runtime
//!   ownership and reuse are universal execution-module invariants, not
//!   family-selectable claims. A dedicated ggml-executor directory
//!   (`models/<module_slug>/executor.rs` or `ggml_executor.rs`) must have a
//!   descriptor row; there is no hand-maintained classification table or
//!   exemption path. Every derived family must reference a resident
//!   runtime-cache primitive in its own module (so a per-request `Runtime::new()`
//!   rebuild has somewhere to be cached). The byte-identity of a cache HIT vs a
//!   fresh build is proved per family by that family's own dev-pack e2e test,
//!   which this static gate backstops.
//!
//! - **K3 (physical-lane identity):**
//!   [`k3_registered_families_reference_physical_execution_lane_identity`]
//!   requires every inventory-derived resident family to derive native-owner keys through
//!   `ExecutionLaneKey`, not the historical coarse CPU-vs-GPU enum. The lane
//!   identity includes provider, physical device, placement and graph backend,
//!   so a runtime built on one card/candidate cannot be handed to another.
//!
//! - **K4 (owner-bound lifetime):** process-resident state must use an admitted
//!   host owner, prepared-runtime owner, or dedicated pinned actor. Family-local
//!   `unsafe impl Send` wrappers and retired thread-affine checkout shapes are
//!   forbidden.

#![cfg(test)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::arch::OpenAsrArchitectureRegistry;
use crate::models::family_source_gates::ProductionSyntax;

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

/// Source-token signatures of an owner-bound resident-runtime primitive. This
/// is the K2 "there is an admitted owner for reused state" signal; a raw map,
/// TLS slot, family-global weight pool, or take/store helper is intentionally
/// not sufficient.
const RESIDENT_CACHE_PRIMITIVES: &[&str] = &[
    "AdmittedPinnedRuntimeActorCheckoutPool",
    "AdmittedPinnedRuntimeActorPool",
    "AdmittedExclusiveObjectPool",
    "AdmittedHostObjectCache",
    "PreparedRuntimeCache",
    "runtime_prepared_registry",
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
        let syntax = ProductionSyntax::collect(file);
        if HOST_F32_LOADER_CALLS
            .iter()
            .any(|call| syntax.calls_or_invokes_method(call))
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

/// Derives the resident-executor family set from the canonical architecture
/// inventory. The physical Rust directory is an explicit identity facet
/// (`module_slug`); the conformance profile remains the public/audit name.
/// Ownership, content-id eviction and graph reuse are supplied by the shared
/// execution module and therefore are not repeated as self-certified family
/// fields.
fn registered_ggml_executor_families() -> BTreeSet<String> {
    let registry = OpenAsrArchitectureRegistry::with_builtins();
    registry
        .validate_references()
        .unwrap_or_else(|error| panic!("canonical architecture inventory is invalid: {error:?}"));
    registry
        .descriptors()
        .iter()
        .map(|descriptor| descriptor.identity.module_slug.to_string())
        .collect()
}

/// K2 completeness lock: the inventory-derived family set must equal the
/// on-disk set of dedicated ggml-executor families. A new executor file with no
/// descriptor row (or a removed family with a stale descriptor) fails here.
#[test]
fn k2_every_ggml_executor_family_is_registered() {
    let models_dir = models_dir();
    let on_disk = on_disk_ggml_executor_families(&models_dir);
    let registered = registered_ggml_executor_families();

    let unregistered: Vec<_> = on_disk.difference(&registered).cloned().collect();
    let stale: Vec<_> = registered.difference(&on_disk).cloned().collect();

    assert!(
        unregistered.is_empty(),
        "K2 resident-runtime gate: these families have a dedicated ggml executor but \
         no canonical architecture descriptor/module_slug: {unregistered:?}"
    );
    assert!(
        stale.is_empty(),
        "K2 resident-runtime gate: these registered module_slugs have no dedicated \
         ggml executor on disk: {stale:?}"
    );
}

/// K2 structural check: every inventory-derived family must reference a
/// resident runtime-cache primitive somewhere in its module directory, so the
/// per-request runtime it builds is actually kept resident and reused.
#[test]
fn k2_registered_families_reference_a_resident_cache() {
    let models_dir = models_dir();
    for family in registered_ggml_executor_families() {
        let family_dir = models_dir.join(&family);
        let mut rs_files = Vec::new();
        collect_rs_files(&family_dir, &mut rs_files);
        let references_cache = rs_files.iter().any(|file| {
            let syntax = ProductionSyntax::collect(file);
            RESIDENT_CACHE_PRIMITIVES
                .iter()
                .any(|primitive| syntax.references_identifier(primitive))
        });
        assert!(
            references_cache,
            "K2 resident-runtime gate: family '{family}' has no file under models/{family}/ \
             referencing a resident runtime-cache primitive ({RESIDENT_CACHE_PRIMITIVES:?})"
        );
    }
}

/// K3: a resident native owner is valid only on the exact execution lane that
/// built it. The source token check complements the Rust key type itself:
/// adding a family to the inventory also requires its family module to derive
/// cache keys through the central lane resolver.
#[test]
fn k3_registered_families_reference_physical_execution_lane_identity() {
    let models_dir = models_dir();
    for family in registered_ggml_executor_families() {
        let family_dir = models_dir.join(&family);
        let mut rs_files = Vec::new();
        collect_rs_files(&family_dir, &mut rs_files);
        let references_lane_key = rs_files
            .iter()
            .any(|file| ProductionSyntax::collect(file).references_identifier("ExecutionLaneKey"));
        let derives_lane_key = rs_files.iter().any(|file| {
            ProductionSyntax::collect(file).calls_or_invokes_method("current_execution_lane_key")
        });
        assert!(
            references_lane_key && derives_lane_key,
            "K3 execution-lane gate: resident family '{family}' does not derive its \
             backend-owner cache identity through ExecutionLaneKey/current_execution_lane_key. \
             A coarse GgmlCpuGraphBackend key aliases providers and physical cards."
        );
    }
}

#[test]
fn k4_family_modules_do_not_bypass_owner_bound_runtime_primitives() {
    let models_dir = models_dir();
    let mut rs_files = Vec::new();
    collect_rs_files(&models_dir, &mut rs_files);
    let forbidden_symbols = [
        "checkout_thread_affine_admitted_object",
        "ThreadAffineAdmittedObjectCache",
        "take_generation_tagged",
        "with_thread_local_cached_mut_by_key",
        "UnloadGenerationGated",
        "BoundedRuntimeCache",
        "DOLPHIN_WEIGHTS_POOL",
    ];
    let mut violations = Vec::new();
    for file in rs_files {
        let relative = models_relative(&models_dir, &file);
        if matches!(
            relative.as_str(),
            "resident_runtime_audit.rs" | "admitted_thread_affine_object_cache.rs"
        ) {
            continue;
        }
        let syntax = ProductionSyntax::collect(&file);
        for symbol in forbidden_symbols {
            if syntax.references_identifier(symbol) || syntax.calls_or_invokes_method(symbol) {
                violations.push(format!("{relative}: {symbol}"));
            }
        }
        for trait_name in ["Send", "Sync"] {
            if syntax.has_unsafe_impl_for(trait_name) {
                violations.push(format!("{relative}: unsafe impl {trait_name}"));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "K4 owner-bound lifetime gate: family code bypasses admitted owner/pinned actor primitives: {violations:?}"
    );
}

#[test]
fn k4_persistent_auxiliary_families_reference_their_declared_owner_shape() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    for (relative, required) in [
        (
            "diarize/embed/policy_runtime.rs",
            "AuxiliaryRuntimeCacheKey",
        ),
        (
            "diarize/segment/policy_runtime.rs",
            "AuxiliaryRuntimeCacheKey",
        ),
        ("models/hymt2/policy_runtime.rs", "PinnedRuntimeActor"),
        (
            "models/firered_punc/policy_runtime.rs",
            "PinnedRuntimeActor",
        ),
    ] {
        let syntax = ProductionSyntax::collect(&root.join(relative));
        assert!(
            syntax.references_identifier(required),
            "K4 auxiliary ownership gate: {relative} does not reference declared owner primitive {required}"
        );
    }
}
