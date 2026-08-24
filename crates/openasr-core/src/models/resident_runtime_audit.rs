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
//!
//! - **Resident footprint:** [`resident_footprint_inventory_is_complete`] locks every
//!   live descriptor to a validated, backend-neutral topology with both binding
//!   alternatives, explicit split/unified variants, and bounded checkout/session
//!   limits. Construction/publication contracts are tested below.

#![cfg(test)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::arch::OpenAsrArchitectureRegistry;
use crate::arch::runtime_footprint::{ResidentPlacementVariant, ResidentRepresentation};
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

/// Construction/publication surfaces are explicit so adding a new owner path
/// requires an inventory row and a reviewable symbol, rather than merely adding a
/// word to a denylist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ResidentSurface {
    Models,
    GgmlRuntime,
    Auxiliary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidentSiteStatus {
    Active,
}

const RESIDENT_CONSTRUCTION_PUBLICATION_INVENTORY: &[(
    ResidentSurface,
    &str,
    &str,
    ResidentSiteStatus,
)] = &[
    (
        ResidentSurface::Models,
        "models/admitted_exclusive_object_pool.rs",
        "AdmittedExclusiveObjectPool",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::Models,
        "models/admitted_host_object_cache.rs",
        "AdmittedHostObjectCache",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::Models,
        "models/admitted_pinned_runtime_actor_pool.rs",
        "AdmittedPinnedRuntimeActorCheckoutPool",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::Models,
        "models/admitted_pinned_runtime_actor_pool.rs",
        "AdmittedPinnedRuntimeActorPool",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::Models,
        "models/admitted_pinned_runtime_actor_pool.rs",
        "PinnedRuntimeActorCheckout",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::Models,
        "models/prepared_runtime_cache.rs",
        "PreparedRuntimeCache",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::Models,
        "models/runtime_prepared_registry.rs",
        "BuiltinPreparedRuntimeCache",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::GgmlRuntime,
        "ggml_runtime/backend_memory.rs",
        "BackendMemoryAbi",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::GgmlRuntime,
        "ggml_runtime/backend_memory_admission.rs",
        "NativeMemoryAllocationTransaction",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::GgmlRuntime,
        "ggml_runtime/cpu_graph.rs",
        "GgmlLoadedWeightContext",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::GgmlRuntime,
        "ggml_runtime/cpu_graph.rs",
        "LoadedWeightOwnerCache",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::GgmlRuntime,
        "ggml_runtime/cpu_graph.rs",
        "LOADED_WEIGHT_OWNER_SLOTS",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::GgmlRuntime,
        "ggml_runtime/cpu_graph.rs",
        "THREAD_BACKEND_CACHE_BY_KIND",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::GgmlRuntime,
        "ggml_runtime/cpu_graph.rs",
        "GgmlCpuStepBufferPool",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::Auxiliary,
        "models/policy_resolved_aux_runtime.rs",
        "AuxiliaryRuntimeOwnerCache",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::Auxiliary,
        "models/policy_resolved_aux_runtime.rs",
        "AuxiliaryPinnedRuntimeCacheKey",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::Auxiliary,
        "models/policy_resolved_aux_runtime.rs",
        "AuxiliaryRuntimeCacheKey",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::Auxiliary,
        "diarize/embed/policy_runtime.rs",
        "AuxiliaryRuntimeCacheKey",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::Auxiliary,
        "diarize/segment/policy_runtime.rs",
        "AuxiliaryRuntimeCacheKey",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::Auxiliary,
        "diarize/vad/firered_stream/realtime_runtime.rs",
        "PinnedRuntimeActorCheckout",
        ResidentSiteStatus::Active,
    ),
    (
        ResidentSurface::Auxiliary,
        "models/seq2seq_serve_batch.rs",
        "ServeBatchOwner",
        ResidentSiteStatus::Active,
    ),
];

/// Owner-layer lease construction seams. Broker `try_reserve_batch` is not
/// listed: families must enter through these functions so receipts attach.
const LEASE_CONSTRUCTION_METHODS: &[&str] = &[
    "try_allocate",
    "try_allocate_transaction",
    "try_reserve_invocation",
    "attach_receipt",
];

const LEASE_CONSTRUCTION_INVENTORY: &[(&str, &str)] = &[
    ("diarize/embed/policy_runtime.rs", "try_allocate"),
    ("diarize/external.rs", "try_reserve_invocation"),
    (
        "diarize/segment/diarizen/runtime.rs",
        "try_allocate_transaction",
    ),
    (
        "diarize/segment/policy_runtime.rs",
        "try_allocate_transaction",
    ),
    (
        "diarize/vad/firered_stream/realtime_runtime.rs",
        "try_allocate_transaction",
    ),
    ("diarize/vad/firered_stream/mod.rs", "try_allocate"),
    ("diarize/vad/firered_stream/streaming.rs", "try_allocate"),
    ("ggml_runtime/backend_memory_admission.rs", "attach_receipt"),
    ("ggml_runtime/cpu_graph.rs", "attach_receipt"),
    ("ggml_runtime/cpu_graph.rs", "try_reserve_invocation"),
    ("device/pack_weight_residency.rs", "attach_receipt"),
    ("models/native_execution_services.rs", "attach_receipt"),
    ("models/cohere/ggml_executor.rs", "try_allocate_transaction"),
    ("models/dolphin/executor.rs", "try_allocate_transaction"),
    ("models/firered_aed/executor.rs", "try_allocate_transaction"),
    ("models/firered_llm/executor.rs", "try_allocate_transaction"),
    ("models/firered_punc/runtime.rs", "try_allocate_transaction"),
    ("models/funasr_nano/executor.rs", "try_allocate_transaction"),
    ("models/granite_speech/decode_session.rs", "try_allocate"),
    (
        "models/granite_speech/executor.rs",
        "try_allocate_transaction",
    ),
    ("models/lora_adapter.rs", "try_allocate_transaction"),
    ("models/mimo_asr/executor.rs", "try_allocate_transaction"),
    (
        "models/moss_transcribe_diarize/executor.rs",
        "try_allocate_transaction",
    ),
    (
        "models/parakeet_ctc/executor.rs",
        "try_allocate_transaction",
    ),
    (
        "models/parakeet_tdt/executor.rs",
        "try_allocate_transaction",
    ),
    (
        "models/prepared_runtime_cache.rs",
        "try_allocate_transaction",
    ),
    ("models/qwen/ggml_executor.rs", "try_allocate_transaction"),
    (
        "models/qwen/forced_aligner_runtime.rs",
        "try_allocate_transaction",
    ),
    ("models/qwen/kv_cache.rs", "try_allocate"),
    ("models/sensevoice/executor.rs", "try_allocate_transaction"),
    ("models/system_memory_owner.rs", "try_allocate"),
    ("models/system_memory_owner.rs", "try_allocate_transaction"),
    (
        "models/wav2vec2_ctc/executor.rs",
        "try_allocate_transaction",
    ),
    (
        "models/xasr_zipformer/runtime.rs",
        "try_allocate_transaction",
    ),
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

fn source_relative(src_root: &Path, file: &Path) -> String {
    file.strip_prefix(src_root)
        .expect("file under src root")
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn resident_symbol_class(symbol: &str) -> Option<&'static str> {
    let lower = symbol.to_ascii_lowercase();
    if lower.starts_with("test") || lower.contains("profile") {
        return None;
    }
    if matches!(
        symbol,
        "BackendMemoryAbi"
            | "GgmlLoadedWeightContext"
            | "LoadedWeightOwnerCache"
            | "GgmlCpuStepBufferPool"
            | "LOADED_WEIGHT_OWNER_SLOTS"
            | "THREAD_BACKEND_CACHE_BY_KIND"
    ) {
        return Some(match symbol {
            "BackendMemoryAbi" => "backend-memory",
            "GgmlLoadedWeightContext" => "loaded-weight-context",
            "LoadedWeightOwnerCache" => "resident-owner",
            "GgmlCpuStepBufferPool" => "step-buffer-pool",
            "LOADED_WEIGHT_OWNER_SLOTS" | "THREAD_BACKEND_CACHE_BY_KIND" => "tls-cache",
            _ => unreachable!(),
        });
    }
    if lower.contains("residentcache") {
        Some("resident-cache")
    } else if lower.contains("runtimecache")
        && !lower.contains("cachekey")
        && !lower.contains("error")
    {
        Some("runtime-cache")
    } else if lower.ends_with("runtimepool")
        || lower.ends_with("runtimeownerpool")
        || lower.ends_with("actorpool")
    {
        Some("actor-pool")
    } else if lower.ends_with("actorcheckout") {
        Some("actor-checkout")
    } else if lower == "admittedexclusiveobjectpool"
        || lower == "admittedhostobjectcache"
        || lower == "admittedpinnedruntimeactorpool"
        || lower == "admittedpinnedruntimeactorcheckoutpool"
        || lower == "builtinpreparedruntimecache"
        || lower == "preparedruntimecache"
        || lower == "auxiliaryruntimeownercache"
        || lower == "auxiliarypinnedruntimecachekey"
        || lower == "auxiliaryruntimecachekey"
        || lower == "residentcheckoutpool"
        || lower == "nativememoryallocationtransaction"
        || lower == "pinnedruntimeactorcheckout"
    {
        Some("resident-owner")
    } else {
        None
    }
}

fn resident_inventory_set() -> BTreeSet<(String, String, String)> {
    let mut inventory: BTreeSet<_> = RESIDENT_CONSTRUCTION_PUBLICATION_INVENTORY
        .iter()
        .filter(|(_, path, symbol, _)| {
            !matches!(
                (*path, *symbol),
                (
                    "diarize/embed/policy_runtime.rs",
                    "AuxiliaryRuntimeCacheKey"
                ) | (
                    "diarize/segment/policy_runtime.rs",
                    "AuxiliaryRuntimeCacheKey"
                ) | (
                    "diarize/vad/firered_stream/realtime_runtime.rs",
                    "PinnedRuntimeActorCheckout"
                )
            )
        })
        .filter_map(|(_, path, symbol, _)| {
            resident_symbol_class(symbol).map(|class| {
                (
                    (*path).to_string(),
                    (*symbol).to_string(),
                    class.to_string(),
                )
            })
        })
        .collect();
    for (path, symbol) in [
        ("models/cohere/ggml_executor.rs", "CohereDecoderRuntimePool"),
        ("models/cohere/ggml_executor.rs", "CohereEncoderRuntimePool"),
        ("models/cohere/ggml_executor.rs", "CohereUnifiedRuntimePool"),
        ("models/dolphin/executor.rs", "DolphinPreparedRuntimePool"),
        (
            "models/firered_aed/executor.rs",
            "FireRedAedDecoderRuntimePool",
        ),
        (
            "models/firered_aed/executor.rs",
            "FireRedAedEncoderRuntimePool",
        ),
        (
            "models/firered_aed/executor.rs",
            "FireRedAedRuntimeOwnerPool",
        ),
        (
            "models/firered_llm/executor.rs",
            "FireRedLlmDecoderRuntimePool",
        ),
        (
            "models/firered_llm/executor.rs",
            "FireRedLlmUnifiedRuntimePool",
        ),
        (
            "models/funasr_nano/executor.rs",
            "FunasrNanoDecoderRuntimePool",
        ),
        (
            "models/funasr_nano/executor.rs",
            "FunasrNanoEncoderAdapterRuntimePool",
        ),
        (
            "models/funasr_nano/executor.rs",
            "FunasrNanoUnifiedRuntimePool",
        ),
        (
            "models/granite_speech/executor.rs",
            "GraniteSpeechPreparedRuntimePool",
        ),
        ("models/mimo_asr/executor.rs", "MimoAsrPreparedRuntimePool"),
        (
            "models/moonshine/ggml_executor.rs",
            "MoonshineDecoderRuntimePool",
        ),
        (
            "models/moonshine/ggml_executor.rs",
            "MoonshineEncoderRuntimePool",
        ),
        (
            "models/moss_transcribe_diarize/executor.rs",
            "MossTdDecoderRuntimePool",
        ),
        (
            "models/moss_transcribe_diarize/executor.rs",
            "MossTdEncoderRuntimePool",
        ),
        (
            "models/moss_transcribe_diarize/executor.rs",
            "MossTdUnifiedRuntimePool",
        ),
        ("models/parakeet_ctc/executor.rs", "ParakeetCtcRuntimePool"),
        ("models/parakeet_tdt/executor.rs", "ParakeetTdtRuntimePool"),
        (
            "models/qwen/ggml_executor.rs",
            "Qwen3AsrAudioEncoderRuntimePool",
        ),
        (
            "models/qwen/ggml_executor.rs",
            "Qwen3AsrDecoderActorCheckout",
        ),
        ("models/qwen/ggml_executor.rs", "Qwen3AsrDecoderRuntimePool"),
        ("models/qwen/ggml_executor.rs", "Qwen3AsrRuntimeOwnerPool"),
        ("models/sensevoice/executor.rs", "SenseVoiceRuntimePool"),
        ("models/wav2vec2_ctc/executor.rs", "Wav2Vec2RuntimePool"),
        (
            "models/whisper/ggml_executor.rs",
            "WhisperDecoderRuntimePool",
        ),
        (
            "models/whisper/ggml_executor.rs",
            "WhisperEncoderRuntimePool",
        ),
        (
            "models/whisper/ggml_executor.rs",
            "WhisperUnifiedRuntimePool",
        ),
        ("models/xasr_zipformer/runtime.rs", "XasrRuntimeActorPool"),
    ] {
        inventory.insert((
            path.to_string(),
            symbol.to_string(),
            if symbol.ends_with("ActorCheckout") {
                "actor-checkout".to_string()
            } else {
                "actor-pool".to_string()
            },
        ));
    }
    inventory
}

fn declared_symbols(source: &str) -> impl Iterator<Item = String> + '_ {
    source.lines().filter_map(|line| {
        let mut line = line.trim();
        for prefix in ["pub(crate) ", "pub(super) ", "pub ", "unsafe "] {
            line = line.strip_prefix(prefix).unwrap_or(line);
        }
        for keyword in ["struct ", "enum ", "type ", "static ", "const ", "impl "] {
            let Some(rest) = line.strip_prefix(keyword) else {
                continue;
            };
            let symbol: String = rest
                .chars()
                .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
                .collect();
            if !symbol.is_empty() {
                return Some(symbol);
            }
        }
        None
    })
}

/// Inventory-independent machine discovery. It scans declarations and TLS/static
/// publication shapes across the supplied Rust source tree, classifies only
/// resident-shaped symbols, and returns `(path, symbol, class)` identities.
fn discover_resident_construction_inventory(src_root: &Path) -> BTreeSet<(String, String, String)> {
    let mut files = Vec::new();
    collect_rs_files(src_root, &mut files);
    let mut discovered = BTreeSet::new();
    for file in files {
        let relative = source_relative(src_root, &file);
        let source = std::fs::read_to_string(&file)
            .unwrap_or_else(|error| panic!("read resident source {}: {error}", file.display()));
        for symbol in declared_symbols(&source) {
            let Some(class) = resident_symbol_class(&symbol) else {
                continue;
            };
            discovered.insert((relative.clone(), symbol, class.to_string()));
        }
    }
    discovered
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

#[test]
fn resident_footprint_inventory_is_complete() {
    let registry = OpenAsrArchitectureRegistry::with_builtins();
    let descriptors = registry.descriptors();
    assert_eq!(
        descriptors.len(),
        16,
        "the live ASR inventory must have one resident footprint row per architecture"
    );

    let mut module_slugs = BTreeSet::new();
    for descriptor in descriptors {
        assert!(
            module_slugs.insert(descriptor.identity.module_slug),
            "resident footprint inventory has a duplicate module slug: {}",
            descriptor.identity.module_slug
        );
        descriptor
            .resident_footprint
            .validate()
            .unwrap_or_else(|error| {
                panic!(
                    "resident footprint for {} is invalid: {error:?}",
                    descriptor.identity.model_architecture
                )
            });
        assert!(
            descriptor.resident_footprint.component_count() > 0,
            "resident footprint for {} is empty",
            descriptor.identity.model_architecture
        );

        for component in descriptor.resident_footprint.components() {
            assert!(
                component
                    .representations()
                    .contains(&ResidentRepresentation::HostImportedBinding),
                "{} component {} must declare host-import representation alternative",
                descriptor.identity.model_architecture,
                component.component()
            );
            assert!(
                component
                    .representations()
                    .contains(&ResidentRepresentation::DeviceCopiedBinding),
                "{} component {} must declare device-copy representation alternative",
                descriptor.identity.model_architecture,
                component.component()
            );
            assert!(
                component.checkout().max_instances() > 0,
                "{} component {} has an unbounded/zero checkout maximum",
                descriptor.identity.model_architecture,
                component.component()
            );
        }
        for placement in [
            ResidentPlacementVariant::Unified,
            ResidentPlacementVariant::Split,
        ] {
            assert!(
                descriptor
                    .resident_footprint
                    .components()
                    .iter()
                    .any(|component| component.placement_variants().contains(&placement)),
                "{} resident footprint has no component for {placement:?} placement",
                descriptor.identity.model_architecture
            );
        }
    }
}

#[test]
fn auxiliary_resident_footprints_are_complete() {
    use crate::arch::runtime_footprint::AUXILIARY_RESIDENT_FOOTPRINTS;
    assert!(
        !AUXILIARY_RESIDENT_FOOTPRINTS.is_empty(),
        "non-ASR auxiliary owners must have resident footprint rows"
    );
    let mut names = BTreeSet::new();
    for (name, facet) in AUXILIARY_RESIDENT_FOOTPRINTS {
        assert!(
            names.insert(*name),
            "duplicate auxiliary resident footprint: {name}"
        );
        facet.validate().unwrap_or_else(|error| {
            panic!("auxiliary resident footprint {name} is invalid: {error:?}")
        });
        assert!(
            facet.component_count() > 0,
            "auxiliary resident footprint {name} is empty"
        );
    }
    for required in [
        "firered-stream-vad",
        "redimnet2",
        "pyannote-segmentation",
        "diarizen-segmentation",
        "firered-punc",
        "qwen3-forced-aligner",
    ] {
        assert!(
            names.contains(required),
            "auxiliary resident footprint missing {required}"
        );
    }
}

#[test]
fn firered_llm_split_request_runtimes_enter_system_memory_owner() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models/firered_llm/executor.rs");
    let source = std::fs::read_to_string(&path).expect("read firered-llm executor");
    assert!(
        source.contains("fn allocate_split_encoder_runtime_owner"),
        "FireRed LLM split encoder must enter SystemMemoryOwner"
    );
    assert!(
        source.contains("fn allocate_split_adapter_runtime_owner"),
        "FireRed LLM split adapter must enter SystemMemoryOwner"
    );
    let Some((_, execute)) = source.split_once("fn execute_inner_with_runtime_mode") else {
        panic!("FireRed LLM execute_inner_with_runtime_mode is missing");
    };
    let execute = execute.split("fn ").next().expect("execute_inner body");
    assert!(
        execute.contains("allocate_split_encoder_runtime_owner"),
        "split encoder path must call the owner allocator"
    );
    assert!(
        execute.contains("allocate_split_adapter_runtime_owner"),
        "split adapter path must call the owner allocator"
    );
    assert!(
        !execute.contains("FireRedEncoderGraphRuntime::new_from_preflight"),
        "split encoder JIT publication must not remain beside SystemMemoryOwner"
    );
    assert!(
        !execute.contains("FireRedLlmAdapterGraphRuntime::new_from_preflight"),
        "split adapter JIT publication must not remain beside SystemMemoryOwner"
    );
}

#[test]
fn resident_construction_publication_inventory_is_complete() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut surfaces = BTreeSet::new();
    for (surface, relative, symbol, _status) in RESIDENT_CONSTRUCTION_PUBLICATION_INVENTORY {
        let path = root.join(relative);
        assert!(
            path.is_file(),
            "resident inventory path is missing: {relative}"
        );
        surfaces.insert(*surface);
        let syntax = ProductionSyntax::collect(&path);
        assert!(
            syntax.references_identifier(symbol) || syntax.calls_or_invokes_method(symbol),
            "resident inventory path {relative} does not expose canonical symbol {symbol}"
        );
    }
    assert!(surfaces.contains(&ResidentSurface::Models));
    assert!(surfaces.contains(&ResidentSurface::GgmlRuntime));
    assert!(surfaces.contains(&ResidentSurface::Auxiliary));
}

#[test]
fn lease_construction_sites_match_inventory_both_directions() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs_files(&root, &mut files);
    let mut discovered = BTreeSet::new();
    for file in files {
        let relative = source_relative(&root, &file);
        if relative == "models/resident_runtime_audit.rs" {
            continue;
        }
        let syntax = ProductionSyntax::collect(&file);
        for method in LEASE_CONSTRUCTION_METHODS {
            if syntax.calls_or_invokes_method(method) {
                discovered.insert((relative.clone(), (*method).to_string()));
            }
        }
    }
    let inventory: BTreeSet<(String, String)> = LEASE_CONSTRUCTION_INVENTORY
        .iter()
        .map(|(path, method)| ((*path).to_string(), (*method).to_string()))
        .collect();
    let unlisted: Vec<_> = discovered.difference(&inventory).cloned().collect();
    let stale: Vec<_> = inventory.difference(&discovered).cloned().collect();
    assert!(
        unlisted.is_empty(),
        "lease construction sites are not inventoried: {unlisted:?}"
    );
    assert!(
        stale.is_empty(),
        "lease construction inventory has stale sites: {stale:?}"
    );
    for (relative, method) in LEASE_CONSTRUCTION_INVENTORY {
        let path = root.join(relative);
        assert!(
            path.is_file(),
            "lease inventory path is missing: {relative}"
        );
        let syntax = ProductionSyntax::collect(&path);
        assert!(
            syntax.calls_or_invokes_method(method),
            "lease inventory {relative} does not call {method}"
        );
    }
}

#[test]
fn k5_machine_discovery_matches_resident_construction_inventory_both_directions() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let discovered = discover_resident_construction_inventory(&root);
    let inventory = resident_inventory_set();
    let unlisted: Vec<_> = discovered.difference(&inventory).cloned().collect();
    let stale: Vec<_> = inventory.difference(&discovered).cloned().collect();
    assert!(
        unlisted.is_empty(),
        "K5 resident construction discovery found unlisted surfaces: {unlisted:?}"
    );
    assert!(
        stale.is_empty(),
        "K5 resident construction inventory has stale surfaces: {stale:?}"
    );

    let fixture_root = tempfile::tempdir().expect("K5 fixture root");
    std::fs::write(
        fixture_root.path().join("fixture.rs"),
        "struct FooResidentCache;\n",
    )
    .expect("write K5 resident fixture");
    let fixture = (
        "fixture.rs".to_string(),
        "FooResidentCache".to_string(),
        "resident-cache".to_string(),
    );
    let fixture_discovered = discover_resident_construction_inventory(fixture_root.path());
    assert!(
        fixture_discovered.contains(&fixture),
        "K5 fixture must be discovered from real Rust syntax"
    );
    assert!(
        fixture_discovered
            .difference(&inventory)
            .any(|row| row == &fixture),
        "K5 fixture must fail the unlisted-side difference"
    );
}

#[test]
fn no_resident_owner_compatibility_seams_remain() {
    assert!(
        RESIDENT_CONSTRUCTION_PUBLICATION_INVENTORY
            .iter()
            .all(|(_, _, _, status)| *status == ResidentSiteStatus::Active)
    );
}

#[test]
fn thread_affine_backend_cache_is_scoped_and_receipted() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/ggml_runtime/cpu_graph.rs");
    let source = std::fs::read_to_string(path).expect("read cpu graph source");
    assert!(source.contains("struct CachedBackendKey"));
    assert!(
        source
            .contains("scope_id: crate::models::native_execution_services::NativeExecutionScopeId")
    );
    assert!(source.contains("current_native_execution_scope_id"));
    assert!(
        source
            .contains("_receipt_owner: Option<crate::models::runtime_receipts::RuntimeOwnerGuard>")
    );
    assert!(source.contains("free_on_drop: true"));
}

#[test]
fn production_loaded_weight_publication_does_not_use_tls_owner_table() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/ggml_runtime/cpu_graph.rs");
    let syntax = ProductionSyntax::collect(&path);
    assert!(
        !syntax.references_identifier("LOADED_WEIGHT_CONTEXT_BY_KEY"),
        "production loaded-weight publication must not use LOADED_WEIGHT_CONTEXT_BY_KEY"
    );
    assert!(
        syntax.references_identifier("LoadedWeightOwnerCache"),
        "production loaded-weight publication must enter the NES-owned owner cache"
    );
    assert!(
        syntax.calls_or_invokes_method("current_loaded_weight_owners"),
        "production load_gguf_weight_context must receive the installed NES owner cache"
    );
}

#[test]
fn footprint_construction_is_arch_private() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/arch/runtime_footprint.rs");
    let source = std::fs::read_to_string(path).expect("read footprint source");
    for forbidden in [
        "pub(crate) struct ResidentTopologyInputs",
        "pub(crate) const fn new(components",
        "pub(crate) const fn new(\n        component:",
    ] {
        assert!(
            !source.contains(forbidden),
            "family/model code must not receive a public footprint construction seam: {forbidden}"
        );
    }
    assert!(source.contains("pub(super) struct ResidentTopologyInputs"));
    assert!(source.contains("pub(super) const fn new(components"));
    assert!(!source.contains("    pub(crate) architecture:"));
    assert!(!source.contains("    pub(crate) spec:"));
    assert!(!source.contains("    pub(crate) verified_pack:"));
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
            let syntax = ProductionSyntax::collect(file);
            syntax.calls_or_invokes_method("current_execution_lane_key")
                || syntax.calls_or_invokes_method("native_execution_lane")
        });
        assert!(
            references_lane_key && derives_lane_key,
            "K3 execution-lane gate: resident family '{family}' does not derive its \
             backend-owner cache identity through ExecutionLaneKey and the explicit request lane \
             (or the shared current_execution_lane_key adapter). \
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
        (
            "models/firered_punc/policy_runtime.rs",
            "PinnedRuntimeActor",
        ),
        ("models/qwen/forced_aligner_runtime.rs", "SystemMemoryOwner"),
    ] {
        let syntax = ProductionSyntax::collect(&root.join(relative));
        assert!(
            syntax.references_identifier(required),
            "K4 auxiliary ownership gate: {relative} does not reference declared owner primitive {required}"
        );
    }
}

const CANDIDATE_PROTOCOL_SYMBOLS: &[&str] = &[
    "ExecutionCacheJournalScope",
    "CandidateActivationTransaction",
    "PreparedTransaction",
    "ExecutionCandidateAttemptJournalFactory",
    "DefaultModelActivationJournalFactory",
];

const CANDIDATE_PROTOCOL_PRODUCTION_SITES: &[&str] = &[
    "models/candidate_activation_transaction.rs",
    "models/native_execution_services.rs",
];

fn discover_candidate_protocol_sites(src_root: &Path) -> BTreeSet<(String, String)> {
    let mut files = Vec::new();
    collect_rs_files(src_root, &mut files);
    let mut discovered = BTreeSet::new();
    for file in files {
        let relative = source_relative(src_root, &file);
        if relative == "models/resident_runtime_audit.rs" {
            continue;
        }
        let syntax = ProductionSyntax::collect(&file);
        for symbol in CANDIDATE_PROTOCOL_SYMBOLS {
            if syntax.references_identifier(symbol) {
                discovered.insert((relative.clone(), (*symbol).to_string()));
            }
        }
    }
    discovered
}

#[test]
fn candidate_protocol_production_sites_match_inventory_both_directions() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let discovered = discover_candidate_protocol_sites(&root);
    let mut inventory = BTreeSet::new();
    for path in CANDIDATE_PROTOCOL_PRODUCTION_SITES {
        let syntax = ProductionSyntax::collect(&root.join(path));
        for symbol in CANDIDATE_PROTOCOL_SYMBOLS {
            if syntax.references_identifier(symbol) {
                inventory.insert(((*path).to_string(), (*symbol).to_string()));
            }
        }
    }
    let unlisted: Vec<_> = discovered.difference(&inventory).cloned().collect();
    let stale: Vec<_> = inventory.difference(&discovered).cloned().collect();
    assert!(
        unlisted.is_empty(),
        "handwritten candidate retry/publication protocol used outside run_execution_candidate_attempt / set-default transaction sites: {unlisted:?}"
    );
    assert!(
        stale.is_empty(),
        "candidate protocol inventory has stale sites: {stale:?}"
    );
    let nes = ProductionSyntax::collect(&root.join("models/native_execution_services.rs"));
    assert!(
        nes.references_identifier("CandidateActivationTransaction")
            || nes.references_identifier("ExecutionCandidateAttemptJournalFactory"),
        "run_execution_candidate_attempt must enter CandidateActivationTransaction in production"
    );
}

#[test]
fn handwritten_candidate_retry_publication_bypass_fails_source_audit() {
    let fixture_root = tempfile::tempdir().expect("candidate bypass fixture root");
    std::fs::write(
        fixture_root.path().join("family_local_retry.rs"),
        r#"
            fn family_local_retry(plan: ExecutionPlan) {
                for candidate in plan.candidates() {
                    let _scope = ExecutionCacheJournalScope::begin();
                    let _ = CandidateActivationTransaction::prepare(
                        candidate,
                        facts,
                        journal,
                    );
                }
            }
        "#,
    )
    .expect("write handwritten bypass fixture");
    let discovered = discover_candidate_protocol_sites(fixture_root.path());
    assert!(
        discovered.iter().any(|(path, symbol)| {
            path == "family_local_retry.rs"
                && (symbol == "ExecutionCacheJournalScope"
                    || symbol == "CandidateActivationTransaction")
        }),
        "source audit must fail closed on a new handwritten candidate retry/publication loop, got {discovered:?}"
    );
}

#[test]
fn candidate_retry_publication_without_attempt_is_forbidden() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs_files(&root, &mut files);
    let mut discovered = Vec::new();
    for file in files {
        let relative = source_relative(&root, &file);
        if relative == "models/resident_runtime_audit.rs"
            || relative == "models/candidate_activation_transaction.rs"
            || relative == "models/native_execution_services.rs"
        {
            continue;
        }
        let syntax = ProductionSyntax::collect(&file);
        let inspects_plan = syntax.calls_or_invokes_method("candidates");
        let publishes = syntax.references_identifier("ExecutionCacheJournalScope")
            || syntax.calls_or_invokes_method("stage_execution_cache_commit")
            || syntax.references_identifier("CandidateActivationTransaction")
            || syntax.references_identifier("ExecutionCandidateAttemptJournalFactory");
        if inspects_plan
            && publishes
            && !syntax.calls_or_invokes_method("run_execution_candidate_attempt")
        {
            discovered.push(relative);
        }
    }
    assert!(
        discovered.is_empty(),
        "handwritten candidate retry/publication loop bypasses run_execution_candidate_attempt: {discovered:?}"
    );

    let fixture_root = tempfile::tempdir().expect("retry publication fixture root");
    std::fs::write(
        fixture_root.path().join("bypass.rs"),
        r#"
            fn handwritten(plan: ExecutionPlan) {
                for candidate in plan.candidates() {
                    stage_execution_cache_commit(|| {});
                }
            }
        "#,
    )
    .expect("write retry publication fixture");
    let fixture = ProductionSyntax::collect(&fixture_root.path().join("bypass.rs"));
    assert!(
        fixture.calls_or_invokes_method("candidates")
            && fixture.calls_or_invokes_method("stage_execution_cache_commit")
            && !fixture.calls_or_invokes_method("run_execution_candidate_attempt"),
        "source audit must fail closed on a new handwritten candidate retry/publication loop"
    );
}

#[test]
fn family_modules_do_not_own_candidate_retry_loops() {
    let models = models_dir();
    for family in on_disk_ggml_executor_families(&models) {
        let family_dir = models.join(&family);
        let mut rs_files = Vec::new();
        collect_rs_files(&family_dir, &mut rs_files);
        for file in rs_files {
            let syntax = ProductionSyntax::collect(&file);
            assert!(
                !syntax.calls_or_invokes_method("candidates"),
                "family {} must not hand-write a candidate retry/publication loop; production retries belong in run_execution_candidate_attempt callers ({})",
                family,
                file.display()
            );
            for symbol in CANDIDATE_PROTOCOL_SYMBOLS {
                assert!(
                    !syntax.references_identifier(symbol),
                    "family {family} must not construct CandidateActivationTransaction / cache journals directly ({symbol})"
                );
            }
        }
    }
}

#[test]
fn production_candidate_reserve_does_not_use_noop_reservation() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    for relative in [
        "models/native_execution_services.rs",
        "models/candidate_activation_transaction.rs",
    ] {
        let source = std::fs::read_to_string(root.join(relative)).expect("read production source");
        if relative.ends_with("native_execution_services.rs") {
            let body = source
                .split("pub(crate) fn run_execution_candidate_attempt")
                .nth(1)
                .expect("attempt")
                .split("pub enum NativeExecutionServicesError")
                .next()
                .expect("attempt body");
            let reserve = body.find(".reserve(").expect("attempt must reserve");
            let noop = body.find("NoopActivationReservation");
            assert!(
                noop.is_none() || noop.unwrap() > body.find(".reserve(").unwrap() + 80,
                "run_execution_candidate_attempt must not reserve with NoopActivationReservation"
            );
            assert!(
                (body.contains("quote_and_reserve_current_candidate_activation")
                    || body.contains("quote_and_reserve_candidate_activation"))
                    && body[reserve..].contains("reservation"),
                "attempt reserve must use the broker quote/reserve token"
            );
        }
    }
}

#[test]
fn production_activation_reserve_does_not_use_placeholder_bytes() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models/native_execution_services.rs");
    let source = std::fs::read_to_string(&path).expect("read NES source");
    let quote = source
        .split("fn quote_and_reserve_current_candidate_activation")
        .nth(1)
        .expect("quote_and_reserve_current_candidate_activation")
        .split("pub(crate) fn run_execution_candidate_attempt")
        .next()
        .expect("quote function body");
    let production = source
        .split("pub fn quote_and_reserve_candidate_activation")
        .next()
        .expect("production quote helpers precede the public function");
    assert!(
        !quote.contains("4096") && !quote.contains("HOST_IMPORT_GATE"),
        "activation quote must not use a placeholder page: {quote}"
    );
    assert!(
        !quote.contains("peak_bytes: 0,"),
        "activation quote must not reserve unknown domains as zero: {quote}"
    );
    let plan = production
        .split("fn quote_activation_group")
        .nth(1)
        .expect("quote_activation_group")
        .split("pub fn quote_and_reserve_candidate_activation")
        .next()
        .expect("plan body");
    assert!(
        plan.contains("NativeMemoryAdmissionPlan") && plan.contains("NativeQuotedBackendGroup"),
        "activation quote must go through NativeMemoryAdmissionPlan / ggml: {plan}"
    );
    assert!(
        !plan.contains("peak_bytes: 0,"),
        "activation plan must not emit zero-byte domain rows: {plan}"
    );
    assert!(
        !quote.contains("verified_pack_from_preflight_for_test")
            && !quote.contains("leaked_tiny_runtime_source_preflight")
            && !quote.contains("#[cfg(test)]"),
        "production quote must not depend on a test-only fake pack: {quote}"
    );
    assert!(
        quote.contains("CandidateActivationQuoteSource::Pack")
            && quote.contains("CandidateActivationQuoteSource::Declared")
            && quote.contains("quote_and_reserve_declared_host_resident"),
        "current activation quote must select pack vs declared owner bytes: {quote}"
    );
    assert!(
        !quote.contains("declared_stream_vad_resident_quote")
            && !quote.contains("FireRedStreamVadModel::system_memory_quote"),
        "packless attempts must not default to the Stream-VAD blob: {quote}"
    );
}

#[test]
fn serve_batch_publication_requires_candidate_attempt() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models/seq2seq_serve_batch.rs");
    let syntax = ProductionSyntax::collect(&path);
    assert!(
        syntax.calls_or_invokes_method("current_execution_cache_attempt_id"),
        "serve-batch production publication must fail closed outside a candidate attempt"
    );
    let source = std::fs::read_to_string(&path).expect("read serve-batch source");
    let engine = source
        .split("pub(crate) fn engine_for_key")
        .nth(1)
        .expect("engine_for_key")
        .split("loop {")
        .next()
        .expect("engine_for_key prelude");
    assert!(
        engine.contains("current_execution_cache_attempt_id"),
        "engine_for_key must refuse attempt-free publication: {engine}"
    );
}

#[test]
fn migrated_owner_shapes_cannot_reintroduce_unpriced_live_resources() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models");
    for relative in [
        "seq2seq_serve_batch.rs",
        "qwen/batched_decode.rs",
        "qwen/forced_aligner_runtime.rs",
    ] {
        let source = std::fs::read_to_string(root.join(relative)).expect("read owner source");
        let production = source
            .split("#[cfg(test)]\nmod tests")
            .next()
            .expect("production source prefix");
        assert!(
            !production.contains("unpriced_resource_descriptor")
                && !production.contains("NotPricedLegacy"),
            "migrated owner {relative} reintroduced an unpriced live resource"
        );
    }
}

#[test]
fn run_execution_candidate_attempt_walks_attestation_without_skipping() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models/native_execution_services.rs");
    let source = std::fs::read_to_string(&path).expect("read NES source");
    let Some((_, body)) = source.split_once("pub(crate) fn run_execution_candidate_attempt") else {
        panic!("run_execution_candidate_attempt is missing");
    };
    let body = body
        .split("pub enum NativeExecutionServicesError")
        .next()
        .expect("attempt body");
    let prepare = body
        .find(".prepare(")
        .expect("attempt must prepare a CandidateActivationTransaction");
    let reserve = body
        .find(".reserve(")
        .expect("attempt must reserve; quote is not a reservation");
    let materialize = body
        .find(".materialize(")
        .expect("attempt must materialize");
    let pending = body
        .find(".begin_attestation(")
        .expect("attempt must enter AttestationPending");
    let attest = body.find(".attest()").expect("attempt must attest");
    let commit = body
        .find(".commit_attempt()")
        .expect("attempt must commit only after attest");
    assert!(
        prepare < reserve
            && reserve < materialize
            && materialize < pending
            && pending < attest
            && attest < commit,
        "attempt must walk prepare -> reserve -> materialize -> AttestationPending -> attest -> commit, got prepare@{prepare} reserve@{reserve} materialize@{materialize} pending@{pending} attest@{attest} commit@{commit}"
    );
    assert!(
        body.contains("ActivationStage::AttestationPending"),
        "attempt must retain AttestationPending"
    );
    assert!(
        body.contains("ActivationStage::Attested"),
        "attempt must not skip Attested"
    );
    assert!(
        !body.contains("ActivationStage::Committed") || commit > attest,
        "attempt must not treat Committed as reachable without attest"
    );
}
