//! One-shot selection and loading of local-activity segmenter packs.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use sha2::Digest;

use super::{LocalActivitySegmenter, PyannoteSegmenter, SegmentError};
use crate::config::VoiceIdSegmenterPreference;
use crate::device::execution_route::{
    ExecutionRouteCacheKey, discrete_vram_admission_budget_for_route,
    enumerate_compute_devices_from_ggml, ranked_preferred_accelerated_devices,
};
use crate::ggml_runtime::{
    AutoGpuPolicy, GgmlCpuGraphBackend, RequestBackendPreference, ResolvedFamilyRuntimeInput,
};
use crate::models::thread_local_runtime_cache::PackContentKey;

static ACTIVE_SEGMENTATION_3_0: LazyLock<
    Mutex<Option<(PackContentKey, Arc<PyannoteSegmenter>, u64)>>,
> = LazyLock::new(|| Mutex::new(None));

const PACK_ENV: &str = "OPENASR_PYANNOTE_PACK";
const INSTALLED_MODEL_ID_HINT: &str = "pyannote-segmentation-3.0";
pub const SEGMENTER_PACK_ID: &str = "pyannote-segmentation-3.0";
pub const DIARIZEN_PACK_ID: &str = super::diarizen::DIARIZEN_MODEL_ID;
const SEGMENTER_ADMISSION_PACK_MULTIPLIER: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SegmenterProvider {
    DiariZen,
    Segmentation3_0,
}

/// Request-scoped execution identity for an auxiliary segmentation model.
///
/// Each candidate carries the exact route used for graph construction,
/// admission, and the resident-runtime cache key. Auto/accelerated requests
/// retain their ranked fallback list, but every attempted graph is exact and
/// therefore can never contaminate another device's cache slot.
#[derive(Debug, Clone)]
pub(crate) struct SegmenterRuntimeInput {
    backend: GgmlCpuGraphBackend,
    candidates: Vec<SegmenterExecutionCandidate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum SegmenterExecutionKey {
    Cpu,
    Metal,
    Route(ExecutionRouteCacheKey),
}

#[derive(Debug, Clone)]
pub(crate) struct SegmenterExecutionCandidate {
    pub key: SegmenterExecutionKey,
    pub backend_preference: Option<RequestBackendPreference>,
    pub vram_budget_bytes: Option<u64>,
}

impl SegmenterRuntimeInput {
    pub(crate) fn resolve(
        preference: Option<RequestBackendPreference>,
    ) -> Result<Self, crate::device::execution_route::ExecutionRouteError> {
        let backend =
            ResolvedFamilyRuntimeInput::resolve(preference.clone(), AutoGpuPolicy::AllBackends)
                .backend();
        let devices = crate::ggml_runtime::ggml_available_devices();
        let inventory = enumerate_compute_devices_from_ggml(&devices);
        let candidates = match backend {
            GgmlCpuGraphBackend::Cpu => vec![SegmenterExecutionCandidate {
                key: SegmenterExecutionKey::Cpu,
                backend_preference: Some(RequestBackendPreference::CpuOnly),
                vram_budget_bytes: None,
            }],
            GgmlCpuGraphBackend::Metal => vec![SegmenterExecutionCandidate {
                key: SegmenterExecutionKey::Metal,
                backend_preference: Some(RequestBackendPreference::Accelerated),
                vram_budget_bytes: None,
            }],
            GgmlCpuGraphBackend::Gpu => {
                let routes = match preference {
                    Some(RequestBackendPreference::Exact(route)) => vec![route],
                    Some(RequestBackendPreference::Accelerated) | None => {
                        ranked_preferred_accelerated_devices(&inventory)
                            .into_iter()
                            .map(|device| device.to_resolved_route())
                            .collect()
                    }
                    Some(RequestBackendPreference::CpuOnly) => Vec::new(),
                };
                if routes.is_empty() {
                    return Err(
                        crate::device::execution_route::ExecutionRouteError::AcceleratedUnavailable,
                    );
                }
                routes
                    .into_iter()
                    .map(|route| SegmenterExecutionCandidate {
                        key: SegmenterExecutionKey::Route(route.cache_key()),
                        vram_budget_bytes: discrete_vram_admission_budget_for_route(
                            &route, &devices,
                        ),
                        backend_preference: Some(RequestBackendPreference::Exact(route)),
                    })
                    .collect()
            }
        };
        Ok(Self {
            backend,
            candidates,
        })
    }

    pub(crate) const fn backend(&self) -> GgmlCpuGraphBackend {
        self.backend
    }

    pub(crate) fn candidates(&self) -> &[SegmenterExecutionCandidate] {
        &self.candidates
    }

    pub(crate) fn minimum_vram_budget_bytes(&self) -> Option<u64> {
        if self.backend != GgmlCpuGraphBackend::Gpu {
            return None;
        }
        self.candidates
            .iter()
            .map(|candidate| candidate.vram_budget_bytes)
            .collect::<Option<Vec<_>>>()
            .and_then(|budgets| budgets.into_iter().min())
    }

    pub(crate) fn execution_keys(&self) -> Vec<SegmenterExecutionKey> {
        self.candidates
            .iter()
            .map(|candidate| candidate.key.clone())
            .collect()
    }
}

/// The adapter selected during request preflight. Holding this value pins the
/// choice for the whole request: inference errors are returned directly and
/// are never interpreted as permission to try the next provider.
pub(crate) struct SelectedSegmenter {
    pub provider: SegmenterProvider,
    pub adapter: Arc<dyn LocalActivitySegmenter>,
}

enum PreparedSegmenterSource {
    DiariZen(super::diarizen::PreparedDiariZenSegmenter),
    Segmentation3_0(PreparedSegmentation3_0),
}

pub(crate) struct PreparedSelectedSegmenter {
    pub provider: SegmenterProvider,
    source: PreparedSegmenterSource,
    admission_bytes: u64,
    admission_backend: GgmlCpuGraphBackend,
    discrete_vram_budget_bytes: Option<u64>,
}

struct PreparedProviderSnapshot {
    source: PreparedSegmenterSource,
    admission_bytes: u64,
    admission_backend: GgmlCpuGraphBackend,
    discrete_vram_budget_bytes: Option<u64>,
}

impl PreparedSelectedSegmenter {
    pub(crate) const fn admission_bytes(&self) -> u64 {
        self.admission_bytes
    }

    pub(crate) const fn admission_backend(&self) -> GgmlCpuGraphBackend {
        self.admission_backend
    }

    pub(crate) const fn discrete_vram_budget_bytes(&self) -> Option<u64> {
        self.discrete_vram_budget_bytes
    }

    #[cfg(test)]
    pub(crate) fn content_id(&self) -> &str {
        match &self.source {
            PreparedSegmenterSource::DiariZen(prepared) => prepared.content_id(),
            PreparedSegmenterSource::Segmentation3_0(prepared) => &prepared.key.pack_content_id,
        }
    }

    pub(crate) fn materialize(self) -> Result<SelectedSegmenter, SegmentError> {
        let adapter: Arc<dyn LocalActivitySegmenter> = match self.source {
            PreparedSegmenterSource::DiariZen(prepared) => {
                prepared.materialize().map_err(|error| {
                    SegmentError::LoadFailed(format!("{DIARIZEN_PACK_ID}: {error}"))
                })?
            }
            PreparedSegmenterSource::Segmentation3_0(prepared) => {
                materialize_segmentation_3_0(prepared)?
            }
        };
        Ok(SelectedSegmenter {
            provider: self.provider,
            adapter,
        })
    }
}

fn segmentation_3_0_path() -> Option<PathBuf> {
    crate::diarize::pack::resolve_pack(PACK_ENV, INSTALLED_MODEL_ID_HINT)
}

pub fn segmenter_pack_installed() -> bool {
    super::diarizen_pack_installed() || segmentation_3_0_path().is_some()
}

/// Resolve the user's model-level preference once. `Auto` is intentionally a
/// provider registry rather than an alias for segmentation-3.0: an installed
/// DiariZen snapshot is preferred ahead of the baseline without changing the
/// diarization module's interface. `Segmentation3_0` filters that registry to
/// the locked permissive baseline and disables the optional provider.
pub(crate) fn prepare_segmenter(
    preference: VoiceIdSegmenterPreference,
    runtime_input: SegmenterRuntimeInput,
) -> Result<PreparedSelectedSegmenter, SegmentError> {
    let (provider, prepared) = select_provider_with(
        preference,
        || {
            super::diarizen::prepare_diarizen_segmenter_snapshot(runtime_input.clone())
                .map(|prepared| {
                    prepared.map(|prepared| PreparedProviderSnapshot {
                        admission_bytes: prepared
                            .pack_bytes()
                            .saturating_mul(SEGMENTER_ADMISSION_PACK_MULTIPLIER),
                        admission_backend: runtime_input.backend(),
                        discrete_vram_budget_bytes: prepared.minimum_vram_budget_bytes(),
                        source: PreparedSegmenterSource::DiariZen(prepared),
                    })
                })
                .map_err(|error| SegmentError::LoadFailed(format!("{DIARIZEN_PACK_ID}: {error}")))
        },
        || {
            let prepared = prepare_segmentation_3_0(preference)?;
            Ok(PreparedProviderSnapshot {
                admission_bytes: prepared
                    .pack_bytes
                    .saturating_mul(SEGMENTER_ADMISSION_PACK_MULTIPLIER),
                admission_backend: GgmlCpuGraphBackend::Cpu,
                discrete_vram_budget_bytes: None,
                source: PreparedSegmenterSource::Segmentation3_0(prepared),
            })
        },
    )?;
    Ok(PreparedSelectedSegmenter {
        provider,
        source: prepared.source,
        admission_bytes: prepared.admission_bytes,
        admission_backend: prepared.admission_backend,
        discrete_vram_budget_bytes: prepared.discrete_vram_budget_bytes,
    })
}

fn select_provider_with<T>(
    preference: VoiceIdSegmenterPreference,
    diarizen: impl FnOnce() -> Result<Option<T>, SegmentError>,
    baseline: impl FnOnce() -> Result<T, SegmentError>,
) -> Result<(SegmenterProvider, T), SegmentError> {
    if preference == VoiceIdSegmenterPreference::Auto
        && let Some(prepared) = diarizen()?
    {
        return Ok((SegmenterProvider::DiariZen, prepared));
    }
    baseline().map(|prepared| (SegmenterProvider::Segmentation3_0, prepared))
}

struct PreparedSegmentation3_0 {
    key: PackContentKey,
    source: SegmenterSourceSnapshot,
    pack_bytes: u64,
}

fn prepare_segmentation_3_0(
    preference: VoiceIdSegmenterPreference,
) -> Result<PreparedSegmentation3_0, SegmentError> {
    let Some(path) = segmentation_3_0_path() else {
        clear_active_segmentation_3_0();
        return Err(SegmentError::MissingPack { preference });
    };
    let (key, source) = snapshot_segmenter_source(&path).map_err(|error| {
        clear_active_segmentation_3_0();
        SegmentError::LoadFailed(format!("{}: {error}", path.display()))
    })?;
    let pack_bytes = source.byte_len();
    Ok(PreparedSegmentation3_0 {
        key,
        source,
        pack_bytes,
    })
}

fn materialize_segmentation_3_0(
    prepared: PreparedSegmentation3_0,
) -> Result<Arc<PyannoteSegmenter>, SegmentError> {
    if let Ok(cache) = ACTIVE_SEGMENTATION_3_0.lock()
        && let Some((cached_key, cached, _)) = cache.as_ref()
        && cached_key == &prepared.key
    {
        return Ok(Arc::clone(cached));
    }
    let built = Arc::new(
        prepared
            .source
            .into_immutable(&prepared.key)
            .and_then(ImmutableSegmenterSource::load)
            .map_err(|error| {
                clear_active_segmentation_3_0();
                SegmentError::LoadFailed(error)
            })?,
    );
    let Ok(mut cache) = ACTIVE_SEGMENTATION_3_0.lock() else {
        return Ok(built);
    };
    if let Some((cached_key, cached, _)) = cache.as_ref()
        && cached_key == &prepared.key
    {
        return Ok(Arc::clone(cached));
    }
    *cache = Some((prepared.key, Arc::clone(&built), prepared.pack_bytes));
    Ok(built)
}

/// Compatibility probe for diagnostics. Production code uses
/// [`resolve_segmenter`] so selection errors retain their typed reason.
pub fn shared_segmenter() -> Option<Arc<PyannoteSegmenter>> {
    prepare_segmentation_3_0(VoiceIdSegmenterPreference::Segmentation3_0)
        .and_then(materialize_segmentation_3_0)
        .ok()
}

enum SegmenterSourceSnapshot {
    Gguf(crate::ggml_runtime::GgmlRuntimeSource),
    Safetensors {
        path: PathBuf,
        file: std::fs::File,
        mmap: memmap2::Mmap,
    },
}

enum ImmutableSegmenterSource {
    Gguf(crate::ggml_runtime::GgmlRuntimeSource),
    Safetensors { path: PathBuf, bytes: Vec<u8> },
}

impl SegmenterSourceSnapshot {
    fn byte_len(&self) -> u64 {
        match self {
            Self::Gguf(source) => source.byte_len(),
            Self::Safetensors { mmap, .. } => mmap.len().try_into().unwrap_or(u64::MAX),
        }
    }

    fn into_immutable(
        self,
        expected_key: &PackContentKey,
    ) -> Result<ImmutableSegmenterSource, String> {
        match self {
            Self::Gguf(source) => {
                let immutable = source
                    .immutable_snapshot_matching_content_id(&expected_key.pack_content_id)
                    .map_err(|error| error.to_string())?;
                Ok(ImmutableSegmenterSource::Gguf(immutable))
            }
            Self::Safetensors {
                path,
                mut file,
                mmap,
            } => {
                file.seek(SeekFrom::Start(0)).map_err(|error| {
                    format!(
                        "could not seek {} for immutable snapshot: {error}",
                        path.display()
                    )
                })?;
                let mut bytes = Vec::with_capacity(mmap.len());
                file.read_to_end(&mut bytes).map_err(|error| {
                    format!(
                        "could not read {} for immutable snapshot: {error}",
                        path.display()
                    )
                })?;
                if bytes.len() != mmap.len() {
                    return Err(format!(
                        "{} changed length after request preflight: expected {}, got {}",
                        path.display(),
                        mmap.len(),
                        bytes.len()
                    ));
                }
                let actual_key =
                    PackContentKey::new(format!("sha256:{:x}", sha2::Sha256::digest(&bytes)));
                if &actual_key != expected_key {
                    return Err(format!(
                        "{} changed after request preflight",
                        path.display()
                    ));
                }
                Ok(ImmutableSegmenterSource::Safetensors { path, bytes })
            }
        }
    }
}

impl ImmutableSegmenterSource {
    fn load(self) -> Result<PyannoteSegmenter, String> {
        match self {
            Self::Gguf(source) => PyannoteSegmenter::from_runtime_source(&source)
                .map_err(|error| format!("{}: {error}", source.path().display())),
            Self::Safetensors { path, bytes } => PyannoteSegmenter::from_safetensors(&bytes)
                .map_err(|error| format!("{}: {error}", path.display())),
        }
    }
}

fn snapshot_segmenter_source(
    path: &Path,
) -> Result<(PackContentKey, SegmenterSourceSnapshot), String> {
    if crate::diarize::pack::is_gguf(path) {
        let source =
            crate::validate_ggml_runtime_source_path(path).map_err(|error| error.to_string())?;
        let key = PackContentKey::for_runtime_source(&source);
        Ok((key, SegmenterSourceSnapshot::Gguf(source)))
    } else {
        let file = std::fs::File::open(path).map_err(|error| error.to_string())?;
        // SAFETY: the read-only mapping owns the file pages independently of
        // the handle after construction and remains pinned in this request
        // snapshot through materialization.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|error| error.to_string())?;
        let key = PackContentKey::new(format!("sha256:{:x}", sha2::Sha256::digest(&mmap)));
        Ok((
            key,
            SegmenterSourceSnapshot::Safetensors {
                path: path.to_path_buf(),
                file,
                mmap,
            },
        ))
    }
}

fn clear_active_segmentation_3_0() {
    if let Ok(mut cache) = ACTIVE_SEGMENTATION_3_0.lock() {
        *cache = None;
    }
}

pub(crate) fn unload_idle_segmenter_caches() {
    clear_active_segmentation_3_0();
    super::diarizen::unload_idle_diarizen_cache();
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubSegmenter;

    impl LocalActivitySegmenter for StubSegmenter {
        fn segment_local_activity(
            &self,
            _samples: &[f32],
            _sample_rate_hz: u32,
            _canceled: &dyn Fn() -> bool,
        ) -> Result<super::super::LocalActivity, SegmentError> {
            Err(SegmentError::Canceled)
        }
    }

    fn stub_segmenter() -> Arc<dyn LocalActivitySegmenter> {
        Arc::new(StubSegmenter)
    }

    fn cpu_runtime_input() -> SegmenterRuntimeInput {
        SegmenterRuntimeInput::resolve(Some(RequestBackendPreference::CpuOnly))
            .expect("CPU runtime input")
    }

    #[test]
    fn forced_baseline_missing_pack_fails_closed_with_typed_error() {
        let home = tempfile::tempdir().unwrap();
        let error = crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_PYANNOTE_PACK", None),
                ("OPENASR_HOME", Some(home.path().as_os_str().to_os_string())),
            ],
            || {
                prepare_segmenter(
                    VoiceIdSegmenterPreference::Segmentation3_0,
                    cpu_runtime_input(),
                )
                .err()
                .expect("missing forced baseline must fail closed")
            },
        );
        assert!(matches!(
            error,
            SegmentError::MissingPack {
                preference: VoiceIdSegmenterPreference::Segmentation3_0
            }
        ));
    }

    #[test]
    fn safetensors_snapshot_fails_closed_on_in_place_rewrite_and_deletion() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = dir.path().join("segmentation.safetensors");
        let content_a = b"segmentation-content-a";
        std::fs::write(&pack, content_a).expect("write a");
        let (key_a, held_a) = snapshot_segmenter_source(&pack).expect("snapshot a");

        std::fs::write(&pack, b"segmentation-content-b").expect("replace b");
        let (key_b, _) = snapshot_segmenter_source(&pack).expect("snapshot b");
        assert_ne!(key_a, key_b, "same-path replacement must miss the cache");

        let error = held_a
            .into_immutable(&key_a)
            .err()
            .expect("in-place rewrite must invalidate the prepared snapshot");
        assert!(error.contains("changed after request preflight"), "{error}");

        std::fs::remove_file(&pack).expect("delete pack");
        assert!(
            snapshot_segmenter_source(&pack).is_err(),
            "deleted pack must not resolve to the old content"
        );
    }

    #[test]
    fn safetensors_snapshot_keeps_the_open_generation_across_atomic_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = dir.path().join("segmentation.safetensors");
        let replacement = dir.path().join("replacement.safetensors");
        let content_a = b"segmentation-content-a";
        std::fs::write(&pack, content_a).expect("write a");
        let (key_a, held_a) = snapshot_segmenter_source(&pack).expect("snapshot a");

        std::fs::write(&replacement, b"segmentation-content-b").expect("write replacement");
        std::fs::rename(&replacement, &pack).expect("atomically replace path");
        let immutable = held_a
            .into_immutable(&key_a)
            .expect("the held descriptor still names generation a");
        let ImmutableSegmenterSource::Safetensors { bytes, .. } = immutable else {
            panic!("fixture must use the safetensors snapshot path");
        };
        assert_eq!(bytes, content_a);
    }

    #[test]
    fn auto_prefers_diarizen_and_absence_falls_back_to_baseline() {
        let preferred = select_provider_with(
            VoiceIdSegmenterPreference::Auto,
            || Ok(Some((stub_segmenter(), 400, GgmlCpuGraphBackend::Metal))),
            || panic!("a present DiariZen snapshot must not load the baseline"),
        )
        .expect("preferred");
        assert_eq!(preferred.0, SegmenterProvider::DiariZen);

        let fallback = select_provider_with(
            VoiceIdSegmenterPreference::Auto,
            || Ok(None),
            || Ok((stub_segmenter(), 40, GgmlCpuGraphBackend::Cpu)),
        )
        .expect("fallback");
        assert_eq!(fallback.0, SegmenterProvider::Segmentation3_0);
        assert_eq!(preferred.1.1, 400);
        assert_eq!(fallback.1.1, 40);
        assert_eq!(preferred.1.2, GgmlCpuGraphBackend::Metal);
        assert_eq!(fallback.1.2, GgmlCpuGraphBackend::Cpu);
    }

    #[test]
    fn auto_does_not_fallback_when_diarizen_is_present_but_broken() {
        let baseline_called = std::cell::Cell::new(false);
        let error = select_provider_with(
            VoiceIdSegmenterPreference::Auto,
            || Err(SegmentError::LoadFailed("broken DiariZen".into())),
            || {
                baseline_called.set(true);
                Ok((stub_segmenter(), 40, GgmlCpuGraphBackend::Cpu))
            },
        )
        .err()
        .expect("broken preferred provider must fail closed");
        assert!(matches!(error, SegmentError::LoadFailed(_)));
        assert!(!baseline_called.get());
    }

    #[test]
    fn forced_baseline_never_probes_diarizen() {
        let selected = select_provider_with(
            VoiceIdSegmenterPreference::Segmentation3_0,
            || panic!("forced baseline must ignore DiariZen"),
            || Ok((stub_segmenter(), 40, GgmlCpuGraphBackend::Cpu)),
        )
        .expect("forced baseline");
        assert_eq!(selected.0, SegmenterProvider::Segmentation3_0);
    }

    #[test]
    fn runtime_identity_separates_coarse_and_exact_routes() {
        let mut route_a = crate::device::execution_route::ResolvedExecutionRoute::cpu();
        route_a.provider = crate::device::execution_route::ExecutionProvider::Cuda;
        route_a.kind = crate::device::execution_route::RouteDeviceKind::Accelerated;
        route_a.stable_id = "CPU-A".to_string();
        let mut route_b = route_a.clone();
        route_b.stable_id = "CPU-B".to_string();
        let exact_a = SegmenterExecutionKey::Route(route_a.cache_key());
        let exact_b = SegmenterExecutionKey::Route(route_b.cache_key());

        assert_ne!(SegmenterExecutionKey::Cpu, exact_a);
        assert_ne!(exact_a, exact_b);
    }
}
