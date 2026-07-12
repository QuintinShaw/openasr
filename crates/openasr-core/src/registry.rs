use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    atomic_file, catalog_security,
    catalog_series::{CatalogSeriesSpec, catalog_series_spec},
    config::DEFAULT_MODEL_ID,
    http,
};

mod resolution;
mod validation;

const DEFAULT_CATALOG_URL: &str = "https://catalog.openasr.org/v1/catalog.json";
const SUPPORTED_CATALOG_SCHEMA_VERSION: u32 = 1;
// Single source of truth for the canonical Hugging Face host: the same constant
// the transport-rewrite layer keys off (`http::HUGGING_FACE_HOST`), so the host
// we build weight URLs against and the host the catalog endpoint rewrites away
// from can never drift apart.
const HUGGING_FACE_BASE_URL: &str = crate::http::HUGGING_FACE_HOST;
const CATALOG_HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const CATALOG_HTTP_TIMEOUT: Duration = Duration::from_secs(60);
pub const CATALOG_FEATURE_SPEAKER_DIARIZATION: &str = "speaker-diarization";
const CATALOG_SPEAKER_EMBEDDER_WESPEAKER_ID: &str = "wespeaker-voxceleb-resnet34-lm";
/// Capability-pack feature key for the optional forced-alignment word-timestamp
/// refinement tier (`--word-timestamps=aligned`). Mirrors
/// `CATALOG_FEATURE_SPEAKER_DIARIZATION`'s role as the shared vocabulary
/// between the catalog and the CLI/server opt-in wiring.
pub const CATALOG_FEATURE_WORD_TIMESTAMPS: &str = "word-timestamps";
/// Capability-pack feature key for the optional punctuation-restoration
/// post-processing stage (restores Chinese full-width marks on an unpunctuated
/// family's transcript). Mirrors `CATALOG_FEATURE_SPEAKER_DIARIZATION` /
/// `CATALOG_FEATURE_WORD_TIMESTAMPS` as the shared catalog<->runtime vocabulary
/// for an opt-in capability pack.
pub const CATALOG_FEATURE_PUNCTUATION: &str = "punctuation";
// Soft-disabled for the initial public release lane. The ModelScope URL
// validation block below stays in place so re-enabling is a one-switch decision.
const MODELSCOPE_CATALOG_MIRRORS_ENABLED: bool = false;

/// The signed **public** catalog projection compiled into the binary — the
/// last-resort offline fallback (see [`load_embedded_signed_catalog`]) so a device
/// that has never been online still shows the model list. This is
/// `catalog.public.json` (the `public:true` models only — the same signed artifact
/// served on catalog.openasr.org), NOT the full `catalog.json` (which also carries
/// staged `public:false` entries): no unreleased model metadata ships in the
/// binary. The path reaches the repo-root `model-registry/`: this crate is
/// workspace-only by design (built as part of the OpenASR binary, never published
/// standalone), so the out-of-crate `include_str!` is intentional.
const EMBEDDED_CATALOG_JSON: &str = include_str!("../../../model-registry/catalog.public.json");
const EMBEDDED_CATALOG_SIGNATURE_JSON: &str =
    include_str!("../../../model-registry/catalog.public.signature.json");

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCard {
    pub id: String,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub default_variant: Option<String>,
    #[serde(default)]
    pub variant: Option<ModelVariantMetadata>,
    pub display_name: String,
    #[serde(default = "default_model_backend")]
    pub backend: String,
    #[serde(default = "default_model_task")]
    pub task: String,
    pub languages: Vec<String>,
    pub size: String,
    #[serde(default = "default_model_recommended_hardware")]
    pub recommended_hardware: String,
    pub license: String,
    #[serde(default = "default_model_features")]
    pub features: Vec<String>,
    #[serde(default = "default_model_quality_profile")]
    pub quality_profile: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelVariantMetadata {
    #[serde(default = "default_model_variant_tag")]
    pub tag: String,
    #[serde(default = "default_model_variant_format")]
    pub format: String,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default = "default_model_variant_role")]
    pub role: Option<String>,
}

fn default_model_backend() -> String {
    "native".to_string()
}

fn default_model_task() -> String {
    "transcription".to_string()
}

fn default_model_recommended_hardware() -> String {
    "CPU or Apple Silicon".to_string()
}

fn default_model_features() -> Vec<String> {
    vec!["transcription".to_string()]
}

fn default_model_quality_profile() -> String {
    "published-oasr".to_string()
}

fn default_model_variant_format() -> String {
    "oasr".to_string()
}

fn default_model_variant_tag() -> String {
    "published".to_string()
}

fn default_model_variant_role() -> Option<String> {
    Some("default".to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRef {
    pub family: String,
    pub tag: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel<'a> {
    pub card: &'a ModelCard,
    pub requested: String,
    pub resolved_id: String,
    pub family: String,
    pub tag: Option<String>,
    pub is_default_variant: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeModelRefSource {
    Catalog,
    Registry,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRuntimeModelRef<'a> {
    pub card: Option<&'a ModelCard>,
    pub requested: String,
    pub model_id: String,
    pub quant: Option<String>,
    pub runtime_model_id: String,
    pub pull: Option<String>,
    pub source: RuntimeModelRefSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCatalog {
    pub schema_version: u32,
    pub generated_at: String,
    pub catalog_url: String,
    pub models: Vec<CatalogModel>,
    /// Downloadable GPU backend plugin packs (HIP / Vulkan / CUDA). A top-level
    /// array authored from day one (design D7), distinct from `models[]`. Absent
    /// in the catalog until the packs land (Phases 3-4); `skip_serializing_if`
    /// keeps the signed catalog byte-identical while empty so the signature and
    /// drift gates stay green.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub backends: Vec<CatalogBackend>,
    /// Curated display labels for language/dialect recognition codes, keyed by
    /// the exact code a model advertises in `languages` (e.g. `zh-sichuan`).
    /// Carried as signed catalog DATA so app surfaces -- including the web app,
    /// which has no `@openasr/shared` dependency -- can render an advertised code
    /// without re-deriving its name. The single source of truth is
    /// `crate::models::language::language_display_label`; a drift test pins the
    /// emitted map back to it (like the canonical quant-tag contract) so Rust and
    /// the catalog cannot disagree. `skip_serializing_if` keeps a label-less
    /// catalog byte-identical while empty.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub language_labels: BTreeMap<String, CatalogLanguageLabel>,
}

/// A localized display label for one language/dialect recognition code in the
/// signed catalog's `language_labels` map. Mirrors
/// `crate::models::language::LanguageDisplayLabel` on the wire (English plus a
/// Simplified-Chinese `zh-CN` name) and is pinned to it by a drift test.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogLanguageLabel {
    pub en: String,
    #[serde(rename = "zh-CN")]
    pub zh_cn: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    #[serde(default)]
    pub kind: CatalogModelKind,
    #[serde(default)]
    pub capability: Option<CatalogCapability>,
    #[serde(default)]
    pub experimental: bool,
    pub display_name: String,
    pub family: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub pull_alias: Option<String>,
    pub size: String,
    pub languages: Vec<String>,
    // Per-model source-language parameter policy, mirroring the resolved
    // `LanguageMode` core dispatches on for this family (see
    // crate::models::language::LanguageMode and
    // crate::models::ggml_family_adapter::LanguageFamilyHint). Derived at
    // catalog-authoring time (tooling/publish-model/scripts/_catalog.py's
    // `language_mode_for_model`) from the model's family (Whisper: from its
    // resolved `languages`), not guessed per release. Absent for kinds core has
    // no source-language axis for (translation-model, capability-pack) -- old
    // clients and packs predating this field also parse fine via the default.
    #[serde(default)]
    pub language_mode: Option<CatalogLanguageMode>,
    // The language conditioned/reported when no explicit selection is made:
    // `specify_only`'s conditioned default, or `fixed_monolingual`'s single
    // language. Unset for `detect_and_specify` (auto stays unresolved until
    // decode-time detection), `detect_implicit`, and `fixed_multilingual`
    // (core exposes no per-request default for either).
    #[serde(default)]
    pub language_default: Option<String>,
    #[serde(default)]
    pub source_langs: Vec<String>,
    #[serde(default)]
    pub target_langs: Vec<String>,
    #[serde(default)]
    pub vendor: Option<String>,
    pub license: String,
    pub license_url: String,
    pub license_class: LicenseClass,
    pub hf_repo: String,
    pub hf_revision: String,
    #[serde(default)]
    pub public: bool,
    pub min_cli_version: String,
    // Optional, author-set (tooling/publish-model/models-core.toml) minimum core
    // RUNTIME version this model needs -- distinct from the publish-time
    // `min_cli_version` floor. A model forward-published before the running build
    // can execute its family (e.g. a new decoder path) sets this so a too-old
    // build gates it exactly like a too-new `min_cli_version`: surfaced as
    // "update to use" and refused at pull time, never hidden or fail-the-catalog
    // (see `availability`). Nullable; only serialized when set so unconstrained
    // models keep the Rust-side serde default (None) and the signed catalog stays
    // byte-identical while empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_core_version: Option<String>,
    // Denormalized signed-catalog wire fields derived from
    // tooling/publish-model/models-core.toml:recommended_quant. Keep all three:
    // Rust pull defaults consume recommended_quant, web/desktop use
    // quants[].recommended, and pull_recommended is the display/copyable token.
    pub recommended_quant: String,
    pub pull_recommended: String,
    // Explicit, author-set display-ranking hints from
    // tooling/publish-model/models-core.toml (`sort_weight`/`recommended`). No
    // threshold is inferred from perf/WER data here; a model opts in only via
    // an explicit catalog value. Higher `sort_weight` sorts first in
    // `models[]`; consumers needing "featured" models filter on `recommended`.
    #[serde(default)]
    pub sort_weight: i64,
    #[serde(default)]
    pub recommended: bool,
    // The UPSTREAM model's original release date (ISO `yyyy-mm-dd`), authored in
    // tooling/publish-model/models-core.toml and distinct from our repack
    // `generated_at`. Nullable: a model opts in only via an explicit catalog
    // value. Consumers use it as a display-sort tiebreaker (newest first within
    // equal `sort_weight`) and to mark recently released models. Only serialized
    // when set so unmarked models keep the Rust-side serde default (None) and the
    // signed catalog stays byte-identical while empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_release_date: Option<String>,
    // Whether the model's transcripts include punctuation -- an architecture/
    // training-corpus property, not a per-release editorial choice. This field
    // is a read-only wire mirror, not an independent declaration: the single
    // Rust-side source of truth is
    // `arch::OpenAsrArchitectureDescriptor::emits_punctuation` (see
    // `arch::emits_punctuation_for_model_architecture`), and catalog authoring
    // (`tooling/publish-model/scripts/_catalog.py`'s `punctuation_for_model` /
    // `PUNCTUATION_BY_FAMILY`) is hand-kept in lockstep with it -- there is no
    // Rust<->Python codegen bridge yet, so
    // `registry/tests/catalog.rs`'s `embedded_catalog_emits_punctuation_matches_family`
    // cross-checks the shipped catalog against the descriptor value for every
    // family both sides know about, to catch drift. `None` means "unknown" (a
    // catalog predating this field, or a kind core has no
    // transcript-punctuation axis for, e.g. capability-pack); consumers must
    // treat `None` as "assume punctuated" (`true`) rather than surfacing a
    // false "no punctuation" notice for an older/omitted entry. Only
    // serialized when set so an unmarked/legacy catalog stays byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emits_punctuation: Option<bool>,
    #[serde(default)]
    pub prose: Option<CatalogProse>,
    // Per-locale tagline/highlights translations of `prose` (first iteration:
    // no `overview`). Absent for a model/locale falls back to the English
    // `prose` fields; consumers should never require a translation to exist.
    #[serde(default)]
    pub prose_locales: Option<BTreeMap<String, CatalogProseLocale>>,
    pub quants: Vec<CatalogQuant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum CatalogModelKind {
    #[default]
    AsrModel,
    CapabilityPack,
    TranslationModel,
}

/// Wire tags for a model's source-language parameter policy, reusing verbatim
/// the tags `LanguageCapability::mode` already serializes on
/// `/v1/capabilities` for the loaded pack (`crate::api::backend::mod`'s
/// `From<LanguageMode> for LanguageCapability`) -- the catalog and the
/// running-model capability surface stay one vocabulary for this axis instead
/// of drifting into two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogLanguageMode {
    /// Decode-time auto-detect plus explicit selection (multilingual Whisper).
    DetectAndSpecify,
    /// Self-detects internally; an explicit hint is rejected (Qwen3-ASR).
    DetectImplicit,
    /// Explicit selection required; `language_default` is used when unset
    /// (Cohere transcribe).
    SpecifyOnly,
    /// Intrinsically a single language; `language_default` names it
    /// (Moonshine, Whisper `*.en`, CTC families).
    FixedMonolingual,
    /// Intrinsically a fixed multilingual set with no per-request selection
    /// (X-ASR zh-en).
    FixedMultilingual,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogCapability {
    pub feature: String,
    pub role: CatalogCapabilityRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CatalogCapabilityRole {
    SpeakerEmbedder,
    SpeakerSegmenter,
    /// A forced-alignment refinement model for the `word-timestamps` feature
    /// (e.g. Qwen3-ForcedAligner-0.6B): consumes a finished transcript's text
    /// plus the source audio and replaces the model family's own approximate
    /// per-word timestamps with aligner-refined spans. Opt-in only; the
    /// family's own approximate timestamps remain the default.
    ForcedAligner,
    /// A punctuation-restoration model for the `punctuation` feature (e.g.
    /// FireRedPunc): a text-in/labels-out BERT classifier that adds Chinese
    /// full-width marks to an unpunctuated family's transcript in a
    /// finalize-only post-process. Opt-in and auto-gated on the ASR model's
    /// `emits_punctuation == Some(false)`; never re-punctuates a punctuating
    /// family.
    PunctuationRestorer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LicenseClass {
    Permissive,
    Noncommercial,
    Gated,
}

/// Whether the running build can use a catalog model, derived from its
/// `min_cli_version`. Models needing a newer OpenASR than the current build are
/// surfaced in listings as [`ModelAvailability::RequiresUpdate`] (not hidden) and
/// refused only at pull time — so an older client still *sees* newer models with a
/// clear "update to use" signal instead of a missing entry or a failed catalog load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelAvailability {
    /// This build satisfies the model's `min_cli_version`.
    Available,
    /// The model needs a newer OpenASR than the running build.
    RequiresUpdate {
        min_cli_version: String,
        current_cli_version: String,
    },
}

/// The OpenASR version of the running build (`CARGO_PKG_VERSION`), used to gate
/// catalog models against their `min_cli_version`.
pub fn current_cli_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

impl CatalogModel {
    pub fn is_market_listed(&self) -> bool {
        self.public
            && matches!(
                self.kind,
                CatalogModelKind::AsrModel | CatalogModelKind::TranslationModel
            )
    }

    /// Classify this model against the running build's version. The build must
    /// clear BOTH version floors the model declares: the publish-time
    /// `min_cli_version` and, when present, the author-set `min_core_version`
    /// runtime floor. The higher of the two unmet floors is reported as the
    /// version to update to. A malformed floor (already rejected at
    /// catalog-validation time) is treated leniently here as satisfied.
    ///
    /// Consumers: the pull path uses this in-repo to refuse a too-new model
    /// (`resolve_catalog_pull_with_profile`). The *listing* consumer — the model
    /// market that shows a too-new model with an "update to use" badge rather than
    /// hiding it — is the desktop/web app; it reads this classifier (or recomputes
    /// from the serialized `min_cli_version` / `min_core_version`). The catalog
    /// itself always loads regardless, so the app receives every model.
    pub fn availability(&self) -> ModelAvailability {
        let Some(current) = parse_semver_triplet(current_cli_version()) else {
            return ModelAvailability::Available;
        };
        // Both floors feed one "you need >= X" answer: keep only the unmet floors
        // and report whichever is highest as the version to update to.
        let unmet = [
            Some(self.min_cli_version.as_str()),
            self.min_core_version.as_deref(),
        ]
        .into_iter()
        .flatten()
        .filter_map(|raw| parse_semver_triplet(raw).map(|parsed| (parsed, raw)))
        .filter(|(parsed, _)| current < *parsed)
        .max_by(|left, right| left.0.cmp(&right.0));
        match unmet {
            Some((_, required)) => ModelAvailability::RequiresUpdate {
                min_cli_version: required.to_string(),
                current_cli_version: current_cli_version().to_string(),
            },
            None => ModelAvailability::Available,
        }
    }
}

impl ModelCatalog {
    /// Best-effort resolve a user-facing model ref -- an id, `pull_alias`, alias,
    /// or series ref, optionally carrying a `:quant` suffix -- to the public
    /// catalog model it names, for surfacing advertised metadata (languages,
    /// `language_mode`, `language_default`) in the CLI. The `:quant` suffix is
    /// stripped (quant does not change the language axis) and the default size is
    /// used for a bare series ref. Returns `None` when the ref matches no public
    /// model -- a local-only or staged (`public:false`) pack -- so callers fall
    /// back to core's fail-closed executor seam rather than inventing a code list.
    pub fn resolve_public_model(&self, model_ref: &str) -> Option<&CatalogModel> {
        let (base, _quant) = parse_catalog_pull_reference(model_ref.trim()).ok()?;
        resolve_catalog_model(self, base, None).ok()
    }

    pub fn capability_packs_for_feature(&self, feature: &str) -> Vec<&CatalogModel> {
        self.models
            .iter()
            .filter(|model| model.public)
            .filter(|model| model.kind == CatalogModelKind::CapabilityPack)
            .filter(|model| {
                model
                    .capability
                    .as_ref()
                    .is_some_and(|capability| capability.feature == feature)
            })
            .collect()
    }

    pub fn speaker_diarization_required_embedder_pack(&self) -> Option<&CatalogModel> {
        self.speaker_diarization_embedder_pack(CATALOG_SPEAKER_EMBEDDER_WESPEAKER_ID)
    }

    fn speaker_diarization_embedder_pack(&self, model_id: &str) -> Option<&CatalogModel> {
        self.capability_packs_for_feature(CATALOG_FEATURE_SPEAKER_DIARIZATION)
            .into_iter()
            .find(|model| {
                model.id == model_id
                    && model.capability.as_ref().is_some_and(|capability| {
                        capability.role == CatalogCapabilityRole::SpeakerEmbedder
                    })
            })
    }

    /// The forced-alignment capability pack for the `word-timestamps` feature
    /// (`--word-timestamps=aligned`), when the catalog carries one. Unlike
    /// diarization's single pinned embedder id, any public pack advertising
    /// `(word-timestamps, ForcedAligner)` qualifies -- there is exactly one
    /// today (Qwen3-ForcedAligner-0.6B) but callers should not hardcode its id.
    pub fn word_timestamps_forced_aligner_pack(&self) -> Option<&CatalogModel> {
        self.capability_packs_for_feature(CATALOG_FEATURE_WORD_TIMESTAMPS)
            .into_iter()
            .find(|model| {
                model.capability.as_ref().is_some_and(|capability| {
                    capability.role == CatalogCapabilityRole::ForcedAligner
                })
            })
    }

    /// The punctuation-restoration capability pack for the `punctuation`
    /// feature, when the catalog carries one. Any public pack advertising
    /// `(punctuation, PunctuationRestorer)` qualifies -- there is exactly one
    /// today (FireRedPunc) but callers should not hardcode its id (mirrors
    /// `word_timestamps_forced_aligner_pack`).
    pub fn punctuation_restorer_pack(&self) -> Option<&CatalogModel> {
        self.capability_packs_for_feature(CATALOG_FEATURE_PUNCTUATION)
            .into_iter()
            .find(|model| {
                model.capability.as_ref().is_some_and(|capability| {
                    capability.role == CatalogCapabilityRole::PunctuationRestorer
                })
            })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogProse {
    #[serde(default)]
    pub tagline: Option<String>,
    #[serde(default)]
    pub overview: Vec<String>,
    #[serde(default)]
    pub highlights: Vec<String>,
}

/// One locale's translation of [`CatalogProse`]. First iteration only covers
/// `tagline` + `highlights` (no `overview`); the publish pipeline
/// (`tooling/publish-model/scripts/_manifest.py`) machine-checks each
/// translation against the English source before it lands here (highlight
/// count, `**`/backtick/emoji parity per highlight, numeric-token parity, and
/// a `source_sha256` staleness check), so a stale or reformatted translation
/// fails catalog regeneration rather than shipping silently.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogProseLocale {
    #[serde(default)]
    pub tagline: Option<String>,
    #[serde(default)]
    pub highlights: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogQuant {
    pub quant: String,
    pub suffix: String,
    pub pull: String,
    pub filename: String,
    pub url: String,
    #[serde(default)]
    pub mirrors: Vec<CatalogMirror>,
    pub sha256: String,
    pub size_bytes: u64,
    #[serde(default)]
    // Generated from CatalogModel::recommended_quant, not an independent
    // authoring source.
    pub recommended: bool,
    #[serde(default)]
    pub perf: Option<CatalogQuantPerf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogMirror {
    pub source: String,
    pub url: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogQuantPerf {
    #[serde(default)]
    pub rtf_cpu: Option<f64>,
    #[serde(default)]
    pub rtf_metal: Option<f64>,
    #[serde(default)]
    pub peak_rss_bytes: Option<u64>,
    #[serde(default)]
    pub jfk_wer_vs_fp16: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogQuantRecommendationProfile {
    pub memory_budget_bytes: Option<u64>,
}

/// A downloadable GPU backend plugin pack (design D7: top-level `backends[]`,
/// authored from day one, no schema_version bump). Unlike a model — one `.oasr`
/// per quant — a backend is a SET of files staged into
/// `OPENASR_HOME/backends/<vendor>/<version>/` and registered with the ggml
/// backend registry at startup (with automatic CPU fallback). The type, pull
/// path, and load path are authored now so populating the catalog with real
/// packs (Phases 3-4) is the only later change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogBackend {
    pub id: String,
    pub vendor: CatalogBackendVendor,
    /// Pack version, pinned to the ggml commit the core was built from so a
    /// plugin is never loaded against a mismatched core ABI.
    pub version: String,
    pub display_name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// Device arch hints this pack targets (HIP `gfx` ids, CUDA SM numbers).
    /// Empty for cross-vendor (Vulkan) or CPU. Drives UI device-match, not the
    /// load decision (the ggml registry score-ranks what actually runs).
    #[serde(default)]
    pub targets: Vec<String>,
    #[serde(default)]
    pub min_driver: Option<String>,
    pub min_cli_version: String,
    pub files: Vec<CatalogBackendFile>,
}

/// One file in a [`CatalogBackend`] pack: the `ggml-<vendor>` plugin, a runtime
/// satellite DLL/shared object, or an archive extracted post-verify.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogBackendFile {
    pub filename: String,
    pub url: String,
    #[serde(default)]
    pub mirrors: Vec<CatalogMirror>,
    pub sha256: String,
    pub size_bytes: u64,
    #[serde(default)]
    pub role: CatalogBackendFileRole,
    /// For `role = archive`: the pack-relative directory the archive extracts
    /// into (e.g. `rocblas/library` for the rocBLAS Tensile set). Ignored for
    /// plugin/runtime files, which stage at the pack root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extract_subdir: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum CatalogBackendFileRole {
    /// A runtime DLL/shared object staged as-is next to the plugin.
    #[default]
    Runtime,
    /// The `ggml-<vendor>` plugin the registry dlopens to register the backend.
    Plugin,
    /// An archive (zip) whose contents are extracted (post sha256 + signature
    /// verify) into `extract_subdir` — e.g. the rocBLAS Tensile `library/` set.
    Archive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CatalogBackendVendor {
    Cpu,
    Vulkan,
    Hip,
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogPullRequest {
    pub reference: String,
    pub quant: Option<String>,
    pub size: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCatalogPull {
    pub requested: String,
    pub model_id: String,
    pub display_name: String,
    pub quant: String,
    pub suffix: String,
    pub pull: String,
    pub filename: String,
    pub url: String,
    pub mirrors: Vec<CatalogMirror>,
    pub hf_revision: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub license: String,
    pub license_url: String,
    pub license_class: LicenseClass,
}

impl ResolvedCatalogPull {
    /// Build a `ResolvedCatalogPull` from a matched `(model, quant)` pair.
    /// `requested` is the only field that isn't derived from `model`/`quant`
    /// -- callers resolving a user-typed reference pass that reference
    /// through verbatim; callers matching by some other identity (e.g. a
    /// local file's sha256/size digest) pass `quant.pull.clone()` so
    /// `requested` still reads as a valid pull spec. Shared by
    /// [`resolve_catalog_pull`] and [`crate::pull::resolve_catalog_pull_by_file_digest`]
    /// so the 12 fields mapped straight from `model`/`quant` can't drift
    /// between the two call sites.
    pub fn from_model_and_quant(
        model: &CatalogModel,
        quant: &CatalogQuant,
        requested: String,
    ) -> Self {
        Self {
            requested,
            model_id: model.id.clone(),
            display_name: model.display_name.clone(),
            quant: quant.quant.clone(),
            suffix: quant.suffix.clone(),
            pull: quant.pull.clone(),
            filename: quant.filename.clone(),
            url: quant.url.clone(),
            mirrors: quant.mirrors.clone(),
            hf_revision: model.hf_revision.clone(),
            sha256: quant.sha256.clone(),
            size_bytes: quant.size_bytes,
            license: model.license.clone(),
            license_url: model.license_url.clone(),
            license_class: model.license_class.clone(),
        }
    }
}

/// A resolved backend-pack pull: the pack identity plus the files to download
/// into `OPENASR_HOME/backends/<vendor>/<version>/`. The download orchestration
/// fetches each file (sha256-verified, then [`crate::pull::preflight_backend_file`]),
/// and archive files extract into their `extract_subdir`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedCatalogBackendPull {
    pub backend_id: String,
    pub vendor: CatalogBackendVendor,
    pub version: String,
    pub display_name: String,
    pub files: Vec<CatalogBackendFile>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BackendResolutionError {
    #[error("The catalog declares no downloadable backends.")]
    NoBackends,
    #[error("Unknown backend '{reference}'. Available backends: {available}.")]
    UnknownBackend {
        reference: String,
        available: String,
    },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModelResolutionError {
    #[error("Invalid model reference '{0}'. Use model or model:tag.")]
    InvalidRef(String),
    #[error("Unknown model: {0}\nRun `openasr list` to see available models.")]
    UnknownModel(String),
    #[error(
        "Model family '{family}' does not have variant tag '{tag}'. Available tags: {available_tags}."
    )]
    UnknownVariantTag {
        family: String,
        tag: String,
        available_tags: String,
    },
    #[error(
        "Model reference '{model_ref}' is ambiguous. Use an explicit tag such as one of: {available_refs}."
    )]
    AmbiguousModelRef {
        model_ref: String,
        available_refs: String,
    },
    #[error(
        "Model family '{family}' has default variant '{default_variant}', but no matching registry card was found."
    )]
    MissingDefaultVariant {
        family: String,
        default_variant: String,
    },
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("Model registry directory was not found: {0}")]
    MissingDirectory(PathBuf),
    #[error("Could not read model registry directory '{path}': {source}")]
    ReadDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Could not read model card '{path}': {source}")]
    ReadCard {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Could not parse model card '{path}': {source}")]
    ParseCard {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("Invalid model card '{path}': {message}")]
    ValidateCard { path: PathBuf, message: String },
    #[error("Invalid model registry: duplicate model id '{model_id}'")]
    DuplicateModelId { model_id: String },
    #[error("Invalid model registry: duplicate variant '{family}:{tag}'")]
    DuplicateVariant { family: String, tag: String },
    #[error(
        "Invalid model registry: family '{family}' default_variant '{default_variant}' does not match any variant tag"
    )]
    MissingDefaultVariant {
        family: String,
        default_variant: String,
    },
    #[error(
        "Invalid model registry: family '{family}' has conflicting default_variant values: '{left}' and '{right}'"
    )]
    ConflictingDefaultVariant {
        family: String,
        left: String,
        right: String,
    },
}

#[derive(Debug, Error)]
pub enum CatalogError {
    #[error(
        "Unsupported model catalog schema_version {found}; update OpenASR to read this catalog."
    )]
    UnsupportedSchema { found: u32 },
    #[error("Could not read model catalog '{catalog_source}': {message}")]
    ReadCatalog {
        catalog_source: String,
        message: String,
    },
    #[error("Could not parse model catalog '{catalog_source}': {source_error}")]
    ParseCatalog {
        catalog_source: String,
        #[source]
        source_error: serde_json::Error,
    },
    #[error("Could not cache model catalog at '{path}': {source}")]
    CacheCatalog {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Could not create OpenASR home directory '{path}': {source}")]
    CreateHome {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Model catalog security check failed for '{catalog_source}': {message}")]
    CatalogSecurity {
        catalog_source: String,
        message: String,
    },
    #[error("Invalid model catalog: {0}")]
    InvalidCatalog(String),
    #[error(
        "Invalid pull reference '{0}'. Use <id> or <id>:<quant>, for example moonshine-tiny:q8."
    )]
    InvalidPullReference(String),
    #[error("Model '{reference}' was not found in the model catalog.")]
    UnknownModel { reference: String },
    #[error(
        "Model '{model_id}' requires OpenASR >= {min_cli_version} (this build is {current_cli_version}). Update OpenASR to use it."
    )]
    ModelRequiresNewerCli {
        model_id: String,
        min_cli_version: String,
        current_cli_version: String,
    },
    #[error("Model reference '{reference}' is ambiguous. Use one of: {available}.")]
    AmbiguousModelRef {
        reference: String,
        available: String,
    },
    #[error("Model '{model_id}' does not provide quant '{quant}'. Available pulls: {available}.")]
    UnknownQuant {
        model_id: String,
        quant: String,
        available: String,
    },
    #[error(
        "Catalog model '{model_id}' has recommended_quant '{quant}', but no matching quant entry."
    )]
    MissingRecommendedQuant { model_id: String, quant: String },
    #[error(
        "Conflicting quant selection: reference requested '{reference_quant}' but --quant requested '{option_quant}'."
    )]
    ConflictingQuant {
        reference_quant: String,
        option_quant: String,
    },
}

#[derive(Debug, Error)]
pub enum RuntimeModelResolutionError {
    #[error(transparent)]
    Registry(#[from] ModelResolutionError),
    #[error(transparent)]
    Catalog(#[from] CatalogError),
}

/// Environment override that points the runtime model registry at an on-disk
/// `model-registry/models` directory instead of deriving it from the signed
/// catalog. Set it for fast `cargo run` iteration against a working tree; it is
/// NEVER set in a bundled/release environment (see [`runtime_registry`]).
pub const OPENASR_REGISTRY_DIR_ENV: &str = "OPENASR_REGISTRY_DIR";

/// The on-disk registry directory a WORKING TREE resolves to, relative to the
/// current directory. This is a build-time / tooling / test convenience only --
/// it is NOT a release runtime source (a deployed binary ships no
/// `model-registry/` tree). The release runtime resolves the registry from the
/// signed catalog via [`runtime_registry`]; the only on-disk path the runtime
/// ever reads is an explicit [`OPENASR_REGISTRY_DIR_ENV`] override.
pub fn default_registry_dir() -> PathBuf {
    PathBuf::from("model-registry/models")
}

/// The explicit dev override directory, when [`OPENASR_REGISTRY_DIR_ENV`] is set
/// to a non-empty value. Absent otherwise, which drives the runtime onto the
/// catalog-derived registry.
fn registry_dir_override() -> Option<PathBuf> {
    std::env::var_os(OPENASR_REGISTRY_DIR_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// The registry directory a test/tooling harness reads from the committed
/// working tree (an absolute path, independent of the process cwd). Kept out of
/// the release runtime deliberately: `env!("CARGO_MANIFEST_DIR")` is a
/// build-machine path that does not exist on a user's device.
#[cfg(test)]
pub(crate) fn test_model_registry_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry/models")
}

pub fn default_catalog_url() -> &'static str {
    DEFAULT_CATALOG_URL
}

// PARITY: must match the desktop TypeScript client's `canonicalQuantTag` exactly.
pub fn canonical_quant_tag(tag: &str) -> &str {
    match tag.trim() {
        "q8" | "q8_0" => "q8_0",
        "q4" | "q4_k" | "q4_k_m" => "q4_k",
        "q3" | "q3_k" => "q3_k",
        "fp16" => "fp16",
        other => other,
    }
}

// PARITY: keep in lockstep with the desktop TypeScript client's `recommendedQuantForDevice`.
// Same contract: pick the
// highest-quality quant (fp16 > q8_0 > q4_k) whose peak RSS fits the budget,
// else the catalog default.
pub fn recommend_catalog_quant(
    model: &CatalogModel,
    profile: CatalogQuantRecommendationProfile,
) -> Result<&CatalogQuant, CatalogError> {
    let recommended = resolve_catalog_quant(model, None)?;
    let Some(memory_budget_bytes) = profile.memory_budget_bytes.filter(|budget| *budget > 0) else {
        return Ok(recommended);
    };
    let Some(recommended_peak_rss) = catalog_quant_peak_rss_bytes(recommended) else {
        return Ok(recommended);
    };
    if recommended_peak_rss <= memory_budget_bytes {
        return Ok(recommended);
    }

    Ok(model
        .quants
        .iter()
        .filter(|quant| {
            catalog_quant_peak_rss_bytes(quant)
                .is_some_and(|peak_rss| peak_rss <= memory_budget_bytes)
        })
        .max_by(|left, right| {
            catalog_quant_quality_rank(left)
                .cmp(&catalog_quant_quality_rank(right))
                .then_with(|| {
                    catalog_quant_peak_rss_bytes(right).cmp(&catalog_quant_peak_rss_bytes(left))
                })
        })
        .unwrap_or(recommended))
}

pub fn default_catalog_cache_path(openasr_home: impl AsRef<Path>) -> PathBuf {
    openasr_home.as_ref().join("catalog.json")
}

/// Loads the model catalog from `catalog_url` (default: [`DEFAULT_CATALOG_URL`]),
/// always through the same fail-closed signature-verification pipeline --
/// remote (`https://`), local (`file://`/bare filesystem path), and the
/// on-disk signed cache all require a matching, valid `catalog.signature.json`
/// sidecar. There is no unsigned/trust-bypass path: a local catalog source is
/// only ever reachable via an explicit `catalog_url` override, and whoever
/// supplies it must sign it (see [`catalog_security::verify_local_catalog_signature_manifest`]
/// and the local-dev key it accepts in addition to the production key).
pub fn load_model_catalog(
    catalog_url: Option<&str>,
    openasr_home: impl AsRef<Path>,
) -> Result<ModelCatalog, CatalogError> {
    let home = openasr_home.as_ref();
    let cache_path = default_catalog_cache_path(home);
    let source = catalog_url.unwrap_or(DEFAULT_CATALOG_URL);

    match read_catalog_source(source) {
        Ok(contents) => match read_and_verify_catalog_manifest(source, home, &contents) {
            Ok(verified) => {
                let catalog = parse_model_catalog(&contents, source)?;
                cache_catalog(home, &cache_path, &contents)?;
                cache_catalog_security(home, &verified.manifest_contents, &verified.signature)?;
                Ok(catalog)
            }
            Err(error) => load_cached_signed_catalog(source, home, &cache_path, error),
        },
        Err(error) => load_cached_signed_catalog(source, home, &cache_path, error),
    }
}

/// Loads a LOCAL catalog file directly from `path`, verifying its adjacent
/// `catalog.signature.json` sidecar against `expected_catalog_url` -- which is
/// deliberately NOT required to be a `file://`/path form of `path` itself.
///
/// This exists for exactly one caller: the CLI's "run from a repo checkout"
/// dev convenience that auto-discovers `model-registry/catalog.json` relative
/// to the current directory / build tree with no `OPENASR_CATALOG_URL`
/// override set (see `openasr-cli`'s `catalog_cli::load_cli_model_catalog`).
/// That file and its committed sidecar ARE the pre-deployment source of truth
/// for the canonical [`DEFAULT_CATALOG_URL`] identity -- the same relationship
/// [`load_embedded_signed_catalog`] has to the binary's embedded snapshot.
///
/// The trust roots are chosen from `expected_catalog_url` itself, through the
/// same [`catalog_security::classify_catalog_identity`] used for every other
/// source (see [`verify_catalog_manifest_for_source`]): when the caller
/// asserts the canonical production (`https://`) identity -- as the repo
/// auto-discovery of `model-registry/catalog.json` does -- ONLY the
/// production key verifies, exactly like a real HTTPS/cached/embedded
/// production catalog. The committed `model-registry/catalog.signature.json`
/// is production-signed, so this is zero-impact for the real, deployed pair.
/// The widely-known public local-dev key is accepted only when
/// `expected_catalog_url` is itself a non-production (local) identity -- i.e.
/// an explicit `--catalog-url file://...`/`OPENASR_CATALOG_URL` override,
/// which goes through [`load_model_catalog`], not this function. A local-dev
/// key bound to the production identity must never be treated as a stand-in
/// for the real production catalog (that would let a malicious CWD override
/// what a careless `cd`-and-run sees as the canonical model list/pull
/// targets); see `registry/tests/catalog.rs`'s
/// `local_catalog_auto_discovery_rejects_dev_key_bound_to_production_identity`.
///
/// Runs the same anti-rollback epoch floor as every other source, scoped to
/// `openasr_home` -- see [`load_embedded_signed_catalog`]'s doc comment for
/// why that is a freshness floor, not a confidentiality mechanism -- except
/// that a local-dev-key verification never touches that floor at all (see
/// [`catalog_security::participates_in_epoch_floor`]).
pub fn load_local_catalog_file_with_identity(
    path: &Path,
    expected_catalog_url: &str,
    openasr_home: impl AsRef<Path>,
) -> Result<ModelCatalog, CatalogError> {
    let home = openasr_home.as_ref();
    let source_label = path.display().to_string();
    let contents = fs::read_to_string(path).map_err(|error| CatalogError::ReadCatalog {
        catalog_source: source_label.clone(),
        message: error.to_string(),
    })?;
    let manifest_path = path.with_file_name(catalog_security::CATALOG_SIGNATURE_FILE_NAME);
    let manifest_contents =
        fs::read_to_string(&manifest_path).map_err(|error| CatalogError::CatalogSecurity {
            catalog_source: source_label.clone(),
            message: format!(
                "could not read signature manifest '{}': {error}",
                manifest_path.display()
            ),
        })?;
    let verified =
        verify_catalog_manifest_for_source(expected_catalog_url, &contents, &manifest_contents)
            .map_err(|error| CatalogError::CatalogSecurity {
                catalog_source: source_label.clone(),
                message: error.to_string(),
            })?;
    catalog_security::enforce_catalog_epoch_for_verified(home, &verified).map_err(|error| {
        CatalogError::CatalogSecurity {
            catalog_source: source_label.clone(),
            message: error.to_string(),
        }
    })?;
    let catalog = parse_model_catalog(&contents, &source_label)?;
    let cache_path = default_catalog_cache_path(home);
    cache_catalog(home, &cache_path, &contents)?;
    cache_catalog_security(home, &manifest_contents, &verified)?;
    Ok(catalog)
}

/// Whether the runtime should prefer the embedded catalog snapshot over the
/// network/cache tier, chosen purely by catalog_epoch freshness. Split out as a
/// pure function so the epoch-max policy is unit-testable without live signing.
///
/// SECURITY: this is a freshness preference only, never a rollback relaxation.
/// The embedded snapshot is only ever *loaded* through
/// [`load_embedded_signed_catalog`], which runs the same `enforce_catalog_epoch`
/// rollback guard as any other source (it refuses an embedded epoch below the
/// stored floor). Here we additionally require the embedded epoch to be STRICTLY
/// newer than the epoch the network/cache tier just established, so a lower/equal
/// embedded snapshot can never displace a newer catalog the device already has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeCatalogChoice {
    Network,
    Embedded,
}

fn choose_runtime_catalog(
    network_epoch: Option<u64>,
    embedded_epoch: Option<u64>,
) -> RuntimeCatalogChoice {
    match (embedded_epoch, network_epoch) {
        (Some(embedded), Some(network)) if embedded > network => RuntimeCatalogChoice::Embedded,
        _ => RuntimeCatalogChoice::Network,
    }
}

/// Resolve the catalog the runtime should use, picking the signature-verified
/// source with the HIGHEST `catalog_epoch` across the network/on-disk-cache tier
/// ([`load_model_catalog`], which already falls back to the on-disk cache and the
/// embedded snapshot when offline) and the embedded signed snapshot as an
/// epoch-max floor.
///
/// Effect: in a release build the embedded epoch is <= production, so the network
/// catalog wins and users get the latest models; in a local preview build that
/// embeds a catalog AHEAD of production, the embedded snapshot wins (test
/// unreleased models with zero infrastructure). Offline with no cache, the
/// embedded snapshot is the permanent floor so the runtime still starts.
///
/// Scoped to the canonical [`default_catalog_url`]: an explicit override URL is
/// honored verbatim, never silently replaced by the bundled catalog. Anti-rollback
/// is unchanged (see [`choose_runtime_catalog`]).
pub fn resolve_runtime_catalog(
    catalog_url: Option<&str>,
    openasr_home: impl AsRef<Path>,
) -> Result<ModelCatalog, CatalogError> {
    let home = openasr_home.as_ref();
    let network = load_model_catalog(catalog_url, home)?;
    // The epoch-max embedded floor only applies to the canonical catalog: an
    // explicit override is authoritative on its own.
    if catalog_url.is_some_and(|url| url != DEFAULT_CATALOG_URL) {
        return Ok(network);
    }
    let embedded_epoch = embedded_catalog_fingerprint().ok().map(|(_, epoch)| epoch);
    // The stored epoch reflects what the network/cache tier just enforced/recorded.
    let network_epoch =
        catalog_security::read_catalog_epoch(&catalog_security::default_catalog_epoch_path(home))
            .ok()
            .flatten();
    match choose_runtime_catalog(network_epoch, embedded_epoch) {
        RuntimeCatalogChoice::Embedded => Ok(load_embedded_signed_catalog(home).unwrap_or(network)),
        RuntimeCatalogChoice::Network => Ok(network),
    }
}

pub fn parse_model_catalog(contents: &str, source: &str) -> Result<ModelCatalog, CatalogError> {
    let catalog: ModelCatalog =
        serde_json::from_str(contents).map_err(|source_error| CatalogError::ParseCatalog {
            catalog_source: source.to_string(),
            source_error,
        })?;
    validate_model_catalog(&catalog)?;
    Ok(catalog)
}

pub fn resolve_catalog_pull(
    catalog: &ModelCatalog,
    request: &CatalogPullRequest,
) -> Result<ResolvedCatalogPull, CatalogError> {
    resolve_catalog_pull_with_profile(catalog, request, None)
}

/// Resolve a backend reference (the backend `id`) against the catalog's
/// `backends[]` to the pack to download. Errors list the available backend ids
/// so a typo gets an actionable message, mirroring model resolution.
pub fn resolve_catalog_backend_pull(
    catalog: &ModelCatalog,
    reference: &str,
) -> Result<ResolvedCatalogBackendPull, BackendResolutionError> {
    if catalog.backends.is_empty() {
        return Err(BackendResolutionError::NoBackends);
    }
    let reference = reference.trim();
    let backend = catalog
        .backends
        .iter()
        .find(|backend| backend.id == reference)
        .ok_or_else(|| BackendResolutionError::UnknownBackend {
            reference: reference.to_string(),
            available: catalog
                .backends
                .iter()
                .map(|backend| backend.id.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        })?;
    Ok(ResolvedCatalogBackendPull {
        backend_id: backend.id.clone(),
        vendor: backend.vendor,
        version: backend.version.clone(),
        display_name: backend.display_name.clone(),
        files: backend.files.clone(),
    })
}

/// Like [`resolve_catalog_pull`], but when the request carries no explicit quant
/// and `device_profile` is `Some`, the default quant becomes the device-recommended
/// one (the largest quant whose peak RSS fits the budget) instead of the catalog's
/// static `recommended_quant`. An explicit `:quant` / `--quant` always wins.
pub fn resolve_catalog_pull_with_profile(
    catalog: &ModelCatalog,
    request: &CatalogPullRequest,
    device_profile: Option<CatalogQuantRecommendationProfile>,
) -> Result<ResolvedCatalogPull, CatalogError> {
    let requested = request.reference.trim();
    if requested.is_empty() {
        return Err(CatalogError::InvalidPullReference(
            request.reference.clone(),
        ));
    }
    let (model_ref, reference_quant) = parse_catalog_pull_reference(requested)?;
    let quant_ref = match (
        reference_quant,
        request
            .quant
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty()),
    ) {
        (Some(left), Some(right)) => {
            if canonical_quant_tag(left) != canonical_quant_tag(right) {
                return Err(CatalogError::ConflictingQuant {
                    reference_quant: left.to_string(),
                    option_quant: right.to_string(),
                });
            }
            Some(canonical_quant_tag(left).to_string())
        }
        (Some(value), _) | (_, Some(value)) => Some(canonical_quant_tag(value).to_string()),
        (None, None) => None,
    };
    let model = resolve_catalog_model(catalog, model_ref, request.size.as_deref())?;
    // Forward-compat gate: the catalog lists models newer than this build can run
    // (so the market can surface them as "update to use"), but actually pulling one
    // is refused with a clear message rather than downloading a pack we can't load.
    if let ModelAvailability::RequiresUpdate {
        min_cli_version,
        current_cli_version,
    } = model.availability()
    {
        return Err(CatalogError::ModelRequiresNewerCli {
            model_id: model.id.clone(),
            min_cli_version,
            current_cli_version,
        });
    }
    let quant = match (quant_ref.as_deref(), device_profile) {
        // No explicit quant + a device profile: pick the device-recommended quant.
        (None, Some(profile)) => recommend_catalog_quant(model, profile)?,
        // Explicit quant, or no profile: keep the static catalog default behavior.
        (explicit, _) => resolve_catalog_quant(model, explicit)?,
    };

    Ok(ResolvedCatalogPull::from_model_and_quant(
        model,
        quant,
        requested.to_string(),
    ))
}

pub fn load_registry(path: impl AsRef<Path>) -> Result<Vec<ModelCard>, RegistryError> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(RegistryError::MissingDirectory(path.to_path_buf()));
    }

    let entries = fs::read_dir(path).map_err(|source| RegistryError::ReadDirectory {
        path: path.to_path_buf(),
        source,
    })?;
    let mut cards = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|source| RegistryError::ReadDirectory {
            path: path.to_path_buf(),
            source,
        })?;
        let card_path = entry.path();
        if card_path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
            continue;
        }

        let contents =
            fs::read_to_string(&card_path).map_err(|source| RegistryError::ReadCard {
                path: card_path.clone(),
                source,
            })?;
        let card: ModelCard =
            toml::from_str(&contents).map_err(|source| RegistryError::ParseCard {
                path: card_path.clone(),
                source,
            })?;
        validation::validate_card(&card_path, &card)?;
        cards.push(card);
    }

    cards.sort_by(|left: &ModelCard, right| {
        match (
            left.id.as_str() == DEFAULT_MODEL_ID,
            right.id.as_str() == DEFAULT_MODEL_ID,
        ) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => left.id.cmp(&right.id),
        }
    });
    validation::validate_unique_ids(&cards)?;
    validation::validate_variant_index(&cards)?;
    Ok(cards)
}

#[derive(Debug, Error)]
pub enum RuntimeRegistryError {
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error(transparent)]
    Catalog(#[from] CatalogError),
}

/// The runtime model registry -- the flat model-id list plus display metadata
/// every server/CLI resolution path needs -- resolved so a RELEASE binary is
/// self-contained and never depends on a source-tree `model-registry/` path.
///
/// Resolution order:
/// 1. [`OPENASR_REGISTRY_DIR_ENV`] override (dev/`cargo run` fast iteration) ->
///    load the on-disk cards. Never set in a bundle/release.
/// 2. Otherwise DERIVE the cards from the signed model catalog: the `catalog` the
///    caller already resolved (carrying the epoch-max embedded floor from
///    [`resolve_runtime_catalog`]) when present, else the signature-verified
///    embedded snapshot ([`load_embedded_signed_catalog`]) as the permanent
///    offline floor. No filesystem source dependency, so a deployed binary with
///    no `model-registry/` directory still resolves and lists models.
///
/// Family/alias resolution stays catalog-first (`resolve_runtime_model_ref`); the
/// derived registry only supplies the flat id list and display metadata, so each
/// derived card is its own family (`family_name() == id`, matching the committed
/// cards) and never collapses `whisper-*` into one ambiguous family.
pub fn runtime_registry(
    catalog: Option<&ModelCatalog>,
) -> Result<Vec<ModelCard>, RuntimeRegistryError> {
    if let Some(dir) = registry_dir_override() {
        return Ok(load_registry(dir)?);
    }
    match catalog {
        Some(catalog) => Ok(model_cards_from_catalog(catalog)?),
        None => {
            let home = crate::home::openasr_home().map_err(|_| {
                RegistryError::MissingDirectory(PathBuf::from(
                    "<embedded catalog: OPENASR_HOME unresolved>",
                ))
            })?;
            let embedded = load_embedded_signed_catalog(&home)?;
            Ok(model_cards_from_catalog(&embedded)?)
        }
    }
}

/// Derive the runtime [`ModelCard`] list from a resolved signed catalog. Every
/// `public` catalog entry becomes one card; the derivation is the empirically
/// verified 1:1 projection of the on-disk cards:
/// - `family = None` so `family_name()` falls back to `id` (the committed cards
///   set no family; using `catalog.family` would collapse `whisper-*` into one
///   family and break resolution/listing -- see [`runtime_registry`]).
/// - `variant.quantization = recommended_quant`; tag/format/role and
///   default_variant/backend/quality_profile are the same constants the on-disk
///   cards default to.
///
/// Non-public (staged) entries are intentionally excluded: the runtime registry
/// only advertises released models.
pub fn model_cards_from_catalog(catalog: &ModelCatalog) -> Result<Vec<ModelCard>, RegistryError> {
    let mut cards: Vec<ModelCard> = catalog
        .models
        .iter()
        .filter(|model| model.public)
        .map(model_card_from_catalog)
        .collect();
    cards.sort_by(|left: &ModelCard, right| {
        match (
            left.id.as_str() == DEFAULT_MODEL_ID,
            right.id.as_str() == DEFAULT_MODEL_ID,
        ) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => left.id.cmp(&right.id),
        }
    });
    validation::validate_unique_ids(&cards)?;
    validation::validate_variant_index(&cards)?;
    Ok(cards)
}

fn model_card_from_catalog(model: &CatalogModel) -> ModelCard {
    ModelCard {
        id: model.id.clone(),
        // Deliberately None: family_name() falls back to id, keeping each model
        // its own family exactly like the committed cards.
        family: None,
        default_variant: Some(default_model_variant_tag()),
        variant: Some(ModelVariantMetadata {
            tag: default_model_variant_tag(),
            format: default_model_variant_format(),
            quantization: Some(model.recommended_quant.clone()),
            role: default_model_variant_role(),
        }),
        display_name: model.display_name.clone(),
        backend: default_model_backend(),
        task: default_model_task(),
        languages: model.languages.clone(),
        size: model.size.clone(),
        recommended_hardware: default_model_recommended_hardware(),
        license: model.license.clone(),
        features: default_model_features(),
        quality_profile: default_model_quality_profile(),
        source: format!(
            "Published OpenASR packs: {HUGGING_FACE_BASE_URL}{}",
            model.hf_repo
        ),
    }
}

fn read_catalog_source(source: &str) -> Result<String, CatalogError> {
    // Transport dispatch shares `classify_catalog_identity` with trust-root
    // selection (`verify_catalog_manifest_for_source`) so the two can never
    // drift apart on a future scheme -- see that function's doc comment.
    if catalog_security::classify_catalog_identity(source)
        == catalog_security::CatalogSourceKind::Remote
    {
        let client = http::blocking_client(CATALOG_HTTP_CONNECT_TIMEOUT, CATALOG_HTTP_TIMEOUT)
            .map_err(|error| CatalogError::ReadCatalog {
                catalog_source: source.to_string(),
                message: http::error_message(&error),
            })?;
        // The catalog (and its sibling signature manifest, which also flows
        // through this function) is served from the OpenASR catalog endpoint
        // (Cloudflare), never Hugging Face. Only the transport host is rewritten:
        // `source` stays the canonical, signed catalog_url everywhere it feeds
        // verification (see `read_and_verify_catalog_manifest`), so a proxy cannot
        // substitute a tampered catalog. Unlike weight downloads (pull.rs), the
        // catalog uses a redirect-following client and the endpoint serves bytes
        // directly, so the per-hop CDN rewrite used by weight downloads is
        // deliberately NOT applied here.
        let response = client
            .get(http::apply_catalog_endpoint(source).as_str())
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .map_err(|error| CatalogError::ReadCatalog {
                catalog_source: source.to_string(),
                message: http::error_message(&error),
            })?;
        return response.text().map_err(|error| CatalogError::ReadCatalog {
            catalog_source: source.to_string(),
            message: http::error_message(&error),
        });
    }

    if let Some(path) = source.strip_prefix("file://") {
        return fs::read_to_string(path).map_err(|error| CatalogError::ReadCatalog {
            catalog_source: source.to_string(),
            message: error.to_string(),
        });
    }

    if source.starts_with("http://") {
        return Err(CatalogError::ReadCatalog {
            catalog_source: source.to_string(),
            message: "catalog URLs must use https://; http:// is not accepted".to_string(),
        });
    }

    fs::read_to_string(source).map_err(|error| CatalogError::ReadCatalog {
        catalog_source: source.to_string(),
        message: error.to_string(),
    })
}

struct VerifiedCatalogManifestContents {
    manifest_contents: String,
    signature: catalog_security::VerifiedCatalogSignature,
}

/// Selects which signing keys a `catalog_url`/identity may be trusted under,
/// via the single shared [`catalog_security::classify_catalog_identity`]:
/// [`catalog_security::CatalogSourceKind::Remote`] (`https://`) sources are
/// restricted to the production-only root (the widely-known local-dev key
/// must never authorize a network catalog), while
/// [`catalog_security::CatalogSourceKind::Local`] (`file://`, a bare
/// filesystem path, or any other non-production identity -- i.e. anything
/// reached only through an explicit local `catalog_url` override, or asserted
/// by a caller as a non-production identity) additionally accepts the public
/// local-dev key. See the doc comment on `CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID`
/// for why that key carries no confidentiality risk, and
/// [`classify_catalog_identity`]'s doc comment for why `read_catalog_source`
/// must classify through the same function.
///
/// [`classify_catalog_identity`]: catalog_security::classify_catalog_identity
fn verify_catalog_manifest_for_source(
    source: &str,
    catalog_contents: &str,
    manifest_contents: &str,
) -> Result<catalog_security::VerifiedCatalogSignature, catalog_security::CatalogSecurityError> {
    match catalog_security::classify_catalog_identity(source) {
        catalog_security::CatalogSourceKind::Remote => {
            catalog_security::verify_catalog_signature_manifest(
                catalog_contents,
                manifest_contents,
                source,
            )
        }
        catalog_security::CatalogSourceKind::Local => {
            catalog_security::verify_local_catalog_signature_manifest(
                catalog_contents,
                manifest_contents,
                source,
            )
        }
    }
}

fn read_and_verify_catalog_manifest(
    source: &str,
    home: &Path,
    contents: &str,
) -> Result<VerifiedCatalogManifestContents, CatalogError> {
    let manifest_source = catalog_security::catalog_signature_source(source);
    let manifest_contents =
        read_catalog_source(&manifest_source).map_err(|error| CatalogError::CatalogSecurity {
            catalog_source: source.to_string(),
            message: error.to_string(),
        })?;
    let verified = match verify_catalog_manifest_for_source(source, contents, &manifest_contents) {
        Ok(verified) => verified,
        Err(error) => {
            return Err(CatalogError::CatalogSecurity {
                catalog_source: source.to_string(),
                message: error.to_string(),
            });
        }
    };
    catalog_security::enforce_catalog_epoch_for_verified(home, &verified).map_err(|error| {
        CatalogError::CatalogSecurity {
            catalog_source: source.to_string(),
            message: error.to_string(),
        }
    })?;
    Ok(VerifiedCatalogManifestContents {
        manifest_contents,
        signature: verified,
    })
}

fn load_cached_signed_catalog(
    source: &str,
    home: &Path,
    cache_path: &Path,
    error: CatalogError,
) -> Result<ModelCatalog, CatalogError> {
    match load_signed_catalog_from_cache(source, home, cache_path, &error) {
        Ok(catalog) => Ok(catalog),
        Err(cache_error) => {
            // Final tier: the signed catalog snapshot compiled into the binary, so
            // a fresh *offline* install with no network and no on-disk cache still
            // shows the (signature-verified) model list. Scoped to the canonical
            // default catalog — an explicit OPENASR_CATALOG_URL override is honoured,
            // not silently replaced with the bundled official catalog.
            if source == DEFAULT_CATALOG_URL
                && let Ok(catalog) = load_embedded_signed_catalog(home)
            {
                return Ok(catalog);
            }
            Err(cache_error)
        }
    }
}

fn load_signed_catalog_from_cache(
    source: &str,
    home: &Path,
    cache_path: &Path,
    error: &CatalogError,
) -> Result<ModelCatalog, CatalogError> {
    let cached =
        fs::read_to_string(cache_path).map_err(|cache_error| CatalogError::ReadCatalog {
            catalog_source: source.to_string(),
            message: format!(
                "{error}; no usable signed cache at '{}': {cache_error}",
                cache_path.display()
            ),
        })?;
    read_and_verify_cached_catalog_manifest(source, home, &cached, error)?;
    parse_model_catalog(&cached, &cache_path.display().to_string())
}

/// Load the signed catalog snapshot embedded in the binary at build time. Used as
/// the last-resort offline fallback (after the network source and the on-disk
/// cache) so a device that has never been online still sees the model list. The
/// embedded bytes are signature-verified against the canonical [`DEFAULT_CATALOG_URL`]
/// and run through the same epoch-rollback guard as any other source, so a stale
/// snapshot can never downgrade a newer catalog the device already cached.
///
/// Also the CLI's network-free source for advertised model metadata (the
/// `openasr show` language block and the `transcribe --language` pre-check): those
/// must never trigger a catalog download, so they prefer a local/env override and
/// fall back to this embedded snapshot rather than [`load_model_catalog`].
pub fn load_embedded_signed_catalog(home: &Path) -> Result<ModelCatalog, CatalogError> {
    let verified = catalog_security::verify_catalog_signature_manifest(
        EMBEDDED_CATALOG_JSON,
        EMBEDDED_CATALOG_SIGNATURE_JSON,
        DEFAULT_CATALOG_URL,
    )
    .map_err(|error| CatalogError::CatalogSecurity {
        catalog_source: DEFAULT_CATALOG_URL.to_string(),
        message: format!("embedded catalog rejected: {error}"),
    })?;
    catalog_security::enforce_catalog_epoch_for_verified(home, &verified).map_err(|error| {
        CatalogError::CatalogSecurity {
            catalog_source: DEFAULT_CATALOG_URL.to_string(),
            message: format!("embedded catalog rejected: {error}"),
        }
    })?;
    parse_model_catalog(EMBEDDED_CATALOG_JSON, "<embedded catalog>")
}

/// The embedded bundled catalog's signature-verified `(catalog_sha256,
/// catalog_epoch)` fingerprint, with no filesystem side effects (unlike
/// [`load_embedded_signed_catalog`], this never touches the on-disk
/// epoch-rollback guard). Used by packaging tooling (the CLI's hidden
/// `catalog-fingerprint` introspection command) to confirm a prebuilt
/// sidecar binary's embedded catalog matches a copied catalog resource
/// before it ships, without needing to run the binary's normal load path.
pub fn embedded_catalog_fingerprint() -> Result<(String, u64), CatalogError> {
    let verified = catalog_security::verify_catalog_signature_manifest(
        EMBEDDED_CATALOG_JSON,
        EMBEDDED_CATALOG_SIGNATURE_JSON,
        DEFAULT_CATALOG_URL,
    )
    .map_err(|error| CatalogError::CatalogSecurity {
        catalog_source: DEFAULT_CATALOG_URL.to_string(),
        message: format!("embedded catalog rejected: {error}"),
    })?;
    Ok((verified.catalog_sha256, verified.catalog_epoch))
}

fn read_and_verify_cached_catalog_manifest(
    source: &str,
    home: &Path,
    cached: &str,
    original_error: &CatalogError,
) -> Result<(), CatalogError> {
    let manifest_path = catalog_security::default_catalog_signature_cache_path(home);
    let manifest_contents =
        fs::read_to_string(&manifest_path).map_err(|cache_error| CatalogError::ReadCatalog {
            catalog_source: source.to_string(),
            message: format!(
                "{original_error}; no usable signed cache manifest at '{}': {cache_error}",
                manifest_path.display()
            ),
        })?;
    let verified = verify_catalog_manifest_for_source(source, cached, &manifest_contents).map_err(
        |error| CatalogError::CatalogSecurity {
            catalog_source: source.to_string(),
            message: format!("{original_error}; cached catalog rejected: {error}"),
        },
    )?;
    catalog_security::enforce_catalog_epoch_for_verified(home, &verified).map_err(|error| {
        CatalogError::CatalogSecurity {
            catalog_source: source.to_string(),
            message: format!("{original_error}; cached catalog rejected: {error}"),
        }
    })
}

fn cache_catalog(home: &Path, cache_path: &Path, contents: &str) -> Result<(), CatalogError> {
    fs::create_dir_all(home).map_err(|source| CatalogError::CreateHome {
        path: home.to_path_buf(),
        source,
    })?;
    atomic_file::write_file_atomically(cache_path, contents.as_bytes()).map_err(|source| {
        CatalogError::CacheCatalog {
            path: cache_path.to_path_buf(),
            source,
        }
    })
}

fn cache_catalog_security(
    home: &Path,
    manifest_contents: &str,
    verified: &catalog_security::VerifiedCatalogSignature,
) -> Result<(), CatalogError> {
    catalog_security::cache_catalog_manifest(home, manifest_contents).map_err(|error| {
        CatalogError::CatalogSecurity {
            catalog_source: catalog_security::default_catalog_signature_cache_path(home)
                .display()
                .to_string(),
            message: error.to_string(),
        }
    })?;
    // Gated by `participates_in_epoch_floor`: a local-dev-key-verified catalog
    // must never advance the shared production anti-rollback floor (see the
    // doc comment on that function for the persistent DoS this closes).
    catalog_security::record_catalog_epoch_for_verified(home, verified).map_err(|error| {
        CatalogError::CatalogSecurity {
            catalog_source: catalog_security::default_catalog_epoch_path(home)
                .display()
                .to_string(),
            message: error.to_string(),
        }
    })
}

fn validate_model_catalog(catalog: &ModelCatalog) -> Result<(), CatalogError> {
    if catalog.schema_version != SUPPORTED_CATALOG_SCHEMA_VERSION {
        return Err(CatalogError::UnsupportedSchema {
            found: catalog.schema_version,
        });
    }
    if catalog.models.is_empty() {
        return Err(CatalogError::InvalidCatalog(
            "catalog must contain at least one model".to_string(),
        ));
    }
    for model in &catalog.models {
        if model.id.trim().is_empty() {
            return Err(CatalogError::InvalidCatalog(
                "model id must not be empty".to_string(),
            ));
        }
        validate_catalog_model_kind(model)?;
        validate_catalog_hf_repo(model)?;
        validate_catalog_min_cli_version_format(model)?;
        validate_catalog_min_core_version_format(model)?;
        if model.hf_revision.len() != 40
            || !model
                .hf_revision
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(CatalogError::InvalidCatalog(format!(
                "model '{}' hf_revision must be a 40 hex character commit sha",
                model.id
            )));
        }
        if model.quants.is_empty() {
            return Err(CatalogError::InvalidCatalog(format!(
                "model '{}' must contain at least one quant",
                model.id
            )));
        }
        if !model
            .quants
            .iter()
            .any(|quant| quant.quant == model.recommended_quant)
        {
            return Err(CatalogError::MissingRecommendedQuant {
                model_id: model.id.clone(),
                quant: model.recommended_quant.clone(),
            });
        }
        for quant in &model.quants {
            if quant.quant.trim().is_empty()
                || quant.suffix.trim().is_empty()
                || quant.pull.trim().is_empty()
            {
                return Err(CatalogError::InvalidCatalog(format!(
                    "model '{}' contains an empty quant selector",
                    model.id
                )));
            }
            if quant.pull != format!("{}:{}", model.id, quant.suffix) {
                return Err(CatalogError::InvalidCatalog(format!(
                    "model '{}' quant '{}' pull must be '<id>:<suffix>'",
                    model.id, quant.quant
                )));
            }
            if quant.filename.contains('/')
                || quant.filename.contains('\\')
                || !quant.filename.ends_with(".oasr")
            {
                return Err(CatalogError::InvalidCatalog(format!(
                    "model '{}' quant '{}' filename must be a local .oasr basename",
                    model.id, quant.quant
                )));
            }
            if quant.size_bytes == 0 {
                return Err(CatalogError::InvalidCatalog(format!(
                    "model '{}' quant '{}' size_bytes must be greater than zero",
                    model.id, quant.quant
                )));
            }
            if !quant.url.starts_with("https://") {
                return Err(CatalogError::InvalidCatalog(format!(
                    "model '{}' quant '{}' URL must use https://",
                    model.id, quant.quant
                )));
            }
            let expected_url = format!(
                "{HUGGING_FACE_BASE_URL}{}/resolve/{}/{}",
                model.hf_repo, model.hf_revision, quant.filename
            );
            if quant.url != expected_url {
                return Err(CatalogError::InvalidCatalog(format!(
                    "model '{}' quant '{}' URL must be pinned to hf_repo, hf_revision, and filename",
                    model.id, quant.quant
                )));
            }
            for mirror in &quant.mirrors {
                validate_catalog_mirror_url(model, quant, mirror)?;
            }
            if quant.sha256.len() != 64
                || !quant.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(CatalogError::InvalidCatalog(format!(
                    "model '{}' quant '{}' sha256 must be 64 hex characters",
                    model.id, quant.quant
                )));
            }
        }
    }
    for backend in &catalog.backends {
        validate_catalog_backend(backend)?;
    }
    Ok(())
}

/// Validate a downloadable backend pack entry: identity fields present, a
/// MAJOR.MINOR.PATCH gate, exactly one plugin file, and per-file integrity
/// (local basename, https URL, non-zero size, 64-hex sha256). Archive files must
/// declare a safe relative `extract_subdir` (no absolute / `..` traversal); the
/// other roles must not. Mirrors the model-quant checks above.
fn validate_catalog_backend(backend: &CatalogBackend) -> Result<(), CatalogError> {
    if backend.id.trim().is_empty() {
        return Err(CatalogError::InvalidCatalog(
            "backend id must not be empty".to_string(),
        ));
    }
    if backend.version.trim().is_empty() {
        return Err(CatalogError::InvalidCatalog(format!(
            "backend '{}' version must not be empty",
            backend.id
        )));
    }
    if backend.display_name.trim().is_empty() {
        return Err(CatalogError::InvalidCatalog(format!(
            "backend '{}' display_name must not be empty",
            backend.id
        )));
    }
    if parse_semver_triplet(&backend.min_cli_version).is_none() {
        return Err(CatalogError::InvalidCatalog(format!(
            "backend '{}' min_cli_version must be MAJOR.MINOR.PATCH",
            backend.id
        )));
    }
    if backend.files.is_empty() {
        return Err(CatalogError::InvalidCatalog(format!(
            "backend '{}' must contain at least one file",
            backend.id
        )));
    }
    let plugin_count = backend
        .files
        .iter()
        .filter(|file| file.role == CatalogBackendFileRole::Plugin)
        .count();
    if plugin_count != 1 {
        return Err(CatalogError::InvalidCatalog(format!(
            "backend '{}' must declare exactly one plugin file (found {plugin_count})",
            backend.id
        )));
    }
    let mut seen_filenames = std::collections::BTreeSet::new();
    for file in &backend.files {
        if file.filename.trim().is_empty()
            || file.filename.contains('/')
            || file.filename.contains('\\')
        {
            return Err(CatalogError::InvalidCatalog(format!(
                "backend '{}' file name '{}' must be a non-empty local basename",
                backend.id, file.filename
            )));
        }
        if !seen_filenames.insert(file.filename.as_str()) {
            return Err(CatalogError::InvalidCatalog(format!(
                "backend '{}' declares duplicate file '{}'",
                backend.id, file.filename
            )));
        }
        if !file.url.starts_with("https://") {
            return Err(CatalogError::InvalidCatalog(format!(
                "backend '{}' file '{}' URL must use https://",
                backend.id, file.filename
            )));
        }
        if file.size_bytes == 0 {
            return Err(CatalogError::InvalidCatalog(format!(
                "backend '{}' file '{}' size_bytes must be greater than zero",
                backend.id, file.filename
            )));
        }
        if file.sha256.len() != 64 || !file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(CatalogError::InvalidCatalog(format!(
                "backend '{}' file '{}' sha256 must be 64 hex characters",
                backend.id, file.filename
            )));
        }
        match file.role {
            CatalogBackendFileRole::Archive => {
                let subdir = file.extract_subdir.as_deref().unwrap_or("").trim();
                if subdir.is_empty() {
                    return Err(CatalogError::InvalidCatalog(format!(
                        "backend '{}' archive '{}' must declare extract_subdir",
                        backend.id, file.filename
                    )));
                }
                let unsafe_path = subdir.starts_with('/')
                    || subdir.starts_with('\\')
                    || subdir.contains(':')
                    || subdir
                        .split(['/', '\\'])
                        .any(|component| component.is_empty() || component == "..");
                if unsafe_path {
                    return Err(CatalogError::InvalidCatalog(format!(
                        "backend '{}' archive '{}' extract_subdir must be a safe relative path",
                        backend.id, file.filename
                    )));
                }
            }
            CatalogBackendFileRole::Plugin | CatalogBackendFileRole::Runtime => {
                if file.extract_subdir.is_some() {
                    return Err(CatalogError::InvalidCatalog(format!(
                        "backend '{}' file '{}' has extract_subdir but is not an archive",
                        backend.id, file.filename
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate_catalog_model_kind(model: &CatalogModel) -> Result<(), CatalogError> {
    match (model.kind, model.capability.as_ref()) {
        (CatalogModelKind::AsrModel, None) => {
            validate_no_translation_metadata(model)?;
            Ok(())
        }
        (CatalogModelKind::AsrModel, Some(_)) => Err(CatalogError::InvalidCatalog(format!(
            "model '{}' has capability metadata but kind is asr-model",
            model.id
        ))),
        (CatalogModelKind::CapabilityPack, None) => Err(CatalogError::InvalidCatalog(format!(
            "model '{}' is kind capability-pack but has no capability metadata",
            model.id
        ))),
        (CatalogModelKind::CapabilityPack, Some(capability)) => {
            if capability.feature.trim().is_empty() {
                return Err(CatalogError::InvalidCatalog(format!(
                    "model '{}' capability.feature must not be empty",
                    model.id
                )));
            }
            validate_no_translation_metadata(model)?;
            Ok(())
        }
        (CatalogModelKind::TranslationModel, Some(_)) => {
            Err(CatalogError::InvalidCatalog(format!(
                "model '{}' has capability metadata but kind is translation-model",
                model.id
            )))
        }
        (CatalogModelKind::TranslationModel, None) => validate_translation_metadata(model),
    }
}

fn validate_no_translation_metadata(model: &CatalogModel) -> Result<(), CatalogError> {
    if !model.source_langs.is_empty() || !model.target_langs.is_empty() {
        return Err(CatalogError::InvalidCatalog(format!(
            "model '{}' has translation metadata but kind is not translation-model",
            model.id
        )));
    }
    Ok(())
}

fn validate_translation_metadata(model: &CatalogModel) -> Result<(), CatalogError> {
    validate_catalog_language_list(model, "source_langs", &model.source_langs)?;
    validate_catalog_language_list(model, "target_langs", &model.target_langs)?;
    for source in &model.source_langs {
        if model.target_langs.iter().any(|target| target == source) {
            return Err(CatalogError::InvalidCatalog(format!(
                "model '{}' translation source_langs and target_langs must not overlap",
                model.id
            )));
        }
    }
    for lang in model.source_langs.iter().chain(model.target_langs.iter()) {
        if !model
            .languages
            .iter()
            .any(|catalog_lang| catalog_lang == lang)
        {
            return Err(CatalogError::InvalidCatalog(format!(
                "model '{}' translation language '{lang}' must also appear in languages",
                model.id
            )));
        }
    }
    Ok(())
}

fn validate_catalog_language_list(
    model: &CatalogModel,
    field: &str,
    langs: &[String],
) -> Result<(), CatalogError> {
    if langs.is_empty() {
        return Err(CatalogError::InvalidCatalog(format!(
            "model '{}' translation {field} must not be empty",
            model.id
        )));
    }
    let mut seen = std::collections::BTreeSet::new();
    for lang in langs {
        if !(2..=3).contains(&lang.len()) || !lang.bytes().all(|byte| byte.is_ascii_lowercase()) {
            return Err(CatalogError::InvalidCatalog(format!(
                "model '{}' translation {field} contains invalid language code '{lang}'",
                model.id
            )));
        }
        if !seen.insert(lang) {
            return Err(CatalogError::InvalidCatalog(format!(
                "model '{}' translation {field} contains duplicate language code '{lang}'",
                model.id
            )));
        }
    }
    Ok(())
}

fn validate_catalog_mirror_url(
    model: &CatalogModel,
    quant: &CatalogQuant,
    mirror: &CatalogMirror,
) -> Result<(), CatalogError> {
    if mirror.source.trim().is_empty() {
        return Err(CatalogError::InvalidCatalog(format!(
            "model '{}' quant '{}' mirror source must not be empty",
            model.id, quant.quant
        )));
    }
    if !http::is_allowed_mirror_host(&mirror.url) {
        return Err(CatalogError::InvalidCatalog(format!(
            "model '{}' quant '{}' mirror URL host is not allowed",
            model.id, quant.quant
        )));
    }
    let parsed = reqwest::Url::parse(&mirror.url).map_err(|source| {
        CatalogError::InvalidCatalog(format!(
            "model '{}' quant '{}' mirror URL is invalid: {source}",
            model.id, quant.quant
        ))
    })?;
    let host = parsed.host_str().unwrap_or_default();
    if mirror.source == "modelscope" && !MODELSCOPE_CATALOG_MIRRORS_ENABLED {
        return Err(CatalogError::InvalidCatalog(format!(
            "model '{}' quant '{}' ModelScope mirrors are disabled; use Hugging Face with the hf-mirror download source",
            model.id, quant.quant
        )));
    }
    if matches!(host, "modelscope.cn" | "www.modelscope.cn") {
        if !MODELSCOPE_CATALOG_MIRRORS_ENABLED {
            return Err(CatalogError::InvalidCatalog(format!(
                "model '{}' quant '{}' ModelScope mirrors are disabled; use Hugging Face with the hf-mirror download source",
                model.id, quant.quant
            )));
        }
        let segments = parsed
            .path_segments()
            .map(|segments| segments.collect::<Vec<_>>())
            .unwrap_or_default();
        let (hf_owner, hf_name) = model.hf_repo.split_once('/').unwrap_or_default();
        let modelscope_owner = hf_owner.to_ascii_lowercase();
        let revision = segments.get(4).copied().unwrap_or_default();
        if segments.len() != 6
            || segments[0] != "models"
            || segments[1] != modelscope_owner
            || segments[2] != hf_name
            || segments[3] != "resolve"
            || revision.len() != 40
            || !revision.chars().all(|ch| ch.is_ascii_hexdigit())
            || segments[5] != quant.filename
        {
            return Err(CatalogError::InvalidCatalog(format!(
                "model '{}' quant '{}' ModelScope mirror URL must use /models/{{lowercase-hf-owner}}/{{hf-repo-name}}/resolve/{{40-hex-revision}}/{{filename}}",
                model.id, quant.quant
            )));
        }
    }
    Ok(())
}

fn validate_catalog_hf_repo(model: &CatalogModel) -> Result<(), CatalogError> {
    let mut parts = model.hf_repo.split('/');
    let owner = parts.next().unwrap_or_default();
    let repo = parts.next().unwrap_or_default();
    if parts.next().is_some() || !is_safe_hf_repo_segment(owner) || !is_safe_hf_repo_segment(repo) {
        return Err(CatalogError::InvalidCatalog(format!(
            "model '{}' hf_repo must use owner/repo with portable characters",
            model.id
        )));
    }
    Ok(())
}

fn is_safe_hf_repo_segment(value: &str) -> bool {
    !value.trim().is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

/// Validate that `min_cli_version` is well-formed (major.minor.patch). The version
/// *comparison* is intentionally NOT enforced here: a model requiring a newer
/// OpenASR than the running build must still load so the model market can list it
/// as "update to use" (see [`CatalogModel::availability`]); it is refused only at
/// pull time (`resolve_catalog_pull_with_profile`), never hidden or fail-the-catalog.
fn validate_catalog_min_cli_version_format(model: &CatalogModel) -> Result<(), CatalogError> {
    if parse_semver_triplet(&model.min_cli_version).is_none() {
        return Err(CatalogError::InvalidCatalog(format!(
            "model '{}' min_cli_version must use major.minor.patch",
            model.id
        )));
    }
    Ok(())
}

/// Validate the optional `min_core_version` gate is well-formed
/// (major.minor.patch) when present. Like `min_cli_version`, the version
/// *comparison* is intentionally NOT enforced here: a model requiring a newer
/// core runtime than the running build must still load so the market can list it
/// as "update to use" (see [`CatalogModel::availability`]); it is refused only at
/// pull time, never hidden or fail-the-catalog. Absent means "no constraint".
fn validate_catalog_min_core_version_format(model: &CatalogModel) -> Result<(), CatalogError> {
    if let Some(min_core_version) = &model.min_core_version
        && parse_semver_triplet(min_core_version).is_none()
    {
        return Err(CatalogError::InvalidCatalog(format!(
            "model '{}' min_core_version must use major.minor.patch",
            model.id
        )));
    }
    Ok(())
}

fn parse_semver_triplet(value: &str) -> Option<(u64, u64, u64)> {
    let core = value
        .trim()
        .split_once('-')
        .map_or(value.trim(), |(core, _)| core);
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

fn parse_catalog_pull_reference(value: &str) -> Result<(&str, Option<&str>), CatalogError> {
    let mut parts = value.split(':');
    let model_ref = parts.next().unwrap_or_default().trim();
    let quant = parts.next().map(str::trim);
    if model_ref.is_empty() || quant.is_some_and(str::is_empty) || parts.next().is_some() {
        return Err(CatalogError::InvalidPullReference(value.to_string()));
    }
    Ok((model_ref, quant))
}

fn resolve_catalog_model<'a>(
    catalog: &'a ModelCatalog,
    model_ref: &str,
    size: Option<&str>,
) -> Result<&'a CatalogModel, CatalogError> {
    let normalized = model_ref.trim();
    let size = size.map(str::trim).filter(|value| !value.is_empty());
    let series = catalog_series_spec(normalized);
    let effective_size = size.or_else(|| series.map(CatalogSeriesSpec::default_size));
    let matches: Vec<&CatalogModel> = catalog
        .models
        .iter()
        .filter(|model| model.public)
        .filter(|model| effective_size.is_none_or(|requested_size| model.size == requested_size))
        .filter(|model| {
            if let Some(spec) = series {
                spec.contains_family_size(&model.family, &model.size)
            } else {
                model.id == normalized
                    || model.pull_alias.as_deref() == Some(normalized)
                    || model.aliases.iter().any(|alias| alias == normalized)
            }
        })
        .collect();

    match matches.as_slice() {
        [model] => Ok(model),
        [] => Err(CatalogError::UnknownModel {
            reference: normalized.to_string(),
        }),
        many => Err(CatalogError::AmbiguousModelRef {
            reference: normalized.to_string(),
            available: many
                .iter()
                .map(|model| model.pull_recommended.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        }),
    }
}

fn resolve_catalog_quant<'a>(
    model: &'a CatalogModel,
    quant_ref: Option<&str>,
) -> Result<&'a CatalogQuant, CatalogError> {
    let selected = quant_ref.unwrap_or(model.recommended_quant.as_str());
    let selected_canonical = canonical_quant_tag(selected);
    model
        .quants
        .iter()
        .find(|quant| {
            canonical_quant_tag(&quant.quant) == selected_canonical
                || canonical_quant_tag(&quant.suffix) == selected_canonical
                || quant.pull == selected
        })
        .ok_or_else(|| CatalogError::UnknownQuant {
            model_id: model.id.clone(),
            quant: selected_canonical.to_string(),
            available: model
                .quants
                .iter()
                .map(|quant| quant.pull.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        })
}

fn catalog_quant_peak_rss_bytes(quant: &CatalogQuant) -> Option<u64> {
    quant
        .perf
        .as_ref()
        .and_then(|perf| perf.peak_rss_bytes)
        .filter(|value| *value > 0)
}

pub(crate) fn quant_quality_rank(quant: &str) -> u8 {
    match canonical_quant_tag(quant) {
        "f32" => 4,
        "fp16" => 3,
        "q8_0" => 2,
        "q4_k" => 1,
        "q3_k" => 0,
        _ => 0,
    }
}

fn catalog_quant_quality_rank(quant: &CatalogQuant) -> u8 {
    quant_quality_rank(&quant.quant)
}

impl ModelCard {
    pub fn family_name(&self) -> &str {
        self.family.as_deref().unwrap_or(&self.id)
    }

    pub fn variant_tag(&self) -> Option<&str> {
        self.variant.as_ref().map(|variant| variant.tag.as_str())
    }

    pub fn variant_format(&self) -> Option<&str> {
        self.variant.as_ref().map(|variant| variant.format.as_str())
    }

    pub fn variant_quantization(&self) -> Option<&str> {
        self.variant
            .as_ref()
            .and_then(|variant| variant.quantization.as_deref())
    }

    pub fn is_default_variant(&self) -> bool {
        self.default_variant
            .as_deref()
            .zip(self.variant_tag())
            .is_some_and(|(default_variant, tag)| default_variant == tag)
    }
}

pub fn parse_model_ref(value: &str) -> Result<ModelRef, ModelResolutionError> {
    resolution::parse_model_ref(value)
}

pub fn model_refs_match_with_optional_tag_alias(requested: &ModelRef, resolved: &ModelRef) -> bool {
    if requested.family != resolved.family {
        return false;
    }

    match (requested.tag.as_deref(), resolved.tag.as_deref()) {
        (None, None) => true,
        (None, Some(_)) => false,
        (Some(requested_tag), Some(resolved_tag)) => {
            canonical_quant_tag(requested_tag) == canonical_quant_tag(resolved_tag)
        }
        (Some(_), None) => false,
    }
}

pub fn model_reference_matches_resolved_source(requested: &str, resolved_source_id: &str) -> bool {
    let Ok(requested_ref) = parse_model_ref(requested) else {
        return false;
    };
    let Ok(resolved_ref) = parse_model_ref(resolved_source_id) else {
        return false;
    };
    model_refs_match_with_optional_tag_alias(&requested_ref, &resolved_ref)
}

pub fn resolve_registry_model_ref<'a>(
    cards: &'a [ModelCard],
    model_ref: &str,
) -> Result<ResolvedModel<'a>, ModelResolutionError> {
    resolution::resolve_registry_model_ref(cards, model_ref)
}

pub fn resolve_runtime_model_ref<'a>(
    cards: &'a [ModelCard],
    catalog: Option<&ModelCatalog>,
    model_ref: &str,
) -> Result<ResolvedRuntimeModelRef<'a>, RuntimeModelResolutionError> {
    if let Some(catalog) = catalog {
        match resolve_catalog_pull(
            catalog,
            &CatalogPullRequest {
                reference: model_ref.to_string(),
                quant: None,
                size: None,
            },
        ) {
            Ok(resolved) => {
                let card = cards.iter().find(|card| card.id == resolved.model_id);
                let runtime_model_id = runtime_model_id(&resolved.model_id, Some(&resolved.quant));
                return Ok(ResolvedRuntimeModelRef {
                    card,
                    requested: model_ref.to_string(),
                    model_id: resolved.model_id,
                    quant: Some(resolved.quant),
                    runtime_model_id,
                    pull: Some(resolved.pull),
                    source: RuntimeModelRefSource::Catalog,
                });
            }
            Err(catalog_error) => {
                return resolve_registry_model_ref(cards, model_ref)
                    .map(runtime_model_ref_from_registry)
                    .map_err(|_| RuntimeModelResolutionError::Catalog(catalog_error));
            }
        }
    }

    resolve_registry_model_ref(cards, model_ref)
        .map(runtime_model_ref_from_registry)
        .map_err(RuntimeModelResolutionError::Registry)
}

fn runtime_model_ref_from_registry<'a>(resolved: ResolvedModel<'a>) -> ResolvedRuntimeModelRef<'a> {
    let quant = resolved
        .card
        .variant_quantization()
        .map(canonical_quant_tag)
        .map(ToOwned::to_owned);
    let runtime_model_id = runtime_model_id(&resolved.card.id, quant.as_deref());
    ResolvedRuntimeModelRef {
        card: Some(resolved.card),
        requested: resolved.requested,
        model_id: resolved.card.id.clone(),
        quant,
        runtime_model_id,
        pull: None,
        source: RuntimeModelRefSource::Registry,
    }
}

fn runtime_model_id(model_id: &str, quant: Option<&str>) -> String {
    quant.map_or_else(
        || model_id.to_string(),
        |quant| format!("{model_id}:{quant}"),
    )
}

#[cfg(test)]
pub(crate) fn test_model_card(id: &str) -> ModelCard {
    ModelCard {
        id: id.to_string(),
        family: None,
        default_variant: None,
        variant: None,
        display_name: id.to_string(),
        backend: "native".to_string(),
        task: "transcription".to_string(),
        languages: vec!["en".to_string()],
        size: "tiny".to_string(),
        recommended_hardware: "CPU".to_string(),
        license: "MIT".to_string(),
        features: vec!["transcription".to_string()],
        quality_profile: "fastest".to_string(),
        source: "Native ASR Core planning metadata".to_string(),
    }
}

#[cfg(test)]
mod tests;
