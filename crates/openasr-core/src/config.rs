use std::{
    fs,
    path::{Path, PathBuf},
    str::FromStr,
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    BackendKind, ExecutionTarget, ModelCard, ModelCatalog, ModelResolutionError, PhraseBiasConfig,
    RuntimeModelResolutionError, atomic_file, resolve_registry_model_ref,
    resolve_runtime_model_ref,
};
use crate::{download_source::DownloadSourcePref, launch_pack::QuantPreference};

pub const DEFAULT_MODEL_ID: &str = "qwen3-asr-0.6b";
/// Quant pinned for the first-run install of `DEFAULT_MODEL_ID`, so the very
/// first download a newcomer triggers is bounded and predictable instead of the
/// auto-picker's largest-that-fits choice. An explicit `openasr pull` still uses
/// the full host-aware quant ladder.
pub const DEFAULT_MODEL_BOOTSTRAP_QUANT: &str = "q4_k";
pub const DEFAULT_BACKEND_ID: &str = "native";
pub const PREFERENCES_SCHEMA_VERSION: u32 = 1;
pub const MAX_INFERENCE_THREADS: u16 = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAsrConfig {
    #[serde(default = "default_model")]
    pub default_model: Option<String>,
    #[serde(default = "default_backend")]
    pub default_backend: Option<String>,
    #[serde(default)]
    pub media: MediaConfig,
    #[serde(default)]
    pub download_source: DownloadSourcePref,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAsrConfigDocument {
    #[serde(flatten)]
    pub config: OpenAsrConfig,
    #[serde(default)]
    pub preferences: Preferences,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Preferences {
    #[serde(default = "default_preferences_version")]
    pub version: u32,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub diarize: bool,
    #[serde(default)]
    pub word_timestamps: bool,
    #[serde(default)]
    pub auto_save: bool,
    #[serde(default)]
    pub launch_at_login: bool,
    #[serde(default = "default_tray_icon")]
    pub tray_icon: bool,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    #[serde(default)]
    pub hotwords: Vec<String>,
    #[serde(default)]
    pub hotword_boost: Option<f32>,
    #[serde(default)]
    pub theme: AppearanceTheme,
    #[serde(default)]
    pub accent_color: Option<String>,
    #[serde(default)]
    pub density: AppearanceDensity,
    #[serde(default = "default_dictation_shortcut")]
    pub dictation_shortcut: Option<String>,
    #[serde(default)]
    pub push_to_talk: bool,
    #[serde(default)]
    pub inference_threads: Option<u16>,
    #[serde(default)]
    pub quant_preference: QuantPreference,
    #[serde(default)]
    pub execution_target: ExecutionTarget,
    #[serde(default)]
    pub history_retention: HistoryRetentionPolicy,
    #[serde(default)]
    pub idle_unload: IdleUnloadPolicy,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryRetentionPolicy {
    #[default]
    Never,
    Last5,
    Week,
    Month,
    Quarter,
    Year,
}

impl HistoryRetentionPolicy {
    pub const fn max_entries(self) -> Option<usize> {
        match self {
            Self::Last5 => Some(5),
            Self::Never | Self::Week | Self::Month | Self::Quarter | Self::Year => None,
        }
    }

    pub const fn max_age_seconds(self) -> Option<u64> {
        match self {
            Self::Week => Some(7 * 24 * 60 * 60),
            Self::Month => Some(30 * 24 * 60 * 60),
            Self::Quarter => Some(90 * 24 * 60 * 60),
            Self::Year => Some(365 * 24 * 60 * 60),
            Self::Never | Self::Last5 => None,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdleUnloadPolicy {
    #[default]
    Never,
    #[serde(rename = "now")]
    Now,
    #[serde(rename = "2m")]
    After2m,
    #[serde(rename = "10m")]
    After10m,
    #[serde(rename = "1h")]
    After1h,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppearanceTheme {
    Light,
    Dark,
    #[default]
    System,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppearanceDensity {
    Compact,
    #[default]
    Comfortable,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaConfig {
    #[serde(default)]
    pub ffmpeg_bin: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigKey {
    DefaultModel,
    DefaultBackend,
    MediaFfmpegBin,
    DownloadSource,
}

impl ConfigKey {
    pub const ALL: &'static [&'static str] = &[
        "default_model",
        "default_backend",
        "media.ffmpeg_bin",
        "download_source",
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::DefaultModel => "default_model",
            Self::DefaultBackend => "default_backend",
            Self::MediaFfmpegBin => "media.ffmpeg_bin",
            Self::DownloadSource => "download_source",
        }
    }
}

impl FromStr for ConfigKey {
    type Err = ConfigError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "default_model" => Ok(Self::DefaultModel),
            "default_backend" => Ok(Self::DefaultBackend),
            "media.ffmpeg_bin" => Ok(Self::MediaFfmpegBin),
            "download_source" => Ok(Self::DownloadSource),
            other => Err(ConfigError::UnknownKey(other.to_string())),
        }
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Unknown config key '{0}'. Use one of: {keys}.", keys = ConfigKey::ALL.join(", "))]
    UnknownKey(String),
    #[error("Unsupported backend '{0}'. Use one of: {backends}.", backends = BackendKind::SELECTABLE.join(", "))]
    UnsupportedBackend(String),
    #[error("Unsupported download source '{0}'. Use one of: auto, hf, hf-mirror.")]
    UnsupportedDownloadSource(String),
    #[error(
        "Backend '{0}' cannot be persisted as default_backend.\nUse `default_backend=mock` and pass `--backend native` explicitly when you need local GGUF runtime execution with fail-closed behavior."
    )]
    UnsupportedDefaultBackend(String),
    #[error("Unsupported preferences schema version {found}. Expected version {expected}.")]
    UnsupportedPreferencesVersion { found: u32, expected: u32 },
    #[error("Invalid preference '{field}': {reason}")]
    InvalidPreference { field: &'static str, reason: String },
    #[error("Unknown model: {0}\nRun `openasr list` to see available models.")]
    UnknownModel(String),
    #[error("{0}")]
    ModelResolution(ModelResolutionError),
    #[error("{0}")]
    RuntimeModelResolution(RuntimeModelResolutionError),
    #[error("Could not read config file '{path}': {source}")]
    ReadConfig {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Could not parse config file '{path}': {source}")]
    ParseConfig {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("Could not create OpenASR home directory '{path}': {source}")]
    CreateHome {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Could not serialize config: {0}")]
    SerializeConfig(serde_json::Error),
    #[error("Could not write config file '{path}': {source}")]
    WriteConfig {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl Default for OpenAsrConfig {
    fn default() -> Self {
        Self {
            default_model: default_model(),
            default_backend: default_backend(),
            media: MediaConfig::default(),
            download_source: DownloadSourcePref::Auto,
        }
    }
}

impl Default for Preferences {
    fn default() -> Self {
        Self {
            version: PREFERENCES_SCHEMA_VERSION,
            language: None,
            diarize: false,
            word_timestamps: false,
            auto_save: false,
            launch_at_login: false,
            tray_icon: default_tray_icon(),
            output_dir: None,
            hotwords: Vec::new(),
            hotword_boost: None,
            theme: AppearanceTheme::System,
            accent_color: None,
            density: AppearanceDensity::Comfortable,
            dictation_shortcut: default_dictation_shortcut(),
            push_to_talk: false,
            inference_threads: None,
            quant_preference: QuantPreference::Auto,
            execution_target: ExecutionTarget::Auto,
            history_retention: HistoryRetentionPolicy::Never,
            idle_unload: IdleUnloadPolicy::Never,
        }
    }
}

impl OpenAsrConfig {
    fn set_key(&mut self, key: ConfigKey, value: String) {
        if let Some(slot) = self.key_slot_mut(key) {
            *slot = Some(value);
        }
    }
    fn key_slot(&self, key: ConfigKey) -> Option<String> {
        match key {
            ConfigKey::DefaultModel => self.default_model.clone(),
            ConfigKey::DefaultBackend => self.default_backend.clone(),
            ConfigKey::MediaFfmpegBin => self.media.ffmpeg_bin.clone(),
            ConfigKey::DownloadSource => Some(render_download_source_pref(&self.download_source)),
        }
    }

    fn key_slot_mut(&mut self, key: ConfigKey) -> Option<&mut Option<String>> {
        match key {
            ConfigKey::DefaultModel => Some(&mut self.default_model),
            ConfigKey::DefaultBackend => Some(&mut self.default_backend),
            ConfigKey::MediaFfmpegBin => Some(&mut self.media.ffmpeg_bin),
            ConfigKey::DownloadSource => None,
        }
    }

    pub fn get(&self, key: ConfigKey) -> Option<String> {
        self.key_slot(key)
    }

    pub fn set(
        &mut self,
        key: ConfigKey,
        value: impl Into<String>,
        registry: &[ModelCard],
    ) -> Result<(), ConfigError> {
        self.set_with_catalog(key, value, registry, None)
    }

    pub fn set_with_catalog(
        &mut self,
        key: ConfigKey,
        value: impl Into<String>,
        registry: &[ModelCard],
        catalog: Option<&ModelCatalog>,
    ) -> Result<(), ConfigError> {
        let value = value.into();
        match key {
            ConfigKey::DefaultModel => {
                let value = resolve_default_model_config_value(registry, catalog, value)?;
                self.set_key(key, value);
            }
            ConfigKey::DefaultBackend => {
                let backend = BackendKind::from_str(&value)
                    .map_err(|_| ConfigError::UnsupportedBackend(value.clone()))?;
                if !matches!(backend, BackendKind::Mock | BackendKind::Native) {
                    return Err(ConfigError::UnsupportedDefaultBackend(value));
                }
                self.set_key(key, value);
            }
            ConfigKey::MediaFfmpegBin => self.set_key(key, value),
            ConfigKey::DownloadSource => {
                self.download_source = DownloadSourcePref::parse_env_value(&value)
                    .ok_or_else(|| ConfigError::UnsupportedDownloadSource(value.clone()))?;
            }
        }
        Ok(())
    }

    pub fn unset(&mut self, key: ConfigKey) {
        if key == ConfigKey::DownloadSource {
            self.download_source = DownloadSourcePref::Auto;
        } else if let Some(slot) = self.key_slot_mut(key) {
            *slot = None;
        }
    }

    pub fn validate(&self, registry: &[ModelCard]) -> Result<(), ConfigError> {
        self.validate_with_catalog(registry, None)
    }

    pub fn validate_with_catalog(
        &self,
        registry: &[ModelCard],
        catalog: Option<&ModelCatalog>,
    ) -> Result<(), ConfigError> {
        if let Some(default_model) = self.default_model.as_deref() {
            validate_default_model_ref(registry, catalog, default_model)?;
        }
        if let Some(default_backend) = self.default_backend.as_deref() {
            let backend = BackendKind::from_str(default_backend)
                .map_err(|_| ConfigError::UnsupportedBackend(default_backend.to_string()))?;
            // `native` is now a valid persisted default (it resolves an installed
            // pack by model id and the CLI consent-pulls a missing one); only
            // non-executable backends are rejected as a saved default.
            if !matches!(backend, BackendKind::Mock | BackendKind::Native) {
                return Err(ConfigError::UnsupportedDefaultBackend(
                    default_backend.to_string(),
                ));
            }
        }
        Ok(())
    }
}

impl OpenAsrConfigDocument {
    pub fn validate(&self, registry: &[ModelCard]) -> Result<(), ConfigError> {
        self.config.validate(registry)?;
        self.preferences.validate()
    }

    pub fn validate_with_catalog(
        &self,
        registry: &[ModelCard],
        catalog: Option<&ModelCatalog>,
    ) -> Result<(), ConfigError> {
        self.config.validate_with_catalog(registry, catalog)?;
        self.preferences.validate()
    }
}

impl Preferences {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.version != PREFERENCES_SCHEMA_VERSION {
            return Err(ConfigError::UnsupportedPreferencesVersion {
                found: self.version,
                expected: PREFERENCES_SCHEMA_VERSION,
            });
        }
        if let Some(language) = self.language.as_deref()
            && language.trim().is_empty()
        {
            return invalid_preference("language", "must be non-empty when set");
        }
        if let Some(output_dir) = self.output_dir.as_deref()
            && output_dir.as_os_str().is_empty()
        {
            return invalid_preference("output_dir", "must be non-empty when set");
        }
        if let Some(accent_color) = self.accent_color.as_deref()
            && accent_color.trim().is_empty()
        {
            return invalid_preference("accent_color", "must be non-empty when set");
        }
        if let Some(shortcut) = self.dictation_shortcut.as_deref()
            && shortcut.trim().is_empty()
        {
            return invalid_preference("dictation_shortcut", "must be non-empty when set");
        }
        if let Some(threads) = self.inference_threads
            && !(1..=MAX_INFERENCE_THREADS).contains(&threads)
        {
            return invalid_preference(
                "inference_threads",
                format!("must be between 1 and {MAX_INFERENCE_THREADS}"),
            );
        }
        if self.hotwords.is_empty() {
            if self.hotword_boost.is_some() {
                return invalid_preference("hotword_boost", "requires at least one hotword entry");
            }
            return Ok(());
        }
        PhraseBiasConfig::from_phrases_with_default_boost(
            self.hotwords.iter().cloned(),
            self.hotword_boost,
        )
        .map_err(|error| ConfigError::InvalidPreference {
            field: "hotwords",
            reason: error.to_string(),
        })?;
        Ok(())
    }
}

pub fn config_path(openasr_home: impl AsRef<Path>) -> PathBuf {
    openasr_home.as_ref().join("config.json")
}

pub fn load_config(openasr_home: impl AsRef<Path>) -> Result<OpenAsrConfig, ConfigError> {
    load_config_document(openasr_home).map(|document| document.config)
}

pub fn load_config_document(
    openasr_home: impl AsRef<Path>,
) -> Result<OpenAsrConfigDocument, ConfigError> {
    let path = config_path(openasr_home);
    match fs::read_to_string(&path) {
        Ok(contents) => serde_json::from_str(&contents)
            .map_err(|source| ConfigError::ParseConfig { path, source }),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            Ok(OpenAsrConfigDocument::default())
        }
        Err(source) => Err(ConfigError::ReadConfig { path, source }),
    }
}

pub fn save_config_document(
    openasr_home: impl AsRef<Path>,
    document: &OpenAsrConfigDocument,
) -> Result<(), ConfigError> {
    let home = openasr_home.as_ref();
    fs::create_dir_all(home).map_err(|source| ConfigError::CreateHome {
        path: home.to_path_buf(),
        source,
    })?;

    let path = config_path(home);
    let contents = serde_json::to_string_pretty(document).map_err(ConfigError::SerializeConfig)?;
    write_config_atomically(&path, format!("{contents}\n").as_bytes())
}

pub fn save_config(
    openasr_home: impl AsRef<Path>,
    config: &OpenAsrConfig,
) -> Result<(), ConfigError> {
    let home = openasr_home.as_ref();
    let mut document = load_config_document(home)?;
    document.config = config.clone();
    save_config_document(home, &document)
}

pub fn save_default_model_selection(
    openasr_home: impl AsRef<Path>,
    model_id: impl Into<String>,
    quant_preference: QuantPreference,
) -> Result<(), ConfigError> {
    let home = openasr_home.as_ref();
    let mut document = load_config_document(home)?;
    document.config.default_model = Some(model_id.into());
    document.preferences.quant_preference = quant_preference;
    save_config_document(home, &document)
}

fn write_config_atomically(path: &Path, contents: &[u8]) -> Result<(), ConfigError> {
    atomic_file::write_file_atomically(path, contents).map_err(|source| ConfigError::WriteConfig {
        path: path.to_path_buf(),
        source,
    })
}

fn default_model() -> Option<String> {
    Some(DEFAULT_MODEL_ID.to_string())
}

fn default_backend() -> Option<String> {
    Some(DEFAULT_BACKEND_ID.to_string())
}

fn default_preferences_version() -> u32 {
    PREFERENCES_SCHEMA_VERSION
}

fn default_tray_icon() -> bool {
    true
}

fn render_download_source_pref(pref: &DownloadSourcePref) -> String {
    match pref {
        DownloadSourcePref::Auto => "auto".to_string(),
        DownloadSourcePref::Pinned { source } => source.as_env_value().to_string(),
    }
}

fn default_dictation_shortcut() -> Option<String> {
    Some("CommandOrControl+Shift+Space".to_string())
}

fn resolve_default_model_config_value(
    registry: &[ModelCard],
    catalog: Option<&ModelCatalog>,
    value: String,
) -> Result<String, ConfigError> {
    let Some(catalog) = catalog else {
        resolve_registry_model_ref(registry, &value).map_err(ConfigError::ModelResolution)?;
        return Ok(value);
    };
    resolve_runtime_model_ref(registry, Some(catalog), &value)
        .map_err(ConfigError::RuntimeModelResolution)?;
    Ok(value)
}

fn validate_default_model_ref(
    registry: &[ModelCard],
    catalog: Option<&ModelCatalog>,
    value: &str,
) -> Result<(), ConfigError> {
    if let Some(catalog) = catalog {
        resolve_runtime_model_ref(registry, Some(catalog), value)
            .map_err(ConfigError::RuntimeModelResolution)?;
    } else {
        resolve_registry_model_ref(registry, value).map_err(ConfigError::ModelResolution)?;
    }
    Ok(())
}

fn invalid_preference(field: &'static str, reason: impl Into<String>) -> Result<(), ConfigError> {
    Err(ConfigError::InvalidPreference {
        field,
        reason: reason.into(),
    })
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod config_tests;
