use std::{
    fs,
    path::{Path, PathBuf},
};

use thiserror::Error;

use crate::{ResponseFormat, recognized_audio_extensions};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchInput {
    pub input_dir: PathBuf,
    pub files: Vec<BatchItem>,
    pub skipped_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchItem {
    pub input_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchOutput {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchFailure {
    pub input_path: PathBuf,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchSummary {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub format: ResponseFormat,
    pub model: String,
    pub backend: String,
    pub files_found: usize,
    pub files_transcribed: usize,
    pub files_skipped: usize,
    pub files_failed: usize,
    pub outputs: Vec<BatchOutput>,
    pub failures: Vec<BatchFailure>,
}

#[derive(Debug, Error)]
pub enum BatchError {
    #[error(
        "Batch input directory not found: {path}\nPlease provide an existing directory containing supported audio or video files."
    )]
    InputNotFound { path: PathBuf },
    #[error(
        "Batch input path is not a directory: {path}\nPlease provide an existing directory containing supported audio or video files."
    )]
    InputNotDirectory { path: PathBuf },
    #[error(
        "Could not read batch input directory: {path}\nPlease check the path and directory permissions."
    )]
    InputMetadata { path: PathBuf },
    #[error(
        "Could not scan batch input directory: {path}\nPlease check the path and directory permissions."
    )]
    ReadDir { path: PathBuf },
    #[error(
        "No supported audio or video files found in: {path}\nSupported extensions: {extensions}."
    )]
    NoSupportedFiles { path: PathBuf, extensions: String },
}

pub fn discover_batch_inputs(input_dir: impl AsRef<Path>) -> Result<BatchInput, BatchError> {
    let input_dir = input_dir.as_ref();
    let mut entries = discover_directory_entries(input_dir)?;
    entries.sort_by(|(left, _), (right, _)| left.cmp(right));

    let (files, skipped_count) = classify_supported_entries(entries);

    if files.is_empty() {
        return Err(BatchError::NoSupportedFiles {
            path: input_dir.to_path_buf(),
            extensions: supported_extensions_sentence(),
        });
    }

    Ok(BatchInput {
        input_dir: input_dir.to_path_buf(),
        files,
        skipped_count,
    })
}

fn discover_directory_entries(input_dir: &Path) -> Result<Vec<(PathBuf, bool)>, BatchError> {
    let metadata = fs::metadata(input_dir).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            BatchError::InputNotFound {
                path: input_dir.to_path_buf(),
            }
        } else {
            BatchError::InputMetadata {
                path: input_dir.to_path_buf(),
            }
        }
    })?;

    if !metadata.is_dir() {
        return Err(BatchError::InputNotDirectory {
            path: input_dir.to_path_buf(),
        });
    }
    let mut entries = Vec::new();
    for entry in fs::read_dir(input_dir).map_err(|_| read_dir_error(input_dir))? {
        let entry = entry.map_err(|_| read_dir_error(input_dir))?;
        let path = entry.path();
        let metadata = entry.metadata().map_err(|_| read_dir_error(input_dir))?;
        entries.push((path, metadata.is_file()));
    }
    Ok(entries)
}

fn read_dir_error(input_dir: &Path) -> BatchError {
    BatchError::ReadDir {
        path: input_dir.to_path_buf(),
    }
}

fn classify_supported_entries(entries: Vec<(PathBuf, bool)>) -> (Vec<BatchItem>, usize) {
    let mut files = Vec::new();
    let mut skipped_count = 0;
    for (path, is_file) in entries {
        if is_file && is_supported_batch_input(&path) {
            files.push(BatchItem { input_path: path });
        } else {
            skipped_count += 1;
        }
    }
    (files, skipped_count)
}

pub fn batch_output_path(
    output_dir: impl AsRef<Path>,
    input_path: impl AsRef<Path>,
    format: ResponseFormat,
) -> PathBuf {
    let input_path = input_path.as_ref();
    let file_name = input_path.file_name().unwrap_or(input_path.as_os_str());
    let mut output_name = file_name.to_os_string();
    output_name.push(".");
    output_name.push(response_format_extension(format));
    output_dir.as_ref().join(output_name)
}

pub fn response_format_extension(format: ResponseFormat) -> &'static str {
    format.output_extension()
}

pub fn render_batch_summary(summary: &BatchSummary) -> String {
    let mut rendered = format!(
        "OpenASR batch transcription\n\nInput directory: {}\nOutput directory: {}\nFormat: {}\nModel: {}\nBackend: {}\nFiles found: {}\nFiles transcribed: {}\nFiles skipped: {}\nFiles failed: {}\n\nOutputs:\n",
        summary.input_dir.display(),
        summary.output_dir.display(),
        summary.format,
        summary.model,
        summary.backend,
        summary.files_found,
        summary.files_transcribed,
        summary.files_skipped,
        summary.files_failed
    );

    for output in &summary.outputs {
        rendered.push_str(&format!(
            "- {} -> {}\n",
            output.input_path.display(),
            output.output_path.display()
        ));
    }

    if !summary.failures.is_empty() {
        rendered.push_str("\nFailures:\n");
        for failure in &summary.failures {
            rendered.push_str(&format!(
                "- {}: {}\n",
                failure.input_path.display(),
                concise_error(&failure.error)
            ));
        }
    }

    rendered
}

fn concise_error(error: &str) -> String {
    error
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("unknown error")
        .trim()
        .to_string()
}

fn is_supported_batch_input(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .is_some_and(|extension| recognized_audio_extensions().contains(&extension.as_str()))
}

fn supported_extensions_sentence() -> String {
    recognized_audio_extensions().join(", ")
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod batch_tests;
