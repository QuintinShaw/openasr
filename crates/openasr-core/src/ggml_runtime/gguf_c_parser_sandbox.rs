use std::{
    env, io,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{
    GgmlRuntimeSource, GgmlRuntimeSourcePathError, GgufMetadata, GgufMetadataReadError,
    GgufTensorIndex, GgufTensorIndexReadError, gguf_tensor_index::GgufTensorIndexSnapshot,
    read_gguf_metadata_from_runtime_source, read_gguf_tensor_index_from_runtime_source,
    validate_ggml_runtime_source_path,
};

pub const GGUF_C_PARSER_SANDBOX_HELPER_ARG: &str = "__openasr-gguf-c-parser-probe";
const SANDBOX_MODE_ENV: &str = "OPENASR_GGUF_C_PARSER_SANDBOX";
const SANDBOX_CHILD_ENV: &str = "OPENASR_GGUF_C_PARSER_SANDBOX_CHILD";
const SANDBOX_TIMEOUT_MS_ENV: &str = "OPENASR_GGUF_C_PARSER_SANDBOX_TIMEOUT_MS";
const DEFAULT_SANDBOX_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_CHILD_STDOUT_BYTES: usize = 128 * 1024 * 1024;
const MAX_CHILD_STDERR_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct GgufCParserProbeOutput {
    metadata: GgufMetadata,
    tensor_index: GgufTensorIndexSnapshot,
}

#[derive(Debug, Error)]
pub enum GgufCParserSandboxError {
    #[error(transparent)]
    InvalidRuntimeSource(#[from] GgmlRuntimeSourcePathError),
    #[error("sandboxed gguf C parser child setup failed: {source}")]
    ChildLimit {
        #[source]
        source: io::Error,
    },
    #[error("could not read gguf metadata from '{path}' in sandbox child: {source}")]
    MetadataRead {
        path: PathBuf,
        #[source]
        source: Box<GgufMetadataReadError>,
    },
    #[error("could not read gguf tensor index from '{path}' in sandbox child: {source}")]
    TensorIndexRead {
        path: PathBuf,
        #[source]
        source: Box<GgufTensorIndexReadError>,
    },
    #[error("could not serialize sandboxed gguf C parser output: {source}")]
    Serialize {
        #[source]
        source: serde_json::Error,
    },
    #[error("could not locate current executable for sandboxed gguf C parser: {source}")]
    CurrentExe {
        #[source]
        source: io::Error,
    },
    #[error("could not spawn sandboxed gguf C parser helper '{helper}': {source}")]
    Spawn { helper: PathBuf, source: io::Error },
    #[error("could not wait for sandboxed gguf C parser helper '{helper}': {source}")]
    Wait { helper: PathBuf, source: io::Error },
    #[error("sandboxed gguf C parser helper timed out for '{path}' after {timeout_ms}ms")]
    Timeout { path: PathBuf, timeout_ms: u64 },
    #[error("could not read sandboxed gguf C parser {stream}: {source}")]
    OutputRead {
        stream: &'static str,
        source: io::Error,
    },
    #[error("sandboxed gguf C parser helper failed for '{path}' with status {status}: {stderr}")]
    HelperFailed {
        path: PathBuf,
        status: String,
        stderr: String,
    },
    #[error("could not decode sandboxed gguf C parser output: {source}")]
    Decode {
        #[source]
        source: serde_json::Error,
    },
    #[error("could not rebuild sandboxed gguf tensor index: {source}")]
    RebuildTensorIndex {
        #[source]
        source: Box<GgufTensorIndexReadError>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SandboxMode {
    Disabled,
    Auto,
    Required,
}

pub(crate) fn load_gguf_metadata_and_tensor_index_with_c_parser_sandbox(
    runtime_source: &GgmlRuntimeSource,
) -> Result<(GgufMetadata, GgufTensorIndex), GgufCParserSandboxError> {
    let mode = sandbox_mode();
    if mode == SandboxMode::Disabled {
        return direct_load(runtime_source);
    }
    let helper =
        env::current_exe().map_err(|source| GgufCParserSandboxError::CurrentExe { source })?;
    if mode == SandboxMode::Auto && !is_openasr_helper_exe(&helper) {
        return direct_load(runtime_source);
    }
    load_with_child(runtime_source, &helper, sandbox_timeout())
}

pub fn render_gguf_c_parser_sandbox_child_output(
    path: &Path,
) -> Result<String, GgufCParserSandboxError> {
    if env::var_os(SANDBOX_CHILD_ENV).is_some() {
        apply_child_limits().map_err(|source| GgufCParserSandboxError::ChildLimit { source })?;
    }
    let runtime_source = validate_ggml_runtime_source_path(path)?;
    let (metadata, tensor_index) = direct_load(&runtime_source)?;
    let output = GgufCParserProbeOutput {
        metadata,
        tensor_index: tensor_index.to_snapshot(),
    };
    serde_json::to_string(&output).map_err(|source| GgufCParserSandboxError::Serialize { source })
}

fn direct_load(
    runtime_source: &GgmlRuntimeSource,
) -> Result<(GgufMetadata, GgufTensorIndex), GgufCParserSandboxError> {
    let metadata = read_gguf_metadata_from_runtime_source(runtime_source).map_err(|source| {
        GgufCParserSandboxError::MetadataRead {
            path: runtime_source.path().to_path_buf(),
            source: Box::new(source),
        }
    })?;
    let tensor_index =
        read_gguf_tensor_index_from_runtime_source(runtime_source).map_err(|source| {
            GgufCParserSandboxError::TensorIndexRead {
                path: runtime_source.path().to_path_buf(),
                source: Box::new(source),
            }
        })?;
    Ok((metadata, tensor_index))
}

fn load_with_child(
    runtime_source: &GgmlRuntimeSource,
    helper: &Path,
    timeout: Duration,
) -> Result<(GgufMetadata, GgufTensorIndex), GgufCParserSandboxError> {
    let mut command = Command::new(helper);
    command
        .arg(GGUF_C_PARSER_SANDBOX_HELPER_ARG)
        .arg(runtime_source.path())
        .env_clear()
        .env(SANDBOX_CHILD_ENV, "1");
    configure_sandbox_child_environment(&mut command, helper);

    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| GgufCParserSandboxError::Spawn {
            helper: helper.to_path_buf(),
            source,
        })?;

    let stdout = child.stdout.take().expect("stdout is configured as piped");
    let stderr = child.stderr.take().expect("stderr is configured as piped");
    let stdout_reader = thread::spawn(move || read_to_end_limited(stdout, MAX_CHILD_STDOUT_BYTES));
    let stderr_reader = thread::spawn(move || read_to_end_limited(stderr, MAX_CHILD_STDERR_BYTES));

    let status =
        wait_with_timeout(&mut child, timeout).map_err(|source| GgufCParserSandboxError::Wait {
            helper: helper.to_path_buf(),
            source,
        })?;
    let Some(status) = status else {
        let _ = child.kill();
        let _ = child.wait();
        let _ = stdout_reader.join();
        let _ = stderr_reader.join();
        return Err(GgufCParserSandboxError::Timeout {
            path: runtime_source.path().to_path_buf(),
            timeout_ms: duration_millis_u64(timeout),
        });
    };

    let stdout = join_reader(stdout_reader, "stdout")?;
    let stderr = join_reader(stderr_reader, "stderr")?;
    if !status.success() {
        return Err(GgufCParserSandboxError::HelperFailed {
            path: runtime_source.path().to_path_buf(),
            status: status.to_string(),
            stderr: String::from_utf8_lossy(&stderr).trim().to_string(),
        });
    }

    let output: GgufCParserProbeOutput = serde_json::from_slice(&stdout)
        .map_err(|source| GgufCParserSandboxError::Decode { source })?;
    let tensor_index = GgufTensorIndex::from_snapshot(output.tensor_index).map_err(|source| {
        GgufCParserSandboxError::RebuildTensorIndex {
            source: Box::new(source),
        }
    })?;
    Ok((output.metadata, tensor_index))
}

#[cfg(windows)]
fn configure_sandbox_child_environment(command: &mut Command, helper: &Path) {
    for key in ["SystemRoot", "WINDIR"] {
        if let Some(value) = env::var_os(key) {
            command.env(key, value);
        }
    }

    let path_entries = sandbox_child_windows_path_entries(helper);
    if path_entries.is_empty() {
        return;
    }

    if let Ok(path) = env::join_paths(path_entries) {
        command.env("PATH", path);
    }
}

#[cfg(not(windows))]
fn configure_sandbox_child_environment(_command: &mut Command, _helper: &Path) {}

#[cfg(windows)]
fn sandbox_child_windows_path_entries(helper: &Path) -> Vec<PathBuf> {
    let mut entries = Vec::new();
    if let Some(parent) = helper.parent() {
        push_unique_existing_dir(&mut entries, parent.to_path_buf());
    }

    for key in ["HIP_PATH", "ROCM_PATH", "ROCM_HOME"] {
        if let Some(root) = env::var_os(key) {
            push_unique_existing_dir(&mut entries, PathBuf::from(root).join("bin"));
        }
    }

    if let Some(path) = env::var_os("PATH") {
        for entry in env::split_paths(&path) {
            if windows_dir_contains_accelerated_runtime_dll(&entry) {
                push_unique_existing_dir(&mut entries, entry);
            }
        }
    }

    entries
}

#[cfg(windows)]
fn push_unique_existing_dir(entries: &mut Vec<PathBuf>, path: PathBuf) {
    if !path.is_dir() || entries.iter().any(|entry| entry == &path) {
        return;
    }
    entries.push(path);
}

#[cfg(windows)]
fn windows_dir_contains_accelerated_runtime_dll(path: &Path) -> bool {
    path.join("rocblas.dll").is_file()
        || path.join("libhipblas.dll").is_file()
        || path.join("amdhip64_7.dll").is_file()
        || path.join("amdhip64.dll").is_file()
}

fn read_to_end_limited(mut reader: impl Read, limit: usize) -> Result<Vec<u8>, io::Error> {
    let mut output = Vec::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            return Ok(output);
        }
        if output.len().saturating_add(read) > limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("sandbox child output exceeded {limit} bytes"),
            ));
        }
        output.extend_from_slice(&buffer[..read]);
    }
}

fn join_reader(
    handle: thread::JoinHandle<Result<Vec<u8>, io::Error>>,
    stream: &'static str,
) -> Result<Vec<u8>, GgufCParserSandboxError> {
    handle
        .join()
        .map_err(|_| GgufCParserSandboxError::OutputRead {
            stream,
            source: io::Error::other("reader thread panicked"),
        })?
        .map_err(|source| GgufCParserSandboxError::OutputRead { stream, source })
}

fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout: Duration,
) -> Result<Option<std::process::ExitStatus>, io::Error> {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if started.elapsed() >= timeout {
            return Ok(None);
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn sandbox_mode() -> SandboxMode {
    match env::var(SANDBOX_MODE_ENV)
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("0" | "off" | "false" | "disabled") => SandboxMode::Disabled,
        Some("1" | "on" | "true" | "required") => SandboxMode::Required,
        Some("auto") | Some("") | None => SandboxMode::Auto,
        Some(_) => SandboxMode::Required,
    }
}

fn is_openasr_helper_exe(path: &Path) -> bool {
    // Accept the bundled sidecar name ("openasr") and any triple-suffixed /
    // resource-dir variant ("openasr-aarch64-apple-darwin", "openasr-cli", ...)
    // so Auto-mode engages the sandbox for every OpenASR helper, not just the
    // exact "openasr" stem. The probe child runs the hidden subcommand on the
    // direct-load path, so this does not recurse.
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| stem == "openasr" || stem.starts_with("openasr-"))
}

fn sandbox_timeout() -> Duration {
    env::var(SANDBOX_TIMEOUT_MS_ENV)
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|millis| *millis > 0)
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_SANDBOX_TIMEOUT)
}

fn duration_millis_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(unix)]
fn apply_child_limits() -> Result<(), io::Error> {
    // Cast to the inferred resource type: libc's RLIMIT_* constants are `u32`
    // on Linux but `c_int` on macOS, so a fixed `as u32` is redundant on one
    // platform and required on the other. `as _` is correct on both.
    set_rlimit(libc::RLIMIT_CORE as _, 0, 0)?;
    set_rlimit(libc::RLIMIT_CPU as _, 30, 30)
}

#[cfg(unix)]
fn set_rlimit(resource: u32, soft: u64, hard: u64) -> Result<(), io::Error> {
    let limit = libc::rlimit {
        rlim_cur: soft,
        rlim_max: hard,
    };
    let result = unsafe { libc::setrlimit(resource as _, &limit) };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(not(unix))]
fn apply_child_limits() -> Result<(), io::Error> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

    #[test]
    fn child_output_round_trips_metadata_and_tensor_index() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("tiny.oasr");
        let spec = TinyGgufFixtureSpec::whisper_oasr_v1_non_streaming_cpu("whisper-small");
        write_tiny_gguf_runtime_source(&path, &spec).expect("write fixture");

        let rendered =
            render_gguf_c_parser_sandbox_child_output(&path).expect("render child output");
        let output: GgufCParserProbeOutput =
            serde_json::from_str(&rendered).expect("decode child output");
        let tensor_index =
            GgufTensorIndex::from_snapshot(output.tensor_index).expect("rebuild tensor index");

        assert_eq!(
            output.metadata.get_string("openasr.model.id"),
            Some("whisper-small")
        );
        assert!(!tensor_index.tensors().is_empty());
        assert_eq!(tensor_index.path(), path.as_path());
    }
}
