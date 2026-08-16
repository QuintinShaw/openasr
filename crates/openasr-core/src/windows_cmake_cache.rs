//! Native ggml source fingerprinting and Windows CMake cache validation.
//!
//! `build.rs` includes this file directly because a build script cannot depend
//! on the crate it configures. The library also compiles it under `cfg(test)` so
//! the regression tests run under `cargo nextest`; tests declared only inside a
//! build script are never collected.

use std::{
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

/// Directories CMake can compile or include for the production ggml library.
///
/// Cargo recursively watches directory paths emitted through
/// `rerun-if-changed`, and the fingerprint walks the same roots. Keep this list
/// as the single source of truth so a newly added native source cannot be
/// omitted from one invalidation mechanism while appearing in the other.
pub(crate) const BUILD_RELEVANT_DIRECTORIES: &[&str] = &["cmake", "include", "src"];

/// Build-relevant files at the vendored ggml root, outside the directories
/// above. Tests/examples are disabled by `build.rs`, so their trees are not
/// native inputs and intentionally do not invalidate the expensive CMake build.
pub(crate) const BUILD_RELEVANT_FILES: &[&str] = &["CMakeLists.txt", "ggml.pc.in"];

pub(crate) const SOURCE_FINGERPRINT_STAMP: &str = "openasr-native-inputs.sha256";

/// Hash every build-relevant native input by normalized relative path and
/// contents. The result is independent of checkout location, directory
/// enumeration order, file mtimes, and git metadata.
pub(crate) fn build_relevant_fingerprint(source_dir: &Path) -> io::Result<String> {
    let mut files = Vec::new();
    for relative in BUILD_RELEVANT_DIRECTORIES {
        collect_regular_files(source_dir, Path::new(relative), &mut files)?;
    }
    for relative in BUILD_RELEVANT_FILES {
        let relative = PathBuf::from(relative);
        ensure_regular_file(&source_dir.join(&relative))?;
        files.push(relative);
    }

    files.sort_by_key(|path| normalized_relative_path(path));

    let mut digest = Sha256::new();
    digest.update(b"openasr-native-inputs-v1\0");
    let mut buffer = [0_u8; 64 * 1024];
    for relative in files {
        let normalized = normalized_relative_path(&relative);
        hash_framed_bytes(&mut digest, normalized.as_bytes());

        let mut file = File::open(source_dir.join(&relative))?;
        let mut file_digest = Sha256::new();
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            file_digest.update(&buffer[..read]);
        }
        digest.update(file_digest.finalize());
    }

    Ok(format!("{:x}", digest.finalize()))
}

/// A missing or different stamp means the private CMake tree cannot be reused.
/// Trimming accepts the newline written by `build.rs` without weakening the
/// exact fingerprint comparison.
pub(crate) fn source_fingerprint_requires_reset(
    stored_fingerprint: Option<&str>,
    expected_fingerprint: &str,
) -> bool {
    stored_fingerprint.is_none_or(|stored| stored.trim() != expected_fingerprint)
}

fn collect_regular_files(
    source_dir: &Path,
    relative_dir: &Path,
    files: &mut Vec<PathBuf>,
) -> io::Result<()> {
    let directory = source_dir.join(relative_dir);
    if !directory.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("native input directory is missing: {}", directory.display()),
        ));
    }

    for entry in fs::read_dir(&directory)? {
        let entry = entry?;
        let relative = relative_dir.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_regular_files(source_dir, &relative, files)?;
        } else if file_type.is_file() {
            files.push(relative);
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "native input must be a regular file or directory: {}",
                    entry.path().display()
                ),
            ));
        }
    }
    Ok(())
}

fn ensure_regular_file(path: &Path) -> io::Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("native input file is missing: {}", path.display()),
        ))
    }
}

fn normalized_relative_path(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn hash_framed_bytes(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

pub(crate) fn cache_matches_contract(
    cache: &str,
    tool_expectations: &[(&str, String)],
    scalar_expectations: &[(&str, String)],
) -> bool {
    tool_expectations.iter().all(|(name, expected)| {
        cache_value(cache, name)
            .is_some_and(|actual| windows_tool_matches(actual, expected.as_str()))
    }) && scalar_expectations.iter().all(|(name, expected)| {
        cache_value(cache, name).is_some_and(|actual| scalar_matches(actual, expected.as_str()))
    })
}

pub(crate) fn cache_value<'a>(cache: &'a str, name: &str) -> Option<&'a str> {
    cache.lines().find_map(|line| {
        let (key_and_type, value) = line.split_once('=')?;
        let (key, _) = key_and_type.split_once(':')?;
        (key == name).then_some(value.trim())
    })
}

fn windows_tool_matches(actual: &str, expected: &str) -> bool {
    let actual = normalize_windows_tool(actual);
    let expected = normalize_windows_tool(expected);
    if expected.contains('/') {
        return actual == expected;
    }

    windows_tool_basename(&actual) == windows_tool_basename(&expected)
}

fn normalize_windows_tool(raw: &str) -> String {
    let mut normalized = raw
        .trim()
        .trim_matches('"')
        .replace('\\', "/")
        .to_ascii_lowercase();
    if let Some(without_prefix) = normalized.strip_prefix("//?/") {
        normalized = without_prefix.to_owned();
    }
    while normalized.contains("//") {
        normalized = normalized.replace("//", "/");
    }
    normalized.trim_end_matches('/').to_owned()
}

fn windows_tool_basename(normalized: &str) -> &str {
    normalized
        .rsplit('/')
        .next()
        .unwrap_or(normalized)
        .strip_suffix(".exe")
        .unwrap_or_else(|| normalized.rsplit('/').next().unwrap_or(normalized))
}

fn scalar_matches(actual: &str, expected: &str) -> bool {
    normalize_scalar(actual) == normalize_scalar(expected)
}

fn normalize_scalar(raw: &str) -> String {
    match raw.trim().to_ascii_uppercase().as_str() {
        "1" | "ON" | "TRUE" | "YES" => "ON".to_owned(),
        "0" | "OFF" | "FALSE" | "NO" => "OFF".to_owned(),
        other => other.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use tempfile::tempdir;

    use super::{
        SOURCE_FINGERPRINT_STAMP, build_relevant_fingerprint, cache_matches_contract,
        source_fingerprint_requires_reset,
    };

    fn write(root: &Path, relative: &str, contents: &str) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().expect("fixture path has a parent"))
            .expect("create fixture directory");
        fs::write(path, contents).expect("write fixture file");
    }

    fn native_fixture(root: &Path, reverse_creation_order: bool) {
        let files = [
            ("CMakeLists.txt", "add_subdirectory(src)\n"),
            ("ggml.pc.in", "prefix=@CMAKE_INSTALL_PREFIX@\n"),
            ("cmake/common.cmake", "set(GGML_VERSION 1)\n"),
            ("include/ggml.h", "void ggml_old(void);\n"),
            ("src/CMakeLists.txt", "add_library(ggml ggml.c)\n"),
            ("src/ggml.c", "void ggml_old(void) {}\n"),
        ];
        if reverse_creation_order {
            for (relative, contents) in files.iter().rev() {
                write(root, relative, contents);
            }
        } else {
            for (relative, contents) in files {
                write(root, relative, contents);
            }
        }
    }

    fn scalar_contract() -> Vec<(&'static str, String)> {
        vec![
            ("CMAKE_BUILD_TYPE", "Release".to_owned()),
            ("BUILD_SHARED_LIBS", "OFF".to_owned()),
            ("GGML_BACKEND_DL", "OFF".to_owned()),
            ("GGML_CUDA", "ON".to_owned()),
            ("GGML_VULKAN", "ON".to_owned()),
            ("GGML_BUILD_TESTS", "OFF".to_owned()),
        ]
    }

    fn cache(compiler: &str, shared: &str, cuda: &str) -> String {
        format!(
            "CMAKE_C_COMPILER:FILEPATH={compiler}\n\
             CMAKE_CXX_COMPILER:FILEPATH={compiler}\n\
             CMAKE_BUILD_TYPE:STRING=Release\n\
             BUILD_SHARED_LIBS:BOOL={shared}\n\
             GGML_BACKEND_DL:BOOL=OFF\n\
             GGML_CUDA:BOOL={cuda}\n\
             GGML_VULKAN:BOOL=ON\n\
             GGML_BUILD_TESTS:BOOL=OFF\n"
        )
    }

    fn cache_with_source(source_dir: &Path, compiler: &str, shared: &str, cuda: &str) -> String {
        format!(
            "CMAKE_HOME_DIRECTORY:INTERNAL={}\n{}",
            source_dir.display(),
            cache(compiler, shared, cuda)
        )
    }

    #[test]
    fn equivalent_windows_compiler_spelling_matches() {
        let actual = cache(
            "D:/Toolchain/VS/VC/Tools/MSVC/14.44/bin/HostX64/x64/cl.exe",
            "OFF",
            "ON",
        );
        let tools = vec![
            (
                "CMAKE_C_COMPILER",
                r"d:\toolchain\vs\vc\tools\msvc\14.44\bin\hostx64\x64\CL.EXE".to_owned(),
            ),
            (
                "CMAKE_CXX_COMPILER",
                r"d:\toolchain\vs\vc\tools\msvc\14.44\bin\hostx64\x64\CL.EXE".to_owned(),
            ),
        ];

        assert!(cache_matches_contract(&actual, &tools, &scalar_contract()));
    }

    #[test]
    fn compiler_version_change_requires_fresh_configure() {
        let actual = cache(
            "D:/Toolchain/VS/VC/Tools/MSVC/14.39/bin/HostX64/x64/cl.exe",
            "OFF",
            "ON",
        );
        let tools = vec![
            (
                "CMAKE_C_COMPILER",
                "D:/Toolchain/VS/VC/Tools/MSVC/14.44/bin/HostX64/x64/cl.exe".to_owned(),
            ),
            (
                "CMAKE_CXX_COMPILER",
                "D:/Toolchain/VS/VC/Tools/MSVC/14.44/bin/HostX64/x64/cl.exe".to_owned(),
            ),
        ];

        assert!(!cache_matches_contract(&actual, &tools, &scalar_contract()));
    }

    #[test]
    fn shared_or_gpu_topology_drift_requires_fresh_configure() {
        let compiler = "D:/Toolchain/VS/VC/Tools/MSVC/14.44/bin/HostX64/x64/cl.exe";
        let tools = vec![
            ("CMAKE_C_COMPILER", compiler.to_owned()),
            ("CMAKE_CXX_COMPILER", compiler.to_owned()),
        ];

        assert!(!cache_matches_contract(
            &cache(compiler, "ON", "ON"),
            &tools,
            &scalar_contract()
        ));
        assert!(!cache_matches_contract(
            &cache(compiler, "OFF", "OFF"),
            &tools,
            &scalar_contract()
        ));
    }

    #[test]
    fn bare_tool_name_accepts_cmake_resolved_path() {
        let actual = cache("C:/LLVM/bin/clang-cl.exe", "OFF", "ON");
        let tools = vec![
            ("CMAKE_C_COMPILER", "clang-cl".to_owned()),
            ("CMAKE_CXX_COMPILER", "clang-cl.exe".to_owned()),
        ];

        assert!(cache_matches_contract(&actual, &tools, &scalar_contract()));
    }

    #[test]
    fn identical_native_contents_at_different_source_path_require_fresh_configure() {
        let first = tempdir().expect("create first source fixture");
        let second = tempdir().expect("create second source fixture");
        native_fixture(first.path(), false);
        native_fixture(second.path(), false);
        assert_eq!(
            build_relevant_fingerprint(first.path()).expect("fingerprint first source"),
            build_relevant_fingerprint(second.path()).expect("fingerprint second source")
        );

        let compiler = "D:/Toolchain/VS/VC/Tools/MSVC/14.44/bin/HostX64/x64/cl.exe";
        let tools = vec![
            ("CMAKE_HOME_DIRECTORY", second.path().display().to_string()),
            ("CMAKE_C_COMPILER", compiler.to_owned()),
            ("CMAKE_CXX_COMPILER", compiler.to_owned()),
        ];
        let actual = cache_with_source(first.path(), compiler, "OFF", "ON");

        assert!(!cache_matches_contract(&actual, &tools, &scalar_contract()));
    }

    #[test]
    fn same_source_path_keeps_compatible_cache() {
        let source_dir = Path::new(r"D:\Staging\openasr-ggml");
        let compiler = "D:/Toolchain/VS/VC/Tools/MSVC/14.44/bin/HostX64/x64/cl.exe";
        let tools = vec![
            ("CMAKE_HOME_DIRECTORY", source_dir.display().to_string()),
            ("CMAKE_C_COMPILER", compiler.to_owned()),
            ("CMAKE_CXX_COMPILER", compiler.to_owned()),
        ];
        let actual = cache_with_source(
            Path::new(r"\\?\d:\STAGING\openasr-ggml\"),
            compiler,
            "OFF",
            "ON",
        );

        assert!(cache_matches_contract(&actual, &tools, &scalar_contract()));
    }

    #[test]
    fn fingerprint_changes_for_native_content_addition_and_deletion() {
        let fixture = tempdir().expect("create fixture tempdir");
        native_fixture(fixture.path(), false);
        let baseline = build_relevant_fingerprint(fixture.path()).expect("fingerprint baseline");

        write(
            fixture.path(),
            "src/ggml.c",
            "void ggml_old(void) {}\nvoid ggml_new(void) {}\n",
        );
        let modified = build_relevant_fingerprint(fixture.path()).expect("fingerprint modified");
        assert_ne!(baseline, modified);

        write(
            fixture.path(),
            "src/new-op.c",
            "void ggml_new_op(void) {}\n",
        );
        let added = build_relevant_fingerprint(fixture.path()).expect("fingerprint added");
        assert_ne!(modified, added);

        fs::remove_file(fixture.path().join("include/ggml.h")).expect("remove native header");
        let deleted = build_relevant_fingerprint(fixture.path()).expect("fingerprint deleted");
        assert_ne!(added, deleted);
    }

    #[test]
    fn fingerprint_is_stable_across_creation_and_enumeration_order() {
        let forward = tempdir().expect("create forward fixture");
        let reverse = tempdir().expect("create reverse fixture");
        native_fixture(forward.path(), false);
        native_fixture(reverse.path(), true);

        assert_eq!(
            build_relevant_fingerprint(forward.path()).expect("fingerprint forward"),
            build_relevant_fingerprint(reverse.path()).expect("fingerprint reverse")
        );
    }

    #[test]
    fn documentation_outside_build_roots_does_not_change_fingerprint() {
        let fixture = tempdir().expect("create fixture tempdir");
        native_fixture(fixture.path(), false);
        let before = build_relevant_fingerprint(fixture.path()).expect("fingerprint before docs");

        write(
            fixture.path(),
            "docs/benchmark-notes.md",
            "not a native build input\n",
        );

        assert_eq!(
            before,
            build_relevant_fingerprint(fixture.path()).expect("fingerprint after docs")
        );
    }

    #[test]
    fn missing_or_mismatched_source_fingerprint_requires_cache_reset() {
        assert_eq!(SOURCE_FINGERPRINT_STAMP, "openasr-native-inputs.sha256");
        let expected = "0123456789abcdef";
        assert!(source_fingerprint_requires_reset(None, expected));
        assert!(source_fingerprint_requires_reset(Some("old"), expected));
        assert!(!source_fingerprint_requires_reset(
            Some("0123456789abcdef\n"),
            expected
        ));
    }
}
