//! Pure validation for the Windows ggml CMake cache contract.
//!
//! `build.rs` includes this file directly because a build script cannot depend
//! on the crate it configures. The library also compiles it under `cfg(test)` so
//! the regression tests run under `cargo nextest`; tests declared only inside a
//! build script are never collected.

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

fn cache_value<'a>(cache: &'a str, name: &str) -> Option<&'a str> {
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
    use super::cache_matches_contract;

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
}
