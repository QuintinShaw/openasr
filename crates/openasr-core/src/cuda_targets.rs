// CUDA GPU-architecture target-list defaults and parsing.
//
// `build.rs` needs this logic to set CMake's `CUDA_ARCHITECTURES` when it
// compiles vendored ggml's CUDA backend, but a build script is not a test
// target: a `#[cfg(test)] mod tests` living only inside `build.rs` is never
// collected by `cargo test`/`cargo nextest`, so a regression there (e.g. the
// default arch list silently narrowing below vendored ggml's own floor, the
// class of bug fixed in #255/#196) would never fail CI. This file is the
// single implementation, `include!`d verbatim by `build.rs` (see its
// `cuda_gpu_targets()`) and compiled into this crate as an ordinary module
// (see `mod cuda_targets;` in `lib.rs`) so the canary test below actually
// runs as part of this crate's normal `cargo nextest` unit-test binary.
//
// Only `std` is used here (no crate-internal imports) so the exact same
// source text compiles cleanly in both contexts. Plain `//` comments only
// (not `//!` inner doc comments): this file is spliced into build.rs mid-file
// via `include!`, where an inner doc comment is a hard error (E0753) since it
// is not the first thing in that compilation unit.

/// Default CUDA arch list for a released binary: sm_75 through sm_90. Floor
/// is 75 (Turing: RTX 20xx, GTX 16xx, T4, 2080 Ti), not 70 (Volta) or lower,
/// because CUDA 13 removed device-code generation for Volta/Pascal/Maxwell
/// outright -- those need a CUDA 12 toolchain, which is a separate build leg
/// (tracked, not this one). 75 itself is deprecated-but-still-buildable on
/// CUDA 13.2: nvcc warns, but still emits working sm_75 SASS and PTX. No
/// explicit `-real`/`-virtual` suffix means CMake emits both device code and
/// PTX for every listed number, so this default also carries 75-virtual PTX
/// that the CUDA driver JITs forward to any newer, unlisted architecture at
/// load time -- the same forward-compat mechanism vendored ggml's own
/// (non-native) CMakeLists.txt default relies on. Upper bound stops at 90
/// (Hopper): 120 (Blackwell) needs CUDA 12.8+ and a CMake new enough for the
/// `f`/`a` suffix regex, which this default does not assume. Override with
/// `OPENASR_CUDA_GPU_TARGETS` for a narrower/wider/newer set.
pub(crate) const DEFAULT_CUDA_GPU_TARGETS: &str = "75;80;86;89;90";

pub(crate) fn cuda_gpu_targets_from_raw(raw: Option<&str>) -> String {
    raw.map(str::trim)
        .filter(|value| !value.is_empty())
        .map(normalize_cuda_gpu_targets)
        .unwrap_or_else(|| DEFAULT_CUDA_GPU_TARGETS.to_string())
}

pub(crate) fn normalize_cuda_gpu_targets(raw: &str) -> String {
    raw.split([',', ';', ' '])
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .strip_prefix("sm_")
                .or_else(|| value.strip_prefix("SM_"))
                .unwrap_or(value)
        })
        .collect::<Vec<_>>()
        .join(";")
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_CUDA_GPU_TARGETS, cuda_gpu_targets_from_raw};

    #[test]
    fn cuda_targets_default_to_common_cloud_and_consumer_arches() {
        assert_eq!(cuda_gpu_targets_from_raw(None), "75;80;86;89;90");
        assert_eq!(cuda_gpu_targets_from_raw(Some("   ")), "75;80;86;89;90");
    }

    #[test]
    fn cuda_targets_accept_common_arch_spellings() {
        assert_eq!(
            cuda_gpu_targets_from_raw(Some("sm_80,86; SM_89 90")),
            "80;86;89;90"
        );
    }

    #[test]
    fn cuda_default_targets_are_not_narrower_than_vendored_ggml_default() {
        // Regression guard for the class of bug in #255/#196: OpenASR's own
        // default CUDA arch list silently excluded sm_75 (Turing), so a
        // released CUDA binary hit "no kernel image available" on every
        // 2080 Ti / T4 / RTX 20xx / GTX 16xx card. Cross-check against
        // vendored ggml's own (non-native) CMakeLists.txt default, which is
        // the actual upstream source of truth this default is trying to
        // track: if a future ggml bump drops or narrows its own sm_75 floor,
        // this test's first assertion fails and flags that our comment/
        // reasoning in `cuda_gpu_targets_from_raw` needs re-deriving; if
        // OpenASR's own default regresses below that floor, the second
        // assertion fails instead.
        //
        // `env!("CARGO_MANIFEST_DIR")` (not a plain relative path) because
        // this file is compiled from two different locations -- as a normal
        // module of this crate, and `include!`d into `build.rs` -- and only
        // an absolute, manifest-anchored path resolves identically both ways.
        let vendored_ggml_cmake = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/third_party/openasr-ggml/src/ggml-cuda/CMakeLists.txt"
        ));
        assert!(
            vendored_ggml_cmake.contains("75-virtual"),
            "vendored ggml CMakeLists.txt no longer defaults to sm_75; re-check \
             whether OpenASR's own CUDA_GPU_TARGETS floor of 75 is still \
             upstream-aligned before changing it"
        );
        assert!(
            cuda_gpu_targets_from_raw(None)
                .split(';')
                .any(|arch| arch == "75"),
            "OpenASR's default CUDA arch list narrowed below upstream ggml's \
             own sm_75 floor"
        );
        // Keep the constant and the canary's own literal from silently
        // drifting apart.
        assert_eq!(cuda_gpu_targets_from_raw(None), DEFAULT_CUDA_GPU_TARGETS);
    }
}
