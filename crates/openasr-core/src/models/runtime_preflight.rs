#[cfg(test)]
pub(crate) use crate::ggml_runtime::load_runtime_source_metadata_and_tensor_index;
#[cfg(test)]
pub(crate) use crate::ggml_runtime::load_runtime_source_metadata_and_tensor_index_from_source;
pub(crate) use crate::ggml_runtime::{
    RuntimeSourceTensorReaderError, build_runtime_tensor_reader_from_preflight,
};

/// Builds an explicit, valid proof for request-only tests that exercise a
/// dispatch/session seam without needing a family-specific tensor fixture.
/// The leaked temporary directory keeps the mapped source alive for the test;
/// callers still receive a real preflight value and never rely on a missing
/// `Option`/path-reopen fallback.
#[cfg(test)]
pub(crate) fn leaked_tiny_runtime_source_preflight()
-> crate::ggml_runtime::GgufRuntimeSourcePreflight {
    use std::collections::BTreeMap;
    use std::sync::OnceLock;

    static PREFLIGHT: OnceLock<crate::ggml_runtime::GgufRuntimeSourcePreflight> = OnceLock::new();
    PREFLIGHT
        .get_or_init(|| {
            // The one intentionally process-lived directory keeps the shared
            // mapped fixture portable to platforms that cannot unlink an open
            // file. Every request-only test clones the same immutable proof.
            let directory = Box::leak(Box::new(tempfile::tempdir().expect("test tempdir")));
            let path = directory.path().join("request-fixture.gguf");
            crate::testing::write_tiny_gguf_runtime_source(
                &path,
                &crate::testing::TinyGgufFixtureSpec::new(BTreeMap::new()),
            )
            .expect("write tiny request fixture");
            crate::ggml_runtime::load_runtime_source_metadata_and_tensor_index(&path)
                .expect("tiny request fixture must pass preflight")
        })
        .clone()
}

/// Wraps a request-only test preflight in the same proof value required by
/// production execution seams. Test fixtures still use a tiny synthetic GGUF,
/// but they cannot silently regress to constructing an execution request from
/// a bare preflight.
#[cfg(test)]
pub(crate) fn verified_pack_from_preflight_for_test(
    preflight: crate::ggml_runtime::GgufRuntimeSourcePreflight,
    model_architecture: &'static str,
) -> crate::models::pack_verifier::VerifiedPack {
    crate::models::pack_verifier::VerifiedPack::from_unverified_preflight_for_test(
        preflight,
        model_architecture,
    )
}
