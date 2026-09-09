#![no_main]

use libfuzzer_sys::fuzz_target;
use openasr_core::{PackCandidate, PackVerifier};

fuzz_target!(|data: &[u8]| {
    let home = match std::env::var_os("OPENASR_HOME") {
        Some(home) => std::path::PathBuf::from(home),
        None => std::env::temp_dir().join("openasr-fuzz-oasr"),
    };
    if std::fs::create_dir_all(&home).is_err() {
        return;
    }
    let path = home.join("fuzz-input.oasr");
    if std::fs::write(&path, data).is_err() {
        return;
    }
    let _ = openasr_core::ggml_runtime::probe_ggml_package_path(&path);
    let _ = PackVerifier.verify_candidate(PackCandidate::new(&path));
});
