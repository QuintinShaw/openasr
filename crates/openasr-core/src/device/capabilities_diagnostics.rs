pub(super) fn env_absent_note() -> Vec<String> {
    vec![
        "No environment hint was present; no SDK is linked and no external probe command is run."
            .to_string(),
    ]
}

pub(super) fn env_present_note(present: &[String]) -> Vec<String> {
    vec![format!(
        "Environment hint(s) present: {}. This does not prove provider readiness.",
        present.join(", ")
    )]
}

pub(super) fn backend_not_linked_note() -> Vec<String> {
    vec!["No SDK is linked and no framework/device enumeration is performed by M58A.".to_string()]
}

pub(super) fn directml_placeholder_note() -> Vec<String> {
    vec![
        "DirectML placeholder is OS-scoped only; no DirectML SDK or adapter probe is used."
            .to_string(),
    ]
}

pub(super) fn rocm_backend_linked_with_devices_note(
    tuning: &str,
    devices: &[String],
) -> Vec<String> {
    vec![format!(
        "ROCm/HIP ggml backend is linked. HIP tuning: {tuning}. ggml reported device(s): {}.",
        devices.join("; ")
    )]
}

pub(super) fn rocm_backend_linked_no_device_note(tuning: &str) -> Vec<String> {
    vec![format!(
        "ROCm/HIP ggml backend is linked, but ggml did not report a ROCm/HIP GPU device. HIP tuning: {tuning}."
    )]
}
