//! Tests for the neural-vs-energy engine-preference resolver shared by the
//! realtime CLI/server surfaces. Numerical parity and provider tests for
//! Stream-VAD itself live in `firered_stream::tests`.

#[test]
fn realtime_vad_prefers_neural_defaults_to_neural_with_env_precedence() {
    let saved = std::env::var("OPENASR_VAD").ok();
    // SAFETY: only this test (within the openasr-core test binary) touches the
    // OPENASR_VAD env; mutations are sequential and the original is restored.
    unsafe { std::env::remove_var("OPENASR_VAD") };

    // Default (no engine, no env) is neural; only an explicit energy/rms opts out;
    // an unrecognized engine falls through to the neural default.
    assert!(super::realtime_vad_prefers_neural(None));
    assert!(super::realtime_vad_prefers_neural(Some("neural")));
    assert!(super::realtime_vad_prefers_neural(Some(
        "definitely-not-an-engine"
    )));
    assert!(!super::realtime_vad_prefers_neural(Some("energy")));
    assert!(!super::realtime_vad_prefers_neural(Some("rms")));

    // OPENASR_VAD wins over the explicit engine in both directions.
    unsafe { std::env::set_var("OPENASR_VAD", "energy") };
    assert!(!super::realtime_vad_prefers_neural(Some("neural")));
    unsafe { std::env::set_var("OPENASR_VAD", "neural") };
    assert!(super::realtime_vad_prefers_neural(Some("energy")));

    match saved {
        Some(value) => unsafe { std::env::set_var("OPENASR_VAD", value) },
        None => unsafe { std::env::remove_var("OPENASR_VAD") },
    }
}
