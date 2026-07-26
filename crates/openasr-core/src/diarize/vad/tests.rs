//! Tests for the neural-vs-energy engine-preference resolver shared by the
//! realtime CLI/server surfaces. Numerical parity and provider tests for
//! Stream-VAD itself live in `firered_stream::tests`.

#[test]
fn realtime_vad_prefers_neural_defaults_to_neural_with_env_precedence() {
    crate::test_process_env::with_test_process_env([("OPENASR_VAD", None)], || {
        // Default (no engine, no env) is neural; only an explicit energy/rms opts out;
        // an unrecognized engine falls through to the neural default.
        assert!(super::realtime_vad_prefers_neural(None));
        assert!(super::realtime_vad_prefers_neural(Some("neural")));
        assert!(super::realtime_vad_prefers_neural(Some(
            "definitely-not-an-engine"
        )));
        assert!(!super::realtime_vad_prefers_neural(Some("energy")));
        assert!(!super::realtime_vad_prefers_neural(Some("rms")));
    });
    crate::test_process_env::with_test_process_env(
        [("OPENASR_VAD", Some("energy".into()))],
        || assert!(!super::realtime_vad_prefers_neural(Some("neural"))),
    );
    crate::test_process_env::with_test_process_env(
        [("OPENASR_VAD", Some("neural".into()))],
        || assert!(super::realtime_vad_prefers_neural(Some("energy"))),
    );
}
