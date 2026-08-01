//! Real-pack end-to-end equivalence for the concurrent long-audio slice
//! pipeline (P1): decoding the SAME recording with the pipeline width forced to
//! 1 (serial), forced to 4 (concurrent), and left to the carry-gated default
//! (env unset; concurrent for this carry-disabled run) must produce a
//! byte-identical transcript.
//!
//! Isolating the concurrency variable: the production serial path threads a
//! cross-slice prompt carry between slices, while the concurrent path is
//! carry-light (it drops that carry). Comparing "serial + carry" against
//! "concurrent + no carry" would confound concurrency with the carry
//! difference. This test therefore disables prompt carry on BOTH runs (via
//! request-level [`LongFormOptions::carry_prompt_across_slices`] = false), so
//! the ONLY thing that differs between the width-1 and width-4 runs is
//! concurrency. Byte-identical output is then evidence that concurrent ggml
//! decoding does not change the transcript.
//!
//! Backend: CPU (deterministic greedy decode) so a run-to-run diff cannot come
//! from GPU nondeterminism. Model: the real moonshine-tiny q8_0 pack -- a
//! `SharedWindow` family, so a ~69s recording genuinely slices into several
//! chunks and the width-4 path engages.
//!
//! Loud-fail prerequisites (never a silent skip; `#[ignore]` is the opt-in):
//! - moonshine-tiny q8_0 pack at `~/.openasr/models/moonshine-tiny/q8_0/...`
//!   or pointed to by `OPENASR_SLICE_PIPELINE_REAL_PACK`;
//! - `fixtures/longform_en_zh.wav` checked in (a ~69s recording).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use openasr_core::{
    ExecutionTarget, LongFormMode, LongFormOptions, TranscriptionBackend, TranscriptionRequest,
};

/// The pipeline width is read from a process-global env var, so the two runs
/// must not race each other or any other test mutating it.
static PIPELINE_WIDTH_ENV_LOCK: Mutex<()> = Mutex::new(());

const MODEL_ID: &str = "moonshine-tiny";
const REAL_PACK_ENV: &str = "OPENASR_SLICE_PIPELINE_REAL_PACK";
const PACK_HOME_RELATIVE_PATH: &str =
    ".openasr/models/moonshine-tiny/q8_0/moonshine-tiny-q8_0.oasr";
const SLICE_PIPELINE_WIDTH_ENV: &str = "OPENASR_SLICE_PIPELINE_WIDTH";

fn resolve_real_pack() -> PathBuf {
    if let Some(value) = std::env::var_os(REAL_PACK_ENV) {
        let path = PathBuf::from(value);
        assert!(
            path.is_file(),
            "{REAL_PACK_ENV} must point to an existing moonshine .oasr pack: {}",
            path.display()
        );
        return path;
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!(
                "slice-pipeline real-pack prerequisites missing: neither HOME nor {REAL_PACK_ENV} \
             is set; #[ignore] is the opt-in, so this test must not silently skip"
            )
        });
    let path = home.join(PACK_HOME_RELATIVE_PATH);
    assert!(
        path.is_file(),
        "slice-pipeline real-pack prerequisites missing; install moonshine-tiny q8_0 (searched \
         {}) or set {REAL_PACK_ENV}",
        path.display()
    );
    path
}

fn longform_audio() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("openasr-core lives under crates/openasr-core")
        .join("fixtures/longform_en_zh.wav");
    assert!(
        path.is_file(),
        "longform_en_zh.wav fixture missing at {}",
        path.display()
    );
    path
}

/// Carry-light long-form options that force several fixed 10s slices, so both
/// the serial and the concurrent run see the SAME multi-slice plan with NO
/// cross-slice prompt carry.
fn carry_light_multi_slice_options() -> LongFormOptions {
    LongFormOptions {
        mode: LongFormMode::Fixed,
        chunk_seconds: 10.0,
        carry_prompt_across_slices: false,
        ..LongFormOptions::default()
    }
}

/// `width` `Some(n)` pins `OPENASR_SLICE_PIPELINE_WIDTH=n`; `None` leaves the
/// variable unset so the run exercises the carry-gated default (this is a
/// carry-disabled run, so the default takes the concurrent pipeline).
fn transcribe_with_width(pack: &Path, audio: &Path, width: Option<usize>) -> String {
    let _lock = PIPELINE_WIDTH_ENV_LOCK.lock().expect("pipeline width lock");
    match width {
        Some(width) => unsafe { std::env::set_var(SLICE_PIPELINE_WIDTH_ENV, width.to_string()) },
        None => unsafe { std::env::remove_var(SLICE_PIPELINE_WIDTH_ENV) },
    }
    let result = openasr_core::NativeBackend.transcribe(
        TranscriptionRequest::new(audio, MODEL_ID)
            .with_model_pack_path(Some(pack.to_path_buf()))
            .with_execution_target(Some(ExecutionTarget::Cpu))
            .with_longform(Some(carry_light_multi_slice_options())),
    );
    unsafe { std::env::remove_var(SLICE_PIPELINE_WIDTH_ENV) };
    result
        .unwrap_or_else(|error| {
            panic!("real native transcription failed at width {width:?}: {error}")
        })
        .text
}

#[test]
#[ignore = "real-pack P1 pipeline: needs moonshine-tiny q8_0 installed (~/.openasr) or \
            OPENASR_SLICE_PIPELINE_REAL_PACK; runs the real ggml decode three times on \
            fixtures/longform_en_zh.wav"]
fn concurrent_width_matches_serial_width_byte_for_byte() {
    let pack = resolve_real_pack();
    let audio = longform_audio();

    // Width 1: serial path, carry disabled -> carry-light serial reference.
    let serial = transcribe_with_width(&pack, &audio, Some(1));
    assert!(
        !serial.trim().is_empty(),
        "the ~69s recording must produce a non-empty transcript"
    );

    // Width 4: concurrent carry-light path over the SAME plan.
    let concurrent = transcribe_with_width(&pack, &audio, Some(4));

    assert_eq!(
        serial, concurrent,
        "concurrent (width=4) and serial (width=1) carry-light transcripts must be \
         byte-identical; the only difference between the runs is concurrency"
    );

    // Env unset: the shipping default for this carry-disabled run, which takes
    // the concurrent pipeline via the carry-gated default width. Its transcript
    // must still match the serial reference byte for byte.
    let default_run = transcribe_with_width(&pack, &audio, None);

    assert_eq!(
        serial, default_run,
        "the carry-gated default (env unset, carry-disabled run -> concurrent) must \
         stay byte-identical to the explicit serial run"
    );
}
