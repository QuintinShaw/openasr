//! OADP Phase 0 end-to-end suite: hand-made `.oadp` adapter packs against the
//! REAL installed moonshine-tiny q8_0 base pack, activated through the
//! production surfaces — the server-side `OPENASR_ADAPTER` env var and the
//! request-level adapter path (`--adapter` plumbing) — and the public
//! transcription API.
//!
//! Oracles:
//! - ZERO adapter (B == 0, all eligible enc+dec targets, f32 AND f16 storage)
//!   must reproduce the real-speech baseline transcript EXACTLY (this is also
//!   the WER sanity gate: identical transcripts == identical WER on the
//!   committed jfk.wav fixture).
//! - A strong non-zero single-target adapter on `dec.blk.0.cross_v.weight`
//!   (the cross-KV precompute side-path) must CHANGE the transcript.
//! - Every base-binding mismatch class fails CLOSED with its specific error.
//!
//! Loud-fail prerequisites: moonshine-tiny q8_0 installed at
//! `~/.openasr/models/moonshine-tiny/q8_0/` (or OPENASR_MOONSHINE_OADP_REAL_PACK);
//! `fixtures/jfk.wav` checked in. `#[ignore]` is the opt-in; the tests never
//! silently skip.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use openasr_core::adapter_pack::{
    LoraAdapterDtype, LoraAdapterWriteRequest, LoraAdapterWriteTarget, OPENASR_ADAPTER_ENV,
    base_pack_model_id, file_sha256_hex, moonshine_lora_targetable_tensors,
    write_lora_adapter_pack,
};
use openasr_core::{ExecutionTarget, TranscriptionBackend, TranscriptionRequest};

static REAL_DECODE_LOCK: Mutex<()> = Mutex::new(());

const MOONSHINE_MODEL_ID: &str = "moonshine-tiny";
const MOONSHINE_OADP_REAL_PACK_ENV: &str = "OPENASR_MOONSHINE_OADP_REAL_PACK";
const MOONSHINE_PACK_HOME_RELATIVE_PATH: &str =
    ".openasr/models/moonshine-tiny/q8_0/moonshine-tiny-q8_0.oasr";
const MOONSHINE_JFK_BASELINE: &str = "And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.";

/// Sets `OPENASR_ADAPTER` for the duration of a scope; removal on drop keeps
/// later tests adapter-free. All uses run under `REAL_DECODE_LOCK`, so the
/// process-global env var is never mutated concurrently.
struct AdapterEnvGuard;

impl AdapterEnvGuard {
    fn set(path: &Path) -> Self {
        unsafe { std::env::set_var(OPENASR_ADAPTER_ENV, path) };
        Self
    }
}

impl Drop for AdapterEnvGuard {
    fn drop(&mut self) {
        unsafe { std::env::remove_var(OPENASR_ADAPTER_ENV) };
    }
}

fn resolve_real_pack() -> PathBuf {
    if let Some(value) = std::env::var_os(MOONSHINE_OADP_REAL_PACK_ENV) {
        let path = PathBuf::from(value);
        assert!(
            path.is_file(),
            "{MOONSHINE_OADP_REAL_PACK_ENV} must point to an existing moonshine .oasr pack: {}",
            path.display()
        );
        return path;
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!(
                "moonshine OADP real-pack prerequisites missing: HOME is not set and \
                 {MOONSHINE_OADP_REAL_PACK_ENV} is not set; #[ignore] is the opt-in, so this \
                 test must not silently skip"
            )
        });
    let path = home.join(MOONSHINE_PACK_HOME_RELATIVE_PATH);
    assert!(
        path.is_file(),
        "moonshine OADP real-pack prerequisites missing; install moonshine-tiny q8_0 (searched \
         {}) or set {MOONSHINE_OADP_REAL_PACK_ENV}",
        path.display()
    );
    path
}

fn jfk_audio() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("openasr-core lives under crates/openasr-core")
        .join("fixtures/jfk.wav");
    assert!(
        path.is_file(),
        "jfk.wav real-speech fixture missing at {}",
        path.display()
    );
    path
}

fn transcribe(pack_path: &Path, audio_path: &Path) -> Result<String, String> {
    openasr_core::NativeBackend
        .transcribe(
            TranscriptionRequest::new(audio_path, MOONSHINE_MODEL_ID)
                .with_model_pack_path(Some(pack_path.to_path_buf()))
                .with_execution_target(Some(ExecutionTarget::Cpu)),
        )
        .map(|transcription| transcription.text)
        .map_err(|error| error.to_string())
}

fn transcribe_ok(pack_path: &Path, audio_path: &Path) -> String {
    transcribe(pack_path, audio_path).unwrap_or_else(|error| {
        panic!(
            "real native transcription failed\npack path: {}\nadapter env: {:?}\nerror: {error}",
            pack_path.display(),
            std::env::var(OPENASR_ADAPTER_ENV).ok(),
        )
    })
}

struct AdapterSpec {
    file_name: &'static str,
    id: &'static str,
    base_model_id: Option<String>,
    base_pack_sha256: Option<String>,
    rank: u32,
    alpha: u32,
    dtype: LoraAdapterDtype,
    min_openasr_version: String,
    targets: Vec<LoraAdapterWriteTarget>,
}

impl AdapterSpec {
    fn new(file_name: &'static str, id: &'static str) -> Self {
        Self {
            file_name,
            id,
            base_model_id: None,
            base_pack_sha256: None,
            rank: 2,
            alpha: 2,
            dtype: LoraAdapterDtype::F32,
            min_openasr_version: env!("CARGO_PKG_VERSION").to_string(),
            targets: Vec::new(),
        }
    }
}

fn constant_targets(
    targets: &[(String, [usize; 2])],
    rank: usize,
    a_fill: f32,
    b_fill: f32,
) -> Vec<LoraAdapterWriteTarget> {
    targets
        .iter()
        .map(
            |(base_tensor, [input_dim, output_dim])| LoraAdapterWriteTarget {
                base_tensor: base_tensor.clone(),
                input_dim: *input_dim,
                output_dim: *output_dim,
                a_values: vec![a_fill; input_dim * rank],
                b_values: vec![b_fill; rank * output_dim],
            },
        )
        .collect()
}

/// Write a `.oadp` into `dir`, defaulting base identity to the real pack.
fn write_adapter(dir: &Path, base_pack: &Path, spec: AdapterSpec) -> PathBuf {
    let output_path = dir.join(spec.file_name);
    let request = LoraAdapterWriteRequest {
        output_path: output_path.clone(),
        id: spec.id.to_string(),
        base_model_id: spec
            .base_model_id
            .unwrap_or_else(|| base_pack_model_id(base_pack).expect("base model id")),
        base_pack_sha256: spec
            .base_pack_sha256
            .unwrap_or_else(|| file_sha256_hex(base_pack).expect("base pack sha256")),
        rank: spec.rank,
        alpha: spec.alpha,
        dtype: spec.dtype,
        min_openasr_version: spec.min_openasr_version,
        targets: spec.targets,
    };
    write_lora_adapter_pack(&request).expect("write .oadp adapter pack");
    output_path
}

#[test]
#[ignore = "real-pack OADP end-to-end: needs moonshine-tiny q8_0 installed (~/.openasr) or OPENASR_MOONSHINE_OADP_REAL_PACK"]
fn zero_adapter_transcript_exactly_matches_base_real_speech() {
    let _guard = REAL_DECODE_LOCK.lock().expect("real decode lock");
    let pack = resolve_real_pack();
    let audio = jfk_audio();
    let dir = tempfile::TempDir::new().expect("tempdir");
    let eligible = moonshine_lora_targetable_tensors(&pack).expect("eligible targets");
    assert!(
        !eligible.is_empty(),
        "moonshine base pack must expose LoRA-eligible tensors"
    );

    let baseline = transcribe_ok(&pack, &audio);
    assert_eq!(
        baseline,
        MOONSHINE_JFK_BASELINE,
        "moonshine real-speech baseline oracle drifted; pack {}",
        pack.display()
    );

    // ZERO adapter over EVERY eligible target (encoder + decoder, incl. all
    // cross projections), in both storage dtypes. Output must be EXACTLY the
    // baseline transcript — this is also the WER sanity gate (identical
    // transcript == identical WER on the committed real-speech fixture).
    for (file_name, dtype) in [
        ("zero-f32.oadp", LoraAdapterDtype::F32),
        ("zero-f16.oadp", LoraAdapterDtype::F16),
    ] {
        let mut spec = AdapterSpec::new(file_name, "zero-adapter");
        spec.dtype = dtype;
        spec.targets = constant_targets(&eligible, 2, 0.05, 0.0);
        let adapter_path = write_adapter(dir.path(), &pack, spec);

        let _env = AdapterEnvGuard::set(&adapter_path);
        let with_zero = transcribe_ok(&pack, &audio);
        assert_eq!(
            with_zero, baseline,
            "zero adapter ({file_name}) must reproduce the base transcript exactly"
        );
    }
}

#[test]
#[ignore = "real-pack OADP end-to-end: needs moonshine-tiny q8_0 installed (~/.openasr) or OPENASR_MOONSHINE_OADP_REAL_PACK"]
fn nonzero_cross_v_adapter_changes_transcript() {
    let _guard = REAL_DECODE_LOCK.lock().expect("real decode lock");
    let pack = resolve_real_pack();
    let audio = jfk_audio();
    let dir = tempfile::TempDir::new().expect("tempdir");
    let eligible = moonshine_lora_targetable_tensors(&pack).expect("eligible targets");
    let cross_v = eligible
        .iter()
        .find(|(name, _)| name == "dec.blk.0.cross_v.weight")
        .cloned()
        .expect("dec.blk.0.cross_v.weight must be eligible");

    let baseline = transcribe_ok(&pack, &audio);
    assert_eq!(baseline, MOONSHINE_JFK_BASELINE);

    // Strong single-target adapter on the cross-V projection: only reachable
    // through the per-utterance cross-KV precompute, so a changed transcript
    // proves that side-path is live through the production env-var surface.
    let mut spec = AdapterSpec::new("cross-v-strong.oadp", "cross-v-strong");
    spec.targets = constant_targets(std::slice::from_ref(&cross_v), 2, 0.1, 0.5);
    let adapter_path = write_adapter(dir.path(), &pack, spec);

    let _env = AdapterEnvGuard::set(&adapter_path);
    let with_adapter = transcribe_ok(&pack, &audio);
    assert_ne!(
        with_adapter, baseline,
        "a strong non-zero cross-V adapter must change the transcript (side-path disconnected?)"
    );
}

#[test]
#[ignore = "real-pack OADP end-to-end: needs moonshine-tiny q8_0 installed (~/.openasr) or OPENASR_MOONSHINE_OADP_REAL_PACK"]
fn adapter_base_binding_mismatches_fail_closed() {
    let _guard = REAL_DECODE_LOCK.lock().expect("real decode lock");
    let pack = resolve_real_pack();
    let audio = jfk_audio();
    let dir = tempfile::TempDir::new().expect("tempdir");
    let eligible = moonshine_lora_targetable_tensors(&pack).expect("eligible targets");
    let first = eligible.first().cloned().expect("eligible target");

    let assert_fails_with = |adapter_path: &Path, expected_fragment: &str| {
        let _env = AdapterEnvGuard::set(adapter_path);
        let error = transcribe(&pack, &audio).expect_err(&format!(
            "adapter '{}' must fail closed (expected error fragment {expected_fragment:?})",
            adapter_path.display()
        ));
        assert!(
            error.contains(expected_fragment),
            "fail-closed error for '{}' must mention {expected_fragment:?}; got: {error}",
            adapter_path.display()
        );
    };

    // (1) base pack sha256 mismatch.
    let mut spec = AdapterSpec::new("sha-mismatch.oadp", "sha-mismatch");
    spec.base_pack_sha256 = Some("f".repeat(64));
    spec.targets = constant_targets(std::slice::from_ref(&first), 2, 0.05, 0.0);
    let path = write_adapter(dir.path(), &pack, spec);
    assert_fails_with(&path, "sha256 mismatch");

    // (1b) the same rejection through the request-level adapter surface (the
    // CLI `--adapter` plumbing): no env var involved, proving the request
    // path reaches the executor's fail-closed validation.
    let error = openasr_core::NativeBackend
        .transcribe(
            TranscriptionRequest::new(&audio, MOONSHINE_MODEL_ID)
                .with_model_pack_path(Some(pack.clone()))
                .with_adapter_path(Some(path.clone()))
                .with_execution_target(Some(ExecutionTarget::Cpu)),
        )
        .map(|transcription| transcription.text)
        .expect_err("request-plumbed adapter with sha mismatch must fail closed");
    assert!(
        error.to_string().contains("sha256 mismatch"),
        "request-plumbed fail-closed error must mention sha256 mismatch; got: {error}"
    );

    // (2) base model id mismatch.
    let mut spec = AdapterSpec::new("model-id-mismatch.oadp", "model-id-mismatch");
    spec.base_model_id = Some("qwen3-asr-0.6b".to_string());
    spec.targets = constant_targets(std::slice::from_ref(&first), 2, 0.05, 0.0);
    let path = write_adapter(dir.path(), &pack, spec);
    assert_fails_with(&path, "base model id mismatch");

    // (3) target tensor outside the moonshine LoRA contract.
    let mut spec = AdapterSpec::new("bad-target.oadp", "bad-target");
    spec.targets = vec![LoraAdapterWriteTarget {
        base_tensor: "dec.emb.weight".to_string(),
        input_dim: 16,
        output_dim: 16,
        a_values: vec![0.0; 16 * 2],
        b_values: vec![0.0; 2 * 16],
    }];
    let path = write_adapter(dir.path(), &pack, spec);
    assert_fails_with(&path, "not a moonshine LoRA target");

    // (4) eligible target name but wrong dims for THIS base pack.
    let (name, [input_dim, output_dim]) = first.clone();
    let mut spec = AdapterSpec::new("dims-mismatch.oadp", "dims-mismatch");
    spec.targets = vec![LoraAdapterWriteTarget {
        base_tensor: name,
        input_dim: input_dim + 1,
        output_dim,
        a_values: vec![0.0; (input_dim + 1) * 2],
        b_values: vec![0.0; 2 * output_dim],
    }];
    let path = write_adapter(dir.path(), &pack, spec);
    assert_fails_with(&path, "dims mismatch");

    // (5) adapter requires a future OpenASR version.
    let mut spec = AdapterSpec::new("future-version.oadp", "future-version");
    spec.min_openasr_version = "999.0.0".to_string();
    spec.targets = constant_targets(std::slice::from_ref(&first), 2, 0.05, 0.0);
    let path = write_adapter(dir.path(), &pack, spec);
    assert_fails_with(&path, "requires OpenASR version");

    // After all the rejected adapters, no-adapter decode still works and the
    // failures left no cached adapter state behind.
    let baseline = transcribe_ok(&pack, &audio);
    assert_eq!(baseline, MOONSHINE_JFK_BASELINE);
}
