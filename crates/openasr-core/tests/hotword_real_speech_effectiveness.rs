use std::path::{Path, PathBuf};
use std::sync::Mutex;

use openasr_core::{
    ExecutionTarget, NativeBackend, PhraseBiasConfig, TranscriptionBackend, TranscriptionRequest,
};

static REAL_DECODE_LOCK: Mutex<()> = Mutex::new(());

const QWEN_MODEL_ID: &str = "qwen3-asr-0.6b";
const MOONSHINE_MODEL_ID: &str = "moonshine-tiny";
const QWEN_VALIDATED_QUANT: &str = "q8_0";
const MOONSHINE_VALIDATED_QUANT: &str = "q8_0";
const EXECUTION_TARGET: ExecutionTarget = ExecutionTarget::Cpu;
const QWEN_PACK_HOME_RELATIVE_PATH: &str =
    ".openasr/models/qwen3-asr-0.6b/q8_0/qwen3-asr-0.6b-q8_0.oasr";
const MOONSHINE_PACK_HOME_RELATIVE_PATH: &str =
    ".openasr/models/moonshine-tiny/q8_0/moonshine-tiny-q8_0.oasr";
const QWEN_PACK_ENV_NAMES: [&str; 3] = [
    "OPENASR_HOTWORD_QWEN_REAL_PACK",
    "OPENASR_QWEN_SERVE_BATCH_REAL_PACK",
    "OPENASR_QWEN_PREFILL_REAL_PACK",
];
const MOONSHINE_PACK_ENV_NAMES: [&str; 3] = [
    "OPENASR_HOTWORD_MOONSHINE_REAL_PACK",
    "OPENASR_MOONSHINE_SERVE_BATCH_REAL_PACK",
    "OPENASR_MOONSHINE_BATCH_REAL_PACK",
];
const CJK_NAME_AUDIO_ENV_NAME: &str = "OPENASR_HOTWORD_CJK_NAME_REAL_AUDIO";
const CJK_NAME_AUDIO_TMP_RELATIVE_PATH: &str = "tmp/hotword-real-case-1781250960.wav";
const QWEN_BASELINE: &str = "欢迎大家来体验达摩院推出的语音识别模型。";
const QWEN_CJK_NAME_BASELINE_MISS: &str = "我有一个朋友叫刁天成。";
const QWEN_CJK_NAME_CORRECTED: &str = "我有一个朋友叫刁天宸。";
const QWEN_NEGATIVE_DAMO: &str = "欢迎大家来体验大摩院推出的语音识别模型。";
const MOONSHINE_JFK_BASELINE: &str = "And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.";
const MOONSHINE_NEGATIVE_AMERICANS: &str = "And so my fellow America's ask not what your country can do for you as what you can do for your country.";

#[test]
#[ignore = "real-pack effectiveness: needs qwen3-asr-0.6b q8_0 .oasr pack; set OPENASR_HOTWORD_QWEN_REAL_PACK or install the local pack"]
fn qwen_real_speech_cjk_hotword_and_negative_boost_affect_decode() {
    let _guard = REAL_DECODE_LOCK.lock().expect("real decode lock");
    let pack_path = resolve_qwen_pack();
    let oracle = DecodeOracle::new(&pack_path, QWEN_VALIDATED_QUANT);
    let audio_path = repo_root().join("fixtures/jfk.wav");

    let baseline = transcribe_text(QWEN_MODEL_ID, oracle, &audio_path, None);
    oracle.assert_text_eq(&baseline, QWEN_BASELINE);

    // Current checked-in speech is already correctly decoded, so default boost is
    // an encoding/end-to-end sanity check rather than a positive correction case.
    let default_hotword = transcribe_text(
        QWEN_MODEL_ID,
        oracle,
        &audio_path,
        Some(default_hotword("达摩院")),
    );
    oracle.assert_text_eq(&default_hotword, QWEN_BASELINE);
    oracle.assert_contains(&default_hotword, "达摩院");

    let suppressed = transcribe_text(
        QWEN_MODEL_ID,
        oracle,
        &audio_path,
        Some(hotword("达摩院", -20.0)),
    );
    oracle.assert_text_eq(&suppressed, QWEN_NEGATIVE_DAMO);
    oracle.assert_contains(&baseline, "达摩院");
    oracle.assert_contains(&suppressed, "大摩院");
    oracle.assert_not_contains(&suppressed, "达摩院");

    let causal_baseline = transcribe_text(QWEN_MODEL_ID, oracle, &audio_path, None);
    oracle.assert_text_eq(&causal_baseline, &baseline);
}

#[test]
#[ignore = "host-local real-pack effectiveness: needs the qwen3-asr-0.6b q8_0 .oasr pack AND the host-local recording tmp/hotword-real-case-1781250960.wav (or OPENASR_HOTWORD_CJK_NAME_REAL_AUDIO)"]
fn qwen_real_speech_cjk_name_hotword_corrects_homophone_at_default_boost() {
    // POSITIVE correction case (real recording): the baseline decode mis-hears
    // the name 刁天宸 as the homophone 刁天成 (pre-bias logit gap 7-10 at the
    // final hanzi across quants). The default-boost hotword must flip it via the
    // depth-scaled continuation boost; this is the case a flat 5.0 boost lost.
    //
    // The recording is the user's own voice, so it stays HOST-LOCAL (gitignored
    // tmp/, or an explicit env path) and is never committed to the repo.
    let _guard = REAL_DECODE_LOCK.lock().expect("real decode lock");
    let pack_path = resolve_qwen_pack();
    let oracle = DecodeOracle::new(&pack_path, QWEN_VALIDATED_QUANT);
    let audio_path = resolve_cjk_name_real_audio();

    let baseline = transcribe_text(QWEN_MODEL_ID, oracle, &audio_path, None);
    oracle.assert_text_eq(&baseline, QWEN_CJK_NAME_BASELINE_MISS);

    let corrected = transcribe_text(
        QWEN_MODEL_ID,
        oracle,
        &audio_path,
        Some(default_hotword("刁天宸")),
    );
    oracle.assert_text_eq(&corrected, QWEN_CJK_NAME_CORRECTED);
    oracle.assert_contains(&corrected, "刁天宸");
    oracle.assert_not_contains(&corrected, "刁天成");

    // The hotword session must not leak into a following unbiased decode.
    let causal_baseline = transcribe_text(QWEN_MODEL_ID, oracle, &audio_path, None);
    oracle.assert_text_eq(&causal_baseline, &baseline);
}

#[test]
#[ignore = "real-pack effectiveness: needs moonshine-tiny q8_0 .oasr pack; set OPENASR_HOTWORD_MOONSHINE_REAL_PACK or install the local pack"]
fn moonshine_real_speech_hotword_and_negative_boost_affect_decode() {
    let _guard = REAL_DECODE_LOCK.lock().expect("real decode lock");
    let pack_path = resolve_moonshine_pack();
    let oracle = DecodeOracle::new(&pack_path, MOONSHINE_VALIDATED_QUANT);
    let audio_path = repo_root().join("fixtures/jfk.wav");

    let baseline = transcribe_text(MOONSHINE_MODEL_ID, oracle, &audio_path, None);
    oracle.assert_text_eq(&baseline, MOONSHINE_JFK_BASELINE);

    // JFK is stable at baseline; default boost confirms the phrase is accepted
    // through real transcription while negative boost proves the decode is biased.
    let default_hotword = transcribe_text(
        MOONSHINE_MODEL_ID,
        oracle,
        &audio_path,
        Some(default_hotword("Americans")),
    );
    oracle.assert_text_eq(&default_hotword, MOONSHINE_JFK_BASELINE);
    oracle.assert_contains(&default_hotword, "Americans");

    let suppressed = transcribe_text(
        MOONSHINE_MODEL_ID,
        oracle,
        &audio_path,
        Some(hotword("Americans", -20.0)),
    );
    oracle.assert_text_eq(&suppressed, MOONSHINE_NEGATIVE_AMERICANS);
    oracle.assert_contains(&baseline, "Americans");
    oracle.assert_contains(&suppressed, "America's");
    oracle.assert_not_contains(&suppressed, "Americans");

    let causal_baseline = transcribe_text(MOONSHINE_MODEL_ID, oracle, &audio_path, None);
    oracle.assert_text_eq(&causal_baseline, &baseline);
}

#[derive(Clone, Copy)]
struct DecodeOracle<'a> {
    pack_path: &'a Path,
    quant: &'static str,
    execution_target: ExecutionTarget,
}

impl<'a> DecodeOracle<'a> {
    fn new(pack_path: &'a Path, quant: &'static str) -> Self {
        Self {
            pack_path,
            quant,
            execution_target: EXECUTION_TARGET,
        }
    }

    fn assert_text_eq(self, observed: &str, expected: &str) {
        assert_eq!(
            observed,
            expected,
            "oracle snapshot mismatch\npack path: {}\nquant: {}\nexecution backend: {}\nexpected text: {expected:?}\nobserved text: {observed:?}",
            self.pack_path.display(),
            self.quant,
            self.execution_target.as_str(),
        );
    }

    fn assert_contains(self, observed: &str, expected_fragment: &str) {
        assert!(
            observed.contains(expected_fragment),
            "oracle snapshot mismatch\npack path: {}\nquant: {}\nexecution backend: {}\nexpected text: transcript containing {expected_fragment:?}\nobserved text: {observed:?}",
            self.pack_path.display(),
            self.quant,
            self.execution_target.as_str(),
        );
    }

    fn assert_not_contains(self, observed: &str, unexpected_fragment: &str) {
        assert!(
            !observed.contains(unexpected_fragment),
            "oracle snapshot mismatch\npack path: {}\nquant: {}\nexecution backend: {}\nexpected text: transcript not containing {unexpected_fragment:?}\nobserved text: {observed:?}",
            self.pack_path.display(),
            self.quant,
            self.execution_target.as_str(),
        );
    }
}

fn transcribe_text(
    model_id: &str,
    oracle: DecodeOracle<'_>,
    audio_path: &Path,
    phrase_bias: Option<PhraseBiasConfig>,
) -> String {
    NativeBackend
        .transcribe(
            TranscriptionRequest::new(audio_path, model_id)
                .with_model_pack_path(Some(oracle.pack_path.to_path_buf()))
                .with_execution_target(Some(oracle.execution_target))
                .with_phrase_bias(phrase_bias),
        )
        .unwrap_or_else(|error| {
            panic!(
                "real native transcription failed\npack path: {}\nquant: {}\nexecution backend: {}\nerror: {error}",
                oracle.pack_path.display(),
                oracle.quant,
                oracle.execution_target.as_str(),
            )
        })
        .text
}

fn default_hotword(phrase: &str) -> PhraseBiasConfig {
    PhraseBiasConfig::from_phrases_with_default_boost([phrase], None)
        .expect("valid default hotword config")
}

fn hotword(phrase: &str, boost: f32) -> PhraseBiasConfig {
    PhraseBiasConfig::from_phrases([(phrase, boost)]).expect("valid hotword config")
}

/// Resolve the HOST-LOCAL real recording for the CJK-name positive-correction
/// case. The audio is the user's own voice saying a real person's name, so it
/// must never be committed (release hygiene: no user/customer audio in the
/// repo); it lives in the gitignored `tmp/` of the checkout, or wherever
/// `OPENASR_HOTWORD_CJK_NAME_REAL_AUDIO` points. Loud-fail like the pack
/// resolvers: `#[ignore]` is the opt-in, so a missing prerequisite must panic
/// with instructions rather than silently pass.
fn resolve_cjk_name_real_audio() -> PathBuf {
    if let Some(value) = std::env::var_os(CJK_NAME_AUDIO_ENV_NAME) {
        let path = PathBuf::from(value);
        assert!(
            path.is_file(),
            "{CJK_NAME_AUDIO_ENV_NAME} must point to an existing wav recording: {}",
            path.display()
        );
        return path;
    }
    let path = repo_root().join(CJK_NAME_AUDIO_TMP_RELATIVE_PATH);
    if path.is_file() {
        return path;
    }
    panic!(
        "CJK-name hotword real-speech test prerequisites missing; #[ignore] is the opt-in, so this test must not silently skip.\nThe recording is host-local user audio and is intentionally NOT committed.\nrequired env var: {CJK_NAME_AUDIO_ENV_NAME}\nsearched path:\n- {}",
        path.display()
    );
}

fn resolve_qwen_pack() -> PathBuf {
    resolve_pack(
        QWEN_MODEL_ID,
        QWEN_VALIDATED_QUANT,
        &QWEN_PACK_ENV_NAMES,
        QWEN_PACK_HOME_RELATIVE_PATH,
    )
}

fn resolve_moonshine_pack() -> PathBuf {
    resolve_pack(
        MOONSHINE_MODEL_ID,
        MOONSHINE_VALIDATED_QUANT,
        &MOONSHINE_PACK_ENV_NAMES,
        MOONSHINE_PACK_HOME_RELATIVE_PATH,
    )
}

fn resolve_pack(
    label: &str,
    validated_quant: &str,
    env_names: &[&str],
    home_relative_path: &str,
) -> PathBuf {
    for env_name in env_names {
        if let Some(value) = std::env::var_os(env_name) {
            let path = PathBuf::from(value);
            assert!(
                path.is_file(),
                "{env_name} must point to an existing {label} {validated_quant} .oasr pack: {}",
                path.display()
            );
            assert_pack_path_matches_quant(label, validated_quant, Some(env_name), &path);
            return path;
        }
    }

    let mut searched_paths = Vec::new();
    if let Some(home_dir) = std::env::var_os("HOME").map(PathBuf::from) {
        let path = home_dir.join(home_relative_path);
        searched_paths.push(path.display().to_string());
        if path.is_file() {
            assert_pack_path_matches_quant(label, validated_quant, None, &path);
            return path;
        }
    } else {
        searched_paths.push(format!("$HOME/{home_relative_path} (HOME is not set)"));
    }

    panic!(
        "{label} hotword real-speech test prerequisites missing; #[ignore] is the opt-in, so this test must not silently skip.\nrequired env vars: {}\nvalidated quant: {validated_quant}\nsearched paths:\n{}",
        env_names.join(", "),
        searched_paths
            .iter()
            .map(|path| format!("- {path}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn assert_pack_path_matches_quant(
    label: &str,
    validated_quant: &str,
    env_name: Option<&str>,
    path: &Path,
) {
    let filename_matches = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains(validated_quant));
    let parent_matches = path
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == validated_quant);

    assert!(
        filename_matches || parent_matches,
        "{} must point to the validated {label} {validated_quant} .oasr pack; got {}",
        env_name.unwrap_or("local pack path"),
        path.display()
    );
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("openasr-core lives under crates/openasr-core")
        .to_path_buf()
}
