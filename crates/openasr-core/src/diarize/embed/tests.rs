//! Parity tests for the WeSpeaker embedder.

use super::WeSpeakerResNet34Model;
use super::fbank::Fbank;
use super::{SpeakerEmbedder, WeSpeakerEmbedder};

const EXPECTED_WESPEAKER_SOURCE_NAME: &str =
    crate::models::wespeaker::package_import::WESPEAKER_EXPECTED_SOURCE_NAME;
const EXPECTED_WESPEAKER_SOURCE_REVISION: &str =
    crate::models::wespeaker::package_import::WESPEAKER_EXPECTED_SOURCE_REVISION;
const EXPECTED_WESPEAKER_CHECKPOINT_SHA256: &str =
    "366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56";

fn read_f32(bytes: &[u8], off: &mut usize, n: usize) -> Vec<f32> {
    let out = bytes[*off..*off + n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    *off += n * 4;
    out
}

fn read_u32(bytes: &[u8], off: &mut usize) -> u32 {
    let value = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
    *off += 4;
    value
}

fn read_string(bytes: &[u8], off: &mut usize) -> String {
    let len = read_u32(bytes, off) as usize;
    let value = std::str::from_utf8(&bytes[*off..*off + len])
        .unwrap()
        .to_string();
    *off += len;
    value
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// Host-local RTF measurement for the published pack's catalog perf entry:
/// embed the committed fixture clip (fbank + full network) and report
/// `rtf_cpu` = wall time / audio seconds, median of 5 warm runs. Run with
/// `--release` when recording numbers.
#[test]
#[ignore = "host-local bench: needs OPENASR_WESPEAKER_PACK; run with --release for catalog numbers"]
fn embedder_rtf_bench_when_pack_present() {
    let Some(embedder) = super::shared_embedder() else {
        eprintln!("skipping: wespeaker pack absent");
        return;
    };
    let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        wav,
        "wespeaker rtf bench",
        "wespeaker rtf bench",
    )
    .expect("fixture wav loads");
    let audio_seconds = samples.len() as f64 / 16_000.0;

    embedder.embed(&samples, 16_000).expect("warm-up embed");
    let mut runs: Vec<f64> = (0..5)
        .map(|_| {
            let start = std::time::Instant::now();
            embedder.embed(&samples, 16_000).expect("timed embed");
            start.elapsed().as_secs_f64()
        })
        .collect();
    runs.sort_by(f64::total_cmp);
    let rtf_cpu = runs[runs.len() / 2] / audio_seconds;
    println!("speaker_embedder rtf_cpu={rtf_cpu:.5} over {audio_seconds:.2}s fixture audio");
}

struct WeSpeakerGoldenCase {
    name: String,
    wav: Vec<f32>,
    frames: usize,
    features: Vec<f32>,
    embedding: Vec<f32>,
}

fn read_wespeaker_golden(path: impl AsRef<std::path::Path>) -> Vec<WeSpeakerGoldenCase> {
    // WSR1: u32 n_cases, source_name string, source_revision string,
    // checkpoint_sha256 string, then per case: u32 name_len, name bytes,
    // u32 n_samples, u32 frames, u32 dim, f32[n_samples] wav,
    // f32[frames*80] fbank, f32[dim] raw embedding.
    let g = std::fs::read(path).unwrap();
    assert_eq!(&g[0..4], b"WSR1", "golden magic");
    let mut off = 4;
    let n_cases = read_u32(&g, &mut off) as usize;
    let source_name = read_string(&g, &mut off);
    assert_eq!(
        source_name, EXPECTED_WESPEAKER_SOURCE_NAME,
        "WeSpeaker golden source_name mismatch: expected {EXPECTED_WESPEAKER_SOURCE_NAME}, got {source_name}; regenerate with tooling/publish-model/scripts/wespeaker_reference.py"
    );
    let source_revision = read_string(&g, &mut off);
    assert_eq!(
        source_revision, EXPECTED_WESPEAKER_SOURCE_REVISION,
        "WeSpeaker golden source_revision mismatch: expected {EXPECTED_WESPEAKER_SOURCE_REVISION}, got {source_revision}; regenerate with tooling/publish-model/scripts/wespeaker_reference.py"
    );
    let checkpoint_sha256 = read_string(&g, &mut off);
    assert_eq!(
        checkpoint_sha256, EXPECTED_WESPEAKER_CHECKPOINT_SHA256,
        "WeSpeaker golden checkpoint_sha256 mismatch: expected {EXPECTED_WESPEAKER_CHECKPOINT_SHA256}, got {checkpoint_sha256}; regenerate from the pinned checkpoint"
    );
    let mut cases = Vec::with_capacity(n_cases);
    for _ in 0..n_cases {
        let name = read_string(&g, &mut off);
        let n_samples = read_u32(&g, &mut off) as usize;
        let frames = read_u32(&g, &mut off) as usize;
        let dim = read_u32(&g, &mut off) as usize;
        let wav = read_f32(&g, &mut off, n_samples);
        let features = read_f32(&g, &mut off, frames * 80);
        let embedding = read_f32(&g, &mut off, dim);
        cases.push(WeSpeakerGoldenCase {
            name,
            wav,
            frames,
            features,
            embedding,
        });
    }
    assert_eq!(off, g.len(), "golden trailing bytes");
    cases
}

#[test]
#[ignore = "needs OPENASR_WESPEAKER_GOLDEN generated by tooling/publish-model/scripts/wespeaker_reference.py"]
fn wespeaker_fbank_matches_torchaudio_reference() {
    let golden = std::env::var("OPENASR_WESPEAKER_GOLDEN").expect("OPENASR_WESPEAKER_GOLDEN");
    let cases = read_wespeaker_golden(golden);
    let fbank = Fbank::wespeaker();
    for case in cases {
        let (features, frames) = fbank.compute(&case.wav);
        assert_eq!(frames, case.frames, "{}", case.name);
        let max_err = features
            .iter()
            .zip(&case.features)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("{} fbank max_abs_err={max_err:.6}", case.name);
        assert!(
            max_err < 2e-2,
            "{} fbank max abs error {max_err}",
            case.name
        );
    }
}

#[test]
#[ignore = "needs OPENASR_WESPEAKER_PACK + OPENASR_WESPEAKER_GOLDEN"]
fn wespeaker_network_matches_pyannote_reference() {
    let pack = std::env::var("OPENASR_WESPEAKER_PACK").expect("OPENASR_WESPEAKER_PACK");
    let golden = std::env::var("OPENASR_WESPEAKER_GOLDEN").expect("OPENASR_WESPEAKER_GOLDEN");
    let model = WeSpeakerResNet34Model::from_safetensors(&std::fs::read(pack).unwrap()).unwrap();
    let cases = read_wespeaker_golden(golden);
    for case in cases {
        let mine = model.forward(&case.features, case.frames).unwrap();
        assert_eq!(mine.len(), case.embedding.len(), "{}", case.name);
        let cos = cosine(&mine, &case.embedding);
        let max_err = mine
            .iter()
            .zip(&case.embedding)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "{} network cosine={cos:.8} max_abs_err={max_err:.6}",
            case.name
        );
        assert!(cos >= 0.9999, "{} network cosine {cos}", case.name);
    }
}

#[test]
#[ignore = "needs OPENASR_WESPEAKER_PACK + OPENASR_WESPEAKER_GOLDEN"]
fn wespeaker_embedder_matches_pyannote_reference() {
    let pack = std::env::var("OPENASR_WESPEAKER_PACK").expect("OPENASR_WESPEAKER_PACK");
    let golden = std::env::var("OPENASR_WESPEAKER_GOLDEN").expect("OPENASR_WESPEAKER_GOLDEN");
    let embedder = WeSpeakerEmbedder::from_safetensors(&std::fs::read(pack).unwrap()).unwrap();
    let cases = read_wespeaker_golden(golden);
    for case in cases {
        let mine = embedder.embed(&case.wav, 16_000).unwrap();
        assert_eq!(mine.dim(), case.embedding.len(), "{}", case.name);
        let cos = cosine(&mine.0, &case.embedding);
        println!("{} e2e cosine={cos:.8}", case.name);
        assert!(cos >= 0.9999, "{} e2e cosine {cos}", case.name);
    }
}

#[test]
#[ignore = "needs OPENASR_WESPEAKER_PACK pointing at the safetensors (uncommitted ~25MB)"]
fn wespeaker_oasr_roundtrip_matches_safetensors() {
    use crate::models::wespeaker::package_import::{
        WeSpeakerImportRequest, convert_local_wespeaker_source_to_runtime_pack,
    };

    let pack = std::env::var("OPENASR_WESPEAKER_PACK").expect("OPENASR_WESPEAKER_PACK");
    let model_st =
        WeSpeakerResNet34Model::from_safetensors(&std::fs::read(&pack).unwrap()).unwrap();

    let out = std::env::temp_dir().join("oasr_wespeaker_roundtrip.oasr");
    let _ = std::fs::remove_file(&out);
    convert_local_wespeaker_source_to_runtime_pack(&WeSpeakerImportRequest {
        source_safetensors: std::path::PathBuf::from(&pack),
        output_root: out.clone(),
        model_id: "wespeaker-roundtrip-test".to_string(),
        source_name: "pyannote/wespeaker-voxceleb-resnet34-LM".to_string(),
        source_revision: "837717ddb9ff5507820346191109dc79c958d614".to_string(),
        license_name: "CC-BY-4.0".to_string(),
        license_source: "https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM"
            .to_string(),
        quantization:
            crate::models::wespeaker::package_import::WeSpeakerRuntimeQuantizationMode::F32,
    })
    .expect("wespeaker .oasr conversion");
    let model_oasr = WeSpeakerResNet34Model::from_oasr(&out).unwrap();

    let t = 218usize;
    let features: Vec<f32> = (0..t * 80)
        .map(|i| ((i as f32) * 0.017).sin() * 0.25)
        .collect();
    let from_st = model_st.forward(&features, t).unwrap();
    let from_oasr = model_oasr.forward(&features, t).unwrap();
    assert_eq!(from_st, from_oasr);
    let _ = std::fs::remove_file(&out);
}
