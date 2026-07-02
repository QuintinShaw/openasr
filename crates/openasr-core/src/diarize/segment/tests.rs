use super::PyannetModel;

/// Parse the `<MAGIC><u32 ndim><u32 dims...><f32 data>` golden format.
fn read_golden(path: &str, magic: &[u8]) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[0..4], magic, "magic");
    let ndim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut dims = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        dims.push(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize);
        off += 4;
    }
    let n: usize = dims.iter().product();
    let data = bytes[off..off + n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (dims, data)
}

#[test]
#[ignore = "stage gate: needs OPENASR_PYANNOTE_{PACK,INPUT,SINCNET,LSTM1}"]
fn pyannet_stage_gates() {
    let pack = std::env::var("OPENASR_PYANNOTE_PACK").expect("pack");
    let input = std::env::var("OPENASR_PYANNOTE_INPUT").expect("input");
    let model = PyannetModel::from_safetensors(&std::fs::read(pack).unwrap()).unwrap();
    let (_, samples) = read_golden(&input, b"PYIN");
    let (sincnet, lstm1, _frames) = model.stages(&samples).unwrap();

    let (_, sinc_ref) = read_golden(&std::env::var("OPENASR_PYANNOTE_SINCNET").unwrap(), b"PYSN");
    let sinc_err = sincnet
        .iter()
        .zip(&sinc_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("sincnet max_abs_err={sinc_err:.5}");

    let (_, lstm_ref) = read_golden(&std::env::var("OPENASR_PYANNOTE_LSTM1").unwrap(), b"PYL1");
    let lstm_err = lstm1
        .iter()
        .zip(&lstm_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("lstm1 max_abs_err={lstm_err:.5}");
}

/// Host-local RTF measurement for the published pack's catalog perf entry:
/// segment the committed fixture clip and report `rtf_cpu` = wall time /
/// audio seconds, median of 5 warm runs. Run with `--release` when recording
/// numbers.
#[test]
#[ignore = "host-local bench: needs OPENASR_PYANNOTE_PACK; run with --release for catalog numbers"]
fn segmenter_rtf_bench_when_pack_present() {
    let Some(segmenter) = super::shared_segmenter() else {
        eprintln!("skipping: pyannote pack absent");
        return;
    };
    let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        wav,
        "pyannote rtf bench",
        "pyannote rtf bench",
    )
    .expect("fixture wav loads");
    let audio_seconds = samples.len() as f64 / 16_000.0;

    segmenter.segment(&samples, 16_000).expect("warm-up run");
    let mut runs: Vec<f64> = (0..5)
        .map(|_| {
            let start = std::time::Instant::now();
            segmenter.segment(&samples, 16_000).expect("timed run");
            start.elapsed().as_secs_f64()
        })
        .collect();
    runs.sort_by(f64::total_cmp);
    let rtf_cpu = runs[runs.len() / 2] / audio_seconds;
    println!("pyannote rtf_cpu={rtf_cpu:.5} over {audio_seconds:.2}s fixture audio");
}

/// Round-trip oracle for Subtask B: converting the real pyannote-seg safetensors
/// to a diarization `.oasr` (GGUF-v0, raw f32) and loading it back through
/// [`PyannetModel::from_oasr`] must reproduce a **byte-identical** forward pass vs
/// the safetensors fast path. A synthetic waveform keeps it self-contained (only
/// the safetensors pack pointed at by `OPENASR_PYANNOTE_PACK` is needed).
#[test]
#[ignore = "needs OPENASR_PYANNOTE_PACK pointing at the safetensors (uncommitted ~6MB)"]
fn oasr_roundtrip_matches_safetensors() {
    use crate::models::pyannote::package_import::{
        PyannoteImportRequest, convert_local_pyannote_source_to_runtime_pack,
    };

    let pack = std::env::var("OPENASR_PYANNOTE_PACK").expect("OPENASR_PYANNOTE_PACK");
    let model_st = PyannetModel::from_safetensors(&std::fs::read(&pack).unwrap()).unwrap();

    let out = std::env::temp_dir().join("oasr_pyannote_roundtrip.oasr");
    let _ = std::fs::remove_file(&out);
    convert_local_pyannote_source_to_runtime_pack(&PyannoteImportRequest {
        source_safetensors: std::path::PathBuf::from(&pack),
        output_root: out.clone(),
        model_id: "pyannote-roundtrip-test".to_string(),
    })
    .expect("pyannote .oasr conversion");
    let model_oasr = PyannetModel::from_oasr(&out).unwrap();

    // Deterministic 1 s synthetic waveform — forward() only does arithmetic.
    let samples: Vec<f32> = (0..16_000)
        .map(|i| ((i as f32) * 0.01).sin() * 0.3)
        .collect();
    let (logp_st, frames_st) = model_st.forward(&samples).unwrap();
    let (logp_oasr, frames_oasr) = model_oasr.forward(&samples).unwrap();
    assert_eq!(frames_st, frames_oasr, "frame count");
    assert_eq!(
        logp_st, logp_oasr,
        "the .oasr round-trip forward must be byte-identical to safetensors"
    );
    let _ = std::fs::remove_file(&out);
}

#[test]
#[ignore = "needs OPENASR_PYANNOTE_PACK + OPENASR_PYANNOTE_INPUT + OPENASR_PYANNOTE_GOLDEN"]
fn pyannet_matches_onnx_reference() {
    let pack = std::env::var("OPENASR_PYANNOTE_PACK").expect("pack");
    let input = std::env::var("OPENASR_PYANNOTE_INPUT").expect("input");
    let golden = std::env::var("OPENASR_PYANNOTE_GOLDEN").expect("golden");

    let model = PyannetModel::from_safetensors(&std::fs::read(pack).unwrap()).unwrap();
    let (in_dims, samples) = read_golden(&input, b"PYIN"); // [1,1,n]
    assert_eq!(in_dims.len(), 3);
    let (y_dims, reference) = read_golden(&golden, b"PYYY"); // [1,frames,7]

    let (logp, frames) = model.forward(&samples).unwrap();
    assert_eq!(frames, y_dims[1], "frame count");
    assert_eq!(logp.len(), reference.len());
    let max_err = logp
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("pyannet max_abs_err={max_err:.5} over {frames} frames");
    assert!(max_err < 1e-2, "pyannet max abs error {max_err} too high");
}
