//! Parity tests for the ReDimNet2-B6 embedder.

use super::RedimNet2Embedder;
use super::SpeakerEmbedder;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[ignore = "host-local bench: needs OPENASR_REDIMNET_PACK; run with --release for catalog numbers"]
#[test]
fn embedder_rtf_bench_when_pack_present() {
    let Some(embedder) = super::shared_embedder() else {
        eprintln!("skipping: redimnet2-b6 pack absent");
        return;
    };
    let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        wav,
        "redimnet rtf bench",
        "redimnet rtf bench",
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

fn redimnet_spike_root() -> Option<std::path::PathBuf> {
    match crate::testing::external_test_fixture_path(
        "OPENASR_REDIMNET_SPIKE_ROOT",
        "ReDimNet parity fixture directory",
    ) {
        Ok(path) => Some(path),
        Err(skip) => {
            eprintln!("skipping: {skip}");
            None
        }
    }
}

/// Plain C-order f32 `.npy` loader (no fortran-order handling needed for the
/// golden embedding dumps), matching the loader in `redimnet::backbone::tests`.
fn read_redimnet_golden_embedding(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(&bytes[..6], b"\x93NUMPY", "npy magic");
    let major = bytes[6];
    let header_len = if major == 1 {
        u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize
    } else {
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize
    };
    let header_start = if major == 1 { 10 } else { 12 };
    let data_start = header_start + header_len;
    bytes[data_start..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "requires local redimnet2-spike assets under tmp/ (not committed)"]
fn redimnet_embedder_matches_python_reference_e2e_jfk() {
    let Some(root) = redimnet_spike_root() else {
        return;
    };
    let pack = root.join("redimnet2-b6-f32.oasr");
    if !pack.exists() {
        eprintln!("skip: {pack:?} not present");
        return;
    }
    let embedder = RedimNet2Embedder::from_oasr(&pack).expect("load redimnet2-b6 f32 pack");
    assert_eq!(embedder.embedding_dim(), 192);
    assert_eq!(embedder.embedding_space_version(), "redimnet2-b6-cn-v1");

    let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        wav,
        "redimnet e2e test",
        "redimnet e2e test",
    )
    .expect("fixture wav loads");

    let mine = embedder.embed(&samples, 16_000).expect("redimnet embed");
    assert_eq!(mine.dim(), 192);

    let golden = read_redimnet_golden_embedding(&root.join("embeddings_b6").join("jfk.npy"));
    assert_eq!(mine.0.len(), golden.len());
    // Golden is the raw (pre-L2-normalize) reference embedding; cosine is
    // scale-invariant so comparing it against `mine`'s normalized vector is
    // still the right check (same convention as `backbone::tests`'
    // `full_pipeline_cosine_gate`).
    let cos = cosine(&mine.0, &golden);
    println!("redimnet e2e jfk cosine={cos:.8}");
    assert!(cos >= 0.9999, "redimnet e2e jfk cosine {cos}");
}
