//! Parity tests for the ReDimNet2-B6 embedder.

use super::{
    EmbedError, REDIMNET_MAX_BATCH_WORKERS, RedimNet2Embedder, RedimNetResidentRuntime,
    SpeakerEmbedder, SpeakerEmbeddingExecutionPlan,
    abort_successful_results_after_terminal_failure, embed_batch_worker_range,
};
use crate::diarize::contract::SpeakerEmbedding;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
fn redimnet_execution_plan_caps_resident_workers_and_divides_cpu_threads() {
    assert_eq!(
        SpeakerEmbeddingExecutionPlan::for_clips(100, 8, 4),
        SpeakerEmbeddingExecutionPlan {
            workers: 4,
            threads_per_runner: 2,
        }
    );
    assert_eq!(
        SpeakerEmbeddingExecutionPlan::for_clips(2, 8, 4),
        SpeakerEmbeddingExecutionPlan {
            workers: 2,
            threads_per_runner: 4,
        }
    );
    assert_eq!(
        SpeakerEmbeddingExecutionPlan::for_clips(1, 8, 4),
        SpeakerEmbeddingExecutionPlan {
            workers: 1,
            threads_per_runner: 8,
        }
    );
    let plan = SpeakerEmbeddingExecutionPlan::for_clips(5, 8, 4);
    assert_eq!(
        (0..plan.workers)
            .map(|worker| plan.worker_range(worker, 5))
            .collect::<Vec<_>>(),
        vec![0..1, 1..2, 2..3, 3..5]
    );
}

#[ignore = "host-local bench: needs OPENASR_REDIMNET_PACK; run with --release for catalog numbers"]
#[test]
fn embedder_rtf_bench_when_pack_present() {
    let services = std::sync::Arc::new(
        crate::NativeExecutionServices::for_local_process().expect("execution services"),
    );
    let Some(runtime) =
        super::PolicyResolvedSpeakerRuntime::load(services).expect("load policy-owned embedder")
    else {
        eprintln!("skipping: redimnet2-b6 pack absent");
        return;
    };
    let embedder = runtime.embedder();
    let wav = std::env::var_os("OPENASR_AUX_BENCH_AUDIO")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
        });
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        wav,
        "redimnet rtf bench",
        "redimnet rtf bench",
    )
    .expect("fixture wav loads");
    let audio_seconds = samples.len() as f64 / 16_000.0;

    let mut last_embedding = embedder.embed(&samples, 16_000).expect("warm-up embed");
    let runs: Vec<f64> = (0..5)
        .map(|_| {
            let start = std::time::Instant::now();
            last_embedding = embedder.embed(&samples, 16_000).expect("timed embed");
            start.elapsed().as_secs_f64()
        })
        .collect();
    let embedding_sha256 = crate::testing::benchmark_sha256_f32(&last_embedding.0);
    let (median_seconds, runs) = crate::testing::benchmark_median_seconds(runs);
    let rtf_cpu = median_seconds / audio_seconds;
    println!(
        "AUX_MODEL_BENCH model=redimnet2 backend=cpu audio_seconds={audio_seconds:.6} median_seconds={median_seconds:.6} rtf={rtf_cpu:.6} embedding_sha256={embedding_sha256} runs={runs:?}"
    );

    let crop_len = samples.len() / 4;
    let clips: Vec<&[f32]> = (0..4)
        .map(|index| &samples[index * crop_len..(index + 1) * crop_len])
        .collect();
    let _ = embedder.embed_batch(&clips, 16_000);
    let mut sequential_runs = Vec::new();
    let mut batch_runs = Vec::new();
    let run_sequential = || {
        let started = std::time::Instant::now();
        for clip in &clips {
            embedder.embed(clip, 16_000).expect("sequential crop");
        }
        started.elapsed().as_secs_f64()
    };
    let run_batch = || {
        let started = std::time::Instant::now();
        let results = embedder.embed_batch(&clips, 16_000);
        assert!(results.into_iter().all(|result| result.is_ok()));
        started.elapsed().as_secs_f64()
    };
    for iteration in 0..5 {
        if iteration % 2 == 0 {
            sequential_runs.push(run_sequential());
            batch_runs.push(run_batch());
        } else {
            batch_runs.push(run_batch());
            sequential_runs.push(run_sequential());
        }
    }
    sequential_runs.sort_by(f64::total_cmp);
    batch_runs.sort_by(f64::total_cmp);
    let sequential = sequential_runs[sequential_runs.len() / 2];
    let batch = batch_runs[batch_runs.len() / 2];
    println!(
        "speaker_embedder crops=4 sequential_s p25={:.5} median={sequential:.5} p75={:.5} batch_s p25={:.5} median={batch:.5} p75={:.5} speedup={:.3}",
        sequential_runs[1],
        sequential_runs[3],
        batch_runs[1],
        batch_runs[3],
        sequential / batch
    );
}

#[ignore = "host-local Pareto bench: needs OPENASR_REDIMNET_PACK, OPENASR_AUX_BENCH_AUDIO, and OPENASR_REDIMNET_BENCH_WORKERS"]
#[test]
fn redimnet_batch_worker_pareto_benchmark() {
    let workers = std::env::var("OPENASR_REDIMNET_BENCH_WORKERS")
        .expect("OPENASR_REDIMNET_BENCH_WORKERS")
        .parse::<usize>()
        .expect("worker count is an integer");
    assert!((1..=REDIMNET_MAX_BATCH_WORKERS).contains(&workers));
    let services = std::sync::Arc::new(
        crate::NativeExecutionServices::for_local_process().expect("execution services"),
    );
    let runtime = super::PolicyResolvedSpeakerRuntime::load(services)
        .expect("load policy-owned embedder")
        .expect("redimnet2-b6 pack is present");
    let audio = crate::testing::external_test_fixture_path(
        "OPENASR_AUX_BENCH_AUDIO",
        "private auxiliary-model benchmark audio",
    )
    .expect("OPENASR_AUX_BENCH_AUDIO");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        audio,
        "redimnet batch Pareto benchmark",
        "redimnet batch Pareto benchmark",
    )
    .expect("load benchmark audio");
    let window_samples = 24_000usize;
    let step_samples = 12_000usize;
    let batch_clips = REDIMNET_MAX_BATCH_WORKERS * 4;
    assert!(
        samples.len() >= (batch_clips - 1) * step_samples + window_samples,
        "benchmark audio must cover one production embedding batch"
    );
    let clips = (0..batch_clips)
        .map(|index| {
            let start = index * step_samples;
            &samples[start..start + window_samples]
        })
        .collect::<Vec<_>>();
    let run = || {
        runtime
            .embedder()
            .embed_batch(&clips, 16_000)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("embed every crop")
    };

    let mut embeddings = run();
    let seconds = (0..3)
        .map(|_| {
            let started = std::time::Instant::now();
            embeddings = run();
            started.elapsed().as_secs_f64()
        })
        .collect::<Vec<_>>();
    let output_sha256 = crate::testing::benchmark_sha256_f32(
        &embeddings
            .iter()
            .flat_map(|embedding| embedding.0.iter().copied())
            .collect::<Vec<_>>(),
    );
    let (median_seconds, seconds) = crate::testing::benchmark_median_seconds(seconds);
    let represented_audio_seconds = (window_samples * clips.len()) as f64 / 16_000.0;
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let plan = SpeakerEmbeddingExecutionPlan::for_clips(clips.len(), available, workers);
    eprintln!(
        "REDIMNET_BATCH_PARETO workers={} threads_per_runner={} audio_seconds={represented_audio_seconds:.6} median_seconds={median_seconds:.6} rtf={:.6} output_sha256={output_sha256} runs={seconds:?}",
        plan.workers,
        plan.threads_per_runner,
        median_seconds / represented_audio_seconds,
    );
}

struct OrderedDefaultEmbedder;

impl SpeakerEmbedder for OrderedDefaultEmbedder {
    fn embed(&self, samples: &[f32], _sr: u32) -> Result<SpeakerEmbedding, EmbedError> {
        Ok(SpeakerEmbedding::l2_normalized(vec![samples[0], 1.0]))
    }

    fn embedding_dim(&self) -> usize {
        2
    }
}

#[test]
fn speaker_embedder_default_batch_preserves_input_order() {
    let clips: Vec<&[f32]> = vec![&[3.0], &[1.0], &[2.0]];
    let embeddings = OrderedDefaultEmbedder.embed_batch(&clips, 16_000);
    let first_components: Vec<f32> = embeddings
        .into_iter()
        .map(|result| result.expect("embedding").0[0])
        .collect();
    assert!(first_components[0] > first_components[2]);
    assert!(first_components[2] > first_components[1]);
}

#[test]
fn speaker_embedder_default_batch_stops_before_work_when_canceled() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    let flag = Arc::new(AtomicBool::new(true));
    let previous = crate::ggml_runtime::arm_thread_job_cancel_flag(Some(Arc::clone(&flag)));
    let clips: Vec<&[f32]> = vec![&[1.0], &[2.0], &[3.0]];
    let results = OrderedDefaultEmbedder.embed_batch(&clips, 16_000);
    assert!(
        results
            .into_iter()
            .all(|result| matches!(result, Err(EmbedError::Canceled)))
    );
    assert!(crate::ggml_runtime::disarm_thread_job_cancel_flag_if_current(&flag, previous));
}

#[test]
fn redimnet_worker_stops_range_after_terminal_backend_failure() {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let samples = [0.0_f32];
    let clips: Vec<&[f32]> = vec![&samples, &samples, &samples];
    let terminal = OnceLock::new();
    let calls = AtomicUsize::new(0);
    let results = embed_batch_worker_range(&clips, None, &terminal, |_| {
        let call = calls.fetch_add(1, Ordering::SeqCst);
        if call == 0 {
            Err(EmbedError::TerminalBackend("device lost".to_string()))
        } else {
            Ok(SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]))
        }
    });

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(matches!(
        &results[0],
        Err(EmbedError::TerminalBackend(reason)) if reason == "device lost"
    ));
    assert!(results[1..].iter().all(|result| matches!(
        result,
        Err(EmbedError::BatchAbortedAfterTerminalBackend(reason)) if reason == "device lost"
    )));
}

#[test]
fn redimnet_terminal_failure_invalidates_successes_from_peer_workers() {
    let mut results = vec![
        Ok(SpeakerEmbedding::l2_normalized(vec![1.0, 0.0])),
        Err(EmbedError::TerminalBackend("device lost".to_string())),
        Ok(SpeakerEmbedding::l2_normalized(vec![0.0, 1.0])),
    ];

    abort_successful_results_after_terminal_failure(&mut results, "device lost");

    assert!(matches!(
        &results[0],
        Err(EmbedError::BatchAbortedAfterTerminalBackend(reason)) if reason == "device lost"
    ));
    assert!(matches!(
        &results[1],
        Err(EmbedError::TerminalBackend(reason)) if reason == "device lost"
    ));
    assert!(matches!(
        &results[2],
        Err(EmbedError::BatchAbortedAfterTerminalBackend(reason)) if reason == "device lost"
    ));
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

fn low_level_redimnet_embed(
    embedder: &RedimNet2Embedder,
    runtime: &mut RedimNetResidentRuntime,
    samples: &[f32],
) -> Result<SpeakerEmbedding, EmbedError> {
    let (features, frames) = embedder.prepare_embedding_input(samples, 16_000)?;
    let raw = runtime
        .forward(&features, frames, Some(1))
        .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
    Ok(SpeakerEmbedding::l2_normalized(raw))
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
    assert_eq!(super::redimnet::config::EMBED_DIM, 192);
    assert_eq!(embedder.embedding_space_version(), "redimnet2-b6-cn-v1");

    let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        wav,
        "redimnet e2e test",
        "redimnet e2e test",
    )
    .expect("fixture wav loads");

    let mut runtime = RedimNetResidentRuntime::new(embedder.shared_weights(), Some(1))
        .expect("construct resident runtime");
    let mine = low_level_redimnet_embed(&embedder, &mut runtime, &samples).expect("redimnet embed");
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

#[test]
#[ignore = "host-local: needs the ReDimNet F32 pack, private audio, and official Python embedding"]
fn redimnet_matches_official_reference_on_aux_audio() {
    let pack = crate::testing::external_test_fixture_path(
        "OPENASR_REDIMNET_F32_PACK",
        "ReDimNet2-B6 F32 runtime pack",
    )
    .expect("OPENASR_REDIMNET_F32_PACK");
    let audio = crate::testing::external_test_fixture_path(
        "OPENASR_AUX_BENCH_AUDIO",
        "private auxiliary-model parity audio",
    )
    .expect("OPENASR_AUX_BENCH_AUDIO");
    let reference = crate::testing::external_test_fixture_path(
        "OPENASR_REDIMNET_REFERENCE_NPY",
        "official ReDimNet2 embedding",
    )
    .expect("OPENASR_REDIMNET_REFERENCE_NPY");

    let embedder = RedimNet2Embedder::from_oasr(&pack).expect("load ReDimNet F32 pack");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        &audio,
        "redimnet official parity",
        "redimnet official parity",
    )
    .expect("load parity audio");
    let mut runtime = RedimNetResidentRuntime::new(embedder.shared_weights(), Some(1))
        .expect("construct resident runtime");
    let actual = low_level_redimnet_embed(&embedder, &mut runtime, &samples)
        .expect("run ReDimNet parity embedding");
    let expected = read_redimnet_golden_embedding(&reference);
    assert_eq!(actual.dim(), expected.len(), "embedding dimension");
    let cos = cosine(&actual.0, &expected);
    eprintln!(
        "REDIMNET_OFFICIAL_PARITY cosine={cos:.8} dim={}",
        actual.dim()
    );
    assert!(cos >= 0.9999, "ReDimNet official cosine {cos}");
}

#[test]
#[ignore = "requires local redimnet2-spike assets under tmp/ (not committed)"]
fn redimnet_batch_matches_single_order() {
    let Some(root) = redimnet_spike_root() else {
        return;
    };
    let pack = root.join("redimnet2-b6-f32.oasr");
    if !pack.exists() {
        eprintln!("skip: {pack:?} not present");
        return;
    }
    let embedder = RedimNet2Embedder::from_oasr(&pack).expect("load pack");
    let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        wav,
        "redimnet batch parity",
        "redimnet batch parity",
    )
    .expect("fixture wav loads");
    let crop = samples.len() / 3;
    let clips: Vec<&[f32]> = (0..3)
        .map(|index| &samples[index * crop..(index + 1) * crop])
        .collect();
    let mut single_runtime = RedimNetResidentRuntime::new(embedder.shared_weights(), Some(1))
        .expect("construct single resident runtime");
    let single: Vec<SpeakerEmbedding> = clips
        .iter()
        .map(|clip| low_level_redimnet_embed(&embedder, &mut single_runtime, clip).expect("single"))
        .collect();

    let mut batch_runtime = RedimNetResidentRuntime::new(embedder.shared_weights(), Some(1))
        .expect("construct batch resident runtime");
    let batch = clips
        .iter()
        .map(|clip| low_level_redimnet_embed(&embedder, &mut batch_runtime, clip))
        .collect::<Vec<_>>();
    for (index, (actual, expected)) in batch.into_iter().zip(single).enumerate() {
        let actual = actual.expect("batch");
        let cos = cosine(&actual.0, &expected.0);
        assert!(
            cos >= 0.999_999,
            "batch item {index} changed embedding: cosine={cos}"
        );
    }
}

#[test]
#[ignore = "requires local redimnet2-spike assets under tmp/ (not committed)"]
fn redimnet_prepare_inherits_job_cancellation() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    let Some(root) = redimnet_spike_root() else {
        return;
    };
    let pack = root.join("redimnet2-b6-f32.oasr");
    if !pack.exists() {
        return;
    }
    let embedder = RedimNet2Embedder::from_oasr(&pack).expect("load pack");
    let samples = vec![0.01_f32; 16_000];
    let flag = Arc::new(AtomicBool::new(true));
    let previous = crate::ggml_runtime::arm_thread_job_cancel_flag(Some(Arc::clone(&flag)));
    let results = (0..3)
        .map(|_| embedder.prepare_embedding_input(&samples, 16_000))
        .collect::<Vec<_>>();
    assert!(
        results
            .into_iter()
            .all(|result| matches!(result, Err(EmbedError::Canceled)))
    );
    assert!(crate::ggml_runtime::disarm_thread_job_cancel_flag_if_current(&flag, previous));
}
