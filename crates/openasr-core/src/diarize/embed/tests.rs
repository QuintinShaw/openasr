//! Parity tests for the ReDimNet2-B6 embedder.

use super::RedimNet2Embedder;
use super::{
    EmbedError, SpeakerEmbedder, SpeakerEmbeddingExecutionPlan,
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
    // `shared_embedder` intentionally keeps a process-wide parsed snapshot.
    // Mirror the daemon owner so the benchmark evicts that snapshot and its
    // worker TLS before native backend static destruction.
    let _runtime_owner = crate::NativeRuntimeShutdownGuard::new();
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

#[ignore = "host-local diagnostic: compares resident and legacy per-call setup"]
#[test]
fn embedder_resident_vs_uncached_bench_when_pack_present() {
    let Some(pack) = std::env::var_os("OPENASR_REDIMNET_PACK") else {
        return;
    };
    let embedder = RedimNet2Embedder::from_oasr(std::path::Path::new(&pack)).expect("load pack");
    let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        wav,
        "redimnet resident comparison",
        "redimnet resident comparison",
    )
    .expect("fixture wav loads");
    embedder.embed(&samples, 16_000).expect("resident warm-up");
    embedder
        .embed_uncached_for_bench(&samples, 16_000)
        .expect("uncached warm-up");
    let mut resident = Vec::new();
    let mut uncached = Vec::new();
    let run_resident = || {
        let started = std::time::Instant::now();
        embedder.embed(&samples, 16_000).expect("resident");
        started.elapsed().as_secs_f64()
    };
    let run_uncached = || {
        let started = std::time::Instant::now();
        embedder
            .embed_uncached_for_bench(&samples, 16_000)
            .expect("uncached");
        started.elapsed().as_secs_f64()
    };
    for iteration in 0..5 {
        if iteration % 2 == 0 {
            resident.push(run_resident());
            uncached.push(run_uncached());
        } else {
            uncached.push(run_uncached());
            resident.push(run_resident());
        }
    }
    resident.sort_by(f64::total_cmp);
    uncached.sort_by(f64::total_cmp);
    println!(
        "speaker_embedder resident_s p25={:.5} median={:.5} p75={:.5} uncached_s p25={:.5} median={:.5} p75={:.5}",
        resident[1], resident[2], resident[3], uncached[1], uncached[2], uncached[3]
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

#[test]
#[ignore = "requires local redimnet2-spike assets under tmp/ (not committed)"]
fn redimnet_batch_matches_single_order_and_runs_concurrently() {
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
    let single: Vec<SpeakerEmbedding> = clips
        .iter()
        .map(|clip| embedder.embed(clip, 16_000).expect("single"))
        .collect();

    super::reset_redim_max_active();
    let batch = embedder.embed_batch(&clips, 16_000);
    for (index, (actual, expected)) in batch.into_iter().zip(single).enumerate() {
        let actual = actual.expect("batch");
        let cos = cosine(&actual.0, &expected.0);
        assert!(
            cos >= 0.999_999,
            "batch item {index} changed embedding: cosine={cos}"
        );
    }
    if std::thread::available_parallelism().is_ok_and(|count| count.get() > 1) {
        assert!(
            super::redim_max_active() > 1,
            "ReDimNet batch did not overlap independent crops"
        );
    }
}

#[test]
#[ignore = "requires local redimnet2-spike assets under tmp/ (not committed)"]
fn redimnet_reuses_resident_runner_and_uploaded_weights() {
    let Some(root) = redimnet_spike_root() else {
        return;
    };
    let pack = root.join("redimnet2-b6-f32.oasr");
    if !pack.exists() {
        return;
    }
    let embedder = RedimNet2Embedder::from_oasr(&pack).expect("load pack");
    let samples = vec![0.01_f32; 16_000];
    let before = super::redimnet::backbone::resident_runtime_build_count();
    embedder.embed(&samples, 16_000).expect("first");
    let after_first = super::redimnet::backbone::resident_runtime_build_count();
    embedder.embed(&samples, 16_000).expect("second");
    let after_second = super::redimnet::backbone::resident_runtime_build_count();
    assert_eq!(after_first, before + 1);
    assert_eq!(
        after_second, after_first,
        "warm call rebuilt resident state"
    );
}

#[test]
#[ignore = "requires OPENASR_REDIMNET_PACK; validates process-owner shutdown and rebuild"]
fn redimnet_rebuilds_worker_runtime_after_process_owner_shutdown() {
    let _test_guard = super::redimnet_runtime_test_lock();
    let Some(pack) = std::env::var_os("OPENASR_REDIMNET_PACK") else {
        eprintln!("skipping: OPENASR_REDIMNET_PACK is not set");
        return;
    };
    let embedder =
        RedimNet2Embedder::from_oasr(std::path::Path::new(&pack)).expect("load ReDimNet pack");
    let samples = vec![0.01_f32; 16_000];

    crate::test_process_env::with_test_process_env(
        [(
            super::REDIMNET_BENCH_WORKERS_ENV,
            Some(std::ffi::OsString::from("1")),
        )],
        || {
            drop(crate::NativeRuntimeShutdownGuard::new());
            let builds_before = super::redimnet::backbone::resident_runtime_build_count();

            embedder.embed(&samples, 16_000).expect("first request");
            let builds_after_first = super::redimnet::backbone::resident_runtime_build_count();
            assert_eq!(builds_after_first, builds_before + 1);
            assert_eq!(super::redimnet_worker_runtime_entry_count(), 1);

            drop(crate::NativeRuntimeShutdownGuard::new());
            assert_eq!(
                super::redimnet_worker_runtime_entry_count(),
                0,
                "process-owner shutdown must eagerly clear persistent worker TLS"
            );

            embedder
                .embed(&samples, 16_000)
                .expect("request after shutdown rebuilds");
            assert_eq!(
                super::redimnet::backbone::resident_runtime_build_count(),
                builds_after_first + 1,
                "the first request after shutdown must rebuild resident state"
            );
            drop(crate::NativeRuntimeShutdownGuard::new());
        },
    );
}

#[test]
#[ignore = "requires OPENASR_REDIMNET_PACK; validates standalone-adapter shutdown"]
fn standalone_redimnet_drop_eagerly_releases_worker_runtime() {
    let _test_guard = super::redimnet_runtime_test_lock();
    let Some(pack) = std::env::var_os("OPENASR_REDIMNET_PACK") else {
        eprintln!("skipping: OPENASR_REDIMNET_PACK is not set");
        return;
    };
    let samples = vec![0.01_f32; 16_000];

    crate::test_process_env::with_test_process_env(
        [(
            super::REDIMNET_BENCH_WORKERS_ENV,
            Some(std::ffi::OsString::from("1")),
        )],
        || {
            super::unload_idle_redimnet_worker_runtimes();
            {
                let embedder = RedimNet2Embedder::from_oasr(std::path::Path::new(&pack))
                    .expect("load ReDimNet pack");
                embedder
                    .embed(&samples, 16_000)
                    .expect("standalone request");
                assert_eq!(super::redimnet_worker_runtime_entry_count(), 1);
            }
            assert_eq!(
                super::redimnet_worker_runtime_entry_count(),
                0,
                "dropping the final standalone embedder must clear persistent worker TLS"
            );
        },
    );
}

#[test]
#[ignore = "requires OPENASR_REDIMNET_PACK; validates terminal-backend eviction"]
fn redimnet_rebuilds_the_runner_after_device_loss_without_retrying_request() {
    let _test_guard = super::redimnet_runtime_test_lock();
    let Some(pack) = std::env::var_os("OPENASR_REDIMNET_PACK") else {
        eprintln!("skipping: OPENASR_REDIMNET_PACK is not set");
        return;
    };
    let embedder =
        RedimNet2Embedder::from_oasr(std::path::Path::new(&pack)).expect("load ReDimNet pack");
    let samples = vec![0.01_f32; 16_000];

    crate::test_process_env::with_test_process_env(
        [(
            super::REDIMNET_BENCH_WORKERS_ENV,
            Some(std::ffi::OsString::from("1")),
        )],
        || {
            super::clear_worker_graph_compute_status_override();
            super::unload_idle_redimnet_worker_runtimes();
            let builds_before = super::redimnet::backbone::resident_runtime_build_count();

            super::install_worker_graph_compute_device_lost();
            let clips: Vec<&[f32]> = vec![&samples, &samples, &samples];
            let results = embedder.embed_batch(&clips, 16_000);
            assert!(matches!(&results[0], Err(EmbedError::TerminalBackend(_))));
            assert!(results[1..].iter().all(|result| matches!(
                result,
                Err(EmbedError::BatchAbortedAfterTerminalBackend(_))
            )));
            assert_eq!(
                super::redimnet::backbone::resident_runtime_build_count(),
                builds_before + 1,
                "the failed batch must not rebuild and retry"
            );

            // The one-shot injection was broadcast because Rayon may choose
            // any worker. Clear unused test overrides before recovery; the
            // production batch path must already have evicted every runtime.
            super::clear_worker_graph_compute_status_override();
            assert_eq!(
                super::redimnet_worker_runtime_entry_count(),
                0,
                "terminal batch failure must evict all resident worker handles"
            );

            let recovered = embedder
                .embed(&samples, 16_000)
                .expect("the next request builds a fresh runner");
            assert_eq!(recovered.dim(), 192);
            assert_eq!(
                super::redimnet::backbone::resident_runtime_build_count(),
                builds_before + 2,
                "only the next request may rebuild after terminal failure"
            );
        },
    );
}

#[test]
#[ignore = "requires local redimnet2-spike assets under tmp/ (not committed)"]
fn redimnet_batch_inherits_job_cancellation() {
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
    let clips: Vec<&[f32]> = vec![&samples, &samples, &samples];
    let flag = Arc::new(AtomicBool::new(true));
    let previous = crate::ggml_runtime::arm_thread_job_cancel_flag(Some(Arc::clone(&flag)));
    let results = embedder.embed_batch(&clips, 16_000);
    assert!(
        results
            .into_iter()
            .all(|result| matches!(result, Err(EmbedError::Canceled)))
    );
    assert!(crate::ggml_runtime::disarm_thread_job_cancel_flag_if_current(&flag, previous));
}
