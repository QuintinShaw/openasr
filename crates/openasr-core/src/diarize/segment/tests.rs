use super::PyannetModel;

#[test]
fn segmenter_working_set_geometry_pins_provider_inference_concurrency() {
    let pyannote = super::segmenter_working_set_geometry(super::SegmenterProvider::Segmentation3_0);
    assert_eq!(
        pyannote.max_parallel_windows,
        super::pyannote_window_worker_count()
    );
    assert_eq!(
        pyannote.inference_peak_bytes_per_window,
        super::pyannet::quoted_forward_peak_bytes(10 * super::SAMPLE_RATE_HZ as usize)
    );
    assert!(
        pyannote.inference_peak_bytes_per_window
            > (pyannote.frames_per_window * super::NUM_CLASSES * std::mem::size_of::<f32>()) as u64
    );

    let diarizen = super::segmenter_working_set_geometry(super::SegmenterProvider::DiariZen);
    assert_eq!(diarizen.max_parallel_windows, 1);
    assert!(diarizen.inference_peak_bytes_per_window > 0);

    assert_eq!(pyannote.window_count(30 * 16_000), 21);
    assert_eq!(diarizen.window_count(30 * 16_000), 10);
    assert_eq!(
        pyannote.activity_frame_count(30 * 16_000),
        super::activity_frame_clock().frame_count_for_samples(30 * 16_000)
    );
    assert_eq!(pyannote.padded_tail_bytes(30 * 16_000), 0);
    assert!(pyannote.padded_tail_bytes(30 * 16_000 + 1) > 0);
}

#[test]
fn bounded_pyannote_window_pool_preserves_order_and_worker_cap() {
    let starts: Vec<usize> = (0..17).collect();
    let output = super::bounded_pyannote_window_map(&starts, &|| false, None, |start| {
        std::thread::sleep(std::time::Duration::from_micros(
            ((17 - start) % 4) as u64 * 50,
        ));
        Ok(start)
    })
    .unwrap();
    assert_eq!(output, starts);
    assert!(
        super::pyannote_window_pool()
            .as_ref()
            .unwrap()
            .current_num_threads()
            <= super::PYANNOTE_MAX_WINDOW_WORKERS
    );
}

#[test]
fn bounded_pyannote_window_pool_checks_cancellation_between_batches() {
    let starts: Vec<usize> = (0..17).collect();
    let checks = std::sync::atomic::AtomicUsize::new(0);
    let result = super::bounded_pyannote_window_map(
        &starts,
        &|| checks.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 0,
        None,
        Ok,
    );
    assert!(matches!(result, Err(super::SegmentError::Canceled)));
    assert_eq!(checks.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[test]
fn accelerated_pyannote_protocol_submits_b4_plus_ordered_tail() {
    let sample_count = 14 * super::SAMPLE_RATE_HZ as usize;
    let pcm = crate::PcmBuffer::from_vec(vec![0.0; sample_count]);
    let frames = super::pyannet::output_frame_count(10 * super::SAMPLE_RATE_HZ as usize);
    let mut submitted = Vec::new();
    let activity = super::segment_pyannote_local_activity_batched(
        pcm.full_slice(),
        super::SAMPLE_RATE_HZ,
        &|| false,
        None,
        4,
        |windows| {
            submitted.push(windows.len());
            Ok(vec![vec![0; frames]; windows.len()])
        },
    )
    .expect("batched sliding-window protocol");
    assert_eq!(submitted, [4, 1]);
    assert_eq!(activity.windows.len(), 5);
    assert!(
        activity
            .windows
            .iter()
            .map(|window| window.start_sample)
            .eq((0..5).map(|index| index * super::SAMPLE_RATE_HZ as usize))
    );
}

#[test]
#[ignore = "needs OPENASR_PYANNOTE_F32_PACK"]
fn parallel_pyannote_windows_match_serial_reference() {
    let pack = std::env::var_os("OPENASR_PYANNOTE_F32_PACK")
        .map(std::path::PathBuf::from)
        .expect("OPENASR_PYANNOTE_F32_PACK");
    let segmenter = super::PyannoteSegmenter::from_oasr(&pack).expect("load F32 segmenter");
    let sample_count = 12 * 16_000;
    let samples: Vec<f32> = (0..sample_count)
        .map(|index| {
            let t = index as f32 / 16_000.0;
            (t * 127.0).sin() * 0.21 + (t * 391.0).cos() * 0.09
        })
        .collect();
    let pcm_samples = samples.clone().into();

    let parallel = super::LocalActivitySegmenter::segment_local_activity(
        &segmenter,
        pcm_samples,
        16_000,
        &|| false,
        None,
    )
    .expect("parallel segmentation");

    let window_samples = (super::DEFAULT_WINDOW_S * 16_000.0) as usize;
    let step_samples = (super::DEFAULT_STEP_S * 16_000.0).round() as usize;
    let starts = super::sliding_window_starts(samples.len(), window_samples, step_samples);
    assert!(
        !starts.is_empty(),
        "auxiliary audio must contain at least one PyanNet window"
    );
    let frame_clock = super::activity_frame_clock();
    let mut windows = Vec::with_capacity(starts.len());
    for start in starts {
        let end = (start + window_samples).min(samples.len());
        let frame_activity = if end - start == window_samples {
            segmenter.infer_window(&samples[start..end])
        } else {
            let mut padded = vec![0.0_f32; window_samples];
            padded[..end - start].copy_from_slice(&samples[start..end]);
            segmenter.infer_window(&padded)
        }
        .expect("serial window inference");
        windows.push(super::LocalActivityWindow {
            start_sample: start,
            frame_activity,
        });
    }
    for window in &mut windows {
        window.frame_activity.truncate(
            frame_clock.frame_count_for_samples(samples.len().saturating_sub(window.start_sample)),
        );
    }
    let speaker_count = super::aggregate_speaker_count(&windows, frame_clock, samples.len());
    let serial = super::LocalActivity {
        frame_clock,
        windows,
        local_speaker_slots: super::MAX_LOCAL_SPEAKERS as u8,
        speaker_count,
    };

    assert_eq!(parallel, serial);
}

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
    let Some(pack) = std::env::var_os("OPENASR_PYANNOTE_PACK") else {
        eprintln!("skipping: pyannote pack absent");
        return;
    };
    let path = std::path::Path::new(&pack);
    let segmenter = if crate::diarize::pack::is_gguf(path) {
        super::PyannoteSegmenter::from_oasr(path).expect("load GGUF pyannote pack")
    } else {
        super::PyannoteSegmenter::from_safetensors(&std::fs::read(path).expect("read pack"))
            .expect("load safetensors pyannote pack")
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

/// Shared private-audio benchmark used by the auxiliary-model Pareto gate.
/// The recording path is always supplied explicitly; no customer audio path is
/// embedded in the repository. Unlike the catalog fixture benchmark above,
/// this exercises the production 10 s / 1 s sliding-window protocol.
#[test]
#[ignore = "host-local: needs OPENASR_PYANNOTE_PACK and OPENASR_AUX_BENCH_AUDIO"]
fn segmentation3_aux_audio_sliding_benchmark() {
    use super::LocalActivitySegmenter;

    let _pack = crate::testing::external_test_fixture_path(
        "OPENASR_PYANNOTE_PACK",
        "segmentation-3.0 runtime pack",
    )
    .expect("OPENASR_PYANNOTE_PACK");
    let audio = crate::testing::external_test_fixture_path(
        "OPENASR_AUX_BENCH_AUDIO",
        "private auxiliary-model benchmark audio",
    )
    .expect("OPENASR_AUX_BENCH_AUDIO");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        &audio,
        "segmentation-3.0 auxiliary benchmark",
        "segmentation-3.0 auxiliary benchmark",
    )
    .expect("load benchmark audio");
    let pcm = crate::PcmBuffer::from_vec(samples);
    let audio_seconds = pcm.len() as f64 / super::SAMPLE_RATE_HZ as f64;
    let backend = std::env::var("OPENASR_AUX_BENCH_BACKEND").unwrap_or_else(|_| "cpu".to_string());
    let execution_intent = segmentation3_benchmark_execution_intent(&backend);
    let services = std::sync::Arc::new(
        crate::NativeExecutionServices::for_local_process().expect("execution services"),
    );
    let segmenter =
        super::PolicyResolvedPyannoteSegmenterRuntime::load_with_intent(services, execution_intent)
            .expect("load policy-resolved segmenter")
            .expect("OPENASR_PYANNOTE_PACK must resolve");
    let run = || {
        segmenter
            .segment_local_activity(pcm.full_slice(), super::SAMPLE_RATE_HZ, &|| false, None)
            .expect("segment benchmark audio")
    };

    let warmup = run();
    let mut last = warmup;
    let seconds = (0..5)
        .map(|_| {
            let started = std::time::Instant::now();
            last = run();
            started.elapsed().as_secs_f64()
        })
        .collect::<Vec<_>>();
    let activity_sha256 = crate::testing::benchmark_sha256_bytes(
        last.windows
            .iter()
            .map(|window| window.frame_activity.as_slice())
            .chain(std::iter::once(last.speaker_count.as_slice())),
    );
    let (median_seconds, seconds) = crate::testing::benchmark_median_seconds(seconds);
    eprintln!(
        "AUX_MODEL_BENCH model=segmentation3 backend={backend} audio_seconds={audio_seconds:.6} median_seconds={median_seconds:.6} rtf={:.6} windows={} activity_sha256={activity_sha256} runs={seconds:?}",
        median_seconds / audio_seconds,
        last.windows.len(),
    );
}

#[test]
#[ignore = "host-local endurance gate: needs OPENASR_PYANNOTE_PACK and a >=15 minute OPENASR_AUX_BENCH_AUDIO"]
fn segmentation3_fifteen_minute_endurance() {
    use super::LocalActivitySegmenter;

    let _pack = crate::testing::external_test_fixture_path(
        "OPENASR_PYANNOTE_PACK",
        "segmentation-3.0 runtime pack",
    )
    .expect("OPENASR_PYANNOTE_PACK");
    let audio = crate::testing::external_test_fixture_path(
        "OPENASR_AUX_BENCH_AUDIO",
        "private auxiliary-model endurance audio",
    )
    .expect("OPENASR_AUX_BENCH_AUDIO");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        &audio,
        "segmentation-3.0 endurance gate",
        "segmentation-3.0 endurance gate",
    )
    .expect("load endurance audio");
    let audio_seconds = samples.len() as f64 / super::SAMPLE_RATE_HZ as f64;
    assert!(audio_seconds >= 15.0 * 60.0, "endurance audio is too short");
    let backend = std::env::var("OPENASR_AUX_BENCH_BACKEND").unwrap_or_else(|_| "cpu".to_string());
    let execution_intent = segmentation3_benchmark_execution_intent(&backend);
    let services = std::sync::Arc::new(
        crate::NativeExecutionServices::for_local_process().expect("execution services"),
    );
    let segmenter =
        super::PolicyResolvedPyannoteSegmenterRuntime::load_with_intent(services, execution_intent)
            .expect("load policy-resolved segmenter")
            .expect("OPENASR_PYANNOTE_PACK must resolve");
    let window_samples = 10 * super::SAMPLE_RATE_HZ as usize;
    let warmup = crate::PcmBuffer::from_vec(samples[..window_samples.min(samples.len())].to_vec());
    segmenter
        .segment_local_activity(warmup.full_slice(), super::SAMPLE_RATE_HZ, &|| false, None)
        .expect("warm segmentation runtime");

    let pcm = crate::PcmBuffer::from_vec(samples);
    let started = std::time::Instant::now();
    let activity = segmenter
        .segment_local_activity(pcm.full_slice(), super::SAMPLE_RATE_HZ, &|| false, None)
        .expect("segment endurance audio");
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let activity_sha256 = crate::testing::benchmark_sha256_bytes(
        activity
            .windows
            .iter()
            .map(|window| window.frame_activity.as_slice())
            .chain(std::iter::once(activity.speaker_count.as_slice())),
    );
    let memory = crate::metrics::process_memory_snapshot();
    let peak_rss_bytes = memory.peak_rss_bytes.unwrap_or(0);
    let current_rss_bytes = memory.current_rss_bytes.unwrap_or(0);
    let phys_footprint_bytes = memory.current_phys_footprint_bytes.unwrap_or(0);
    let peak_phys_footprint_bytes = memory.peak_phys_footprint_bytes.unwrap_or(0);
    eprintln!(
        "AUX_MODEL_ENDURANCE model=segmentation3 backend={backend} audio_seconds={audio_seconds:.6} elapsed_seconds={elapsed_seconds:.6} rtf={:.6} peak_rss_bytes={peak_rss_bytes} current_rss_bytes={current_rss_bytes} phys_footprint_bytes={phys_footprint_bytes} peak_phys_footprint_bytes={peak_phys_footprint_bytes} windows={} activity_sha256={activity_sha256}",
        elapsed_seconds / audio_seconds,
        activity.windows.len(),
    );
}

fn segmentation3_benchmark_execution_intent(
    backend: &str,
) -> crate::device::execution_policy::ExecutionIntent {
    match backend {
        "cpu" => crate::device::execution_policy::ExecutionIntent::CpuOnly,
        "metal" => crate::device::execution_policy::ExecutionIntent::ConstrainedAcceleratedOnly(
            crate::device::execution_policy::AcceleratedDeviceConstraint::Provider(
                crate::device::execution_route::ExecutionProvider::Metal,
            ),
        ),
        "cuda" => crate::device::execution_policy::ExecutionIntent::ConstrainedAcceleratedOnly(
            crate::device::execution_policy::AcceleratedDeviceConstraint::Provider(
                crate::device::execution_route::ExecutionProvider::Cuda,
            ),
        ),
        "vulkan" => crate::device::execution_policy::ExecutionIntent::ConstrainedAcceleratedOnly(
            crate::device::execution_policy::AcceleratedDeviceConstraint::Provider(
                crate::device::execution_route::ExecutionProvider::Vulkan,
            ),
        ),
        other => {
            panic!("OPENASR_AUX_BENCH_BACKEND must be cpu, metal, cuda, or vulkan; got '{other}'")
        }
    }
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

#[test]
#[ignore = "needs OPENASR_PYANNOTE_F32_PACK + OPENASR_PYANNOTE_INPUT + OPENASR_PYANNOTE_GOLDEN and Metal"]
fn pyannet_metal_matches_onnx_reference() {
    let pack = std::env::var("OPENASR_PYANNOTE_F32_PACK").expect("f32 pack");
    let input = std::env::var("OPENASR_PYANNOTE_INPUT").expect("input");
    let golden = std::env::var("OPENASR_PYANNOTE_GOLDEN").expect("golden");
    let (in_dims, samples) = read_golden(&input, b"PYIN");
    assert_eq!(in_dims.len(), 3);
    let (y_dims, reference) = read_golden(&golden, b"PYYY");
    let model = PyannetModel::from_oasr(std::path::Path::new(&pack)).expect("PyanNet pack");
    let mut runtime = super::pyannet_ggml::PyannetGgmlRuntime::new(
        model,
        crate::ggml_runtime::GgmlCpuGraphBackend::Metal,
        crate::device::execution_policy::ExecutionPlacement::Hybrid,
    )
    .expect("Metal runtime");
    let (actual, frames) = runtime.forward(&samples).expect("Metal forward");
    assert_eq!(frames, y_dims[1], "frame count");
    assert_eq!(actual.len(), reference.len());
    let max_abs = actual
        .iter()
        .zip(&reference)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let class_mismatches = actual
        .chunks_exact(super::NUM_CLASSES)
        .zip(reference.chunks_exact(super::NUM_CLASSES))
        .filter(|(actual, expected)| row_argmax(actual) != row_argmax(expected))
        .count();
    eprintln!(
        "PYANNET_METAL_OFFICIAL frames={frames} max_abs={max_abs:.8} class_mismatches={class_mismatches}"
    );
    assert!(max_abs < 1e-2, "Metal max abs error {max_abs} too high");
    assert_eq!(
        class_mismatches, 0,
        "Metal must preserve the official powerset class"
    );
}

#[test]
#[ignore = "needs OPENASR_PYANNOTE_F32_PACK + OPENASR_PYANNOTE_INPUT + OPENASR_PYANNOTE_GOLDEN and OPENASR_PYANNOTE_BENCH_BACKEND=cuda|vulkan"]
fn pyannet_exact_gpu_matches_cpu_and_onnx_reference() {
    use crate::device::execution_route::ExecutionProvider;

    let requested = std::env::var("OPENASR_PYANNOTE_BENCH_BACKEND")
        .expect("OPENASR_PYANNOTE_BENCH_BACKEND must be cuda or vulkan")
        .trim()
        .to_ascii_lowercase();
    let provider = match requested.as_str() {
        "cuda" => ExecutionProvider::Cuda,
        "vulkan" => ExecutionProvider::Vulkan,
        other => panic!("OPENASR_PYANNOTE_BENCH_BACKEND must be cuda or vulkan; got '{other}'"),
    };
    let route = crate::device::execution_route::enumerate_compute_devices_from_ggml(
        &crate::ggml_runtime::ggml_available_devices(),
    )
    .into_iter()
    .find(|device| device.provider == provider)
    .unwrap_or_else(|| panic!("requested PyanNet provider '{requested}' is unavailable"))
    .to_resolved_route();
    let route_identity = route.isolation_key();
    let _route_guard = crate::ggml_runtime::install_request_backend_override(Some(
        crate::ggml_runtime::RequestBackendPreference::Exact(route),
    ));

    let pack = std::env::var("OPENASR_PYANNOTE_F32_PACK").expect("f32 pack");
    let input = std::env::var("OPENASR_PYANNOTE_INPUT").expect("input");
    let golden = std::env::var("OPENASR_PYANNOTE_GOLDEN").expect("golden");
    let (in_dims, samples) = read_golden(&input, b"PYIN");
    assert_eq!(in_dims.len(), 3);
    let (y_dims, reference) = read_golden(&golden, b"PYYY");
    let pack = std::path::Path::new(&pack);
    let cpu_model = PyannetModel::from_oasr(pack).expect("CPU PyanNet pack");
    let (features, feature_frames) = cpu_model
        .frontend_features(&samples)
        .expect("CPU PyanNet frontend");
    let (cpu, cpu_frames) = cpu_model.forward(&samples).expect("CPU PyanNet forward");
    assert_eq!(feature_frames, cpu_frames, "frontend/CPU frame count");
    let mut runtime = super::pyannet_ggml::PyannetGgmlRuntime::new(
        PyannetModel::from_oasr(pack).expect("exact GPU PyanNet pack"),
        crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
        crate::device::execution_policy::ExecutionPlacement::Hybrid,
    )
    .expect("exact GPU runtime");
    let feature_batch = [
        features.as_slice(),
        features.as_slice(),
        features.as_slice(),
        features.as_slice(),
    ];
    let mut actual_batch = runtime
        .forward_features_batch(&feature_batch, feature_frames)
        .expect("exact GPU batched recurrent/classifier forward");
    assert_eq!(actual_batch.len(), feature_batch.len());
    let actual = actual_batch.remove(0);
    for duplicate in actual_batch {
        let duplicate_max_abs = duplicate
            .iter()
            .zip(&actual)
            .map(|(duplicate, actual)| (duplicate - actual).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            duplicate_max_abs < 1e-6,
            "GPU batch lanes diverged by {duplicate_max_abs}"
        );
    }
    let frames = feature_frames;
    assert_eq!(frames, cpu_frames, "CPU/GPU frame count");
    assert_eq!(frames, y_dims[1], "GPU/oracle frame count");
    assert_eq!(actual.len(), reference.len());
    let (max_abs, mean_abs) = actual.iter().zip(&cpu).fold(
        (0.0_f32, 0.0_f64),
        |(maximum, total), (actual, expected)| {
            let error = (actual - expected).abs();
            (maximum.max(error), total + f64::from(error))
        },
    );
    let mean_abs = mean_abs / actual.len().max(1) as f64;
    let oracle_max_abs = actual
        .iter()
        .zip(&reference)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let cpu_class_mismatches = actual
        .chunks_exact(super::NUM_CLASSES)
        .zip(cpu.chunks_exact(super::NUM_CLASSES))
        .filter(|(actual, expected)| row_argmax(actual) != row_argmax(expected))
        .count();
    let oracle_class_mismatches = actual
        .chunks_exact(super::NUM_CLASSES)
        .zip(reference.chunks_exact(super::NUM_CLASSES))
        .filter(|(actual, expected)| row_argmax(actual) != row_argmax(expected))
        .count();
    eprintln!(
        "PYANNET_EXACT_GPU_OFFICIAL provider={requested} placement=hybrid exact_route={route_identity} frames={frames} cpu_max_abs={max_abs:.9} cpu_mean_abs={mean_abs:.9} oracle_max_abs={oracle_max_abs:.9} cpu_class_mismatches={cpu_class_mismatches} oracle_class_mismatches={oracle_class_mismatches}"
    );
    assert!(max_abs < 7e-3, "CPU/{requested} max abs {max_abs} too high");
    assert!(
        mean_abs < 1.2e-3,
        "CPU/{requested} mean abs {mean_abs} too high"
    );
    assert!(
        oracle_max_abs < 1e-2,
        "{requested}/oracle max abs {oracle_max_abs} too high"
    );
    assert_eq!(cpu_class_mismatches, 0, "GPU must preserve CPU classes");
    assert_eq!(
        oracle_class_mismatches, 0,
        "GPU must preserve official powerset classes"
    );
}

#[test]
#[ignore = "host-local diagnostic: needs OPENASR_PYANNOTE_F32_PACK + OPENASR_AUX_BENCH_AUDIO and Metal"]
fn pyannet_cpu_and_metal_stay_within_measured_near_tie_envelope_on_aux_audio() {
    let pack = std::env::var("OPENASR_PYANNOTE_F32_PACK").expect("f32 pack");
    let audio = std::env::var("OPENASR_AUX_BENCH_AUDIO").expect("aux audio");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        &audio,
        "PyanNet CPU/Metal diagnostic",
        "PyanNet CPU/Metal diagnostic",
    )
    .expect("load auxiliary audio");
    let pack = std::path::Path::new(&pack);
    let reference = PyannetModel::from_oasr(pack).expect("reference PyanNet pack");
    let mut ggml_cpu = super::pyannet_ggml::PyannetGgmlRuntime::new(
        PyannetModel::from_oasr(pack).expect("ggml CPU PyanNet pack"),
        crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
        crate::device::execution_policy::ExecutionPlacement::CpuOnly,
    )
    .expect("ggml CPU runtime");
    let mut metal = super::pyannet_ggml::PyannetGgmlRuntime::new(
        PyannetModel::from_oasr(pack).expect("Metal PyanNet pack"),
        crate::ggml_runtime::GgmlCpuGraphBackend::Metal,
        crate::device::execution_policy::ExecutionPlacement::Hybrid,
    )
    .expect("Metal runtime");
    let window_samples = (super::DEFAULT_WINDOW_S * super::SAMPLE_RATE_HZ as f64) as usize;
    let step_samples = (super::DEFAULT_STEP_S * super::SAMPLE_RATE_HZ as f64).round() as usize;
    let starts = super::sliding_window_starts(samples.len(), window_samples, step_samples);
    assert!(
        !starts.is_empty(),
        "auxiliary audio must contain at least one PyanNet window"
    );
    let mut ggml_cpu_errors = Vec::with_capacity(starts.len() * 589 * super::NUM_CLASSES);
    let mut metal_errors = Vec::with_capacity(starts.len() * 589 * super::NUM_CLASSES);
    let mut backend_errors = Vec::with_capacity(starts.len() * 589 * super::NUM_CLASSES);
    let mut ggml_cpu_class_mismatches = 0usize;
    let mut class_mismatches = Vec::new();
    let mut activity_mismatches = 0usize;
    let mut max_metal_error = (0.0f32, 0usize, 0usize, 0usize, 0.0f32, 0.0f32);

    for (window_index, start) in starts.iter().copied().enumerate() {
        let end = (start + window_samples).min(samples.len());
        let mut padded = Vec::new();
        let window = if end - start == window_samples {
            &samples[start..end]
        } else {
            padded.resize(window_samples, 0.0);
            padded[..end - start].copy_from_slice(&samples[start..end]);
            padded.as_slice()
        };
        let (cpu_logp, cpu_frames) = reference.forward(window).expect("reference forward");
        let (ggml_cpu_logp, ggml_cpu_frames) = ggml_cpu.forward(window).expect("ggml CPU forward");
        let (metal_logp, metal_frames) = metal.forward(window).expect("Metal forward");
        assert_eq!(
            cpu_frames, ggml_cpu_frames,
            "window {window_index} ggml CPU frame count"
        );
        assert_eq!(
            cpu_frames, metal_frames,
            "window {window_index} frame count"
        );
        for (frame_index, ((cpu_row, ggml_cpu_row), metal_row)) in cpu_logp
            .chunks_exact(super::NUM_CLASSES)
            .zip(ggml_cpu_logp.chunks_exact(super::NUM_CLASSES))
            .zip(metal_logp.chunks_exact(super::NUM_CLASSES))
            .enumerate()
        {
            for (class, ((cpu, ggml_cpu), metal)) in
                cpu_row.iter().zip(ggml_cpu_row).zip(metal_row).enumerate()
            {
                let ggml_cpu_error = (cpu - ggml_cpu).abs();
                let metal_error = (cpu - metal).abs();
                ggml_cpu_errors.push(ggml_cpu_error);
                metal_errors.push(metal_error);
                backend_errors.push((ggml_cpu - metal).abs());
                if metal_error > max_metal_error.0 {
                    max_metal_error = (metal_error, window_index, frame_index, class, *cpu, *metal);
                }
            }
            let cpu_class = row_argmax(cpu_row);
            let ggml_cpu_class = row_argmax(ggml_cpu_row);
            let metal_class = row_argmax(metal_row);
            ggml_cpu_class_mismatches += usize::from(cpu_class != ggml_cpu_class);
            if cpu_class != metal_class {
                activity_mismatches +=
                    usize::from(super::POWERSET[cpu_class] != super::POWERSET[metal_class]);
                class_mismatches.push((
                    window_index,
                    frame_index,
                    cpu_class,
                    metal_class,
                    row_top_margin(cpu_row),
                    row_top_margin(metal_row),
                    cpu_row
                        .iter()
                        .zip(metal_row)
                        .map(|(cpu, metal)| (cpu - metal).abs())
                        .fold(0.0f32, f32::max),
                ));
            }
        }
    }
    let (ggml_cpu_max_abs, ggml_cpu_p99_abs) = sorted_max_and_p99(&mut ggml_cpu_errors);
    let (metal_max_abs, metal_p99_abs) = sorted_max_and_p99(&mut metal_errors);
    let (backend_max_abs, backend_p99_abs) = sorted_max_and_p99(&mut backend_errors);
    eprintln!(
        "PYANNET_CPU_METAL_DIAGNOSTIC windows={} values={} ggml_cpu_max_abs={ggml_cpu_max_abs:.8} ggml_cpu_p99_abs={ggml_cpu_p99_abs:.8} ggml_cpu_class_mismatches={ggml_cpu_class_mismatches} metal_max_abs={metal_max_abs:.8} metal_p99_abs={metal_p99_abs:.8} backend_max_abs={backend_max_abs:.8} backend_p99_abs={backend_p99_abs:.8} class_mismatches={} activity_mismatches={} max_metal_error={max_metal_error:?} first_mismatches={:?}",
        starts.len(),
        metal_errors.len(),
        class_mismatches.len(),
        activity_mismatches,
        class_mismatches.iter().take(12).collect::<Vec<_>>()
    );
    // The accelerated runtime now includes SincNet as well as the recurrent
    // stack. Its direct ggml convolutions use a different, deterministic f32
    // reduction order from the scalar family oracle; the full-graph regression
    // pins that bounded numeric envelope while the class assertion below keeps
    // the semantic contract exact.
    assert!(
        ggml_cpu_max_abs < 5e-4,
        "ggml CPU full-graph max abs {ggml_cpu_max_abs} drifted from family math"
    );
    assert_eq!(
        ggml_cpu_class_mismatches, 0,
        "ggml CPU graph must preserve every family-math powerset class"
    );
    assert!(
        metal_max_abs < 4e-2,
        "Metal max abs {metal_max_abs} regressed"
    );
    assert!(
        metal_p99_abs < 5e-3,
        "Metal p99 abs {metal_p99_abs} regressed"
    );
    let compared_frames = metal_errors.len() / super::NUM_CLASSES;
    assert!(
        class_mismatches.len() as f64 / compared_frames.max(1) as f64 <= 1e-4,
        "Metal powerset mismatch rate exceeded the measured near-tie envelope"
    );
    assert!(
        activity_mismatches as f64 / compared_frames.max(1) as f64 <= 1e-4,
        "Metal activity mismatch rate exceeded the measured near-tie envelope"
    );
    assert!(
        class_mismatches
            .iter()
            .all(|mismatch| mismatch.4 <= mismatch.6),
        "Metal changed a powerset class whose reference margin exceeded its row error"
    );
}

fn sorted_max_and_p99(values: &mut [f32]) -> (f32, f32) {
    values.sort_by(f32::total_cmp);
    let max = values.last().copied().unwrap_or(0.0);
    let p99_index = values.len().saturating_sub(1).saturating_mul(99) / 100;
    (max, values.get(p99_index).copied().unwrap_or(0.0))
}

fn row_argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn row_top_margin(row: &[f32]) -> f32 {
    let mut top = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for value in row.iter().copied() {
        if value > top {
            second = top;
            top = value;
        } else if value > second {
            second = value;
        }
    }
    top - second
}
