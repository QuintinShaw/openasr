use super::*;

use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

fn external_path(env: &str) -> PathBuf {
    std::env::var_os(env)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{env} must point at the external DiariZen parity fixture"))
}

struct BenchmarkBackend {
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    requested_provider: &'static str,
    accelerated_fp16_tolerances: bool,
    expected_backend_name: Option<String>,
    exact_route_identity: Option<String>,
    _route_guard: Option<crate::ggml_runtime::RequestBackendOverrideGuard>,
}

impl BenchmarkBackend {
    fn assert_runtime(&self, runtime: &DiariZenRuntime) {
        let backend = runtime.backend_label();
        if let Some(expected_backend_name) = &self.expected_backend_name {
            assert_eq!(
                backend,
                format!("Gpu:{expected_backend_name}"),
                "the exact DiariZen route must initialize the requested ggml device"
            );
        } else {
            assert!(
                backend
                    .to_ascii_lowercase()
                    .contains(self.requested_provider),
                "requested DiariZen provider '{}' but constructed backend '{backend}'",
                self.requested_provider
            );
        }
        eprintln!(
            "DIARIZEN_BACKEND_RECEIPT requested_provider={} placement=full_device exact_route={} backend={backend}",
            self.requested_provider,
            self.exact_route_identity.as_deref().unwrap_or("coarse")
        );
    }
}

fn assert_selected_backend_compute(
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    observed: &crate::GgmlExecutionPlacementSummary,
    operation: &str,
) {
    if backend != crate::ggml_runtime::GgmlCpuGraphBackend::Metal {
        return;
    }
    assert!(
        !observed.observed_compute_nodes_by_backend.is_empty()
            && observed
                .observed_compute_nodes_by_backend
                .keys()
                .all(|backend| {
                    let backend = backend.to_ascii_lowercase();
                    backend.starts_with("mtl") || backend.contains("metal")
                }),
        "{operation} explicit Metal route observed non-Metal compute: {:?}",
        observed.observed_compute_nodes_by_backend,
    );
}

fn benchmark_backend() -> BenchmarkBackend {
    let requested = std::env::var("OPENASR_DIARIZEN_BENCH_BACKEND")
        .unwrap_or_else(|_| "cpu".to_string())
        .trim()
        .to_ascii_lowercase();
    let (backend, provider) = match requested.as_str() {
        "cpu" => (
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
            crate::device::execution_route::ExecutionProvider::Cpu,
        ),
        "metal" => (
            crate::ggml_runtime::GgmlCpuGraphBackend::Metal,
            crate::device::execution_route::ExecutionProvider::Metal,
        ),
        "cuda" => (
            crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            crate::device::execution_route::ExecutionProvider::Cuda,
        ),
        "vulkan" => (
            crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            crate::device::execution_route::ExecutionProvider::Vulkan,
        ),
        backend => panic!(
            "OPENASR_DIARIZEN_BENCH_BACKEND must be cpu, metal, cuda, or vulkan; got '{backend}'"
        ),
    };
    let mut expected_backend_name = None;
    let mut exact_route_identity = None;
    let route_guard = if matches!(
        provider,
        crate::device::execution_route::ExecutionProvider::Cuda
            | crate::device::execution_route::ExecutionProvider::Vulkan
    ) {
        let route = crate::device::execution_route::enumerate_compute_devices_from_ggml(
            &crate::ggml_runtime::ggml_available_devices(),
        )
        .into_iter()
        .find(|device| device.provider == provider)
        .unwrap_or_else(|| panic!("requested DiariZen provider '{requested}' is unavailable"))
        .to_resolved_route();
        expected_backend_name = Some(route.stable_id.clone());
        exact_route_identity = Some(route.isolation_key());
        Some(crate::ggml_runtime::install_request_backend_override(Some(
            crate::ggml_runtime::RequestBackendPreference::Exact(route),
        )))
    } else {
        None
    };
    BenchmarkBackend {
        backend,
        requested_provider: match provider {
            crate::device::execution_route::ExecutionProvider::Cpu => "cpu",
            crate::device::execution_route::ExecutionProvider::Metal => "metal",
            crate::device::execution_route::ExecutionProvider::Cuda => "cuda",
            crate::device::execution_route::ExecutionProvider::Vulkan => "vulkan",
            _ => unreachable!("DiariZen benchmark parser admits only concrete providers"),
        },
        accelerated_fp16_tolerances: matches!(
            provider,
            crate::device::execution_route::ExecutionProvider::Cuda
                | crate::device::execution_route::ExecutionProvider::Vulkan
        ),
        expected_backend_name,
        exact_route_identity,
        _route_guard: route_guard,
    }
}

fn golden_limits(name: &str, accelerated_fp16: bool) -> (f32, f32) {
    let (max_limit, mean_limit) = match name {
        "wavlm_layer_23" => (3.0, 0.14),
        name if name.starts_with("wavlm_layer_") => (0.55, 0.035),
        "weighted_layer_sum_raw" => (1.0, 0.05),
        "projection_raw" => (1.5, 0.18),
        "projection_norm" | "conformer_layer_00" | "conformer_layer_01" | "conformer_layer_02"
        | "conformer_layer_03" => (0.08, 0.008),
        "logits" => (0.04, 0.015),
        _ => (0.05, 0.005),
    };
    if !accelerated_fp16 {
        return (max_limit, mean_limit);
    }
    // On the frozen FP16 pack, exact CUDA and Vulkan kernels have a wider
    // audited accumulation envelope through 24 WavLM layers than the CPU
    // reference path. Keep the CPU/PyTorch envelope above unchanged and use
    // fixed, evidence-backed multipliers for those two routes. Every stage's
    // observed max/mean remains in the host-local evidence log, while the
    // downstream powerset/activity/segment gates remain exact.
    let multiplier = match name {
        "wavlm_layer_23" | "weighted_layer_sum_raw" | "projection_raw" => 2.5,
        _ => 2.0,
    };
    (max_limit * multiplier, mean_limit * multiplier)
}

fn npy_bytes(npz: &std::path::Path, name: &str) -> Vec<u8> {
    let file = std::fs::File::open(npz).expect("open external npz fixture");
    let mut archive = zip::ZipArchive::new(file).expect("parse external npz fixture");
    let mut entry = archive
        .by_name(&format!("{name}.npy"))
        .unwrap_or_else(|_| panic!("npz array '{name}'"));
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes).expect("read npy entry");
    bytes
}

fn npy_payload(bytes: &[u8]) -> (&str, &[u8]) {
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        _ => panic!("unsupported npy version {major}"),
    };
    let header_end = header_start + header_len;
    (
        std::str::from_utf8(&bytes[header_start..header_end]).expect("npy header utf8"),
        &bytes[header_end..],
    )
}

fn npy_f32(npz: &std::path::Path, name: &str) -> Vec<f32> {
    let bytes = npy_bytes(npz, name);
    let (header, payload) = npy_payload(&bytes);
    assert!(header.contains("'<f4'"), "{name}: {header}");
    let values = payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect::<Vec<_>>();
    if !header.contains("'fortran_order': True") {
        return values;
    }
    let shape_start = header.find("'shape': (").expect("npy shape") + "'shape': (".len();
    let shape_end = header[shape_start..].find(')').expect("npy shape end") + shape_start;
    let shape = header[shape_start..shape_end]
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect::<Vec<_>>();
    assert_eq!(shape.iter().product::<usize>(), values.len());
    let mut c_order = vec![0.0_f32; values.len()];
    for (c_flat, output) in c_order.iter_mut().enumerate() {
        let mut remainder = c_flat;
        let mut fortran_flat = 0;
        for axis in (0..shape.len()).rev() {
            let index = remainder % shape[axis];
            remainder /= shape[axis];
            let stride = shape[..axis].iter().product::<usize>();
            fortran_flat += index * stride;
        }
        *output = values[fortran_flat];
    }
    c_order
}

fn npy_i64(npz: &std::path::Path, name: &str) -> Vec<i64> {
    let bytes = npy_bytes(npz, name);
    let (header, payload) = npy_payload(&bytes);
    assert!(header.contains("'<i8'"), "{name}: {header}");
    payload
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("i64 chunk")))
        .collect()
}

fn diff(actual: &[f32], expected: &[f32]) -> (f32, f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let difference = (actual - expected).abs();
        max_abs = max_abs.max(difference);
        sum_abs += difference as f64;
    }
    (max_abs, (sum_abs / actual.len().max(1) as f64) as f32)
}

fn synthetic_exact_window() -> Vec<f32> {
    (0..super::config::WINDOW_SAMPLES)
        .map(|index| {
            let time = index as f64 / super::config::SAMPLE_RATE_HZ as f64;
            let duration =
                super::config::WINDOW_SAMPLES as f64 / super::config::SAMPLE_RATE_HZ as f64;
            let envelope = (time / 0.08).min(1.0) * ((duration - time) / 0.08).min(1.0);
            ((0.31 * (2.0 * std::f64::consts::PI * 173.0 * time).sin()
                + 0.17 * (2.0 * std::f64::consts::PI * 421.0 * time + 0.23).sin()
                + 0.09 * (2.0 * std::f64::consts::PI * 37.0 * time).cos()
                + 0.03 * (2.0 * std::f64::consts::PI * (83.0 + 7.0 * time) * time).sin())
                * envelope) as f32
        })
        .collect()
}

#[test]
fn median_filter_uses_scipy_reflect_edges() {
    let mut activity = vec![0_u8; 7 * LOCAL_SPEAKERS];
    for frame in [0, 1, 2] {
        activity[frame * LOCAL_SPEAKERS] = 1;
    }
    let filtered = median_filter_activity(&activity, 7);
    assert_eq!(
        filtered
            .chunks_exact(LOCAL_SPEAKERS)
            .map(|row| row[0])
            .collect::<Vec<_>>(),
        vec![1, 1, 1, 0, 0, 0, 0]
    );
}

#[test]
fn feature_tiles_use_the_minimal_exact_receptive_field() {
    for expected_frames in [1, 2, 31, 64, 128, 799] {
        let samples = runtime::feature_input_samples(expected_frames)
            .expect("valid output frame count has an inverse receptive field");
        assert_eq!(config::output_frames(samples), expected_frames);
        if samples > 0 {
            assert_eq!(
                config::output_frames(samples - 1),
                expected_frames - 1,
                "one fewer source sample must not manufacture the last feature frame"
            );
        }
    }
}

#[test]
fn postprocess_preserves_frame_geometry() {
    let output = postprocess_logits(&[0.0; 2 * POWERSET_CLASSES], 2);
    assert_eq!(output.0.len(), 2);
    assert_eq!(output.1.len(), 2 * LOCAL_SPEAKERS);
}

#[test]
fn local_activity_owns_complete_windows_without_copying_the_source() {
    let samples: crate::PcmSlice =
        vec![0.0_f32; super::config::WINDOW_SAMPLES + super::config::WINDOW_STEP_SAMPLES / 2]
            .into();
    let source_identity = samples.backing_identity();
    let mut observed = Vec::new();
    let frames = super::config::output_frames(super::config::WINDOW_SAMPLES);
    let local = super::super::segment_diarizen_local_activity(
        samples,
        super::config::SAMPLE_RATE_HZ,
        &|| false,
        None,
        |window| {
            observed.push((window.backing_identity(), window.len()));
            Ok(DiariZenWindowOutput {
                frame_count: frames,
                logits: vec![0.0; frames * POWERSET_CLASSES],
                powerset_class: vec![0; frames],
                activity: vec![0; frames * LOCAL_SPEAKERS],
            })
        },
    )
    .expect("owned DiariZen windows");

    assert_eq!(local.windows.len(), 2);
    assert_eq!(observed.len(), 2);
    assert_eq!(
        observed[0],
        (source_identity, super::config::WINDOW_SAMPLES)
    );
    assert_ne!(
        observed[1].0, source_identity,
        "only the padded tail should allocate a new backing"
    );
    assert_eq!(observed[1].1, super::config::WINDOW_SAMPLES);
}

#[test]
#[ignore = "requires OPENASR_DIARIZEN_LEGACY_PACK"]
fn legacy_long_name_pack_is_rejected_by_the_compact_schema_runtime() {
    let pack = external_path("OPENASR_DIARIZEN_LEGACY_PACK");
    assert!(
        DiariZenSegmenter::probe_oasr(&pack).is_err(),
        "a pre-compact-v2 pack must never enter the native runtime"
    );
}

/// One-second geometry is intentionally test-only: it exercises the exact
/// production graph against a compact PyTorch stage dump, while the public
/// constructor remains pinned to the checkpoint's 16-second window.
#[test]
#[ignore = "requires OPENASR_DIARIZEN_PACK and OPENASR_DIARIZEN_GOLDEN"]
fn native_graph_matches_external_pytorch_golden() {
    let pack = external_path("OPENASR_DIARIZEN_PACK");
    let golden = external_path("OPENASR_DIARIZEN_GOLDEN");
    let benchmark_backend = benchmark_backend();
    DiariZenSegmenter::probe_oasr(&pack).expect("strict pack probe");
    let waveform = npy_f32(&golden, "waveform");
    let mut runtime = DiariZenRuntime::new(&pack, 16_000, true, Some(benchmark_backend.backend))
        .expect("construct test-geometry runtime");
    benchmark_backend.assert_runtime(&runtime);
    let execution_placement = crate::GgmlExecutionTelemetryCollector::new();
    let _execution_placement_guard = execution_placement.install();
    let trace = runtime.infer_trace(&waveform).expect("native trace");
    let observed = execution_placement.snapshot();
    eprintln!(
        "DIARIZEN_OFFICIAL_PLACEMENT provider={} backend={} observed_compute_nodes={:?} observed_nodes={:?}",
        benchmark_backend.requested_provider,
        runtime.backend_label(),
        observed.observed_compute_nodes_by_backend,
        observed.observed_nodes_by_backend,
    );
    assert_selected_backend_compute(
        benchmark_backend.backend,
        &observed,
        "DiariZen official parity",
    );

    let mut numeric_violations = Vec::new();
    for (name, actual) in &trace {
        if name == "weighted_layer_sum_raw" {
            // The golden carries a trailing singleton dimension; flat order is
            // identical to the native [hidden, frame] tensor.
        }
        let expected = npy_f32(&golden, name);
        let (max_abs, mean_abs) = diff(actual, &expected);
        eprintln!("{name}: max_abs={max_abs}, mean_abs={mean_abs}");
        // Dense FP16 rounding accumulates across the 24-layer WavLM, most
        // visibly in its final layer. The learned layer mixture, projection,
        // and LayerNorm contract that drift again, so keep those downstream
        // limits tight and require exact powerset argmax below.
        let (max_limit, mean_limit) =
            golden_limits(name, benchmark_backend.accelerated_fp16_tolerances);
        if max_abs > max_limit || mean_abs > mean_limit {
            numeric_violations.push(format!(
                "{name} drift: max_abs={max_abs} (limit {max_limit}), mean_abs={mean_abs} (limit {mean_limit})"
            ));
        }
    }

    let logits = trace
        .iter()
        .find(|(name, _)| name == "logits")
        .map(|(_, values)| values.as_slice())
        .expect("logits trace");
    let frames = super::config::output_frames(waveform.len());
    let (classes, activity) = postprocess_logits(logits, frames);
    let expected_classes = npy_i64(&golden, "powerset_class")
        .into_iter()
        .map(|value| value as u8)
        .collect::<Vec<_>>();
    assert_eq!(classes, expected_classes, "powerset argmax parity");

    let mut expected_logits = vec![-1.0e9_f32; frames * POWERSET_CLASSES];
    for (frame, class) in expected_classes.iter().enumerate() {
        expected_logits[frame * POWERSET_CLASSES + *class as usize] = 0.0;
    }
    let (_, expected_activity) = postprocess_logits(&expected_logits, frames);
    assert_eq!(
        activity, expected_activity,
        "median-filtered activity parity"
    );
    assert_eq!(
        decode_segments(&activity, frames),
        decode_segments(&expected_activity, frames),
        "segment-boundary parity"
    );
    assert!(
        numeric_violations.is_empty(),
        "numeric golden violations:\n{}",
        numeric_violations.join("\n")
    );
}

#[test]
#[ignore = "requires OPENASR_DIARIZEN_PACK; native release benchmark"]
fn native_fp16_exact_window_benchmark() {
    let pack = external_path("OPENASR_DIARIZEN_PACK");
    let samples = synthetic_exact_window();
    let benchmark_backend = benchmark_backend();
    let mut runtime =
        DiariZenRuntime::new(&pack, samples.len(), false, Some(benchmark_backend.backend))
            .expect("construct production runtime");
    benchmark_backend.assert_runtime(&runtime);
    let allocation_bytes = runtime
        .prepared_graph_allocation_bytes()
        .expect("direct benchmark backend uses the liveness-aware graph allocator");
    assert!(
        allocation_bytes <= 96 * 1024 * 1024,
        "the fixed 16-second production graph must stay below the audited 96 MiB ceiling; got {allocation_bytes} bytes"
    );

    let warmup = runtime.infer(&samples).expect("warmup inference");
    assert_eq!(warmup.logits.len(), warmup.frame_count * POWERSET_CLASSES);
    let mut seconds = Vec::with_capacity(5);
    let mut checksum = 0.0_f64;
    for _ in 0..5 {
        let started = Instant::now();
        let output = runtime.infer(&samples).expect("timed inference");
        seconds.push(started.elapsed().as_secs_f64());
        checksum += output.logits.iter().map(|value| *value as f64).sum::<f64>();
        std::hint::black_box(output);
    }
    seconds.sort_by(f64::total_cmp);
    let median_seconds = seconds[seconds.len() / 2];
    let audio_seconds = super::config::WINDOW_SAMPLES as f64 / super::config::SAMPLE_RATE_HZ as f64;
    eprintln!(
        "DIARIZEN_NATIVE_BENCH median_seconds={median_seconds:.6} rtf={:.6} graph_allocation_bytes={allocation_bytes} runs={seconds:?} checksum={checksum:.6}",
        median_seconds / audio_seconds
    );
    assert!(median_seconds.is_finite() && median_seconds > 0.0);
    assert!(checksum.is_finite());
}

#[test]
#[ignore = "requires OPENASR_DIARIZEN_PACK; native release benchmark"]
fn native_fp16_sixty_second_window_throughput_benchmark() {
    let pack = external_path("OPENASR_DIARIZEN_PACK");
    let samples = synthetic_exact_window();
    let benchmark_backend = benchmark_backend();
    let mut runtime =
        DiariZenRuntime::new(&pack, samples.len(), false, Some(benchmark_backend.backend))
            .expect("construct production runtime");
    benchmark_backend.assert_runtime(&runtime);
    runtime.infer(&samples).expect("warmup inference");

    // Match pyannote Inference.slide: all complete 16 s windows at the pinned
    // 1.6 s step, followed by one zero-padded tail when needed.
    let recording_samples = 60 * super::config::SAMPLE_RATE_HZ as usize;
    let complete = if recording_samples >= super::config::WINDOW_SAMPLES {
        (recording_samples - super::config::WINDOW_SAMPLES) / super::config::WINDOW_STEP_SAMPLES + 1
    } else {
        0
    };
    let has_tail = recording_samples < super::config::WINDOW_SAMPLES
        || !(recording_samples - super::config::WINDOW_SAMPLES)
            .is_multiple_of(super::config::WINDOW_STEP_SAMPLES);
    let windows = complete + usize::from(has_tail);
    assert_eq!(windows, 29);

    let started = Instant::now();
    let mut checksum = 0.0_f64;
    for _ in 0..windows {
        let output = runtime.infer(&samples).expect("window inference");
        checksum += output.logits.iter().map(|value| *value as f64).sum::<f64>();
        std::hint::black_box(output);
    }
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "DIARIZEN_NATIVE_60S_BENCH elapsed_seconds={elapsed:.6} windows={windows} effective_rtf={:.6} checksum={checksum:.6}",
        elapsed / 60.0
    );
    assert!(elapsed.is_finite() && elapsed > 0.0);
    assert!(checksum.is_finite());
}

/// Private-audio production-window benchmark for the auxiliary-model Pareto
/// gate. The same recording can be supplied to every auxiliary model without
/// baking its path or content into the repository.
#[test]
#[ignore = "host-local: needs OPENASR_DIARIZEN_PACK and OPENASR_AUX_BENCH_AUDIO"]
fn diarizen_aux_audio_sliding_benchmark() {
    let pack = external_path("OPENASR_DIARIZEN_PACK");
    let audio = crate::testing::external_test_fixture_path(
        "OPENASR_AUX_BENCH_AUDIO",
        "private auxiliary-model benchmark audio",
    )
    .expect("OPENASR_AUX_BENCH_AUDIO");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        &audio,
        "DiariZen auxiliary benchmark",
        "DiariZen auxiliary benchmark",
    )
    .expect("load benchmark audio");
    let pcm = crate::PcmBuffer::from_vec(samples);
    let audio_seconds = pcm.len() as f64 / super::config::SAMPLE_RATE_HZ as f64;
    let benchmark_backend = benchmark_backend();
    let mut runtime = DiariZenRuntime::new(
        &pack,
        super::config::WINDOW_SAMPLES,
        false,
        Some(benchmark_backend.backend),
    )
    .expect("construct production runtime");
    benchmark_backend.assert_runtime(&runtime);
    let mut run = || {
        super::super::segment_diarizen_local_activity(
            pcm.full_slice(),
            super::config::SAMPLE_RATE_HZ,
            &|| false,
            None,
            |window| {
                runtime
                    .infer(&window)
                    .map_err(|error| super::super::SegmentError::Inference(error.to_string()))
            },
        )
        .expect("segment benchmark audio")
    };

    let warmup = run();
    let mut last = warmup;
    let seconds = (0..5)
        .map(|_| {
            let started = Instant::now();
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
        "AUX_MODEL_BENCH model=diarizen provider={} placement=full_device backend={} audio_seconds={audio_seconds:.6} median_seconds={median_seconds:.6} rtf={:.6} windows={} activity_sha256={activity_sha256} runs={seconds:?}",
        benchmark_backend.requested_provider,
        runtime.backend_label(),
        median_seconds / audio_seconds,
        last.windows.len(),
    );
}

#[test]
#[ignore = "host-local endurance gate: needs OPENASR_DIARIZEN_PACK and a >=15 minute OPENASR_AUX_BENCH_AUDIO"]
fn diarizen_fifteen_minute_endurance() {
    let pack = external_path("OPENASR_DIARIZEN_PACK");
    let audio = crate::testing::external_test_fixture_path(
        "OPENASR_AUX_BENCH_AUDIO",
        "private auxiliary-model endurance audio",
    )
    .expect("OPENASR_AUX_BENCH_AUDIO");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        &audio,
        "DiariZen endurance gate",
        "DiariZen endurance gate",
    )
    .expect("load endurance audio");
    let audio_seconds = samples.len() as f64 / super::config::SAMPLE_RATE_HZ as f64;
    assert!(audio_seconds >= 15.0 * 60.0, "endurance audio is too short");
    let benchmark_backend = benchmark_backend();
    let mut runtime = DiariZenRuntime::new(
        &pack,
        super::config::WINDOW_SAMPLES,
        false,
        Some(benchmark_backend.backend),
    )
    .expect("construct production runtime");
    benchmark_backend.assert_runtime(&runtime);
    let execution_placement = crate::GgmlExecutionTelemetryCollector::new();
    let _execution_placement_guard = execution_placement.install();
    runtime
        .infer(&samples[..super::config::WINDOW_SAMPLES])
        .expect("warm DiariZen runtime");

    let pcm = crate::PcmBuffer::from_vec(samples);
    let started = Instant::now();
    let activity = super::super::segment_diarizen_local_activity(
        pcm.full_slice(),
        super::config::SAMPLE_RATE_HZ,
        &|| false,
        None,
        |window| {
            runtime
                .infer(&window)
                .map_err(|error| super::super::SegmentError::Inference(error.to_string()))
        },
    )
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
    let observed = execution_placement.snapshot();
    eprintln!(
        "AUX_MODEL_ENDURANCE model=diarizen provider={} placement=full_device backend={} audio_seconds={audio_seconds:.6} elapsed_seconds={elapsed_seconds:.6} rtf={:.6} peak_rss_bytes={peak_rss_bytes} current_rss_bytes={current_rss_bytes} phys_footprint_bytes={phys_footprint_bytes} peak_phys_footprint_bytes={peak_phys_footprint_bytes} windows={} activity_sha256={activity_sha256} observed_compute_nodes={:?} observed_nodes={:?}",
        benchmark_backend.requested_provider,
        runtime.backend_label(),
        elapsed_seconds / audio_seconds,
        activity.windows.len(),
        observed.observed_compute_nodes_by_backend,
        observed.observed_nodes_by_backend,
    );
    assert_selected_backend_compute(benchmark_backend.backend, &observed, "DiariZen endurance");
}

#[test]
#[ignore = "host-local: needs OPENASR_DIARIZEN_PACK and OPENASR_AUX_BENCH_AUDIO"]
fn diarizen_cpu_and_metal_activity_stays_semantically_close() {
    let pack = external_path("OPENASR_DIARIZEN_PACK");
    let audio = crate::testing::external_test_fixture_path(
        "OPENASR_AUX_BENCH_AUDIO",
        "private auxiliary-model benchmark audio",
    )
    .expect("OPENASR_AUX_BENCH_AUDIO");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        &audio,
        "DiariZen backend parity",
        "DiariZen backend parity",
    )
    .expect("load parity audio");
    let pcm = crate::PcmBuffer::from_vec(samples);
    let run = |backend| {
        let mut runtime =
            DiariZenRuntime::new(&pack, super::config::WINDOW_SAMPLES, false, Some(backend))
                .expect("construct parity runtime");
        super::super::segment_diarizen_local_activity(
            pcm.full_slice(),
            super::config::SAMPLE_RATE_HZ,
            &|| false,
            None,
            |window| {
                runtime
                    .infer(&window)
                    .map_err(|error| super::super::SegmentError::Inference(error.to_string()))
            },
        )
        .expect("segment parity audio")
    };

    let cpu = run(crate::ggml_runtime::GgmlCpuGraphBackend::Cpu);
    let metal = run(crate::ggml_runtime::GgmlCpuGraphBackend::Metal);
    assert_eq!(cpu.frame_clock, metal.frame_clock);
    assert_eq!(cpu.local_speaker_slots, metal.local_speaker_slots);
    assert_eq!(cpu.windows.len(), metal.windows.len());
    assert_eq!(cpu.speaker_count.len(), metal.speaker_count.len());

    let mut frame_count = 0usize;
    let mut exact_mask_mismatches = 0usize;
    let mut active_count_mismatches = 0usize;
    for (cpu_window, metal_window) in cpu.windows.iter().zip(&metal.windows) {
        assert_eq!(cpu_window.start_sample, metal_window.start_sample);
        assert_eq!(
            cpu_window.frame_activity.len(),
            metal_window.frame_activity.len()
        );
        for (&cpu_mask, &metal_mask) in cpu_window
            .frame_activity
            .iter()
            .zip(&metal_window.frame_activity)
        {
            frame_count += 1;
            exact_mask_mismatches += usize::from(cpu_mask != metal_mask);
            active_count_mismatches +=
                usize::from(cpu_mask.count_ones() != metal_mask.count_ones());
        }
    }
    let aggregate_mismatches = cpu
        .speaker_count
        .iter()
        .zip(&metal.speaker_count)
        .filter(|(cpu_count, metal_count)| cpu_count != metal_count)
        .count();
    let exact_mask_rate = exact_mask_mismatches as f64 / frame_count as f64;
    let active_count_rate = active_count_mismatches as f64 / frame_count as f64;
    let aggregate_rate = aggregate_mismatches as f64 / cpu.speaker_count.len() as f64;
    eprintln!(
        "DIARIZEN_BACKEND_PARITY frames={frame_count} exact_mask_mismatches={exact_mask_mismatches} exact_mask_rate={exact_mask_rate:.8} active_count_mismatches={active_count_mismatches} active_count_rate={active_count_rate:.8} aggregate_mismatches={aggregate_mismatches} aggregate_rate={aggregate_rate:.8}"
    );

    assert!(
        active_count_rate <= 0.01,
        "CPU/Metal active-speaker counts diverged for {:.3}% of window frames",
        active_count_rate * 100.0
    );
    assert!(
        aggregate_rate <= 0.01,
        "CPU/Metal aggregated speaker counts diverged for {:.3}% of recording frames",
        aggregate_rate * 100.0
    );
}

#[test]
#[ignore = "host-local: needs OPENASR_DIARIZEN_PACK, OPENASR_AUX_BENCH_AUDIO, and exact cuda or vulkan backend"]
fn diarizen_cpu_and_exact_gpu_activity_stays_semantically_close() {
    let pack = external_path("OPENASR_DIARIZEN_PACK");
    let audio = crate::testing::external_test_fixture_path(
        "OPENASR_AUX_BENCH_AUDIO",
        "private auxiliary-model benchmark audio",
    )
    .expect("OPENASR_AUX_BENCH_AUDIO");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        &audio,
        "DiariZen exact GPU parity",
        "DiariZen exact GPU parity",
    )
    .expect("load parity audio");
    let pcm = crate::PcmBuffer::from_vec(samples);
    let benchmark_backend = benchmark_backend();
    assert!(
        benchmark_backend.accelerated_fp16_tolerances,
        "this parity gate requires OPENASR_DIARIZEN_BENCH_BACKEND=cuda or vulkan"
    );

    let run = |backend| {
        let mut runtime =
            DiariZenRuntime::new(&pack, super::config::WINDOW_SAMPLES, false, Some(backend))
                .expect("construct parity runtime");
        if backend == benchmark_backend.backend {
            benchmark_backend.assert_runtime(&runtime);
        }
        super::super::segment_diarizen_local_activity(
            pcm.full_slice(),
            super::config::SAMPLE_RATE_HZ,
            &|| false,
            None,
            |window| {
                runtime
                    .infer(&window)
                    .map_err(|error| super::super::SegmentError::Inference(error.to_string()))
            },
        )
        .expect("segment parity audio")
    };

    let cpu = run(crate::ggml_runtime::GgmlCpuGraphBackend::Cpu);
    let accelerated = run(benchmark_backend.backend);
    assert_eq!(cpu.frame_clock, accelerated.frame_clock);
    assert_eq!(cpu.local_speaker_slots, accelerated.local_speaker_slots);
    assert_eq!(cpu.windows.len(), accelerated.windows.len());
    assert_eq!(cpu.speaker_count.len(), accelerated.speaker_count.len());

    let mut frame_count = 0usize;
    let mut exact_mask_mismatches = 0usize;
    let mut active_count_mismatches = 0usize;
    for (cpu_window, accelerated_window) in cpu.windows.iter().zip(&accelerated.windows) {
        assert_eq!(cpu_window.start_sample, accelerated_window.start_sample);
        assert_eq!(
            cpu_window.frame_activity.len(),
            accelerated_window.frame_activity.len()
        );
        for (&cpu_mask, &accelerated_mask) in cpu_window
            .frame_activity
            .iter()
            .zip(&accelerated_window.frame_activity)
        {
            frame_count += 1;
            exact_mask_mismatches += usize::from(cpu_mask != accelerated_mask);
            active_count_mismatches +=
                usize::from(cpu_mask.count_ones() != accelerated_mask.count_ones());
        }
    }
    assert!(frame_count > 0, "parity audio produced no activity frames");
    let aggregate_mismatches = cpu
        .speaker_count
        .iter()
        .zip(&accelerated.speaker_count)
        .filter(|(cpu_count, accelerated_count)| cpu_count != accelerated_count)
        .count();
    let exact_mask_rate = exact_mask_mismatches as f64 / frame_count as f64;
    let active_count_rate = active_count_mismatches as f64 / frame_count as f64;
    let aggregate_rate = aggregate_mismatches as f64 / cpu.speaker_count.len() as f64;
    eprintln!(
        "DIARIZEN_EXACT_GPU_PARITY provider={} placement=full_device frames={frame_count} exact_mask_mismatches={exact_mask_mismatches} exact_mask_rate={exact_mask_rate:.8} active_count_mismatches={active_count_mismatches} active_count_rate={active_count_rate:.8} aggregate_mismatches={aggregate_mismatches} aggregate_rate={aggregate_rate:.8}",
        benchmark_backend.requested_provider,
    );

    assert!(
        active_count_rate <= 0.01,
        "CPU/{} active-speaker counts diverged for {:.3}% of window frames",
        benchmark_backend.requested_provider,
        active_count_rate * 100.0
    );
    assert!(
        aggregate_rate <= 0.01,
        "CPU/{} aggregated speaker counts diverged for {:.3}% of recording frames",
        benchmark_backend.requested_provider,
        aggregate_rate * 100.0
    );
}
