//! Numerical parity (Rust causal forward pass vs. a numpy reference
//! reproduction of the upstream torch `DetectModel` with `N2 = 0`) and
//! provider smoke tests for Stream-VAD.

use super::model::FireRedStreamVadModel;
use super::provider::FireRedStreamVadProvider;

/// Golden fixture: a 3 s (48,000-sample) excerpt of `fixtures/jfk.wav`, plus
/// reference per-10ms-frame speech probabilities from a numpy reproduction of
/// the upstream `DetectModel`
/// forward with `N2 = 0` (no lookahead) run against the vendored
/// `Stream-VAD/model.pth.tar` + `Stream-VAD/cmvn.ark` checkpoint (there is no
/// upstream Python streaming-VAD "batch" entrypoint to diff against directly;
/// the reference forward is the same math this module implements, checked
/// independently against the checkpoint's raw tensors). Binary layout: magic
/// `"FRSG"`, `u32 n_samples`, `u32 n_frames`, `f32[n_samples]` samples,
/// `f32[n_frames]` reference probs (all little-endian).
fn golden() -> (Vec<f32>, Vec<f32>) {
    const GOLDEN: &[u8] = include_bytes!("../assets/firered_stream_vad_16k_golden.bin");
    assert_eq!(&GOLDEN[0..4], b"FRSG", "golden magic");
    let n_samples = u32::from_le_bytes(GOLDEN[4..8].try_into().unwrap()) as usize;
    let n_frames = u32::from_le_bytes(GOLDEN[8..12].try_into().unwrap()) as usize;
    let mut off = 12;
    let mut read = |n: usize| -> Vec<f32> {
        let out = GOLDEN[off..off + n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        off += n * 4;
        out
    };
    let samples = read(n_samples);
    let probs = read(n_frames);
    (samples, probs)
}

fn max_abs_diff_with_location(got: &[f32], want: &[f32]) -> (f32, usize) {
    assert_eq!(got.len(), want.len(), "frame count mismatch");
    let mut worst = 0.0f32;
    let mut worst_idx = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            worst_idx = i;
        }
    }
    (worst, worst_idx)
}

fn mean_abs_diff(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "frame count mismatch");
    got.iter()
        .zip(want)
        .map(|(actual, expected)| f64::from((actual - expected).abs()))
        .sum::<f64>()
        / got.len().max(1) as f64
}

struct BenchmarkBackend {
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    requested_provider: &'static str,
    exact_route_identity: Option<String>,
    _route_guard: Option<crate::ggml_runtime::RequestBackendOverrideGuard>,
}

fn benchmark_backend() -> BenchmarkBackend {
    let requested = std::env::var("OPENASR_FIRERED_STREAM_VAD_BENCH_BACKEND")
        .unwrap_or_else(|_| "cpu".to_string())
        .trim()
        .to_ascii_lowercase();
    let (backend, provider, requested_provider) = match requested.as_str() {
        "cpu" => (
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
            crate::device::execution_route::ExecutionProvider::Cpu,
            "cpu",
        ),
        "metal" => (
            crate::ggml_runtime::GgmlCpuGraphBackend::Metal,
            crate::device::execution_route::ExecutionProvider::Metal,
            "metal",
        ),
        "cuda" => (
            crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            crate::device::execution_route::ExecutionProvider::Cuda,
            "cuda",
        ),
        "vulkan" => (
            crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            crate::device::execution_route::ExecutionProvider::Vulkan,
            "vulkan",
        ),
        backend => panic!(
            "OPENASR_FIRERED_STREAM_VAD_BENCH_BACKEND must be cpu, metal, cuda, or vulkan; got '{backend}'"
        ),
    };
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
        .unwrap_or_else(|| {
            panic!("requested FireRedVAD provider '{requested_provider}' is unavailable")
        })
        .to_resolved_route();
        exact_route_identity = Some(route.isolation_key());
        Some(crate::ggml_runtime::install_request_backend_override(Some(
            crate::ggml_runtime::RequestBackendPreference::Exact(route),
        )))
    } else {
        None
    };
    BenchmarkBackend {
        backend,
        requested_provider,
        exact_route_identity,
        _route_guard: route_guard,
    }
}

fn streaming_probabilities_for_backend(
    model: &'static FireRedStreamVadModel,
    samples: &[f32],
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
) -> Result<Vec<f32>, String> {
    let default_chunk_samples = super::provider::offline_chunk_samples_for_backend(backend);
    let chunk_seconds = std::env::var("OPENASR_FIRERED_STREAM_VAD_BENCH_CHUNK_SECONDS")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("OPENASR_FIRERED_STREAM_VAD_BENCH_CHUNK_SECONDS must be an integer")
        })
        .unwrap_or_else(|| default_chunk_samples / super::frontend::SAMPLE_RATE_HZ as usize)
        .max(1);
    let chunk_samples = chunk_seconds.saturating_mul(super::frontend::SAMPLE_RATE_HZ as usize);
    let mut streaming = super::streaming::FireRedStreamingVad::from_model(model);
    let placement = if backend == crate::ggml_runtime::GgmlCpuGraphBackend::Cpu {
        crate::device::execution_policy::ExecutionPlacement::CpuOnly
    } else {
        crate::device::execution_policy::ExecutionPlacement::FullDevice
    };
    let mut runtime = (backend != crate::ggml_runtime::GgmlCpuGraphBackend::Cpu)
        .then(|| super::ggml_runtime::FireRedStreamVadGgmlRuntime::new(model, backend, placement))
        .transpose()
        .map_err(|error| error.to_string())?;
    let mut probabilities = Vec::with_capacity(samples.len().div_ceil(160));
    for chunk in samples.chunks(chunk_samples) {
        let next = if let Some(runtime) = runtime.as_mut() {
            streaming
                .accept_f32_chunk_with(chunk, |features, frames, cache| {
                    runtime.forward_chunk(features, frames, cache)
                })
                .map_err(|error| error.to_string())?
        } else {
            streaming.accept_f32_chunk(chunk)
        };
        probabilities.extend(next);
    }
    Ok(probabilities)
}

/// Read the one-dimensional little-endian f32 NPY emitted by the pinned
/// upstream FireRedVAD reference harness. Keeping this test-only parser local
/// avoids adding a production dependency for a host-local parity fixture.
fn read_reference_probabilities(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    assert_eq!(&bytes[..6], b"\x93NUMPY", "npy magic");
    let major = bytes[6];
    let (header_start, header_len) = if major == 1 {
        (
            10,
            u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize,
        )
    } else {
        (
            12,
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
        )
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .expect("npy header utf8");
    assert!(header.contains("'<f4'"), "expected <f4 npy, got {header}");
    assert!(
        header.contains("'fortran_order': False"),
        "expected C-order npy"
    );
    let data_start = header_start + header_len;
    assert!(
        (bytes.len() - data_start).is_multiple_of(4),
        "npy f32 payload alignment"
    );
    bytes[data_start..]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn forward_pass_matches_reference_within_tolerance() {
    let (samples, reference_probs) = golden();
    let model = FireRedStreamVadModel::embedded().expect("vendored firered Stream-VAD weights");
    let probs = model.probabilities(&samples);

    assert_eq!(probs.len(), reference_probs.len());
    let (max_diff, at) = max_abs_diff_with_location(&probs, &reference_probs);
    assert!(
        max_diff < 1e-3,
        "max abs prob error {max_diff} at frame {at} (got {}, want {}) exceeds tolerance",
        probs[at],
        reference_probs[at],
    );
}

#[test]
fn metal_graph_matches_cpu_probabilities_and_product_spans_on_macos() {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return;
    }
    let (samples, _) = golden();
    let model = FireRedStreamVadModel::embedded().expect("vendored firered Stream-VAD weights");
    let (features, frames) = model.cmvn_features(&samples);
    let mut cpu_cache = super::model::FireRedStreamVadCache::new();
    let cpu = model.forward_chunk(&features, frames, &mut cpu_cache);
    let mut runtime = super::ggml_runtime::FireRedStreamVadGgmlRuntime::new(
        &model,
        crate::ggml_runtime::GgmlCpuGraphBackend::Metal,
        crate::device::execution_policy::ExecutionPlacement::FullDevice,
    )
    .expect("Apple Silicon must construct the FireRedVAD Metal graph");
    let metal = runtime
        .forward_chunk(
            &features,
            frames,
            &mut super::model::FireRedStreamVadCache::new(),
        )
        .expect("Metal FireRedVAD forward");
    let (max_abs, worst_frame) = max_abs_diff_with_location(&metal, &cpu);
    assert!(
        max_abs < 5e-3,
        "Metal/CPU probability max abs {max_abs} at frame {worst_frame}"
    );
    let options = crate::LongFormOptions::default();
    assert_eq!(
        super::provider::spans_from_probs(&metal, samples.len(), &options),
        super::provider::spans_from_probs(&cpu, samples.len(), &options),
        "Metal and CPU probabilities must produce identical product speech spans"
    );
}

#[test]
#[ignore = "host-local: set OPENASR_FIRERED_STREAM_VAD_BENCH_BACKEND=cuda or vulkan"]
fn exact_gpu_graph_matches_cpu_probabilities_and_product_spans() {
    let benchmark_backend = benchmark_backend();
    assert_eq!(
        benchmark_backend.backend,
        crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
        "this parity gate requires OPENASR_FIRERED_STREAM_VAD_BENCH_BACKEND=cuda or vulkan"
    );
    let (samples, _) = golden();
    let model = super::shared_model().expect("vendored FireRedVAD weights");
    let cpu = streaming_probabilities_for_backend(
        model,
        &samples,
        crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
    )
    .expect("run FireRedVAD CPU reference");
    let accelerated =
        streaming_probabilities_for_backend(model, &samples, benchmark_backend.backend)
            .expect("run FireRedVAD exact GPU route");
    let (max_abs, worst_frame) = max_abs_diff_with_location(&accelerated, &cpu);
    let mean_abs = mean_abs_diff(&accelerated, &cpu);
    eprintln!(
        "FIRERED_VAD_EXACT_GPU_PARITY provider={} placement=full-device exact_route={} frames={} max_abs={max_abs:.9} mean_abs={mean_abs:.9} worst_frame={worst_frame}",
        benchmark_backend.requested_provider,
        benchmark_backend
            .exact_route_identity
            .as_deref()
            .expect("exact GPU route identity"),
        accelerated.len(),
    );
    assert!(
        accelerated
            .iter()
            .all(|probability| probability.is_finite() && (0.0..=1.0).contains(probability)),
        "exact GPU probabilities must be finite and in [0, 1]"
    );
    assert!(
        max_abs < 5e-3,
        "CPU/{} max abs probability difference {max_abs} at frame {worst_frame} exceeds tolerance",
        benchmark_backend.requested_provider,
    );
    assert!(
        mean_abs < 1e-4,
        "CPU/{} mean abs probability difference {mean_abs} exceeds tolerance",
        benchmark_backend.requested_provider,
    );
    let options = crate::LongFormOptions::default();
    assert_eq!(
        super::provider::spans_from_probs(&accelerated, samples.len(), &options),
        super::provider::spans_from_probs(&cpu, samples.len(), &options),
        "CPU and exact GPU probabilities must yield identical speech spans"
    );
}

#[test]
#[ignore = "host-local: needs OPENASR_FIRERED_STREAM_VAD_REFERENCE_AUDIO and OPENASR_FIRERED_STREAM_VAD_REFERENCE_NPY"]
fn firered_stream_vad_matches_official_reference_on_aux_audio() {
    let audio = match crate::testing::external_test_fixture_path(
        "OPENASR_FIRERED_STREAM_VAD_REFERENCE_AUDIO",
        "official FireRedVAD parity audio",
    ) {
        Ok(path) => path,
        Err(skip) => {
            eprintln!("skipping: {skip}");
            return;
        }
    };
    let reference = match crate::testing::external_test_fixture_path(
        "OPENASR_FIRERED_STREAM_VAD_REFERENCE_NPY",
        "official FireRedVAD probability reference",
    ) {
        Ok(path) => path,
        Err(skip) => {
            eprintln!("skipping: {skip}");
            return;
        }
    };
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        &audio,
        "FireRedVAD official parity",
        "FireRedVAD official parity",
    )
    .expect("load official parity audio");
    let reference_probs = read_reference_probabilities(&reference);
    let model = super::shared_model().expect("vendored FireRedVAD weights");
    let benchmark_backend = benchmark_backend();
    let backend = benchmark_backend.backend;
    let execution_placement = crate::GgmlExecutionTelemetryCollector::new();
    let _execution_placement_guard = execution_placement.install();
    let probabilities = streaming_probabilities_for_backend(model, &samples, backend)
        .expect("run policy-equivalent FireRedVAD");
    let observed = execution_placement.snapshot();
    let (max_abs, worst_frame) = max_abs_diff_with_location(&probabilities, &reference_probs);
    let mut absolute_errors = probabilities
        .iter()
        .zip(&reference_probs)
        .map(|(actual, expected)| (actual - expected).abs())
        .collect::<Vec<_>>();
    absolute_errors.sort_by(f32::total_cmp);
    let mean_abs = absolute_errors
        .iter()
        .map(|value| f64::from(*value))
        .sum::<f64>()
        / absolute_errors.len().max(1) as f64;
    let p99_abs = absolute_errors[(absolute_errors.len() - 1) * 99 / 100];
    let threshold_disagreements = probabilities
        .iter()
        .zip(&reference_probs)
        .filter(|(actual, expected)| (**actual >= 0.5) != (**expected >= 0.5))
        .count();
    let options = crate::LongFormOptions::default();
    let native_spans = super::provider::spans_from_probs(&probabilities, samples.len(), &options);
    let reference_spans =
        super::provider::spans_from_probs(&reference_probs, samples.len(), &options);
    let memory = crate::metrics::process_memory_snapshot();
    eprintln!(
        "FIRERED_VAD_OFFICIAL_PARITY provider={} placement={} exact_route={} backend={backend:?} frames={} max_abs={max_abs:.9} p99_abs={p99_abs:.9} mean_abs={mean_abs:.9} worst_frame={worst_frame} actual_at_worst={:.9} reference_at_worst={:.9} threshold_disagreements={threshold_disagreements} speech_spans={} observed_compute_nodes={:?} current_rss_bytes={:?} peak_rss_bytes={:?} current_phys_footprint_bytes={:?} peak_phys_footprint_bytes={:?}",
        benchmark_backend.requested_provider,
        if backend == crate::ggml_runtime::GgmlCpuGraphBackend::Cpu {
            "cpu-only"
        } else {
            "full-device"
        },
        benchmark_backend
            .exact_route_identity
            .as_deref()
            .unwrap_or("coarse"),
        probabilities.len(),
        probabilities[worst_frame],
        reference_probs[worst_frame],
        native_spans.len(),
        observed.observed_compute_nodes_by_backend,
        memory.current_rss_bytes,
        memory.peak_rss_bytes,
        memory.current_phys_footprint_bytes,
        memory.peak_phys_footprint_bytes,
    );
    assert!(
        // Torch/BLAS and Rust use different f32 reduction orders. The DFSMN
        // has a fixed 152-frame receptive field, so this is a local numerical
        // bound rather than a sequence-length-dependent drift allowance.
        max_abs < 5e-3,
        "official probability max abs {max_abs} at frame {worst_frame} exceeds tolerance"
    );
    assert!(
        mean_abs < 1e-4,
        "official probability mean abs {mean_abs} exceeds tolerance"
    );
    assert_eq!(
        native_spans, reference_spans,
        "official and native probabilities must produce identical product speech spans despite {threshold_disagreements} near-threshold frame(s)"
    );
    if backend == crate::ggml_runtime::GgmlCpuGraphBackend::Metal {
        assert!(
            !observed.observed_compute_nodes_by_backend.is_empty()
                && observed
                    .observed_compute_nodes_by_backend
                    .keys()
                    .all(|backend| {
                        let backend = backend.to_ascii_lowercase();
                        backend.starts_with("mtl") || backend.contains("metal")
                    }),
            "explicit Metal FireRedVAD route observed non-Metal compute: {:?}",
            observed.observed_compute_nodes_by_backend
        );
    }
}

#[test]
#[ignore = "host-local benchmark: needs OPENASR_FIRERED_STREAM_VAD_REFERENCE_AUDIO and Metal"]
fn firered_stream_vad_cpu_metal_aux_benchmark() {
    let audio = crate::testing::external_test_fixture_path(
        "OPENASR_FIRERED_STREAM_VAD_REFERENCE_AUDIO",
        "FireRedVAD CPU/Metal benchmark audio",
    )
    .expect("OPENASR_FIRERED_STREAM_VAD_REFERENCE_AUDIO");
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        &audio,
        "FireRedVAD CPU/Metal benchmark",
        "FireRedVAD CPU/Metal benchmark",
    )
    .expect("load benchmark audio");
    let model = super::shared_model().expect("vendored FireRedVAD weights");
    let audio_seconds = samples.len() as f64 / super::frontend::SAMPLE_RATE_HZ as f64;
    let mut outputs = Vec::new();
    for backend in [
        crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
        crate::ggml_runtime::GgmlCpuGraphBackend::Metal,
    ] {
        streaming_probabilities_for_backend(model, &samples[..samples.len().min(16_000)], backend)
            .expect("warm FireRedVAD backend");
        let mut probabilities = Vec::new();
        let seconds = (0..5)
            .map(|_| {
                let started = std::time::Instant::now();
                probabilities = streaming_probabilities_for_backend(model, &samples, backend)
                    .expect("run FireRedVAD backend");
                started.elapsed().as_secs_f64()
            })
            .collect::<Vec<_>>();
        let probability_sha256 = crate::testing::benchmark_sha256_f32(&probabilities);
        let (median_seconds, seconds) = crate::testing::benchmark_median_seconds(seconds);
        eprintln!(
            "AUX_MODEL_BENCH model=fireredvad backend={backend:?} audio_seconds={audio_seconds:.6} median_seconds={median_seconds:.6} rtf={:.6} frames={} probability_sha256={probability_sha256} runs={seconds:?}",
            median_seconds / audio_seconds,
            probabilities.len(),
        );
        outputs.push(probabilities);
    }
    let (max_abs, worst_frame) = max_abs_diff_with_location(&outputs[1], &outputs[0]);
    let options = crate::LongFormOptions::default();
    assert!(
        max_abs < 5e-3,
        "CPU/Metal probability max abs {max_abs} at frame {worst_frame}"
    );
    assert_eq!(
        super::provider::spans_from_probs(&outputs[1], samples.len(), &options),
        super::provider::spans_from_probs(&outputs[0], samples.len(), &options),
        "CPU and Metal must preserve product speech spans"
    );
}

#[test]
fn chunked_streaming_forward_matches_batch_forward_bit_close() {
    // The load-bearing invariant this whole module exists for: chunking the
    // same audio through the cached streaming forward must reproduce the
    // whole-utterance batch forward, since Stream-VAD has no lookahead.
    use super::streaming::FireRedStreamingVad;

    let (samples, _) = golden();
    let model = FireRedStreamVadModel::embedded().expect("vendored firered Stream-VAD weights");
    let batch_probs = model.probabilities(&samples);

    let pcm: Vec<i16> = samples
        .iter()
        .map(|s| (s * 32_768.0).clamp(-32_768.0, 32_767.0) as i16)
        .collect();
    let mut streaming = FireRedStreamingVad::shared().expect("shared Stream-VAD streaming model");
    let mut streamed_last = 0.0f32;
    // An odd, non-frame-aligned chunk size (37 samples) to stress the
    // raw-buffer bookkeeping.
    for frame in pcm.chunks(37) {
        streamed_last = streaming.accept_frame(frame);
    }
    let (max_diff, _) = max_abs_diff_with_location(
        &[streamed_last],
        &[*batch_probs.last().expect("non-empty batch probs")],
    );
    assert!(
        max_diff < 1e-4,
        "chunked streaming forward diverged from batch forward by {max_diff}"
    );
}

#[test]
fn offline_f32_chunking_matches_every_batch_probability() {
    use super::streaming::FireRedStreamingVad;

    let (samples, _) = golden();
    let model = FireRedStreamVadModel::embedded().expect("vendored firered Stream-VAD weights");
    let batch = model.probabilities(&samples);
    let mut streaming = FireRedStreamingVad::shared().expect("shared Stream-VAD streaming model");
    let mut chunked = Vec::new();
    for chunk in samples.chunks(7_913) {
        chunked.extend(streaming.accept_f32_chunk(chunk));
    }
    let (max_diff, at) = max_abs_diff_with_location(&chunked, &batch);
    assert!(
        max_diff < 1e-5,
        "offline f32 chunking diverged from batch by {max_diff} at frame {at}"
    );
}

#[test]
fn probabilities_are_finite_and_in_unit_range() {
    let (samples, _) = golden();
    let model = FireRedStreamVadModel::embedded().expect("vendored firered Stream-VAD weights");
    let probs = model.probabilities(&samples);
    assert!(!probs.is_empty());
    assert!(
        probs
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
    );
}

#[test]
fn empty_audio_returns_no_probabilities() {
    let model = FireRedStreamVadModel::embedded().expect("vendored firered Stream-VAD weights");
    assert!(model.probabilities(&[]).is_empty());
}

#[test]
fn shared_model_loads() {
    assert!(super::shared_model().is_some());
}

#[test]
fn provider_shared_computes_speech_slices_on_golden_clip() {
    use crate::longform::{LongFormOptions, LongFormVadProvider};

    let (samples, _) = golden();
    let provider = FireRedStreamVadProvider::shared().expect("shared Stream-VAD provider");
    let options = LongFormOptions::default();
    let slices = provider
        .compute_speech_slices(&samples, super::frontend::SAMPLE_RATE_HZ, &options)
        .expect("speech slices");
    assert!(!slices.is_empty(), "expected at least one speech span");
    for slice in &slices {
        assert!(slice.end_sample > slice.start_sample);
        assert!(slice.end_sample <= samples.len());
    }
}

#[test]
fn provider_rejects_wrong_sample_rate() {
    use crate::longform::{LongFormOptions, LongFormVadProvider};

    let provider = FireRedStreamVadProvider::shared().expect("shared Stream-VAD provider");
    let samples = vec![0.0f32; 8_000];
    let err = provider
        .compute_speech_slices(&samples, 8_000, &LongFormOptions::default())
        .expect_err("wrong sample rate must fail closed");
    assert!(err.contains("16000"));
}

#[test]
fn provider_cancellable_path_stops_before_frontend_work() {
    let provider = FireRedStreamVadProvider::shared().expect("shared Stream-VAD provider");
    let samples = vec![0.0f32; 32_000];
    let error = provider
        .compute_speech_slices_cancellable(
            &samples,
            super::frontend::SAMPLE_RATE_HZ,
            &crate::LongFormOptions::default(),
            &|| true,
        )
        .expect_err("pre-canceled VAD must stop");
    assert!(matches!(
        error,
        super::provider::FireRedStreamVadError::Canceled
    ));
}

#[test]
fn longform_provider_contract_preserves_typed_cancellation() {
    use crate::longform::{LongFormVadProvider, LongFormVadProviderError};

    let provider = FireRedStreamVadProvider::shared().expect("shared Stream-VAD provider");
    let samples = vec![0.0f32; 32_000];
    let error = LongFormVadProvider::compute_speech_slices_cancellable(
        &provider,
        &samples,
        super::frontend::SAMPLE_RATE_HZ,
        &crate::LongFormOptions::default(),
        &|| true,
    )
    .expect_err("shared long-form contract must preserve provider cancellation");
    assert!(matches!(error, LongFormVadProviderError::Canceled));
}

/// Host-local endurance benchmark over a real >=15-minute recording (not part of the
/// default gate; run explicitly with `--ignored` on a machine that has the
/// fixture). Prints wall-clock forward-pass time and RTF (`elapsed / audio_s`)
/// to stdout with `--nocapture`.
#[test]
#[ignore = "host-local: requires OPENASR_FIRERED_STREAM_VAD_BENCH_AUDIO with >=15 minutes of 16 kHz mono audio"]
fn firered_stream_vad_fifteen_minute_endurance() {
    let path = match crate::testing::external_test_fixture_path(
        "OPENASR_FIRERED_STREAM_VAD_BENCH_AUDIO",
        "Stream-VAD benchmark audio fixture",
    ) {
        Ok(path) => path,
        Err(skip) => {
            eprintln!("skipping: {skip}");
            return;
        }
    };
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        &path,
        "Stream-VAD RTF benchmark",
        "Stream-VAD RTF benchmark",
    )
    .expect("load real endurance wav fixture");
    let audio_seconds = samples.len() as f64 / super::frontend::SAMPLE_RATE_HZ as f64;
    assert!(
        audio_seconds >= 15.0 * 60.0,
        "endurance audio is only {audio_seconds:.3}s"
    );

    let model = super::shared_model().expect("vendored Stream-VAD weights");
    let benchmark_backend = benchmark_backend();
    let backend = benchmark_backend.backend;
    // Warm up (page-in, allocator warm) before timing.
    let _ =
        streaming_probabilities_for_backend(model, &samples[..samples.len().min(16_000)], backend)
            .expect("warm FireRedVAD backend");

    let mut probs = Vec::new();
    let seconds = (0..5)
        .map(|_| {
            let started = std::time::Instant::now();
            probs = streaming_probabilities_for_backend(model, &samples, backend)
                .expect("run FireRedVAD endurance backend");
            started.elapsed().as_secs_f64()
        })
        .collect::<Vec<_>>();
    let probability_sha256 = crate::testing::benchmark_sha256_f32(&probs);
    let (median_seconds, seconds) = crate::testing::benchmark_median_seconds(seconds);
    let rtf = median_seconds / audio_seconds;
    let memory = crate::metrics::process_memory_snapshot();
    let peak_rss_bytes = memory.peak_rss_bytes.unwrap_or(0);
    let current_rss_bytes = memory.current_rss_bytes.unwrap_or(0);
    let phys_footprint_bytes = memory.current_phys_footprint_bytes.unwrap_or(0);
    let peak_phys_footprint_bytes = memory.peak_phys_footprint_bytes.unwrap_or(0);
    let chunk_seconds = std::env::var("OPENASR_FIRERED_STREAM_VAD_BENCH_CHUNK_SECONDS")
        .unwrap_or_else(|_| "1".to_string());
    println!(
        "AUX_MODEL_ENDURANCE model=fireredvad provider={} placement={} exact_route={} backend={backend:?} chunk_seconds={chunk_seconds} audio_seconds={audio_seconds:.6} median_seconds={median_seconds:.6} rtf={rtf:.6} peak_rss_bytes={peak_rss_bytes} current_rss_bytes={current_rss_bytes} phys_footprint_bytes={phys_footprint_bytes} peak_phys_footprint_bytes={peak_phys_footprint_bytes} frames={} probability_sha256={probability_sha256} runs={seconds:?}",
        benchmark_backend.requested_provider,
        if backend == crate::ggml_runtime::GgmlCpuGraphBackend::Cpu {
            "cpu-only"
        } else {
            "full-device"
        },
        benchmark_backend
            .exact_route_identity
            .as_deref()
            .unwrap_or("coarse"),
        probs.len(),
    );
    assert!(!probs.is_empty());
}
