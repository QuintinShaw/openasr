use super::*;

use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

fn external_path(env: &str) -> PathBuf {
    std::env::var_os(env)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{env} must point at the external DiariZen parity fixture"))
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
fn postprocess_preserves_frame_geometry() {
    let output = postprocess_logits(&[0.0; 2 * POWERSET_CLASSES], 2);
    assert_eq!(output.0.len(), 2);
    assert_eq!(output.1.len(), 2 * LOCAL_SPEAKERS);
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
    DiariZenSegmenter::probe_oasr(&pack).expect("strict pack probe");
    let segmenter = DiariZenSegmenter::from_oasr(&pack).expect("construct production adapter");
    let waveform = npy_f32(&golden, "waveform");
    assert!(matches!(
        segmenter.infer_window(&waveform, super::config::SAMPLE_RATE_HZ),
        Err(DiariZenSegmenterError::WindowSize {
            expected: super::config::WINDOW_SAMPLES,
            actual: 16_000,
        })
    ));
    assert!(matches!(
        segmenter.infer_window(
            &vec![0.0; super::config::WINDOW_SAMPLES],
            super::config::SAMPLE_RATE_HZ / 2,
        ),
        Err(DiariZenSegmenterError::UnsupportedSampleRate { actual: 8_000 })
    ));
    let mut runtime = DiariZenRuntime::new(
        &pack,
        16_000,
        true,
        Some(crate::ggml_runtime::GgmlCpuGraphBackend::Cpu),
    )
    .expect("construct test-geometry runtime");
    let trace = runtime.infer_trace(&waveform).expect("native trace");

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
        let (max_limit, mean_limit) = match name.as_str() {
            "wavlm_layer_23" => (3.0, 0.14),
            name if name.starts_with("wavlm_layer_") => (0.55, 0.035),
            "weighted_layer_sum_raw" => (1.0, 0.05),
            "projection_raw" => (1.5, 0.18),
            "projection_norm" | "conformer_layer_00" | "conformer_layer_01"
            | "conformer_layer_02" | "conformer_layer_03" => (0.08, 0.008),
            "logits" => (0.04, 0.015),
            _ => (0.05, 0.005),
        };
        assert!(
            max_abs <= max_limit && mean_abs <= mean_limit,
            "{name} drift: max_abs={max_abs} (limit {max_limit}), mean_abs={mean_abs} (limit {mean_limit})"
        );
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
}

#[test]
#[ignore = "requires OPENASR_DIARIZEN_PACK; native release benchmark"]
fn native_fp16_exact_window_benchmark() {
    let pack = external_path("OPENASR_DIARIZEN_PACK");
    let segmenter = DiariZenSegmenter::from_oasr(&pack).expect("construct production runtime");
    let samples = synthetic_exact_window();

    let warmup = segmenter
        .infer_window(&samples, super::config::SAMPLE_RATE_HZ)
        .expect("warmup inference");
    assert_eq!(warmup.logits.len(), warmup.frame_count * POWERSET_CLASSES);

    let mut seconds = Vec::with_capacity(5);
    let mut checksum = 0.0_f64;
    for _ in 0..5 {
        let started = Instant::now();
        let output = segmenter
            .infer_window(&samples, super::config::SAMPLE_RATE_HZ)
            .expect("timed inference");
        seconds.push(started.elapsed().as_secs_f64());
        checksum += output.logits.iter().map(|value| *value as f64).sum::<f64>();
        std::hint::black_box(output);
    }
    seconds.sort_by(f64::total_cmp);
    let median_seconds = seconds[seconds.len() / 2];
    let audio_seconds = super::config::WINDOW_SAMPLES as f64 / super::config::SAMPLE_RATE_HZ as f64;
    eprintln!(
        "DIARIZEN_NATIVE_BENCH median_seconds={median_seconds:.6} rtf={:.6} runs={seconds:?} checksum={checksum:.6}",
        median_seconds / audio_seconds
    );
    assert!(median_seconds.is_finite() && median_seconds > 0.0);
    assert!(checksum.is_finite());
}

#[test]
#[ignore = "requires OPENASR_DIARIZEN_PACK; validates failed-compute recovery"]
fn native_runtime_rebuilds_only_the_poisoned_graph_after_abort() {
    let _test_guard = diarizen_runtime_test_lock();
    let pack = external_path("OPENASR_DIARIZEN_PACK");
    let segmenter = DiariZenSegmenter::from_oasr(&pack).expect("construct production runtime");
    let samples = synthetic_exact_window();

    install_worker_graph_compute_abort();
    let error = segmenter
        .infer_window(&samples, super::config::SAMPLE_RATE_HZ)
        .expect_err("injected abort must fail this inference");
    assert!(error.is_canceled());

    let recovered = segmenter
        .infer_window(&samples, super::config::SAMPLE_RATE_HZ)
        .expect("the same resident weights must rebuild a clean graph");
    assert_eq!(
        recovered.logits.len(),
        recovered.frame_count * POWERSET_CLASSES
    );
}

#[test]
#[ignore = "requires OPENASR_DIARIZEN_PACK; validates process-owner shutdown and rebuild"]
fn native_runtime_rebuilds_after_process_owner_shutdown() {
    let _test_guard = diarizen_runtime_test_lock();
    let pack = external_path("OPENASR_DIARIZEN_PACK");
    let segmenter = DiariZenSegmenter::from_oasr(&pack).expect("construct production runtime");
    let samples = synthetic_exact_window();

    drop(crate::NativeRuntimeShutdownGuard::new());
    segmenter
        .infer_window(&samples, super::config::SAMPLE_RATE_HZ)
        .expect("first request");
    assert_eq!(diarizen_worker_runtime_entry_count(), 1);

    drop(crate::NativeRuntimeShutdownGuard::new());
    assert_eq!(
        diarizen_worker_runtime_entry_count(),
        0,
        "process-owner shutdown must eagerly clear persistent worker TLS"
    );

    segmenter
        .infer_window(&samples, super::config::SAMPLE_RATE_HZ)
        .expect("request after shutdown rebuilds");
    assert_eq!(
        diarizen_worker_runtime_entry_count(),
        1,
        "the first request after shutdown must rebuild resident state"
    );
    drop(crate::NativeRuntimeShutdownGuard::new());
}

#[test]
#[ignore = "requires OPENASR_DIARIZEN_PACK; validates standalone-adapter shutdown"]
fn standalone_segmenter_drop_eagerly_releases_worker_runtime() {
    let _test_guard = diarizen_runtime_test_lock();
    unload_idle_worker_runtimes();
    let pack = external_path("OPENASR_DIARIZEN_PACK");
    let samples = synthetic_exact_window();

    {
        let segmenter = DiariZenSegmenter::from_oasr(&pack).expect("construct production runtime");
        segmenter
            .infer_window(&samples, super::config::SAMPLE_RATE_HZ)
            .expect("standalone request");
        assert_eq!(diarizen_worker_runtime_entry_count(), 1);
    }

    assert_eq!(
        diarizen_worker_runtime_entry_count(),
        0,
        "dropping the final standalone adapter must clear persistent worker TLS"
    );
}

#[test]
#[ignore = "requires OPENASR_DIARIZEN_PACK; validates terminal-backend eviction"]
fn native_runtime_rebuilds_the_runner_after_device_loss_without_retrying_request() {
    let _test_guard = diarizen_runtime_test_lock();
    let pack = external_path("OPENASR_DIARIZEN_PACK");
    let segmenter = DiariZenSegmenter::from_oasr(&pack).expect("construct production runtime");
    let samples = synthetic_exact_window();

    install_worker_graph_compute_device_lost();
    let error = segmenter
        .infer_window(&samples, super::config::SAMPLE_RATE_HZ)
        .expect_err("device loss must fail this request without retry");
    assert!(matches!(
        error,
        DiariZenSegmenterError::Graph {
            source: crate::ggml_runtime::GgmlCpuGraphError::DeviceLost,
            ..
        }
    ));
    assert_eq!(
        diarizen_worker_runtime_entry_count(),
        0,
        "a terminal backend handle must be evicted on its owner worker"
    );

    let recovered = segmenter
        .infer_window(&samples, super::config::SAMPLE_RATE_HZ)
        .expect("the next request builds a fresh runner");
    assert_eq!(
        recovered.logits.len(),
        recovered.frame_count * POWERSET_CLASSES
    );
}

#[test]
#[ignore = "requires OPENASR_DIARIZEN_PACK; native release benchmark"]
fn native_fp16_sixty_second_window_throughput_benchmark() {
    let pack = external_path("OPENASR_DIARIZEN_PACK");
    let segmenter = DiariZenSegmenter::from_oasr(&pack).expect("construct production runtime");
    let samples = synthetic_exact_window();
    segmenter
        .infer_window(&samples, super::config::SAMPLE_RATE_HZ)
        .expect("warmup inference");

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
        let output = segmenter
            .infer_window(&samples, super::config::SAMPLE_RATE_HZ)
            .expect("window inference");
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
