//! WeSpeaker ResNet speaker embedder (256-d), ggml-graph backend.
//!
//! Family architecture id is `wespeaker-resnet`. Size (34/152/221/293) lives in
//! pack metadata; the graph builder is parameterized from that table.

pub(crate) mod backbone;
pub(crate) mod config;
pub(crate) mod frontend;
mod ops;

use super::{EmbedError, weights::Weights};
use backbone::WeSpeakerResNetModel;
use frontend::WeSpeakerFrontend;

pub(crate) use backbone::WeSpeakerResidentRuntime;

const SAMPLE_RATE_HZ: u32 = 16_000;

pub struct WeSpeakerEmbedder {
    model: WeSpeakerResNetModel,
    frontend: WeSpeakerFrontend,
}

impl WeSpeakerEmbedder {
    #[cfg(test)]
    pub(crate) fn from_oasr(path: &std::path::Path) -> Result<Self, EmbedError> {
        let model = WeSpeakerResNetModel::from_oasr(path)
            .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
        Ok(Self {
            model,
            frontend: WeSpeakerFrontend::new(),
        })
    }

    pub(crate) fn from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    ) -> Result<Self, EmbedError> {
        let model = WeSpeakerResNetModel::from_preflight(preflight)
            .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
        Ok(Self {
            model,
            frontend: WeSpeakerFrontend::new(),
        })
    }

    pub(crate) fn config(&self) -> config::ResNetConfig {
        self.model.config()
    }

    pub(crate) fn persistent_host_commitment_bytes(&self) -> Result<u64, EmbedError> {
        let model_bytes = self
            .model
            .persistent_host_commitment_bytes()
            .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
        let frontend_bytes = self
            .frontend
            .persistent_host_commitment_bytes()
            .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
        model_bytes.checked_add(frontend_bytes).ok_or_else(|| {
            EmbedError::Unavailable("wespeaker persistent host byte sum overflow".to_string())
        })
    }

    pub(crate) fn quoted_persistent_host_commitment_bytes(
        tensor_index: &crate::GgufTensorIndex,
    ) -> Result<u64, EmbedError> {
        let model_bytes =
            WeSpeakerResNetModel::quoted_persistent_host_commitment_bytes(tensor_index)
                .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
        let frontend_bytes = WeSpeakerFrontend::quoted_persistent_host_commitment_bytes()
            .map_err(|error| EmbedError::Unavailable(error.to_string()))?;
        model_bytes.checked_add(frontend_bytes).ok_or_else(|| {
            EmbedError::Unavailable("wespeaker quoted host byte sum overflow".to_string())
        })
    }

    pub(crate) fn prepare_embedding_input(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<(Vec<f32>, usize), EmbedError> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(EmbedError::UnsupportedSampleRate(sample_rate_hz));
        }
        if crate::ggml_runtime::thread_job_cancel_flag()
            .as_ref()
            .is_some_and(super::cancel_requested)
        {
            return Err(EmbedError::Canceled);
        }
        let (features, frames) = self.frontend.compute(samples);
        if frames == 0 {
            return Err(EmbedError::TooShort);
        }
        if config::post_stride_time_len(frames) < 2 {
            return Err(EmbedError::TooShort);
        }
        Ok((
            WeSpeakerFrontend::features_to_ggml_layout(&features, frames),
            frames,
        ))
    }

    pub(crate) fn shared_weights(&self) -> std::sync::Arc<Weights> {
        self.model.shared_weights()
    }
}

#[cfg(test)]
impl WeSpeakerEmbedder {
    pub(crate) fn compute_fbank(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        self.frontend.compute(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::execution_policy::ExecutionPlacement;
    use crate::diarize::contract::SpeakerEmbedding;
    use crate::ggml_runtime::GgmlCpuGraphBackend;
    use std::path::{Path, PathBuf};

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn spike_root() -> Option<PathBuf> {
        match crate::testing::external_test_fixture_path(
            "OPENASR_WESPEAKER_SPIKE_ROOT",
            "WeSpeaker parity fixture directory",
        ) {
            Ok(path) => Some(path),
            Err(skip) => {
                eprintln!("skipping: {skip}");
                None
            }
        }
    }

    fn load_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        assert_eq!(&bytes[..6], b"\x93NUMPY", "npy magic");
        let major = bytes[6];
        let header_len = if major == 1 {
            u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize
        } else {
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize
        };
        let header_start = if major == 1 { 10 } else { 12 };
        let header = std::str::from_utf8(&bytes[header_start..header_start + header_len]).unwrap();
        assert!(header.contains("'<f4'"), "expected <f4 npy, got {header}");
        let shape_start = header.find("'shape':").expect("shape key");
        let paren = header[shape_start..].find('(').unwrap() + shape_start;
        let close = header[paren..].find(')').unwrap() + paren;
        let shape: Vec<usize> = header[paren + 1..close]
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        let data_start = header_start + header_len;
        let values: Vec<f32> = bytes[data_start..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (shape, values)
    }

    fn golden_cases(root: &Path) -> Vec<String> {
        let mut names = Vec::new();
        let dir = root.join("golden");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return names;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(stem) = name.strip_suffix(".embedding.npy") {
                names.push(stem.to_string());
            }
        }
        names.sort();
        names
    }

    fn run_backend(backend: GgmlCpuGraphBackend) {
        let Some(root) = spike_root() else {
            return;
        };
        let pack = root.join("wespeaker-resnet34-f32.oasr");
        if !pack.exists() {
            eprintln!("skipping: missing {}", pack.display());
            return;
        }
        let embedder = WeSpeakerEmbedder::from_oasr(&pack).expect("load wespeaker pack");
        let placement = if backend == GgmlCpuGraphBackend::Cpu {
            ExecutionPlacement::CpuOnly
        } else {
            ExecutionPlacement::FullDevice
        };
        let mut runtime = WeSpeakerResidentRuntime::new(
            embedder.shared_weights(),
            embedder.config(),
            Some(1),
            backend,
            placement,
        )
        .expect("construct wespeaker resident runtime");
        let mut ran = 0usize;
        for name in golden_cases(&root) {
            let wav_path = root.join("golden").join(format!("{name}.wav.npy"));
            let fbank_path = root.join("golden").join(format!("{name}.fbank.npy"));
            let emb_path = root.join("golden").join(format!("{name}.embedding.npy"));
            if !wav_path.exists() || !fbank_path.exists() || !emb_path.exists() {
                continue;
            }
            let (_, wav) = load_npy_f32(&wav_path);
            let (fbank_shape, fbank_ref) = load_npy_f32(&fbank_path);
            let (_, emb_ref) = load_npy_f32(&emb_path);
            let (fbank, frames) = embedder.compute_fbank(&wav);
            assert_eq!(
                frames,
                fbank_shape.first().copied().unwrap_or(0),
                "{name} frames"
            );
            let fbank_cos = cosine(&fbank, &fbank_ref);
            let fbank_max = max_abs_diff(&fbank, &fbank_ref);
            println!("wespeaker frontend {name} cosine={fbank_cos:.8} max_abs={fbank_max:.6e}");
            assert!(
                fbank_cos >= 0.999 && fbank_max < 1e-3,
                "{name} frontend cosine={fbank_cos} max_abs={fbank_max}"
            );
            let (features, frames) = embedder
                .prepare_embedding_input(&wav, 16_000)
                .expect("prepare");
            let raw = runtime
                .forward(&features, frames, Some(1))
                .expect("wespeaker forward");
            let embedding = SpeakerEmbedding::l2_normalized(raw);
            let cos = cosine(&embedding.0, &emb_ref);
            println!(
                "wespeaker e2e {name} backend={backend:?} cosine={cos:.8} dim={}",
                embedding.dim()
            );
            assert!(
                cos >= 0.999,
                "{name} backend={backend:?} cosine {cos} below 0.999"
            );
            ran += 1;
        }
        assert!(ran > 0, "no golden cases under {}", root.display());
    }

    #[test]
    #[ignore = "host-local: needs OPENASR_WESPEAKER_SPIKE_ROOT with converted f32 pack and dump_reference goldens"]
    fn wespeaker_resnet34_matches_pytorch_on_cpu() {
        run_backend(GgmlCpuGraphBackend::Cpu);
    }

    #[test]
    #[ignore = "host-local: needs OPENASR_WESPEAKER_SPIKE_ROOT; Metal correctness not a perf gate"]
    fn wespeaker_resnet34_matches_pytorch_on_metal() {
        run_backend(GgmlCpuGraphBackend::Gpu);
    }
}
