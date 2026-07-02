use std::{fmt, path::Path, str::FromStr};

use serde::Serialize;

use crate::probe_wav_duration;

pub mod suite;

pub use suite::{
    RegressionFinding, RegressionKind, SuiteBaseline, SuiteConfig, SuiteEntry, SuiteEntryMetrics,
    Tolerances, check_quant_ordering, check_vs_cpp, compare_to_baseline, quant_rank,
    render_suite_json, render_suite_markdown,
};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BenchmarkResult {
    pub input: String,
    pub model: String,
    pub backend: String,
    pub elapsed_ms: u128,
    pub audio_duration_seconds: Option<f64>,
    pub real_time_factor: Option<f64>,
    pub text_length: usize,
    pub segment_count: usize,
    pub chunk_count: Option<usize>,
    pub skipped_silent_chunks: Option<usize>,
    pub duplicate_merge_count: Option<usize>,
    pub provenance: Option<Vec<String>>,
    pub output_format: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkFormat {
    Text,
    Json,
    Markdown,
}

impl BenchmarkFormat {
    pub const ALL: &'static [&'static str] = &["text", "json", "markdown"];
}

impl fmt::Display for BenchmarkFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Text => "text",
            Self::Json => "json",
            Self::Markdown => "markdown",
        })
    }
}

impl FromStr for BenchmarkFormat {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            "markdown" => Ok(Self::Markdown),
            other => Err(format!(
                "Unsupported benchmark format '{other}'. Use one of: {}.",
                Self::ALL.join(", ")
            )),
        }
    }
}

pub fn render_benchmark(
    result: &BenchmarkResult,
    format: BenchmarkFormat,
) -> Result<String, serde_json::Error> {
    match format {
        BenchmarkFormat::Text => Ok(render_text(result)),
        BenchmarkFormat::Json => serde_json::to_string_pretty(result),
        BenchmarkFormat::Markdown => Ok(render_markdown(result)),
    }
}

pub fn probe_audio_duration_seconds(path: impl AsRef<Path>) -> Option<f64> {
    probe_wav_duration(path)
}

fn render_text(result: &BenchmarkResult) -> String {
    let base = format!(
        "OpenASR benchmark\n\nInput: {}\nModel: {}\nBackend: {}\nElapsed: {} ms\nAudio duration: {}\nReal-time factor: {}\nText length: {} chars\nSegments: {}\nOutput format: {}\n",
        result.input,
        result.model,
        result.backend,
        result.elapsed_ms,
        format_duration(result.audio_duration_seconds),
        format_real_time_factor(result.real_time_factor),
        result.text_length,
        result.segment_count,
        result.output_format
    );
    format!(
        "{base}Chunks: {}\nSkipped silent chunks: {}\nDuplicate merges: {}\nProvenance: {}\n",
        format_optional_usize(result.chunk_count),
        format_optional_usize(result.skipped_silent_chunks),
        format_optional_usize(result.duplicate_merge_count),
        format_provenance(result.provenance.as_deref()),
    )
}

fn render_markdown(result: &BenchmarkResult) -> String {
    let base = format!(
        "# OpenASR Benchmark\n\n| Field | Value |\n| --- | --- |\n| Input | {} |\n| Model | {} |\n| Backend | {} |\n| Elapsed | {} ms |\n| Audio duration | {} |\n| Real-time factor | {} |\n| Text length | {} chars |\n| Segments | {} |\n| Output format | {} |\n",
        result.input,
        result.model,
        result.backend,
        result.elapsed_ms,
        format_duration(result.audio_duration_seconds),
        format_real_time_factor(result.real_time_factor),
        result.text_length,
        result.segment_count,
        result.output_format
    );
    format!(
        "{base}| Chunks | {} |\n| Skipped silent chunks | {} |\n| Duplicate merges | {} |\n| Provenance | {} |\n",
        format_optional_usize(result.chunk_count),
        format_optional_usize(result.skipped_silent_chunks),
        format_optional_usize(result.duplicate_merge_count),
        format_provenance(result.provenance.as_deref()),
    )
}

fn format_duration(value: Option<f64>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| format!("{value:.2} s"))
}

fn format_real_time_factor(value: Option<f64>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| format!("{value:.3}x"))
}

fn format_optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| value.to_string())
}

fn format_provenance(value: Option<&[String]>) -> String {
    value.map_or_else(
        || "unknown".to_string(),
        |items| {
            if items.is_empty() {
                "none".to_string()
            } else {
                items.join(", ")
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::*;

    #[test]
    fn renders_text_without_transcript_content() {
        let result = sample_result();

        let rendered = render_benchmark(&result, BenchmarkFormat::Text).unwrap();

        assert!(rendered.contains("OpenASR benchmark"));
        assert!(rendered.contains("Input: fixtures/jfk.wav"));
        assert!(rendered.contains("Audio duration: 2.50 s"));
        assert!(rendered.contains("Real-time factor: 0.005x"));
        assert!(rendered.contains("Chunks: unknown"));
        assert!(!rendered.contains("hello transcript"));
    }

    #[test]
    fn renders_json_with_nullable_duration_fields() {
        let mut result = sample_result();
        result.audio_duration_seconds = None;
        result.real_time_factor = None;

        let rendered = render_benchmark(&result, BenchmarkFormat::Json).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();

        assert_eq!(parsed["audio_duration_seconds"], serde_json::Value::Null);
        assert_eq!(parsed["real_time_factor"], serde_json::Value::Null);
        assert_eq!(parsed["chunk_count"], serde_json::Value::Null);
        assert_eq!(parsed["provenance"], serde_json::Value::Null);
    }

    #[test]
    fn renders_markdown_table() {
        let rendered = render_benchmark(&sample_result(), BenchmarkFormat::Markdown).unwrap();

        assert!(rendered.contains("# OpenASR Benchmark"));
        assert!(rendered.contains("| Field | Value |"));
        assert!(rendered.contains("| Model | whisper-tiny |"));
    }

    #[test]
    fn wav_duration_probe_reads_pcm_duration() {
        let temp = tempfile::tempdir().unwrap();
        let wav = temp.path().join("tone.wav");
        write_test_wav(&wav, 16_000, 1, 16, 16_000);

        let duration = probe_audio_duration_seconds(&wav).unwrap();

        assert!((duration - 1.0).abs() < 0.001);
    }

    #[test]
    fn wav_duration_probe_returns_none_for_non_wav() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("sample.mp3");
        fs::write(&path, "not a wav").unwrap();

        assert_eq!(probe_audio_duration_seconds(path), None);
    }

    fn sample_result() -> BenchmarkResult {
        BenchmarkResult {
            input: "fixtures/jfk.wav".to_string(),
            model: "whisper-tiny".to_string(),
            backend: "mock".to_string(),
            elapsed_ms: 12,
            audio_duration_seconds: Some(2.5),
            real_time_factor: Some(0.0048),
            text_length: 42,
            segment_count: 1,
            chunk_count: None,
            skipped_silent_chunks: None,
            duplicate_merge_count: None,
            provenance: None,
            output_format: "text".to_string(),
        }
    }

    fn write_test_wav(
        path: &Path,
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
        frames: u32,
    ) {
        let data_size = frames * channels as u32 * (bits_per_sample as u32 / 8);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_size).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * channels as u32 * (bits_per_sample as u32 / 8);
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        let block_align = channels * (bits_per_sample / 8);
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&bits_per_sample.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_size.to_le_bytes());
        bytes.resize(bytes.len() + data_size as usize, 0);

        fs::write(path, bytes).unwrap();
    }
}
