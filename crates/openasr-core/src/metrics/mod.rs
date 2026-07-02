//! Shared measurement utilities for the performance regression harness.
//!
//! These are first-class core utilities (WER/CER + peak-RSS), reusable beyond
//! the harness and kept dependency-free.

pub mod peak_rss;
pub mod wer;

pub use peak_rss::peak_rss_bytes;
pub use wer::{WerCounts, cer_counts, normalize_text, wer, wer_counts, word_prefix_error_rate};
