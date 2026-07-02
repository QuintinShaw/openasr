//! The interval-only vocabulary shared between the VAD, diarization, and ASR
//! stages. Stages exchange these values and never tensors or model identity,
//! which keeps "who said what" a pure interval computation decoupled from any
//! model.

/// A half-open time span on the original-audio clock, in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeRange {
    pub start_s: f64,
    pub end_s: f64,
}

impl TimeRange {
    pub fn new(start_s: f64, end_s: f64) -> Self {
        Self { start_s, end_s }
    }

    pub fn duration_s(&self) -> f64 {
        (self.end_s - self.start_s).max(0.0)
    }

    /// Length of the overlap with `other` in seconds (0 if disjoint).
    pub fn intersection_s(&self, other: &TimeRange) -> f64 {
        (self.end_s.min(other.end_s) - self.start_s.max(other.start_s)).max(0.0)
    }

    pub fn overlaps(&self, other: &TimeRange) -> bool {
        self.intersection_s(other) > 0.0
    }
}

/// A speech region detected by the VAD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeechSegment {
    pub range: TimeRange,
}

/// An opaque, session-relative, arrival-order speaker label. It carries no
/// identity and is not stable across recordings — privacy by construction (see
/// plan §7). Any human-facing name is a separate, optional enrollment step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpeakerId(pub u32);

impl SpeakerId {
    /// Canonical `SPEAKER_NN` rendering for transcripts (matches the convention
    /// cohere already emits).
    pub fn label(&self) -> String {
        format!("SPEAKER_{:02}", self.0)
    }
}

/// A contiguous span attributed to one speaker. `overlap` marks regions the
/// segmenter flagged as overlapping speech.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerTurn {
    pub range: TimeRange,
    pub speaker: SpeakerId,
    pub overlap: bool,
}

/// An L2-normalized speaker embedding. The dimension is model-dependent and read
/// at runtime from the pack (WeSpeaker ResNet34 = 256).
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerEmbedding(pub Vec<f32>);

impl SpeakerEmbedding {
    /// Build from raw values, L2-normalizing. A zero vector stays zero.
    pub fn l2_normalized(mut values: Vec<f32>) -> Self {
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for v in &mut values {
                *v /= norm;
            }
        }
        Self(values)
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// Cosine similarity. Inputs are L2-normalized, so this is a dot product;
    /// it stays correct (just un-normalized) if they are not.
    ///
    /// Precondition: both embeddings have the same dimension (same model). A
    /// mismatch is a programming error — `zip` would otherwise silently compare
    /// only the shared prefix.
    pub fn cosine(&self, other: &SpeakerEmbedding) -> f32 {
        debug_assert_eq!(
            self.0.len(),
            other.0.len(),
            "cosine on embeddings of different dimensions"
        );
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
    }
}

/// How many speakers a diarizer should find.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum DiarizeHint {
    /// Estimate the count from the data.
    #[default]
    Auto,
    /// A known speaker count.
    NumSpeakers(u8),
    /// A clustering distance threshold (cosine dissimilarity).
    Threshold(f32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersection_and_overlap() {
        let a = TimeRange::new(0.0, 2.0);
        let b = TimeRange::new(1.5, 3.0);
        assert!((a.intersection_s(&b) - 0.5).abs() < 1e-9);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&TimeRange::new(2.0, 3.0)));
    }

    #[test]
    fn speaker_label_is_padded() {
        assert_eq!(SpeakerId(0).label(), "SPEAKER_00");
        assert_eq!(SpeakerId(12).label(), "SPEAKER_12");
    }

    #[test]
    fn embedding_normalizes_and_cosine_self_is_one() {
        let e = SpeakerEmbedding::l2_normalized(vec![3.0, 4.0]);
        assert!((e.0[0] - 0.6).abs() < 1e-6 && (e.0[1] - 0.8).abs() < 1e-6);
        assert!((e.cosine(&e) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_embedding_stays_zero() {
        let e = SpeakerEmbedding::l2_normalized(vec![0.0, 0.0]);
        assert_eq!(e.0, vec![0.0, 0.0]);
    }
}
