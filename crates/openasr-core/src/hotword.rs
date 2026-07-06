use std::collections::BTreeMap;

use thiserror::Error;

pub const MAX_PHRASE_BIAS_ENTRIES: usize = 128;
pub const MAX_PHRASE_BIAS_PHRASE_CHARS: usize = 128;
pub const MAX_PHRASE_BIAS_TOTAL_CHARS: usize = 4096;
pub const MAX_PHRASE_BIAS_BOOST: f32 = 20.0;

/// Default per-phrase boost applied when a caller supplies hotword phrases
/// without an explicit boost. Single source of truth for every surface (CLI,
/// batch HTTP, realtime WS, saved config); each surface re-exports this rather
/// than redefining the literal.
///
/// The configured value is the per-step BASE strength, not the only strength:
/// the decoder applies exactly this value to a phrase's entry token, and scales
/// it by the matched-prefix depth (`boost * (matched_len + 1)`, clamped to
/// [`MAX_PHRASE_BIAS_BOOST`]) for mid-phrase continuation tokens — so default
/// 5.0 means 5 at entry, then up to 10/15/20 deeper into the phrase. Escalated
/// positive boosts are additionally gated on acoustic plausibility; see
/// `apply_phrase_bias_to_logits` in `models::phrase_bias_decode`.
///
/// CTC families (sensevoice / parakeet-ctc / wav2vec2-ctc) decode hotwords
/// through a prefix-beam search with a context graph instead
/// (`models::ctc_prefix_beam`); there the value is the per-matched-token context
/// climb. For CTC, boosts up to ~10 are the recommended range: the measured CJK
/// homophone correction lands cleanly at the default, while boosts near the cap
/// (20.0) exhibit a known drop pattern -- a confusable character ADJACENT to the
/// hotword (e.g. the `叫` right before `刁天宸`) can be squeezed out of the
/// transcript, because hypotheses that skip it into the hotword hold a transient
/// context lead while the narrow beam prunes the keep-it-then-match alternative.
pub const DEFAULT_PHRASE_BIAS_BOOST: f32 = 5.0;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum PhraseBiasError {
    #[error("phrase bias phrase must be non-empty after trimming")]
    EmptyPhrase,
    #[error("phrase bias phrase has {char_count} chars, max is {max_chars}")]
    PhraseTooLong { char_count: usize, max_chars: usize },
    #[error(
        "phrase bias boost must be finite, non-zero, and within [-{max_boost}, {max_boost}] (negative suppresses the phrase), got {value}"
    )]
    InvalidBoost { value: f32, max_boost: f32 },
    #[error("phrase bias config has {count} entries, max is {max_entries}")]
    TooManyEntries { count: usize, max_entries: usize },
    #[error("phrase bias config uses {total_chars} total phrase chars, max is {max_chars}")]
    TotalPhraseBudgetExceeded {
        total_chars: usize,
        max_chars: usize,
    },
    #[error(
        "phrase bias config contains duplicate normalized phrases at entries {first_index} and {duplicate_index}"
    )]
    DuplicatePhrase {
        first_index: usize,
        duplicate_index: usize,
    },
}

/// Per-phrase bias strength. A POSITIVE value favors the phrase (boost); a
/// NEGATIVE value suppresses it (anti-context — discourage a confusable term).
/// Magnitudes are bounded by [`MAX_PHRASE_BIAS_BOOST`]; zero is rejected.
///
/// This is the per-step BASE strength: the decoder applies it as-is to the
/// phrase's entry token and scales it with matched-prefix depth for
/// continuation tokens (both signs), clamped to [`MAX_PHRASE_BIAS_BOOST`].
#[derive(Debug, Clone, Copy)]
pub struct PhraseBiasBoost(f32);

impl PhraseBiasBoost {
    pub fn new(value: f32) -> Result<Self, PhraseBiasError> {
        if !value.is_finite() || value == 0.0 || value.abs() > MAX_PHRASE_BIAS_BOOST {
            return Err(PhraseBiasError::InvalidBoost {
                value,
                max_boost: MAX_PHRASE_BIAS_BOOST,
            });
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl PartialEq for PhraseBiasBoost {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for PhraseBiasBoost {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhraseBiasEntry {
    phrase: String,
    canonical_phrase: String,
    boost: PhraseBiasBoost,
}

impl PhraseBiasEntry {
    pub fn new(phrase: impl Into<String>, boost: f32) -> Result<Self, PhraseBiasError> {
        let phrase = normalize_phrase(&phrase.into())?;
        let char_count = phrase.chars().count();
        if char_count > MAX_PHRASE_BIAS_PHRASE_CHARS {
            return Err(PhraseBiasError::PhraseTooLong {
                char_count,
                max_chars: MAX_PHRASE_BIAS_PHRASE_CHARS,
            });
        }
        Ok(Self {
            canonical_phrase: canonicalize_phrase(&phrase),
            phrase,
            boost: PhraseBiasBoost::new(boost)?,
        })
    }

    pub fn phrase(&self) -> &str {
        &self.phrase
    }

    pub fn boost(&self) -> f32 {
        self.boost.get()
    }

    fn canonical_phrase(&self) -> &str {
        &self.canonical_phrase
    }

    fn char_count(&self) -> usize {
        self.phrase.chars().count()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PhraseBiasConfig {
    entries: Vec<PhraseBiasEntry>,
}

impl PhraseBiasConfig {
    pub fn new(entries: Vec<PhraseBiasEntry>) -> Result<Self, PhraseBiasError> {
        validate_entries(&entries)?;
        Ok(Self { entries })
    }

    pub fn from_phrases<I, P>(entries: I) -> Result<Self, PhraseBiasError>
    where
        I: IntoIterator<Item = (P, f32)>,
        P: Into<String>,
    {
        let entries = entries
            .into_iter()
            .map(|(phrase, boost)| PhraseBiasEntry::new(phrase, boost))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(entries)
    }

    /// Build a config from `phrases` that all share `boost`, defaulting to
    /// [`DEFAULT_PHRASE_BIAS_BOOST`] when `boost` is `None`. The single place
    /// every surface (CLI, batch HTTP, saved config) turns a phrase list plus an
    /// optional boost into a config, so the default-boost policy cannot drift.
    pub fn from_phrases_with_default_boost<I, P>(
        phrases: I,
        boost: Option<f32>,
    ) -> Result<Self, PhraseBiasError>
    where
        I: IntoIterator<Item = P>,
        P: Into<String>,
    {
        let boost = boost.unwrap_or(DEFAULT_PHRASE_BIAS_BOOST);
        Self::from_phrases(phrases.into_iter().map(|phrase| (phrase, boost)))
    }

    pub fn entries(&self) -> &[PhraseBiasEntry] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn validate_entries(entries: &[PhraseBiasEntry]) -> Result<(), PhraseBiasError> {
    if entries.len() > MAX_PHRASE_BIAS_ENTRIES {
        return Err(PhraseBiasError::TooManyEntries {
            count: entries.len(),
            max_entries: MAX_PHRASE_BIAS_ENTRIES,
        });
    }

    let mut total_chars = 0usize;
    let mut first_index_by_canonical = BTreeMap::<&str, usize>::new();
    for (index, entry) in entries.iter().enumerate() {
        total_chars = total_chars.saturating_add(entry.char_count());
        if total_chars > MAX_PHRASE_BIAS_TOTAL_CHARS {
            return Err(PhraseBiasError::TotalPhraseBudgetExceeded {
                total_chars,
                max_chars: MAX_PHRASE_BIAS_TOTAL_CHARS,
            });
        }
        if let Some(first_index) = first_index_by_canonical.insert(entry.canonical_phrase(), index)
        {
            return Err(PhraseBiasError::DuplicatePhrase {
                first_index,
                duplicate_index: index,
            });
        }
    }

    Ok(())
}

fn normalize_phrase(value: &str) -> Result<String, PhraseBiasError> {
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return Err(PhraseBiasError::EmptyPhrase);
    }
    Ok(normalized)
}

fn canonicalize_phrase(value: &str) -> String {
    value.chars().flat_map(char::to_lowercase).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_constructor_trims_and_collapses_whitespace() {
        let entry = PhraseBiasEntry::new("  OpenASR\tcore\ncontract  ", 4.5).unwrap();
        assert_eq!(entry.phrase(), "OpenASR core contract");
        assert_eq!(entry.boost(), 4.5);
    }

    #[test]
    fn entry_constructor_rejects_empty_phrase_without_echoing_input() {
        let error = PhraseBiasEntry::new(" \t\n ", 1.0).unwrap_err();
        assert_eq!(error, PhraseBiasError::EmptyPhrase);
        assert!(!error.to_string().contains("\\t"));
    }

    #[test]
    fn entry_constructor_rejects_overlong_phrase() {
        let phrase = "x".repeat(MAX_PHRASE_BIAS_PHRASE_CHARS + 1);
        let error = PhraseBiasEntry::new(phrase, 1.0).unwrap_err();
        assert!(matches!(
            error,
            PhraseBiasError::PhraseTooLong {
                char_count,
                max_chars: MAX_PHRASE_BIAS_PHRASE_CHARS
            } if char_count == MAX_PHRASE_BIAS_PHRASE_CHARS + 1
        ));
    }

    #[test]
    fn entry_constructor_rejects_zero_non_finite_or_out_of_magnitude_boost() {
        for boost in [0.0, 20.1, -20.1, f32::INFINITY, f32::NAN] {
            assert!(matches!(
                PhraseBiasEntry::new("openasr", boost),
                Err(PhraseBiasError::InvalidBoost { .. })
            ));
        }
        // Positive favors, negative suppresses (anti-context); magnitude boundary ok.
        for boost in [1.0, 20.0, -1.0, -20.0] {
            assert!(PhraseBiasEntry::new("openasr", boost).is_ok());
        }
    }

    #[test]
    fn config_rejects_too_many_entries() {
        let entries = (0..=MAX_PHRASE_BIAS_ENTRIES)
            .map(|index| PhraseBiasEntry::new(format!("term-{index}"), 1.0).unwrap())
            .collect::<Vec<_>>();
        let error = PhraseBiasConfig::new(entries).unwrap_err();
        assert!(matches!(
            error,
            PhraseBiasError::TooManyEntries {
                count,
                max_entries: MAX_PHRASE_BIAS_ENTRIES
            } if count == MAX_PHRASE_BIAS_ENTRIES + 1
        ));
    }

    #[test]
    fn config_rejects_total_phrase_budget_overflow() {
        let entries = (0..40)
            .map(|index| {
                let mut phrase = "x".repeat(MAX_PHRASE_BIAS_PHRASE_CHARS - 3);
                phrase.push_str(&format!("{index:03}"));
                PhraseBiasEntry::new(phrase, 1.0).unwrap()
            })
            .collect::<Vec<_>>();
        let error = PhraseBiasConfig::new(entries).unwrap_err();
        assert!(matches!(
            error,
            PhraseBiasError::TotalPhraseBudgetExceeded { .. }
        ));
    }

    #[test]
    fn config_rejects_duplicate_normalized_phrases_without_echoing_phrase() {
        let error =
            PhraseBiasConfig::from_phrases([(" OpenASR   Core ", 1.0), ("openasr core", 2.0)])
                .unwrap_err();
        assert_eq!(
            error,
            PhraseBiasError::DuplicatePhrase {
                first_index: 0,
                duplicate_index: 1
            }
        );
        assert!(!error.to_string().contains("OpenASR Core"));
    }

    #[test]
    fn empty_config_is_valid_and_reports_empty() {
        let config = PhraseBiasConfig::new(Vec::new()).unwrap();
        assert!(config.is_empty());
        assert!(config.entries().is_empty());
    }
}
