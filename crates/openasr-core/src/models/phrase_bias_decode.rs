use std::collections::BTreeMap;

use thiserror::Error;

use crate::PhraseBiasConfig;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TokenPhraseBias {
    /// Alternative token-id sequences for one phrase. A phrase can tokenize more
    /// than one way depending on where it lands in running text — byte-level BPE
    /// (whisper/qwen) folds a preceding space into a word's first token, so
    /// "openasr" mid-sentence is a different token sequence than at an utterance
    /// boundary. Any variant matching the decoded suffix activates the boost, so
    /// the hotword is biased regardless of which surface form is about to emit.
    variants: Vec<Vec<u32>>,
    boost: f32,
}

impl TokenPhraseBias {
    pub(crate) fn new(variants: Vec<Vec<u32>>, boost: f32) -> Option<Self> {
        let variants: Vec<Vec<u32>> = variants
            .into_iter()
            .filter(|variant| !variant.is_empty())
            .collect();
        if variants.is_empty() || !boost.is_finite() || boost == 0.0 {
            return None;
        }
        Some(Self { variants, boost })
    }

    pub(crate) fn variants(&self) -> &[Vec<u32>] {
        &self.variants
    }

    pub(crate) fn boost(&self) -> f32 {
        self.boost
    }
}

pub(crate) trait PhraseBiasTokenEncoder {
    /// The primary (standalone) tokenization of a phrase, or `Ok(None)` if this
    /// tokenizer cannot encode phrase-bias entries at all.
    fn encode_phrase_bias_tokens(&self, phrase: &str) -> Result<Option<Vec<u32>>, String>;

    /// All token-id variants to match the phrase against during decode. Defaults
    /// to just the primary tokenization; tokenizers whose running-text form
    /// differs from the standalone one (byte-level BPE leading space) override
    /// this to also return the alternative form, so a hotword is biased whether
    /// it lands at an utterance boundary or mid-sentence.
    fn encode_phrase_bias_variants(&self, phrase: &str) -> Result<Option<Vec<Vec<u32>>>, String> {
        Ok(self
            .encode_phrase_bias_tokens(phrase)?
            .map(|token_ids| vec![token_ids]))
    }
}

/// Byte-level BPE phrase-bias variants: the standalone tokenization plus the
/// leading-space form. Byte-level BPE (whisper/qwen) folds a preceding space
/// into a word's first token, so a phrase mid-sentence tokenizes differently
/// than at an utterance boundary; emitting both forms lets the bias fire in
/// either position. De-duplicated when the two forms coincide.
pub(crate) fn encode_bpe_phrase_bias_variants<F, E>(
    phrase: &str,
    mut encode: F,
) -> Result<Vec<Vec<u32>>, String>
where
    F: FnMut(&str) -> Result<Vec<u32>, E>,
    E: std::fmt::Display,
{
    let standalone = encode(phrase).map_err(|error| error.to_string())?;
    let leading_space = encode(&format!(" {phrase}")).map_err(|error| error.to_string())?;
    let mut variants = vec![standalone];
    if !variants.contains(&leading_space) {
        variants.push(leading_space);
    }
    Ok(variants)
}

/// Typed failure of [`build_token_phrase_biases`]. Lets callers classify the
/// failure by variant instead of sniffing a stringified message: a tokenizer
/// that cannot encode a phrase at all is `Unsupported`, anything else (an
/// encoder error, or an empty token sequence) is `TokenizationFailed`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum PhraseBiasBuildError {
    #[error("tokenizer cannot encode phrase-bias entries")]
    Unsupported,
    #[error("{reason}")]
    TokenizationFailed { reason: String },
}

pub(crate) fn build_token_phrase_biases<E: PhraseBiasTokenEncoder + ?Sized>(
    phrase_bias: Option<&PhraseBiasConfig>,
    encoder: &E,
) -> Result<Vec<TokenPhraseBias>, PhraseBiasBuildError> {
    let Some(phrase_bias) = phrase_bias.filter(|config| !config.is_empty()) else {
        return Ok(Vec::new());
    };

    let mut biases = Vec::with_capacity(phrase_bias.entries().len());
    for entry in phrase_bias.entries() {
        let variants = encoder
            .encode_phrase_bias_variants(entry.phrase())
            .map_err(|reason| PhraseBiasBuildError::TokenizationFailed { reason })?
            .ok_or(PhraseBiasBuildError::Unsupported)?;
        let Some(bias) = TokenPhraseBias::new(variants, entry.boost()) else {
            return Err(PhraseBiasBuildError::TokenizationFailed {
                reason: "tokenizer produced an empty phrase-bias token sequence".to_string(),
            });
        };
        biases.push(bias);
    }
    Ok(biases)
}

/// Sequence-aware phrase biasing shared by the seq2seq and CTC decode paths.
///
/// For each phrase (and each of its tokenization variants), find the next token
/// that continues the longest suffix of `decoded_so_far` matching a prefix of
/// the phrase, and adjust that token by the phrase's boost — positive to favor
/// the phrase, negative to suppress it (anti-context). The adjustment is gated
/// on phrase progress (only a phrase's next expected token is nudged, never
/// every phrase token on every step) and each token takes the single strongest
/// opinion about it (largest magnitude), never the sum — bounding the
/// spurious-insertion / WER risk of over-biasing.
///
/// The applied boost scales with match depth: `boost * (matched_prefix_len + 1)`,
/// clamped to [`crate::MAX_PHRASE_BIAS_BOOST`]. With no decoded prefix the entry
/// token gets exactly the configured boost (so the always-active entry nudge
/// stays small), but once the decode has already emitted part of the phrase the
/// evidence that the phrase is being spoken is strong, and the continuation gets
/// proportionally stronger help. This mirrors accumulated path scores in
/// FST-based contextual biasing (sherpa-style) collapsed onto a greedy decode,
/// and is what makes a real CJK homophone correction (`刁天成` -> `刁天宸`,
/// observed pre-bias logit gap 7-10 at the final hanzi) win at the default
/// boost without inflating the entry-token nudge.
///
/// Two explicit guard rails on the scaling:
///
/// - POSITIVE escalation requires acoustic plausibility. A scaled-up
///   (depth-1+) positive boost only applies when the candidate's UNBIASED logit is
///   within [`CONTINUATION_PLAUSIBILITY_MARGIN`] of the unbiased maximum;
///   otherwise the flat configured boost applies, exactly as before depth
///   scaling existed. Without this gate, a long (4+ token) hotword whose entry
///   nudge tipped a near-tie would have its continuations railroaded at
///   +2x/+3x/+4x against arbitrarily strong contrary acoustics (up to the
///   clamp), turning one borderline entry mistake into a whole hallucinated
///   phrase. The gate breaks that chain while still letting the real homophone
///   case (gap 7-10, well inside the margin) win.
/// - NEGATIVE boosts (anti-context) DO scale with depth, deliberately and
///   ungated: if the decode got into a suppressed phrase despite the entry
///   suppression, pushing harder against its continuation steers the decode
///   off the phrase before it completes. Suppression only ever pushes a
///   specific expected token DOWN, so it cannot insert garbage; its magnitude
///   stays clamped to [`crate::MAX_PHRASE_BIAS_BOOST`].
pub(crate) fn apply_phrase_bias_to_logits(
    logits: &mut [f32],
    decoded_so_far: &[u32],
    phrase_biases: &[TokenPhraseBias],
) {
    if phrase_biases.is_empty() {
        return;
    }

    // Resolve one adjustment per candidate token, taking the STRONGEST single
    // opinion (largest magnitude) of any phrase/variant whose next expected token
    // it is — the most-favoring positive or the most-suppressing negative, never
    // the sum. Phrases colliding on a shared continuation token (or one phrase's
    // multiple tokenization variants pointing at it) must not stack into a
    // super-additive value that buries the acoustic evidence; the perturbation on
    // any token stays within a single phrase's envelope. Non-candidate tokens are
    // untouched, so the empty-config default path stays byte-identical.
    let max_unbiased_logit = max_finite_logit(logits);
    let mut boost_by_token: BTreeMap<u32, f32> = BTreeMap::new();
    for bias in phrase_biases {
        for variant in bias.variants() {
            if let Some((next_token, matched_prefix_len)) =
                next_phrase_continuation_token(variant, decoded_so_far)
            {
                let candidate_logit = logits
                    .get(usize::try_from(next_token).unwrap_or(usize::MAX))
                    .copied()
                    .unwrap_or(f32::NEG_INFINITY);
                let applied = depth_scaled_boost(
                    bias.boost(),
                    matched_prefix_len,
                    candidate_logit,
                    max_unbiased_logit,
                );
                let entry = boost_by_token.entry(next_token).or_insert(0.0);
                if applied.abs() > entry.abs() {
                    *entry = applied;
                }
            }
        }
    }

    for (token_id, boost) in boost_by_token {
        add_token_boost(logits, token_id, boost);
    }
}

/// How far (in logits) a continuation candidate may trail the unbiased argmax
/// and still receive the ESCALATED (depth-scaled) positive boost; beyond it,
/// only the flat configured boost applies. Sized from the measured real
/// homophone correction (pre-bias gap 7.0 q4_k / 10.3 q8_0 at the final hanzi
/// of 刁天宸) with headroom, and below the 15-20 escalated default boosts whose
/// railroading it exists to stop.
const CONTINUATION_PLAUSIBILITY_MARGIN: f32 = 12.0;

/// Boost scaled by how much of the phrase is already decoded (see
/// [`apply_phrase_bias_to_logits`]), clamped to the global per-token envelope.
/// Entry tokens (no matched prefix) keep exactly the configured boost. A
/// positive boost only escalates when the candidate is acoustically plausible
/// (within [`CONTINUATION_PLAUSIBILITY_MARGIN`] of the unbiased max); a
/// negative boost escalates unconditionally (suppression cannot insert).
fn depth_scaled_boost(
    boost: f32,
    matched_prefix_len: usize,
    candidate_logit: f32,
    max_unbiased_logit: f32,
) -> f32 {
    if matched_prefix_len == 0 {
        return boost;
    }
    let plausible = candidate_logit >= max_unbiased_logit - CONTINUATION_PLAUSIBILITY_MARGIN;
    if boost > 0.0 && !plausible {
        return boost;
    }
    let scale = (matched_prefix_len + 1) as f32;
    (boost * scale).clamp(-crate::MAX_PHRASE_BIAS_BOOST, crate::MAX_PHRASE_BIAS_BOOST)
}

fn max_finite_logit(logits: &[f32]) -> f32 {
    logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max)
}

/// The phrase's next expected token given the decoded suffix, with the length
/// of the matched prefix (0 = entry token, nothing matched yet).
fn next_phrase_continuation_token(
    phrase_tokens: &[u32],
    decoded_so_far: &[u32],
) -> Option<(u32, usize)> {
    if phrase_tokens.is_empty() {
        return None;
    }

    // If the full phrase was just emitted, it is satisfied for now: do not
    // immediately re-nudge its first token, which would push the model to repeat
    // a one-shot hotword. It can re-activate later, once the decoded suffix no
    // longer ends with the complete phrase.
    if decoded_so_far.len() >= phrase_tokens.len()
        && &decoded_so_far[decoded_so_far.len() - phrase_tokens.len()..] == phrase_tokens
    {
        return None;
    }

    let max_prefix_len = phrase_tokens
        .len()
        .saturating_sub(1)
        .min(decoded_so_far.len());
    for prefix_len in (0..=max_prefix_len).rev() {
        let decoded_suffix = &decoded_so_far[decoded_so_far.len() - prefix_len..];
        if decoded_suffix == &phrase_tokens[..prefix_len] {
            return phrase_tokens
                .get(prefix_len)
                .copied()
                .map(|token| (token, prefix_len));
        }
    }
    None
}

fn add_token_boost(logits: &mut [f32], token_id: u32, boost: f32) {
    let Some(index) = usize::try_from(token_id).ok() else {
        return;
    };
    let Some(logit) = logits.get_mut(index) else {
        return;
    };
    if logit.is_finite() {
        *logit += boost;
    }
}

impl PhraseBiasTokenEncoder for () {
    fn encode_phrase_bias_tokens(&self, _phrase: &str) -> Result<Option<Vec<u32>>, String> {
        Ok(None)
    }
}

pub(crate) fn encode_sentencepiece_phrase_bias_tokens(
    phrase: &str,
    token_to_id: &BTreeMap<String, u32>,
    tokenizer_name: &str,
) -> Result<Option<Vec<u32>>, String> {
    let words = phrase.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() {
        return Ok(None);
    }

    let mut token_ids = Vec::new();
    for word in words {
        let piece = format!("\u{2581}{word}");
        match encode_exact_subpieces(&piece, token_to_id) {
            Some(mut ids) => token_ids.append(&mut ids),
            None if token_ids.is_empty() => {
                let Some(mut ids) = encode_exact_subpieces(word, token_to_id) else {
                    return Err(format!(
                        "{tokenizer_name} tokenizer cannot encode phrase-bias word"
                    ));
                };
                token_ids.append(&mut ids);
            }
            None => {
                return Err(format!(
                    "{tokenizer_name} tokenizer cannot encode phrase-bias word"
                ));
            }
        }
    }

    Ok((!token_ids.is_empty()).then_some(token_ids))
}

pub(crate) fn encode_character_ctc_phrase_bias_tokens(
    phrase: &str,
    token_to_id: &BTreeMap<String, u32>,
    word_delimiter: &str,
    uppercase: bool,
    tokenizer_name: &str,
) -> Result<Option<Vec<u32>>, String> {
    let normalized = phrase.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return Ok(None);
    }

    let mut ids = Vec::new();
    for ch in normalized.chars() {
        if ch == ' ' {
            ids.push(*token_to_id.get(word_delimiter).ok_or_else(|| {
                format!("{tokenizer_name} tokenizer is missing word delimiter token")
            })?);
            continue;
        }
        let token = if uppercase {
            ch.to_uppercase().collect::<String>()
        } else {
            ch.to_string()
        };
        ids.push(*token_to_id.get(&token).ok_or_else(|| {
            format!("{tokenizer_name} tokenizer cannot encode phrase-bias character")
        })?);
    }
    Ok((!ids.is_empty()).then_some(ids))
}

fn encode_exact_subpieces(piece: &str, token_to_id: &BTreeMap<String, u32>) -> Option<Vec<u32>> {
    let mut ids = Vec::new();
    let mut cursor = 0usize;
    while cursor < piece.len() {
        let mut matched = None;
        for end in piece
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(piece.len()))
            .filter(|end| *end > cursor)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            let candidate = &piece[cursor..end];
            if let Some(token_id) = token_to_id.get(candidate).copied() {
                matched = Some((token_id, end));
                break;
            }
        }
        let (token_id, next_cursor) = matched?;
        ids.push(token_id);
        cursor = next_cursor;
    }
    Some(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PhraseBiasConfig;

    struct SyntheticEncoder;

    impl PhraseBiasTokenEncoder for SyntheticEncoder {
        fn encode_phrase_bias_tokens(&self, phrase: &str) -> Result<Option<Vec<u32>>, String> {
            Ok(match phrase {
                "open asr" => Some(vec![1, 2]),
                "core" => Some(vec![3]),
                _ => None,
            })
        }
    }

    #[test]
    fn biases_first_and_continuation_tokens() {
        let config = PhraseBiasConfig::from_phrases([("open asr", 4.0)]).unwrap();
        let biases = build_token_phrase_biases(Some(&config), &SyntheticEncoder).unwrap();

        let mut first = vec![0.0; 8];
        apply_phrase_bias_to_logits(&mut first, &[], &biases);
        assert_eq!(first[1], 4.0);

        // Depth-1 continuation: boost scales with the matched prefix (4.0 * 2).
        let mut next = vec![0.0; 8];
        apply_phrase_bias_to_logits(&mut next, &[1], &biases);
        assert_eq!(next[2], 8.0);
    }

    #[test]
    fn gates_on_decoded_progress_with_per_phrase_boost() {
        let config = PhraseBiasConfig::from_phrases([("open asr", 2.0), ("core", 5.0)]).unwrap();
        let biases = build_token_phrase_biases(Some(&config), &SyntheticEncoder).unwrap();

        // Nothing decoded yet: only each phrase's FIRST token is boosted, by its
        // own boost (not a shared global maximum).
        let mut start = vec![0.0; 8];
        apply_phrase_bias_to_logits(&mut start, &[], &biases);
        assert_eq!(start[1], 2.0); // "open asr" -> token 1
        assert_eq!(start[3], 5.0); // "core" -> token 3
        assert_eq!(start[2], 0.0); // continuation token of "open asr" not yet boosted

        // After token 1 is emitted, the continuation token 2 is boosted (depth-1
        // scaled: 2.0 * 2) and the already-emitted token 1 is not.
        let mut mid = vec![0.0; 8];
        apply_phrase_bias_to_logits(&mut mid, &[1], &biases);
        assert_eq!(mid[2], 4.0);
        assert_eq!(mid[1], 0.0);
    }

    fn vocab(pairs: &[(&str, u32)]) -> BTreeMap<String, u32> {
        pairs
            .iter()
            .map(|(token, id)| ((*token).to_string(), *id))
            .collect()
    }

    const SP: &str = "\u{2581}"; // SentencePiece word-start marker U+2581

    #[test]
    fn sentencepiece_encoder_prefixes_each_word_and_concatenates() {
        let token_to_id = vocab(&[(&format!("{SP}open"), 10), (&format!("{SP}asr"), 11)]);
        let ids = encode_sentencepiece_phrase_bias_tokens("open asr", &token_to_id, "test")
            .unwrap()
            .unwrap();
        assert_eq!(ids, vec![10, 11]);
    }

    #[test]
    fn sentencepiece_encoder_greedily_splits_into_longest_subpieces() {
        // "▁unknown" is not a single token; greedy longest-match yields ▁un + known.
        let token_to_id = vocab(&[(&format!("{SP}un"), 13), ("known", 14)]);
        let ids = encode_sentencepiece_phrase_bias_tokens("unknown", &token_to_id, "test")
            .unwrap()
            .unwrap();
        assert_eq!(ids, vec![13, 14]);
    }

    #[test]
    fn sentencepiece_encoder_falls_back_to_bare_form_only_for_the_first_word() {
        // ▁core absent but bare "core" present: allowed because it is the first word.
        let token_to_id = vocab(&[("core", 30)]);
        let ids = encode_sentencepiece_phrase_bias_tokens("core", &token_to_id, "test")
            .unwrap()
            .unwrap();
        assert_eq!(ids, vec![30]);
    }

    #[test]
    fn sentencepiece_encoder_errors_on_unencodable_non_first_word() {
        // First word encodes; the second has neither ▁zzz nor a subpiece decomposition.
        let token_to_id = vocab(&[(&format!("{SP}open"), 10)]);
        let error =
            encode_sentencepiece_phrase_bias_tokens("open zzz", &token_to_id, "test").unwrap_err();
        assert!(error.contains("cannot encode"));
    }

    #[test]
    fn character_ctc_encoder_maps_chars_delimiter_and_uppercases() {
        let token_to_id = vocab(&[("H", 1), ("I", 2), ("|", 3)]);
        let ids = encode_character_ctc_phrase_bias_tokens("hi", &token_to_id, "|", true, "test")
            .unwrap()
            .unwrap();
        assert_eq!(ids, vec![1, 2]);

        let spaced =
            encode_character_ctc_phrase_bias_tokens("h i", &token_to_id, "|", true, "test")
                .unwrap()
                .unwrap();
        assert_eq!(spaced, vec![1, 3, 2]);
    }

    #[test]
    fn character_ctc_encoder_errors_on_missing_delimiter_or_char() {
        let no_delim = vocab(&[("H", 1), ("I", 2)]);
        assert!(
            encode_character_ctc_phrase_bias_tokens("h i", &no_delim, "|", true, "test")
                .unwrap_err()
                .contains("word delimiter")
        );

        let missing_char = vocab(&[("H", 1), ("|", 3)]);
        assert!(
            encode_character_ctc_phrase_bias_tokens("hx", &missing_char, "|", true, "test")
                .unwrap_err()
                .contains("cannot encode")
        );
    }

    #[test]
    fn add_token_boost_ignores_out_of_range_token_ids() {
        // A phrase whose continuation token id is past the vocab must not panic or
        // mutate any logit (defense-in-depth for a tokenizer/vocab mismatch).
        let bias = TokenPhraseBias::new(vec![vec![99]], 5.0).unwrap();
        let mut logits = vec![0.0_f32; 8];
        apply_phrase_bias_to_logits(&mut logits, &[], std::slice::from_ref(&bias));
        assert!(logits.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn add_token_boost_does_not_resurrect_suppressed_neg_infinity_logits() {
        let bias = TokenPhraseBias::new(vec![vec![1]], crate::MAX_PHRASE_BIAS_BOOST).unwrap();
        let mut logits = vec![0.0_f32; 4];
        logits[1] = f32::NEG_INFINITY; // e.g. a suppressed token
        apply_phrase_bias_to_logits(&mut logits, &[], std::slice::from_ref(&bias));
        assert!(logits[1].is_infinite() && logits[1] < 0.0);
    }

    #[test]
    fn max_boost_is_added_to_a_finite_logit() {
        let bias = TokenPhraseBias::new(vec![vec![2]], crate::MAX_PHRASE_BIAS_BOOST).unwrap();
        let mut logits = vec![0.0_f32; 4];
        logits[2] = 1.0;
        apply_phrase_bias_to_logits(&mut logits, &[], std::slice::from_ref(&bias));
        assert_eq!(logits[2], 1.0 + crate::MAX_PHRASE_BIAS_BOOST);
    }

    #[test]
    fn colliding_phrase_boosts_take_the_strongest_not_the_sum() {
        // Two phrases whose next token is the same id (1). The token gets the
        // strongest single boost (15), never the sum (27) nor the global cap (20).
        let biases = [
            TokenPhraseBias::new(vec![vec![1]], 15.0).unwrap(),
            TokenPhraseBias::new(vec![vec![1]], 12.0).unwrap(),
        ];
        let mut logits = vec![0.0_f32; 4];
        apply_phrase_bias_to_logits(&mut logits, &[], &biases);
        assert_eq!(logits[1], 15.0);
    }

    #[test]
    fn non_colliding_phrase_boosts_keep_their_own_boost() {
        // Distinct tokens: each keeps its own boost.
        let biases = [
            TokenPhraseBias::new(vec![vec![1]], 6.0).unwrap(),
            TokenPhraseBias::new(vec![vec![2]], 7.0).unwrap(),
        ];
        let mut logits = vec![0.0_f32; 4];
        apply_phrase_bias_to_logits(&mut logits, &[], &biases);
        assert_eq!(logits[1], 6.0);
        assert_eq!(logits[2], 7.0);
    }

    #[test]
    fn any_tokenization_variant_activates_the_boost() {
        // A phrase with two tokenization variants ([1,2] and [5,2]): whichever form
        // the model is mid-emitting, the shared continuation token (2) is boosted.
        let bias = TokenPhraseBias::new(vec![vec![1, 2], vec![5, 2]], 4.0).unwrap();

        let mut via_a = vec![0.0_f32; 8];
        apply_phrase_bias_to_logits(&mut via_a, &[1], std::slice::from_ref(&bias));
        assert_eq!(via_a[2], 8.0);

        let mut via_b = vec![0.0_f32; 8];
        apply_phrase_bias_to_logits(&mut via_b, &[5], std::slice::from_ref(&bias));
        assert_eq!(via_b[2], 8.0);
    }

    #[test]
    fn completed_phrase_is_not_immediately_re_nudged() {
        // After the full phrase [1,2] is emitted its first token is NOT re-boosted
        // (anti-repetition); mid-phrase it still is.
        let bias = TokenPhraseBias::new(vec![vec![1, 2]], 5.0).unwrap();

        let mut done = vec![0.0_f32; 8];
        apply_phrase_bias_to_logits(&mut done, &[9, 1, 2], std::slice::from_ref(&bias));
        assert!(done.iter().all(|value| *value == 0.0));

        let mut mid = vec![0.0_f32; 8];
        apply_phrase_bias_to_logits(&mut mid, &[1], std::slice::from_ref(&bias));
        assert_eq!(mid[2], 10.0);
    }

    #[test]
    fn boost_scales_with_match_depth_and_clamps_at_the_global_cap() {
        // Depth scaling: entry = 1x, depth-1 = 2x, depth-2 = 3x ... clamped to
        // MAX_PHRASE_BIAS_BOOST. With boost 8: entry 8, depth-1 16, depth-2
        // would be 24 -> clamped to 20.
        let bias = TokenPhraseBias::new(vec![vec![1, 2, 3]], 8.0).unwrap();

        let mut entry = vec![0.0_f32; 8];
        apply_phrase_bias_to_logits(&mut entry, &[], std::slice::from_ref(&bias));
        assert_eq!(entry[1], 8.0);

        let mut depth1 = vec![0.0_f32; 8];
        apply_phrase_bias_to_logits(&mut depth1, &[1], std::slice::from_ref(&bias));
        assert_eq!(depth1[2], 16.0);

        let mut depth2 = vec![0.0_f32; 8];
        apply_phrase_bias_to_logits(&mut depth2, &[1, 2], std::slice::from_ref(&bias));
        assert_eq!(depth2[3], crate::MAX_PHRASE_BIAS_BOOST);
    }

    #[test]
    fn implausible_continuation_is_not_railroaded_keeps_flat_boost() {
        // Adversarial long-hotword case: phrase [1,2,3,4] (boost 5), and the
        // decode has already emitted [1,2] (e.g. the entry nudge tipped a
        // near-tie). At depth 2 the naive scaled boost would be 15 — enough to
        // flip the acoustically-correct token 7 (logit 20.0) if the candidate
        // sat at 6.0 (gap 14). The plausibility gate (margin 12) must refuse to
        // escalate and fall back to the flat configured boost, so the decode is
        // NOT railroaded into the rest of the phrase.
        let bias = TokenPhraseBias::new(vec![vec![1, 2, 3, 4]], 5.0).unwrap();
        let mut logits = vec![0.0_f32; 8];
        logits[7] = 20.0; // strong contrary acoustic evidence
        logits[3] = 6.0; // candidate trails the max by 14 > margin
        apply_phrase_bias_to_logits(&mut logits, &[1, 2], std::slice::from_ref(&bias));
        assert_eq!(logits[3], 6.0 + 5.0); // flat boost, not 6.0 + 15.0
        assert!(
            logits[3] < logits[7],
            "wrong entry must not chain-force the phrase"
        );
    }

    #[test]
    fn plausible_continuation_still_gets_the_escalated_boost() {
        // The measured real case shape: the candidate trails the argmax by ~10
        // (within the margin of 12), so the depth-2 escalated boost (15) applies
        // and flips the homophone.
        let bias = TokenPhraseBias::new(vec![vec![1, 2, 3]], 5.0).unwrap();
        let mut logits = vec![0.0_f32; 8];
        logits[7] = 10.0; // homophone continuation, pre-bias winner
        logits[3] = 0.0; // hotword continuation, gap 10 <= margin
        apply_phrase_bias_to_logits(&mut logits, &[1, 2], std::slice::from_ref(&bias));
        assert_eq!(logits[3], 15.0);
        assert!(logits[3] > logits[7]);
    }

    #[test]
    fn negative_escalation_ignores_the_plausibility_gate() {
        // Suppression deepens regardless of how far the continuation trails the
        // argmax: pushing an unwanted token further DOWN cannot insert garbage.
        let bias = TokenPhraseBias::new(vec![vec![1, 2]], -6.0).unwrap();
        let mut logits = vec![0.0_f32; 8];
        logits[5] = 30.0; // candidate 2 trails by 30, far beyond the margin
        logits[2] = 0.0;
        apply_phrase_bias_to_logits(&mut logits, &[1], std::slice::from_ref(&bias));
        assert_eq!(logits[2], -12.0); // -6 * 2, still depth-scaled
    }

    #[test]
    fn mixed_depth_collision_takes_the_strongest_scaled_opinion() {
        // Token 3 is both the ENTRY token of phrase B (boost 5 -> applied 5) and
        // the depth-2 continuation of phrase A (boost 5 -> applied 15, plausible
        // here). Abs-max selection over the APPLIED (depth-scaled) values must
        // pick 15 — the deeper match carries more decoded evidence.
        let biases = [
            TokenPhraseBias::new(vec![vec![1, 2, 3]], 5.0).unwrap(),
            TokenPhraseBias::new(vec![vec![3, 4]], 5.0).unwrap(),
        ];
        let mut logits = vec![0.0_f32; 8];
        apply_phrase_bias_to_logits(&mut logits, &[1, 2], &biases);
        assert_eq!(logits[3], 15.0);
    }

    #[test]
    fn negative_boost_scales_with_match_depth_for_suppression() {
        // Anti-context deepens too: depth-1 suppression is -6 * 2 = -12.
        let bias = TokenPhraseBias::new(vec![vec![1, 2]], -6.0).unwrap();
        let mut depth1 = vec![0.0_f32; 8];
        depth1[2] = 5.0;
        apply_phrase_bias_to_logits(&mut depth1, &[1], std::slice::from_ref(&bias));
        assert_eq!(depth1[2], 5.0 - 12.0);
    }

    #[test]
    fn negative_boost_suppresses_its_continuation_token() {
        // Anti-context: a negative-boost phrase pushes its continuation token DOWN.
        let bias = TokenPhraseBias::new(vec![vec![1]], -10.0).unwrap();
        let mut logits = vec![0.0_f32; 4];
        logits[1] = 5.0;
        apply_phrase_bias_to_logits(&mut logits, &[], std::slice::from_ref(&bias));
        assert_eq!(logits[1], -5.0); // 5.0 + (-10.0)
    }

    #[test]
    fn strongest_opinion_wins_on_a_shared_token_across_signs() {
        // A favored (+5) and a suppressed (-12) phrase collide on token 1: the
        // larger-magnitude opinion (suppress) wins; the two do not cancel or sum.
        let biases = [
            TokenPhraseBias::new(vec![vec![1]], 5.0).unwrap(),
            TokenPhraseBias::new(vec![vec![1]], -12.0).unwrap(),
        ];
        let mut logits = vec![0.0_f32; 4];
        logits[1] = 1.0;
        apply_phrase_bias_to_logits(&mut logits, &[], &biases);
        assert_eq!(logits[1], 1.0 - 12.0);
    }
}
