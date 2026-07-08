//! FireRedPunc (`FireRedTeam/FireRedPunc`) architecture constants and the
//! pack-metadata contract.
//!
//! FireRedPunc is a BERT-style bidirectional encoder (initialised from
//! `chinese-lert-base`) with a token-level classification head: for each input
//! subword it predicts which punctuation mark, if any, follows that token. It
//! is a text-in / labels-out post-processor, not an ASR model -- no audio
//! frontend, no autoregressive decode. Apache-2.0.
//!
//! The released label space is exactly five Chinese full-width classes (see
//! [`PUNC_LABELS`]); the model architecturally cannot emit English half-width
//! marks, so the OpenASR integration is Chinese-only by construction.

use thiserror::Error;

/// `general.architecture` value stamped into the FireRedPunc `.oasr` pack. The
/// pull-time contract dispatches on this string, so ASR/translation packs fall
/// through to their own family adapters.
pub(crate) const FIRERED_PUNC_ARCHITECTURE_VALUE: &str = "firered-punc";

pub(crate) const FIRERED_PUNC_BLOCK_COUNT_KEY: &str = "firered-punc.block_count";
pub(crate) const FIRERED_PUNC_EMBEDDING_LENGTH_KEY: &str = "firered-punc.embedding_length";
pub(crate) const FIRERED_PUNC_FEED_FORWARD_LENGTH_KEY: &str = "firered-punc.feed_forward_length";
pub(crate) const FIRERED_PUNC_ATTENTION_HEAD_COUNT_KEY: &str = "firered-punc.attention.head_count";
pub(crate) const FIRERED_PUNC_ATTENTION_LAYER_NORM_EPSILON_KEY: &str =
    "firered-punc.attention.layer_norm_epsilon";
pub(crate) const FIRERED_PUNC_CONTEXT_LENGTH_KEY: &str = "firered-punc.context_length";
pub(crate) const FIRERED_PUNC_VOCAB_SIZE_KEY: &str = "firered-punc.vocab_size";
pub(crate) const FIRERED_PUNC_LABEL_COUNT_KEY: &str = "firered-punc.label_count";
pub(crate) const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";

/// chinese-lert-base hyper-parameters (`config.json`): 12-layer BERT-base,
/// hidden 768, 12 heads, intermediate 3072, GELU, LayerNorm eps 1e-12, learned
/// absolute positions up to 512, two token-type segments.
pub(crate) const FIRERED_PUNC_EXPECTED_LAYERS: usize = 12;
pub(crate) const FIRERED_PUNC_EXPECTED_D_MODEL: usize = 768;
pub(crate) const FIRERED_PUNC_EXPECTED_FFN_DIM: usize = 3072;
pub(crate) const FIRERED_PUNC_EXPECTED_HEADS: usize = 12;
pub(crate) const FIRERED_PUNC_EXPECTED_VOCAB_SIZE: usize = 21_128;
pub(crate) const FIRERED_PUNC_EXPECTED_MAX_POSITIONS: usize = 512;
pub(crate) const FIRERED_PUNC_TYPE_VOCAB_SIZE: usize = 2;
pub(crate) const FIRERED_PUNC_LAYER_NORM_EPSILON: f32 = 1.0e-12;

/// The five punctuation classes from upstream `out_dict`, in label-id order.
/// Index 0 is "no punctuation after this token" (`<space>` in `out_dict`); the
/// remaining four are the Chinese full-width comma, period, question mark, and
/// exclamation mark. This ordering is the contract with the trained classifier
/// head and must not be reordered.
pub(crate) const PUNC_LABELS: [Option<char>; 5] =
    [None, Some('，'), Some('。'), Some('？'), Some('！')];

/// Number of punctuation classes (`PUNC_LABELS.len()`), i.e. the classifier
/// head output width.
pub(crate) const FIRERED_PUNC_LABEL_COUNT: usize = PUNC_LABELS.len();

/// Returns the punctuation mark for a predicted label id, or `None` for the
/// "no punctuation" class (label 0) and any id outside the trained label space.
pub(crate) fn punctuation_for_label(label_id: usize) -> Option<char> {
    PUNC_LABELS.get(label_id).copied().flatten()
}

/// Validated FireRedPunc pack geometry, read from the GGUF metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FireRedPuncExecutionMetadata {
    pub layers: usize,
    pub d_model: usize,
    pub ffn_dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_positions: usize,
    pub label_count: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FireRedPuncConfigError {
    #[error("firered-punc pack is missing required metadata key '{0}'")]
    MissingMetadata(String),
    #[error("firered-punc metadata key '{key}' has an unexpected type")]
    MetadataType { key: String },
    #[error(
        "firered-punc geometry mismatch: {field} is {got}, expected {expected} for chinese-lert-base"
    )]
    UnexpectedGeometry {
        field: &'static str,
        got: usize,
        expected: usize,
    },
    #[error("firered-punc embedding_length {d_model} is not divisible by head_count {heads}")]
    HeadDimNotDivisible { d_model: usize, heads: usize },
}

impl FireRedPuncExecutionMetadata {
    /// Asserts the pack geometry matches the chinese-lert-base checkpoint the
    /// runtime graph is written against. FireRedPunc ships a single published
    /// checkpoint, so an off-geometry pack is a conversion bug, not a variant.
    pub(crate) fn assert_expected_chinese_lert_base(self) -> Result<(), FireRedPuncConfigError> {
        let checks = [
            ("layers", self.layers, FIRERED_PUNC_EXPECTED_LAYERS),
            ("d_model", self.d_model, FIRERED_PUNC_EXPECTED_D_MODEL),
            ("ffn_dim", self.ffn_dim, FIRERED_PUNC_EXPECTED_FFN_DIM),
            ("heads", self.heads, FIRERED_PUNC_EXPECTED_HEADS),
            (
                "vocab_size",
                self.vocab_size,
                FIRERED_PUNC_EXPECTED_VOCAB_SIZE,
            ),
            (
                "max_positions",
                self.max_positions,
                FIRERED_PUNC_EXPECTED_MAX_POSITIONS,
            ),
            ("label_count", self.label_count, FIRERED_PUNC_LABEL_COUNT),
        ];
        for (field, got, expected) in checks {
            if got != expected {
                return Err(FireRedPuncConfigError::UnexpectedGeometry {
                    field,
                    got,
                    expected,
                });
            }
        }
        Ok(())
    }
}
