//! MiMo-V2.5-ASR (`XiaomiMiMo/MiMo-V2.5-ASR` + `XiaomiMiMo/MiMo-Audio-Tokenizer`)
//! model family: mel -> 32L rope audio-tokenizer encoder (conv stem, skip@L3)
//! -> 8-level RVQ encode (first 8 codebooks only) -> 8-codebook embedding sum
//! -> 6L bidirectional input-local transformer (per 4-frame group) -> group
//! downcast -> ChatML + `<|sosp|>`/`<|eosp|>` prompt splice -> 36L Qwen2
//! backbone (qkv-bias, no QK-norm, reusing `qwen::llm_transformer`'s
//! shared machinery) driven through the ONE shared greedy decode loop. MIT.
//!
//! Pack import surface: this family is the inventory's
//! `OpenAsrPackImportSurface::ExternalTooling` case -- `.oasr` packing is
//! Python-only tooling (`tooling/mimo-asr/convert_mimo_asr.py`), not a Rust
//! `CoreConvert` importer. The split stops at tensor production: the external
//! script writes the full public envelope (routing keys, tokenizer id, build
//! provenance), and every pack it emits still passes the SAME production
//! `PackVerifier` + this module's `runtime_contract` validator (metadata +
//! tensor + tokenizer) at publish staging, install-time admission, and the
//! direct run ingress -- there is no bypass and no second, weaker gate.

mod audio_tokenizer_graph;
pub(crate) mod capacity;
mod decode_prompt;
pub(crate) mod executor;
mod input_local_graph;
mod llm_transformer;
mod mel_frontend;
pub(crate) mod runtime_contract;
mod rvq;
mod tensor_names;
mod tokenizer;
use crate::arch::MIMO_ASR_GGML_ARCHITECTURE_ID;
use crate::models::pack_quant::{QuantizedAxis, TensorQuantizationContract, TensorRole};

pub(crate) const AUDIO_ENCODER_TENSOR_NAME_PREFIXES: &[&str] = &[
    "audiotok.",
    "inlocal.",
    "speech_embd.",
    "speech_group_proj.",
];

pub(crate) const TENSOR_QUANTIZATION_CONTRACT: TensorQuantizationContract =
    TensorQuantizationContract::SemanticRolesV1 {
        model_architecture: MIMO_ASR_GGML_ARCHITECTURE_ID,
        classify: classify_mimo_asr_quant_tensor_role,
        quantized_axis: QuantizedAxis::First,
    };

fn classify_mimo_asr_quant_tensor_role(name: &str) -> TensorRole {
    if name.ends_with(".weight")
        && AUDIO_ENCODER_TENSOR_NAME_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
    {
        TensorRole::AcousticEncoderMatrix
    } else if name.ends_with(".weight") {
        TensorRole::TextDecoderMatrix
    } else {
        TensorRole::NonQuantizable
    }
}
