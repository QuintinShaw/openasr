//! MiMo-V2.5-ASR (`XiaomiMiMo/MiMo-V2.5-ASR` + `XiaomiMiMo/MiMo-Audio-Tokenizer`)
//! model family: mel -> 32L rope audio-tokenizer encoder (conv stem, skip@L3)
//! -> 8-level RVQ encode (first 8 codebooks only) -> 8-codebook embedding sum
//! -> 6L bidirectional input-local transformer (per 4-frame group) -> group
//! downcast -> ChatML + `<|sosp|>`/`<|eosp|>` prompt splice -> 36L Qwen2
//! backbone (qkv-bias, no QK-norm, reusing `qwen::llm_transformer`'s
//! shared machinery) driven through the ONE shared greedy decode loop. MIT.
//!
//! Conversion (`.oasr` packing) is Python-only tooling
//! (`tooling/mimo-asr/convert_mimo_asr.py`), not a Rust importer -- unlike
//! `firered_llm`, which has its own `package_import.rs`. This module is the
//! P2.2 runtime only: mel/encoder/RVQ/input-local/LLM graphs, tokenizer,
//! decode-policy registration, and the dedicated executor.

mod audio_tokenizer_graph;
mod decode_prompt;
pub(crate) mod executor;
mod input_local_graph;
mod llm_transformer;
mod mel_frontend;
pub(crate) mod runtime_contract;
mod rvq;
mod tensor_names;
mod tokenizer;
