//! FireRedASR2-LLM (`FireRedTeam/FireRedASR2-LLM`) model family.
//!
//! Encoder-Adapter-LLM ASR: the `firered-aed` Conformer encoder (identical
//! architecture to `firered-aed-l-v2`, independently-trained weights -- see
//! [`package_import`]'s module doc) feeds a 2x frame-stacking Adapter (2
//! `Linear` + `ReLU`) that splices into a Qwen2-7B-Instruct decoder's prompt
//! embedding stream (the checkpoint's Qwen2 weights are PEFT-LoRA-finetuned;
//! the LoRA increment is merged into the base weights by the python data-prep
//! stage before this importer ever runs -- see `tooling/publish-model/
//! scripts/firered_llm_merge_lora.py`). ChatML prompt + ASR-specific
//! `<speech>` placeholder token, ASCII+CJK Qwen2 BPE tokenizer. Apache-2.0.
//!
//! Stage status:
//! - The checkpoint-to-GGUF importer lives in [`package_import`] and is
//!   complete (encoder + adapter + LoRA-merged-Qwen2 branches, all hparam
//!   metadata a runtime executor will need).
//! - The Qwen2-parameterized LLM transformer (qwen3-asr's `llm_transformer`
//!   has QK-norm and no qkv-bias; Qwen2 needs the opposite), the Adapter ggml
//!   graph, the dedicated executor, and the `firered-llm.greedy.seq2seq.v0`
//!   decode-policy registration do not exist yet -- a pack produced by this
//!   importer is not yet runnable by `openasr transcribe`.

mod adapter_graph;
mod decode_prompt;
pub(crate) mod executor;
mod llm_transformer;
pub mod package_import;
pub(crate) mod runtime_contract;
pub(crate) mod tensor_names;
mod tokenizer;

pub use package_import::{
    FireRedLlmImportRequest, FireRedLlmImportResult, FireRedLlmQuantizationMode,
    convert_local_firered_llm_source_to_runtime_pack,
};
