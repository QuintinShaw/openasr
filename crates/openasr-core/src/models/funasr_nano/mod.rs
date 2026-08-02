//! Fun-ASR-Nano-2512 (`FunAudioLLM/Fun-ASR-Nano-2512`, Apache-2.0) model family:
//! a FunASR SAN-M/DFSMN audio encoder (50 `enc.blk` + 20 `tp.blk`, LayerNorm eps
//! 1e-5) -> a 2-layer transformer adaptor (512 -> 2048 -> 1024 MLP + 2 standard
//! transformer blocks) -> a stock Qwen3-0.6B decoder (QK-norm, no attention
//! bias, GQA, tied embeddings). The published `model.pt` carries no CTC decoder
//! (a training-only auxiliary branch), so the runtime path is purely
//! encoder -> adaptor -> Qwen3 greedy decode.
//!
//! Stage status: the SAN-M [`encoder_graph`], the [`adapter_graph`] bridge, the
//! Qwen3 [`llm_transformer`] (reusing `qwen`'s family-agnostic decoder
//! machinery byte-for-byte), the dedicated [`executor`], and the
//! decode-policy/executor/tensor-contract registration in `arch/mod.rs` are all
//! implemented and registered as a builtin architecture -- a pack runs
//! end-to-end. Pack import is via the python `tooling/publish-model/scripts/
//! funasr_nano_pt_to_safetensors.py` converter (external tooling); publication
//! to the model catalog is a later step.

pub(crate) mod adapter_graph;
pub(crate) mod capacity;
pub(crate) mod decode_prompt;
pub(crate) mod encoder_graph;
pub(crate) mod executor;
pub(crate) mod llm_transformer;
pub(crate) mod runtime_contract;
pub(crate) mod tensor_names;
pub(crate) mod tokenizer;
