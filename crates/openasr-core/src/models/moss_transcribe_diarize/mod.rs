//! MOSS-Transcribe-Diarize (`OpenMOSS/MOSS-Transcribe-Diarize`, 0.9B) model
//! family: a joint transcription + speaker-diarization ASR model built from
//! a Whisper-Medium-architecture audio encoder, a small `VQAdaptor` bridge
//! (a plain 3-layer MLP+LayerNorm despite the "VQ" name -- there is no
//! vector-quantization codebook in this checkpoint), and a Qwen3-0.6B
//! decoder. `[S01]`-style speaker labels and inline timestamps are ordinary
//! BPE tokens the Qwen3 decoder emits freely as part of its transcript text;
//! this family does not parse or structure them (that is a product-layer
//! concern, out of scope for the core engine).
//!
//! Stage status: the checkpoint-to-GGUF importer ([`package_import`]) and the
//! full ggml execution graph (Whisper encoder reuse via [`encoder_graph`],
//! the [`adaptor_graph`] bridge, Qwen3 decoder reuse via [`llm_decoder`], and
//! decode-policy/executor/tensor-contract registration in [`executor`] and
//! `arch/mod.rs`) are both implemented and registered as a builtin
//! architecture -- a pack produced by this importer runs end-to-end through
//! `openasr transcribe --model-pack <pack>` (CPU; the Metal path has a known
//! encoder-numerics defect, see the arch descriptor's `auto_gpu_policy`
//! doc). What is NOT yet wired: a public `openasr model-pack import`
//! subcommand (the importer above is reachable only from Rust/tests, same
//! pre-CLI-wiring stage `qwen3-forced-aligner` was at) and publication to
//! the model catalog/registry (see `tooling/publish-model/models-core.toml`'s
//! `moss-transcribe-diarize` entry, staged `release_public` but not yet
//! public).

mod adaptor_graph;
mod decode_prompt;
mod encoder_graph;
pub(crate) mod executor;
mod graph_config;
mod llm_decoder;
pub(crate) mod package_import;
mod prompt_embedding;
pub(crate) mod runtime_contract;
pub(crate) mod tensor_names;
mod tokenizer;

// Not yet consumed by any CLI/tooling entry point (see the module doc above
// for stage status) -- re-exported now so a future CLI `model-pack import`
// case can reach these without touching this file again. Matches every
// other family module's `pub use` shape in this crate (e.g. `firered_llm`,
// `mimo_asr`), which stay unused the same way until their own CLI wiring
// lands.
#[allow(unused_imports)]
pub use package_import::{
    MossTdImportRequest, MossTdImportResult, MossTdQuantizationMode,
    convert_local_moss_transcribe_diarize_source_to_runtime_pack,
};
