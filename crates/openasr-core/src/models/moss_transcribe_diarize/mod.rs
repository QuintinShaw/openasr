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
//! Stage status: only the checkpoint-to-GGUF importer
//! ([`package_import`]) exists so far -- see that module's doc comment for
//! the full tensor-mapping contract and exactly what is and is not wired up
//! yet. The ggml execution graph (Whisper encoder reuse, the adaptor
//! bridge, Qwen3 decoder reuse, decode-policy registration) has not been
//! implemented; a pack produced by this importer is not yet runnable by
//! `openasr transcribe`.

pub(crate) mod package_import;
pub(crate) mod tensor_names;

// Not yet consumed by any CLI/tooling entry point (see the module doc above
// for stage status) -- re-exported now so the runtime-wiring follow-up and a
// future CLI `model-pack import` case can reach these without touching this
// file again. Matches every other family module's `pub use` shape in this
// crate (e.g. `firered_llm`, `mimo_asr`), which stay unused the same way
// until their own executor/CLI wiring lands.
#[allow(unused_imports)]
pub use package_import::{
    MossTdImportRequest, MossTdImportResult, MossTdQuantizationMode,
    convert_local_moss_transcribe_diarize_source_to_runtime_pack,
};
