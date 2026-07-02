//! Moonshine inference-side dynamic LoRA (OADP Phase 0).
//!
//! The generic resolver, fail-closed validation, and per-(adapter, base) cache
//! live in [`crate::models::lora_adapter`]; this module keeps only the
//! Moonshine-named entry points and the Moonshine LoRA-target predicate. The
//! arena slot bundles are built directly from [`LoraSlot`] in the encoder/decoder
//! graph modules.

use std::path::Path;
use std::sync::Arc;

use crate::adapter_pack::is_moonshine_lora_target_tensor_name;
use crate::models::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;
use crate::models::lora_adapter::{
    LoraResolveError, ResolvedLoraAdapter, adapter_cache_fingerprint, resolve_lora_adapter,
};

pub(crate) use crate::models::lora_adapter::{LoraSlot, new_lora_slot_tensors};
// Only named outside this module in tests; the inference path flows
// `lora_adapter::LoraTarget` directly via `ResolvedLoraAdapter::target`.
#[cfg(test)]
pub(crate) use crate::models::lora_adapter::LoraTarget as MoonshineLoraTarget;

/// Moonshine-named alias for the shared resolved adapter / error types.
pub(crate) type MoonshineLoraAdapter = ResolvedLoraAdapter;
pub(crate) type MoonshineLoraError = LoraResolveError;

const MOONSHINE_LORA_ALLOWED_TARGETS: &str = "{enc,dec}.blk.<n>.{attn_q,attn_k,attn_v,attn_o,ffn_up,ffn_down}.weight and \
     dec.blk.<n>.cross_{q,k,v,o}.weight";

/// Cache-key component for the moonshine runtime caches: empty when no adapter is
/// active.
pub(crate) fn moonshine_adapter_cache_fingerprint(
    adapter: Option<&MoonshineLoraAdapter>,
) -> String {
    adapter_cache_fingerprint(adapter)
}

/// Resolve the active adapter (request-level `--adapter` path, falling back to
/// the `OPENASR_ADAPTER` env var) for a moonshine execution. Returns `Ok(None)`
/// when no adapter is configured; otherwise the adapter must load AND bind to
/// this exact base pack or the whole transcription fails (fail-closed).
pub(crate) fn resolve_moonshine_lora_adapter(
    request_adapter_path: Option<&Path>,
    preflight: &GgmlAsrRuntimeSourcePreflight,
) -> Result<Option<Arc<MoonshineLoraAdapter>>, MoonshineLoraError> {
    resolve_lora_adapter(
        request_adapter_path,
        preflight,
        is_moonshine_lora_target_tensor_name,
        "moonshine",
        MOONSHINE_LORA_ALLOWED_TARGETS,
    )
}

#[cfg(test)]
pub(crate) fn moonshine_lora_adapter_for_test(
    fingerprint: String,
    targets: Vec<(String, MoonshineLoraTarget)>,
) -> MoonshineLoraAdapter {
    crate::models::lora_adapter::lora_adapter_for_test(fingerprint, targets)
}
