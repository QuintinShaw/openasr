//! HTTP route handlers and domain clusters extracted from `lib.rs`.
//!
//! Pure code-motion; each submodule pulls shared crate-root items via
//! `use super::*` and re-exports its public surface back through `lib.rs`.

pub(crate) mod config;
pub(crate) mod history;
pub(crate) mod models_api;
pub(crate) mod pairing;
pub(crate) mod pull_jobs;
pub(crate) mod speakers;
pub(crate) mod transcription;
pub(crate) mod translation;
