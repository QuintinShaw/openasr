//! Dolphin `small.cn` dialect model family (WeNet-format E-Branchformer encoder).
//!
//! WIP: only the encoder graph and its dev parity harness live here so far. The
//! executor/frontend/decoder wiring lands separately.

pub(crate) mod encoder_graph;

#[cfg(test)]
mod parity;
