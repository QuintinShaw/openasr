//! RNN-T greedy search for X-ASR.

use super::decoder::XasrDecoder;
use super::joiner::{XasrJoiner, XasrJoinerScratch};
use super::tokenizer::XasrZipformerTokenizer;

pub(crate) const DEFAULT_MAX_SYMBOLS_PER_FRAME: usize = 8;

/// Runtime-minted proof and the exact host-visible rows consumed by one XASR
/// selection compute. A speculative blank graph has several rows under one
/// compute; `output_index` distinguishes the row actually used for each
/// selection without pretending that the graph ran once per frame.
pub(crate) struct XasrSelectionEvidence {
    rows: Vec<crate::ggml_runtime::GgmlSelectionEvidenceRef>,
    logits: Vec<f32>,
    vocab_size: usize,
    row_count: usize,
}

impl XasrSelectionEvidence {
    pub(super) fn new(
        rows: Vec<crate::ggml_runtime::GgmlSelectionEvidenceRef>,
        logits: Vec<f32>,
        vocab_size: usize,
        row_count: usize,
    ) -> Result<Self, String> {
        let expected = vocab_size
            .checked_mul(row_count)
            .ok_or_else(|| "xasr selection evidence shape overflowed".to_string())?;
        if vocab_size == 0 || row_count == 0 || logits.len() != expected || rows.len() != row_count
        {
            return Err(format!(
                "xasr selection evidence has {} logits and {} witnesses, expected {row_count} rows of {vocab_size}",
                logits.len(),
                rows.len()
            ));
        }
        Ok(Self {
            rows,
            logits,
            vocab_size,
            row_count,
        })
    }

    fn row(
        &self,
        output_index: usize,
    ) -> Option<(&[f32], crate::ggml_runtime::GgmlSelectionEvidenceRef)> {
        if output_index >= self.row_count {
            return None;
        }
        let start = output_index.checked_mul(self.vocab_size)?;
        let end = start.checked_add(self.vocab_size)?;
        Some((self.logits.get(start..end)?, *self.rows.get(output_index)?))
    }
}

pub(crate) trait XasrGreedyDecodeBackend {
    fn project_encoder_frame(&mut self, frame: &[f32]) -> Result<(), String>;
    fn project_decoder_context(&mut self, context: &[u32]) -> Result<(), String>;
    fn next_token(&mut self) -> Result<u32, String>;
    fn token_probability(&self, token: u32) -> Result<f32, String>;

    /// Consume proof for the immediately preceding scalar or speculative
    /// selection compute. Normal inference does not retain rows; an explicit
    /// request receipt makes the device backend retain them until this call.
    fn take_selection_evidence(&mut self) -> Option<XasrSelectionEvidence> {
        None
    }

    fn speculative_blank_prefix_len(
        &mut self,
        _context: Option<&[u32]>,
        _encoder_frames: &[f32],
        _frame_count: usize,
        _encoder_dim: usize,
    ) -> Result<Option<usize>, String> {
        Ok(None)
    }
}

/// Record one joiner selection that has a ggml compute witness. Host joiner
/// rows have no native evidence and are skipped. Device-head speculative
/// blanks and scalar recomputes keep their readback-bound receipt steps.
fn record_selection_receipt(
    evidence: Option<&XasrSelectionEvidence>,
    output_index: usize,
    token_id: u32,
) {
    let Some(receipt) =
        crate::models::native_execution_services::current_execution_receipt_collector()
    else {
        return;
    };
    let Some((row, compute)) = evidence.and_then(|evidence| evidence.row(output_index)) else {
        return;
    };
    let step_index = receipt.begin_next_decode_step(Some(compute));
    receipt.record_top_k_last_max(step_index, row);
    receipt.record_token(step_index, token_id, false);
    receipt.finish_decode_step(step_index);
}

struct HostXasrGreedyDecodeBackend<'a> {
    decoder: &'a XasrDecoder,
    joiner: &'a XasrJoiner,
    scratch: XasrJoinerScratch,
}

impl<'a> HostXasrGreedyDecodeBackend<'a> {
    fn new(decoder: &'a XasrDecoder, joiner: &'a XasrJoiner) -> Self {
        Self {
            decoder,
            joiner,
            scratch: joiner.scratch(),
        }
    }
}

impl XasrGreedyDecodeBackend for HostXasrGreedyDecodeBackend<'_> {
    fn project_encoder_frame(&mut self, frame: &[f32]) -> Result<(), String> {
        self.joiner.project_encoder_frame(frame, &mut self.scratch)
    }

    fn project_decoder_context(&mut self, context: &[u32]) -> Result<(), String> {
        let decoder_state = self.decoder.decode_context(context)?;
        self.joiner
            .project_decoder_state(&decoder_state, &mut self.scratch)
    }

    fn next_token(&mut self) -> Result<u32, String> {
        let logits = self.joiner.logits_from_projected(&mut self.scratch)?;
        argmax(logits).ok_or_else(|| "xasr joiner produced no logits".to_string())
    }

    fn token_probability(&self, token: u32) -> Result<f32, String> {
        self.joiner.token_probability(&self.scratch, token)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrGreedyDecodeResult {
    pub token_ids: Vec<u32>,
    /// Absolute encoder frame each token was emitted on (parallel to
    /// `token_ids`).
    pub emit_frames: Vec<usize>,
    /// Joiner softmax probability of each emitted token (parallel to
    /// `token_ids`).
    pub emit_probabilities: Vec<f32>,
    /// Total encoder frames the emission frames index into.
    pub encoder_frames: usize,
    pub text: String,
}

pub(crate) fn greedy_decode_frames(
    encoder_frames: &[f32],
    frame_count: usize,
    encoder_dim: usize,
    decoder: &XasrDecoder,
    joiner: &XasrJoiner,
    tokenizer: &XasrZipformerTokenizer,
    blank_id: u32,
) -> Result<XasrGreedyDecodeResult, String> {
    greedy_decode_frames_with_limit(
        encoder_frames,
        frame_count,
        encoder_dim,
        decoder,
        joiner,
        tokenizer,
        blank_id,
        DEFAULT_MAX_SYMBOLS_PER_FRAME,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn greedy_decode_frames_with_limit(
    encoder_frames: &[f32],
    frame_count: usize,
    encoder_dim: usize,
    decoder: &XasrDecoder,
    joiner: &XasrJoiner,
    tokenizer: &XasrZipformerTokenizer,
    blank_id: u32,
    max_symbols_per_frame: usize,
) -> Result<XasrGreedyDecodeResult, String> {
    let mut context = decoder.initial_context();
    let mut emitted = Vec::new();
    let mut emit_frames = Vec::new();
    let mut emit_probabilities = Vec::new();
    greedy_decode_frames_incremental(
        encoder_frames,
        frame_count,
        encoder_dim,
        decoder,
        joiner,
        blank_id,
        max_symbols_per_frame,
        &mut context,
        &mut emitted,
        &mut emit_frames,
        &mut emit_probabilities,
        0,
        &|| false,
    )?;
    let text = tokenizer.decode(&emitted)?;
    Ok(XasrGreedyDecodeResult {
        token_ids: emitted,
        emit_frames,
        emit_probabilities,
        encoder_frames: frame_count,
        text,
    })
}

/// Greedy RNN-T over `frame_count` encoder frames, continuing from the given
/// decoder `context` and appending to `emitted`. Each emission also records
/// its absolute encoder frame (`frame_offset` + local index) into
/// `emit_frames` and its joiner softmax probability into
/// `emit_probabilities`, both kept parallel to `emitted` — the alignment and
/// the per-token score transducers get for free.
///
/// Per-step cost discipline: the encoder projection is computed once per
/// frame, and the decoder state + its projection are recomputed only after a
/// non-blank emission changes the context — across the (overwhelmingly
/// common) blank-only frames, each step runs just the vocab output linear.
/// The probability is computed only on emission (non-blank), so blank-only
/// frames pay nothing extra.
#[allow(clippy::too_many_arguments)]
pub(crate) fn greedy_decode_frames_incremental(
    encoder_frames: &[f32],
    frame_count: usize,
    encoder_dim: usize,
    decoder: &XasrDecoder,
    joiner: &XasrJoiner,
    blank_id: u32,
    max_symbols_per_frame: usize,
    context: &mut Vec<u32>,
    emitted: &mut Vec<u32>,
    emit_frames: &mut Vec<usize>,
    emit_probabilities: &mut Vec<f32>,
    frame_offset: usize,
    is_canceled: &dyn Fn() -> bool,
) -> Result<usize, String> {
    let mut backend = HostXasrGreedyDecodeBackend::new(decoder, joiner);
    greedy_decode_frames_incremental_with_backend(
        encoder_frames,
        frame_count,
        encoder_dim,
        &mut backend,
        blank_id,
        max_symbols_per_frame,
        context,
        emitted,
        emit_frames,
        emit_probabilities,
        frame_offset,
        is_canceled,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn greedy_decode_frames_incremental_with_backend<B: XasrGreedyDecodeBackend>(
    encoder_frames: &[f32],
    frame_count: usize,
    encoder_dim: usize,
    backend: &mut B,
    blank_id: u32,
    max_symbols_per_frame: usize,
    context: &mut Vec<u32>,
    emitted: &mut Vec<u32>,
    emit_frames: &mut Vec<usize>,
    emit_probabilities: &mut Vec<f32>,
    frame_offset: usize,
    is_canceled: &dyn Fn() -> bool,
) -> Result<usize, String> {
    let expected = frame_count
        .checked_mul(encoder_dim)
        .ok_or_else(|| "xasr greedy encoder shape overflow".to_string())?;
    if encoder_frames.len() != expected {
        return Err(format!(
            "xasr greedy got {} encoder values, expected {expected}",
            encoder_frames.len()
        ));
    }
    let start_len = emitted.len();
    let mut decoder_projection_valid = false;
    let mut frame_idx = 0usize;
    while frame_idx < frame_count {
        // The token-control loop remains host-side on every backend, so poll at
        // each encoder-frame boundary in addition to the shared graph-abort
        // callback used by device graph execution.
        if is_canceled() {
            return Err(format!(
                "xasr-zipformer decode canceled at encoder frame {frame_idx}"
            ));
        }
        let remaining_frames = frame_count - frame_idx;
        let remaining_values = &encoder_frames[frame_idx * encoder_dim..];
        let speculative_context = (!decoder_projection_valid).then_some(context.as_slice());
        let speculative_blank_prefix_len = backend.speculative_blank_prefix_len(
            speculative_context,
            remaining_values,
            remaining_frames,
            encoder_dim,
        )?;
        let speculative_evidence = backend.take_selection_evidence();
        if speculative_blank_prefix_len.is_none() && speculative_evidence.is_some() {
            return Err(
                "xasr backend produced selection evidence without a speculative result".to_string(),
            );
        }
        if let Some(blank_prefix_len) = speculative_blank_prefix_len {
            if blank_prefix_len > remaining_frames {
                return Err("xasr speculative blank prefix exceeds remaining frames".to_string());
            }
            for output_index in 0..blank_prefix_len {
                record_selection_receipt(speculative_evidence.as_ref(), output_index, blank_id);
            }
            if speculative_context.is_some() {
                decoder_projection_valid = true;
            }
            frame_idx += blank_prefix_len;
            if frame_idx == frame_count {
                break;
            }
        }
        let frame = &encoder_frames[frame_idx * encoder_dim..(frame_idx + 1) * encoder_dim];
        backend.project_encoder_frame(frame)?;
        for _ in 0..max_symbols_per_frame {
            if !decoder_projection_valid {
                backend.project_decoder_context(context)?;
                decoder_projection_valid = true;
            }
            let token_id = backend.next_token()?;
            let selection_evidence = backend.take_selection_evidence();
            if token_id == blank_id {
                record_selection_receipt(selection_evidence.as_ref(), 0, token_id);
                break;
            }
            let probability = backend.token_probability(token_id)?;
            record_selection_receipt(selection_evidence.as_ref(), 0, token_id);
            emitted.push(token_id);
            emit_frames.push(frame_offset + frame_idx);
            emit_probabilities.push(probability);
            context.remove(0);
            context.push(token_id);
            decoder_projection_valid = false;
        }
        frame_idx += 1;
    }
    Ok(emitted.len() - start_len)
}

pub(super) fn argmax(values: &[f32]) -> Option<u32> {
    values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::xasr_zipformer::decoder::XasrDecoder;
    use crate::models::xasr_zipformer::joiner::XasrJoiner;
    use crate::models::xasr_zipformer::weights::{
        NamedTensor, StoredLinear, XasrDecoderWeights, XasrJoinerWeights,
    };

    #[test]
    fn argmax_uses_last_index_on_exact_ties() {
        assert_eq!(argmax(&[3.0, 7.0, 7.0, 2.0]), Some(2));
        assert_eq!(argmax(&[f32::NAN, 7.0, 7.0]), Some(2));
        assert_eq!(
            argmax(&[2.0, 1.0, 5.0, 5.0]),
            Some(3),
            "XASR host last-max must keep the last equal maximum"
        );
    }

    #[test]
    fn greedy_emits_until_blank_and_advances_context() {
        let tokenizer = XasrZipformerTokenizer::new(
            vec![
                "<blk>".to_string(),
                "\u{2581}A".to_string(),
                "\u{2581}B".to_string(),
            ],
            0,
        )
        .unwrap();
        let decoder = XasrDecoder::new(decoder_weights(), 2, 0);
        let joiner = XasrJoiner::new(joiner_weights());
        let result = greedy_decode_frames_with_limit(
            &[1.0, 0.0, 0.0, 1.0],
            2,
            2,
            &decoder,
            &joiner,
            &tokenizer,
            0,
            1,
        )
        .unwrap();
        assert_eq!(result.token_ids, vec![1, 2]);
        assert_eq!(result.text, "A B");
        assert_eq!(result.emit_frames, vec![0, 1]);
        assert_eq!(result.encoder_frames, 2);
        assert_eq!(result.emit_probabilities.len(), 2);
        // The fixture joiner separates the winner by 8 logits; its softmax
        // probability must reflect near-certainty.
        assert!(result.emit_probabilities.iter().all(|p| *p > 0.99));
    }

    #[test]
    fn incremental_emit_frames_are_offset_to_absolute_stream_frames() {
        let decoder = XasrDecoder::new(decoder_weights(), 2, 0);
        let joiner = XasrJoiner::new(joiner_weights());
        let mut context = decoder.initial_context();
        let mut emitted = Vec::new();
        let mut emit_frames = Vec::new();
        let mut emit_probabilities = Vec::new();
        greedy_decode_frames_incremental(
            &[1.0, 0.0, 0.0, 1.0],
            2,
            2,
            &decoder,
            &joiner,
            0,
            1,
            &mut context,
            &mut emitted,
            &mut emit_frames,
            &mut emit_probabilities,
            7,
            &|| false,
        )
        .unwrap();
        assert_eq!(emitted.len(), emit_frames.len());
        assert_eq!(emitted.len(), emit_probabilities.len());
        assert_eq!(emit_frames, vec![7, 8]);
    }

    #[test]
    fn greedy_decode_polls_cancellation_at_frame_boundaries() {
        let decoder = XasrDecoder::new(decoder_weights(), 2, 0);
        let joiner = XasrJoiner::new(joiner_weights());
        let mut context = decoder.initial_context();
        let mut emitted = Vec::new();
        let mut emit_frames = Vec::new();
        let mut emit_probabilities = Vec::new();
        // Already canceled before the first frame: the dedicated transducer
        // loop must fail closed without emitting anything, mirroring the
        // shared cooperative cancellation contract (the parakeet-tdt precedent).
        let error = greedy_decode_frames_incremental(
            &[1.0, 0.0, 0.0, 1.0],
            2,
            2,
            &decoder,
            &joiner,
            0,
            1,
            &mut context,
            &mut emitted,
            &mut emit_frames,
            &mut emit_probabilities,
            0,
            &|| true,
        )
        .expect_err("a canceled decode must fail closed");
        assert!(error.contains("canceled"), "{error}");
        assert!(
            emitted.is_empty(),
            "cancel polling must remain frame-local and emit nothing"
        );
    }

    struct SpeculativeTestBackend {
        blank_prefix: usize,
        projected_frames: Vec<Vec<f32>>,
        projected_contexts: usize,
        tokens: std::collections::VecDeque<u32>,
    }

    impl XasrGreedyDecodeBackend for SpeculativeTestBackend {
        fn project_encoder_frame(&mut self, frame: &[f32]) -> Result<(), String> {
            self.projected_frames.push(frame.to_vec());
            Ok(())
        }

        fn project_decoder_context(&mut self, _context: &[u32]) -> Result<(), String> {
            self.projected_contexts += 1;
            Ok(())
        }

        fn next_token(&mut self) -> Result<u32, String> {
            self.tokens
                .pop_front()
                .ok_or_else(|| "speculative test backend ran out of tokens".to_string())
        }

        fn token_probability(&self, _token: u32) -> Result<f32, String> {
            Ok(0.75)
        }

        fn speculative_blank_prefix_len(
            &mut self,
            context: Option<&[u32]>,
            _encoder_frames: &[f32],
            _frame_count: usize,
            _encoder_dim: usize,
        ) -> Result<Option<usize>, String> {
            if context.is_some() {
                self.projected_contexts += 1;
            }
            Ok(Some(self.blank_prefix))
        }
    }

    #[test]
    fn speculative_blank_prefix_skips_only_confirmed_frames_then_uses_scalar_path() {
        let mut backend = SpeculativeTestBackend {
            blank_prefix: 2,
            projected_frames: Vec::new(),
            projected_contexts: 0,
            tokens: [1, 0].into_iter().collect(),
        };
        let mut context = vec![0, 0];
        let mut emitted = Vec::new();
        let mut frames = Vec::new();
        let mut probabilities = Vec::new();

        let count = greedy_decode_frames_incremental_with_backend(
            &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
            3,
            2,
            &mut backend,
            0,
            2,
            &mut context,
            &mut emitted,
            &mut frames,
            &mut probabilities,
            7,
            &|| false,
        )
        .expect("speculative blank prefix decode");

        assert_eq!(count, 1);
        assert_eq!(emitted, vec![1]);
        assert_eq!(frames, vec![9]);
        assert_eq!(probabilities, vec![0.75]);
        assert_eq!(context, vec![0, 1]);
        assert_eq!(backend.projected_frames, vec![vec![30.0, 31.0]]);
        assert_eq!(backend.projected_contexts, 2);
    }

    #[test]
    fn speculative_blank_prefix_rejects_backend_overrun() {
        let mut backend = SpeculativeTestBackend {
            blank_prefix: 3,
            projected_frames: Vec::new(),
            projected_contexts: 0,
            tokens: std::collections::VecDeque::new(),
        };
        let mut context = vec![0, 0];
        let mut emitted = Vec::new();
        let mut frames = Vec::new();
        let mut probabilities = Vec::new();

        let error = greedy_decode_frames_incremental_with_backend(
            &[1.0, 2.0, 3.0, 4.0],
            2,
            2,
            &mut backend,
            0,
            1,
            &mut context,
            &mut emitted,
            &mut frames,
            &mut probabilities,
            0,
            &|| false,
        )
        .expect_err("speculative prefix beyond remaining frames must fail closed");

        assert!(error.contains("exceeds remaining frames"), "{error}");
        assert!(emitted.is_empty());
    }

    struct ReceiptEvidenceBackend {
        blank_prefix: Option<usize>,
        speculative_evidence: Option<XasrSelectionEvidence>,
        scalar: std::collections::VecDeque<(u32, f32, XasrSelectionEvidence)>,
        last_probability: f32,
        last_evidence: Option<XasrSelectionEvidence>,
    }

    impl XasrGreedyDecodeBackend for ReceiptEvidenceBackend {
        fn project_encoder_frame(&mut self, _frame: &[f32]) -> Result<(), String> {
            Ok(())
        }

        fn project_decoder_context(&mut self, _context: &[u32]) -> Result<(), String> {
            Ok(())
        }

        fn next_token(&mut self) -> Result<u32, String> {
            let (token, probability, evidence) = self
                .scalar
                .pop_front()
                .ok_or_else(|| "receipt evidence backend ran out of scalar rows".to_string())?;
            self.last_probability = probability;
            self.last_evidence = Some(evidence);
            Ok(token)
        }

        fn token_probability(&self, _token: u32) -> Result<f32, String> {
            Ok(self.last_probability)
        }

        fn take_selection_evidence(&mut self) -> Option<XasrSelectionEvidence> {
            self.last_evidence.take()
        }

        fn speculative_blank_prefix_len(
            &mut self,
            _context: Option<&[u32]>,
            _encoder_frames: &[f32],
            _frame_count: usize,
            _encoder_dim: usize,
        ) -> Result<Option<usize>, String> {
            self.last_evidence = self.speculative_evidence.take();
            Ok(self.blank_prefix.take())
        }
    }

    fn mint_selection_evidence(rows: &[&[f32]]) -> XasrSelectionEvidence {
        assert!(!rows.is_empty());
        let width = rows[0].len();
        assert!(width > 0 && rows.iter().all(|row| row.len() == width));
        let values = rows
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        let mut runner = crate::ggml_runtime::GgmlCpuGraphRunner::new(
            crate::ggml_runtime::GgmlCpuGraphConfig::conservative_default(),
        )
        .expect("CPU graph runner");
        let mut graph = runner.start_graph();
        let output = graph
            .new_tensor_1d_f32(values.len(), "xasr_receipt_rows")
            .expect("row tensor");
        graph.set_input(output).expect("row input");
        graph.set_output(output).expect("row output");
        graph
            .set_f32_slice(output, &values, "xasr_receipt_rows")
            .expect("row upload");
        let observed = graph
            .compute_output_f32_rows_with_evidence(output, width, rows.len())
            .expect("row compute");
        let (values, evidence) = observed.into_parts();
        XasrSelectionEvidence::new(
            evidence.expect("installed receipt must mint row witnesses"),
            values,
            width,
            rows.len(),
        )
        .expect("valid row evidence")
    }

    #[test]
    fn speculative_and_scalar_selections_bind_distinct_runtime_minted_rows() {
        let receipt = crate::NativeExecutionReceiptCollector::new();
        receipt.set_trace_mode(crate::NativeExecutionTraceMode::Cold);
        receipt.begin_candidate_attempt();
        let _guard = crate::models::native_execution_services::install_execution_receipt_collector(
            Some(receipt.clone()),
        );
        let speculative =
            mint_selection_evidence(&[&[4.0, 0.0, 0.0], &[3.0, 1.0, 0.0], &[0.0, 2.0, 1.0]]);
        let scalar_non_blank = mint_selection_evidence(&[&[0.0, 2.0, 1.0]]);
        let scalar_blank = mint_selection_evidence(&[&[3.0, 1.0, 0.0]]);
        let mut backend = ReceiptEvidenceBackend {
            blank_prefix: Some(2),
            speculative_evidence: Some(speculative),
            scalar: [(1, 0.75, scalar_non_blank), (0, 0.75, scalar_blank)]
                .into_iter()
                .collect(),
            last_probability: 0.0,
            last_evidence: None,
        };
        let mut context = vec![0, 0];
        let mut emitted = Vec::new();
        let mut frames = Vec::new();
        let mut probabilities = Vec::new();

        greedy_decode_frames_incremental_with_backend(
            &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
            3,
            2,
            &mut backend,
            0,
            2,
            &mut context,
            &mut emitted,
            &mut frames,
            &mut probabilities,
            0,
            &|| false,
        )
        .expect("receipt-bound XASR decode");
        receipt.finish_candidate_attempt(true);

        let snapshot = receipt.snapshot();
        assert!(!snapshot.trace.invalid_binding);
        let tokens = snapshot
            .trace
            .jsonl
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|event| event.get("event").and_then(serde_json::Value::as_str) == Some("token"))
            .collect::<Vec<_>>();
        assert_eq!(tokens.len(), 4);
        assert_eq!(
            tokens
                .iter()
                .map(|event| event["token_id"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 0, 1, 0]
        );
        assert_eq!(
            tokens
                .iter()
                .map(|event| {
                    (
                        event["compute"]["output_index"].as_u64().unwrap(),
                        event["compute"]["output_count"].as_u64().unwrap(),
                    )
                })
                .collect::<Vec<_>>(),
            vec![(0, 3), (1, 3), (0, 1), (0, 1)]
        );
    }

    fn decoder_weights() -> XasrDecoderWeights {
        XasrDecoderWeights {
            embedding: StoredLinear {
                name: "emb".to_string(),
                input_dim: 2,
                output_dim: 3,
                values: vec![
                    0.0, 0.0, // blank
                    1.0, 0.0, // token 1
                    0.0, 1.0, // token 2
                ],
                native: None,
            },
            conv_weight: NamedTensor {
                name: "conv".to_string(),
                dims: vec![2, 2, 2],
                values: vec![
                    0.0, 0.0, 1.0, 0.0, // out0 reads second token channel 0
                    0.0, 0.0, 0.0, 1.0, // out1 reads second token channel 1
                ],
            },
            groups: 1,
        }
    }

    fn joiner_weights() -> XasrJoinerWeights {
        XasrJoinerWeights {
            encoder_proj_weight: identity("enc", 2),
            encoder_proj_bias: vec![0.0, 0.0],
            decoder_proj_weight: StoredLinear {
                name: "dec".to_string(),
                input_dim: 2,
                output_dim: 2,
                values: vec![-1.0, 0.0, 0.0, -1.0],
                native: None,
            },
            decoder_proj_bias: vec![0.0, 0.0],
            output_linear_weight: StoredLinear {
                name: "out".to_string(),
                input_dim: 2,
                output_dim: 3,
                values: vec![
                    -4.0, -4.0, // blank
                    4.0, -4.0, // token 1
                    -4.0, 4.0, // token 2
                ],
                native: None,
            },
            output_linear_bias: vec![0.0, 0.0, 0.0],
        }
    }

    fn identity(name: &str, dim: usize) -> StoredLinear {
        let mut values = vec![0.0_f32; dim * dim];
        for i in 0..dim {
            values[i * dim + i] = 1.0;
        }
        StoredLinear {
            name: name.to_string(),
            input_dim: dim,
            output_dim: dim,
            values,
            native: None,
        }
    }
}
