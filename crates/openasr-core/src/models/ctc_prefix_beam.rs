//! CTC prefix-beam search with a hotword context graph.
//!
//! The frame-argmax greedy path (`ctc_greedy_decode`) is a maximum-*alignment*
//! decoder: it cannot trade a locally-worse frame for a globally-better label
//! sequence, so it cannot be biased toward a hotword without the per-frame logit
//! nudges that (measured) either do nothing or shred the transcript. This module
//! is the maximum-*label-sequence* decoder that hotwords actually need: a
//! log-domain CTC prefix-beam search (blank/non-blank prefix probabilities merged
//! per the standard Graves recurrence) whose beam scores carry an independent
//! CONTEXT score driven by an Aho-Corasick context graph over the hotword token
//! sequences (sherpa-onnx / icefall / WeNet share this shape; the code here is an
//! independent reimplementation).
//!
//! The context graph rewards a beam for advancing along a hotword one token at a
//! time and REFUNDS the reward when a partial match breaks (the reward is stored
//! relative to the graph node's accumulated score, so a fail-transition to a
//! shallower node subtracts exactly the boost granted for the now-broken prefix).
//! A completed hotword banks a permanent completion bonus. Because the boost lives
//! in a path-level score rather than in the per-frame logits, a same-sounding
//! competitor (`刁天成` vs the hotword `刁天宸`) is beaten by accumulated path
//! evidence rather than by railroading a single frame's argmax.
//!
//! This path runs ONLY when the caller supplies a non-empty context graph; with
//! no hotwords the CTC decode stays on the untouched greedy path, so the no-bias
//! behavior is byte-for-byte unchanged and there is zero performance regression.

use std::collections::HashMap;

use crate::models::ctc_greedy_decode::{
    CtcGreedyDecodeError, CtcGreedyDecodeResult, CtcTokenFrameSpan,
};
use crate::models::phrase_bias_decode::TokenPhraseBias;

/// Beam width. CTC posteriors are peaky, so a handful of hypotheses already
/// covers every path with non-negligible mass; the width only has to be large
/// enough to keep the hotword hypothesis AND its same-sounding competitor alive
/// side by side until the accumulated context score separates them. 8 is the
/// upper end of the 4-8 range shared by sherpa-onnx / icefall for streaming CTC,
/// chosen for that homophone headroom; the per-frame cost is trivial next to the
/// encoder matmul that produced the logits.
const CTC_PREFIX_BEAM_WIDTH: usize = 8;

/// How many top acoustic tokens each frame contributes as expansion candidates.
/// The full hotword token set is ALWAYS a candidate on top of this (so a hotword
/// continuation is explorable even when its acoustic rank is low), so this only
/// has to cover the ordinary competing labels; 8 keeps the frame work bounded on
/// large vocabularies (sensevoice ~25k) while never dropping a plausible label.
const CTC_ACOUSTIC_CANDIDATE_TOP_K: usize = 8;

/// Aho-Corasick context graph over hotword token-id sequences.
///
/// Node 0 is the root. `node_score[n]` is the accumulated boost of the token
/// prefix that reaches `n` from the root; the per-edge boost is the strongest
/// (largest-magnitude, mirroring the greedy path's collision rule) opinion of any
/// hotword whose prefix uses that edge. `output_score[n]` is the extra completion
/// bonus realized on arrival at `n` (its own end-of-word bonus plus those of every
/// shorter hotword ending here through the fail chain).
pub(crate) struct CtcContextGraph {
    goto: Vec<HashMap<u32, usize>>,
    fail: Vec<usize>,
    node_score: Vec<f32>,
    output_score: Vec<f32>,
    /// Every token id that labels an edge anywhere in the graph -- the tokens a
    /// beam must always be allowed to explore regardless of acoustic rank.
    edge_tokens: Vec<u32>,
}

impl CtcContextGraph {
    /// Build the context graph from the resolved per-phrase token biases. Returns
    /// `None` when there is nothing to bias (no phrases / only empty variants), so
    /// the caller stays on the greedy path. Only POSITIVE boosts build context
    /// (a hotword to favor); negative "anti-context" biases are not representable
    /// as a prefix-beam reward here and are ignored on the CTC path.
    pub(crate) fn from_token_phrase_biases(biases: &[TokenPhraseBias]) -> Option<Self> {
        let mut goto: Vec<HashMap<u32, usize>> = vec![HashMap::new()];
        let mut token_score: Vec<f32> = vec![0.0];
        let mut is_end_bonus: Vec<f32> = vec![0.0];

        for bias in biases {
            let boost = bias.boost();
            if boost <= 0.0 {
                continue;
            }
            for variant in bias.variants() {
                if variant.is_empty() {
                    continue;
                }
                let mut node = 0usize;
                for &token in variant {
                    node = match goto[node].get(&token).copied() {
                        Some(child) => {
                            // Strongest opinion wins on a shared edge (same rule
                            // the greedy path used for colliding boosts).
                            if boost > token_score[child] {
                                token_score[child] = boost;
                            }
                            child
                        }
                        None => {
                            let child = goto.len();
                            goto.push(HashMap::new());
                            token_score.push(boost);
                            is_end_bonus.push(0.0);
                            goto[node].insert(token, child);
                            child
                        }
                    };
                }
                // Completion bonus for the whole hotword = its accumulated boost,
                // realized once on arrival at the end node. Strongest opinion when
                // two hotwords share an end node.
                let end_word_boost = boost * variant.len() as f32;
                if end_word_boost > is_end_bonus[node] {
                    is_end_bonus[node] = end_word_boost;
                }
            }
        }

        if goto.len() == 1 {
            return None;
        }

        let node_count = goto.len();
        // node_score in BFS (increasing depth) order: child = parent + edge.
        let mut node_score = vec![0.0f32; node_count];
        // Collect edge tokens and a BFS order.
        let mut edge_tokens = Vec::new();
        let mut order = Vec::with_capacity(node_count);
        let mut queue = std::collections::VecDeque::new();
        for (&token, &child) in &goto[0] {
            edge_tokens.push(token);
            node_score[child] = token_score[child];
            queue.push_back(child);
        }
        while let Some(node) = queue.pop_front() {
            order.push(node);
            let children: Vec<(u32, usize)> = goto[node].iter().map(|(&t, &c)| (t, c)).collect();
            for (token, child) in children {
                edge_tokens.push(token);
                node_score[child] = node_score[node] + token_score[child];
                queue.push_back(child);
            }
        }
        edge_tokens.sort_unstable();
        edge_tokens.dedup();

        // Aho-Corasick fail links via BFS from the root's children.
        let mut fail = vec![0usize; node_count];
        let mut bfs = std::collections::VecDeque::new();
        for &child in goto[0].values() {
            fail[child] = 0;
            bfs.push_back(child);
        }
        while let Some(node) = bfs.pop_front() {
            let children: Vec<(u32, usize)> = goto[node].iter().map(|(&t, &c)| (t, c)).collect();
            for (token, child) in children {
                // fail(child) = deepest proper suffix that is a graph prefix.
                let mut f = fail[node];
                loop {
                    if let Some(&next) = goto[f].get(&token) {
                        fail[child] = next;
                        break;
                    }
                    if f == 0 {
                        fail[child] = 0;
                        break;
                    }
                    f = fail[f];
                }
                bfs.push_back(child);
            }
        }

        // output_score(n) = own end bonus + output_score(fail(n)); computed in
        // BFS (shallow-to-deep) order so fail(n) is always resolved first.
        let mut output_score = vec![0.0f32; node_count];
        for &node in &order {
            output_score[node] = is_end_bonus[node] + output_score[fail[node]];
        }

        Some(Self {
            goto,
            fail,
            node_score,
            output_score,
            edge_tokens,
        })
    }

    fn root(&self) -> usize {
        0
    }

    /// Advance one token from `node`, returning the destination node and the
    /// context-score DELTA to add to the beam. The delta is
    /// `node_score[dest] - node_score[node] + output_score[dest]`: a match climbs
    /// (positive per-token boost), a broken prefix fails back to a shallower node
    /// (negative delta = refund of the boost granted for the broken prefix), and a
    /// completed hotword adds its one-shot completion bonus via `output_score`.
    fn step(&self, node: usize, token: u32) -> (usize, f32) {
        let mut cur = node;
        let dest = loop {
            if let Some(&next) = self.goto[cur].get(&token) {
                break next;
            }
            if cur == 0 {
                break 0;
            }
            cur = self.fail[cur];
        };
        let delta = self.node_score[dest] - self.node_score[node] + self.output_score[dest];
        (dest, delta)
    }
}

/// One emitted token on a beam's winning alignment: the frame it was first
/// appended and the log-posterior seen there (surfaced as per-token confidence).
#[derive(Debug, Clone)]
struct TokenEmit {
    token_id: u32,
    start_frame: usize,
    log_prob: f32,
}

#[derive(Debug, Clone)]
struct Beam {
    /// log P(prefix, last alignment step was blank).
    p_blank: f32,
    /// log P(prefix, last alignment step was a non-blank label).
    p_non_blank: f32,
    context_node: usize,
    context_score: f32,
    emits: Vec<TokenEmit>,
}

impl Beam {
    fn total_log_prob(&self) -> f32 {
        log_sum_exp2(self.p_blank, self.p_non_blank)
    }

    /// Score used for pruning and final selection: acoustic prefix mass plus the
    /// independent context reward.
    fn score(&self) -> f32 {
        self.total_log_prob() + self.context_score
    }
}

/// Run a log-domain CTC prefix-beam search biased by `graph`. `frame_logits[t]`
/// is the length-`vocab_size` logit row for frame `t`. Produces the same
/// `CtcGreedyDecodeResult` shape as the greedy path (token ids, per-token frame
/// spans, frame count, detokenized text) so every downstream consumer -- word
/// timestamps, per-word confidence, streaming -- is unchanged.
pub(crate) fn run_ctc_prefix_beam_decode<E>(
    blank_token_id: u32,
    vocab_size: usize,
    graph: &CtcContextGraph,
    frame_logits: &[&[f32]],
    decode_text_token_ids: impl Fn(&[u32]) -> Result<String, E>,
    map_err: impl Fn(E) -> CtcGreedyDecodeError,
) -> Result<CtcGreedyDecodeResult, CtcGreedyDecodeError> {
    if frame_logits.is_empty() {
        return Err(CtcGreedyDecodeError::EmptyFrames);
    }
    let blank = blank_token_id as usize;

    // Beams keyed by the collapsed label prefix. Iterate keys in a deterministic
    // (sorted) order every frame so timestamps/first-write-wins merges are stable.
    let mut beams: HashMap<Vec<u32>, Beam> = HashMap::new();
    beams.insert(
        Vec::new(),
        Beam {
            p_blank: 0.0, // log(1): CTC begins in the blank state.
            p_non_blank: f32::NEG_INFINITY,
            context_node: graph.root(),
            context_score: 0.0,
            emits: Vec::new(),
        },
    );

    let mut log_probs = vec![0.0f32; vocab_size];
    for (frame, &row) in frame_logits.iter().enumerate() {
        if row.len() != vocab_size {
            return Err(CtcGreedyDecodeError::FrameWidthMismatch {
                frame,
                got: row.len(),
                expected: vocab_size,
            });
        }
        log_softmax_into(row, &mut log_probs, frame)?;
        let blank_lp = log_probs[blank];
        let candidates = frame_candidate_tokens(&log_probs, blank, graph);

        let mut next: HashMap<Vec<u32>, Beam> = HashMap::new();
        let mut ordered: Vec<(&Vec<u32>, &Beam)> = beams.iter().collect();
        ordered.sort_by(|a, b| a.0.cmp(b.0));

        for (prefix, beam) in ordered {
            // (1) blank -> same prefix, folded into p_blank.
            {
                let entry = next
                    .entry(prefix.clone())
                    .or_insert_with(|| carry_beam(beam));
                entry.p_blank = log_sum_exp3(
                    entry.p_blank,
                    beam.p_blank + blank_lp,
                    beam.p_non_blank + blank_lp,
                );
            }

            // (2) repeat the last label without a blank between -> same prefix,
            //     into p_non_blank.
            if let Some(&last) = prefix.last() {
                let last_lp = log_probs[last as usize];
                let entry = next
                    .entry(prefix.clone())
                    .or_insert_with(|| carry_beam(beam));
                entry.p_non_blank = log_sum_exp2(entry.p_non_blank, beam.p_non_blank + last_lp);
            }

            // (3) extend by each candidate label.
            for &token in &candidates {
                let token_lp = log_probs[token as usize];
                let mut extended = prefix.clone();
                extended.push(token);
                let is_repeat = prefix.last() == Some(&token);
                let entry = next
                    .entry(extended)
                    .or_insert_with(|| extend_beam(beam, graph, token, frame, token_lp));
                // A label identical to the prefix's last one can only start a new
                // token when a blank separated them (p_blank), else it merges into
                // the repeat case above.
                let contribution = if is_repeat {
                    beam.p_blank + token_lp
                } else {
                    log_sum_exp2(beam.p_blank, beam.p_non_blank) + token_lp
                };
                entry.p_non_blank = log_sum_exp2(entry.p_non_blank, contribution);
            }
        }

        beams = prune_beams(next, CTC_PREFIX_BEAM_WIDTH);
    }

    let best = beams
        .into_iter()
        .max_by(|a, b| {
            a.1.score()
                .partial_cmp(&b.1.score())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        })
        .map(|(_, beam)| beam)
        .ok_or(CtcGreedyDecodeError::EmptyFrames)?;

    let frame_count = frame_logits.len();
    let token_ids: Vec<u32> = best.emits.iter().map(|emit| emit.token_id).collect();
    let mut token_spans = Vec::with_capacity(best.emits.len());
    for (index, emit) in best.emits.iter().enumerate() {
        // Cover this token's frame span up to the next token's onset (or the end
        // of the utterance) so downstream word-timestamp assembly sees contiguous,
        // monotonic frames, matching the greedy run-based spans.
        let end_frame = best
            .emits
            .get(index + 1)
            .map(|next_emit| next_emit.start_frame.max(emit.start_frame + 1))
            .unwrap_or(frame_count)
            .max(emit.start_frame + 1);
        token_spans.push(CtcTokenFrameSpan {
            token_id: emit.token_id,
            start_frame: emit.start_frame,
            end_frame,
            probability: emit.log_prob.exp().clamp(0.0, 1.0),
        });
    }

    let text = decode_text_token_ids(&token_ids).map_err(map_err)?;
    Ok(CtcGreedyDecodeResult {
        token_ids,
        token_spans,
        frame_count,
        text,
    })
}

/// Carry a beam forward under a same-prefix transition (blank / repeat): context
/// state and emitted alignment are unchanged; only the probabilities accumulate.
fn carry_beam(beam: &Beam) -> Beam {
    Beam {
        p_blank: f32::NEG_INFINITY,
        p_non_blank: f32::NEG_INFINITY,
        context_node: beam.context_node,
        context_score: beam.context_score,
        emits: beam.emits.clone(),
    }
}

/// Create the beam for a one-token extension: step the context graph, add the
/// score delta, and append the emitted token at this frame.
///
/// The context state (node + accumulated score) is a DETERMINISTIC function of the
/// prefix's token sequence -- the Aho-Corasick automaton run over the tokens.
/// This is load-bearing: beams are keyed and merged by prefix, so two alignments
/// reaching the same label prefix MUST agree on its context, or a first-write-wins
/// merge would let one alignment's context clobber the other's. Any acoustic,
/// per-frame ("is this token plausible here") gating of the context reward would
/// make the context path-dependent and break that invariant. Over-insertion at
/// large boosts is held off instead at the CANDIDATE stage (see
/// [`frame_candidate_tokens`]): a hotword token is only force-explored on a frame
/// whose acoustics support it, which controls WHICH prefixes exist without
/// perturbing the context OF any prefix.
fn extend_beam(
    parent: &Beam,
    graph: &CtcContextGraph,
    token: u32,
    frame: usize,
    token_lp: f32,
) -> Beam {
    let (context_node, context_delta) = graph.step(parent.context_node, token);
    let mut emits = parent.emits.clone();
    emits.push(TokenEmit {
        token_id: token,
        start_frame: frame,
        log_prob: token_lp,
    });
    Beam {
        p_blank: f32::NEG_INFINITY,
        p_non_blank: f32::NEG_INFINITY,
        context_node,
        context_score: parent.context_score + context_delta,
        emits,
    }
}

/// Keep the `width` highest-scoring beams (score = acoustic mass + context).
fn prune_beams(next: HashMap<Vec<u32>, Beam>, width: usize) -> HashMap<Vec<u32>, Beam> {
    if next.len() <= width {
        return next;
    }
    let mut scored: Vec<(Vec<u32>, Beam)> = next.into_iter().collect();
    scored.sort_by(|a, b| {
        b.1.score()
            .partial_cmp(&a.1.score())
            .unwrap_or(std::cmp::Ordering::Equal)
            // Deterministic tie-break so pruning is reproducible.
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(width);
    scored.into_iter().collect()
}

/// How far (in log-prob, == logit units) a hotword edge token may trail the
/// frame's best acoustic token and still be force-explored as a candidate. The
/// same-sounding substitution the feature exists for (the measured
/// `刁天成` -> `刁天宸` case sits 7-10 logits behind at the decisive character)
/// stays inside this margin, so the hotword continuation is always explored where
/// it plausibly occurs; a hotword token with no acoustic support at a frame (a
/// silence/blank frame, an unrelated word) falls outside it and is NOT
/// force-explored, so even a max (20.0) boost cannot hallucinate a hotword into
/// frames that do not sound like it. Shares the value and rationale of the greedy
/// path's `CONTINUATION_PLAUSIBILITY_MARGIN`. Gating exploration (not the context
/// score) keeps each prefix's context a deterministic function of its tokens.
const CTC_CANDIDATE_PLAUSIBILITY_MARGIN: f32 = 12.0;

/// Candidate expansion labels for one frame: the top-K acoustic tokens (blank
/// excluded -- it is handled by the same-prefix branch) unioned with every hotword
/// edge token that is acoustically plausible at this frame (within
/// [`CTC_CANDIDATE_PLAUSIBILITY_MARGIN`] of the frame's best token). The union
/// keeps a low-acoustic-rank-but-plausible hotword continuation explorable; the
/// plausibility filter keeps a boost from forcing a hotword token into a frame
/// that does not support it.
fn frame_candidate_tokens(log_probs: &[f32], blank: usize, graph: &CtcContextGraph) -> Vec<u32> {
    let mut frame_max = f32::NEG_INFINITY;
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(CTC_ACOUSTIC_CANDIDATE_TOP_K + 1);
    for (index, &lp) in log_probs.iter().enumerate() {
        if lp > frame_max {
            frame_max = lp;
        }
        if index == blank {
            continue;
        }
        if top.len() < CTC_ACOUSTIC_CANDIDATE_TOP_K {
            top.push((index, lp));
            if top.len() == CTC_ACOUSTIC_CANDIDATE_TOP_K {
                top.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        } else if lp > top[0].1 {
            top[0] = (index, lp);
            top.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    let plausibility_floor = frame_max - CTC_CANDIDATE_PLAUSIBILITY_MARGIN;
    let mut tokens: Vec<u32> = top.iter().map(|(index, _)| *index as u32).collect();
    for &token in &graph.edge_tokens {
        let index = token as usize;
        if index != blank && index < log_probs.len() && log_probs[index] >= plausibility_floor {
            tokens.push(token);
        }
    }
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

/// log-softmax of `row` into `out`, fail-closed on any non-finite logit (matching
/// the greedy path's `NonFiniteLogits`).
fn log_softmax_into(
    row: &[f32],
    out: &mut [f32],
    frame: usize,
) -> Result<(), CtcGreedyDecodeError> {
    let mut max = f32::NEG_INFINITY;
    for &value in row {
        if !value.is_finite() {
            return Err(CtcGreedyDecodeError::NonFiniteLogits { frame });
        }
        if value > max {
            max = value;
        }
    }
    let mut sum = 0.0f32;
    for &value in row {
        sum += (value - max).exp();
    }
    let log_sum = max + sum.ln();
    for (slot, &value) in out.iter_mut().zip(row.iter()) {
        *slot = value - log_sum;
    }
    Ok(())
}

fn log_sum_exp2(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

fn log_sum_exp3(a: f32, b: f32, c: f32) -> f32 {
    log_sum_exp2(log_sum_exp2(a, b), c)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB: usize = 6; // ids 0..=4 real, 5 = blank
    const BLANK: u32 = 5;

    fn graph(biases: &[(Vec<u32>, f32)]) -> CtcContextGraph {
        let biases: Vec<TokenPhraseBias> = biases
            .iter()
            .map(|(tokens, boost)| TokenPhraseBias::new(vec![tokens.clone()], *boost).unwrap())
            .collect();
        CtcContextGraph::from_token_phrase_biases(&biases).expect("non-empty graph")
    }

    /// One-hot-ish logit row peaking at `id` with margin `margin`.
    fn frame(id: u32, margin: f32) -> Vec<f32> {
        let mut row = vec![0.0f32; VOCAB];
        row[id as usize] = margin;
        row
    }

    fn decode_ids(ids: &[u32]) -> Result<String, std::convert::Infallible> {
        Ok(ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(","))
    }

    fn run(
        g: &CtcContextGraph,
        rows: &[Vec<f32>],
    ) -> Result<CtcGreedyDecodeResult, CtcGreedyDecodeError> {
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        run_ctc_prefix_beam_decode(BLANK, VOCAB, g, &refs, decode_ids, |never| match never {})
    }

    /// Un-biased greedy decode (the baseline every hotword flip is measured
    /// against).
    fn greedy(rows: &[Vec<f32>]) -> Vec<u32> {
        use crate::models::ctc_greedy_decode::{CtcGreedyDecodeConfig, run_ctc_greedy_decode};
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        run_ctc_greedy_decode(
            CtcGreedyDecodeConfig {
                blank_token_id: BLANK,
                vocab_size: VOCAB,
                phrase_biases: Vec::new(),
            },
            &refs,
            decode_ids,
            |never| match never {},
        )
        .unwrap()
        .token_ids
    }

    #[test]
    fn empty_graph_is_none() {
        assert!(CtcContextGraph::from_token_phrase_biases(&[]).is_none());
        let neg = TokenPhraseBias::new(vec![vec![1]], -5.0).unwrap();
        // A purely-negative bias contributes no positive context edges.
        assert!(CtcContextGraph::from_token_phrase_biases(std::slice::from_ref(&neg)).is_none());
    }

    #[test]
    fn context_graph_multi_hotword_shared_prefix_scores() {
        // Two hotwords sharing the prefix token 1: [1,2] and [1,3].
        let g = graph(&[(vec![1, 2], 4.0), (vec![1, 3], 6.0)]);
        // From root, stepping 1 climbs; the shared edge takes the strongest boost.
        let (n1, d1) = g.step(g.root(), 1);
        assert_eq!(d1, 6.0); // strongest opinion on the shared edge
        let (_n12, d2) = g.step(n1, 2);
        // Edge into [1,2] climbs 4 and completes the hotword, realizing its
        // completion bonus (boost * len = 4 * 2 = 8): 4 + 8 = 12.
        assert_eq!(d2, 12.0);
        // Completing [1,2] realizes its completion bonus (boost * len = 4*2 = 8)
        // on top of the per-token climb.
        let (n1b, _) = g.step(g.root(), 1);
        let (_n13, d3) = g.step(n1b, 3);
        // edge climb 6 + completion bonus (6*2=12) = 18.
        assert_eq!(d3, 6.0 + 12.0);
    }

    #[test]
    fn context_graph_refunds_a_broken_partial_match() {
        // Hotword [1,2,3]; after matching 1,2 (climbed 2*boost) a mismatching
        // token that is not in the graph fails back to the root and REFUNDS the
        // accumulated boost.
        let g = graph(&[(vec![1, 2, 3], 5.0)]);
        let (n1, a) = g.step(g.root(), 1);
        let (n2, b) = g.step(n1, 2);
        assert_eq!(a, 5.0);
        assert_eq!(b, 5.0);
        // token 4 is not in the graph: fail back to root, delta = -node_score(n2).
        let (dest, refund) = g.step(n2, 4);
        assert_eq!(dest, g.root());
        assert_eq!(refund, -10.0); // gives back the 2*5 granted for [1,2]
    }

    #[test]
    fn context_graph_overlapping_suffix_uses_fail_link() {
        // Hotwords [1,2,3] and [2,3]: after [1,2] a 3 completes [1,2,3]; but a
        // fail-link also lets a bare [2,3] be recognized. Verify a [2,3] match
        // banks its completion bonus via the fail/output chain.
        let g = graph(&[(vec![1, 2, 3], 3.0), (vec![2, 3], 7.0)]);
        // Walk 1 -> 2: at node for [1,2], fail link points at node [2].
        let (n1, _) = g.step(g.root(), 1);
        let (_n12, _) = g.step(n1, 2);
        // Independently walking 2 -> 3 from root completes [2,3].
        let (n2, _) = g.step(g.root(), 2);
        let (_n23, d) = g.step(n2, 3);
        // climb 7 + completion bonus (7*2 = 14).
        assert_eq!(d, 7.0 + 14.0);
    }

    #[test]
    fn beam_matches_greedy_with_a_no_op_hotword_on_peaky_logits() {
        // Consistency: on peaky one-hot logits the prefix beam collapses to the
        // greedy label sequence. The hotword here (token 4) never appears in the
        // audio, so the context score is inert and the result is the greedy one.
        let g = graph(&[(vec![4], 5.0)]);
        let rows = [
            frame(1, 12.0),
            frame(1, 12.0),
            frame(BLANK, 12.0),
            frame(2, 12.0),
            frame(3, 12.0),
        ];
        let r = run(&g, &rows).unwrap();
        assert_eq!(r.token_ids, vec![1, 2, 3]);
        assert_eq!(r.text, "1,2,3");
        assert_eq!(r.frame_count, 5);
    }

    #[test]
    fn hotword_flips_a_close_acoustic_competitor() {
        // Frame 0 barely prefers token 2 over token 3 (a homophone-style near-tie);
        // frames then settle. Without a hotword the decode picks 2; a hotword [3]
        // with a healthy boost flips the winning label to 3 via path evidence.
        let near_tie = {
            let mut row = vec![0.0f32; VOCAB];
            row[2] = 1.2; // acoustic winner
            row[3] = 1.0; // hotword continuation, just behind
            row
        };
        let rows = [near_tie, frame(BLANK, 12.0)];

        assert_eq!(greedy(&rows), vec![2]);

        let hot = graph(&[(vec![3], 5.0)]);
        let biased = run(&hot, &rows).unwrap();
        assert_eq!(biased.token_ids, vec![3]);
    }

    #[test]
    fn multi_token_hotword_beats_homophone_on_the_last_token() {
        // Hotword [1,2,3]. Frames strongly emit 1 then 2; the last frame narrowly
        // prefers a homophone token 4 over the hotword's final token 3. The
        // accumulated + completion context score must carry 3 to victory.
        let last = {
            let mut row = vec![0.0f32; VOCAB];
            row[4] = 2.0; // homophone acoustic winner
            row[3] = 0.0; // hotword final token, behind by 2 logits
            row
        };
        let rows = [
            frame(1, 12.0),
            frame(BLANK, 12.0),
            frame(2, 12.0),
            frame(BLANK, 12.0),
            last,
        ];

        assert_eq!(greedy(&rows), vec![1, 2, 4]);

        let hot = graph(&[(vec![1, 2, 3], 5.0)]);
        let biased = run(&hot, &rows).unwrap();
        assert_eq!(biased.token_ids, vec![1, 2, 3]);
    }

    #[test]
    fn produces_monotonic_token_spans_and_confidences() {
        let g = graph(&[(vec![4], 5.0)]);
        let rows = [
            frame(1, 12.0),
            frame(1, 12.0),
            frame(BLANK, 12.0),
            frame(2, 12.0),
        ];
        let r = run(&g, &rows).unwrap();
        assert_eq!(r.token_ids, vec![1, 2]);
        assert_eq!(r.token_spans.len(), 2);
        // Spans are contiguous and monotonic; the last runs to frame_count.
        assert!(r.token_spans[0].start_frame <= r.token_spans[0].end_frame);
        assert!(r.token_spans[0].end_frame <= r.token_spans[1].start_frame);
        assert_eq!(r.token_spans[1].end_frame, r.frame_count);
        for span in &r.token_spans {
            assert!((0.0..=1.0).contains(&span.probability));
        }
    }

    #[test]
    fn rejects_non_finite_logits() {
        let g = graph(&[(vec![4], 5.0)]);
        let mut bad = frame(1, 12.0);
        bad[2] = f32::NAN;
        assert_eq!(
            run(&g, &[frame(1, 12.0), bad]),
            Err(CtcGreedyDecodeError::NonFiniteLogits { frame: 1 })
        );
    }

    #[test]
    fn rejects_empty_frames() {
        let g = graph(&[(vec![4], 5.0)]);
        assert_eq!(run(&g, &[]), Err(CtcGreedyDecodeError::EmptyFrames));
    }

    #[test]
    fn rejects_frame_width_mismatch() {
        let g = graph(&[(vec![4], 5.0)]);
        let good = frame(1, 12.0);
        let short = vec![0.0f32; VOCAB - 1];
        let refs: [&[f32]; 2] = [good.as_slice(), short.as_slice()];
        assert_eq!(
            run_ctc_prefix_beam_decode(BLANK, VOCAB, &g, &refs, decode_ids, |never| match never {}),
            Err(CtcGreedyDecodeError::FrameWidthMismatch {
                frame: 1,
                got: VOCAB - 1,
                expected: VOCAB
            })
        );
    }

    #[test]
    fn detokenize_error_is_propagated_fail_closed() {
        let g = graph(&[(vec![4], 5.0)]);
        let row = frame(1, 12.0);
        let refs: [&[f32]; 1] = [row.as_slice()];
        let result = run_ctc_prefix_beam_decode(
            BLANK,
            VOCAB,
            &g,
            &refs,
            |_ids| Err("tokenizer exploded".to_string()),
            |reason| CtcGreedyDecodeError::DetokenizeFailed { reason },
        );
        assert_eq!(
            result,
            Err(CtcGreedyDecodeError::DetokenizeFailed {
                reason: "tokenizer exploded".to_string()
            })
        );
    }
}
