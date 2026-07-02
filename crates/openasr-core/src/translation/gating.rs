#[derive(Debug, Clone, PartialEq)]
pub struct StabilityGateConfig {
    pub min_source_chars: usize,
    pub min_source_stable_ms: u64,
    pub min_asr_repeats: usize,
    pub min_emit_interval_ms: u64,
    pub max_emit_interval_ms: u64,
    pub max_similarity_without_refresh: f32,
}

impl Default for StabilityGateConfig {
    fn default() -> Self {
        Self {
            min_source_chars: 4,
            min_source_stable_ms: 180,
            min_asr_repeats: 2,
            min_emit_interval_ms: 180,
            max_emit_interval_ms: 600,
            max_similarity_without_refresh: 0.96,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StabilityGateInput<'a> {
    pub source_text: &'a str,
    pub observed_at_ms: u64,
    pub finalized: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StabilityGateDecision {
    pub should_enqueue: bool,
    pub reason: StabilityGateReason,
    pub stability: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityGateReason {
    Final,
    Ready,
    TooShort,
    UnstableTail,
    WaitingForRepeat,
    WaitingForStableTime,
    MinEmitInterval,
    TooSimilar,
    MaxIntervalRefresh,
}

#[derive(Debug, Clone)]
pub struct StabilityGate {
    config: StabilityGateConfig,
    last_source: Option<String>,
    stable_since_ms: u64,
    repeat_count: usize,
    last_emitted_source: Option<String>,
    last_emit_ms: Option<u64>,
}

impl Default for StabilityGate {
    fn default() -> Self {
        Self::new(StabilityGateConfig::default())
    }
}

impl StabilityGate {
    pub fn new(config: StabilityGateConfig) -> Self {
        Self {
            config,
            last_source: None,
            stable_since_ms: 0,
            repeat_count: 0,
            last_emitted_source: None,
            last_emit_ms: None,
        }
    }

    pub fn observe(&mut self, input: StabilityGateInput<'_>) -> StabilityGateDecision {
        let source = input.source_text.trim();
        if input.finalized {
            self.mark_emitted(source, input.observed_at_ms);
            return StabilityGateDecision {
                should_enqueue: true,
                reason: StabilityGateReason::Final,
                stability: 1.0,
            };
        }

        self.update_source_stability(source, input.observed_at_ms);
        let source_chars = source.chars().filter(|ch| !ch.is_whitespace()).count();
        if has_unstable_tail(source) {
            return self.decision(false, StabilityGateReason::UnstableTail, source_chars);
        }
        if source_chars < self.config.min_source_chars {
            return self.decision(false, StabilityGateReason::TooShort, source_chars);
        }

        let since_emit = self
            .last_emit_ms
            .map(|last| input.observed_at_ms.saturating_sub(last));
        let max_interval_due = matches!(
            since_emit,
            Some(elapsed) if elapsed >= self.config.max_emit_interval_ms
        );
        if max_interval_due {
            self.mark_emitted(source, input.observed_at_ms);
            return self.decision(true, StabilityGateReason::MaxIntervalRefresh, source_chars);
        }

        if self.repeat_count < self.config.min_asr_repeats {
            return self.decision(false, StabilityGateReason::WaitingForRepeat, source_chars);
        }
        let stable_for = input.observed_at_ms.saturating_sub(self.stable_since_ms);
        if stable_for < self.config.min_source_stable_ms {
            return self.decision(
                false,
                StabilityGateReason::WaitingForStableTime,
                source_chars,
            );
        }
        if matches!(
            since_emit,
            Some(elapsed) if elapsed < self.config.min_emit_interval_ms
        ) {
            return self.decision(false, StabilityGateReason::MinEmitInterval, source_chars);
        }
        if let Some(previous) = self.last_emitted_source.as_deref() {
            let similarity = source_similarity(previous, source);
            if similarity >= self.config.max_similarity_without_refresh {
                return StabilityGateDecision {
                    should_enqueue: false,
                    reason: StabilityGateReason::TooSimilar,
                    stability: stability_from_chars(source_chars),
                };
            }
        }

        self.mark_emitted(source, input.observed_at_ms);
        self.decision(true, StabilityGateReason::Ready, source_chars)
    }

    fn update_source_stability(&mut self, source: &str, observed_at_ms: u64) {
        if self.last_source.as_deref() == Some(source) {
            self.repeat_count = self.repeat_count.saturating_add(1);
            return;
        }
        self.last_source = Some(source.to_string());
        self.stable_since_ms = observed_at_ms;
        self.repeat_count = 1;
    }

    fn mark_emitted(&mut self, source: &str, observed_at_ms: u64) {
        self.last_emitted_source = Some(source.to_string());
        self.last_emit_ms = Some(observed_at_ms);
    }

    fn decision(
        &self,
        should_enqueue: bool,
        reason: StabilityGateReason,
        source_chars: usize,
    ) -> StabilityGateDecision {
        StabilityGateDecision {
            should_enqueue,
            reason,
            stability: stability_from_chars(source_chars),
        }
    }
}

fn has_unstable_tail(source: &str) -> bool {
    let trimmed = source.trim_end();
    let Some(last) = trimmed.chars().next_back() else {
        return true;
    };
    if last.is_ascii_alphabetic() {
        return true;
    }
    if last.is_ascii_digit() {
        let digit_count = trimmed
            .chars()
            .rev()
            .take_while(|ch| ch.is_ascii_digit())
            .count();
        return digit_count <= 1;
    }
    false
}

fn stability_from_chars(source_chars: usize) -> f32 {
    ((source_chars.min(12) as f32) / 12.0).clamp(0.0, 0.99)
}

fn source_similarity(left: &str, right: &str) -> f32 {
    if left == right {
        return 1.0;
    }
    let left_chars = left.chars().collect::<Vec<_>>();
    let right_chars = right.chars().collect::<Vec<_>>();
    if left_chars.is_empty() || right_chars.is_empty() {
        return 0.0;
    }
    let common_prefix = left_chars
        .iter()
        .zip(&right_chars)
        .take_while(|(left, right)| left == right)
        .count();
    (2.0 * common_prefix as f32) / (left_chars.len() + right_chars.len()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provisional_requires_repeats_and_stable_time() {
        let mut gate = StabilityGate::default();
        assert_eq!(
            gate.observe(StabilityGateInput {
                source_text: "我们需要",
                observed_at_ms: 0,
                finalized: false,
            })
            .reason,
            StabilityGateReason::WaitingForRepeat
        );
        assert_eq!(
            gate.observe(StabilityGateInput {
                source_text: "我们需要",
                observed_at_ms: 100,
                finalized: false,
            })
            .reason,
            StabilityGateReason::WaitingForStableTime
        );
        let ready = gate.observe(StabilityGateInput {
            source_text: "我们需要",
            observed_at_ms: 200,
            finalized: false,
        });
        assert!(ready.should_enqueue);
        assert_eq!(ready.reason, StabilityGateReason::Ready);
    }

    #[test]
    fn delays_latin_and_single_digit_tails() {
        let mut gate = StabilityGate::default();
        assert_eq!(
            gate.observe(StabilityGateInput {
                source_text: "版本 OpenAS",
                observed_at_ms: 200,
                finalized: false,
            })
            .reason,
            StabilityGateReason::UnstableTail
        );
        assert_eq!(
            gate.observe(StabilityGateInput {
                source_text: "第 3",
                observed_at_ms: 400,
                finalized: false,
            })
            .reason,
            StabilityGateReason::UnstableTail
        );
    }

    #[test]
    fn final_bypasses_provisional_gate() {
        let mut gate = StabilityGate::default();
        let decision = gate.observe(StabilityGateInput {
            source_text: "短",
            observed_at_ms: 0,
            finalized: true,
        });
        assert!(decision.should_enqueue);
        assert_eq!(decision.reason, StabilityGateReason::Final);
        assert_eq!(decision.stability, 1.0);
    }

    #[test]
    fn suppresses_too_similar_refresh_until_max_interval() {
        let mut gate = StabilityGate::new(StabilityGateConfig {
            min_source_stable_ms: 0,
            min_asr_repeats: 1,
            min_emit_interval_ms: 0,
            ..StabilityGateConfig::default()
        });
        assert!(
            gate.observe(StabilityGateInput {
                source_text: "我们需要保持流式路径",
                observed_at_ms: 0,
                finalized: false,
            })
            .should_enqueue
        );
        assert_eq!(
            gate.observe(StabilityGateInput {
                source_text: "我们需要保持流式路径",
                observed_at_ms: 100,
                finalized: false,
            })
            .reason,
            StabilityGateReason::TooSimilar
        );
        let refresh = gate.observe(StabilityGateInput {
            source_text: "我们需要保持流式路径",
            observed_at_ms: 700,
            finalized: false,
        });
        assert!(refresh.should_enqueue);
        assert_eq!(refresh.reason, StabilityGateReason::MaxIntervalRefresh);
    }
}
