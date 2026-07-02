use std::env;
use std::time::Instant;

pub(crate) const OPENASR_WHISPER_GGML_TRACE_ENV: &str = "OPENASR_WHISPER_GGML_TRACE";

pub(crate) const WHISPER_GGML_TRACE_DECODE_STEP_INTERVAL: usize = 16;

#[derive(Debug, Clone)]
pub(crate) struct WhisperGgmlTrace {
    enabled: bool,
    trace_start: Instant,
}

impl WhisperGgmlTrace {
    pub(crate) fn from_env() -> Self {
        let enabled = env::var(OPENASR_WHISPER_GGML_TRACE_ENV)
            .ok()
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                !normalized.is_empty()
                    && !matches!(normalized.as_str(), "0" | "false" | "off" | "no")
            })
            .unwrap_or(false);
        Self {
            enabled,
            trace_start: Instant::now(),
        }
    }

    pub(crate) fn run_stage<T, E, F>(&self, stage: &'static str, op: F) -> Result<T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        let span = self.start_stage(stage);
        let result = op();
        if result.is_ok() {
            span.finish_ok();
        } else {
            span.finish_err();
        }
        result
    }

    pub(crate) fn start_stage(&self, stage: &'static str) -> WhisperGgmlTraceStage<'_> {
        if self.enabled {
            self.emit(stage, "start", "na", None, "");
        }
        WhisperGgmlTraceStage {
            trace: self,
            stage,
            stage_start: Instant::now(),
        }
    }

    pub(crate) fn emit_decode_step_progress(
        &self,
        event: &'static str,
        step_index: usize,
        token_count: usize,
        completed_steps: usize,
        decode_loop_start: Instant,
    ) {
        if !self.enabled {
            return;
        }
        self.emit(
            "decode_loop",
            event,
            "ok",
            Some(decode_loop_start),
            &format!(
                "step_index={step_index} token_count={token_count} completed_steps={completed_steps} interval={WHISPER_GGML_TRACE_DECODE_STEP_INTERVAL}"
            ),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_decode_step_metrics(
        &self,
        status: &'static str,
        step_index: usize,
        token_count: usize,
        plan_cache_status: &'static str,
        plan_cache_hit: bool,
        plan_build_ms: u128,
        decoder_graph_run_ms: u128,
        logits_ms: u128,
        total_ms: u128,
        decode_loop_start: Instant,
    ) {
        if !self.enabled {
            return;
        }
        self.emit(
            "decode_step",
            "step_metrics",
            status,
            Some(decode_loop_start),
            &format!(
                "step_index={step_index} token_count={token_count} plan_cache={plan_cache_status} plan_cache_hit={} plan_cache_miss={} plan_build_ms={plan_build_ms} decoder_graph_run_ms={decoder_graph_run_ms} logits_ms={logits_ms} total_ms={total_ms}",
                usize::from(plan_cache_hit),
                usize::from(!plan_cache_hit),
            ),
        );
    }

    fn emit(
        &self,
        stage: &'static str,
        event: &'static str,
        status: &'static str,
        stage_start: Option<Instant>,
        extra: &str,
    ) {
        if !self.enabled {
            return;
        }
        let mut line = format!(
            "openasr_whisper_ggml_trace stage={stage} event={event} status={status} t_ms={}",
            self.trace_start.elapsed().as_millis()
        );
        if let Some(started_at) = stage_start {
            line.push_str(&format!(" dt_ms={}", started_at.elapsed().as_millis()));
        }
        if !extra.is_empty() {
            line.push(' ');
            line.push_str(extra);
        }
        eprintln!("{line}");
    }
}

#[derive(Debug)]
pub(crate) struct WhisperGgmlTraceStage<'a> {
    trace: &'a WhisperGgmlTrace,
    stage: &'static str,
    stage_start: Instant,
}

impl WhisperGgmlTraceStage<'_> {
    pub(crate) fn finish_ok(self) {
        self.trace
            .emit(self.stage, "end", "ok", Some(self.stage_start), "");
    }

    pub(crate) fn finish_err(self) {
        self.trace
            .emit(self.stage, "end", "err", Some(self.stage_start), "");
    }

    pub(crate) fn finish_with_extra(self, status: &'static str, extra: &str) {
        self.trace
            .emit(self.stage, "end", status, Some(self.stage_start), extra);
    }
}
