//! Stage-weighted file-transcription progress.
//!
//! Two layers, both driven only by real pipeline events:
//! 1. **Stage fraction** -- real completion of the stage currently running
//!    (window/batch/sample/token/segment progress when available; otherwise
//!    indeterminate, never a fabricated climb).
//! 2. **Overall fraction** -- planned-stage costs (profile-estimated effort)
//!    weighted so the bar reflects remaining work mix, not fixed stage
//!    percentages.
//!
//! Without a real event the overall fraction **must not increase**. The
//! registry only advances when a stage is entered, a sub-progress report
//! lands, or a stage completes. Time alone never moves the bar.
//!
//! Wire compatibility: [`NativeTranscriptionProgress::fraction`] equals
//! `overall_fraction`, and [`NativeTranscriptionPhase`] is the nearest legacy
//! label for the current stage (decode / assemble / align).

use std::sync::Mutex;

use crate::config::VoiceIdSegmenterPreference;

/// Bound on concurrent progress entries (same capacity rationale as the
/// historical progress registry: in-flight runs only; oldest-entry eviction
/// if a handle leaks).
const PROGRESS_REGISTRY_CAPACITY: usize = 64;

/// Coarse legacy phase labels kept for pre-stage-aware clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeTranscriptionPhase {
    /// Decoding audio slices (also covers prepare / load / diarize / identity).
    Decode,
    /// Post-decode assembly-adjacent work (punctuate / project / persist).
    Assemble,
    /// Forced-align refine (word timestamps).
    Align,
}

impl NativeTranscriptionPhase {
    /// Stable lowercase label for the wire contract and optional UI phase text.
    pub fn label(self) -> &'static str {
        match self {
            NativeTranscriptionPhase::Decode => "decode",
            NativeTranscriptionPhase::Assemble => "assemble",
            NativeTranscriptionPhase::Align => "align",
        }
    }
}

/// Pipeline stage names that can appear in a progress plan. Stages that will
/// not run for a request must not be present in that request's plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TranscriptionStage {
    Prepare,
    LoadModel,
    Diarize,
    IdentifySpeakers,
    Decode,
    Punctuate,
    Align,
    Project,
    Persist,
}

impl TranscriptionStage {
    /// Stable snake_case wire label.
    pub fn label(self) -> &'static str {
        match self {
            TranscriptionStage::Prepare => "prepare",
            TranscriptionStage::LoadModel => "load_model",
            TranscriptionStage::Diarize => "diarize",
            TranscriptionStage::IdentifySpeakers => "identify_speakers",
            TranscriptionStage::Decode => "decode",
            TranscriptionStage::Punctuate => "punctuate",
            TranscriptionStage::Align => "align",
            TranscriptionStage::Project => "project",
            TranscriptionStage::Persist => "persist",
        }
    }

    /// Nearest legacy [`NativeTranscriptionPhase`] for clients that only read
    /// `phase`.
    pub fn legacy_phase(self) -> NativeTranscriptionPhase {
        match self {
            TranscriptionStage::Prepare
            | TranscriptionStage::LoadModel
            | TranscriptionStage::Diarize
            | TranscriptionStage::IdentifySpeakers
            | TranscriptionStage::Decode => NativeTranscriptionPhase::Decode,
            TranscriptionStage::Punctuate
            | TranscriptionStage::Project
            | TranscriptionStage::Persist => NativeTranscriptionPhase::Assemble,
            TranscriptionStage::Align => NativeTranscriptionPhase::Align,
        }
    }
}

/// Snapshot of one in-flight native transcription (or post-hoc refine).
#[derive(Debug, Clone, PartialEq)]
pub struct NativeTranscriptionProgress {
    /// Legacy coarse phase (mapped from [`Self::stage`]).
    pub phase: NativeTranscriptionPhase,
    /// Legacy overall progress in `0..=1` (= [`Self::overall_fraction`]).
    pub fraction: f32,
    /// Current pipeline stage.
    pub stage: TranscriptionStage,
    /// Real completion of the current stage in `0..=1`, or `None` when the
    /// stage is indeterminate (no honest sub-progress available yet).
    pub stage_fraction: Option<f32>,
    /// Optional unit counters (windows, samples, segments, ...).
    pub completed_units: Option<u64>,
    pub total_units: Option<u64>,
    /// Cost-weighted overall completion in `0..=1`.
    pub overall_fraction: f32,
    /// True when the current stage has no honest fraction yet.
    pub indeterminate: bool,
    /// Optional human-readable detail (not required by the wire contract).
    pub detail: Option<String>,
}

impl NativeTranscriptionProgress {
    /// Construct a fully-specified snapshot (tests + internal publishers).
    pub fn new(
        stage: TranscriptionStage,
        stage_fraction: Option<f32>,
        overall_fraction: f32,
        completed_units: Option<u64>,
        total_units: Option<u64>,
        detail: Option<String>,
    ) -> Self {
        let overall = overall_fraction.clamp(0.0, 1.0);
        let indeterminate = stage_fraction.is_none();
        Self {
            phase: stage.legacy_phase(),
            fraction: overall,
            stage,
            stage_fraction,
            completed_units,
            total_units,
            overall_fraction: overall,
            indeterminate,
            detail,
        }
    }
}

/// Legacy (pre-multi-request) id-less progress read.
#[derive(Debug, Clone, PartialEq)]
pub enum LegacyNativeTranscriptionProgress {
    Idle,
    Single(NativeTranscriptionProgress),
    Ambiguous { active_count: usize },
}

/// Coarse backend class for the first-cut cost profile (Metal vs CPU-ish).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProgressBackendClass {
    #[default]
    AutoOrCpu,
    Accelerated,
}

/// Segmenter kind when external diarization is in the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProgressSegmenterKind {
    #[default]
    Auto,
    /// pyannote segmentation-3.0 style local activity.
    Segmentation3_0,
    /// DiariZen-class heavier windowed segmenter (when selected by Auto/install).
    DiariZen,
}

impl ProgressSegmenterKind {
    pub fn from_preference(preference: VoiceIdSegmenterPreference) -> Self {
        match preference {
            VoiceIdSegmenterPreference::Auto => ProgressSegmenterKind::Auto,
            VoiceIdSegmenterPreference::Segmentation3_0 => ProgressSegmenterKind::Segmentation3_0,
        }
    }
}

/// Inputs for building a request-scoped progress plan. Stages that will not
/// run must be left off (their flags false).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgressPlanInput {
    pub audio_duration_s: f32,
    pub voice_id: bool,
    /// External segment/embed/cluster path (not in-decoder speakers).
    pub external_diarize: bool,
    pub segmenter: ProgressSegmenterKind,
    pub punctuate: bool,
    pub align: bool,
    pub backend: ProgressBackendClass,
    /// Include a terminal `persist` stage (server write path). Core-only runs
    /// leave this false.
    pub persist: bool,
}

impl ProgressPlanInput {
    /// Minimal plan for post-hoc forced-align refine (independent operation).
    pub fn post_hoc_align(audio_duration_s: f32, backend: ProgressBackendClass) -> Self {
        Self {
            audio_duration_s,
            voice_id: false,
            external_diarize: false,
            segmenter: ProgressSegmenterKind::Auto,
            punctuate: false,
            align: true,
            backend,
            persist: false,
        }
    }
}

/// One planned stage with its estimated effort cost (abstract work units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlannedStage {
    pub stage: TranscriptionStage,
    pub estimated_cost: f64,
}

/// Ordered plan of stages that will actually run for this request.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgressPlan {
    stages: Vec<PlannedStage>,
    /// When false, cost weights are provisional (audio duration still unknown).
    /// Overall fraction stays at 0 until a duration-aware plan replaces it —
    /// otherwise fixed early-stage costs dominate and the bar jumps to ~70%.
    duration_known: bool,
}

fn forced_align_rtf(backend: ProgressBackendClass) -> f64 {
    match backend {
        ProgressBackendClass::Accelerated => 0.055,
        // The Q8_0 quality-safe profile measured 0.141 median RTF and 0.146
        // over sixteen consecutive segments on the production private-audio
        // gate. Keep a small conservative margin for the final NAR graph.
        ProgressBackendClass::AutoOrCpu => 0.15,
    }
}

impl ProgressPlan {
    pub fn stages(&self) -> &[PlannedStage] {
        &self.stages
    }

    pub fn total_cost(&self) -> f64 {
        self.stages.iter().map(|s| s.estimated_cost).sum()
    }

    /// True once audio duration is known and stage costs are trustworthy.
    pub fn duration_known(&self) -> bool {
        self.duration_known
    }

    pub fn contains(&self, stage: TranscriptionStage) -> bool {
        self.stages.iter().any(|s| s.stage == stage)
    }

    pub fn cost_of(&self, stage: TranscriptionStage) -> Option<f64> {
        self.stages
            .iter()
            .find(|s| s.stage == stage)
            .map(|s| s.estimated_cost)
    }

    /// Build the plan from request facts. Missing optional stages are omitted
    /// entirely so their weight never dilutes overall.
    pub fn build(input: ProgressPlanInput) -> Self {
        let duration = input.audio_duration_s.max(0.0) as f64;
        // Duration is "known" only when the caller supplied a positive length.
        // Zero means provisional (pre-prepare) and must not drive overall %.
        let duration_known = input.audio_duration_s.is_finite() && input.audio_duration_s > 0.0;
        let accelerated = matches!(input.backend, ProgressBackendClass::Accelerated);
        let mut stages = Vec::with_capacity(9);

        // Order matches the real native pipeline: model resolve / pack verify
        // first, then audio prepare, then optional diarize/identity before
        // decode. (Product stage names still use prepare/load_model labels.)
        stages.push(PlannedStage {
            stage: TranscriptionStage::LoadModel,
            estimated_cost: if accelerated { 0.35 } else { 0.55 },
        });
        stages.push(PlannedStage {
            stage: TranscriptionStage::Prepare,
            estimated_cost: 0.05 + duration * 0.002,
        });

        if input.external_diarize {
            // This stage includes the selected activity model, vendored VAD,
            // ReDimNet embedding windows, and clustering. Earlier weights only
            // priced the segmenter and made the bar reach 100% before the
            // embedding loop. These production-geometry profiles price the
            // complete stage; per-window callbacks provide the live fraction.
            // `Auto` remains a provisional stand-in until prepare pins a
            // provider.
            let rtf = match input.segmenter {
                ProgressSegmenterKind::DiariZen => {
                    if accelerated {
                        0.89
                    } else {
                        1.31
                    }
                }
                ProgressSegmenterKind::Segmentation3_0 | ProgressSegmenterKind::Auto => {
                    // Segmentation3 and ReDimNet are fixed-CPU stages even if
                    // the primary ASR request later resolves an accelerator.
                    0.66
                }
            };
            stages.push(PlannedStage {
                stage: TranscriptionStage::Diarize,
                estimated_cost: 0.15 + duration * rtf,
            });
        }

        // Identity stage order matches the real pipeline:
        // - external Voice ID: Identify runs on the speaker timeline before decode
        // - in-decoder Voice ID: Identify runs after Decode/Punctuate/Align on
        //   assembled scopes (Decode -> Punctuate -> Align -> Identify -> Project)
        let identify_cost = if input.voice_id {
            let rtf = if accelerated { 0.04 } else { 0.08 };
            Some(0.10 + duration * rtf)
        } else {
            None
        };
        if let Some(cost) = identify_cost
            && input.external_diarize
        {
            stages.push(PlannedStage {
                stage: TranscriptionStage::IdentifySpeakers,
                estimated_cost: cost,
            });
        }

        // decode dominates for plain ASR; RTF is a coarse profile, not a
        // measured ETA and not a fixed percent of the bar.
        let decode_rtf = if accelerated { 0.08 } else { 0.30 };
        stages.push(PlannedStage {
            stage: TranscriptionStage::Decode,
            estimated_cost: 0.20 + duration * decode_rtf,
        });

        if input.punctuate {
            stages.push(PlannedStage {
                stage: TranscriptionStage::Punctuate,
                estimated_cost: 0.05 + duration * 0.001,
            });
        }

        if input.align {
            stages.push(PlannedStage {
                stage: TranscriptionStage::Align,
                estimated_cost: 0.20 + duration * forced_align_rtf(input.backend),
            });
        }

        // InDecoder identity needs punctuated / aligned word anchors when those
        // stages run, so it sits after them and before dual-view projection.
        if let Some(cost) = identify_cost
            && !input.external_diarize
        {
            stages.push(PlannedStage {
                stage: TranscriptionStage::IdentifySpeakers,
                estimated_cost: cost,
            });
        }

        stages.push(PlannedStage {
            stage: TranscriptionStage::Project,
            estimated_cost: 0.02,
        });

        if input.persist {
            stages.push(PlannedStage {
                stage: TranscriptionStage::Persist,
                estimated_cost: 0.02,
            });
        }

        // Guard: every cost must be strictly positive so overall is well-defined.
        for stage in &mut stages {
            if stage.estimated_cost <= 0.0 {
                stage.estimated_cost = 1e-6;
            }
        }

        Self {
            stages,
            duration_known,
        }
    }

    /// Post-hoc FA-only plan: align (+ project). No diarize/decode weight.
    pub fn post_hoc_align(audio_duration_s: f32, backend: ProgressBackendClass) -> Self {
        let input = ProgressPlanInput::post_hoc_align(audio_duration_s, backend);
        let duration = input.audio_duration_s.max(0.0) as f64;
        let duration_known = audio_duration_s.is_finite() && audio_duration_s > 0.0;
        Self {
            stages: vec![
                PlannedStage {
                    stage: TranscriptionStage::Align,
                    estimated_cost: (0.20 + duration * forced_align_rtf(backend)).max(1e-6),
                },
                PlannedStage {
                    stage: TranscriptionStage::Project,
                    estimated_cost: 0.02,
                },
            ],
            // Post-hoc refine always has a known recording length (or treats
            // zero as unknown and keeps overall at 0 until duration is set).
            duration_known,
        }
    }
}

/// Pure overall-fraction math used by the registry and unit tests.
///
/// ```text
/// sum(completed_stage_estimated_cost)
///   + current_stage_estimated_cost * stage_fraction
/// ────────────────────────────────────────────────
/// sum(all_planned_stage_estimated_cost)
/// ```
///
/// When `stage_fraction` is `None` (indeterminate), the current stage
/// contributes **zero** -- the bar does not invent progress.
pub fn compute_overall_fraction(
    plan: &ProgressPlan,
    completed_stage_costs: f64,
    current_stage_cost: f64,
    stage_fraction: Option<f32>,
) -> f32 {
    let total = plan.total_cost();
    if total <= 0.0 {
        return 0.0;
    }
    let current = match stage_fraction {
        Some(frac) => current_stage_cost * (frac.clamp(0.0, 1.0) as f64),
        None => 0.0,
    };
    let raw = (completed_stage_costs + current) / total;
    raw.clamp(0.0, 1.0) as f32
}

/// Duration-weighted stage fraction for forced-align (or similar) loops over
/// segments of unequal audio length. Empty total duration yields 1.0 when any
/// work was "done", else 0.0.
pub fn duration_weighted_fraction(completed_duration_s: f64, total_duration_s: f64) -> f32 {
    if total_duration_s <= 0.0 {
        return if completed_duration_s > 0.0 { 1.0 } else { 0.0 };
    }
    (completed_duration_s / total_duration_s).clamp(0.0, 1.0) as f32
}

// ── Registry ────────────────────────────────────────────────────────────────

struct ProgressState {
    plan: ProgressPlan,
    /// Stages that were entered and fully completed. Never includes a stage
    /// the pipeline has not actually run -- `enter_stage` must not invent
    /// completion for an unentered plan prefix.
    completed_stages: Vec<TranscriptionStage>,
    /// Index of the current stage in `plan.stages`, if any has been entered.
    current_index: Option<usize>,
    stage_fraction: Option<f32>,
    completed_units: Option<u64>,
    total_units: Option<u64>,
    overall_fraction: f32,
    indeterminate: bool,
    detail: Option<String>,
}

impl ProgressState {
    fn new(plan: ProgressPlan) -> Self {
        Self {
            plan,
            completed_stages: Vec::new(),
            current_index: None,
            stage_fraction: None,
            completed_units: None,
            total_units: None,
            overall_fraction: 0.0,
            indeterminate: true,
            detail: None,
        }
    }

    fn current_stage(&self) -> TranscriptionStage {
        self.current_index
            .and_then(|i| self.plan.stages.get(i).map(|s| s.stage))
            .unwrap_or(TranscriptionStage::Prepare)
    }

    fn current_stage_cost(&self) -> f64 {
        self.current_index
            .and_then(|i| self.plan.stages.get(i).map(|s| s.estimated_cost))
            .unwrap_or(0.0)
    }

    /// Sum of planned costs for stages that actually completed. Stages that
    /// left the plan (or were never in it) contribute zero.
    fn completed_stage_costs(&self) -> f64 {
        self.completed_stages
            .iter()
            .filter_map(|stage| self.plan.cost_of(*stage))
            .sum()
    }

    fn mark_completed(&mut self, stage: TranscriptionStage) {
        if !self.completed_stages.contains(&stage) {
            self.completed_stages.push(stage);
        }
    }

    fn snapshot(&self) -> NativeTranscriptionProgress {
        let stage = self.current_stage();
        NativeTranscriptionProgress::new(
            stage,
            self.stage_fraction,
            self.overall_fraction,
            self.completed_units,
            self.total_units,
            self.detail.clone(),
        )
    }

    fn recompute_overall(&mut self) {
        // Provisional (duration-unknown) plans must not publish cost-weighted
        // overall: fixed early-stage costs would dominate (e.g. 73% at decode
        // start). Stage labels still advance; the bar stays at 0 until a
        // duration-aware plan is installed.
        if !self.plan.duration_known {
            self.overall_fraction = 0.0;
            return;
        }
        // When the open stage is already fully complete it lives in
        // `completed_stages`; do not also multiply its cost by stage_fraction.
        let (completed, current_cost, stage_fraction) = if let Some(idx) = self.current_index {
            let stage = self.plan.stages[idx].stage;
            if self.completed_stages.contains(&stage) {
                (self.completed_stage_costs(), 0.0, Some(0.0))
            } else {
                (
                    self.completed_stage_costs(),
                    self.current_stage_cost(),
                    self.stage_fraction,
                )
            }
        } else {
            (self.completed_stage_costs(), 0.0, Some(0.0))
        };
        let next = compute_overall_fraction(&self.plan, completed, current_cost, stage_fraction);
        // Monotonic within a duration-known plan: a later report never moves
        // the bar backward. Plan *replacement* rewrites overall honestly
        // (see `replace_plan`) and may lower the floor.
        self.overall_fraction = self.overall_fraction.max(next);
    }

    /// Enter `stage` (must be in the plan). Completes the previously open stage
    /// only -- never marks unentered plan-prefix stages as done (that would
    /// fake overall progress when the pipeline order differs from a stale plan).
    fn enter_stage(&mut self, stage: TranscriptionStage, indeterminate: bool) {
        let Some(idx) = self.plan.stages.iter().position(|s| s.stage == stage) else {
            // Stage not in plan: ignore (caller bug); do not invent weight.
            return;
        };

        if self.current_index == Some(idx) {
            if indeterminate {
                self.stage_fraction = None;
                self.indeterminate = true;
                self.completed_units = None;
                self.total_units = None;
            }
            self.recompute_overall();
            return;
        }

        // Fold only the previously open stage into completed work.
        if let Some(prev) = self.current_index
            && prev != idx
        {
            let prev_stage = self.plan.stages[prev].stage;
            self.mark_completed(prev_stage);
        }

        // Re-opening a previously completed stage (e.g. LoadModel again for
        // auxiliary Voice ID packs) must un-complete it so sub-progress can
        // contribute honestly; the first visit's cost is not double-counted
        // once this visit completes again.
        self.completed_stages.retain(|s| *s != stage);

        self.current_index = Some(idx);
        self.stage_fraction = if indeterminate { None } else { Some(0.0) };
        self.indeterminate = indeterminate;
        self.completed_units = None;
        self.total_units = None;
        self.detail = None;
        self.recompute_overall();
    }

    fn set_fraction(
        &mut self,
        fraction: f32,
        completed_units: Option<u64>,
        total_units: Option<u64>,
        detail: Option<String>,
    ) {
        let frac = fraction.clamp(0.0, 1.0);
        // Stage fraction is also monotonic within a stage.
        self.stage_fraction = Some(
            self.stage_fraction
                .map(|prev| prev.max(frac))
                .unwrap_or(frac),
        );
        self.indeterminate = false;
        if completed_units.is_some() {
            self.completed_units = completed_units;
        }
        if total_units.is_some() {
            self.total_units = total_units;
        }
        if detail.is_some() {
            self.detail = detail;
        }
        self.recompute_overall();
    }

    fn complete_current(&mut self) {
        if let Some(idx) = self.current_index {
            let stage = self.plan.stages[idx].stage;
            self.stage_fraction = Some(1.0);
            self.indeterminate = false;
            self.mark_completed(stage);
            self.recompute_overall();
        }
    }

    /// Replace the plan while preserving stages already finished (matched by
    /// name). Unentered stages are never force-completed just because they
    /// appear before the open stage in the new order.
    ///
    /// Overall is **recomputed honestly** from the new costs (may go down).
    /// Monotonic max only applies within a stable plan after this rewrite —
    /// otherwise a provisional duration=0 plan permanently floors the bar at
    /// ~70% when real duration weights land.
    fn replace_plan(&mut self, plan: ProgressPlan) {
        let mut finished = self.completed_stages.clone();
        // Stages strictly before current were advanced past and are finished
        // even if complete_current was not called (enter of the next stage
        // folds them). Reconstruct from the old plan index for safety.
        if let Some(idx) = self.current_index {
            for planned in self.plan.stages.iter().take(idx) {
                if !finished.contains(&planned.stage) {
                    finished.push(planned.stage);
                }
            }
        }
        let open = self.current_index.map(|i| self.plan.stages[i].stage);
        let open_frac = self.stage_fraction;
        let units = (self.completed_units, self.total_units);
        let detail = self.detail.clone();
        let open_was_complete = open.is_some_and(|stage| finished.contains(&stage))
            || matches!(open_frac, Some(f) if f >= 1.0 - 1e-6);

        self.plan = plan;
        self.completed_stages = finished
            .into_iter()
            .filter(|stage| self.plan.contains(*stage))
            .collect();
        if open_was_complete
            && let Some(stage) = open
            && self.plan.contains(stage)
        {
            self.mark_completed(stage);
        }
        self.current_index = None;
        self.stage_fraction = None;
        self.completed_units = None;
        self.total_units = None;
        self.detail = None;
        // Drop the previous floor so the new weights can correct a provisional
        // (duration-unknown) overshoot. recompute_overall then sets the honest
        // value (0 while still provisional, else completed/total + current).
        self.overall_fraction = 0.0;

        if let Some(stage) = open
            && self.plan.contains(stage)
        {
            let idx = self
                .plan
                .stages
                .iter()
                .position(|s| s.stage == stage)
                .expect("contains");
            self.current_index = Some(idx);
            if open_was_complete {
                self.stage_fraction = Some(1.0);
                self.indeterminate = false;
            } else {
                self.stage_fraction = open_frac;
                self.indeterminate = open_frac.is_none();
            }
            self.completed_units = units.0;
            self.total_units = units.1;
            self.detail = detail;
        }
        self.recompute_overall();
    }
}

struct ProgressRegistry {
    entries: Vec<(String, ProgressState)>,
}

impl ProgressRegistry {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn get(&self, id: &str) -> Option<NativeTranscriptionProgress> {
        self.entries
            .iter()
            .find(|(entry_id, _)| entry_id == id)
            .map(|(_, state)| state.snapshot())
    }

    fn get_mut(&mut self, id: &str) -> Option<&mut ProgressState> {
        self.entries
            .iter_mut()
            .find(|(entry_id, _)| entry_id == id)
            .map(|(_, state)| state)
    }

    fn insert(&mut self, id: &str, plan: ProgressPlan) {
        if let Some(state) = self.get_mut(id) {
            *state = ProgressState::new(plan);
            return;
        }
        if self.entries.len() >= PROGRESS_REGISTRY_CAPACITY {
            self.entries.remove(0);
        }
        self.entries
            .push((id.to_string(), ProgressState::new(plan)));
    }

    fn remove(&mut self, id: &str) {
        self.entries.retain(|(entry_id, _)| entry_id != id);
    }
}

static PROGRESS_REGISTRY: Mutex<ProgressRegistry> = Mutex::new(ProgressRegistry::new());

fn with_registry<T>(f: impl FnOnce(&mut ProgressRegistry) -> T) -> T {
    let mut registry = PROGRESS_REGISTRY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f(&mut registry)
}

/// Test-only: wipe every registry entry so aggregate legacy reads are
/// isolated under plain `cargo test` (which shares one process). Prefer
/// `cargo nextest` for real isolation; this is a belt-and-braces cleanup.
#[cfg(test)]
pub(crate) fn clear_progress_registry_for_test() {
    with_registry(|reg| reg.entries.clear());
}

/// Serializes tests that inspect the process-global aggregate registry under
/// plain `cargo test` (which runs tests in one process). `cargo nextest`
/// isolates per test process so this is a no-op race-wise there.
#[cfg(test)]
pub(crate) fn progress_registry_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Progress of the in-flight native transcription with this `id`, or `None`
/// when no such run is currently active.
pub fn native_transcription_progress_for_id(id: &str) -> Option<NativeTranscriptionProgress> {
    with_registry(|reg| reg.get(id))
}

pub fn native_transcription_progress() -> LegacyNativeTranscriptionProgress {
    with_registry(|reg| match reg.entries.as_slice() {
        [] => LegacyNativeTranscriptionProgress::Idle,
        [(_, state)] => LegacyNativeTranscriptionProgress::Single(state.snapshot()),
        entries => LegacyNativeTranscriptionProgress::Ambiguous {
            active_count: entries.len(),
        },
    })
}

/// RAII handle: removes the registry entry on drop (completion / cancel /
/// panic). Creating a handle does not publish; the first `enter`/`report`
/// does after `install`.
pub struct ProgressRegistryHandle {
    id: Option<String>,
}

impl ProgressRegistryHandle {
    pub fn new(id: Option<String>) -> Self {
        Self { id }
    }
}

impl Drop for ProgressRegistryHandle {
    fn drop(&mut self) {
        if let Some(id) = &self.id {
            with_registry(|reg| reg.remove(id));
        }
    }
}

/// Request-scoped progress reporter. All methods are no-ops for `id: None`
/// (detached / uncancellable contexts never publish).
#[derive(Debug, Clone)]
pub struct ProgressReporter {
    id: Option<String>,
}

impl ProgressReporter {
    /// Install a fresh plan for `id` (replacing any prior state for that id).
    pub fn install(id: Option<String>, plan: ProgressPlan) -> Self {
        if let Some(ref rid) = id {
            with_registry(|reg| reg.insert(rid, plan));
        }
        Self { id }
    }

    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn replace_plan(&self, plan: ProgressPlan) {
        let Some(id) = self.id.as_deref() else {
            return;
        };
        with_registry(|reg| {
            if let Some(state) = reg.get_mut(id) {
                state.replace_plan(plan);
            } else {
                reg.insert(id, plan);
            }
        });
    }

    pub fn enter_stage(&self, stage: TranscriptionStage) {
        self.enter_stage_inner(stage, false);
    }

    pub fn enter_stage_indeterminate(&self, stage: TranscriptionStage) {
        self.enter_stage_inner(stage, true);
    }

    fn enter_stage_inner(&self, stage: TranscriptionStage, indeterminate: bool) {
        let Some(id) = self.id.as_deref() else {
            return;
        };
        with_registry(|reg| {
            if let Some(state) = reg.get_mut(id) {
                state.enter_stage(stage, indeterminate);
            }
        });
    }

    /// Report real stage completion in `0..=1`. Does nothing if no stage open.
    pub fn report_fraction(&self, fraction: f32) {
        self.report(fraction, None, None, None);
    }

    pub fn report_units(&self, completed: u64, total: u64) {
        let fraction = if total == 0 {
            1.0
        } else {
            (completed as f32 / total as f32).clamp(0.0, 1.0)
        };
        self.report(fraction, Some(completed), Some(total), None);
    }

    pub fn report(
        &self,
        fraction: f32,
        completed_units: Option<u64>,
        total_units: Option<u64>,
        detail: Option<String>,
    ) {
        let Some(id) = self.id.as_deref() else {
            return;
        };
        with_registry(|reg| {
            if let Some(state) = reg.get_mut(id) {
                state.set_fraction(fraction, completed_units, total_units, detail);
            }
        });
    }

    pub fn complete_stage(&self) {
        let Some(id) = self.id.as_deref() else {
            return;
        };
        with_registry(|reg| {
            if let Some(state) = reg.get_mut(id) {
                state.complete_current();
            }
        });
    }

    /// Flash a short stage from 0 to 1 (e.g. project).
    pub fn complete_stage_brief(&self, stage: TranscriptionStage) {
        self.enter_stage(stage);
        self.report_fraction(1.0);
        self.complete_stage();
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn plain_plan(duration_s: f32, align: bool) -> ProgressPlan {
        ProgressPlan::build(ProgressPlanInput {
            audio_duration_s: duration_s,
            voice_id: false,
            external_diarize: false,
            segmenter: ProgressSegmenterKind::Auto,
            punctuate: false,
            align,
            backend: ProgressBackendClass::AutoOrCpu,
            persist: false,
        })
    }

    #[test]
    fn overall_is_monotonic_as_stages_and_fractions_advance() {
        let plan = plain_plan(60.0, true);
        let mut completed = 0.0f64;
        let mut overall = 0.0f32;
        for planned in plan.stages() {
            // enter at 0
            let at_start =
                compute_overall_fraction(&plan, completed, planned.estimated_cost, Some(0.0));
            assert!(at_start >= overall - 1e-6, "{at_start} < {overall}");
            overall = overall.max(at_start);
            // mid
            let mid = compute_overall_fraction(&plan, completed, planned.estimated_cost, Some(0.5));
            assert!(mid >= overall - 1e-6);
            overall = overall.max(mid);
            // done
            let done =
                compute_overall_fraction(&plan, completed, planned.estimated_cost, Some(1.0));
            assert!(done >= overall - 1e-6);
            overall = overall.max(done);
            completed += planned.estimated_cost;
        }
        assert!((overall - 1.0).abs() < 1e-5);
    }

    #[test]
    fn no_event_leaves_overall_unchanged() {
        let plan = plain_plan(30.0, false);
        let decode_cost = plan.cost_of(TranscriptionStage::Decode).unwrap();
        let prepare_cost = plan.cost_of(TranscriptionStage::Prepare).unwrap();
        let load_cost = plan.cost_of(TranscriptionStage::LoadModel).unwrap();
        let completed = prepare_cost + load_cost;
        let a = compute_overall_fraction(&plan, completed, decode_cost, Some(0.4));
        // Recomputing the same inputs (no new event) yields the same value.
        let b = compute_overall_fraction(&plan, completed, decode_cost, Some(0.4));
        assert!((a - b).abs() < 1e-9);
        // Indeterminate current stage contributes zero -- overall holds at
        // completed/total, never invents a climb.
        let indet = compute_overall_fraction(&plan, completed, decode_cost, None);
        let at_zero = compute_overall_fraction(&plan, completed, decode_cost, Some(0.0));
        assert!((indet - at_zero).abs() < 1e-9);
    }

    #[test]
    fn forced_align_duration_weight_is_not_fifty_fifty() {
        // Two segments: 1s and 9s -- after the short one, fraction is 0.1 not 0.5.
        let total = 1.0 + 9.0;
        let after_first = duration_weighted_fraction(1.0, total);
        assert!((after_first - 0.1).abs() < 1e-6);
        let after_both = duration_weighted_fraction(total, total);
        assert!((after_both - 1.0).abs() < 1e-6);
    }

    #[test]
    fn plan_without_forced_align_omits_align_weight() {
        let with = plain_plan(120.0, true);
        let without = plain_plan(120.0, false);
        assert!(with.contains(TranscriptionStage::Align));
        assert!(!without.contains(TranscriptionStage::Align));
        assert!(with.total_cost() > without.total_cost());
        // Decode weight relative share is higher when align is absent -- but
        // absolute decode cost is identical (not a fixed percent).
        assert_eq!(
            with.cost_of(TranscriptionStage::Decode),
            without.cost_of(TranscriptionStage::Decode)
        );
    }

    #[test]
    fn plan_omits_diarize_when_voice_id_external_off() {
        let plan = plain_plan(10.0, false);
        assert!(!plan.contains(TranscriptionStage::Diarize));
        assert!(!plan.contains(TranscriptionStage::IdentifySpeakers));
    }

    #[test]
    fn plan_includes_diarize_and_identity_before_decode_when_external() {
        let plan = ProgressPlan::build(ProgressPlanInput {
            audio_duration_s: 60.0,
            voice_id: true,
            external_diarize: true,
            segmenter: ProgressSegmenterKind::Segmentation3_0,
            punctuate: false,
            align: false,
            backend: ProgressBackendClass::AutoOrCpu,
            persist: false,
        });
        let labels: Vec<_> = plan.stages().iter().map(|s| s.stage).collect();
        let diarize = labels
            .iter()
            .position(|s| *s == TranscriptionStage::Diarize)
            .unwrap();
        let identify = labels
            .iter()
            .position(|s| *s == TranscriptionStage::IdentifySpeakers)
            .unwrap();
        let decode = labels
            .iter()
            .position(|s| *s == TranscriptionStage::Decode)
            .unwrap();
        assert!(diarize < identify);
        assert!(identify < decode);
    }

    #[test]
    fn plan_places_identity_after_decode_for_in_decoder_voice_id() {
        let plan = ProgressPlan::build(ProgressPlanInput {
            audio_duration_s: 60.0,
            voice_id: true,
            external_diarize: false,
            segmenter: ProgressSegmenterKind::Auto,
            punctuate: true,
            align: true,
            backend: ProgressBackendClass::AutoOrCpu,
            persist: false,
        });
        let labels: Vec<_> = plan.stages().iter().map(|s| s.stage).collect();
        assert!(!plan.contains(TranscriptionStage::Diarize));
        let decode = labels
            .iter()
            .position(|s| *s == TranscriptionStage::Decode)
            .unwrap();
        let punctuate = labels
            .iter()
            .position(|s| *s == TranscriptionStage::Punctuate)
            .unwrap();
        let align = labels
            .iter()
            .position(|s| *s == TranscriptionStage::Align)
            .unwrap();
        let identify = labels
            .iter()
            .position(|s| *s == TranscriptionStage::IdentifySpeakers)
            .unwrap();
        let project = labels
            .iter()
            .position(|s| *s == TranscriptionStage::Project)
            .unwrap();
        assert!(
            decode < punctuate && punctuate < align && align < identify && identify < project,
            "in-decoder plan must be Decode->Punctuate->Align->Identify->Project: {labels:?}"
        );
    }

    #[test]
    fn plan_diarize_weight_uses_resolved_diarizen_segmenter() {
        let auto = ProgressPlan::build(ProgressPlanInput {
            audio_duration_s: 100.0,
            voice_id: true,
            external_diarize: true,
            segmenter: ProgressSegmenterKind::Auto,
            punctuate: false,
            align: false,
            backend: ProgressBackendClass::AutoOrCpu,
            persist: false,
        });
        let diarizen = ProgressPlan::build(ProgressPlanInput {
            audio_duration_s: 100.0,
            voice_id: true,
            external_diarize: true,
            segmenter: ProgressSegmenterKind::DiariZen,
            punctuate: false,
            align: false,
            backend: ProgressBackendClass::AutoOrCpu,
            persist: false,
        });
        assert!(
            diarizen.cost_of(TranscriptionStage::Diarize).unwrap()
                > auto.cost_of(TranscriptionStage::Diarize).unwrap(),
            "resolved DiariZen must weigh heavier than provisional Auto"
        );
    }

    #[test]
    fn plan_accelerated_backend_reduces_decode_cost() {
        let cpu = ProgressPlan::build(ProgressPlanInput {
            audio_duration_s: 100.0,
            voice_id: false,
            external_diarize: false,
            segmenter: ProgressSegmenterKind::Auto,
            punctuate: false,
            align: false,
            backend: ProgressBackendClass::AutoOrCpu,
            persist: false,
        });
        let accel = ProgressPlan::build(ProgressPlanInput {
            audio_duration_s: 100.0,
            voice_id: false,
            external_diarize: false,
            segmenter: ProgressSegmenterKind::Auto,
            punctuate: false,
            align: false,
            backend: ProgressBackendClass::Accelerated,
            persist: false,
        });
        assert!(
            accel.cost_of(TranscriptionStage::Decode).unwrap()
                < cpu.cost_of(TranscriptionStage::Decode).unwrap()
        );
    }

    #[test]
    fn measured_auxiliary_stage_costs_preserve_backend_and_provider_ordering() {
        let build = |segmenter, backend| {
            ProgressPlan::build(ProgressPlanInput {
                audio_duration_s: 100.0,
                voice_id: true,
                external_diarize: true,
                segmenter,
                punctuate: true,
                align: true,
                backend,
                persist: false,
            })
        };
        let seg3_cpu = build(
            ProgressSegmenterKind::Segmentation3_0,
            ProgressBackendClass::AutoOrCpu,
        );
        let seg3_accel = build(
            ProgressSegmenterKind::Segmentation3_0,
            ProgressBackendClass::Accelerated,
        );
        let diarizen_cpu = build(
            ProgressSegmenterKind::DiariZen,
            ProgressBackendClass::AutoOrCpu,
        );
        let diarizen_accel = build(
            ProgressSegmenterKind::DiariZen,
            ProgressBackendClass::Accelerated,
        );

        assert_eq!(
            seg3_cpu.cost_of(TranscriptionStage::Diarize),
            seg3_accel.cost_of(TranscriptionStage::Diarize),
            "Segmentation3 and ReDimNet remain fixed-CPU work"
        );
        assert!(
            diarizen_accel.cost_of(TranscriptionStage::Diarize)
                < diarizen_cpu.cost_of(TranscriptionStage::Diarize)
        );
        assert!(
            diarizen_accel.cost_of(TranscriptionStage::Diarize)
                > seg3_accel.cost_of(TranscriptionStage::Diarize)
        );
        assert!(
            seg3_accel.cost_of(TranscriptionStage::Align)
                < seg3_cpu.cost_of(TranscriptionStage::Align)
        );
        assert_eq!(
            seg3_accel.cost_of(TranscriptionStage::Punctuate),
            seg3_cpu.cost_of(TranscriptionStage::Punctuate),
            "the measured punctuation cost is backend-independent at plan time"
        );
        let close = |actual: f64, expected: f64| {
            assert!(
                (actual - expected).abs() <= 1e-9,
                "expected {expected}, got {actual}"
            );
        };
        close(
            seg3_cpu
                .cost_of(TranscriptionStage::Diarize)
                .expect("diarize stage"),
            66.15,
        );
        close(
            diarizen_cpu
                .cost_of(TranscriptionStage::Diarize)
                .expect("diarize stage"),
            131.15,
        );
        close(
            diarizen_accel
                .cost_of(TranscriptionStage::Diarize)
                .expect("diarize stage"),
            89.15,
        );
        close(
            seg3_cpu
                .cost_of(TranscriptionStage::Punctuate)
                .expect("punctuate stage"),
            0.15,
        );
        close(
            seg3_cpu
                .cost_of(TranscriptionStage::Align)
                .expect("align stage"),
            15.20,
        );
        close(
            seg3_accel
                .cost_of(TranscriptionStage::Align)
                .expect("align stage"),
            5.70,
        );
        for (backend, full) in [
            (ProgressBackendClass::AutoOrCpu, &seg3_cpu),
            (ProgressBackendClass::Accelerated, &seg3_accel),
        ] {
            let post_hoc = ProgressPlan::post_hoc_align(100.0, backend);
            assert_eq!(
                post_hoc.cost_of(TranscriptionStage::Align),
                full.cost_of(TranscriptionStage::Align),
                "post-hoc and in-pipeline forced alignment must share one calibration"
            );
        }
    }

    #[test]
    fn post_hoc_align_plan_has_no_decode_weight() {
        let plan = ProgressPlan::post_hoc_align(45.0, ProgressBackendClass::Accelerated);
        assert!(plan.contains(TranscriptionStage::Align));
        assert!(plan.contains(TranscriptionStage::Project));
        assert!(!plan.contains(TranscriptionStage::Decode));
        assert!(!plan.contains(TranscriptionStage::Diarize));
    }

    #[test]
    fn enter_stage_does_not_complete_unentered_plan_prefix() {
        // Stale plan with Identify before Decode (external-style). Jumping to
        // Decode must not mark Identify complete -- that stage never ran.
        let plan = ProgressPlan::build(ProgressPlanInput {
            audio_duration_s: 30.0,
            voice_id: true,
            external_diarize: true,
            segmenter: ProgressSegmenterKind::Segmentation3_0,
            punctuate: false,
            align: false,
            backend: ProgressBackendClass::AutoOrCpu,
            persist: false,
        });
        let identify_cost = plan.cost_of(TranscriptionStage::IdentifySpeakers).unwrap();
        let load_cost = plan.cost_of(TranscriptionStage::LoadModel).unwrap();
        let prepare_cost = plan.cost_of(TranscriptionStage::Prepare).unwrap();
        let diarize_cost = plan.cost_of(TranscriptionStage::Diarize).unwrap();
        let total = plan.total_cost();

        let mut state = ProgressState::new(plan);
        state.enter_stage(TranscriptionStage::LoadModel, false);
        state.complete_current();
        state.enter_stage(TranscriptionStage::Prepare, false);
        state.complete_current();
        state.enter_stage(TranscriptionStage::Diarize, false);
        state.complete_current();
        // Skip IdentifySpeakers -- jump straight to Decode (the InDecoder bug).
        state.enter_stage(TranscriptionStage::Decode, false);
        state.recompute_overall();

        let completed = state.completed_stage_costs();
        assert!(
            !state
                .completed_stages
                .contains(&TranscriptionStage::IdentifySpeakers),
            "unentered IdentifySpeakers must not be marked complete"
        );
        assert!(
            (completed - (load_cost + prepare_cost + diarize_cost)).abs() < 1e-9,
            "completed costs must exclude skipped Identify: got {completed}"
        );
        // Overall must not include identify weight.
        let expected_at_decode_start =
            (load_cost + prepare_cost + diarize_cost) as f32 / total as f32;
        assert!(
            (state.overall_fraction - expected_at_decode_start).abs() < 1e-4,
            "overall {} must not include unentered identify (expected ~{})",
            state.overall_fraction,
            expected_at_decode_start
        );
        // identify still in total so the skipped weight is honest remaining work
        // until the plan is revised or Identify is entered later.
        assert!(identify_cost > 0.0);
    }

    #[test]
    fn provisional_duration_unknown_keeps_overall_at_zero() {
        // Matches the pre-prepare install path (audio_duration_s: 0.0).
        let provisional = ProgressPlan::build(ProgressPlanInput {
            audio_duration_s: 0.0,
            voice_id: false,
            external_diarize: false,
            segmenter: ProgressSegmenterKind::Auto,
            punctuate: false,
            align: false,
            backend: ProgressBackendClass::AutoOrCpu,
            persist: false,
        });
        assert!(!provisional.duration_known());
        let mut state = ProgressState::new(provisional);
        state.enter_stage(TranscriptionStage::LoadModel, true);
        state.complete_current();
        state.enter_stage(TranscriptionStage::Prepare, true);
        state.complete_current();
        state.enter_stage(TranscriptionStage::Decode, false);
        state.set_fraction(0.0, None, None, None);
        assert!(
            state.overall_fraction.abs() < 1e-6,
            "provisional plan must not publish inflated overall, got {}",
            state.overall_fraction
        );
    }

    #[test]
    fn replace_plan_with_real_duration_corrects_overall_downward() {
        // Simulate the QA bug: provisional d=0 would have been ~73% at decode
        // start if overall were published; after duration-aware replace_plan
        // the bar must sit near completed/total (~4%), not stick at 73%.
        let provisional = ProgressPlan::build(ProgressPlanInput {
            audio_duration_s: 0.0,
            voice_id: false,
            external_diarize: false,
            segmenter: ProgressSegmenterKind::Auto,
            punctuate: false,
            align: false,
            backend: ProgressBackendClass::AutoOrCpu,
            persist: false,
        });
        let mut state = ProgressState::new(provisional);
        state.enter_stage(TranscriptionStage::LoadModel, false);
        state.complete_current();
        state.enter_stage(TranscriptionStage::Prepare, false);
        state.complete_current();
        state.enter_stage(TranscriptionStage::Decode, false);
        assert!(state.overall_fraction.abs() < 1e-6);

        let real = ProgressPlan::build(ProgressPlanInput {
            audio_duration_s: 59.0,
            voice_id: false,
            external_diarize: false,
            segmenter: ProgressSegmenterKind::Auto,
            punctuate: false,
            align: false,
            backend: ProgressBackendClass::AutoOrCpu,
            persist: false,
        });
        let load = real.cost_of(TranscriptionStage::LoadModel).unwrap();
        let prep = real.cost_of(TranscriptionStage::Prepare).unwrap();
        let total = real.total_cost();
        let expected = (load + prep) as f32 / total as f32;
        state.replace_plan(real);
        assert!(
            (state.overall_fraction - expected).abs() < 1e-3,
            "after replace_plan overall {} must be ~{} (not provisional overshoot)",
            state.overall_fraction,
            expected
        );
        assert!(
            state.overall_fraction < 0.15,
            "decode start for ~1min audio must stay well below the old 73% bug"
        );
    }

    #[test]
    fn reporter_is_monotonic_and_clears_on_drop() {
        let _serial = progress_registry_test_lock();
        let id = "progress-reporter-monotonic";
        assert_eq!(native_transcription_progress_for_id(id), None);
        {
            let _handle = ProgressRegistryHandle::new(Some(id.to_string()));
            let reporter = ProgressReporter::install(Some(id.to_string()), plain_plan(30.0, true));
            reporter.enter_stage(TranscriptionStage::Prepare);
            reporter.report_fraction(1.0);
            let after_prepare = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(after_prepare.stage, TranscriptionStage::Prepare);
            assert!(after_prepare.overall_fraction > 0.0);

            reporter.enter_stage(TranscriptionStage::LoadModel);
            reporter.report_fraction(0.5);
            let mid_load = native_transcription_progress_for_id(id).unwrap();
            assert!(mid_load.overall_fraction >= after_prepare.overall_fraction);
            assert_eq!(mid_load.phase, NativeTranscriptionPhase::Decode);

            // No new event: re-read is stable.
            let again = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(again.overall_fraction, mid_load.overall_fraction);

            // Lower report must not regress overall.
            reporter.report_fraction(0.1);
            let after_lower = native_transcription_progress_for_id(id).unwrap();
            assert_eq!(after_lower.overall_fraction, mid_load.overall_fraction);

            reporter.enter_stage(TranscriptionStage::Decode);
            reporter.report_units(400, 1000);
            let decode_mid = native_transcription_progress_for_id(id).unwrap();
            assert!(decode_mid.overall_fraction >= after_lower.overall_fraction);
            assert_eq!(decode_mid.completed_units, Some(400));
            assert_eq!(decode_mid.total_units, Some(1000));
            assert!((decode_mid.stage_fraction.unwrap() - 0.4).abs() < 1e-5);

            reporter.enter_stage(TranscriptionStage::Align);
            assert_eq!(
                native_transcription_progress_for_id(id).unwrap().phase,
                NativeTranscriptionPhase::Align
            );
            // Duration-weighted FA: 1s of 10s.
            reporter.report_fraction(duration_weighted_fraction(1.0, 10.0));
            let fa_mid = native_transcription_progress_for_id(id).unwrap();
            assert!((fa_mid.stage_fraction.unwrap() - 0.1).abs() < 1e-5);

            reporter.complete_stage_brief(TranscriptionStage::Project);
            let done = native_transcription_progress_for_id(id).unwrap();
            assert!(done.overall_fraction <= 1.0);
            assert!(done.overall_fraction >= fa_mid.overall_fraction);
        }
        assert_eq!(native_transcription_progress_for_id(id), None);
    }

    #[test]
    fn detached_reporter_never_publishes() {
        let _serial = progress_registry_test_lock();
        let _handle = ProgressRegistryHandle::new(None);
        let reporter = ProgressReporter::install(None, plain_plan(10.0, false));
        reporter.enter_stage(TranscriptionStage::Decode);
        reporter.report_fraction(0.5);
        assert_eq!(
            native_transcription_progress_for_id("detached-progress-probe"),
            None
        );
    }

    #[test]
    fn legacy_aggregate_idle_single_ambiguous() {
        let _serial = progress_registry_test_lock();
        clear_progress_registry_for_test();
        assert_eq!(
            native_transcription_progress(),
            LegacyNativeTranscriptionProgress::Idle
        );
        let id_a = "legacy-progress-a";
        let id_b = "legacy-progress-b";
        let _ha = ProgressRegistryHandle::new(Some(id_a.to_string()));
        let ra = ProgressReporter::install(Some(id_a.to_string()), plain_plan(5.0, false));
        ra.enter_stage(TranscriptionStage::Decode);
        ra.report_fraction(0.33);
        match native_transcription_progress() {
            LegacyNativeTranscriptionProgress::Single(p) => {
                assert!((p.fraction - p.overall_fraction).abs() < 1e-9);
                assert!((p.stage_fraction.unwrap() - 0.33).abs() < 1e-5);
            }
            other => panic!("expected Single, got {other:?}"),
        }
        let _hb = ProgressRegistryHandle::new(Some(id_b.to_string()));
        let rb = ProgressReporter::install(Some(id_b.to_string()), plain_plan(5.0, false));
        rb.enter_stage(TranscriptionStage::Decode);
        assert_eq!(
            native_transcription_progress(),
            LegacyNativeTranscriptionProgress::Ambiguous { active_count: 2 }
        );
        clear_progress_registry_for_test();
    }

    #[test]
    fn registry_evicts_oldest_at_capacity() {
        let _serial = progress_registry_test_lock();
        clear_progress_registry_for_test();
        let ids: Vec<String> = (0..=PROGRESS_REGISTRY_CAPACITY)
            .map(|i| format!("cap-progress-{i}"))
            .collect();
        for id in &ids {
            ProgressReporter::install(Some(id.clone()), plain_plan(1.0, false));
        }
        assert_eq!(native_transcription_progress_for_id(&ids[0]), None);
        for id in &ids[1..] {
            assert!(native_transcription_progress_for_id(id).is_some());
        }
        with_registry(|reg| {
            for id in &ids[1..] {
                reg.remove(id);
            }
        });
    }

    #[test]
    fn stage_labels_are_stable_snake_case() {
        assert_eq!(
            TranscriptionStage::IdentifySpeakers.label(),
            "identify_speakers"
        );
        assert_eq!(TranscriptionStage::LoadModel.label(), "load_model");
    }
}
