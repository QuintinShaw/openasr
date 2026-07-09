//! Local-only, dependency-free timestamp and coarse stage-timing helpers for
//! daemon/server/CLI logs.
//!
//! `daemon.log` (stdout+stderr of `openasr serve`, captured by the desktop
//! sidecar -- see `openasr-app/apps/desktop/src-tauri/src/sidecar.rs`) had no
//! timestamps at all: server boot, model-pack loading, and realtime warm-up
//! were each a plain `println!`/`eprintln!` with no timing, so diagnosing how
//! long any of it took meant guessing from wall-clock reads of unrelated
//! surrounding events. This module is the single place that formats a dual
//! timestamp for every new stage-boundary log line added alongside it:
//!
//! - a wall-clock ISO 8601 UTC instant, for correlating against other
//!   timestamped logs (crash reporters, desktop-side logs, support bundles);
//! - a monotonic "ms since this process's first stage-timing call" counter,
//!   immune to wall-clock adjustments (NTP step, DST, manual clock changes)
//!   corrupting an elapsed-time calculation.
//!
//! Deliberately dependency-free: no `chrono`/`time` crate pulled in for one
//! narrow purpose, matching this workspace's existing posture (see the
//! `symphonia`/`rubato` feature-trimming comments in the workspace
//! `Cargo.toml`). Every function here only ever writes to local stderr --
//! never a network call -- matching the no-telemetry product promise; see
//! `AGENTS.md` and `docs/agents/domain.md`.
//!
//! This is intentionally additive: existing log lines this workspace's tests
//! match on exact text (e.g. the `"OpenASR server listening on http://"`
//! banner `crates/openasr-cli/tests/cli.rs` waits for) are left byte-for-byte
//! unchanged. Stage timing is emitted as new, separate lines instead of
//! rewriting anything a consumer might already parse.

use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static PROCESS_START: OnceLock<Instant> = OnceLock::new();

/// Pins the monotonic zero point for this process on first call (idempotent
/// afterwards). Call this as early as possible in `main` (before any other
/// stage timing) so "ms since start" reads as true process uptime rather than
/// "time since the first log line happened to fire"; every other function in
/// this module also calls it lazily, so correctness never depends on the
/// caller remembering to do this -- only the precision of the very first
/// delta does.
pub fn process_start() -> Instant {
    *PROCESS_START.get_or_init(Instant::now)
}

fn monotonic_ms_since_start() -> u128 {
    process_start().elapsed().as_millis()
}

/// Wall-clock UTC timestamp, `YYYY-MM-DDTHH:MM:SS.mmmZ` (ISO 8601, millisecond
/// precision).
pub fn now_iso8601() -> String {
    let since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO);
    format_iso8601(since_epoch)
}

fn format_iso8601(since_epoch: Duration) -> String {
    let total_seconds = since_epoch.as_secs();
    let millis = since_epoch.subsec_millis();
    let days = (total_seconds / 86_400) as i64;
    let secs_of_day = total_seconds % 86_400;
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day / 60) % 60;
    let second = secs_of_day % 60;
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

/// Days-since-Unix-epoch -> proleptic Gregorian (year, month, day). Howard
/// Hinnant's well-known dependency-free `civil_from_days` algorithm:
/// <http://howardhinnant.github.io/date_algorithms.html#civil_from_days>.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = if month <= 2 { y + 1 } else { y };
    (year, month, day)
}

/// The `[<iso8601> +<mono_ms>ms]` prefix shared by every timestamped line this
/// module writes.
pub fn prefix() -> String {
    format!("[{} +{}ms]", now_iso8601(), monotonic_ms_since_start())
}

/// Logs one timestamped line to stderr: `<prefix> <component>: <message>`.
/// Always on -- not gated by an env var -- this is baseline daemon-log
/// observability, not opt-in profiling (contrast with the existing
/// `OPENASR_<FAMILY>_PROFILE`-gated fine-grained hooks such as
/// `diarize::embed::wespeaker`'s `OPENASR_WESPEAKER_PROFILE`). Cheap: one
/// `SystemTime`/`Instant` read plus one `eprintln!`; callers must still only
/// call this at coarse boundaries (stage/request granularity), never inside a
/// per-frame or per-token hot loop.
pub fn log_event(component: &str, message: impl std::fmt::Display) {
    eprintln!("{} {component}: {message}", prefix());
}

/// Convenience over [`log_event`] for the common "named stage finished, here
/// is how long it took" line: `<prefix> <component>: stage=<stage>
/// duration_ms=<elapsed>`.
pub fn log_stage(component: &str, stage: &str, elapsed: Duration) {
    log_event(
        component,
        format_args!(
            "stage={stage} duration_ms={:.3}",
            elapsed.as_secs_f64() * 1000.0
        ),
    );
}

/// Whether `OPENASR_TIMING` opts into the finer-grained detail tier (e.g.
/// per-longform-slice decode timing, model-resolution sub-stage timing) on
/// top of the coarse, always-on lines `log_event`/`log_stage` produce. Reread
/// on every call (like the existing `OPENASR_<FAMILY>_PROFILE` gates) rather
/// than cached, since this is checked at most once per request/stage, never
/// in a per-frame loop.
pub fn detail_enabled() -> bool {
    std::env::var("OPENASR_TIMING")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

/// [`log_event`], but only when [`detail_enabled`] -- for finer breakdowns
/// that would otherwise be too noisy to leave on by default (e.g. one line
/// per longform slice on a long recording).
pub fn log_detail_event(component: &str, message: impl std::fmt::Display) {
    if detail_enabled() {
        log_event(component, message);
    }
}

/// [`log_stage`], but only when [`detail_enabled`].
pub fn log_detail_stage(component: &str, stage: &str, elapsed: Duration) {
    if detail_enabled() {
        log_stage(component, stage, elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_from_days_matches_known_dates() {
        // Unix epoch itself.
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // A well-known leap day.
        assert_eq!(civil_from_days(19_051), (2022, 2, 28));
        assert_eq!(civil_from_days(19_052), (2022, 3, 1));
        // 2026-07-10 is 20_644 days after the epoch.
        assert_eq!(civil_from_days(20_644), (2026, 7, 10));
        // A leap-year Feb 29 (2024).
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
    }

    #[test]
    fn format_iso8601_renders_expected_shape() {
        let rendered = format_iso8601(Duration::from_millis(20_644 * 86_400_000 + 12_345_678));
        assert_eq!(rendered, "2026-07-10T03:25:45.678Z");
    }

    #[test]
    fn now_iso8601_has_the_expected_length_and_terminators() {
        let rendered = now_iso8601();
        assert_eq!(rendered.len(), "2026-07-10T12:34:56.789Z".len());
        assert!(rendered.ends_with('Z'));
        assert!(rendered.contains('T'));
    }

    #[test]
    fn prefix_contains_brackets_and_a_monotonic_millisecond_marker() {
        let rendered = prefix();
        assert!(rendered.starts_with('['));
        assert!(rendered.contains("ms]"));
    }

    #[test]
    fn monotonic_ms_since_start_is_nondecreasing() {
        let first = monotonic_ms_since_start();
        std::thread::sleep(Duration::from_millis(5));
        let second = monotonic_ms_since_start();
        assert!(second >= first);
    }

    #[test]
    fn log_event_and_log_stage_do_not_panic() {
        // These write to stderr; the only contract under test is that they
        // never panic on ordinary inputs (this is a hot-adjacent path called
        // from server boot, model-pack load, and the request path).
        log_event("test_component", "message with duration_ms=1.5");
        log_stage("test_component", "example_stage", Duration::from_millis(42));
    }

    #[test]
    fn detail_enabled_defaults_to_false_when_unset() {
        // Best-effort rather than setting/unsetting the env var here: mutating
        // process env from a test races other tests in the same process
        // (`std::env::set_var` is not safe to call concurrently across
        // threads), and no other test in this workspace touches
        // `OPENASR_TIMING`, so an ordinary local/CI run has it unset.
        if std::env::var("OPENASR_TIMING").is_err() {
            assert!(!detail_enabled());
        }
    }

    #[test]
    fn log_detail_event_and_log_detail_stage_do_not_panic() {
        log_detail_event("test_component", "detail message");
        log_detail_stage("test_component", "detail_stage", Duration::from_millis(1));
    }
}
