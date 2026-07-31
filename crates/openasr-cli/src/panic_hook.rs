//! Startup-diagnostics panic hook for `openasr serve`.
//!
//! `daemon.log` is the durable record of what `openasr serve` did --
//! [`openasr_core::stage_timing`]'s doc describes it as "stdout+stderr of
//! `openasr serve`, captured by the desktop sidecar", and every boot/request
//! diagnostic line this binary emits already goes through
//! `stage_timing::log_event` so every line shares one timestamp format. The
//! default panic hook does not: it writes an unstructured, multi-line
//! `thread '...' panicked at ...:\n<message>` blob straight to raw stderr,
//! with no timestamp and no `daemon.log`-consistent shape. Worse, for a panic
//! inside a detached `tokio::spawn`'d task (e.g. the idle-unload reaper, the
//! realtime boot warmup) whose `JoinHandle` nobody polls, that unstructured
//! blob is the ONLY signal a panic happened at all -- tokio's own per-task
//! `catch_unwind` keeps the process alive and swallows everything else about
//! it, and a manually-launched `openasr serve` (no desktop sidecar capturing
//! stdio into a file) has nothing durable at all unless the operator happened
//! to redirect stderr themselves.
//!
//! This installs a panic hook, as early as possible in `openasr serve`'s
//! startup, that logs the message/location/thread through the *same*
//! `stage_timing::log_event` channel as every other boot line before chaining
//! to the previous (default) hook. It changes nothing about whether or how
//! the process continues -- only makes the fact of the panic land in
//! `daemon.log`, timestamped, in the same grep-able shape as everything
//! around it.

use std::panic::PanicHookInfo;

/// Extracts the human-readable panic message from a hook's payload. Matches
/// the two shapes `panic!`/`.unwrap()`/`.expect()` actually produce -- a
/// `&'static str` literal or an owned `String` -- and returns `None` for
/// anything else (a custom payload type has no reliable text form, so this
/// deliberately does not guess at one).
fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> Option<&str> {
    if let Some(message) = payload.downcast_ref::<&str>() {
        Some(message)
    } else {
        payload.downcast_ref::<String>().map(String::as_str)
    }
}

/// Pure formatter for the one line this hook logs. Kept separate from the
/// hook itself -- which cannot be unit tested without actually panicking --
/// so the line shape is covered by ordinary tests instead. Strips control
/// characters (a panic message is developer-authored code text, not user
/// content, but a stray newline would still split one panic into multiple
/// `daemon.log` lines and break line-oriented tailing/grepping) and falls
/// back to an explicit placeholder for each field that's unavailable, rather
/// than omitting it, so a reader can tell "logged but empty" apart from "this
/// hook version doesn't capture that field yet".
fn format_panic_log_line(
    message: Option<&str>,
    location: Option<&str>,
    thread_name: Option<&str>,
) -> String {
    let sanitized_message = message.map(|message| {
        message
            .chars()
            .map(|ch| if ch.is_control() { ' ' } else { ch })
            .collect::<String>()
    });
    format!(
        "thread={} location={} message={}",
        thread_name.unwrap_or("<unnamed>"),
        location.unwrap_or("<unknown>"),
        sanitized_message
            .as_deref()
            .unwrap_or("<non-string payload>"),
    )
}

/// Builds the log line for a live `PanicHookInfo`. The only piece of this
/// module that touches the real hook type, so the field-extraction logic
/// itself doesn't need duplicating between the hook and its tests.
fn panic_log_line(info: &PanicHookInfo<'_>) -> String {
    let message = panic_payload_message(info.payload());
    let location = info
        .location()
        .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()));
    let thread_name = std::thread::current().name().map(str::to_string);
    format_panic_log_line(message, location.as_deref(), thread_name.as_deref())
}

/// Installs the startup-diagnostics panic hook for `openasr serve`. Chains to
/// (rather than replaces) whatever hook was previously registered -- the
/// default one, unless something else already swapped it in -- so panic
/// behavior (abort/unwind, the default text dump) is unchanged; the only
/// difference is that the panic also lands in `daemon.log`, through the
/// shared `stage_timing` channel every other boot line uses. Must run before
/// anything else in `openasr serve`'s startup that could plausibly panic, so
/// no startup panic escapes unlogged.
pub fn install() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Logging must never itself panic -- that would recurse straight into
        // the default hook's abort/unwind path without ever reaching it.
        // `stage_timing::log_event` is a plain `eprintln!` (infallible) and
        // every step above it here (`panic_log_line`, payload downcasting,
        // formatting) is infallible too.
        openasr_core::stage_timing::log_event("panic", panic_log_line(info));
        default_hook(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::format_panic_log_line;

    #[test]
    fn formats_a_full_panic_with_message_and_location() {
        let line = format_panic_log_line(Some("boom"), Some("src/main.rs:12:5"), Some("main"));
        assert_eq!(line, "thread=main location=src/main.rs:12:5 message=boom");
    }

    #[test]
    fn falls_back_to_a_placeholder_when_the_message_is_missing() {
        let line = format_panic_log_line(None, Some("src/main.rs:12:5"), Some("main"));
        assert!(line.contains("message=<non-string payload>"));
    }

    #[test]
    fn falls_back_to_a_placeholder_when_the_location_is_missing() {
        let line = format_panic_log_line(Some("boom"), None, Some("main"));
        assert!(line.contains("location=<unknown>"));
    }

    #[test]
    fn falls_back_to_a_placeholder_when_the_thread_name_is_missing() {
        let line = format_panic_log_line(Some("boom"), Some("src/main.rs:12:5"), None);
        assert!(line.contains("thread=<unnamed>"));
    }

    #[test]
    fn sanitizes_control_characters_in_the_message_so_one_panic_is_one_line() {
        let line = format_panic_log_line(
            Some("first line\nsecond line\ttabbed"),
            Some("src/main.rs:12:5"),
            Some("main"),
        );
        assert_eq!(
            line,
            "thread=main location=src/main.rs:12:5 message=first line second line tabbed"
        );
    }
}
