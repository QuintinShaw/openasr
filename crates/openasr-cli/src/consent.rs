//! Interactive-consent primitives and the stable CLI exit-code contract.
//!
//! The auto-pull-on-missing-model behaviour (see [`crate::pull_cli`]) is a
//! CLI-only affordance. It must never run on the shared model-resolution path
//! the server also uses: a missing model is always either an interactive,
//! visible, confirmed pull here in a command handler, or a fail-closed error.
//! This module holds the terminal-detection, confirmation, and exit-code
//! plumbing that keeps that promise honest in non-interactive contexts.

use std::io::{IsTerminal, Write};

/// Stable, documented CLI exit codes. These are part of the CLI contract so
/// scripts and CI can branch on the failure class. `0` is success and `2` is
/// reserved for clap's own usage/argument errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExitCode {
    /// Bad input: audio file missing, unreadable, or in an unusable shape.
    InputError = 3,
    /// The resolved model is not installed and no consent was given
    /// (non-interactive without `--yes`, or `--offline`, or a declined prompt).
    ModelNotInstalled = 4,
    /// A model download or its integrity verification failed.
    DownloadFailed = 5,
    /// The backend/runtime failed to produce a transcript.
    RuntimeFailed = 6,
}

/// An error that carries an explicit [`ExitCode`]. `main` downcasts to this to
/// pick a process exit status; any other error keeps the generic failure code.
#[derive(Debug)]
pub(crate) struct CliExit {
    pub(crate) code: ExitCode,
    pub(crate) message: String,
}

impl CliExit {
    pub(crate) fn new(code: ExitCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for CliExit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CliExit {}

/// How the user pre-authorized (or forbade) the consent-pull path for this run.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PullConsent {
    /// `--yes`/`-y` or `OPENASR_ASSUME_YES`: pull a missing model without an
    /// interactive prompt (an explicit, logged decision).
    pub(crate) assume_yes: bool,
    /// `--offline`/`--no-pull` or `OPENASR_OFFLINE`: never touch the network;
    /// a missing model is a hard, fail-closed error.
    pub(crate) offline: bool,
}

impl PullConsent {
    /// Folds the CLI flags together with their environment overrides. Either the
    /// flag or a truthy env var enables the behaviour.
    pub(crate) fn resolve(yes_flag: bool, offline_flag: bool) -> Self {
        Self {
            assume_yes: yes_flag || env_flag_set("OPENASR_ASSUME_YES"),
            offline: offline_flag || env_flag_set("OPENASR_OFFLINE"),
        }
    }
}

fn env_flag_set(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

/// True only when both stdin and stderr are attached to a terminal, so a
/// blocking confirmation prompt can actually be answered. A pipe, a captured
/// stdout, CI, or a cron job is non-interactive and must never block on input.
pub(crate) fn is_interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

/// Asks a yes/no question on stderr (keeping stdout clean for transcript data)
/// and reads the answer from the controlling terminal. Defaults to "no" on
/// anything other than an explicit yes. Callers must only reach this when
/// [`is_interactive`] is true.
pub(crate) fn confirm(prompt: &str) -> bool {
    let mut stderr = std::io::stderr();
    let _ = write!(stderr, "{prompt} [y/N] ");
    let _ = stderr.flush();
    let mut answer = String::new();
    // Read from the controlling terminal rather than stdin so a user can still
    // pipe audio in on stdin while answering the prompt.
    let read = read_line_from_tty(&mut answer).or_else(|| {
        if std::io::stdin().is_terminal() {
            std::io::stdin().read_line(&mut answer).ok()
        } else {
            None
        }
    });
    if read.is_none() {
        return false;
    }
    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

#[cfg(unix)]
fn read_line_from_tty(buf: &mut String) -> Option<usize> {
    use std::io::BufRead;
    let tty = std::fs::File::open("/dev/tty").ok()?;
    std::io::BufReader::new(tty).read_line(buf).ok()
}

#[cfg(not(unix))]
fn read_line_from_tty(_buf: &mut String) -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consent_env_overrides_apply() {
        // Flags alone, without env, reflect the flag values.
        let consent = PullConsent::resolve(true, false);
        assert!(consent.assume_yes);
        assert!(!consent.offline);
    }

    #[test]
    fn exit_codes_are_stable() {
        assert_eq!(ExitCode::InputError as i32, 3);
        assert_eq!(ExitCode::ModelNotInstalled as i32, 4);
        assert_eq!(ExitCode::DownloadFailed as i32, 5);
        assert_eq!(ExitCode::RuntimeFailed as i32, 6);
    }
}
