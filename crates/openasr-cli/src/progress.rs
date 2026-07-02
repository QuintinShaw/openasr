//! Shared model-pull progress UX so `pull` and the transcribe/live consent-pull
//! render downloads identically: an `indicatif` bar on a TTY (percent, rate, ETA,
//! human-readable sizes), and periodic plain lines on a pipe/CI where a bar would
//! just spew control codes.

use indicatif::{ProgressBar, ProgressStyle};
use openasr_core::PullProgress;
use std::io::IsTerminal;

/// Renders [`PullProgress`] events. Drive it with `|event| reporter.on(event)`.
pub(crate) struct PullReporter {
    pull: String,
    bar: Option<ProgressBar>,
    last_plain: u64,
}

impl PullReporter {
    pub(crate) fn new(pull: &str) -> Self {
        Self {
            pull: pull.to_string(),
            bar: None,
            last_plain: 0,
        }
    }

    pub(crate) fn on(&mut self, event: PullProgress) {
        match event {
            PullProgress::UsingInstalled { path } => {
                eprintln!("Already installed: {}", path.display());
            }
            PullProgress::DownloadStarted {
                bytes_total,
                resume_from,
            } => {
                if bytes_total > 0 && std::io::stderr().is_terminal() {
                    let bar = ProgressBar::new(bytes_total);
                    bar.set_style(
                        ProgressStyle::with_template(
                            "{msg} [{bar:30}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA {eta})",
                        )
                        .expect("static progress template")
                        .progress_chars("=>-"),
                    );
                    bar.set_message(format!("Downloading {}", self.pull));
                    bar.set_position(resume_from);
                    self.bar = Some(bar);
                } else if resume_from > 0 {
                    eprintln!(
                        "Resuming {} from {resume_from}/{bytes_total} bytes",
                        self.pull
                    );
                } else {
                    eprintln!("Downloading {} ({bytes_total} bytes)", self.pull);
                }
            }
            PullProgress::Downloading {
                bytes_done,
                bytes_total,
            } => {
                if let Some(bar) = &self.bar {
                    bar.set_position(bytes_done);
                } else if bytes_done == bytes_total
                    || bytes_done.saturating_sub(self.last_plain) >= 8 * 1024 * 1024
                {
                    self.last_plain = bytes_done;
                    eprintln!("Downloaded {bytes_done}/{bytes_total} bytes");
                }
            }
            PullProgress::Verifying { .. } => {
                self.clear_bar();
                eprintln!("Verifying {}", self.pull);
            }
            PullProgress::Installed { path } => {
                self.clear_bar();
                eprintln!("Installed {} at {}", self.pull, path.display());
            }
        }
    }

    fn clear_bar(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
    }
}
