//! Shuts this process down when the process that spawned it (identified by
//! `openasr serve --parent-pid <pid>`) disappears without cleanly stopping
//! this one first.
//!
//! `openasr serve` is normally torn down by a supervisor (a desktop app's
//! process supervisor, or whatever launched a remote-compute server) via a
//! graceful path: an explicit stop call, `Drop`, or a Unix
//! SIGTERM/SIGINT/SIGHUP the supervisor forwards. All of those assume the
//! supervisor gets a chance to run its own shutdown code. A SIGKILL, a hard
//! crash, or a Force Quit/End Task on the supervisor skips every one of them,
//! leaving this process listening forever -- a silently orphaned daemon (and,
//! for a remote-compute server, a still-open network listener) that nothing
//! cleans up short of a reboot, because the next supervisor launch binds a
//! fresh ephemeral port rather than noticing the leftover.
//!
//! `--parent-pid` closes that gap independent of *how* the supervisor dies:
//! this process polls whether `parent_pid` is still alive and joins the same
//! graceful shutdown path as Ctrl-C / SIGTERM as soon as it is not, so ggml
//! backends can drop before process teardown. Calling `process::exit` here
//! skipped Rust `Drop` and aborted in ggml Metal atexit
//! (`GGML_ASSERT([rsets->data count] == 0)`).

use std::{thread, time::Duration};

/// How often to poll the parent for liveness. Frequent enough that an orphan
/// is cleaned up within a few seconds of the supervisor dying; infrequent
/// enough that it costs nothing meaningful over a daemon's lifetime.
const POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Receiver fired when the watched parent disappears. The serve loop awaits
/// this alongside Ctrl-C / SIGTERM so every exit path drops backends first.
pub(crate) type ParentShutdown = tokio::sync::oneshot::Receiver<()>;

/// Spawns a background thread that signals graceful shutdown shortly after
/// `parent_pid` disappears. A no-op if `parent_pid` is 0 (never a valid pid,
/// and clap's `Option<u32>` cannot itself reject it), so a malformed launch
/// arg degrades to "watchdog disabled" rather than an instant self-kill.
pub(crate) fn spawn(parent_pid: u32) -> Option<ParentShutdown> {
    if parent_pid == 0 {
        return None;
    }
    let (tx, rx) = tokio::sync::oneshot::channel();
    let spawned = thread::Builder::new()
        .name("parent-death-watchdog".to_string())
        .spawn(move || {
            loop {
                if !parent_is_alive(parent_pid) {
                    eprintln!(
                        "openasr serve: parent process {parent_pid} is gone; shutting down to avoid an orphaned daemon."
                    );
                    let _ = tx.send(());
                    return;
                }
                thread::sleep(POLL_INTERVAL);
            }
        });
    // A failure to spawn the watchdog thread (exhausted OS resources) must not
    // take the whole daemon down with it -- log and keep serving without the
    // safety net rather than aborting a healthy startup.
    if let Err(error) = spawned {
        eprintln!("openasr serve: could not start parent-death watchdog: {error}");
        return None;
    }
    Some(rx)
}

/// Completes when the operator sends Ctrl-C / SIGTERM or the parent watchdog
/// fires. Shared by `openasr serve` so parent-loss and interactive stop both
/// return from the server future instead of `process::exit`.
pub(crate) async fn wait_for_shutdown(parent_gone: Option<ParentShutdown>) {
    let parent_lost = async {
        match parent_gone {
            Some(rx) => {
                let _ = rx.await;
            }
            None => std::future::pending::<()>().await,
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = unix_terminate() => {}
        _ = parent_lost => {}
    }
}

#[cfg(unix)]
async fn unix_terminate() {
    match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
        Ok(mut signal) => {
            let _ = signal.recv().await;
        }
        Err(_) => std::future::pending::<()>().await,
    }
}

#[cfg(not(unix))]
async fn unix_terminate() {
    std::future::pending::<()>().await
}

/// Unix liveness probe: `kill(pid, 0)` sends no signal, it only asks the
/// kernel whether `pid` exists and is one this process may signal. Mirrors
/// the stale-pull-lock probe in `openasr-core`'s `pull.rs`.
#[cfg(unix)]
fn parent_is_alive(pid: u32) -> bool {
    // SAFETY: signal 0 is the documented no-op "check for existence" form of
    // kill(2); it has no effect on the target process.
    let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if result == 0 {
        return true;
    }
    // ESRCH: no such process -- the parent is gone. Any other errno (e.g.
    // EPERM for a pid owned by another user) is treated as "still alive":
    // fail closed toward NOT self-killing on an inconclusive probe.
    std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

/// Windows liveness probe: open the pid with a query-only access right and
/// read its exit code. A process that has exited keeps its pid reserved as
/// long as anyone still holds an open handle to it, so "OpenProcess
/// succeeded" is not proof of life -- only `STILL_ACTIVE` is. Mirrors the
/// equivalent probe in `openasr-core`'s `pull.rs`.
#[cfg(windows)]
fn parent_is_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    const STILL_ACTIVE: u32 = 259;

    // SAFETY: OpenProcess with a query-only access right is a read-only probe;
    // the handle (if any) is closed before returning.
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            // No process object for this pid at all -- definitely gone.
            return false;
        }
        let mut exit_code: u32 = 0;
        let queried = GetExitCodeProcess(handle, &mut exit_code);
        CloseHandle(handle);
        // queried == 0 -> status unreadable; be conservative and treat as
        // alive rather than risk a false-positive self-kill.
        queried == 0 || exit_code == STILL_ACTIVE
    }
}

#[cfg(not(any(unix, windows)))]
fn parent_is_alive(_pid: u32) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_is_a_noop_for_pid_zero() {
        // Must return promptly and must not spawn a thread that calls
        // process::exit on the test binary.
        spawn(0);
    }

    #[test]
    fn current_process_is_alive() {
        assert!(parent_is_alive(std::process::id()));
    }

    #[cfg(unix)]
    #[test]
    fn exited_child_is_not_alive() {
        let mut child = std::process::Command::new("true")
            .spawn()
            .expect("spawn `true`");
        let pid = child.id();
        child.wait().expect("wait for child");
        assert!(!parent_is_alive(pid));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn parent_loss_signals_shutdown_without_exiting() {
        let mut child = std::process::Command::new("true")
            .spawn()
            .expect("spawn `true`");
        let pid = child.id();
        child.wait().expect("wait for child");
        let rx = spawn(pid).expect("watchdog should start for a real pid");
        tokio::time::timeout(Duration::from_secs(8), rx)
            .await
            .expect("watchdog should notice the dead parent")
            .expect("watchdog oneshot should fire");
    }
}
