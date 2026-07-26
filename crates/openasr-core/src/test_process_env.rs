//! Test-only serialization and restoration for process environment overrides.
//!
//! Process environment is shared by the whole test binary. Keep every test
//! override under one lock, and restore it while the lock is still held so a
//! parallel test can never observe a partially restored environment.

#![cfg(test)]

use std::ffi::{OsStr, OsString};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

static TEST_PROCESS_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn lock() -> MutexGuard<'static, ()> {
    TEST_PROCESS_ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// RAII scope for process environment overrides in unit tests.
///
/// The lock is acquired before reading or changing any key. Drop restores keys
/// in reverse order, then releases the lock. This also makes restoration work
/// during unwinding without catching or suppressing the original panic.
pub(crate) struct TestProcessEnvGuard {
    _lock: MutexGuard<'static, ()>,
    previous: Vec<(OsString, Option<OsString>)>,
}

impl TestProcessEnvGuard {
    pub(crate) fn new<I, K>(overrides: I) -> Self
    where
        I: IntoIterator<Item = (K, Option<OsString>)>,
        K: AsRef<OsStr>,
    {
        let lock = lock();
        let previous = overrides
            .into_iter()
            .map(|(key, value)| {
                let key = key.as_ref().to_os_string();
                let old = std::env::var_os(&key);
                match value {
                    Some(value) => {
                        #[expect(unsafe_code, reason = "test-only process env override")]
                        unsafe {
                            std::env::set_var(&key, value);
                        }
                    }
                    None => {
                        #[expect(unsafe_code, reason = "test-only process env override")]
                        unsafe {
                            std::env::remove_var(&key);
                        }
                    }
                }
                (key, old)
            })
            .collect();
        Self {
            _lock: lock,
            previous,
        }
    }
}

impl Drop for TestProcessEnvGuard {
    fn drop(&mut self) {
        for (key, previous) in self.previous.drain(..).rev() {
            match previous {
                Some(value) => {
                    #[expect(unsafe_code, reason = "test-only process env restore")]
                    unsafe {
                        std::env::set_var(key, value);
                    }
                }
                None => {
                    #[expect(unsafe_code, reason = "test-only process env restore")]
                    unsafe {
                        std::env::remove_var(key);
                    }
                }
            }
        }
    }
}

pub(crate) fn with_test_process_env<T, I, K>(overrides: I, run: impl FnOnce() -> T) -> T
where
    I: IntoIterator<Item = (K, Option<OsString>)>,
    K: AsRef<OsStr>,
{
    let _guard = TestProcessEnvGuard::new(overrides);
    run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::mpsc;
    use std::time::Duration;

    const KEY_UNSET: &str = "OPENASR_TEST_PROCESS_ENV_GUARD_UNSET_7C6E";
    const KEY_PREEXISTING: &str = "OPENASR_TEST_PROCESS_ENV_GUARD_PREEXISTING_7C6E";
    const KEY_MULTI_A: &str = "OPENASR_TEST_PROCESS_ENV_GUARD_MULTI_A_7C6E";
    const KEY_MULTI_B: &str = "OPENASR_TEST_PROCESS_ENV_GUARD_MULTI_B_7C6E";
    const KEY_PANIC: &str = "OPENASR_TEST_PROCESS_ENV_GUARD_PANIC_7C6E";
    const KEY_CONCURRENT: &str = "OPENASR_TEST_PROCESS_ENV_GUARD_CONCURRENT_7C6E";

    #[test]
    fn unset_set_and_restore() {
        let original = std::env::var_os(KEY_UNSET);
        with_test_process_env([(KEY_UNSET, Some(OsString::from("set")))], || {
            assert_eq!(std::env::var_os(KEY_UNSET), Some(OsString::from("set")));
        });
        assert_eq!(std::env::var_os(KEY_UNSET), original);
    }

    #[test]
    fn preexisting_value_is_restored() {
        let original = std::env::var_os(KEY_PREEXISTING);
        with_test_process_env([(KEY_PREEXISTING, Some(OsString::from("before")))], || {
            assert_eq!(
                std::env::var_os(KEY_PREEXISTING),
                Some(OsString::from("before"))
            );
        });
        assert_eq!(std::env::var_os(KEY_PREEXISTING), original);
    }

    #[test]
    fn multi_key_values_restore_in_reverse_order() {
        let original_a = std::env::var_os(KEY_MULTI_A);
        let original_b = std::env::var_os(KEY_MULTI_B);
        with_test_process_env(
            [
                (KEY_MULTI_A, Some(OsString::from("during"))),
                (KEY_MULTI_B, None),
            ],
            || {
                assert_eq!(
                    std::env::var_os(KEY_MULTI_A),
                    Some(OsString::from("during"))
                );
                assert_eq!(std::env::var_os(KEY_MULTI_B), None);
            },
        );
        assert_eq!(std::env::var_os(KEY_MULTI_A), original_a);
        assert_eq!(std::env::var_os(KEY_MULTI_B), original_b);
    }

    #[test]
    fn panic_restores_and_poison_recovery_keeps_panic_visible() {
        let original = std::env::var_os(KEY_PANIC);
        let panic_result = catch_unwind(AssertUnwindSafe(|| {
            with_test_process_env([(KEY_PANIC, Some(OsString::from("panic")))], || {
                assert_eq!(std::env::var_os(KEY_PANIC), Some(OsString::from("panic")));
                panic!("expected test panic");
            });
        }));
        assert!(panic_result.is_err());
        assert_eq!(std::env::var_os(KEY_PANIC), original);
        with_test_process_env([(KEY_PANIC, Some(OsString::from("after")))], || {
            assert_eq!(std::env::var_os(KEY_PANIC), Some(OsString::from("after")));
        });
        assert_eq!(std::env::var_os(KEY_PANIC), original);
    }

    #[test]
    fn concurrent_overrides_wait_for_restore_before_entering() {
        let original = std::env::var_os(KEY_CONCURRENT);
        let (first_entered_tx, first_entered_rx) = mpsc::channel();
        let (release_first_tx, release_first_rx) = mpsc::channel();
        let first = std::thread::spawn(move || {
            with_test_process_env([(KEY_CONCURRENT, Some(OsString::from("first")))], || {
                first_entered_tx.send(()).expect("signal first guard");
                release_first_rx.recv().expect("release first guard");
                assert_eq!(
                    std::env::var_os(KEY_CONCURRENT),
                    Some(OsString::from("first"))
                );
            });
        });
        first_entered_rx.recv().expect("first guard entered");

        let (second_attempted_tx, second_attempted_rx) = mpsc::channel();
        let (second_entered_tx, second_entered_rx) = mpsc::channel();
        let second = std::thread::spawn(move || {
            second_attempted_tx.send(()).expect("signal second attempt");
            with_test_process_env([(KEY_CONCURRENT, Some(OsString::from("second")))], || {
                second_entered_tx.send(()).expect("signal second guard");
                assert_eq!(
                    std::env::var_os(KEY_CONCURRENT),
                    Some(OsString::from("second"))
                );
            });
        });
        second_attempted_rx.recv().expect("second guard attempted");
        assert!(
            second_entered_rx
                .recv_timeout(Duration::from_millis(50))
                .is_err(),
            "second guard entered before the first restored the process environment"
        );

        release_first_tx.send(()).expect("release first guard");
        second_entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("second guard should enter after the first restores");
        first.join().expect("first override thread");
        second.join().expect("second override thread");
        assert_eq!(std::env::var_os(KEY_CONCURRENT), original);
    }
}
