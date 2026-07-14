use std::{env, path::PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum OpenAsrHomeError {
    #[error(
        "Could not determine the OpenASR home directory. Set OPENASR_HOME to a writable directory."
    )]
    MissingHome,
}

// Resolution contract: OPENASR_HOME overrides, otherwise <user home>/.openasr.
// Downstream installers that cannot link this crate replicate this rule (e.g.
// the desktop NSIS uninstaller's app-data cleanup); keep them in sync when
// changing it.
pub fn openasr_home() -> Result<PathBuf, OpenAsrHomeError> {
    resolve_openasr_home(env::var_os("OPENASR_HOME"), user_home_dir())
}

pub fn resolve_openasr_home(
    openasr_home: Option<std::ffi::OsString>,
    user_home: Option<PathBuf>,
) -> Result<PathBuf, OpenAsrHomeError> {
    if let Some(path) = openasr_home.filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }

    user_home
        .map(|path| path.join(".openasr"))
        .ok_or(OpenAsrHomeError::MissingHome)
}

fn user_home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::*;

    #[test]
    fn openasr_home_override_wins() {
        let temp = tempfile::tempdir().unwrap();
        let resolved = resolve_openasr_home(
            Some(OsString::from(temp.path())),
            Some(PathBuf::from("/unused")),
        )
        .unwrap();

        assert_eq!(resolved, temp.path());
    }

    #[test]
    fn openasr_home_defaults_under_user_home() {
        let resolved = resolve_openasr_home(None, Some(PathBuf::from("/tmp/example"))).unwrap();

        assert_eq!(resolved, PathBuf::from("/tmp/example/.openasr"));
    }
}
