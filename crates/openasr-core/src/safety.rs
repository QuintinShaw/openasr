use std::path::{Component, Path};

pub fn current_platform_key() -> String {
    let os = match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "windows",
        "linux" => "linux",
        other => other,
    };
    format!("{os}-{}", std::env::consts::ARCH)
}

pub fn validate_platform_key(platform: &str) -> Result<(), String> {
    validate_platform_key_field("platform", platform)
}

pub fn validate_platform_key_field(field: &str, platform: &str) -> Result<(), String> {
    require_non_empty_value(field, platform)?;
    let Some((os, arch)) = platform.split_once('-') else {
        return Err(format!("platform '{platform}' must use <os>-<arch>"));
    };
    if platform.split('-').count() != 2 {
        return Err(format!("platform '{platform}' must use <os>-<arch>"));
    }
    if !["darwin", "linux", "windows"].contains(&os) {
        return Err(format!("platform '{platform}' uses unsupported os '{os}'"));
    }
    if !["aarch64", "x86_64"].contains(&arch) {
        return Err(format!(
            "platform '{platform}' uses unsupported arch '{arch}'"
        ));
    }
    Ok(())
}

pub fn validate_safe_relative_path(field: &str, value: &str) -> Result<(), String> {
    require_non_empty_value(field, value)?;
    if looks_like_windows_drive_path(value) {
        return Err(format!("{field} must be a portable relative path"));
    }
    let path = Path::new(value);
    if path.is_absolute() {
        return Err(format!("{field} must be a relative path"));
    }
    if value.contains('\\') {
        return Err(format!("{field} must use portable '/' path separators"));
    }
    if value.ends_with('/') {
        return Err(format!("{field} must not end with '/'"));
    }
    let mut has_normal_component = false;
    for component in path.components() {
        match component {
            Component::Normal(_) => has_normal_component = true,
            Component::ParentDir => {
                return Err(format!("{field} must not contain '..'"));
            }
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!("{field} must be a safe relative path"));
            }
        }
    }
    if !has_normal_component {
        return Err(format!("{field} must be a safe relative path"));
    }
    Ok(())
}

pub fn validate_sha256(field: &str, value: &str) -> Result<(), String> {
    if value.len() == 64 && value.chars().all(|character| character.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(format!("{field} must be exactly 64 hex characters"))
    }
}

fn require_non_empty_value(field: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    Ok(())
}

fn looks_like_windows_drive_path(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_key_is_deterministic_for_current_target() {
        let expected_os = match std::env::consts::OS {
            "macos" => "darwin",
            "windows" => "windows",
            "linux" => "linux",
            other => other,
        };
        assert_eq!(
            current_platform_key(),
            format!("{expected_os}-{}", std::env::consts::ARCH)
        );
    }

    #[test]
    fn safe_relative_path_validation_rejects_unsafe_values() {
        assert!(validate_safe_relative_path("path", "assets/model.bin").is_ok());
        assert!(
            validate_safe_relative_path("path", "../model.bin")
                .unwrap_err()
                .contains("must not contain '..'")
        );
        let absolute_path = if cfg!(windows) {
            "C:\\tmp\\model.bin"
        } else {
            "/tmp/model.bin"
        };
        assert!(
            validate_safe_relative_path("path", absolute_path)
                .unwrap_err()
                .contains("relative path")
        );
    }

    #[test]
    fn sha256_validation_requires_exact_hex() {
        assert!(
            validate_sha256(
                "sha256",
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            )
            .is_ok()
        );
        assert!(
            validate_sha256("sha256", "abc123")
                .unwrap_err()
                .contains("64 hex")
        );
    }
}
