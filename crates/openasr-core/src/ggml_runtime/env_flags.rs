fn normalize_env_bool(raw: Option<&str>) -> Option<bool> {
    match raw.map(str::trim).map(str::to_ascii_lowercase) {
        Some(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on") => Some(true),
        Some(value) if matches!(value.as_str(), "0" | "false" | "no" | "off" | "none") => {
            Some(false)
        }
        _ => None,
    }
}

pub(crate) fn env_var_truthy(var_name: &str) -> bool {
    normalize_env_bool(std::env::var(var_name).ok().as_deref()) == Some(true)
}

pub(crate) fn env_toggle_with_raw(
    disable_raw: Option<&str>,
    enable_raw: Option<&str>,
    default_enabled: bool,
) -> bool {
    if normalize_env_bool(disable_raw) == Some(true) {
        return false;
    }
    if let Some(value) = normalize_env_bool(enable_raw) {
        return value;
    }
    default_enabled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_toggle_respects_disable_precedence() {
        for value in [
            Some("1"),
            Some("true"),
            Some("TRUE"),
            Some("yes"),
            Some("on"),
        ] {
            assert!(!env_toggle_with_raw(value, Some("1"), true));
        }
    }

    #[test]
    fn env_toggle_parses_enable_when_present() {
        for value in [
            Some("1"),
            Some("true"),
            Some("TRUE"),
            Some("yes"),
            Some("on"),
        ] {
            assert!(env_toggle_with_raw(None, value, false));
        }
        for value in [
            Some("0"),
            Some("false"),
            Some("FALSE"),
            Some("no"),
            Some("off"),
        ] {
            assert!(!env_toggle_with_raw(None, value, true));
        }
    }

    #[test]
    fn env_toggle_uses_default_for_unrecognized_or_missing_enable() {
        assert!(env_toggle_with_raw(None, None, true));
        assert!(!env_toggle_with_raw(None, None, false));
        assert!(env_toggle_with_raw(None, Some("maybe"), true));
        assert!(!env_toggle_with_raw(None, Some("maybe"), false));
    }
}
