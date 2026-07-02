//! Standalone CI gate for the `openasr-system-audio` crate: it compiles the real
//! platform backend for the host and is cross-checked against Windows in CI,
//! exercising the crate's public `support_status()` contract.

#[cfg(test)]
mod tests {
    #[test]
    fn platform_support_contract_is_well_formed() {
        let support = openasr_system_audio::support_status();
        assert!(!support.platform.trim().is_empty());
        assert!(!support.label.trim().is_empty());
        assert!(!support.detail.trim().is_empty());
    }
}
