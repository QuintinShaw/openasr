//! Cross-language quant-tag contract.
//!
//! The desktop TypeScript `canonicalQuantTag` must canonicalize exactly like
//! `openasr_core::canonical_quant_tag`. Both sides consume the same JSON
//! fixture; if the mapping ever changes, change the fixture first and both
//! test suites will hold the implementations in lockstep.

use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    input: String,
    canonical: String,
}

#[test]
fn canonical_quant_tag_matches_shared_fixture() {
    let raw = include_str!("fixtures/quant_tag_cases.json");
    let fixture: Fixture = serde_json::from_str(raw).expect("quant tag fixture parses");
    assert!(
        fixture.cases.len() >= 10,
        "fixture must keep meaningful coverage"
    );
    for case in &fixture.cases {
        assert_eq!(
            openasr_core::canonical_quant_tag(&case.input),
            case.canonical,
            "canonical_quant_tag({:?}) drifted from the shared contract",
            case.input
        );
    }
}
