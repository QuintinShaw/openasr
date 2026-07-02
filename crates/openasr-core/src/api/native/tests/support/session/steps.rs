use super::*;

pub(super) enum TestOnlyStreamingStep {
    Partial { revision: u64, text: &'static str },
    Final { revision: u64, text: &'static str },
    PostFinalSameText { revision: u64, text: &'static str },
    PostFinalRevision { revision: u64, text: &'static str },
}

pub(super) fn initial_script() -> VecDeque<TestOnlyStreamingStep> {
    VecDeque::from([
        TestOnlyStreamingStep::Partial {
            revision: 1,
            text: "hel",
        },
        TestOnlyStreamingStep::Partial {
            revision: 2,
            text: "hello wor",
        },
        TestOnlyStreamingStep::Final {
            revision: 3,
            text: "hello world",
        },
        TestOnlyStreamingStep::PostFinalSameText {
            revision: 4,
            text: "hello world",
        },
        TestOnlyStreamingStep::PostFinalRevision {
            revision: 5,
            text: "hello, world",
        },
    ])
}
