use super::{TranscriptLifecycleResult, TranscriptSegmentState, TranscriptUpdate};

pub(super) fn reject_out_of_order_or_stable_same_revision(
    state: &TranscriptSegmentState,
    update: &TranscriptUpdate,
    final_update: bool,
) -> Option<TranscriptLifecycleResult> {
    if update.revision < state.revision {
        return Some(TranscriptLifecycleResult::IgnoredOutOfOrder {
            current_revision: state.revision,
            incoming_revision: update.revision,
        });
    }

    if update.revision == state.revision {
        if final_update && !state.finalized {
            return None;
        }
        return Some(TranscriptLifecycleResult::IgnoredOutOfOrder {
            current_revision: state.revision,
            incoming_revision: update.revision,
        });
    }

    None
}

pub(super) fn reject_no_change_after_final(
    state: &TranscriptSegmentState,
    update: &TranscriptUpdate,
) -> Option<TranscriptLifecycleResult> {
    if state.finalized && update.text == state.text {
        return Some(TranscriptLifecycleResult::IgnoredNoChange {
            current_revision: state.revision,
        });
    }
    None
}
