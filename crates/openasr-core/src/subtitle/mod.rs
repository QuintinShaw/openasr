//! Unified subtitle / reading-timeline projection for finished transcriptions.
//!
//! Product pipeline (single fact source):
//!
//! ```text
//! word anchors -> speaker attribution -> reading segments / subtitle cues -> SRT/VTT
//! ```
//!
//! - Reading segments are speaker-merged paragraphs for the manuscript view.
//! - Subtitle cues are short presentation units for SRT/VTT and on-screen display.
//! - Both views share the same attributed word timeline; SRT/VTT never invent a
//!   second segmentation.

mod anchors;
pub mod cues;
mod reading;
mod refine;
mod timeline;

pub use anchors::{
    WordAnchorIssue, WordAnchorQuality, WordAnchorValidation, validate_word_anchors,
};
pub use cues::{resegment_segments_into_cues, resegment_transcription_cues, segment_into_cues};
pub use reading::merge_reading_segments;
pub use timeline::{
    ForcedAlignmentDecision, TimelinePrecisionPolicy, TimelineProjectOptions, TimelineQuality,
    decide_forced_alignment, project_transcription, strip_unrequested_word_timestamps,
    timed_cues_for_export,
};
