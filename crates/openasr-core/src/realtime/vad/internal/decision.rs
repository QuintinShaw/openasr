use super::{RealtimeAudioFrame, VadDecision, VadFrameDecision};

pub(super) fn decision_is_speech(decision: VadDecision, threshold: f32) -> bool {
    match decision {
        VadDecision::Speech => true,
        VadDecision::Silence => false,
        VadDecision::Probability(probability) => probability >= threshold,
    }
}

pub(super) fn frame_decision_from_energy(
    frame: &RealtimeAudioFrame,
    threshold: f32,
) -> VadFrameDecision {
    let rms = normalized_rms(frame.samples());
    VadFrameDecision {
        decision: if rms >= threshold {
            VadDecision::Speech
        } else {
            VadDecision::Silence
        },
        rms: Some(rms),
    }
}

fn normalized_rms(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean_square = samples
        .iter()
        .map(|sample| {
            let normalized = *sample as f64 / i16::MAX as f64;
            normalized * normalized
        })
        .sum::<f64>()
        / samples.len() as f64;
    mean_square.sqrt() as f32
}
