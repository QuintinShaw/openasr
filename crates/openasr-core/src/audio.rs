mod decode;
mod errors;
mod prepare;
mod probe;
mod symphonia_decode;
mod types;
mod validation;

use std::path::Path;

use crate::BackendKind;

pub use errors::{AudioInputError, AudioPreparationError};
pub(crate) use types::RECOGNIZED_EXTENSIONS;
pub use types::{AudioInputInfo, AudioInputIssue, AudioPreparationOptions, PreparedAudioInput};

pub fn recognized_audio_extensions() -> &'static [&'static str] {
    RECOGNIZED_EXTENSIONS
}

pub fn probe_audio_input(path: impl AsRef<Path>) -> Result<AudioInputInfo, AudioInputError> {
    let path = path.as_ref();
    validation::validate_regular_file(path)?;
    Ok(probe::probe_audio_details(path))
}

pub fn validate_audio_input(path: impl AsRef<Path>) -> Result<AudioInputInfo, AudioInputError> {
    probe_audio_input(path)
}

pub fn prepare_audio_input(
    path: impl AsRef<Path>,
    options: &AudioPreparationOptions,
) -> Result<PreparedAudioInput, AudioPreparationError> {
    let info = probe_audio_input(path)?;

    match options.backend {
        BackendKind::Mock => {
            let prepared_path = info.path.clone();
            Ok(PreparedAudioInput {
                original: info,
                samples: types::PreparedAudioSamples::Path(prepared_path),
                temp_dir: None,
            })
        }
        BackendKind::Native => prepare::prepare_external_input(info, options),
    }
}

pub fn probe_wav_duration(path: impl AsRef<Path>) -> Option<f64> {
    decode::probe_wav_duration_inner(path.as_ref())
        .ok()
        .flatten()
}

#[cfg(test)]
mod tests;
