use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::api::backend::Transcription;

mod json;
mod renderers;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    Json,
    VerboseJson,
    Srt,
    Vtt,
    Markdown,
}

impl ResponseFormat {
    pub const ALL: &'static [&'static str] =
        &["text", "json", "srt", "vtt", "verbose_json", "markdown"];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
            Self::VerboseJson => "verbose_json",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Markdown => "markdown",
        }
    }

    pub const fn output_extension(self) -> &'static str {
        match self {
            Self::Text => "txt",
            Self::Json => "json",
            Self::VerboseJson => "verbose.json",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::Markdown => "md",
        }
    }
}

impl fmt::Display for ResponseFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ResponseFormat {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            "verbose_json" => Ok(Self::VerboseJson),
            "srt" => Ok(Self::Srt),
            "vtt" => Ok(Self::Vtt),
            "markdown" => Ok(Self::Markdown),
            other => Err(format!(
                "Unsupported response format '{other}'. Use one of: {}.",
                Self::ALL.join(", ")
            )),
        }
    }
}

pub fn render_transcription(
    transcription: &Transcription,
    format: ResponseFormat,
) -> Result<String, serde_json::Error> {
    renderers::render(transcription, format)
}

#[cfg(test)]
mod tests;
