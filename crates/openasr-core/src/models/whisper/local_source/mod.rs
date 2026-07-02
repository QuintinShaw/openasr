use std::{
    collections::BTreeSet,
    fmt,
    path::{Path, PathBuf},
};

use serde::{
    Deserializer,
    de::{MapAccess, SeqAccess, Visitor},
};
use thiserror::Error;

mod safetensors;
pub(super) mod source_io;

pub use safetensors::{SafetensorsHeaderV0, SafetensorsTensorHeaderV0, load_safetensors_header_v0};

#[derive(Debug, Error)]
pub enum WhisperLocalSourceError {
    #[error("could not read model source file '{path}': {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not write model source file '{path}': {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not parse model source artifact '{path}': {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("{0}")]
    Validate(String),
}

pub(super) fn validate_error(message: String) -> WhisperLocalSourceError {
    WhisperLocalSourceError::Validate(message)
}

pub(super) fn checked_u64_add_with_context(
    left: u64,
    right: u64,
    context: impl Into<String>,
) -> Result<u64, WhisperLocalSourceError> {
    left.checked_add(right)
        .ok_or_else(|| validate_error(context.into()))
}

pub(super) fn tensor_validation_error(
    name: &str,
    detail: impl fmt::Display,
) -> WhisperLocalSourceError {
    validate_error(format!("tensor '{name}' {detail}"))
}

pub(super) fn reject_duplicate_json_keys(contents: &str) -> Result<(), serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_str(contents);
    deserializer.deserialize_any(DuplicateKeyVisitor)?;
    deserializer.end()
}

#[derive(Clone, Copy)]
struct DuplicateKeyVisitor;

impl<'de> serde::de::DeserializeSeed<'de> for DuplicateKeyVisitor {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for DuplicateKeyVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON without duplicate object keys")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<(), E> {
        Ok(())
    }
    fn visit_i64<E>(self, _value: i64) -> Result<(), E> {
        Ok(())
    }
    fn visit_u64<E>(self, _value: u64) -> Result<(), E> {
        Ok(())
    }
    fn visit_f64<E>(self, _value: f64) -> Result<(), E> {
        Ok(())
    }
    fn visit_str<E>(self, _value: &str) -> Result<(), E> {
        Ok(())
    }
    fn visit_borrowed_str<E>(self, _value: &'de str) -> Result<(), E> {
        Ok(())
    }
    fn visit_string<E>(self, _value: String) -> Result<(), E> {
        Ok(())
    }
    fn visit_none<E>(self) -> Result<(), E> {
        Ok(())
    }
    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_some<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(self)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<(), A::Error>
    where
        A: SeqAccess<'de>,
    {
        while seq.next_element_seed(DuplicateKeyVisitor)?.is_some() {}
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<(), A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut keys = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate JSON object key `{key}`"
                )));
            }
            map.next_value_seed(DuplicateKeyVisitor)?;
        }
        Ok(())
    }
}
