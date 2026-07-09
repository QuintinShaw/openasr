//! Model-agnostic JSON hardening helpers shared by every safetensors header
//! parser in the tree (the shared `local_source_import` importer used by 14+
//! model families, and the whisper-specific local-source path). Kept here
//! rather than under `models/whisper/` so no model family logic leaks into
//! infrastructure that every family depends on.
//!
//! `serde_json` silently keeps the *last* value for a duplicate object key,
//! which would let a crafted safetensors header smuggle a tensor definition
//! past a naive "one entry per name" review while a byte-identical duplicate
//! key sits unused earlier in the file. Untrusted safetensors headers must be
//! rejected outright instead.

use std::collections::BTreeSet;
use std::fmt;

use serde::Deserializer;
use serde::de::{MapAccess, SeqAccess, Visitor};

/// Parse `contents` far enough to detect duplicate JSON object keys at any
/// nesting depth, without allocating a full DOM. Returns `Err` on the first
/// duplicate key found (or on any structural parse error, since a caller is
/// expected to also run a normal `serde_json::from_str` afterwards for the
/// typed decode).
pub(crate) fn reject_duplicate_json_keys(contents: &str) -> Result<(), serde_json::Error> {
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

#[cfg(test)]
mod tests {
    use super::reject_duplicate_json_keys;

    #[test]
    fn accepts_json_without_duplicate_keys() {
        reject_duplicate_json_keys(r#"{"a": 1, "b": {"c": 2}}"#)
            .expect("no duplicate keys should be accepted");
    }

    #[test]
    fn rejects_duplicate_top_level_key() {
        let error = reject_duplicate_json_keys(r#"{"a": 1, "a": 2}"#)
            .expect_err("duplicate top-level key must be rejected");
        assert!(error.to_string().contains("duplicate JSON object key"));
    }

    #[test]
    fn rejects_duplicate_nested_key() {
        let error = reject_duplicate_json_keys(r#"{"a": {"x": 1, "x": 2}}"#)
            .expect_err("duplicate nested key must be rejected");
        assert!(error.to_string().contains("duplicate JSON object key"));
    }

    #[test]
    fn rejects_duplicate_key_inside_array_of_objects() {
        let error = reject_duplicate_json_keys(r#"[{"a": 1}, {"a": 1, "a": 2}]"#)
            .expect_err("duplicate key nested in an array element must be rejected");
        assert!(error.to_string().contains("duplicate JSON object key"));
    }
}
