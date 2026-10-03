//! Minimal `NSKeyedArchiver` decoder for the profiler's `streamData`.
//!
//! `streamData` is a keyed archive whose values are themselves keyed archives
//! stored as `NSData` (e.g. every `APSCounterData[i]`). [`ArchiveValue::Data`]
//! keeps those blobs; call [`decode`] on them to descend.

use std::collections::BTreeMap;
use std::io::Cursor;

use plist::{Dictionary, Uid, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum ArchiveValue {
    Null,
    Bool(bool),
    Integer(i128),
    Real(f64),
    String(String),
    Data(Vec<u8>),
    Array(Vec<ArchiveValue>),
    Dictionary(BTreeMap<String, ArchiveValue>),
}

impl ArchiveValue {
    pub fn get(&self, key: &str) -> Option<&ArchiveValue> {
        match self {
            Self::Dictionary(map) => map.get(key),
            _ => None,
        }
    }

    pub fn as_dictionary(&self) -> Option<&BTreeMap<String, ArchiveValue>> {
        match self {
            Self::Dictionary(map) => Some(map),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[ArchiveValue]> {
        match self {
            Self::Array(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_data(&self) -> Option<&[u8]> {
        match self {
            Self::Data(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Integer(value) => u64::try_from(*value).ok(),
            Self::Real(value) if *value >= 0.0 => Some(*value as u64),
            Self::Bool(value) => Some(*value as u64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => i64::try_from(*value)
                .ok()
                .or_else(|| u64::try_from(*value).ok().map(|value| value as i64)),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Integer(value) => Some(*value as f64),
            Self::Real(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            Self::Integer(value) => Some(*value != 0),
            _ => None,
        }
    }

    /// Decode a `Data` value that holds a nested keyed archive.
    pub fn nested(&self) -> Option<ArchiveValue> {
        self.as_data().and_then(decode)
    }
}

/// Decode a keyed archive (or a plain binary plist) into its root value.
pub fn decode(bytes: &[u8]) -> Option<ArchiveValue> {
    let value = Value::from_reader(Cursor::new(bytes)).ok()?;
    let Some(archive) = value.as_dictionary() else {
        return Some(plain(&value));
    };
    if archive.get("$archiver").and_then(Value::as_string) != Some("NSKeyedArchiver") {
        return Some(plain(&value));
    }
    let objects = archive.get("$objects").and_then(Value::as_array)?;
    let root = archive
        .get("$top")
        .and_then(Value::as_dictionary)
        .and_then(|top| top.get("root"))?;
    Some(resolve(objects, root, 0))
}

fn plain(value: &Value) -> ArchiveValue {
    match value {
        Value::Boolean(value) => ArchiveValue::Bool(*value),
        Value::Integer(value) => value
            .as_signed()
            .map(|value| ArchiveValue::Integer(value.into()))
            .or_else(|| value.as_unsigned().map(|value| ArchiveValue::Integer(value.into())))
            .unwrap_or(ArchiveValue::Null),
        Value::Real(value) => ArchiveValue::Real(*value),
        Value::String(value) => ArchiveValue::String(value.clone()),
        Value::Data(value) => ArchiveValue::Data(value.clone()),
        Value::Array(values) => ArchiveValue::Array(values.iter().map(plain).collect()),
        Value::Dictionary(map) => ArchiveValue::Dictionary(
            map.iter()
                .map(|(key, value)| (key.clone(), plain(value)))
                .collect(),
        ),
        _ => ArchiveValue::Null,
    }
}

fn resolve(objects: &[Value], value: &Value, depth: usize) -> ArchiveValue {
    if depth > 64 {
        return ArchiveValue::Null;
    }
    let value = match value {
        Value::Uid(uid) => match object(objects, *uid) {
            Some(value) => value,
            None => return ArchiveValue::Null,
        },
        other => other,
    };
    match value {
        Value::String(value) if value == "$null" => ArchiveValue::Null,
        Value::Dictionary(map) => resolve_object(objects, map, depth),
        Value::Array(values) => ArchiveValue::Array(
            values
                .iter()
                .map(|value| resolve(objects, value, depth + 1))
                .collect(),
        ),
        other => plain(other),
    }
}

fn resolve_object(objects: &[Value], map: &Dictionary, depth: usize) -> ArchiveValue {
    if let (Some(keys), Some(values)) = (
        map.get("NS.keys").and_then(Value::as_array),
        map.get("NS.objects").and_then(Value::as_array),
    ) {
        return ArchiveValue::Dictionary(
            keys.iter()
                .zip(values)
                .filter_map(|(key, value)| {
                    let key = match resolve(objects, key, depth + 1) {
                        ArchiveValue::String(key) => key,
                        ArchiveValue::Integer(key) => key.to_string(),
                        _ => return None,
                    };
                    Some((key, resolve(objects, value, depth + 1)))
                })
                .collect(),
        );
    }
    if let Some(values) = map.get("NS.objects").and_then(Value::as_array) {
        return ArchiveValue::Array(
            values
                .iter()
                .map(|value| resolve(objects, value, depth + 1))
                .collect(),
        );
    }
    if let Some(string) = map.get("NS.string").and_then(Value::as_string) {
        return ArchiveValue::String(string.to_owned());
    }
    if let Some(data) = map.get("NS.data").and_then(Value::as_data) {
        return ArchiveValue::Data(data.to_vec());
    }
    if map.contains_key("$class") {
        return ArchiveValue::Dictionary(
            map.iter()
                .filter(|(key, _)| key.as_str() != "$class")
                .map(|(key, value)| (key.clone(), resolve(objects, value, depth + 1)))
                .collect(),
        );
    }
    plain(&Value::Dictionary(map.clone()))
}

fn object(objects: &[Value], uid: Uid) -> Option<&Value> {
    objects.get(uid.get() as usize)
}
