//! `TranscriptionRequest` (`api_server/openai/protocol.py`) from the multipart
//! form's fields, as the Python route builds it: each field but the declared
//! text ones `json.loads`-ed when it parses (`name[]` fields collected as a
//! list of strings), then
//! validated by pydantic's lax rules for the declared fields, with pydantic's
//! error text. Unknown fields pass through, in form order, as the request's
//! `extra` (pydantic's `model_extra`).

use serde_json::Value;

use crate::adapters::{py_float_repr, py_value_repr};
use crate::generation_kwargs::py_float;

/// A field's value as `json.loads` gives it: JSON, or one of the floats
/// Python's JSON reads and JSON proper doesn't (`NaN`, `Infinity`).
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    Json(Value),
    Float(f64),
}

impl FieldValue {
    /// `json.loads(text)`, or the text itself where that raises.
    pub fn from_form(text: String) -> FieldValue {
        match text.trim() {
            "NaN" => return FieldValue::Float(f64::NAN),
            "Infinity" => return FieldValue::Float(f64::INFINITY),
            "-Infinity" => return FieldValue::Float(f64::NEG_INFINITY),
            _ => {}
        }
        match serde_json::from_str::<Value>(&text) {
            Ok(v) => FieldValue::Json(v),
            Err(_) => FieldValue::Json(Value::String(text)),
        }
    }

    fn repr(&self) -> String {
        match self {
            FieldValue::Json(v) => py_value_repr(v),
            FieldValue::Float(f) => py_float_repr(*f),
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            FieldValue::Float(_) => "float",
            FieldValue::Json(v) => match v {
                Value::Null => "NoneType",
                Value::Bool(_) => "bool",
                Value::Number(n) if n.is_f64() => "float",
                Value::Number(_) => "int",
                Value::String(_) => "str",
                Value::Array(_) => "list",
                Value::Object(_) => "dict",
            },
        }
    }

    pub fn into_kwarg(self) -> crate::generation_kwargs::Kwarg {
        match self {
            FieldValue::Json(v) => crate::generation_kwargs::Kwarg::Json(v),
            FieldValue::Float(f) => crate::generation_kwargs::Kwarg::Float(f),
        }
    }
}

/// `TranscriptionRequest`, validated.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionRequest {
    pub model: Option<String>,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub response_format: String,
    pub temperature: Option<f64>,
    pub timestamp_granularities: Option<Vec<String>>,
    pub stream: Option<bool>,
    /// Python's int; JSON caps it far below where the range check refuses it.
    pub seed: Option<i128>,
    /// The fields the model doesn't declare, in form order.
    pub extra: Vec<(String, FieldValue)>,
}

impl Default for TranscriptionRequest {
    fn default() -> Self {
        Self {
            model: None,
            language: None,
            prompt: None,
            response_format: "json".to_string(),
            temperature: Some(0.0),
            timestamp_granularities: None,
            stream: Some(false),
            seed: None,
            extra: Vec::new(),
        }
    }
}

impl TranscriptionRequest {
    pub fn extra(&self, key: &str) -> Option<&FieldValue> {
        self.extra.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

/// One pydantic error: where, what, and the input.
struct LaxError {
    loc: String,
    kind: &'static str,
    msg: &'static str,
    input: FieldValue,
}

fn fail(loc: &str, kind: &'static str, msg: &'static str, input: &FieldValue) -> LaxError {
    LaxError { loc: loc.to_string(), kind, msg, input: input.clone() }
}

const STRING_TYPE: (&str, &str) = ("string_type", "Input should be a valid string");

fn lax_str(loc: &str, v: &FieldValue) -> Result<String, LaxError> {
    match v {
        FieldValue::Json(Value::String(s)) => Ok(s.clone()),
        _ => Err(fail(loc, STRING_TYPE.0, STRING_TYPE.1, v)),
    }
}

fn lax_float(loc: &str, v: &FieldValue) -> Result<f64, LaxError> {
    match v {
        FieldValue::Float(f) => Ok(*f),
        FieldValue::Json(Value::Bool(b)) => Ok(f64::from(u8::from(*b))),
        FieldValue::Json(Value::Number(n)) => Ok(n.as_f64().unwrap_or(f64::NAN)),
        FieldValue::Json(Value::String(s)) => py_float(s).ok_or_else(|| {
            fail(loc, "float_parsing", "Input should be a valid number, unable to parse string as a number", v)
        }),
        _ => Err(fail(loc, "float_type", "Input should be a valid number", v)),
    }
}

fn lax_int(loc: &str, v: &FieldValue) -> Result<i128, LaxError> {
    let from_float = |f: f64| {
        if !f.is_finite() {
            Err(fail(loc, "finite_number", "Input should be a finite number", v))
        } else if f.fract() != 0.0 {
            Err(fail(loc, "int_from_float", "Input should be a valid integer, got a number with a fractional part", v))
        } else {
            Ok(f as i128)
        }
    };
    match v {
        FieldValue::Float(f) => from_float(*f),
        FieldValue::Json(Value::Bool(b)) => Ok(i128::from(*b)),
        FieldValue::Json(Value::Number(n)) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => Ok(i128::from(i)),
            (_, Some(u)) => Ok(i128::from(u)),
            _ => from_float(n.as_f64().unwrap_or(f64::NAN)),
        },
        FieldValue::Json(Value::String(s)) => {
            let parsing = || {
                fail(loc, "int_parsing", "Input should be a valid integer, unable to parse string as an integer", v)
            };
            let t = s.trim();
            // digits with `_` between them, and a fraction of zeros
            static INT: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
            let int = INT.get_or_init(|| regex::Regex::new(r"^[+-]?[0-9](?:_?[0-9])*(?:\.0*)?$").unwrap());
            if !int.is_match(t) {
                return Err(parsing());
            }
            t.split('.').next().unwrap_or("").replace('_', "").parse::<i128>().map_err(|_| parsing())
        }
        _ => Err(fail(loc, "int_type", "Input should be a valid integer", v)),
    }
}

fn lax_bool(loc: &str, v: &FieldValue) -> Result<bool, LaxError> {
    let parsing = || fail(loc, "bool_parsing", "Input should be a valid boolean, unable to interpret input", v);
    match v {
        FieldValue::Json(Value::Bool(b)) => Ok(*b),
        FieldValue::Json(Value::Number(n)) if !n.is_f64() => match n.as_i64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(parsing()),
        },
        FieldValue::Json(Value::Number(n)) => match n.as_f64() {
            Some(0.0) => Ok(false),
            Some(1.0) => Ok(true),
            _ => Err(fail(loc, "bool_type", "Input should be a valid boolean", v)),
        },
        FieldValue::Json(Value::String(s)) => match s.to_ascii_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Ok(false),
            "1" | "on" | "t" | "true" | "y" | "yes" => Ok(true),
            _ => Err(parsing()),
        },
        _ => Err(fail(loc, "bool_type", "Input should be a valid boolean", v)),
    }
}

fn lax_str_list(loc: &str, v: &FieldValue, errors: &mut Vec<LaxError>) -> Option<Vec<String>> {
    let FieldValue::Json(Value::Array(items)) = v else {
        errors.push(fail(loc, "list_type", "Input should be a valid list", v));
        return None;
    };
    let mut out = Vec::new();
    let before = errors.len();
    for (i, item) in items.iter().enumerate() {
        match lax_str(&format!("{loc}.{i}"), &FieldValue::Json(item.clone())) {
            Ok(s) => out.push(s),
            Err(e) => errors.push(e),
        }
    }
    (errors.len() == before).then_some(out)
}

/// pydantic's `input_value`: the repr, cut to its first 25 and last 24
/// characters around `...` past 50.
fn truncated_repr(v: &FieldValue) -> String {
    let repr = v.repr();
    let chars: Vec<char> = repr.chars().collect();
    if chars.len() <= 50 {
        return repr;
    }
    let head: String = chars[..25].iter().collect();
    let tail: String = chars[chars.len() - 24..].iter().collect();
    format!("{head}...{tail}")
}

/// `str(ValidationError)`. `errors_url` is pydantic's docs prefix for its
/// version (`https://errors.pydantic.dev/2.14/v/`), which the text ends each
/// error with.
fn render(errors: &[LaxError], errors_url: &str) -> String {
    let n = errors.len();
    let mut out = format!("{n} validation error{} for TranscriptionRequest", if n == 1 { "" } else { "s" });
    for e in errors {
        out.push_str(&format!(
            "\n{}\n  {} [type={}, input_value={}, input_type={}]\n    For further information visit {}{}",
            e.loc,
            e.msg,
            e.kind,
            truncated_repr(&e.input),
            e.input.type_name(),
            errors_url,
            e.kind
        ));
    }
    out
}

/// `TranscriptionRequest(**fields)`: the request, or pydantic's error text.
pub fn validate(fields: Vec<(String, FieldValue)>, errors_url: &str) -> Result<TranscriptionRequest, String> {
    let mut req = TranscriptionRequest::default();
    let mut errors = Vec::new();
    let field = |name: &str| fields.iter().find(|(k, _)| k == name).map(|(_, v)| v);
    let null = |v: &FieldValue| matches!(v, FieldValue::Json(Value::Null));
    // pydantic reports in the model's field order
    for name in ["model", "language", "prompt"] {
        if let Some(v) = field(name).filter(|v| !null(v)) {
            match lax_str(name, v) {
                Ok(s) => match name {
                    "model" => req.model = Some(s),
                    "language" => req.language = Some(s),
                    _ => req.prompt = Some(s),
                },
                Err(e) => errors.push(e),
            }
        }
    }
    if let Some(v) = field("response_format") {
        match lax_str("response_format", v) {
            Ok(s) => req.response_format = s,
            Err(e) => errors.push(e),
        }
    }
    if let Some(v) = field("temperature") {
        req.temperature = if null(v) {
            None
        } else {
            match lax_float("temperature", v) {
                Ok(f) => Some(f),
                Err(e) => {
                    errors.push(e);
                    None
                }
            }
        };
    }
    if let Some(v) = field("timestamp_granularities").filter(|v| !null(v)) {
        req.timestamp_granularities = lax_str_list("timestamp_granularities", v, &mut errors);
    }
    if let Some(v) = field("stream") {
        req.stream = if null(v) {
            None
        } else {
            match lax_bool("stream", v) {
                Ok(b) => Some(b),
                Err(e) => {
                    errors.push(e);
                    None
                }
            }
        };
    }
    if let Some(v) = field("seed").filter(|v| !null(v)) {
        match lax_int("seed", v) {
            Ok(i) => req.seed = Some(i),
            Err(e) => errors.push(e),
        }
    }
    if !errors.is_empty() {
        return Err(render(&errors, errors_url));
    }
    const DECLARED: [&str; 8] =
        ["model", "language", "prompt", "response_format", "temperature", "timestamp_granularities", "stream", "seed"];
    req.extra = fields.into_iter().filter(|(k, _)| !DECLARED.contains(&k.as_str())).collect();
    Ok(req)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const URL: &str = "https://errors.pydantic.dev/2.14/v/";

    fn fields(v: Value) -> Vec<(String, FieldValue)> {
        v.as_object().unwrap().iter().map(|(k, v)| (k.clone(), FieldValue::Json(v.clone()))).collect()
    }

    #[test]
    fn lax_coercion_matches_pydantic() {
        let req = validate(fields(json!({"temperature": " 1.5 ", "seed": "1_0", "stream": "TRUE", "x": [1]})), URL).unwrap();
        assert_eq!((req.temperature, req.seed, req.stream), (Some(1.5), Some(10), Some(true)));
        assert_eq!(req.extra, vec![("x".to_string(), FieldValue::Json(json!([1])))]);
        let req = validate(fields(json!({"seed": true, "temperature": true, "stream": 1.0})), URL).unwrap();
        assert_eq!((req.temperature, req.seed, req.stream), (Some(1.0), Some(1), Some(true)));
    }

    #[test]
    fn errors_read_as_pydantics_in_field_order() {
        let err = validate(fields(json!({"seed": "x", "prompt": 1.5})), URL).unwrap_err();
        assert_eq!(
            err,
            "2 validation errors for TranscriptionRequest\nprompt\n  Input should be a valid string \
             [type=string_type, input_value=1.5, input_type=float]\n    For further information visit \
             https://errors.pydantic.dev/2.14/v/string_type\nseed\n  Input should be a valid integer, unable to \
             parse string as an integer [type=int_parsing, input_value='x', input_type=str]\n    For further \
             information visit https://errors.pydantic.dev/2.14/v/int_parsing"
        );
        let long = validate(fields(json!({"temperature": "x".repeat(60)})), URL).unwrap_err();
        assert!(long.contains("input_value='xxxxxxxxxxxxxxxxxxxxxxxx...xxxxxxxxxxxxxxxxxxxxxxx', input_type=str"), "{long}");
    }
}
