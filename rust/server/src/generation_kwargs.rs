//! The client-facing generation knobs, checked before a request is submitted:
//! a port of `mstar/utils/generation_kwargs.py`'s `normalize_generation_kwargs`
//! and the sampler's knob rules (`engine/resources/sampler/config.py`), with
//! the same refusal text. `null` is the same as leaving a key out, a value is
//! coerced to the knob's type only when it denotes exactly that value, and a
//! value out of range is refused. A multi-stage model's `<stage>_<knob>`
//! (`talker_temperature`) follows the rule for `<knob>`.
//!
//! The kwargs are an ordered list, as Python's dict is: the first bad knob in
//! insertion order is the one refused.

use serde_json::{Map, Number, Value};

use crate::adapters::py_value_repr;

/// A kwarg's value, with the floats JSON cannot hold (Python's `float("nan")`).
#[derive(Debug, Clone, PartialEq)]
pub enum Kwarg {
    Json(Value),
    Float(f64),
    /// An integer past JSON's range, as Python's int holds it.
    Int(i128),
}

impl Kwarg {
    fn repr(&self) -> String {
        match self {
            Kwarg::Json(v) => py_value_repr(v),
            Kwarg::Float(f) => crate::adapters::py_float_repr(*f),
            Kwarg::Int(i) => i.to_string(),
        }
    }
}

/// Insertion-ordered kwargs, as a Python dict.
pub type Kwargs = Vec<(String, Kwarg)>;

/// `mk.setdefault(key, value)`.
pub fn set_default(mk: &mut Kwargs, key: &str, value: Kwarg) {
    if !mk.iter().any(|(k, _)| k == key) {
        mk.push((key.to_string(), value));
    }
}

pub fn remove(mk: &mut Kwargs, key: &str) -> Option<Kwarg> {
    let i = mk.iter().position(|(k, _)| k == key)?;
    Some(mk.remove(i).1)
}

pub fn get<'a>(mk: &'a Kwargs, key: &str) -> Option<&'a Kwarg> {
    mk.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

type Rule<T> = (&'static str, &'static str, fn(T) -> bool);

/// knob -> (rule text, predicate), as `REAL_KNOB_RULES`.
const REAL_RULES: &[Rule<f64>] = &[
    ("temperature", ">= 0 (0 is greedy)", |v| v >= 0.0),
    ("top_p", "in (0, 1] (1 is off; greedy is temperature 0)", |v| 0.0 < v && v <= 1.0),
    ("min_p", "in [0, 1] (0 is off)", |v| (0.0..=1.0).contains(&v)),
    ("repetition_penalty", "> 0 (1 is off)", |v| v > 0.0),
];
/// `INT_KNOB_RULES` and `max_output_tokens`.
const INT_RULES: &[Rule<i128>] =
    &[("top_k", ">= 0 (0 is off)", |v| v >= 0), ("max_output_tokens", ">= 1", |v| v >= 1)];
/// `_BOOL_KNOBS`: the sampler's, and `do_sample`, `return_token_ids` and
/// `enable_thinking` (what a chat template reads; `"false"` is truthy there).
const BOOL_KNOBS: &[&str] =
    &["ignore_eos", "penalize_prompt", "return_logprobs", "do_sample", "return_token_ids", "enable_thinking"];
/// Other names for `max_output_tokens`, in precedence order after it.
const MAX_OUTPUT_TOKENS_ALIASES: &[&str] = &["max_new_tokens", "max_completion_tokens", "max_tokens"];

/// The knob `key` names: itself, or a `<stage>_<knob>` suffix of one.
fn knob<'a, T>(key: &str, rules: &'a [Rule<T>]) -> Option<&'a Rule<T>> {
    rules.iter().find(|(name, _, _)| key == *name || key.ends_with(&format!("_{name}")))
}

/// Python's `float(s)` of a string: surrounding whitespace, `_` between digits,
/// and `inf` / `infinity` / `nan` in any case. `None` where Python raises.
pub fn py_float(s: &str) -> Option<f64> {
    static NUMBER: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let number = NUMBER.get_or_init(|| {
        let digits = r"[0-9](?:_?[0-9])*";
        regex::Regex::new(&format!(
            r"^[+-]?(?:(?:{digits})(?:\.(?:{digits})?)?|\.{digits})(?:[eE][+-]?{digits})?$"
        ))
        .unwrap()
    });
    let t = s.trim();
    let lower = t.to_ascii_lowercase();
    let unsigned = lower.trim_start_matches(['+', '-']);
    if lower.len() - unsigned.len() <= 1 && matches!(unsigned, "inf" | "infinity" | "nan") {
        let negative = lower.starts_with('-');
        let f = if unsigned == "nan" { f64::NAN } else { f64::INFINITY };
        return Some(if negative { -f } else { f });
    }
    if !number.is_match(t) {
        return None;
    }
    t.replace('_', "").parse().ok()
}

/// An integral JSON number as Python's int (an int, or a float with no fractional part).
fn as_py_int(v: &Value) -> Option<i128> {
    match v {
        Value::Number(n) => n
            .as_i64()
            .map(i128::from)
            .or_else(|| n.as_u64().map(i128::from))
            .or_else(|| n.as_f64().filter(|f| f.fract() == 0.0 && f.abs() < 1.7e38).map(|f| f as i128)),
        _ => None,
    }
}

fn int_value(i: i128) -> Kwarg {
    match (i64::try_from(i), u64::try_from(i)) {
        (Ok(v), _) => Kwarg::Json(Value::from(v)),
        (_, Ok(v)) => Kwarg::Json(Value::from(v)),
        _ => Kwarg::Int(i),
    }
}

/// `_coerce_real`: a numeric string as its float; anything else unchanged.
fn coerce_real(v: Kwarg) -> Kwarg {
    match &v {
        Kwarg::Json(Value::String(s)) => match py_float(s) {
            Some(f) => match Number::from_f64(f) {
                Some(n) => Kwarg::Json(Value::Number(n)),
                None => Kwarg::Float(f),
            },
            None => v,
        },
        _ => v,
    }
}

/// `_coerce_int`: an integer-valued number or string as its int; anything else unchanged.
fn coerce_int(v: Kwarg) -> Kwarg {
    static INTEGRAL: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let integral = INTEGRAL.get_or_init(|| regex::Regex::new(r"^[+-]?[0-9]+(?:\.0*)?$").unwrap());
    match &v {
        Kwarg::Json(Value::Number(n)) if n.is_f64() => match n.as_f64() {
            Some(f) if f.fract() == 0.0 && f.is_finite() && f.abs() < 1.7e38 => int_value(f as i128),
            _ => v,
        },
        Kwarg::Json(Value::String(s)) => {
            let text = s.trim();
            if !integral.is_match(text) {
                return v;
            }
            match text.split('.').next().unwrap_or("").parse::<i128>() {
                Ok(i) => int_value(i),
                Err(_) => v,
            }
        }
        _ => v,
    }
}

fn number(v: &Kwarg) -> Option<f64> {
    match v {
        Kwarg::Json(Value::Number(n)) => n.as_f64(),
        Kwarg::Float(f) => Some(*f),
        Kwarg::Int(i) => Some(*i as f64),
        _ => None,
    }
}

fn integer(v: &Kwarg) -> Option<i128> {
    match v {
        Kwarg::Json(v @ Value::Number(n)) if !n.is_f64() => as_py_int(v),
        Kwarg::Int(i) => Some(*i),
        _ => None,
    }
}

/// A Python int as a kwarg: JSON when it fits.
pub fn int(i: i128) -> Kwarg {
    int_value(i)
}

/// `check_seed`.
fn check_seed(v: &Kwarg) -> Result<(), String> {
    let Some(seed) = integer(v) else {
        return Err(format!("seed must be an integer, got {}.", v.repr()));
    };
    if !(i128::from(i64::MIN)..=i128::from(i64::MAX)).contains(&seed) {
        return Err(format!("seed must fit in a signed 64-bit integer, got {seed}."));
    }
    Ok(())
}

/// `normalize_generation_kwargs`: nulls dropped, the length aliases folded into
/// `max_output_tokens`, and every known knob coerced and checked. The error is
/// the Python refusal's text.
pub fn normalize(mk: Kwargs) -> Result<Kwargs, String> {
    let mut kwargs: Kwargs = mk.into_iter().filter(|(_, v)| *v != Kwarg::Json(Value::Null)).collect();
    for alias in MAX_OUTPUT_TOKENS_ALIASES {
        if let Some(value) = remove(&mut kwargs, alias) {
            set_default(&mut kwargs, "max_output_tokens", value);
        }
    }
    for (key, value) in kwargs.iter_mut() {
        if key == "seed" {
            *value = coerce_int(value.clone());
            check_seed(value)?;
        } else if let Some((_, rule, ok)) = knob(key, REAL_RULES) {
            *value = coerce_real(value.clone());
            match (value.clone(), number(value)) {
                (Kwarg::Json(Value::Bool(_)), _) | (_, None) => {
                    return Err(format!("{key} must be a number {rule}; got {}", value.repr()));
                }
                (_, Some(f)) if !f.is_finite() || !ok(f) => {
                    return Err(format!("{key} must be a finite number {rule}; got {}", value.repr()));
                }
                _ => {}
            }
        } else if let Some((_, rule, ok)) = knob(key, INT_RULES) {
            *value = coerce_int(value.clone());
            if !integer(value).is_some_and(ok) {
                return Err(format!("{key} must be an integer {rule}; got {}", value.repr()));
            }
        } else if BOOL_KNOBS.iter().any(|name| key == name || key.ends_with(&format!("_{name}")))
            && !matches!(value, Kwarg::Json(Value::Bool(_)))
        {
            return Err(format!("{key} must be true or false; got {}", value.repr()));
        }
    }
    Ok(kwargs)
}

/// The kwargs as the JSON map a submission carries. A float JSON cannot hold
/// only survives a key that names no knob, which no model reads; it goes as
/// `null`.
pub fn into_map(mk: Kwargs) -> Map<String, Value> {
    mk.into_iter()
        .map(|(k, v)| {
            let v = match v {
                Kwarg::Json(v) => v,
                Kwarg::Float(f) => Number::from_f64(f).map_or(Value::Null, Value::Number),
                Kwarg::Int(i) => Number::from_f64(i as f64).map_or(Value::Null, Value::Number),
            };
            (k, v)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn kw(pairs: Value) -> Kwargs {
        pairs.as_object().unwrap().iter().map(|(k, v)| (k.clone(), Kwarg::Json(v.clone()))).collect()
    }

    #[test]
    fn numeric_strings_are_coerced_and_aliases_fold_into_max_output_tokens() {
        let out = into_map(normalize(kw(json!({"temperature": "0.5", "top_k": "5.0", "max_tokens": 9, "x": null}))).unwrap());
        assert_eq!(Value::Object(out), json!({"temperature": 0.5, "top_k": 5, "max_output_tokens": 9}));
    }

    #[test]
    fn a_bad_knob_is_refused_with_pythons_text() {
        let err = |v: Value| normalize(kw(v)).unwrap_err();
        assert_eq!(err(json!({"temperature": "hot"})), "temperature must be a number >= 0 (0 is greedy); got 'hot'");
        assert_eq!(err(json!({"talker_top_p": 2})), "talker_top_p must be a finite number in (0, 1] (1 is off; greedy is temperature 0); got 2");
        assert_eq!(err(json!({"top_k": 1.5})), "top_k must be an integer >= 0 (0 is off); got 1.5");
        assert_eq!(err(json!({"ignore_eos": "yes"})), "ignore_eos must be true or false; got 'yes'");
        assert_eq!(err(json!({"seed": true})), "seed must be an integer, got True.");
        assert_eq!(err(json!({"temperature": "inf"})), "temperature must be a finite number >= 0 (0 is greedy); got inf");
    }

    #[test]
    fn py_float_reads_what_python_float_reads() {
        for (s, f) in [(" 1.5 ", 1.5), ("1_000", 1000.0), ("-.5", -0.5), ("5.", 5.0), ("1e3", 1000.0), ("+Infinity", f64::INFINITY)] {
            assert_eq!(py_float(s), Some(f), "{s}");
        }
        assert!(py_float("nan").unwrap().is_nan());
        for s in ["", "1__0", "_1", "1_", "abc", "1e", "--1", "0x10", "in f"] {
            assert_eq!(py_float(s), None, "{s}");
        }
    }
}
