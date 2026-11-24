//! Shared `openpi_client.msgpack_numpy` codec for `ws/openpi.rs` and `actions.rs`.
//!
//! Decodes both `bytes` (`Mp::Binary`) and `str` (`Mp::String`) marker keys
//! (`__ndarray__`, `__npgeneric__`, `data`, `dtype`, `shape`) while encoding
//! arrays with `Mp::Binary` marker keys (matching `use_bin_type=True`).

use rmpv::Value as Mp;

use crate::adapters::{py_float_repr, RawArray};

/// A numeric dtype as numpy names one: its kind (`b`, `i`, `u`, `f`), size and byte order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dtype {
    pub kind: char,
    pub size: usize,
    pub little: bool,
}

impl Dtype {
    /// `dtype.str`.
    pub fn str(&self) -> String {
        let order = if self.size == 1 {
            '|'
        } else if self.little {
            '<'
        } else {
            '>'
        };
        format!("{order}{}{}", self.kind, self.size)
    }

    /// How numpy prints it: its name in the native order, else its `str`.
    pub fn name(&self) -> String {
        if !self.little && self.size > 1 {
            return self.str();
        }
        match self.kind {
            'b' => "bool".to_string(),
            'i' => format!("int{}", self.size * 8),
            'u' => format!("uint{}", self.size * 8),
            _ => format!("float{}", self.size * 8),
        }
    }
}

/// A value as `msgpack.unpackb(raw=False, object_hook=_unpack_array)` gives it.
#[derive(Debug, Clone)]
pub enum Obs {
    Nil,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Obs>),
    Map(Vec<(Obs, Obs)>),
    Array {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    },
    Scalar {
        dtype: Dtype,
        value: f64,
        int: Option<i128>,
    },
    Other,
}

impl Obs {
    pub fn get(&self, key: &str) -> Option<&Obs> {
        let Obs::Map(pairs) = self else { return None };
        pairs
            .iter()
            .rev()
            .find(|(k, _)| matches!(k, Obs::Str(s) if s == key))
            .map(|(_, v)| v)
    }

    /// Look up a `msgpack_numpy` marker key, accepting both `bytes` and `str` keys.
    fn get_marker_key<'a>(pairs: &'a [(Obs, Obs)], key: &str) -> Option<&'a Obs> {
        pairs
            .iter()
            .rev()
            .find(|(k, _)| {
                matches!(k, Obs::Bytes(b) if b == key.as_bytes())
                    || matches!(k, Obs::Str(s) if s == key)
            })
            .map(|(_, v)| v)
    }

    /// `type(value).__name__`.
    pub fn type_name(&self) -> String {
        match self {
            Obs::Nil => "NoneType".into(),
            Obs::Bool(_) => "bool".into(),
            Obs::Int(_) => "int".into(),
            Obs::Float(_) => "float".into(),
            Obs::Str(_) => "str".into(),
            Obs::Bytes(_) => "bytes".into(),
            Obs::List(_) => "list".into(),
            Obs::Map(_) => "dict".into(),
            Obs::Array { .. } => "ndarray".into(),
            Obs::Scalar { dtype, .. } => dtype.name(),
            Obs::Other => "ExtType".into(),
        }
    }

    /// Python's `repr`.
    pub fn repr(&self) -> String {
        match self {
            Obs::Nil => "None".into(),
            Obs::Bool(b) => if *b { "True" } else { "False" }.into(),
            Obs::Int(i) => i.to_string(),
            Obs::Float(f) => py_float_repr(*f),
            Obs::Str(s) => crate::media::py_repr(s),
            Obs::Bytes(b) => bytes_repr(b),
            Obs::List(items) => {
                format!(
                    "[{}]",
                    items.iter().map(Obs::repr).collect::<Vec<_>>().join(", ")
                )
            }
            Obs::Map(pairs) => format!(
                "{{{}}}",
                pairs
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k.repr(), v.repr()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            other => format!("<{}>", other.type_name()),
        }
    }
}

/// Python's `repr(bytes)`.
fn bytes_repr(b: &[u8]) -> String {
    let quote = if b.contains(&b'\'') && !b.contains(&b'"') {
        '"'
    } else {
        '\''
    };
    let mut out = format!("b{quote}");
    for &c in b {
        match c {
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            c if c as char == quote => {
                out.push('\\');
                out.push(c as char);
            }
            0x20..=0x7e => out.push(c as char),
            c => out.push_str(&format!("\\x{c:02x}")),
        }
    }
    out.push(quote);
    out
}

/// `_numeric_dtype`: a dtype string for bool, an integer or a float.
pub fn numeric_dtype(name: &Obs) -> Result<Dtype, String> {
    let Obs::Str(s) = name else {
        return Err(format!(
            "an array's dtype must be a dtype string, not {}",
            name.repr()
        ));
    };
    let named = |kind: char, size: usize| Some(Dtype { kind, size, little: true });
    let parsed = match s.as_str() {
        "bool" | "?" | "b1" => named('b', 1),
        "int8" | "i1" | "byte" => named('i', 1),
        "uint8" | "u1" | "ubyte" | "B" => named('u', 1),
        "int16" | "short" | "h" => named('i', 2),
        "uint16" | "ushort" | "H" => named('u', 2),
        "int32" | "intc" | "i" => named('i', 4),
        "uint32" | "uintc" | "I" => named('u', 4),
        "int64" | "int" | "long" | "l" | "q" | "intp" => named('i', 8),
        "uint64" | "ulong" | "L" | "Q" | "uintp" => named('u', 8),
        "float16" | "half" | "e" => named('f', 2),
        "float32" | "single" | "f" => named('f', 4),
        "float64" | "double" | "float" | "d" => named('f', 8),
        other => {
            // a type string: [<>|=]?<kind><size>
            let (order, rest) = match other.chars().next() {
                Some(c @ ('<' | '>' | '|' | '=')) => (c, &other[1..]),
                _ => ('=', other),
            };
            let mut chars = rest.chars();
            let first = chars.next();
            let tail = chars.as_str();
            if first == Some('O') && tail.is_empty() {
                return Err("array dtype |O is not bool, integer or float".to_string());
            }
            match (first, tail.parse::<usize>()) {
                (Some(kind), Ok(size)) if "biufcmMOSUV".contains(kind) => {
                    if !"biuf".contains(kind) {
                        // numpy's dtype.str: no byte order for bytes, voids and objects
                        let order = if "OSV".contains(kind) || (size == 1 && kind != 'U') {
                            '|'
                        } else if order == '>' {
                            '>'
                        } else {
                            '<'
                        };
                        return Err(format!(
                            "array dtype {order}{kind}{size} is not bool, integer or float"
                        ));
                    }
                    let valid = match kind {
                        'b' => size == 1,
                        'f' => [2, 4, 8].contains(&size),
                        _ => [1, 2, 4, 8].contains(&size),
                    };
                    valid.then_some(Dtype {
                        kind,
                        size,
                        little: order != '>',
                    })
                }
                _ => None,
            }
        }
    };
    parsed.ok_or_else(|| format!("unknown array dtype {}", crate::media::py_repr(s)))
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let frac = (h & 0x3ff) as f32;
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 => {
            if frac == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        e => sign * (1.0 + frac / 1024.0) * 2f32.powi(e as i32 - 15),
    }
}

/// `f32` rounded to the nearest half, as numpy's float16 arithmetic rounds it.
fn round_to_f16(x: f32) -> f32 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let a = x.abs();
    if a >= 65520.0 {
        return x.signum() * f32::INFINITY;
    }
    // the spacing of halves around `a`: subnormal below 2^-14
    let exp = a.log2().floor().max(-14.0);
    let ulp = 2f32.powf(exp - 10.0);
    let q = a / ulp;
    let r = q.round();
    let r = if (q - q.floor() - 0.5).abs() < f32::EPSILON && r % 2.0 != 0.0 {
        r - 1.0
    } else {
        r
    };
    x.signum() * r * ulp
}

/// Element `i` of an array's data as f64 (and its integer, for an integer dtype).
pub fn element(dtype: Dtype, data: &[u8], i: usize) -> (f64, Option<i128>) {
    let b = &data[i * dtype.size..(i + 1) * dtype.size];
    let mut buf = [0u8; 8];
    buf[..dtype.size].copy_from_slice(b);
    if !dtype.little {
        buf[..dtype.size].reverse();
    }
    let u = u64::from_le_bytes(buf);
    match (dtype.kind, dtype.size) {
        ('b', _) => ((u != 0) as u8 as f64, Some((u != 0) as i128)),
        ('u', _) => (u as f64, Some(u as i128)),
        ('i', n) => {
            let shift = 64 - 8 * n as u32;
            let v = ((u << shift) as i64) >> shift;
            (v as f64, Some(v as i128))
        }
        ('f', 2) => (f16_to_f32(u as u16) as f64, None),
        ('f', 4) => (f32::from_bits(u as u32) as f64, None),
        _ => (f64::from_bits(u), None),
    }
}

/// `_unpack_array`: an array or a numpy scalar, or the map as it is.
fn unpack_array(pairs: Vec<(Obs, Obs)>) -> Result<Obs, String> {
    if Obs::get_marker_key(&pairs, "__ndarray__").is_some() {
        let (data, dtype, shape) = (
            Obs::get_marker_key(&pairs, "data"),
            Obs::get_marker_key(&pairs, "dtype"),
            Obs::get_marker_key(&pairs, "shape"),
        );
        let (Some(data), Some(dtype), Some(shape)) = (data, dtype, shape) else {
            return Err("an array needs data, dtype and shape".into());
        };
        let dtype = numeric_dtype(dtype)?;
        let dims: Option<Vec<usize>> = match shape {
            Obs::List(items) => items
                .iter()
                .map(|n| match n {
                    Obs::Int(i) if *i >= 0 => usize::try_from(*i).ok(),
                    _ => None,
                })
                .collect(),
            _ => None,
        };
        let (Obs::Bytes(data), Some(dims)) = (data, dims) else {
            return Err("an array needs bytes data and a shape of non-negative ints".into());
        };
        let want = dims.iter().try_fold(dtype.size, |acc, d| acc.checked_mul(*d));
        if want != Some(data.len()) {
            let shape = match dims.len() {
                1 => format!("({},)", dims[0]),
                _ => format!(
                    "({})",
                    dims.iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            };
            return Err(format!(
                "an array of shape {shape} and dtype {} is not {} bytes",
                dtype.str(),
                data.len()
            ));
        }
        return Ok(Obs::Array {
            dtype,
            shape: dims,
            data: data.clone(),
        });
    }
    if Obs::get_marker_key(&pairs, "__npgeneric__").is_some() {
        let (Some(data), Some(dtype)) = (
            Obs::get_marker_key(&pairs, "data"),
            Obs::get_marker_key(&pairs, "dtype"),
        ) else {
            return Err("a scalar needs data and dtype".into());
        };
        let (value, int) = match data {
            Obs::Bool(b) => (*b as u8 as f64, Some(*b as i128)),
            Obs::Int(i) => (*i as f64, Some(*i)),
            Obs::Float(f) => (*f, None),
            other => {
                return Err(format!(
                    "a scalar's data must be a number, not {}",
                    other.type_name()
                ))
            }
        };
        let dtype = numeric_dtype(dtype)?;
        if let (Some(i), 'i' | 'u') = (int, dtype.kind) {
            let bits = 8 * dtype.size as u32;
            let (lo, hi) = if dtype.kind == 'u' {
                (0, (1i128 << bits) - 1)
            } else {
                (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
            };
            if !(lo..=hi).contains(&i) {
                return Err(format!(
                    "Python integer {i} out of bounds for {}",
                    dtype.name()
                ));
            }
        }
        let int = match dtype.kind {
            'i' | 'u' => Some(int.unwrap_or(value.trunc() as i128)),
            'b' => Some((value != 0.0) as i128),
            _ => None,
        };
        return Ok(Obs::Scalar { dtype, value, int });
    }
    Ok(Obs::Map(pairs))
}

fn from_msgpack(v: Mp) -> Result<Obs, String> {
    Ok(match v {
        Mp::Nil => Obs::Nil,
        Mp::Boolean(b) => Obs::Bool(b),
        Mp::Integer(i) => Obs::Int(
            i.as_i64()
                .map(i128::from)
                .or_else(|| i.as_u64().map(i128::from))
                .unwrap_or(0),
        ),
        Mp::F32(f) => Obs::Float(f as f64),
        Mp::F64(f) => Obs::Float(f),
        Mp::String(s) => match s.into_str() {
            Some(s) => Obs::Str(s),
            None => return Err("'utf-8' codec can't decode the string".into()),
        },
        Mp::Binary(b) => Obs::Bytes(b),
        Mp::Array(items) => {
            Obs::List(items.into_iter().map(from_msgpack).collect::<Result<_, _>>()?)
        }
        Mp::Map(pairs) => unpack_array(
            pairs
                .into_iter()
                .map(|(k, v)| Ok((from_msgpack(k)?, from_msgpack(v)?)))
                .collect::<Result<_, String>>()?,
        )?,
        Mp::Ext(..) => Obs::Other,
    })
}

/// Unpack a raw `msgpack_numpy` payload without the `"the observation is not openpi's msgpack: "` prefix.
pub fn unpack_raw(data: &[u8]) -> Result<Obs, String> {
    let mut cursor = data;
    let value = rmpv::decode::read_value(&mut cursor).map_err(|e| e.to_string())?;
    if !cursor.is_empty() {
        return Err("unpack(b) received extra data.".into());
    }
    from_msgpack(value)
}

/// `unpack`: an observation, or why it isn't one (with openpi's error prefix).
pub fn unpack_openpi(data: &[u8]) -> Result<Obs, String> {
    unpack_raw(data).map_err(|e| format!("the observation is not openpi's msgpack: {e}"))
}

/// Pack a numpy ndarray into an `Mp::Map` with `Mp::Binary` marker keys.
pub fn pack_ndarray(dtype: &str, shape: &[usize], data: Vec<u8>) -> Mp {
    Mp::Map(vec![
        (Mp::Binary(b"__ndarray__".to_vec()), Mp::Boolean(true)),
        (Mp::Binary(b"data".to_vec()), Mp::Binary(data)),
        (Mp::Binary(b"dtype".to_vec()), Mp::String(dtype.into())),
        (
            Mp::Binary(b"shape".to_vec()),
            Mp::Array(shape.iter().map(|&n| Mp::from(n as u64)).collect()),
        ),
    ])
}

/// Encode an `rmpv::Value` to msgpack bytes.
pub fn encode(value: &Mp) -> Vec<u8> {
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, value).expect("pack into a Vec");
    out
}

/// numpy's `float.astype(uint8)` on x86-64: through an int32, keeping its low byte.
fn to_u8(x: f64) -> u8 {
    if x.is_nan() || x.abs() >= 2147483648.0 {
        0
    } else {
        (x.trunc() as i32) as u8
    }
}

/// Convert a 3D image slice (`[3, H, W]` or `[H, W, 3]`, uint8 or float) into `RawArray` (`|u1`, `[3, H, W]`).
pub fn image_slice(
    dtype: Dtype,
    shape: &[usize],
    data: &[u8],
    key: &str,
) -> Result<RawArray, String> {
    if shape.len() != 3 {
        return Err(format!("{key} must be an image array of 3 dimensions"));
    }
    let n: usize = shape.iter().product();
    let bytes: Vec<u8> = match (dtype.kind, dtype.size) {
        // `255 * value` in the array's own precision, then the cast
        ('f', 2) => (0..n)
            .map(|i| to_u8(round_to_f16(255.0 * element(dtype, data, i).0 as f32) as f64))
            .collect(),
        ('f', 4) => (0..n)
            .map(|i| to_u8((255.0f32 * element(dtype, data, i).0 as f32) as f64))
            .collect(),
        ('f', _) => (0..n)
            .map(|i| to_u8(255.0 * element(dtype, data, i).0))
            .collect(),
        ('u', 1) => data.to_vec(),
        _ => return Err(format!("{key} must be uint8 or float, not {}", dtype.name())),
    };
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    let (chw, out) = if h == 3 {
        (shape.to_vec(), bytes)
    } else {
        // [H, W, C] -> [C, H, W]
        let mut out = vec![0u8; n];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..c {
                    out[ch * h * w + y * w + x] = bytes[(y * w + x) * c + ch];
                }
            }
        }
        (vec![c, h, w], out)
    };
    if chw[0] != 3 || chw.contains(&0) {
        let dims = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "{key} must be an RGB image, [H, W, 3] or [3, H, W]; got [{dims}]"
        ));
    }
    Ok(RawArray {
        dtype: "|u1".to_string(),
        shape: chw,
        data: out,
    })
}

/// `_image`: an observation image as RGB uint8 `[3, H, W]`, floats scaled by 255.
pub fn image(value: &Obs, key: &str) -> Result<RawArray, String> {
    let Obs::Array { dtype, shape, data } = value else {
        return Err(format!("{key} must be an image array of 3 dimensions"));
    };
    image_slice(*dtype, shape, data, key)
}

/// `_numbers`: a value as a flat run of numbers.
pub fn numbers(value: &Obs, key: &str) -> Result<Vec<f64>, String> {
    let numeric = || format!("{key} must be numeric");
    fn flat(v: &Obs, out: &mut Vec<f64>) -> bool {
        match v {
            Obs::Bool(b) => out.push(*b as u8 as f64),
            Obs::Int(i) => out.push(*i as f64),
            Obs::Float(f) => out.push(*f),
            Obs::Scalar { value, .. } => out.push(*value),
            Obs::Array { dtype, shape, data } => {
                let n: usize = shape.iter().product();
                out.extend((0..n).map(|i| element(*dtype, data, i).0));
            }
            Obs::List(items) => return items.iter().all(|item| flat(item, out)),
            _ => return false,
        }
        true
    }
    let mut out = Vec::new();
    if flat(value, &mut out) {
        Ok(out)
    } else {
        Err(numeric())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ndarray_bin(dtype: &str, shape: &[u64], data: Vec<u8>) -> Mp {
        Mp::Map(vec![
            (Mp::Binary(b"__ndarray__".to_vec()), Mp::Boolean(true)),
            (Mp::Binary(b"data".to_vec()), Mp::Binary(data)),
            (Mp::Binary(b"dtype".to_vec()), Mp::String(dtype.into())),
            (
                Mp::Binary(b"shape".to_vec()),
                Mp::Array(shape.iter().map(|n| Mp::from(*n)).collect()),
            ),
        ])
    }

    fn ndarray_str(dtype: &str, shape: &[u64], data: Vec<u8>) -> Mp {
        Mp::Map(vec![
            (Mp::String("__ndarray__".into()), Mp::Boolean(true)),
            (Mp::String("data".into()), Mp::Binary(data)),
            (Mp::String("dtype".into()), Mp::String(dtype.into())),
            (
                Mp::String("shape".into()),
                Mp::Array(shape.iter().map(|n| Mp::from(*n)).collect()),
            ),
        ])
    }

    #[test]
    fn decodes_both_binary_and_string_marker_keys() {
        let bytes: Vec<u8> = [1.5f32, -2.25f32]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        for map in [
            ndarray_bin("<f4", &[2], bytes.clone()),
            ndarray_str("<f4", &[2], bytes.clone()),
        ] {
            let encoded = encode(&map);
            let Obs::Array { dtype, shape, data } = unpack_raw(&encoded).unwrap() else {
                panic!("expected Obs::Array");
            };
            assert_eq!(dtype.str(), "<f4");
            assert_eq!(shape, vec![2]);
            assert_eq!(data, bytes);
        }

        let scalar_str = Mp::Map(vec![
            (Mp::String("__npgeneric__".into()), Mp::Boolean(true)),
            (Mp::String("data".into()), Mp::from(7i64)),
            (Mp::String("dtype".into()), Mp::String("<i4".into())),
        ]);
        let Obs::Scalar { dtype, int, .. } = unpack_raw(&encode(&scalar_str)).unwrap() else {
            panic!("expected Obs::Scalar");
        };
        assert_eq!(dtype.str(), "<i4");
        assert_eq!(int, Some(7));
    }

    #[test]
    fn pack_ndarray_emits_binary_marker_keys_and_roundtrips() {
        let data: Vec<u8> = [0.25f32, 0.75f32]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let packed = encode(&pack_ndarray("<f4", &[1, 2], data.clone()));
        let Mp::Map(pairs) = rmpv::decode::read_value(&mut &packed[..]).unwrap() else {
            panic!("expected Mp::Map");
        };
        assert_eq!(pairs[0].0, Mp::Binary(b"__ndarray__".to_vec()));
        let Obs::Array {
            dtype,
            shape,
            data: out,
        } = unpack_raw(&packed).unwrap()
        else {
            panic!("expected Obs::Array");
        };
        assert_eq!(dtype.str(), "<f4");
        assert_eq!(shape, vec![1, 2]);
        assert_eq!(out, data);
    }

    #[test]
    fn rejects_non_numeric_dtypes_malformed_lengths_and_extra_bytes() {
        let bad_len = encode(&ndarray_str("<f4", &[2], vec![0; 4]));
        assert_eq!(
            unpack_openpi(&bad_len).unwrap_err(),
            "the observation is not openpi's msgpack: an array of shape (2,) and dtype <f4 is not 4 bytes"
        );
        let bad_dtype = encode(&ndarray_str("<U3", &[1], vec![0; 12]));
        assert_eq!(
            unpack_openpi(&bad_dtype).unwrap_err(),
            "the observation is not openpi's msgpack: array dtype <U3 is not bool, integer or float"
        );
        let obj_dtype = encode(&ndarray_bin("|O", &[1], vec![0; 8]));
        assert!(unpack_raw(&obj_dtype)
            .unwrap_err()
            .contains("is not bool, integer or float"));

        let mut trailing = encode(&Mp::Map(vec![]));
        trailing.push(0x01);
        assert_eq!(
            unpack_raw(&trailing).unwrap_err(),
            "unpack(b) received extra data."
        );
    }
}
