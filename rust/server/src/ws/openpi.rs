//! openpi's WebSocket policy protocol at the server's root, a port of
//! `api_server/openpi.py`: one metadata frame on connect, then one
//! observation in and one action chunk out per call, arrays in
//! `openpi_client.msgpack_numpy`'s encoding. A call that fails is answered
//! with its error as a text frame and the socket closes, as openpi's server
//! does. The deployment's `openpi:` block, checked against its model, comes
//! from the launcher as `MSTAR_OPENPI` ([`Policy`]).

use std::sync::OnceLock;
use std::time::Instant;

use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use rmpv::Value as Mp;
use serde::Deserialize;

use crate::adapters::{py_float_repr, RawArray, SubmitArgs};
use crate::serving::{rid, try_result_stream, AppState, Out};

/// The deployment's `OpenpiPolicy`.
#[derive(Debug, Deserialize)]
pub struct Policy {
    pub image_keys: Vec<String>,
    pub state_keys: Vec<String>,
    pub state_kwarg: String,
    pub action_horizon: usize,
    pub action_dim: usize,
    pub action_dims: usize,
    /// The metadata frame, packed as the Python side packs it, base64.
    pub metadata: String,
}

pub fn policy() -> Option<&'static Policy> {
    static POLICY: OnceLock<Option<Policy>> = OnceLock::new();
    POLICY
        .get_or_init(|| {
            let raw = std::env::var("MSTAR_OPENPI").ok().filter(|s| !s.is_empty())?;
            Some(serde_json::from_str(&raw).expect("MSTAR_OPENPI is the launcher's JSON"))
        })
        .as_ref()
}

// ── openpi_client.msgpack_numpy ─────────────────────────────────────────────

/// A numeric dtype as numpy names one: its kind (`b`, `i`, `u`, `f`), size and byte order.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Dtype {
    kind: char,
    size: usize,
    little: bool,
}

impl Dtype {
    /// `dtype.str`.
    fn str(&self) -> String {
        let order = if self.size == 1 { '|' } else if self.little { '<' } else { '>' };
        format!("{order}{}{}", self.kind, self.size)
    }

    /// How numpy prints it: its name in the native order, else its `str`.
    fn name(&self) -> String {
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
enum Obs {
    Nil,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Obs>),
    Map(Vec<(Obs, Obs)>),
    Array { dtype: Dtype, shape: Vec<usize>, data: Vec<u8> },
    Scalar { dtype: Dtype, value: f64, int: Option<i128> },
    Other,
}

impl Obs {
    fn get(&self, key: &str) -> Option<&Obs> {
        let Obs::Map(pairs) = self else { return None };
        pairs.iter().rev().find(|(k, _)| matches!(k, Obs::Str(s) if s == key)).map(|(_, v)| v)
    }

    fn get_bytes_key<'a>(pairs: &'a [(Obs, Obs)], key: &[u8]) -> Option<&'a Obs> {
        pairs.iter().rev().find(|(k, _)| matches!(k, Obs::Bytes(b) if b == key)).map(|(_, v)| v)
    }

    /// `type(value).__name__`.
    fn type_name(&self) -> String {
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
    fn repr(&self) -> String {
        match self {
            Obs::Nil => "None".into(),
            Obs::Bool(b) => if *b { "True" } else { "False" }.into(),
            Obs::Int(i) => i.to_string(),
            Obs::Float(f) => py_float_repr(*f),
            Obs::Str(s) => crate::media::py_repr(s),
            Obs::Bytes(b) => bytes_repr(b),
            Obs::List(items) => format!("[{}]", items.iter().map(Obs::repr).collect::<Vec<_>>().join(", ")),
            Obs::Map(pairs) => format!(
                "{{{}}}",
                pairs.iter().map(|(k, v)| format!("{}: {}", k.repr(), v.repr())).collect::<Vec<_>>().join(", ")
            ),
            other => format!("<{}>", other.type_name()),
        }
    }
}

/// Python's `repr(bytes)`.
fn bytes_repr(b: &[u8]) -> String {
    let quote = if b.contains(&b'\'') && !b.contains(&b'"') { '"' } else { '\'' };
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
fn numeric_dtype(name: &Obs) -> Result<Dtype, String> {
    let Obs::Str(s) = name else {
        return Err(format!("an array's dtype must be a dtype string, not {}", name.repr()));
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
            match (chars.next(), chars.as_str().parse::<usize>()) {
                (Some(kind), Ok(size)) if "biufcmMOSUV".contains(kind) => {
                    if !"biuf".contains(kind) {
                        // numpy's dtype.str: no byte order for bytes, voids and objects
                        let order = if "OSV".contains(kind) || (size == 1 && kind != 'U') { '|' } else if order == '>' { '>' } else { '<' };
                        return Err(format!("array dtype {order}{kind}{size} is not bool, integer or float"));
                    }
                    let valid = match kind {
                        'b' => size == 1,
                        'f' => [2, 4, 8].contains(&size),
                        _ => [1, 2, 4, 8].contains(&size),
                    };
                    valid.then_some(Dtype { kind, size, little: order != '>' })
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
        31 => if frac == 0.0 { sign * f32::INFINITY } else { f32::NAN },
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
    let r = if (q - q.floor() - 0.5).abs() < f32::EPSILON && r % 2.0 != 0.0 { r - 1.0 } else { r };
    x.signum() * r * ulp
}

/// Element `i` of an array's data as f64 (and its integer, for an integer dtype).
fn element(dtype: Dtype, data: &[u8], i: usize) -> (f64, Option<i128>) {
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
    if Obs::get_bytes_key(&pairs, b"__ndarray__").is_some() {
        let (data, dtype, shape) = (
            Obs::get_bytes_key(&pairs, b"data"),
            Obs::get_bytes_key(&pairs, b"dtype"),
            Obs::get_bytes_key(&pairs, b"shape"),
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
                _ => format!("({})", dims.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
            };
            return Err(format!("an array of shape {shape} and dtype {} is not {} bytes", dtype.str(), data.len()));
        }
        return Ok(Obs::Array { dtype, shape: dims, data: data.clone() });
    }
    if Obs::get_bytes_key(&pairs, b"__npgeneric__").is_some() {
        let (Some(data), Some(dtype)) = (Obs::get_bytes_key(&pairs, b"data"), Obs::get_bytes_key(&pairs, b"dtype")) else {
            return Err("a scalar needs data and dtype".into());
        };
        let (value, int) = match data {
            Obs::Bool(b) => (*b as u8 as f64, Some(*b as i128)),
            Obs::Int(i) => (*i as f64, Some(*i)),
            Obs::Float(f) => (*f, None),
            other => return Err(format!("a scalar's data must be a number, not {}", other.type_name())),
        };
        let dtype = numeric_dtype(dtype)?;
        if let (Some(i), 'i' | 'u') = (int, dtype.kind) {
            let bits = 8 * dtype.size as u32;
            let (lo, hi) = if dtype.kind == 'u' { (0, (1i128 << bits) - 1) } else { (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1) };
            if !(lo..=hi).contains(&i) {
                return Err(format!("Python integer {i} out of bounds for {}", dtype.name()));
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
        Mp::Integer(i) => Obs::Int(i.as_i64().map(i128::from).or_else(|| i.as_u64().map(i128::from)).unwrap_or(0)),
        Mp::F32(f) => Obs::Float(f as f64),
        Mp::F64(f) => Obs::Float(f),
        Mp::String(s) => match s.into_str() {
            Some(s) => Obs::Str(s),
            None => return Err("'utf-8' codec can't decode the string".into()),
        },
        Mp::Binary(b) => Obs::Bytes(b),
        Mp::Array(items) => Obs::List(items.into_iter().map(from_msgpack).collect::<Result<_, _>>()?),
        Mp::Map(pairs) => unpack_array(
            pairs.into_iter().map(|(k, v)| Ok((from_msgpack(k)?, from_msgpack(v)?))).collect::<Result<_, String>>()?,
        )?,
        Mp::Ext(..) => Obs::Other,
    })
}

/// `unpack`: an observation, or why it isn't one.
fn unpack(data: &[u8]) -> Result<Obs, String> {
    let fail = |e: String| format!("the observation is not openpi's msgpack: {e}");
    let mut cursor = data;
    let value = rmpv::decode::read_value(&mut cursor).map_err(|e| fail(e.to_string()))?;
    if !cursor.is_empty() {
        return Err(fail("unpack(b) received extra data.".into()));
    }
    from_msgpack(value).map_err(fail)
}

/// `pack` of the reply: `actions` as `msgpack_numpy` writes an array, and the timings.
fn pack_reply(actions: &[f32], rows: usize, cols: usize, infer_ms: f64, prev_total_ms: Option<f64>) -> Vec<u8> {
    let data: Vec<u8> = actions.iter().flat_map(|f| f.to_le_bytes()).collect();
    let array = Mp::Map(vec![
        (Mp::Binary(b"__ndarray__".to_vec()), Mp::Boolean(true)),
        (Mp::Binary(b"data".to_vec()), Mp::Binary(data)),
        (Mp::Binary(b"dtype".to_vec()), Mp::String("<f4".into())),
        (Mp::Binary(b"shape".to_vec()), Mp::Array(vec![Mp::from(rows as u64), Mp::from(cols as u64)])),
    ]);
    let mut timing = vec![(Mp::String("infer_ms".into()), Mp::F64(infer_ms))];
    if let Some(prev) = prev_total_ms {
        timing.push((Mp::String("prev_total_ms".into()), Mp::F64(prev)));
    }
    let reply = Mp::Map(vec![
        (Mp::String("actions".into()), array),
        (Mp::String("server_timing".into()), Mp::Map(timing)),
    ]);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &reply).expect("pack into a Vec");
    out
}

// ── observation -> request ───────────────────────────────────────────────────

/// numpy's `float.astype(uint8)` on x86-64: through an int32, keeping its low byte.
fn to_u8(x: f64) -> u8 {
    if x.is_nan() || x.abs() >= 2147483648.0 {
        0
    } else {
        (x.trunc() as i32) as u8
    }
}

/// `_image`: an observation image as RGB uint8 `[3, H, W]`, floats scaled by 255.
fn image(value: &Obs, key: &str) -> Result<RawArray, String> {
    let Obs::Array { dtype, shape, data } = value else {
        return Err(format!("{key} must be an image array of 3 dimensions"));
    };
    if shape.len() != 3 {
        return Err(format!("{key} must be an image array of 3 dimensions"));
    }
    let n: usize = shape.iter().product();
    let bytes: Vec<u8> = match (dtype.kind, dtype.size) {
        // `255 * value` in the array's own precision, then the cast
        ('f', 2) => (0..n).map(|i| to_u8(round_to_f16(255.0 * element(*dtype, data, i).0 as f32) as f64)).collect(),
        ('f', 4) => (0..n).map(|i| to_u8((255.0f32 * element(*dtype, data, i).0 as f32) as f64)).collect(),
        ('f', _) => (0..n).map(|i| to_u8(255.0 * element(*dtype, data, i).0)).collect(),
        ('u', 1) => data.clone(),
        _ => return Err(format!("{key} must be uint8 or float, not {}", dtype.name())),
    };
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    let (chw, out) = if h == 3 {
        (shape.clone(), bytes)
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
        let dims = shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", ");
        return Err(format!("{key} must be an RGB image, [H, W, 3] or [3, H, W]; got [{dims}]"));
    }
    Ok(RawArray { dtype: "|u1".to_string(), shape: chw, data: out })
}

/// `_numbers`: a value as a flat run of numbers.
fn numbers(value: &Obs, key: &str) -> Result<Vec<f64>, String> {
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
    if flat(value, &mut out) { Ok(out) } else { Err(numeric()) }
}

/// `OpenpiPolicy.request`: the submission an observation makes.
fn request(policy: &Policy, obs: &Obs) -> Result<SubmitArgs, String> {
    if !matches!(obs, Obs::Map(_)) {
        return Err("an observation is a map of names to values".into());
    }
    let missing: Vec<&str> = policy
        .image_keys
        .iter()
        .chain(&policy.state_keys)
        .filter(|k| obs.get(k).is_none())
        .map(String::as_str)
        .collect();
    if !missing.is_empty() {
        return Err(format!("the observation has no {}", missing.join(", ")));
    }
    let prompt = match obs.get("prompt") {
        Some(Obs::Str(s)) => s.clone(),
        Some(Obs::Bytes(b)) => String::from_utf8(b.clone()).map_err(|e| format!("'utf-8' codec can't decode the prompt: {e}"))?,
        _ => return Err("the observation needs a prompt string".into()),
    };
    let images = policy.image_keys.iter().map(|k| image(obs.get(k).unwrap(), k)).collect::<Result<Vec<_>, _>>()?;
    let mut state = Vec::new();
    for k in &policy.state_keys {
        state.extend(numbers(obs.get(k).unwrap(), k)?);
    }
    let mut mk = serde_json::Map::new();
    mk.insert(
        policy.state_kwarg.clone(),
        state.iter().map(|f| serde_json::Number::from_f64(*f).map_or(serde_json::Value::Null, serde_json::Value::Number)).collect(),
    );
    match obs.get("seed") {
        None | Some(Obs::Nil) => {}
        Some(Obs::Int(i)) => {
            mk.insert("seed".into(), i64::try_from(*i).map_or(serde_json::Value::from(*i as f64), serde_json::Value::from));
        }
        Some(Obs::Scalar { dtype, int: Some(i), .. }) if dtype.kind != 'b' => {
            mk.insert("seed".into(), i64::try_from(*i).map_or(serde_json::Value::from(*i as f64), serde_json::Value::from));
        }
        _ => return Err("seed must be an integer".into()),
    }
    Ok(SubmitArgs {
        text: Some(prompt),
        input_modalities: vec!["image".into(), "text".into()],
        output_modalities: vec!["action".into()],
        model_kwargs: mk,
        raw_inputs: [("image".to_string(), images)].into_iter().collect(),
        ..SubmitArgs::default()
    })
}

/// A failure's text for the client: a refusal says why; anything else only
/// its error id, with the details in the server log (`client_error_text`).
enum Failure {
    Refused(String),
    Internal(String),
}

impl Failure {
    fn text(self) -> String {
        match self {
            Failure::Refused(text) => text,
            Failure::Internal(detail) => {
                let id = &uuid::Uuid::new_v4().simple().to_string()[..12];
                eprintln!("openpi request failed [error id {id}]: {detail}");
                format!("internal error (error id {id})")
            }
        }
    }
}

/// `_infer`: the observation's action chunk, cut to the deployment's dims.
async fn infer(st: &AppState, policy: &Policy, obs: &Obs) -> Result<Vec<f32>, Failure> {
    let args = request(policy, obs).map_err(Failure::Refused)?;
    let mut stream = try_result_stream(st, &args, &rid("openpi"), true).map_err(|(_, m)| Failure::Refused(m))?;
    let mut data = None;
    while let Some(item) = stream.next().await {
        match item {
            Out::Error { message, .. } => return Err(Failure::Refused(message)),
            Out::Chunk(c) if c.modality == "action" => data = Some(c.data),
            Out::Chunk(_) => {}
        }
    }
    let data = data.ok_or_else(|| Failure::Internal("RuntimeError: the request finished without actions".into()))?;
    let values: Vec<f32> = data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let (h, d) = (policy.action_horizon, policy.action_dim);
    if data.len() % 4 != 0 || values.len() != h * d {
        return Err(Failure::Internal(format!(
            "RuntimeError: the model returned {} action values, not {h}x{d}",
            data.len() / 4
        )));
    }
    Ok(values.chunks(d).flat_map(|row| row[..policy.action_dims].to_vec()).collect())
}

/// `GET /` as a WebSocket: refused at the handshake unless the config has an `openpi:` block.
pub async fn openpi_ws(State(st): State<AppState>, ws: WebSocketUpgrade) -> Response {
    let Some(policy) = policy() else {
        return StatusCode::FORBIDDEN.into_response();
    };
    let limit = super::max_message_bytes();
    ws.max_message_size(limit).max_frame_size(limit).on_upgrade(move |socket| serve(socket, st, policy))
}

/// One robot's connection, as openpi's `WebsocketPolicyServer` serves one.
async fn serve(mut socket: WebSocket, st: AppState, policy: &'static Policy) {
    use base64::Engine;
    let metadata = base64::engine::general_purpose::STANDARD.decode(&policy.metadata).unwrap_or_default();
    if socket.send(Message::Binary(metadata)).await.is_err() {
        return;
    }
    let mut prev_total_ms = None;
    loop {
        // as openpi's server times it: prev_total_ms spans the wait for the observation too
        let start = Instant::now();
        let message = match socket.recv().await {
            Some(Ok(m)) => m,
            _ => return,
        };
        let observation = match message {
            Message::Binary(b) => Ok(b),
            Message::Close(_) => return,
            Message::Ping(_) | Message::Pong(_) => continue,
            Message::Text(_) => Err(Failure::Refused("an observation is a binary msgpack frame".into())),
        };
        let infer_start = Instant::now();
        let result = match observation {
            Ok(bytes) => match crate::blocking(move || unpack(&bytes)).await {
                Ok(obs) => infer(&st, policy, &obs).await,
                Err(e) => Err(Failure::Refused(e)),
            },
            Err(e) => Err(e),
        };
        let actions = match result {
            Ok(actions) => actions,
            Err(failure) => {
                // openpi's contract: the error as text, then close
                let _ = socket.send(Message::Text(failure.text())).await;
                let _ = socket
                    .send(Message::Close(Some(CloseFrame {
                        code: 1011,
                        reason: "request failed; the error is in the previous frame".into(),
                    })))
                    .await;
                return;
            }
        };
        let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
        let reply = pack_reply(&actions, policy.action_horizon, policy.action_dims, infer_ms, prev_total_ms);
        if socket.send(Message::Binary(reply)).await.is_err() {
            return;
        }
        prev_total_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
    }
}

/// `GET /healthz`: `/health` as openpi's policy server answers its probe, on a
/// server that serves the openpi socket; elsewhere there is no such route.
pub async fn healthz(state: State<AppState>) -> Response {
    if policy().is_none() {
        return crate::detail_error(404, "Not Found");
    }
    let health = crate::health(state).await;
    if !health.status().is_success() {
        return health;
    }
    "OK\n".into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ndarray(dtype: &str, shape: &[u64], data: Vec<u8>) -> Mp {
        Mp::Map(vec![
            (Mp::Binary(b"__ndarray__".to_vec()), Mp::Boolean(true)),
            (Mp::Binary(b"data".to_vec()), Mp::Binary(data)),
            (Mp::Binary(b"dtype".to_vec()), Mp::String(dtype.into())),
            (Mp::Binary(b"shape".to_vec()), Mp::Array(shape.iter().map(|n| Mp::from(*n)).collect())),
        ])
    }

    fn obs(pairs: Vec<(&str, Mp)>) -> Obs {
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &Mp::Map(pairs.into_iter().map(|(k, v)| (Mp::String(k.into()), v)).collect())).unwrap();
        unpack(&buf).unwrap()
    }

    #[test]
    fn an_hwc_float_image_becomes_chw_uint8_as_numpy_casts_it() {
        let pixels: Vec<f32> = vec![0.0, 0.5, 1.0, 0.2, 0.4, 0.6];
        let data = pixels.iter().flat_map(|f| f.to_le_bytes()).collect();
        let o = obs(vec![("cam", ndarray("<f4", &[1, 2, 3], data))]);
        let img = image(o.get("cam").unwrap(), "cam").unwrap();
        assert_eq!(img.shape, vec![3, 1, 2]);
        // 255*0.5 = 127.5 -> 127; 255*0.2 = 51.000004 -> 51
        assert_eq!(img.data, vec![0, 51, 127, 102, 255, 153]);
    }

    #[test]
    fn a_bad_array_is_refused_with_pythons_text() {
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &ndarray("<f4", &[2], vec![0; 4])).unwrap();
        assert_eq!(unpack(&buf).unwrap_err(), "the observation is not openpi's msgpack: an array of shape (2,) and dtype <f4 is not 4 bytes");
        buf.clear();
        rmpv::encode::write_value(&mut buf, &ndarray("<U3", &[1], vec![0; 12])).unwrap();
        assert_eq!(unpack(&buf).unwrap_err(), "the observation is not openpi's msgpack: array dtype <U3 is not bool, integer or float");
    }

    #[test]
    fn the_reply_packs_actions_as_msgpack_numpy() {
        let out = pack_reply(&[1.0, 2.0], 1, 2, 3.5, None);
        let Mp::Map(pairs) = rmpv::decode::read_value(&mut &out[..]).unwrap() else { panic!() };
        assert_eq!(pairs[0].0, Mp::String("actions".into()));
        let Mp::Map(array) = &pairs[0].1 else { panic!() };
        assert_eq!(array[0].0, Mp::Binary(b"__ndarray__".to_vec()));
    }
}
