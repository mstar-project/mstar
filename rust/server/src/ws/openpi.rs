//! openpi's WebSocket policy protocol at the server's root, a port of
//! `api_server/openpi.py`: one metadata frame on connect, then one
//! observation in and one action chunk out per call, arrays in
//! `openpi_client.msgpack_numpy`'s encoding. A call that fails is answered
//! with its error as a text frame and the socket closes, as openpi's server
//! does. The deployment's `openpi:` block, checked against its model, comes
//! from the launcher as `MSTAR_OPENPI` ([`Policy`]).

use std::sync::OnceLock;
use std::time::Instant;

use axum::extract::ws::{CloseFrame, Message, WebSocket};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use rmpv::Value as Mp;
use serde::Deserialize;

use crate::adapters::SubmitArgs;
use crate::msgpack_numpy::{encode, image, numbers, pack_ndarray, unpack_openpi as unpack, Obs};
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

/// `pack` of the reply: `actions` as `msgpack_numpy` writes an array, and the timings.
fn pack_reply(actions: &[f32], rows: usize, cols: usize, infer_ms: f64, prev_total_ms: Option<f64>) -> Vec<u8> {
    let data: Vec<u8> = actions.iter().flat_map(|f| f.to_le_bytes()).collect();
    let array = pack_ndarray("<f4", &[rows, cols], data);
    let mut timing = vec![(Mp::String("infer_ms".into()), Mp::F64(infer_ms))];
    if let Some(prev) = prev_total_ms {
        timing.push((Mp::String("prev_total_ms".into()), Mp::F64(prev)));
    }
    let reply = Mp::Map(vec![
        (Mp::String("actions".into()), array),
        (Mp::String("server_timing".into()), Mp::Map(timing)),
    ]);
    encode(&reply)
}

// ── observation -> request ───────────────────────────────────────────────────

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
    let values: Vec<f32> = data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
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
pub async fn openpi_ws(State(st): State<AppState>, ws: super::Upgrade) -> Response {
    let ws = match super::upgrade(ws) {
        Ok(ws) => ws,
        Err(not_found) => return not_found,
    };
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
            Some(Err(e)) => {
                if let Some(close) = super::read_failure_close(&e) {
                    let _ = socket.send(close).await;
                }
                return;
            }
            None => return,
        };
        let observation = match message {
            Message::Binary(b) => Ok(b),
            Message::Close(_) => {
                super::answer_close(&mut socket).await;
                return;
            }
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
