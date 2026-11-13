//! Positronic protocol v2 session wire. The model emits command dictionaries
//! as a msgpack `action` chunk; the Praxis adapter supplies their contents.

use std::sync::OnceLock;
use std::time::Instant;

use axum::extract::ws::{CloseFrame, Message, WebSocket};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use rmpv::Value as Mp;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::adapters::SubmitArgs;
use crate::serving::{rid, try_result_stream, AppState, Out};

#[derive(Deserialize)]
pub struct Policy {
    meta: serde_json::Map<String, Value>,
    prompt_key: String,
    observation_format: String,
    control_mode: String,
}

fn policy() -> Option<&'static Policy> {
    static POLICY: OnceLock<Option<Policy>> = OnceLock::new();
    POLICY.get_or_init(|| {
        std::env::var("MSTAR_POSITRONIC").ok().filter(|s| !s.is_empty())
            .map(|s| serde_json::from_str(&s).expect("MSTAR_POSITRONIC is the launcher's JSON"))
    }).as_ref()
}

fn pack(v: &Value) -> Vec<u8> {
    rmp_serde::to_vec_named(v).expect("encode Positronic frame")
}

fn error(reason: &str) -> Vec<u8> {
    pack(&json!({"status": "error", "error": reason}))
}

fn get<'a>(value: &'a Mp, key: &str) -> Option<&'a Mp> {
    let Mp::Map(pairs) = value else { return None; };
    pairs.iter().rev().find(|(k, _)| k.as_str() == Some(key)).map(|(_, v)| v)
}

fn request(data: &[u8], session_id: &str) -> Result<Option<Mp>, String> {
    rmp_serde::from_slice::<serde::de::IgnoredAny>(data)
        .map_err(|_| "invalid msgpack request".to_string())?;
    let mut cursor = data;
    let value = rmpv::decode::read_value(&mut cursor).map_err(|_| "invalid msgpack request".to_string())?;
    if !cursor.is_empty() { return Err("invalid msgpack request".into()); }
    if !matches!(value, Mp::Map(_)) || get(&value, "session_id").and_then(Mp::as_str) != Some(session_id) {
        return Err("The session ID does not belong to this connection".into());
    }
    if get(&value, "end_session") == Some(&Mp::Boolean(true)) {
        return Ok(None);
    }
    let Some(obs) = get(&value, "observation").filter(|v| matches!(v, Mp::Map(_))) else {
        return Err("a v2 request carries an observation dict".into());
    };
    Ok(Some(obs.clone()))
}

fn prompt(obs: &Mp, policy: &Policy) -> Result<String, String> {
    let value = if policy.observation_format == "praxis1" {
        get(obs, "task").filter(|v| !matches!(v, Mp::Nil) && v.as_str() != Some(""))
            .or_else(|| get(obs, &policy.prompt_key))
    } else { get(obs, &policy.prompt_key) };
    match value {
        Some(Mp::String(s)) => s.as_str().map(str::to_string),
        Some(Mp::Binary(b)) => Some(String::from_utf8_lossy(b).into_owned()),
        _ => None,
    }.ok_or_else(|| format!("the observation needs a {} string", policy.prompt_key))
}

async fn infer(st: &AppState, policy: &Policy, obs: &Mp, session_id: &str, reset: bool, frame: &[u8], prior_rows: usize) -> Result<(Mp, String), String> {
    let prompt = prompt(obs, policy)?;
    let mut model_kwargs = serde_json::Map::new();
    if policy.observation_format == "praxis1" {
        model_kwargs.insert("control_mode".into(), Value::String(policy.control_mode.clone()));
        model_kwargs.insert("executed_rows".into(), Value::from(if reset { 0 } else { prior_rows }));
    }
    let args = SubmitArgs {
        text: Some(prompt.clone()),
        episode: Some(session_id.to_string()),
        reset_episode: reset,
        positronic_frame: (policy.observation_format == "praxis1").then(|| frame.to_vec()),
        input_modalities: if policy.observation_format == "praxis1" {
            vec!["image".into(), "text".into()]
        } else { vec!["text".into()] },
        output_modalities: vec!["action".into()],
        model_kwargs,
        ..SubmitArgs::default()
    };
    let mut stream = try_result_stream(st, &args, &rid("positronic"), true).map_err(|(_, m)| m)?;
    let mut data = None;
    while let Some(item) = stream.next().await {
        match item {
            Out::Error { message, .. } => return Err(message),
            Out::Chunk(c) if c.modality == "action" => data = Some(c.data),
            Out::Chunk(_) => {}
        }
    }
    let data = data.ok_or("the request finished without actions")?;
    let mut cursor = data.as_slice();
    let rows = rmpv::decode::read_value(&mut cursor).map_err(|_| "invalid command result".to_string())?;
    if !cursor.is_empty() { return Err("invalid command result".into()); }
    if !matches!(&rows, Mp::Array(items) if items.iter().all(|item| matches!(item, Mp::Map(_)))) {
        return Err("the model must return a list of command dictionaries".into());
    }
    Ok((rows, prompt))
}

pub async fn positronic_ws(State(st): State<AppState>, ws: super::Upgrade) -> Response {
    let ws = match super::upgrade(ws) {
        Ok(ws) => ws,
        Err(not_found) => return not_found,
    };
    let Some(policy) = policy() else { return StatusCode::FORBIDDEN.into_response(); };
    let limit = super::max_message_bytes();
    ws.max_message_size(limit).max_frame_size(limit).on_upgrade(move |socket| serve(socket, st, policy))
}

async fn end(st: &AppState, id: &str) -> Result<(), String> {
    if let Some(bridge) = &st.bridge {
        bridge.end_episode(id).await?;
    }
    Ok(())
}

async fn serve(mut socket: WebSocket, st: AppState, policy: &'static Policy) {
    let id = uuid::Uuid::new_v4().simple().to_string();
    let mut prior_rows = 0;
    let loading = pack(&json!({"status": "loading", "message": "Loading policy..."}));
    let ready = pack(&json!({
        "status": "ready", "meta": policy.meta, "protocol_version": 2,
        "session_id": id, "max_in_flight": 1,
    }));
    if socket.send(Message::Binary(loading)).await.is_err() || socket.send(Message::Binary(ready)).await.is_err() {
        if let Err(e) = end(&st, &id).await { eprintln!("positronic episode cleanup failed: {e}"); }
        return;
    }
    loop {
        let message = match socket.recv().await {
            Some(Ok(m)) => m,
            Some(Err(e)) => {
                if let Some(close) = super::read_failure_close(&e) { let _ = socket.send(close).await; }
                break;
            }
            None => break,
        };
        let data = match message {
            Message::Binary(data) => data,
            Message::Close(_) => { super::answer_close(&mut socket).await; break; }
            Message::Ping(_) | Message::Pong(_) => continue,
            Message::Text(_) => {
                let reason = "Text frames are not served. Send each observation as one binary msgpack frame.";
                let _ = socket.send(Message::Binary(error(reason))).await;
                let _ = socket.send(Message::Close(Some(CloseFrame { code: 1008, reason: reason.into() }))).await;
                break;
            }
        };
        let started = Instant::now();
        let obs = match request(&data, &id) {
            Ok(Some(obs)) => obs,
            Ok(None) => {
                match end(&st, &id).await {
                    Ok(()) => { let _ = socket.send(Message::Binary(pack(&json!({"session_id": id, "end_session": true})))).await; }
                    Err(e) => failed(&mut socket, &e).await,
                }
                return;
            }
            Err(e) => { failed(&mut socket, &e).await; break; }
        };
        let reset = get(&obs, "reset_context") == Some(&Mp::Boolean(true))
            || get(&obs, "__reset_context__") == Some(&Mp::Boolean(true));
        if reset { prior_rows = 0; }
        let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
        let infer_started = Instant::now();
        let (rows, effective_prompt) = match infer(&st, policy, &obs, &id, reset, &data, prior_rows).await {
            Ok(rows) => rows,
            Err(e) => { failed(&mut socket, &e).await; break; }
        };
        prior_rows = rows.as_array().map_or(0, Vec::len);
        let infer_ms = infer_started.elapsed().as_secs_f64() * 1000.0;
        let timing = Mp::Map(vec![
            (Mp::from("served_ms"), Mp::F64((started.elapsed().as_secs_f64() * 10000.0).round() / 10.0)),
            (Mp::from("decode_ms"), Mp::F64((decode_ms * 10.0).round() / 10.0)),
            (Mp::from("infer_ms"), Mp::F64((infer_ms * 10.0).round() / 10.0)),
            (Mp::from("queued_ms"), Mp::F64(0.0)),
        ]);
        let reply = Mp::Map(vec![
            (Mp::from("result"), rows),
            (Mp::from("effective_prompt"), Mp::from(effective_prompt)),
            (Mp::from("prompt_original"), Mp::Nil),
            (Mp::from("timing"), timing),
        ]);
        let mut packed = Vec::new();
        rmpv::encode::write_value(&mut packed, &reply).expect("pack into Vec");
        if socket.send(Message::Binary(packed)).await.is_err() { break; }
    }
    if let Err(e) = end(&st, &id).await { eprintln!("positronic episode cleanup failed: {e}"); }
}

async fn failed(socket: &mut WebSocket, reason: &str) {
    let _ = socket.send(Message::Binary(error(reason))).await;
    let _ = socket.send(Message::Close(Some(CloseFrame { code: 1011, reason: "request failed".into() }))).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_id_is_required_for_observations_and_end() {
        let id = "0123456789abcdef0123456789abcdef";
        let call = pack(&json!({"session_id": id, "observation": {"prompt": "go"}}));
        assert_eq!(get(&request(&call, id).unwrap().unwrap(), "prompt").and_then(Mp::as_str), Some("go"));
        assert_eq!(request(&call, "other").unwrap_err(), "The session ID does not belong to this connection");
        let end = pack(&json!({"session_id": id, "end_session": true}));
        assert_eq!(request(&end, id).unwrap(), None);
    }
}
