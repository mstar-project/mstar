//! `/generate` over one persistent WebSocket, for control loops: a port of
//! `entrypoint.generate_ws`.
//!
//! Each message is a request with `/generate`'s fields, as a JSON text frame
//! (files' `data` base64) or a msgpack binary frame (raw bytes). Every chunk
//! comes back in the message's encoding as `{request_id, modality, data,
//! metadata}`, then `{request_id, finish: true}`; a refusal or a failure is
//! `{request_id, error, status}` and no finish. Messages may be pipelined:
//! they are submitted in the order sent, and their replies may interleave.
//! Closing the socket aborts what it submitted.
//!
//! A request to a model with client streams ends its input at open, as on
//! `/generate`, unless the opening carries `end_input: false` (or a list of
//! the streams it ends): then it stays open as a live call until the client
//! ends its input. `append` feeds it, `end_input` ends it. The
//! appends reach the API server over the bridge's `live` operations
//! (`taking_input`, `stream_input`, `reject_input`); one the model finds late
//! comes back on the socket's notice route.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use axum::extract::ws::{Message, WebSocket};
use axum::extract::State;
use axum::response::Response;
use futures::future::{BoxFuture, FutureExt, Shared};
use futures::{SinkExt, StreamExt};
use rmpv::Value as Mp;
use serde_json::{Map, Value};
use tokio::sync::{mpsc, oneshot, watch};

use crate::adapters::{Part, SubmitArgs};
use crate::bridge::StreamItem;
use crate::serving::{AppState, Out, OutStream};

/// The model's client streams (`APIServer.client_streams`), from the launcher's
/// `MSTAR_CLIENT_STREAMS`; a model with any keeps a request's input open.
fn takes_stream_input() -> bool {
    static STREAMS: OnceLock<bool> = OnceLock::new();
    *STREAMS.get_or_init(|| {
        std::env::var("MSTAR_CLIENT_STREAMS")
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
            .is_some_and(|streams| !streams.is_empty())
    })
}

/// A message value, as `json.loads` or `msgpack.unpackb(raw=False)` gives it.
#[derive(Debug, Clone, PartialEq)]
enum Val {
    Nil,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Val>),
    Map(Vec<(String, Val)>),
}

impl Val {
    fn from_json(v: Value) -> Val {
        match v {
            Value::Null => Val::Nil,
            Value::Bool(b) => Val::Bool(b),
            Value::Number(n) => match (n.as_i64(), n.as_u64()) {
                (Some(i), _) => Val::Int(i.into()),
                (_, Some(u)) => Val::Int(u.into()),
                _ => Val::Float(n.as_f64().unwrap_or(f64::NAN)),
            },
            Value::String(s) => Val::Str(s),
            Value::Array(items) => Val::List(items.into_iter().map(Val::from_json).collect()),
            Value::Object(m) => Val::Map(m.into_iter().map(|(k, v)| (k, Val::from_json(v))).collect()),
        }
    }

    fn from_msgpack(v: Mp) -> Result<Val, String> {
        Ok(match v {
            Mp::Nil => Val::Nil,
            Mp::Boolean(b) => Val::Bool(b),
            Mp::Integer(i) => Val::Int(i.as_i64().map(i128::from).or_else(|| i.as_u64().map(i128::from)).unwrap_or(0)),
            Mp::F32(f) => Val::Float(f as f64),
            Mp::F64(f) => Val::Float(f),
            Mp::String(s) => Val::Str(s.into_str().ok_or("'utf-8' codec can't decode a string")?),
            Mp::Binary(b) => Val::Bytes(b),
            Mp::Array(items) => Val::List(items.into_iter().map(Val::from_msgpack).collect::<Result<_, _>>()?),
            Mp::Map(pairs) => Val::Map(
                pairs
                    .into_iter()
                    .map(|(k, v)| {
                        let key = match k {
                            Mp::String(s) => s.into_str().unwrap_or_default(),
                            other => other.to_string(),
                        };
                        Ok((key, Val::from_msgpack(v)?))
                    })
                    .collect::<Result<_, String>>()?,
            ),
            Mp::Ext(..) => Val::Nil,
        })
    }

    /// As the JSON a submission carries; bytes, which JSON can't hold, as null.
    fn to_json(&self) -> Value {
        match self {
            Val::Nil | Val::Bytes(_) => Value::Null,
            Val::Bool(b) => Value::Bool(*b),
            Val::Int(i) => i64::try_from(*i).map(Value::from).unwrap_or_else(|_| Value::from(*i as f64)),
            Val::Float(f) => serde_json::Number::from_f64(*f).map_or(Value::Null, Value::Number),
            Val::Str(s) => Value::String(s.clone()),
            Val::List(items) => Value::Array(items.iter().map(Val::to_json).collect()),
            Val::Map(pairs) => Value::Object(pairs.iter().map(|(k, v)| (k.clone(), v.to_json())).collect()),
        }
    }

    fn get(&self, key: &str) -> Option<&Val> {
        let Val::Map(pairs) = self else { return None };
        pairs.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    fn truthy(&self) -> bool {
        match self {
            Val::Nil => false,
            Val::Bool(b) => *b,
            Val::Int(i) => *i != 0,
            Val::Float(f) => *f != 0.0,
            Val::Str(s) => !s.is_empty(),
            Val::Bytes(b) => !b.is_empty(),
            Val::List(l) => !l.is_empty(),
            Val::Map(m) => !m.is_empty(),
        }
    }
}

/// A field of a reply, in the order the Python route writes it.
enum Field {
    Str(String),
    OptStr(Option<String>),
    Int(i64),
    True,
    Bytes(Vec<u8>),
    Json(Value),
}

fn json_to_mp(v: &Value) -> Mp {
    match v {
        Value::Null => Mp::Nil,
        Value::Bool(b) => Mp::Boolean(*b),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => Mp::from(i),
            (_, Some(u)) => Mp::from(u),
            _ => Mp::F64(n.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(s) => Mp::String(s.clone().into()),
        Value::Array(items) => Mp::Array(items.iter().map(json_to_mp).collect()),
        Value::Object(m) => Mp::Map(m.iter().map(|(k, v)| (Mp::String(k.clone().into()), json_to_mp(v))).collect()),
    }
}

/// A reply in the encoding of the message it answers: msgpack for a binary
/// frame, JSON text (`data` base64) for a text one.
fn encode(fields: Vec<(&str, Field)>, binary: bool) -> Message {
    use base64::Engine;
    if binary {
        let map = Mp::Map(
            fields
                .into_iter()
                .map(|(k, v)| {
                    let v = match v {
                        Field::Str(s) => Mp::String(s.into()),
                        Field::OptStr(s) => s.map_or(Mp::Nil, |s| Mp::String(s.into())),
                        Field::Int(i) => Mp::from(i),
                        Field::True => Mp::Boolean(true),
                        Field::Bytes(b) => Mp::Binary(b),
                        Field::Json(v) => json_to_mp(&v),
                    };
                    (Mp::String(k.into()), v)
                })
                .collect(),
        );
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, &map).expect("pack into a Vec");
        return Message::Binary(out);
    }
    let mut obj = Map::new();
    for (k, v) in fields {
        let v = match v {
            Field::Str(s) => Value::String(s),
            Field::OptStr(s) => s.map_or(Value::Null, Value::String),
            Field::Int(i) => Value::from(i),
            Field::True => Value::Bool(true),
            Field::Bytes(b) => Value::String(base64::engine::general_purpose::STANDARD.encode(b)),
            Field::Json(v) => v,
        };
        obj.insert(k.to_string(), v);
    }
    Message::Text(Value::Object(obj).to_string())
}

/// `{request_id, error[, status]}`.
fn error_reply(request_id: Option<&str>, error: &str, status: Option<u16>, binary: bool) -> Message {
    let mut fields = vec![("request_id", Field::OptStr(request_id.map(str::to_string))), ("error", Field::Str(error.to_string()))];
    if let Some(status) = status {
        fields.push(("status", Field::Int(status.into())));
    }
    encode(fields, binary)
}

/// `{request_id, append: true, error}`: an append that reached no live call.
fn late_reply(request_id: &str, error: &str, binary: bool) -> Message {
    encode(
        vec![("request_id", Field::Str(request_id.to_string())), ("append", Field::True), ("error", Field::Str(error.to_string()))],
        binary,
    )
}

/// `_ws_modalities`: a comma-separated string or a list of strings.
fn ws_modalities(value: Option<&Val>, key: &str) -> Result<Option<Vec<String>>, String> {
    match value {
        None => Ok(None),
        Some(Val::Str(s)) => Ok(Some(s.split(',').map(|m| crate::adapters::py_strip(m).to_string()).filter(|m| !m.is_empty()).collect())),
        Some(Val::List(items)) if items.iter().all(|m| matches!(m, Val::Str(_))) => {
            Ok(Some(items.iter().map(|m| if let Val::Str(s) = m { s.clone() } else { unreachable!() }).collect()))
        }
        Some(_) => Err(format!("{key} must be a comma-separated string or a list of strings")),
    }
}

/// `_ws_end_input`: true for all of the call's client streams, or the ones
/// named; `absent` when the message has none.
fn ws_end_input(value: Option<&Val>, absent: bool) -> Result<Value, String> {
    match value {
        None => Ok(Value::Bool(absent)),
        Some(Val::Bool(b)) => Ok(Value::Bool(*b)),
        Some(Val::List(items)) if items.iter().all(|v| matches!(v, Val::Str(_))) => {
            Ok(Value::Array(items.iter().map(Val::to_json).collect()))
        }
        Some(_) => Err("end_input must be true, false or a list of client stream names".into()),
    }
}

/// `_ws_is_append`: the message carries `append`, or ends input and carries none.
fn is_append(message: &Val) -> bool {
    let ends = match message.get("end_input") {
        Some(Val::Bool(true)) => true,
        Some(Val::List(items)) => !items.is_empty(),
        _ => false,
    };
    message.get("append").is_some() || (ends && message.get("text").is_none() && message.get("files").is_none())
}

/// `_ws_model_kwargs`: an object, or one as JSON text.
fn ws_model_kwargs(value: Option<&Val>) -> Result<Option<Map<String, Value>>, String> {
    let value = match value {
        Some(Val::Str(s)) => Val::from_json(serde_json::from_str::<Value>(s).map_err(|e| e.to_string())?),
        Some(v) => v.clone(),
        None => return Ok(None),
    };
    match value {
        Val::Nil => Ok(None),
        Val::Map(_) => Ok(value.to_json().as_object().cloned()),
        _ => Err("model_kwargs must be a JSON object".into()),
    }
}

/// Saved upload paths, by modality.
type PathsByModality = BTreeMap<String, Vec<String>>;

/// A message's saved uploads until a submission or an append hands them on.
/// Dropped before then (a refusal, or the socket closing while the message
/// waits or decodes), they are removed, as Python's `finally` clauses and
/// `_decode_ws_files_off_loop` remove them. While an append's round trip may
/// have handed them over (`in_flight`), they go after the delay
/// `schedule_upload_cleanup` gives the preprocess worker to read them.
struct Uploads {
    dir: PathBuf,
    paths: PathsByModality,
    in_flight: bool,
}

impl Uploads {
    /// The API server has them: its preprocess worker deletes them once loaded.
    fn hand_over(&mut self) {
        self.paths.clear();
    }
}

impl Drop for Uploads {
    fn drop(&mut self) {
        if self.paths.is_empty() {
            return;
        }
        if let (true, Ok(runtime)) = (self.in_flight, tokio::runtime::Handle::try_current()) {
            let (dir, paths) = (self.dir.clone(), std::mem::take(&mut self.paths));
            runtime.spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                crate::remove_uploads(&dir, &paths);
            });
            return;
        }
        crate::remove_uploads(&self.dir, &self.paths);
    }
}

/// `decode_files` off the async workers. Its uploads come back owned, so a
/// decode whose message is dropped meanwhile removes what it saved.
async fn decode_owned(files: Option<Val>, dir: PathBuf) -> Result<(Uploads, Vec<Part>), String> {
    crate::blocking(move || {
        decode_files(files.as_ref(), &dir).map(|(paths, parts)| (Uploads { dir, paths, in_flight: false }, parts))
    })
    .await
}

/// `_decode_ws_files`: a message's files saved under the upload directory,
/// grouped by the modality of each name, and their parts. A message refused
/// halfway leaves no files behind.
fn decode_files(files: Option<&Val>, upload_dir: &Path) -> Result<(PathsByModality, Vec<Part>), String> {
    let entries = match files {
        None | Some(Val::Nil) => return Ok((BTreeMap::new(), Vec::new())),
        Some(Val::List(items)) => items,
        Some(_) => return Err("files must be a list of {name, data} objects".into()),
    };
    let mut paths = PathsByModality::new();
    let mut parts = Vec::new();
    let mut saved: Vec<PathBuf> = Vec::new();
    let result = (|| {
        for entry in entries {
            let Some(Val::Str(name)) = entry.get("name") else {
                return Err("each file needs a string name".to_string());
            };
            let modality = crate::media::modality_from_filename(name);
            if modality == "unknown" {
                return Err(format!("Cannot determine modality for file: {name}"));
            }
            let data = match entry.get("data") {
                Some(Val::Bytes(b)) => b.clone(),
                Some(Val::Str(s)) => crate::media::decode_b64_lenient(s).map_err(|e| e.to_string())?,
                _ => return Err(format!("file {} carries no data", crate::media::py_repr(name))),
            };
            let base = Path::new(name).file_name().map(|b| b.to_string_lossy().into_owned()).filter(|b| !b.is_empty());
            std::fs::create_dir_all(upload_dir).map_err(|e| e.to_string())?;
            let path = upload_dir.join(format!("{}_{}", uuid::Uuid::new_v4(), base.as_deref().unwrap_or("upload")));
            std::fs::write(&path, &data).map_err(|e| {
                let _ = std::fs::remove_file(&path);
                e.to_string()
            })?;
            saved.push(path.clone());
            let list = paths.entry(modality.to_string()).or_default();
            parts.push(Part { modality: modality.to_string(), text: None, index: list.len(), role: None });
            list.push(path.to_string_lossy().into_owned());
        }
        Ok(())
    })();
    if let Err(e) = result {
        for path in saved {
            let _ = std::fs::remove_file(path);
        }
        return Err(e);
    }
    Ok((paths, parts))
}

/// `_ws_file_streams`: per modality, the client stream each file names; `None`
/// when none names one.
fn file_streams(files: Option<&Val>, parts: &[Part]) -> Result<Option<Value>, String> {
    let entries: &[Val] = match files {
        Some(Val::List(items)) => items,
        _ => &[],
    };
    let mut streams: Map<String, Value> = Map::new();
    let mut named = false;
    for (entry, part) in entries.iter().zip(parts) {
        let stream = match entry.get("stream") {
            None | Some(Val::Nil) => Value::Null,
            Some(Val::Str(s)) => {
                named = true;
                Value::String(s.clone())
            }
            Some(_) => return Err("a file's stream must be a client stream's name".into()),
        };
        match streams.entry(part.modality.clone()).or_insert_with(|| Value::Array(Vec::new())) {
            Value::Array(list) => list.push(stream),
            _ => unreachable!(),
        }
    }
    Ok(named.then_some(Value::Object(streams)))
}

/// `_ws_input_layout`: an explicit list is the layout (a text prompt keeps its
/// slot); else the order the files and text arrived in.
fn input_layout(text: Option<&str>, input_modalities: Option<Vec<String>>, mut parts: Vec<Part>) -> (Vec<String>, Vec<Part>) {
    let text = text.filter(|t| !t.is_empty());
    if let Some(t) = text {
        parts.push(Part { modality: "text".into(), text: Some(t.to_string()), index: 0, role: None });
    }
    if let Some(mut mods) = input_modalities {
        if text.is_some() && !mods.iter().any(|m| m == "text") {
            mods.push("text".into());
        }
        if text.is_none() {
            mods.retain(|m| m != "text");
        }
        return (mods, Vec::new());
    }
    (parts.iter().map(|p| p.modality.clone()).collect(), parts)
}

/// What a refusal raised: the client's (a 400 with its text) or the server's
/// (a 500 naming only an error id).
fn refusal(text: String) -> (u16, String) {
    (400, crate::scrub_paths(&text))
}

/// A live call's submission: `None` until it is submitted or refused, then
/// its `sub` (or `None`, when it never was).
type CallSlot = watch::Receiver<Option<Option<u64>>>;

/// The message before this one's submission (or refusal), for this one to wait on.
type Turn = Shared<BoxFuture<'static, ()>>;

struct Socket {
    st: AppState,
    out: mpsc::UnboundedSender<Message>,
    live: Mutex<HashMap<String, CallSlot>>,
    appends: Mutex<HashMap<String, Turn>>,
    notices: Option<u64>,
}

impl Socket {
    fn send(&self, message: Message) {
        let _ = self.out.send(message);
    }
}

fn turn_pair() -> (oneshot::Sender<()>, Turn) {
    let (tx, rx) = oneshot::channel::<()>();
    (tx, rx.map(|_| ()).boxed().shared())
}

/// Lets the next message submit once the one before this one has.
fn release_turn(turn: oneshot::Sender<()>, after: Option<Turn>) {
    tokio::spawn(async move {
        if let Some(after) = after {
            after.await;
        }
        let _ = turn.send(());
    });
}

/// `serve_one`: decode, wait the message's turn, submit, and stream the replies.
async fn serve_one(
    sock: Arc<Socket>,
    message: Val,
    binary: bool,
    call: Option<watch::Sender<Option<Option<u64>>>>,
    after: Option<Turn>,
    turn: oneshot::Sender<()>,
) {
    let request_id = match message.get("request_id") {
        Some(Val::Str(s)) => Some(s.clone()),
        _ => None,
    };
    let upload_dir = sock.st.upload_dir.clone();
    let mut turn = Some(turn);
    let prepared: Result<(SubmitArgs, Uploads), (u16, String)> = async {
        let out_mods = ws_modalities(message.get("output_modalities"), "output_modalities").map_err(refusal)?.unwrap_or_else(|| vec!["text".into()]);
        let text = match message.get("text") {
            None => None,
            Some(Val::Str(s)) => Some(s.clone()),
            Some(_) => return Err(refusal("text must be a string".into())),
        };
        // an opening is one-shot unless it says otherwise
        let end_input = ws_end_input(message.get("end_input"), true).map_err(refusal)?;
        let has_files = message.get("files").is_some_and(Val::truthy);
        if text.as_deref().is_none_or(str::is_empty) && !has_files && (end_input == Value::Bool(true) || !takes_stream_input()) {
            // a one-shot opening: the client may have meant a live one
            let hint = if takes_stream_input() { " (open a live call with end_input: false)" } else { "" };
            return Err(refusal(format!("message carries neither text nor files{hint}")));
        }
        let (uploads, parts) = decode_owned(message.get("files").cloned(), upload_dir.clone()).await.map_err(refusal)?;
        let mut args = SubmitArgs { file_paths: uploads.paths.clone(), ..SubmitArgs::default() };
        let streams = file_streams(message.get("files"), &parts);
        let modalities = ws_modalities(message.get("input_modalities"), "input_modalities");
        let kwargs = ws_model_kwargs(message.get("model_kwargs"));
        let failed = streams.as_ref().err().or(modalities.as_ref().err()).or(kwargs.as_ref().err()).cloned();
        if let Some(e) = failed {
            return Err(refusal(e)); // dropping the uploads removes them
        }
        let (in_mods, parts) = input_layout(text.as_deref(), modalities.unwrap(), parts);
        args.text = text;
        args.input_modalities = in_mods;
        args.output_modalities = out_mods;
        args.model_kwargs = kwargs.unwrap().unwrap_or_default();
        args.parts = parts;
        args.end_input = Some(end_input);
        args.file_streams = streams.unwrap();
        Ok((args, uploads))
    }
    .await;
    let (args, mut uploads) = match prepared {
        Ok(prepared) => prepared,
        Err((status, error)) => {
            if let Some(call) = &call {
                let _ = call.send(Some(None));
            }
            release_turn(turn.take().unwrap(), after);
            sock.send(error_reply(request_id.as_deref(), &error, Some(status), binary));
            finish_call(&sock, request_id.as_deref(), call.is_some());
            return;
        }
    };
    // messages decode side by side but are submitted in the order they were sent
    if let Some(after) = after {
        after.await;
    }
    let rid = request_id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    if let Some(bridge) = &sock.st.bridge {
        bridge.wait_for_request_id(&rid, crate::serving::DUPLICATE_ID_GRACE).await;
    }
    let submitted = if call.is_some() {
        crate::serving::live_result_stream(&sock.st, &args, &rid)
    } else {
        crate::serving::try_result_stream(&sock.st, &args, &rid, true).map(|s| (s, None))
    };
    crate::schedule_upload_cleanup(&upload_dir, &args);
    uploads.hand_over();
    let mut stream: OutStream = match submitted {
        Ok((stream, sub)) => {
            if let Some(call) = &call {
                let _ = call.send(Some(sub));
            }
            stream
        }
        Err((status, error)) => {
            if let Some(call) = &call {
                let _ = call.send(Some(None));
            }
            let _ = turn.take().unwrap().send(());
            // refused before it had an id: an unnamed message's error names none, as on Python's route
            sock.send(error_reply(request_id.as_deref(), &error, Some(status), binary));
            finish_call(&sock, request_id.as_deref(), call.is_some());
            return;
        }
    };
    let _ = turn.take().unwrap().send(());
    let mut failed = false;
    while let Some(item) = stream.next().await {
        match item {
            Out::Error { status, message, refused } => {
                failed = true;
                let id = if refused { request_id.as_deref() } else { Some(rid.as_str()) };
                sock.send(error_reply(id, &message, Some(status), binary));
            }
            Out::Chunk(c) => sock.send(encode(
                vec![
                    ("request_id", Field::Str(rid.clone())),
                    ("modality", Field::Str(c.modality)),
                    ("data", Field::Bytes(c.data)),
                    ("metadata", Field::Json(if c.metadata.is_null() { Value::Object(Map::new()) } else { c.metadata })),
                ],
                binary,
            )),
        }
    }
    if !failed {
        sock.send(encode(vec![("request_id", Field::Str(rid)), ("finish", Field::True)], binary));
    }
    finish_call(&sock, request_id.as_deref(), call.is_some());
}

/// A live call's end: appends no longer find it by its id.
fn finish_call(sock: &Socket, request_id: Option<&str>, live: bool) {
    if let (true, Some(rid)) = (live, request_id) {
        sock.live.lock().unwrap().remove(rid);
        sock.appends.lock().unwrap().remove(rid);
    }
}

/// `_ws_append_record`: an append's record with its null fields dropped.
fn append_record(value: Option<&Val>) -> Result<Vec<(String, Val)>, String> {
    let record = match value {
        None => return Ok(Vec::new()),
        Some(Val::List(items)) => return Ok(vec![("files".into(), Val::List(items.clone()))]),
        Some(Val::Map(pairs)) => pairs.iter().filter(|(_, v)| *v != Val::Nil).cloned().collect::<Vec<_>>(),
        Some(_) => return Err("append must be an object or a list of files".into()),
    };
    let mut unknown: Vec<&str> = record.iter().map(|(k, _)| k.as_str()).filter(|k| !["text", "files", "model_kwargs"].contains(k)).collect();
    if !unknown.is_empty() {
        unknown.sort_unstable();
        unknown.dedup();
        let list = unknown.iter().map(|k| crate::media::py_repr(k)).collect::<Vec<_>>().join(", ");
        return Err(format!("an append carries only ['files', 'model_kwargs', 'text']; got [{list}]"));
    }
    Ok(record)
}

/// `append_one`: feed an append to its live call, after the call's previous one.
async fn append_one(sock: Arc<Socket>, message: Val, binary: bool, mut call: CallSlot, before: Option<Turn>) {
    let rid = match message.get("request_id") {
        Some(Val::Str(s)) => s.clone(),
        _ => return,
    };
    let late = format!("request {} is not taking input", crate::media::py_repr(&rid));
    if let Some(before) = before {
        before.await;
    }
    // in order behind the call's own submission, which may still be running
    let sub = loop {
        if let Some(sub) = *call.borrow() {
            break sub;
        }
        if call.changed().await.is_err() {
            break None;
        }
    };
    let (Some(sub), Some(bridge)) = (sub, sock.st.bridge.clone()) else {
        sock.send(late_reply(&rid, &late, binary));
        return;
    };
    let timeout = crate::BRIDGE_PROBE_TIMEOUT * 5;
    // checked before decoding, so a bad append to a call that has ended is late
    let taking = bridge.live(sub, "taking", Map::new(), timeout).await;
    if !taking.as_ref().and_then(|m| m.get("taking")).and_then(Value::as_bool).unwrap_or(false) {
        sock.send(late_reply(&rid, &late, binary));
        return;
    }
    let upload_dir = sock.st.upload_dir.clone();
    let mut uploads: Option<Uploads> = None;
    let prepared: Result<Map<String, Value>, String> = async {
        if message.get("append").is_some() {
            let mut misplaced: Vec<&str> = ["files", "model_kwargs", "text"].into_iter().filter(|k| message.get(k).is_some()).collect();
            misplaced.sort_unstable();
            if !misplaced.is_empty() {
                return Err(format!("an append carries {} under 'append'", misplaced.join(", ")));
            }
        }
        let record = Val::Map(append_record(message.get("append"))?);
        let end_input = ws_end_input(message.get("end_input"), false)?;
        let text = match record.get("text") {
            None => None,
            Some(Val::Str(s)) => Some(s.clone()),
            Some(_) => return Err("text must be a string".into()),
        };
        let kwargs = ws_model_kwargs(record.get("model_kwargs"))?;
        let (saved, parts) = decode_owned(record.get("files").cloned(), upload_dir.clone()).await?;
        let file_paths = &uploads.insert(saved).paths;
        let streams = file_streams(record.get("files"), &parts)?;
        let ends = end_input != Value::Bool(false) && end_input != Value::Array(Vec::new());
        if file_paths.is_empty() && text.as_deref().is_none_or(str::is_empty) && kwargs.as_ref().is_none_or(Map::is_empty) && !ends {
            return Err("an append needs text, files, model_kwargs or end_input".into());
        }
        let mut op = Map::new();
        op.insert("file_paths".into(), serde_json::to_value(file_paths).unwrap());
        op.insert("end_input".into(), end_input);
        op.insert("text".into(), text.filter(|t| !t.is_empty()).map_or(Value::Null, Value::String));
        op.insert("file_streams".into(), streams.unwrap_or(Value::Null));
        op.insert("model_kwargs".into(), kwargs.map_or(Value::Null, Value::Object));
        op.insert("notice".into(), sock.notices.map_or(Value::Null, Value::from));
        op.insert("tag".into(), serde_json::json!({"request_id": rid, "binary": binary}));
        Ok(op)
    }
    .await;
    let rejection = match prepared {
        Err(text) => Some((crate::scrub_paths(&text), 400)),
        Ok(op) => {
            if let Some(u) = &mut uploads {
                u.in_flight = true; // the socket may close before the reply says who has them
            }
            let reply = bridge.live(sub, "append", op, timeout).await;
            if let (Some(u), Some(_)) = (&mut uploads, &reply) {
                u.in_flight = false; // answered: taken, or not
            }
            match reply {
                Some(meta) if meta.get("accepted").and_then(Value::as_bool) == Some(true) => {
                    if let Some(u) = &mut uploads {
                        u.hand_over();
                    }
                    None
                }
                Some(meta) if meta.get("error").is_some() => Some((
                    meta["error"].as_str().unwrap_or("").to_string(),
                    meta.get("status").and_then(Value::as_u64).unwrap_or(400) as u16,
                )),
                _ => {
                    // the call ended meanwhile: leave it be
                    sock.send(late_reply(&rid, &late, binary));
                    None
                }
            }
        }
    };
    if let Some((text, status)) = rejection {
        let mut op = Map::new();
        op.insert("error".into(), Value::String(format!("append rejected: {text}")));
        op.insert("status".into(), Value::from(status));
        let rejected = bridge.live(sub, "reject", op, timeout).await;
        // the call's stream sends the error and no finish; if it had ended, the append was late
        if !rejected.as_ref().and_then(|m| m.get("rejected")).and_then(Value::as_bool).unwrap_or(false) {
            sock.send(late_reply(&rid, &late, binary));
        }
    }
    drop(uploads); // what the API server didn't take
}

/// `GET /generate/ws`.
pub async fn generate_ws(State(st): State<AppState>, ws: super::Upgrade) -> Response {
    let ws = match super::upgrade(ws) {
        Ok(ws) => ws,
        Err(not_found) => return not_found,
    };
    let limit = super::max_message_bytes();
    ws.max_message_size(limit).max_frame_size(limit).on_upgrade(move |socket| serve(socket, st))
}

fn parse_frame(message: Message) -> Option<(Result<Val, String>, bool)> {
    match message {
        Message::Text(t) => Some((serde_json::from_str::<Value>(&t).map(Val::from_json).map_err(|e| e.to_string()), false)),
        Message::Binary(b) => {
            let mut cursor = &b[..];
            let parsed = rmpv::decode::read_value(&mut cursor)
                .map_err(|e| e.to_string())
                .and_then(|v| if cursor.is_empty() { Ok(v) } else { Err("unpack(b) received extra data.".into()) })
                .and_then(Val::from_msgpack);
            Some((parsed, true))
        }
        _ => None,
    }
}

async fn serve(socket: WebSocket, st: AppState) {
    let (mut sink, mut source) = socket.split();
    let (out, mut outbox) = mpsc::unbounded_channel::<Message>();
    let writer = tokio::spawn(async move {
        while let Some(m) = outbox.recv().await {
            if sink.send(m).await.is_err() {
                break;
            }
        }
        // the reader stops at the client's Close; its reply goes out with this half
        super::answer_close(&mut sink).await;
    });
    let mut notice_rx = None;
    let notices = match (&st.bridge, takes_stream_input()) {
        (Some(bridge), true) => {
            let (sub, rx) = bridge.notices();
            notice_rx = Some(rx);
            Some(sub)
        }
        _ => None,
    };
    let sock = Arc::new(Socket { st: st.clone(), out: out.clone(), live: Mutex::new(HashMap::new()), appends: Mutex::new(HashMap::new()), notices });
    let mut tasks = tokio::task::JoinSet::new();
    if let Some(mut rx) = notice_rx {
        // an append the model found late: answered as late on its own socket
        let out = out.clone();
        tasks.spawn(async move {
            while let Some(StreamItem::Chunk(c)) = rx.recv().await {
                let rid = c.metadata.get("request_id").and_then(Value::as_str).unwrap_or("").to_string();
                let binary = c.metadata.get("binary").and_then(Value::as_bool).unwrap_or(false);
                let _ = out.send(late_reply(&rid, &String::from_utf8_lossy(&c.data), binary));
            }
        });
    }
    let mut previous_turn: Option<Turn> = None;
    loop {
        let frame = match source.next().await {
            Some(Ok(frame)) => frame,
            Some(Err(e)) => {
                if let Some(close) = super::read_failure_close(&e) {
                    let _ = out.send(close);
                }
                break;
            }
            None => break,
        };
        // a finished message's task goes now, not with the socket, as Python's tasks.discard
        while tasks.try_join_next().is_some() {}
        if matches!(frame, Message::Close(_)) {
            break;
        }
        let Some((parsed, binary)) = parse_frame(frame) else { continue };
        let message = match parsed {
            Ok(Val::Map(pairs)) => {
                // a null field is an absent one, to routing and validation alike
                let mut pairs: Vec<(String, Val)> = pairs.into_iter().filter(|(_, v)| *v != Val::Nil).collect();
                pairs.retain(|(k, v)| !(k == "request_id" && *v == Val::Str(String::new())));
                Val::Map(pairs)
            }
            Ok(_) => {
                sock.send(error_reply(None, "message must be an object", None, binary));
                continue;
            }
            Err(e) => {
                sock.send(error_reply(None, &format!("undecodable frame: {}", crate::scrub_paths(&e)), None, binary));
                continue;
            }
        };
        let request_id = match message.get("request_id") {
            None => None,
            Some(Val::Str(s)) => Some(s.clone()),
            Some(_) => {
                sock.send(error_reply(None, "request_id must be a string", None, binary));
                continue;
            }
        };
        if let Some(rid) = &request_id {
            if let Err(e) = crate::check_request_id(rid) {
                sock.send(error_reply(Some(rid), &e, Some(400), binary));
                continue;
            }
        }
        if is_append(&message) {
            let call = request_id.as_ref().and_then(|rid| sock.live.lock().unwrap().get(rid).cloned());
            let Some(call) = call else {
                let shown = request_id.as_deref().map_or("None".to_string(), crate::media::py_repr);
                let fields = vec![
                    ("request_id", Field::OptStr(request_id.clone())),
                    ("append", Field::True),
                    ("error", Field::Str(format!("request {shown} is not taking input on this socket"))),
                ];
                sock.send(encode(fields, binary));
                continue;
            };
            // a task per append, chained per call
            let rid = request_id.clone().unwrap();
            let before = sock.appends.lock().unwrap().get(&rid).cloned();
            let (done_tx, done) = turn_pair();
            sock.appends.lock().unwrap().insert(rid, done);
            let sock2 = sock.clone();
            tasks.spawn(async move {
                append_one(sock2, message, binary, call, before).await;
                let _ = done_tx.send(());
            });
            continue;
        }
        if let Some(rid) = &request_id {
            if sock.live.lock().unwrap().contains_key(rid) {
                // refused as any id in flight is, and the call goes on
                sock.send(error_reply(Some(rid), &crate::bridge::request_id_in_use(rid), Some(409), binary));
                continue;
            }
        }
        let mut call = None;
        if takes_stream_input() && !matches!(message.get("end_input"), None | Some(Val::Bool(true))) {
            // a live call: appends find it by id, from the moment it is read
            let Some(rid) = &request_id else {
                sock.send(error_reply(None, "a live call needs a request_id", None, binary));
                continue;
            };
            let (tx, rx) = watch::channel(None);
            sock.live.lock().unwrap().insert(rid.clone(), rx);
            call = Some(tx);
        }
        let (turn_tx, turn) = turn_pair();
        let after = previous_turn.replace(turn);
        tasks.spawn(serve_one(sock.clone(), message, binary, call, after, turn_tx));
    }
    // closing the socket aborts whatever is still in flight: the tasks' result
    // streams abort their submissions as they drop
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    if let (Some(sub), Some(bridge)) = (notices, &st.bridge) {
        bridge.close_notices(sub);
    }
    drop(sock);
    drop(out);
    let _ = writer.await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_append_is_named_by_append_or_by_ending_input_with_nothing_else() {
        let msg = |v: Value| Val::from_json(v);
        assert!(is_append(&msg(serde_json::json!({"request_id": "a", "append": {"text": "x"}}))));
        assert!(is_append(&msg(serde_json::json!({"request_id": "a", "end_input": true}))));
        assert!(!is_append(&msg(serde_json::json!({"request_id": "a", "end_input": true, "text": "x"}))));
        assert!(!is_append(&msg(serde_json::json!({"request_id": "a", "end_input": []}))));
    }

    fn scratch() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mstar_genws_test_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn saved(dir: &Path, name: &str) -> PathsByModality {
        let path = dir.join(name);
        std::fs::write(&path, b"x").unwrap();
        BTreeMap::from([("image".to_string(), vec![path.to_string_lossy().into_owned()])])
    }

    #[test]
    fn uploads_not_handed_over_are_removed_when_dropped() {
        let dir = scratch();
        drop(Uploads { dir: dir.clone(), paths: saved(&dir, "a.png"), in_flight: false });
        let mut handed = Uploads { dir: dir.clone(), paths: saved(&dir, "b.png"), in_flight: false };
        handed.hand_over();
        drop(handed);
        assert!(!dir.join("a.png").exists());
        assert!(dir.join("b.png").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_decode_whose_message_is_dropped_removes_what_it_saved() {
        let dir = scratch();
        let files = Val::List(vec![Val::Map(vec![
            ("name".into(), Val::Str("big.png".into())),
            ("data".into(), Val::Bytes(vec![7; 256 << 20])),
        ])]);
        let decode = tokio::spawn(decode_owned(Some(files), dir.clone()));
        let count = || std::fs::read_dir(&dir).unwrap().count();
        // dropped once the decode has saved something
        while count() == 0 {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        decode.abort();
        let _ = decode.await;
        // the blocking decode runs on; its output is dropped once it returns
        for _ in 0..400 {
            if count() == 0 {
                std::fs::remove_dir_all(&dir).unwrap();
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        panic!("the dropped decode's upload is still there");
    }

    #[test]
    fn the_layout_keeps_a_text_slot_as_generate_does() {
        let parts = vec![Part { modality: "image".into(), text: None, index: 0, role: None }];
        assert_eq!(input_layout(Some("hi"), None, parts.clone()).0, vec!["image", "text"]);
        assert_eq!(input_layout(Some("hi"), Some(vec!["image".into()]), parts.clone()).0, vec!["image", "text"]);
        assert_eq!(input_layout(None, Some(vec!["image".into(), "text".into()]), parts).0, vec!["image"]);
    }
}
