//! Realtime transcription on a model that hears audio as it comes (one
//! `ClientAudio` client stream and no turn tokens): a port of
//! `serving_realtime.RealtimeLiveTranscription`. Each item is one engine
//! request, opened by its first audio and ended by its commit, whose text goes
//! out as deltas; nothing is re-decoded. If the item's call fails (or can't
//! open), the client gets an `error`, the rest of the item's audio is dropped,
//! and its commit reports it failed.
//!
//! The session is the re-decoding one's ([`super::Realtime`]), with this call
//! in place of its steps: the same events, formats, caps and turn detection.

use axum::extract::ws::WebSocket;
use futures::StreamExt;
use serde_json::{json, Map, Value};

use super::conversation::{truthy, Facts};
use super::vad::SAMPLE_RATE;
use super::{call_error, error_event, event, object, open_call, stream_audio, ItemDecoder, Realtime, Session};
use crate::adapters::{OpenAIAdapter, SubmitArgs};
use crate::serving::{AppState, Out, OutStream};
use crate::utf8::Utf8Stream;

/// A re-decoding session's knobs, which a live transcription has no use for.
const REDECODE_KNOBS: [&str; 3] = ["chunk_seconds", "unfixed_chunks", "unfixed_tokens"];

const ENDED_EARLY: &str = "this item's transcription ended before its commit";

/// The open item's call.
pub(super) struct Live {
    facts: &'static Facts,
    /// the call's submission while it takes the item's audio
    sub: Option<u64>,
    /// the call's output, until it ends
    output: Option<OutStream>,
    text: Utf8Stream,
    /// the `error` event of this item's failed call (`None`: not failed)
    failure: Option<Value>,
}

impl Live {
    /// A transcription session on `facts`' stream.
    pub(super) fn session(st: AppState, adapter: &'static dyn OpenAIAdapter, socket: WebSocket, mut s: Session, facts: &'static Facts) -> Realtime<Live> {
        // the stream's own bound on one item's length, not the re-decoding path's
        if let Some(max_items) = facts.max_items {
            s.max_item_seconds = max_items as f64 / facts.rate as f64;
        }
        let d = Live { facts, sub: None, output: None, text: Utf8Stream::default(), failure: None };
        Realtime { st, adapter, socket, s, d }
    }
}

impl ItemDecoder for Live {
    /// All the audio in: the data worker cuts it on the model's frame.
    fn take_chunk(&self, s: &mut Session) -> Option<Vec<f32>> {
        s.flush()
    }

    /// Put `chunk` on the item's call, opening it with the item's first audio.
    async fn step(rt: &mut Realtime<Self>, chunk: Vec<f32>, _final: bool) {
        if rt.d.failure.is_some() || (rt.d.sub.is_none() && !rt.open().await) {
            return; // the item has failed: its error is sent, and its commit reports it
        }
        let (sub, facts) = (rt.d.sub.expect("open"), rt.d.facts);
        match stream_audio(&rt.st, sub, &facts.stream, &chunk, &rt.s.id).await {
            Ok(true) => {}
            Ok(false) => return, // the call has ended: its output says why
            Err(error) => {
                rt.send(error).await;
                return;
            }
        }
        rt.s.streamed_samples += chunk.len();
        rt.s.chunk_id += 1;
    }

    /// End the item: its stream ends, and the rest of its text arrives.
    async fn commit(rt: &mut Realtime<Self>) {
        if let Some(tail) = rt.s.flush() {
            Self::step(rt, tail, true).await;
        }
        let (sub, mut output) = (rt.d.sub.take(), rt.d.output.take());
        let ended = match (sub, rt.st.bridge.clone()) {
            (None, _) => true,
            (Some(sub), bridge) => {
                let mut op = Map::new();
                op.insert("end_input".into(), Value::Bool(true));
                let reply = match bridge {
                    Some(bridge) => bridge.live(sub, "append", op, crate::BRIDGE_PROBE_TIMEOUT * 5).await,
                    None => None,
                };
                let meta = reply.as_ref().and_then(Value::as_object);
                match meta.filter(|m| m.get("error").is_none()).and_then(|m| m.get("accepted")).and_then(Value::as_bool) {
                    Some(accepted) => accepted,
                    None => {
                        // whether the call's input ended is unknown, so its output
                        // may never end: dropping it aborts the call
                        output = None;
                        rt.fail(super::failed_in_server("commit", meta, &rt.s.id)).await;
                        true
                    }
                }
            }
        };
        if let Some(mut output) = output {
            while let Some(item) = output.next().await {
                Self::relay(rt, Some(item)).await;
            }
        }
        if !ended {
            rt.fail(error_event(ENDED_EARLY, "server_error", None)).await;
        }
        let item_id = rt.s.item_id.clone();
        rt.send(event("input_audio_buffer.committed", json!({"item_id": item_id}))).await;
        match rt.d.failure.take() {
            Some(failure) => {
                let failed = event(
                    "conversation.item.input_audio_transcription.failed",
                    json!({"item_id": item_id, "content_index": 0, "error": failure["error"]}),
                );
                rt.send(failed).await;
            }
            None => {
                let text = rt.current_text();
                rt.emit(&text, 0).await;
                let completed = event("conversation.item.input_audio_transcription.completed", json!({"item_id": item_id, "transcript": text}));
                rt.send(completed).await;
            }
        }
        rt.s.clear();
    }

    /// The re-decoding knobs and a prompt do nothing here, and an open item's
    /// call keeps the language it opened with.
    fn refuse_update(rt: &Realtime<Self>, session: &Map<String, Value>) -> Result<(), String> {
        let knobs = object(session.get("mstar"), "mstar")?;
        if let Some(name) = REDECODE_KNOBS.iter().find(|n| knobs.is_some_and(|k| k.contains_key(**n))) {
            return Err(format!("mstar.{name} does not apply here: this model hears audio as it comes and re-decodes nothing"));
        }
        let empty = Map::new();
        let transcription = object(session.get("input_audio_transcription"), "input_audio_transcription")?.unwrap_or(&empty);
        if truthy(transcription.get("prompt")) {
            return Err("input_audio_transcription.prompt is not supported by this model".into());
        }
        if let (Some(_), Some(language)) = (rt.d.sub, transcription.get("language")) {
            // compared as `(language or None) != (current or None)`
            let sent = truthy(Some(language)).then(|| language.clone());
            let current = rt.s.request.language.clone().filter(|l| !l.is_empty()).map(Value::String);
            if sent != current {
                return Err("commit or clear the input audio buffer before changing the language".into());
            }
        }
        Ok(())
    }

    /// A live session has no re-decoding knobs.
    fn view(&self, view: &mut Value) {
        if let Some(Value::Object(knobs)) = view.get_mut("mstar") {
            for name in REDECODE_KNOBS {
                knobs.remove(name);
            }
        }
    }

    /// A clear drops the item's call too: what it heard can't be unheard.
    /// Dropping its output aborts a call still running.
    fn clear(&mut self) {
        self.sub = None;
        self.output = None;
        self.failure = None;
    }

    fn output(&mut self) -> Option<&mut OutStream> {
        self.output.as_mut()
    }

    /// The call's text as deltas of the item, until the call ends (`None`).
    async fn relay(rt: &mut Realtime<Self>, item: Option<Out>) {
        match item {
            Some(Out::Chunk(chunk)) if chunk.modality == "text" => {
                let piece = rt.d.text.feed(&chunk.data);
                rt.s.raw_hypothesis.push_str(&piece);
                let text = rt.current_text();
                rt.emit(&text, 0).await;
                // the step this text came from covers its chunk of audio
                let step = chunk.metadata.get("step").and_then(Value::as_i64).unwrap_or(rt.s.chunk_id);
                let frame = rt.d.facts.frame_samples as f64 / rt.d.facts.rate as f64;
                let heard = rt.s.streamed_samples as f64 / SAMPLE_RATE as f64;
                let partial = event(
                    "mstar.transcription.partial",
                    json!({
                        "item_id": rt.s.item_id, "text": text, "chunk_id": step,
                        "audio_seconds": ((step + 1) as f64 * frame).min(heard), "created": crate::serving::now(),
                    }),
                );
                rt.send(partial).await;
            }
            Some(Out::Chunk(_)) => {}
            Some(Out::Error { status, message, .. }) => {
                let error = call_error(&rt.st, status, &message);
                rt.fail(error).await;
            }
            None => {
                rt.d.output = None;
                if rt.d.sub.is_some() {
                    // not ended by the commit
                    rt.fail(error_event(ENDED_EARLY, "server_error", None)).await;
                }
            }
        }
    }
}

impl Realtime<Live> {
    /// The item's call failed, or ended before its commit: it has ended, so it
    /// takes no more of the item's audio.
    async fn fail(&mut self, error: Value) {
        self.d.sub = None;
        if self.d.failure.is_none() {
            self.d.failure = Some(error.clone());
            self.send(error).await;
        }
    }

    /// Open the item's call; false, with its `error` sent, if it was refused.
    async fn open(&mut self) -> bool {
        let mut model_kwargs = Map::new();
        if let Some(language) = self.s.request.language.clone().filter(|l| !l.is_empty()) {
            model_kwargs.insert("language".into(), Value::from(language));
        }
        if let Err(message) = self.st.served.select(self.s.request.model.as_deref(), &mut model_kwargs) {
            let error = super::invalid(&super::clip(&super::scrub_paths(&message, &self.st.upload_dir)));
            self.fail(error).await;
            return false;
        }
        let args = SubmitArgs {
            output_modalities: vec!["text".into()],
            model_kwargs,
            end_input: Some(Value::Bool(false)),
            ..SubmitArgs::default()
        };
        match open_call(&self.st, &args, &self.s.id).await {
            Ok((sub, output)) => {
                self.d.sub = Some(sub);
                self.d.output = Some(output);
                self.d.text = Utf8Stream::default();
                true
            }
            Err(error) => {
                self.fail(error).await;
                false
            }
        }
    }
}
