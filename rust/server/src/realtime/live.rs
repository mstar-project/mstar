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
use super::{call_error, error_event, event, object, open_call, stream_audio, Realtime, Session};
use crate::adapters::{OpenAIAdapter, SubmitArgs};
use crate::serving::{AppState, Out, OutStream};
use crate::utf8::Utf8Stream;

/// A re-decoding session's knobs, which a live transcription has no use for.
const REDECODE_KNOBS: [&str; 3] = ["chunk_seconds", "unfixed_chunks", "unfixed_tokens"];

const ENDED_EARLY: &str = "this item's transcription ended before its commit";

/// The open item's call.
pub(super) struct LiveCall {
    facts: &'static Facts,
    /// the call's submission while it takes the item's audio
    sub: Option<u64>,
    /// the call's output, until it ends
    pub(super) output: Option<OutStream>,
    text: Utf8Stream,
    /// the `error` event of this item's failed call (`None`: not failed)
    failure: Option<Value>,
}

impl LiveCall {
    /// A transcription session on `facts`' stream.
    pub(super) fn session(st: AppState, adapter: &'static dyn OpenAIAdapter, socket: WebSocket, mut s: Session, facts: &'static Facts) -> Realtime {
        // the stream's own bound on one item's length, not the re-decoding path's
        if let Some(max_items) = facts.max_items {
            s.max_item_seconds = max_items as f64 / facts.rate as f64;
        }
        let call = LiveCall { facts, sub: None, output: None, text: Utf8Stream::default(), failure: None };
        Realtime { st, adapter, socket, s, live: Some(call) }
    }
}

impl Realtime {
    fn call(&mut self) -> &mut LiveCall {
        self.live.as_mut().expect("a live transcription")
    }

    /// The session as `transcription_session.*` shows it; a live one has no
    /// re-decoding knobs.
    pub(super) fn view(&self) -> Value {
        let mut view = self.s.view();
        if self.live.is_some() {
            if let Some(Value::Object(knobs)) = view.get_mut("mstar") {
                for name in REDECODE_KNOBS {
                    knobs.remove(name);
                }
            }
        }
        view
    }

    /// What a live session's update refuses before the rest applies: the
    /// re-decoding knobs and a prompt do nothing here, and an open item's call
    /// keeps the language it opened with.
    pub(super) fn refuse_live(&self, session: &Map<String, Value>) -> Result<(), String> {
        let knobs = object(session.get("mstar"), "mstar")?;
        if let Some(name) = REDECODE_KNOBS.iter().find(|n| knobs.is_some_and(|k| k.contains_key(**n))) {
            return Err(format!("mstar.{name} does not apply here: this model hears audio as it comes and re-decodes nothing"));
        }
        let empty = Map::new();
        let transcription = object(session.get("input_audio_transcription"), "input_audio_transcription")?.unwrap_or(&empty);
        if truthy(transcription.get("prompt")) {
            return Err("input_audio_transcription.prompt is not supported by this model".into());
        }
        let open = self.live.as_ref().is_some_and(|c| c.sub.is_some());
        if let (true, Some(language)) = (open, transcription.get("language")) {
            // compared as `(language or None) != (current or None)`
            let sent = truthy(Some(language)).then(|| language.clone());
            let current = self.s.request.language.clone().filter(|l| !l.is_empty()).map(Value::String);
            if sent != current {
                return Err("commit or clear the input audio buffer before changing the language".into());
            }
        }
        Ok(())
    }

    /// The item's call failed, or ended before its commit: it has ended, so it
    /// takes no more of the item's audio.
    async fn fail(&mut self, error: Value) {
        let call = self.call();
        call.sub = None;
        if call.failure.is_none() {
            call.failure = Some(error.clone());
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
                let call = self.call();
                call.sub = Some(sub);
                call.output = Some(output);
                call.text = Utf8Stream::default();
                true
            }
            Err(error) => {
                self.fail(error).await;
                false
            }
        }
    }

    /// One item of the call's output: its text as deltas of the item, until
    /// the call ends (`None`).
    pub(super) async fn relay(&mut self, item: Option<Out>) {
        match item {
            Some(Out::Chunk(chunk)) if chunk.modality == "text" => {
                let piece = self.call().text.feed(&chunk.data);
                self.s.raw_hypothesis.push_str(&piece);
                let text = self.current_text();
                self.emit(&text, 0).await;
                // the step this text came from covers its chunk of audio
                let step = chunk.metadata.get("step").and_then(Value::as_i64).unwrap_or(self.s.chunk_id);
                let frame = self.call().facts.frame_samples as f64 / self.call().facts.rate as f64;
                let heard = self.s.streamed_samples as f64 / SAMPLE_RATE as f64;
                let partial = event(
                    "mstar.transcription.partial",
                    json!({
                        "item_id": self.s.item_id, "text": text, "chunk_id": step,
                        "audio_seconds": ((step + 1) as f64 * frame).min(heard), "created": crate::serving::now(),
                    }),
                );
                self.send(partial).await;
            }
            Some(Out::Chunk(_)) => {}
            Some(Out::Error { status, message, .. }) => {
                let error = call_error(&self.st, status, &message);
                self.fail(error).await;
            }
            None => {
                let call = self.call();
                call.output = None;
                if call.sub.is_some() {
                    // not ended by the commit
                    self.fail(error_event(ENDED_EARLY, "server_error", None)).await;
                }
            }
        }
    }

    /// Put `chunk` on the item's call, opening it with the item's first audio.
    pub(super) async fn live_step(&mut self, chunk: Vec<f32>) {
        if self.call().failure.is_some() || (self.call().sub.is_none() && !self.open().await) {
            return; // the item has failed: its error is sent, and its commit reports it
        }
        let (sub, facts) = (self.call().sub.expect("open"), self.call().facts);
        match stream_audio(&self.st, sub, &facts.stream, &chunk, &self.s.id).await {
            Ok(true) => {}
            Ok(false) => return, // the call has ended: its output says why
            Err(error) => {
                self.send(error).await;
                return;
            }
        }
        self.s.streamed_samples += chunk.len();
        self.s.chunk_id += 1;
    }

    /// End the item: its stream ends, and the rest of its text arrives.
    pub(super) async fn live_commit(&mut self) {
        if let Some(tail) = self.s.flush() {
            self.live_step(tail).await;
        }
        let call = self.call();
        let (sub, output) = (call.sub.take(), call.output.take());
        let ended = match (sub, self.st.bridge.clone()) {
            (None, _) => true,
            (Some(sub), Some(bridge)) => {
                let mut op = Map::new();
                op.insert("end_input".into(), Value::Bool(true));
                let reply = bridge.live(sub, "append", op, crate::BRIDGE_PROBE_TIMEOUT * 5).await;
                reply.as_ref().and_then(|m| m.get("accepted")).and_then(Value::as_bool) == Some(true)
            }
            (Some(_), None) => false,
        };
        if let Some(mut output) = output {
            while let Some(item) = output.next().await {
                self.relay(Some(item)).await;
            }
        }
        if !ended {
            self.fail(error_event(ENDED_EARLY, "server_error", None)).await;
        }
        let item_id = self.s.item_id.clone();
        self.send(event("input_audio_buffer.committed", json!({"item_id": item_id}))).await;
        match self.call().failure.take() {
            Some(failure) => {
                let failed = event(
                    "conversation.item.input_audio_transcription.failed",
                    json!({"item_id": item_id, "content_index": 0, "error": failure["error"]}),
                );
                self.send(failed).await;
            }
            None => {
                let text = self.current_text();
                self.emit(&text, 0).await;
                let completed = event("conversation.item.input_audio_transcription.completed", json!({"item_id": item_id, "transcript": text}));
                self.send(completed).await;
            }
        }
        self.s.clear();
    }

    /// A clear drops the item's call too: what it heard can't be unheard.
    /// Dropping its output aborts a call still running.
    pub(super) fn drop_call(&mut self) {
        if let Some(call) = &mut self.live {
            call.sub = None;
            call.output = None;
            call.failure = None;
        }
    }
}
