//! A transcription's windows stitched into one transcript (`merge_windows`),
//! and the response formats and stream events of `serving_transcriptions.py`.

use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use super::drive::Window;
use super::{Segment, Transcript, TranscriptionRequest, Word};
use crate::adapters::{py_strip, OpenAIAdapter};

pub const SSE_DONE: &str = "data: [DONE]\n\n";

fn num(x: f64) -> Value {
    serde_json::Number::from_f64(x).map_or(Value::Null, Value::Number)
}

/// `merge_windows`: texts joined by a space, segments and words shifted by
/// their window's start (segments also carry the temperature and compression
/// ratio the window was accepted at; a window without timestamps is one
/// segment), the language from the first window that reported one, and the
/// text as the adapter finishes it.
pub fn merge_windows(windows: &[Window], adapter: &dyn OpenAIAdapter, req: &TranscriptionRequest) -> Transcript {
    let mut texts = Vec::new();
    let mut segments: Vec<Segment> = Vec::new();
    let mut words: Vec<Word> = Vec::new();
    let mut language: Option<String> = None;
    for window in windows {
        let t = window.transcript.clone().unwrap_or_else(|| adapter.parse_transcript(&window.raw_text, req));
        if !t.text.is_empty() {
            texts.push(t.text.clone());
        }
        // a window of unknown length (a headerless upload to an unbounded model) gets none
        let untimed = if !t.text.is_empty() && window.duration != 0.0 {
            vec![Segment::new(0.0, window.duration, t.text.clone())]
        } else {
            Vec::new()
        };
        let window_segments = if t.segments.is_empty() { untimed } else { t.segments.clone() };
        for seg in window_segments {
            segments.push(Segment {
                start: seg.start + window.offset,
                end: seg.end + window.offset,
                temperature: Some(window.temperature.unwrap_or(req.temperature.filter(|t| *t != 0.0).unwrap_or(0.0))),
                compression_ratio: Some(window.compression_ratio),
                ..seg
            });
        }
        for word in &t.words {
            words.push(Word { start: word.start + window.offset, end: word.end + window.offset, ..word.clone() });
        }
        if language.as_deref().is_none_or(str::is_empty) {
            language = t.language.clone().filter(|l| !l.is_empty()).or(language);
        }
    }
    for (i, seg) in segments.iter_mut().enumerate() {
        seg.id = Some(i);
    }
    Transcript {
        text: adapter.final_text(texts.join(" ")),
        language: language.filter(|l| !l.is_empty()).or_else(|| req.language.clone()),
        segments,
        words,
        unfinished: false,
    }
}

/// `_fmt_time`: `HH:MM:SS<sep>mmm`, the milliseconds rounded half to even.
fn fmt_time(seconds: f64, sep: char) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round_ties_even() as i64;
    let (hours, rem) = (total_ms / 3_600_000, total_ms % 3_600_000);
    let (minutes, rem) = (rem / 60_000, rem % 60_000);
    let (secs, ms) = (rem / 1000, rem % 1000);
    format!("{hours:02}:{minutes:02}:{secs:02}{sep}{ms:03}")
}

pub fn render_srt(segments: &[Segment]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for (i, seg) in segments.iter().enumerate() {
        lines.push((i + 1).to_string());
        lines.push(format!("{} --> {}", fmt_time(seg.start, ','), fmt_time(seg.end, ',')));
        lines.push(py_strip(&seg.text).to_string());
        lines.push(String::new());
    }
    lines.join("\n")
}

pub fn render_vtt(segments: &[Segment]) -> String {
    let mut lines = vec!["WEBVTT".to_string(), String::new()];
    for seg in segments {
        lines.push(format!("{} --> {}", fmt_time(seg.start, '.'), fmt_time(seg.end, '.')));
        lines.push(py_strip(&seg.text).to_string());
        lines.push(String::new());
    }
    lines.join("\n")
}

/// `_whole`: one segment spanning the file, for a timed format on a model
/// that emitted no timestamps.
fn whole(transcript: &Transcript, duration: Option<f64>) -> Vec<Segment> {
    vec![Segment { id: Some(0), ..Segment::new(0.0, duration.unwrap_or(0.0), transcript.text.clone()) }]
}

fn usage(duration: Option<f64>) -> Value {
    let seconds = duration.filter(|d| *d != 0.0).map_or(0, |d| d.ceil() as i64);
    json!({"type": "duration", "seconds": seconds})
}

fn verbose(transcript: &Transcript, req: &TranscriptionRequest, duration: Option<f64>) -> Value {
    let granularities = req.timestamp_granularities.clone().unwrap_or_default();
    let mut body = json!({
        "task": "transcribe",
        "language": transcript.language,
        "duration": duration.map(num),
        "text": transcript.text,
    });
    if granularities.is_empty() || granularities.iter().any(|g| g == "segment") {
        let fallback = req.temperature.filter(|t| *t != 0.0).unwrap_or(0.0);
        body["segments"] = transcript
            .segments
            .iter()
            .enumerate()
            .map(|(i, seg)| {
                json!({
                    "id": seg.id.unwrap_or(i),
                    "seek": 0,
                    "start": num(seg.start),
                    "end": num(seg.end),
                    "text": seg.text,
                    "tokens": [],
                    "temperature": num(seg.temperature.unwrap_or(fallback)),
                    "avg_logprob": 0.0,
                    "compression_ratio": num(seg.compression_ratio.unwrap_or(0.0)),
                    "no_speech_prob": 0.0,
                })
            })
            .collect();
    }
    if granularities.iter().any(|g| g == "word") {
        body["words"] = transcript
            .words
            .iter()
            .map(|w| json!({"word": w.word, "start": num(w.start), "end": num(w.end)}))
            .collect();
    }
    body["usage"] = usage(duration);
    body
}

fn plain(text: String) -> Response {
    ([(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response()
}

/// The non-streamed response in `fmt`.
pub fn response(fmt: &str, transcript: &Transcript, req: &TranscriptionRequest, duration: Option<f64>) -> Response {
    let timed = |render: fn(&[Segment]) -> String| {
        if transcript.segments.is_empty() {
            render(&whole(transcript, duration))
        } else {
            render(&transcript.segments)
        }
    };
    match fmt {
        "text" => plain(transcript.text.clone()),
        "json" => Json(json!({"text": transcript.text})).into_response(),
        "srt" => plain(timed(render_srt)),
        "vtt" => plain(timed(render_vtt)),
        _ => Json(verbose(transcript, req, duration)).into_response(),
    }
}

fn quoted(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_default()
}

// The events are written field by field, in the Python route's key order.

pub fn delta_event(delta: &str) -> String {
    format!("data: {{\"type\":\"transcript.text.delta\",\"delta\":{}}}\n\n", quoted(delta))
}

pub fn done_event(text: &str, duration: Option<f64>) -> String {
    let seconds = duration.filter(|d| *d != 0.0).map_or(0, |d| d.ceil() as i64);
    format!(
        "data: {{\"type\":\"transcript.text.done\",\"text\":{},\"usage\":{{\"type\":\"duration\",\"seconds\":{seconds}}}}}\n\n",
        quoted(text)
    )
}

/// `_error_event`: in band, since the stream has sent its 200.
pub fn error_event(message: &str, status: u16) -> String {
    format!(
        "data: {{\"type\":\"error\",\"error\":{{\"message\":{},\"type\":{},\"code\":{status}}}}}\n\n",
        quoted(message),
        quoted(crate::error_type(status))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srt_and_vtt_read_as_pythons() {
        let segs = vec![Segment::new(0.0, 1.5, " Hello ".to_string()), Segment::new(3661.0005, 3662.0, "world".to_string())];
        assert_eq!(render_srt(&segs), "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n2\n01:01:01,000 --> 01:01:02,000\nworld\n");
        assert_eq!(render_vtt(&segs[..1]), "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello\n");
    }

    #[test]
    fn events_keep_pythons_key_order() {
        assert_eq!(delta_event("é"), "data: {\"type\":\"transcript.text.delta\",\"delta\":\"é\"}\n\n");
        assert_eq!(
            error_event("boom", 503),
            "data: {\"type\":\"error\",\"error\":{\"message\":\"boom\",\"type\":\"server_error\",\"code\":503}}\n\n"
        );
    }
}
