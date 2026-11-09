//! The speech-to-text adapters (`WhisperAdapter`, `Qwen3ASRAdapter`, the
//! FastConformer ones, `HiggsAudioAdapter`, `OmnilingualASRAdapter`): how an
//! upload's window becomes a submission, and how the model's raw text becomes
//! a transcript. The language tables they check against are the models' own,
//! handed over by the launcher ([`crate::transcription::facts`]).

use super::*;
use crate::generation_kwargs::{self as gk, Kwarg, Kwargs};
use crate::transcription::{facts, Transcript, TranscriptionPolicy, TranscriptionRequest, Segment, Word};

/// `str.splitlines()`.
fn py_splitlines(s: &str) -> Vec<&str> {
    let breaks = |c: char| matches!(c, '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}');
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if breaks(c) {
            lines.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' && chars.peek().is_some_and(|(_, n)| *n == '\n') {
                chars.next();
                end += 1;
            }
            start = end;
        }
    }
    if start < s.len() {
        lines.push(&s[start..]);
    }
    lines
}

/// `mstar.model.utils.base_language`: `en-US`, `en_us` or `EN` -> `en`.
fn base_language(code: &Kwarg) -> Result<String, String> {
    match code {
        Kwarg::Json(Value::String(s)) => Ok(s.replace('_', "-").split('-').next().unwrap_or("").to_lowercase()),
        other => Err(format!("a language is a code such as 'en'; got {}", kwarg_repr(other))),
    }
}

fn kwarg_repr(v: &Kwarg) -> String {
    match v {
        Kwarg::Json(v) => py_value_repr(v),
        Kwarg::Float(f) => py_float_repr(*f),
        Kwarg::Int(i) => i.to_string(),
    }
}

fn kwarg_truthy(v: &Kwarg) -> bool {
    match v {
        Kwarg::Json(v) => truthy(v),
        Kwarg::Float(f) => *f != 0.0,
        Kwarg::Int(i) => *i != 0,
    }
}

/// `x or default` for a kwarg.
fn or<'a>(v: Option<&'a Kwarg>, default: &'a Kwarg) -> &'a Kwarg {
    v.filter(|v| kwarg_truthy(v)).unwrap_or(default)
}

fn s(text: &str) -> Kwarg {
    Kwarg::Json(Value::String(text.to_string()))
}

/// `_transcription_kwargs`: the OpenAI fields every ASR model shares, onto
/// `model_kwargs`, normalized as the Python adapter normalizes them.
fn transcription_kwargs(req: &TranscriptionRequest) -> Result<Kwargs, String> {
    let mut mk: Kwargs = req.extra.iter().map(|(k, v)| (k.clone(), v.clone().into_kwarg())).collect();
    if let Some(language) = req.language.as_deref().filter(|l| !l.is_empty()) {
        gk::set_default(&mut mk, "language", s(language));
    }
    if let Some(prompt) = req.prompt.as_deref().filter(|p| !p.is_empty()) {
        gk::set_default(&mut mk, "initial_prompt", s(prompt));
    }
    if let Some(t) = req.temperature {
        let value = serde_json::Number::from_f64(t).map_or(Kwarg::Float(t), |n| Kwarg::Json(Value::Number(n)));
        gk::set_default(&mut mk, "temperature", value);
    }
    if let Some(seed) = req.seed {
        gk::set_default(&mut mk, "seed", gk::int(seed));
    }
    let granularities = req.timestamp_granularities.as_deref().unwrap_or(&[]);
    if granularities.iter().any(|g| g == "word") {
        gk::set_default(&mut mk, "timestamps", s("word"));
    } else if !granularities.is_empty() || ["verbose_json", "srt", "vtt"].contains(&req.response_format.as_str()) {
        gk::set_default(&mut mk, "timestamps", s("segment"));
    }
    gk::normalize(mk)
}

/// `_untimed_transcription_kwargs`: a model with no conditioning text and no
/// timestamps; `task` is the caller's to read when the model `translates`.
fn untimed_transcription_kwargs(req: &TranscriptionRequest, model: &str, translates: bool) -> Result<Kwargs, String> {
    let mut mk = transcription_kwargs(req)?;
    gk::remove(&mut mk, "initial_prompt");
    if req.timestamp_granularities.as_ref().is_some_and(|g| !g.is_empty()) {
        return Err(format!("{model} does not produce timestamps; leave timestamp_granularities unset"));
    }
    gk::remove(&mut mk, "timestamps");
    if !translates {
        let task = gk::remove(&mut mk, "task");
        if !matches!(&task, None | Some(Kwarg::Json(Value::Null))) && task != Some(s("transcribe")) {
            return Err(format!("{model} only transcribes; task must be 'transcribe'; got {}", kwarg_repr(&task.unwrap())));
        }
    }
    Ok(mk)
}

fn audio_args(audio_path: &str, text: String, input_modalities: &[&str], mk: Kwargs) -> SubmitArgs {
    SubmitArgs {
        text: Some(text),
        file_paths: BTreeMap::from([("audio".to_string(), vec![audio_path.to_string()])]),
        input_modalities: input_modalities.iter().map(|m| m.to_string()).collect(),
        output_modalities: vec!["text".to_string()],
        model_kwargs: gk::into_map(mk),
        ..SubmitArgs::default()
    }
}

/// The FastConformer models' text: a SentencePiece vocabulary starts it with a
/// space, which the parsed transcript drops, so the deltas drop it too.
fn untimed_stream_text(raw: &str) -> String {
    raw.trim_start_matches(py_space).to_string()
}

/// Whisper (large-v3, large-v3-turbo): its own token stream rendered as text,
/// `<|xx|>` language and `<|s.ss|>` timestamp tokens lifted into the transcript.
pub struct Whisper;

impl OpenAIAdapter for Whisper {
    fn supports_transcriptions(&self) -> bool {
        true
    }
    fn transcription_policy(&self) -> TranscriptionPolicy {
        TranscriptionPolicy { max_audio_seconds: Some(30.0), long_form: "parallel", seeks_by_timestamps: true, ..TranscriptionPolicy::DEFAULT }
    }
    fn transcription_to_request(&self, req: &TranscriptionRequest, audio_path: &str) -> Result<SubmitArgs, String> {
        // conditioned by its forced token prompt, so `prompt` goes as `initial_prompt`
        Ok(audio_args(audio_path, String::new(), &["audio", "text"], transcription_kwargs(req)?))
    }
    fn stream_text(&self, raw: &str) -> String {
        let raw = raw.split_once(WORDS_MARKER).map_or(raw, |(head, _)| head);
        control_token().replace_all(&strip_partial_control_token(raw), "").into_owned()
    }
    fn parse_transcript(&self, text: &str, req: &TranscriptionRequest) -> Transcript {
        let (text, timed_words) = text.split_once(WORDS_MARKER).unwrap_or((text, ""));
        let parsed = parse_timestamped(text);
        let clean = parsed.parts.iter().map(|p| py_strip(p)).filter(|p| !p.is_empty()).collect::<Vec<_>>().join(" ");
        let mut segments = parsed.segments;
        for (i, seg) in segments.iter_mut().enumerate() {
            seg.id = Some(i);
        }
        let words = if timed_words.is_empty() {
            Vec::new()
        } else {
            parse_timestamped(timed_words)
                .segments
                .into_iter()
                .map(|seg| Word { word: seg.text, start: seg.start, end: seg.end })
                .collect()
        };
        Transcript {
            text: clean,
            language: parsed.language.or_else(|| req.language.clone()),
            segments,
            words,
            unfinished: parsed.unfinished,
        }
    }
}

/// `WORDS_MARKER` (`model/whisper/config.py`): word timings follow it.
const WORDS_MARKER: &str = "<|startoflm|>";

fn control_token() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"<\|([^|<>]*)\|>").unwrap())
}

/// `_PARTIAL_CONTROL_TOKEN.sub("", raw)`: a control token cut off by the end of
/// the stream so far. Python's `$` also matches before a last newline.
fn strip_partial_control_token(raw: &str) -> String {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"<(\|[^|<>]*\|?)?\z").unwrap());
    if let Some(m) = re.find(raw) {
        return format!("{}{}", &raw[..m.start()], &raw[m.end()..]);
    }
    if let Some(head) = raw.strip_suffix('\n') {
        if let Some(m) = re.find(head) {
            return format!("{}\n", &head[..m.start()]);
        }
    }
    raw.to_string()
}

/// `re.match(pattern, token)` for an anchored pattern, `$` matching before a last newline too.
fn full_match(re: &regex::Regex, token: &str) -> bool {
    re.is_match(token) || token.strip_suffix('\n').is_some_and(|t| re.is_match(t))
}

struct Timestamped {
    language: Option<String>,
    segments: Vec<Segment>,
    parts: Vec<String>,
    unfinished: bool,
}

/// `_parse_timestamped`: the language, the closed segments, the text pieces in
/// order, and whether a segment was left open.
fn parse_timestamped(text: &str) -> Timestamped {
    static TIMESTAMP: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    static LANGUAGE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let timestamp = TIMESTAMP.get_or_init(|| regex::Regex::new(r"^\d+\.\d{2}\z").unwrap());
    let language_re = LANGUAGE.get_or_init(|| regex::Regex::new(r"^[a-z]{2,3}\z").unwrap());
    let mut out = Timestamped { language: None, segments: Vec::new(), parts: Vec::new(), unfinished: false };
    let mut start: Option<f64> = None;
    let mut buffer = String::new();
    let mut cursor = 0;
    for caps in control_token().captures_iter(text) {
        let whole = caps.get(0).unwrap();
        let span = &text[cursor..whole.start()];
        cursor = whole.end();
        if start.is_some() {
            buffer.push_str(span);
        } else {
            out.parts.push(span.to_string());
        }
        let token = &caps[1];
        if full_match(timestamp, token) {
            let stamp: f64 = token.trim_end_matches('\n').parse().unwrap_or(0.0);
            match start {
                None => start = Some(stamp),
                Some(begin) => {
                    let segment_text = py_strip(&buffer).to_string();
                    if !segment_text.is_empty() {
                        out.segments.push(Segment::new(begin, stamp, segment_text.clone()));
                        out.parts.push(segment_text);
                    }
                    buffer.clear();
                    start = None;
                }
            }
        } else if out.language.is_none() && full_match(language_re, token) {
            out.language = Some(token.to_string());
        }
    }
    let tail = &text[cursor..];
    if start.is_some() {
        buffer.push_str(tail);
        let open = py_strip(&buffer);
        if !open.is_empty() {
            // an open segment at the end of the stream: its text stays, with no end
            out.parts.push(open.to_string());
        }
    } else {
        out.parts.push(tail.to_string());
    }
    out.unfinished = start.is_some();
    out
}

/// `ASR_TEXT_TAG` (`model/qwen3_asr/constants.py`).
const ASR_TEXT_TAG: &str = "<asr_text>";

/// Qwen3-ASR: an LLM decoder that writes `language {Name}<asr_text>{text}`,
/// or the text alone when the language is forced. Hears 20 minutes, but long
/// uploads run as 30 s windows at once.
pub struct Qwen3ASR;

impl OpenAIAdapter for Qwen3ASR {
    fn supports_transcriptions(&self) -> bool {
        true
    }
    fn transcription_policy(&self) -> TranscriptionPolicy {
        TranscriptionPolicy {
            max_audio_seconds: Some(1200.0),
            window_seconds: Some(30.0),
            long_form: "parallel",
            // 2.4 is Whisper's bound for 30 s; a 20 minute transcript can pass it
            compression_ratio_threshold: None,
            ..TranscriptionPolicy::DEFAULT
        }
    }
    fn transcription_to_request(&self, req: &TranscriptionRequest, audio_path: &str) -> Result<SubmitArgs, String> {
        let mk = transcription_kwargs(req)?;
        // `str(kwargs.get("initial_prompt") or "")`
        let prompt = match gk::get(&mk, "initial_prompt").filter(|v| kwarg_truthy(v)) {
            Some(Kwarg::Json(Value::String(p))) => p.clone(),
            Some(other) => kwarg_repr(other),
            None => String::new(),
        };
        if prompt.contains("<|audio_pad|>") {
            return Err("the prompt must not contain <|audio_pad|>".to_string());
        }
        Ok(audio_args(audio_path, String::new(), &["audio", "text"], mk))
    }
    fn stream_text(&self, raw: &str) -> String {
        // a detected language opens the stream as `language X<asr_text>`: hold the
        // text back until the tag is in
        if let Some((_, text)) = raw.split_once(ASR_TEXT_TAG) {
            return text.to_string();
        }
        let head = raw.trim_start_matches(py_space);
        if head.starts_with("language") && head.chars().count() < 40 && !head.contains('\n') {
            return String::new();
        }
        raw.to_string()
    }
    fn final_text(&self, text: String) -> String {
        fix_repetitions(&text)
    }
    fn parse_transcript(&self, text: &str, req: &TranscriptionRequest) -> Transcript {
        let raw = py_strip(text);
        let Some((meta, body)) = raw.split_once(ASR_TEXT_TAG) else {
            return Transcript { language: req.language.clone(), ..Transcript::text(raw) };
        };
        let mut language: Option<String> = None;
        for line in py_splitlines(meta) {
            let line = py_strip(line);
            if line.to_lowercase().starts_with("language ") {
                let name = py_strip(line.get("language ".len()..).unwrap_or(""));
                language = (!name.is_empty()).then(|| name.to_string());
                break;
            }
        }
        if language.as_deref().is_some_and(|l| l.to_lowercase() == "none") {
            language = None;
        }
        Transcript {
            language: req.language.clone().filter(|l| !l.is_empty()).or_else(|| qwen3_language_code(language)),
            ..Transcript::text(py_strip(body))
        }
    }
}

/// `_qwen3_language_code`: the ISO code for a language name Qwen3-ASR wrote; a
/// name outside its list passes through.
fn qwen3_language_code(name: Option<String>) -> Option<String> {
    let name = name.filter(|n| !n.is_empty())?;
    let lowered = name.to_lowercase();
    for (code, canonical) in &facts().language_codes {
        if canonical.to_lowercase() == lowered {
            return Some(code.clone());
        }
    }
    Some(name)
}

/// `fix_repetitions` (`model/qwen3_asr/repetition.py`, after qwen-asr's
/// `detect_and_fix_repetitions`): a character run of more than 20 becomes one
/// character, then each leftmost shortest pattern of up to 20 characters
/// repeated 20 or more times becomes one copy. Run tables per period stand in
/// for the reference's backreference, so it is linear in the text.
pub fn fix_repetitions(text: &str) -> String {
    const THRESHOLD: usize = 20;
    const MAX_PATTERN: usize = 20;
    let chars: Vec<char> = text.chars().collect();
    let mut collapsed: Vec<char> = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let mut j = i;
        while j < chars.len() && chars[j] == chars[i] {
            j += 1;
        }
        let run = j - i;
        collapsed.extend(std::iter::repeat_n(chars[i], if run > THRESHOLD { 1 } else { run }));
        i = j;
    }
    let n = collapsed.len();
    // agree[p][i]: how many positions from i on match the character p further on
    let mut agree = vec![vec![0u32; n + 1]; MAX_PATTERN + 1];
    for (p, row) in agree.iter_mut().enumerate().skip(1) {
        for i in (0..n.saturating_sub(p)).rev() {
            row[i] = if collapsed[i] == collapsed[i + p] { row[i + 1] + 1 } else { 0 };
        }
    }
    // the leftmost start from `from` on of a pattern repeated THRESHOLD times, its
    // shortest period, and the copies that follow it
    let find = |from: usize| {
        // the reference looks for a start only up to 2 * THRESHOLD characters from the end
        let last = n.checked_sub(2 * THRESHOLD)?;
        (from..=last).find_map(|start| {
            (1..=MAX_PATTERN.min(n - start))
                .map(|p| (start, p, agree[p][start] as usize / p))
                .find(|(_, _, copies)| *copies >= THRESHOLD - 1)
        })
    };
    let mut out = String::new();
    let mut base = 0;
    while let Some((start, p, copies)) = find(base) {
        out.extend(&collapsed[base..start + p]);
        base = start + p * (copies + 1);
    }
    out.extend(&collapsed[base..]);
    out
}

/// NVIDIA Canary: recognition, and translation to or from English.
pub struct Canary;

impl OpenAIAdapter for Canary {
    fn supports_transcriptions(&self) -> bool {
        true
    }
    fn transcription_policy(&self) -> TranscriptionPolicy {
        TranscriptionPolicy { max_audio_seconds: Some(40.0), long_form: "parallel", compression_ratio_threshold: None, ..TranscriptionPolicy::DEFAULT }
    }
    fn transcription_to_request(&self, req: &TranscriptionRequest, audio_path: &str) -> Result<SubmitArgs, String> {
        let mut mk = untimed_transcription_kwargs(req, "Canary", true)?;
        // Whisper's convention: translate means into English
        let task = gk::remove(&mut mk, "task");
        let translate = task == Some(s("translate"));
        if !matches!(&task, None | Some(Kwarg::Json(Value::Null))) && task != Some(s("transcribe")) && !translate {
            return Err(format!("task must be 'transcribe' or 'translate'; got {}", kwarg_repr(&task.unwrap())));
        }
        let en = s("en");
        let source = or(gk::get(&mk, "language"), &en).clone();
        let fallback = if translate { en.clone() } else { source.clone() };
        let target = or(gk::get(&mk, "target_language"), &fallback).clone();
        let (source, target) = check_task(&source, &target)?;
        set_kwarg(&mut mk, "language", s(&source));
        set_kwarg(&mut mk, "target_language", s(&target));
        if let Some(p) = gk::get(&mk, "punctuation") {
            if !matches!(p, Kwarg::Json(Value::Bool(_))) {
                return Err(format!("punctuation must be true or false; got {}", kwarg_repr(p)));
            }
        }
        Ok(audio_args(audio_path, String::new(), &["audio", "text"], mk))
    }
    fn stream_text(&self, raw: &str) -> String {
        untimed_stream_text(raw)
    }
}

/// `mk[key] = value`, keeping the key's place.
fn set_kwarg(mk: &mut Kwargs, key: &str, value: Kwarg) {
    match mk.iter_mut().find(|(k, _)| k == key) {
        Some((_, v)) => *v = value,
        None => mk.push((key.to_string(), value)),
    }
}

/// `canary/config.py`'s `check_task` over the checkpoint's languages.
fn check_task(source: &Kwarg, target: &Kwarg) -> Result<(String, String), String> {
    let (source, target) = (base_language(source)?, base_language(target)?);
    let languages = &facts().languages;
    for lang in [&source, &target] {
        if !languages.contains(lang) {
            return Err(format!(
                "Canary does not support language {}; supported: {}",
                media::py_repr(lang),
                languages.join(", ")
            ));
        }
    }
    if source != target && source != "en" && target != "en" {
        return Err(format!(
            "Canary translates to or from English only; got {} -> {}",
            media::py_repr(&source),
            media::py_repr(&target)
        ));
    }
    Ok((source, target))
}

/// NVIDIA Canary-Qwen: English recognition by an LLM under its trained instruction.
pub struct CanaryQwen;

impl OpenAIAdapter for CanaryQwen {
    fn supports_transcriptions(&self) -> bool {
        true
    }
    fn transcription_policy(&self) -> TranscriptionPolicy {
        TranscriptionPolicy { max_audio_seconds: Some(40.0), long_form: "parallel", compression_ratio_threshold: None, ..TranscriptionPolicy::DEFAULT }
    }
    fn transcription_to_request(&self, req: &TranscriptionRequest, audio_path: &str) -> Result<SubmitArgs, String> {
        let mk = untimed_transcription_kwargs(req, "Canary-Qwen", false)?;
        let en = s("en");
        if base_language(or(gk::get(&mk, "language"), &en))? != "en" {
            return Err("Canary-Qwen transcribes English only".to_string());
        }
        Ok(audio_args(audio_path, String::new(), &["audio", "text"], mk))
    }
    fn stream_text(&self, raw: &str) -> String {
        untimed_stream_text(raw)
    }
}

/// NVIDIA Parakeet / Nemotron ASR. Parakeet-TDT detects the language and
/// refuses only one it doesn't know; the launcher hands its list over as
/// [`facts`]' `languages`, which Nemotron's deployment leaves empty.
pub struct Transducer;

impl OpenAIAdapter for Transducer {
    fn supports_transcriptions(&self) -> bool {
        true
    }
    fn transcription_policy(&self) -> TranscriptionPolicy {
        TranscriptionPolicy { max_audio_seconds: Some(600.0), long_form: "parallel", compression_ratio_threshold: None, ..TranscriptionPolicy::DEFAULT }
    }
    fn transcription_to_request(&self, req: &TranscriptionRequest, audio_path: &str) -> Result<SubmitArgs, String> {
        let mk = untimed_transcription_kwargs(req, "this model", false)?;
        if facts().detects_language {
            check_detected_language(gk::get(&mk, "language"))?;
        }
        Ok(audio_args(audio_path, String::new(), &["audio", "text"], mk))
    }
    fn stream_text(&self, raw: &str) -> String {
        untimed_stream_text(raw)
    }
}

/// `parakeet/config.py`'s `check_detected_language`.
fn check_detected_language(language: Option<&Kwarg>) -> Result<(), String> {
    let Some(language) = language.filter(|l| !matches!(l, Kwarg::Json(Value::Null))) else { return Ok(()) };
    if *language == s("") {
        return Ok(());
    }
    let base = base_language(language)?;
    let languages = &facts().languages;
    if base != "auto" && !languages.contains(&base) {
        return Err(format!(
            "unsupported language {}; Parakeet detects one of: {}",
            kwarg_repr(language),
            languages.join(", ")
        ));
    }
    Ok(())
}

/// Higgs-Audio v3 STT: an instruction-following decoder, so `prompt` is the instruction.
pub struct HiggsAudio;

impl OpenAIAdapter for HiggsAudio {
    fn supports_transcriptions(&self) -> bool {
        true
    }
    fn transcription_to_request(&self, req: &TranscriptionRequest, audio_path: &str) -> Result<SubmitArgs, String> {
        let mut mk = transcription_kwargs(req)?;
        gk::remove(&mut mk, "initial_prompt");
        Ok(audio_args(audio_path, req.prompt.clone().unwrap_or_default(), &["audio", "text"], mk))
    }
}

/// Omnilingual ASR CTC: a clip of up to 20 s heard whole, a longer one as the
/// fewest windows of at most 16 s, cut at the quietest 0.1 s within 5 s of each
/// boundary, run at once.
pub struct OmnilingualASR;

impl OpenAIAdapter for OmnilingualASR {
    fn supports_transcriptions(&self) -> bool {
        true
    }
    fn transcription_policy(&self) -> TranscriptionPolicy {
        TranscriptionPolicy {
            max_audio_seconds: Some(40.0),
            window_seconds: Some(16.0),
            whole_seconds: Some(20.0),
            balanced_windows: true,
            window_search_seconds: Some(5.0),
            window_frame_seconds: Some(0.1),
            long_form: "parallel",
            compression_ratio_threshold: None,
            ..TranscriptionPolicy::DEFAULT
        }
    }
    fn transcription_to_request(&self, req: &TranscriptionRequest, audio_path: &str) -> Result<SubmitArgs, String> {
        Ok(audio_args(audio_path, String::new(), &["audio"], transcription_kwargs(req)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whisper_lifts_language_segments_and_an_open_segment() {
        let req = TranscriptionRequest::default();
        let t = Whisper.parse_transcript("<|en|><|0.00|> Hello<|1.50|><|1.50|> world<|2.00|><|2.00|> and", &req);
        assert_eq!(t.language.as_deref(), Some("en"));
        assert_eq!(t.text, "Hello world and");
        assert_eq!(t.segments.len(), 2);
        assert_eq!((t.segments[1].start, t.segments[1].end, t.segments[1].text.as_str()), (1.5, 2.0, "world"));
        assert!(t.unfinished);
        assert_eq!(Whisper.stream_text("<|en|> Hi <|0.0"), " Hi ");
    }

    #[test]
    fn fix_repetitions_collapses_runs_and_patterns() {
        assert_eq!(fix_repetitions(&"a".repeat(30)), "a");
        assert_eq!(fix_repetitions(&format!("x{}y{}", "ab".repeat(25), "z".repeat(40))), "xabyz");
        let short = "ab".repeat(10);
        assert_eq!(fix_repetitions(&short), short, "fewer than 20 copies stay");
    }

    #[test]
    fn splitlines_is_pythons() {
        assert_eq!(py_splitlines("a\r\nb\rc\u{2028}d\n"), vec!["a", "b", "c", "d"]);
        assert_eq!(py_splitlines(""), Vec::<&str>::new());
    }
}
