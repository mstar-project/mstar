//! Chat messages as a prompt (`flatten_messages`), and the chat translations
//! several adapters share: the tool-call refusal, the standard sampling fields, a
//! text-only model's refusals.

use super::*;

/// Flatten OpenAI chat `messages` into (text, file_paths, input_modalities, parts).
///
/// `parts` is the ordered sequence as written; the other three derive from
/// it: text newline-joined, attachments persisted under `upload_dir` and
/// grouped by modality, `input_modalities` the per-part modality sequence.
/// So an attachment's position and a repeated modality both survive. Each
/// text part carries its message's role, so a reply stays its own turn. The rules
/// are `flatten_messages` in `api_server/openai/adapters.py`, line for line.
pub fn flatten_messages(
    messages: &[crate::protocol::ChatMessage],
    upload_dir: &Path,
    allow_remote: bool,
) -> Result<(Option<String>, BTreeMap<String, Vec<String>>, Vec<String>, Vec<Part>), String> {
    let mut parts: Vec<Part> = Vec::new();
    // A refused chat is never submitted, so a refusal below drops `saved`
    // unkept, which deletes the attachments saved before it.
    let mut saved = crate::SavedUploads::new(upload_dir);

    for msg in messages {
        let role = match msg.role.as_str() {
            "developer" => "system", // OpenAI's newer name for system
            role @ ("system" | "user" | "assistant") => role,
            other => {
                return Err(format!(
                    "a message's role must be system, developer, user or assistant, not '{other}'"
                ))
            }
        };
        let content = match &msg.content {
            None => continue,
            Some(Content::Text(s)) => {
                if !s.is_empty() {
                    add_text(&mut parts, s, role);
                }
                continue;
            }
            Some(Content::Parts(content)) => content,
        };
        for part in content {
            let obj = match part.as_object() {
                Some(o) => o,
                None => continue,
            };
            let ptype = obj.get("type").and_then(Value::as_str).unwrap_or("");
            match ptype {
                "text" => {
                    if let Some(t) = obj.get("text").and_then(Value::as_str) {
                        if !t.is_empty() {
                            add_text(&mut parts, t, role);
                        }
                    }
                }
                "image_url" | "video_url" | "audio_url" => {
                    let url = nested_url(obj, ptype);
                    if !url.is_empty() {
                        let fallback = ptype.trim_end_matches("_url");
                        let (m, p) = media::resolve_media_ref(&url, upload_dir, allow_remote)?;
                        let m = if m == "unknown" { fallback.to_string() } else { m };
                        add_file(&mut parts, &mut saved.paths, m, p);
                    }
                }
                "input_audio" => {
                    // OpenAI-native audio input: base64 + format.
                    let ia = obj.get("input_audio").and_then(Value::as_object);
                    if let Some(ia) = ia {
                        let data = ia.get("data").and_then(Value::as_str).unwrap_or("");
                        let fmt = ia.get("format").and_then(Value::as_str).unwrap_or("wav");
                        if !data.is_empty() {
                            let (m, p) = media::save_base64(data, fmt, "audio", upload_dir)?;
                            add_file(&mut parts, &mut saved.paths, m, p);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let texts: Vec<&str> = parts.iter().filter_map(|p| p.text.as_deref()).collect();
    let text = if texts.is_empty() { None } else { Some(texts.join("\n")) };
    let input_modalities = parts.iter().map(|p| p.modality.clone()).collect();
    Ok((text, saved.keep(), input_modalities, parts))
}

pub(super) fn add_file(
    parts: &mut Vec<Part>,
    file_paths: &mut BTreeMap<String, Vec<String>>,
    modality: String,
    path: String,
) {
    // shortcut: always a user turn, so the model can't tell an image it made from one it was
    // sent, and a reply that is only an image leaves no assistant turn; a feature like
    // multi-turn editing would give the turn break between two attachments a text slot
    let paths = file_paths.entry(modality.clone()).or_default();
    parts.push(Part {
        modality,
        text: None,
        index: paths.len(),
        role: Some("user".to_string()),
    });
    paths.push(path);
}

pub(super) fn add_text(parts: &mut Vec<Part>, text: &str, role: &str) {
    // Adjacent text parts were newline-joined before ordering was kept; merge
    // them here, across messages of one role too (the layout has no slot for
    // the boundary between them), so only text an attachment or a role change
    // separates gets its own part.
    if let Some(last) = parts.last_mut() {
        if last.modality == "text" && last.role.as_deref() == Some(role) {
            let merged = last.text.get_or_insert_with(String::new);
            merged.push('\n');
            merged.push_str(text);
            return;
        }
    }
    parts.push(Part {
        modality: "text".to_string(),
        text: Some(text.to_string()),
        index: 0,
        role: Some(role.to_string()),
    });
}

/// `{ "<key>": { "url": "..." } }` -> the url string (or "").
pub(super) fn nested_url(obj: &Map<String, Value>, key: &str) -> String {
    obj.get(key)
        .and_then(Value::as_object)
        .and_then(|o| o.get("url"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

/// `max_completion_tokens`, else `max_tokens`, as Python reads them: an
/// explicit 0 is kept, so validation refuses it.
pub(super) fn chat_max_tokens(req: &ChatCompletionRequest) -> Option<i64> {
    req.max_completion_tokens.or(req.max_tokens)
}

/// Text out of a chat's messages, with the standard sampling fields
/// (`BagelAdapter` / `Qwen3_5Adapter.chat_to_request`).
pub(super) fn text_chat(
    req: &ChatCompletionRequest,
    upload_dir: &Path,
    allow_remote: bool,
) -> Result<SubmitArgs, String> {
    let (text, file_paths, in_mods, parts) = flatten_messages(&req.messages, upload_dir, allow_remote)?;
    let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
    apply_sampling(
        &mut mk,
        Sampling {
            temperature: req.temperature.clone(),
            top_p: req.top_p.clone(),
            seed: req.seed,
            max_tokens: chat_max_tokens(req),
            temperature_key: "temperature",
            top_p_key: "top_p",
            max_tokens_key: Some("max_output_tokens"),
        },
    );
    Ok(SubmitArgs {
        tokens: None,
        report_usage: false,
        text,
        file_paths,
        input_modalities: in_mods,
        output_modalities: vec!["text".to_string()],
        model_kwargs: mk,
        parts,
        ..SubmitArgs::default()
    })
}

/// A chat whose `chat_template_kwargs` keys are read as flat ones, as
/// `_flatten_template_kwargs` reads them (Qwen3.5, Cosmos3-Edge): how vLLM-style
/// clients send template variables. A flat key wins, except one the refusal
/// dropped as off. Read before the attachments are saved, since a malformed
/// value refuses the request.
pub(super) fn template_kwargs_chat(req: &ChatCompletionRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
    let mut req = req.clone();
    let template_kwargs = match req.extra.remove("chat_template_kwargs") {
        Some(Value::Object(m)) => m,
        Some(v) if truthy(&v) => return Err("chat_template_kwargs must be an object".to_string()),
        _ => Map::new(),
    };
    let mut args = text_chat(&req, upload_dir, allow_remote)?;
    for (key, value) in template_kwargs {
        set_default(&mut args.model_kwargs, &key, value);
    }
    Ok(args)
}

/// `TextChatAdapter.chat_to_request`: text-only LLMs, which refuse what they
/// cannot serve rather than dropping it. Remote media is never fetched.
pub(super) fn text_only_chat(req: &ChatCompletionRequest, upload_dir: &Path) -> Result<SubmitArgs, String> {
    let (text, file_paths, in_mods, parts) = flatten_messages(&req.messages, upload_dir, false)?;
    let refuse = |message: String| {
        // a refused chat is never submitted, so nothing else removes its attachments
        for path in file_paths.values().flatten() {
            let _ = std::fs::remove_file(path);
        }
        Err(message)
    };
    let other: std::collections::BTreeSet<&str> =
        in_mods.iter().map(String::as_str).filter(|m| *m != "text").collect();
    if !other.is_empty() {
        let listed: Vec<String> = other.iter().map(|m| crate::media::py_repr(m)).collect();
        return refuse(format!("this model takes text only; the messages carry [{}]", listed.join(", ")));
    }
    if text.as_deref().is_none_or(str::is_empty) {
        return refuse("messages carry no text content".to_string());
    }
    for (name, value) in [("max_tokens", req.max_tokens), ("max_completion_tokens", req.max_completion_tokens)] {
        if let Some(v) = value.filter(|v| *v < 1) {
            return refuse(format!("{name} must be at least 1, got {v}"));
        }
    }
    // n, stop and the unserved fields were refused, or dropped, by refuse_unserved_chat
    let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
    apply_sampling(
        &mut mk,
        Sampling {
            temperature: req.temperature.clone(),
            top_p: req.top_p.clone(),
            seed: req.seed,
            max_tokens: chat_max_tokens(req),
            temperature_key: "temperature",
            top_p_key: "top_p",
            max_tokens_key: Some("max_output_tokens"),
        },
    );
    Ok(SubmitArgs {
        tokens: None,
        report_usage: false,
        text,
        file_paths: BTreeMap::new(),
        // one per part, all text past the refusal: a model that rebuilds its turns from the modalities keeps them all
        input_modalities: in_mods,
        output_modalities: vec!["text".to_string()],
        model_kwargs: mk,
        parts,
        ..SubmitArgs::default()
    })
}

/// Fields no chat adapter serves, refused when on (`best_of` has its own off
/// values).
const UNSERVED_CHAT_FIELDS: [&str; 6] =
    ["logprobs", "top_logprobs", "logit_bias", "presence_penalty", "frequency_penalty", "stop_token_ids"];

/// OpenAI request bookkeeping that changes no reply, dropped from a chat's
/// passthrough: `CHAT_ENVELOPE_KEYS` in `api_server/openai/adapters.py`.
const CHAT_ENVELOPE_FIELDS: [&str; 6] =
    ["stream_options", "parallel_tool_calls", "user", "store", "metadata", "service_tier"];

/// Refuse a chat that asks for more than one plain reply: more choices, stop
/// sequences, log probs, sampling penalties, or a tool call. A JSON
/// `response_format` becomes `structured_outputs`. The rules are `_refuse_unserved_chat` in
/// `api_server/openai/adapters.py`. `mk` is the request's passthrough; an off
/// value of one of these fields (`logprobs: false`, `tools: []`) is dropped
/// from it, before any `chat_template_kwargs` key is merged in.
pub(super) fn refuse_unserved_chat(req: &ChatCompletionRequest, mk: &mut BTreeMap<String, Value>) -> Result<(), String> {
    if let Some(n) = req.n.filter(|&n| n != 1) {
        return Err(format!("n={n}: this model returns one choice"));
    }
    if req.stop.as_ref().is_some_and(truthy) {
        return Err(
            "stop sequences are not supported; the reply ends at a stop token or max_tokens".to_string(),
        );
    }
    if let Some(name) = UNSERVED_CHAT_FIELDS.iter().find(|&&f| mk.get(f).is_some_and(truthy)) {
        return Err(format!("{name} is not supported"));
    }
    // Python's `not in (None, 1)`, where 1.0 and True equal 1
    if let Some(v) = mk.get("best_of") {
        if !(v.is_null() || v.as_f64() == Some(1.0) || *v == Value::Bool(true)) {
            return Err("best_of is not supported; this model returns one choice".to_string());
        }
    }
    // json_object / json_schema become the engine's grammar request (`structured_outputs`)
    if let Some(v) = mk.remove("response_format") {
        if let Some(structured) = response_format_request(&v)? {
            if mk.get("structured_outputs").is_some_and(|x| !x.is_null()) {
                return Err("send response_format or structured_outputs, not both".to_string());
            }
            mk.insert("structured_outputs".to_string(), structured);
        }
    }
    let listed = |field: &str| match mk.get(field) {
        None | Some(Value::Null) => false,
        Some(Value::Array(t)) => !t.is_empty(),
        Some(_) => true,
    };
    let says_none = |field: &str| mk.get(field).and_then(Value::as_str) == Some("none");
    let choice = match mk.get("tool_choice") {
        Some(Value::String(c)) => c == "required",
        Some(Value::Object(_)) => true,
        _ => false,
    };
    let function_call = match mk.get("function_call") {
        None | Some(Value::Null) => false,
        Some(Value::String(c)) => c != "none" && c != "auto",
        Some(_) => true,
    };
    if (listed("tools") && !says_none("tool_choice"))
        || (listed("functions") && !says_none("function_call"))
        || choice
        || function_call
    {
        return Err("tool calling is not supported for this model".to_string());
    }
    let off = ["best_of", "tools", "tool_choice", "functions", "function_call"];
    for name in UNSERVED_CHAT_FIELDS.iter().chain(&off).chain(&CHAT_ENVELOPE_FIELDS) {
        mk.remove(*name);
    }
    Ok(())
}

/// OpenAI `response_format` as a `structured_outputs` value, None for text:
/// `response_format_request` in `engine/resources/sampler/grammar.py`.
fn response_format_request(v: &Value) -> Result<Option<Value>, String> {
    let obj = match v {
        Value::Null => return Ok(None),
        Value::Object(o) => o,
        _ => return Err(format!("response_format must be an object; got {v}")),
    };
    match obj.get("type").and_then(Value::as_str) {
        Some("text") => Ok(None),
        Some("json_object") => Ok(Some(serde_json::json!({"json_object": true}))),
        Some("json_schema") => match obj.get("json_schema").and_then(|s| s.get("schema")) {
            Some(schema) if !schema.is_null() => Ok(Some(serde_json::json!({"json": schema}))),
            _ => Err("response_format json_schema needs json_schema.schema".to_string()),
        },
        other => Err(format!(
            "response_format type must be text, json_object or json_schema; got {}",
            other.map_or("none".to_string(), |t| format!("'{t}'"))
        )),
    }
}

/// Chat `audio.voice` wins over a top-level `voice` (Qwen3-Omni).
pub(super) fn chat_voice(req: &ChatCompletionRequest) -> Option<String> {
    if let Some(Value::Object(a)) = &req.audio {
        if let Some(v) = a.get("voice").and_then(Value::as_str) {
            return Some(v.to_string());
        }
    }
    req.extra
        .get("voice")
        .and_then(Value::as_str)
        .map(str::to_string)
}
