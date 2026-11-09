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
                        // An `unknown` MIME type stays unknown, as in Python, for
                        // the submission to refuse; the part's kind is no evidence.
                        let (m, p) = media::resolve_media_ref(&url, upload_dir, allow_remote)?;
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
        text,
        file_paths,
        input_modalities: in_mods,
        output_modalities: vec!["text".to_string()],
        model_kwargs: mk,
        parts,
    })
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
    if let Some(n) = req.n.filter(|n| *n != 1) {
        return refuse(format!("n={n}: this model returns one choice"));
    }
    if req.stop.as_ref().is_some_and(truthy) {
        return refuse(
            "stop sequences are not supported; the reply ends at a stop token or max_tokens".to_string(),
        );
    }
    let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
    for name in ["logprobs", "top_logprobs", "logit_bias", "presence_penalty", "frequency_penalty", "stop_token_ids"] {
        if let Some(v) = mk.get(name).filter(|v| truthy(v)) {
            return refuse(format!("{name} is not supported; got {}", py_value_repr(v)));
        }
        mk.remove(name);
    }
    if mk.remove("best_of").is_some_and(|v| !(v.is_null() || v == 1)) {
        return refuse("best_of is not supported; this model returns one choice".to_string());
    }
    if mk.remove("response_format").is_some_and(|v| !(v.is_null() || v == serde_json::json!({"type": "text"}))) {
        return refuse("response_format needs constrained decoding; only {'type': 'text'} is served".to_string());
    }
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
        text,
        file_paths: BTreeMap::new(),
        input_modalities: vec!["text".to_string()],
        output_modalities: vec!["text".to_string()],
        model_kwargs: mk,
        parts,
    })
}

/// Refuse a chat that asks the model to call a tool. The rules are
/// `_refuse_tool_calls` in `api_server/openai/adapters.py`.
pub(super) fn refuse_tool_calls(req: &ChatCompletionRequest) -> Result<(), String> {
    let listed = |field: &str| match req.extra.get(field) {
        None | Some(Value::Null) => false,
        Some(Value::Array(t)) => !t.is_empty(),
        Some(_) => true,
    };
    let says_none = |field: &str| req.extra.get(field).and_then(Value::as_str) == Some("none");
    let choice = match req.extra.get("tool_choice") {
        Some(Value::String(c)) => c == "required",
        Some(Value::Object(_)) => true,
        _ => false,
    };
    let function_call = match req.extra.get("function_call") {
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
    Ok(())
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
