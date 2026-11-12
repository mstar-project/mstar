//! A Responses request as the chat request it asks the model's adapter for
//! (`serving_responses.to_chat_request`).

use serde_json::{json, Map, Value};

use crate::adapters::{py_value_repr, truthy, OpenAIAdapter};
use crate::protocol::ChatCompletionRequest;
use crate::translate::{self, Effort};

const FIELDS: &[&str] = &[
    "model",
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "max_output_tokens",
    "reasoning",
    "temperature",
    "top_p",
    "stream",
    "store",
    "include",
    "text",
    "truncation",
    "service_tier",
    "top_logprobs",
    "metadata",
    "user",
    "prompt_cache_key",
    "safety_identifier",
    "stream_options",
    "previous_response_id",
    "conversation",
    "prompt",
    "background",
    "max_tool_calls",
    "client_metadata",
    "prefix_cache",
];

/// Python's `==` on JSON values, where `False == 0 == 0.0` and `True == 1`.
fn py_eq(a: &Value, b: &Value) -> bool {
    let num = |v: &Value| match v {
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        Value::Number(n) => n.as_f64(),
        _ => None,
    };
    match (num(a), num(b)) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

/// The chat request a Responses body asks `adapter` for.
pub fn to_chat_request(
    body: &Value,
    adapter: &dyn OpenAIAdapter,
) -> Result<ChatCompletionRequest, String> {
    let Value::Object(body) = body else {
        return Err("the request body must be a JSON object".to_string());
    };
    translate::refuse_unknown(body, FIELDS)?;
    let null = Value::Null;
    let get = |key: &str| body.get(key).unwrap_or(&null);
    // set to anything but their off value, these need what a stateless server does not have
    let off: [(&str, &[Value]); 9] = [
        ("previous_response_id", &[Value::Null]),
        ("conversation", &[Value::Null]),
        ("prompt", &[Value::Null]),
        ("store", &[Value::Null, Value::Bool(false)]),
        ("background", &[Value::Null, Value::Bool(false)]),
        ("max_tool_calls", &[Value::Null]),
        ("top_logprobs", &[Value::Null, json!(0)]),
        ("truncation", &[Value::Null, json!("disabled")]),
        (
            "service_tier",
            &[Value::Null, json!("auto"), json!("default")],
        ),
    ];
    for (key, values) in off {
        let v = get(key);
        if !values.iter().any(|o| py_eq(v, o)) {
            return Err(format!(
                "{key}={} is not served: responses are stateless and not stored",
                py_value_repr(v)
            ));
        }
    }
    let include = Some(get("include")).filter(|v| truthy(v));
    if let Some(include) = include {
        let ok = include
            .as_array()
            .is_some_and(|items| items.iter().all(|i| i == "reasoning.encrypted_content"));
        if !ok {
            return Err(format!(
                "include takes only 'reasoning.encrypted_content'; got {}",
                py_value_repr(include)
            ));
        }
    }
    let text = Some(get("text"))
        .filter(|v| truthy(v))
        .cloned()
        .unwrap_or(json!({}));
    let text_ok = text.as_object().is_some_and(|t| {
        let format = t
            .get("format")
            .filter(|f| truthy(f))
            .cloned()
            .unwrap_or(json!({"type": "text"}));
        format.get("type") == Some(&Value::from("text"))
            && t.get("verbosity")
                .is_none_or(|v| v.is_null() || v == "medium")
    });
    if !text_ok {
        return Err(format!(
            "text takes the text format only, structured output needs constrained decoding; got {}",
            py_value_repr(&text)
        ));
    }

    let mut messages = Vec::new();
    if truthy(get("instructions")) {
        messages.push(json!({"role": "system", "content": get("instructions")}));
    }
    messages.extend(input(get("input"))?);
    // the model names a LoRA adapter as on chat completions; with none served any name runs the model
    let mut chat = Map::new();
    chat.insert("model".into(), get("model").clone());
    chat.insert("messages".into(), Value::Array(messages));
    chat.insert(
        "stream".into(),
        if truthy(get("stream")) {
            get("stream").clone()
        } else {
            Value::Bool(false)
        },
    );
    if !get("max_output_tokens").is_null() {
        chat.insert("max_tokens".into(), get("max_output_tokens").clone());
    }
    for key in ["temperature", "top_p", "parallel_tool_calls"] {
        if !get(key).is_null() {
            chat.insert(key.into(), get(key).clone());
        }
    }
    translate::prefix_cache(body, &mut chat)?;
    if truthy(get("tools")) {
        chat.insert("tools".into(), tools(get("tools"))?);
    }
    if !get("tool_choice").is_null() {
        chat.insert("tool_choice".into(), tool_choice(get("tool_choice"))?);
    }
    chat.extend(reasoning(get("reasoning"), adapter)?);
    serde_json::from_value(Value::Object(chat)).map_err(|e| e.to_string())
}

/// The text of a string or a list of text parts.
fn parts_text(content: &Value, place: &str) -> Result<String, String> {
    let parts = match content {
        Value::String(s) => return Ok(s.clone()),
        Value::Array(parts) => parts,
        _ => {
            return Err(format!(
                "{place} must be a string or a list of content parts"
            ))
        }
    };
    let mut texts = String::new();
    for part in parts {
        let kind = part.get("type");
        match (
            kind.and_then(Value::as_str),
            part.get("text"),
            part.get("refusal"),
        ) {
            (Some("input_text" | "output_text"), Some(Value::String(t)), _) => texts.push_str(t),
            (Some("refusal"), _, Some(Value::String(t))) => texts.push_str(t),
            _ => {
                let reads = match kind.and_then(Value::as_str) {
                    Some("input_image" | "input_file" | "input_audio") => " (the model reads text)",
                    _ => "",
                };
                return Err(format!(
                    "{place} takes text parts only; got {}{reads}",
                    py_value_repr(kind.unwrap_or(&Value::Null))
                ));
            }
        }
    }
    Ok(texts)
}

fn input(items: &Value) -> Result<Vec<Value>, String> {
    let items = match items {
        Value::String(s) => return Ok(vec![json!({"role": "user", "content": s})]),
        Value::Array(items) if !items.is_empty() => items,
        _ => return Err("input must be a string or a non-empty list of items".to_string()),
    };
    let mut out: Vec<Value> = Vec::new();
    // the assistant turn being put together
    let mut turn: Option<Map<String, Value>> = None;
    let flush = |turn: &mut Option<Map<String, Value>>, out: &mut Vec<Value>| {
        if let Some(t) = turn.take() {
            out.push(Value::Object(t));
        }
    };
    let assistant = || Map::from_iter([("role".to_string(), Value::from("assistant"))]);
    for item in items {
        let Value::Object(fields) = item else {
            return Err(format!(
                "input items are objects; got {}",
                py_value_repr(item)
            ));
        };
        let null = Value::Null;
        let get = |key: &str| fields.get(key).unwrap_or(&null);
        let kind = fields
            .get("type")
            .cloned()
            .unwrap_or(Value::from("message"));
        match kind.as_str() {
            Some("message") => {
                let role = get("role");
                let Some(role @ ("user" | "assistant" | "system" | "developer")) = role.as_str() else {
                    return Err(format!(
                        "a message's role must be user, assistant, system or developer, not {}",
                        py_value_repr(role)
                    ));
                };
                let text = parts_text(get("content"), &format!("a {role} message's content"))?;
                if role == "assistant" {
                    if turn.as_ref().is_some_and(|t| t.get("content").is_some_and(truthy) || t.get("tool_calls").is_some_and(truthy)) {
                        flush(&mut turn, &mut out);
                    }
                    turn.get_or_insert_with(assistant).insert("content".into(), Value::from(text));
                } else {
                    flush(&mut turn, &mut out);
                    out.push(json!({"role": role, "content": text}));
                }
            }
            Some("reasoning") => {
                let texts = |key: &str| -> Vec<Value> {
                    get(key).as_array().map_or(Vec::new(), |parts| {
                        parts.iter().filter(|p| p.is_object()).map(|p| p.get("text").cloned().unwrap_or(Value::Null)).collect()
                    })
                };
                let mut found = texts("content");
                if found.is_empty() {
                    found = texts("summary");
                }
                if !found.iter().all(Value::is_string) {
                    return Err(format!(
                        "a reasoning item's content and summary are text parts; got {}",
                        py_value_repr(item)
                    ));
                }
                if found.is_empty() && truthy(get("encrypted_content")) {
                    return Err("a reasoning item with only encrypted_content cannot be read; send its content".to_string());
                }
                flush(&mut turn, &mut out);
                let joined: Vec<&str> = found.iter().filter_map(Value::as_str).collect();
                let mut t = assistant();
                t.insert("reasoning_content".into(), Value::from(joined.join("\n")));
                turn = Some(t);
            }
            Some("function_call") => {
                if !["call_id", "name", "arguments"].iter().all(|k| get(k).is_string()) {
                    return Err("a function_call item needs its call_id, name and arguments".to_string());
                }
                let call = json!({"id": get("call_id"), "type": "function",
                                  "function": {"name": get("name"), "arguments": get("arguments")}});
                let t = turn.get_or_insert_with(assistant);
                match t.entry("tool_calls").or_insert_with(|| Value::Array(Vec::new())) {
                    Value::Array(calls) => calls.push(call),
                    other => *other = Value::Array(vec![call]),
                }
            }
            Some("function_call_output") => {
                let Some(call_id) = get("call_id").as_str() else {
                    return Err("a function_call_output item needs its call_id".to_string());
                };
                flush(&mut turn, &mut out);
                let output = fields.get("output").cloned().unwrap_or(Value::from(""));
                out.push(json!({"role": "tool", "tool_call_id": call_id,
                                "content": parts_text(&output, "a function_call_output's output")?}));
            }
            _ => {
                return Err(format!(
                    "input item type {} is not served; send message, function_call, function_call_output or reasoning items",
                    py_value_repr(&kind)
                ))
            }
        }
    }
    flush(&mut turn, &mut out);
    if out.last().and_then(|m| m.get("role")) == Some(&Value::from("assistant")) {
        return Err("the input ends with an assistant turn; end with a user message or a function_call_output".to_string());
    }
    Ok(out)
}

fn tools(tools: &Value) -> Result<Value, String> {
    let Value::Array(tools) = tools else {
        return Err("tools must be a list".to_string());
    };
    let mut out = Vec::new();
    for tool in tools {
        let kind = tool.get("type").unwrap_or(&Value::Null);
        if kind != "function" {
            return Err(format!(
                "tools take function tools only; {} is not served",
                py_value_repr(kind)
            ));
        }
        let Some(name) = tool.get("name").and_then(Value::as_str) else {
            return Err(format!(
                "a function tool needs a name; got {}",
                py_value_repr(tool)
            ));
        };
        if tool.get("strict").is_some_and(truthy) {
            return Err(format!(
                "tool {}: strict schemas need constrained decoding; send strict: false",
                crate::media::py_repr(name)
            ));
        }
        let parameters = tool
            .get("parameters")
            .filter(|p| truthy(p))
            .cloned()
            .unwrap_or(json!({"type": "object", "properties": {}}));
        let mut function = json!({"name": name, "parameters": parameters});
        if let Some(d) = tool.get("description").filter(|d| truthy(d)) {
            function["description"] = d.clone();
        }
        out.push(json!({"type": "function", "function": function}));
    }
    Ok(Value::Array(out))
}

fn tool_choice(choice: &Value) -> Result<Value, String> {
    if ["auto", "none", "required"].iter().any(|c| choice == *c) {
        return Ok(choice.clone());
    }
    if choice.get("type") == Some(&Value::from("function")) {
        if let Some(name) = choice.get("name").filter(|n| n.is_string()) {
            return Ok(json!({"type": "function", "function": {"name": name}}));
        }
    }
    Err(format!(
        "tool_choice takes 'auto', 'none', 'required' or a function by name; got {}",
        py_value_repr(choice)
    ))
}

fn reasoning(reasoning: &Value, adapter: &dyn OpenAIAdapter) -> Result<Map<String, Value>, String> {
    if reasoning.is_null() {
        return Ok(Map::new());
    }
    let Value::Object(r) = reasoning else {
        return Err(format!(
            "reasoning must be an object; got {}",
            py_value_repr(reasoning)
        ));
    };
    for key in ["summary", "generate_summary"] {
        if let Some(v) = r.get(key).filter(|v| !(v.is_null() || *v == "auto")) {
            return Err(format!(
                "reasoning.{key}={}: no summary is made; the reasoning comes whole",
                py_value_repr(v)
            ));
        }
    }
    let effort = r.get("effort").unwrap_or(&Value::Null);
    let level = match effort.as_str() {
        _ if effort.is_null() => return Ok(Map::new()),
        Some("none") => {
            return Ok(Map::from_iter([(
                "enable_thinking".to_string(),
                Value::Bool(false),
            )]))
        }
        Some("minimal" | "low") => Effort::Low,
        Some("medium") => Effort::Medium,
        Some("high") => Effort::High,
        Some("xhigh") => Effort::Top,
        _ => {
            return Err(format!(
                "reasoning.effort takes none, minimal, low, medium, high or xhigh; got {}",
                py_value_repr(effort)
            ))
        }
    };
    translate::template_level(
        level,
        adapter,
        &format!("reasoning.effort={}", py_value_repr(effort)),
    )
}
