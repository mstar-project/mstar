//! A Messages request as the chat request it asks the model's adapter for
//! (`serving_messages.to_chat_request`).

use serde_json::{json, Map, Value};

use crate::adapters::{py_type_name, py_value_repr, truthy, OpenAIAdapter};
use crate::protocol::ChatCompletionRequest;
use crate::reply::MAX_STOP_CHARS;
use crate::translate::{self, Effort};

const FIELDS: &[&str] = &[
    "model",
    "messages",
    "system",
    "max_tokens",
    "stop_sequences",
    "stream",
    "temperature",
    "top_p",
    "top_k",
    "tools",
    "tool_choice",
    "thinking",
    "metadata",
    "context_management",
    "output_config",
    "prefix_cache",
];
const COUNT_FIELDS: &[&str] = &[
    "model",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "thinking",
    "context_management",
    "output_config",
];
/// The stop matcher scans every stop on each token.
const MAX_STOPS: usize = 16;

/// Which route's body is read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Route {
    Messages,
    /// `count_tokens`, whose body has no `max_tokens` and generates nothing.
    CountTokens,
}

/// A Messages request translated onto chat.
pub struct Translated {
    pub chat: ChatCompletionRequest,
    pub stop_sequences: Vec<String>,
}

/// The chat request a Messages body asks `adapter` for.
pub fn to_chat_request(
    body: &Value,
    route: Route,
    adapter: &dyn OpenAIAdapter,
) -> Result<Translated, String> {
    let Value::Object(body) = body else {
        return Err("the request body must be a JSON object".to_string());
    };
    translate::refuse_unknown(
        body,
        if route == Route::CountTokens {
            COUNT_FIELDS
        } else {
            FIELDS
        },
    )?;
    let get = |key: &str| body.get(key).filter(|v| !v.is_null());
    let mut messages = system(body.get("system"))?;
    messages.extend(turns(body.get("messages"))?);
    // the model names a LoRA adapter as on chat completions; with none served any name runs the model
    let mut chat = Map::new();
    chat.insert(
        "model".into(),
        body.get("model").cloned().unwrap_or(Value::Null),
    );
    chat.insert("messages".into(), Value::Array(messages));
    if route == Route::Messages {
        let max_tokens = body.get("max_tokens").unwrap_or(&Value::Null);
        // as_i64 is None for a float, as Python refuses 5.0
        if !max_tokens.as_i64().is_some_and(|n| n >= 1) {
            return Err(format!(
                "max_tokens is required, a positive integer; got {}",
                py_value_repr(max_tokens)
            ));
        }
        chat.insert("max_tokens".into(), max_tokens.clone());
        for key in ["temperature", "top_p", "top_k"] {
            if let Some(v) = get(key) {
                chat.insert(key.into(), v.clone());
            }
        }
        translate::prefix_cache(body, &mut chat)?;
        chat.insert(
            "stream".into(),
            get("stream")
                .filter(|s| truthy(s))
                .cloned()
                .unwrap_or(Value::Bool(false)),
        );
    }
    if let Some(t) = body.get("tools").filter(|t| truthy(t)) {
        chat.insert("tools".into(), tools(t)?);
    }
    if let Some(choice) = get("tool_choice") {
        chat.extend(tool_choice(choice)?);
    }
    if let Some(thinking) = thinking(get("thinking"), body.get("max_tokens"))? {
        chat.insert("enable_thinking".into(), Value::Bool(thinking));
    }
    if keeps_all_thinking(get("context_management"))? {
        chat.insert("clear_thinking".into(), Value::Bool(false));
    }
    chat.extend(output_config(get("output_config"), adapter)?);
    let stop_sequences = match body.get("stop_sequences").filter(|s| truthy(s)) {
        None => Vec::new(),
        Some(Value::Array(items))
            if items
                .iter()
                .all(|s| s.as_str().is_some_and(|s| !s.is_empty())) =>
        {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        }
        Some(other) => {
            return Err(format!(
                "stop_sequences must be a list of non-empty strings; got {}",
                py_value_repr(other)
            ))
        }
    };
    if stop_sequences.len() > MAX_STOPS
        || stop_sequences
            .iter()
            .any(|s| s.chars().count() > MAX_STOP_CHARS)
    {
        return Err(format!("stop_sequences takes at most {MAX_STOPS} strings of at most {MAX_STOP_CHARS} characters"));
    }
    let chat = serde_json::from_value(Value::Object(chat)).map_err(|e| e.to_string())?;
    Ok(Translated {
        chat,
        stop_sequences,
    })
}

/// `kind`'s name for a block, as a refusal quotes it: its `type`, or Python's
/// name for what the block is when it has none.
fn block_kind(block: &Value) -> String {
    match block.get("type").filter(|k| truthy(k)) {
        Some(kind) => py_value_repr(kind),
        None => crate::media::py_repr(py_type_name(block)),
    }
}

fn reads_text(kind: Option<&Value>) -> &'static str {
    match kind.and_then(Value::as_str) {
        Some("image" | "document") => " (the model reads text)",
        _ => "",
    }
}

/// The text of a string or a list of text blocks, a block a line.
fn text(blocks: &Value, place: &str) -> Result<String, String> {
    let items = match blocks {
        Value::String(s) => return Ok(s.clone()),
        Value::Array(items) => items,
        _ => {
            return Err(format!(
                "{place} must be a string or a list of content blocks"
            ))
        }
    };
    let mut texts = Vec::new();
    for block in items {
        let kind = block.get("type");
        match (kind.and_then(Value::as_str), block.get("text")) {
            (Some("text"), Some(Value::String(t))) => texts.push(t.as_str()),
            _ => {
                return Err(format!(
                    "{place} takes text blocks only; got a {} block{}",
                    block_kind(block),
                    reads_text(kind)
                ))
            }
        }
    }
    Ok(texts.join("\n"))
}

fn system(system: Option<&Value>) -> Result<Vec<Value>, String> {
    let Some(system) = system.filter(|s| !s.is_null()) else {
        return Ok(Vec::new());
    };
    let text = text(system, "system")?;
    Ok(if text.is_empty() {
        Vec::new()
    } else {
        vec![json!({"role": "system", "content": text})]
    })
}

fn turns(messages: Option<&Value>) -> Result<Vec<Value>, String> {
    let Some(Value::Array(messages)) = messages.filter(|m| truthy(m)) else {
        return Err("messages must be a non-empty list".to_string());
    };
    let mut out = Vec::new();
    for m in messages {
        let role = m.get("role").unwrap_or(&Value::Null);
        let content = m.get("content").unwrap_or(&Value::Null);
        match role.as_str() {
            Some("system") => {
                out.extend(system(Some(content))?);
                continue;
            }
            Some("user" | "assistant") => {}
            _ => {
                return Err(format!(
                    "a message's role must be user, assistant or system, not {}",
                    py_value_repr(role)
                ))
            }
        }
        let blocks = match content {
            Value::String(s) => vec![json!({"type": "text", "text": s})],
            Value::Array(items) => items.clone(),
            _ => {
                return Err(format!(
                    "a message's content must be a string or a list of blocks; got {}",
                    py_value_repr(content)
                ))
            }
        };
        if role == "user" {
            out.extend(user(&blocks)?);
        } else {
            out.push(assistant(&blocks)?);
        }
    }
    if messages.last().and_then(|m| m.get("role")) == Some(&Value::from("assistant")) {
        return Err(
            "a final assistant message (prefill) is not supported; end with a user turn"
                .to_string(),
        );
    }
    Ok(out)
}

/// A user turn: each tool result is a tool turn, text between them a user turn.
fn user(blocks: &[Value]) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    for block in blocks {
        let kind = block.get("type");
        match kind.and_then(Value::as_str) {
            Some("text") => texts.push(text(&Value::Array(vec![block.clone()]), "a user message")?),
            Some("tool_result") => {
                if !texts.is_empty() {
                    out.push(json!({"role": "user", "content": texts.join("\n")}));
                    texts.clear();
                }
                let Some(id) = block.get("tool_use_id").and_then(Value::as_str) else {
                    return Err("a tool_result block needs its tool_use_id".to_string());
                };
                out.push(
                    json!({"role": "tool", "tool_call_id": id, "content": tool_result(block)?}),
                );
            }
            _ => {
                let kind_repr = py_value_repr(kind.unwrap_or(&Value::Null));
                return Err(format!(
                    "a user message takes text and tool_result blocks; got {kind_repr}{}",
                    reads_text(kind)
                ));
            }
        }
    }
    if !texts.is_empty() {
        out.push(json!({"role": "user", "content": texts.join("\n")}));
    }
    Ok(out)
}

/// A tool result's content for the template: its text, or, when it is only tool
/// references (a tool search's answer), the references by name, which the
/// template expands into the tools' definitions.
fn tool_result(block: &Value) -> Result<Value, String> {
    let mut content = block
        .get("content")
        .filter(|c| truthy(c))
        .cloned()
        .unwrap_or(Value::from(""));
    if let Value::Array(items) = &content {
        let is_ref = |b: &Value| b.get("type") == Some(&Value::from("tool_reference"));
        let refs: Vec<&Value> = items.iter().filter(|b| is_ref(b)).collect();
        for r in &refs {
            if !r.get("tool_name").is_some_and(Value::is_string) {
                return Err(format!(
                    "a tool_reference block needs its tool_name; got {}",
                    py_value_repr(r)
                ));
            }
        }
        if !refs.is_empty() && refs.len() == items.len() {
            return Ok(Value::Array(
                refs.iter()
                    .map(|r| json!({"type": "tool_reference", "name": r["tool_name"]}))
                    .collect(),
            ));
        }
        content = Value::Array(
            items
                .iter()
                .map(|b| {
                    if is_ref(b) {
                        json!({"type": "text", "text": format!("Tool loaded: {}", b["tool_name"].as_str().unwrap_or_default())})
                    } else {
                        b.clone()
                    }
                })
                .collect(),
        );
    }
    let result = text(&content, "a tool_result's content")?;
    // the template has no flag for it
    Ok(Value::from(if block.get("is_error").is_some_and(truthy) {
        format!("Error: {result}")
    } else {
        result
    }))
}

fn assistant(blocks: &[Value]) -> Result<Value, String> {
    let (mut texts, mut thinking, mut calls) = (Vec::new(), Vec::new(), Vec::new());
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => texts.push(text(
                &Value::Array(vec![block.clone()]),
                "an assistant message",
            )?),
            Some("thinking") => match block.get("thinking") {
                Some(Value::String(t)) => thinking.push(t.clone()),
                _ => return Err("a thinking block needs its thinking text".to_string()),
            },
            Some("tool_use") => {
                let (Some(id), Some(name)) = (
                    block.get("id").and_then(Value::as_str),
                    block.get("name").and_then(Value::as_str),
                ) else {
                    return Err("a tool_use block needs its id and name".to_string());
                };
                let input = block.get("input").unwrap_or(&Value::Null);
                if !input.is_object() {
                    return Err(format!(
                        "a tool_use block's input must be an object; got {}",
                        py_value_repr(input)
                    ));
                }
                calls.push(json!({"id": id, "type": "function", "function": {"name": name, "arguments": input}}));
            }
            _ => {
                let kind = py_value_repr(block.get("type").unwrap_or(&Value::Null));
                return Err(format!(
                    "an assistant message takes text, thinking and tool_use blocks; got {kind}"
                ));
            }
        }
    }
    let joined = texts.join("\n");
    let mut message = json!({"role": "assistant", "content": if joined.is_empty() { Value::Null } else { Value::from(joined) }});
    if !thinking.is_empty() {
        message["reasoning_content"] = Value::from(thinking.join("\n"));
    }
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls);
    }
    Ok(message)
}

fn tools(tools: &Value) -> Result<Value, String> {
    let Value::Array(tools) = tools else {
        return Err("tools must be a list".to_string());
    };
    let mut out = Vec::new();
    for tool in tools {
        let custom = tool
            .as_object()
            .is_some_and(|t| t.get("type").is_none_or(|k| k == "custom"));
        if !custom {
            let kind = if tool.is_object() {
                tool.get("type").unwrap_or(&Value::Null)
            } else {
                tool
            };
            return Err(format!(
                "tools take client tools with an input_schema; {} is a server tool",
                py_value_repr(kind)
            ));
        }
        let (Some(name), Some(schema)) = (
            tool.get("name").filter(|n| n.is_string()),
            tool.get("input_schema").filter(|s| s.is_object()),
        ) else {
            return Err(format!(
                "a tool needs a name and an input_schema object; got {}",
                py_value_repr(tool)
            ));
        };
        let mut function = json!({"name": name, "parameters": schema});
        if let Some(d) = tool.get("description").filter(|d| truthy(d)) {
            function["description"] = d.clone();
        }
        if tool.get("defer_loading").is_some_and(truthy) {
            function["defer_loading"] = Value::Bool(true); // listed once a tool_reference loads it
        }
        out.push(json!({"type": "function", "function": function}));
    }
    Ok(Value::Array(out))
}

fn tool_choice(choice: &Value) -> Result<Map<String, Value>, String> {
    let kind = choice.get("type").and_then(Value::as_str);
    let chat = match kind {
        Some("auto") => Value::from("auto"),
        Some("any") => Value::from("required"),
        Some("none") => Value::from("none"),
        Some("tool") => match choice.get("name") {
            Some(Value::String(name)) => json!({"type": "function", "function": {"name": name}}),
            _ => return Err("tool_choice of type tool needs a name".to_string()),
        },
        _ => {
            return Err(format!(
                "tool_choice takes type auto, any, tool or none; got {}",
                py_value_repr(choice)
            ))
        }
    };
    let mut out = Map::new();
    out.insert("tool_choice".into(), chat);
    if choice.get("disable_parallel_tool_use").is_some_and(truthy) {
        out.insert("parallel_tool_calls".into(), Value::Bool(false));
    }
    Ok(out)
}

/// `context_management`: whether it asks to keep every thinking block. Other
/// edits would take content out of the prompt, which is not served.
fn keeps_all_thinking(config: Option<&Value>) -> Result<bool, String> {
    let Some(config) = config else {
        return Ok(false);
    };
    let edits = match config {
        Value::Object(c) if c.len() == 1 => c.get("edits").and_then(Value::as_array),
        _ => None,
    };
    let Some(edits) = edits else {
        return Err(format!(
            "context_management takes a list of edits; got {}",
            py_value_repr(config)
        ));
    };
    for edit in edits {
        let served = edit.as_object().is_some_and(|e| {
            e.len() == 2
                && e.get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|k| k.starts_with("clear_thinking_"))
                && e.get("keep") == Some(&Value::from("all"))
        });
        if !served {
            return Err(format!(
                "context_management edit {} is not served; only clear_thinking with keep 'all' is",
                py_value_repr(edit)
            ));
        }
    }
    Ok(!edits.is_empty())
}

fn output_config(
    config: Option<&Value>,
    adapter: &dyn OpenAIAdapter,
) -> Result<Map<String, Value>, String> {
    let Some(config) = config else {
        return Ok(Map::new());
    };
    let Value::Object(c) = config else {
        return Err(format!(
            "output_config must be an object; got {}",
            py_value_repr(config)
        ));
    };
    if c.contains_key("format") {
        return Err("output_config.format (structured output) needs constrained decoding, which is not served".to_string());
    }
    if c.keys().any(|k| k != "effort") {
        let mut keys: Vec<Value> = c.keys().map(|k| Value::from(k.as_str())).collect();
        keys.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
        return Err(format!(
            "output_config takes effort only; got {}",
            py_value_repr(&Value::Array(keys))
        ));
    }
    let Some(effort) = c.get("effort").filter(|e| !e.is_null()) else {
        return Ok(Map::new());
    };
    let effort_level = match effort.as_str() {
        Some("low") => Effort::Low,
        Some("medium") => Effort::Medium,
        Some("high") => Effort::High,
        Some("xhigh" | "max") => Effort::Top,
        _ => {
            return Err(format!(
                "output_config.effort takes low, medium, high, xhigh or max; got {}",
                py_value_repr(effort)
            ))
        }
    };
    translate::template_level(
        effort_level,
        adapter,
        &format!("output_config.effort={}", py_value_repr(effort)),
    )
}

fn thinking(thinking: Option<&Value>, max_tokens: Option<&Value>) -> Result<Option<bool>, String> {
    let Some(thinking) = thinking else {
        return Ok(None);
    };
    match thinking.get("type").and_then(Value::as_str) {
        Some("disabled") => return Ok(Some(false)),
        Some("adaptive") => return Ok(Some(true)),
        Some("enabled") => {}
        _ => {
            return Err(format!(
                "thinking takes type enabled, adaptive or disabled; got {}",
                py_value_repr(thinking)
            ))
        }
    }
    let budget = thinking.get("budget_tokens").unwrap_or(&Value::Null);
    let Some(n) = budget.as_i64().filter(|&n| n >= 1024) else {
        return Err(format!(
            "thinking.budget_tokens must be an integer of at least 1024; got {}",
            py_value_repr(budget)
        ));
    };
    if let Some(max) = max_tokens.and_then(Value::as_i64) {
        if n >= max {
            return Err(format!(
                "thinking.budget_tokens ({n}) must be less than max_tokens ({max})"
            ));
        }
    }
    // a target, not a bound: the model's reasoning is bounded by max_tokens only
    Ok(Some(true))
}
