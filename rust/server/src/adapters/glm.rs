//! GLM-5.2 / GLM-5.3 chat: the port of `GlmChatAdapter` and the request side of
//! `glm_chat.py`. The messages with their roles, and `tools`, go to the
//! checkpoint's chat template, which the model's `process_prompt` renders
//! behind the bridge; the reply comes back through [`glm_reply::ReplyParser`].
//!
//! `tool_choice` is `"auto"` (default), `"none"` (the tools stay out of the
//! prompt), `"required"` or a named function. With no grammar, the prompt
//! forces the call: it ends in `<tool_call>` and the name, which `"required"`
//! over several tools leaves to the model (a name it makes up stays text). The
//! call opens the reply, so a forced call does not reason, and
//! `enable_thinking` with it is a 400. `parallel_tool_calls: false` ends the
//! request at the end of its first call.

use std::sync::OnceLock;

use serde::Deserialize;

use super::glm_reply::{self, ReplyParser, Tools};
use super::*;
use crate::generation_kwargs as gk;
use crate::reply::OutputParser;

/// What the launcher knows of the model's chat template, as
/// `MSTAR_CHAT_TEMPLATE` (`chat_template_facts`).
#[derive(Debug, Default, Deserialize)]
pub struct TemplateFacts {
    /// GLM-5.3's template has no thinking switch: it always reasons.
    #[serde(default)]
    pub always_thinks: bool,
    /// The `chat_template_kwargs` keys the template reads (`GLM_TEMPLATE_KWARGS`).
    #[serde(default)]
    pub template_kwargs: Vec<String>,
}

pub fn template_facts() -> &'static TemplateFacts {
    static FACTS: OnceLock<TemplateFacts> = OnceLock::new();
    FACTS.get_or_init(|| match std::env::var("MSTAR_CHAT_TEMPLATE") {
        Ok(s) if !s.is_empty() => {
            serde_json::from_str(&s).expect("MSTAR_CHAT_TEMPLATE is the launcher's JSON")
        }
        _ => TemplateFacts::default(),
    })
}

/// GLM-5.2 and GLM-5.3: chat with roles, tools and reasoning.
pub struct GlmChat;

/// `GlmChatAdapter._options`: what a request's tool and template fields ask for.
struct Options {
    tools: Option<Value>,
    template_kwargs: Map<String, Value>,
    /// The call a forced `tool_choice` opens the reply with: the function's
    /// name, or `""` for the model to name one.
    forced: Option<String>,
}

impl OpenAIAdapter for GlmChat {
    fn supports_chat(&self) -> bool {
        true
    }

    fn serves_tool_calls(&self) -> bool {
        true
    }

    fn serves_stop_strings(&self) -> bool {
        true
    }

    fn chat_args(
        &self,
        req: &ChatCompletionRequest,
        upload_dir: &Path,
        _allow_remote: bool,
    ) -> Result<SubmitArgs, String> {
        self.stop_strings(req)?;
        // flatten_messages refuses a tool turn; the template takes each turn's role
        // from template_messages below, so the base checks read a tool result as user text
        let mut checked = req.clone();
        for m in checked.messages.iter_mut().filter(|m| m.role == "tool") {
            m.role = "user".to_string();
        }
        let mut args = text_only_chat(&checked, upload_dir)?;
        // as TextChatAdapter normalizes, before the template switches leave the
        // top level: a flat `enable_thinking: 0` is refused there
        let kwargs = args
            .model_kwargs
            .into_iter()
            .map(|(k, v)| (k, gk::Kwarg::Json(v)))
            .collect();
        let mut mk = gk::into_map(gk::normalize(kwargs)?);
        let Options {
            tools,
            template_kwargs,
            forced,
        } = options(&mut mk)?;
        let facts = template_facts();
        let thinking = template_kwargs.get("enable_thinking");
        if facts.always_thinks && thinking.is_some_and(|t| !truthy(t)) {
            // the template has no switch, so off would still reason
            return Err(
                "this model always reasons; enable_thinking cannot turn it off".to_string(),
            );
        }
        mk.insert("messages".to_string(), template_messages(&req.messages)?);
        if let Some(tools) = tools {
            mk.insert("tools".to_string(), tools);
        }
        if let Some(name) = forced {
            // GLM-5.3's prompt ends inside <think>: an empty reasoning, as the template
            // renders a turn without one
            let think = if facts.always_thinks {
                glm_reply::THINK_END
            } else {
                ""
            };
            mk.insert(
                "reply_prefix".to_string(),
                Value::from(format!("{think}{}{name}", glm_reply::CALL_START)),
            );
        }
        mk.insert(
            "chat_template_kwargs".to_string(),
            Value::Object(template_kwargs),
        );
        args.model_kwargs = mk;
        Ok(args)
    }

    /// `stop`, as OpenAI takes it: a string or up to 4. The chat route matches
    /// them on the decoded answer and ends the request at the first.
    fn stop_strings(&self, req: &ChatCompletionRequest) -> Result<Vec<String>, String> {
        let stops: Vec<String> = match &req.stop {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(items)) => items
                .iter()
                .map(|s| s.as_str().map(str::to_string))
                .collect::<Option<_>>()
                .ok_or("stop must be a string or a list of strings")?,
            Some(_) => return Err("stop must be a string or a list of strings".to_string()),
        };
        if stops.len() > 4 {
            return Err(format!("stop takes at most 4 strings, got {}", stops.len()));
        }
        if stops.iter().any(String::is_empty) {
            return Err("a stop string cannot be empty".to_string());
        }
        Ok(stops)
    }

    fn output_parser(
        &self,
        req: &ChatCompletionRequest,
    ) -> Result<Option<Box<dyn OutputParser>>, String> {
        // normalized as chat_args reads them, so the parser and the prompt agree
        // (a null switch is unset in both)
        let kwargs = req
            .extra
            .clone()
            .into_iter()
            .map(|(k, v)| (k, gk::Kwarg::Json(v)))
            .collect();
        let mut mk = gk::into_map(gk::normalize(kwargs)?);
        let single_call = mk.get("parallel_tool_calls") == Some(&Value::Bool(false));
        let Options {
            tools,
            template_kwargs,
            forced,
        } = options(&mut mk)?;
        // truthiness, as the GLM-5.2 template reads it
        let thinking = template_facts().always_thinks
            || template_kwargs.get("enable_thinking").is_none_or(truthy);
        let params = tools
            .as_ref()
            .map(tool_parameters)
            .transpose()?
            .map(|o| o.params);
        Ok(Some(Box::new(ReplyParser::new(
            thinking,
            params,
            forced,
            single_call,
        ))))
    }
}

/// Takes `tools`, `tool_choice` and the template's keys out of `mk`.
fn options(mk: &mut Map<String, Value>) -> Result<Options, String> {
    if mk.contains_key("reply_prefix") {
        return Err("reply_prefix is not a request field; tool_choice sets it".to_string());
    }
    // false ends the reply at its first call; read, not sent on
    match mk.shift_remove("parallel_tool_calls") {
        None | Some(Value::Null) | Some(Value::Bool(_)) => {}
        Some(v) => {
            return Err(format!(
                "parallel_tool_calls must be a boolean, got {}",
                py_value_repr(&v)
            ))
        }
    }
    let mut tools = mk.shift_remove("tools").filter(truthy);
    let choice = mk.shift_remove("tool_choice").unwrap_or(Value::Null);
    let names = match &tools {
        Some(tools) => tool_parameters(tools)?.names,
        None => Vec::new(),
    };
    let facts = template_facts();
    let reads = |key: &str| facts.template_kwargs.iter().any(|k| k == key);
    let mut template_kwargs = match mk.shift_remove("chat_template_kwargs").filter(truthy) {
        None => Map::new(),
        Some(Value::Object(m)) => m.into_iter().filter(|(k, _)| reads(k)).collect(),
        Some(_) => return Err("chat_template_kwargs must be an object".to_string()),
    };
    for key in &facts.template_kwargs {
        if let Some(v) = mk.shift_remove(key.as_str()) {
            template_kwargs.entry(key.clone()).or_insert(v);
        }
    }
    let mut forced = None;
    let required = choice == "required";
    if choice == "none" {
        tools = None;
    } else if required || choice.is_object() {
        if names.is_empty() {
            return Err(format!(
                "tool_choice {} needs tools",
                py_value_repr(&choice)
            ));
        }
        let name = if required {
            if names.len() == 1 {
                names[0].clone()
            } else {
                String::new()
            }
        } else {
            let function = choice
                .get("function")
                .filter(|f| f.is_object() && choice.get("type") == Some(&Value::from("function")));
            match function.and_then(|f| f.get("name")).and_then(Value::as_str) {
                Some(name) if names.iter().any(|n| n == name) => name.to_string(),
                _ => {
                    let listed =
                        Value::Array(names.iter().map(|n| Value::from(n.as_str())).collect());
                    return Err(format!(
                        "tool_choice {} names no function in tools {}",
                        py_value_repr(&choice),
                        py_value_repr(&listed)
                    ));
                }
            }
        };
        if template_kwargs.get("enable_thinking").is_some_and(truthy) {
            return Err(
                "a forced tool call opens the reply, which leaves no room to reason; \
                        leave enable_thinking unset, or use tool_choice 'auto'"
                    .to_string(),
            );
        }
        if !facts.always_thinks {
            // the prompt ends in <think></think>, as a turn that does not reason
            template_kwargs.insert("enable_thinking".to_string(), Value::Bool(false));
        }
        forced = Some(name);
    } else if !(choice.is_null() || choice == "auto") {
        return Err(format!(
            "tool_choice takes 'auto', 'none', 'required' or {{'type': 'function', 'function': {{'name': ...}}}}; got {}",
            py_value_repr(&choice)
        ));
    }
    Ok(Options {
        tools,
        template_kwargs,
        forced,
    })
}

/// The function tools a request offers (`glm_chat.tool_parameters`).
pub struct Offered {
    /// Each one's parameter properties, by name.
    pub params: Tools,
    /// Their names, in the order the request first gives each.
    pub names: Vec<String>,
}

pub fn tool_parameters(tools: &Value) -> Result<Offered, String> {
    let Value::Array(tools) = tools else {
        return Err(format!("tools must be a list, got {}", py_type_name(tools)));
    };
    let (mut params, mut names) = (Tools::new(), Vec::new());
    for tool in tools {
        let function = tool
            .as_object()
            .and_then(|t| t.get("function"))
            .and_then(Value::as_object);
        let typed = tool
            .as_object()
            .and_then(|t| t.get("type"))
            .is_none_or(|t| t == "function");
        let Some((fn_, name)) = function
            .filter(|_| typed)
            .and_then(|f| Some((f, f.get("name")?.as_str()?)))
            .filter(|(_, n)| !n.is_empty())
        else {
            return Err(format!(
                "tools take {{'type': 'function', 'function': {{'name': ...}}}}, got {}",
                py_value_repr(tool)
            ));
        };
        let properties = match fn_.get("parameters").filter(|p| truthy(p)) {
            None => Some(Map::new()),
            Some(Value::Object(p)) => match p.get("properties").filter(|v| truthy(v)) {
                None => Some(Map::new()),
                Some(Value::Object(props)) => Some(props.clone()),
                Some(_) => None,
            },
            Some(_) => None,
        };
        let Some(properties) = properties else {
            return Err(format!(
                "tool {}: parameters must be a JSON schema object with object properties",
                crate::media::py_repr(name)
            ));
        };
        if !params.contains_key(name) {
            names.push(name.to_string());
        }
        params.insert(name.to_string(), properties);
    }
    Ok(Offered { params, names })
}

/// The messages as the templates read them: a call's `arguments` as the
/// object the template iterates (clients send a JSON string), no null fields
/// (GLM-5.2's template prints `None`), and `developer` turns as `system`.
pub fn template_messages(messages: &[crate::protocol::ChatMessage]) -> Result<Value, String> {
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let role = if m.role == "developer" {
            "system"
        } else {
            m.role.as_str()
        };
        if !matches!(role, "system" | "user" | "assistant" | "tool") {
            // the templates skip any other role: the model would answer without that turn
            return Err(format!(
                "role {} is not supported; use system, user, assistant or tool",
                crate::media::py_repr(&m.role)
            ));
        }
        let mut msg = Map::new();
        msg.insert("role".to_string(), Value::from(role));
        match &m.content {
            None => {}
            Some(Content::Text(s)) => {
                msg.insert("content".to_string(), Value::from(s.as_str()));
            }
            Some(Content::Parts(parts)) => {
                msg.insert("content".to_string(), Value::Array(parts.clone()));
            }
        }
        for (k, v) in m.extra.iter().filter(|(_, v)| !v.is_null()) {
            msg.insert(k.clone(), v.clone());
        }
        if let Some(calls) = msg.get("tool_calls").filter(|c| truthy(c)) {
            let calls = match calls {
                Value::Array(items) => items.clone(),
                // iterated as Python iterates it: a dict's keys
                Value::Object(m) => m.keys().map(|k| Value::from(k.as_str())).collect(),
                other => vec![other.clone()],
            };
            let calls = calls
                .iter()
                .map(call_for_template)
                .collect::<Result<Vec<_>, _>>()?;
            msg.insert("tool_calls".to_string(), Value::Array(calls));
        }
        out.push(Value::Object(msg));
    }
    Ok(Value::Array(out))
}

fn call_for_template(call: &Value) -> Result<Value, String> {
    let (Some(obj), Some(Value::Object(function))) = (call.as_object(), call.get("function"))
    else {
        return Err(format!(
            "tool_calls entries need a function object, got {}",
            py_value_repr(call)
        ));
    };
    if function
        .get("name")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        return Err(format!(
            "tool_calls entries need a function name, got {}",
            py_value_repr(call)
        ));
    }
    let mut function = function.clone();
    let arguments = match function.get("arguments") {
        Some(Value::String(s)) if py_strip(s).is_empty() => Value::Object(Map::new()),
        Some(Value::String(s)) => serde_json::from_str(s).map_err(|_| {
            format!(
                "tool call arguments are not JSON: {}",
                crate::media::py_repr(s)
            )
        })?,
        Some(v) => v.clone(),
        None => Value::Null,
    };
    if !arguments.is_object() {
        return Err(format!(
            "tool call arguments must be a JSON object, got {}",
            py_value_repr(&arguments)
        ));
    }
    function.insert("arguments".to_string(), arguments);
    let mut obj = obj.clone();
    obj.insert("function".to_string(), Value::Object(function));
    Ok(Value::Object(obj))
}
