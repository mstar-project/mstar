//! The names an OpenAI request's `model` field may carry, as
//! `api_server/served_models.py`'s `ServedModels` reads them: the base model's
//! name (or its checkpoint repo) runs the base model, and with LoRA on (a
//! `resources: lora:` block in the serving config) an adapter's name runs that
//! adapter. The launcher passes the lists in `MSTAR_SERVED_MODELS`.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::adapters::py_value_repr;

/// `lora_adapter`, the model kwarg a request runs under an adapter by
/// (`LORA_ADAPTER_KWARG`).
const LORA_ADAPTER_KWARG: &str = "lora_adapter";

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ServedModels {
    /// Names that select the base model: the config's model name and its checkpoint repo.
    pub base: Vec<String>,
    /// The served adapters, in the config's order (`/v1/models` lists them so).
    #[serde(default)]
    pub adapters: Vec<String>,
    #[serde(default)]
    pub lora: bool,
}

impl ServedModels {
    /// `MSTAR_SERVED_MODELS`, or the base model alone (no LoRA) when unset.
    pub fn from_env(model_name: &str) -> Self {
        match std::env::var("MSTAR_SERVED_MODELS").ok().filter(|s| !s.is_empty()) {
            Some(json) => serde_json::from_str(&json).expect("MSTAR_SERVED_MODELS is a served-models object"),
            None => ServedModels { base: vec![model_name.to_string()], adapters: Vec::new(), lora: false },
        }
    }

    fn lora_on(&self) -> bool {
        self.lora || !self.adapters.is_empty()
    }

    /// `adapter_for`: the adapter a request's `model` names; `None` runs the base
    /// model. Without LoRA the field isn't read; with it, any other name that is
    /// not a served adapter is refused.
    fn adapter_for(&self, requested: Option<&str>) -> Result<Option<String>, String> {
        let Some(requested) = requested.filter(|r| !r.is_empty()) else {
            return Ok(None);
        };
        if !self.lora_on() || self.base.iter().any(|b| b == requested) {
            return Ok(None);
        }
        self.check(&Value::from(requested))?;
        Ok(Some(requested.to_string()))
    }

    /// `check`: refuse an adapter name that is not served.
    fn check(&self, adapter: &Value) -> Result<(), String> {
        if adapter.as_str().is_some_and(|a| self.adapters.iter().any(|s| s == a)) {
            return Ok(());
        }
        let mut served = self.adapters.clone();
        served.sort();
        let served = Value::from(served);
        Err(format!(
            "unknown LoRA adapter {}; this deployment serves {}",
            py_value_repr(adapter),
            py_value_repr(&served)
        ))
    }

    /// `select`: put the adapter the request runs under in `model_kwargs`, the one
    /// its `model` names or its `lora_adapter` kwarg, each checked; both, naming
    /// different adapters, is refused. Returns that adapter, `None` for the base
    /// model. A refusal is the 400's text.
    pub fn select(&self, requested: Option<&str>, model_kwargs: &mut Map<String, Value>) -> Result<Option<String>, String> {
        let adapter = self.adapter_for(requested)?;
        let explicit = model_kwargs.get(LORA_ADAPTER_KWARG).filter(|v| !v.is_null()).cloned();
        let Some(explicit) = explicit else {
            if let Some(adapter) = &adapter {
                model_kwargs.insert(LORA_ADAPTER_KWARG.to_string(), Value::from(adapter.clone()));
            }
            return Ok(adapter);
        };
        self.check(&explicit)?;
        if let Some(adapter) = &adapter {
            if explicit.as_str() != Some(adapter) {
                return Err(format!(
                    "model {} and lora_adapter {} name different adapters",
                    py_value_repr(&Value::from(requested.unwrap_or_default())),
                    py_value_repr(&explicit)
                ));
            }
        }
        Ok(explicit.as_str().map(str::to_string))
    }

    /// `/v1/models`' cards: the base model, then each adapter naming it as its parent.
    pub fn cards(&self, model_name: &str, created: i64) -> Value {
        let mut data = vec![serde_json::json!({"id": model_name, "object": "model", "created": created, "owned_by": "mstar"})];
        data.extend(self.adapters.iter().map(|name| {
            serde_json::json!({"id": name, "object": "model", "created": created, "owned_by": "mstar", "parent": model_name})
        }));
        serde_json::json!({"object": "list", "data": data})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn served() -> ServedModels {
        ServedModels { base: vec!["qwen".into(), "Qwen/Qwen3.5-0.8B".into()], adapters: vec!["sql".into(), "chat".into()], lora: true }
    }

    #[test]
    fn a_model_name_selects_its_adapter_as_python_does() {
        let mut mk = Map::new();
        assert_eq!(served().select(Some("sql"), &mut mk), Ok(Some("sql".into())));
        assert_eq!(mk["lora_adapter"], "sql");
        for base in [None, Some(""), Some("qwen"), Some("Qwen/Qwen3.5-0.8B")] {
            let mut mk = Map::new();
            assert_eq!(served().select(base, &mut mk), Ok(None), "{base:?}");
            assert!(mk.is_empty());
        }
        let mut mk = Map::new();
        assert_eq!(
            served().select(Some("nope"), &mut mk),
            Err("unknown LoRA adapter 'nope'; this deployment serves ['chat', 'sql']".to_string())
        );
        let mut mk: Map<String, Value> = serde_json::from_str(r#"{"lora_adapter": "chat"}"#).unwrap();
        assert_eq!(
            served().select(Some("sql"), &mut mk),
            Err("model 'sql' and lora_adapter 'chat' name different adapters".to_string())
        );
        let mut mk: Map<String, Value> = serde_json::from_str(r#"{"lora_adapter": 5}"#).unwrap();
        assert_eq!(
            served().select(None, &mut mk),
            Err("unknown LoRA adapter 5; this deployment serves ['chat', 'sql']".to_string())
        );
        // without LoRA, `model` is not read
        let mut mk = Map::new();
        assert_eq!(ServedModels { base: vec!["qwen".into()], ..Default::default() }.select(Some("x"), &mut mk), Ok(None));
    }
}
