//! The names an OpenAI request's `model` field may carry, as
//! `api_server/served_models.py`'s `ServedModels` reads them: the base model's
//! name (or its checkpoint repo) runs the base model, and with LoRA on (a
//! `resources: lora:` block in the serving config) an adapter's name runs that
//! adapter. The launcher passes the lists in `MSTAR_SERVED_MODELS`; the API
//! server pushes the adapters over the bridge after every runtime load or
//! unload (`ServedModels.subscribe`), so selection needs no round trip.

use std::sync::RwLock;

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::adapters::py_value_repr;

/// `lora_adapter`, the model kwarg a request runs under an adapter by
/// (`LORA_ADAPTER_KWARG`).
const LORA_ADAPTER_KWARG: &str = "lora_adapter";

#[derive(Deserialize)]
struct Launched {
    base: Vec<String>,
    #[serde(default)]
    adapters: Vec<String>,
    #[serde(default)]
    lora: bool,
}

#[derive(Debug, Default)]
pub struct ServedModels {
    /// Names that select the base model: the config's model name and its checkpoint repo.
    base: Vec<String>,
    /// The served adapters, in the order the API server holds them (`/v1/models`
    /// lists them so): the config's, then runtime loads.
    adapters: RwLock<Vec<String>>,
    /// LoRA on (a `lora:` block, or adapters at launch), as `ServedModels.lora`.
    lora: bool,
}

/// One request's view of the served adapters.
struct Snapshot<'a> {
    served: &'a ServedModels,
    adapters: Vec<String>,
}

impl ServedModels {
    pub fn new(base: Vec<String>, adapters: Vec<String>, lora: bool) -> Self {
        let lora = lora || !adapters.is_empty();
        ServedModels { base, adapters: RwLock::new(adapters), lora }
    }

    /// `MSTAR_SERVED_MODELS`, or the base model alone (no LoRA) when unset.
    pub fn from_env(model_name: &str) -> Self {
        match std::env::var("MSTAR_SERVED_MODELS").ok().filter(|s| !s.is_empty()) {
            Some(json) => {
                let l: Launched = serde_json::from_str(&json).expect("MSTAR_SERVED_MODELS is a served-models object");
                ServedModels::new(l.base, l.adapters, l.lora)
            }
            None => ServedModels::new(vec![model_name.to_string()], Vec::new(), false),
        }
    }

    /// The API server's served adapters after a change (the bridge's `served` push).
    pub fn set_adapters(&self, adapters: Vec<String>) {
        *self.adapters.write().unwrap() = adapters;
    }

    fn snapshot(&self) -> Snapshot<'_> {
        Snapshot { served: self, adapters: self.adapters.read().unwrap().clone() }
    }

    /// `select`: put the adapter the request runs under in `model_kwargs`, the one
    /// its `model` names or its `lora_adapter` kwarg, each checked; both, naming
    /// different adapters, is refused. Returns that adapter, `None` for the base
    /// model. A refusal is the 400's text.
    pub fn select(&self, requested: Option<&str>, model_kwargs: &mut Map<String, Value>) -> Result<Option<String>, String> {
        self.snapshot().select(requested, model_kwargs)
    }

    /// `/v1/models`' cards: the base model, then each adapter naming it as its parent.
    pub fn cards(&self, model_name: &str, created: i64) -> Value {
        let mut data = vec![serde_json::json!({"id": model_name, "object": "model", "created": created, "owned_by": "mstar"})];
        data.extend(self.snapshot().adapters.iter().map(|name| {
            serde_json::json!({"id": name, "object": "model", "created": created, "owned_by": "mstar", "parent": model_name})
        }));
        serde_json::json!({"object": "list", "data": data})
    }
}

impl Snapshot<'_> {
    fn lora_on(&self) -> bool {
        self.served.lora
    }

    /// `adapter_for`: the adapter a request's `model` names; `None` runs the base
    /// model. Without LoRA the field isn't read; with it, any other name that is
    /// not a served adapter is refused.
    fn adapter_for(&self, requested: Option<&str>) -> Result<Option<String>, String> {
        let Some(requested) = requested.filter(|r| !r.is_empty()) else {
            return Ok(None);
        };
        if !self.lora_on() || self.served.base.iter().any(|b| b == requested) {
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

    fn select(&self, requested: Option<&str>, model_kwargs: &mut Map<String, Value>) -> Result<Option<String>, String> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn served() -> ServedModels {
        ServedModels::new(vec!["qwen".into(), "Qwen/Qwen3.5-0.8B".into()], vec!["sql".into(), "chat".into()], true)
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
        assert_eq!(ServedModels::new(vec!["qwen".into()], Vec::new(), false).select(Some("x"), &mut mk), Ok(None));
    }

    #[test]
    fn a_pushed_set_is_what_requests_select_from() {
        let served = served();
        served.set_adapters(vec!["chat".into(), "new".into()]);
        let mut mk = Map::new();
        assert_eq!(served.select(Some("new"), &mut mk), Ok(Some("new".into())));
        assert_eq!(
            served.select(Some("sql"), &mut Map::new()),
            Err("unknown LoRA adapter 'sql'; this deployment serves ['chat', 'new']".to_string())
        );
        assert_eq!(served.cards("qwen", 0)["data"].as_array().unwrap().len(), 3);
    }
}
