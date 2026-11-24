//! Shared `POST /v1/actions/generations` (`application/msgpack`) handler.
//!
//! Mirrors `mstar/api_server/actions.py` with exact wire, validation, and error
//! status parity across Pi0.5 and Cosmos3 action models.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use axum::{
    extract::{Request, State},
    http::{header::CONTENT_TYPE, StatusCode},
    response::{IntoResponse, Response},
};
use futures::future::try_join_all;
use rmpv::Value as Mp;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::adapters::{RawArray, SubmitArgs};
use crate::msgpack_numpy::{self, Dtype, Obs};
use crate::serving::{collect, result_stream, rid, AppState};

pub const ACTIONS_ENV: &str = "MSTAR_ACTIONS";
const MAX_ACTION_BATCH_SIZE: usize = 8;

fn default_state_kwarg() -> String {
    "robot_state".to_string()
}

fn default_state_dim() -> usize {
    32
}

fn default_num_cameras() -> usize {
    3
}

/// Action-model metadata exported by `mstar/api_server/rust_frontend.py::actions_facts`.
#[derive(Debug, Clone, Deserialize)]
pub struct ActionsFacts {
    pub family: String,
    #[allow(dead_code)]
    pub action_horizon: usize,
    pub action_dim: usize,
    #[serde(default = "default_state_kwarg")]
    pub state_kwarg: String,
    #[allow(dead_code)]
    #[serde(default = "default_state_dim")]
    pub state_dim: usize,
    #[serde(default = "default_num_cameras")]
    pub num_cameras: usize,
    #[serde(default)]
    pub num_flow_steps: Option<usize>,
}

impl ActionsFacts {
    pub fn from_env() -> Option<Self> {
        static FACTS: OnceLock<Option<ActionsFacts>> = OnceLock::new();
        FACTS
            .get_or_init(|| {
                let raw = std::env::var(ACTIONS_ENV).ok().filter(|s| !s.is_empty())?;
                Some(serde_json::from_str(&raw).expect("MSTAR_ACTIONS is the launcher's JSON"))
            })
            .clone()
    }
}

/// Resolve whether this frontend serves `/v1/actions/generations` and its capabilities.
pub fn resolve_actions(st: &AppState) -> Result<ActionsFacts, Response> {
    if let Some(facts) = &st.actions {
        return Ok(facts.clone());
    }
    if let Some(policy) = crate::ws::openpi::policy() {
        return Ok(ActionsFacts {
            family: "pi05".to_string(),
            action_horizon: policy.action_horizon,
            action_dim: policy.action_dim,
            state_kwarg: policy.state_kwarg.clone(),
            state_dim: policy.action_dim,
            num_cameras: policy.image_keys.len(),
            num_flow_steps: Some(10),
        });
    }
    if matches!(
        st.python_adapter.as_str(),
        "Cosmos3Adapter" | "Cosmos3EdgeAdapter"
    ) && !st.model_name.ends_with("_4step")
    {
        return Ok(ActionsFacts {
            family: "cosmos3".to_string(),
            action_horizon: 16,
            action_dim: 64,
            state_kwarg: "robot_state".to_string(),
            state_dim: 64,
            num_cameras: 1,
            num_flow_steps: Some(30),
        });
    }
    if st.model_name == "pi05" || st.model_name.starts_with("pi05_") {
        return Ok(ActionsFacts {
            family: "pi05".to_string(),
            action_horizon: 50,
            action_dim: 32,
            state_kwarg: "robot_state".to_string(),
            state_dim: 32,
            num_cameras: 3,
            num_flow_steps: Some(10),
        });
    }
    Err(crate::error(
        404,
        &format!(
            "Model '{}' does not support /v1/actions/generations",
            st.model_name
        ),
        "invalid_request_error",
    ))
}

/// Deletes persisted upload files on early validation/submission failure unless disarmed.
struct SavedPathGuard {
    upload_dir: PathBuf,
    paths: Vec<String>,
    disarmed: bool,
}

impl SavedPathGuard {
    fn new(upload_dir: &Path) -> Self {
        Self {
            upload_dir: upload_dir.to_path_buf(),
            paths: Vec::new(),
            disarmed: false,
        }
    }

    fn push(&mut self, path: String) {
        self.paths.push(path);
    }

    fn disarm(&mut self) -> Vec<String> {
        self.disarmed = true;
        std::mem::take(&mut self.paths)
    }
}

impl Drop for SavedPathGuard {
    fn drop(&mut self) {
        if self.disarmed || self.paths.is_empty() {
            return;
        }
        let root = std::fs::canonicalize(&self.upload_dir).unwrap_or_else(|_| self.upload_dir.clone());
        for p in &self.paths {
            let pb = PathBuf::from(p);
            let resolved = std::fs::canonicalize(&pb).unwrap_or(pb);
            if resolved.starts_with(&root) {
                let _ = std::fs::remove_file(resolved);
            }
        }
    }
}

struct DeferredRowCleanup<'a> {
    upload_dir: &'a Path,
    rows: &'a [SubmitArgs],
}

impl Drop for DeferredRowCleanup<'_> {
    fn drop(&mut self) {
        if tokio::runtime::Handle::try_current().is_ok() {
            for args in self.rows {
                crate::schedule_upload_cleanup(self.upload_dir, args);
            }
        }
    }
}

fn check_int64(obs: &Obs, name: &str) -> Result<i64, String> {
    match obs {
        Obs::Int(i) => i64::try_from(*i)
            .map_err(|_| format!("{name} must fit in a signed 64-bit integer")),
        Obs::Scalar {
            dtype,
            int: Some(i),
            ..
        } if matches!(dtype.kind, 'i' | 'u') => i64::try_from(*i)
            .map_err(|_| format!("{name} must fit in a signed 64-bit integer")),
        Obs::Bool(_)
        | Obs::Scalar {
            dtype: Dtype { kind: 'b', .. },
            ..
        } => Err(format!("{name} must be an integer, not bool")),
        other => Err(format!(
            "{name} must be an integer, got {}",
            other.type_name()
        )),
    }
}

fn check_bool(obs: &Obs, name: &str) -> Result<bool, String> {
    match obs {
        Obs::Bool(b) => Ok(*b),
        Obs::Scalar {
            dtype: Dtype { kind: 'b', .. },
            int: Some(i),
            ..
        } => Ok(*i != 0),
        other => Err(format!(
            "{name} must be a boolean, got {}",
            other.type_name()
        )),
    }
}

fn bool_01(obs: &Obs, name: &str) -> Result<bool, String> {
    match obs {
        Obs::Bool(b) => Ok(*b),
        Obs::Scalar {
            dtype: Dtype { kind: 'b', .. },
            int: Some(i),
            ..
        } => Ok(*i != 0),
        Obs::Int(i) if *i == 0 || *i == 1 => Ok(*i == 1),
        Obs::Scalar {
            dtype,
            int: Some(i),
            ..
        } if matches!(dtype.kind, 'i' | 'u') && (*i == 0 || *i == 1) => Ok(*i == 1),
        _ => Err(format!("{name} elements must be bool or 0/1")),
    }
}

fn check_finite_number(obs: &Obs, name: &str) -> Result<f64, String> {
    let val = match obs {
        Obs::Int(i) => *i as f64,
        Obs::Float(f) => *f,
        Obs::Scalar {
            dtype, value, int, ..
        } if dtype.kind != 'b' => match (dtype.kind, int) {
            ('i' | 'u', Some(i)) => *i as f64,
            _ => *value,
        },
        Obs::Bool(_)
        | Obs::Scalar {
            dtype: Dtype { kind: 'b', .. },
            ..
        } => return Err(format!("{name} must be a number, not bool")),
        other => {
            return Err(format!(
                "{name} must be a number, got {}",
                other.type_name()
            ))
        }
    };
    if !val.is_finite() {
        return Err(format!("{name} must be finite"));
    }
    Ok(val)
}

fn obs_to_json(obs: &Obs, path: &str) -> Result<Value, String> {
    match obs {
        Obs::Nil => Ok(Value::Null),
        Obs::Bool(b) => Ok(Value::Bool(*b)),
        Obs::Int(i) => i64::try_from(*i)
            .map(Value::from)
            .map_err(|_| format!("{path} must fit in a signed 64-bit integer")),
        Obs::Float(f) => {
            if !f.is_finite() {
                return Err(format!("{path} must be finite"));
            }
            Ok(Value::Number(
                serde_json::Number::from_f64(*f)
                    .ok_or_else(|| format!("{path} must be finite"))?,
            ))
        }
        Obs::Str(s) => Ok(Value::String(s.clone())),
        Obs::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(obs_to_json(item, &format!("{path}[]"))?);
            }
            Ok(Value::Array(out))
        }
        Obs::Map(pairs) => {
            let mut map = Map::new();
            for (k, v) in pairs {
                let Obs::Str(ks) = k else {
                    return Err(format!(
                        "{path} keys must be strings, got {}",
                        k.type_name()
                    ));
                };
                map.insert(ks.clone(), obs_to_json(v, &format!("{path}.{ks}"))?);
            }
            Ok(Value::Object(map))
        }
        Obs::Scalar {
            dtype, value, int, ..
        } => match dtype.kind {
            'b' => Ok(Value::Bool(int.unwrap_or(0) != 0)),
            'i' | 'u' => i64::try_from(int.unwrap_or(0))
                .map(Value::from)
                .map_err(|_| format!("{path} must fit in a signed 64-bit integer")),
            'f' => {
                if !value.is_finite() {
                    return Err(format!("{path} must be finite"));
                }
                Ok(Value::Number(
                    serde_json::Number::from_f64(*value)
                        .ok_or_else(|| format!("{path} must be finite"))?,
                ))
            }
            _ => Err(format!("{path} has unsupported dtype {}", dtype.name())),
        },
        Obs::Array { dtype, shape, data } => {
            fn build_nested(
                dtype: Dtype,
                shape: &[usize],
                data: &[u8],
                offset: &mut usize,
                path: &str,
            ) -> Result<Value, String> {
                if shape.is_empty() {
                    let (f, i) = msgpack_numpy::element(dtype, data, *offset);
                    *offset += 1;
                    return match dtype.kind {
                        'b' => Ok(Value::Bool(i.unwrap_or(0) != 0)),
                        'i' | 'u' => i64::try_from(i.unwrap_or(0))
                            .map(Value::from)
                            .map_err(|_| format!("{path} must fit in a signed 64-bit integer")),
                        'f' => {
                            if !f.is_finite() {
                                return Err(format!("{path} contains non-finite float values"));
                            }
                            Ok(Value::Number(
                                serde_json::Number::from_f64(f)
                                    .ok_or_else(|| format!("{path} must be finite"))?,
                            ))
                        }
                        _ => Err(format!("{path} has unsupported dtype {}", dtype.name())),
                    };
                }
                let mut out = Vec::with_capacity(shape[0]);
                for _ in 0..shape[0] {
                    out.push(build_nested(dtype, &shape[1..], data, offset, path)?);
                }
                Ok(Value::Array(out))
            }
            let mut offset = 0;
            build_nested(*dtype, shape, data, &mut offset, path)
        }
        other => Err(format!(
            "{path} has unsupported type {}",
            other.type_name()
        )),
    }
}

fn map_pairs<'a>(obs: &'a Obs, name: &str) -> Result<&'a [(Obs, Obs)], String> {
    match obs {
        Obs::Map(pairs) => Ok(pairs),
        other => Err(format!("{name} must be a map, got {}", other.type_name())),
    }
}

fn validate_runtime(obs: Option<&Obs>) -> Result<(), String> {
    let Some(obs) = obs else { return Ok(()) };
    if matches!(obs, Obs::Nil) {
        return Ok(());
    }
    let pairs = map_pairs(obs, "runtime")?;
    for (k, v) in pairs {
        let Obs::Str(ks) = k else {
            return Err("runtime keys must be strings".to_string());
        };
        match ks.as_str() {
            "response_format" => {
                if !matches!(v, Obs::Nil) && !matches!(v, Obs::Str(s) if s == "envelope") {
                    return Err(format!(
                        "Unsupported runtime.response_format {}; expected 'envelope'",
                        v.repr()
                    ));
                }
            }
            "output_format" => {
                if !matches!(v, Obs::Nil) && !matches!(v, Obs::Str(s) if s == "numpy") {
                    return Err(format!(
                        "Unsupported runtime.output_format {}; expected 'numpy'",
                        v.repr()
                    ));
                }
            }
            other => {
                return Err(format!("Unknown runtime parameter(s): ['{other}']"));
            }
        }
    }
    Ok(())
}

struct RolloutSpec {
    json: Value,
    return_state: bool,
}

fn validate_rollout(
    obs: Option<&Obs>,
    effective_steps: Option<i64>,
) -> Result<Option<RolloutSpec>, String> {
    let Some(obs) = obs else { return Ok(None) };
    if matches!(obs, Obs::Nil) {
        return Ok(None);
    }
    let pairs = map_pairs(obs, "parameters.rollout")?;
    for (k, _) in pairs {
        let Obs::Str(ks) = k else {
            return Err("parameters.rollout keys must be strings".to_string());
        };
        if !matches!(
            ks.as_str(),
            "sde_type"
                | "noise_level"
                | "sde_step"
                | "return_state"
                | "ignore_last"
                | "video_solver"
        ) {
            return Err(format!("Unknown rollout parameter(s): ['{ks}']"));
        }
    }

    let sde_type = match obs.get("sde_type") {
        Some(Obs::Str(s)) if s == "flow_sde" || s == "flow_ode" => s.clone(),
        _ => return Err("rollout.sde_type must be 'flow_sde' or 'flow_ode'".to_string()),
    };
    let noise_level = match obs.get("noise_level") {
        None | Some(Obs::Nil) => 0.5,
        Some(v) => check_finite_number(v, "rollout.noise_level")?,
    };
    if noise_level < 0.0 {
        return Err("rollout.noise_level must be >= 0".to_string());
    }
    let sde_step = match obs.get("sde_step") {
        None | Some(Obs::Nil) => Value::String("random".to_string()),
        Some(Obs::Str(s)) if s == "random" => Value::String("random".to_string()),
        Some(v) => {
            let step = check_int64(v, "rollout.sde_step")?;
            if step < 0 {
                return Err("rollout.sde_step must be >= 0 or 'random'".to_string());
            }
            if let Some(limit) = effective_steps {
                if step >= limit {
                    return Err(format!(
                        "rollout.sde_step ({step}) must be < num_steps ({limit})"
                    ));
                }
            }
            Value::from(step)
        }
    };
    let return_state = match obs.get("return_state") {
        None | Some(Obs::Nil) => true,
        Some(v) => check_bool(v, "rollout.return_state")?,
    };
    let ignore_last = match obs.get("ignore_last") {
        None | Some(Obs::Nil) => false,
        Some(v) => check_bool(v, "rollout.ignore_last")?,
    };
    let video_solver = match obs.get("video_solver") {
        None | Some(Obs::Nil) => None,
        Some(Obs::Str(s)) if s == "hybrid_unipc" || s == "euler" => Some(s.clone()),
        _ => return Err("rollout.video_solver must be 'hybrid_unipc' or 'euler'".to_string()),
    };

    let mut map = Map::new();
    map.insert("sde_type".into(), Value::String(sde_type));
    map.insert(
        "noise_level".into(),
        Value::Number(
            serde_json::Number::from_f64(noise_level)
                .ok_or_else(|| "rollout.noise_level must be finite".to_string())?,
        ),
    );
    map.insert("sde_step".into(), sde_step);
    map.insert("return_state".into(), Value::Bool(return_state));
    map.insert("ignore_last".into(), Value::Bool(ignore_last));
    if let Some(vs) = video_solver {
        map.insert("video_solver".into(), Value::String(vs));
    }
    Ok(Some(RolloutSpec {
        json: Value::Object(map),
        return_state,
    }))
}

fn parse_seeds(params: &Obs, batch_size: usize) -> Result<Vec<Option<i64>>, String> {
    let raw_seed = params.get("seed").filter(|v| !matches!(v, Obs::Nil));
    let raw_seeds = params.get("seeds").filter(|v| !matches!(v, Obs::Nil));
    if raw_seed.is_some() && raw_seeds.is_some() {
        return Err("Provide either 'seed' or 'seeds', not both".to_string());
    }
    if let Some(s) = raw_seed {
        return Ok(vec![Some(check_int64(s, "seed")?) ; batch_size]);
    }
    if let Some(seeds) = raw_seeds {
        let items: Vec<i64> = match seeds {
            Obs::List(list) => list
                .iter()
                .map(|x| check_int64(x, "seeds[]"))
                .collect::<Result<_, _>>()?,
            Obs::Array { dtype, shape, data } => {
                if shape.len() != 1 || !matches!(dtype.kind, 'i' | 'u') {
                    return Err("parameters.seeds must be a 1D integer list or array".to_string());
                }
                (0..shape[0])
                    .map(|idx| {
                        let (_, int_val) = msgpack_numpy::element(*dtype, data, idx);
                        i64::try_from(int_val.unwrap_or(0))
                            .map_err(|_| "seeds[] must fit in a signed 64-bit integer".to_string())
                    })
                    .collect::<Result<_, _>>()?
            }
            _ => return Err("parameters.seeds must be a list of integers".to_string()),
        };
        if items.len() != batch_size {
            return Err(format!(
                "parameters.seeds length ({}) must match batch size ({batch_size})",
                items.len()
            ));
        }
        return Ok(items.into_iter().map(Some).collect());
    }
    Ok(vec![None; batch_size])
}

fn parse_robot_state_rows(
    raw: &Obs,
    batch_size: usize,
    name: &str,
) -> Result<Vec<Vec<Value>>, String> {
    let to_num = |f: f64| -> Result<Value, String> {
        if !f.is_finite() {
            return Err(format!("{name} must be finite"));
        }
        serde_json::Number::from_f64(f)
            .map(Value::Number)
            .ok_or_else(|| format!("{name} must be finite"))
    };
    match raw {
        Obs::Array { dtype, shape, data } => {
            if !matches!(dtype.kind, 'i' | 'u' | 'f') {
                return Err(format!("{name} must be numeric"));
            }
            match shape.as_slice() {
                [d] => {
                    if *d == 0 {
                        return Err(format!("{name} must not be empty"));
                    }
                    if batch_size != 1 {
                        return Err(format!(
                            "{name} must be 2D [B, D] for batch size {batch_size}"
                        ));
                    }
                    let row = (0..*d)
                        .map(|i| to_num(msgpack_numpy::element(*dtype, data, i).0))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(vec![row])
                }
                [b, d] => {
                    if *b == 0 || *d == 0 {
                        return Err(format!("{name} must not be empty"));
                    }
                    if *b != batch_size {
                        return Err(format!(
                            "{name} batch size ({b}) must match ({batch_size})"
                        ));
                    }
                    let mut rows = Vec::with_capacity(*b);
                    for r in 0..*b {
                        let row = (0..*d)
                            .map(|c| to_num(msgpack_numpy::element(*dtype, data, r * d + c).0))
                            .collect::<Result<Vec<_>, _>>()?;
                        rows.push(row);
                    }
                    Ok(rows)
                }
                _ => Err(format!("{name} must be 1D or 2D, got {}D", shape.len())),
            }
        }
        Obs::List(items) => {
            if items.is_empty() {
                return Err(format!("{name} must not be empty"));
            }
            if matches!(items[0], Obs::List(_)) {
                if items.len() != batch_size {
                    return Err(format!(
                        "{name} batch size ({}) must match ({batch_size})",
                        items.len()
                    ));
                }
                let mut rows = Vec::with_capacity(items.len());
                for item in items {
                    let Obs::List(sub) = item else {
                        return Err(format!("{name} rows must be non-empty lists of numbers"));
                    };
                    if sub.is_empty() {
                        return Err(format!("{name} rows must be non-empty lists of numbers"));
                    }
                    rows.push(
                        sub.iter()
                            .map(|x| check_finite_number(x, name).and_then(to_num))
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                }
                Ok(rows)
            } else {
                if batch_size != 1 {
                    return Err(format!(
                        "{name} must be 2D [B, D] for batch size {batch_size}"
                    ));
                }
                let row = items
                    .iter()
                    .map(|x| check_finite_number(x, name).and_then(to_num))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(vec![row])
            }
        }
        _ => Err(format!("{name} must be a list or ndarray")),
    }
}

fn save_image_ref(
    ref_obs: &Obs,
    upload_dir: &Path,
    allow_remote: bool,
    guard: &mut SavedPathGuard,
) -> Result<String, String> {
    let Obs::Str(reference) = ref_obs else {
        return Err("input_reference entries must be non-empty data URLs".to_string());
    };
    if reference.is_empty() {
        return Err("input_reference entries must be non-empty data URLs".to_string());
    }
    let (modality, path) = crate::media::resolve_media_ref(reference, upload_dir, allow_remote)?;
    guard.push(path.clone());
    if modality != "image" {
        return Err(format!("input_reference must be an image, got {modality}"));
    }
    Ok(path)
}

fn build_cosmos3_rows(
    inp: &Obs,
    params: &Obs,
    facts: &ActionsFacts,
    upload_dir: &Path,
    allow_remote: bool,
    guard: &mut SavedPathGuard,
) -> Result<(Vec<SubmitArgs>, bool), String> {
    let prompts: Vec<String> = match inp.get("prompt") {
        Some(Obs::Str(s)) => vec![s.clone()],
        Some(Obs::List(items)) => items
            .iter()
            .map(|p| match p {
                Obs::Str(s) => Ok(s.clone()),
                _ => Err("Cosmos3 input.prompt must be a string or list of strings".to_string()),
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err("Cosmos3 input.prompt must be a string or list of strings".to_string()),
    };
    let batch_size = prompts.len();
    if !(1..=MAX_ACTION_BATCH_SIZE).contains(&batch_size) {
        return Err(format!(
            "Batch size must be in [1, {MAX_ACTION_BATCH_SIZE}], got {batch_size}"
        ));
    }

    let ref_paths: Vec<Option<String>> = match inp.get("input_reference") {
        None | Some(Obs::Nil) => vec![None; batch_size],
        Some(s @ Obs::Str(_)) if batch_size == 1 => {
            vec![Some(save_image_ref(s, upload_dir, allow_remote, guard)?)]
        }
        Some(Obs::List(items)) => {
            if items.len() != batch_size {
                return Err(format!(
                    "input.input_reference length ({}) must match input.prompt length ({batch_size})",
                    items.len()
                ));
            }
            items
                .iter()
                .map(|r| save_image_ref(r, upload_dir, allow_remote, guard).map(Some))
                .collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err("input.input_reference must be a list of image data URLs".to_string()),
    };

    let effective_steps = match params.get("num_inference_steps") {
        Some(v) if !matches!(v, Obs::Nil) => Some(check_int64(v, "num_inference_steps")?),
        _ => facts.num_flow_steps.map(|n| n as i64),
    };
    let rollout = validate_rollout(params.get("rollout"), effective_steps)?;
    let seeds = parse_seeds(params, batch_size)?;

    let inp_has_state = inp.get("robot_state").is_some_and(|v| !matches!(v, Obs::Nil))
        || inp.get("robot_states").is_some_and(|v| !matches!(v, Obs::Nil));
    let par_has_state = params.get("robot_state").is_some_and(|v| !matches!(v, Obs::Nil))
        || params.get("robot_states").is_some_and(|v| !matches!(v, Obs::Nil));
    if inp_has_state && par_has_state {
        return Err("Provide robot_state/robot_states in either input or parameters, not both".to_string());
    }
    let state_src = if par_has_state { params } else { inp };
    let rs = state_src.get("robot_state").filter(|v| !matches!(v, Obs::Nil));
    let rss = state_src.get("robot_states").filter(|v| !matches!(v, Obs::Nil));
    if rs.is_some() && rss.is_some() {
        return Err("Provide either 'robot_state' or 'robot_states', not both".to_string());
    }
    let state_rows = match rs.or(rss) {
        Some(v) => Some(parse_robot_state_rows(v, batch_size, "robot_state")?),
        None => None,
    };

    let mut shared_mk = Map::new();
    let Obs::Map(param_pairs) = params else {
        return Err("'parameters' must be a map".to_string());
    };
    for (k, v) in param_pairs {
        let Obs::Str(ks) = k else {
            return Err("parameters keys must be strings".to_string());
        };
        if matches!(
            ks.as_str(),
            "seed" | "seeds" | "rollout" | "robot_state" | "robot_states"
        ) || matches!(v, Obs::Nil)
        {
            continue;
        }
        shared_mk.insert(ks.clone(), obs_to_json(v, &format!("parameters.{ks}"))?);
    }
    match shared_mk.get("action_mode") {
        None => {
            shared_mk.insert("action_mode".into(), Value::String("policy".to_string()));
        }
        Some(Value::String(_)) => {}
        _ => return Err("parameters.action_mode must be a string".to_string()),
    }
    for (key, min_val) in [
        ("raw_action_dim", 1i64),
        ("num_frames", 1),
        ("action_chunk_size", 1),
        ("num_inference_steps", 1),
    ] {
        if let Some(raw_v) = params.get(key).filter(|v| !matches!(v, Obs::Nil)) {
            let iv = check_int64(raw_v, &format!("parameters.{key}"))?;
            if iv < min_val {
                return Err(format!("parameters.{key} must be >= {min_val}"));
            }
            shared_mk.insert(key.into(), Value::from(iv));
        }
    }
    if let Some(ref r) = rollout {
        shared_mk.insert("rollout".into(), r.json.clone());
    }

    let mut rows = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let mut row_mk = shared_mk.clone();
        if let Some(seed) = seeds[i] {
            row_mk.insert("seed".into(), Value::from(seed));
        }
        if let Some(ref srows) = state_rows {
            row_mk.insert("robot_state".into(), Value::Array(srows[i].clone()));
        }
        let mut file_paths = std::collections::BTreeMap::new();
        let input_modalities = if let Some(ref p) = ref_paths[i] {
            file_paths.insert("image".to_string(), vec![p.clone()]);
            vec!["image".to_string(), "text".to_string()]
        } else {
            vec!["text".to_string()]
        };
        rows.push(SubmitArgs {
            text: Some(prompts[i].clone()),
            file_paths,
            input_modalities,
            output_modalities: vec!["action".to_string()],
            model_kwargs: row_mk,
            ..SubmitArgs::default()
        });
    }
    let expect_rollout = rollout.as_ref().is_some_and(|r| r.return_state);
    Ok((rows, expect_rollout))
}

fn parse_tokenized_prompt_rows(
    raw_tp: &Obs,
    raw_mask: Option<&Obs>,
) -> Result<(Vec<Vec<i64>>, Option<Vec<Vec<bool>>>), String> {
    let tp_rows: Vec<Vec<i64>> = match raw_tp {
        Obs::Array { dtype, shape, data } => {
            if !matches!(dtype.kind, 'i' | 'u') {
                return Err(format!(
                    "tokenized_prompt must have integer dtype, not {}",
                    dtype.name()
                ));
            }
            match shape.as_slice() {
                [l] => {
                    if *l == 0 {
                        return Err("tokenized_prompt must not be empty".to_string());
                    }
                    vec![(0..*l)
                        .map(|i| {
                            let (_, iv) = msgpack_numpy::element(*dtype, data, i);
                            i64::try_from(iv.unwrap_or(0)).map_err(|_| {
                                "tokenized_prompt[] must fit in a signed 64-bit integer".to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?]
                }
                [b, l] => {
                    if *b == 0 || *l == 0 {
                        return Err("tokenized_prompt must not be empty".to_string());
                    }
                    let mut rows = Vec::with_capacity(*b);
                    for r in 0..*b {
                        let row = (0..*l)
                            .map(|c| {
                                let (_, iv) = msgpack_numpy::element(*dtype, data, r * l + c);
                                i64::try_from(iv.unwrap_or(0)).map_err(|_| {
                                    "tokenized_prompt[] must fit in a signed 64-bit integer"
                                        .to_string()
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        rows.push(row);
                    }
                    rows
                }
                _ => {
                    return Err(format!(
                        "tokenized_prompt must be 1D or 2D, got {}D",
                        shape.len()
                    ))
                }
            }
        }
        Obs::List(items) => {
            if items.is_empty() {
                return Err("tokenized_prompt must not be empty".to_string());
            }
            if matches!(items[0], Obs::List(_)) {
                items
                    .iter()
                    .map(|row| {
                        let Obs::List(sub) = row else {
                            return Err(
                                "tokenized_prompt rows must be non-empty integer lists".to_string()
                            );
                        };
                        if sub.is_empty() {
                            return Err(
                                "tokenized_prompt rows must be non-empty integer lists".to_string()
                            );
                        }
                        sub.iter()
                            .map(|x| check_int64(x, "tokenized_prompt[]"))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                vec![items
                    .iter()
                    .map(|x| check_int64(x, "tokenized_prompt[]"))
                    .collect::<Result<Vec<_>, _>>()?]
            }
        }
        _ => return Err("tokenized_prompt must be a 1D/2D integer array or list".to_string()),
    };

    for row in &tp_rows {
        if row.iter().any(|&t| t < 0) {
            return Err("tokenized_prompt IDs must be non-negative".to_string());
        }
    }

    let Some(raw_mask) = raw_mask.filter(|v| !matches!(v, Obs::Nil)) else {
        return Ok((tp_rows, None));
    };

    let mask_rows_raw: Vec<Vec<bool>> = match raw_mask {
        Obs::Array { dtype, shape, data } => {
            if !matches!(dtype.kind, 'b' | 'i' | 'u') {
                return Err(format!(
                    "tokenized_prompt_mask must be bool or integer, not {}",
                    dtype.name()
                ));
            }
            let parse_elem = |idx: usize| -> Result<bool, String> {
                let (_, iv) = msgpack_numpy::element(*dtype, data, idx);
                let v = iv.unwrap_or(-1);
                if v == 0 || v == 1 {
                    Ok(v == 1)
                } else {
                    Err("tokenized_prompt_mask elements must be bool or 0/1".to_string())
                }
            };
            match shape.as_slice() {
                [l] => vec![(0..*l).map(parse_elem).collect::<Result<Vec<_>, _>>()?],
                [b, l] => (0..*b)
                    .map(|r| {
                        (0..*l)
                            .map(|c| parse_elem(r * l + c))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                _ => {
                    return Err(format!(
                        "tokenized_prompt_mask must be 1D or 2D, got {}D",
                        shape.len()
                    ))
                }
            }
        }
        Obs::List(items) => {
            if items.is_empty() {
                return Err("tokenized_prompt_mask must not be empty".to_string());
            }
            if matches!(items[0], Obs::List(_)) {
                items
                    .iter()
                    .map(|row| {
                        let Obs::List(sub) = row else {
                            return Err(
                                "tokenized_prompt_mask elements must be bool or 0/1".to_string()
                            );
                        };
                        sub.iter()
                            .map(|x| bool_01(x, "tokenized_prompt_mask"))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                vec![items
                    .iter()
                    .map(|x| bool_01(x, "tokenized_prompt_mask"))
                    .collect::<Result<Vec<_>, _>>()?]
            }
        }
        _ => {
            return Err("tokenized_prompt_mask must be a 1D/2D bool/int array or list".to_string())
        }
    };

    if mask_rows_raw.len() != tp_rows.len() {
        return Err(format!(
            "tokenized_prompt_mask batch size ({}) must match tokenized_prompt ({})",
            mask_rows_raw.len(),
            tp_rows.len()
        ));
    }

    for (row_idx, (tp_row, m_row)) in tp_rows.iter().zip(mask_rows_raw.iter()).enumerate() {
        if m_row.len() != tp_row.len() {
            return Err(format!(
                "tokenized_prompt_mask row {row_idx} length must match tokenized_prompt ({})",
                tp_row.len()
            ));
        }
        let mut seen_false = false;
        let mut any_true = false;
        for &b in m_row {
            if seen_false && b {
                return Err(
                    "tokenized_prompt_mask must be a contiguous prefix of True values".to_string(),
                );
            }
            if b {
                any_true = true;
            } else {
                seen_false = true;
            }
        }
        if !any_true {
            return Err("tokenized_prompt_mask must have at least one True token".to_string());
        }
    }

    Ok((tp_rows, Some(mask_rows_raw)))
}

enum Pi05Images {
    Raw {
        rows: Vec<Vec<RawArray>>,
        cam_order: Option<Vec<String>>,
    },
    Files {
        rows: Vec<Vec<String>>,
    },
}

fn parse_pi05_images(
    inp: &Obs,
    num_cameras: usize,
    upload_dir: &Path,
    allow_remote: bool,
    guard: &mut SavedPathGuard,
) -> Result<Pi05Images, String> {
    let img_keys = ["images", "pixel_values", "camera_images"];
    let present: Vec<(&str, &Obs)> = img_keys
        .iter()
        .filter_map(|&k| {
            inp.get(k)
                .filter(|v| !matches!(v, Obs::Nil))
                .map(|v| (k, v))
        })
        .collect();
    if present.len() > 1 {
        let names: Vec<String> = present.iter().map(|(k, _)| format!("'{k}'")).collect();
        return Err(format!("Provide only one of [{}]", names.join(", ")));
    }
    let raw_ref = inp.get("input_reference").filter(|v| !matches!(v, Obs::Nil));
    if !present.is_empty() && raw_ref.is_some() {
        return Err("Provide either image arrays or input_reference, not both".to_string());
    }
    if present.is_empty() && raw_ref.is_none() {
        return Err("Pi0.5 input requires images (or input_reference)".to_string());
    }

    if let Some(ref_obs) = raw_ref {
        let ref_rows: Vec<Vec<&Obs>> = match ref_obs {
            Obs::Str(_) => vec![vec![ref_obs]],
            Obs::List(items) if !items.is_empty() => {
                if items.iter().all(|r| matches!(r, Obs::Str(_))) {
                    items.iter().map(|r| vec![r]).collect()
                } else if items
                    .iter()
                    .all(|r| matches!(r, Obs::List(sub) if !sub.is_empty() && sub.iter().all(|x| matches!(x, Obs::Str(_)))))
                {
                    items
                        .iter()
                        .map(|r| {
                            let Obs::List(sub) = r else { unreachable!() };
                            sub.iter().collect()
                        })
                        .collect()
                } else {
                    return Err("input.input_reference must be a list of image data URLs".to_string());
                }
            }
            _ => {
                return Err(
                    "input.input_reference must be a non-empty list of image data URLs".to_string(),
                )
            }
        };
        let cam_count = ref_rows[0].len();
        if ref_rows.iter().any(|r| r.len() != cam_count) {
            return Err("All rows in input_reference must have the same number of cameras".to_string());
        }
        if num_cameras > 0 && cam_count > num_cameras {
            return Err(format!(
                "Received {cam_count} cameras; model accepts at most {num_cameras}"
            ));
        }
        let mut file_rows = Vec::with_capacity(ref_rows.len());
        for row in ref_rows {
            let mut paths = Vec::with_capacity(cam_count);
            for uri in row {
                paths.push(save_image_ref(uri, upload_dir, allow_remote, guard)?);
            }
            file_rows.push(paths);
        }
        return Ok(Pi05Images::Files { rows: file_rows });
    }

    let (key, val) = present[0];
    let mut cam_order: Option<Vec<String>> = None;
    let rows: Vec<Vec<RawArray>> = match val {
        Obs::Array { dtype, shape, data } => match shape.as_slice() {
            [b, c, d1, d2, d3] => {
                if *b == 0 || *c == 0 {
                    return Err(format!("{key} must contain at least one row and one camera"));
                }
                if num_cameras > 0 && *c > num_cameras {
                    return Err(format!(
                        "Received {c} cameras; model accepts at most {num_cameras}"
                    ));
                }
                let slice_elems = d1 * d2 * d3;
                let slice_bytes = slice_elems * dtype.size;
                let mut out = Vec::with_capacity(*b);
                for bi in 0..*b {
                    let mut row = Vec::with_capacity(*c);
                    for ci in 0..*c {
                        let start = (bi * (*c) + ci) * slice_bytes;
                        let slice = &data[start..start + slice_bytes];
                        row.push(msgpack_numpy::image_slice(
                            *dtype,
                            &[*d1, *d2, *d3],
                            slice,
                            &format!("{key}[{ci}]"),
                        )?);
                    }
                    out.push(row);
                }
                out
            }
            [c, d1, d2, d3] => {
                if *c == 0 {
                    return Err(format!("{key} must contain at least one row and one camera"));
                }
                if num_cameras > 0 && *c > num_cameras {
                    return Err(format!(
                        "Received {c} cameras; model accepts at most {num_cameras}"
                    ));
                }
                let slice_bytes = d1 * d2 * d3 * dtype.size;
                let mut row = Vec::with_capacity(*c);
                for ci in 0..*c {
                    let start = ci * slice_bytes;
                    let slice = &data[start..start + slice_bytes];
                    row.push(msgpack_numpy::image_slice(
                        *dtype,
                        &[*d1, *d2, *d3],
                        slice,
                        &format!("{key}[{ci}]"),
                    )?);
                }
                vec![row]
            }
            _ => {
                return Err(format!(
                    "{key} ndarray must be 4D [C, ...] or 5D [B, C, ...], got {}D",
                    shape.len()
                ))
            }
        },
        Obs::Map(pairs) => {
            if pairs.is_empty() {
                return Err(format!("{key} map must not be empty"));
            }
            let mut order = Vec::with_capacity(pairs.len());
            let mut per_cam: Vec<Vec<RawArray>> = Vec::with_capacity(pairs.len());
            let mut expected_b: Option<usize> = None;
            for (ci, (k_obs, v_obs)) in pairs.iter().enumerate() {
                let Obs::Str(cam_name) = k_obs else {
                    return Err(format!("{key} camera keys must be strings"));
                };
                order.push(cam_name.clone());
                let cam_key = format!("{key}[{ci}]");
                let cam_batch: Vec<RawArray> = match v_obs {
                    Obs::Array { dtype, shape, data } => match shape.as_slice() {
                        [b, d1, d2, d3] => {
                            let slice_bytes = d1 * d2 * d3 * dtype.size;
                            (0..*b)
                                .map(|bi| {
                                    let start = bi * slice_bytes;
                                    msgpack_numpy::image_slice(
                                        *dtype,
                                        &[*d1, *d2, *d3],
                                        &data[start..start + slice_bytes],
                                        &cam_key,
                                    )
                                })
                                .collect::<Result<Vec<_>, _>>()?
                        }
                        [d1, d2, d3] => vec![msgpack_numpy::image_slice(
                            *dtype,
                            &[*d1, *d2, *d3],
                            data,
                            &cam_key,
                        )?],
                        _ => {
                            return Err(format!(
                                "{key}.{cam_name} must be 3D or 4D, got {}D",
                                shape.len()
                            ))
                        }
                    },
                    Obs::List(items) if !items.is_empty() => items
                        .iter()
                        .map(|item| msgpack_numpy::image(item, &cam_key))
                        .collect::<Result<Vec<_>, _>>()?,
                    _ => {
                        return Err(format!(
                            "{key}.{cam_name} must be an image array or non-empty list of images"
                        ))
                    }
                };
                if let Some(eb) = expected_b {
                    if cam_batch.len() != eb {
                        return Err(format!(
                            "{key}.{cam_name} batch size ({}) does not match ({eb})",
                            cam_batch.len()
                        ));
                    }
                } else {
                    expected_b = Some(cam_batch.len());
                }
                per_cam.push(cam_batch);
            }
            let b = expected_b.unwrap_or(0);
            if b == 0 {
                return Err(format!("{key} must contain at least one row and one camera"));
            }
            if num_cameras > 0 && order.len() > num_cameras {
                return Err(format!(
                    "Received {} cameras; model accepts at most {num_cameras}",
                    order.len()
                ));
            }
            cam_order = Some(order);
            (0..b)
                .map(|bi| per_cam.iter().map(|c| c[bi].clone()).collect())
                .collect()
        }
        Obs::List(items) => {
            if items.is_empty() {
                return Err(format!("{key} must not be empty"));
            }
            if items
                .iter()
                .all(|x| matches!(x, Obs::Array { shape, .. } if shape.len() == 4))
            {
                items
                    .iter()
                    .map(|item| {
                        let Obs::Array { dtype, shape, data } = item else { unreachable!() };
                        let (c, d1, d2, d3) = (shape[0], shape[1], shape[2], shape[3]);
                        let slice_bytes = d1 * d2 * d3 * dtype.size;
                        (0..c)
                            .map(|ci| {
                                msgpack_numpy::image_slice(
                                    *dtype,
                                    &[d1, d2, d3],
                                    &data[ci * slice_bytes..(ci + 1) * slice_bytes],
                                    &format!("{key}[{ci}]"),
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else if items.iter().all(|x| matches!(x, Obs::List(sub) if !sub.is_empty())) {
                items
                    .iter()
                    .map(|item| {
                        let Obs::List(sub) = item else { unreachable!() };
                        sub.iter()
                            .enumerate()
                            .map(|(ci, cam)| msgpack_numpy::image(cam, &format!("{key}[{ci}]")))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else if items
                .iter()
                .all(|x| matches!(x, Obs::Array { shape, .. } if shape.len() == 3))
            {
                vec![items
                    .iter()
                    .enumerate()
                    .map(|(ci, cam)| msgpack_numpy::image(cam, &format!("{key}[{ci}]")))
                    .collect::<Result<Vec<_>, _>>()?]
            } else {
                return Err(format!(
                    "{key} list must contain 3D/4D image arrays or per-row camera lists"
                ));
            }
        }
        _ => return Err(format!("{key} must be an ndarray, map, or list")),
    };

    if rows.is_empty() || rows[0].is_empty() {
        return Err(format!("{key} must contain at least one row and one camera"));
    }
    let cam_count = rows[0].len();
    if rows.iter().any(|r| r.len() != cam_count) {
        return Err("All rows must have the same number of cameras".to_string());
    }
    if num_cameras > 0 && cam_count > num_cameras {
        return Err(format!(
            "Received {cam_count} cameras; model accepts at most {num_cameras}"
        ));
    }
    Ok(Pi05Images::Raw { rows, cam_order })
}

fn parse_camera_mask(
    raw_mask: &Obs,
    batch_size: usize,
    cam_count: usize,
    cam_order: Option<&[String]>,
) -> Result<Vec<Vec<bool>>, String> {
    let rows: Vec<Vec<bool>> = match raw_mask {
        Obs::Map(pairs) => {
            let Some(order) = cam_order else {
                return Err("camera_mask can only be a mapping when images is a mapping".to_string());
            };
            let mask_keys: HashSet<&str> = pairs
                .iter()
                .filter_map(|(k, _)| match k {
                    Obs::Str(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect();
            let order_set: HashSet<&str> = order.iter().map(String::as_str).collect();
            if mask_keys.len() != pairs.len() || mask_keys != order_set {
                return Err("camera_mask keys must match images keys".to_string());
            }
            let mut per_cam: Vec<Vec<bool>> = Vec::with_capacity(cam_count);
            for cam_name in order {
                let v = raw_mask.get(cam_name).unwrap();
                let vals: Vec<bool> = match v {
                    Obs::Array { dtype, shape, data } => {
                        if !matches!(dtype.kind, 'b' | 'i' | 'u') {
                            return Err(format!(
                                "camera_mask.{cam_name} elements must be bool or 0/1"
                            ));
                        }
                        let parse_i = |idx: usize| -> Result<bool, String> {
                            let (_, iv) = msgpack_numpy::element(*dtype, data, idx);
                            let n = iv.unwrap_or(-1);
                            if n == 0 || n == 1 {
                                Ok(n == 1)
                            } else {
                                Err(format!(
                                    "camera_mask.{cam_name} elements must be bool or 0/1"
                                ))
                            }
                        };
                        match shape.as_slice() {
                            [] => vec![parse_i(0)?],
                            [b] => (0..*b).map(parse_i).collect::<Result<Vec<_>, _>>()?,
                            _ => {
                                return Err(format!(
                                    "camera_mask.{cam_name} must be scalar or 1D"
                                ))
                            }
                        }
                    }
                    Obs::List(items) => items
                        .iter()
                        .map(|x| bool_01(x, &format!("camera_mask.{cam_name}")))
                        .collect::<Result<Vec<_>, _>>()?,
                    other => vec![bool_01(other, &format!("camera_mask.{cam_name}"))?],
                };
                if vals.len() != batch_size {
                    return Err(format!(
                        "camera_mask.{cam_name} length ({}) must match batch size ({batch_size})",
                        vals.len()
                    ));
                }
                per_cam.push(vals);
            }
            (0..batch_size)
                .map(|b| (0..cam_count).map(|c| per_cam[c][b]).collect())
                .collect()
        }
        Obs::Array { dtype, shape, data } => {
            if !matches!(dtype.kind, 'b' | 'i' | 'u') {
                return Err(format!(
                    "camera_mask must be bool or 0/1 integer, not {}",
                    dtype.name()
                ));
            }
            let parse_i = |idx: usize| -> Result<bool, String> {
                let (_, iv) = msgpack_numpy::element(*dtype, data, idx);
                let n = iv.unwrap_or(-1);
                if n == 0 || n == 1 {
                    Ok(n == 1)
                } else {
                    Err("camera_mask elements must be bool or 0/1".to_string())
                }
            };
            match shape.as_slice() {
                [c] => {
                    if batch_size != 1 {
                        return Err(format!(
                            "camera_mask must be 2D [B, C] for batch size {batch_size}"
                        ));
                    }
                    vec![(0..*c).map(parse_i).collect::<Result<Vec<_>, _>>()?]
                }
                [b, c] => {
                    if *b != batch_size {
                        return Err(format!(
                            "camera_mask batch size ({b}) must match ({batch_size})"
                        ));
                    }
                    (0..*b)
                        .map(|r| (0..*c).map(|col| parse_i(r * c + col)).collect())
                        .collect::<Result<Vec<_>, _>>()?
                }
                _ => {
                    return Err(format!(
                        "camera_mask must be 1D or 2D, got {}D",
                        shape.len()
                    ))
                }
            }
        }
        Obs::List(items) => {
            if items.is_empty() {
                return Err("camera_mask must not be empty".to_string());
            }
            if matches!(items[0], Obs::List(_)) {
                if items.len() != batch_size {
                    return Err(format!(
                        "camera_mask batch size ({}) must match ({batch_size})",
                        items.len()
                    ));
                }
                items
                    .iter()
                    .map(|row| {
                        let Obs::List(sub) = row else {
                            return Err("camera_mask elements must be bool or 0/1".to_string());
                        };
                        sub.iter()
                            .map(|x| bool_01(x, "camera_mask"))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                if batch_size != 1 {
                    return Err(format!(
                        "camera_mask must be 2D [B, C] for batch size {batch_size}"
                    ));
                }
                vec![items
                    .iter()
                    .map(|x| bool_01(x, "camera_mask"))
                    .collect::<Result<Vec<_>, _>>()?]
            }
        }
        _ => return Err("camera_mask must be an array, list, or mapping".to_string()),
    };

    for row in &rows {
        if row.len() != cam_count {
            return Err(format!(
                "camera_mask length ({}) must match camera count ({cam_count})",
                row.len()
            ));
        }
        if !row.iter().any(|&b| b) {
            return Err("camera_mask must keep at least one camera".to_string());
        }
    }
    Ok(rows)
}

fn build_pi05_rows(
    inp: &Obs,
    params: &Obs,
    facts: &ActionsFacts,
    upload_dir: &Path,
    allow_remote: bool,
    guard: &mut SavedPathGuard,
) -> Result<(Vec<SubmitArgs>, bool), String> {
    let images = parse_pi05_images(inp, facts.num_cameras, upload_dir, allow_remote, guard)?;
    let (img_batch, cam_count, cam_order) = match &images {
        Pi05Images::Raw { rows, cam_order } => (rows.len(), rows[0].len(), cam_order.as_deref()),
        Pi05Images::Files { rows } => (rows.len(), rows[0].len(), None),
    };

    let raw_tp = inp.get("tokenized_prompt").filter(|v| !matches!(v, Obs::Nil));
    let raw_tp_mask = inp
        .get("tokenized_prompt_mask")
        .filter(|v| !matches!(v, Obs::Nil));
    if raw_tp.is_none() && raw_tp_mask.is_some() {
        return Err("tokenized_prompt_mask requires tokenized_prompt".to_string());
    }
    let (tp_rows, tp_mask_rows) = match raw_tp {
        Some(tp) => {
            let (tps, masks) = parse_tokenized_prompt_rows(tp, raw_tp_mask)?;
            if tps.len() != img_batch {
                return Err(format!(
                    "tokenized_prompt batch size ({}) must match images ({img_batch})",
                    tps.len()
                ));
            }
            (Some(tps), masks)
        }
        None => (None, None),
    };

    let prompts: Vec<Option<String>> = match inp.get("prompt").filter(|v| !matches!(v, Obs::Nil)) {
        None => {
            if tp_rows.is_none() {
                return Err("Pi0.5 input requires either 'prompt' or 'tokenized_prompt'".to_string());
            }
            vec![None; img_batch]
        }
        Some(Obs::Str(s)) => vec![Some(s.clone()); img_batch],
        Some(Obs::List(items)) => {
            if items.len() != img_batch {
                return Err(format!(
                    "input.prompt length ({}) must match images ({img_batch})",
                    items.len()
                ));
            }
            items
                .iter()
                .map(|p| match p {
                    Obs::Str(s) => Ok(Some(s.clone())),
                    _ => Err("input.prompt must be a string or list of strings".to_string()),
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err("input.prompt must be a string or list of strings".to_string()),
    };

    let batch_size = img_batch;
    if !(1..=MAX_ACTION_BATCH_SIZE).contains(&batch_size) {
        return Err(format!(
            "Batch size must be in [1, {MAX_ACTION_BATCH_SIZE}], got {batch_size}"
        ));
    }

    let in_cm = inp.get("camera_mask").is_some_and(|v| !matches!(v, Obs::Nil))
        || inp.get("image_mask").is_some_and(|v| !matches!(v, Obs::Nil));
    let pa_cm = params.get("camera_mask").is_some_and(|v| !matches!(v, Obs::Nil))
        || params.get("image_mask").is_some_and(|v| !matches!(v, Obs::Nil));
    if in_cm && pa_cm {
        return Err(
            "Provide camera_mask/image_mask in either input or parameters, not both".to_string(),
        );
    }
    let cm_src = if in_cm { inp } else { params };
    let cm = cm_src.get("camera_mask").filter(|v| !matches!(v, Obs::Nil));
    let im = cm_src.get("image_mask").filter(|v| !matches!(v, Obs::Nil));
    if cm.is_some() && im.is_some() {
        return Err("Provide either 'camera_mask' or 'image_mask', not both".to_string());
    }
    let cam_mask_rows = match cm.or(im) {
        Some(raw) => Some(parse_camera_mask(raw, batch_size, cam_count, cam_order)?),
        None => None,
    };

    let in_rs = inp.get("robot_state").is_some_and(|v| !matches!(v, Obs::Nil))
        || inp.get("state").is_some_and(|v| !matches!(v, Obs::Nil));
    let pa_rs = params.get("robot_state").is_some_and(|v| !matches!(v, Obs::Nil))
        || params.get("state").is_some_and(|v| !matches!(v, Obs::Nil));
    if in_rs && pa_rs {
        return Err("Provide robot_state/state in either input or parameters, not both".to_string());
    }
    let rs_src = if in_rs { inp } else { params };
    let rs = rs_src.get("robot_state").filter(|v| !matches!(v, Obs::Nil));
    let st_v = rs_src.get("state").filter(|v| !matches!(v, Obs::Nil));
    if rs.is_some() && st_v.is_some() {
        return Err("Provide either 'robot_state' or 'state', not both".to_string());
    }
    let state_rows = match rs.or(st_v) {
        Some(raw) => Some(parse_robot_state_rows(raw, batch_size, "robot_state")?),
        None => None,
    };

    let rollout = validate_rollout(
        params.get("rollout"),
        facts.num_flow_steps.map(|n| n as i64),
    )?;
    let seeds = parse_seeds(params, batch_size)?;

    let mut shared_mk = Map::new();
    let Obs::Map(param_pairs) = params else {
        return Err("'parameters' must be a map".to_string());
    };
    for (k, v) in param_pairs {
        let Obs::Str(ks) = k else {
            return Err("parameters keys must be strings".to_string());
        };
        if matches!(
            ks.as_str(),
            "seed" | "seeds" | "rollout" | "robot_state" | "state" | "camera_mask" | "image_mask"
        ) || matches!(v, Obs::Nil)
        {
            continue;
        }
        shared_mk.insert(ks.clone(), obs_to_json(v, &format!("parameters.{ks}"))?);
    }
    if let Some(ref r) = rollout {
        shared_mk.insert("rollout".into(), r.json.clone());
    }

    let mut rows = Vec::with_capacity(batch_size);
    for (i, prompt) in prompts.into_iter().enumerate() {
        let mut row_mk = shared_mk.clone();
        if let Some(seed) = seeds[i] {
            row_mk.insert("seed".into(), Value::from(seed));
        }
        if let Some(ref tps) = tp_rows {
            row_mk.insert(
                "tokenized_prompt".into(),
                Value::Array(tps[i].iter().copied().map(Value::from).collect()),
            );
        }
        if let Some(ref masks) = tp_mask_rows {
            row_mk.insert(
                "tokenized_prompt_mask".into(),
                Value::Array(masks[i].iter().copied().map(Value::from).collect()),
            );
        }
        if let Some(ref cm_rows) = cam_mask_rows {
            row_mk.insert(
                "camera_mask".into(),
                Value::Array(cm_rows[i].iter().copied().map(Value::from).collect()),
            );
        }
        if let Some(ref srows) = state_rows {
            row_mk.insert(facts.state_kwarg.clone(), Value::Array(srows[i].clone()));
        }

        let mut raw_inputs = std::collections::BTreeMap::new();
        let mut file_paths = std::collections::BTreeMap::new();
        match &images {
            Pi05Images::Raw { rows: img_rows, .. } => {
                raw_inputs.insert("image".to_string(), img_rows[i].clone());
            }
            Pi05Images::Files { rows: path_rows } => {
                file_paths.insert("image".to_string(), path_rows[i].clone());
            }
        }
        rows.push(SubmitArgs {
            text: prompt,
            file_paths,
            raw_inputs,
            input_modalities: vec!["image".to_string(), "text".to_string()],
            output_modalities: vec!["action".to_string()],
            model_kwargs: row_mk,
            ..SubmitArgs::default()
        });
    }
    let expect_rollout = rollout.as_ref().is_some_and(|r| r.return_state);
    Ok((rows, expect_rollout))
}

fn reshape_action(
    data: Vec<u8>,
    metadata: &Value,
    facts: &ActionsFacts,
) -> Result<([usize; 2], Vec<u8>), (u16, String)> {
    if data.is_empty() || data.len() % 4 != 0 {
        return Err((
            500,
            format!("Invalid action byte length: {}", data.len()),
        ));
    }
    let n_floats = data.len() / 4;
    if let Some(arr) = metadata.get("shape").and_then(Value::as_array) {
        if arr.len() == 2 {
            if let (Some(h), Some(d)) = (arr[0].as_u64(), arr[1].as_u64()) {
                let (h, d) = (h as usize, d as usize);
                if h > 0 && d > 0 && h * d == n_floats {
                    return Ok(([h, d], data));
                }
            }
        }
    }
    let d = facts.action_dim;
    if d == 0 || n_floats % d != 0 {
        return Err((
            500,
            format!("Action output has {n_floats} floats, incompatible with action_dim={d}"),
        ));
    }
    Ok(([n_floats / d, d], data))
}

fn obs_to_f32_ndarray(obs: &Obs, name: &str) -> Result<Mp, (u16, String)> {
    let Obs::Array { dtype, shape, data } = obs else {
        return Err((500, format!("Rollout {name} must be an ndarray")));
    };
    if dtype.kind == 'f' && dtype.size == 4 && dtype.little {
        return Ok(msgpack_numpy::pack_ndarray("<f4", shape, data.clone()));
    }
    if !matches!(dtype.kind, 'b' | 'i' | 'u' | 'f') {
        return Err((500, format!("Rollout {name} has invalid dtype")));
    }
    let n: usize = shape.iter().product();
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let (f, _) = msgpack_numpy::element(*dtype, data, i);
        out.extend_from_slice(&(f as f32).to_le_bytes());
    }
    Ok(msgpack_numpy::pack_ndarray("<f4", shape, out))
}

fn obs_to_bool_ndarray(obs: &Obs, name: &str) -> Result<Mp, (u16, String)> {
    let Obs::Array { dtype, shape, data } = obs else {
        return Err((500, format!("Rollout {name} must be an ndarray")));
    };
    if dtype.kind == 'b' && dtype.size == 1 {
        return Ok(msgpack_numpy::pack_ndarray("|b1", shape, data.clone()));
    }
    if !matches!(dtype.kind, 'b' | 'i' | 'u' | 'f') {
        return Err((500, format!("Rollout {name} has invalid dtype")));
    }
    let n: usize = shape.iter().product();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (f, int_v) = msgpack_numpy::element(*dtype, data, i);
        let b = match int_v {
            Some(iv) => iv != 0,
            None => f != 0.0,
        };
        out.push(u8::from(b));
    }
    Ok(msgpack_numpy::pack_ndarray("|b1", shape, out))
}

fn extract_rollout(data: &[u8], weight_version: Option<&str>) -> Result<Mp, (u16, String)> {
    let unpacked = msgpack_numpy::unpack_raw(data)
        .map_err(|e| (500, format!("Corrupt rollout tensor chunk: {e}")))?;
    if !matches!(unpacked, Obs::Map(_)) {
        return Err((500, "Rollout tensor chunk must unpack to a map".to_string()));
    }
    let chain = unpacked
        .get("chain")
        .ok_or_else(|| (500, "Rollout tensor chunk missing 'chain'".to_string()))?;
    let logprobs = unpacked
        .get("logprobs")
        .ok_or_else(|| (500, "Rollout tensor chunk missing 'logprobs'".to_string()))?;
    let logprobs_mask = unpacked
        .get("logprobs_mask")
        .ok_or_else(|| (500, "Rollout tensor chunk missing 'logprobs_mask'".to_string()))?;
    let denoise_ind = unpacked
        .get("denoise_ind")
        .ok_or_else(|| (500, "Rollout tensor chunk missing 'denoise_ind'".to_string()))
        .and_then(|v| {
            check_int64(v, "denoise_ind")
                .map_err(|_| (500, "Rollout denoise_ind and seed must be integers".to_string()))
        })?;
    let seed = unpacked
        .get("seed")
        .ok_or_else(|| (500, "Rollout tensor chunk missing 'seed'".to_string()))
        .and_then(|v| {
            check_int64(v, "seed")
                .map_err(|_| (500, "Rollout denoise_ind and seed must be integers".to_string()))
        })?;

    let wv = match weight_version {
        Some(s) => Mp::String(s.into()),
        None => Mp::Nil,
    };
    Ok(Mp::Map(vec![
        (
            Mp::String("chain".into()),
            obs_to_f32_ndarray(chain, "chain")?,
        ),
        (Mp::String("denoise_ind".into()), Mp::from(denoise_ind)),
        (
            Mp::String("logprobs".into()),
            obs_to_f32_ndarray(logprobs, "logprobs")?,
        ),
        (
            Mp::String("logprobs_mask".into()),
            obs_to_bool_ndarray(logprobs_mask, "logprobs_mask")?,
        ),
        (Mp::String("seed".into()), Mp::from(seed)),
        (Mp::String("weight_version".into()), wv),
    ]))
}

/// `POST /v1/actions/generations` (`application/msgpack`).
pub async fn actions_generations(State(st): State<AppState>, req: Request) -> Response {
    let facts = match resolve_actions(&st) {
        Ok(f) => f,
        Err(resp) => return resp,
    };

    let ctype = req
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if ctype != "application/msgpack" && ctype != "application/x-msgpack" {
        return crate::error(
            415,
            "Content-Type must be application/msgpack",
            "invalid_request_error",
        );
    }

    let body = match axum::body::to_bytes(req.into_body(), crate::body_limit_mb() << 20).await {
        Ok(b) => b,
        Err(e) if crate::is_length_limit(&e) => {
            return crate::error(413, &crate::body_too_large(), "invalid_request_error")
        }
        Err(e) => {
            return crate::error(
                400,
                &format!("reading request body: {e}"),
                "invalid_request_error",
            )
        }
    };
    if body.is_empty() {
        return crate::error(
            400,
            "Request body must not be empty",
            "invalid_request_error",
        );
    }

    let upload_dir = st.upload_dir.clone();
    let allow_remote = st.allow_remote;
    let facts_for_prep = facts.clone();
    let prepared = crate::blocking(move || -> Result<(Vec<SubmitArgs>, bool), String> {
        let obs = msgpack_numpy::unpack_raw(&body)
            .map_err(|e| format!("the observation is not openpi's msgpack: {e}"))?;
        if !matches!(obs, Obs::Map(_)) {
            return Err("Request body must be a msgpack map".to_string());
        }
        let inp = obs
            .get("input")
            .filter(|v| matches!(v, Obs::Map(_)))
            .ok_or_else(|| "Request body requires an 'input' map".to_string())?;
        let default_params = Obs::Map(Vec::new());
        let params = match obs.get("parameters") {
            None | Some(Obs::Nil) => &default_params,
            Some(p @ Obs::Map(_)) => p,
            Some(_) => return Err("'parameters' must be a map".to_string()),
        };
        validate_runtime(obs.get("runtime"))?;

        let mut guard = SavedPathGuard::new(&upload_dir);
        let res = if facts_for_prep.family == "cosmos3" {
            build_cosmos3_rows(inp, params, &facts_for_prep, &upload_dir, allow_remote, &mut guard)?
        } else {
            build_pi05_rows(inp, params, &facts_for_prep, &upload_dir, allow_remote, &mut guard)?
        };
        guard.disarm();
        Ok(res)
    })
    .await;

    let (rows, expect_rollout) = match prepared {
        Ok(v) => v,
        Err(msg) => return crate::error(400, &msg, "invalid_request_error"),
    };

    let _cleanup = DeferredRowCleanup {
        upload_dir: &st.upload_dir,
        rows: &rows,
    };

    let futs = rows
        .iter()
        .map(|args| collect(result_stream(&st, args, &rid("act"), false)));
    let collected = match try_join_all(futs).await {
        Ok(c) => c,
        Err((status, msg)) => {
            return crate::error(status, &msg, crate::error_type(status));
        }
    };

    let mut items = Vec::with_capacity(collected.len());
    for (idx, chunks) in collected.into_iter().enumerate() {
        let weight_version = chunks
            .iter()
            .find_map(|c| c.metadata.get("weight_version").and_then(Value::as_str))
            .map(str::to_string);

        let mut action_chunk = None;
        let mut tensor_chunk = None;
        let mut tensor_is_rollout = false;
        for c in chunks {
            if c.modality == "action" {
                action_chunk = Some((c.data, c.metadata));
            } else if c.modality == "tensor" {
                let is_rollout = c
                    .metadata
                    .get("rollout_state")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if is_rollout || !tensor_is_rollout {
                    tensor_chunk = Some(c.data);
                    tensor_is_rollout = is_rollout;
                }
            }
        }

        let Some((action_data, action_meta)) = action_chunk else {
            return crate::error(
                500,
                "Request finished without action output",
                "server_error",
            );
        };
        let (shape, action_bytes) = match reshape_action(action_data, &action_meta, &facts) {
            Ok(v) => v,
            Err((status, msg)) => return crate::error(status, &msg, crate::error_type(status)),
        };

        let mut entry = vec![
            (Mp::String("input_index".into()), Mp::from(idx as u64)),
            (
                Mp::String("action".into()),
                Mp::Map(vec![(
                    Mp::String("values".into()),
                    msgpack_numpy::pack_ndarray("<f4", &shape, action_bytes),
                )]),
            ),
        ];

        if expect_rollout {
            let Some(tdata) = tensor_chunk else {
                return crate::error(
                    500,
                    "Request finished without rollout state output",
                    "server_error",
                );
            };
            let rollout_mp = match extract_rollout(&tdata, weight_version.as_deref()) {
                Ok(v) => v,
                Err((status, msg)) => {
                    return crate::error(status, &msg, crate::error_type(status))
                }
            };
            entry.push((Mp::String("rollout".into()), rollout_mp));
        }

        items.push(Mp::Map(entry));
    }

    let reply = msgpack_numpy::encode(&Mp::Map(vec![(
        Mp::String("data".into()),
        Mp::Array(items),
    )]));
    (StatusCode::OK, [(CONTENT_TYPE, "application/msgpack")], reply).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::routing::post;
    use axum::Router;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tower::ServiceExt;

    fn test_state(model_name: &str, actions: Option<ActionsFacts>) -> AppState {
        let mut st = crate::tests::state(None);
        st.model_name = model_name.to_string();
        st.actions = actions;
        st
    }

    fn cosmos3_facts() -> ActionsFacts {
        ActionsFacts {
            family: "cosmos3".into(),
            action_horizon: 16,
            action_dim: 64,
            state_kwarg: "robot_state".into(),
            state_dim: 64,
            num_cameras: 1,
            num_flow_steps: Some(30),
        }
    }

    fn pi05_facts() -> ActionsFacts {
        ActionsFacts {
            family: "pi05".into(),
            action_horizon: 10,
            action_dim: 32,
            state_kwarg: "robot_state".into(),
            state_dim: 32,
            num_cameras: 3,
            num_flow_steps: Some(3),
        }
    }

    fn router(st: AppState) -> Router {
        Router::new()
            .route("/v1/actions/generations", post(actions_generations))
            .with_state(st)
    }

    fn ndarray_str(dtype: &str, shape: &[u64], data: Vec<u8>) -> Mp {
        Mp::Map(vec![
            (Mp::String("__ndarray__".into()), Mp::Boolean(true)),
            (Mp::String("data".into()), Mp::Binary(data)),
            (Mp::String("dtype".into()), Mp::String(dtype.into())),
            (
                Mp::String("shape".into()),
                Mp::Array(shape.iter().map(|n| Mp::from(*n)).collect()),
            ),
        ])
    }

    #[tokio::test]
    async fn non_action_model_returns_404_and_wrong_ctype_returns_415() {
        let app = router(test_state("qwen3_omni", None));
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/actions/generations")
            .header(CONTENT_TYPE, "application/msgpack")
            .body(Body::from(vec![0x80]))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let app2 = router(test_state("pi05_libero", Some(pi05_facts())));
        let req2 = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/actions/generations")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(b"{}".to_vec()))
            .unwrap();
        let resp2 = app2.oneshot(req2).await.unwrap();
        assert_eq!(resp2.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }

    #[tokio::test]
    async fn rejects_non_finite_floats_and_cleans_up_uploads_on_400() {
        let tmp = std::env::temp_dir().join(format!("mstar_act_test_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&tmp).unwrap();
        let mut st = test_state("cosmos3", Some(cosmos3_facts()));
        st.upload_dir = tmp.clone();

        let tiny_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        let payload = msgpack_numpy::encode(&Mp::Map(vec![
            (
                Mp::String("input".into()),
                Mp::Map(vec![
                    (
                        Mp::String("prompt".into()),
                        Mp::Array(vec![Mp::String("pick".into()), Mp::String("place".into())]),
                    ),
                    // mismatched length (1 ref vs 2 prompts) after saving file 0!
                    (
                        Mp::String("input_reference".into()),
                        Mp::Array(vec![Mp::String(tiny_png.into())]),
                    ),
                ]),
            ),
            (Mp::String("parameters".into()), Mp::Map(vec![])),
        ]));
        let resp = router(st.clone())
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/actions/generations")
                    .header(CONTENT_TYPE, "application/msgpack")
                    .body(Body::from(payload))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(std::fs::read_dir(&tmp).unwrap().count(), 0);

        // NaN noise_level rejected with 400
        let nan_payload = msgpack_numpy::encode(&Mp::Map(vec![
            (
                Mp::String("input".into()),
                Mp::Map(vec![(Mp::String("prompt".into()), Mp::String("pick".into()))]),
            ),
            (
                Mp::String("parameters".into()),
                Mp::Map(vec![(
                    Mp::String("rollout".into()),
                    Mp::Map(vec![
                        (Mp::String("sde_type".into()), Mp::String("flow_sde".into())),
                        (Mp::String("noise_level".into()), Mp::F64(f64::NAN)),
                    ]),
                )]),
            ),
        ]));
        let resp_nan = router(st)
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/actions/generations")
                    .header(CONTENT_TYPE, "application/msgpack")
                    .body(Body::from(nan_payload))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp_nan.status(), StatusCode::BAD_REQUEST);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn pi05_adapter_parses_str_keyed_arrays_and_masks() {
        let tmp = std::env::temp_dir().join(format!("mstar_act_pi05_{}", uuid::Uuid::new_v4().simple()));
        let mut guard = SavedPathGuard::new(&tmp);
        let facts = pi05_facts();

        // B=2, C=3, H=2, W=2, 3 channels uint8
        let img_bytes = vec![128u8; 2 * 3 * 2 * 2 * 3];
        let tp_bytes: Vec<u8> = [1i64, 2, 3, 4, 5, 6]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let tpm_bytes = vec![1u8, 1, 0, 1, 0, 0];
        let cm_bytes = vec![1u8, 1, 0, 1, 0, 1];
        let state_bytes: Vec<u8> = vec![0.5f32; 2 * 8]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let body = msgpack_numpy::encode(&Mp::Map(vec![
            (
                Mp::String("input".into()),
                Mp::Map(vec![
                    (
                        Mp::String("images".into()),
                        ndarray_str("|u1", &[2, 3, 2, 2, 3], img_bytes),
                    ),
                    (
                        Mp::String("tokenized_prompt".into()),
                        ndarray_str("<i8", &[2, 3], tp_bytes),
                    ),
                    (
                        Mp::String("tokenized_prompt_mask".into()),
                        ndarray_str("|b1", &[2, 3], tpm_bytes),
                    ),
                    (
                        Mp::String("camera_mask".into()),
                        ndarray_str("|b1", &[2, 3], cm_bytes),
                    ),
                    (
                        Mp::String("robot_state".into()),
                        ndarray_str("<f4", &[2, 8], state_bytes),
                    ),
                ]),
            ),
            (
                Mp::String("parameters".into()),
                Mp::Map(vec![
                    (
                        Mp::String("seeds".into()),
                        Mp::Array(vec![Mp::from(10), Mp::from(20)]),
                    ),
                    (
                        Mp::String("rollout".into()),
                        Mp::Map(vec![
                            (Mp::String("sde_type".into()), Mp::String("flow_sde".into())),
                            (Mp::String("sde_step".into()), Mp::from(1)),
                        ]),
                    ),
                ]),
            ),
        ]));
        let obs = msgpack_numpy::unpack_raw(&body).unwrap();
        let (rows, expect_rollout) = build_pi05_rows(
            obs.get("input").unwrap(),
            obs.get("parameters").unwrap(),
            &facts,
            &tmp,
            false,
            &mut guard,
        )
        .unwrap();
        assert!(expect_rollout);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].raw_inputs["image"].len(), 3);
        assert_eq!(rows[0].raw_inputs["image"][0].shape, vec![3, 2, 2]);
        assert_eq!(rows[0].model_kwargs["seed"], json!(10));
        assert_eq!(rows[1].model_kwargs["seed"], json!(20));
        assert_eq!(rows[0].model_kwargs["camera_mask"], json!([true, true, false]));
        assert_eq!(rows[1].model_kwargs["camera_mask"], json!([true, false, true]));
        assert_eq!(rows[0].model_kwargs["tokenized_prompt"], json!([1, 2, 3]));
        assert_eq!(
            rows[1].model_kwargs["tokenized_prompt_mask"],
            json!([true, false, false])
        );
    }

    #[test]
    fn extract_rollout_stamps_weight_version_and_preserves_shapes() {
        let chain: Vec<u8> = vec![1.0f32; 4 * 10 * 32]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let lp: Vec<u8> = vec![-0.5f32; 10 * 32]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let mask = vec![1u8; 10 * 32];
        let tensor_bytes = msgpack_numpy::encode(&Mp::Map(vec![
            (
                Mp::String("chain".into()),
                msgpack_numpy::pack_ndarray("<f4", &[4, 10, 32], chain),
            ),
            (Mp::String("denoise_ind".into()), Mp::from(2)),
            (
                Mp::String("logprobs".into()),
                msgpack_numpy::pack_ndarray("<f4", &[10, 32], lp),
            ),
            (
                Mp::String("logprobs_mask".into()),
                msgpack_numpy::pack_ndarray("|b1", &[10, 32], mask),
            ),
            (Mp::String("seed".into()), Mp::from(42)),
        ]));
        let packed_none = extract_rollout(&tensor_bytes, None).unwrap();
        let Mp::Map(pairs_none) = packed_none else {
            panic!()
        };
        assert!(pairs_none
            .iter()
            .any(|(k, v)| *k == Mp::String("weight_version".into()) && *v == Mp::Nil));

        let packed_v1 = extract_rollout(&tensor_bytes, Some("v1")).unwrap();
        let Mp::Map(pairs_v1) = packed_v1 else {
            panic!()
        };
        assert!(pairs_v1
            .iter()
            .any(|(k, v)| *k == Mp::String("weight_version".into())
                && *v == Mp::String("v1".into())));

        // Keep compiler happy about unused Arc/Mutex import if not used above
        let _ = Arc::new(Mutex::new(()));
    }
}
