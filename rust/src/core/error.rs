use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("unknown node '{0}' in walk '{1}'")]
    UnknownNode(String, String),
    #[error("unknown loop '{0}' in walk '{1}'")]
    UnknownLoop(String, String),
    #[error("unknown walk '{0}'")]
    UnknownWalk(String),
    #[error("node '{node}' is not ready (missing inputs: {missing:?})")]
    NodeNotReady { node: String, missing: Vec<String> },
    #[error("node '{0}' was not scheduled; complete_node without take_node_inputs")]
    NodeNotScheduled(String),
    #[error("duplicate node name '{0}' in walk '{1}'")]
    DuplicateNode(String, String),
    #[error("invalid walk spec: {0}")]
    InvalidSpec(String),
}

pub type Result<T> = std::result::Result<T, CoreError>;
