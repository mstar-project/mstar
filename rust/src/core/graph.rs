use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::error::{CoreError, Result};

/// Special edge destination: route the edge's tensors to the client.
pub const EMIT_TO_CLIENT: &str = "EMIT_TO_CLIENT";
/// Special edge destination: drop the value (side-effect-only outputs).
pub const EMPTY_DESTINATION: &str = "EMPTY_DESTINATION";

/// A directed dataflow edge. `name` is both the producing node's output name
/// and the input name at the destination (as in `mstar/graph/base.py`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeSpec {
    pub next_node: String,
    pub name: String,
    /// Persisted edges become walk outputs handed to the policy on WalkDone
    /// (mstar's "persist signals" flowing back to the conductor).
    #[serde(default)]
    pub persist: bool,
    #[serde(default)]
    pub output_modality: Option<String>,
    /// Streaming edge (mstar's `StreamingGraphEdge`): the value goes into
    /// the stream buffer of the connection targeting this partition instead
    /// of a node in this walk. The producer is otherwise unaware.
    #[serde(default)]
    pub target_partition: Option<String>,
}

/// A unit of computation. Executed on the data plane (Python/torch); the
/// control plane only tracks its readiness and routes its outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSpec {
    pub name: String,
    pub input_names: BTreeSet<String>,
    #[serde(default)]
    pub outputs: Vec<EdgeSpec>,
}

/// A loop over a body section. `outputs` snapshot the last iteration's
/// values; `accumulated_outputs` collect a value per iteration and emit the
/// whole sequence at termination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopSpec {
    pub name: String,
    pub body: Box<Section>,
    pub max_iters: u32,
    #[serde(default)]
    pub outputs: Vec<EdgeSpec>,
    #[serde(default)]
    pub accumulated_outputs: Vec<EdgeSpec>,
}

/// The composable graph structure a model declares per walk. Sequential and
/// Parallel exist for construction ergonomics and (later) worker-graph
/// splitting; runtime readiness is pure dataflow, so compilation flattens
/// them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Section {
    Node(NodeSpec),
    Sequential { sections: Vec<Section> },
    Parallel { sections: Vec<Section> },
    Loop(LoopSpec),
}

/// A loop after compilation: membership plus termination/emission spec.
#[derive(Debug, Clone)]
pub struct CompiledLoop {
    pub name: String,
    pub max_iters: u32,
    /// Direct member nodes — nodes whose INNERMOST enclosing loop is this one.
    pub members: BTreeSet<String>,
    pub outputs: Vec<EdgeSpec>,
    pub accumulated_outputs: Vec<EdgeSpec>,
    /// Enclosing loop, for nesting (mstar's loop-inside-loop): an inner loop
    /// is an *entity* of its parent's iteration — the parent's iteration
    /// completes only once the inner loop has fully terminated.
    pub parent: Option<usize>,
    /// Direct child loops (indices into `CompiledWalk::loops`).
    pub children: Vec<usize>,
}

/// A walk's graph, flattened for execution: nodes by name, loops, and the
/// node -> loop membership index.
#[derive(Debug, Clone)]
pub struct CompiledWalk {
    pub name: String,
    pub nodes: BTreeMap<String, NodeSpec>,
    pub loops: Vec<CompiledLoop>,
    pub node_loop: BTreeMap<String, usize>,
}

impl CompiledWalk {
    pub fn compile(walk_name: &str, section: &Section) -> Result<Self> {
        let mut walk = CompiledWalk {
            name: walk_name.to_string(),
            nodes: BTreeMap::new(),
            loops: Vec::new(),
            node_loop: BTreeMap::new(),
        };
        walk.collect(section, None)?;
        walk.validate()?;
        Ok(walk)
    }

    fn collect(&mut self, section: &Section, enclosing_loop: Option<usize>) -> Result<()> {
        match section {
            Section::Node(node) => {
                if self.nodes.contains_key(&node.name) {
                    return Err(CoreError::DuplicateNode(
                        node.name.clone(),
                        self.name.clone(),
                    ));
                }
                self.nodes.insert(node.name.clone(), node.clone());
                if let Some(loop_idx) = enclosing_loop {
                    self.loops[loop_idx].members.insert(node.name.clone());
                    self.node_loop.insert(node.name.clone(), loop_idx);
                }
                Ok(())
            }
            Section::Sequential { sections } | Section::Parallel { sections } => {
                for s in sections {
                    self.collect(s, enclosing_loop)?;
                }
                Ok(())
            }
            Section::Loop(spec) => {
                if spec.max_iters == 0 {
                    return Err(CoreError::InvalidSpec(format!(
                        "loop '{}' has max_iters == 0",
                        spec.name
                    )));
                }
                let loop_idx = self.loops.len();
                self.loops.push(CompiledLoop {
                    name: spec.name.clone(),
                    max_iters: spec.max_iters,
                    members: BTreeSet::new(),
                    outputs: spec.outputs.clone(),
                    accumulated_outputs: spec.accumulated_outputs.clone(),
                    parent: enclosing_loop,
                    children: Vec::new(),
                });
                if let Some(outer) = enclosing_loop {
                    self.loops[outer].children.push(loop_idx);
                }
                self.collect(&spec.body, Some(loop_idx))?;
                if self.loops[loop_idx].members.is_empty()
                    && self.loops[loop_idx].children.is_empty()
                {
                    return Err(CoreError::InvalidSpec(format!(
                        "loop '{}' has an empty body",
                        spec.name
                    )));
                }
                Ok(())
            }
        }
    }

    fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(CoreError::InvalidSpec(format!(
                "walk '{}' has no nodes",
                self.name
            )));
        }
        // Every internal edge destination must exist.
        let edge_iter = self
            .nodes
            .values()
            .flat_map(|n| n.outputs.iter())
            .chain(self.loops.iter().flat_map(|l| l.outputs.iter()))
            .chain(self.loops.iter().flat_map(|l| l.accumulated_outputs.iter()));
        for edge in edge_iter {
            // Streaming edges leave this walk (their next_node lives in the
            // target partition's walk); everything else must resolve here.
            if edge.target_partition.is_none()
                && edge.next_node != EMIT_TO_CLIENT
                && edge.next_node != EMPTY_DESTINATION
                && !self.nodes.contains_key(&edge.next_node)
            {
                return Err(CoreError::UnknownNode(
                    edge.next_node.clone(),
                    self.name.clone(),
                ));
            }
        }
        Ok(())
    }

    pub fn loop_index(&self, loop_name: &str) -> Result<usize> {
        self.loops
            .iter()
            .position(|l| l.name == loop_name)
            .ok_or_else(|| CoreError::UnknownLoop(loop_name.to_string(), self.name.clone()))
    }
}

/// A model's full set of named walks (`prefill`, `decode`, ...), compiled.
/// This is what `get_graph_walk_graphs()` returns in Python mstar.
#[derive(Debug, Clone)]
pub struct WalkSet {
    pub walks: BTreeMap<String, Arc<CompiledWalk>>,
}

impl WalkSet {
    /// Build from the JSON spec the Python side sends:
    /// `{"walk_name": <Section>, ...}`.
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: BTreeMap<String, Section> = serde_json::from_str(json)
            .map_err(|e| CoreError::InvalidSpec(format!("bad walk-set JSON: {e}")))?;
        let mut walks = BTreeMap::new();
        for (name, section) in &raw {
            walks.insert(name.clone(), Arc::new(CompiledWalk::compile(name, section)?));
        }
        Ok(Self { walks })
    }

    pub fn get(&self, walk: &str) -> Result<Arc<CompiledWalk>> {
        self.walks
            .get(walk)
            .cloned()
            .ok_or_else(|| CoreError::UnknownWalk(walk.to_string()))
    }
}
