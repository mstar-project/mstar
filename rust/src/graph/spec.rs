//! Compiled, interned graph spec — the immutable half of Python's
//! GraphNode/Loop/registry objects. Built once per worker graph, shared by
//! every request behind an `Arc`. Splitting this out is what removes the
//! per-request `deepcopy(section)`.

use rustc_hash::FxHashMap;
use std::sync::Arc;

pub type Sym = u32; // interned string (node name, signal name, worker id)
pub type NodeId = u32;
pub type LoopId = u32;

pub const MAX_INPUTS: usize = 64; // readiness is a u64 bitmask

#[derive(Default)]
pub struct StrToId {
    map: FxHashMap<Box<str>, Sym>,
    names: Vec<Box<str>>,
}

impl StrToId {
    pub fn intern(&mut self, s: &str) -> Sym {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.names.len() as Sym;
        let boxed: Box<str> = s.into();
        self.names.push(boxed.clone());
        self.map.insert(boxed, id);
        id
    }
    pub fn get(&self, s: &str) -> Option<Sym> {
        self.map.get(s).copied()
    }
    pub fn name(&self, id: Sym) -> &str {
        &self.names[id as usize]
    }
    /// For an id that might not be one of ours -- notably `Sym::MAX`, which
    /// the fanout invents for a destination with no sharding group. Every
    /// other Sym comes from `intern` and is valid by construction, so the
    /// infallible `name` stays the normal path.
    pub fn try_name(&self, id: Sym) -> Option<&str> {
        self.names.get(id as usize).map(|s| &**s)
    }
}

/// Where an output edge goes, resolved at compile time so the hot path never
/// string-compares against EMIT_TO_CLIENT / EMPTY_DESTINATION.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dest {
    Local(NodeId), // a node in this worker graph
    External(Sym), // a node owned elsewhere
    EmitToClient,
    Empty, // EMPTY_DESTINATION: persist-only, routed nowhere
}

impl Dest {
    pub fn is_to_worker(&self) -> bool {
        match self {
            Dest::Local(_) => true,
            Dest::External(_) => true,
            Dest::EmitToClient => false,
            Dest::Empty => false,
        }
    }
}

#[derive(Clone)]
pub struct EdgeSpec {
    pub name: Sym,
    pub dest: Dest,
    pub dest_sym: Sym,
    pub persist: bool,
    pub new_token: bool,
    pub streaming: bool,
    pub modality: Sym,
    /// Index of `name` in the destination node's input list (Local only).
    pub dest_slot: u8,
}

pub struct NodeSpec {
    pub name: Sym,
    pub async_enabled: bool,
    pub inputs: Vec<Sym>,
    pub full_mask: u64,
    pub streaming_mask: u64,
    /// Inputs an enclosing loop re-injects every iteration (Python's
    /// `_persist_for_loop`), so clearing the node's slots must NOT dereference
    /// them. Purely structural -- the loop's declared `external_inputs`, minus
    /// the streamed ones, which are never re-injected -- so it is a mask
    /// computed once here rather than a Vec rebuilt per rid per pass.
    pub held_mask: u64,
    /// True when every input is streaming — Python's `only_streaming_inputs`,
    /// which seeds `ready_for_streaming`.
    pub only_streaming: bool,
    pub outputs: Vec<EdgeSpec>,
    /// Innermost enclosing loop, if any.
    pub loop_id: Option<LoopId>,
}

impl NodeSpec {
    pub fn slot_of(&self, name: Sym) -> Option<u8> {
        self.inputs.iter().position(|&s| s == name).map(|i| i as u8)
    }
}

pub struct LoopSpec {
    pub name: Sym,
    pub max_iters: u32,
    pub parent: Option<LoopId>,
    /// Entities this loop's registry manages directly. A node inside a child
    /// loop belongs to the child, not here — matching Python's
    /// `GraphStateRegistry._set_managed_entities`, which stops at a Loop.
    pub member_nodes: Vec<NodeId>,
    pub child_loops: Vec<LoopId>,
    pub outputs: Vec<EdgeSpec>,
    pub accumulated: Vec<EdgeSpec>,
    pub output_names: Vec<Sym>,
    pub accum_names: Vec<Sym>,
    /// Python's `Loop._loop_back_inputs` / `_external_inputs`, computed by
    /// `GraphSection.get_inputs_outputs()` and handed over verbatim — the real
    /// port's compile seam does the same rather than re-deriving them.
    pub loop_back: Vec<(Sym, NodeId)>,
    pub external_inputs: Vec<(Sym, NodeId)>,
}

impl LoopSpec {
    pub fn n_entities(&self) -> u32 {
        (self.member_nodes.len() + self.child_loops.len()) as u32
    }
}

pub struct CompiledGraph {
    pub nodes: Vec<NodeSpec>,
    pub loops: Vec<LoopSpec>,
    pub by_name: FxHashMap<Sym, NodeId>,
    pub loop_by_name: FxHashMap<Sym, LoopId>,
    pub root_nodes: Vec<NodeId>,
    pub root_loops: Vec<LoopId>,
}

impl CompiledGraph {
    pub fn node(&self, id: NodeId) -> &NodeSpec {
        &self.nodes[id as usize]
    }
    pub fn lp(&self, id: LoopId) -> &LoopSpec {
        &self.loops[id as usize]
    }
    pub fn n_root_entities(&self) -> u32 {
        (self.root_nodes.len() + self.root_loops.len()) as u32
    }
    pub fn n_ready_words(&self) -> usize {
        self.nodes.len().div_ceil(64).max(1)
    }
    /// Outer -> inner chain of enclosing loops, Python's `_get_loop_order`.
    pub fn loop_order(&self, lid: LoopId) -> Vec<LoopId> {
        let mut chain = vec![lid];
        let mut cur = lid;
        while let Some(p) = self.lp(cur).parent {
            chain.push(p);
            cur = p;
        }
        chain.reverse();
        chain
    }
}

pub type GraphRef = Arc<CompiledGraph>;
