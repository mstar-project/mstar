
//! Compile a worker graph's spec, handed over from Python as plain dicts, into
//! the interned, index-addressed `CompiledGraph` the walk state runs against.

use std::sync::Arc;

use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use crate::graph::spec::*;

#[derive(FromPyObject)]
pub struct EdgeArg {
    #[pyo3(item)] pub name: String,
    #[pyo3(item)] pub dest: String,
    #[pyo3(item)] pub persist: bool,
    #[pyo3(item)] pub new_token: bool,
    #[pyo3(item)] pub streaming: bool,
    #[pyo3(item)] pub modality: String,
}

#[derive(FromPyObject)]
pub struct NodeArg {
    #[pyo3(item)] pub name: String,
    #[pyo3(item)] pub async_enabled: bool,
    #[pyo3(item)] pub inputs: Vec<String>,
    #[pyo3(item)] pub streaming_inputs: Vec<String>,
    #[pyo3(item)] pub outputs: Vec<EdgeArg>,
}

#[derive(FromPyObject)]
pub struct LoopArg {
    #[pyo3(item)] pub name: String,
    #[pyo3(item)] pub max_iters: u32,
    #[pyo3(item)] pub parent: Option<String>,
    /// Directly-owned nodes only — a node inside a child loop belongs there.
    #[pyo3(item)] pub member_nodes: Vec<String>,
    #[pyo3(item)] pub outputs: Vec<EdgeArg>,
    #[pyo3(item)] pub accumulated: Vec<EdgeArg>,
    /// Verbatim `Loop._loop_back_inputs` / `_external_inputs`.
    #[pyo3(item)] pub loop_back: Vec<(String, String)>,
    #[pyo3(item)] pub external_inputs: Vec<(String, String)>,
}

/// Must match `mstar/graph/special_destinations.py`.
pub const EMIT_TO_CLIENT: &str = "emit_to_client";
pub const EMPTY_DESTINATION: &str = "";

/// Compile one worker graph's spec into a shared `CompiledGraph`.
pub fn compile_one(
    it: &mut StrToId,
    nodes: &[NodeArg],
    loops: &[LoopArg],
) -> PyResult<GraphRef> {
    let local: FxHashMap<&str, NodeId> = nodes.iter().enumerate()
        .map(|(i, n)| (n.name.as_str(), i as NodeId)).collect();

    let mk_edge = |it: &mut StrToId, e: &EdgeArg, local: &FxHashMap<&str, NodeId>| {
        let dest_sym = it.intern(&e.dest);
        let dest = if e.dest == EMIT_TO_CLIENT {
            Dest::EmitToClient
        } else if e.dest == EMPTY_DESTINATION {
            Dest::Empty
        } else if let Some(&id) = local.get(e.dest.as_str()) {
            Dest::Local(id)
        } else {
            Dest::External(dest_sym)
        };
        let dest_slot = match dest {
            Dest::Local(id) => nodes[id as usize].inputs.iter()
                .position(|s| *s == e.name).unwrap_or(0) as u8,
            _ => 0,
        };
        EdgeSpec {
            name: it.intern(&e.name), dest,
            dest_sym, persist: e.persist,
            new_token: e.new_token, streaming: e.streaming,
            modality: it.intern(&e.modality), dest_slot,
        }
    };

    let mut node_specs = Vec::with_capacity(nodes.len());
    for n in nodes {
        if n.inputs.len() > MAX_INPUTS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "node {} has {} inputs; readiness mask holds {}",
                n.name, n.inputs.len(), MAX_INPUTS)));
        }
        let name = it.intern(&n.name);
        let inputs: Vec<Sym> = n.inputs.iter().map(|s| it.intern(s)).collect();
        let full = if inputs.len() == 64 { u64::MAX } else { (1u64 << inputs.len()) - 1 };
        let mut streaming_mask = 0u64;
        for s in &n.streaming_inputs {
            let sym = it.intern(s);
            if let Some(i) = inputs.iter().position(|&x| x == sym) {
                streaming_mask |= 1 << i;
            }
        }
        let outputs = n.outputs.iter().map(|e| mk_edge(it, e, &local)).collect();
        node_specs.push(NodeSpec {
            name, inputs, full_mask: full, streaming_mask,
            async_enabled: n.async_enabled,
            only_streaming: full != 0 && streaming_mask == full,
            outputs, loop_id: None,
        });
    }

    // Loops: resolve parents by name (order-independent), then derive
    // child_loops and each node's innermost loop.
    let loop_idx: FxHashMap<&str, LoopId> = loops.iter().enumerate()
        .map(|(i, l)| (l.name.as_str(), i as LoopId)).collect();
    let mut loop_specs = Vec::with_capacity(loops.len());
    for l in loops {
        let parent = match &l.parent {
            Some(p) => Some(*loop_idx.get(p.as_str()).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("unknown parent loop {p}"))
            })?),
            None => None,
        };
        let member_nodes: Vec<NodeId> = l.member_nodes.iter()
            .map(|m| local[m.as_str()]).collect();
        let outputs: Vec<EdgeSpec> = l.outputs.iter().map(|e| mk_edge(it, e, &local)).collect();
        let accumulated: Vec<EdgeSpec> = l.accumulated.iter().map(|e| mk_edge(it, e, &local)).collect();
        let resolve = |it: &mut StrToId, v: &Vec<(String, String)>| -> Vec<(Sym, NodeId)> {
            v.iter().filter_map(|(n, d)| local.get(d.as_str()).map(|&id| (it.intern(n), id))).collect()
        };
        let loop_back = resolve(it, &l.loop_back);
        let external_inputs = resolve(it, &l.external_inputs);
        loop_specs.push(LoopSpec {
            name: it.intern(&l.name), max_iters: l.max_iters, parent,
            member_nodes, child_loops: Vec::new(),
            output_names: outputs.iter().map(|e| e.name).collect(),
            accum_names: accumulated.iter().map(|e| e.name).collect(),
            outputs, accumulated, loop_back, external_inputs,
        });
    }
    for i in 0..loop_specs.len() {
        if let Some(p) = loop_specs[i].parent {
            loop_specs[p as usize].child_loops.push(i as LoopId);
        }
    }
    // A node belongs to exactly one loop. Two claims used to resolve
    // last-writer-wins, the outermost, and deadlocked nested loops.
    for (i, l) in loop_specs.iter().enumerate() {
        for &m in &l.member_nodes {
            if let Some(prev) = node_specs[m as usize].loop_id {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "node {} is claimed by loops {} and {}; member_nodes must \
                     list only the nodes a loop owns directly",
                    it.name(node_specs[m as usize].name),
                    it.name(loop_specs[prev as usize].name),
                    it.name(l.name),
                )));
            }
            node_specs[m as usize].loop_id = Some(i as LoopId);
        }
    }

    let owned: Vec<NodeId> = loop_specs.iter().flat_map(|l| l.member_nodes.clone()).collect();
    let root_nodes: Vec<NodeId> = (0..nodes.len() as NodeId)
        .filter(|n| !owned.contains(n)).collect();
    let root_loops: Vec<LoopId> = loop_specs.iter().enumerate()
        .filter(|(_, l)| l.parent.is_none()).map(|(i, _)| i as LoopId).collect();

    let by_name = node_specs.iter().enumerate().map(|(i, n)| (n.name, i as NodeId)).collect();
    let loop_by_name = loop_specs.iter().enumerate().map(|(i, l)| (l.name, i as LoopId)).collect();
    Ok(Arc::new(CompiledGraph {
        nodes: node_specs, loops: loop_specs, by_name, loop_by_name, root_nodes, root_loops,
    }))
}