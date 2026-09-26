//! Compiled, interned graph spec — the immutable half of Python's
//! GraphNode/Loop/registry objects. Built once per worker graph, shared by
//! every request behind an `Arc`. Splitting this out is what removes the
//! per-request `deepcopy(section)`.

use rustc_hash::FxHashMap;

pub type Sym = u32; // interned string (node name, signal name, worker id)

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
    pub fn name(&self, id: Sym) -> &str {
        &self.names[id as usize]
    }
}
