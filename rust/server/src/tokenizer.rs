//! Tokenize / detokenize via the HF `tokenizers` crate — the same
//! `tokenizer.json` a Python server loads, but off the GIL. Incremental
//! detokenization is the streaming-critical piece: as each generated
//! token-id arrives, emit only the *new* text, handling multi-byte /
//! partial-token boundaries.

use std::sync::Arc;

use tokenizers::Tokenizer;

pub struct Tok {
    inner: Tokenizer,
}

impl Tok {
    pub fn from_file(path: &str) -> Result<Self, String> {
        Tokenizer::from_file(path)
            .map(|inner| Self { inner })
            .map_err(|e| e.to_string())
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).unwrap_or_default()
    }

}

/// A streaming detokenizer that emits the incremental text per token,
/// buffering incomplete pieces (partial UTF-8 / sub-word) until they resolve
/// — the correct behavior for per-token SSE. Owns an `Arc<Tok>` so a single
/// instance can be threaded through an async stream's state (both the live
/// bridge path and the mock path use it — one decode implementation, no
/// hand-rolled prefix-diff at the call site).
///
/// Prefix-diff strategy: keep the running id list, re-decode, and emit the
/// suffix beyond what was already emitted. O(n) per step for clarity; the O(1)
/// `tokenizers::DecodeStream` is the drop-in when stream length warrants it.
pub struct IncrementalDecoder {
    tok: Arc<Tok>,
    ids: Vec<u32>,
    emitted: String,
}

impl IncrementalDecoder {
    pub fn new(tok: Arc<Tok>) -> Self {
        Self {
            tok,
            ids: Vec::new(),
            emitted: String::new(),
        }
    }

    /// Feed one token-id; return the new text to stream (None if this token
    /// didn't yet resolve to emittable text — e.g. a partial multi-byte char).
    pub fn step(&mut self, id: u32) -> Option<String> {
        self.ids.push(id);
        let full = self.tok.decode(&self.ids);
        if full.len() <= self.emitted.len() || !full.starts_with(&self.emitted) {
            // The decode didn't extend our stable prefix yet; hold.
            return None;
        }
        let piece = full[self.emitted.len()..].to_string();
        // decode() already yields valid UTF-8, so `full` is safe to commit.
        self.emitted = full;
        Some(piece)
    }
}
