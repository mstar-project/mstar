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
        // A token that ends mid-character decodes to a trailing U+FFFD; hold
        // it until the character completes, as vLLM and SGLang do. Committing
        // it would also stall the stream: later decodes no longer start with
        // `emitted`.
        if full.ends_with('\u{FFFD}')
            || full.len() <= self.emitted.len()
            || !full.starts_with(&self.emitted)
        {
            return None;
        }
        let piece = full[self.emitted.len()..].to_string();
        self.emitted = full;
        Some(piece)
    }

    /// At the end of the stream: whatever `step` still holds (a character
    /// that never completed comes out as U+FFFD).
    pub fn finish(&mut self) -> Option<String> {
        let full = self.tok.decode(&self.ids);
        let rest = full.strip_prefix(self.emitted.as_str())?.to_string();
        self.emitted = full;
        (!rest.is_empty()).then_some(rest)
    }
}

/// A byte-level BPE with one token per byte and no merges, so every byte of a
/// multi-byte character is its own token.
#[cfg(test)]
pub(crate) fn byte_tok() -> Arc<Tok> {
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;

    let vocab = ByteLevel::alphabet()
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), i as u32))
        .collect();
    let bpe = BPE::builder().vocab_and_merges(vocab, vec![]).build().unwrap();
    let mut inner = Tokenizer::new(bpe);
    inner.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    inner.with_decoder(Some(ByteLevel::new(false, false, false)));
    Arc::new(Tok { inner })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_characters_stream_whole() {
        let tok = byte_tok();
        let ids = tok.encode("a中😀b");
        assert_eq!(ids.len(), 9);
        let mut d = IncrementalDecoder::new(tok);
        let pieces: Vec<String> = ids.iter().filter_map(|&i| d.step(i)).collect();
        assert_eq!(pieces, vec!["a", "中", "😀", "b"]);
        assert_eq!(d.finish(), None);
    }

    #[test]
    fn unfinished_character_flushes_at_the_end() {
        let tok = byte_tok();
        let ids = tok.encode("ok😀");
        let mut d = IncrementalDecoder::new(tok);
        let pieces: String = ids[..ids.len() - 1].iter().filter_map(|&i| d.step(i)).collect();
        assert_eq!(pieces, "ok");
        assert_eq!(d.finish().as_deref(), Some("\u{FFFD}"));
    }
}
