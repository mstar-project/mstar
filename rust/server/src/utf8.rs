//! Incremental UTF-8 decoding of a request's text chunks.
//!
//! A byte-level BPE token can end inside a character, so a text chunk is not
//! UTF-8 on its own. This decodes the concatenation of the chunks the way the
//! Python frontend does (`codecs.getincrementaldecoder("utf-8")("replace")`):
//! an incomplete trailing sequence waits for the next chunk, each maximal
//! invalid subsequence becomes one U+FFFD, and `finish` replaces whatever is
//! still held.

#[derive(Default)]
pub struct Utf8Stream {
    held: Vec<u8>,
}

impl Utf8Stream {
    /// The text `bytes` completes, holding back an incomplete last character.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        if self.held.is_empty() {
            if let Ok(s) = std::str::from_utf8(bytes) {
                return s.to_owned();
            }
        }
        let mut buf = std::mem::take(&mut self.held);
        buf.extend_from_slice(bytes);
        let mut out = String::with_capacity(buf.len());
        let mut rest = &buf[..];
        loop {
            match std::str::from_utf8(rest) {
                Ok(s) => {
                    out.push_str(s);
                    break;
                }
                Err(e) => {
                    let (valid, after) = rest.split_at(e.valid_up_to());
                    out.push_str(std::str::from_utf8(valid).unwrap());
                    match e.error_len() {
                        Some(n) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            rest = &after[n..];
                        }
                        // the input ends inside a character: wait for the rest
                        None => {
                            self.held = after.to_vec();
                            break;
                        }
                    }
                }
            }
        }
        out
    }

    /// At the end of the stream: the held bytes of a character that never
    /// completed, as U+FFFD (empty when nothing is held).
    pub fn finish(&mut self) -> String {
        let held = std::mem::take(&mut self.held);
        String::from_utf8_lossy(&held).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(chunks: &[&[u8]]) -> (Vec<String>, String) {
        let mut d = Utf8Stream::default();
        let pieces = chunks.iter().map(|c| d.push(c)).collect();
        (pieces, d.finish())
    }

    #[test]
    fn a_character_split_across_chunks_comes_out_whole_once_complete() {
        // 🙂 = F0 9F 99 82, split 1 + 2 + 1
        let (pieces, tail) = stream(&[b"a\xF0", b"\x9F\x99", b"\x82b"]);
        assert_eq!(pieces, ["a", "", "🙂b"]);
        assert_eq!(tail, "");
    }

    #[test]
    fn an_emoji_zwj_sequence_byte_by_byte_joins_to_the_text() {
        let text = "👩‍👩‍👧‍👦 and 𠀋𪚥 ok";
        let mut d = Utf8Stream::default();
        let joined: String = text.as_bytes().iter().map(|b| d.push(&[*b])).collect();
        assert_eq!(joined + &d.finish(), text);
    }

    // Expected strings are what Python's incremental decoder with "replace"
    // gives for the same chunks.
    #[test]
    fn invalid_bytes_are_replaced_as_python_replaces_them() {
        // a held prefix that the next chunk shows is invalid
        assert_eq!(stream(&[b"\xF0\x9F\x98", b"A"]), (vec!["".into(), "\u{FFFD}A".into()], "".into()));
        // a byte that can never start a character
        assert_eq!(stream(&[b"x\xFFy"]), (vec!["x\u{FFFD}y".into()], "".into()));
        // an overlong lead and a lone continuation byte: one U+FFFD each
        assert_eq!(stream(&[b"\xE0\x80z"]), (vec!["\u{FFFD}\u{FFFD}z".into()], "".into()));
        // a surrogate's encoding
        assert_eq!(stream(&[b"\xED\xA0\x80"]), (vec!["\u{FFFD}\u{FFFD}\u{FFFD}".into()], "".into()));
    }

    #[test]
    fn a_character_cut_off_by_the_end_of_the_stream_is_one_replacement() {
        let (pieces, tail) = stream(&[b"ok\xE4\xBD"]);
        assert_eq!(pieces, ["ok"]);
        assert_eq!(tail, "\u{FFFD}");
    }
}
