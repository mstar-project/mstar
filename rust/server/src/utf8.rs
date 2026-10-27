//! Decode a model's text chunks without splitting characters. Twin of
//! `mstar/utils/utf8_stream.py`: byte-level detokenizers send each token's
//! raw bytes as its own chunk, so one character can span two chunks, and a
//! per-chunk `from_utf8_lossy` turns both halves into U+FFFD.

/// One text stream's incremental decoder: `feed` returns the complete
/// characters so far and holds back an incomplete tail, `flush` releases that
/// tail when the stream ends (as U+FFFD if it never completed). Invalid bytes
/// become U+FFFD as in `String::from_utf8_lossy`.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn feed(&mut self, data: &[u8]) -> String {
        self.pending.extend_from_slice(data);
        let mut out = String::new();
        let mut start = 0;
        while start < self.pending.len() {
            match std::str::from_utf8(&self.pending[start..]) {
                Ok(s) => {
                    out.push_str(s);
                    start = self.pending.len();
                }
                Err(e) => {
                    let valid = start + e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[start..valid]).unwrap_or_default());
                    match e.error_len() {
                        Some(n) => {
                            out.push('\u{FFFD}');
                            start = valid + n;
                        }
                        None => {
                            start = valid;
                            break;
                        }
                    }
                }
            }
        }
        self.pending.drain(..start);
        out
    }

    pub fn flush(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "中文 😀" cut inside 中 and inside the emoji; the same cases as
    /// `test/modular/test_utf8_split_tokens.py`.
    fn pieces() -> Vec<Vec<u8>> {
        let data = "中文 😀".as_bytes();
        vec![data[..1].to_vec(), data[1..4].to_vec(), data[4..8].to_vec(), data[8..].to_vec()]
    }

    #[test]
    fn split_characters_come_out_whole() {
        let mut s = Utf8Stream::default();
        let out: Vec<String> = pieces().iter().map(|p| s.feed(p)).collect();
        assert_eq!(out, vec!["", "中", "文 ", "😀"]);
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn one_byte_at_a_time() {
        let mut s = Utf8Stream::default();
        let out: String = "a中😀b".as_bytes().iter().map(|b| s.feed(&[*b])).collect();
        assert_eq!(out + &s.flush(), "a中😀b");
    }

    #[test]
    fn truncated_tail_flushes_as_one_replacement() {
        let mut s = Utf8Stream::default();
        let emoji = "😀".as_bytes();
        assert_eq!(s.feed(b"ok"), "ok");
        assert_eq!(s.feed(&emoji[..3]), "");
        assert_eq!(s.flush(), "\u{FFFD}");
        assert_eq!(s.feed(b"x"), "x");
    }

    /// Expected strings are Python's `bytes.decode("utf-8", "replace")`.
    #[test]
    fn invalid_bytes_match_from_utf8_lossy_and_python() {
        let cases: [(&[u8], &str); 4] = [
            (b"a\xffb", "a\u{FFFD}b"),
            (b"\xe4\x41", "\u{FFFD}A"),
            (b"\xc0\x80z", "\u{FFFD}\u{FFFD}z"),
            (b"\xed\xa0\x80!", "\u{FFFD}\u{FFFD}\u{FFFD}!"),
        ];
        for (bad, python) in cases {
            let mut s = Utf8Stream::default();
            let got = s.feed(bad) + &s.flush();
            assert_eq!(got, String::from_utf8_lossy(bad), "{bad:?}");
            assert_eq!(got, python, "{bad:?}");
        }
    }
}
