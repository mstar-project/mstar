//! Sentence chunking for `/v1/audio/speech`, ported from
//! `api_server/openai/speech_chunking.py` (`split_sentences`) and
//! `serving_speech.py` (`_plan_chunks`, `_chunk_kwargs`).
//!
//! Long inputs are split into sentence groups that are synthesized as separate
//! requests and played back in order. The rules are Python's: lengths count
//! characters (`len` of a str), and whitespace is what `str.isspace` (and so
//! `re`'s `\s` and `str.split()`) calls whitespace.

use std::sync::OnceLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::adapters::SpeechChunking;

const LATIN_ENDS: [char; 4] = ['.', '!', '?', ';'];
const CLOSERS: [char; 6] = ['"', '\'', '”', '’', ')', ']'];
const CJK_ENDS: [char; 4] = ['。', '！', '？', '；'];
const CLAUSE_ENDS: [char; 6] = [',', ';', ':', '，', '；', '：'];

/// Python's `str.isspace`: Unicode White_Space plus the separators U+001C..U+001F.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `unicodedata.category(c).startswith("M")`.
fn is_mark(c: char) -> bool {
    static MARK: OnceLock<Regex> = OnceLock::new();
    MARK.get_or_init(|| Regex::new(r"\p{M}").unwrap()).is_match(c.encode_utf8(&mut [0; 4]))
}

fn strip(s: &[char]) -> &[char] {
    let start = s.iter().position(|&c| !is_space(c)).unwrap_or(s.len());
    let end = s.iter().rposition(|&c| !is_space(c)).map_or(start, |i| i + 1);
    &s[start..end]
}

/// `" ".join(s.split())`.
fn squeeze(s: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    for word in s.split(|&c| is_space(c)).filter(|w| !w.is_empty()) {
        if !out.is_empty() {
            out.push(' ');
        }
        out.extend_from_slice(word);
    }
    out
}

/// Where each match of `_SENTENCE` ends: the shortest run of at least one
/// character up to a Latin terminator (and any closing quotes or brackets)
/// before whitespace or the end, a CJK terminator, a paragraph break, or the
/// end. The matches tile the text.
fn sentence_ends(text: &[char]) -> Vec<usize> {
    let n = text.len();
    let mut ends = Vec::new();
    let mut start = 0;
    while start < n {
        let mut end = n;
        for e in start + 1..n {
            let c = text[e];
            if LATIN_ENDS.contains(&c) {
                let k = e + 1 + text[e + 1..].iter().take_while(|c| CLOSERS.contains(c)).count();
                if k == n || is_space(text[k]) {
                    end = k;
                    break;
                }
            } else if CJK_ENDS.contains(&c) {
                end = e + 1;
                break;
            } else if c == '\n' {
                let rest = text[e + 1..].iter().skip_while(|&&c| c == ' ' || c == '\t');
                if rest.take(1).any(|&c| c == '\n') {
                    end = e;
                    break;
                }
            }
        }
        ends.push(end);
        start = end;
    }
    ends
}

/// Near-equal pieces of at most `max_chars`, none starting on a combining mark.
fn cut_evenly(text: &[char], max_chars: usize) -> Vec<Vec<char>> {
    let mut pieces = Vec::new();
    let mut start = 0;
    while text.len() - start > max_chars {
        let left = text.len() - start;
        let n = left.div_ceil(max_chars);
        let end = start + left.div_ceil(n);
        let mut cut = end;
        while cut > start && is_mark(text[cut]) {
            cut -= 1;
        }
        if cut == start {
            cut = end;
        }
        pieces.push(text[start..cut].to_vec());
        start = cut;
    }
    pieces.push(text[start..].to_vec());
    pieces
}

/// `re.split(r"(?<=[,;:，；：])\s*", s)`: a cut after each clause mark, which
/// drops the whitespace that follows it.
fn split_clauses(s: &[char]) -> Vec<&[char]> {
    let mut parts = Vec::new();
    let (mut last, mut i) = (0, 1);
    while i <= s.len() {
        if CLAUSE_ENDS.contains(&s[i - 1]) {
            parts.push(&s[last..i]);
            let mut j = i;
            while j < s.len() && is_space(s[j]) {
                j += 1;
            }
            last = j;
            i = j;
        }
        i += 1;
    }
    parts.push(&s[last.min(s.len())..]);
    parts
}

/// One over-long sentence split at clause boundaries, then at spaces, then
/// (for text with neither) into near-equal pieces of at most `max_chars`.
fn hard_wrap(sentence: Vec<char>, max_chars: usize) -> Vec<Vec<char>> {
    let mut pieces = vec![sentence];
    let splitters: [fn(&[char]) -> Vec<&[char]>; 2] =
        [split_clauses, |s| s.split(|&c| is_space(c)).collect()];
    for split in splitters {
        let mut wrapped = Vec::new();
        for piece in pieces {
            if piece.len() <= max_chars {
                wrapped.push(piece);
                continue;
            }
            let mut current: Vec<char> = Vec::new();
            for part in split(&piece).into_iter().map(strip).filter(|p| !p.is_empty()) {
                if !current.is_empty() && current.len() + 1 + part.len() > max_chars {
                    wrapped.push(std::mem::replace(&mut current, part.to_vec()));
                } else {
                    if !current.is_empty() {
                        current.push(' ');
                    }
                    current.extend_from_slice(part);
                }
            }
            if !current.is_empty() {
                wrapped.push(current);
            }
        }
        pieces = wrapped;
    }
    pieces
        .into_iter()
        .flat_map(|p| if p.len() > max_chars { cut_evenly(&p, max_chars) } else { vec![p] })
        .collect()
}

/// Two sentences of one chunk: a space after a Latin sentence, none after a CJK one.
fn join(left: &mut Vec<char>, right: &[char]) {
    if !left.last().is_some_and(|c| CJK_ENDS.contains(c)) {
        left.push(' ');
    }
    left.extend_from_slice(right);
}

/// `split_sentences`: `text` grouped into sentence chunks of at most
/// `max_chars` characters. Sentences are cut only when one alone is longer;
/// a trailing fragment under `min_chars` is merged into its predecessor.
/// `[text]` when nothing needs splitting, `[]` for blank input.
pub fn split_sentences(text: &str, max_chars: usize, min_chars: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let text = strip(&chars);
    if text.is_empty() {
        return Vec::new();
    }
    if text.len() <= max_chars {
        return vec![text.iter().collect()];
    }
    let mut sentences = Vec::new();
    let mut start = 0;
    for end in sentence_ends(text) {
        let piece = squeeze(&text[start..end]);
        start = end;
        if piece.is_empty() {
            continue;
        }
        if piece.len() > max_chars {
            sentences.extend(hard_wrap(piece, max_chars));
        } else {
            sentences.push(piece);
        }
    }

    let mut chunks: Vec<Vec<char>> = Vec::new();
    let mut current: Vec<char> = Vec::new();
    for sentence in sentences {
        if !current.is_empty() && current.len() + 1 + sentence.len() > max_chars {
            chunks.push(std::mem::replace(&mut current, sentence));
        } else if current.is_empty() {
            current = sentence;
        } else {
            join(&mut current, &sentence);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    if let [.., before, last] = chunks.as_slice() {
        if last.len() < min_chars && before.len() + 1 + last.len() <= max_chars * 5 / 4 {
            let last = chunks.pop().unwrap();
            join(chunks.last_mut().unwrap(), &last);
        }
    }
    chunks.into_iter().map(|c| c.into_iter().collect()).collect()
}

/// `_plan_chunks`: the texts to synthesize, in playback order (`[input]` when
/// unchunked). `requested` is the request's `sentence_chunking`: `false`
/// never splits, any other value always does, and absent or `null` splits
/// inputs of at least the adapter's `min_chars`.
pub fn plan_chunks(
    requested: Option<&Value>,
    chunking: &SpeechChunking,
    input: &str,
) -> Result<Vec<String>, String> {
    let requested = requested.filter(|v| !v.is_null());
    let short = chunking.min_chars.is_none_or(|min| input.chars().count() < min);
    if requested == Some(&Value::Bool(false)) || (requested.is_none() && short) {
        return Ok(vec![input.to_string()]);
    }
    let chunks = split_sentences(input, chunking.max_chars, 24);
    if chunks.len() > chunking.max_pieces {
        return Err(format!(
            "input splits into {} sentence chunks; at most {} are allowed",
            chunks.len(),
            chunking.max_pieces
        ));
    }
    Ok(if chunks.is_empty() { vec![input.to_string()] } else { chunks })
}

/// `_chunk_kwargs`: a piece's model kwargs, the request's except that
/// `sentence_chunking` is dropped and a client seed advances per piece
/// (`offset_seed`, int64 wrapping), so piece 0 matches the unchunked request.
pub fn chunk_kwargs(model_kwargs: &Map<String, Value>, index: usize) -> Map<String, Value> {
    let mut kwargs = model_kwargs.clone();
    kwargs.remove("sentence_chunking");
    if index > 0 {
        if let Some(seed) = kwargs.get("seed").and_then(Value::as_i64) {
            kwargs.insert("seed".to_string(), Value::from(seed.wrapping_add(index as i64)));
        }
    }
    kwargs
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const CHUNKED: SpeechChunking =
        SpeechChunking { min_chars: Some(600), max_chars: 400, lookahead: 2, max_pieces: 32 };

    fn sentence(i: usize) -> String {
        format!("Sentence number {i} goes on for a little while, so that it has some length to it.")
    }

    #[test]
    fn short_text_is_one_chunk_and_blank_is_none() {
        assert_eq!(split_sentences("  Hello there.  ", 400, 24), ["Hello there."]);
        assert!(split_sentences(" \u{1c}\t ", 400, 24).is_empty());
    }

    #[test]
    fn sentences_group_up_to_the_limit_and_a_short_tail_merges() {
        let text: String = (0..12).map(sentence).collect::<Vec<_>>().join(" ") + " Okay.";
        let chunks = split_sentences(&text, 400, 24);
        assert!(chunks.iter().all(|c| c.chars().count() <= 400 * 5 / 4), "{chunks:?}");
        assert!(chunks.last().unwrap().ends_with("to it. Okay."), "{chunks:?}");
        assert_eq!(chunks.join(" "), text);
    }

    #[test]
    fn a_terminator_needs_whitespace_or_closers_after_it() {
        let text = format!("{} Version 3.5 is out.\" Next one!) {}", "x".repeat(370), "y".repeat(20));
        let chunks = split_sentences(&text, 400, 24);
        let want = [format!("{} Version 3.5 is out.\"", "x".repeat(370)), format!("Next one!) {}", "y".repeat(20))];
        assert_eq!(chunks, want);
    }

    #[test]
    fn cjk_sentences_join_without_a_space() {
        let text = "海は広い。".repeat(100);
        let chunks = split_sentences(&text, 400, 24);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks, ["海は広い。".repeat(79), "海は広い。".repeat(21)]);
    }

    #[test]
    fn a_long_run_without_spaces_is_cut_evenly_and_not_before_a_mark() {
        // even cuts land on a mark at 301 and 601, and move back one
        let text = format!("xy{}", "a\u{301}".repeat(450));
        let chunks = split_sentences(&text, 400, 24);
        assert_eq!(chunks.iter().map(|c| c.chars().count()).collect::<Vec<_>>(), [300, 300, 302]);
        assert!(chunks.iter().all(|c| !is_mark(c.chars().next().unwrap())), "{chunks:?}");
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn clause_marks_cut_before_spaces() {
        let text = format!("{},{}", "a ".repeat(150).trim_end(), " b".repeat(100));
        let chunks = split_sentences(&text, 400, 24);
        assert_eq!(chunks, [format!("{},", "a ".repeat(150).trim_end()), "b ".repeat(100).trim_end().to_string()]);
    }

    #[test]
    fn chunking_follows_the_request_and_the_adapter_threshold() {
        let long = (0..12).map(sentence).collect::<Vec<_>>().join(" ");
        assert!(long.chars().count() >= 600);
        let plan = |requested: Option<Value>, text: &str, chunking: &SpeechChunking| {
            plan_chunks(requested.as_ref(), chunking, text).unwrap().len()
        };
        assert!(plan(None, &long, &CHUNKED) > 1);
        assert_eq!(plan(Some(json!(false)), &long, &CHUNKED), 1);
        assert!(plan(Some(json!(null)), &long, &CHUNKED) > 1);
        let unchunked = SpeechChunking { min_chars: None, ..CHUNKED };
        assert_eq!(plan(None, &long, &unchunked), 1);
        // anything but false asks for it, as Python's `is False` test reads it
        assert!(plan(Some(json!(0)), &long, &unchunked) > 1);
        assert_eq!(plan(Some(json!(true)), "   ", &unchunked), 1);
    }

    #[test]
    fn too_many_pieces_is_refused_with_the_python_text() {
        let text = (0..200).map(sentence).collect::<Vec<_>>().join(" ");
        let err = plan_chunks(None, &CHUNKED, &text).unwrap_err();
        assert!(err.starts_with("input splits into ") && err.ends_with(" sentence chunks; at most 32 are allowed"), "{err}");
    }

    #[test]
    fn piece_kwargs_drop_the_switch_and_offset_the_seed() {
        let mk: Map<String, Value> =
            serde_json::from_value(json!({"seed": i64::MAX, "sentence_chunking": true, "voice": "ryan"})).unwrap();
        assert_eq!(Value::Object(chunk_kwargs(&mk, 0)), json!({"seed": i64::MAX, "voice": "ryan"}));
        assert_eq!(Value::Object(chunk_kwargs(&mk, 2)), json!({"seed": i64::MIN + 1, "voice": "ryan"}));
    }
}
