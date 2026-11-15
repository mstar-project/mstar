//! Sentence chunking for `/v1/audio/speech`, ported from
//! `api_server/openai/speech_chunking.py` (`split_sentences`) and
//! `serving_speech.py` (`_plan_chunks`, `_chunk_kwargs`).
//!
//! Long inputs are split into sentence groups that are synthesized as separate
//! requests and played back in order. The rules are Python's: lengths count
//! characters (`len` of a str), and whitespace is what `str.isspace` (and so
//! `re`'s `\s` and `str.split()`) calls whitespace.

use serde_json::{Map, Value};

use crate::adapters::SpeechChunking;
use crate::unicode_marks::MARK_RANGES;

const LATIN_ENDS: [char; 4] = ['.', '!', '?', ';'];
const CLOSERS: [char; 6] = ['"', '\'', '”', '’', ')', ']'];
const CJK_ENDS: [char; 4] = ['。', '！', '？', '；'];
// A Latin clause mark cuts only before whitespace; a full-width one always does.
const LATIN_CLAUSE_ENDS: [char; 3] = [',', ';', ':'];
const CJK_CLAUSE_ENDS: [char; 3] = ['，', '；', '：'];

/// Python's `str.isspace`: Unicode White_Space plus the separators U+001C..U+001F.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `unicodedata.category(c).startswith("M")`, over the table
/// `scripts/gen_unicode_marks.py` reads out of CPython's `unicodedata` (Unicode
/// 15.0.0 on the 3.12 the repository tests on, and on 3.13). `regex`'s `\p{M}`
/// asks the same question of the crate's own Unicode tables, which are newer
/// and differ on 51 code points, so the two frontends would cut long text in
/// different places whenever one of those lands on a cut.
/// `test/modular/test_speech_chunking.py` compares the table with the running
/// interpreter's `unicodedata`, over every code point.
fn is_mark(c: char) -> bool {
    let c = c as u32;
    // the ranges are sorted and disjoint, so the last one starting at or below
    // `c` is the only one that can hold it
    let after = MARK_RANGES.partition_point(|&(low, _)| low <= c);
    after > 0 && c <= MARK_RANGES[after - 1].1
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

/// `re.split(r"(?<=[,;:])\s+|(?<=[，；：])\s*", s)`: a cut after each clause
/// mark, which drops the whitespace that follows it. A Latin mark cuts only
/// when whitespace follows it, so "4,500,000" and "9:45" stay whole; a
/// full-width mark needs no space after it, so its cut can take none.
fn split_clauses(s: &[char]) -> Vec<&[char]> {
    let mut parts = Vec::new();
    let (mut last, mut i) = (0, 1);
    while i <= s.len() {
        let mark = s[i - 1];
        let cuts = if LATIN_CLAUSE_ENDS.contains(&mark) {
            i < s.len() && is_space(s[i])
        } else {
            CJK_CLAUSE_ENDS.contains(&mark)
        };
        if cuts {
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
///
/// Panics on a `max_chars` of zero, which Python refuses with
/// `ValueError("max_chars must be positive")`. It comes from the adapter, so a
/// zero is a bad build rather than anything a client can ask for.
pub fn split_sentences(text: &str, max_chars: usize, min_chars: usize) -> Vec<String> {
    assert!(max_chars > 0, "max_chars must be positive");
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

/// How an adapter cuts long input (its `split_speech_text`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Splitter {
    /// `split_sentences(text, max_chars=speech_chunk_max_chars)`.
    Sentences,
    /// `CosyVoice3Adapter.split_speech_text`.
    CosyVoice3,
}

/// `CosyVoice3Adapter.split_speech_text`: Chinese text as the reference's
/// `split_paragraph(lang="zh")` cuts it, anything else as sentences sized by
/// an estimate of its BPE tokens.
fn cosyvoice3_split(text: &str, max_chars: usize) -> Vec<String> {
    if text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)) {
        return cosyvoice3_split_chinese(text);
    }
    // `_estimated_tokens`: an upper estimate of the Qwen2 BPE token count
    let tokens: f64 = text
        .chars()
        .map(|c| if (c as u32) < 0x80 { 0.25 } else if (c as u32) < 0x800 { 0.5 } else { 1.0 })
        .sum();
    let chars_per_token = text.chars().count() as f64 / tokens.max(1.0);
    let max = max_chars.min(80usize.max((80.0 * chars_per_token) as usize));
    split_sentences(text, max, (20.0 * chars_per_token) as usize)
}

/// `re.findall` of `CosyVoice3Adapter._ZH_SENTENCE` (or, with `commas`,
/// `_ZH_CLAUSE`): the shortest run up to a CJK sentence or clause mark (with
/// one closing quote), or a Latin one before whitespace or the end, then the
/// rest; and the empty match at the end that `findall` also returns.
fn cosyvoice3_findall(text: &[char], commas: bool) -> Vec<Vec<char>> {
    let cjk = |c: char| "。？！；：、".contains(c) || (commas && c == '，');
    let latin = |c: char| ".?!;".contains(c) || (commas && c == ',');
    let n = text.len();
    let mut found = Vec::new();
    let mut start = 0;
    while start < n {
        let mut end = n;
        for i in start..n {
            if cjk(text[i]) {
                end = i + 1 + usize::from(text.get(i + 1).is_some_and(|&q| q == '"' || q == '”'));
                break;
            }
            if latin(text[i]) && (i + 1 == n || is_space(text[i + 1])) {
                end = i + 1;
                break;
            }
        }
        found.push(text[start..end].to_vec());
        start = end;
    }
    found.push(Vec::new());
    found
}

/// `CosyVoice3Adapter._group_chinese`.
fn cosyvoice3_group(sentences: Vec<Vec<char>>) -> Vec<Vec<char>> {
    let mut pieces: Vec<Vec<char>> = Vec::new();
    let mut current: Vec<char> = Vec::new();
    for sentence in sentences {
        if current.len() + sentence.len() > 80 && current.len() > 60 {
            pieces.push(std::mem::take(&mut current));
        }
        current.extend(sentence);
    }
    if !strip(&current).is_empty() {
        match pieces.last_mut() {
            Some(last) if current.len() < 20 => last.extend(current),
            _ => pieces.push(current),
        }
    }
    pieces.into_iter().map(|p| strip(&p).to_vec()).collect()
}

/// `CosyVoice3Adapter._split_chinese`: a piece over 300 characters is cut again
/// at commas, after its clauses are cut evenly into at most 80.
fn cosyvoice3_split_chinese(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut pieces = Vec::new();
    for piece in cosyvoice3_group(cosyvoice3_findall(&chars, false)) {
        if piece.len() <= 300 {
            pieces.push(piece);
            continue;
        }
        let clauses = cosyvoice3_findall(&piece, true);
        pieces.extend(cosyvoice3_group(clauses.iter().flat_map(|c| cut_evenly(c, 80)).collect()));
    }
    pieces.into_iter().map(|p| p.into_iter().collect()).collect()
}

/// `_plan_chunks`: the texts to synthesize, in playback order (`[input]` when
/// unchunked). `requested` is the request's `sentence_chunking`: `false`
/// never splits, any other value always does, and absent or `null` splits
/// inputs of at least the adapter's `min_chars`.
///
/// A `max_chars` of zero is refused here with the text Python's `ValueError`
/// carries, and only where Python would raise it: an input this never splits is
/// returned whole on both sides. The route answers the refusal as the Python
/// one answers that `ValueError`, a 400 with the message.
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
    if chunking.max_chars == 0 {
        return Err("max_chars must be positive".to_string());
    }
    let chunks = match chunking.splitter {
        Splitter::Sentences => split_sentences(input, chunking.max_chars, 24),
        Splitter::CosyVoice3 => cosyvoice3_split(input, chunking.max_chars),
    };
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
        SpeechChunking { min_chars: Some(600), max_chars: 400, lookahead: 2, max_pieces: 32, splitter: Splitter::Sentences };

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
    fn the_mark_table_is_canonical_and_answers_for_the_interpreter() {
        for (i, &(low, high)) in MARK_RANGES.iter().enumerate() {
            assert!(low <= high && high <= char::MAX as u32, "{i}: {low:#x}..={high:#x}");
            if i > 0 {
                let before = MARK_RANGES[i - 1].1;
                assert!(before + 1 < low, "{i}: {before:#x} runs into {low:#x}");
            }
        }
        // the ends of a range, a code point between two, and non-marks around them
        assert!(is_mark('\u{300}') && is_mark('\u{36f}') && is_mark('\u{310}'));
        assert!(!is_mark('\u{2ff}') && !is_mark('\u{370}') && !is_mark('a'));
        // unassigned in the interpreter's Unicode 15.0.0, a mark in the one
        // `regex` carries: the table answers for the interpreter. A table
        // regenerated from an interpreter whose Unicode has U+0897 assigned
        // moves this case and the even-cut one below.
        assert!(!is_mark('\u{897}') && is_mark('\u{898}'));
    }

    #[test]
    fn an_even_cut_on_a_code_point_only_regex_calls_a_mark_does_not_move() {
        // `unicodedata` has U+0897 unassigned, so Python cuts at 267 and so must we
        let text = format!("{}\u{897}{}", "a".repeat(267), "b".repeat(533));
        let chunks = split_sentences(&text, 400, 24);
        assert_eq!(chunks.iter().map(|c| c.chars().count()).collect::<Vec<_>>(), [267, 267, 267]);
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    #[should_panic(expected = "max_chars must be positive")]
    fn a_max_chars_of_zero_is_refused_as_python_refuses_it() {
        split_sentences("a. b.", 0, 24);
    }

    #[test]
    fn a_max_chars_of_zero_is_a_refusal_where_python_raises_and_nowhere_else() {
        let zero = SpeechChunking { max_chars: 0, ..CHUNKED };
        let err = plan_chunks(None, &zero, &(0..12).map(sentence).collect::<Vec<_>>().join(" "));
        assert_eq!(err.unwrap_err(), "max_chars must be positive");
        // what Python never splits it never measures either
        assert_eq!(plan_chunks(Some(&json!(false)), &zero, "Hi.").unwrap(), ["Hi."]);
        assert_eq!(plan_chunks(None, &zero, "Hi.").unwrap(), ["Hi."]);
    }

    #[test]
    fn clause_marks_cut_before_spaces() {
        let text = format!("{},{}", "a ".repeat(150).trim_end(), " b".repeat(100));
        let chunks = split_sentences(&text, 400, 24);
        assert_eq!(chunks, [format!("{},", "a ".repeat(150).trim_end()), "b ".repeat(100).trim_end().to_string()]);
    }

    #[test]
    fn a_clause_mark_with_no_space_after_it_keeps_a_number_whole() {
        // Python's `test_overlong_sentence_keeps_its_numbers_whole`, piece for piece
        let sentence = "The budget for the whole program across all regional offices is $4,500,000 \
                        and the review starts at 9:45 on the third floor near the main entrance hall";
        assert_eq!(split_sentences(sentence, 60, 0), [
            "The budget for the whole program across all regional offices",
            "is $4,500,000 and the review starts at 9:45 on the third",
            "floor near the main entrance hall",
        ]);
        // a full-width comma still cuts with no space after it
        let zh = "第一部分已经完成，第二部分正在进行，第三部分将在下个月开始，第四部分还在计划之中";
        assert_eq!(split_sentences(zh, 20, 0), [
            "第一部分已经完成， 第二部分正在进行，",
            "第三部分将在下个月开始，",
            "第四部分还在计划之中",
        ]);
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
