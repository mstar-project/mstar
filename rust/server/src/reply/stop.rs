//! Stop strings, matched on the decoded answer as it streams.

/// Ends a text at the first of its stops to appear. `feed` holds back the end
/// that may be the start of a stop (at most the longest stop less one
/// character) and returns the text that may go out, with the stop once one has
/// appeared; the text from the stop on never goes out. The request caps the
/// stops' number and length, as each feed scans every one.
pub struct StopMatcher {
    stops: Vec<String>,
    held: String,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            held: String::new(),
        }
    }

    pub fn feed(&mut self, fed: &str) -> (String, Option<String>) {
        let mut text = std::mem::take(&mut self.held);
        text.push_str(fed);
        // the stop that completes first ends the text; of two that end together,
        // the longer started first
        let hit = self
            .stops
            .iter()
            .filter_map(|s| {
                text.find(s.as_str())
                    .map(|i| (i + s.len(), std::cmp::Reverse(s.len()), s))
            })
            .min();
        if let Some((end, std::cmp::Reverse(len), stop)) = hit {
            text.truncate(end - len);
            return (text, Some(stop.clone()));
        }
        let keep = self
            .stops
            .iter()
            .map(|s| started(&text, s))
            .max()
            .unwrap_or(0);
        self.held = text.split_off(text.len() - keep);
        (text, None)
    }

    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

/// Length of the longest end of `text` that begins `stop`, short of all of it.
fn started(text: &str, stop: &str) -> usize {
    let Some(&first) = stop.as_bytes().first() else {
        return 0;
    };
    let from = text.len().saturating_sub(stop.len() - 1);
    // such an end starts at a byte equal to the stop's first, which no UTF-8
    // continuation byte is, so it starts a character
    text.as_bytes()[from..]
        .iter()
        .enumerate()
        .filter(|&(_, &b)| b == first)
        .map(|(i, _)| &text[from + i..])
        .find(|end| stop.starts_with(end))
        .map_or(0, str::len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stopped(stops: &[&str], text: &str, size: usize) -> (String, Option<String>) {
        let mut m = StopMatcher::new(stops.iter().map(|s| s.to_string()).collect());
        let chars: Vec<char> = text.chars().collect();
        let mut out = String::new();
        for piece in chars.chunks(size) {
            let (emitted, stop) = m.feed(&piece.iter().collect::<String>());
            out.push_str(&emitted);
            if stop.is_some() {
                return (out, stop);
            }
        }
        out.push_str(&m.flush());
        (out, None)
    }

    #[test]
    fn a_stop_string_ends_the_text_in_any_chunking() {
        for (stops, text, want, stop) in [
            (&["END"][..], "abcENDdef", "abc", Some("END")),
            (&["END"], "abcENdef", "abcENdef", None),
            (&["xyz", "y"], "axyz", "ax", Some("y")), // the stop that completes first wins
            (&["bc", "abc"], "zabcd", "z", Some("abc")), // of two ending together, the longer
            (&["\n\n"], "a\nb\n\nc", "a\nb", Some("\n\n")),
            (&["é😀"], "aé😀b", "a", Some("é😀")),
        ] {
            for size in [1, 2, 3, text.chars().count()] {
                let got = stopped(stops, text, size);
                assert_eq!(
                    got,
                    (want.to_string(), stop.map(str::to_string)),
                    "{text:?} in {size}s"
                );
            }
        }
    }

    #[test]
    fn a_stop_string_holds_back_only_what_may_start_it() {
        let mut m = StopMatcher::new(vec!["</answer>".to_string()]);
        assert_eq!(m.feed("plain text"), ("plain text".to_string(), None));
        assert_eq!(m.feed(" and </ans"), (" and ".to_string(), None));
        assert_eq!(m.feed("wer"), (String::new(), None));
        assert_eq!(m.feed(">!"), (String::new(), Some("</answer>".to_string())));
    }
}
