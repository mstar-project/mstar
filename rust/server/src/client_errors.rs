//! Client-visible text for an error: `mstar/utils/client_errors.py`'s
//! `scrub_paths`, which cuts each absolute path to its base name. Its regular
//! expressions use a lookbehind and a backreference, which `regex` lacks, so
//! this scans for the same matches.

/// `os.path.basename(path) or "<path>"`.
fn base_name(path: &str) -> &str {
    match path.rsplit('/').next() {
        Some(b) if !b.is_empty() => b,
        _ => "<path>",
    }
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `_ABS_PATH`'s path characters.
fn path_char(c: char) -> bool {
    !(c.is_whitespace() || "'\"`,;:()[]{}<>|".contains(c))
}

/// `_QUOTED_PATH.sub`: a quoted absolute path, taken whole.
fn scrub_quoted(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let q = chars[i];
        if matches!(q, '\'' | '"' | '`') {
            let mut j = i + 1;
            let starts = chars[j..].iter().take(7).collect::<String>() == "file://";
            if starts {
                j += 7;
            }
            if chars.get(j) == Some(&'/') {
                let mut k = j;
                while k < chars.len() && !matches!(chars[k], '\'' | '"' | '`' | '\n') {
                    k += 1;
                }
                if chars.get(k) == Some(&q) {
                    let path: String = chars[i + 1..k].iter().collect();
                    out.push(q);
                    out.push_str(base_name(&path));
                    out.push(q);
                    i = k + 1;
                    continue;
                }
            }
        }
        out.push(q);
        i += 1;
    }
    out
}

/// `scrub_paths`: each absolute file path as its base name, and a directory
/// path (one ending in "/") as `<path>`.
pub fn scrub_paths(text: &str) -> String {
    let text = scrub_quoted(text);
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let prev = i.checked_sub(1).map(|p| chars[p]);
        let file_url = chars[i..].iter().take(8).collect::<String>() == "file:///" && !prev.is_some_and(is_word);
        let bare = chars[i] == '/' && !prev.is_some_and(|p| is_word(p) || ".:/)]".contains(p));
        let start = if file_url { Some(i + 7) } else if bare { Some(i) } else { None };
        if let Some(slash) = start {
            let mut k = slash + 1;
            while k < chars.len() && path_char(chars[k]) {
                k += 1;
            }
            if k > slash + 1 {
                let path: String = chars[i..k].iter().collect();
                out.push_str(base_name(&path));
                i = k;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::scrub_paths;

    #[test]
    fn paths_become_their_base_names_as_python_cuts_them() {
        assert_eq!(scrub_paths("open /srv/up/a.wav failed"), "open a.wav failed");
        assert_eq!(scrub_paths("bad '/srv/my dir/a b.png'"), "bad 'a b.png'");
        assert_eq!(scrub_paths("see file:///srv/x.txt, ok"), "see x.txt, ok");
        assert_eq!(scrub_paths("dir /srv/up/ gone"), "dir <path> gone");
        assert_eq!(scrub_paths("a/b and http://h/p and 3/4"), "a/b and http://h/p and 3/4");
    }
}
