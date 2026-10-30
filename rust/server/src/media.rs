//! Media decode/encode helpers, ported from mstar's `api_server/media_io.py`.
//!
//! Two directions:
//!  * **Inbound** — turn media referenced by an OpenAI-style request (`data:`
//!    URLs or base64 blobs) into files under the server's `upload_dir`, so the
//!    Python data-worker can read them by path (the same contract `/generate`
//!    uses for multipart uploads). `http(s)` fetch is SSRF surface: it is off
//!    unless `MSTAR_ALLOW_REMOTE=1`, and limited as in `save_remote_url` when
//!    on. Paths on the server are refused.
//!  * **Outbound** — wrap raw model audio output (16-bit PCM, no header) into a
//!    real container (WAV by default) and encode PNG image bytes as a `data:`
//!    URL for OpenAI chat image output.
//!
//! Only the stdlib + `base64` + `uuid` are needed (plus `url` and `rustls` for
//! the opt-in remote fetch, a minimal HTTP/1.1 GET whose every read is bounded
//! in size and by the fetch deadline); compressed audio containers
//! (mp3/flac/ogg) degrade to WAV since the base install has no encoder — the
//! Python side is where those are produced when `soundfile` is present.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::time::{Duration, Instant};

use base64::Engine as _;
use uuid::Uuid;

/// Map a MIME type (top-level or full) to a file extension for persisting.
fn ext_for(mime: &str) -> &'static str {
    match mime.to_ascii_lowercase().as_str() {
        "image/png" => ".png",
        "image/jpeg" | "image/jpg" => ".jpg",
        "image/webp" => ".webp",
        "image/gif" => ".gif",
        "image/bmp" => ".bmp",
        "image/tiff" => ".tiff",
        "audio/wav" | "audio/x-wav" | "audio/wave" => ".wav",
        "audio/mpeg" | "audio/mp3" => ".mp3",
        "audio/flac" => ".flac",
        "audio/ogg" => ".ogg",
        "audio/opus" => ".opus",
        "audio/aac" => ".aac",
        "audio/m4a" => ".m4a",
        "video/mp4" => ".mp4",
        "video/webm" => ".webm",
        "video/quicktime" => ".mov",
        "video/x-matroska" => ".mkv",
        "video/x-msvideo" | "video/avi" => ".avi",
        other => match other.split('/').next().unwrap_or("") {
            "image" => ".png",
            "audio" => ".wav",
            "video" => ".mp4",
            _ => ".bin",
        },
    }
}

/// Reverse of `ext_for`: infer a modality string from a local path extension.
fn modality_from_ext(suffix: &str) -> &'static str {
    match suffix.to_ascii_lowercase().as_str() {
        ".png" | ".jpg" | ".jpeg" | ".webp" | ".gif" | ".bmp" | ".tiff" => "image",
        ".wav" | ".mp3" | ".flac" | ".ogg" | ".opus" | ".aac" | ".m4a" => "audio",
        ".mp4" | ".webm" | ".mov" | ".mkv" | ".avi" => "video",
        _ => "unknown",
    }
}

/// Infer a modality string from a filename's extension (for `/generate`
/// multipart uploads, matching mstar's `_detect_modality`).
pub fn modality_from_filename(name: &str) -> &'static str {
    let suffix = Path::new(name)
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    modality_from_ext(&suffix)
}

/// Map a MIME type to one of the modality strings (image/audio/video).
pub fn modality_from_mime(mime: &str) -> &'static str {
    match mime.split('/').next().unwrap_or("").to_ascii_lowercase().as_str() {
        "image" => "image",
        "audio" => "audio",
        "video" => "video",
        _ => "unknown",
    }
}

// ---------------------------------------------------------------------------
// Inbound: persist request media into upload_dir, return (modality, path)
// ---------------------------------------------------------------------------

/// Decode base64 the way Python's `base64.b64decode(..., validate=False)` does:
/// non-alphabet bytes are discarded rather than erroring. In practice the only
/// such bytes are the interior newlines of line-wrapped payloads (what the
/// `base64` CLI and `base64.encodebytes` emit by default), so stripping ASCII
/// whitespace before a strict decode matches Python without accepting garbage.
fn decode_b64_lenient(payload: &str) -> Result<Vec<u8>, base64::DecodeError> {
    let compact: String = payload.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    base64::engine::general_purpose::STANDARD.decode(compact.as_bytes())
}

fn save_bytes(raw: &[u8], mime: &str, upload_dir: &Path) -> std::io::Result<(String, String)> {
    std::fs::create_dir_all(upload_dir)?;
    let path = upload_dir.join(format!("{}{}", Uuid::new_v4(), ext_for(mime)));
    std::fs::write(&path, raw)?;
    Ok((modality_from_mime(mime).to_string(), path.to_string_lossy().into_owned()))
}

/// Persist a `data:<mime>;base64,<payload>` URL. Returns (modality, path).
fn save_data_url(data_url: &str, upload_dir: &Path) -> Result<(String, String), String> {
    let (header, payload) = data_url
        .split_once(',')
        .ok_or_else(|| "Malformed data URL: missing payload".to_string())?;
    if payload.is_empty() {
        return Err("Malformed data URL: missing payload".to_string());
    }
    // header = "data:<mime>[;base64]"
    let mime = header
        .strip_prefix("data:")
        .unwrap_or(header)
        .split(';')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("application/octet-stream");
    let raw = decode_b64_lenient(payload)
        .map_err(|e| format!("data URL base64 decode failed: {e}"))?;
    save_bytes(&raw, mime, upload_dir).map_err(|e| e.to_string())
}

/// Persist a bare base64 blob with a known `fmt` (e.g. "wav"). Returns
/// (modality_hint, path). Mirrors `save_base64` for OpenAI-native `input_audio`.
pub fn save_base64(
    b64: &str,
    fmt: &str,
    modality_hint: &str,
    upload_dir: &Path,
) -> Result<(String, String), String> {
    std::fs::create_dir_all(upload_dir).map_err(|e| e.to_string())?;
    let raw = decode_b64_lenient(b64)
        .map_err(|e| format!("input_audio base64 decode failed: {e}"))?;
    // Sanitize the client-controlled `format`: keep only alphanumerics, so
    // a value like `../../etc/x` cannot inject path separators into the
    // upload path (arbitrary write, mirroring the multipart filename fix).
    let clean: String = fmt
        .trim_start_matches('.')
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect();
    let ext = if clean.is_empty() {
        ".bin".to_string()
    } else {
        format!(".{clean}")
    };
    let path = upload_dir.join(format!("{}{}", Uuid::new_v4(), ext));
    std::fs::write(&path, &raw).map_err(|e| e.to_string())?;
    Ok((modality_hint.to_string(), path.to_string_lossy().into_owned()))
}

const REMOTE_MAX_BYTES: u64 = 64 << 20;
const REMOTE_MAX_REDIRECTS: usize = 3;
const REMOTE_USER_AGENT: &str = "mstar";

// Refused unless the host is allowlisted: loopback, private, link-local (cloud
// metadata at 169.254.169.254), shared/CGNAT, multicast, reserved. The same
// list is in mstar/api_server/media_io.py.
const BLOCKED_V4: [(Ipv4Addr, u32); 11] = [
    (Ipv4Addr::new(0, 0, 0, 0), 8),
    (Ipv4Addr::new(10, 0, 0, 0), 8),
    (Ipv4Addr::new(100, 64, 0, 0), 10),
    (Ipv4Addr::new(127, 0, 0, 0), 8),
    (Ipv4Addr::new(169, 254, 0, 0), 16),
    (Ipv4Addr::new(172, 16, 0, 0), 12),
    (Ipv4Addr::new(192, 0, 0, 0), 24),
    (Ipv4Addr::new(192, 168, 0, 0), 16),
    (Ipv4Addr::new(198, 18, 0, 0), 15),
    (Ipv4Addr::new(224, 0, 0, 0), 4),
    (Ipv4Addr::new(240, 0, 0, 0), 4),
];
const BLOCKED_V6: [(Ipv6Addr, u32); 7] = [
    (Ipv6Addr::UNSPECIFIED, 128),
    (Ipv6Addr::LOCALHOST, 128),
    (Ipv6Addr::new(0x64, 0xff9b, 1, 0, 0, 0, 0, 0), 48),
    (Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 0), 7),
    (Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 0), 10),
    (Ipv6Addr::new(0xfec0, 0, 0, 0, 0, 0, 0, 0), 10),
    (Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0), 8),
];

/// The IPv4 address an IPv4-mapped, 6to4 (2002::/16), NAT64 (64:ff9b::/96) or
/// IPv4-compatible (::/96) address reaches. NAT64 is not refused outright: on an
/// IPv6-only network with DNS64 every IPv4-only host resolves into it.
fn embedded_v4(v6: Ipv6Addr) -> Option<Ipv4Addr> {
    let bits = u128::from(v6);
    if let Some(v4) = v6.to_ipv4_mapped() {
        Some(v4)
    } else if bits >> 112 == 0x2002 {
        Some(Ipv4Addr::from((bits >> 80) as u32))
    } else if bits >> 32 == 0x64_ff9b_0000_0000_0000_0000 || bits >> 32 == 0 {
        Some(Ipv4Addr::from(bits as u32))
    } else {
        None
    }
}

fn blocked_address(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => BLOCKED_V4
            .iter()
            .any(|&(net, len)| u32::from(v4) >> (32 - len) == u32::from(net) >> (32 - len)),
        IpAddr::V6(v6) => match embedded_v4(v6) {
            Some(v4) => blocked_address(IpAddr::V4(v4)),
            None => BLOCKED_V6
                .iter()
                .any(|&(net, len)| u128::from(v6) >> (128 - len) == u128::from(net) >> (128 - len)),
        },
    }
}

/// Both URL parsers delete these anywhere in a URL, so a host written with them
/// would match the allowlist as if they were not there; such URLs are refused.
const URL_DROPPED: [char; 3] = ['\t', '\r', '\n'];

/// The host of `url` as written (lower-cased, without brackets or a trailing
/// dot), or `None` when the text has no `//` authority.
fn written_host(url: &str) -> Option<String> {
    let after_scheme = match url.split_once(':') {
        Some((scheme, rest)) if !scheme.is_empty() && scheme.bytes().all(|b| b.is_ascii_alphanumeric() || b"+-.".contains(&b)) => rest,
        _ => url,
    };
    let rest = after_scheme.strip_prefix("//")?;
    let authority = &rest[..rest.find(['/', '\\', '?', '#']).unwrap_or(rest.len())];
    let host_port = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let host = match host_port.strip_prefix('[') {
        Some(v6) => &v6[..v6.find(']')?],
        None => host_port.rsplit_once(':').map_or(host_port, |(h, _)| h),
    };
    Some(host.trim_end_matches('.').to_ascii_lowercase())
}

/// Limits on remote fetch: `MSTAR_REMOTE_ALLOWED_HOSTS` (comma-separated hosts
/// and `*.suffix` patterns; when set, only those hosts are fetched) and
/// `MSTAR_REMOTE_MAX_BYTES`.
struct RemotePolicy {
    allowed_hosts: Vec<String>,
    max_bytes: u64,
    timeout: Duration,
}

impl RemotePolicy {
    fn from_env() -> &'static RemotePolicy {
        static POLICY: OnceLock<RemotePolicy> = OnceLock::new();
        POLICY.get_or_init(|| RemotePolicy {
            allowed_hosts: std::env::var("MSTAR_REMOTE_ALLOWED_HOSTS")
                .unwrap_or_default()
                .split(',')
                .map(|h| {
                    let h = h.trim().trim_matches(|c| c == '[' || c == ']');
                    h.trim_end_matches('.').to_ascii_lowercase()
                })
                .filter(|h| !h.is_empty())
                .collect(),
            max_bytes: std::env::var("MSTAR_REMOTE_MAX_BYTES")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(REMOTE_MAX_BYTES),
            timeout: Duration::from_secs(30),
        })
    }

    /// `host` (as the URL parser prints it) equals an entry, or ends in `.suffix`
    /// for a `*.suffix` entry. Only a host the URL wrote that same way can match,
    /// so another spelling of a listed address (`2130706433`, `0x7f000001`,
    /// `127.1`) gets the address check instead, as in the Python frontend.
    fn allowlisted(&self, host: &str, written: Option<&str>) -> bool {
        written == Some(host)
            && self
                .allowed_hosts
                .iter()
                .any(|p| host == p || (p.starts_with("*.") && host.ends_with(&p[1..])))
    }
}

/// Log a fetch failure's details server-side; the client gets only `reason`.
fn fetch_error(host: &str, reason: &str, detail: impl std::fmt::Display) -> String {
    eprintln!("remote media fetch from {host} failed: {reason} {detail}");
    format!("remote media fetch failed: {reason}")
}

fn transport_error(host: &str, deadline: Instant, fired: bool, detail: impl std::fmt::Display) -> String {
    let reason = if fired || Instant::now() >= deadline { "timed out" } else { "connection failed" };
    fetch_error(host, reason, detail)
}

/// Download an `http(s)` URL into `upload_dir` (mstar's `save_remote_url`).
/// Hosts outside `MSTAR_REMOTE_ALLOWED_HOSTS` (when set) are refused, and so
/// is a host that is not allowlisted and resolves to a blocked address; the
/// connection goes to the vetted addresses. Every redirect hop is checked again
/// (at most `REMOTE_MAX_REDIRECTS`), the body is streamed to disk and capped at
/// `MSTAR_REMOTE_MAX_BYTES`, and the whole fetch takes at most 30 s. Errors
/// are short messages; the details are logged only.
fn save_remote_url(
    url: &str,
    upload_dir: &Path,
    policy: &RemotePolicy,
) -> Result<(String, String), String> {
    let deadline = Instant::now() + policy.timeout;
    if url.contains(URL_DROPPED) {
        return Err(fetch_error("", "invalid URL", format!("{url:?}")));
    }
    let mut written = written_host(url);
    let mut url = url::Url::parse(url).map_err(|e| fetch_error("", "invalid URL", e))?;
    let mut host = String::new();
    for _ in 0..=REMOTE_MAX_REDIRECTS {
        host = url
            .host_str()
            .unwrap_or("")
            .trim_start_matches('[')
            .trim_end_matches(']')
            .trim_end_matches('.')
            .to_ascii_lowercase();
        let allowlisted = policy.allowlisted(&host, written.as_deref());
        if !matches!(url.scheme(), "http" | "https") || host.is_empty() {
            return Err(fetch_error(&host, "invalid URL", &url));
        }
        if !policy.allowed_hosts.is_empty() && !allowlisted {
            return Err(fetch_error(&host, "host not allowed", "not in MSTAR_REMOTE_ALLOWED_HOSTS"));
        }
        let addrs = url
            .socket_addrs(|| None)
            .map_err(|e| fetch_error(&host, "connection failed", e))?;
        if !allowlisted && addrs.iter().any(|a| blocked_address(a.ip())) {
            return Err(fetch_error(&host, "host not allowed", format!("resolves to {addrs:?}")));
        }
        match fetch_hop(&url, &host, &addrs, upload_dir, policy.max_bytes, deadline)? {
            Hop::Redirect(next, location) => {
                // only a Location with neither a scheme nor an authority (`//`, or
                // `\\` as the URL parser also reads it) stays on this host as written
                let first = location.split(['/', '\\', '?', '#']).next().unwrap_or("");
                let relative = !first.contains(':')
                    && !matches!(location.as_bytes(), [b'/' | b'\\', b'/' | b'\\', ..]);
                if !relative {
                    written = written_host(&location);
                } else if next.host_str() != url.host_str() {
                    written = None;
                }
                url = next;
            }
            Hop::Saved(modality, path) => return Ok((modality, path)),
        }
    }
    Err(fetch_error(&host, "too many redirects", ""))
}

// Response line and header bounds, the same as Python's http.client.
const MAX_LINE: usize = 65536;
const MAX_HEADERS: usize = 100;

enum Hop {
    /// The next URL and the `Location` text it came from.
    Redirect(url::Url, String),
    Saved(String, String),
}

enum BodyError {
    TooLarge,
    Read(io::Error),
    Store(io::Error),
}

trait Conn: Read + Write {}
impl<T: Read + Write> Conn for T {}

/// Shuts a connected socket down at the deadline, so no read or write on it
/// (TLS handshake, headers, chunk framing, body) can outlast the fetch timeout.
struct Watchdog {
    fired: Arc<AtomicBool>,
    _cancel: mpsc::Sender<()>,
}

impl Watchdog {
    fn arm(sock: &TcpStream, deadline: Instant) -> io::Result<Watchdog> {
        let sock = sock.try_clone()?;
        let (cancel, cancelled) = mpsc::channel::<()>();
        let fired = Arc::new(AtomicBool::new(false));
        let flag = fired.clone();
        let left = deadline.saturating_duration_since(Instant::now());
        std::thread::spawn(move || {
            // dropping the Watchdog disconnects the channel and ends this wait early
            if let Err(mpsc::RecvTimeoutError::Timeout) = cancelled.recv_timeout(left) {
                flag.store(true, Ordering::SeqCst);
                let _ = sock.shutdown(Shutdown::Both);
            }
        });
        Ok(Watchdog { fired, _cancel: cancel })
    }

    fn fired(&self) -> bool {
        self.fired.load(Ordering::SeqCst)
    }
}

fn time_left(deadline: Instant) -> io::Result<Duration> {
    let left = deadline.saturating_duration_since(Instant::now());
    if left.is_zero() {
        Err(io::ErrorKind::TimedOut.into())
    } else {
        Ok(left)
    }
}

fn invalid(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what.to_string())
}

fn tls_config() -> Arc<rustls::ClientConfig> {
    static CONFIG: OnceLock<Arc<rustls::ClientConfig>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let roots = rustls::RootCertStore { roots: webpki_roots::TLS_SERVER_ROOTS.to_vec() };
            let config = rustls::ClientConfig::builder_with_provider(
                rustls::crypto::ring::default_provider().into(),
            )
            .with_protocol_versions(&[&rustls::version::TLS12, &rustls::version::TLS13])
            .expect("ring supports TLS 1.2 and 1.3")
            .with_root_certificates(roots)
            .with_no_client_auth();
            Arc::new(config)
        })
        .clone()
}

/// Connect to the first reachable vetted address (each attempt gets only what
/// is left of the deadline) and arm the watchdog before any TLS handshake.
fn connect(
    url: &url::Url,
    host: &str,
    addrs: &[SocketAddr],
    deadline: Instant,
) -> io::Result<(Box<dyn Conn>, Watchdog)> {
    let mut last = io::Error::new(io::ErrorKind::NotFound, "no address");
    for addr in addrs {
        let sock = match TcpStream::connect_timeout(addr, time_left(deadline)?) {
            Ok(sock) => sock,
            Err(e) => {
                last = e;
                continue;
            }
        };
        let watchdog = Watchdog::arm(&sock, deadline)?;
        if url.scheme() != "https" {
            return Ok((Box::new(sock), watchdog));
        }
        let name = rustls::pki_types::ServerName::try_from(host)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?
            .to_owned();
        let tls = rustls::ClientConnection::new(tls_config(), name).map_err(io::Error::other)?;
        return Ok((Box::new(rustls::StreamOwned::new(tls, sock)), watchdog));
    }
    Err(last)
}

/// One line of at most `MAX_LINE` bytes, without its line ending.
fn read_line(r: &mut impl BufRead) -> io::Result<String> {
    let mut line = Vec::new();
    r.take(MAX_LINE as u64 + 1).read_until(b'\n', &mut line)?;
    if line.len() > MAX_LINE {
        return Err(invalid("line too long"));
    }
    if line.is_empty() {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    Ok(String::from_utf8_lossy(&line).trim_end_matches(['\r', '\n']).to_string())
}

/// The status code and headers (names lower-cased) of the final response,
/// skipping `100 Continue`.
fn read_head(r: &mut impl BufRead) -> io::Result<(u16, Vec<(String, String)>)> {
    loop {
        let status = read_line(r)?;
        let mut parts = status.splitn(3, ' ');
        let (version, code) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
        let code = match code.parse::<u16>() {
            Ok(c) if version.starts_with("HTTP/") && code.len() == 3 && (100..=999).contains(&c) => c,
            _ => return Err(invalid("bad status line")),
        };
        let mut headers = Vec::new();
        loop {
            let line = read_line(r)?;
            if line.is_empty() {
                break;
            }
            if headers.len() == MAX_HEADERS {
                return Err(invalid("too many headers"));
            }
            if let Some((name, value)) = line.split_once(':') {
                headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
            }
        }
        if code != 100 {
            return Ok((code, headers));
        }
    }
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
}

/// Copy `want` bytes (to EOF when `None`) from `r` to `out`, failing once more
/// than `max` body bytes have arrived in all.
fn copy_some(
    r: &mut impl Read,
    out: &mut impl Write,
    want: Option<u64>,
    total: &mut u64,
    max: u64,
) -> Result<(), BodyError> {
    let mut buf = [0u8; 1 << 14];
    let mut left = want;
    loop {
        let cap = match left {
            Some(0) => return Ok(()),
            Some(n) => n.min(buf.len() as u64) as usize,
            None => buf.len(),
        };
        let n = match r.read(&mut buf[..cap]) {
            Ok(0) if left.is_none() => return Ok(()),
            Ok(0) => return Err(BodyError::Read(io::ErrorKind::UnexpectedEof.into())),
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            // a TLS peer that closes without close_notify, which Python's ssl also accepts
            Err(e) if left.is_none() && e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(BodyError::Read(e)),
        };
        *total += n as u64;
        if *total > max {
            return Err(BodyError::TooLarge);
        }
        out.write_all(&buf[..n]).map_err(BodyError::Store)?;
        if let Some(l) = left.as_mut() {
            *l -= n as u64;
        }
    }
}

fn copy_chunked(r: &mut impl BufRead, out: &mut impl Write, max: u64) -> Result<(), BodyError> {
    let mut total = 0u64;
    loop {
        let line = read_line(r).map_err(BodyError::Read)?;
        let size = line.split(';').next().unwrap_or("").trim();
        let size = u64::from_str_radix(size, 16).map_err(|_| BodyError::Read(invalid("bad chunk size")))?;
        if size == 0 {
            // trailers up to the blank line; EOF there is accepted, as in Python
            for _ in 0..=MAX_HEADERS {
                match read_line(r) {
                    Ok(line) if !line.is_empty() => continue,
                    Ok(_) => return Ok(()),
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
                    Err(e) => return Err(BodyError::Read(e)),
                }
            }
            return Err(BodyError::Read(invalid("too many trailers")));
        }
        if total.saturating_add(size) > max {
            return Err(BodyError::TooLarge);
        }
        copy_some(r, out, Some(size), &mut total, max)?;
        if !read_line(r).map_err(BodyError::Read)?.is_empty() {
            return Err(BodyError::Read(invalid("bad chunk end")));
        }
    }
}

/// The MIME type from `Content-Type`, or from the URL path extension when it
/// is missing or generic (mstar's fallback).
fn mime_for(content_type: Option<&str>, url: &url::Url) -> String {
    let mime = content_type.unwrap_or("").split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    if !mime.is_empty() && mime != "application/octet-stream" {
        return mime;
    }
    let suffix = PathBuf::from(url.path())
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    match modality_from_ext(&suffix) {
        "image" => format!("image/{}", suffix.trim_start_matches('.')),
        "audio" => format!("audio/{}", suffix.trim_start_matches('.')),
        "video" => format!("video/{}", suffix.trim_start_matches('.')),
        _ => "application/octet-stream".to_string(),
    }
}

/// One GET to the vetted `addrs`: the next URL for a redirect, or the body
/// saved under `upload_dir`.
fn fetch_hop(
    url: &url::Url,
    host: &str,
    addrs: &[SocketAddr],
    upload_dir: &Path,
    max_bytes: u64,
    deadline: Instant,
) -> Result<Hop, String> {
    let (stream, watchdog) =
        connect(url, host, addrs, deadline).map_err(|e| transport_error(host, deadline, false, e))?;
    let mut r = BufReader::new(stream);
    let target = &url[url::Position::BeforePath..url::Position::AfterQuery];
    let host_header = &url[url::Position::BeforeHost..url::Position::AfterPort];
    let request = format!(
        "GET {target} HTTP/1.1\r\nHost: {host_header}\r\nUser-Agent: {REMOTE_USER_AGENT}\r\n\
         Accept-Encoding: identity\r\nConnection: close\r\n\r\n"
    );
    let head = r
        .get_mut()
        .write_all(request.as_bytes())
        .and_then(|_| r.get_mut().flush())
        .and_then(|_| read_head(&mut r));
    let (code, headers) = head.map_err(|e| transport_error(host, deadline, watchdog.fired(), e))?;
    match code {
        301 | 302 | 303 | 307 | 308 => {
            let location = header(&headers, "location")
                .ok_or_else(|| fetch_error(host, "bad redirect", format!("{code} without Location")))?;
            if location.contains(URL_DROPPED) {
                return Err(fetch_error(host, "invalid URL", format!("{location:?}")));
            }
            return url
                .join(location)
                .map(|next| Hop::Redirect(next, location.to_string()))
                .map_err(|e| fetch_error(host, "invalid URL", e));
        }
        200..=299 => {}
        _ => return Err(fetch_error(host, &format!("upstream returned {code}"), "")),
    }
    let too_large = format!("too large (limit {max_bytes} bytes)");
    let chunked = header(&headers, "transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked"));
    // as http.client: a non-numeric length is ignored; one past u64 is still a length, over any cap
    let length = match header(&headers, "content-length") {
        Some(v) if !chunked && !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) => {
            Some(v.parse::<u64>().unwrap_or(u64::MAX))
        }
        _ => None,
    };
    if let Some(length) = length.filter(|&n| n > max_bytes) {
        return Err(fetch_error(host, &too_large, format!("Content-Length {length}")));
    }
    let mime = mime_for(header(&headers, "content-type"), url);
    std::fs::create_dir_all(upload_dir).map_err(|e| fetch_error(host, "could not save the file", e))?;
    let path = upload_dir.join(format!("{}{}", Uuid::new_v4(), ext_for(&mime)));
    let copied = std::fs::File::create(&path).map_err(BodyError::Store).and_then(|mut f| {
        if chunked {
            copy_chunked(&mut r, &mut f, max_bytes)?;
        } else {
            copy_some(&mut r, &mut f, length, &mut 0, max_bytes)?;
        }
        // a body that runs to EOF also "ends" when the watchdog shuts the socket
        if watchdog.fired() {
            return Err(BodyError::Read(io::ErrorKind::TimedOut.into()));
        }
        Ok(())
    });
    match copied {
        Ok(()) => Ok(Hop::Saved(modality_from_mime(&mime).to_string(), path.to_string_lossy().into_owned())),
        Err(e) => {
            let _ = std::fs::remove_file(&path);
            Err(match e {
                BodyError::TooLarge => fetch_error(host, &too_large, "body exceeded the limit"),
                BodyError::Read(e) => transport_error(host, deadline, watchdog.fired(), e),
                BodyError::Store(e) => fetch_error(host, "could not save the file", e),
            })
        }
    }
}

/// Resolve a media reference from a request: a data URL, or an `http(s)` URL
/// when `allow_remote` is set (remote fetch is SSRF surface, off by default).
/// Returns `(modality, path)`. A request cannot name a file on the server, not
/// even one under `upload_dir`: plain paths and other URL schemes are refused
/// without accessing client-named server files.
pub fn resolve_media_ref(
    reference: &str,
    upload_dir: &Path,
    allow_remote: bool,
) -> Result<(String, String), String> {
    if reference.starts_with("data:") {
        return save_data_url(reference, upload_dir);
    }
    let scheme = reference.split(':').next().unwrap_or("").to_ascii_lowercase();
    if scheme == "http" || scheme == "https" {
        if !allow_remote {
            return Err("Remote media fetch is disabled on this server".to_string());
        }
        return save_remote_url(reference, upload_dir, RemotePolicy::from_env());
    }
    Err("server file paths are refused; send media as a data URL or a file upload".to_string())
}

// ---------------------------------------------------------------------------
// Outbound: wrap raw model output for client surfaces
// ---------------------------------------------------------------------------

/// Wrap raw little-endian 16-bit PCM (the model's audio output) into a WAV blob.
pub fn pcm16_to_wav_bytes(pcm: &[u8], sample_rate: u32, num_channels: u16) -> Vec<u8> {
    let bits: u16 = 16;
    let byte_rate = sample_rate * num_channels as u32 * bits as u32 / 8;
    let block_align = num_channels * bits / 8;
    let data_len = pcm.len() as u32;
    let riff_len = 36 + data_len;

    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

/// Encode raw 16-bit PCM into `fmt`. Returns `(bytes, mime_type)`. `wav` and
/// `pcm` are produced here; compressed formats need an encoder the Rust
/// frontend doesn't carry, so they degrade to WAV (the Python side produces
/// compressed containers when `soundfile` is available).
pub fn pcm16_to_container(pcm: &[u8], sample_rate: u32, fmt: &str) -> (Vec<u8>, &'static str) {
    match fmt.to_ascii_lowercase().as_str() {
        "pcm" => (pcm.to_vec(), "audio/pcm"),
        "wav" => (pcm16_to_wav_bytes(pcm, sample_rate, 1), "audio/wav"),
        _ => (pcm16_to_wav_bytes(pcm, sample_rate, 1), "audio/wav"),
    }
}

/// A 44-byte WAV header with streaming (unknown-length) size fields. Emit this,
/// then 16-bit PCM frames as they arrive, to stream TTS over one HTTP response.
/// The `0xFFFFFFFF` placeholders signal an open-ended stream.
pub fn wav_stream_header(sample_rate: u32, num_channels: u16) -> Vec<u8> {
    let bits: u16 = 16;
    let byte_rate = sample_rate * num_channels as u32 * bits as u32 / 8;
    let block_align = num_channels * bits / 8;

    let mut out = Vec::with_capacity(44);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    out
}

/// Encode PNG image bytes (the model's image output) as a data URL.
pub fn png_to_data_url(png_bytes: &[u8]) -> String {
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png_bytes)
    )
}

/// base64-encode arbitrary bytes (audio deltas, container blobs) for JSON.
pub fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::Mutex;

    const PNG: &[u8] = b"\x89PNG\r\n\x1a\n0123456789";
    const LEAKY: &[u8] = b"upstream says: db.internal.corp 10.1.2.3 secret-token";
    const TOO_LARGE: &str = "too large (limit 1000 bytes)";

    enum Reply {
        Bytes(Vec<u8>),
        Hang,
        Trickle,
        Stall,
        /// write the prefix, then one byte every 50 ms for 3 s
        Drip(&'static [u8], u8),
    }

    fn http(status: &str, headers: &str, body: &[u8]) -> Reply {
        let head = format!(
            "HTTP/1.1 {status}\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        Reply::Bytes([head.as_bytes(), body].concat())
    }

    fn chunked(total: usize) -> Reply {
        let mut out = b"HTTP/1.1 200 OK\r\nContent-Type: video/mp4\r\n".to_vec();
        out.extend_from_slice(b"Transfer-Encoding: chunked\r\n\r\n");
        for _ in 0..total / 500 {
            out.extend_from_slice(&[b"1f4\r\n", &[0u8; 500][..], b"\r\n"].concat());
        }
        out.extend_from_slice(b"0\r\n\r\n");
        Reply::Bytes(out)
    }

    fn route(path: &str, port: u16) -> Reply {
        let redirect = |status: &str, to: &str| http(status, &format!("Location: {to}\r\n"), b"");
        match path {
            "/a.png" => http("200 OK", "Content-Type: image/png\r\n", PNG),
            "/untyped.png" => http("200 OK", "", PNG),
            "/generic.png" => http("200 OK", "Content-Type: application/octet-stream\r\n", PNG),
            "/r1" => redirect("301 Moved Permanently", "/r2"),
            "/r2" => redirect("307 Temporary Redirect", &format!("http://127.0.0.1:{port}/r3")),
            "/r3" => redirect("308 Permanent Redirect", "a.png"),
            "/loop" => redirect("302 Found", "/loop"),
            "/to-localhost" => redirect("302 Found", &format!("http://localhost:{port}/a.png")),
            "/to-short-form" => redirect("302 Found", &format!("//127.1:{port}/a.png")),
            "/to-backslash-authority" => redirect("302 Found", &format!("\\\\127.1:{port}/a.png")),
            "/to-tab" => redirect("302 Found", &format!("http://127.0.0.\t1:{port}/a.png")),
            "/to-file" => redirect("302 Found", "file:///etc/passwd"),
            // a body this size would never finish
            "/big.png" => Reply::Bytes(b"HTTP/1.1 200 OK\r\nContent-Length: 1000000000000\r\n\r\n".to_vec()),
            "/at.mp4" => chunked(1000),
            "/over.mp4" => chunked(5000),
            "/hang" => Reply::Hang,
            "/trickle" => Reply::Trickle,
            "/stall" => Reply::Stall,
            "/drip-head" => Reply::Drip(b"HTTP/1.1 200 OK\r\nX-Slow: ", b'a'),
            "/drip-chunk" => Reply::Drip(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n", b'0'),
            "/long-chunk-line" => {
                let mut out = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
                out.extend(std::iter::repeat(b'0').take(100_000));
                out.extend_from_slice(b"1\r\nx\r\n0\r\n\r\n");
                Reply::Bytes(out)
            }
            "/huge-length.png" => {
                Reply::Bytes(b"HTTP/1.1 200 OK\r\nContent-Length: 18446744073709551616\r\n\r\nabc".to_vec())
            }
            "/negative-chunk" => {
                Reply::Bytes(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n-1\r\nabc".to_vec())
            }
            "/huge-negative-chunk" => Reply::Bytes(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n-10000000000000000000000000\r\nabc".to_vec(),
            ),
            "/a.png;v=2?t=1" => http("200 OK", "Content-Type: image/png\r\n", PNG),
            "/truncated.png" => Reply::Bytes(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabc".to_vec()),
            "/eof.png" => Reply::Bytes([b"HTTP/1.0 200 OK\r\nContent-Type: image/png\r\n\r\n", PNG].concat()),
            status => http(&format!("{} Nope", &status[1..]), "", LEAKY),
        }
    }

    /// A local HTTP server answering with `route`; returns its port and the request heads it saw.
    fn serve() -> (u16, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let log = log.clone();
                std::thread::spawn(move || answer(stream, port, &log));
            }
        });
        (port, seen)
    }

    fn answer(mut s: TcpStream, port: u16, log: &Mutex<Vec<String>>) {
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            if s.read(&mut byte).unwrap_or(0) == 0 {
                return;
            }
            head.push(byte[0]);
        }
        let head = String::from_utf8_lossy(&head).into_owned();
        let path = head.split_whitespace().nth(1).unwrap_or("").to_string();
        log.lock().unwrap().push(head);
        match route(&path, port) {
            Reply::Bytes(out) => {
                let _ = s.write_all(&out);
            }
            Reply::Hang => std::thread::sleep(Duration::from_secs(3)),
            Reply::Trickle => {
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n");
                for _ in 0..100 {
                    if s.write_all(b"x").is_err() {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            Reply::Stall => {
                std::thread::sleep(Duration::from_millis(700));
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nx");
                std::thread::sleep(Duration::from_secs(3));
            }
            Reply::Drip(prefix, byte) => {
                let _ = s.write_all(prefix);
                for _ in 0..60 {
                    if s.write_all(&[byte]).is_err() {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
    }

    fn policy(hosts: &[&str], max_bytes: u64, timeout: Duration) -> RemotePolicy {
        let allowed_hosts = hosts.iter().map(|h| h.to_string()).collect();
        RemotePolicy { allowed_hosts, max_bytes, timeout }
    }

    fn local() -> RemotePolicy {
        policy(&["127.0.0.1"], REMOTE_MAX_BYTES, Duration::from_secs(30))
    }

    /// Fetch into a fresh upload dir; a failed fetch must leave nothing behind.
    fn fetch(url: &str, policy: &RemotePolicy) -> Result<(String, Vec<u8>), String> {
        let dir = std::env::temp_dir().join(format!("mstar-remote-{}", Uuid::new_v4()));
        let out = save_remote_url(url, &dir, policy).map(|(m, p)| (m, std::fs::read(p).unwrap()));
        if out.is_err() {
            let left = std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0);
            assert_eq!(left, 0, "upload left behind");
        }
        let _ = std::fs::remove_dir_all(&dir);
        out
    }

    fn refused(url: &str, policy: &RemotePolicy, reason: &str) {
        assert_eq!(fetch(url, policy).unwrap_err(), format!("remote media fetch failed: {reason}"));
    }

    #[test]
    fn internal_addresses_are_blocked() {
        for a in [
            "127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254",
            "100.100.100.200", "0.0.0.0", "224.0.0.1", "239.1.1.1", "255.255.255.255", "::1", "::",
            "fe80::1", "fd00:ec2::254", "ff02::1", "::ffff:127.0.0.1", "::ffff:169.254.169.254",
            "64:ff9b::7f00:1", "64:ff9b::a9fe:a9fe", "64:ff9b:1::808:808", "2002:7f00:1::",
            "2002:a00:1:2::3", "::127.0.0.1", "::169.254.169.254",
        ] {
            assert!(blocked_address(a.parse().unwrap()), "{a}");
        }
        for a in [
            "8.8.8.8", "1.1.1.1", "2606:4700:4700::1111", "::ffff:8.8.8.8", "64:ff9b::808:808",
            "2002:808:808::1", "::8.8.8.8",
        ] {
            assert!(!blocked_address(a.parse().unwrap()), "{a}");
        }
    }

    #[test]
    fn wildcard_entries_match_subdomains_only() {
        let p = policy(&["*.media.test", "example.org"], 1, Duration::ZERO);
        for host in ["cdn.media.test", "a.b.media.test", "example.org"] {
            assert!(p.allowlisted(host, Some(host)), "{host}");
        }
        for host in ["media.test", "evilmedia.test", "www.example.org"] {
            assert!(!p.allowlisted(host, Some(host)), "{host}");
        }
    }

    /// The host each URL is matched against, or `None` when another spelling
    /// keeps it from matching any allowlist entry. Same table as
    /// `test_only_the_canonical_spelling_matches_the_allowlist` in Python.
    fn allowlist_host(raw: &str) -> Option<String> {
        let url = url::Url::parse(raw).ok()?;
        let host = url.host_str()?.trim_start_matches('[').trim_end_matches(']').trim_end_matches('.');
        (written_host(raw).as_deref() == Some(host)).then(|| host.to_string())
    }

    #[test]
    fn only_the_canonical_spelling_matches_the_allowlist() {
        for (raw, host) in [
            ("http://127.0.0.1:8000/a.png", "127.0.0.1"),
            ("http://127.0.0.1./a.png", "127.0.0.1"),
            ("http://user@Example.ORG/a.png", "example.org"),
            ("https://[::1]:8443/a.png", "::1"),
            ("http://[::ffff:7f00:1]/a.png", "::ffff:7f00:1"),
            ("http://0xg/a.png", "0xg"),
        ] {
            assert_eq!(allowlist_host(raw).as_deref(), Some(host), "{raw}");
        }
        for raw in [
            "http://2130706433/a.png",
            "http://0x7f000001/a.png",
            "http://127.1/a.png",
            "http://127.0.0.01/a.png",
            "http://0177.0.0.1/a.png",
            "http://[0:0:0:0:0:0:0:1]/a.png",
            "http://[::ffff:127.0.0.1]/a.png",
            "http://%31%32%37.0.0.1/a.png",
            "http://b\u{fc}cher.example/a.png",
        ] {
            assert_eq!(allowlist_host(raw), None, "{raw}");
        }
    }

    #[test]
    fn other_spellings_of_an_allowlisted_address_are_refused() {
        let (port, seen) = serve();
        for host in ["2130706433", "0x7f000001", "127.1", "127.0.0.01"] {
            refused(&format!("http://{host}:{port}/a.png"), &local(), "host not allowed");
        }
        let numeric = policy(&["2130706433"], REMOTE_MAX_BYTES, Duration::from_secs(30));
        refused(&format!("http://2130706433:{port}/a.png"), &numeric, "host not allowed");
        refused(&format!("http://127.0.0.1:{port}/to-short-form"), &local(), "host not allowed");
        refused(&format!("http://127.0.0.1:{port}/to-backslash-authority"), &local(), "host not allowed");
        // the parsers drop tabs and newlines, which would make `127.0.0.\t1` read as 127.0.0.1
        refused(&format!("http://127.0.0.\t1:{port}/a.png"), &local(), "invalid URL");
        refused(&format!("http://127.0.0.1:{port}/to-tab"), &local(), "invalid URL");
        let paths: Vec<String> = seen.lock().unwrap().iter().map(|h| h.split(' ').nth(1).unwrap().to_string()).collect();
        assert_eq!(paths, ["/to-short-form", "/to-backslash-authority", "/to-tab"], "only redirecting hops reach the server");
    }

    #[test]
    fn allowlisted_host_is_fetched() {
        let (port, seen) = serve();
        let (modality, body) = fetch(&format!("http://127.0.0.1:{port}/a.png"), &local()).unwrap();
        assert_eq!((modality.as_str(), body.as_slice()), ("image", PNG));
        assert!(seen.lock().unwrap()[0].to_ascii_lowercase().contains("\r\nuser-agent: mstar\r\n"));
    }

    #[test]
    fn hosts_outside_the_allowlist_are_refused() {
        let (port, seen) = serve();
        refused(&format!("http://localhost:{port}/a.png"), &local(), "host not allowed");
        let other = policy(&["example.org"], REMOTE_MAX_BYTES, Duration::from_secs(30));
        refused(&format!("http://127.0.0.1:{port}/a.png"), &other, "host not allowed");
        assert!(seen.lock().unwrap().is_empty());
    }

    #[test]
    fn internal_addresses_are_refused_unless_allowlisted() {
        let (port, seen) = serve();
        let open = policy(&[], REMOTE_MAX_BYTES, Duration::from_secs(30));
        for host in ["127.0.0.1", "localhost", "[::1]", "169.254.169.254", "10.0.0.1"] {
            refused(&format!("http://{host}:{port}/a.png"), &open, "host not allowed");
        }
        assert!(seen.lock().unwrap().is_empty());
    }

    #[test]
    fn redirect_to_a_host_outside_the_allowlist_is_refused() {
        let (port, seen) = serve();
        refused(&format!("http://127.0.0.1:{port}/to-localhost"), &local(), "host not allowed");
        refused(&format!("http://127.0.0.1:{port}/to-file"), &local(), "invalid URL");
        assert_eq!(seen.lock().unwrap().len(), 2);
    }

    #[test]
    fn redirects_are_followed_up_to_the_cap() {
        let (port, seen) = serve();
        assert_eq!(fetch(&format!("http://127.0.0.1:{port}/r1"), &local()).unwrap().1, PNG);
        seen.lock().unwrap().clear();
        refused(&format!("http://127.0.0.1:{port}/loop"), &local(), "too many redirects");
        assert_eq!(seen.lock().unwrap().len(), REMOTE_MAX_REDIRECTS + 1);
    }

    #[test]
    fn content_length_over_the_cap_is_refused_before_reading() {
        let (port, _) = serve();
        let small = policy(&["127.0.0.1"], 1000, Duration::from_secs(5));
        let started = Instant::now();
        refused(&format!("http://127.0.0.1:{port}/big.png"), &small, TOO_LARGE);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn body_without_content_length_is_capped_while_streaming() {
        let (port, _) = serve();
        let small = policy(&["127.0.0.1"], 1000, Duration::from_secs(30));
        refused(&format!("http://127.0.0.1:{port}/over.mp4"), &small, TOO_LARGE);
        let (modality, body) = fetch(&format!("http://127.0.0.1:{port}/at.mp4"), &small).unwrap();
        assert_eq!((modality.as_str(), body.len()), ("video", 1000));
    }

    #[test]
    fn upstream_errors_become_a_short_message() {
        let (port, _) = serve();
        for status in [403, 404, 500, 502] {
            let reason = format!("upstream returned {status}");
            refused(&format!("http://127.0.0.1:{port}/{status}"), &local(), &reason);
        }
    }

    #[test]
    fn the_whole_fetch_is_bounded_by_the_timeout() {
        let (port, _) = serve();
        let quick = policy(&["127.0.0.1"], REMOTE_MAX_BYTES, Duration::from_millis(500));
        for path in ["/hang", "/trickle"] {
            let started = Instant::now();
            refused(&format!("http://127.0.0.1:{port}{path}"), &quick, "timed out");
            assert!(started.elapsed() < Duration::from_secs(2), "{path}");
        }
    }

    #[test]
    fn a_read_that_stalls_late_still_ends_at_the_deadline() {
        let (port, _) = serve();
        let one_second = policy(&["127.0.0.1"], REMOTE_MAX_BYTES, Duration::from_secs(1));
        let started = Instant::now();
        refused(&format!("http://127.0.0.1:{port}/stall"), &one_second, "timed out");
        assert!(started.elapsed() < Duration::from_millis(1500));
    }

    #[test]
    fn trickled_headers_and_chunk_framing_end_at_the_deadline() {
        let (port, _) = serve();
        let quick = policy(&["127.0.0.1"], REMOTE_MAX_BYTES, Duration::from_millis(500));
        for path in ["/drip-head", "/drip-chunk"] {
            let started = Instant::now();
            refused(&format!("http://127.0.0.1:{port}{path}"), &quick, "timed out");
            assert!(started.elapsed() < Duration::from_millis(1500), "{path}");
        }
    }

    #[test]
    fn malformed_bodies_are_refused() {
        let (port, _) = serve();
        // a chunk-size line longer than MAX_LINE, negative chunk sizes, and a body shorter than its Content-Length
        for path in ["/long-chunk-line", "/negative-chunk", "/huge-negative-chunk", "/truncated.png"] {
            refused(&format!("http://127.0.0.1:{port}{path}"), &local(), "connection failed");
        }
    }

    #[test]
    fn a_content_length_past_u64_is_too_large() {
        let (port, _) = serve();
        let small = policy(&["127.0.0.1"], 1000, Duration::from_secs(5));
        refused(&format!("http://127.0.0.1:{port}/huge-length.png"), &small, TOO_LARGE);
    }

    #[test]
    fn the_request_target_keeps_params_and_query() {
        let (port, seen) = serve();
        assert_eq!(fetch(&format!("http://127.0.0.1:{port}/a.png;v=2?t=1"), &local()).unwrap().1, PNG);
        assert!(seen.lock().unwrap()[0].starts_with("GET /a.png;v=2?t=1 HTTP/1.1\r\n"));
    }

    #[test]
    fn unparseable_urls_are_a_short_message() {
        for url in ["http://[db.internal.corp]/a.png", "http://127.0.0.1:99999/a.png"] {
            refused(url, &local(), "invalid URL");
        }
    }

    #[test]
    fn a_body_that_runs_to_eof_is_fetched() {
        let (port, _) = serve();
        let (modality, body) = fetch(&format!("http://127.0.0.1:{port}/eof.png"), &local()).unwrap();
        assert_eq!((modality.as_str(), body.as_slice()), ("image", PNG));
    }

    #[test]
    fn a_storage_failure_is_a_short_message() {
        let (port, _) = serve();
        let occupied = std::env::temp_dir().join(format!("mstar-remote-{}", Uuid::new_v4()));
        std::fs::write(&occupied, b"not a directory").unwrap();
        let out = save_remote_url(&format!("http://127.0.0.1:{port}/a.png"), &occupied, &local());
        let _ = std::fs::remove_file(&occupied);
        assert_eq!(out.unwrap_err(), "remote media fetch failed: could not save the file");
    }

    #[test]
    fn a_missing_or_generic_content_type_falls_back_to_the_url_extension() {
        let (port, _) = serve();
        for path in ["/untyped.png", "/generic.png"] {
            let (modality, body) = fetch(&format!("http://127.0.0.1:{port}{path}"), &local()).unwrap();
            assert_eq!((modality.as_str(), body.as_slice()), ("image", PNG), "{path}");
        }
    }

    #[test]
    fn connection_failure_does_not_echo_transport_details() {
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        refused(&format!("http://127.0.0.1:{port}/a.png"), &local(), "connection failed");
    }
}
