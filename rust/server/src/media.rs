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
//! Only the stdlib + `base64` + `uuid` are needed (plus `url` and `ureq` for
//! the opt-in remote fetch: a GET bounded in size and by the fetch deadline,
//! which reaches only the addresses this module has vetted); compressed audio
//! containers (mp3/flac/ogg) degrade to WAV since the base install has no
//! encoder — the Python side is where those are produced when `soundfile` is
//! present.

use std::io::{self, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::Value;
use ureq::unversioned::resolver::ResolvedSocketAddrs;
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

/// A file's modality by its name, for `/generate` and `/generate/ws`
/// uploads: `entrypoint._detect_modality`'s table, which differs from the
/// media paths' (no `.opus`).
pub fn modality_from_filename(name: &str) -> &'static str {
    let suffix = Path::new(name)
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()).to_lowercase())
        .unwrap_or_default();
    match suffix.as_str() {
        ".png" | ".jpg" | ".jpeg" | ".webp" | ".bmp" | ".tiff" | ".gif" => "image",
        ".wav" | ".mp3" | ".flac" | ".ogg" | ".m4a" | ".aac" => "audio",
        ".mp4" | ".avi" | ".mov" | ".mkv" | ".webm" => "video",
        _ => "unknown",
    }
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

/// Client base64 decoded as the Python frontend's `_decode_base64_payload`
/// does: whitespace anywhere is dropped (line-wrapped payloads), missing `=`
/// padding is restored (some OpenAI clients strip it), and anything else
/// outside the standard alphabet is refused.
pub(crate) fn decode_b64_lenient(payload: &str) -> Result<Vec<u8>, base64::DecodeError> {
    use base64::engine::general_purpose::{GeneralPurpose, GeneralPurposeConfig};
    use base64::engine::DecodePaddingMode;
    // Python ignores the unused low bits of the last symbol too
    const LENIENT: GeneralPurpose = GeneralPurpose::new(
        &base64::alphabet::STANDARD,
        GeneralPurposeConfig::new()
            .with_decode_padding_mode(DecodePaddingMode::Indifferent)
            .with_decode_allow_trailing_bits(true),
    );
    // Python's `str.split()` whitespace: Unicode's plus the C0 separators
    let compact: String = payload
        .chars()
        .filter(|c| !c.is_whitespace() && !('\u{1c}'..='\u{1f}').contains(c))
        .collect();
    LENIENT.decode(compact.as_bytes())
}

/// `subprocess.run(cmd, capture_output=True, timeout=timeout)`: the child's
/// output, or an `Err` once it has run past `timeout` (it is killed, as
/// `subprocess.run` kills it).
fn run_with_timeout(cmd: &mut std::process::Command, timeout: std::time::Duration) -> Result<std::process::Output, String> {
    use std::io::Read;
    use std::process::Stdio;
    let name = cmd.get_program().to_string_lossy().into_owned();
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{name} failed: {e}"))?;
    // drained as the child writes, so a full pipe cannot stall it
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut out = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut out);
            }
            out
        })
    };
    let stdout = drain(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let stderr = drain(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let deadline = std::time::Instant::now() + timeout;
    let status = loop {
        match child.try_wait().map_err(|e| format!("{name} failed: {e}"))? {
            Some(status) => break status,
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{name} timed out after {} seconds", timeout.as_secs()));
            }
            None => std::thread::sleep(std::time::Duration::from_millis(10)),
        }
    };
    Ok(std::process::Output {
        status,
        stdout: stdout.join().unwrap_or_default(),
        stderr: stderr.join().unwrap_or_default(),
    })
}

/// `mux_mp4_with_pcm16`'s limits on the frame-rate query and the mux.
const FFPROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const FFMPEG_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// `media_io.mux_mp4_with_pcm16`: an H.264 mp4 and raw interleaved 16-bit PCM
/// into one mp4 with an AAC track, the video stream-copied, by the same
/// `ffprobe` and `ffmpeg` commands. `video_fps` is the content's true rate: when
/// it differs from the container's, the video timestamps are rescaled so
/// playback stays in sync with the audio. Blocking; an `Err` leaves the caller
/// the video alone, as Python falls back to it.
pub fn mux_mp4_with_pcm16(
    video_mp4: &[u8],
    pcm: &[u8],
    sample_rate: i64,
    num_channels: i64,
    video_fps: Option<f64>,
) -> Result<Vec<u8>, String> {
    use std::os::unix::fs::DirBuilderExt;
    let dir = std::env::temp_dir().join(format!("mstar_mux_{}", Uuid::new_v4().simple()));
    // private, as `TemporaryDirectory`'s: the generated clip is the client's
    std::fs::DirBuilder::new().mode(0o700).create(&dir).map_err(|e| e.to_string())?;
    let result = (|| {
        let (vpath, apath, opath) = (dir.join("v.mp4"), dir.join("a.pcm"), dir.join("out.mp4"));
        std::fs::write(&vpath, video_mp4).map_err(|e| e.to_string())?;
        std::fs::write(&apath, pcm).map_err(|e| e.to_string())?;
        let mut itsscale = None;
        if let Some(fps) = video_fps {
            let mut probe = std::process::Command::new("ffprobe");
            probe
                .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate",
                       "-of", "csv=p=0"])
                .arg(&vpath);
            // a query that times out fails the mux; one that answers nothing leaves the rate
            let probe = run_with_timeout(&mut probe, FFPROBE_TIMEOUT)?;
            let out = String::from_utf8_lossy(&probe.stdout);
            if let Some((num, den)) = out.trim().split_once('/') {
                if let (Ok(num), Ok(den)) = (num.parse::<f64>(), den.parse::<f64>()) {
                    let container = num / den;
                    if den != 0.0 && container > 0.0 && (container - fps).abs() > 1e-3 {
                        itsscale = Some(container / fps);
                    }
                }
            }
        }
        let mut cmd = std::process::Command::new("ffmpeg");
        cmd.args(["-y", "-v", "error"]);
        if let Some(scale) = itsscale {
            cmd.args(["-itsscale".to_string(), format!("{scale:.6}")]);
        }
        cmd.arg("-i").arg(&vpath);
        cmd.args(["-f", "s16le", "-ar", &sample_rate.to_string(), "-ac", &num_channels.to_string(), "-i"]);
        cmd.arg(&apath);
        cmd.args(["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]);
        cmd.arg(&opath);
        let out = run_with_timeout(&mut cmd, FFMPEG_TIMEOUT).map_err(|e| format!("ffmpeg mux failed: {e}"))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(format!("ffmpeg mux failed: {}", stderr.trim().chars().take(500).collect::<String>()));
        }
        std::fs::read(&opath).map_err(|e| e.to_string())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// `media_io.BARE_REF_AUDIO_REFUSAL`.
pub const BARE_REF_AUDIO_REFUSAL: &str =
    "server file paths are refused; ref_audio must be a data URL, an http(s) URL or base64 audio the model can decode";

/// A request's reference clip once saved, for the API server to check with the
/// model's loader (`media_io.check_saved_ref_audio`) before anything is submitted.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RefAudioCheck {
    pub path: String,
    /// The library the model's loader opens it with (`check_ref_audio`'s `decoder`).
    pub decoder: &'static str,
    /// Sent as bare base64, which could be a server path as well as audio.
    pub bare: bool,
}

/// Save a speech request's `ref_audio` as `adapters._resolve_ref_audio` does,
/// up to its check: a data URL, an http(s) URL when remote fetch is allowed, or
/// bare base64 (saved as `.wav`). A refusal is `(status, message)`.
pub fn save_ref_audio(
    reference: &Value,
    upload_dir: &Path,
    allow_remote: bool,
    decoder: &'static str,
) -> Result<RefAudioCheck, (u16, String)> {
    let Value::String(reference) = reference else {
        let kind = match reference {
            Value::Bool(_) => "bool",
            Value::Number(n) if n.is_f64() => "float",
            Value::Number(_) => "int",
            Value::Array(_) => "list",
            Value::Object(_) => "dict",
            _ => "NoneType",
        };
        return Err((400, format!("ref_audio must be a data URL or base64 string, got {kind}")));
    };
    let scheme = reference.split(':').next().unwrap_or("").to_ascii_lowercase();
    if reference.starts_with("data:") || scheme == "http" || scheme == "https" {
        let (_, path) = resolve_media_ref(reference, upload_dir, allow_remote).map_err(|e| (400, e))?;
        return Ok(RefAudioCheck { path, decoder, bare: false });
    }
    let raw = decode_b64_lenient(reference).map_err(|_| (400, BARE_REF_AUDIO_REFUSAL.to_string()))?;
    let stored = std::fs::create_dir_all(upload_dir).and_then(|()| {
        let path = upload_dir.join(format!("{}.wav", Uuid::new_v4()));
        std::fs::write(&path, raw).map(|()| path)
    });
    match stored {
        Ok(path) => Ok(RefAudioCheck { path: path.to_string_lossy().into_owned(), decoder, bare: true }),
        Err(e) => {
            eprintln!("could not store a ref_audio upload under {}: {e}", upload_dir.display());
            Err((500, "the server could not store ref_audio".to_string()))
        }
    }
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

/// A transport failure: `timed out` once the deadline has passed (the fetch ran
/// out of time even if the error itself does not say so), `connection failed`
/// before it.
fn transport_error(host: &str, deadline: Instant, detail: impl std::fmt::Display) -> String {
    let reason = if Instant::now() >= deadline { "timed out" } else { "connection failed" };
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
        // the same spelling the pinned resolver checks the hop's URI against
        host = normalized_host(url.host_str().unwrap_or(""));
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

/// `host` as the allowlist and the address check spell it: lower-cased, without
/// brackets around an IPv6 literal or a trailing dot.
fn normalized_host(host: &str) -> String {
    host.trim_start_matches('[')
        .trim_end_matches(']')
        .trim_end_matches('.')
        .to_ascii_lowercase()
}

/// Answers every lookup with the one address `save_remote_url` has already
/// vetted, and refuses any other host. ureq's connector reaches only what a
/// resolver returns, so the name cannot resolve to something else between the
/// address check and the connect.
#[derive(Debug)]
struct Pinned {
    host: String,
    addr: SocketAddr,
}

impl ureq::unversioned::resolver::Resolver for Pinned {
    fn resolve(
        &self,
        uri: &ureq::http::Uri,
        _config: &ureq::config::Config,
        _timeout: ureq::unversioned::transport::NextTimeout,
    ) -> Result<ResolvedSocketAddrs, ureq::Error> {
        // one hop is one vetted host; redirects go back through save_remote_url
        if normalized_host(uri.host().unwrap_or("")) != self.host {
            return Err(ureq::Error::HostNotFound);
        }
        let mut out = self.empty();
        out.push(self.addr);
        Ok(out)
    }
}

/// Whether a failed attempt concerns only the address tried, so the next
/// vetted address may still answer. These are the kinds ureq's own connector
/// retries, but it reads them from the connect alone while this reads them
/// from the whole call: a route that disappears after the request was sent
/// retries the GET on the next vetted address, where the socket client
/// returned. Both stay inside the checked host, its vetted addresses and the
/// fetch deadline, and only a call that returned no response is retried.
fn addr_specific(e: &ureq::Error) -> bool {
    matches!(
        e,
        ureq::Error::Io(io)
            if matches!(
                io.kind(),
                io::ErrorKind::ConnectionRefused
                    | io::ErrorKind::HostUnreachable
                    | io::ErrorKind::NetworkUnreachable
                    | io::ErrorKind::AddrNotAvailable
            )
    )
}

/// A failed call as one of the fetch's short reasons. The detail goes to the
/// log only: ureq's messages name the host, the address and the upstream.
fn request_error(host: &str, deadline: Instant, max_bytes: u64, e: &ureq::Error) -> String {
    match e {
        ureq::Error::Timeout(_) => fetch_error(host, "timed out", e),
        ureq::Error::BodyExceedsLimit(_) => {
            fetch_error(host, &format!("too large (limit {max_bytes} bytes)"), e)
        }
        ureq::Error::BadUri(_) | ureq::Error::Http(_) => fetch_error(host, "invalid URL", e),
        ureq::Error::Io(io) => io_error(host, deadline, io),
        // a protocol error (bad framing, an unreadable Content-Length, too many
        // headers), a refused connection, or anything ureq adds later
        _ => transport_error(host, deadline, e),
    }
}

fn io_error(host: &str, deadline: Instant, e: &io::Error) -> String {
    if e.kind() == io::ErrorKind::TimedOut {
        return fetch_error(host, "timed out", e);
    }
    transport_error(host, deadline, e)
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

/// The client settings for one hop. `proxy(None)` matters: ureq's default
/// config takes a proxy from `ALL_PROXY` / `HTTPS_PROXY` / `HTTP_PROXY`, and a
/// fetch sent to a proxy would reach neither the vetted addresses nor the
/// checked host.
fn hop_config(left: Duration) -> ureq::config::Config {
    ureq::config::Config::builder()
        .proxy(None)
        .http_status_as_error(false)
        .max_redirects(0)
        .max_redirects_will_error(false)
        .user_agent(REMOTE_USER_AGENT)
        .timeout_global(Some(left))
        // a fetch is one connection: nothing is kept for a later request
        .max_idle_connections(0)
        .max_idle_connections_per_host(0)
        .build()
}

/// One GET to the vetted `addrs`: the next URL for a redirect, or the body
/// saved under `upload_dir`.
///
/// ureq does the HTTP/1.1 and TLS work. The redirect loop stays in
/// `save_remote_url` (`max_redirects(0)`, and a 3xx is a response rather than
/// an error), so every hop is checked against the allowlist and the blocked
/// ranges, the `Location` is read as the server wrote it, and the hop count
/// stays at `REMOTE_MAX_REDIRECTS`.
///
/// The addresses are tried one call at a time, each with `timeout_global` set
/// to what is left of the fetch deadline, so the handshake, the headers, the
/// chunk framing and the body are all bounded by it and several addresses
/// cannot spend it more than once. ureq's own address loop would: it moves on
/// from a refusal without re-reading the clock or shrinking the next
/// address's share (`try_connect_with` in its `transport::tcp`).
fn fetch_hop(
    url: &url::Url,
    host: &str,
    addrs: &[SocketAddr],
    upload_dir: &Path,
    max_bytes: u64,
    deadline: Instant,
) -> Result<Hop, String> {
    let mut last: Option<ureq::Error> = None;
    for addr in addrs {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(fetch_error(host, "timed out", "the fetch ran out of time"));
        }
        let agent = ureq::Agent::with_parts(
            hop_config(left),
            ureq::unversioned::transport::DefaultConnector::new(),
            Pinned { host: host.to_string(), addr: *addr },
        );
        let call = agent
            .get(url.as_str())
            // the cap counts the bytes written to disk, so take the body as it is
            .header("accept-encoding", "identity")
            .call();
        match call {
            Ok(resp) => return read_hop(resp, url, host, upload_dir, max_bytes, deadline),
            Err(e) if addr_specific(&e) => last = Some(e),
            Err(e) => return Err(request_error(host, deadline, max_bytes, &e)),
        }
    }
    match last {
        Some(e) => Err(request_error(host, deadline, max_bytes, &e)),
        None => Err(fetch_error(host, "connection failed", "no vetted address")),
    }
}

/// The redirect this response points at, or its body saved under `upload_dir`.
fn read_hop(
    mut resp: ureq::http::Response<ureq::Body>,
    url: &url::Url,
    host: &str,
    upload_dir: &Path,
    max_bytes: u64,
    deadline: Instant,
) -> Result<Hop, String> {
    let code = resp.status().as_u16();
    match code {
        301 | 302 | 303 | 307 | 308 => {
            let location = resp
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| fetch_error(host, "bad redirect", format!("{code} without Location")))?
                .to_string();
            if location.contains(URL_DROPPED) {
                return Err(fetch_error(host, "invalid URL", format!("{location:?}")));
            }
            return url
                .join(&location)
                .map(|next| Hop::Redirect(next, location))
                .map_err(|e| fetch_error(host, "invalid URL", e));
        }
        200..=299 => {}
        _ => return Err(fetch_error(host, &format!("upstream returned {code}"), "")),
    }
    let too_large = format!("too large (limit {max_bytes} bytes)");
    // as http.client: a non-numeric length is ignored; one past u64 is still a
    // length, over any cap. The header is read as well as the framing ureq
    // chose from it, so a length it does not frame a body with (a 204's, say)
    // is still checked.
    let declared = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()))
        .map(|v| v.parse::<u64>().unwrap_or(u64::MAX));
    for length in [resp.body().content_length(), declared].into_iter().flatten() {
        if length > max_bytes {
            return Err(fetch_error(host, &too_large, format!("Content-Length {length}")));
        }
    }
    let mime = mime_for(resp.headers().get("content-type").and_then(|v| v.to_str().ok()), url);
    std::fs::create_dir_all(upload_dir).map_err(|e| fetch_error(host, "could not save the file", e))?;
    let path = upload_dir.join(format!("{}{}", Uuid::new_v4(), ext_for(&mime)));
    // Two things ureq's reader is laxer about than the socket client was, both
    // bounded by the fetch deadline and neither reaching the file: it consumes
    // chunk trailers without counting their number or their bytes (the old
    // reader stopped at 100 trailers and a 64 KiB line), and it reads a reset
    // connection as the end of a body that has no declared length, where the
    // old reader refused anything but a clean EOF. A body short of its
    // `Content-Length` is still refused.
    let copied = std::fs::File::create(&path).map_err(BodyError::Store).and_then(|mut f| {
        // `limit` stops the read one byte past the cap, so the count below
        // always reaches `max_bytes` first and names the refusal
        let mut body = resp.body_mut().with_config().limit(max_bytes.saturating_add(1)).reader();
        let mut buf = [0u8; 1 << 14];
        let mut total = 0u64;
        loop {
            let n = match body.read(&mut buf) {
                Ok(0) => return Ok(()),
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(BodyError::Read(e)),
            };
            total += n as u64;
            if total > max_bytes {
                return Err(BodyError::TooLarge);
            }
            f.write_all(&buf[..n]).map_err(BodyError::Store)?;
        }
    });
    match copied {
        Ok(()) => Ok(Hop::Saved(modality_from_mime(&mime).to_string(), path.to_string_lossy().into_owned())),
        Err(e) => {
            let _ = std::fs::remove_file(&path);
            Err(match e {
                BodyError::TooLarge => fetch_error(host, &too_large, "body exceeded the limit"),
                BodyError::Read(e) => io_error(host, deadline, &e),
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

/// Encode raw 16-bit PCM into `fmt` (checked by [`check_audio_format`]), as
/// `media_io.pcm16_to_container`. Returns `(bytes, mime_type)`; the compressed
/// formats come from the libsndfile Python's soundfile ships ([`crate::sndfile`]).
pub fn pcm16_to_container(pcm: &[u8], sample_rate: u32, fmt: &str) -> Result<(Vec<u8>, &'static str), String> {
    match fmt {
        "pcm" => Ok((pcm.to_vec(), "audio/pcm")),
        "wav" => Ok((pcm16_to_wav_bytes(pcm, sample_rate, 1), "audio/wav")),
        _ => Ok((crate::sndfile::encode(pcm, sample_rate, fmt)?, crate::sndfile::mime(fmt))),
    }
}

/// `media_io.audio_formats`: a stream carries an open-ended WAV or bare PCM;
/// otherwise the compressed containers too, when libsndfile is loaded, and opus
/// only at one of its native rates.
pub(crate) fn audio_formats(stream: bool, sample_rate: u32, compressed: bool) -> Vec<&'static str> {
    let mut formats = vec!["wav", "pcm"];
    if !stream && compressed {
        formats.extend(["mp3", "flac"]);
        if crate::sndfile::OPUS_RATES.contains(&sample_rate) {
            formats.push("opus");
        }
    }
    formats
}

/// `response_format` lower-cased (`wav` when empty), or the 400's text, as
/// `media_io.check_audio_format`.
pub fn check_audio_format(fmt: &str, stream: bool, sample_rate: u32) -> Result<String, String> {
    check_format(fmt, stream, sample_rate, crate::sndfile::available())
}

fn check_format(fmt: &str, stream: bool, sample_rate: u32, compressed: bool) -> Result<String, String> {
    let fmt = if fmt.is_empty() { "wav".to_string() } else { fmt.to_lowercase() };
    let allowed = audio_formats(stream, sample_rate, compressed);
    if allowed.contains(&fmt.as_str()) {
        return Ok(fmt);
    }
    let how = if stream { "streamed" } else { "produced" };
    Err(format!(
        "response_format {} cannot be {how} by this server; use one of: {}",
        py_repr(&fmt),
        allowed.join(", ")
    ))
}

/// Is `response_format` one this frontend can stream — `media_io.audio_formats
/// (stream=True)`, which is `("wav", "pcm")` on both frontends? Read by the
/// speech route to answer Python's `stream_format` rule: the switch only forces
/// streaming for a format that is actually streamable, so the SDK's default mp3
/// still falls back to one whole body.
pub fn is_streamable_audio_format(fmt: &str) -> bool {
    let fmt = if fmt.is_empty() { "wav".to_string() } else { fmt.to_lowercase() };
    matches!(fmt.as_str(), "wav" | "pcm")
}

/// A client's rejected JSON value named by its type, for a refusal that cannot
/// quote it. The Python frontend's `describe_rejected` says the same words.
///
/// Both frontends quote a string, which is easy to agree on. Reading any other
/// type back would mean reproducing Python's `repr` of it, and serde_json's
/// number model does not allow that: an integer outside `i64`/`u64` arrives as
/// an `f64`, so does `-0`, a float whose decimal needs 17 digits can land one
/// ULP off without `float_roundtrip`, and an object's keys come out sorted,
/// `Value` being a `BTreeMap` without `preserve_order`. A client that sent the
/// wrong type does not need its value read back, so name the type instead.
pub fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// Python's `repr` of a str, for a refusal that quotes the client's value as
/// the Python frontend's does.
pub fn py_repr(s: &str) -> String {
    static UNPRINTABLE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let unprintable = UNPRINTABLE
        .get_or_init(|| regex::Regex::new(r"[\p{Cc}\p{Cf}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]").unwrap());
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::from(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if c != ' ' && unprintable.is_match(c.encode_utf8(&mut [0; 4])) => {
                out.push_str(&match c as u32 {
                    n @ ..=0xff => format!("\\x{n:02x}"),
                    n @ ..=0xffff => format!("\\u{n:04x}"),
                    n => format!("\\U{n:08x}"),
                });
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
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
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};

    #[test]
    fn a_format_is_refused_with_the_python_text() {
        assert_eq!(check_format("WAV", true, 24000, true).unwrap(), "wav");
        assert_eq!(check_format("", false, 24000, false).unwrap(), "wav");
        assert_eq!(
            check_format("MP3", true, 24000, true).unwrap_err(),
            "response_format 'mp3' cannot be streamed by this server; use one of: wav, pcm"
        );
        assert_eq!(check_format("opus", false, 24000, true).unwrap(), "opus");
        assert_eq!(
            check_format("opus", false, 44100, true).unwrap_err(),
            "response_format 'opus' cannot be produced by this server; use one of: wav, pcm, mp3, flac"
        );
        // no libsndfile: as the Python frontend without soundfile
        assert_eq!(
            check_format("mp3", false, 24000, false).unwrap_err(),
            "response_format 'mp3' cannot be produced by this server; use one of: wav, pcm"
        );
    }

    #[test]
    fn base64_decodes_what_the_python_frontend_decodes() {
        // each payload's outcome in `media_io._decode_base64_payload` on CPython 3.12
        let a: Option<&[u8]> = Some(b"a");
        for (payload, want) in [
            ("YQ==", a),
            ("YQ", a), // the client stripped the padding
            ("YQ=", a),
            ("YR==", a), // unused low bits set
            ("YQ\r\n==\n", a),
            ("Y Q\u{a0}\u{b}\u{1c}\u{85}\u{3000}", a),
            ("YWJjZA", Some(&b"abcd"[..])),
            ("", Some(&b""[..])),
            ("Y", None),
            ("Y==", None),
            ("YQ===", None),
            ("YQ==YQ==", None),
            ("====", None),
            ("-_", None),
            ("YQ\u{200b}", None),
        ] {
            assert_eq!(decode_b64_lenient(payload).ok().as_deref(), want, "{payload:?}");
        }
    }

    #[test]
    fn repr_quotes_and_escapes_as_python_does() {
        for (s, want) in [
            ("mp3", "'mp3'"),
            ("it's", "\"it's\""),
            ("'\"", "'\\'\"'"),
            ("a\\b\n\t", "'a\\\\b\\n\\t'"),
            ("\u{0}\u{7f}\u{a0}\u{200b}", "'\\x00\\x7f\\xa0\\u200b'"),
            ("海 é", "'海 é'"),
        ] {
            assert_eq!(py_repr(s), want, "{s:?}");
        }
    }

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
        /// wait, then write: a hop that costs time before it redirects
        Delayed(Vec<u8>, Duration),
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
            "/to-stall" => redirect("302 Found", "/stall"),
            "/slow-to-stall" => {
                let Reply::Bytes(out) = redirect("302 Found", "/stall") else { unreachable!() };
                Reply::Delayed(out, Duration::from_millis(600))
            }
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
            "/empty-but-long.png" => {
                Reply::Bytes(b"HTTP/1.1 204 No Content\r\nContent-Length: 5000\r\n\r\n".to_vec())
            }
            "/text-length.png" => {
                Reply::Bytes(b"HTTP/1.1 200 OK\r\nContent-Length: abc\r\n\r\nabc".to_vec())
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
            Reply::Delayed(out, wait) => {
                std::thread::sleep(wait);
                let _ = s.write_all(&out);
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
        let head = seen.lock().unwrap()[0].to_ascii_lowercase();
        assert!(head.contains("\r\nuser-agent: mstar\r\n"), "{head:?}");
        // one identity Accept-Encoding, so the size cap counts the stored bytes
        assert_eq!(head.matches("\r\naccept-encoding: identity\r\n").count(), 1, "{head:?}");
        assert_eq!(head.matches("accept-encoding").count(), 1, "{head:?}");
    }

    /// The fetch reaches the address it was handed and nothing else. The name
    /// here is in a TLD that never resolves, so a body that arrives can only
    /// have come from the vetted address, and a hop for a host that was not
    /// vetted is refused before any connection.
    #[test]
    fn the_fetch_goes_only_to_the_vetted_address() {
        let (port, seen) = serve();
        let vetted = [SocketAddr::from(([127, 0, 0, 1], port))];
        let dir = std::env::temp_dir().join(format!("mstar-remote-{}", Uuid::new_v4()));
        let deadline = Instant::now() + Duration::from_secs(5);
        let url = url::Url::parse("http://pinned.invalid/a.png").unwrap();
        let hop = fetch_hop(&url, "pinned.invalid", &vetted, &dir, REMOTE_MAX_BYTES, deadline).unwrap();
        let Hop::Saved(modality, path) = hop else { panic!("expected a saved body") };
        assert_eq!((modality.as_str(), std::fs::read(&path).unwrap().as_slice()), ("image", PNG));
        // reached with the name it was asked for, not with the address
        assert!(seen.lock().unwrap()[0].to_ascii_lowercase().contains("\r\nhost: pinned.invalid\r\n"));
        let _ = std::fs::remove_dir_all(&dir);

        let other = url::Url::parse("http://other.invalid/a.png").unwrap();
        let refusal = fetch_hop(&other, "pinned.invalid", &vetted, &dir, REMOTE_MAX_BYTES, deadline);
        assert!(matches!(&refusal, Err(e) if e == "remote media fetch failed: connection failed"));
        assert_eq!(seen.lock().unwrap().len(), 1, "a host that was not vetted reaches no server");
    }

    /// A proxy in the environment does not divert a fetch: ureq's default config
    /// would take one, and the request would then go to the proxy instead of the
    /// vetted address, with neither the host nor the address checked.
    #[test]
    fn a_proxy_in_the_environment_is_ignored() {
        let (port, seen) = serve();
        // a port nothing listens on: were the proxy used, the fetch would fail
        let sink = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let vars = ["ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY"];
        let was: Vec<_> = vars.iter().map(|v| (v, std::env::var(v).ok())).collect();
        for var in vars {
            std::env::set_var(var, format!("http://127.0.0.1:{sink}"));
        }
        assert!(hop_config(Duration::from_secs(5)).proxy().is_none());
        let fetched = fetch(&format!("http://127.0.0.1:{port}/a.png"), &local());
        for (var, before) in was {
            match before {
                Some(v) => std::env::set_var(var, v),
                None => std::env::remove_var(var),
            }
        }
        assert_eq!(fetched.unwrap().1, PNG);
        assert!(seen.lock().unwrap()[0].starts_with("GET /a.png HTTP/1.1\r\n"), "went straight to the host");
    }

    /// The deadline covers the whole fetch, not one hop: each hop gets only what
    /// is left of it. The first hop spends 600 ms of a 1 s budget before it
    /// redirects, so the stall behind it has about 400 ms; a hop that started
    /// the budget again would run to about 1.6 s instead.
    #[test]
    fn the_deadline_is_not_reset_by_a_redirect() {
        let (port, seen) = serve();
        let one_second = policy(&["127.0.0.1"], REMOTE_MAX_BYTES, Duration::from_secs(1));
        let started = Instant::now();
        refused(&format!("http://127.0.0.1:{port}/slow-to-stall"), &one_second, "timed out");
        let took = started.elapsed();
        assert!(took > Duration::from_millis(600), "the first hop was not reached: {took:?}");
        assert!(took < Duration::from_millis(1300), "the deadline restarted: {took:?}");
        assert_eq!(seen.lock().unwrap().len(), 2, "the redirect was followed, then it timed out");
    }

    /// Every vetted address is tried in turn, and none is tried once the
    /// deadline has passed. That the deadline also covers them all together
    /// comes from the one-address-per-call shape of `fetch_hop`, which keeps
    /// ureq's own address loop (and its missing elapsed check) out of the way;
    /// a refused connection returns too quickly to show the difference here.
    #[test]
    fn the_addresses_are_tried_in_turn_within_one_deadline() {
        let (port, seen) = serve();
        // bound then dropped: nothing listens there, so the connect is refused
        let refuses = |_| {
            let p = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
            SocketAddr::from(([127, 0, 0, 1], p))
        };
        let addrs = [refuses(0), refuses(1), SocketAddr::from(([127, 0, 0, 1], port))];
        let dir = std::env::temp_dir().join(format!("mstar-remote-{}", Uuid::new_v4()));
        let url = url::Url::parse("http://pinned.invalid/a.png").unwrap();
        let started = Instant::now();
        let hop =
            fetch_hop(&url, "pinned.invalid", &addrs, &dir, REMOTE_MAX_BYTES, started + Duration::from_secs(5))
                .unwrap();
        let Hop::Saved(_, path) = hop else { panic!("expected a saved body") };
        assert_eq!(std::fs::read(&path).unwrap(), PNG, "the third address answered");
        assert_eq!(seen.lock().unwrap().len(), 1);

        // past the deadline no address is tried at all
        let out = fetch_hop(&url, "pinned.invalid", &addrs, &dir, REMOTE_MAX_BYTES, Instant::now());
        assert!(matches!(&out, Err(e) if e == "remote media fetch failed: timed out"), "not refused");
        assert_eq!(seen.lock().unwrap().len(), 1, "nothing is tried past the deadline");
        assert!(started.elapsed() < Duration::from_secs(2), "{:?}", started.elapsed());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A `Content-Length` over the cap is refused even when ureq does not frame
    /// the body with it (a 204 carrying one is invalid, and was refused before).
    #[test]
    fn a_declared_length_over_the_cap_is_refused_whatever_the_framing() {
        let (port, _) = serve();
        let small = policy(&["127.0.0.1"], 1000, Duration::from_secs(5));
        refused(&format!("http://127.0.0.1:{port}/empty-but-long.png"), &small, TOO_LARGE);
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
        // an over-long chunk-size line, negative chunk sizes, and a body shorter
        // than its Content-Length. ureq bounds the whole size line, extensions
        // included, at 20 bytes, so it refuses a long one sooner than the Python
        // frontend's 64 KiB line bound does; a valid chunk extension past 20
        // bytes is refused here and read there.
        for path in ["/long-chunk-line", "/negative-chunk", "/huge-negative-chunk", "/truncated.png"] {
            refused(&format!("http://127.0.0.1:{port}{path}"), &local(), "connection failed");
        }
    }

    /// A `Content-Length` the HTTP client cannot read is refused before any of
    /// the body is kept. ureq rejects such a response outright, so both of these
    /// are `connection failed` here. The Python frontend answers `too large` for
    /// the one past 64 bits and reads the non-numeric one to EOF, so this is
    /// where the two frontends differ; nothing is written either way.
    #[test]
    fn an_unreadable_content_length_is_refused() {
        let (port, _) = serve();
        let small = policy(&["127.0.0.1"], 1000, Duration::from_secs(5));
        for path in ["/huge-length.png", "/text-length.png"] {
            refused(&format!("http://127.0.0.1:{port}{path}"), &small, "connection failed");
        }
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
    /// `media_io.audio_formats(stream=True)` is `("wav", "pcm")`; an unset
    /// `response_format` reads as `wav`, as `(req.response_format or "wav")` does.
    #[test]
    fn only_wav_and_pcm_can_be_streamed() {
        for fmt in ["wav", "WAV", "pcm", "PCM", ""] {
            assert!(is_streamable_audio_format(fmt), "{fmt:?}");
        }
        for fmt in ["mp3", "MP3", "flac", "opus", "aac", "ogg", "nonsense"] {
            assert!(!is_streamable_audio_format(fmt), "{fmt:?}");
        }
    }
}
