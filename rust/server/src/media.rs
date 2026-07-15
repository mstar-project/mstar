//! Media decode/encode helpers, ported from mstar's `api_server/media_io.py`.
//!
//! Two directions:
//!  * **Inbound** — turn media referenced by an OpenAI-style request (`data:`
//!    URLs or base64 blobs) into files under the server's `upload_dir`, so the
//!    Python data-worker can read them by path (the same contract `/generate`
//!    uses for multipart uploads). `http(s)` fetch is SSRF surface and is gated
//!    off by default here (data-URL / base64 / local path only).
//!  * **Outbound** — wrap raw model audio output (16-bit PCM, no header) into a
//!    real container (WAV by default) and encode PNG image bytes as a `data:`
//!    URL for OpenAI chat image output.
//!
//! Only the stdlib + `base64` + `uuid` are needed; compressed audio containers
//! (mp3/flac/ogg) degrade to WAV since the base install has no encoder — the
//! Python side is where those are produced when `soundfile` is present.

use std::path::{Path, PathBuf};

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
        ".mp4" | ".webm" | ".mov" | ".mkv" => "video",
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
    let raw = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
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
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("input_audio base64 decode failed: {e}"))?;
    let ext = if fmt.is_empty() {
        ".bin".to_string()
    } else {
        format!(".{}", fmt.trim_start_matches('.'))
    };
    let path = upload_dir.join(format!("{}{}", Uuid::new_v4(), ext));
    std::fs::write(&path, &raw).map_err(|e| e.to_string())?;
    Ok((modality_hint.to_string(), path.to_string_lossy().into_owned()))
}

/// Download an `http(s)` URL into `upload_dir` (mstar's `save_remote_url`;
/// 30 s timeout). Modality from the Content-Type, falling back to the URL
/// path's extension. Gated by `allow_remote` — SSRF surface.
fn save_remote_url(url: &str, upload_dir: &Path) -> Result<(String, String), String> {
    let resp = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .get(url)
        .call()
        .map_err(|e| format!("remote media fetch failed: {e}"))?;
    let mut mime = resp
        .header("content-type")
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    let mut raw = Vec::new();
    resp.into_reader()
        .read_to_end(&mut raw)
        .map_err(|e| format!("remote media read failed: {e}"))?;
    if mime.is_empty() || mime == "application/octet-stream" {
        // Infer from the URL path extension (mstar's fallback).
        let path = url.split('?').next().unwrap_or(url);
        let suffix = PathBuf::from(path)
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy()))
            .unwrap_or_default();
        mime = match modality_from_ext(&suffix) {
            "image" => format!("image/{}", suffix.trim_start_matches('.')),
            "audio" => format!("audio/{}", suffix.trim_start_matches('.')),
            "video" => format!("video/{}", suffix.trim_start_matches('.')),
            _ => "application/octet-stream".to_string(),
        };
    }
    save_bytes(&raw, &mime, upload_dir).map_err(|e| e.to_string())
}

/// Resolve a media reference (data URL, `http(s)` URL, or local path).
/// Returns `(modality, path)`. Local paths pass through unchanged (modality
/// from extension). `http(s)` is refused unless `allow_remote` is set — remote
/// fetch is SSRF surface, off by default, matching mstar's `allow_remote` gate.
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
        return save_remote_url(reference, upload_dir);
    }
    // Treat as a local filesystem path.
    let suffix = PathBuf::from(reference)
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    Ok((modality_from_ext(&suffix).to_string(), reference.to_string()))
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
