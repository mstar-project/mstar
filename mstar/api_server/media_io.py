"""Media decode/encode helpers shared by the native and OpenAI-compatible APIs.

Two directions:

* **Inbound** — turn media referenced by an OpenAI-style request (``data:`` URLs,
  ``http(s)`` URLs, or base64 blobs) into files under the API server's
  ``upload_dir``, so ``model.load_image`` / ``load_audio`` / ``load_video`` can
  read them by path. This is the same contract ``/generate`` already uses for
  multipart uploads.
* **Outbound** — wrap raw model audio output (16-bit PCM, no container header)
  into a real audio container (WAV by default) and encode image bytes (PNG) as a
  ``data:`` URL for OpenAI chat image output.

Only stdlib + numpy are required. ``mp3`` / ``flac`` / ``ogg`` encoding is opt-in
and degrades to WAV when the optional ``soundfile`` backend is unavailable, so
the base install stays slim.
"""

from __future__ import annotations

import base64
import http.client
import io
import ipaddress
import logging
import math
import os
import re
import socket
import ssl
import threading
import time
import wave
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse, urlsplit
from uuid import uuid4

import numpy as np

# MIME (top-level type or full type) -> file extension used when persisting.
_MIME_TO_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/aac": ".aac",
    "audio/m4a": ".m4a",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
}

# Output audio container -> (extension, mime type) for OpenAI audio responses.
AUDIO_FORMAT_MIME: dict[str, str] = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/ogg",
}

# response_format -> soundfile (format, subtype); wav and pcm need no encoder
_SOUNDFILE_AUDIO: dict[str, tuple[str, str | None]] = {
    "mp3": ("MP3", None),
    "flac": ("FLAC", None),
    "opus": ("OGG", "OPUS"),
}


# libsndfile's Opus encoder takes only these rates (44.1 kHz fails)
_OPUS_RATES = (8000, 12000, 16000, 24000, 48000)


def audio_formats(stream: bool = False, sample_rate: int | None = None) -> tuple[str, ...]:
    """The ``response_format`` values this frontend can produce.

    A stream carries an open-ended WAV or bare PCM, so only those two; the
    compressed containers need the optional ``soundfile`` (the ``audio`` extra),
    and Opus also needs one of its native sample rates when ``sample_rate`` is given.
    """
    if stream:
        return ("wav", "pcm")
    try:
        import soundfile  # type: ignore  # noqa: F401
    except ImportError:
        return ("wav", "pcm")
    return ("wav", "pcm", *(
        f for f in _SOUNDFILE_AUDIO
        if f != "opus" or sample_rate is None or int(sample_rate) in _OPUS_RATES
    ))


def check_audio_format(fmt: str | None, stream: bool = False, sample_rate: int | None = None) -> str:
    """``fmt`` lower-cased (``wav`` when unset); ValueError when it cannot be produced."""
    fmt = (fmt or "wav").lower()
    allowed = audio_formats(stream, sample_rate)
    if fmt not in allowed:
        how = "streamed" if stream else "produced"
        raise ValueError(
            f"response_format {fmt!r} cannot be {how} by this server; use one of: {', '.join(allowed)}"
        )
    return fmt


def modality_from_mime(mime: str) -> str:
    """Map a MIME type to one of our modality strings (image/audio/video)."""
    top = (mime or "").split("/", 1)[0].lower()
    if top in ("image", "audio", "video"):
        return top
    return "unknown"


def _ext_for(mime: str, fallback: str = ".bin") -> str:
    mime = (mime or "").lower()
    if mime in _MIME_TO_EXT:
        return _MIME_TO_EXT[mime]
    # Fall back to the top-level type's most common extension.
    top = mime.split("/", 1)[0]
    return {"image": ".png", "audio": ".wav", "video": ".mp4"}.get(top, fallback)


# ---------------------------------------------------------------------------
# Inbound: persist request media into upload_dir, return (modality, path)
# ---------------------------------------------------------------------------

def _save_bytes(raw: bytes, mime: str, upload_dir: Path) -> tuple[str, str]:
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid4()}{_ext_for(mime)}"
    path.write_bytes(raw)
    return modality_from_mime(mime), str(path)


def _decode_base64_payload(payload: str) -> bytes:
    """Decode client-provided base64 after normalizing common wire variants."""
    # OpenAI-compatible clients may omit padding or wrap the payload at a
    # fixed column width. Normalize both forms before asking the decoder to
    # validate the alphabet, so malformed requests fail as client errors.
    cleaned = "".join(payload.split())
    padded = cleaned + "=" * (-len(cleaned) % 4)
    return base64.b64decode(padded, validate=True)


def save_data_url(data_url: str, upload_dir: Path) -> tuple[str, str]:
    """Persist a ``data:<mime>;base64,<payload>`` URL. Returns (modality, path)."""
    header, _, payload = data_url.partition(",")
    if not payload:
        raise ValueError("Malformed data URL: missing payload")
    mime = header[len("data:"):].split(";", 1)[0] or "application/octet-stream"
    raw = _decode_base64_payload(payload)
    return _save_bytes(raw, mime, upload_dir)


def save_base64(b64: str, fmt: str, modality_hint: str, upload_dir: Path) -> tuple[str, str]:
    """Persist a bare base64 blob with a known ``fmt`` (e.g. ``"wav"``)."""
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    raw = _decode_base64_payload(b64)
    # Sanitize the client-controlled fmt: alphanumerics only, so it cannot
    # inject path separators into the upload path.
    clean = "".join(c for c in fmt.lstrip(".") if c.isalnum())
    ext = "." + clean if clean else ".bin"
    path = upload_dir / f"{uuid4()}{ext}"
    path.write_bytes(raw)
    return modality_hint, str(path)


# Telephone band, the lowest rate speech is recorded or coded at. A clip below it
# is not a voice reference, and resampling it up to the probe's or the model's
# rate multiplies its decoded size by that ratio (a 64 KB WAV at 1 Hz is 2 GB at
# 16 kHz).
MIN_REF_AUDIO_SAMPLE_RATE = 8000
# Voice cloning needs seconds of speech. Twice the longest reference a model
# takes (Qwen3-TTS refuses over 30 s, Chatterbox keeps 30 s), so this never
# refuses a clip a model would use; without it a 66 MB FLAC probe-decodes to
# about 23 GB.
MAX_REF_AUDIO_SECONDS = 60.0
# The probe decode runs in the request thread pool (32 threads); this bounds
# how many decode at once.
REF_AUDIO_DECODE_SLOTS = 4
_REF_AUDIO_PROBE_RATE = 16000
_ref_audio_decodes = threading.BoundedSemaphore(REF_AUDIO_DECODE_SLOTS)


class RefAudioTooLongError(ValueError):
    """A clip that opens as audio but runs past ``MAX_REF_AUDIO_SECONDS``."""


def save_base64_audio(b64: str, upload_dir: Path, decoder: str) -> str | None:
    """Persist bare base64 that passes :func:`check_ref_audio`; return its path,
    or None (leaving nothing behind) when it is not base64 or not loadable audio.

    A path such as ``/etc/passwd`` is valid base64 too, so the alphabet cannot
    tell them apart; the decoder can. A clip that is audio but too long raises
    :class:`RefAudioTooLongError`.
    """
    try:
        _, path = save_base64(b64, "wav", "audio", upload_dir)
    except ValueError:
        return None
    try:
        check_ref_audio(path, decoder)
    except RefAudioTooLongError:
        raise
    except ValueError:
        return None
    return path


def check_ref_audio(path: str, decoder: str) -> None:
    """Open the reference clip at ``path`` the way the model's ``load_audio``
    will; on failure delete it and raise ValueError.

    ``decoder`` is the library that ``load_audio`` uses, ``"soundfile"`` or
    ``"torchcodec"``. torchcodec decodes the clip, not just its header: some
    FFmpeg builds parse a container header (MAT4, MAT5) they cannot decode. A
    clip whose header rate is below ``MIN_REF_AUDIO_SAMPLE_RATE``, or whose
    header duration is over ``MAX_REF_AUDIO_SECONDS``, is refused before any
    decode; the decode itself stops just past ``MAX_REF_AUDIO_SECONDS``, so a
    header that hides or understates the length cannot make it unbounded.
    """
    try:
        if decoder == "torchcodec":
            from torchcodec.decoders import AudioDecoder

            header = AudioDecoder(path).metadata
            _require_ref_audio_rate(header.sample_rate)
            _require_ref_audio_seconds(header.duration_seconds_from_header)
            with _ref_audio_decodes:
                # Mono at a fixed rate: a small compressed upload can declare many
                # channels at a high rate, and decoding it natively takes gigabytes.
                samples = AudioDecoder(
                    path, sample_rate=_REF_AUDIO_PROBE_RATE, num_channels=1
                ).get_samples_played_in_range(0.0, MAX_REF_AUDIO_SECONDS + 1.0)
            if samples.duration_seconds == 0:
                raise ValueError("reference audio has no samples in its first seconds")
            _require_ref_audio_seconds(samples.pts_seconds + samples.duration_seconds)
        else:
            import soundfile

            from mstar.model.media_limits import count_sound_file_seconds, sound_file_seconds

            info = soundfile.info(path)
            _require_ref_audio_rate(info.samplerate)
            seconds = sound_file_seconds(info.frames, info.samplerate)
            if seconds is None:
                # no sample count in the header (a FLAC written to a pipe)
                with _ref_audio_decodes:
                    seconds = count_sound_file_seconds(path, MAX_REF_AUDIO_SECONDS)
            _require_ref_audio_seconds(seconds)
    except RefAudioTooLongError:
        Path(path).unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 — not loadable here, so the model cannot load it either
        Path(path).unlink(missing_ok=True)
        raise ValueError(
            f"ref_audio must be audio the model can decode, at {MIN_REF_AUDIO_SAMPLE_RATE} Hz or more"
        ) from exc


def _require_ref_audio_rate(rate: int | None) -> None:
    if not rate or rate < MIN_REF_AUDIO_SAMPLE_RATE:
        raise ValueError(f"reference audio at {rate} Hz is below {MIN_REF_AUDIO_SAMPLE_RATE} Hz")


def _require_ref_audio_seconds(seconds: float | None) -> None:
    if seconds is not None and seconds > MAX_REF_AUDIO_SECONDS:
        raise RefAudioTooLongError(
            f"ref_audio must be {MAX_REF_AUDIO_SECONDS:g} s or shorter; voice cloning needs a few seconds of speech"
        )


logger = logging.getLogger(__name__)

REMOTE_MAX_BYTES = 64 * 2**20
REMOTE_MAX_REDIRECTS = 3
REMOTE_USER_AGENT = "mstar"
_URL_SAFE = "".join(map(chr, range(0x21, 0x7F)))  # printable ASCII, so existing %-escapes stay
# Refused unless the host is allowlisted: loopback, private, link-local (cloud
# metadata at 169.254.169.254), shared/CGNAT, multicast, reserved. The same list
# is in rust/server/src/media.rs.
_BLOCKED_NETS = tuple(ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12",
    "192.0.0.0/24", "192.168.0.0/16", "198.18.0.0/15", "224.0.0.0/4", "240.0.0.0/4",
    "::/128", "::1/128", "64:ff9b:1::/48", "fc00::/7", "fe80::/10", "fec0::/10", "ff00::/8",
))
# IPv6 forms that reach the IPv4 address in their low 32 bits: NAT64 and IPv4-compatible.
_V4_IN_LOW_BITS = (ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("::/96"))
_IPV4_NUMBER = re.compile(r"[0-9]+|0x[0-9a-f]*")
# Both URL parsers delete these anywhere in a URL, so a host written with them
# would match the allowlist as if they were not there; such URLs are refused.
_URL_DROPPED = "\t\r\n"


class RemoteFetchError(ValueError):
    """A remote media fetch was refused or failed; the message is safe to return to the client."""


def _fetch_error(host: str, reason: str, detail: object = "") -> RemoteFetchError:
    logger.warning("remote media fetch from %s failed: %s %s", host, reason, detail)
    return RemoteFetchError(f"remote media fetch failed: {reason}")


def _blocked_address(addr: str) -> bool:
    ip = ipaddress.ip_address(addr)
    if ip.version == 6:
        # IPv4-mapped, 6to4 (2002::/16), NAT64 and IPv4-compatible addresses are
        # judged by the IPv4 address they carry. NAT64 is not refused outright:
        # on an IPv6-only network with DNS64 every IPv4-only host resolves into it.
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip.sixtofour is not None:
            ip = ip.sixtofour
        elif any(ip in net for net in _V4_IN_LOW_BITS):
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return any(ip in net for net in _BLOCKED_NETS)


def _canonical_host(host: str) -> bool:
    """Whether ``host`` (lower-cased, no brackets or trailing dot) is spelled the way
    the Rust frontend's URL parser prints it: an ASCII name, dotted-decimal IPv4 or
    compressed IPv6. ``2130706433``, ``0x7f000001``, ``127.1`` and ``::ffff:127.0.0.1``
    are not; a name whose last label is a number is an IPv4 address to that parser."""
    if not host.isascii() or "%" in host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return not _IPV4_NUMBER.fullmatch(host.rsplit(".", 1)[-1])
    if ip.version == 6 and ip.ipv4_mapped is not None:  # the URL parser prints these in hex
        v4 = int(ip.ipv4_mapped)
        return host == f"::ffff:{v4 >> 16:x}:{v4 & 0xFFFF:x}"
    return ip.compressed == host


def _host_allowlisted(host: str, patterns: list[str]) -> bool:
    """``host`` equals an entry, or ends in ``.suffix`` for a ``*.suffix`` entry. Only a
    canonically spelled host can match, so another spelling of a listed address gets the
    address check instead, as in the Rust frontend."""
    return _canonical_host(host) and any(
        host == p or (p.startswith("*.") and host.endswith(p[1:])) for p in patterns)


def _time_left(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError
    return left


class _Watchdog:
    """Shuts a connected socket down at ``deadline``, so no read or write on it
    (TLS handshake, headers, chunk framing, body) can outlast the fetch timeout.
    It holds a duplicate of the socket, so the descriptor it shuts down cannot
    have been reused by another connection in the meantime."""

    def __init__(self, sock: socket.socket, deadline: float):
        self._sock, self._lock, self.fired = sock.dup(), threading.Lock(), False
        self._timer = threading.Timer(max(deadline - time.monotonic(), 0.0), self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self):
        with self._lock:
            if self._sock is not None:
                self.fired = True
                try:
                    self._sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def close(self):
        self._timer.cancel()
        with self._lock:
            if self._sock is not None:
                self._sock.close()
                self._sock = None


class _PinnedConnection(http.client.HTTPConnection):
    """Connects to addresses vetted up front, so a second DNS answer cannot move
    the connection; the Host header and TLS name stay the URL's host. Each
    connect attempt gets only what is left of ``deadline``, and a watchdog
    bounds everything after it."""

    def __init__(self, host: str, port: int, addrs: list, tls: bool, deadline: float):
        super().__init__(host, port, timeout=_time_left(deadline))
        if tls:  # the Host header leaves out port 443, as HTTPSConnection does
            self.default_port = http.client.HTTPS_PORT
        self._addrs, self._tls, self._deadline = addrs, tls, deadline
        self.watchdog: _Watchdog | None = None

    def connect(self):
        err: OSError = OSError("no address")
        for addr in self._addrs:
            try:
                sock = socket.create_connection(addr[:2], _time_left(self._deadline))
            except OSError as e:
                err = e
                continue
            self.watchdog = _Watchdog(sock, self._deadline)
            self.sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self.host) if self._tls else sock
            return
        raise err

    def timed_out(self) -> bool:
        return self.watchdog is not None and self.watchdog.fired


def save_remote_url(url: str, upload_dir: Path, timeout: float = 30.0) -> tuple[str, str]:
    """Download an ``http(s)`` URL into ``upload_dir``. Returns (modality, path).

    With ``MSTAR_REMOTE_ALLOWED_HOSTS`` set (comma-separated hosts and
    ``*.suffix`` patterns) only those hosts are fetched. A host that is not
    allowlisted is also refused when it resolves to a blocked address, and the
    connection goes to the vetted address. Every redirect hop is checked again
    (at most ``REMOTE_MAX_REDIRECTS``), the body is streamed to disk and capped
    at ``MSTAR_REMOTE_MAX_BYTES``, and ``timeout`` bounds the whole fetch
    (name resolution aside). Failures raise :class:`RemoteFetchError` with a
    short message; the details are logged server-side only.
    """
    patterns = [p.strip().strip("[]").rstrip(".").lower() for p in
                os.getenv("MSTAR_REMOTE_ALLOWED_HOSTS", "").split(",") if p.strip()]
    raw_max = os.getenv("MSTAR_REMOTE_MAX_BYTES", "").strip()
    max_bytes = int(raw_max) if raw_max.isdigit() else REMOTE_MAX_BYTES
    deadline = time.monotonic() + timeout
    host = ""
    for _ in range(REMOTE_MAX_REDIRECTS + 1):
        if any(c in url for c in _URL_DROPPED):  # urlsplit would drop them, also from the host
            raise _fetch_error("", "invalid URL", repr(url))
        try:
            parsed = urlsplit(url)  # not urlparse, which would drop ;params from the path
            host = (parsed.hostname or "").rstrip(".")
            port = parsed.port
        except ValueError as e:  # a bad port, or a bracketed host that is not an address
            raise _fetch_error("", "invalid URL", e) from None
        allowlisted = _host_allowlisted(host, patterns)
        if parsed.scheme not in ("http", "https") or not host:
            raise _fetch_error(host, "invalid URL", url)
        if patterns and not allowlisted:
            raise _fetch_error(host, "host not allowed", "not in MSTAR_REMOTE_ALLOWED_HOSTS")
        conn = None
        try:
            tls = parsed.scheme == "https"
            port = port or (443 if tls else 80)
            addrs = [info[4] for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]
            if not allowlisted and any(_blocked_address(a[0]) for a in addrs):
                raise _fetch_error(host, "host not allowed", f"resolves to {[a[0] for a in addrs]}")
            conn = _PinnedConnection(host, port, addrs, tls, deadline)
            # percent-encode spaces, controls and non-ASCII, as the Rust frontend's URL parser does
            target = quote((parsed.path or "/") + (f"?{parsed.query}" if parsed.query else ""), safe=_URL_SAFE)
            conn.request("GET", target, headers={"User-Agent": REMOTE_USER_AGENT})
            resp = conn.getresponse()
            if resp.status in (301, 302, 303, 307, 308):
                location = resp.getheader("Location")
                if not location:
                    raise _fetch_error(host, "bad redirect", f"{resp.status} without Location")
                if any(c in location for c in _URL_DROPPED):
                    raise _fetch_error(host, "invalid URL", repr(location))
                url = urljoin(url, location)
                continue
            if not 200 <= resp.status < 300:
                raise _fetch_error(host, f"upstream returned {resp.status}")
            return _stream_to_upload(resp, conn, url, Path(upload_dir), max_bytes, host)
        except RemoteFetchError:
            raise
        # before ValueError: a failed certificate check (SSLCertVerificationError) is both;
        # OverflowError: http.client takes a chunk size like -1000...0 and fails reading it
        except (OSError, http.client.HTTPException, OverflowError) as e:
            timed_out = (isinstance(e, TimeoutError) or time.monotonic() >= deadline
                         or (conn is not None and conn.timed_out()))
            raise _fetch_error(host, "timed out" if timed_out else "connection failed", repr(e)) from None
        except ValueError as e:  # a bad port, an unencodable host or path
            raise _fetch_error(host, "invalid URL", e) from None
        finally:
            if conn is not None:
                conn.close()
                if conn.watchdog is not None:
                    conn.watchdog.close()
    raise _fetch_error(host, "too many redirects")


def _stream_to_upload(resp, conn: _PinnedConnection, url: str, upload_dir: Path, max_bytes: int, host: str):
    too_large = f"too large (limit {max_bytes} bytes)"
    if resp.length is not None and resp.length > max_bytes:  # None when chunked or without Content-Length
        raise _fetch_error(host, too_large, f"Content-Length {resp.length}")
    mime = (resp.getheader("Content-Type") or "").split(";", 1)[0].strip().lower()
    if mime in ("", "application/octet-stream"):
        # Infer from the URL path extension.
        suffix = Path(urlparse(url).path).suffix.lower()
        rev = {v: k for k, v in _MIME_TO_EXT.items()}
        mime = rev.get(suffix, "application/octet-stream")
    path = upload_dir / f"{uuid4()}{_ext_for(mime)}"
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        f = open(path, "wb", buffering=0)
    except OSError as e:
        raise _fetch_error(host, "could not save the file", e) from None
    size = 0
    try:
        with f:
            while chunk := resp.read1(1 << 16):
                size += len(chunk)
                if size > max_bytes:
                    raise _fetch_error(host, too_large, "body exceeded the limit")
                try:
                    view = memoryview(chunk)
                    while view:  # an unbuffered write may take only part of the chunk
                        written = f.write(view)
                        if not written:
                            raise OSError("the file stopped accepting bytes")
                        view = view[written:]
                except OSError as e:
                    raise _fetch_error(host, "could not save the file", e) from None
        if resp.length:  # the peer closed before sending all of Content-Length
            raise http.client.IncompleteRead(b"", resp.length)
        if conn.timed_out():  # a body that runs to EOF also "ends" when the watchdog shuts the socket
            raise TimeoutError
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return modality_from_mime(mime), str(path)


def resolve_media_ref(ref: str, upload_dir: Path, *, allow_remote: bool = False) -> tuple[str, str]:
    """Resolve a media reference from a request: a data URL, or an http(s) URL
    when remote fetch is allowed. Returns ``(modality, path)``.

    A request cannot name a file on the server, not even one under
    ``upload_dir``: plain paths and other URL schemes are refused without
    accessing client-named server files, so a caller can neither read files nor
    probe for them.
    """
    if ref.startswith("data:"):
        return save_data_url(ref, upload_dir)
    # not urlparse: it raises on a malformed host with parser text the client would see
    if ref.split(":", 1)[0].lower() in ("http", "https"):
        allow_remote = allow_remote or os.getenv("MSTAR_ALLOW_REMOTE", "0") == "1"
        if not allow_remote:
            raise ValueError("Remote media fetch is disabled on this server")
        return save_remote_url(ref, upload_dir)
    raise ValueError("server file paths are refused; send media as a data URL or a file upload")


# How long a submitted request's uploads outlive it: the data worker reads them
# from its queue after the submission returns. The Rust frontend waits 60 s too.
UPLOAD_CLEANUP_DELAY_S = 60.0


def remove_uploads(file_paths: dict[str, list[str]] | None, upload_dir: Path) -> None:
    """Delete a request's files under ``upload_dir`` now. A path elsewhere is
    left alone: no route puts one in ``file_paths`` today, and a file the
    server did not save is never the request's to delete."""
    root = Path(upload_dir).resolve()
    for paths in (file_paths or {}).values():
        for path in paths:
            p = Path(path)
            if not p.resolve().is_relative_to(root):
                continue
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def schedule_upload_cleanup(file_paths: dict[str, list[str]] | None, upload_dir: Path) -> None:
    """:func:`remove_uploads` after :data:`UPLOAD_CLEANUP_DELAY_S`, for a
    request that was submitted and may not have been read yet."""
    if not file_paths:
        return

    def _cleanup() -> None:
        time.sleep(UPLOAD_CLEANUP_DELAY_S)
        remove_uploads(file_paths, upload_dir)

    threading.Thread(target=_cleanup, daemon=True).start()


# ---------------------------------------------------------------------------
# Outbound: wrap raw model output for client surfaces
# ---------------------------------------------------------------------------

# torchaudio's resampling kernel holds (target / g) * (rate / g + 2 * width)
# floats, g = gcd(rate, target), width its filter's half-length (default
# lowpass_filter_width 6, rolloff 0.99): 3.0M for the 47,952 Hz pull-down rate
# to 16 kHz, but 1.15G for a 48,001 Hz header resampled to 24 kHz
MAX_RESAMPLE_KERNEL = 1 << 24
MAX_SAMPLE_RATE = 768_000


def resample_kernel_size(rate: int, target: int) -> int:
    """Floats in torchaudio's default sinc kernel resampling ``rate`` to ``target``."""
    g = math.gcd(rate, target)
    orig, new = rate // g, target // g
    width = math.ceil(6 * orig / (min(orig, new) * 0.99))
    return new * (orig + 2 * width)


def check_resample_rate(rate: int, target: int) -> None:
    """Refuse a file rate outside ``1..MAX_SAMPLE_RATE`` Hz or whose
    resampling to ``target`` would cost more than :data:`MAX_RESAMPLE_KERNEL`
    (every real rate, pull-down rates included, passes)."""
    if not 1 <= rate <= MAX_SAMPLE_RATE or resample_kernel_size(rate, target) > MAX_RESAMPLE_KERNEL:
        raise ValueError(f"unsupported sample rate {rate} Hz; resample the audio to a standard rate")


# Longest request audio any decoder here produces (transcription uploads, chat
# and /generate audio, reference clips): the transcription route's limit.
MAX_AUDIO_SECONDS = 30 * 60
_SOUNDFILE_BLOCK_SECONDS = 10
# libsndfile's MP3 decoder is seamless across reads only at whole 1152-sample
# MPEG frames; a read that splits a frame glitches the samples after it
_SOUNDFILE_BLOCK_ALIGN = 1152


class AudioRefusedError(ValueError):
    """Audio a decoder opened and refuses (too long, or an unsupported rate):
    callers re-raise it rather than fall back to another decoder."""


class AudioTooLongError(AudioRefusedError):
    """Audio that decodes past the limit, whatever its header declares."""

    status_code = 413

    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        super().__init__(f"audio is longer than the {max_seconds:g} s limit")


def read_soundfile_capped(path: str, max_seconds: float = MAX_AUDIO_SECONDS) -> tuple[np.ndarray, int]:
    """libsndfile decode of ``path``: float32 mono at the file's own rate.

    Reads in blocks and counts the frames it decodes, so a header that hides
    or understates the length cannot make the read unbounded; past
    ``max_seconds`` it raises :class:`AudioTooLongError`. A rate outside
    ``1..MAX_SAMPLE_RATE`` is refused (:class:`AudioRefusedError`) before
    any read. A container libsndfile cannot read raises soundfile's own error.
    """
    import soundfile as sf

    with sf.SoundFile(path) as f:
        rate = f.samplerate
        # the frame cap scales with the rate; the range alone here, the
        # kernel to the caller's target is checked after the decode
        try:
            check_resample_rate(rate, rate)
        except ValueError as exc:
            raise AudioRefusedError(str(exc)) from None
        limit = int(max_seconds * rate)
        block = max(1, _SOUNDFILE_BLOCK_SECONDS * rate // _SOUNDFILE_BLOCK_ALIGN) * _SOUNDFILE_BLOCK_ALIGN
        parts, frames = [], 0
        while True:
            data = f.read(min(block, limit + 1 - frames), dtype="float32", always_2d=True)
            if not len(data):
                break
            frames += len(data)
            if frames > limit:
                raise AudioTooLongError(max_seconds)
            parts.append(data.mean(axis=1))
    return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)), rate


def decode_torchcodec_capped(path: str, sample_rate: int, max_seconds: float = MAX_AUDIO_SECONDS):
    """FFmpeg (torchcodec) decode of ``path``: a float32 mono tensor at
    ``sample_rate``, the same samples ``get_all_samples()`` returns.

    The decode stops just past ``max_seconds`` of play time, and the audio is
    refused with :class:`AudioTooLongError` when it plays longer or holds
    more samples than that. The decoder resamples every frame to
    ``sample_rate``, so what is counted is decoded output, not the rate or
    duration the container declares.
    """
    return torchcodec_samples_capped(path, max_seconds, sample_rate=sample_rate, num_channels=1).data[0]


def torchcodec_samples_capped(path: str, max_seconds: float = MAX_AUDIO_SECONDS, **decoder_args):
    """``AudioDecoder(path, **decoder_args)``'s samples from the stream's first
    frame, as ``get_all_samples()`` returns them, decoding just past
    ``max_seconds`` of play time; refused with :class:`AudioTooLongError` when
    the audio plays longer or holds more samples than that."""
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(path, **decoder_args)
    # play time counts from the stream's first timestamp, which need not be 0
    start = max(0.0, getattr(decoder.metadata, "begin_stream_seconds_from_header", None) or 0.0)
    samples = decoder.get_samples_played_in_range(start, start + max_seconds + 1.0)
    rate = decoder_args.get("sample_rate") or samples.sample_rate
    played = samples.pts_seconds - start + samples.duration_seconds
    if played > max_seconds or samples.data.shape[-1] > max_seconds * rate:
        raise AudioTooLongError(max_seconds)
    return samples


def decode_audio(path: str, sample_rate: int = 16000, max_seconds: float = MAX_AUDIO_SECONDS):
    """Decode an audio file to a float32 mono numpy waveform at ``sample_rate``,
    refused with :class:`AudioTooLongError` past ``max_seconds``.

    libsndfile (``soundfile``) first — no FFmpeg needed for wav/flac/ogg/mp3 —
    then torchcodec for the rest (m4a, webm, ...).
    """
    try:
        audio, sr = read_soundfile_capped(path, max_seconds)
    except AudioRefusedError:
        raise
    except Exception:  # noqa: BLE001 — not a libsndfile container
        return decode_torchcodec_capped(path, sample_rate, max_seconds).numpy().astype(np.float32)
    check_resample_rate(sr, sample_rate)
    if sr != sample_rate:
        try:
            import torch
            import torchaudio

            audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, sample_rate).numpy()
        except ImportError:
            idx = np.linspace(0, len(audio) - 1, int(round(len(audio) * sample_rate / sr)))
            audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return audio


def window_stop(start: int, total: int, step: int, balanced: bool) -> int:
    """Nominal end of the window starting at ``start``: ``step`` samples on,
    or, ``balanced``, an equal share of what is left over the fewest windows
    of at most ``step`` (so the last window is never a sliver)."""
    if not balanced:
        return min(total, start + step)
    remaining = total - start
    windows = -(-remaining // step)
    return start + -(-remaining // windows)


def split_windows(
    audio, window_seconds: float, sample_rate: int = 16000,
    search_seconds: float = 0.0, frame_seconds: float = 0.02, balanced: bool = False,
) -> list:
    """Cut a waveform into consecutive windows of at most ``window_seconds``;
    concatenating them gives the input back sample for sample.

    With ``search_seconds`` a cut is moved back from its nominal boundary to
    the end of the quietest ``frame_seconds`` frame within that span (the
    last such frame on a tie, so silence cuts at the boundary): the cut lands
    in a pause rather than on a word when there is one to find. ``balanced``
    places the nominal boundaries as :func:`window_stop` does.
    """
    audio = np.asarray(audio)
    step = int(round(window_seconds * sample_rate))
    search = int(round(search_seconds * sample_rate))
    frame = max(1, int(round(frame_seconds * sample_rate)))
    pieces, start = [], 0
    while start < len(audio):
        stop = window_stop(start, len(audio), step, balanced)
        if stop < len(audio) and search > 0:
            num_frames = (min(search, stop - start - frame)) // frame
            if num_frames > 0:
                span = audio[stop - num_frames * frame:stop].astype(np.float32)
                energy = np.square(span.reshape(num_frames, frame)).mean(axis=1)
                quietest = num_frames - 1 - int(np.argmin(energy[::-1]))
                stop = stop - num_frames * frame + (quietest + 1) * frame
        pieces.append(audio[start:stop])
        start = stop
    return pieces


class StreamResampler:
    """Resample a stream that arrives in pieces of any size (``push``), with
    the same output however it was split. A Kaiser-windowed sinc lowpass at
    the lower Nyquist, applied polyphase: output ``k`` is centered on input
    time ``k * rate_in / rate_out``, so timing is kept exactly. The passband
    is flat to 0.1 dB up to 0.9 of the lower Nyquist and 3 dB down at 0.94
    (7.2 and 7.5 kHz at 24 to 16 kHz), close to what a 16 kHz recording
    holds; anything above that Nyquist, which would fold back, is down 46 dB
    or more, and 100 dB from 1.06 of it on. Each output waits for half a
    kernel of input past its center (72 samples, 3 ms, at 24 to 16 kHz); ``flush``
    emits the rest as if the input went on as
    silence, ``ceil(n_in * rate_out / rate_in)`` samples in all, and starts
    over. The kernel has ``2 * half_width * max(up, down) + 1`` taps for the
    reduced ratio ``up/down``, so callers bound the rates."""

    def __init__(self, rate_in: int, rate_out: int, half_width: int = 48, beta: float = 10.0, rolloff: float = 0.95):
        g = math.gcd(rate_in, rate_out)
        self.up, self.down = rate_out // g, rate_in // g
        self.half = half_width * max(self.up, self.down)        # kernel half length, upsampled samples
        n = np.arange(-self.half, self.half + 1)
        cutoff = rolloff * 0.5 / max(self.up, self.down)          # cycles per upsampled sample
        self.kernel = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(n.size, beta) * self.up
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float64)
        self._base = 0      # input index of _buf[0]
        self._n_in = 0      # input samples pushed
        self._k_out = 0     # output samples emitted

    def push(self, x: np.ndarray) -> np.ndarray:
        self._buf = np.concatenate([self._buf, np.asarray(x, dtype=np.float64)])
        self._n_in += int(np.size(x))
        # output k is final once every input within half a kernel of it is in
        return self._emit(-(-(self._n_in * self.up - self.half) // self.down))

    def held(self, extra: int = 0) -> int:
        """Output samples ``flush`` would still emit, after ``extra`` more
        input samples."""
        return -(-((self._n_in + extra) * self.up) // self.down) - self._k_out

    def flush(self) -> np.ndarray:
        out = self._emit(-(-(self._n_in * self.up) // self.down))
        self.reset()
        return out

    def _emit(self, k_end: int) -> np.ndarray:
        up, down, half = self.up, self.down, self.half
        if k_end <= self._k_out or not self._buf.size:
            n, self._k_out = max(0, k_end - self._k_out), max(self._k_out, k_end)
            return np.zeros(n, dtype=np.float32)
        span = half // up + 1
        taps = np.arange(-span, span + 1)[None, :]
        out = []
        for k0 in range(self._k_out, k_end, 4096):              # bounds the gather to 4096 rows
            t = np.arange(k0, min(k0 + 4096, k_end))[:, None] * down   # centers, upsampled
            ns = t // up + taps                                  # candidate input indices
            idx = t - ns * up + half
            ok = (idx >= 0) & (idx <= 2 * half) & (ns >= 0) & (ns < self._n_in)
            vals = self._buf[np.clip(ns - self._base, 0, self._buf.size - 1)]
            out.append((np.where(ok, vals * self.kernel[np.clip(idx, 0, 2 * half)], 0.0)).sum(axis=1))
        self._k_out = k_end
        keep = max(self._base, -(-(k_end * down - half) // up))  # first input the next output needs
        self._buf = self._buf[keep - self._base:]
        self._base = keep
        return np.concatenate(out).astype(np.float32)


def write_wav(audio, path: str, sample_rate: int = 16000) -> str:
    """Float waveform -> 16-bit PCM WAV file at ``path``. The inverse of
    reading PCM16 as ``int16 / 32768``, so 16-bit audio comes back sample for
    sample."""
    scaled = np.rint(np.asarray(audio, dtype=np.float64) * 32768.0)
    pcm = np.clip(scaled, -32768, 32767).astype("<i2").tobytes()
    Path(path).write_bytes(pcm16_to_wav_bytes(pcm, sample_rate))
    return path


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Wrap raw little-endian 16-bit PCM (the model's audio output) into a WAV blob."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)
    return buf.getvalue()


def pcm16_to_container(pcm: bytes, sample_rate: int, fmt: str = "wav") -> tuple[bytes, str]:
    """Encode raw 16-bit PCM into ``fmt``. Returns ``(bytes, mime_type)``.

    ``wav`` and ``pcm`` use the stdlib (the bytes are already PCM_16); the
    compressed formats use the optional ``soundfile``. A format this server
    cannot produce raises ValueError (see :func:`check_audio_format`) rather
    than coming back as WAV under another name.
    """
    fmt = check_audio_format(fmt, sample_rate=sample_rate)
    if fmt == "wav":
        return pcm16_to_wav_bytes(pcm, sample_rate), AUDIO_FORMAT_MIME["wav"]
    if fmt == "pcm":
        return pcm, AUDIO_FORMAT_MIME["pcm"]

    import soundfile as sf  # type: ignore

    sf_format, subtype = _SOUNDFILE_AUDIO[fmt]
    audio = np.frombuffer(pcm, dtype="<i2")
    buf = io.BytesIO()
    sf.write(buf, audio, int(sample_rate), format=sf_format, subtype=subtype)
    return buf.getvalue(), AUDIO_FORMAT_MIME[fmt]


def mux_mp4_with_pcm16(
    video_mp4: bytes, pcm: bytes, sample_rate: int, num_channels: int = 2,
    video_fps: float | None = None,
) -> bytes:
    """Mux an H.264 mp4 and raw interleaved 16-bit PCM into one mp4 with an AAC
    audio track (stream-copies the video). ``video_fps`` is the true frame rate
    of the content: when it differs from the container's rate the video
    timestamps are rescaled (no re-encode) so playback runs at the real rate
    and stays in sync with the audio track. Requires the ``ffmpeg`` binary;
    raises on failure — callers fall back to the video-only mp4."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        vpath = Path(tmp) / "v.mp4"
        apath = Path(tmp) / "a.pcm"
        opath = Path(tmp) / "out.mp4"
        vpath.write_bytes(video_mp4)
        apath.write_bytes(pcm)
        itsscale = None
        if video_fps:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", str(vpath)],
                capture_output=True, text=True, timeout=30, check=False,
            )
            try:
                num, den = probe.stdout.strip().split("/")
                container_fps = float(num) / float(den)
                if container_fps > 0 and abs(container_fps - video_fps) > 1e-3:
                    itsscale = container_fps / float(video_fps)
            except (ValueError, ZeroDivisionError):
                pass
        cmd = ["ffmpeg", "-y", "-v", "error"]
        if itsscale is not None:
            cmd += ["-itsscale", f"{itsscale:.6f}"]
        cmd += [
            "-i", str(vpath),
            "-f", "s16le", "-ar", str(int(sample_rate)), "-ac", str(int(num_channels)), "-i", str(apath),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(opath),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed: {proc.stderr.strip()[:500]}")
        return opath.read_bytes()


def wav_stream_header(sample_rate: int, num_channels: int = 1, bits: int = 16) -> bytes:
    """A 44-byte WAV header with streaming (unknown-length) size fields.

    Used to stream TTS audio over a single HTTP response: emit this header, then
    16-bit PCM frames as they arrive. The 0xFFFFFFFF placeholders signal an
    open-ended stream, which players and the OpenAI client's
    ``stream_to_file`` handle.
    """
    import struct

    byte_rate = sample_rate * num_channels * bits // 8
    block_align = num_channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


def png_to_data_url(png_bytes: bytes) -> str:
    """Encode PNG image bytes (the model's image output) as a data URL."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
