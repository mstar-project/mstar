"""Remote media fetch (MSTAR_ALLOW_REMOTE=1): host allowlist, address checks on
every redirect hop, a size cap while streaming, a whole-fetch timeout, and short
4xx messages that do not echo upstream details. Every server here is local."""
import ipaddress
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from mstar.api_server import media_io

PNG = b"\x89PNG\r\n\x1a\n" + bytes(32)
LEAKY = b"upstream says: db.internal.corp 10.1.2.3 secret-token"


class _Upstream:
    """A local HTTP server; ``routes`` maps a path to a function writing the reply."""

    def __init__(self, host="127.0.0.1"):
        self.requests, self.routes = [], {}
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                upstream.requests.append((self.path, dict(self.headers)))
                upstream.routes[self.path](self)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer((host, 0), Handler)
        self.server.daemon_threads = True
        self.server.handle_error = lambda *args: None  # the client hangs up on refused bodies
        self.host, self.port = self.server.server_address[:2]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def url(self, path, host=None):
        return f"http://{host or self.host}:{self.port}{path}"


def reply(status=200, body=b"", headers=(("Content-Type", "image/png"),), length=None):
    def handle(h):
        h.send_response(status)
        for key, value in headers:
            h.send_header(key, value)
        h.send_header("Content-Length", str(len(body) if length is None else length))
        h.end_headers()
        h.wfile.write(body)
    return handle


def redirect(location, status=302):
    return reply(status, headers=(("Location", location),))


def chunked(total, piece=500):
    def handle(h):
        h.send_response(200)
        h.send_header("Content-Type", "video/mp4")
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        for _ in range(total // piece):
            h.wfile.write(b"%x\r\n%s\r\n" % (piece, bytes(piece)))
        h.wfile.write(b"0\r\n\r\n")
    return handle


def trickle(h):
    h.send_response(200)
    h.send_header("Content-Type", "image/png")
    h.send_header("Content-Length", "100")
    h.end_headers()
    for _ in range(100):
        h.wfile.write(b"x")
        time.sleep(0.05)


def raw(data):
    """Write ``data`` as the whole reply, then close the connection."""
    def handle(h):
        h.close_connection = True
        h.wfile.write(data)
    return handle


def drip(prefix, byte):
    """Write ``prefix``, then one byte every 50 ms for 3 s."""
    def handle(h):
        h.close_connection = True
        h.wfile.write(prefix)
        for _ in range(60):
            h.wfile.write(byte)
            time.sleep(0.05)
    return handle


def stall(h):
    # late headers and one byte, then silence: the stalled read gets only what is left of the timeout
    time.sleep(0.7)
    h.send_response(200)
    h.send_header("Content-Type", "image/png")
    h.send_header("Content-Length", "100")
    h.end_headers()
    h.wfile.write(b"x")
    time.sleep(3)


@pytest.fixture
def upstream(monkeypatch):
    monkeypatch.delenv("MSTAR_REMOTE_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("MSTAR_REMOTE_MAX_BYTES", raising=False)
    servers = []

    def make(host="127.0.0.1"):
        servers.append(_Upstream(host))
        return servers[-1]

    yield make
    for s in servers:
        s.server.shutdown()
        s.server.server_close()


@pytest.fixture
def uploads(tmp_path):
    return tmp_path / "uploads"


def refused(url, uploads, reason, **kw):
    with pytest.raises(media_io.RemoteFetchError) as e:
        media_io.save_remote_url(url, uploads, **kw)
    assert str(e.value) == f"remote media fetch failed: {reason}"
    assert not uploads.exists() or not any(uploads.iterdir())


def resolve_as(monkeypatch, answers):
    """Resolve names through ``answers`` (name -> address, or an iterator of addresses)."""
    real = socket.getaddrinfo

    def fake(host, *args, **kw):
        answer = answers.get(host, host)
        return real(answer if isinstance(answer, str) else next(answer), *args, **kw)
    monkeypatch.setattr(media_io.socket, "getaddrinfo", fake)


def block_only(monkeypatch, net):
    """Treat only ``net`` as internal, so 127.0.0.1 can stand in for a public host."""
    monkeypatch.setattr(media_io, "_BLOCKED_NETS", (ipaddress.ip_network(net),))


@pytest.mark.parametrize("addr", [
    "127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254", "100.100.100.200", "0.0.0.0",
    "224.0.0.1", "239.1.1.1", "255.255.255.255", "::1", "::", "fe80::1", "fe80::1%2", "fd00:ec2::254",
    "ff02::1", "::ffff:127.0.0.1", "::ffff:169.254.169.254",
    # NAT64, 6to4 and IPv4-compatible forms of internal IPv4 addresses, and local-use NAT64
    "64:ff9b::7f00:1", "64:ff9b::a9fe:a9fe", "64:ff9b:1::808:808", "2002:7f00:1::", "2002:a00:1:2::3",
    "::127.0.0.1", "::169.254.169.254",
])
def test_internal_addresses_are_blocked(addr):
    assert media_io._blocked_address(addr)


@pytest.mark.parametrize("addr", [
    "8.8.8.8", "1.1.1.1", "2606:4700:4700::1111", "::ffff:8.8.8.8", "64:ff9b::808:808", "2002:808:808::1",
    "::8.8.8.8",
])
def test_public_addresses_are_not_blocked(addr):
    assert not media_io._blocked_address(addr)


# The same table as only_the_canonical_spelling_matches_the_allowlist in rust/server/src/media.rs.
@pytest.mark.parametrize("url, host", [
    ("http://127.0.0.1:8000/a.png", "127.0.0.1"),
    ("http://127.0.0.1./a.png", "127.0.0.1"),
    ("http://user@Example.ORG/a.png", "example.org"),
    ("https://[::1]:8443/a.png", "::1"),
    ("http://[::ffff:7f00:1]/a.png", "::ffff:7f00:1"),
    ("http://0xg/a.png", "0xg"),
    ("http://2130706433/a.png", None),
    ("http://0x7f000001/a.png", None),
    ("http://127.1/a.png", None),
    ("http://127.0.0.01/a.png", None),
    ("http://0177.0.0.1/a.png", None),
    ("http://[0:0:0:0:0:0:0:1]/a.png", None),
    ("http://[::ffff:127.0.0.1]/a.png", None),
    ("http://%31%32%37.0.0.1/a.png", None),
    ("http://b\u00fccher.example/a.png", None),
])
def test_only_the_canonical_spelling_matches_the_allowlist(url, host):
    written = urlsplit(url).hostname.rstrip(".")
    assert media_io._host_allowlisted(written, [written]) == (host is not None)
    if host is not None:
        assert written == host


def test_other_spellings_of_an_allowlisted_address_are_refused(upstream, uploads, monkeypatch):
    up = upstream()
    up.routes["/a.png"] = reply(body=PNG)
    up.routes["/r"] = redirect(f"//127.1:{up.port}/a.png")
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    for host in ("2130706433", "0x7f000001", "127.1", "127.0.0.01"):
        refused(up.url("/a.png", host=host), uploads, "host not allowed")
    refused(up.url("/r"), uploads, "host not allowed")
    # urlsplit drops tabs and newlines, which would make 127.0.0.\t1 read as 127.0.0.1
    up.routes["/to-tab"] = redirect(f"http://127.0.0.\t1:{up.port}/a.png")
    refused(up.url("/a.png", host="127.0.0.\t1"), uploads, "invalid URL")
    refused(up.url("/to-tab"), uploads, "invalid URL")
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "2130706433")
    refused(up.url("/a.png", host="2130706433"), uploads, "host not allowed")
    assert [p for p, _ in up.requests] == ["/r", "/to-tab"]


def test_allowlisted_host_is_fetched(upstream, uploads, monkeypatch):
    up = upstream()
    up.routes["/a.png"] = reply(body=PNG)
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "example.org, 127.0.0.1")
    modality, path = media_io.save_remote_url(up.url("/a.png"), uploads)
    assert modality == "image" and path.endswith(".png") and open(path, "rb").read() == PNG
    assert up.requests[0][1]["User-Agent"] == "mstar"


def test_hosts_outside_the_allowlist_are_refused(upstream, uploads, monkeypatch):
    up = upstream()
    up.routes["/a.png"] = reply(body=PNG)
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    refused(up.url("/a.png", host="localhost"), uploads, "host not allowed")
    assert up.requests == []


def test_wildcard_entries_match_subdomains_only(upstream, uploads, monkeypatch):
    up = upstream()
    up.routes["/a.png"] = reply(body=PNG)
    resolve_as(monkeypatch, {n: "127.0.0.1" for n in ("cdn.media.test", "media.test", "evilmedia.test")})
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "*.media.test")
    media_io.save_remote_url(up.url("/a.png", host="cdn.media.test"), uploads)
    assert up.requests[0][1]["Host"] == f"cdn.media.test:{up.port}"
    for host in ("media.test", "evilmedia.test"):
        refused(up.url("/a.png", host=host), uploads / host, "host not allowed")
    assert len(up.requests) == 1


def test_internal_addresses_are_refused_unless_allowlisted(upstream, uploads):
    up = upstream()
    up.routes["/a.png"] = reply(body=PNG)
    for host in ("127.0.0.1", "localhost", "[::1]", "169.254.169.254", "10.0.0.1"):
        refused(up.url("/a.png", host=host), uploads, "host not allowed")
    assert up.requests == []


def test_connects_to_the_vetted_address(upstream, uploads, monkeypatch):
    # a second DNS answer pointing inside is never used: the connection goes to the vetted one
    public, internal = upstream("127.0.0.1"), upstream("127.0.0.2")
    block_only(monkeypatch, "127.0.0.2/32")
    resolve_as(monkeypatch, {"rebind.test": iter(["127.0.0.1"] + ["127.0.0.2"] * 10)})
    public.routes["/a.png"] = internal.routes["/a.png"] = reply(body=PNG)
    media_io.save_remote_url(f"http://rebind.test:{public.port}/a.png", uploads)
    assert public.requests[0][1]["Host"] == f"rebind.test:{public.port}" and internal.requests == []


def test_redirect_to_an_internal_address_is_refused(upstream, uploads, monkeypatch):
    public, internal = upstream("127.0.0.1"), upstream("127.0.0.2")
    block_only(monkeypatch, "127.0.0.2/32")
    internal.routes["/a.png"] = reply(body=PNG)
    public.routes["/r"] = redirect(internal.url("/a.png"))
    refused(public.url("/r"), uploads, "host not allowed")
    assert internal.requests == []


def test_redirect_to_a_host_outside_the_allowlist_is_refused(upstream, uploads, monkeypatch):
    up = upstream()
    up.routes["/a.png"] = reply(body=PNG)
    up.routes["/r"] = redirect(up.url("/a.png", host="localhost"))
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    refused(up.url("/r"), uploads, "host not allowed")
    assert [p for p, _ in up.requests] == ["/r"]


def test_redirects_are_followed_up_to_the_cap(upstream, uploads, monkeypatch):
    up = upstream()
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    up.routes.update({"/r1": redirect("/r2", 301), "/r2": redirect(up.url("/r3"), 307),
                      "/r3": redirect("a.png", 308), "/a.png": reply(body=PNG), "/loop": redirect("/loop")})
    _, path = media_io.save_remote_url(up.url("/r1"), uploads)
    assert open(path, "rb").read() == PNG
    up.requests.clear()
    refused(up.url("/loop"), uploads / "loop", "too many redirects")
    assert len(up.requests) == media_io.REMOTE_MAX_REDIRECTS + 1


@pytest.mark.parametrize("location", ["file:///etc/passwd", "ftp://127.0.0.1/x", "data:image/png;base64,AAAA"])
def test_redirect_to_a_non_http_url_is_refused(upstream, uploads, monkeypatch, location):
    up = upstream()
    up.routes["/r"] = redirect(location)
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    refused(up.url("/r"), uploads, "invalid URL")


def test_content_length_over_the_cap_is_refused_before_reading(upstream, uploads, monkeypatch):
    up = upstream()
    up.routes["/big.png"] = reply(body=PNG, length=10**12)  # a body this size would never finish
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    monkeypatch.setenv("MSTAR_REMOTE_MAX_BYTES", "1000")
    started = time.monotonic()
    refused(up.url("/big.png"), uploads, "too large (limit 1000 bytes)", timeout=5)
    assert time.monotonic() - started < 2


def test_body_without_content_length_is_capped_while_streaming(upstream, uploads, monkeypatch):
    up = upstream()
    up.routes.update({"/at.mp4": chunked(1000), "/over.mp4": chunked(5000)})
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    monkeypatch.setenv("MSTAR_REMOTE_MAX_BYTES", "1000")
    refused(up.url("/over.mp4"), uploads, "too large (limit 1000 bytes)")
    modality, path = media_io.save_remote_url(up.url("/at.mp4"), uploads)
    assert modality == "video" and len(open(path, "rb").read()) == 1000
    assert "Content-Length" not in up.requests[0][1]


@pytest.mark.parametrize("status", [403, 404, 500, 502])
def test_upstream_errors_become_a_short_message(upstream, uploads, monkeypatch, status):
    up = upstream()
    up.routes["/a.png"] = reply(status, body=LEAKY, headers=(("Content-Type", "text/plain"),))
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    refused(up.url("/a.png"), uploads, f"upstream returned {status}")


@pytest.mark.parametrize("route", [lambda h: time.sleep(3), trickle], ids=["no-response", "trickled-body"])
def test_the_whole_fetch_is_bounded_by_the_timeout(upstream, uploads, monkeypatch, route):
    up = upstream()
    up.routes["/slow.png"] = route
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    started = time.monotonic()
    refused(up.url("/slow.png"), uploads, "timed out", timeout=0.5)
    assert time.monotonic() - started < 2


def test_a_read_that_stalls_late_still_ends_at_the_deadline(upstream, uploads, monkeypatch):
    up = upstream()
    up.routes["/stall.png"] = stall
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    started = time.monotonic()
    refused(up.url("/stall.png"), uploads, "timed out", timeout=1.0)
    assert time.monotonic() - started < 1.5


@pytest.mark.parametrize("route", [
    drip(b"HTTP/1.1 200 OK\r\nX-Slow: ", b"a"),
    drip(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n", b"0"),
], ids=["headers", "chunk-size-line"])
def test_trickled_headers_and_chunk_framing_end_at_the_deadline(upstream, uploads, monkeypatch, route):
    up = upstream()
    up.routes["/slow.png"] = route
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    started = time.monotonic()
    refused(up.url("/slow.png"), uploads, "timed out", timeout=0.5)
    assert time.monotonic() - started < 1.5


@pytest.mark.parametrize("route", [
    raw(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + b"0" * 100_000 + b"1\r\nx\r\n0\r\n\r\n"),
    raw(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n-1\r\nabc"),
    raw(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n-10000000000000000000000000\r\nabc"),
    raw(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabc"),
], ids=["chunk-size-line-too-long", "negative-chunk-size", "huge-negative-chunk-size", "shorter-than-content-length"])
def test_malformed_bodies_are_refused(upstream, uploads, monkeypatch, route):
    up = upstream()
    up.routes["/bad.png"] = route
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    refused(up.url("/bad.png"), uploads, "connection failed")


def test_a_content_length_past_64_bits_is_too_large(upstream, uploads, monkeypatch):
    up = upstream()
    up.routes["/huge.png"] = raw(b"HTTP/1.1 200 OK\r\nContent-Length: 18446744073709551616\r\n\r\nabc")
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    monkeypatch.setenv("MSTAR_REMOTE_MAX_BYTES", "1000")
    refused(up.url("/huge.png"), uploads, "too large (limit 1000 bytes)")


def test_the_request_target_keeps_params_and_query_and_is_encoded(upstream, uploads, monkeypatch):
    up = upstream()
    up.routes["/a.png;v=2?t=1"] = up.routes["/my%20cat%C3%A9.png"] = reply(body=PNG)
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    media_io.save_remote_url(up.url("/a.png;v=2?t=1"), uploads)
    media_io.save_remote_url(up.url("/my cat\u00e9.png"), uploads)
    assert [p for p, _ in up.requests] == ["/a.png;v=2?t=1", "/my%20cat%C3%A9.png"]


@pytest.mark.parametrize("tls,port,host", [
    (True, 443, "example.com"), (True, 8443, "example.com:8443"),
    (False, 80, "example.com"), (False, 8080, "example.com:8080"),
])
def test_the_host_header_leaves_out_the_default_port(tls, port, host):
    # signed URLs (SigV4, GCS) sign the exact Host value
    conn = media_io._PinnedConnection("example.com", port, [], tls, time.monotonic() + 5)
    sent = []
    conn.send = sent.append  # capture the request instead of connecting
    conn.request("GET", "/a.png")
    assert f"\r\nHost: {host}\r\n".encode() in b"".join(sent)


@pytest.mark.parametrize("url", ["http://[db.internal.corp]/a.png", "http://127.0.0.1:99999/a.png"])
def test_unparseable_urls_are_a_short_message(uploads, monkeypatch, url):
    refused(url, uploads, "invalid URL")
    monkeypatch.setenv("MSTAR_ALLOW_REMOTE", "1")
    with pytest.raises(media_io.RemoteFetchError, match="^remote media fetch failed: invalid URL$"):
        media_io.resolve_media_ref(url, uploads)


@pytest.mark.parametrize("route", [
    raw(b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s"
        % (len(PNG), PNG)),
    raw(b"HTTP/1.0 200 OK\r\nContent-Type: image/png\r\n\r\n" + PNG),
], ids=["connection-close", "runs-to-eof"])
def test_a_body_that_ends_the_connection_is_fetched(upstream, uploads, monkeypatch, route):
    up = upstream()
    up.routes["/a.png"] = route
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    modality, path = media_io.save_remote_url(up.url("/a.png"), uploads)
    assert modality == "image" and open(path, "rb").read() == PNG


def test_a_storage_failure_is_a_short_message(upstream, tmp_path, monkeypatch):
    up = upstream()
    up.routes["/a.png"] = reply(body=PNG)
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    occupied = tmp_path / "uploads"
    occupied.write_bytes(b"not a directory")
    with pytest.raises(media_io.RemoteFetchError) as e:
        media_io.save_remote_url(up.url("/a.png"), occupied)
    assert str(e.value) == "remote media fetch failed: could not save the file"


class _ShortWrites:
    """An unbuffered file whose writes take at most ``step`` bytes, then none once ``cap`` is reached."""

    def __init__(self, f, step, cap):
        self.f, self.step, self.cap = f, step, cap

    def write(self, data):
        n = min(len(data), self.step, self.cap - self.f.tell())
        return self.f.write(data[:n]) if n > 0 else 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.f.close()


@pytest.mark.parametrize("cap", [1 << 20, 5], ids=["short-writes", "file-full"])
def test_a_partial_write_is_finished_or_refused(upstream, uploads, monkeypatch, cap):
    up = upstream()
    up.routes["/a.png"] = reply(body=PNG)
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    monkeypatch.setattr(media_io, "open", lambda *a, **kw: _ShortWrites(open(*a, **kw), 3, cap), raising=False)
    if cap > len(PNG):
        _, path = media_io.save_remote_url(up.url("/a.png"), uploads)
        assert open(path, "rb").read() == PNG
    else:
        refused(up.url("/a.png"), uploads, "could not save the file")


@pytest.mark.parametrize("headers", [(), (("Content-Type", "application/octet-stream"),)], ids=["none", "generic"])
def test_a_missing_or_generic_content_type_falls_back_to_the_url_extension(upstream, uploads, monkeypatch, headers):
    up = upstream()
    up.routes["/a.png"] = reply(body=PNG, headers=headers)
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    modality, path = media_io.save_remote_url(up.url("/a.png"), uploads)
    assert modality == "image" and path.endswith(".png")


def test_connection_failure_does_not_echo_transport_details(uploads, monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # closed again: nothing listens there
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1,db.internal.corp")
    refused(f"http://127.0.0.1:{port}/a.png", uploads, "connection failed")

    def no_such_name(host, *args, **kw):
        raise socket.gaierror(socket.EAI_NONAME, f"{host}: Name or service not known")
    monkeypatch.setattr(media_io.socket, "getaddrinfo", no_such_name)
    refused("http://db.internal.corp/a.png", uploads, "connection failed")


def test_a_failed_certificate_check_is_a_connection_failure(upstream, uploads, monkeypatch):
    # SSLCertVerificationError is a ValueError too; it must not read as a bad URL
    class Untrusted:
        def wrap_socket(self, sock, server_hostname):
            raise ssl.SSLCertVerificationError(1, f"certificate verify failed for {server_hostname}")

    up = upstream()
    monkeypatch.setattr(media_io.ssl, "create_default_context", Untrusted)
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    refused(f"https://127.0.0.1:{up.port}/a.png", uploads, "connection failed")


def test_remote_fetch_stays_off_by_default(upstream, uploads, monkeypatch):
    up = upstream()
    up.routes["/a.png"] = reply(body=PNG)
    monkeypatch.delenv("MSTAR_ALLOW_REMOTE", raising=False)
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "127.0.0.1")
    with pytest.raises(ValueError, match="Remote media fetch is disabled"):
        media_io.resolve_media_ref(up.url("/a.png"), uploads)
    assert up.requests == []
    monkeypatch.setenv("MSTAR_ALLOW_REMOTE", "1")
    assert media_io.resolve_media_ref(up.url("/a.png"), uploads)[0] == "image"
