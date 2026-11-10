"""``MSTAR_MAX_BODY_MB``, read the way mstar-server reads it (no GPU)."""

import pytest

from mstar.api_server.body_limit import max_body_bytes

MIB = 2**20


@pytest.mark.parametrize(
    "raw, mb",
    [(None, 128), ("", 128), ("16", 16), (" 16 ", 16), ("0", 0), ("abc", 128), ("1.5", 128), ("-1", 128)],
)
def test_max_body_bytes_parses_like_the_rust_frontend(monkeypatch, raw, mb):
    if raw is None:
        monkeypatch.delenv("MSTAR_MAX_BODY_MB", raising=False)
    else:
        monkeypatch.setenv("MSTAR_MAX_BODY_MB", raw)
    assert max_body_bytes() == mb * MIB
