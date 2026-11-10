"""What the Python frontend answered, for the parity suites to hold mstar-server to.

These suites compared mstar-server with the FastAPI routes request for request.
Those routes are gone; each test's Python side is replayed from
``python_oracle/<module>.json``, recorded from them before they went (the
recording code is in this file's history). A test asks ``python_side()`` once per
Python answer it needs, in the order it needs them.

Values are stored as JSON with their Python types tagged, so ``bytes``, tuples,
numpy arrays and non-string dict keys come back as they were recorded. A new case
has no recording to replay: write its expectation out instead.
"""

from __future__ import annotations

import base64
import contextvars
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "python_oracle"

# the running test's node id and how many answers it has asked for
_test: contextvars.ContextVar[list | None] = contextvars.ContextVar("python_oracle_test", default=None)
_loaded: dict[str, dict] = {}


def _decode(v):
    if isinstance(v, list):
        return [_decode(x) for x in v]
    if not isinstance(v, dict):
        return v
    if len(v) == 1:
        (tag, body), = v.items()
        if tag == "__bytes__":
            return base64.b64decode(body)
        if tag == "__tuple__":
            return tuple(_decode(x) for x in body)
        if tag == "__set__":
            return {_decode(x) for x in body}
        if tag == "__dict__":
            return {_decode(k): _decode(x) for k, x in body}
        if tag in ("__ndarray__", "__npscalar__"):
            import numpy as np

            if tag == "__ndarray__":
                dtype, shape, data = body
                return np.frombuffer(base64.b64decode(data), dtype=dtype).reshape(shape).copy()
            dtype, data = body
            return np.frombuffer(base64.b64decode(data), dtype=dtype)[0]
    return {k: _decode(x) for k, x in v.items()}


def python_side():
    """The Python frontend's answer recorded for this call of this test."""
    state = _test.get()
    if state is None:
        raise RuntimeError("python_side is called from inside a test")
    nodeid, n = state[0], state[1]
    state[1] += 1
    key = f"{nodeid.split('::', 1)[1]}#{n}"
    module = Path(nodeid.split("::", 1)[0]).stem
    if module not in _loaded:
        path = FIXTURES / f"{module}.json"
        _loaded[module] = json.loads(path.read_text()) if path.exists() else {}
    if key not in _loaded[module]:
        pytest.fail(f"no recorded Python answer {key!r} in python_oracle/{module}.json; "
                    "a new case needs its expectation written out")
    return _decode(_loaded[module][key])


@pytest.fixture(autouse=True)
def _python_oracle_test(request):
    token = _test.set([request.node.nodeid, 0])
    yield
    _test.reset(token)
