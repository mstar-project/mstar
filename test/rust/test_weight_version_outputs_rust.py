"""The Rust runtime stamps the weight version on its result frames, tensor and
inline token alike, read off a real inbox; nothing before a version is set.
The API server side is in ``test/modular/test_weight_version_outputs.py``."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modular"))

import pytest

pytest.importorskip("mstar_rust")

from mstar_rust import ZmqCommunicator
from test_delivery_chunk_counts import (  # noqa: E402
    RUNTIME_KWARGS,
    SHAPES,
    WORKER,
    drive,
    tensor_info,
    worker_graphs,
)

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.api_server.request_types import ResultTokens
from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.communication.wire import decode
from mstar.graph.runtime import rust as rust_runtime


def _frames(tmp_path, inline, version):
    prefills, decode_steps = SHAPES["plain"]
    inbox = ZmqCommunicator("api_server", str(tmp_path))
    ZmqCommunicator("conductor", str(tmp_path))  # the reports need a peer
    book = RustTensorBookkeeping()
    runtime = rust_runtime.RustGraphRuntime(
        my_worker_graphs=worker_graphs(decode_steps), bookkeeping=book,
        communicator=SimpleNamespace(_inner=ZmqCommunicator(WORKER, str(tmp_path))),
        **RUNTIME_KWARGS,
    )
    if version is not None:
        runtime.set_weight_version(version)

    def put(uuid):
        book.put_tensor(uuid, tensor_info(uuid))
        book.increment_ref(uuid, 1)

    drive(runtime, put, prefills, decode_steps, inline=inline)
    out = []
    first = inbox.recv_timeout(1000)
    while first is not None:
        out.append(decode(first).body)
        first = inbox.recv_timeout(50)
    assert len(out) == prefills + decode_steps
    assert sum(isinstance(f, ResultTokens) for f in out) == (decode_steps if inline else 0)
    return out


@pytest.mark.parametrize("inline", [False, True], ids=["tensor", "inline"])
def test_no_version_leaves_the_metadata_empty(tmp_path, inline):
    assert [f.metadata for f in _frames(tmp_path, inline, None)] == [{}] * 4


@pytest.mark.parametrize("inline", [False, True], ids=["tensor", "inline"])
def test_every_frame_carries_the_version_once_set(tmp_path, inline):
    frames = _frames(tmp_path, inline, "v7")
    assert [f.metadata for f in frames] == [{"weight_version": "v7"}] * 4


def test_log_probs_and_the_version_share_the_metadata(tmp_path):
    """A tensor frame with a log-prob carries both keys, the log-prob first;
    an inline frame keeps its log-probs in their own column and the version
    in its metadata."""
    from test_delivery_chunk_counts import new_runtime
    from test_delivery_chunk_counts_rust import _script

    runtime, put, sent = new_runtime(5, tmp_path)
    runtime.set_weight_version("v9")
    _script(runtime, put)
    chunks, _ = sent()

    assert all(c.metadata.get("weight_version") == "v9" for c in chunks)
    with_lp = [c.metadata for c in chunks if not isinstance(c, ResultTokens) and "logprobs" in c.metadata]
    assert len(with_lp) == 3
    assert all(list(m) == ["logprobs", "weight_version"] for m in with_lp)
    tokens = [c for c in chunks if isinstance(c, ResultTokens)]
    assert tokens and all(c.metadata == {"weight_version": "v9"} for c in tokens)
    assert any(c.logprobs for c in tokens)
