"""MSTAR_DETOK_PROC: off-process detok is byte-identical + fails safe.

Pure CPU, no GPU. Exercises the real detok child process with the real Qwen
tokenizer-only model (loaded from the local HF cache) plus in-process checks of
the deferral build sites and the inline-fallback path.

Run:
    HF_HOME=<hf-cache-dir> HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" \
    PYTHONPATH=<worktree> <venv>/bin/python -m pytest \
        test/modular/test_detok_proc_identity.py -x -q
or just execute the file directly (it self-runs without pytest).
"""

from __future__ import annotations

import os
import queue
import time
from types import SimpleNamespace

import torch

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.api_server.detok_proc import DetokClient
from mstar.api_server.request_types import PendingDetok, ResultChunk

MODEL_NAME = "qwen3_omni"

try:
    import pytest

    @pytest.fixture(scope="module")
    def model():
        return _build_model()
except ImportError:  # direct-run without pytest installed
    pytest = None


def _build_model():
    from mstar.model.registry import HF_MODELS, get_model_class
    return get_model_class(MODEL_NAME)(
        model_path_hf=HF_MODELS[MODEL_NAME]["model_path_hf"],
        cache_dir=None,
    )


def _stub_worker(model, detok_client=None):
    """A PreprocessWorkerThread with only the attributes the emit/build paths
    touch — no queues, no sockets (mirrors the other modular emit tests)."""
    w = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    w.model = model
    w.detok_client = detok_client
    w._detok_enabled = detok_client is not None
    w.out_queue = queue.Queue()
    return w


def _inline_ref(model, ints, dtype, dims, modality="text"):
    return model.postprocess(
        torch.tensor(ints, dtype=dtype).reshape(dims), modality
    )


# ---------------------------------------------------------------------------
# 1. Build-site deferral is byte-identical to the inline path.
# ---------------------------------------------------------------------------

def test_build_inline_chunks_defer_is_byte_identical(model):
    """_build_inline_chunks with the flag on must attach a PendingDetok that,
    once postprocessed, yields the SAME bytes as the flag-off path — including
    the multi-tensor_info split (each tensor_info its own chunk/slice)."""
    ids = model.tokenizer.encode("The quick brown fox jumps over 13 lazy dogs.")
    # Two tensor_info entries: split the token stream across them.
    split = len(ids) // 2
    ti = [
        SimpleNamespace(dtype=torch.long, dims=(split,)),
        SimpleNamespace(dtype=torch.long, dims=(len(ids) - split,)),
    ]
    result = SimpleNamespace(
        request_id="rid-A",
        modality="text",
        graph_edge=SimpleNamespace(name="text_output", tensor_info=ti),
        metadata={"inline_values": {"text_output": list(ids)}},
    )

    off = _stub_worker(model)
    chunks_off = off._build_inline_chunks(result)
    assert [c.pending_detok for c in chunks_off] == [None, None]

    on = _stub_worker(model, detok_client=SimpleNamespace())  # enables defer
    chunks_on = on._build_inline_chunks(result)

    assert len(chunks_on) == len(chunks_off) == 2
    for c_on, c_off in zip(chunks_on, chunks_off, strict=False):
        assert c_on.data == b""
        assert c_on.pending_detok is not None
        pd = c_on.pending_detok
        assert _inline_ref(model, pd.ints, pd.dtype, pd.dims) == c_off.data
        # Metadata + modality unchanged by deferral.
        assert c_on.metadata == c_off.metadata
        assert c_on.modality == c_off.modality
    print("[1] build-site deferral byte-identical (multi-tensor_info split) OK")


# ---------------------------------------------------------------------------
# 2. Real child: interleaved 2-rid stream, byte-identical + per-rid order.
# ---------------------------------------------------------------------------

def test_child_roundtrip_identity_and_order(model):
    client = DetokClient(
        model=model, model_name=MODEL_NAME, cache_dir=None,
        model_kwargs=None, out_queue=queue.Queue(),
    )
    try:
        # Two rids, token stream decoded one token at a time, interleaved.
        texts = {
            "rid-1": model.tokenizer.encode("Off-process detok keeps bytes."),
            "rid-2": model.tokenizer.encode("Second stream, interleaved words!"),
        }
        submitted: dict[str, list[bytes]] = {r: [] for r in texts}
        n = 0
        # Interleave: rid-1[0], rid-2[0], rid-1[1], rid-2[1], ...
        maxlen = max(len(v) for v in texts.values())
        for i in range(maxlen):
            for rid, ids in texts.items():
                if i >= len(ids):
                    continue
                tok = ids[i]
                pd = PendingDetok(ints=[tok], dtype=torch.long, dims=(1,))
                chunk = ResultChunk(
                    request_id=rid, modality="text", data=b"", pending_detok=pd,
                )
                submitted[rid].append(_inline_ref(model, pd.ints, pd.dtype, pd.dims))
                assert client.submit(chunk) is True
                n += 1

        got = _drain(client._out_queue, n, timeout=90)  # child build + decode
        # Per-rid delivery order must match submit order, bytes identical.
        by_rid: dict[str, list[bytes]] = {r: [] for r in texts}
        for c in got:
            assert c.pending_detok is None
            by_rid[c.request_id].append(c.data)
        for rid in texts:
            assert by_rid[rid] == submitted[rid], f"order/bytes mismatch {rid}"
        # And the concatenation decodes to the original strings' bytes.
        for rid, ids in texts.items():
            assert b"".join(by_rid[rid]) == b"".join(
                model.postprocess(torch.tensor([t], dtype=torch.long), "text")
                for t in ids
            )
        print(f"[2] child round-trip: {n} chunks, 2 rids interleaved, "
              "byte-identical + in per-rid order OK")

        # 3. Abort cleanup: drop a rid, then keep streaming another — no hang.
        client.drop_rid("rid-1")
        pd = PendingDetok(ints=list(texts["rid-2"][:3]), dtype=torch.long,
                          dims=(3,))
        c = ResultChunk(request_id="rid-3", modality="text", data=b"",
                        pending_detok=pd)
        assert client.submit(c) is True
        got = _drain(client._out_queue, 1, timeout=30)
        assert got[0].data == _inline_ref(model, pd.ints, pd.dtype, pd.dims)
        print("[3] drop_rid (abort) + continued streaming OK")
    finally:
        client.shutdown()
    assert not client._proc.is_alive()
    print("[4] clean shutdown OK")


# ---------------------------------------------------------------------------
# 5. Failure fallback: child killed -> inline recovery, no hang, correct bytes.
# ---------------------------------------------------------------------------

def test_child_death_falls_back_inline(model):
    out_q: queue.Queue = queue.Queue()
    client = DetokClient(
        model=model, model_name=MODEL_NAME, cache_dir=None,
        model_kwargs=None, out_queue=out_q,
    )
    # Wait until the child is actually up and serving (one warm round-trip),
    # so the kill lands on a live, ready process.
    pd = PendingDetok(ints=[9707], dtype=torch.long, dims=(1,))
    assert client.submit(
        ResultChunk(request_id="warm", modality="text", data=b"", pending_detok=pd)
    )
    _drain(out_q, 1, timeout=90)

    # Kill the child hard; the receiver thread must recover in-flight + future
    # work inline.
    client._proc.kill()
    client._proc.join(timeout=5)

    ref = _inline_ref(model, [9707, 11, 1879], torch.long, (3,))
    recovered = []
    # Some submits may still succeed (queued) before failure is observed, some
    # return False -> caller does inline. Either way every chunk must surface
    # with correct bytes and the request must not hang.
    for _ in range(5):
        pd = PendingDetok(ints=[9707, 11, 1879], dtype=torch.long, dims=(3,))
        chunk = ResultChunk(request_id="rid-x", modality="text", data=b"",
                            pending_detok=pd)
        if not client.submit(chunk):
            # Inline path the real _emit_chunk would take on submit()==False.
            chunk.data = model.postprocess(
                torch.tensor(pd.ints, dtype=pd.dtype).reshape(pd.dims), "text"
            )
            chunk.pending_detok = None
            recovered.append(chunk)
    # Give the receiver's death-drain a moment to flush any outstanding inline.
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            recovered.append(out_q.get(timeout=0.2))
        except queue.Empty:
            if not client.healthy():
                break
    assert not client.healthy(), "client should be permanently failed"
    assert recovered, "no chunks recovered after child death (hang!)"
    for c in recovered:
        assert c.data == ref and c.pending_detok is None
    client.shutdown()
    print(f"[5] child-death inline fallback: {len(recovered)} chunk(s) "
          "recovered, correct bytes, no hang OK")


def _drain(q: queue.Queue, n: int, timeout: float) -> list:
    out = []
    deadline = time.time() + timeout
    while len(out) < n and time.time() < deadline:
        try:
            out.append(q.get(timeout=0.2))
        except queue.Empty:
            continue
    assert len(out) == n, f"expected {n} chunks, got {len(out)} in {timeout}s"
    return out


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    print("building tokenizer-only model...")
    m = _build_model()
    test_build_inline_chunks_defer_is_byte_identical(m)
    test_child_roundtrip_identity_and_order(m)
    test_child_death_falls_back_inline(m)
    print("\nALL DETOK_PROC FUNCTIONAL CHECKS PASSED")
