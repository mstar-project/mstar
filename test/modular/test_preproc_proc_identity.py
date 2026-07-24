"""MSTAR_PREPROC_PROC: off-process request preprocessing is byte-identical +
fails safe.

Pure CPU, no GPU. Exercises the real preproc child pool with the real Qwen
tokenizer/processor model (loaded from the local HF cache) on a real small image,
plus in-process checks of the wire round-trip, in-order admission, cancellation,
and the inline-fallback path.

Run:
    HF_HOME=<hf-cache-dir> HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" \
    PYTHONPATH=<worktree> <venv>/bin/python -m pytest \
        test/modular/test_preproc_proc_identity.py -x -q
or just execute the file directly (it self-runs without pytest).
"""

from __future__ import annotations

import os
import tempfile
import time

import torch

from mstar.api_server.preproc_proc import (
    PreprocClient,
    _tensor_from_wire,
    _tensor_to_wire,
    preprocess_tensors,
)
from mstar.api_server.request_types import PreprocessInput

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


def _make_image(path: str, h: int = 56, w: int = 72) -> None:
    """Write a deterministic RGB PNG (a gradient) so decode is reproducible."""
    from PIL import Image
    yy, xx = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing="ij"
    )
    r = (xx * 255 // max(1, w - 1)).to(torch.uint8)
    g = (yy * 255 // max(1, h - 1)).to(torch.uint8)
    b = ((xx + yy) * 255 // max(1, w + h - 2)).to(torch.uint8)
    arr = torch.stack([r, g, b], dim=-1).numpy()  # HWC uint8
    Image.fromarray(arr, mode="RGB").save(path)


def _make_input(rid: str, image_path: str) -> PreprocessInput:
    return PreprocessInput(
        request_id=rid,
        text="Describe this image.",
        file_paths={"image": [image_path]},
        input_modalities=["image", "text"],
        output_modalities=["text"],
        model_kwargs={},
    )


def _assert_tensors_identical(a: dict, b: dict, ctx: str) -> None:
    assert a.keys() == b.keys(), f"{ctx}: key mismatch {a.keys()} vs {b.keys()}"
    for k, av in a.items():
        assert len(av) == len(b[k]), f"{ctx}: len mismatch for {k}"
        for i, (ta, tb) in enumerate(zip(av, b[k], strict=False)):
            assert ta.dtype == tb.dtype, f"{ctx}: dtype {k}[{i}]"
            assert tuple(ta.shape) == tuple(tb.shape), f"{ctx}: shape {k}[{i}]"
            assert torch.equal(ta, tb), f"{ctx}: values differ {k}[{i}]"


def _poll(client: PreprocClient, want: int, timeout: float) -> list:
    got: list = []
    deadline = time.time() + timeout
    while len(got) < want and time.time() < deadline:
        got.extend(client.get_ready())
        if len(got) < want:
            time.sleep(0.05)
    return got


# ---------------------------------------------------------------------------
# 1. Wire round-trip is bitwise identical for every dtype (no numpy bf16 gap).
# ---------------------------------------------------------------------------

def test_wire_roundtrip_bitwise_identical():
    cases = [
        torch.randn(3, 5, dtype=torch.float32),
        torch.randn(4, dtype=torch.float64),
        (torch.randn(2, 3) * 100).to(torch.float16),
        (torch.randn(2, 3) * 100).to(torch.bfloat16),
        torch.randint(-9, 9, (6,), dtype=torch.int64),
        torch.randint(0, 255, (2, 2), dtype=torch.uint8),
        torch.tensor([True, False, True]),
        torch.empty(0, 7, dtype=torch.float32),  # zero-element edge case
    ]
    for t in cases:
        rebuilt = _tensor_from_wire(_tensor_to_wire(t))
        assert rebuilt.dtype == t.dtype
        assert tuple(rebuilt.shape) == tuple(t.shape)
        # bitwise: compare the raw bytes (torch.equal treats NaN!=NaN; here no
        # NaN, but bytes are the strongest identity check anyway).
        assert (
            rebuilt.contiguous().flatten().view(torch.uint8).numpy().tobytes()
            == t.contiguous().flatten().view(torch.uint8).numpy().tobytes()
        )
    print(f"[1] wire round-trip bitwise-identical across {len(cases)} dtypes OK")


# ---------------------------------------------------------------------------
# 2. Real child pool: produced tensors byte-identical to the inline path.
# ---------------------------------------------------------------------------

def test_pool_preprocess_byte_identical(model):
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "im.png")
        _make_image(img)
        inp = _make_input("rid-A", img)

        # Inline reference (the flag-off path).
        ref_tensors, ref_meta = preprocess_tensors(model, inp, "cpu")

        client = PreprocClient(
            model_name=MODEL_NAME, cache_dir=None, model_kwargs=None,
            num_procs=2, num_threads=1,
        )
        try:
            assert client.submit(inp) is True
            got = _poll(client, 1, timeout=120)  # child build + preprocess
            assert len(got) == 1, "no completion from pool (hang?)"
            kind, out_inp, tensors, meta = got[0]
            assert kind == "ok"
            assert out_inp is inp  # original input object handed back
            _assert_tensors_identical(ref_tensors, tensors, "pool-vs-inline")
            assert meta == ref_meta
        finally:
            client.shutdown()
    print("[2] pool preprocess byte-identical to inline (real image) OK")


# ---------------------------------------------------------------------------
# 3. In-order admission across a burst.
# ---------------------------------------------------------------------------

def test_pool_in_submit_order(model):
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "im.png")
        _make_image(img)
        rids = [f"rid-{i}" for i in range(6)]

        client = PreprocClient(
            model_name=MODEL_NAME, cache_dir=None, model_kwargs=None,
            num_procs=3, num_threads=1,
        )
        try:
            for rid in rids:
                assert client.submit(_make_input(rid, img)) is True
            got = _poll(client, len(rids), timeout=180)
            assert len(got) == len(rids), f"got {len(got)}/{len(rids)}"
            order = [c[1].request_id for c in got]
            assert order == rids, f"admission order {order} != submit order {rids}"
        finally:
            client.shutdown()
    print("[3] pool admits completions in submit order OK")


# ---------------------------------------------------------------------------
# 4. Cancellation: a rid aborted before admission is dropped, not admitted.
# ---------------------------------------------------------------------------

def test_pool_cancel_before_admit(model):
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "im.png")
        _make_image(img)
        client = PreprocClient(
            model_name=MODEL_NAME, cache_dir=None, model_kwargs=None,
            num_procs=1, num_threads=1,
        )
        try:
            keep0 = _make_input("keep-0", img)
            drop = _make_input("drop-1", img)
            keep2 = _make_input("keep-2", img)
            assert client.submit(keep0)
            assert client.submit(drop)
            assert client.submit(keep2)
            # Abort the middle request while it is still in flight in the pool.
            client.cancel_rid("drop-1")
            got = _poll(client, 2, timeout=180)
            rids = [c[1].request_id for c in got]
            assert "drop-1" not in rids, f"cancelled rid admitted: {rids}"
            assert rids == ["keep-0", "keep-2"], rids
            # Marker self-cleared once its job drained.
            assert "drop-1" not in client._cancelled
        finally:
            client.shutdown()
    print("[4] cancel-before-admit drops the completion, order preserved OK")


# ---------------------------------------------------------------------------
# 5. Child death -> permanent inline fallback, exactly-once, no hang.
# ---------------------------------------------------------------------------

def test_pool_death_falls_back_inline(model):
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "im.png")
        _make_image(img)
        client = PreprocClient(
            model_name=MODEL_NAME, cache_dir=None, model_kwargs=None,
            num_procs=2, num_threads=1,
        )
        # Warm: one full round-trip so the children are up and serving.
        assert client.submit(_make_input("warm", img))
        assert len(_poll(client, 1, timeout=120)) == 1

        # Submit more, then kill every child hard before they finish.
        pending = [_make_input(f"rid-{i}", img) for i in range(4)]
        submitted = [inp for inp in pending if client.submit(inp)]
        for p in client._procs:
            p.kill()
            p.join(timeout=5)

        # get_ready must recover every outstanding job as an inline completion,
        # exactly once, and then the client is permanently failed.
        recovered = _poll(client, len(submitted), timeout=30)
        assert not client._all_alive()
        assert client._failed, "client should be permanently failed"
        rids = [c[1].request_id for c in recovered]
        assert all(c[0] == "inline" for c in recovered), \
            f"expected inline recovery, got {[c[0] for c in recovered]}"
        assert sorted(rids) == sorted(inp.request_id for inp in submitted), \
            f"recovery set mismatch: {rids}"
        assert len(rids) == len(set(rids)), f"duplicate recovery (not once): {rids}"

        # Future submits go inline (return False) and are not double-processed.
        assert client.submit(_make_input("after", img)) is False
        assert client.get_ready() == []

        # Each recovered request preprocesses byte-identically via the inline path.
        ref, _ = preprocess_tensors(model, submitted[0], "cpu")
        again, _ = preprocess_tensors(model, submitted[0], "cpu")
        _assert_tensors_identical(ref, again, "inline-recovery")
        client.shutdown()
    print(f"[5] child-death inline fallback: {len(recovered)} recovered "
          "exactly-once, no hang OK")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    test_wire_roundtrip_bitwise_identical()
    print("building tokenizer/processor model...")
    m = _build_model()
    test_pool_preprocess_byte_identical(m)
    test_pool_in_submit_order(m)
    test_pool_cancel_before_admit(m)
    test_pool_death_falls_back_inline(m)
    print("\nALL PREPROC_PROC FUNCTIONAL CHECKS PASSED")
