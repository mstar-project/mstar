"""CPU parity harness for MSTAR_CODEC_CHUNK_EMIT (item c, speech-floor).

The optimization stages Talker codec frames on the producer and writes them into
the Code2Wav StreamBuffer in one batched put per chunk boundary
(StreamBuffer.stage + flush_pending) instead of one put per frame. Correctness
gate: the CONSUMER must see byte-identical chunk windows either way.

This drives a real StreamBuffer with the real LeftContextChunkPolicy (chunk=25,
left_context=25, the Qwen3-Omni codec edge) two ways over the same frame stream
and asserts every popped window is identical. No GPU needed.

Run:  python -m pytest test/modular/test_codec_chunk_emit_parity.py -q
  or:  python test/modular/test_codec_chunk_emit_parity.py
"""

import torch

from mstar.streaming.chunk_policy import LeftContextChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer

CHUNK = 25
LEFT_CTX = 25
NUM_CODES = 16


def _make_buffer():
    return StreamBuffer(
        request_id="r",
        edge_name="codec_tokens",
        from_partition="Talker",
        policy=LeftContextChunkPolicy(chunk=CHUNK, left_context=LEFT_CTX),
    )


def _frames(n):
    # Distinct per-frame content so any misordering/duplication is caught.
    return [
        (f"u{i}", torch.arange(i * NUM_CODES, (i + 1) * NUM_CODES))
        for i in range(n)
    ]


def _drain_all_windows(sbuf):
    """Pop every ready chunk, returning the list of window tensors."""
    windows = []
    while sbuf.has_chunk_ready():
        chunk = sbuf.pop_chunk()
        data = chunk.data.get("data")
        windows.append((None if data is None else data.clone(),
                        chunk.start_offset, chunk.is_final))
    return windows


def _run_per_frame(frames, done=True):
    sbuf = _make_buffer()
    out = []
    for tid, item in frames:
        sbuf.pre_read_register(tid)
        sbuf.put(tid, item)
        out.extend(_drain_all_windows(sbuf))
    if done:
        sbuf.signal_done()
        out.extend(_drain_all_windows(sbuf))
    return out


def _run_coalesced(frames, done=True):
    sbuf = _make_buffer()
    size = sbuf.policy.coalesce_size()
    out = []
    for tid, item in frames:
        sbuf.pre_read_register(tid)   # registration stays per-frame
        sbuf.stage(tid, item)
        if sbuf.num_pending() >= size:
            sbuf.flush_pending()
        out.extend(_drain_all_windows(sbuf))
    if done:
        sbuf.signal_done()            # flushes the < chunk remainder
        out.extend(_drain_all_windows(sbuf))
    return out


def _assert_identical(a, b, label):
    assert len(a) == len(b), f"{label}: window count {len(a)} != {len(b)}"
    for i, (wa, wb) in enumerate(zip(a, b, strict=False)):
        ta, oa, fa = wa
        tb, ob, fb = wb
        assert oa == ob and fa == fb, (
            f"{label} chunk {i}: meta (offset {oa}/{ob}, final {fa}/{fb})"
        )
        if ta is None or tb is None:
            assert ta is None and tb is None, f"{label} chunk {i}: None mismatch"
        else:
            assert torch.equal(ta, tb), f"{label} chunk {i}: window contents"


def test_parity_frame_counts():
    # Exact multiples, partial tails, below-first-chunk, and long streams.
    for n in [0, 1, 24, 25, 26, 49, 50, 51, 75, 99, 100, 137, 250]:
        frames = _frames(n)
        pf = _run_per_frame(frames)
        co = _run_coalesced(frames)
        _assert_identical(pf, co, f"n={n}")

    # Coalesced windows must also equal a straight HF-style chunked slicing of
    # the flat frame stream for a representative length (independent oracle on
    # the first few windows: first=frames[0:25], then frames[0:50], [25:75]...).
    n = 100
    flat = [item for _, item in _frames(n)]
    co = _run_coalesced(_frames(n))
    # window 0: first chunk, no context -> frames[0:25]
    w0 = co[0][0]
    assert torch.equal(w0, torch.stack(flat[0:25]))
    # window 1: frames[0:50] (chunk+left_context, overlap with window 0 tail)
    w1 = co[1][0]
    assert torch.equal(w1, torch.stack(flat[0:50]))
    # window 2: advance by chunk=25 -> frames[25:75]
    w2 = co[2][0]
    assert torch.equal(w2, torch.stack(flat[25:75]))


def test_parity_without_done():
    # Mid-stream (no producer_done): coalesced holds the < chunk remainder in
    # staging, so it must match per-frame which also can't pop a partial chunk.
    for n in [10, 25, 40, 60]:
        _assert_identical(
            _run_per_frame(_frames(n), done=False),
            _run_coalesced(_frames(n), done=False),
            f"nodone n={n}",
        )


if __name__ == "__main__":
    test_parity_frame_counts()
    test_parity_without_done()
    print("codec chunk-emit parity: OK")
