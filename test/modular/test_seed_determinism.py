"""Unit tests for the per-request RNG seed used by the CodePredictor engine.

The CodePredictor's CUDA-graph sampling buffer was previously initialized with
``torch.randint(0, 2**32, ...)`` on every replay, which made the sampler
non-reproducible across runs and (more importantly for upcoming TP work)
non-coherent across tensor-parallel ranks. These tests pin the contract of
the deterministic seed path:

  * ``req_id_to_seed`` is a stable, side-effect-free hash of the request id.
  * Within a runner, the seed advances per (rid, decode_step) — identical rid
    in two consecutive decode steps yields different seeds.
  * Across two fresh runner instances (i.e. a process restart, or two TP
    rank workers built from the same config), the same rid sequence yields
    the same seeds.
  * Distinct rids in the same batch yield distinct seeds.
  * Removing a request resets the per-rid step counter.

The tests are CPU-only — no CUDA, no model — and exercise
``CodePredictorCudaGraphRunner._update_sampling_buffers`` directly with a
hand-rolled minimal state dict.
"""
from __future__ import annotations

import types

import torch

from mminf.engine.code_predictor_engine import CodePredictorCudaGraphRunner
from mminf.utils.sampling import Sampler, req_id_to_seed

# ---------------------------------------------------------------------------
# req_id_to_seed: pure-function contract
# ---------------------------------------------------------------------------

def test_req_id_to_seed_is_deterministic():
    assert req_id_to_seed("req-a") == req_id_to_seed("req-a")


def test_req_id_to_seed_distinct_for_distinct_ids():
    seeds = {req_id_to_seed(f"req-{i}") for i in range(64)}
    assert len(seeds) == 64


def test_req_id_to_seed_fits_in_32_bits():
    for rid in ("a", "abc", "x" * 256, ""):
        s = req_id_to_seed(rid)
        assert 0 <= s < 2**32


def test_req_id_to_seed_matches_md5_contract():
    """Pin the exact hash so callers (e.g. external clients reproducing
    server-side noise) can rely on the documented md5-low-4-bytes mapping."""
    import hashlib

    expected = int.from_bytes(hashlib.md5(b"hello").digest()[:4], "little")
    assert req_id_to_seed("hello") == expected


# ---------------------------------------------------------------------------
# CodePredictorCudaGraphRunner._update_sampling_buffers
# ---------------------------------------------------------------------------

MAX_BS = 8
VOCAB = 1024


def _make_runner(device: str = "cpu") -> CodePredictorCudaGraphRunner:
    """Construct a runner with the minimal state needed by
    ``_update_sampling_buffers`` only. Skips the real ``__init__`` so we
    don't need a real submodule / cuda_graph machinery."""
    runner = CodePredictorCudaGraphRunner.__new__(CodePredictorCudaGraphRunner)
    runner.cp_cfg = types.SimpleNamespace(vocab_size=VOCAB)
    runner.sampler = Sampler()
    runner.device = torch.device(device)
    runner._step_count = {}
    runner._shared_bufs = {
        "temperature_buf": torch.zeros(MAX_BS, dtype=torch.float32),
        "top_k_buf": torch.zeros(MAX_BS, dtype=torch.int32),
        "top_p_buf": torch.zeros(MAX_BS, dtype=torch.float32),
        "seed_buf": torch.zeros(MAX_BS, dtype=torch.long),
        "offset_buf": torch.zeros(MAX_BS, dtype=torch.long),
    }
    return runner


def _seeds(runner: CodePredictorCudaGraphRunner, n: int) -> torch.Tensor:
    return runner._shared_bufs["seed_buf"][:n].clone()


def test_update_sampling_buffers_distinct_per_request():
    rids = ["alpha", "beta", "gamma"]
    runner = _make_runner()
    runner._update_sampling_buffers(rids, padded_bs=4)
    seeds = _seeds(runner, 3).tolist()
    assert len(set(seeds)) == 3, f"expected 3 distinct seeds, got {seeds}"


def test_update_sampling_buffers_advances_per_decode_step():
    runner = _make_runner()
    runner._update_sampling_buffers(["alpha"], padded_bs=2)
    s1 = runner._shared_bufs["seed_buf"][0].item()
    runner._update_sampling_buffers(["alpha"], padded_bs=2)
    s2 = runner._shared_bufs["seed_buf"][0].item()
    assert s1 != s2, (
        "Same rid across consecutive decode steps must yield different seeds; "
        "otherwise FlashInfer would reuse the same RNG slot per token."
    )


def test_update_sampling_buffers_reproducible_across_runners():
    """Two fresh runner instances (mimicking process restart or two TP ranks
    built from the same config) must produce identical seeds for the same
    rid sequence."""
    rids = ["alpha", "beta", "gamma"]

    r1 = _make_runner()
    r1._update_sampling_buffers(rids, padded_bs=4)
    seeds_1 = _seeds(r1, 3)

    r2 = _make_runner()
    r2._update_sampling_buffers(rids, padded_bs=4)
    seeds_2 = _seeds(r2, 3)

    assert torch.equal(seeds_1, seeds_2)


def test_update_sampling_buffers_pads_with_last_entry():
    """Padding slots beyond ``len(request_ids)`` must repeat the last seed
    so every captured-graph slot is well-defined (matches the
    temperature/top_k/top_p padding convention)."""
    runner = _make_runner()
    runner._update_sampling_buffers(["alpha", "beta"], padded_bs=4)
    seeds = runner._shared_bufs["seed_buf"][:4].tolist()
    assert seeds[2] == seeds[1] == seeds[3]
    assert seeds[0] != seeds[1]


def test_update_sampling_buffers_empty_request_ids_does_not_crash():
    runner = _make_runner()
    runner._update_sampling_buffers([], padded_bs=2)
    seeds = runner._shared_bufs["seed_buf"][:2].tolist()
    assert seeds == [0, 0]


def test_offset_buf_reset_each_call():
    """The offset buffer must be reset each call — it advances inside the
    captured graph (one step per codebook iteration) and must start at 0
    for each new decode step."""
    runner = _make_runner()
    runner._shared_bufs["offset_buf"][:] = 99  # poison
    runner._update_sampling_buffers(["alpha"], padded_bs=2)
    assert torch.all(runner._shared_bufs["offset_buf"] == 0)


# ---------------------------------------------------------------------------
# CodePredictorEngine.remove_request — step-count cleanup
# ---------------------------------------------------------------------------

def test_remove_request_clears_step_count_in_runner():
    """The engine-level ``remove_request`` cleanup is exercised here at the
    runner level: pop the rid from ``_step_count`` and verify a subsequent
    decode step for that rid restarts from step 0 (matching what a fresh
    runner would produce). This is the invariant that
    ``CodePredictorEngine.remove_request`` enforces; the AREngine-level
    superclass call is covered separately in integration tests."""
    runner = _make_runner()
    runner._update_sampling_buffers(["alpha", "beta"], padded_bs=2)
    assert "alpha" in runner._step_count and "beta" in runner._step_count

    runner._step_count.pop("alpha", None)

    assert "alpha" not in runner._step_count
    assert "beta" in runner._step_count

    fresh = _make_runner()
    fresh._update_sampling_buffers(["alpha"], padded_bs=1)
    expected = fresh._shared_bufs["seed_buf"][0].item()

    runner._update_sampling_buffers(["alpha"], padded_bs=1)
    actual = runner._shared_bufs["seed_buf"][0].item()
    assert actual == expected
