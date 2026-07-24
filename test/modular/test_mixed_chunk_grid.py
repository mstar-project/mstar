"""W5-P3 mixed-step capture grid growth (MSTAR_MIXED_CHUNK_SIZES).

CPU-only, no GPU, no model weights, no CUDA-graph capture. Three surfaces:

  * ``ThinkerSubmodule.MIXED_BATCH_CHUNK_SIZES`` — the property that turns the
    env var into the capture-grid list ``get_cuda_graph_configs`` iterates.
  * ``MicroScheduler._max_chunk_tokens()`` — the scheduler-side mirror that
    the G1 chunk-size gate (``_chunk_entry_passes_gates``) reads; must agree
    with the submodule's grid or the scheduler either rejects chunks the
    capture actually supports, or (worse) admits a chunk into an uncaptured
    bucket (the UNCAP-IMA failure mode).
  * ``Worker._compute_coadmit_budget`` — the MSTAR_COADMIT budget clamp;
    must auto-widen with the grid instead of the pre-fix hardcoded 512.

Both the submodule property and the scheduler's ``_resolve_mixed_chunk_sizes``
independently parse the SAME env var (duplicated on purpose — the scheduler
must not import the model submodule, same reasoning as
``qwen3_omni_model._PREFILL_CHUNK_BUCKETS``); this file checks they agree.

The scheduler-side resolution is cached at module scope for the process
lifetime (boot-time flag — capture happens once, not a dynflag). Tests that
vary the env var mid-process reset the private cache directly
(``micro_scheduler._mixed_chunk_sizes_cache = None``), the same pattern this
file already uses for ``_mixed_min_decode_cached``.
"""
import os

import pytest

torch = pytest.importorskip("torch")  # module import pulls torch transitively

from mstar.model.qwen3_omni.submodules import ThinkerSubmodule
from mstar.worker import micro_scheduler
from mstar.worker.micro_scheduler import MicroScheduler, ReadyNodeEntry
from mstar.worker.worker import Worker

_ENV = "MSTAR_MIXED_CHUNK_SIZES"


@pytest.fixture
def clean_grid():
    saved = os.environ.get(_ENV)
    os.environ.pop(_ENV, None)
    micro_scheduler._mixed_chunk_sizes_cache = None
    yield
    if saved is None:
        os.environ.pop(_ENV, None)
    else:
        os.environ[_ENV] = saved
    micro_scheduler._mixed_chunk_sizes_cache = None


def _set_grid(raw: str | None):
    if raw is None:
        os.environ.pop(_ENV, None)
    else:
        os.environ[_ENV] = raw
    micro_scheduler._mixed_chunk_sizes_cache = None  # boot-time: force re-resolve


class _DummySubmodule(ThinkerSubmodule):
    """Only the env-parsing property under test; skip the real __init__."""

    def __init__(self):
        pass


# --- ThinkerSubmodule.MIXED_BATCH_CHUNK_SIZES -------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, [256, 288, 512]),                        # unset -> byte-identical default
        ("", [256, 288, 512]),                           # empty -> default
        ("256,288,512,1024,2048", [256, 288, 512, 1024, 2048]),
        ("128", [128, 256, 288, 512]),                    # grow-only: union, never shrinks ceiling
        ("bogus,1024", [256, 288, 512]),                  # malformed -> falls back to default
        ("0,-5,1024", [256, 288, 512, 1024]),             # non-positive values filtered
        ("  256 , 512 ,1536 ", [256, 288, 512, 1536]),    # whitespace tolerant
    ],
)
def test_submodule_grid_parsing(clean_grid, raw, expected):
    if raw is None:
        os.environ.pop(_ENV, None)
    else:
        os.environ[_ENV] = raw
    assert _DummySubmodule().MIXED_BATCH_CHUNK_SIZES == expected


def test_submodule_default_is_exact_prior_list(clean_grid):
    # No env at all -> the literal list that shipped before this change.
    os.environ.pop(_ENV, None)
    assert _DummySubmodule().MIXED_BATCH_CHUNK_SIZES == [256, 288, 512]


# --- MicroScheduler._max_chunk_tokens() mirrors the submodule --------------

@pytest.mark.parametrize(
    "raw",
    [None, "", "256,288,512,1024,2048", "128", "bogus,1024", "0,-5,1024"],
)
def test_scheduler_grid_agrees_with_submodule(clean_grid, raw):
    _set_grid(raw)
    submodule_grid = _DummySubmodule().MIXED_BATCH_CHUNK_SIZES
    assert MicroScheduler._max_chunk_tokens() == max(submodule_grid)


def test_scheduler_max_chunk_tokens_default(clean_grid):
    _set_grid(None)
    assert MicroScheduler._max_chunk_tokens() == 512


def test_scheduler_max_chunk_tokens_grows(clean_grid):
    _set_grid("256,288,512,1024,2048")
    assert MicroScheduler._max_chunk_tokens() == 2048


# --- G1 gate (_chunk_entry_passes_gates) tracks the grid --------------------

class _FakeFwdInfo:
    def __init__(self, chunk_len):
        self.step_metadata = (
            {"prefill_chunk_len": chunk_len} if chunk_len is not None else {}
        )
        self.sampling_config = {}


class _FakeWGM:
    def __init__(self, fwd_by_rid):
        self._fwd_by_rid = fwd_by_rid

    def get_fwd_info(self, request_id, node_partition):
        return self._fwd_by_rid[request_id]


def _entry(rid, walk="prefill_text"):
    return ReadyNodeEntry(rid, "wg0", walk)


def test_gate_rejects_1024_chunk_by_default(clean_grid):
    _set_grid(None)
    sched = MicroScheduler(engine_manager=object())
    wgm = _FakeWGM({"c0": _FakeFwdInfo(1024)})
    assert not sched._chunk_entry_passes_gates(wgm, "Thinker", None, _entry("c0"))


def test_gate_admits_1024_chunk_once_grid_grows(clean_grid):
    _set_grid("256,288,512,1024,2048")
    sched = MicroScheduler(engine_manager=object())
    wgm = _FakeWGM({"c0": _FakeFwdInfo(1024)})
    assert sched._chunk_entry_passes_gates(wgm, "Thinker", None, _entry("c0"))


def test_gate_still_rejects_beyond_grown_ceiling(clean_grid):
    # Grid grown to 2048, but a 4096-token chunk is still bigger than the max
    # captured bucket -> still rejected (the UNCAP-IMA wall just moved, it
    # didn't disappear).
    _set_grid("256,288,512,1024,2048")
    sched = MicroScheduler(engine_manager=object())
    wgm = _FakeWGM({"c0": _FakeFwdInfo(4096)})
    assert not sched._chunk_entry_passes_gates(wgm, "Thinker", None, _entry("c0"))


def test_gate_rejects_unchunked_prefill_regardless_of_grid(clean_grid):
    # No prefill_chunk_len metadata at all -> always rejected (G6), independent
    # of the chunk-size grid.
    _set_grid("256,288,512,1024,2048")
    sched = MicroScheduler(engine_manager=object())
    wgm = _FakeWGM({"c0": _FakeFwdInfo(None)})
    assert not sched._chunk_entry_passes_gates(wgm, "Thinker", None, _entry("c0"))


# --- MSTAR_COADMIT budget clamp auto-widens ---------------------------------

def test_coadmit_clamp_default_is_544(clean_grid):
    _set_grid(None)
    os.environ.pop("MSTAR_COADMIT_BUDGET_TOKENS", None)
    # _compute_coadmit_budget doesn't touch `self` in its body — safe to call
    # unbound, same as the manual verification used while developing this.
    assert Worker._compute_coadmit_budget(None) == 32 + 512


def test_coadmit_clamp_widens_with_grid(clean_grid):
    _set_grid("256,288,512,1024,2048")
    os.environ.pop("MSTAR_COADMIT_BUDGET_TOKENS", None)
    assert Worker._compute_coadmit_budget(None) == 32 + 2048


def test_coadmit_explicit_want_below_clamp_is_respected(clean_grid):
    _set_grid("256,288,512,1024,2048")
    os.environ["MSTAR_COADMIT_BUDGET_TOKENS"] = "100"
    try:
        assert Worker._compute_coadmit_budget(None) == 100
    finally:
        os.environ.pop("MSTAR_COADMIT_BUDGET_TOKENS", None)
