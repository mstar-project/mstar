"""W5-P3-lite: VISION prefill chunk riding a captured ``thinker_mixed`` step.

CPU-only, no GPU, no model weights. Two surfaces are covered:

  * ``mixed_batch_vision_enabled()`` gating — the flag is a strict extension of
    MSTAR_MIXED_BATCH and only fires with MSTAR_MIXED_BATCH +
    MSTAR_CHUNKED_PREFILL_V2_VISION also on.
  * ``MicroScheduler._try_assemble_mixed`` chunk-walk selection — a
    ``prefill_vision`` chunk row is admitted as the mixed step's single chunk
    row ONLY when the vision flag is on; otherwise the chunk row stays
    ``prefill_text`` (P2 behavior), and a lone vision chunk never assembles.

The scheduler is driven with minimal fakes for the WorkerGraphsManager surface
``_try_assemble_mixed`` touches (partition lookup, per-request fwd_info, and a
queue that pops ready nodes). No engine, no cache, no CUDA.
"""
import os

import pytest

torch = pytest.importorskip("torch")  # module import pulls torch transitively

from mstar.model.qwen3_omni import qwen3_omni_model as _qom
from mstar.model.qwen3_omni.qwen3_omni_model import (
    mark_mixed_vision_provisioned,
    mixed_batch_spec_enabled,
    mixed_batch_vision_enabled,
    mixed_vision_capture_provisioned,
)
from mstar.worker.micro_scheduler import MicroScheduler, ReadyNodeEntry

# --- flag gating -------------------------------------------------------------
_FLAG_ENV = (
    "MSTAR_MIXED_BATCH_VISION",
    "MSTAR_MIXED_BATCH",
    "MSTAR_MIXED_SPEC",
    "MSTAR_CHUNKED_PREFILL_V2_VISION",
    # V2 budgeted admission knobs.
    "MSTAR_MIXED_BUDGET_TOKENS",
    "MSTAR_MIXED_BUDGET_MIN_DECODE",
    "MSTAR_MIXED_MIN_DECODE",
    "MSTAR_MIXED_SINGLE_CHUNK",
)


@pytest.fixture
def clean_flags():
    saved = {k: os.environ.get(k) for k in _FLAG_ENV}
    for k in _FLAG_ENV:
        os.environ.pop(k, None)
    # The capture-provisioning latch is a process global; reset it to the
    # pre-capture state so a test that marks it can't leak into the next (which
    # would then read a stale record instead of the None -> live-flag fallback).
    saved_prov = _qom._MIXED_VISION_PROVISIONED
    _qom._MIXED_VISION_PROVISIONED = None
    yield
    _qom._MIXED_VISION_PROVISIONED = saved_prov
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_vision_flag_requires_all_three(clean_flags):
    # None set -> off.
    assert not mixed_batch_vision_enabled()
    # Each subset that misses a prerequisite stays off.
    os.environ["MSTAR_MIXED_BATCH_VISION"] = "1"
    assert not mixed_batch_vision_enabled()
    os.environ["MSTAR_MIXED_BATCH"] = "1"
    assert not mixed_batch_vision_enabled()  # still missing V2_VISION
    del os.environ["MSTAR_MIXED_BATCH"]
    os.environ["MSTAR_CHUNKED_PREFILL_V2_VISION"] = "1"
    assert not mixed_batch_vision_enabled()  # missing MIXED_BATCH
    # All three -> on.
    os.environ["MSTAR_MIXED_BATCH"] = "1"
    assert mixed_batch_vision_enabled()


# --- scheduler chunk-walk selection -----------------------------------------
class _FakeNode:
    def __init__(self, name):
        self.name = name


class _FakeQueue:
    def __init__(self, node):
        self._node = node

    def pop_ready_nodes(self, request_id, node_names):
        return [_FakeNode(node_names[0])]


class _FakeFwdInfo:
    def __init__(self, chunk_len):
        # A chunked prefill row carries prefill_chunk_len; unchunked = absent.
        self.step_metadata = (
            {"prefill_chunk_len": chunk_len} if chunk_len is not None else {}
        )
        # No repetition penalty -> passes the sampler gate.
        self.sampling_config = {}


class _FakeWGM:
    """Minimal WorkerGraphsManager surface for ``_try_assemble_mixed``."""

    def __init__(self, node_name, fwd_by_rid):
        self._node_name = node_name
        self._fwd_by_rid = fwd_by_rid
        # One shared queue keyed by worker_graph_id; every rid uses "wg0".
        self.queues = {"wg0": _FakeQueue(node_name)}

    def get_partition_for_node(self, node_name):
        return "p0"

    def get_fwd_info(self, request_id, node_partition):
        return self._fwd_by_rid[request_id]


def _scheduler():
    # engine_manager is unused by _try_assemble_mixed; pass a sentinel.
    return MicroScheduler(engine_manager=object())


def _entries_and_wgm(chunk_walk):
    """Two decode rows + one chunk row (walk = chunk_walk, C=256)."""
    node = "Thinker"
    fwd = {
        "d0": _FakeFwdInfo(None),
        "d1": _FakeFwdInfo(None),
        "c0": _FakeFwdInfo(256),
    }
    entries = {
        node: [
            ReadyNodeEntry("d0", "wg0", "thinker_decode"),
            ReadyNodeEntry("d1", "wg0", "thinker_decode"),
            ReadyNodeEntry("c0", "wg0", chunk_walk),
        ]
    }
    return entries, _FakeWGM(node, fwd)


def _set(**flags):
    for k in _FLAG_ENV:
        os.environ.pop(k, None)
    for k, v in flags.items():
        os.environ[k] = v


def test_vision_chunk_admitted_only_under_flag(clean_flags):
    sched = _scheduler()

    # Vision flag OFF (only MIXED_BATCH): a prefill_vision chunk is NOT a valid
    # mixed chunk row -> no mixed batch assembles.
    _set(MSTAR_MIXED_BATCH="1")
    entries, wgm = _entries_and_wgm("prefill_vision")
    assert sched._try_assemble_mixed(wgm, entries, max_batch_size=32) is None

    # Vision flag ON: the same prefill_vision chunk now assembles a mixed batch.
    _set(
        MSTAR_MIXED_BATCH="1",
        MSTAR_MIXED_BATCH_VISION="1",
        MSTAR_CHUNKED_PREFILL_V2_VISION="1",
    )
    entries, wgm = _entries_and_wgm("prefill_vision")
    batch = sched._try_assemble_mixed(wgm, entries, max_batch_size=32)
    assert batch is not None
    assert batch.graph_walk == "thinker_mixed"
    # 2 decode rows + 1 vision chunk row popped.
    assert set(batch.node_objects.keys()) == {"d0", "d1", "c0"}


def test_text_chunk_always_admitted(clean_flags):
    # A prefill_text chunk mixes under MIXED_BATCH regardless of the vision flag
    # (P2 behavior preserved).
    sched = _scheduler()
    _set(MSTAR_MIXED_BATCH="1")
    entries, wgm = _entries_and_wgm("prefill_text")
    batch = sched._try_assemble_mixed(wgm, entries, max_batch_size=32)
    assert batch is not None
    assert batch.graph_walk == "thinker_mixed"
    assert set(batch.node_objects.keys()) == {"d0", "d1", "c0"}


# --- has_mixed_opportunity peek ---------------------------------------------
# The peek is what the worker uses to decide whether to break a decode spec
# chain into the non-speculative path (so get_next_batch can assemble the mixed
# batch). During a spec chain the decode rids are absent from the ready queue,
# so the peek only confirms a mixable CHUNK is ready on the decode's node.
class _FakeEngine:
    def engine_type(self):
        return None

    def check_ready(self, node_name, request_id, fwd_info):
        return True


class _FakeEngineManager:
    def get_engine(self, node_name):
        return _FakeEngine()


class _PeekWGM:
    """WorkerGraphsManager surface for has_mixed_opportunity: a ready scan
    (get_ready_node_names), per-rid walk + fwd_info, and per_request_info."""

    def __init__(self, node_name, ready_by_rid, walk_by_rid, fwd_by_rid):
        self._node_name = node_name
        self._walk_by_rid = walk_by_rid
        self._fwd_by_rid = fwd_by_rid
        self.per_request_info = {rid: object() for rid in ready_by_rid}
        self.queues = {"wg0": _PeekQueue(ready_by_rid)}

    def get_partition_for_node(self, node_name):
        return "p0"

    def get_graph_walk(self, request_id, node_partition):
        return self._walk_by_rid[request_id]

    def get_fwd_info(self, request_id, node_partition):
        return self._fwd_by_rid[request_id]


class _PeekQueue:
    def __init__(self, ready_by_rid):
        # rid -> set of ready node names
        self._ready = ready_by_rid
        self.popped = []  # (rid, node_name) actually popped

    def get_ready_node_names(self):
        return {rid: set(names) for rid, names in self._ready.items()}

    def pop_ready_nodes(self, request_id, node_names):
        # Mirror WorkerGraphQueues.pop_ready_nodes: discard the ready mark and
        # return a node object. Returns [] if the rid has no such ready name
        # (simulates a raced removal).
        out = []
        ready = self._ready.get(request_id, set())
        for name in node_names:
            if name in ready:
                ready.discard(name)
                self.popped.append((request_id, name))
                out.append(_FakeNode(name))
        return out


def _peek_scheduler():
    sched = MicroScheduler(engine_manager=_FakeEngineManager())
    # Only rank-0 nodes can initiate; the peek honors the same gate.
    sched.tp_rank_zero_nodes = {"Thinker"}
    return sched


def test_has_mixed_opportunity_true_when_chunk_ready(clean_flags):
    # Decode rids are mid-chain (absent from the ready scan); a chunk row on the
    # decode node IS ready. The peek should report an opportunity so the worker
    # breaks the chain into the non-spec mixed path.
    _set(MSTAR_MIXED_BATCH="1")
    sched = _peek_scheduler()
    wgm = _PeekWGM(
        "Thinker",
        ready_by_rid={"c0": {"Thinker"}},          # only the chunk is ready
        walk_by_rid={"c0": "prefill_text"},
        fwd_by_rid={"c0": _FakeFwdInfo(256)},
    )
    assert sched.has_mixed_opportunity(wgm, ("Thinker", "thinker_decode"))


def test_has_mixed_opportunity_false_flag_off(clean_flags):
    # Flag off -> peek always False so the default yield-away path is unchanged.
    _set()  # clears all flags
    sched = _peek_scheduler()
    wgm = _PeekWGM(
        "Thinker",
        ready_by_rid={"c0": {"Thinker"}},
        walk_by_rid={"c0": "prefill_text"},
        fwd_by_rid={"c0": _FakeFwdInfo(256)},
    )
    assert not sched.has_mixed_opportunity(wgm, ("Thinker", "thinker_decode"))


def test_has_mixed_opportunity_false_unchunked_prefill(clean_flags):
    # A full unchunked prefill (no prefill_chunk_len) is not mixable -> no
    # opportunity; the worker keeps the normal yield-away.
    _set(MSTAR_MIXED_BATCH="1")
    sched = _peek_scheduler()
    wgm = _PeekWGM(
        "Thinker",
        ready_by_rid={"c0": {"Thinker"}},
        walk_by_rid={"c0": "prefill_text"},
        fwd_by_rid={"c0": _FakeFwdInfo(None)},      # unchunked
    )
    assert not sched.has_mixed_opportunity(wgm, ("Thinker", "thinker_decode"))


# --- MSTAR_MIXED_SPEC flag gating -------------------------------------------
def test_mixed_spec_flag_requires_mixed_batch(clean_flags):
    # MSTAR_MIXED_SPEC implies MSTAR_MIXED_BATCH: the spec-chain fold has nothing
    # to fold in without mixed steps, so it stays off unless BOTH are set.
    _set()
    assert not mixed_batch_spec_enabled()
    _set(MSTAR_MIXED_SPEC="1")            # spec on, mixed off -> still off
    assert not mixed_batch_spec_enabled()
    _set(MSTAR_MIXED_BATCH="1")           # mixed on, spec off -> still off
    assert not mixed_batch_spec_enabled()
    _set(MSTAR_MIXED_BATCH="1", MSTAR_MIXED_SPEC="1")  # both -> on
    assert mixed_batch_spec_enabled()


# --- pop_mixed_chunk_for_spec (mid-chain chunk pop) -------------------------
# The worker calls this DURING a live decode spec chain to obtain the single
# chunk row to fold into the next speculative (thinker_mixed) batch. Only the
# chunk is popped; the decode rids are mid-chain and continue via speculation.
def test_pop_mixed_chunk_for_spec_pops_only_chunk(clean_flags):
    _set(MSTAR_MIXED_BATCH="1", MSTAR_MIXED_SPEC="1")
    sched = _peek_scheduler()
    wgm = _PeekWGM(
        "Thinker",
        ready_by_rid={"c0": {"Thinker"}},          # only the chunk is ready
        walk_by_rid={"c0": "prefill_text"},
        fwd_by_rid={"c0": _FakeFwdInfo(256)},
    )
    got = sched.pop_mixed_chunk_for_spec(wgm, ("Thinker", "thinker_decode"))
    assert got is not None
    node, rid, wg_id, chunk_len = got
    assert rid == "c0"
    assert wg_id == "wg0"
    assert chunk_len == 256
    assert node.name == "Thinker"
    # The chunk node was actually removed from the ready queue.
    assert wgm.queues["wg0"].popped == [("c0", "Thinker")]
    assert "Thinker" not in wgm.queues["wg0"].get_ready_node_names()["c0"]


def test_pop_mixed_chunk_for_spec_none_when_spec_flag_off(clean_flags):
    # MSTAR_MIXED_BATCH on but MSTAR_MIXED_SPEC off: the mid-chain pop is
    # unreachable (0cc7c71 break-the-chain behavior stays intact). Nothing pops.
    _set(MSTAR_MIXED_BATCH="1")
    sched = _peek_scheduler()
    wgm = _PeekWGM(
        "Thinker",
        ready_by_rid={"c0": {"Thinker"}},
        walk_by_rid={"c0": "prefill_text"},
        fwd_by_rid={"c0": _FakeFwdInfo(256)},
    )
    assert sched.pop_mixed_chunk_for_spec(wgm, ("Thinker", "thinker_decode")) is None
    assert wgm.queues["wg0"].popped == []          # queue untouched


def test_pop_mixed_chunk_for_spec_none_on_unchunked(clean_flags):
    # A full unchunked prefill fails the chunk gates -> no chunk to fold, queue
    # untouched so the normal path still schedules the prefill.
    _set(MSTAR_MIXED_BATCH="1", MSTAR_MIXED_SPEC="1")
    sched = _peek_scheduler()
    wgm = _PeekWGM(
        "Thinker",
        ready_by_rid={"c0": {"Thinker"}},
        walk_by_rid={"c0": "prefill_text"},
        fwd_by_rid={"c0": _FakeFwdInfo(None)},      # unchunked
    )
    assert sched.pop_mixed_chunk_for_spec(wgm, ("Thinker", "thinker_decode")) is None
    assert wgm.queues["wg0"].popped == []


# --- V2 budgeted admission (MSTAR_MIXED_BUDGET_TOKENS) -----------------------
# The budget is a per-step token cap on the mixed step (n_decode 1-token rows +
# the C-token chunk). It gates only chunks that ALREADY exist (unchunked spans
# are still excluded by the mixable gate); it never routes short prefills
# through the chunk path (that was the closed MSTAR_MIXED_SINGLE_CHUNK). The
# peek and the pop must apply the SAME cap so they fold the same chunk.
from mstar.model.qwen3_omni.qwen3_omni_model import mixed_budget_tokens


def test_mixed_budget_tokens_parse(clean_flags):
    _set()
    assert mixed_budget_tokens() == 0            # unset -> off
    _set(MSTAR_MIXED_BUDGET_TOKENS="512")
    assert mixed_budget_tokens() == 512
    _set(MSTAR_MIXED_BUDGET_TOKENS="0")
    assert mixed_budget_tokens() == 0            # explicit 0 -> off
    _set(MSTAR_MIXED_BUDGET_TOKENS="-5")
    assert mixed_budget_tokens() == 0            # negative -> off
    _set(MSTAR_MIXED_BUDGET_TOKENS="junk")
    assert mixed_budget_tokens() == 0            # unparseable -> off


def _budget_wgm(chunk_len):
    return _PeekWGM(
        "Thinker",
        ready_by_rid={"c0": {"Thinker"}},
        walk_by_rid={"c0": "prefill_text"},
        fwd_by_rid={"c0": _FakeFwdInfo(chunk_len)},
    )


def test_budget_peek_fits(clean_flags):
    # n_decode + C <= budget -> opportunity. Floor disabled so only the budget
    # cap is under test.
    _set(
        MSTAR_MIXED_BATCH="1", MSTAR_MIXED_SPEC="1",
        MSTAR_MIXED_BUDGET_TOKENS="512", MSTAR_MIXED_BUDGET_MIN_DECODE="0",
    )
    sched = _peek_scheduler()
    assert sched.has_mixed_opportunity(
        _budget_wgm(256), ("Thinker", "thinker_decode"),
        n_decode=8, budget_tokens=512,
    )  # 8 + 256 = 264 <= 512


def test_budget_peek_over_budget(clean_flags):
    # n_decode + C > budget -> no opportunity (the chunk is too big for this
    # decode side this step; a later, smaller decode side folds it).
    _set(
        MSTAR_MIXED_BATCH="1", MSTAR_MIXED_SPEC="1",
        MSTAR_MIXED_BUDGET_TOKENS="200", MSTAR_MIXED_BUDGET_MIN_DECODE="0",
    )
    sched = _peek_scheduler()
    assert not sched.has_mixed_opportunity(
        _budget_wgm(256), ("Thinker", "thinker_decode"),
        n_decode=8, budget_tokens=200,
    )  # 8 + 256 = 264 > 200


def test_budget_off_ignores_cap(clean_flags):
    # budget_tokens=0 (off) -> the cap never binds; a huge chunk still folds
    # (P2 / yield-boundary behavior unchanged).
    _set(MSTAR_MIXED_BATCH="1", MSTAR_MIXED_SPEC="1")
    sched = _peek_scheduler()
    assert sched.has_mixed_opportunity(
        _budget_wgm(512), ("Thinker", "thinker_decode"),
        n_decode=8, budget_tokens=0,
    )


def test_budget_pop_mirrors_peek(clean_flags):
    # The pop applies the SAME budget cap as the peek: over budget -> None and
    # the queue is untouched; within budget -> the chunk pops.
    _set(
        MSTAR_MIXED_BATCH="1", MSTAR_MIXED_SPEC="1",
        MSTAR_MIXED_BUDGET_TOKENS="200", MSTAR_MIXED_BUDGET_MIN_DECODE="0",
    )
    sched = _peek_scheduler()
    wgm = _budget_wgm(256)
    assert sched.pop_mixed_chunk_for_spec(
        wgm, ("Thinker", "thinker_decode"), n_decode=8, budget_tokens=200,
    ) is None                                    # 264 > 200
    assert wgm.queues["wg0"].popped == []        # queue untouched

    wgm2 = _budget_wgm(256)
    got = sched.pop_mixed_chunk_for_spec(
        wgm2, ("Thinker", "thinker_decode"), n_decode=8, budget_tokens=512,
    )
    assert got is not None and got[3] == 256     # 264 <= 512 -> pops


def test_budget_enables_occupancy_floor(clean_flags):
    # The budget policy inherits the single-chunk occupancy floor (default 24):
    # a small decode side is skipped even when a chunk is ready, so ramp-up
    # admits on the standalone path instead of throttling on fold slots.
    _set(MSTAR_MIXED_BATCH="1", MSTAR_MIXED_SPEC="1",
         MSTAR_MIXED_BUDGET_TOKENS="4096")
    sched = _peek_scheduler()
    assert not sched.has_mixed_opportunity(
        _budget_wgm(256), ("Thinker", "thinker_decode"),
        n_decode=8, budget_tokens=4096,
    )  # 8 < 24 floor
    sched2 = _peek_scheduler()
    assert sched2.has_mixed_opportunity(
        _budget_wgm(256), ("Thinker", "thinker_decode"),
        n_decode=24, budget_tokens=4096,
    )  # 24 >= floor, 24 + 256 <= 4096


def test_min_decode_override_precedence(clean_flags):
    # BUDGET_MIN_DECODE wins over MIXED_MIN_DECODE wins over the eager default.
    _set(MSTAR_MIXED_BATCH="1", MSTAR_MIXED_BUDGET_TOKENS="512",
         MSTAR_MIXED_MIN_DECODE="10", MSTAR_MIXED_BUDGET_MIN_DECODE="5")
    assert _peek_scheduler()._mixed_min_decode() == 5
    _set(MSTAR_MIXED_BATCH="1", MSTAR_MIXED_BUDGET_TOKENS="512",
         MSTAR_MIXED_MIN_DECODE="10")
    assert _peek_scheduler()._mixed_min_decode() == 10
    _set(MSTAR_MIXED_BATCH="1", MSTAR_MIXED_BUDGET_TOKENS="512")
    assert _peek_scheduler()._mixed_min_decode() == 24   # eager default
    _set(MSTAR_MIXED_BATCH="1")                          # no eager policy
    assert _peek_scheduler()._mixed_min_decode() == 0


# --- IMA safety: capture-provisioning latch ---------------------------------
# The routing gate (_mixed_chunk_walks) must key off what the thinker_mixed
# capture ACTUALLY provisioned (deepstack statics), not the live env flag. This
# blocks the UNCAP-IMA hazard: MSTAR_DYNFLAGS can flip MSTAR_MIXED_BATCH_VISION on
# AFTER a vision-off boot, and routing a prefill_vision chunk into the text-only
# capture (no deepstack static buffer to copy into) would IMA / corrupt output.
def test_provisioned_query_falls_back_to_flag_before_capture(clean_flags):
    # Pre-capture (record None): the query reflects the live flag intent so CPU
    # tests and boot ordering behave as written.
    assert _qom._MIXED_VISION_PROVISIONED is None
    _set(
        MSTAR_MIXED_BATCH="1",
        MSTAR_MIXED_BATCH_VISION="1",
        MSTAR_CHUNKED_PREFILL_V2_VISION="1",
    )
    assert mixed_vision_capture_provisioned()
    _set(MSTAR_MIXED_BATCH="1")  # flag off -> fallback off
    assert not mixed_vision_capture_provisioned()


def test_provisioned_record_is_authoritative_after_capture(clean_flags):
    # After capture recorded a value, the live flag no longer changes the query.
    mark_mixed_vision_provisioned(True)
    _set(MSTAR_MIXED_BATCH="1")  # flag OFF but capture DID provision
    assert mixed_vision_capture_provisioned()
    mark_mixed_vision_provisioned(False)
    _set(
        MSTAR_MIXED_BATCH="1",
        MSTAR_MIXED_BATCH_VISION="1",
        MSTAR_CHUNKED_PREFILL_V2_VISION="1",
    )  # flag ON but capture did NOT provision
    assert not mixed_vision_capture_provisioned()


def test_runtime_on_flip_after_vision_off_boot_is_ima_safe(clean_flags):
    # Boot with vision OFF -> capture records False. A later dynflag ON-flip must
    # NOT make the scheduler route a prefill_vision chunk into the unprovisioned
    # (text-signature) thinker_mixed graph.
    sched = _scheduler()
    mark_mixed_vision_provisioned(False)  # booted vision-off
    _set(
        MSTAR_MIXED_BATCH="1",
        MSTAR_MIXED_BATCH_VISION="1",       # flipped on at runtime
        MSTAR_CHUNKED_PREFILL_V2_VISION="1",
    )
    entries, wgm = _entries_and_wgm("prefill_vision")
    # A lone vision chunk is NOT admitted -> no mixed batch assembles.
    assert sched._try_assemble_mixed(wgm, entries, max_batch_size=32) is None
    # A text chunk still folds (the mixed capture itself is fine).
    entries, wgm = _entries_and_wgm("prefill_text")
    assert sched._try_assemble_mixed(wgm, entries, max_batch_size=32) is not None


def test_provisioned_boot_routes_vision_even_if_flag_flipped_off(clean_flags):
    # Boot vision-ON -> capture records True. Routing stays IMA-safe (the graph
    # HAS deepstack), and honors a safe-direction runtime OFF-flip.
    sched = _scheduler()
    mark_mixed_vision_provisioned(True)
    _set(
        MSTAR_MIXED_BATCH="1",
        MSTAR_MIXED_BATCH_VISION="1",
        MSTAR_CHUNKED_PREFILL_V2_VISION="1",
    )
    entries, wgm = _entries_and_wgm("prefill_vision")
    assert sched._try_assemble_mixed(wgm, entries, max_batch_size=32) is not None
    # Safe-direction OFF-flip at runtime: stop routing vision (graph could still
    # replay it, but the A/B asked for off).
    os.environ["MSTAR_MIXED_BATCH_VISION"] = "0"
    entries, wgm = _entries_and_wgm("prefill_vision")
    assert sched._try_assemble_mixed(wgm, entries, max_batch_size=32) is None
