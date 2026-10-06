"""A session's second request continues where the first one stopped.

This is the behaviour everything else is in service of, driven through the real
pieces: the model's ``SessionsConfig``, the engine's marking of which resources
hold session state, and one ``StepRunner`` over a real KV cache and the position
counters above it. The assertions are the position ids the plan actually
produces and the pages the stream actually holds, because those are what a wrong
answer would come from — a resumed turn placed at 0 writes over the context it
was supposed to continue, and nothing raises when it does.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.engine.resources import (
    PositionConfig,
    StepContext,
    StepRunner,
)
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.config import PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.model.sessions import (
    SessionResourceConfig,
    SessionsConfig,
)

KV = "kv"
ROPE = "rope"
NODE = "LLM"
WALK = "prefill"
SESSION = "s"
PAGE_SIZE = 16


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del kwargs
        return transfer_info == self.get_kv_transfer_info()

    def remove_request(self, request_id):
        del request_id

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


class _Node:
    """One node's KV and positions, marked up by the engine from a model's
    ``SessionsConfig`` exactly as a worker would at load."""

    def __init__(self, max_state: int | None = None):
        device = torch.device("cpu")
        self.kv = KVManager(
            cfg=PagedKVConfig(
                num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
                max_num_pages=64, page_size=PAGE_SIZE,
            ),
            name=KV, joint_comm_group=None, transfer_engine_info=None,
            device=device, dtype=torch.float32,
        )
        self.rope = RopeManager(
            config=PositionConfig(kv_cache=KV), device=device,
            dtype=torch.float32,
        )
        self.runner = StepRunner(
            {KV: self.kv, ROPE: self.rope},
            node_resources={NODE: [KV, ROPE]},
        )
        # the model names the cache only; the engine works out that the
        # counters over it have to come along
        config = SessionsConfig(resources={KV: SessionResourceConfig(
            max_state=max_state,
        )})
        engine = Engine.__new__(Engine)
        engine._resources = {KV: self.kv, ROPE: self.rope}
        engine._runner = self.runner
        engine._open_session_state({KV: object(), ROPE: object()}, config)
        self.free_at_start = self.kv._arena.num_free

    # -- lifecycle, as the worker drives it ------------------------------

    def start(self, rid: str, session: str | None = SESSION) -> None:
        self.runner.ingest_request(rid, session_id=session)

    def finish(self, rid: str, session: str | None = SESSION) -> None:
        self.runner.remove_request(rid, session_id=session)

    def end_session(self, session: str = SESSION) -> None:
        self.runner.remove_session(session)

    # -- one step ---------------------------------------------------------

    def step(self, rid: str, span: int, label: str = "main") -> list[int]:
        """Admit, plan and commit one step; give back its position ids."""
        step = SubmoduleStep(
            steps={KV: KVStep(), ROPE: PositionStep()},
            segments=[Segment(rid, label, span)],
        )
        ctx = StepContext(
            request_ids=(rid,), graph_walk=WALK, slot=0, capture=False,
        )
        step.set_ctx(ctx)
        assert self.runner.admit(step).outcome.ok
        pos_ids = self.runner.plan(step)[ROPE][label].tolist()
        self.runner.commit(step)
        return pos_ids

    def stream(self, rid: str, label: str = "main"):
        return self.kv._streams[rid][label]


# ── the engine's marking ────────────────────────────────────────────────────

def test_naming_the_cache_carries_the_counters_over_it_too():
    node = _Node()

    assert node.kv.session_config is not None
    assert node.rope.session_config is not None
    assert node.runner.session_resource_keys() == [KV, ROPE]


# ── continuation ────────────────────────────────────────────────────────────

def test_a_resumed_turn_is_placed_after_everything_the_session_holds():
    node = _Node()

    node.start("r0")
    assert node.step("r0", 20) == list(range(20))
    node.finish("r0")

    node.start("r1")
    second = node.step("r1", 4)

    assert second == [20, 21, 22, 23], (
        "the resumed turn was placed on top of the context it should continue"
    )


def test_the_same_request_without_a_session_starts_over():
    node = _Node()
    node.start("r0", session=None)
    node.step("r0", 20)
    node.finish("r0", session=None)

    node.start("r1", session=None)

    assert node.step("r1", 4) == [0, 1, 2, 3]


def test_the_resumed_turn_attends_the_pages_the_first_one_wrote():
    node = _Node()
    node.start("r0")
    node.step("r0", 20)
    pages = list(node.stream("r0").page_indices)
    node.finish("r0")

    node.start("r1")
    node.step("r1", 4)

    resumed = node.stream("r1")
    assert resumed.page_indices[:len(pages)] == pages
    assert resumed.stored_len == 24, "the turn did not extend the session's span"
    node.kv.assert_pages_conserved()


def test_a_third_turn_keeps_accumulating():
    node = _Node()
    spans = [16, 8, 4]
    placed = []
    for i, span in enumerate(spans):
        rid = f"r{i}"
        node.start(rid)
        placed.append(node.step(rid, span))
        node.finish(rid)

    assert [ids[0] for ids in placed] == [0, 16, 24]
    assert node.kv.session_state_size(SESSION) == 2  # 28 tokens over 16-page


def test_a_turn_that_is_still_running_holds_the_state_itself():
    node = _Node()
    node.start("r0")
    node.step("r0", 20)

    # nothing is parked under the session while its request has the state
    assert node.kv.session_state_size(SESSION) == 0

    node.finish("r0")

    assert node.kv.session_state_size(SESSION) == 2


# ── the end of a session ────────────────────────────────────────────────────

def test_ending_the_session_gives_every_page_back():
    node = _Node()
    node.start("r0")
    node.step("r0", 100)
    node.finish("r0")
    assert node.kv._arena.num_free < node.free_at_start

    node.end_session()

    assert node.kv._arena.num_free == node.free_at_start
    node.kv.assert_pages_conserved()


def test_a_turn_after_the_session_ended_starts_from_nothing():
    node = _Node()
    node.start("r0")
    node.step("r0", 20)
    node.finish("r0")
    node.end_session()

    node.start("r1")

    assert node.step("r1", 4) == [0, 1, 2, 3]


# ── the budget ──────────────────────────────────────────────────────────────

def test_a_session_over_budget_is_cleared_and_the_next_turn_starts_over():
    node = _Node(max_state=2)
    node.start("r0")
    node.step("r0", 100)  # 7 pages, well past the 2 it may hold

    node.finish("r0")

    assert node.kv.session_state_size(SESSION) == 0
    assert node.kv._arena.num_free == node.free_at_start
    node.start("r1")
    assert node.step("r1", 4) == [0, 1, 2, 3]


def test_the_clear_takes_the_counters_with_the_pages():
    # the counters address the pages: clearing one and not the other would put
    # the next turn's tokens past a context that is no longer there
    node = _Node(max_state=2)
    node.start("r0")
    node.step("r0", 100)

    node.finish("r0")

    assert node.kv.session_state_size(SESSION) == 0
    assert node.rope.session_state_size(SESSION) == 0


def test_the_cleared_session_owes_the_next_turn_an_explanation():
    node = _Node(max_state=2)
    node.start("r0")
    node.step("r0", 100)

    node.finish("r0")

    error = node.runner.take_session_error(SESSION)
    assert error is not None and "budget" in error
    assert node.kv.session_state_size(SESSION) == 0


def test_a_session_inside_its_budget_is_left_alone():
    node = _Node(max_state=8)
    node.start("r0")
    node.step("r0", 100)

    node.finish("r0")

    assert node.kv.session_state_size(SESSION) == 7
    assert node.runner.take_session_error(SESSION) is None
    node.start("r1")
    assert node.step("r1", 1) == [100]
