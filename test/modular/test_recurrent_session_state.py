"""Recurrent state held across a session.

A hybrid model's turn leaves state in its recurrent slots (a GDN layer's state
matrix and conv window) as well as in its KV. A session parks the slots under a
reserved rid when its request ends and hands them to the next request, exactly
as the KV manager does with pages, so the resumed turn continues the recurrence
instead of starting from zeros.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.step import Segment, StepContext


def _pool(max_slots: int = 8) -> RecurrentStatePool:
    spec = RecurrentStateSpec(
        "gdn_state", {"llm"},
        RecurrentStateConfig(
            num_layers=2,
            blocks={
                "state": RecurrentBlockConfig(shape=(2, 4, 4), dtype=torch.float32),
                "conv": RecurrentBlockConfig(shape=(8, 3), dtype=torch.float32),
            },
            max_slots=max_slots,
        ),
    )
    return RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cpu")))


def _run(pool: RecurrentStatePool, rid: str, label: str = "main") -> int:
    """Admit, plan and commit one step for ``rid``, writing a marker into its
    slot as a backend would. Returns the slot."""
    step = RecurrentStep(segments=(Segment(rid, label, 1),))
    ctx = StepContext(request_ids=(rid,), graph_walk="decode", slot=0, capture=False)
    assert pool.admit(step, ctx).ok
    pool.plan(step, ctx)
    index = pool._slots[rid][label].index
    pool.block("state", 0)[index].fill_(float(index))
    pool.commit(step, ctx)
    return index


def _conserved(pool: RecurrentStatePool) -> None:
    """Every slot but the sink is either free or held, exactly once."""
    held = [s.index for labels in pool._slots.values() for s in labels.values()]
    everything = sorted(held + pool._free)
    assert everything == list(range(1, pool.config.max_slots)), everything


def test_retain_parks_the_slots_under_the_session():
    pool = _pool()
    pool.ingest_request("r0")
    index = _run(pool, "r0")

    pool.retain_session_state("r0", "s")

    assert "r0" not in pool._slots
    assert pool.session_state_size("s") == 1
    assert pool._slots[pool.session_rid("s")]["main"].index == index
    _conserved(pool)


def test_the_next_request_resumes_the_state_the_last_one_left():
    pool = _pool()
    pool.ingest_request("r0")
    index = _run(pool, "r0")
    pool.retain_session_state("r0", "s")

    pool.ingest_request("r1")
    pool.adopt_session_state("r1", "s")

    slot = pool._slots["r1"]["main"]
    # the same slot, its contents untouched, and still marked as holding state
    # so the backend resumes from it rather than from zeros
    assert slot.index == index
    assert slot.has_state is True
    assert torch.all(pool.block("state", 0)[index] == float(index))
    assert pool.session_state_size("s") == 0
    _conserved(pool)


def test_a_parked_slot_is_out_of_circulation():
    pool = _pool(max_slots=3)  # two usable slots
    pool.ingest_request("r0")
    _run(pool, "r0")
    pool.retain_session_state("r0", "s")

    assert pool.num_free_slots == 1


def test_adopting_nothing_leaves_the_request_alone():
    pool = _pool()
    pool.ingest_request("r0")

    pool.adopt_session_state("r0", "fresh")

    assert pool._slots["r0"] == {}
    _conserved(pool)


def test_a_slot_the_request_already_had_is_given_back_on_adoption():
    pool = _pool()
    pool.ingest_request("r0")
    _run(pool, "r0")
    pool.retain_session_state("r0", "s")
    pool.ingest_request("r1")
    stray = _run(pool, "r1")  # a slot of its own under the same label

    pool.adopt_session_state("r1", "s")

    assert pool._slots["r1"]["main"].index != stray
    assert stray in pool._free
    _conserved(pool)


def test_ending_the_session_zeroes_and_frees_its_slots():
    pool = _pool()
    pool.ingest_request("r0")
    index = _run(pool, "r0")
    pool.retain_session_state("r0", "s")

    pool.remove_session("s")

    assert pool.session_state_size("s") == 0
    assert index in pool._free
    # a later request handed this slot must not resume the session's state
    assert torch.all(pool.block("state", 0)[index] == 0)
    _conserved(pool)


def test_every_label_comes_along():
    pool = _pool()
    pool.ingest_request("r0")
    _run(pool, "r0", "main")
    _run(pool, "r0", "cfg")

    pool.retain_session_state("r0", "s")
    pool.ingest_request("r1")
    pool.adopt_session_state("r1", "s")

    assert sorted(pool._slots["r1"]) == ["cfg", "main"]
    _conserved(pool)
