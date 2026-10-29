"""Draft-verify over recurrent state, on CPU.

A verify step runs its whole block through a recurrent layer, which updates its state in
place; the verdict says only afterwards how much of the block to keep. With verify blocks the
pool's state stays a checkpoint before the last block's kept tokens, the next step replays
them, and the draft-verify resource settles how many from the verdict, on the device, inside
the forward (``RecurrentStatePool.settle_verify``). Through the engine, the toy with a KDA
layer must emit what its plain decode emits whatever the drafter gets right, and must not
without the settle. A node whose recurrent state can't be settled refuses to load. (GDN's verify
steps, which settle in the GDN resource, are test_gdn_verify.py's.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draft_verify_toy as toy  # noqa: E402

from mstar.engine.resources.base import RewindKind  # noqa: E402
from mstar.engine.resources.recurrent.config import (  # noqa: E402
    DeltaNetGeometry,
    RecurrentStateConfig,
)

CPU = torch.device("cpu")
PROMPTS = [[3, 9, 12, 40], [5, 11], [20, 21, 22, 23, 24, 25], [60, 2, 33]]
MAX_TOKENS = 16


@pytest.fixture(autouse=True)
def _threads():
    # the comparisons are exact; keep the CPU reductions' order fixed
    torch.set_num_threads(2)


def _run(k: int, corrupt: int = 0, batched: bool = True, patch=None):
    sub = toy.ToySubmodule(k=k, corrupt=corrupt, linear="kda")
    # float64 around the KDA layer, whose recurrence runs in fp32 on either path
    eng = toy.engine(sub, CPU, dtype=torch.float64)
    if patch is not None:
        patch(eng)
    drv = toy.Driver(eng, sub)
    for rid, prompt in enumerate(PROMPTS):
        drv.add(rid, prompt, MAX_TOKENS)
    return drv.run(list(range(len(PROMPTS))), batched=batched), eng


@pytest.fixture(scope="module")
def plain():
    torch.set_num_threads(2)
    out, _ = _run(0)
    # the KDA layer moves the stream off the attention-only toy's
    assert out != {rid: toy.reference(prompt, MAX_TOKENS) for rid, prompt in enumerate(PROMPTS)}
    return out


@pytest.mark.parametrize(("k", "corrupt"), [(1, 0), (3, 0), (3, 2), (2, 1)],
                         ids=["k1", "k3", "k3_mixed", "k2_reject_all"])
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "one_per_step"])
def test_verify_emits_what_plain_decode_emits(plain, k, corrupt, batched):
    out, eng = _run(k, corrupt, batched)
    assert out == plain
    dv, pool = eng._resources[toy.DV], eng._resources[toy.STATE]
    # the pool is what the engine settles; the KDA resource only plans
    assert dv._settlers == [pool]
    assert pool.rewind_kind is RewindKind.DEVICE
    assert eng._resources[toy.LIN].rewind_kind is RewindKind.NONE
    assert dv.rows_verified > 0
    if corrupt != 1:
        assert dv.drafts_accepted > 0
    if corrupt:
        assert dv.drafts_accepted < k * dv.rows_verified


def test_an_engine_loaded_without_its_model_binds_the_walk_before_its_first_verify(plain):
    """Without the model the engine asks a walk for its policy at the walk's first step; the
    verify in that step's forward already reads the sampler and the settles it binds."""
    sub = toy.ToySubmodule(k=3, corrupt=2, linear="kda")
    eng = toy.engine(sub, CPU, dtype=torch.float64, bare=True)
    dv = eng._resources[toy.DV]
    assert dv._sampler is None and dv._settlers == []
    drv = toy.Driver(eng, sub)
    for rid, prompt in enumerate(PROMPTS):
        drv.add(rid, prompt, MAX_TOKENS)
    assert drv.run(list(range(len(PROMPTS)))) == plain
    assert dv._settlers == [eng._resources[toy.STATE]]


def test_without_the_settle_the_streams_come_apart(plain):
    """The control: a pool that never learns how much each block kept replays the wrong
    prefix (all of the first block's tokens, then whatever the count says)."""
    def no_settle(eng):
        eng._resources[toy.STATE].settle_verify = lambda accepted, label="main": None

    out, eng = _run(3, corrupt=2, patch=no_settle)
    assert eng._resources[toy.DV].drafts_accepted > 0
    assert out != plain


def test_a_settle_with_an_off_by_one_count_comes_apart(plain):
    """And one that keeps one token too many of a block."""
    def off_by_one(eng):
        pool = eng._resources[toy.STATE]
        settle = pool.settle_verify
        pool.settle_verify = lambda accepted, label="main": settle(accepted + 1, label)

    out, _ = _run(3, corrupt=2, patch=off_by_one)
    assert out != plain


def test_a_pool_without_verify_blocks_keeps_the_node_from_loading():
    sub = toy.ToySubmodule(k=3, linear="kda")
    with pytest.raises(ValueError, match="declare no rewind_kind"):
        toy.engine(sub, CPU, verify_blocks=0)


def test_a_walk_drafting_past_the_pools_blocks_is_refused():
    sub = toy.ToySubmodule(k=3, linear="kda")
    with pytest.raises(ValueError, match="verify blocks for 2 drafts, but a walk drafts 3"):
        toy.engine(sub, CPU, verify_blocks=2)


def test_a_pool_with_verify_blocks_refuses_a_gdn_layer():
    """Verify blocks are KDA's replay; GDN settles in its own resource and keeps none."""
    from types import SimpleNamespace

    from mstar.engine.resources.linear_attn.base import LinearAttnManager
    from mstar.engine.resources.linear_attn.config import (
        LinearAttnConfig,
        LinearAttnSpec,
        LinearAttnVariant,
    )

    geometry = DeltaNetGeometry(2, 2, 16, 16, 4)
    pool = RecurrentStateConfig(num_layers=1, max_slots=4, blocks=geometry.to_blocks(max_drafts=2))
    spec = LinearAttnSpec(resource_key="lin", nodes={"llm"}, config=LinearAttnConfig(
        recurrent_state="state", variant=LinearAttnVariant.GDN))
    info = SimpleNamespace(dependency=lambda key: SimpleNamespace(config=pool), joint_comm_group=None)
    with pytest.raises(ValueError, match="which only KDA replays"):
        LinearAttnManager.build(spec, info)
