"""Each state op's fused kernel writes what its reference writes, on a real pool, on every format it lists (GPU only).

A fused kernel addresses the cache by raw page and slot, so a wrong stride or a tensor passed in the wrong place writes
another request's entries, or attends entries no step wrote, with no error. The reference goes through the contract's
own ``add``, so holding the kernel to it holds it to the engine's addressing: the mutated arguments and the step's slots
within tolerance, and every other byte of the array unchanged.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, ArrayStep, Attend, Causal, KVEntry, Read
from mstar.engine.state.calls import ArrayHandle
from mstar.engine.state.state_op import STATE_OPS, bind_state_ops, conformance_problems
from mstar.model.components import qwen3_lm
from mstar.model.qwen3_omni.components import attention as omni_attention
from mstar.model.qwen3_omni.components.rope import compute_3d_cos_sin, compute_rope_freqs

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernels are CUDA's")

HQ, HKV, D, LAYERS, PAGE = 8, 2, 128, 3, 16
EPS, THETA = 1e-6, 1e6
# bf16 rounding, which both the kernel and the reference carry: what the kernel's own tests allow
TOLERANCE = 2e-2


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", lambda *args, **kwargs: None, raising=False)


def _manager() -> KVManager:
    return KVManager(
        cfg=PagedKVConfig(
            num_layers=LAYERS, num_kv_heads=HKV, num_qo_heads=HQ, head_dim=D, max_seq_len=256, max_num_pages=64,
            page_size=PAGE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=torch.device("cuda"), dtype=torch.bfloat16,
    )


def _array() -> Array:
    return Array(
        "kv", nodes={"llm"}, layers=frozenset(range(LAYERS)), entry=KVEntry(heads=HKV, head_dim=D), max_len=256,
        reads={"attn": Read(mask=Causal(), op=Attend(q_heads=HQ))},
    )


def _opened_step(kv: KVManager, lengths: dict[int, int]) -> ArrayHandle:
    """A handle on real pages with a planned step over ``lengths`` tokens per request, and some history before it."""
    for rid in lengths:
        kv.ingest_request(rid, KVReqConfig())
    history = KVStep(segments=tuple(Segment(rid, "main", 5) for rid in lengths))
    ctx = StepContext(request_ids=tuple(lengths), graph_walk="prefill", slot=0, capture=False)
    assert kv.admit(history, ctx).ok
    kv.plan(history, ctx)
    kv.commit(history, ctx)
    kv.kv_cache.tensor.copy_(torch.randn_like(kv.kv_cache.tensor.float()).to(kv.kv_cache.tensor.dtype))
    segments = tuple(Segment(rid, "main", n) for rid, n in lengths.items())
    step = KVStep(segments=segments)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    # the read hands back its q, so conformance compares what an op that reads passes to its read
    attn = SimpleNamespace(requires_kv_write=True, run=lambda q, **kwargs: q.clone())
    handle = ArrayHandle(_array(), {"kv": kv, "attn": attn}, None)
    bind_state_ops({"kv": handle})
    contract_step = SubmoduleStep(segments=segments, state={"kv": ArrayStep()})
    contract_step.set_ctx(ctx)
    handle.begin_step(contract_step)
    return handle


def _thinker_args(handle: ArrayHandle, layer: int, tokens: int) -> tuple:
    gen = torch.Generator("cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=gen).to(torch.bfloat16)

    t = torch.arange(tokens, device="cuda", dtype=torch.float)
    grids = torch.stack([torch.full_like(t, 700.0), t % 5 + 700, t % 3 + 700])
    cos, sin = compute_3d_cos_sin(
        grids, compute_rope_freqs(D, THETA, device="cuda"), (24, 20, 20), target_dtype=torch.bfloat16,
    )
    return (
        handle, layer, randn(tokens, HQ, D), randn(tokens, HKV, D), randn(tokens, HKV, D), randn(D), randn(D), cos, sin,
        EPS, EPS,
    )


def _plain_rope_args(handle: ArrayHandle, layer: int, tokens: int) -> tuple:
    gen = torch.Generator("cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=gen).to(torch.bfloat16)

    # the positions run past the step's tokens, as a bucket's buffer does
    pos_ids = torch.arange(700, 700 + tokens + 3, device="cuda", dtype=torch.int32)
    return (
        handle, layer, randn(tokens, HQ, D), randn(tokens, HKV, D), randn(tokens, HKV, D), randn(D), randn(D), pos_ids,
        THETA, EPS, EPS,
    )


# How to call each op on a real handle.
ARGUMENTS = {
    omni_attention.qk_norm_rope_table_attend: _thinker_args,
    qwen3_lm.qk_norm_rope_kv_write: _plain_rope_args,
}


def test_every_op_in_the_tree_has_a_conformance_case():
    """Read from the source, since `STATE_OPS` holds only what this process imported, and runs on CPU, where the cases
    themselves are skipped."""
    decorated = {
        node.name
        for path in Path("mstar").rglob("*.py") for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "state_op"
            for d in node.decorator_list
        )
    }
    assert decorated == {call.op.name for call in ARGUMENTS} == set(STATE_OPS) & decorated


@needs_cuda
@pytest.mark.parametrize("call", list(ARGUMENTS), ids=lambda call: call.op.name)
@pytest.mark.parametrize("layer", [0, LAYERS - 1])
@pytest.mark.parametrize("lengths", [{1: 1}, {1: 7}, {1: 3, 2: 21}, {1: 40, 2: 1, 3: 9}], ids=str)
def test_a_fused_kernel_conforms_to_its_reference(call, layer, lengths):
    for fmt in call.op.formats:
        assert fmt.devices is None or "cuda" in fmt.devices, "this test runs on CUDA, so it couldn't cover that format"
    handle = _opened_step(_manager(), lengths)
    assert call.op.name in handle.fused_ops, "a listed format should run the kernel"
    args = ARGUMENTS[call](handle, layer, sum(lengths.values()))
    assert conformance_problems(call, args, TOLERANCE) == []


@needs_cuda
def test_the_debug_check_passes_the_thinkers_kernel_on_real_pages(monkeypatch):
    monkeypatch.setattr(sys.modules["mstar.engine.state.state_op"], "_DEBUG_ASSERTS", True)
    handle = _opened_step(_manager(), {1: 3, 2: 4})
    omni_attention.qk_norm_rope_table_attend(*_thinker_args(handle, 1, 7))


@needs_cuda
def test_the_debug_check_passes_the_plain_rope_kernel_on_real_pages(monkeypatch):
    monkeypatch.setattr(sys.modules["mstar.engine.state.state_op"], "_DEBUG_ASSERTS", True)
    handle = _opened_step(_manager(), {1: 3, 2: 4})
    qwen3_lm.qk_norm_rope_kv_write(*_plain_rope_args(handle, 1, 7))
