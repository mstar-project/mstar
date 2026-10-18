"""Fused MoE on an AMD GPU against torch references.

Covers the hipified align op (vs the torch version), the Triton backend with
default, ROCm-default and every shipped tuned config, and AITER's fused MoE,
each vs an fp32 torch MoE at Qwen3-Omni's expert sizes (E=128, Thinker hidden
2048 / inter 768 / top-8, Talker 1024 / 384 / top-6). Eager, graph replay and
compiled.

    pytest test/rocm/test_fused_moe.py
"""

from __future__ import annotations

import importlib.util
import json

import pytest
import torch
import torch.nn.functional as F

from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm kernels run on AMD GPUs")

DEV = torch.device("cuda")
THINKER = (128, 768, 2048, 8)  # E, inter, hidden, top_k
TALKER = (128, 384, 1024, 6)
HAS_AITER = importlib.util.find_spec("aiter") is not None


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _problem(dims):
    from mstar.utils.fused_moe.config import MoEProblem

    return MoEProblem(*dims)


def _inputs(dims, num_tokens, seed=0):
    e, inter, hidden, top_k = dims
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(num_tokens, hidden, device=DEV, dtype=torch.bfloat16, generator=g)
    w1 = torch.randn(e, 2 * inter, hidden, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02
    w2 = torch.randn(e, hidden, inter, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02
    logits = torch.randn(num_tokens, e, device=DEV, generator=g)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # fp32, as TopKRouter returns
    return x, w1, w2, weights, ids


def _reference(x, w1, w2, weights, ids):
    """fp32 reference MoE, per expert silu(x W_gate) * (x W_up) W_down, weighted and summed."""
    out = torch.zeros(x.shape[0], w2.shape[1], device=x.device, dtype=torch.float32)
    x32 = x.float()
    for e in ids.unique().tolist():
        rows, slots = torch.where(ids == e)
        gate, up = (x32[rows] @ w1[e].float().T).chunk(2, dim=-1)
        y = (F.silu(gate) * up) @ w2[e].float().T
        out.index_add_(0, rows, y * weights[rows, slots, None].float())
    return out


def _assert_matches(out, ref):
    assert out.shape == ref.shape and torch.isfinite(out).all()
    # bf16 intermediates against an fp32 reference
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    cos = F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0)
    assert cos > 0.9999, cos


# ---------------------------------------------------------------------------
# Align op
# ---------------------------------------------------------------------------


def _align_buffers(numel, num_experts, block_size):
    max_padded = numel + num_experts * (block_size - 1)
    return (
        torch.full((max_padded,), -1, dtype=torch.int32, device=DEV),
        torch.full((-(-max_padded // block_size),), -1, dtype=torch.int32, device=DEV),
        torch.empty(1, dtype=torch.int32, device=DEV),
    )


@pytest.mark.parametrize("num_tokens,top_k,num_experts", [
    (1, 8, 128), (7, 8, 128), (64, 8, 128), (128, 8, 128), (1000, 8, 128), (4096, 8, 128),
    (1, 6, 128), (33, 6, 128), (2048, 6, 128),
    (4, 8, 64), (100, 8, 64),  # small-batch kernel (E <= 64, numel < 1024)
])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("sentinels", [False, True])
def test_align_op_matches_torch(num_tokens, top_k, num_experts, block_size, sentinels):
    from mstar.utils.fused_moe import align

    assert align._cuda_op_available(), "the align op did not build on ROCm"
    ids = torch.randint(0, num_experts, (num_tokens, top_k), device=DEV, dtype=torch.int32)
    if sentinels:  # ids >= num_experts are skipped slots
        ids = torch.where(torch.rand(ids.shape, device=DEV) < 0.3, num_experts, ids)
    numel = ids.numel()

    hip = _align_buffers(numel, num_experts, block_size)
    torch.ops._mstar_moe_C.moe_align_block_size(ids, num_experts, block_size, *hip)
    ref = _align_buffers(numel, num_experts, block_size)
    align._moe_align_block_size_torch(ids, block_size, num_experts, *ref)

    total = int(ref[2].item())
    assert int(hip[2].item()) == total
    nblocks = total // block_size
    assert torch.equal(hip[1][:nblocks], ref[1][:nblocks])
    # Order within an expert's span is unspecified (atomics), so compare each
    # expert's slots as a set keyed by (expert, slot).
    def keyed(sorted_ids):
        expert = hip[1][:nblocks].long().repeat_interleave(block_size)
        return (expert * (numel + 1) + sorted_ids[:total].long()).sort().values

    assert torch.equal(keyed(hip[0]), keyed(ref[0]))
    assert (hip[0][total:] == numel).all()


def test_align_op_uses_torchs_hip_runtime():
    from mstar.utils.fused_moe import align

    assert align._cuda_op_available()
    with open("/proc/self/maps") as f:
        libs = {line.rsplit(None, 1)[-1] for line in f if "libamdhip64" in line}
    assert len(libs) == 1, libs


# ---------------------------------------------------------------------------
# Triton backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dims", [THINKER, TALKER], ids=["thinker", "talker"])
@pytest.mark.parametrize("num_tokens", [1, 3, 16, 64, 256, 1024])
def test_triton_backend_matches_reference(dims, num_tokens):
    from mstar.utils.fused_moe.backend import TritonFusedMoE
    from mstar.utils.fused_moe.runner import fused_experts

    x, w1, w2, weights, ids = _inputs(dims, num_tokens)
    ref = _reference(x, w1, w2, weights, ids)
    backend = TritonFusedMoE(_problem(dims))
    _assert_matches(backend(x, w1, w2, weights, ids), ref)
    _assert_matches(fused_experts(x, w1, w2, weights, ids), ref)  # the kernel's own default


def _shipped_tables():
    from mstar.utils.fused_moe.config import CONFIG_DIR, MoEProblem, config_file_name, device_config_name

    name = device_config_name()
    out = []
    for dims in (THINKER, TALKER):
        path = CONFIG_DIR / config_file_name(MoEProblem(*dims), name)
        if path.is_file():
            table = json.loads(path.read_text())
            out += [(dims, int(m), cfg) for m, cfg in table.items() if m.isdigit()]
    return out


def test_every_shipped_config_matches_reference():
    """A tuned entry with a bad compiler-option combination compiles and runs
    but computes garbage (kpack=2 with 16-wide K tiles did on gfx942)."""
    from mstar.utils.fused_moe.runner import fused_experts

    entries = _shipped_tables()
    if not entries:
        pytest.skip(f"no tuned table for {torch.cuda.get_device_name()}")
    for dims, m, cfg in entries:
        x, w1, w2, weights, ids = _inputs(dims, m)
        out = fused_experts(x, w1, w2, weights, ids, config=cfg)
        try:
            _assert_matches(out, _reference(x, w1, w2, weights, ids))
        except AssertionError as e:
            raise AssertionError(f"dims {dims} M={m} config {cfg}: {e}") from e


# ---------------------------------------------------------------------------
# AITER backend
# ---------------------------------------------------------------------------


def _aiter_backend(dims, w1, w2):
    from mstar.utils.fused_moe.backend_aiter import AiterFusedMoE

    backend = AiterFusedMoE(_problem(dims))
    p1, p2 = torch.nn.Parameter(w1.clone(), requires_grad=False), torch.nn.Parameter(w2.clone(), requires_grad=False)
    backend.prepare_weights(p1, p2)
    return backend, p1, p2


@pytest.mark.skipif(not HAS_AITER, reason="aiter not installed")
@pytest.mark.parametrize("dims", [THINKER, TALKER], ids=["thinker", "talker"])
@pytest.mark.parametrize("num_tokens", [1, 3, 16, 64, 256, 1024])
def test_aiter_backend_matches_reference(dims, num_tokens):
    x, w1, w2, weights, ids = _inputs(dims, num_tokens)
    backend, p1, p2 = _aiter_backend(dims, w1, w2)
    _assert_matches(backend(x, p1, p2, weights, ids), _reference(x, w1, w2, weights, ids))


@pytest.mark.skipif(not HAS_AITER, reason="aiter not installed")
def test_aiter_backend_refuses_unaligned_intermediate():
    from mstar.utils.fused_moe.backend_aiter import AiterFusedMoE
    from mstar.utils.fused_moe.config import MoEProblem

    with pytest.raises(ValueError, match="multiple of 128"):
        AiterFusedMoE(MoEProblem(128, 192, 1024, 6))  # the Talker's experts at TP2


# ---------------------------------------------------------------------------
# Capture and replay, and compile
# ---------------------------------------------------------------------------


def _backends(dims, w1, w2):
    from mstar.utils.fused_moe.backend import TritonFusedMoE

    out = [("triton", TritonFusedMoE(_problem(dims)), w1, w2)]
    if HAS_AITER:
        backend, p1, p2 = _aiter_backend(dims, w1, w2)
        out.append(("aiter", backend, p1, p2))
    return out


@pytest.mark.parametrize("dims", [THINKER, TALKER], ids=["thinker", "talker"])
@pytest.mark.parametrize("num_tokens", [1, 8, 32, 512])
def test_backend_capture_replay(dims, num_tokens):
    """A graph captured with one routing should replay another one, since the
    align op and kernels read routing from device buffers."""
    x, w1, w2, weights, ids = _inputs(dims, num_tokens, seed=0)
    for name, backend, b1, b2 in _backends(dims, w1, w2):
        sx, sw, si = x.clone(), weights.clone(), ids.clone()
        backend(sx, b1, b2, sw, si)  # warm up so compile/JIT happens outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = backend(sx, b1, b2, sw, si)
        for seed in (1, 2):
            nx, _, _, nw, ni = _inputs(dims, num_tokens, seed=seed)
            sx.copy_(nx), sw.copy_(nw), si.copy_(ni)
            graph.replay()
            torch.cuda.synchronize()
            try:
                _assert_matches(out, _reference(nx, w1, w2, nw, ni))
            except AssertionError as e:
                raise AssertionError(f"{name}: {e}") from e
        del graph


@pytest.mark.parametrize("backend_kind", ["triton", "aiter"])
def test_compiled_block_has_no_graph_breaks(backend_kind):
    import torch._dynamo

    from mstar.model.components import SparseMoeBlock
    from mstar.model.components.moe import prepare_moe_weights

    if backend_kind == "aiter" and not HAS_AITER:
        pytest.skip("aiter not installed")
    e, inter, hidden, top_k = THINKER
    block = SparseMoeBlock(hidden, e, top_k, inter, moe_backend=backend_kind).to(DEV, torch.bfloat16)
    with torch.no_grad():
        for p in block.parameters():
            p.normal_(std=0.02)
    prepare_moe_weights(block)
    x = torch.randn(8, hidden, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        eager = block(x)
        explained = torch._dynamo.explain(block)(x)
        assert explained.graph_break_count == 0, [str(b.reason)[:200] for b in explained.break_reasons]
        torch._dynamo.reset()
        compiled = torch.compile(block, fullgraph=True)(x)
    torch.testing.assert_close(compiled, eager, atol=2e-2, rtol=2e-2)
