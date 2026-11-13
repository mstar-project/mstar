"""The fp8 kernels compile for sm_90 with the arguments their launchers pass, on any host.

Triton type-checks a kernel, and the helpers it calls, when a GPU first launches it; the CPU suite stands
recorders in for the kernels, so a helper called with the wrong arguments or a pointer of a type its loads can't
take only fails on a GPU. Here the real launchers run over recorders and Triton's compiler builds every launch
they made, with the argument types it would derive on the GPU."""

import pytest
import torch

pytest.importorskip("triton.backends.nvidia")
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402
from triton.runtime.jit import mangle_type  # noqa: E402

from mstar.utils import dense_fp8, quant_fp8  # noqa: E402
from mstar.utils.fp8_block import FP8_DTYPE  # noqa: E402
from mstar.utils.fused_moe import kernels, runner  # noqa: E402
from mstar.utils.situ import SiTU  # noqa: E402

BLOCK = (128, 128)


class _Launches:
    """Stands in for a ``@triton.jit`` kernel: records each launch's arguments instead of running it."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((args, kwargs))


def _record(monkeypatch, module, name: str) -> _Launches:
    launches = _Launches(getattr(module, name))
    monkeypatch.setattr(module, name, launches)
    return launches


def _compile(launches: _Launches) -> None:
    """Compile each distinct launch in ``launches`` for sm_90."""
    assert launches.calls, launches.fn
    names, done = launches.fn.arg_names, set()
    for args, kwargs in launches.calls:
        values = dict(zip(names, args, strict=False)) | {k: v for k, v in kwargs.items() if k in names}
        consts = {names[i]: values[names[i]] for i in launches.fn.constexprs}
        signature = {n: "constexpr" if n in consts else mangle_type(v) for n, v in values.items()}
        key = repr((signature, consts))
        if key in done:
            continue
        done.add(key)
        try:
            triton.compile(ASTSource(fn=launches.fn, signature=signature, constexprs=consts),
                           target=GPUTarget("cuda", 90, 32))
        except (RuntimeError, FileNotFoundError) as exc:
            if "ptxas" in str(exc).lower():
                pytest.skip(f"this host can't run ptxas: {exc}")
            raise


def _experts(tokens: int, experts: int = 8, hidden: int = 256, inter: int = 128, top_k: int = 2):
    """``fused_experts_fp8``'s arguments as a served block passes them (e4m3 bytes, fp32 scales, int64 ids)."""
    return (
        torch.zeros(tokens, hidden, dtype=torch.bfloat16),
        torch.zeros(experts, 2 * inter, hidden, dtype=torch.uint8),
        torch.zeros(experts, hidden, inter, dtype=torch.uint8),
        torch.ones(experts, 2 * inter // BLOCK[0], hidden // BLOCK[1]),
        torch.ones(experts, hidden // BLOCK[0], inter // BLOCK[1]),
        torch.ones(tokens, top_k),
        torch.arange(tokens * top_k).reshape(tokens, top_k) % experts,
    )


@pytest.mark.parametrize("activation", ["silu", SiTU(4.0, 25.0).spec], ids=["silu", "situ"])
def test_the_fused_fp8_experts_launches_compile(monkeypatch, activation):
    """Every kernel ``fused_experts_fp8`` launches, for SwiGLU with a clamp and for a SiTU (which takes the
    activation and the quantization as two launches)."""
    quant = _record(monkeypatch, quant_fp8, "per_token_group_quant_fp8_kernel")
    gemm = _record(monkeypatch, kernels, "fused_moe_kernel_fp8_w8a8")
    act = _record(monkeypatch, kernels, "act_and_mul_kernel")
    act_quant = _record(monkeypatch, kernels, "act_and_mul_quant_fp8_kernel")
    reduce = _record(monkeypatch, kernels, "moe_sum_reduce_kernel")
    runner.fused_experts_fp8(*_experts(tokens=4), block_size=BLOCK, activation=activation, swiglu_limit=10.0)
    assert len(gemm.calls) == 2  # gate/up, then down
    for launches in (quant, gemm, act if activation != "silu" else act_quant, reduce):
        _compile(launches)
    assert not (act_quant if activation != "silu" else act).calls


@pytest.mark.parametrize("activation", ["silu", "gelu", SiTU(1.5).spec, SiTU(4.0, 25.0).spec])
def test_the_gated_activation_launches_compile(monkeypatch, activation):
    """``act_and_mul_triton`` (all slots, and the rows an alignment lists) and ``act_and_mul_quant_fp8`` for each
    activation the experts name."""
    quant = _record(monkeypatch, quant_fp8, "per_token_group_quant_fp8_kernel")
    act = _record(monkeypatch, kernels, "act_and_mul_kernel")
    act_quant = _record(monkeypatch, kernels, "act_and_mul_quant_fp8_kernel")
    gate_up = torch.zeros(6, 512, dtype=torch.bfloat16)
    out = torch.zeros(6, 256, dtype=torch.bfloat16)
    kernels.act_and_mul_triton(gate_up, out, activation=activation)
    kernels.act_and_mul_triton(gate_up, out, activation=activation, rows=torch.arange(8, dtype=torch.int32))
    _compile(act)
    kernels.act_and_mul_quant_fp8(gate_up, 128, activation=activation)
    situ = SiTU.parse(activation) is not None
    _compile(quant if situ else act_quant)
    assert not (act_quant if situ else quant).calls


@pytest.mark.parametrize("glu", [False, True], ids=["linear", "glu"])
@pytest.mark.parametrize("tokens", [2, 16], ids=["fma", "dot"])
def test_the_dense_fp8_w8a16_launch_compiles(monkeypatch, tokens, glu):
    """A served layer hands ``fp8_block`` its weight as e4m3 (what the stored bytes stand for): the w8a16
    kernel, which loads bytes, compiles for it on its FMA and dot tiles."""
    cpu = torch.device("cpu")
    monkeypatch.setitem(dense_fp8._COUNTERS, cpu, torch.zeros(dense_fp8._MAX_TILES, dtype=torch.int32))
    w8a16 = _record(monkeypatch, dense_fp8, "_w8a16_kernel")
    n, k = 256, 2048
    weight = torch.zeros(2 * n if glu else n, k, dtype=FP8_DTYPE)
    scale = torch.ones(weight.shape[0] // BLOCK[0], k // BLOCK[1])
    dense_fp8._linear(torch.zeros(tokens, k, dtype=torch.bfloat16), weight, scale, BLOCK, glu)
    (launch,) = w8a16.calls
    assert launch[1]["DOT"] is (tokens > 2) and launch[1]["SPLIT"] > 1
    _compile(w8a16)
