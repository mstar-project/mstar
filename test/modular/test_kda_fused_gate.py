"""The fused KDA kernels with either output gate, on CPU in Triton's interpreter: Kimi K3.1's
full-rank gate at one TP8 rank's shape (12 heads of 128) and GLM-5.3's low-rank one (8 heads of
128). The layer serves prefill, decode and verify steps (each verdict settled) through the KDA
resource twice, once per kernel bundle, and the fused bundle tracks the torch reference.

Each case runs this file as a script in a subprocess with ``TRITON_INTERPRET=1``, which Triton
reads when the kernels are defined. The CUDA parity tests are in test_kda_triton.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None

GEOMETRIES = {
    "k31": dict(num_heads=12, head_dim=128, full_rank_output_gate=True),
    "glm5": dict(num_heads=8, head_dim=128, full_rank_output_gate=False),
}
HIDDEN = 256  # the in-projection's width is not under test
SPEC_K = 3
POOL, KDA = "kda_state", "kda"
# (spans, walk, verdicts of a verify step or None)
SERVE = [
    ({"a": 5, "b": 70}, "prefill", None),  # under the conv width; past one chunk
    ({"a": 1, "b": 1, "c": 2}, "prefill", None),  # continues beside a new prompt
    *[({"a": 1, "b": 1, "c": 1}, "decode", None)] * 3,
]
VERIFY = [
    ({"a": 5, "b": 70}, "prefill", None),
    ({"a": 4, "b": 4}, "decode", [1, 3]),  # k + 1 tokens
    ({"a": 2, "b": 2}, "decode", [0, 1]),
    ({"a": 1, "b": 1}, "decode", [0, 0]),
    ({"a": 4, "b": 4}, "decode", [2, 0]),
]


@pytest.mark.parametrize("walks", ["serve", "verify"])
@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_fused_kernels_match_the_reference_in_the_interpreter(geometry, walks):
    pytest.importorskip("triton")
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "TRITON_INTERPRET": "1", "OMP_NUM_THREADS": "2",
           "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")])}
    run = subprocess.run([sys.executable, __file__, geometry, walks], env=env,
                         capture_output=True, text=True, check=False, timeout=110)
    assert run.returncode == 0, (run.stdout[-2000:], run.stderr[-4000:])


# -- the case, in the interpreter ----------------------------------------------


if triton is not None:

    @triton.jit
    def _exact_sigmoid(x):
        return 1.0 / (1.0 + tl.exp2(x * -1.4426950408889634))


def _interpreter_patches():
    """Triton 3.6's interpreter, made to compute what the GPU does: numpy 2 refuses a 1-element
    array as an index (a loop over a loaded count); fp32 -> bf16 truncates where the GPU rounds
    to nearest even; its bf16 dot multiplies the raw bits; it has no inline PTX, which
    ``kda_triton._sigmoid`` is, so the same formula with an exact reciprocal stands in."""
    import numpy as np
    import triton.runtime.interpreter as ti

    patch = ti._patch_lang_tensor

    def patched(tensor, scope):
        patch(tensor, scope)
        scope.set_attr(tensor, "__index__", lambda self: int(self.handle.data.reshape(-1)[0]))

    ti._patch_lang_tensor = patched

    cast = ti.InterpreterBuilder.cast_impl

    def rounded(self, src, dst_type):
        if src.dtype.scalar != tl.float32 or dst_type.scalar != tl.bfloat16:
            return cast(self, src, dst_type)
        data = torch.from_numpy(np.ascontiguousarray(src.data, dtype=np.float32))
        return ti.TensorHandle(data.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16),
                               tl.bfloat16)

    ti.InterpreterBuilder.cast_impl = rounded
    dot = ti.InterpreterBuilder.create_dot

    def fp32(x):
        if x.dtype.scalar != tl.bfloat16:
            return x
        data = ti._convert_float(x.data, tl.bfloat16, tl.float32, None).view(np.float32)
        return ti.TensorHandle(data, tl.float32)

    ti.InterpreterBuilder.create_dot = lambda self, a, b, *args: dot(self, fp32(a), fp32(b), *args)

    from mstar.engine.resources.linear_attn import kda_triton

    kda_triton._sigmoid = _exact_sigmoid
    kda_triton._num_sms = lambda device: 132  # an H200's


def _layer(geometry: str):
    from mstar.model.components.distributed import ParallelKimiDeltaAttention
    from mstar.model.components.kda import KDAConfig

    torch.manual_seed(0)
    layer = ParallelKimiDeltaAttention(KDAConfig(hidden_size=HIDDEN, **GEOMETRIES[geometry]),
                                       dtype=torch.bfloat16)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if "conv1d" in name:
                p.normal_(0.0, 0.3)
            elif name.endswith("A_log"):
                p.normal_(0.0, 0.5)
            elif name.endswith("dt_bias"):
                p.uniform_(-8.0, 2.0)  # decay from ~0 to the lower bound
            elif "norm" in name:
                p.normal_(1.0, 0.1)
            else:
                p.normal_(0.0, 0.06)
    layer.process_weights_after_loading("cpu")
    # the gated-normed output itself, not its o_proj
    layer._project_out = lambda out, hidden_states: out
    return layer.requires_grad_(False)


class _Arm:
    """One layer, its pool (state and conv tails random) and its KDA resource on ``kernels``."""

    def __init__(self, geometry: str, kernels, speculative_tokens: int):
        from mstar.engine.resources.base import EngineResourceInfo, build_resource
        from mstar.engine.resources.linear_attn.config import (
            LinearAttnBackend,
            LinearAttnConfig,
            LinearAttnSpec,
            LinearAttnVariant,
        )
        from mstar.engine.resources.recurrent.config import (
            DeltaNetGeometry,
            RecurrentStateConfig,
            RecurrentStateSpec,
        )

        self.layer = layer = _layer(geometry)
        cpu = torch.device("cpu")
        geometry = DeltaNetGeometry(
            num_k_heads=layer.num_heads, num_v_heads=layer.num_heads,
            head_k_dim=layer.head_dim, head_v_dim=layer.head_dim, conv_kernel_size=4,
        )
        pool_spec = RecurrentStateSpec(POOL, {"llm"}, RecurrentStateConfig(
            num_layers=1, max_slots=5, blocks=geometry.to_blocks(
                conv_dtype=torch.bfloat16, speculative_tokens=speculative_tokens),
        ))
        kda_spec = LinearAttnSpec(KDA, {"llm"}, LinearAttnConfig(
            recurrent_state=POOL, variant=LinearAttnVariant.KDA, backend=LinearAttnBackend.TRITON,
        ))
        self.pool = build_resource(pool_spec, EngineResourceInfo(device=cpu))
        self.kda = build_resource(kda_spec, EngineResourceInfo(
            device=cpu, dependencies={POOL: pool_spec}))
        self.kda.kernels = kernels
        gen = torch.Generator().manual_seed(1)
        for name in ("state", "conv"):
            block = self.pool.block(name, 0)
            block.copy_(torch.randn(block.shape, generator=gen) * (0.1 if name == "state" else 0.5))
        layer.bind_resources({POOL: self.pool, KDA: self.kda})

    def step(self, spans: dict[str, int], walk: str, verdicts, x: torch.Tensor) -> torch.Tensor:
        """One step through the layer and the resources' lifecycle; a verify step's verdicts
        settle through the layer's ``set_prefix_len``."""
        from mstar.engine.resources.linear_attn.config import LinearAttnStep
        from mstar.engine.resources.recurrent.config import RecurrentStep
        from mstar.engine.resources.step import Segment, StepContext

        pool, kda = self.pool, self.kda
        verify = verdicts is not None
        segments = tuple(Segment(rid, "main", n) for rid, n in spans.items())
        for rid in spans:
            if rid not in pool._slots:
                pool.ingest_request(rid)
        ctx = StepContext(request_ids=tuple(spans), graph_walk=walk, slot=0, capture=False)
        pool_step = RecurrentStep(segments=segments)
        assert pool.admit(pool_step, ctx).ok
        ctx.plan_results[POOL] = pool.plan(pool_step, ctx)
        kda_step = LinearAttnStep(segments=segments, speculative=verify)
        kda.plan(kda_step, ctx)
        assert kda.current_plan().is_decode == (walk == "decode" and not verify)
        out = self.layer(x, 0)
        if verify:
            self.layer.mix.set_prefix_len(torch.tensor(verdicts))
        kda.commit(kda_step, ctx)
        pool.commit(pool_step, ctx)
        return out


def _rel(a, b) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _case(geometry: str, walks: str) -> None:
    """Both bundles step by step. After a prefill, whose dots round their operands to bf16, the
    fused pool takes the reference's, so each recurrent step starts from the same state: those
    round where the reference does and match it to fp32 summation order."""
    _interpreter_patches()
    from mstar.engine.resources.linear_attn import kda_triton
    from mstar.engine.resources.linear_attn.kda_torch import TorchKDAKernels

    steps = SERVE if walks == "serve" else VERIFY
    k = SPEC_K if walks == "verify" else 0
    fused = _Arm(geometry, kda_triton.TritonKDAKernels(), k)
    ref = _Arm(geometry, TorchKDAKernels(), k)
    width = fused.layer.num_heads * fused.layer.head_dim
    gen = torch.Generator().manual_seed(2)
    for i, (spans, walk, verdicts) in enumerate(steps):
        x = torch.randn(sum(spans.values()), HIDDEN, generator=gen).to(torch.bfloat16)
        out = fused.step(spans, walk, verdicts, x)
        want = ref.step(spans, walk, verdicts, x)
        what = f"{geometry} step {i} ({walk}{', verify' if verdicts else ''})"
        assert out.shape == want.shape == (x.shape[0], width), (what, out.shape)
        prefill = walk == "prefill"
        err = _rel(out, want)
        assert err < (1e-2 if prefill else 1e-3), f"{what}: output rel err {err:.2e}"
        got, wanted = fused.pool._blocks, ref.pool._blocks
        state_err = _rel(got["state"], wanted["state"])
        assert state_err < (1e-2 if prefill else 5e-5), f"{what}: state rel err {state_err:.2e}"
        # the conv tails and the cached verify inputs are copies of projection rows
        for name, block in wanted.items():
            if name != "state":
                assert torch.equal(got[name], block), f"{what}: {name} differs"
        print(f"{what}: output rel err {err:.2e}, state {state_err:.2e}")
        if prefill:
            got["state"].copy_(wanted["state"])


if __name__ == "__main__":
    _case(*sys.argv[1:])
