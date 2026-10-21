"""Which kernel the RMSNorm, RoPE and sampling ops reach on each platform.

The ops' "cuda" kernels serve NVIDIA and AMD GPUs alike (torch calls both
"cuda" devices), so each picks its implementation from ``current_platform``:
FlashInfer on NVIDIA, exactly as before, and FlashInfer's ROCm fork on AMD. These
tests point ``current_platform`` at a chosen platform and replace ``flashinfer``
with a recording stand-in, so they run without a GPU or FlashInfer.
"""

import sys
import types

import pytest
import torch

from mstar import platforms
from mstar.engine.resources.position import rope as rope_ops
from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.rms_norm import rms_norm as norm_ops
from mstar.engine.resources.sampler import utils as sampler_utils
from mstar.platforms import CudaPlatform, RocmPlatform

_ROCM_BACKENDS = (
    "mstar.engine.resources.rms_norm.rocm",
    "mstar.engine.resources.position.rocm",
    "mstar.engine.resources.sampler.rocm",
)
_DISPATCHES = (
    norm_ops._rmsnorm,
    norm_ops._fused_add_rmsnorm,
    norm_ops.run_fused_add_rms_norm,
    rope_ops._rope,
    sampler_utils._top_p_sampling_from_probs,
    sampler_utils._top_k_top_p_sampling_from_probs,
)


class _Recorder:
    def __init__(self):
        self.calls = []

    def fn(self, name, result=None):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return result
        call.__name__ = name
        return call


def _fake_flashinfer(rec: _Recorder, rocm: bool, seeded: bool = True) -> types.ModuleType:
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(
        rmsnorm=rec.fn("norm.rmsnorm", result=torch.zeros(1)),
        fused_add_rmsnorm=rec.fn("norm.fused_add_rmsnorm"),
        gemma_fused_add_rmsnorm=rec.fn("norm.gemma_fused_add_rmsnorm"),
    )
    fi.rope = types.SimpleNamespace(
        apply_rope_pos_ids_inplace=rec.fn("rope.apply_rope_pos_ids_inplace"),
        apply_llama31_rope_pos_ids_inplace=rec.fn("rope.apply_llama31_rope_pos_ids_inplace"),
        apply_rope_with_cos_sin_cache_inplace=rec.fn("rope.apply_rope_with_cos_sin_cache_inplace"),
    )

    def top_p_sampling_from_probs(probs, top_p, deterministic=True, seed=None, offset=None):
        rec.calls.append(("sampling.top_p", (probs, top_p), dict(seed=seed, offset=offset)))
        return "drawn"

    def top_k_top_p_sampling_from_probs(probs, top_k, top_p, deterministic=True, seed=None, offset=None):
        rec.calls.append(("sampling.top_k_top_p", (probs, top_k, top_p), dict(seed=seed, offset=offset)))
        return "drawn"

    def unseeded(probs, top_k, top_p, deterministic=True):
        return "drawn"

    fi.sampling = types.SimpleNamespace(
        top_p_sampling_from_probs=top_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs=top_k_top_p_sampling_from_probs if seeded else unseeded,
    )
    if rocm:
        fi.rocm = types.ModuleType("flashinfer.rocm")
    return fi


@pytest.fixture
def stack(monkeypatch):
    """``stack(platform, rocm_fork=..., seeded=...)``: point the platform and
    ``flashinfer`` at stand-ins; returns the call recorder."""
    def use(platform, rocm_fork=True, seeded=True):
        rec = _Recorder()
        fi = _fake_flashinfer(rec, rocm=rocm_fork, seeded=seeded)
        monkeypatch.setattr(platforms, "resolve_platform", lambda: platform)
        monkeypatch.setitem(sys.modules, "flashinfer", fi)
        for sub in ("norm", "rope", "sampling"):
            monkeypatch.setitem(sys.modules, f"flashinfer.{sub}", getattr(fi, sub))
        if rocm_fork:
            monkeypatch.setitem(sys.modules, "flashinfer.rocm", fi.rocm)
        else:
            monkeypatch.delitem(sys.modules, "flashinfer.rocm", raising=False)
        for name in _ROCM_BACKENDS:
            monkeypatch.delitem(sys.modules, name, raising=False)
        for dispatch in _DISPATCHES:
            dispatch.reset()
        return rec

    yield use
    for dispatch in _DISPATCHES:
        dispatch.reset()
    for name in _ROCM_BACKENDS:
        sys.modules.pop(name, None)


# --- RMSNorm ----------------------------------------------------------------


def test_nvidia_rms_norm_is_flashinfers_kernel(stack):
    rec = stack(CudaPlatform(), rocm_fork=False)
    x, w = torch.ones(2, 3, 8), torch.ones(8)
    norm_ops._rmsnorm(x, w, eps=1e-5)
    [(name, args, kwargs)] = rec.calls
    assert name == "norm.rmsnorm" and args[0] is x and kwargs == {"eps": 1e-5}


def test_amd_rms_norm_hands_the_fork_2d_rows(stack):
    rec = stack(RocmPlatform())
    rec_result = torch.arange(48.0).view(6, 8)
    sys.modules["flashinfer"].norm.rmsnorm = rec.fn("norm.rmsnorm", result=rec_result)
    x = torch.ones(2, 3, 8)
    out = norm_ops._rmsnorm(x, torch.ones(8), eps=1e-5)
    [(name, args, kwargs)] = rec.calls
    assert name == "norm.rmsnorm" and args[0].shape == (6, 8) and kwargs == {"eps": 1e-5}
    assert out.shape == x.shape and torch.equal(out.view(6, 8), rec_result)
    x2 = torch.ones(4, 8)
    norm_ops._rmsnorm(x2, torch.ones(8), eps=1e-5)
    assert rec.calls[-1][1][0] is x2, "2-D input goes through unchanged"


@pytest.mark.parametrize("platform", [CudaPlatform(), RocmPlatform()], ids=repr)
@pytest.mark.parametrize("gemma", [False, True])
def test_fused_add_rms_norm_is_flashinfers_in_place_kernel(stack, platform, gemma):
    rec = stack(platform, rocm_fork=platform.is_rocm())
    h, r, w = torch.ones(2, 8), torch.ones(2, 8), torch.ones(8)
    norm_ops._fused_add_rmsnorm(h, r, w, 1e-5, gemma)
    [(name, args, _)] = rec.calls
    assert name == ("norm.gemma_fused_add_rmsnorm" if gemma else "norm.fused_add_rmsnorm")
    assert args[0] is h and args[1] is r and args[2] is w and args[3] == 1e-5


@pytest.mark.parametrize("gemma", [False, True])
def test_nvidia_forward_residual_calls_flashinfer_directly(stack, monkeypatch, gemma):
    """NVIDIA keeps the direct in-place call, outside any op."""
    rec = stack(CudaPlatform(), rocm_fork=False)

    def op(*args):
        raise AssertionError("NVIDIA went through mstar::fused_add_rms_norm")

    monkeypatch.setattr(torch.ops.mstar, "fused_add_rms_norm", op)
    h, r, w = torch.ones(2, 8), torch.ones(2, 8), torch.ones(8)
    norm_ops.run_fused_add_rms_norm(h, r, w, 1e-5, gemma)
    [(name, args, _)] = rec.calls
    assert name == ("norm.gemma_fused_add_rmsnorm" if gemma else "norm.fused_add_rmsnorm")
    assert args[0] is h and args[1] is r


def test_amd_forward_residual_goes_through_the_op(stack, monkeypatch):
    stack(RocmPlatform())
    calls = []
    monkeypatch.setattr(
        torch.ops.mstar, "fused_add_rms_norm", lambda *args: calls.append(args),
    )
    h, r, w = torch.ones(2, 8), torch.ones(2, 8), torch.ones(8)
    norm_ops.run_fused_add_rms_norm(h, r, w, 1e-5, True)
    assert calls == [(h, r, w, 1e-5, True)]


def test_fused_add_rms_norm_fallback_follows_flashinfer_numerics():
    torch.manual_seed(0)
    h0 = torch.randn(5, 64, dtype=torch.bfloat16)
    r0 = torch.randn(5, 64, dtype=torch.bfloat16) * 3
    w = torch.randn(64, dtype=torch.bfloat16)
    for gemma in (False, True):
        h, r = h0.clone(), r0.clone()
        torch.ops.mstar.fused_add_rms_norm(h, r, w, 1e-6, gemma)
        total = h0.float() + r0.float()
        scale = w.float() + (1.0 if gemma else 0.0)
        expected = total * torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + 1e-6) * scale
        assert torch.equal(r, total.bfloat16())
        assert torch.equal(h, expected.bfloat16())


def test_forward_residual_on_cpu_keeps_the_unfused_path(monkeypatch):
    """Only GPU tensors take the fused op; elsewhere the add and norm run apart."""
    from mstar.model.components.norm import RMSNorm

    def fused(*args):
        raise AssertionError("the fused op ran on CPU tensors")

    monkeypatch.setattr(torch.ops.mstar, "fused_add_rms_norm", fused)
    norm = RMSNorm(16)
    h, r = torch.randn(3, 16), torch.randn(3, 16)
    out, new_r = norm.forward_residual(h, r)
    assert torch.equal(new_r, h + r)


# --- RoPE -------------------------------------------------------------------


_ROPE_ARGS = dict(rotary_dim=None, interleave=False, rope_scale=1.0, rope_theta=1e4)


@pytest.mark.parametrize("llama31", [False, True])
def test_nvidia_rope_is_flashinfers_kernel(stack, llama31):
    rec = stack(CudaPlatform(), rocm_fork=False)
    q, k, pos = torch.ones(2, 4, 8), torch.ones(2, 2, 8), torch.arange(2)
    extra = (1.0, 4.0, 8192.0) if llama31 else (None, None, None)
    rope_ops._rope(q, k, pos, None, *_ROPE_ARGS.values(), *extra)
    [(name, args, kwargs)] = rec.calls
    if llama31:
        assert name == "rope.apply_llama31_rope_pos_ids_inplace"
        assert kwargs["low_freq_factor"] == 1.0 and kwargs["old_context_len"] == 8192.0
    else:
        assert name == "rope.apply_rope_pos_ids_inplace"
    assert args == (q, k, pos)


@pytest.mark.parametrize("interleave", [False, True])
def test_amd_rope_reads_the_cos_sin_cache(stack, interleave):
    rec = stack(RocmPlatform())
    q, k, pos = torch.ones(2, 4, 8), torch.ones(2, 2, 8), torch.arange(2)
    cache = torch.zeros(16, 8)
    rope_ops._rope(q, k, pos, cache, 8, interleave, 1.0, 1e4, None, None, None)
    [(name, args, kwargs)] = rec.calls
    assert name == "rope.apply_rope_with_cos_sin_cache_inplace"
    assert args == (pos, q, k, 8, cache) and kwargs == {"is_neox": not interleave}


def test_amd_without_the_fork_names_it(stack):
    stack(RocmPlatform(), rocm_fork=False)
    with pytest.raises(RuntimeError, match="amd-flashinfer"):
        norm_ops._rmsnorm(torch.ones(1, 8), torch.ones(8), eps=1e-5)


class _CacheRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, rotary_dim, rope_scale, rope_theta, dtype, **kwargs):
        self.calls.append((rotary_dim, rope_scale, rope_theta, dtype, kwargs))
        return torch.zeros(4, rotary_dim)


@pytest.mark.parametrize("platform", [CudaPlatform(), RocmPlatform()], ids=repr)
def test_only_amd_rope_builds_a_cache(monkeypatch, platform):
    monkeypatch.setattr(platforms, "resolve_platform", lambda: platform)
    build = _CacheRecorder()
    monkeypatch.setattr(RopeManager, "_build_rope_cache", build)
    config = PositionConfig(
        kv_cache="kv", rope_theta=5e5, rope_scale=32.0,
        low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192,
    )
    manager = RopeManager(config, torch.device("cuda"), max_seq_len=4, head_dim=128)
    if platform.is_cuda():
        assert build.calls == [] and manager._rope_cache is None
        return
    assert build.calls == [(128, 32.0, 5e5, torch.float32, {
        "llama31": (1.0, 4.0, 8192), "compute_dtype": torch.float64,
    })]
    assert manager._rope_cache_key == (128, 32.0, 5e5, (1.0, 4.0, 8192))


def test_amd_rope_takes_the_cache_only_for_its_own_arguments(monkeypatch):
    rocm = RocmPlatform()
    monkeypatch.setattr(platforms, "resolve_platform", lambda: rocm)
    monkeypatch.setattr(RopeManager, "_build_rope_cache", _CacheRecorder())
    calls = []
    monkeypatch.setattr(
        torch.ops.mstar, "rope_apply_qk_inplace", lambda *args, **kwargs: calls.append(args),
    )
    config = PositionConfig(kv_cache="kv", rope_theta=1e4, rotary_dim=32)
    manager = RopeManager(config, torch.device("cuda"), max_seq_len=4, head_dim=64)
    manager._current_pos_ids["text"] = torch.arange(3)
    q, k = torch.ones(3, 2, 64, dtype=torch.bfloat16), torch.ones(3, 1, 64, dtype=torch.bfloat16)
    # attention layers pass Llama 3.1 defaults, a no-op at scale 1
    manager.apply_qk(q, k, "text", rope_theta=1e4, low_freq_factor=1.0,
                     high_freq_factor=1.0, old_context_len=8192)
    manager.apply_qk(q, k, "text", rope_theta=1e6)
    manager.apply_qk(q, k, "text", rope_scale=8.0, low_freq_factor=1.0,
                     high_freq_factor=4.0, old_context_len=8192)
    assert [args[3] is manager._rope_cache for args in calls] == [True, False, False]
    assert [args[3] for args in calls[1:]] == [None, None]


def test_amd_rope_rotates_from_a_rope_table_dtype_table(monkeypatch):
    """A config that asks for a rounded table (rope_table_dtype) gets it on AMD
    GPUs too, in place of the fp32 cache."""
    rocm = RocmPlatform()
    monkeypatch.setattr(platforms, "resolve_platform", lambda: rocm)
    build = _CacheRecorder()
    monkeypatch.setattr(RopeManager, "_build_rope_cache", build)
    calls = []
    monkeypatch.setattr(
        torch.ops.mstar, "rope_apply_qk_inplace", lambda *args, **kwargs: calls.append(args),
    )
    config = PositionConfig(kv_cache="kv", rope_theta=1e6, rotary_dim=8, rope_table_dtype=torch.bfloat16)
    manager = RopeManager(config, torch.device("cuda"), max_seq_len=4, head_dim=8)
    assert manager._rope_cache is None
    assert [call[3] for call in build.calls] == [torch.bfloat16]
    manager._current_pos_ids["text"] = torch.arange(3)
    q, k = torch.ones(3, 1, 8, dtype=torch.bfloat16), torch.ones(3, 1, 8, dtype=torch.bfloat16)
    manager.apply_qk(q, k, "text")
    assert [args[3] is manager._rope_table for args in calls] == [True]


def test_amd_rope_off_the_cache_runs_flashinfers_on_the_fly_kernels(stack):
    rec = stack(RocmPlatform())
    q, k, pos = torch.ones(2, 4, 8), torch.ones(2, 2, 8), torch.arange(2)
    rope_ops._rope(q, k, pos, None, 8, False, 1.0, 1e6, 1.0, 1.0, 8192)
    rope_ops._rope(q, k, pos, None, 8, False, 32.0, 5e5, 1.0, 4.0, 8192)
    names = [name for name, _, _ in rec.calls]
    assert names == [
        "rope.apply_rope_pos_ids_inplace",  # the Llama 3.1 blend is a no-op at scale 1
        "rope.apply_llama31_rope_pos_ids_inplace",
    ]
    assert rec.calls[0][2]["rope_theta"] == 1e6


def test_llama31_cache_matches_flashinfers_frequencies():
    """The cache's angles are FlashInfer's: ``pos_enc.cuh`` blends the scaled and
    unscaled frequency by ``clamp(freq * a + b, 0, 1)``."""
    import math

    manager = RopeManager(PositionConfig(kv_cache="kv"), torch.device("cpu"), max_seq_len=64, head_dim=128)
    low, high, old, scale, theta, dim = 1.0, 4.0, 8192, 32.0, 5e5, 128
    cache = manager._build_rope_cache(
        dim, scale, theta, torch.float32, llama31=(low, high, old), compute_dtype=torch.float64,
    )
    freq = theta ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    smooth_a = old / (2 * math.pi * high - 2 * math.pi * low)
    smooth_b = -1.0 / (high / low - 1.0)
    smooth = (freq * smooth_a + smooth_b).clamp(0, 1)
    freq = (1 - smooth) * (freq / scale) + smooth * freq
    angles = torch.arange(64, dtype=torch.float64)[:, None] * freq[None, :]
    expected = torch.cat((angles.cos(), angles.sin()), dim=-1).float()
    assert cache.dtype == torch.float32
    torch.testing.assert_close(cache, expected, rtol=0, atol=1e-6)


# --- sampling ---------------------------------------------------------------


def test_nvidia_sampling_is_flashinfers_function(stack):
    stack(CudaPlatform(), rocm_fork=False)
    fi = sys.modules["flashinfer"]
    assert sampler_utils._top_k_top_p_sampling_from_probs._resolve() is (
        fi.sampling.top_k_top_p_sampling_from_probs
    )
    assert sampler_utils._top_p_sampling_from_probs._resolve() is fi.sampling.top_p_sampling_from_probs


def test_amd_sampling_passes_tensor_seeds_through(stack):
    rec = stack(RocmPlatform())
    probs, k, p = torch.ones(2, 4), torch.ones(2), torch.ones(2)
    seed, offset = torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long)
    out = sampler_utils._top_k_top_p_sampling_from_probs(
        probs, k, p, deterministic=True, seed=seed, offset=offset,
    )
    assert out == "drawn"
    [(name, args, kwargs)] = rec.calls
    assert name == "sampling.top_k_top_p" and kwargs["seed"] is seed and kwargs["offset"] is offset


def test_amd_sampling_runs_outside_dynamo(stack):
    """The fork raises when dynamo traces into it; behind ``torch.compiler.disable``
    a traced caller graph-breaks there instead."""
    stack(RocmPlatform())
    seen = []

    def body(x):
        seen.append(torch.compiler.is_compiling())
        return x

    sys.modules["flashinfer"].sampling.top_p_sampling_from_probs = (
        lambda probs, top_p, deterministic=True, seed=None, offset=None: body(probs) + 1
    )
    # resolved outside the trace, as a warmed-up worker has it
    sampler_utils._top_p_sampling_from_probs._resolve()

    def step(probs):
        return sampler_utils._top_p_sampling_from_probs(probs, None, seed=None, offset=None) * 2

    torch._dynamo.reset()
    out = torch.compile(step, backend="eager")(torch.ones(3))
    assert torch.equal(out, torch.full((3,), 4.0)) and seen == [False]


def test_amd_sampling_requires_tensor_seeds(stack):
    stack(RocmPlatform(), seeded=False)
    with pytest.raises(RuntimeError, match="seed/offset"):
        sampler_utils._top_k_top_p_sampling_from_probs(torch.ones(1, 4), 1, 1.0)


def test_prep_autotune_drops_configs_wider_than_a_block(monkeypatch):
    """32 warps is 2048 threads with AMD's 64-wide wavefronts, past the 1024 a
    block may hold; with NVIDIA's 32-wide warps every config fits."""
    configs = sampler_utils._fused_sampling_prep_kernel.configs
    for warp_size, expected in ((32, len(configs)), (64, len(configs) - 1)):
        monkeypatch.setattr(
            torch.cuda, "get_device_properties",
            lambda device, ws=warp_size: types.SimpleNamespace(warp_size=ws),
        )
        kept = sampler_utils._launchable(configs, torch.device("cuda"))
        assert len(kept) == expected
        assert all(c.num_warps * warp_size <= 1024 for c in kept)
