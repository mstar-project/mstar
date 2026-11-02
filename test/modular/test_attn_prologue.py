"""The packed attention prologue (``attn.prologue``): its reference does what a
Qwen3.5 layer does unfused, and the rotary tables it reads agree with the
step's. CPU only."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mstar.engine.resources.attn import prologue
from mstar.engine.resources.kv.manager import KVManager
from mstar.model.qwen3_5.components.language_model import Qwen3_5Attention, Qwen3_5LanguageModel
from mstar.model.qwen3_5.components.rope import apply_partial_mrope, compute_3d_cos_sin, compute_inv_freq
from mstar.utils.h2d import PinnedStager

T, H, KH, D, HALF = 5, 4, 2, 32, 4


def _args(gated=True):
    shape = {"tokens": T, "heads": H, "kv_heads": KH, "head_dim": D, "half": HALF}
    args = list(prologue._inputs(shape, torch.float32, "cpu", torch.Generator().manual_seed(0)))
    if not gated:
        args[0] = args[0][:, H * D:].contiguous()
        args[-1] = False
    return args


def _unfused(qkv, qw, kw, cos, sin, pos, gated):
    q_size = H * D * (2 if gated else 1)
    q, k, v = qkv.split([q_size, KH * D, KH * D], -1)
    gate = None
    if gated:
        q, gate = q.reshape(T, H, 2 * D).chunk(2, -1)
    else:
        q = q.reshape(T, H, D)

    def norm(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w)

    c, s = torch.cat([cos[pos], cos[pos]], -1), torch.cat([sin[pos], sin[pos]], -1)
    q, k = apply_partial_mrope(norm(q, qw), norm(k.reshape(T, KH, D), kw), c, s)
    return q, gate, k, v.reshape(T, KH, D)


def test_the_reference_is_the_layers_unfused_math():
    for gated in (True, False):
        args = _args(gated)
        q, gate, k, v = prologue.QKV_NORM_ROPE_CACHE(*args)
        qkv, qw, kw, cos, sin, pos, key_cache, value_cache, slots = args[:9]
        eq, egate, ek, ev = _unfused(qkv, qw, kw, cos, sin, pos, gated)
        torch.testing.assert_close(q, eq, rtol=0, atol=0)
        torch.testing.assert_close(k, ek, rtol=0, atol=0)
        torch.testing.assert_close(v, ev, rtol=0, atol=0)
        if gated:
            torch.testing.assert_close(gate, egate, rtol=0, atol=0)
        else:
            assert gate.numel() == 0
        block = key_cache.shape[1]
        torch.testing.assert_close(key_cache[slots // block, slots % block], ek, rtol=0, atol=0)
        torch.testing.assert_close(value_cache[slots // block, slots % block], ev, rtol=0, atol=0)
        assert v.data_ptr() != qkv.data_ptr()


def test_no_implementation_off_amd_gpus():
    assert prologue.QKV_NORM_ROPE_CACHE.unavailable_reason() is not None


def test_the_rope_tables_are_the_steps_tables():
    """One position grid per token: the three MRoPE grids agree, so the 1D
    table rounds to what ``build_cos_sin`` gives a decode step."""
    inv_freq = compute_inv_freq(64, 1e7)
    pos = torch.tensor([0, 1, 17, 4095])
    cos, sin = compute_3d_cos_sin(pos[None].expand(3, -1), inv_freq, [11, 11, 10], torch.bfloat16)
    freqs = inv_freq[None, :].float() * torch.arange(4096).float()[:, None]
    table_cos, table_sin = freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)
    assert torch.equal(cos[:, :32], table_cos[pos]) and torch.equal(cos[:, 32:], table_cos[pos])
    assert torch.equal(sin[:, :32], table_sin[pos])


def _manager(flat: bool) -> KVManager:
    kv = KVManager.__new__(KVManager)
    kv.kv_cache = SimpleNamespace(page_size=16)
    kv._plan_stager = PinnedStager(torch.long)
    kv._flat_slots = False
    kv._device = torch.device("cpu")
    kv._cg_max_seq_len = 8
    kv._static_plan_states = {}
    kv._staged_outputs = {}
    kv._current_plan_states = {}
    kv._preplan_states = {}
    kv._default_label = "main"
    if flat:
        kv.stage_flat_slots()
    return kv


def _decode_step(kv: KVManager) -> None:
    indptrs = SimpleNamespace(is_decode=True, decode_pages=[3, 5, 7], decode_offsets=np.array([0, 15, 2]))
    lease = SimpleNamespace(slot=0, bucket=SimpleNamespace(num_tokens=4))
    kv._setup_plan_states({"main": indptrs}, SimpleNamespace(is_preplan=False), lease)


def test_flat_slots_are_staged_only_for_a_layer_that_asked():
    plain = _manager(flat=False)
    _decode_step(plain)
    assert plain.write_flat_slots() is None

    asked = _manager(flat=True)
    _decode_step(asked)
    # the padding row writes the sink page
    assert asked._current_plan_states["main"].token_to_slot[:4].tolist() == [3 * 16, 5 * 16 + 15, 7 * 16 + 2, 0]
    assert asked.write_flat_slots().tolist() == [48, 95, 114]


def _stack(world_size: int, enabled: bool):
    asked = []

    def enable():
        asked.append(True)
        return enabled

    layers = []
    for _ in range(2):
        attn = Qwen3_5Attention.__new__(Qwen3_5Attention)
        attn.q_norm = object()
        attn.attend = SimpleNamespace(enable=enable)
        layers.append(SimpleNamespace(self_attn=attn))
    # a delta-net layer has no say in it
    layers.append(SimpleNamespace(self_attn=object()))
    return SimpleNamespace(_packed=None, comm_group=SimpleNamespace(world_size=world_size), layers=layers), asked


@pytest.mark.parametrize(
    ("world_size", "enabled", "packed", "asks"), [(1, True, True, 2), (1, False, False, 1), (2, True, False, 0)],
)
def test_which_stacks_take_the_packed_prologue(world_size, enabled, packed, asks):
    stack, asked = _stack(world_size, enabled)
    assert Qwen3_5LanguageModel.packed_prologue.fget(stack) is packed
    # settled once, so the slots are asked for once
    assert Qwen3_5LanguageModel.packed_prologue.fget(stack) is packed
    assert len(asked) == asks
