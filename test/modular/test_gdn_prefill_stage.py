"""The fused GDN prefill stage: its guard, how a wrapper picks it, and on a GPU its
values against the eager path it stands in for, alone and through the chunked kernel."""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.linear_attn import gdn_stage, wrappers
from mstar.engine.resources.linear_attn.gdn_stage import gdn_stage_applies, stage_gdn_prefill_inputs
from mstar.engine.resources.linear_attn.wrappers import (
    GDNDecodeWrapper,
    GDNMixedWrapper,
    GDNPrefillWrapper,
)
from mstar.engine.resources.recurrent import DeltaNetGeometry
from mstar.utils.causal_conv1d import PAD_SLOT_ID

_CUDA = torch.device("cuda")
_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _geometry(head_k_dim=128, head_v_dim=128):
    return DeltaNetGeometry(
        num_k_heads=16, num_v_heads=32, head_k_dim=head_k_dim, head_v_dim=head_v_dim,
        conv_kernel_size=4,
    )


@pytest.fixture
def blackwell(monkeypatch):
    monkeypatch.setattr(
        gdn_stage, "current_platform",
        SimpleNamespace(has_cuda_capability=lambda major, minor=0, device=None: major <= 10),
    )


def test_guard_takes_a_blackwell_gpu_with_one_head_dim(blackwell):
    assert gdn_stage_applies(_CUDA, _geometry())


def test_guard_refuses_cpu_and_older_gpus(monkeypatch, blackwell):
    assert not gdn_stage_applies(torch.device("cpu"), _geometry())
    monkeypatch.setattr(
        gdn_stage, "current_platform",
        SimpleNamespace(has_cuda_capability=lambda major, minor=0, device=None: major <= 9),
    )
    assert not gdn_stage_applies(_CUDA, _geometry())


def test_guard_refuses_shapes_the_kernel_does_not_take(blackwell):
    assert not gdn_stage_applies(_CUDA, _geometry(head_k_dim=64, head_v_dim=128))
    assert not gdn_stage_applies(_CUDA, _geometry(head_k_dim=1024, head_v_dim=1024))


def _prefill(stage_inputs, **kwargs):
    return GDNPrefillWrapper(
        device=torch.device("cpu"), pad_slot_id=PAD_SLOT_ID, stage_inputs=stage_inputs, **kwargs,
    )


def test_a_staged_wrapper_tells_the_caller_not_to_prepare():
    staged, plain = _prefill(True), _prefill(False)
    assert staged.qk_l2norm_in_kernel and staged.takes_strided_inputs
    assert not plain.qk_l2norm_in_kernel and not plain.takes_strided_inputs


@pytest.mark.parametrize("stage_inputs", [False, True])
def test_mixed_wrapper_normalizes_decode_rows_once(stage_inputs):
    mixed = GDNMixedWrapper(
        device=torch.device("cpu"), pad_slot_id=PAD_SLOT_ID, stage_inputs=stage_inputs,
    )
    assert mixed.qk_l2norm_in_kernel is stage_inputs
    # with the stage the caller skips the norm, so the decode kernel does it;
    # without it, the caller did, and doing it again would change the values
    assert mixed._decode._qk_l2norm is stage_inputs
    assert mixed._prefill._stage_inputs is stage_inputs
    assert mixed.takes_strided_inputs is stage_inputs


def test_decode_wrapper_is_untouched():
    decode = GDNDecodeWrapper(device=torch.device("cpu"), pad_slot_id=PAD_SLOT_ID)
    assert decode.qk_l2norm_in_kernel and decode.takes_strided_inputs


def test_run_goes_through_the_stage_with_the_plan_mask(monkeypatch):
    wrapper = _prefill(True, accelerator_graph=True, num_tokens=32, bs=2)
    mask = torch.ones(34)
    wrapper._plan_state = SimpleNamespace(
        slots=torch.zeros(2, dtype=torch.int32), has_state=torch.zeros(2, dtype=torch.bool),
        num_rows=2, cu_seqlens=torch.tensor([0, 16, 32]), token_mask=mask,
    )
    seen = {}

    def stage(q, k, v, a, b, a_log, dt_bias, token_mask, room, normalize):
        seen.update(room=room, normalize=normalize, mask=token_mask)
        return q, k, v, a, b

    def chunk(**kw):
        seen["alpha"] = kw["g"]
        return kw["v"], torch.zeros_like(kw["initial_state"])

    monkeypatch.setattr(wrappers, "stage_gdn_prefill_inputs", stage)
    monkeypatch.setattr("flashinfer.gdn_prefill.chunk_gated_delta_rule", chunk)
    q = torch.zeros(32, 16, 128)
    v = torch.zeros(32, 32, 128)
    a = torch.full((32, 32), 3.0)
    state = torch.zeros(4, 32, 128, 128)
    out = wrapper.run(q, q, v, a, a, state, torch.zeros(32), torch.zeros(32))
    assert seen["room"] == 2 and seen["normalize"] and seen["mask"] is mask
    assert seen["alpha"] is a
    assert out.shape[0] == 32


def _eager(q, k, v, a, b, a_log, dt_bias, mask, room):
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(q.dtype)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype)
    g = -torch.exp(a_log.float()) * torch.nn.functional.softplus(a.float() + dt_bias.float())
    beta = torch.sigmoid(b.float())
    q, k, v, g, beta = (
        torch.cat((t, t.new_zeros((room, *t.shape[1:]))), dim=0) for t in (q, k, v, g, beta)
    )
    keep = mask[: q.shape[0]].bool()
    v = torch.where(keep.view(-1, 1, 1), v, torch.zeros_like(v))
    g = torch.where(keep.view(-1, 1), g, torch.zeros_like(g))
    beta = torch.where(keep.view(-1, 1), beta, torch.zeros_like(beta))
    return q, k, v, torch.exp(g), beta


# one bf16 ulp at the magnitudes a unit vector's elements have
_BF16_TOL = 2e-3


@_needs_gpu
@pytest.mark.parametrize("tokens, room, real", [(32, 2, 32), (512, 8, 401), (4096, 16, 4000)])
def test_stage_matches_eager_on_strided_views(tokens, room, real):
    torch.manual_seed(tokens)
    wide = torch.randn(tokens, 12352, dtype=torch.bfloat16, device=_CUDA)
    q = wide[:, :2048].view(tokens, 16, 128)
    k = wide[:, 2048:4096].view(tokens, 16, 128)
    v = wide[:, 4096:8192].view(tokens, 32, 128)
    a, b = wide[:, 12288:12320], wide[:, 12320:]
    a_log = torch.randn(32, device=_CUDA)
    dt_bias = torch.randn(32, device=_CUDA)
    mask = torch.zeros(tokens + room, device=_CUDA)
    mask[:real] = 1
    ref = _eager(q, k, v, a, b, a_log, dt_bias, mask, room)
    got = stage_gdn_prefill_inputs(q, k, v, a, b, a_log, dt_bias, mask, room, normalize=True)
    for name, x, y in zip(("q", "k", "v", "alpha", "beta"), ref, got, strict=True):
        assert x.shape == y.shape, name
        assert y.is_contiguous(), name
        assert (x[:real].float() - y[:real].float()).abs().max() < _BF16_TOL, name
    assert (got.q[real:] == 0).all() and (got.v[real:] == 0).all()
    assert (got.alpha[real:] == 1).all() and (got.beta[real:] == 0).all()


@_needs_gpu
def test_stage_skips_the_norm_when_asked():
    q = torch.randn(32, 16, 128, dtype=torch.bfloat16, device=_CUDA)
    v = torch.randn(32, 32, 128, dtype=torch.bfloat16, device=_CUDA)
    a = torch.randn(32, 32, dtype=torch.bfloat16, device=_CUDA)
    mask = torch.ones(32, device=_CUDA)
    got = stage_gdn_prefill_inputs(
        q, q, v, a, a, torch.zeros(32, device=_CUDA), torch.zeros(32, device=_CUDA), mask, 0,
        normalize=False,
    )
    assert torch.equal(got.q, q)


@_needs_gpu
def test_chunked_kernel_agrees_through_either_path():
    """The same step through both wrappers, captured bucket width and all."""
    torch.manual_seed(0)
    tokens, rows = 256, 4
    spans = [100, 60, 0, 0]
    kwargs = dict(
        device=_CUDA, pad_slot_id=PAD_SLOT_ID, accelerator_graph=True,
        num_tokens=tokens, bs=rows,
    )
    q = torch.randn(tokens, 16, 128, dtype=torch.bfloat16, device=_CUDA)
    k = torch.randn(tokens, 16, 128, dtype=torch.bfloat16, device=_CUDA)
    v = torch.randn(tokens, 32, 128, dtype=torch.bfloat16, device=_CUDA)
    a = torch.randn(tokens, 32, dtype=torch.bfloat16, device=_CUDA)
    b = torch.randn(tokens, 32, dtype=torch.bfloat16, device=_CUDA)
    a_log = torch.randn(32, device=_CUDA)
    dt_bias = torch.randn(32, device=_CUDA)
    slots = torch.tensor([1, 2, 0, 0], dtype=torch.int32, device=_CUDA)
    has_state = torch.zeros(rows, dtype=torch.bool, device=_CUDA)

    def run(stage_inputs):
        wrapper = GDNPrefillWrapper(stage_inputs=stage_inputs, **kwargs)
        wrapper.plan(list(spans), slots, has_state)
        state = torch.zeros(4, 32, 128, 128, device=_CUDA)
        if stage_inputs:
            qq, kk = q, k
        else:
            qq = torch.nn.functional.normalize(q.float(), dim=-1).to(q.dtype)
            kk = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype)
        return wrapper.run(qq, kk, v, a, b, state, a_log, dt_bias), state

    out_ref, state_ref = run(False)
    out_new, state_new = run(True)
    real = sum(spans)
    assert (out_ref[:real].float() - out_new[:real].float()).abs().max() < 0.05
    assert (state_ref[1:3] - state_new[1:3]).abs().max() < 0.05
