"""The AttnRes read sites on FlashInfer's kimi_k3_attn_res, on CPU: the token-major residual stack, and the model's
reads through a stand-in for FlashInfer's call that keeps its contract (the addend added into ``prefix`` in place,
the block written, the mix normed in fp32). Through the real engine the tiny model's prefill and decode logits
match the reference with it, so the in-place stream never lands in a tensor something else still reads. The
kernel itself against the Triton read: ``test_attn_res_flashinfer_gpu.py`` (SM100 / SM103)."""
import types

import pytest
import torch

from mstar.model.components.norm import RMSNorm
from mstar.model.kimi_k3.components import attn_res
from mstar.model.kimi_k3.components.attn_res import attn_res_read, attn_res_read_norm
from mstar.model.kimi_k3.tests import engine_helpers as H
from mstar.model.kimi_k3.tests.test_engine_cpu import PROMPT, _greedy


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    H.install_cpu_stubs(monkeypatch)
    torch.set_num_threads(2)


def _fake_flashinfer(calls):
    """FlashInfer's ``kimi_k3_attn_res`` as its docstring states it, in torch."""

    def kimi_k3_attn_res(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, out, *, num_blocks,
                         block_write_idx=-1, eps=1e-5, output_norm_eps=1e-5, enable_pdl=False):
        calls.append({"m": num_blocks, "write": block_write_idx, "delta": delta is not None, "pdl": enable_pdl,
                      "blocks": blocks.shape[1], "eps": (eps, output_norm_eps)})
        assert blocks.stride(2) == 1 and prefix.shape == out.shape == (blocks.shape[0], blocks.shape[2])
        assert out.is_contiguous() and qk_weight.dim() == 1 and 0 <= num_blocks <= blocks.shape[1]
        if delta is not None:
            prefix.add_(delta)
        if block_write_idx >= 0:
            blocks[:, block_write_idx].copy_(prefix)
        v = torch.cat((blocks[:, :num_blocks].float(), prefix.float().unsqueeze(1)), dim=1)
        k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        probs = (k * (norm_weight.float() * qk_weight.float())).sum(-1).softmax(-1)
        mixed = torch.einsum("tm,tmd->td", probs, v)
        normed = mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + output_norm_eps)
        out.copy_(normed * output_norm_weight.float())
        return out

    return types.SimpleNamespace(kimi_k3_attn_res=kimi_k3_attn_res)


def _serve_with(monkeypatch, calls):
    """The model's reads take the stand-in wherever the stack fits FlashInfer's programs."""
    fake = _fake_flashinfer(calls)
    monkeypatch.setattr(attn_res, "_fi", lambda device: fake)
    monkeypatch.setattr(attn_res, "fi_serves",
                        lambda stack, prefix, weights, addend=None: stack.shape[1] <= attn_res.FI_MAX_BLOCKS)


def test_a_read_site_over_the_token_major_stack_is_the_read_its_write_and_its_norm():
    """``attn_res_read_norm`` without FlashInfer: the block-major read over ``stack.transpose(0, 1)``, the stream
    written into ``stack[:, m]`` at a block start, then the norm."""
    g = torch.Generator().manual_seed(4)
    t, d, blocks = 5, 32, 4
    stack = torch.randn(t, blocks, d, generator=g)
    stack[:, 2:] = float("nan")  # past m: never read
    prefix, addend = torch.randn(t, d, generator=g), torch.randn(t, d, generator=g)
    norm_w, proj_w = 1 + 0.1 * torch.randn(d, generator=g), torch.randn(1, d, generator=g)
    norm = RMSNorm(d, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(d, generator=g))
        mixed, stream = attn_res_read(stack.transpose(0, 1).clone(), 2, prefix, norm_w, proj_w, 1e-5, addend=addend)
        got, got_stream = attn_res_read_norm(stack, 2, prefix, norm_w, proj_w, 1e-5, norm, addend=addend, write=2)
    torch.testing.assert_close(got_stream, prefix + addend, atol=0, rtol=0)
    torch.testing.assert_close(got, norm(mixed), atol=0, rtol=0)
    torch.testing.assert_close(stack[:, 2], stream, atol=0, rtol=0)
    assert torch.isnan(stack[:, 3]).all()


def test_flashinfers_call_adds_in_place_writes_the_block_and_norms(monkeypatch):
    """Where FlashInfer serves, one call per read site: the stream is ``prefix`` with the addend added in place,
    the block start's write and the output norm (weight and epsilon) go to it, PDL on."""
    calls = []
    fake = _fake_flashinfer(calls)
    g = torch.Generator().manual_seed(5)
    t, d = 3, 16
    stack = torch.randn(t, 8, d, generator=g)
    prefix, addend = torch.randn(t, d, generator=g), torch.randn(t, d, generator=g)
    norm = RMSNorm(d, eps=3e-6)
    want_stream = prefix + addend
    monkeypatch.setattr(attn_res, "_fi", lambda device: fake)
    monkeypatch.setattr(attn_res, "fi_serves", lambda *a, **k: True)
    with torch.no_grad():
        out, stream = attn_res_read_norm(stack, 4, prefix, torch.ones(d), torch.ones(1, d), 1e-5, norm,
                                         addend=addend, write=4)
    assert stream is prefix and torch.equal(prefix, want_stream) and torch.equal(stack[:, 4], want_stream)
    assert calls == [{"m": 4, "write": 4, "delta": True, "pdl": True, "blocks": 8, "eps": (1e-5, 3e-6)}]
    assert out.shape == (t, d)


def test_the_model_on_flashinfers_reads_matches_the_reference(monkeypatch):
    """Prefill and four decodes through the engine with every read site on FlashInfer's call (in place): the logits
    match the reference's, so the stream FlashInfer adds into is always one the forward owns. One call per read
    site, the block starts writing their block, PDL on."""
    calls = []
    _serve_with(monkeypatch, calls)
    engine, _, model, sd = H.build_engine()
    cfg = model.config
    _greedy(engine, sd, cfg, PROMPT, steps=4)
    layers, starts = cfg.num_hidden_layers, sum(cfg.is_attn_res_block_start(i) for i in range(cfg.num_hidden_layers))
    steps = 5
    assert len(calls) == steps * (2 * layers + 1)
    assert all(c["pdl"] for c in calls)
    assert sum(c["write"] >= 0 for c in calls) == steps * starts
    # the first read of a step has no addend (the embedding is the stream) and writes block 0
    assert calls[0] == {**calls[0], "m": 0, "write": 0, "delta": False}


def test_attn_res_backend_triton_never_calls_flashinfer(monkeypatch):
    calls = []
    _serve_with(monkeypatch, calls)
    engine, _, model, sd = H.build_engine(attn_res_backend="triton")
    assert model.config.attn_res_backend == "triton"
    _greedy(engine, sd, model.config, PROMPT, steps=1)
    assert calls == []


def test_attn_res_backend_takes_auto_or_triton():
    from mstar.model.kimi_k3.config import KimiK3ModelConfig

    assert KimiK3ModelConfig.reduced().attn_res_backend == "auto"
    with pytest.raises(ValueError, match="attn_res_backend"):
        KimiK3ModelConfig.reduced(attn_res_backend="flashinfer")


def test_a_call_flashinfer_refuses_falls_back_to_the_triton_read(monkeypatch):
    """A call FlashInfer refuses on the host (no program for it, a precondition) runs the Triton read, the write and
    the norm instead, and isn't offered to FlashInfer again; an addend it can't take (not bf16) never is."""
    refused = []

    def kimi_k3_attn_res(*args, **kwargs):
        refused.append(kwargs["num_blocks"])
        raise NotImplementedError("no program registered")

    monkeypatch.setattr(attn_res, "_fi", lambda device: types.SimpleNamespace(kimi_k3_attn_res=kimi_k3_attn_res))
    # FlashInfer's dtype rule for the addend kept, the rest waived (a CPU tensor)
    monkeypatch.setattr(attn_res, "fi_serves",
                        lambda stack, prefix, weights, addend=None: addend is None or addend.dtype == torch.bfloat16)
    monkeypatch.setattr(attn_res, "_FI_REFUSED", set())
    g = torch.Generator().manual_seed(6)
    t, d = 4, 16
    norm = RMSNorm(d, eps=1e-5)
    norm_w, proj_w = 1 + 0.1 * torch.randn(d, generator=g), torch.randn(1, d, generator=g)
    with torch.no_grad():
        for _ in range(2):
            stack = torch.randn(t, 8, d, generator=g)
            prefix = torch.randn(t, d, generator=g).bfloat16()
            addend = torch.randn(t, d, generator=g).bfloat16()
            want = attn_res_read_norm(stack.clone(), 3, prefix.clone(), norm_w, proj_w, 1e-5, norm, addend=addend,
                                      write=3, flashinfer=False)
            got = attn_res_read_norm(stack, 3, prefix, norm_w, proj_w, 1e-5, norm, addend=addend, write=3)
            assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))
            assert torch.equal(stack[:, 3], want[1])
        assert refused == [3]  # offered once, then the Triton read for that call shape
        attn_res_read_norm(torch.randn(t, 8, d), 3, prefix, norm_w, proj_w, 1e-5, norm, addend=addend.float())
    assert refused == [3]


def test_a_flashinfer_that_cant_build_its_programs_is_off_for_the_process(monkeypatch):
    """B300 slot 10: flashinfer 0.7.2's JIT build fails (the wheel ships no generated sources). The first failing
    call runs the Triton read, the write and the norm, and FlashInfer is off for the device from then on (one
    warning); a CUDA error is not swallowed."""
    tried = []

    def kimi_k3_attn_res(*args, **kwargs):
        tried.append(kwargs["num_blocks"])
        raise RuntimeError("Ninja build failed. Ninja output: ninja: error: 'cake_kimi_k3_attn_res_x.cu' missing")

    fake = types.SimpleNamespace(kimi_k3_attn_res=kimi_k3_attn_res)
    monkeypatch.setattr(attn_res, "_FI", {torch.device("cpu"): fake})
    monkeypatch.setattr(attn_res, "_FI_REFUSED", set())
    # FlashInfer's shape rules waived (a CPU tensor); whether it serves still asks _fi
    monkeypatch.setattr(attn_res, "fi_serves", lambda stack, prefix, weights, addend=None: attn_res._fi(
        prefix.device) is not None)
    g = torch.Generator().manual_seed(7)
    t, d = 3, 16
    norm = RMSNorm(d, eps=1e-5)
    norm_w, proj_w = 1 + 0.1 * torch.randn(d, generator=g), torch.randn(1, d, generator=g)
    with torch.no_grad():
        for m in (2, 5):
            stack, prefix, addend = torch.randn(t, 8, d, generator=g), torch.randn(t, d, generator=g), torch.randn(
                t, d, generator=g)
            want = attn_res_read_norm(stack.clone(), m, prefix.clone(), norm_w, proj_w, 1e-5, norm, addend=addend,
                                      write=m, flashinfer=False)
            got = attn_res_read_norm(stack, m, prefix, norm_w, proj_w, 1e-5, norm, addend=addend, write=m)
            assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))
    assert tried == [2] and attn_res._FI[torch.device("cpu")] is None

    def faults(*args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(attn_res, "_FI", {torch.device("cpu"): types.SimpleNamespace(kimi_k3_attn_res=faults)})
    with pytest.raises(RuntimeError, match="CUDA error"), torch.no_grad():
        attn_res_read_norm(torch.randn(t, 8, d), 1, torch.randn(t, d), norm_w, proj_w, 1e-5, norm,
                           addend=torch.randn(t, d))
