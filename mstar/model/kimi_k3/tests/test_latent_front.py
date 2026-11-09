"""The latent MoE's front, CPU side: ``latent_moe_front: split`` (the router's GEMM forked to the side stream ahead of
the shared experts, the latent's on the main one) and ``flashinfer`` (the block's wiring with a torch stand-in for
FlashInfer's program: router logits, latent, shared SiTU activation from one call; the load's refusal where the
program can't run), and the config switch. FlashInfer's program runs on SM100 only (test_latent_front_b300.py)."""
import pytest
import torch
import torch.nn.functional as F

from mstar.engine import side_stream
from mstar.model.kimi_k3.components import latent_front
from mstar.model.kimi_k3.config import KimiK3ModelConfig
from mstar.model.kimi_k3.tests.test_components import _loaded_latent_moe


def _reference_launch(calls):
    """The program's contract in torch: fp32 logits, the latent and SiTU(gate, up) from the fp32 products."""
    def launch(x, weight, experts, latent, out_logits, out_latent, out_act):
        calls.append(x.shape[0])
        h = F.linear(x.float(), weight.float())
        i_local = (weight.shape[0] - experts - latent) // 2
        gate, up = h[:, experts + latent:experts + latent + i_local], h[:, experts + latent + i_local:]
        out_logits.copy_(h[:, :experts])
        out_latent.copy_(h[:, experts:experts + latent])
        out_act.copy_(4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * 25 * torch.tanh(up / 25))
    return launch


def _block(**cfg):
    block, _ = _loaded_latent_moe(KimiK3ModelConfig.reduced(**cfg))
    block.process_weights_after_loading(None)
    assert block._in_weight is not None
    return block


@pytest.mark.parametrize("kernel_routes", [False, True])
def test_the_split_front_forks_the_router_ahead_of_the_shared_experts(monkeypatch, kernel_routes):
    """With the shared overlap the router's GEMM is forked to the side stream first and the shared experts second,
    the latent's GEMM runs on the main stream, and the layer's output is the packed GEMM's (the logits go to an
    experts kernel that routes); past the overlap's rows, one stream and the packed GEMM, no fork."""
    from types import SimpleNamespace

    block = _block(latent_moe_front="split")
    seen = []
    if kernel_routes:
        def from_logits(z, logits):
            seen.append(logits)
            weights, selected, _ = block.gate.__class__.route(block.gate, logits)
            return block.experts_forward(z, weights, selected, reduce_results=False)

        block._experts_kernel = SimpleNamespace(routing=object(), from_logits=from_logits)
    x = torch.randn(3, block.model_hidden_size, generator=torch.Generator().manual_seed(7))
    block.latent_moe_front, block.moe_overlap = "gemm", "off"
    with torch.no_grad():
        want = block(x)
    block.latent_moe_front, block.moe_overlap = "split", "shared"
    forks, fork = [], side_stream.fork

    def recorded(fn, *args):
        forks.append(fn.__name__)
        return fork(fn, *args)

    monkeypatch.setattr(side_stream, "fork", recorded)
    with torch.no_grad():
        got = block(x)
    assert forks == ["_router_logits", "_shared_partial"]
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    if kernel_routes:
        torch.testing.assert_close(seen[-1], F.linear(x, block.gate.weight), atol=1e-5, rtol=1e-5)
    block.shared_experts_overlap_max_tokens = 2
    with torch.no_grad():
        block(x)
    assert forks == ["_router_logits", "_shared_partial"], "past the overlap's rows: the packed GEMM, no fork"


def test_the_front_serves_the_block_in_one_call_on_one_stream(monkeypatch):
    """Up to MAX_TOKENS rows the block takes the router's logits, the latent and the shared activation from one
    front call, on one stream (the shared experts' activation is the front's), and the layer's output is the
    packed GEMM path's; past them, the packed GEMMs as before."""
    block = _block()
    calls = []
    monkeypatch.setattr(latent_front, "_launch", _reference_launch(calls))
    monkeypatch.setattr(latent_front, "MAX_TOKENS", 4)
    g = torch.Generator().manual_seed(5)
    for rows in (1, 3, 4):
        x = torch.randn(rows, block.model_hidden_size, generator=g)
        block._front = False
        block.moe_overlap = "off"
        with torch.no_grad():
            want = block(x)
        block._front = True
        block.moe_overlap = "shared"  # the front's steps run on one stream whatever the overlap

        def no_side_stream(*a, **k):
            raise AssertionError("a front step forked the side stream")

        monkeypatch.setattr(side_stream, "fork", no_side_stream)
        with torch.no_grad():
            got = block(x)
        monkeypatch.undo()
        monkeypatch.setattr(latent_front, "_launch", _reference_launch(calls))
        monkeypatch.setattr(latent_front, "MAX_TOKENS", 4)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    assert calls == [1, 3, 4]
    x = torch.randn(5, block.model_hidden_size, generator=g)
    with torch.no_grad():
        block(x)
    assert calls == [1, 3, 4], "past MAX_TOKENS rows the packed GEMM runs"


def test_a_kernel_routed_front_hands_its_logits_to_the_experts_kernel(monkeypatch):
    """With the experts kernel routing from the logits, the front's fp32 logits go to ``from_logits`` and the
    block's router never runs, as on the packed GEMM path."""
    from types import SimpleNamespace

    block = _block()
    monkeypatch.setattr(latent_front, "_launch", _reference_launch([]))
    x = torch.randn(3, block.model_hidden_size, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        want = block(x)
    seen = []

    def from_logits(z, logits):
        seen.append(logits)
        weights, selected, _ = block.gate.__class__.route(block.gate, logits)
        return block.experts_forward(z, weights, selected, reduce_results=False)

    def no_router(*a, **k):
        raise AssertionError("the block's router ran")

    block._experts_kernel = SimpleNamespace(routing=object(), from_logits=from_logits)
    monkeypatch.setattr(block.gate, "route", no_router)
    block._front = True
    with torch.no_grad():
        got = block(x)
    (logits,) = seen
    torch.testing.assert_close(logits, F.linear(x, block.gate.weight), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_the_load_refuses_a_front_the_program_cant_run():
    """``latent_moe_front: flashinfer`` on a block the program can't serve (here: a CPU fp32 reduced block) stops
    the load naming why; ``gemm`` (the default) never asks."""
    block = _block()
    block._setup_front()
    assert not block._front
    block = _block(latent_moe_front="flashinfer")
    with pytest.raises(RuntimeError, match="latent_moe_front: flashinfer can't serve"):
        block._setup_front()
    assert not block._front


def test_unavailable_names_each_mismatch():
    from mstar.utils.situ import SiTU

    rows = latent_front.EXPERTS + latent_front.LATENT + 2 * latent_front.SHARED // 8
    situ = SiTU(beta=4.0, linear_beta=25.0)
    w = torch.empty(rows, latent_front.HIDDEN, dtype=torch.bfloat16)
    assert "CUDA" in latent_front.unavailable(w, 896, 3584, situ, 8)
    assert "no packed input weight" in latent_front.unavailable(None, 896, 3584, situ, 8)
    if torch.cuda.is_available():
        w = w.cuda()
        assert "geometry" in latent_front.unavailable(w, 896, 3584, situ, 4)
        assert "SiTU" in latent_front.unavailable(w, 896, 3584, SiTU(beta=1.0), 8)
        assert "rows" in latent_front.unavailable(w[:-64], 896, 3584, situ, 8)


def test_latent_moe_front_is_a_serving_switch():
    assert KimiK3ModelConfig.reduced().latent_moe_front == "gemm"
    assert KimiK3ModelConfig.reduced(latent_moe_front="flashinfer").latent_moe_front == "flashinfer"
    assert KimiK3ModelConfig.reduced(latent_moe_front="split").latent_moe_front == "split"
    with pytest.raises(ValueError, match="latent_moe_front"):
        KimiK3ModelConfig.reduced(latent_moe_front="cake")
