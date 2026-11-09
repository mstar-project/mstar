"""FlashInfer's Kimi-K3 LatentMoE front (``latent_front``) on a B300 / B200 (SM100 / SM103, 148 SMs), at K3.1's
TP8 rank geometry (router 896, latent 3584, shared gate_up 2 x 768, hidden 7168): its three outputs against torch
in fp32 at every decode bucket the program pads to, and a CUDA graph replay against eager. Skips elsewhere (H200
has no program).

    pytest mstar/model/kimi_k3/tests/test_latent_front_b300.py -q -rA
"""
import pytest
import torch
import torch.nn.functional as F

from mstar.model.kimi_k3.components import latent_front as LF


def _available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from flashinfer.experimental.kimi_k3_latent_moe import cake_backend

        return all(cake_backend.generated_program_available(torch.device("cuda"), "front", 8, n)
                   for n in LF.WARM_TOKENS) and not LF.missing_sources()
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _available(), reason="FlashInfer's Kimi-K3 front runs on SM100 / SM103 only, "
                                "with its sources installed")
I_LOCAL = LF.SHARED // 8


@pytest.fixture(scope="module")
def weight():
    g = torch.Generator(device="cuda").manual_seed(0)
    rows = LF.EXPERTS + LF.LATENT + 2 * I_LOCAL
    w = (torch.randn(rows, LF.HIDDEN, device="cuda", generator=g) * 0.02).bfloat16()
    LF.warm(w)
    return w


def _reference(x, w):
    h = F.linear(x.float(), w.float())
    e, lat = LF.EXPERTS, LF.LATENT
    gate, up = h[:, e + lat:e + lat + I_LOCAL], h[:, e + lat + I_LOCAL:]
    act = 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * 25 * torch.tanh(up / 25)
    return h[:, :e], h[:, e:e + lat], act


def _close(got, want, rel, name):
    err = (got.float() - want).abs()
    bound = rel * want.abs() + rel * want.abs().mean()
    bad = err > bound
    assert not bool(bad.any()), f"{name}: {int(bad.sum())} of {err.numel()} off, max err {err.max().item():.3g}"


@pytest.mark.parametrize("rows", [1, 2, 7, 8, 16, 33, 64, 128])
def test_the_front_matches_torch(weight, rows):
    """Logits within fp32 summation order, the latent and the activation within a bf16 rounding (plus the
    program's tanh / sigmoid), at each padded row count's programs and an unpadded count."""
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, LF.HIDDEN, device="cuda", generator=g).bfloat16()
    logits, latent, act = torch.ops.mstar.k31_latent_front(x, weight, LF.EXPERTS, LF.LATENT)
    torch.cuda.synchronize()
    want_logits, want_latent, want_act = _reference(x, weight)
    assert logits.dtype == torch.float32 and latent.dtype == act.dtype == torch.bfloat16
    assert act.shape == (rows, I_LOCAL)
    _close(logits, want_logits, 1e-3, "logits")
    _close(latent, want_latent, 2 ** -7, "latent")
    _close(act, want_act, 2 ** -6, "act")


def test_a_captured_front_replays_new_inputs_as_eager(weight):
    """Captured once, replayed with new rows written into its input: eager's outputs, bit for bit."""
    x = torch.zeros(8, LF.HIDDEN, device="cuda", dtype=torch.bfloat16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.ops.mstar.k31_latent_front(x, weight, LF.EXPERTS, LF.LATENT)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = torch.ops.mstar.k31_latent_front(x, weight, LF.EXPERTS, LF.LATENT)
    g = torch.Generator(device="cuda").manual_seed(9)
    for _ in range(3):
        x.copy_(torch.randn(8, LF.HIDDEN, device="cuda", generator=g).bfloat16())
        graph.replay()
        want = torch.ops.mstar.k31_latent_front(x.clone(), weight, LF.EXPERTS, LF.LATENT)
        torch.cuda.synchronize()
        for a, b in zip(outs, want, strict=True):
            assert torch.equal(a, b)
