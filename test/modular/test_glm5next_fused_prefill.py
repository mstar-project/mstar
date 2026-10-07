"""Fused KDA prefill (kda_triton) vs the torch chunk_kda reference — parity on CUDA.

The oracle is ``Glm5NextLinearAttention.prefill`` run per span in fp32 against in-place slot
views, as the reference bundle does. The fused path must be no less accurate than that
reference under the engine's bf16 autocast.
"""
from dataclasses import dataclass

import pytest
import torch

from mstar.engine.resources.linear_attn import kda_triton
from mstar.engine.resources.linear_attn.kda import KDAPlan
from mstar.model.glm5_next import fused_decode
from mstar.model.glm5_next.components.attention import Glm5NextKdaAttention
from mstar.model.glm5_next.kda import Glm5NextKdaConfig, TorchKDAKernels

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and fused_decode._HAS_TRITON),
    reason="fused kernels need CUDA + triton",
)

HEADS, HEAD_DIM, HIDDEN, SLOTS = 8, 128, 4096, 12


@pytest.fixture
def fp32_matmul():
    """mstar.engine sets TF32 matmuls globally; the oracle runs in true fp32."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(prev)


@dataclass
class Span:
    """One row of a step: where its tokens sit and which slot holds its state."""
    slot: int
    q_start: int
    q_len: int


class _Pool:
    """One layer's pool blocks, as ``RecurrentStatePool.block`` hands them out."""

    def __init__(self, state, conv):
        self.blocks = {"state": state, "conv": conv}

    def block(self, name, layer):
        return self.blocks[name]


class _Kda:
    """The KDA resource's ``run`` on a hand-built plan."""

    def __init__(self, plan, kernels):
        self.plan, self.kernels = plan, kernels

    def current_plan(self, label=None):
        return self.plan

    def run(self, qkv, g, beta, conv, state, params, gate=None, label=None, spec=None):
        return self.kernels.run_paged(qkv, g, beta, self.plan, conv, state, params, gate=gate)


def _plan(spans) -> KDAPlan:
    lens = [s.q_len for s in spans]
    slots = torch.tensor([s.slot for s in spans], dtype=torch.int32, device="cuda")
    return KDAPlan(
        slot_ids=slots, has_state=torch.ones_like(slots, dtype=torch.bool), spans=tuple(lens),
        num_tokens=sum(lens), is_decode=False,
        layout=torch.tensor(kda_triton.varlen_layout(lens), dtype=torch.int32, device="cuda"),
    )


def _kda(seed=0, head_dim=HEAD_DIM):
    """One TP8 rank's KDA layer with gates spread like a trained model's: per-channel decay
    from ~0 (long memory) to the lower bound (forgets within a token)."""
    torch.manual_seed(seed)
    kda = Glm5NextKdaAttention(Glm5NextKdaConfig(
        hidden_size=HIDDEN, linear_num_heads=HEADS, linear_head_dim=head_dim),
        dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        for name, p in kda.named_parameters():
            if "conv1d" in name:
                p.normal_(0.0, 0.3)
            elif name.endswith("A_log"):
                p.normal_(0.0, 0.5)
            elif name.endswith("dt_bias"):
                p.uniform_(-8.0, 2.0)
            elif "norm" in name:
                p.normal_(1.0, 0.1)
            else:
                p.normal_(0.0, 0.02)
    kda.process_weights_after_loading("cuda")
    return kda


def _spans(lengths, seed=0):
    slots = torch.randperm(SLOTS - 1, generator=torch.Generator().manual_seed(seed))[:len(lengths)] + 1
    spans, start = [], 0
    for n, slot in zip(lengths, slots.tolist(), strict=True):
        spans.append(Span(slot=slot, q_start=start, q_len=n))
        start += n
    return spans, start


def _pools(fresh, seed=1):
    torch.manual_seed(seed)
    rec = torch.randn(SLOTS, HEADS, HEAD_DIM, HEAD_DIM, device="cuda") * 0.1
    conv = torch.randn(SLOTS, 3 * HEADS * HEAD_DIM, 3, device="cuda").to(torch.bfloat16)
    if fresh:
        rec.zero_(), conv.zero_()
    return rec, conv


def _reference(kda, x, spans, rec, conv, autocast=False):
    """Per-span torch prefill against in-place slot views; the pool is V-first."""
    ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast)
    outs = []
    with torch.no_grad(), ctx:
        for s in spans:
            out, _, _ = kda.prefill(x[s.q_start:s.q_start + s.q_len].unsqueeze(0),
                                    recurrent_state=rec[s.slot].transpose(-1, -2).unsqueeze(0),
                                    conv_state=conv[s.slot].unsqueeze(0))
            outs.append(out.squeeze(0))
    return torch.cat(outs)


def _paged(kda, x, spans, rec, conv, kernels=None):
    """``x`` through the layer's paged path, the pool's blocks being ``rec`` and ``conv``."""
    kda.pool, kda.kda = _Pool(rec, conv), _Kda(_plan(spans), kernels or kda_triton.TritonKDAKernels())
    with torch.no_grad():
        return kda.forward_paged(x, 0)


def _fused(kda, x, spans, rec, conv):
    return _paged(kda, x, spans, rec, conv)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _check(fused, served, oracle, what, floor):
    """The fused result is at most 1.5x as far from the fp32 oracle as the served path,
    or within ``floor`` (relative Frobenius) when the served path is closer than that."""
    err, served_err = _rel(fused, oracle), _rel(served, oracle)
    assert err <= max(1.5 * served_err, floor), (
        f"{what}: fused rel err {err:.2e} vs served {served_err:.2e}")


@pytest.mark.parametrize("fresh", [True, False])
@pytest.mark.parametrize("lengths", [
    (1,), (3,), (64,), (65,), (300,), (2048,),
    (5, 64, 1, 130, 2, 700),  # mixed: below the conv width, chunk edges, multi-chunk
    (63, 127, 128, 129),  # either side of the chunk boundaries
])
def test_kda_prefill_matches_reference(lengths, fresh, fp32_matmul):
    spans, tokens = _spans(lengths)
    x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    _compare(_kda(), x, spans, fresh)


@pytest.mark.parametrize("case", ["identical_tokens", "gate_at_lower_bound", "gate_near_zero"])
def test_kda_prefill_stress(case, fp32_matmul):
    """Inputs at the edges of the math. Identical tokens make every key the same direction,
    the triangular solve's worst case; gates pinned at the lower bound give the largest
    decay factors the block split has to hold in fp32; gates near zero, the longest memory."""
    kda = _kda(seed=3)
    x = torch.randn(2048, HIDDEN, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        if case == "identical_tokens":
            x = x[:1].expand(2048, -1).contiguous()
        elif case == "gate_at_lower_bound":
            kda.forget_gate.dt_bias.fill_(30.0)
            kda.forget_gate.A_log.fill_(1.0)
        else:
            kda.forget_gate.dt_bias.fill_(-30.0)
    spans, _ = _spans((2048,))
    _compare(kda, x, spans, fresh=False)


def _compare(kda, x, spans, fresh):
    """The fused prefill against the fp32 oracle and the served path, on every pool slot."""
    rec, conv = _pools(fresh)
    rec_o, conv_o = rec.clone(), conv.clone()
    oracle = _reference(kda, x, spans, rec_o, conv_o)
    rec_s, conv_s = rec.clone(), conv.clone()
    served = _reference(kda, x, spans, rec_s, conv_s, autocast=True)

    out = _fused(kda, x, spans, rec, conv)

    _check(out, served, oracle, "output", floor=1e-2)
    touched = [s.slot for s in spans]
    _check(rec[touched], rec_s[touched], rec_o[touched], "recurrent state", floor=1e-2)
    # The conv tail is the raw pre-conv projection rows: exact up to the projection GEMM.
    torch.testing.assert_close(conv[touched], conv_o[touched], rtol=1e-2, atol=1e-2)
    untouched = [i for i in range(SLOTS) if i not in touched]
    assert torch.equal(rec[untouched], rec_o[untouched])
    assert torch.equal(conv[untouched], conv_o[untouched])


def test_kda_prefill_continue_then_decode(fp32_matmul):
    """Two fused prefills (100 + 200) and a fused decode step reproduce one 300-token
    reference prefill and a reference decode step: the prefill leaves the slot in the
    layout decode reads."""
    kda = _kda()
    x = torch.randn(301, HIDDEN, device="cuda", dtype=torch.bfloat16)
    rec, conv = _pools(fresh=True)
    rec_o, conv_o = rec.clone(), conv.clone()
    slot = 5
    whole, _ = _spans((300,))
    whole[0].slot = slot
    oracle = _reference(kda, x[:300], whole, rec_o, conv_o)
    with torch.no_grad():
        r, c = rec_o[[slot]].transpose(-1, -2), conv_o[[slot]]
        oracle_step = kda.decode_step(x[300:].unsqueeze(1), r, c).squeeze(1)
        rec_o[slot], conv_o[slot] = r[0].transpose(-1, -2), c[0]

    first = [Span(slot=slot, q_start=0, q_len=100)]
    second = [Span(slot=slot, q_start=0, q_len=200)]
    out = torch.cat([_fused(kda, x[:100], first, rec, conv),
                     _fused(kda, x[100:300], second, rec, conv)])
    slots = torch.tensor([slot], dtype=torch.int32, device="cuda")
    kda.kda.plan = KDAPlan(slot_ids=slots, has_state=torch.ones_like(slots, dtype=torch.bool),
                           spans=(1,), num_tokens=1, is_decode=True)
    with torch.no_grad():
        step = kda.forward_paged(x[300:], 0)

    assert _rel(out, oracle) < 1e-2
    assert _rel(step, oracle_step) < 2e-2
    assert _rel(rec[slot], rec_o[slot]) < 1e-2
    torch.testing.assert_close(conv[slot], conv_o[slot], rtol=1e-2, atol=1e-2)


def test_reference_bundle_matches_the_fused_prefill():
    """The torch bundle a model falls back to computes what the fused one does, from the
    same plan, at a head width the fused prefill takes."""
    assert kda_triton.supported(32)
    kda = _kda(seed=4, head_dim=32)
    spans = [Span(slot=2, q_start=0, q_len=70), Span(slot=1, q_start=70, q_len=9)]
    x = torch.randn(79, HIDDEN, device="cuda", dtype=torch.bfloat16)
    pools = [(torch.zeros(3, HEADS, 32, 32, device="cuda"),
              torch.zeros(3, 3 * HEADS * 32, 3, device="cuda", dtype=torch.bfloat16))
             for _ in range(2)]
    out = _paged(kda, x, spans, *pools[0])
    ref = _paged(kda, x, spans, *pools[1], kernels=TorchKDAKernels())
    assert _rel(out, ref) < 2e-2
    assert _rel(pools[0][0], pools[1][0]) < 2e-2
    torch.testing.assert_close(pools[0][1], pools[1][1], rtol=1e-2, atol=1e-2)
