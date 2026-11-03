"""The persistent decode kernels (mstar/kernels/megakernel): which devices and batches take them, and on a
GPU each kernel against an fp32 reference of the ops it fuses, at Qwen3.5-0.8B's shapes with random weights.

The references round where the captured graph rounds (each kernel's bf16 output), so the tolerances are a
few bf16 ulps of the outputs' scale. Each kernel also runs twice from the same inputs: the second launch
must agree with the first, which it can only do if the first left its counters and accumulators zeroed.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from mstar.kernels import megakernel as mk_mod
from mstar.kernels.megakernel import megakernel_applies

_CUDA = torch.device("cuda")


@pytest.mark.parametrize(
    "capability, rows, expected",
    [((10, 3), 1, True), ((10, 0), 2, True), ((10, 3), 3, False), ((10, 3), 4, False), ((10, 3), 0, False),
     ((9, 0), 1, False), ((12, 0), 1, False), (None, 1, False)],
)
def test_guard_is_blackwell_and_small_batches_only(monkeypatch, capability, rows, expected):
    monkeypatch.setattr(mk_mod, "current_platform", SimpleNamespace(cuda_capability=lambda device=None: capability))
    assert megakernel_applies(_CUDA, rows) is expected


def test_guard_refuses_cpu():
    assert not megakernel_applies(torch.device("cpu"), 1)


def _needs_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("needs an sm_90+ GPU (acquire loads, the kernels' counters)")


H, I, EPS = 1024, 3584, 1e-6
HV, DK, DV = 16, 128, 128
CONV = 2 * HV * DK + HV * DV
NIN = CONV + HV * DV + 2 * HV
QH, KVH, D, ROT, PAGE = 8, 2, 256, 64, 128


def _bf(t):
    return t.bfloat16().float()


def _norm(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1 + w.float())


def _mlp_ref(h, res, w_post, w_gu, w_dt, w_next):
    r1 = h.float() + res.float()
    x = _bf(_norm(r1, w_post))
    gu = _bf(x @ w_gu.float().t())
    a = _bf(F.silu(gu[:, :I]) * gu[:, I:])
    r2 = _bf(r1) + _bf(a @ w_dt.float())
    return _norm(r2, w_next).bfloat16(), r2.bfloat16()


def _gdn_ref(x, w_in, conv_w, conv, slots, a_log, dt_bias, state, norm_w, w_out):
    """One layer; updates ``conv`` and ``state`` in place, as the kernel does."""
    s = slots.long()
    y = _bf(x.float() @ w_in.float().t())
    win = torch.cat([conv[s].float(), y[:, :CONV, None]], -1)
    act = _bf(F.silu((win * conv_w.float()[None]).sum(-1)))
    conv[s] = win[:, :, 1:].bfloat16()
    q = act[:, :HV * DK].view(-1, HV, DK)
    k = act[:, HV * DK:2 * HV * DK].view(-1, HV, DK)
    v = act[:, 2 * HV * DK:].view(-1, HV, DV)
    z = y[:, CONV:CONV + HV * DV].view(-1, HV, DV)
    a, b = y[:, NIN - 2 * HV:NIN - HV], y[:, NIN - HV:]
    g = torch.exp(-torch.exp(a_log) * F.softplus(a + dt_bias))
    beta = torch.sigmoid(b)
    q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * DK ** -0.5
    k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    S = state[s] * g[:, :, None, None]
    vnew = (v - (S @ k[..., None])[..., 0]) * beta[..., None]
    S = S + vnew[..., None] * k[:, :, None, :]
    state[s] = S
    core = _bf((S @ q[..., None])[..., 0])
    xn = _bf(core * torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + EPS))
    go = _bf(norm_w * xn * F.silu(z))
    return (go.flatten(1) @ w_out.float().t()).bfloat16()


def _rope(t, cos, sin):
    h = ROT // 2
    rot = torch.cat([-t[..., h:ROT], t[..., :h]], -1)
    return torch.cat([t[..., :ROT] * cos + rot * sin, t[..., ROT:]], -1)


def _attn_ref(x, w_qkv, qn, kn, cos, sin, kv, table, lens, w_o):
    """One layer; writes the new K/V into ``kv`` as the kernel does."""
    y = _bf(x.float() @ w_qkv.float().t())
    qg = y[:, :QH * 2 * D].view(-1, QH, 2 * D)
    q, gate = qg[..., :D], qg[..., D:]
    k = y[:, QH * 2 * D:QH * 2 * D + KVH * D].view(-1, KVH, D)
    v = y[:, QH * 2 * D + KVH * D:].view(-1, KVH, D)
    c, s = cos.float()[:, None], sin.float()[:, None]
    q = _bf(_rope(_bf(_norm(q, qn)), c, s))
    k = _bf(_rope(_bf(_norm(k, kn)), c, s))
    outs = []
    for b in range(x.shape[0]):
        L = int(lens[b])
        page, slot = int(table[b, (L - 1) // PAGE]), (L - 1) % PAGE
        kv[page, 0, slot] = k[b].bfloat16()
        kv[page, 1, slot] = v[b].bfloat16()
        pos = torch.arange(L, device=x.device)
        pages = table[b, pos // PAGE].long()
        K, V = kv[pages, 0, pos % PAGE].float(), kv[pages, 1, pos % PAGE].float()  # [L, KVH, D]
        qb = q[b].view(KVH, QH // KVH, D)
        att = torch.softmax(torch.einsum("gqd,lgd->gql", qb, K) * D ** -0.5, -1)
        outs.append(_bf(torch.einsum("gql,lgd->gqd", att, V)).reshape(QH, D))
    o = torch.stack(outs)
    go = _bf(o * torch.sigmoid(gate))
    return (go.flatten(1) @ w_o.float().t()).bfloat16()


def _rand(*shape, scale=0.02):
    return (torch.randn(*shape, device=_CUDA) * scale).bfloat16()


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("rows", [1, 2, 4])
def test_mlp_kernel_matches_reference(rows):
    _needs_sm90()
    from mstar.kernels.megakernel.mlp import MlpMegakernel

    torch.manual_seed(0)
    w_gu, w_dt = _rand(2 * I, H), _rand(I, H)
    w_post, w_next = _rand(H, scale=0.1), _rand(H, scale=0.1)
    h0, r0 = _rand(rows, H, scale=1.0), _rand(rows, H, scale=1.0)
    ref_h, ref_r = _mlp_ref(h0, r0, w_post, w_gu, w_dt, w_next)
    mk = MlpMegakernel(H, I)
    for _ in range(2):
        h, r = h0.clone(), r0.clone()
        mk(h, r, w_post, w_gu, w_dt, w_next, EPS)
        torch.cuda.synchronize()
        assert _rel(h, ref_h) < 1e-2 and _rel(r, ref_r) < 1e-2
    assert int(mk.cnt.item()) == 0 and not mk.acc.any()


def _gdn_weights():
    return dict(w_in=_rand(NIN, H), conv_w=_rand(CONV, 4, scale=0.3),
                a_log=torch.rand(HV, device=_CUDA) - 0.5, dt_bias=torch.rand(HV, device=_CUDA) - 0.5,
                norm_w=torch.rand(DV, device=_CUDA) + 0.5, w_out=_rand(H, HV * DV))


@pytest.mark.parametrize("rows", [1, 2])
def test_gdn_kernel_matches_reference(rows):
    _needs_sm90()
    from mstar.kernels.megakernel.gdn import GdnMegakernel

    torch.manual_seed(0)
    w = _gdn_weights()
    slots = torch.arange(1, rows + 1, dtype=torch.int32, device=_CUDA) * 2
    conv0 = _rand(8, CONV, 3, scale=0.5)
    state0 = torch.randn(8, HV, DV, DK, device=_CUDA) * 0.02
    x = _rand(rows, H, scale=1.0)
    conv_r, state_r = conv0.clone(), state0.clone()
    ref = _gdn_ref(x, w["w_in"], w["conv_w"], conv_r, slots, w["a_log"], w["dt_bias"], state_r, w["norm_w"], w["w_out"])
    mk = GdnMegakernel(H, HV, DK, DV, HV, NIN)
    for _ in range(2):
        conv, state = conv0.clone(), state0.clone()
        out = torch.empty(rows, H, device=_CUDA, dtype=torch.bfloat16)
        mk(x, w["w_in"], w["conv_w"], conv, slots, w["a_log"], w["dt_bias"], state, w["norm_w"], w["w_out"], out, EPS)
        torch.cuda.synchronize()
        assert _rel(out, ref) < 2e-2
        # the two older taps shift exactly; the new one is this step's projection, a bf16 ulp apart
        assert torch.equal(conv[..., :2], conv_r[..., :2])
        assert (conv.float() - conv_r.float()).abs().max().item() <= 0.0625
        assert (state - state_r).abs().max().item() < 1e-3
    untouched = torch.ones(8, dtype=torch.bool, device=_CUDA)
    untouched[slots.long()] = False
    assert torch.equal(state[untouched], state0[untouched])


def _attn_setup(rows, L):
    pages_per_row = -(-L // PAGE) + 1
    kv = _rand(1 + rows * pages_per_row, 2, PAGE, KVH, D, scale=1.0)
    perm = torch.randperm(rows * pages_per_row, device=_CUDA).int() + 1
    table = perm.view(rows, pages_per_row).contiguous()
    lens = torch.tensor([L - 7 * b for b in range(rows)], dtype=torch.int32, device=_CUDA)
    pos = (lens - 1).float()[:, None]
    inv = 1.0 / (1e7 ** (torch.arange(0, ROT, 2, device=_CUDA).float() / ROT))
    ang = torch.cat([pos * inv, pos * inv], -1)
    return kv, table, lens, ang.cos().bfloat16(), ang.sin().bfloat16()


@pytest.mark.parametrize("rows, L", [(1, 300), (2, 1029)])
def test_attention_kernel_matches_reference(rows, L):
    _needs_sm90()
    from mstar.kernels.megakernel.attn import AttnMegakernel

    torch.manual_seed(0)
    w_qkv, w_o = _rand(QH * 2 * D + 2 * KVH * D, H), _rand(H, QH * D)
    qn, kn = _rand(D, scale=0.1), _rand(D, scale=0.1)
    kv0, table, lens, cos, sin = _attn_setup(rows, L)
    x = _rand(rows, H, scale=1.0)
    kv_r = kv0.clone()
    ref = _attn_ref(x, w_qkv, qn, kn, cos, sin, kv_r, table, lens, w_o)
    mk = AttnMegakernel(H, QH, KVH, D, ROT, PAGE, max_len=2048)
    for _ in range(2):
        kv = kv0.clone()
        out = torch.empty(rows, H, device=_CUDA, dtype=torch.bfloat16)
        mk(x, w_qkv, qn, kn, cos, sin, kv, table, lens, w_o, out, EPS)
        torch.cuda.synchronize()
        assert _rel(out, ref) < 2e-2
        assert (kv.float() - kv_r.float()).abs().max().item() <= 0.0625


def _block_modules():
    """Stand-ins with the attributes BlockMegakernel reads off the model's modules."""
    ns = SimpleNamespace
    gdn = []
    for _ in range(3):
        w = _gdn_weights()
        gdn.append(ns(num_v_heads=HV, num_k_heads=HV, head_k_dim=DK, head_v_dim=DV, key_dim=HV * DK,
                      value_dim=HV * DV, conv_dim=CONV, in_proj_fused=ns(weight=w["w_in"]),
                      conv1d=ns(weight=w["conv_w"][:, None]), A_log=w["a_log"], dt_bias=w["dt_bias"],
                      norm=ns(weight=w["norm_w"]), out_proj=ns(weight=w["w_out"])))
    attn = ns(num_heads=QH, num_kv_heads=KVH, head_dim=D, qkv_proj=ns(weight=_rand(QH * 2 * D + 2 * KVH * D, H)),
              q_norm=ns(weight=_rand(D, scale=0.1)), k_norm=ns(weight=_rand(D, scale=0.1)),
              o_proj=ns(weight=_rand(H, QH * D)))
    mlps = [ns(gate_up_proj=ns(weight=_rand(2 * I, H)), down_proj=ns(weight=_rand(H, I))) for _ in range(4)]
    post = [ns(weight=_rand(H, scale=0.1)) for _ in range(4)]
    nxt = [ns(weight=_rand(H, scale=0.1)) for _ in range(4)]
    return gdn, attn, mlps, post, nxt


@pytest.mark.parametrize("rows", [1, 2])
def test_block_kernel_matches_the_stage_kernels(rows):
    """The block is the stage kernels' phases in one launch: same inputs, same outputs and writes."""
    _needs_sm90()
    from mstar.kernels.megakernel.attn import AttnMegakernel
    from mstar.kernels.megakernel.block import BlockMegakernel
    from mstar.kernels.megakernel.gdn import GdnMegakernel
    from mstar.kernels.megakernel.mlp import MlpMegakernel

    torch.manual_seed(0)
    cfg = {"hidden_size": H, "intermediate_size": I}
    gdn, attn, mlps, post, nxt = _block_modules()
    slots = torch.arange(1, rows + 1, dtype=torch.int32, device=_CUDA)
    conv0 = _rand(3, 8, CONV, 3, scale=0.5)
    state0 = torch.randn(3, 8, HV, DV, DK, device=_CUDA) * 0.02
    kv0, table, lens, cos, sin = _attn_setup(rows, 600)
    res0 = _rand(rows, H, scale=1.0)
    h0 = _norm(res0.float(), _rand(H, scale=0.1)).bfloat16()

    # the stage kernels, layer by layer
    conv, state, kv = conv0.clone(), state0.clone(), kv0.clone()
    g_mk, m_mk = GdnMegakernel(H, HV, DK, DV, HV, NIN), MlpMegakernel(H, I)
    a_mk = AttnMegakernel(H, QH, KVH, D, ROT, PAGE, 2048)
    x, res = h0.clone(), res0.clone()
    for i in range(4):
        out = torch.empty_like(x)
        if i < 3:
            m = gdn[i]
            g_mk(x, m.in_proj_fused.weight, m.conv1d.weight[:, 0], conv[i], slots, m.A_log, m.dt_bias, state[i],
                 m.norm.weight, m.out_proj.weight, out, EPS)
        else:
            a_mk(x, attn.qkv_proj.weight, attn.q_norm.weight, attn.k_norm.weight, cos, sin, kv, table, lens,
                 attn.o_proj.weight, out, EPS)
        m_mk(out, res, post[i].weight, mlps[i].gate_up_proj.weight, mlps[i].down_proj.weight.t().contiguous(),
             nxt[i].weight, EPS)
        x = out
    want = (x, res, conv, state, kv)

    blk = BlockMegakernel(cfg, gdn, attn, mlps, post, nxt, max_len=2048)
    for _ in range(2):
        conv, state, kv = conv0.clone(), state0.clone(), kv0.clone()
        h_out, r_out = torch.empty_like(h0), torch.empty_like(res0)
        blk(h0, res0, r_out, h_out, conv, state, 0, slots, cos, sin, kv, table, lens, EPS)
        torch.cuda.synchronize()
        assert _rel(h_out, want[0]) < 1e-2 and _rel(r_out, want[1]) < 1e-2
        assert (conv.float() - want[2].float()).abs().max().item() <= 0.0625
        assert (state - want[3]).abs().max().item() < 1e-3
        assert (kv.float() - want[4].float()).abs().max().item() <= 0.0625
    assert not blk.cnt.any() and not blk.scratch.acc_a.any() and not blk.scratch.acc_m.any()
