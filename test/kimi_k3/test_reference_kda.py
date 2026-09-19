"""KDA reference parity (CPU): against fla's naive recurrence and against the HF
``KimiDeltaAttention`` module run with torch stand-ins for the Triton kernels."""
import torch

from mstar.model.kimi_k3.reference.kda import (
    KDAWeights,
    from_v_first,
    kda_gate,
    kda_layer_forward,
    kda_recurrent,
    l2norm,
    patch_hf_modeling_for_cpu,
    short_conv,
)

torch.manual_seed(0)


def test_recurrence_matches_fla_naive():
    from fla.ops.kda.naive import naive_recurrent_kda
    b, t, h, d = 1, 11, 3, 16
    q, k = l2norm(torch.randn(t, h, d)), l2norm(torch.randn(t, h, d))
    v = torch.randn(t, h, d)
    g_log = -5.0 * torch.sigmoid(torch.randn(t, h, d))
    beta = torch.sigmoid(torch.randn(t, h))
    s0 = torch.randn(h, d, d)
    o_ref, s_ref = naive_recurrent_kda(q[None], k[None], v[None], g_log[None], beta[None],
                                       initial_state=s0[None], output_final_state=True)
    o, s = kda_recurrent(q, k, v, g_log, beta, s0)
    torch.testing.assert_close(o, o_ref[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(s, s_ref[0], rtol=1e-5, atol=1e-5)


def test_short_conv_matches_conv1d_and_streams():
    d, w, t = 6, 4, 9
    weight = torch.randn(d, 1, w)
    x = torch.randn(t, d)
    ref = torch.nn.functional.conv1d(x.t()[None], weight, padding=w - 1, groups=d)[0, :, :t].t()
    ref = torch.nn.functional.silu(ref)
    y, cache = short_conv(x, weight)
    torch.testing.assert_close(y, ref, rtol=1e-6, atol=1e-6)
    assert torch.equal(cache, x[-w:].t())
    # streaming: first 5 tokens, then 4 with the cache
    y1, c1 = short_conv(x[:5], weight)
    y2, c2 = short_conv(x[5:], weight, c1)
    torch.testing.assert_close(torch.cat([y1, y2]), ref, rtol=1e-6, atol=1e-6)
    assert torch.equal(c2, cache)
    # token-by-token decode
    ys, c = [], None
    for i in range(t):
        yi, c = short_conv(x[i : i + 1], weight, c)
        ys.append(yi)
    torch.testing.assert_close(torch.cat(ys), ref, rtol=1e-6, atol=1e-6)


def test_layer_matches_hf_and_decode_is_consistent(hf_modeling, hf_text_config):
    patch_hf_modeling_for_cpu(hf_modeling)
    m = hf_modeling.KimiDeltaAttention(hf_text_config, layer_idx=0).float().eval()
    for p in m.parameters():
        with torch.no_grad():
            p.normal_(std=0.1)
    with torch.no_grad():
        m.dt_bias.normal_(std=0.5)
        m.A_log.copy_(torch.log(torch.empty(m.num_heads).uniform_(1, 16)))
    t = 7
    x = torch.randn(1, t, hf_text_config.hidden_size)
    with torch.no_grad():
        ref = m(x)  # chunk mode, no cache
    w = KDAWeights.from_hf_module(m)
    out, st = kda_layer_forward(w, x[0])
    torch.testing.assert_close(out, ref[0], rtol=1e-4, atol=1e-4)
    # prefill T-1 then decode 1 token == full prefill's last row; states agree with the
    # HF cache produced by the stand-in kernels (V-first layout in the cache)
    cache = hf_modeling.KimiDynamicCache(hf_text_config)
    with torch.no_grad():
        ref_pre = m(x[:, :-1], cache_params=cache)
        ref_step = m(x[:, -1:], cache_params=cache)
    out_pre, st_pre = kda_layer_forward(w, x[0, :-1])
    out_step, st_full = kda_layer_forward(w, x[0, -1:], st_pre)
    torch.testing.assert_close(out_pre, ref_pre[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_step, ref_step[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_step, ref[0, -1:], rtol=1e-4, atol=1e-4)
    hf_state = cache.recurrent_states[0][0]  # [H, V, K]
    torch.testing.assert_close(from_v_first(hf_state), st_full.recurrent, rtol=1e-4, atol=1e-4)
    hf_conv_q = cache.conv_states[0][0][0]  # [D, W]
    torch.testing.assert_close(hf_conv_q, st_full.conv_q, rtol=1e-5, atol=1e-5)


def test_gate_bounds():
    g = kda_gate(torch.randn(4, 2, 8) * 10, torch.zeros(2), torch.zeros(16), -5.0)
    assert (g <= 0).all() and (g >= -5.0).all()
