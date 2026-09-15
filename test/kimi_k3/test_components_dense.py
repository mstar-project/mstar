"""The serving components (TP-capable modules, trivial comm group) loaded through the
M* streaming loader must reproduce the HF model on the tiny checkpoint (CPU, fp32)."""
import pytest
import torch

from mstar.model.kimi_k3.components.language_model import KimiK3ForCausalLM
from mstar.model.kimi_k3.config import KimiK3Config
from mstar.model.kimi_k3.reference.kda import patch_hf_modeling_for_cpu
from mstar.model.loader import load_weights


@pytest.fixture(scope="module")
def hf_model(tiny_dir, hf_modeling, hf_text_config):
    from safetensors.torch import load_file
    patch_hf_modeling_for_cpu(hf_modeling)
    model = hf_modeling.KimiLinearForCausalLM(hf_text_config).eval()
    model.config._attn_implementation = "eager"
    sd = load_file(str(tiny_dir / "model.safetensors"))
    lm = {k[len("language_model."):]: v for k, v in sd.items() if k.startswith("language_model.")}
    model.load_state_dict(lm, strict=False)
    return model.float()


@pytest.fixture(scope="module")
def mstar_model(tiny_dir):
    cfg = KimiK3Config.from_hf_dir(tiny_dir).text
    with torch.device("meta"):
        model = KimiK3ForCausalLM(cfg)
    model = model.to(torch.float32)
    model.to_empty(device="cpu")
    loaded = load_weights(model, tiny_dir, device="cpu")
    params = dict(model.named_parameters())
    missing = sorted(set(params) - loaded)
    assert not missing, missing[:10]
    return model.eval()


def test_dense_forward_matches_hf(hf_model, mstar_model):
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (12,))
    with torch.no_grad():
        ref = hf_model(input_ids=ids[None]).logits[0]
        out, _ = mstar_model.forward_dense(ids)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)


def test_dense_incremental_and_generation(hf_model, mstar_model):
    torch.manual_seed(1)
    ids = torch.randint(0, 1000, (9,))
    with torch.no_grad():
        full, _ = mstar_model.forward_dense(ids)
        l1, st = mstar_model.forward_dense(ids[:5])
        l2, _ = mstar_model.forward_dense(ids[5:], st)
        hf_out = hf_model.generate(ids[None], max_new_tokens=6, do_sample=False, use_cache=True)[0, 9:].tolist()
    torch.testing.assert_close(torch.cat([l1, l2]), full, rtol=1e-4, atol=1e-3)
    assert mstar_model.generate_dense(ids, 6) == hf_out


def test_kda_paged_kernel_matches_dense(mstar_model, tiny_dir):
    """The paged reference kernel (slot-indexed, varlen) agrees with the dense path and
    carries state across a prefill + decode split."""
    from mstar.model.kimi_k3.components.kda import KDAParams  # noqa: F401
    from mstar.model.kimi_k3.reference.kda import from_v_first
    cfg = KimiK3Config.from_hf_dir(tiny_dir).text
    layer = mstar_model.model.layers[0].self_attn
    torch.manual_seed(2)
    t1, t2 = 7, 4
    x = torch.randn(t1 + t2, cfg.hidden_size)
    # dense: sequence A (t1 tokens) then continue with 2 tokens; sequence B (t2 tokens)
    out_a, st_a = layer.forward_dense(x[:t1])
    xa2 = torch.randn(2, cfg.hidden_size)
    out_a2, st_a2 = layer.forward_dense(xa2, st_a)
    out_b, st_b = layer.forward_dense(x[t1:])
    # paged: slots 1 (A) and 2 (B), packed prefill of both, then a decode of A only
    p = layer.params()
    n_slots = 3
    conv = torch.zeros(n_slots, 3 * layer.projection_size, cfg.kda_conv_kernel_size)
    rec = torch.zeros(n_slots, layer.num_heads, layer.head_dim, layer.head_dim)
    qkv, g_raw, beta_raw = layer._project(x)
    o = layer.kernels.run_lists(qkv, g_raw, beta_raw, [0, t1, t1 + t2], [1, 2], [False, False], conv, rec, p)
    out_paged = layer._finish(x, o)
    torch.testing.assert_close(out_paged[:t1], out_a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_paged[t1:], out_b, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(from_v_first(rec[1]), st_a.recurrent, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(from_v_first(rec[2]), st_b.recurrent, rtol=1e-4, atol=1e-4)
    qkv2, g2, b2 = layer._project(xa2)
    o2 = layer.kernels.run_lists(qkv2, g2, b2, [0, 2], [1], [True], conv, rec, p)
    torch.testing.assert_close(layer._finish(xa2, o2), out_a2, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(from_v_first(rec[1]), st_a2.recurrent, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(conv[1, : layer.projection_size], st_a2.conv_q, rtol=1e-5, atol=1e-5)


def test_kda_params_bundle_is_cached_and_invalidated(mstar_model):
    """``params()`` (the concatenated conv weights the kernels take) is built once and reused
    -- rebuilding it was a launch per layer per decode step -- and rebuilt after the weights
    move (``_apply``), are reloaded, or are written in place."""
    layer = mstar_model.model.layers[0].self_attn
    p1 = layer.params()
    assert layer.params() is p1
    expected = torch.cat([layer.q_conv1d.weight[:, 0], layer.k_conv1d.weight[:, 0], layer.v_conv1d.weight[:, 0]])
    assert torch.equal(p1.conv_weight, expected) and p1.A_log is layer.A_log
    with torch.no_grad():
        layer.q_conv1d.weight.mul_(2.0)  # in-place write bumps the version counter
    p2 = layer.params()
    assert p2 is not p1 and torch.equal(p2.conv_weight[: layer.projection_size], layer.q_conv1d.weight[:, 0])
    layer._apply(lambda t: t)  # a device/dtype move rebinds the parameters
    assert layer.params() is not p2
    p3 = layer.params()
    layer.load_state_dict(layer.state_dict())
    assert layer.params() is not p3
    with torch.no_grad():
        layer.q_conv1d.weight.div_(2.0)


def test_llm_submodule_caps_prefill_batches_only(mstar_model, tiny_dir):
    """Prefill steps are bounded in requests (their transient memory grows with the tokens in
    the step); decode is left to the captured graph buckets."""
    from mstar.model.kimi_k3.submodules import KimiK3LLMSubmodule
    cfg = KimiK3Config.from_hf_dir(tiny_dir)
    sub = KimiK3LLMSubmodule(language_model=mstar_model, config=cfg, cuda_graphs=False)
    assert sub.max_batch_size("prefill") == 8 and sub.max_batch_size("decode") is None
    sub = KimiK3LLMSubmodule(language_model=mstar_model, config=cfg, cuda_graphs=False, max_prefill_batch_size=3)
    assert sub.max_batch_size("prefill") == 3
    sub = KimiK3LLMSubmodule(language_model=mstar_model, config=cfg, cuda_graphs=False, max_prefill_batch_size=None)
    assert sub.max_batch_size("prefill") is None
