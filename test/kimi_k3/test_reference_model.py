"""Whole-model parity on the tiny checkpoint's real weights (CPU, fp32): the reference
forward vs the HF model, full prefill vs incremental decode, and greedy generation."""

import pytest
import torch
from safetensors.torch import load_file

from mstar.model.kimi_k3.config import KimiK3Config
from mstar.model.kimi_k3.reference.kda import patch_hf_modeling_for_cpu
from mstar.model.kimi_k3.reference.model import ModelWeights, reference_forward, reference_generate


@pytest.fixture(scope="module")
def hf_model(tiny_dir, hf_modeling, hf_text_config):
    patch_hf_modeling_for_cpu(hf_modeling)
    torch.manual_seed(0)
    model = hf_modeling.KimiLinearForCausalLM(hf_text_config).eval()
    # the modeling file forces flash_attention_2 in __init__; run eager on CPU
    model.config._attn_implementation = "eager"
    sd = load_file(str(tiny_dir / "model.safetensors"))
    lm = {k[len("language_model."):]: v for k, v in sd.items() if k.startswith("language_model.")}
    missing, unexpected = model.load_state_dict(lm, strict=False)
    assert not unexpected, unexpected[:5]
    assert not [k for k in missing if "rotary" not in k], missing[:5]
    return model.float()


@pytest.fixture(scope="module")
def ref_weights(hf_model, tiny_dir):
    cfg = KimiK3Config.from_hf_dir(tiny_dir).text
    return ModelWeights.from_hf_model(hf_model, cfg)


def test_full_forward_matches_hf(hf_model, ref_weights):
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (13,))
    with torch.no_grad():
        ref = hf_model(input_ids=ids[None]).logits[0]
        out, _ = reference_forward(ref_weights, ids)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)


def test_incremental_decode_matches_full(ref_weights):
    torch.manual_seed(1)
    ids = torch.randint(0, 1000, (10,))
    with torch.no_grad():
        full, _ = reference_forward(ref_weights, ids)
        l1, st = reference_forward(ref_weights, ids[:6])
        l2, st = reference_forward(ref_weights, ids[6:9], st)
        l3, _ = reference_forward(ref_weights, ids[9:], st)
    inc = torch.cat([l1, l2, l3])
    torch.testing.assert_close(inc, full, rtol=1e-4, atol=1e-3)


def test_greedy_generation_matches_hf(hf_model, ref_weights, hf_modeling, hf_text_config):
    torch.manual_seed(2)
    ids = torch.randint(0, 1000, (5,))
    n = 6
    with torch.no_grad():
        hf_out = hf_model.generate(ids[None], max_new_tokens=n, do_sample=False, use_cache=True)[0, 5:].tolist()
    ref_out = reference_generate(ref_weights, ids, n)
    assert ref_out == hf_out, (ref_out, hf_out)
