"""The Kimi LLM submodule over its real resources, prefill then decode.

Checks the loaded weights produce the same prefill logits as a reference
forward through plain SDPA at the DeepSeek-correct scale, that the decode loop
advances, and that two identical runs agree.
"""

import pytest
import torch
from kimi_harness import (
    DEVICE,
    cleanup,
    forward_step,
    load_submodule,
    logits_step,
    open_request,
)
from kimi_reference import (
    make_model,
    ref_deepseek_mla,
    write_checkpoint,
)

from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.kimi_k2_7.submodules import KimiLLMSubmodule

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="submodule e2e needs a GPU (real FlashInfer paged cache)",
)


def _naive_config():
    """The naive path: an NHD cache under the paged backend, so the reference
    forward can be plain materialized MLA."""
    cfg = KimiK2Config.reduced()
    cfg.mla_absorb = False
    return cfg


def _reference_logits(submodule, cfg, prompt):
    """The same loaded weights, layer by layer, through the SDPA reference."""
    model = submodule.language_model.model
    pos = torch.arange(prompt.shape[0], device=DEVICE)
    with torch.no_grad():
        hidden = model.embed_tokens(prompt)
        for layer in model.layers:
            residual = hidden
            hidden = residual + ref_deepseek_mla(
                layer.self_attn, cfg, layer.input_layernorm(hidden), pos
            )
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = model.norm(hidden)
        return submodule.lm_head(hidden[-1:])


def test_submodule_prefill_decode_over_real_paged_cache(tmp_path):
    cfg = _naive_config()
    write_checkpoint(tmp_path, cfg, seed=0)
    model = make_model(cfg, tmp_path)

    submodule, resources = load_submodule(model)
    try:
        assert isinstance(submodule, KimiLLMSubmodule)
        assert model.get_submodule("LLM") is submodule
        # derived buffers must not survive meta -> to_empty as garbage
        assert list(submodule.language_model.named_buffers()) == []
        param = next(submodule.language_model.parameters())
        assert param.device.type == "cuda" and param.dtype == torch.bfloat16

        open_request(model, resources, "r0", greedy=True)
        prompt = torch.randint(0, cfg.vocab_size, (6,), device=DEVICE)

        logits = logits_step(submodule, resources, "prefill", {"r0": prompt})
        assert logits.shape == (1, cfg.vocab_size)
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(
            logits, _reference_logits(submodule, cfg, prompt), rtol=5e-2, atol=5e-2,
        )

        token = logits.argmax(-1)
        generated = [int(token.item())]
        for _ in range(4):
            out = forward_step(submodule, resources, "decode", {"r0": token})
            token = out["r0"]["new_token"][0]
            generated.append(int(token.item()))
    finally:
        cleanup(resources)

    assert len(generated) == 5
    assert all(0 <= t < cfg.vocab_size for t in generated), generated


def test_submodule_paged_decode_is_deterministic(tmp_path):
    cfg = _naive_config()
    write_checkpoint(tmp_path, cfg, seed=1)
    model = make_model(cfg, tmp_path)
    prompt = torch.randint(0, cfg.vocab_size, (5,), device=DEVICE)

    tokens = []
    for _ in range(2):
        # a fresh resource set each time, so neither run inherits the other's pages
        submodule, resources = load_submodule(model)
        try:
            open_request(model, resources, "r0", greedy=True)
            logits = logits_step(submodule, resources, "prefill", {"r0": prompt})
            tokens.append(int(logits.argmax(-1).item()))
        finally:
            cleanup(resources)
    assert tokens[0] == tokens[1]
