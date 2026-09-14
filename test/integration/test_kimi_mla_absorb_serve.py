"""The absorbed path serves the same numbers as the naive one.

Same checkpoint, loaded twice: once with ``mla_absorb=False`` (materialized
per-head K/V, NHD cache, paged backend) and once with ``mla_absorb=True``
(one compressed latent per token, MLA cache, absorbed backend). The absorbed
forward folds ``kv_b_proj`` into Q/O, so agreeing here is what says the
absorption is algebraically exact end to end.
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
from kimi_reference import make_model, write_checkpoint

from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.kimi_k2_7.submodules import KimiLLMSubmodule

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the absorbed serve path runs on GPU",
)


def _config(mla_absorb: bool) -> KimiK2Config:
    cfg = KimiK2Config.reduced()
    cfg.mla_absorb = mla_absorb
    return cfg


def _prefill_logits(cfg, checkpoint_dir, prompt):
    model = make_model(cfg, checkpoint_dir)
    submodule, resources = load_submodule(model)
    try:
        assert isinstance(submodule, KimiLLMSubmodule)
        open_request(model, resources, "r0", greedy=True)
        logits = logits_step(submodule, resources, "prefill", {"r0": prompt})
        return logits, submodule, resources
    except Exception:
        cleanup(resources)
        raise


def test_absorbed_serve_matches_naive_reference(tmp_path):
    cfg_naive = _config(mla_absorb=False)
    assert KimiK2Config.reduced().mla_absorb is False  # the reduced default
    write_checkpoint(tmp_path, cfg_naive, seed=0)

    prompt = torch.randint(0, cfg_naive.vocab_size, (6,), device=DEVICE)

    naive_logits, _naive_sub, naive_res = _prefill_logits(
        cfg_naive, tmp_path, prompt
    )
    try:
        assert naive_logits.shape == (1, cfg_naive.vocab_size)
        assert torch.isfinite(naive_logits).all()
        naive_logits = naive_logits.clone()
    finally:
        cleanup(naive_res)

    cfg_absorb = _config(mla_absorb=True)
    absorbed_logits, submodule, resources = _prefill_logits(
        cfg_absorb, tmp_path, prompt
    )
    try:
        # the absorbed projections must have been built from kv_b_proj on load
        buffers = {name for name, _ in submodule.language_model.named_buffers()}
        assert any("w_kc" in name for name in buffers)
        assert any("fused_qkv_a_proj" in name for name in buffers)

        assert absorbed_logits.shape == (1, cfg_absorb.vocab_size)
        assert torch.isfinite(absorbed_logits).all()
        torch.testing.assert_close(
            absorbed_logits, naive_logits, rtol=5e-2, atol=5e-2
        )

        token = absorbed_logits.argmax(-1)
        generated = [int(token.item())]
        for _ in range(4):
            out = forward_step(submodule, resources, "decode", {"r0": token})
            token = out["r0"]["new_token"][0]
            tok = int(token.item())
            assert 0 <= tok < cfg_absorb.vocab_size
            generated.append(tok)
    finally:
        cleanup(resources)

    assert len(generated) == 5
