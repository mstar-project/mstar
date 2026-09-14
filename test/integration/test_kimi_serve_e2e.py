"""The serving loop end to end: prompt bytes in, generated tokens out.

Covers the pieces the engine drives around the forward — ``process_prompt``,
the sampler resource, ``postprocess``'s output rebinding, and ``check_stop``
terminating on max_tokens — and that two identical runs agree.
"""

import pytest
import torch
from kimi_harness import (
    DEVICE,
    cleanup,
    forward_step,
    load_submodule,
    open_request,
)
from kimi_reference import write_checkpoint

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.kimi_k2_7.config import SAMPLER, KimiK2Config
from mstar.model.kimi_k2_7.kimi_model import KimiK2Model
from mstar.model.kimi_k2_7.submodules import KimiLLMSubmodule

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="serve e2e needs a GPU",
)


def _fwd_info(model, max_tokens):
    return CurrentForwardPassInfo(
        request_id="r0", graph_walk="decode", fwd_index=0, random_seed=0,
        max_tokens=max_tokens,
        resource_configs=model.get_request_resource_configs({}),
        dynamic_loop_iter_counts={},
    )


def _run_generation(model, cfg, prompt_ids, max_tokens):
    submodule, resources = load_submodule(model)
    info = _fwd_info(model, max_tokens)
    generated: list[int] = []
    stopped = False
    try:
        assert isinstance(submodule, KimiLLMSubmodule)
        assert list(submodule.language_model.named_buffers()) == []
        open_request(model, resources, "r0", greedy=True)

        out = forward_step(submodule, resources, "prefill", {"r0": prompt_ids})
        token = out["r0"]["new_token"][0]
        generated.append(int(token.item()))

        for iteration in range(max_tokens + 4):  # slack; check_stop must break
            out = forward_step(submodule, resources, "decode", {"r0": token})
            token = out["r0"]["new_token"][0]
            outputs = {"new_token": [token]}
            submodule.postprocess("r0", info, outputs)  # rebinds text_inputs
            assert outputs["text_inputs"] is outputs["new_token"]
            info.dynamic_loop_iter_counts["decode_loop"] = iteration
            stop = submodule.check_stop("r0", info, outputs)
            generated.append(int(token.item()))
            if stop:
                stopped = True
                break
    finally:
        cleanup(resources)
    return generated, stopped


def _model(tmp_path, cfg, seed):
    write_checkpoint(tmp_path, cfg, seed=seed)
    return KimiK2Model(
        model_path_hf=str(tmp_path), config_variant="reduced",
        tokenizer_mode="byte",
    )


def test_serve_path_prefill_decode_loop(tmp_path):
    cfg = KimiK2Config.reduced()
    model = _model(tmp_path, cfg, seed=0)
    assert model.config.vocab_size == 256

    prompt_ids = model.process_prompt(
        "hello kimi", ["text"], ["text"]
    )["text_inputs"][0].to(DEVICE)
    assert prompt_ids.tolist() == list("hello kimi".encode())
    assert prompt_ids.max().item() < cfg.vocab_size

    # the sampler resource is what the submodule samples through now
    assert SAMPLER in model.get_request_resource_configs({})

    max_tokens = 6
    generated, stopped = _run_generation(model, cfg, prompt_ids, max_tokens)

    assert stopped, "decode loop did not terminate via check_stop"
    # check_stop fires after exactly max_tokens decode steps
    assert len(generated) == 1 + max_tokens, generated
    assert all(0 <= t < cfg.vocab_size for t in generated), generated

    out_bytes = model.postprocess(torch.tensor(generated), "text")
    assert isinstance(out_bytes, bytes)
    assert len(out_bytes) == len(generated)


def test_serve_path_is_deterministic(tmp_path):
    cfg = KimiK2Config.reduced()
    model = _model(tmp_path, cfg, seed=1)
    prompt_ids = model.process_prompt(
        "serve", ["text"], ["text"]
    )["text_inputs"][0].to(DEVICE)

    runs = [
        _run_generation(model, cfg, prompt_ids, max_tokens=5)[0] for _ in range(2)
    ]
    assert runs[0] == runs[1], runs
    assert len(runs[0]) == 1 + 5
