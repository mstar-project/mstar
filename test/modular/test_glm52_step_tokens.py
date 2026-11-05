"""GLM-5.2's long-context startup checks."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.model.glm52.config import Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402


def _submodule(config) -> Glm52LLMSubmodule:
    sub = object.__new__(Glm52LLMSubmodule)
    sub.config = config
    return sub


def _kv(pages, page_size):
    import types

    return types.SimpleNamespace(config=types.SimpleNamespace(
        max_num_pages=pages, page_size=page_size, tokens_per_page=page_size))


@pytest.mark.parametrize("change, resources, match", [
    ({"prefill_chunk_tokens": 0}, {}, "must be >= 1"),
    ({}, {"kv": _kv(4096, 96)}, "kv page_size=96 must be a power of two"),
])
def test_long_context_bind_refuses_what_it_cannot_serve(change, resources, match):
    cfg = Glm52ModelConfig(dsa_long_context=True, max_seq_len=262144, **change)
    with pytest.raises(ValueError, match=match):
        _submodule(cfg).bind_node_resources(resources)


def test_long_context_captures_no_prefill_bucket_past_the_chunk(torch_flag):
    import torch

    chunk = max(Glm52LLMSubmodule.PREFILL_TOKEN_BUCKETS) // 2
    cfg = Glm52ModelConfig(dsa_long_context=True, max_seq_len=262144, prefill_chunk_tokens=chunk)
    sub = _submodule(cfg)
    sub._moe_capture_blocked = lambda tp_world_size: False
    torch_flag(torch._dynamo.config, "recompile_limit", torch._dynamo.config.recompile_limit)
    configs = sub.get_accelerator_graph_configs(torch.device("cpu"))
    prefill = [n for c in configs if c.capture_graph_walk == "prefill"
               for n in c.capture_token_lengths]
    assert prefill and max(prefill) <= chunk


def test_long_context_captures_no_prefill_bucket_past_one_sparse_plan(torch_flag):
    """A captured prefill attends in one sparse plan, which FlashInfer takes up to
    sparse_mla.MAX_ROWS rows; a longer bucket runs eager, whose plan takes chunks."""
    import torch

    from mstar.engine.resources.dsa import sparse_mla

    cfg = Glm52ModelConfig(dsa_long_context=True, max_seq_len=262144,
                           prefill_chunk_tokens=4 * sparse_mla.MAX_ROWS,
                           prefill_batched_token_buckets=[1024, 2 * sparse_mla.MAX_ROWS])
    sub = _submodule(cfg)
    sub._moe_capture_blocked = lambda tp_world_size: False
    torch_flag(torch._dynamo.config, "recompile_limit", torch._dynamo.config.recompile_limit)
    configs = sub.get_accelerator_graph_configs(torch.device("cpu"))
    prefill = [n for c in configs if c.capture_graph_walk == "prefill"
               for n in c.capture_token_lengths]
    assert 1024 in prefill and max(prefill) <= sparse_mla.MAX_ROWS
