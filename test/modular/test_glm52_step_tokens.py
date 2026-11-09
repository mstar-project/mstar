"""GLM-5.2's long-context config and its startup checks."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.model.glm52.config import Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.glm52_model import Glm52Model  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402


def _submodule(config) -> Glm52LLMSubmodule:
    sub = object.__new__(Glm52LLMSubmodule)
    sub.config = config
    return sub


def test_longctx_config_window_fits_its_pages():
    """The long-context yaml builds the DSA path, and its cache (the latents and, on the same
    pages, the index keys) holds a full window."""
    import yaml

    path = Path(__file__).resolve().parents[2] / "configs" / "glm52_tp8_longctx.yaml"
    cfg = yaml.safe_load(path.read_text())
    model = Glm52Model("", tokenizer_mode="byte", config_variant="full", **cfg["model_kwargs"])
    assert model.config.dsa_long_context
    assert model.config.dsa_shard_prefill
    # every captured prefill bucket runs in one pass, never in row chunks
    ladders = cfg["prefill_token_buckets"]["LLM"]["prefill"]
    assert max(max(ladder) for ladder in ladders.values()) <= model.config.prefill_chunk_tokens
    assert model.config.max_seq_len == cfg["max_seq_len"]
    assert set(cfg["resources"]) == {"kv"}
    res = cfg["resources"]["kv"]
    # the manager holds one page back as its sink
    assert (res["max_num_pages"] - 1) * res["page_size"] >= model.config.max_seq_len


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


def _captured_prefill(sub, ladder=None) -> list[int]:
    """The prefill token counts the submodule captures at any batch size, once the engine has
    applied ``ladder`` (a serving config's ``prefill_token_buckets``)."""
    import torch

    from mstar.engine.accelerator_graph_config import apply_token_ladders

    sub._moe_capture_blocked = lambda tp_world_size: False
    configs = sub.get_accelerator_graph_configs(torch.device("cpu"))
    if ladder is not None:
        apply_token_ladders("LLM", configs, {"prefill": ladder}, lambda walk, rows: None,
                            lambda config: config.capture_batch_sizes)
    return [n for c in configs if c.capture_graph_walk == "prefill"
            for bs in c.capture_batch_sizes for n in c.get_total_tokens(bs)]


def test_long_context_captures_no_prefill_bucket_past_the_chunk():
    chunk = max(Glm52LLMSubmodule.PREFILL_TOKEN_BUCKETS) // 2
    cfg = Glm52ModelConfig(dsa_long_context=True, max_seq_len=262144, prefill_chunk_tokens=chunk)
    prefill = _captured_prefill(_submodule(cfg))
    assert prefill and max(prefill) <= chunk


def test_long_context_captures_no_prefill_bucket_past_one_sparse_plan():
    """A captured prefill attends in one sparse plan, which FlashInfer takes up to
    sparse_mla.MAX_ROWS rows; a longer bucket runs eager, whose plan takes chunks, whatever
    ladder the deployment names."""
    from mstar.engine.resources.dsa import sparse_mla

    cfg = Glm52ModelConfig(dsa_long_context=True, max_seq_len=262144,
                           prefill_chunk_tokens=4 * sparse_mla.MAX_ROWS)
    prefill = _captured_prefill(_submodule(cfg), {1: [1024], 2: [1024, 2 * sparse_mla.MAX_ROWS]})
    assert 1024 in prefill and max(prefill) <= sparse_mla.MAX_ROWS
