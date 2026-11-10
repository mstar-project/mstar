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


@pytest.mark.parametrize("name", ["glm52_tp8_longctx.yaml", "glm52_tp8_longctx_mtp.yaml"])
def test_longctx_config_window_fits_its_pages(name):
    """The long-context yamls build the DSA path, and their cache (the latents and, on the same
    pages, the index keys) holds a full window."""
    import yaml

    path = Path(__file__).resolve().parents[2] / "configs" / name
    cfg = yaml.safe_load(path.read_text())
    model = Glm52Model("", tokenizer_mode="byte", config_variant="full", **cfg["model_kwargs"])
    assert model.config.dsa_long_context
    assert model.config.mtp_num_draft_tokens == (3 if "mtp" in name else 0)
    assert model.config.dsa_shard_prefill
    # every captured prefill bucket runs in one pass, never in row chunks (MTP over DSA captures none)
    ladders = cfg.get("prefill_token_buckets", {}).get("LLM", {}).get("prefill", {})
    assert max((n for ladder in ladders.values() for n in ladder), default=0) <= model.config.prefill_window_tokens
    assert model.config.max_seq_len == cfg["max_seq_len"]
    assert set(cfg["resources"]) == {"kv"}
    res = cfg["resources"]["kv"]
    # the manager holds one page back as its sink
    assert (res["max_num_pages"] - 1) * res["page_size"] >= model.config.max_seq_len


def test_long_context_bind_refuses_an_empty_prefill_window():
    cfg = Glm52ModelConfig(dsa_long_context=True, max_seq_len=262144, prefill_window_tokens=0)
    with pytest.raises(ValueError, match="must be >= 1"):
        _submodule(cfg).bind_node_resources({})


def test_long_context_captures_no_prefill_bucket_past_the_chunk():
    import torch

    chunk = max(Glm52LLMSubmodule.PREFILL_TOKEN_BUCKETS) // 2
    cfg = Glm52ModelConfig(dsa_long_context=True, max_seq_len=262144, prefill_window_tokens=chunk)
    sub = _submodule(cfg)
    sub._moe_capture_blocked = lambda tp_world_size: False
    configs = sub.get_accelerator_graph_configs(torch.device("cpu"))
    prefill = [n for c in configs if c.capture_graph_walk == "prefill"
               for ladder in c.capture_token_lengths.values() for n in ladder]
    assert prefill and max(prefill) <= chunk

