"""GLM-5.2 declares a prefill token budget per step from model_kwargs."""
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


def _budget(walk="prefill", **kwargs):
    model = Glm52Model("", tokenizer_mode="byte", config_variant="full", **kwargs)
    return _submodule(model.config).max_step_tokens(walk)


def test_no_budget_by_default():
    assert Glm52ModelConfig().prefill_max_step_tokens is None
    assert _budget() is None


def test_budget_from_model_kwargs_on_prefill_only():
    assert _budget(prefill_max_step_tokens=3000) == 3000
    assert _budget(prefill_max_step_tokens="3000") == 3000
    assert _budget("decode", prefill_max_step_tokens=3000) is None


@pytest.mark.parametrize(("buckets", "expected"), [
    ({"prefill_token_buckets": [64, 2048], "prefill_batched_token_buckets": [1024, 4096]}, 4096),
    ({"prefill_token_buckets": [64, 2048]}, 2048),
    ({}, max(Glm52LLMSubmodule.PREFILL_TOKEN_BUCKETS)),
])
def test_auto_budget_is_the_largest_captured_bucket(buckets, expected):
    assert _budget(prefill_max_step_tokens="auto", **buckets) == expected


@pytest.mark.parametrize("name", ["glm52_tp8.yaml", "glm52_tp8_mtp.yaml"])
def test_shipped_config_prefill_steps_fit_the_largest_bucket(name):
    import yaml

    path = Path(__file__).resolve().parents[2] / "configs" / name
    kwargs = yaml.safe_load(path.read_text())["model_kwargs"]
    assert _budget(**kwargs) == max(kwargs["prefill_batched_token_buckets"])


def test_longctx_config_window_fits_its_pages():
    """The long-context yaml builds the DSA path, and both caches hold a full window."""
    import yaml

    path = Path(__file__).resolve().parents[2] / "configs" / "glm52_tp8_longctx.yaml"
    cfg = yaml.safe_load(path.read_text())
    model = Glm52Model("", tokenizer_mode="byte", config_variant="full", **cfg["model_kwargs"])
    assert model.config.dsa_long_context
    assert model.config.dsa_shard_prefill
    # every step of the captured prefill buckets runs in one pass, never in row chunks
    assert _submodule(model.config).max_step_tokens("prefill") <= model.config.prefill_chunk_tokens
    assert model.config.max_seq_len == cfg["max_seq_len"]
    for key in ("kv", "kv_index"):
        res = cfg["resources"][key]
        assert res["max_num_pages"] * res["page_size"] >= model.config.max_seq_len
