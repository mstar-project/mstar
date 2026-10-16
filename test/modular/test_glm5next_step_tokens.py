"""GLM-5.3 declares a prefill token budget per step from model_kwargs."""
import sys

sys.path.insert(0, ".")

from mstar.model.glm5_next.config import Glm5NextModelConfig  # noqa: E402
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel  # noqa: E402
from mstar.model.glm5_next.submodules import (  # noqa: E402
    PREFILL_TOKEN_BUCKETS,
    Glm5NextLLMSubmodule,
)


def _submodule(config) -> Glm5NextLLMSubmodule:
    sub = object.__new__(Glm5NextLLMSubmodule)
    sub.config = config
    return sub


def _budget(walk="prefill", **kwargs):
    model = Glm5NextModel("x", config_variant="reduced", tokenizer_mode="byte", **kwargs)
    return _submodule(model.config).max_step_tokens(walk)


def test_no_budget_by_default():
    assert Glm5NextModelConfig().prefill_max_step_tokens is None
    assert _budget() is None


def test_budget_from_model_kwargs_on_prefill_only():
    assert _budget(prefill_max_step_tokens=3000) == 3000
    assert _budget(prefill_max_step_tokens="3000") == 3000
    assert _budget("decode", prefill_max_step_tokens=3000) is None


def test_auto_budget_is_the_largest_captured_bucket():
    config = Glm5NextModelConfig(prefill_max_step_tokens="auto")
    assert _submodule(config).max_step_tokens("prefill") == max(PREFILL_TOKEN_BUCKETS)
    config.prefill_token_buckets = [64, 2048]
    assert _submodule(config).max_step_tokens("prefill") == 2048


def test_captured_prefill_caps_the_rows_of_a_step(monkeypatch):
    """An eager prefill's last layer would compile the MoE decode kernels for its
    row count mid-serve, so a step takes no more rows than a captured one."""
    from types import SimpleNamespace

    from mstar.model.glm5_next import submodules
    from mstar.model.glm5_next.config import KDA_STATE

    monkeypatch.setattr(submodules, "_fused_kda", lambda config: True)
    config = Glm5NextModelConfig(prefill_graphs=True, prefill_capture_batch_sizes=[1, 2])
    sub = _submodule(config)
    pool = SimpleNamespace(num_free_slots=5, config=SimpleNamespace(usable_slots=8))
    sub.node_resources = {KDA_STATE: pool}

    assert sub.max_batch_size("prefill") == 2
    assert sub.max_batch_size("decode") == 8
    pool.num_free_slots = 1
    assert sub.max_batch_size("prefill") == 1
    config.prefill_graphs = False
    pool.num_free_slots = 5
    assert sub.max_batch_size("prefill") == 5, "eager prefill: only the free slots"
