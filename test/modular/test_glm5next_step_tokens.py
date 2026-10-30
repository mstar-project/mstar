"""GLM-5.3 caps a prefill step's rows at what a captured prefill takes."""
import sys

sys.path.insert(0, ".")

from mstar.model.glm5_next.config import Glm5NextModelConfig  # noqa: E402
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule  # noqa: E402


def _submodule(config) -> Glm5NextLLMSubmodule:
    sub = object.__new__(Glm5NextLLMSubmodule)
    sub.config = config
    return sub


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
