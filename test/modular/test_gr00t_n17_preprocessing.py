import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from mstar.model.gr00t_n17.preprocessing import Gr00tN17Preprocessor

_GR00T = Path.home() / (".cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/"
                        "2fc962b973bccdd5d8ce4f67cc63b264d6886495")

# Tiny two-embodiment checkpoint. "grip" is constant under q01/q99 but not under min/max.
KWARGS = {
    "use_percentiles": True, "use_mean_std": False, "apply_sincos_state_encoding": False, "exclude_state": False,
    "modality_configs": {
        "bot": {"state": {"modality_keys": ["arm", "grip"]}, "action": {"delta_indices": [0, 1, 2]}},
        "other": {"state": {"modality_keys": ["j"]}, "action": {"delta_indices": list(range(5))}},
    },
}
STATS = {
    "bot": {"state": {
        "grip": {"q01": [5.0], "q99": [5.0], "min": [0.0], "max": [10.0]},
        "arm": {"q01": [0.0, -1.0], "q99": [2.0, 1.0], "min": [-2.0, -3.0], "max": [4.0, 3.0]},
    }},
    "other": {"state": {"j": {"q01": [0.0], "q99": [1.0], "min": [0.0], "max": [1.0]}}},
}
IDS = {"bot": 7, "other": 2, "unused": 9}
# (1, T=4, D): midpoint, at q01, at q99, outside
ARM = [[[1.0, 0.0], [0.0, -1.0], [2.0, 1.0], [9.0, -9.0]]]
GRIP = [[[7.5], [5.0], [5.0], [-100.0]]]


def _pre(kwargs=KWARGS, max_state_dim=5, max_action_horizon=6):
    return Gr00tN17Preprocessor(kwargs, STATS, IDS, max_state_dim, max_action_horizon)


def test_state_normalises_orders_and_pads():
    # grip -> 0 proves q01/q99 bounds: min/max would give 0.5 for the midpoint
    arm, grip = [[0, 0], [-1, -1], [1, 1], [1, -1]], [0, 0, 0, 0]
    pre = _pre()
    # input dict order is grip-first; output must follow modality_keys (arm, grip)
    s = pre.state("bot", {"grip": np.array(GRIP), "arm": np.array(ARM)})
    assert s.shape == (1, 4, 5) and s.dtype == torch.float32
    expected = torch.tensor([[a + [g, 0, 0] for a, g in zip(arm, grip, strict=True)]], dtype=torch.float32)
    torch.testing.assert_close(s, expected)


def test_tag_is_case_insensitive():
    pre = _pre()
    state = {"arm": np.array(ARM), "grip": np.array(GRIP)}
    assert torch.equal(pre.state("BoT", state), pre.state("bot", state))
    assert torch.equal(pre.action_mask("OTHER", 1), pre.action_mask("other", 1))


@pytest.mark.parametrize("tag, state, max_state_dim, match", [
    ("robot", {"arm": np.zeros(2), "grip": np.zeros(1)}, 5, "not in this checkpoint"),
    ("unused", {"arm": np.zeros(2), "grip": np.zeros(1)}, 5, "not in this checkpoint"),
    ("bot", {"arm": np.zeros(3), "grip": np.zeros(1)}, 5, "expected 2"),
    ("bot", {"arm": np.zeros(2), "grip": np.zeros(1)}, 2, "more than max_state_dim"),
])
def test_bad_request_raises(tag, state, max_state_dim, match):
    with pytest.raises(ValueError, match=match):
        _pre(max_state_dim=max_state_dim).state(tag, state)


def test_embodiment_id_and_action_mask():
    pre = _pre()
    e = pre.embodiment_id("bot", 3)
    assert e.dtype == torch.long and e.tolist() == [7, 7, 7]
    assert pre.embodiment_id("other", 1).tolist() == [2]
    m = pre.action_mask("bot", 2)
    assert m.shape == (2, 6) and m.dtype == torch.float32
    assert m.tolist() == [[1, 1, 1, 0, 0, 0]] * 2
    assert pre.action_mask("other", 1).tolist() == [[1, 1, 1, 1, 1, 0]]


def _set(path, value):
    kwargs = copy.deepcopy(KWARGS)
    *parents, last = path
    node = kwargs
    for p in parents:
        node = node[p]
    node[last] = value
    return kwargs


@pytest.mark.parametrize("kwargs, match", [
    (_set(["use_mean_std"], True), "unsupported"),
    (_set(["apply_sincos_state_encoding"], True), "unsupported"),
    (_set(["exclude_state"], True), "unsupported"),
    (_set(["use_percentiles"], False), "unsupported"),
    (_set(["clip_outliers"], False), "unsupported"),
    (_set(["modality_configs", "bot", "state", "mean_std_embedding_keys"], ["arm"]), "unsupported"),
    (_set(["modality_configs", "other", "action", "delta_indices"], list(range(7))), "exceed"),
])
def test_rejects_a_variant_it_does_not_implement(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _pre(kwargs)


@pytest.mark.skipif(not _GR00T.exists(), reason="checkpoint not cached")
def test_the_checkpoint_loads():
    pre = Gr00tN17Preprocessor.from_pretrained(_GR00T, max_state_dim=132, max_action_horizon=40)
    assert len(pre.state_keys) == 8
    tag = "oxe_droid_relative_eef_relative_joint"
    assert pre.embodiment_id(tag, 2).tolist() == [24, 24]
    assert pre.action_mask(tag, 1).sum() == 40
    # the reference's short enum name is not a checkpoint tag (D06)
    with pytest.raises(ValueError, match="xdof_relative_eef_relative_joint"):
        pre.embodiment_id("XDOF", 1)
    lows = {k: lo for k, (lo, _) in pre.state_bounds[tag].items()}
    s = pre.state(tag, {k: lo[None] for k, lo in lows.items()})
    n = sum(lo.size for lo in lows.values())
    assert s.shape == (1, 132) and torch.all(s[:, :n] <= 0) and torch.all(s[:, n:] == 0)
