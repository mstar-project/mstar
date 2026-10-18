import copy
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from mstar.model.gr00t_n17.preprocessing import Gr00tN17Preprocessor

_GR00T = Path.home() / (".cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/"
                        "2fc962b973bccdd5d8ce4f67cc63b264d6886495")
_COSMOS = Path.home() / (".cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/"
                         "9ce19a195e423419c349abfc86fd07178b230561")

# Tiny two-embodiment checkpoint. "grip" is constant under q01/q99 but not under min/max.
KWARGS = {
    "use_percentiles": True, "use_mean_std": False, "apply_sincos_state_encoding": False, "exclude_state": False,
    "shortest_image_edge": 256, "crop_fraction": 0.95,
    "modality_configs": {
        "bot": {"video": {"modality_keys": ["front", "wrist"]}, "state": {"modality_keys": ["arm", "grip"]},
                "action": {"delta_indices": [0, 1, 2]}},
        "other": {"video": {"modality_keys": ["cam"]}, "state": {"modality_keys": ["j"]},
                  "action": {"delta_indices": list(range(5))}},
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


def test_group_without_stats_is_zero_filled_at_input_width():
    # xdof's eef in the real checkpoint: 0-dim stats, 9-dim state -> 9 zeros
    stats = copy.deepcopy(STATS)
    stats["bot"]["state"]["arm"] = {"q01": [], "q99": []}
    pre = Gr00tN17Preprocessor(KWARGS, stats, IDS, 5, 6)
    s = pre.state("bot", {"arm": np.full((1, 2), 7.0), "grip": np.zeros((1, 1))})
    assert s.tolist() == [[0, 0, 0, 0, 0]]


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
    (_set(["use_albumentations"], False), "unsupported"),
    (_set(["letter_box_transform"], True), "unsupported"),
    (_set(["formalize_language"], False), "unsupported"),
    (_set(["modality_configs", "bot", "state", "mean_std_embedding_keys"], ["arm"]), "unsupported"),
    (_set(["modality_configs", "other", "action", "delta_indices"], list(range(7))), "exceed"),
])
def test_rejects_a_variant_it_does_not_implement(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _pre(kwargs)



@pytest.mark.parametrize("hw, out_hw", [
    ((180, 320), (256, 455)),  # 256x455 -> crop 243x432 at (6, 11) -> 256x455
    ((320, 180), (455, 256)),  # portrait
    ((256, 256), (256, 256)),  # first resize skipped (scale 1)
    ((512, 512), (256, 256)),
    ((1, 4), (256, 1024)),     # tiny frame: crop keeps >= 1 px
    ((512, 513), (256, 256)),  # 256.5 rounds half to even; half-up would give 257 wide
])
def test_image_output_size(hw, out_hw):
    pytest.importorskip("cv2")
    out = _pre().image(np.zeros((*hw, 3), np.uint8))
    assert out.shape == (*out_hw, 3) and out.dtype == np.uint8


def test_image_constant_frame_stays_constant():
    pytest.importorskip("cv2")
    out = _pre().image(np.full((180, 320, 3), (10, 128, 250), np.uint8))
    assert np.all(out == np.array([10, 128, 250], np.uint8))


@pytest.mark.parametrize("hw", [(180, 320), (256, 256), (480, 640)])
def test_image_matches_hand_written_area_chain(hw):
    cv2 = pytest.importorskip("cv2")
    img = np.random.default_rng(0).integers(0, 256, (*hw, 3), dtype=np.uint8)

    def chain(interp):
        def edge(x):
            h, w = x.shape[:2]
            s = 256 / min(h, w)
            return x if s == 1 else cv2.resize(x, (round(w * s), round(h * s)), interpolation=interp)
        x = edge(img)
        h, w = x.shape[:2]
        ch, cw = int(h * 0.95), int(w * 0.95)
        return edge(x[(h - ch) // 2 : (h - ch) // 2 + ch, (w - cw) // 2 : (w - cw) // 2 + cw])

    out = _pre().image(img)
    assert np.array_equal(out, chain(cv2.INTER_AREA))
    assert not np.array_equal(out, chain(cv2.INTER_LINEAR))  # negative control: the filter matters

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


class _StubProcessor:
    """Records what the preprocessor hands the backbone processor."""

    def __init__(self):
        self.tokenizer = types.SimpleNamespace(padding_side="right")
        self.conversations, self.calls = [], []

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
        self.conversations.append(conversation)
        return conversation[0]["content"][-1]["text"]

    def __call__(self, text, images, **kwargs):
        self.calls.append((text, images))
        return {"input_ids": torch.zeros(len(text), 3, dtype=torch.long)}


def _video(b, t, view):  # constant frames, one colour per (item, step, view)
    x = np.zeros((b, t, 180, 320, 3), np.uint8)
    for i in range(b):
        for j in range(t):
            x[i, j] = 100 * i + 10 * j + view
    return x


def test_call_feeds_the_processor_in_reference_order():
    pytest.importorskip("cv2")
    stub = _StubProcessor()
    pre = Gr00tN17Preprocessor(KWARGS, STATS, IDS, 5, 6, processor=stub)
    assert stub.tokenizer.padding_side == "left"
    video = {"wrist": _video(2, 2, 1), "front": _video(2, 2, 0)}  # dict order is not modality order
    state = {"arm": np.zeros((2, 1, 2)), "grip": np.zeros((2, 1, 1))}
    out = pre("bot", video, ["Pick up the RED cup!", "go."], state)
    [(texts, images)] = stub.calls  # one call for the whole batch
    assert texts == ["pick up the red cup", "go"]
    content = stub.conversations[0][0]["content"]
    assert [c["type"] for c in content] == ["image"] * 4 + ["text"]
    # time-major, views in modality order, item after item; constant frames stay constant
    assert [int(f[0, 0, 0]) for f in images] == [100 * b + 10 * t + v for b in (0, 1) for t in (0, 1) for v in (0, 1)]
    assert all(f.shape == (3, 256, 455) and torch.all(f == f[0, 0, 0]) for f in images)
    assert out["input_ids"].shape[0] == 2 and out["state"].shape == (2, 1, 5)
    assert out["embodiment_id"].tolist() == [7, 7] and out["action_mask"].shape == (2, 6)


def test_call_rejects_language_batch_mismatch():
    pre = Gr00tN17Preprocessor(KWARGS, STATS, IDS, 5, 6, processor=_StubProcessor())
    with pytest.raises(ValueError, match="instructions for a batch of 2"):
        pre("other", {"cam": _video(2, 1, 0)}, ["only one"], {"j": np.zeros((2, 1, 1))})


@pytest.mark.skipif(not (_GR00T.exists() and _COSMOS.exists()), reason="checkpoints not cached")
def test_call_with_the_real_processor():
    pytest.importorskip("cv2")
    pre = Gr00tN17Preprocessor.from_pretrained(_GR00T, 132, 40, backbone_path=_COSMOS)
    tag = "oxe_droid_relative_eef_relative_joint"
    video = {k: np.zeros((2, 2, 180, 320, 3), np.uint8) for k in pre.video_keys[tag]}
    state = {k: np.zeros((2, 1, lo.size)) for k, (lo, _) in pre.state_bounds[tag].items()}
    out = pre(tag, video, ["pick up the red cup and put it in the bowl", "stop"], state)
    L = out["input_ids"].shape[1]
    assert out["input_ids"].shape == out["attention_mask"].shape == (2, L)
    assert out["attention_mask"][1, 0] == 0 and out["attention_mask"][:, -1].tolist() == [1, 1]  # left padding
    assert out["image_grid_thw"].tolist() == [[1, 16, 28]] * 8
    assert out["pixel_values"].shape[0] == 8 * 448
