import copy
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from mstar.model.gr00t_n17.preprocessing import Gr00tN17Preprocessor

from .helpers import BACKBONE, GR00T, needs_checkpoint

# The 20 recorded DROID requests the decode's parity bar is measured on (~10 MB each).
_DUMPS = sorted((Path.home() / "gr00t-runs/dump").glob("req_*.pt"))

# Tiny three-embodiment checkpoint. "grip" is constant under q01/q99 but not under min/max.
KWARGS = {
    "use_percentiles": True, "use_mean_std": False, "apply_sincos_state_encoding": False, "exclude_state": False,
    "shortest_image_edge": 256, "crop_fraction": 0.95,
    "modality_configs": {
        "bot": {"video": {"modality_keys": ["front", "wrist"], "delta_indices": [0, 1]},
                "state": {"modality_keys": ["arm", "grip"]},
                "action": {"delta_indices": [0, 1, 2], "modality_keys": ["arm", "grip"],
                           "action_configs": [
                               {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "arm"},
                               {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "grip"}]}},
        "other": {"video": {"modality_keys": ["cam"], "delta_indices": [0, 1]},
                "state": {"modality_keys": ["j"]},
                  "action": {"delta_indices": list(range(5)), "modality_keys": ["j"],
                             "action_configs": [
                                 {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "j"}]}},
        # the SE(3) robot: three groups of three different widths with the relative
        # eef one in the *middle*, so no group's offset is its own dim and none of
        # the three offsets is 0
        "eefbot": {"video": {"modality_keys": ["cam"], "delta_indices": [0, 1]},
                "state": {"modality_keys": ["grip", "pose", "joints"]},
                   "action": {"delta_indices": [0, 1], "modality_keys": ["grip", "pose", "joints"],
                              "action_configs": [
                                  {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "grip"},
                                  {"rep": "RELATIVE", "type": "EEF", "format": "XYZ_ROT6D", "state_key": "pose"},
                                  {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT",
                                   "state_key": "joints"}]}},
    },
}
STATS = {
    "bot": {
        "state": {
            "grip": {"q01": [5.0], "q99": [5.0], "min": [0.0], "max": [10.0]},
            "arm": {"q01": [0.0, -1.0], "q99": [2.0, 1.0], "min": [-2.0, -3.0], "max": [4.0, 3.0]},
        },
        # the absolute stats also fix each group's decoded **width**, even for a
        # relative group, so "arm" needs them here as well as under relative_action
        "action": {
            "grip": {"q01": [5.0], "q99": [5.0], "min": [0.0], "max": [10.0]},
            "arm": {"q01": [0.0, -1.0], "q99": [2.0, 1.0], "min": [-2.0, -3.0], "max": [4.0, 3.0]},
        },
        # a relative group reads min/max off *these*, never the percentiles above
        "relative_action": {
            "arm": {"q01": [-0.5, -0.5], "q99": [0.5, 0.5], "min": [-1.0, -2.0], "max": [1.0, 2.0]},
        },
    },
    "other": {
        "state": {"j": {"q01": [0.0], "q99": [1.0], "min": [0.0], "max": [1.0]}},
        "action": {"j": {"q01": [0.0], "q99": [1.0], "min": [0.0], "max": [1.0]}},
    },
    # every bound a decode is allowed to read is [-1, 1] here, which makes the
    # unnormalise the identity and leaves the decode's *slicing* and its SE(3)
    # arithmetic as the only things a value can be wrong about. The bounds a
    # decode must **not** read are deliberately elsewhere: "pose"'s percentiles
    # are [0, 10], so a percentile read of this relative group lands nowhere near.
    "eefbot": {
        "state": {
            "grip": {"q01": [-1.0], "q99": [1.0]},
            "pose": {"q01": [-1.0] * 9, "q99": [1.0] * 9},
            "joints": {"q01": [-1.0, -1.0], "q99": [1.0, 1.0]},
        },
        "action": {
            "grip": {"q01": [-1.0], "q99": [1.0], "min": [0.0], "max": [10.0]},
            "pose": {"q01": [0.0] * 9, "q99": [10.0] * 9, "min": [0.0] * 9, "max": [10.0] * 9},
            "joints": {"q01": [-1.0, -1.0], "q99": [1.0, 1.0], "min": [0.0, 0.0], "max": [4.0, 4.0]},
        },
        "relative_action": {
            "pose": {"q01": [0.0] * 9, "q99": [4.0] * 9, "min": [-1.0] * 9, "max": [1.0] * 9},
        },
    },
}
IDS = {"bot": 7, "other": 2, "eefbot": 3, "unused": 9}
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

@pytest.mark.skipif(not GR00T.exists(), reason="checkpoint not cached")
def test_the_checkpoint_loads():
    pre = Gr00tN17Preprocessor.from_pretrained(GR00T, max_state_dim=132, max_action_horizon=40)
    assert len(pre.state_keys) == 8
    tag = "oxe_droid_relative_eef_relative_joint"
    assert pre.embodiment_id(tag, 2).tolist() == [24, 24]
    assert pre.action_mask(tag, 1).sum() == 40
    # the reference's short enum name is not a checkpoint tag
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


@needs_checkpoint
def test_call_with_the_real_processor():
    pytest.importorskip("cv2")
    pre = Gr00tN17Preprocessor.from_pretrained(GR00T, 132, 40, backbone_path=BACKBONE)
    tag = "oxe_droid_relative_eef_relative_joint"
    video = {k: np.zeros((2, 2, 180, 320, 3), np.uint8) for k in pre.video_keys[tag]}
    state = {k: np.zeros((2, 1, lo.size)) for k, (lo, _) in pre.state_bounds[tag].items()}
    out = pre(tag, video, ["pick up the red cup and put it in the bowl", "stop"], state)
    L = out["input_ids"].shape[1]
    assert out["input_ids"].shape == out["attention_mask"].shape == (2, L)
    assert out["attention_mask"][1, 0] == 0 and out["attention_mask"][:, -1].tolist() == [1, 1]  # left padding
    assert out["image_grid_thw"].tolist() == [[1, 16, 28]] * 8
    assert out["pixel_values"].shape[0] == 8 * 448


# ----------------------------------------------------------------------------
# The decode: `unapply`. An inverse of the wrong thing still returns plausible commands
# of the right shape, so each test pins one choice and, where tempting, the wrong answer.
# ----------------------------------------------------------------------------

# (9,) XYZ_ROT6D poses (position, then a rotation matrix's first two rows): the identity
# frame, and a quarter turn about z at (1, 2, 3).
_IDENTITY_POSE = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
_TURNED_POSE = [1.0, 2.0, 3.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0]


def _pose_chunk(xyz, rot6d):
    """An ``eefbot`` chunk (1, 2, 12) asking for this relative pose twice; its relative
    bounds are [-1, 1], so the unnormalise is the identity and the value is the pose."""
    chunk = np.zeros((1, 2, 12), dtype=np.float32)
    chunk[0, :, 1:10] = [*xyz, *rot6d]
    return chunk


@pytest.mark.parametrize("x, relative, percentile", [
    (-1.0, [-1.0, -2.0], [0.0, -1.0]),
    (0.0, [0.0, 0.0], [1.0, 0.0]),
    (1.0, [1.0, 2.0], [2.0, 1.0]),
])
def test_a_relative_group_unnormalises_against_the_relative_action_bounds(x, relative, percentile):
    """A relative group's bounds are ``relative_action``'s min/max, not the ``action``
    q01/q99 ``use_percentiles`` picks for the rest: the reference's override replaces the
    group's stats (``state_action_processor.py:195-204``). The wrong read raises nothing and
    on DROID rescales each command by 0.03x-3.8x. Zero state, so only the bounds are in it."""
    out = _pre().unapply("bot", np.full((1, 3, 3), x, np.float32), {"arm": np.zeros((1, 1, 2), np.float32)})
    assert out["arm"][0, 0].tolist() == relative
    # `percentile` is the same unnormalise off `action`'s q01/q99, derived here so
    # the negative control follows the fixture instead of being a second literal
    q01, q99 = np.array(STATS["bot"]["action"]["arm"]["q01"]), np.array(STATS["bot"]["action"]["arm"]["q99"])
    assert percentile == ((x + 1) / 2 * (q99 - q01) + q01).tolist() != relative


def test_a_relative_groups_bounds_may_be_one_per_timestep():
    """``relative_action`` bounds are (T, D) in the real checkpoint (DROID ``eef_9d``:
    (40, 9)) and broadcast over the batch, not time (``utils.py:131-146``). Steps 0 and 1
    get different bounds, so a decode that read only the first row returns 1.0 for step 1."""
    stats = copy.deepcopy(STATS)
    stats["eefbot"]["relative_action"]["pose"] = {"min": [[-1.0] * 9, [-2.0] * 9], "max": [[1.0] * 9, [2.0] * 9]}
    pre = Gr00tN17Preprocessor(KWARGS, stats, IDS, 12, 6)
    # +1 on every dimension -> each step's own `max`, so (1, ..) then (2, ..)
    chunk = np.ones((1, 2, 12), np.float32)
    out = pre.unapply("eefbot", chunk, {"pose": np.array([[_IDENTITY_POSE]], np.float32)})
    assert out["pose"][0, 0, :3].tolist() == [1.0, 1.0, 1.0]
    assert out["pose"][0, 1, :3].tolist() == [2.0, 2.0, 2.0]


@pytest.mark.parametrize("x, minmax_would_give", [(-1.0, 0.0), (0.0, 5.0), (1.0, 10.0), (100.0, 10.0)])
def test_an_absolute_groups_constant_dimension_decodes_to_the_constant(x, minmax_would_give):
    """``grip`` has q01 == q99 == 5 and min/max 0/10. An absolute group reads the
    percentiles, so it decodes to 5 whatever arrives (min/max gives the second column), and
    the 1e-8 range floor (``state_action_processor.py:169``) is not carried into the inverse.
    ``x = 100`` is the clip (``utils.py:146``)."""
    out = _pre().unapply("bot", np.full((1, 3, 3), x, np.float32), {"arm": np.zeros((1, 1, 2), np.float32)})
    assert out["grip"].tolist() == [[[5.0]] * 3]
    # the min/max read, derived off the fixture's own min/max: a whole-range answer
    # at three of the four inputs, and the floored one would be 5 + 5e-09 at all four
    lo, hi = STATS["bot"]["action"]["grip"]["min"][0], STATS["bot"]["action"]["grip"]["max"][0]
    assert minmax_would_give == (np.clip(x, -1.0, 1.0) + 1.0) / 2.0 * (hi - lo) + lo


def test_each_joint_group_is_read_at_its_own_offset():
    """``eefbot``'s groups are 1, 9 and 2 wide (offsets 0, 1, 10), every column holds a
    different number and every bound is [-1, 1], so each group must come back as exactly its
    own slice. The identity frame makes the SE(3) translation a copy; the NaN columns past
    12 are padding nothing may read."""
    v = np.array([(j + 1) / 16 for j in range(12)], np.float32)
    chunk = np.full((1, 2, 14), np.nan, np.float32)
    chunk[0, 0, :12], chunk[0, 1, :12] = v, -v
    out = _pre().unapply("eefbot", chunk, {"pose": np.array([[_IDENTITY_POSE]], np.float32)})
    assert np.array_equal(out["grip"], [[[v[0]], [-v[0]]]])
    assert np.array_equal(out["pose"][..., :3], [[v[1:4], -v[1:4]]])
    assert np.array_equal(out["joints"], [[v[10:12], -v[10:12]]])


def test_the_horizon_and_the_width_are_cut_before_anything_reads_them():
    """The decoder's chunk is 132 wide over 40 steps for every embodiment; NaN padding
    makes a read one step or one column too far return NaN, not a plausible number."""
    chunk = np.full((2, 6, 7), np.nan, np.float32)
    chunk[:, :3, :3] = 0.0
    out = _pre().unapply("bot", chunk, {"arm": np.zeros((2, 1, 2), np.float32)})
    assert [v.shape for v in out.values()] == [(2, 3, 2), (2, 3, 1)]
    assert all(np.isfinite(v).all() for v in out.values())


@pytest.mark.parametrize("shape", [(1, 2, 3), (1, 3, 2), (1, 0, 0)])
def test_a_chunk_too_short_or_too_narrow_for_the_embodiment_is_refused(shape):
    """Numpy slices past the end silently: a narrow group shifts every later group's
    offset, and a short chunk returns fewer steps than the robot asked for."""
    with pytest.raises(ValueError, match="decodes 3 dims over 3 steps"):
        _pre().unapply("bot", np.zeros(shape, np.float32), {"arm": np.zeros((1, 1, 2), np.float32)})


def test_a_chunk_that_is_not_batched_is_refused():
    """A ``(T, D)`` chunk read as ``(B, T)`` would decode the horizon as the batch;
    ``postprocess`` adds the batch axis, so ``unapply`` refuses one without it."""
    with pytest.raises(ValueError, match=r"wants \(B, T, D\)"):
        _pre().unapply("bot", np.zeros((3, 3), np.float32), {"arm": np.zeros((1, 1, 2), np.float32)})


def test_a_joint_group_with_no_absolute_statistics_is_refused_and_named():
    """A group's decoded width comes from its absolute stats even when it is relative, so
    empty ones leave nothing to decode into. The reference fails one step later, on a numpy
    broadcast; ours names the group, which is the operator's cue to pick another tag."""
    stats = copy.deepcopy(STATS)
    stats["bot"]["action"]["arm"] = {"q01": [], "q99": [], "min": [], "max": []}
    pre = Gr00tN17Preprocessor(KWARGS, stats, IDS, 5, 6)
    with pytest.raises(ValueError, match="no action statistics for joint group 'arm'"):
        pre.unapply("bot", np.zeros((1, 3, 3), np.float32), {"arm": np.zeros((1, 1, 2), np.float32)})


@pytest.mark.skipif(not GR00T.exists(), reason="checkpoint not cached")
def test_the_checkpoints_xdof_embodiments_cannot_be_decoded_at_all():
    """On disk: both xdof embodiments ship empty ``action`` stats for the wrist eef groups
    beside a full (40, 9) ``relative_action``, so they cannot be served; the other six decode."""
    pre = Gr00tN17Preprocessor.from_pretrained(GR00T, max_state_dim=132, max_action_horizon=40)
    broken = {tag for tag, dims in pre.action_dims.items() if 0 in dims.values()}
    assert broken == {"xdof_relative_eef_relative_joint", "xdof_relative_eef_relative_joint_subtask"}
    for tag in sorted(broken):
        with pytest.raises(ValueError, match="no action statistics for joint group 'left_wrist_eef'"):
            pre.unapply(tag, np.zeros((1, 40, 132), np.float32), None)
    # and the other six decode: 9 + 1 + 7 for DROID, over its own 40-step horizon.
    # `eef_9d`'s reference frame has to be a real pose, so it is the identity.
    droid = "oxe_droid_relative_eef_relative_joint"
    state = {k: np.zeros((1, 1, d), np.float32) for k, d in pre.action_dims[droid].items()}
    state["eef_9d"] = np.array([[_IDENTITY_POSE]], np.float32)
    out = pre.unapply(droid, np.zeros((1, 40, 132), np.float32), state)
    assert [v.shape for v in out.values()] == [(1, 40, 9), (1, 40, 1), (1, 40, 7)]


def test_a_relative_non_eef_group_adds_the_states_last_timestep():
    """A joint group adds the raw state's **last** timestep (``action_chunking.py:383-387``).
    The state moves between its steps and the chunk rows differ, so reading the first step
    or broadcasting one action step is caught."""
    chunk = np.zeros((1, 3, 3), np.float32)
    chunk[0, :, 0] = [1.0, -1.0, 0.0]  # arm dim 0 -> relative +1, -1, 0
    state = {"arm": np.array([[[10.0, 0.0], [20.0, 0.0]]], np.float32)}
    out = _pre().unapply("bot", chunk, state)
    assert out["arm"][0, :, 0].tolist() == [21.0, 19.0, 20.0]


def test_an_eef_group_composes_the_reference_frame_rather_than_adding_it():
    """``XYZ_ROT6D`` poses compose, ``T_abs = T_ref @ T_rel`` (``action_chunking.py:641-653``),
    which differs from adding whenever the frame is rotated: a metre along x under a quarter
    turn about z lands at (1, 3, 3); added, it lands at (2, 2, 3) with a rot6d that is not a
    rotation."""
    chunk = _pose_chunk([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    out = _pre().unapply("eefbot", chunk, {"pose": np.array([[_TURNED_POSE]], np.float32)})
    composed = [1.0, 3.0, 3.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0]
    added = (np.array(_TURNED_POSE) + np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])).tolist()
    assert np.array_equal(out["pose"], [[composed] * 2])
    assert composed != added


def test_the_eef_reference_frame_is_the_states_last_timestep():
    """The frame is the state's ``s[-1]`` (``state_action_processor.py:511-513``). The
    recorded requests carry one timestep, so only this test tells it from ``s[0]``."""
    chunk = _pose_chunk([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    pre = _pre()
    both = np.array([[_IDENTITY_POSE, _TURNED_POSE]], np.float32)
    assert np.array_equal(
        pre.unapply("eefbot", chunk, {"pose": both})["pose"],
        pre.unapply("eefbot", chunk, {"pose": np.array([[_TURNED_POSE]], np.float32)})["pose"],
    )
    assert not np.array_equal(
        pre.unapply("eefbot", chunk, {"pose": both})["pose"],
        pre.unapply("eefbot", chunk, {"pose": np.array([[_IDENTITY_POSE]], np.float32)})["pose"],
    )


def test_an_eef_decode_is_the_same_whether_the_state_came_over_json_or_not():
    """JSON delivers the state as float64; the reference's frame Gram-Schmidt runs in
    float32 (``gr00t_policy.py:319-321``). On this non-orthonormal frame a float64 frame
    moves ``eef_9d`` by 1.5e-8, with no error."""
    pose = {"pose": np.array([[[0.1, 0.2, 0.3, 0.3, 0.7, 0.1, -0.7, 0.3, 0.2]]], np.float32)}
    wire = json.loads(json.dumps({k: v.tolist() for k, v in pose.items()}))
    chunk = _pose_chunk([0.1, 0.2, 0.3], [0.9, 0.1, 0.1, 0.1, 0.9, 0.1])
    pre = _pre()
    assert np.array_equal(pre.unapply("eefbot", chunk, wire)["pose"], pre.unapply("eefbot", chunk, pose)["pose"])


@pytest.mark.parametrize("state, match", [
    (None, "raw state is required"),
    ({"grip": np.zeros((1, 1, 1), np.float32)}, "raw state is required"),
    ({"arm": np.zeros((1, 1, 3), np.float32)}, "does not match the action's"),
    ({"arm": np.zeros((2, 1, 2), np.float32)}, "does not match the action's"),
])
def test_a_relative_group_without_a_usable_state_is_refused(state, match):
    """``state`` is optional for all-absolute embodiments, so a relative one must refuse a
    missing or mismatched one itself. Each case can come from a client; each would otherwise
    be a numpy broadcast, and a (1, 1, 1) state broadcasts silently onto every joint."""
    with pytest.raises(ValueError, match=match):
        _pre().unapply("bot", np.zeros((1, 3, 3), np.float32), state)


@pytest.mark.parametrize("rot6d, frame, match", [
    ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0] * 9, "does not define a direction"),
    ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0], "two rows are parallel"),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], _TURNED_POSE, "does not define a direction"),
    ([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], _TURNED_POSE, "two rows are parallel"),
])
def test_a_degenerate_rotation_is_refused_rather_than_decoded_to_nan(rot6d, frame, match):
    """The frame (client state) and the relative rotation (decoder output) both go through
    Gram-Schmidt and neither is guaranteed a rotation. A zero first row, or a second row
    parallel to the first, divides by zero and gives a NaN pose (the reference raises from
    scipy instead), so both are refused."""
    with pytest.raises(ValueError, match=match):
        _pre().unapply("eefbot", _pose_chunk([1.0, 0.0, 0.0], rot6d), {"pose": np.array([[frame]], np.float32)})


@pytest.mark.skipif(not (GR00T.exists() and len(_DUMPS) == 20), reason="checkpoint or the 20 requests missing")
def test_the_decode_matches_the_reference_on_the_twenty_recorded_droid_requests():
    """Our decode against the reference's own ``decoded_action`` on 20 recorded DROID
    requests, byte-identical inputs. Measured bars, never to be loosened:

    * ``gripper_position`` and ``joint_position``: **bitwise equal**. Neither path has a
      float the reference's lacks, so anything less is a defect.
    * ``eef_9d``: within one float32 ulp of the reference frame. numpy 1.26 and 2.5 disagree
      on a float32 3-vector ``np.dot`` (0.0 vs 3.7e-9), which moves one request by 3.3e-9;
      The 3.3e-9 asserted is this checkpoint's instance of that bound and stays the bar;
      the median bar (1e-15) keeps a regression on the typical request from hiding behind it.
    """
    pre = Gr00tN17Preprocessor.from_pretrained(GR00T, max_state_dim=132, max_action_horizon=40)
    eef = []
    for path in _DUMPS:
        req = torch.load(path, map_location="cpu", weights_only=False)
        ours = pre.unapply(req["embodiment_tag"], req["action_pred"].float().numpy(), req["raw_obs"]["state"])
        ref = req["decoded_action"]
        assert list(ours) == list(ref) == ["eef_9d", "gripper_position", "joint_position"]
        assert [v.shape for v in ours.values()] == [ref[k].shape for k in ours]
        assert np.array_equal(ours["gripper_position"], ref["gripper_position"]), path
        assert np.array_equal(ours["joint_position"], ref["joint_position"]), path
        eef.append(float(np.abs(ours["eef_9d"] - ref["eef_9d"]).max()))
    assert max(eef) <= 3.3e-09, eef
    assert float(np.median(eef)) <= 1e-15, eef
