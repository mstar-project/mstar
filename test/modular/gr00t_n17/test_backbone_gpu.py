"""GR00T N1.7's backbone on the GPU, one stage at a time: ``vision_encoder``, ``LLM`` and
``vl_bridge``, each our served node in bf16 against NVIDIA's reference run in fp32.

The recordings are NVIDIA's reference run: its stage outputs at seed 7 on 20 DROID requests, in
fp32 (``~/gr00t-runs/oracle_fp32_seed7``) and bf16 (``oracle_bf16_seed7``), plus the requests'
inputs (``~/gr00t-runs/dump``). The floor is the bf16 run against the fp32 one, measured per
request; the bar is **mean abs error within 1.2x of it**. Never max abs (a handful of elements
dominates it), never a picked constant. Each pass has a control that must land past the bar.

Traps that would give a test that passes and says nothing:

1. Each stage is handed the bf16 run's own input to that stage, so the ratio does not also read
   the gap between two upstream stacks. The vision stage reads the dump's ``pixel_values``.
2. Each node is built by ``Gr00tN17Model.get_submodule`` (strict load), cast and run under
   ``autocast_scope`` as ``engine_manager.py`` and ``engine.py`` do, both skipped for a node that
   sets ``disable_autocast`` (the bridge). Autocast runs LayerNorm in fp32, so it changes numbers.
3. Attention is a per-segment SDPA stand-in planned from the node's ``declare_step``; the ragged
   kernels are not exercised, so a kernel swap does not blur the stage bar.
4. One node on the device at a time, per class-scoped fixture. The LLM is built in fp32 on the
   host first, so it briefly needs several GB of host RAM.

A stage whose tensors the recordings lack skips and names them. Besides ``vl_embeds`` (the
bridge's output) it reads ``backbone_features``, the LLM's output and the input of the action
head's ``vlln`` (``gr00t_n1d7.py:175``), ``[1, n, 2048]``; and ``vision_embeds`` and
``deepstack_embeds``, Qwen3-VL's ``visual`` output ``pooler_output`` ``[448, 2048]`` and its three
``deepstack_features`` stacked to ``[3, 448, 2048]``, read in a forward hook on ``visual`` because
``get_image_features`` replaces both with per-image splits once the call returns.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_runner import autocast_scope
from mstar.engine.resources.step import SubmoduleStep
from mstar.model.gr00t_n17.config import BRIDGE_ATTN, LLM_ATTN, Gr00tN17Config
from mstar.model.gr00t_n17.gr00t_n17_model import LLM, VISION_ENCODER, VL_BRIDGE, Gr00tN17Model
from mstar.model.qwen3_5.config import VISION_ATTN
from mstar.model.submodule_base import ModelInputsFromEngine, NodeSubmodule

from .helpers import BACKBONE, GR00T, SegmentAttention

_RUNS = Path.home() / "gr00t-runs"
_ORACLE_FP32, _ORACLE_BF16 = _RUNS / "oracle_fp32_seed7", _RUNS / "oracle_bf16_seed7"
_DUMP = _RUNS / "dump"
_REQUESTS = 20

# What each stage reads off the two runs; the batched two are recorded with a batch dim of 1.
_VISION_KEYS = ("vision_embeds", "deepstack_embeds")
_LLM_KEY = "backbone_features"
_BRIDGE_KEY = "vl_embeds"
_BATCHED = frozenset({_LLM_KEY, _BRIDGE_KEY})

_DEVICE = torch.device("cuda:0")
_BF16 = torch.bfloat16

_BAR = 1.2  # a multiple of the floor measured here, never an absolute tolerance
# Two requests, not 20: a control only has to show the bar can fail. One per (text rows, image
# rows) bucket, the two `test_action_gen_gpu.py` anchors on.
_CONTROL_REQUESTS = (0, 8)
_TAP_SHIFTS = (-1, 1)

_WALK = "prefill"
_FWD = CurrentForwardPassInfo(request_id="r0", graph_walk=_WALK, fwd_index=0, random_seed=0, max_tokens=0)
_ENGINE = ModelInputsFromEngine(request_ids=["r0"], per_request_info={})


def _complete(directory: Path) -> bool:
    """All 20 requests, counted: a partial set would quietly shrink the sample."""
    return directory.is_dir() and len(list(directory.glob("req_*.pt"))) == _REQUESTS


def _recorded_keys() -> frozenset[str]:
    """The tensors **every** file of both oracles holds. Memory-mapped, so nothing is read."""
    if not (_complete(_ORACLE_FP32) and _complete(_ORACLE_BF16)):
        return frozenset()
    return frozenset.intersection(*(
        frozenset(torch.load(path, map_location="cpu", mmap=True, weights_only=True))
        for directory in (_ORACLE_FP32, _ORACLE_BF16)
        for path in sorted(directory.glob("req_*.pt"))
    ))


_RECORDED = _recorded_keys()

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and GR00T.exists() and bool(BACKBONE) and _complete(_DUMP)),
    reason=f"the backbone stages need CUDA, the cached checkpoint and the {_REQUESTS} dumps in {_DUMP}",
)


def _needs(*keys: str) -> pytest.MarkDecorator:
    missing = [key for key in keys if key not in _RECORDED]
    return pytest.mark.skipif(
        bool(missing),
        reason=(
            f"{_ORACLE_BF16.name} and {_ORACLE_FP32.name} do not both record {missing} on all "
            f"{_REQUESTS} requests; the module docstring says what they must hold"
        ),
    )


# ----------------------------------------------------------------------------
# One recorded request
# ----------------------------------------------------------------------------


@dataclass
class _Request:
    """One recorded DROID request: NVIDIA's inputs, and both runs' tensors at one stage,
    memory-mapped. ``fp32`` is the answer; ``bf16`` is the floor's other half and, by trap 1,
    what the next stage is handed."""

    index: int
    input_ids: torch.Tensor  # [n]
    grid: torch.Tensor  # [frames, 3]
    pixel_values: torch.Tensor  # [patches, 1536], what both oracles were handed
    fp32: dict[str, torch.Tensor]
    bf16: dict[str, torch.Tensor]

    def is_image(self, config: Gr00tN17Config) -> torch.Tensor:
        """The reference's own predicate (``qwen3_backbone.py:364``)."""
        return self.input_ids == config.text.image_token_id


def _load(index: int, keys: tuple[str, ...]) -> _Request:
    name = f"req_{index:02d}.pt"
    dump = torch.load(_DUMP / name, map_location="cpu", mmap=True, weights_only=False)["collated"]["inputs"]
    sides = {}
    for side, directory, dtype in (("fp32", _ORACLE_FP32, torch.float32), ("bf16", _ORACLE_BF16, _BF16)):
        recorded = torch.load(directory / name, map_location="cpu", mmap=True, weights_only=True)
        sides[side] = {key: recorded[key][0] if key in _BATCHED else recorded[key] for key in keys}
        # a run recorded at the wrong dtype would make the floor zero, or the answer a rounding
        wrong = {key: t.dtype for key, t in sides[side].items() if t.dtype != dtype}
        assert not wrong, f"{directory.name}/{name}: {wrong}, expected {dtype}"
    return _Request(
        index=index,
        input_ids=dump["input_ids"].reshape(-1),
        grid=dump["image_grid_thw"],
        pixel_values=dump["pixel_values"],
        **sides,
    )


# ----------------------------------------------------------------------------
# The bar
# ----------------------------------------------------------------------------


def _error_and_floor(ours: torch.Tensor, want: torch.Tensor, rounded: torch.Tensor) -> tuple[float, float]:
    """Mean abs error of ours against the fp32 run, and the floor: the bf16 run against it."""
    assert ours.shape == want.shape, (tuple(ours.shape), tuple(want.shape))
    want = want.float()
    floor = (rounded.float() - want).abs().mean().item()
    assert floor > 0, "the two runs agree exactly, so they are not a bf16 run and an fp32 run"
    return (ours.float().cpu() - want).abs().mean().item(), floor


def _within_bar(ours: torch.Tensor, want: torch.Tensor, rounded: torch.Tensor, what: str) -> None:
    error, floor = _error_and_floor(ours, want, rounded)
    assert error <= _BAR * floor, (
        f"{what}: mean abs error {error:.3e} past the bar {_BAR * floor:.3e} "
        f"({_BAR}x its measured bf16 floor {floor:.3e}; {error / floor:.2f}x)"
    )


def _past_bar(ours: torch.Tensor, want: torch.Tensor, rounded: torch.Tensor, what: str) -> None:
    """A control: the bar has to reject it, or the pass beside it measures nothing."""
    error, floor = _error_and_floor(ours, want, rounded)
    assert error > _BAR * floor, (
        f"{what}: the control's mean abs error {error:.3e} is inside the bar {_BAR * floor:.3e} "
        f"({error / floor:.2f}x the floor {floor:.3e}), so the bar cannot see this mistake"
    )


# ----------------------------------------------------------------------------
# Building and driving a node
# ----------------------------------------------------------------------------


@dataclass
class _Built:
    """A holder, so the fixture's teardown can drop the node while pytest still holds the value."""

    node: NodeSubmodule | None
    config: Gr00tN17Config


def _build(name: str) -> Iterator[_Built]:
    """The node as the server builds it (trap 2). The model is dropped once the node is out:
    ``get_submodule`` caches every node it builds, which would keep this one alive past the class."""
    model = Gr00tN17Model(model_path_hf=str(GR00T), backbone_path_hf=BACKBONE)
    node = model.get_submodule(name, device=str(_DEVICE), autocast_dtype=_BF16)
    # an opted-out node keeps its built dtype (`engine_manager.py`)
    cast = {} if node.disable_autocast else {"dtype": _BF16}
    built = _Built(node=node.to(device=_DEVICE, **cast), config=model.config)
    del model, node
    gc.collect()
    yield built
    built.node = None
    gc.collect()
    torch.cuda.empty_cache()


class _SegmentSdpa(SegmentAttention):
    """The ragged attention resource's stand-in (trap 3), planned from the node's declared step."""

    def __init__(self, step: SubmoduleStep, key: str):
        super().__init__(tuple(segment.span for segment in step.segments), causal=step.steps[key].causal)

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        assert q.shape[0] == sum(self.seq_lengths), (q.shape[0], self.seq_lengths)
        # `enable_gqa` only where KV heads are grouped, as transformers passes it
        self.gqa = q.shape[1] != k.shape[1]
        return super().run(q, k, v)


def _bind(node: NodeSubmodule, key: str, step: SubmoduleStep) -> None:
    node.bind_node_resources({key: _SegmentSdpa(step, key)})


def _staged(node: NodeSubmodule, key: str, inputs: dict) -> dict:
    """``prepare_inputs`` -> ``declare_step`` -> its resource -> ``preprocess``, as the engine runs them."""
    prepared = node.prepare_inputs(_WALK, _FWD, inputs)
    _bind(node, key, node.declare_step(_WALK, ["r0"], [prepared]))
    return node.preprocess(_WALK, _ENGINE, [prepared])


def _forward(node: NodeSubmodule, tensors: dict) -> dict:
    """The forward, under the scope the engine runs it in (``engine.py``'s ``autocast_scope``)."""
    with torch.no_grad(), autocast_scope(None if node.disable_autocast else _BF16, _DEVICE.type):
        return node.forward(_WALK, _ENGINE, **tensors)


def _vision_inputs(request: _Request) -> dict:
    return {"pixel_values": [request.pixel_values], "image_grid_thw": [request.grid]}


def _llm_inputs(request: _Request) -> dict:
    """The prompt, and the bf16 run's own vision outputs (trap 1)."""
    return {
        "input_ids": [request.input_ids],
        "image_grid_thw": [request.grid],
        "vision_embeds": [request.bf16["vision_embeds"]],
        "deepstack_embeds": list(request.bf16["deepstack_embeds"].unbind(0)),
    }


def _bridge_inputs(request: _Request, config: Gr00tN17Config) -> dict:
    """The bf16 run's own LLM output (trap 1), and its image rows by the reference's predicate."""
    return {
        "backbone_features": [request.bf16[_LLM_KEY]],
        "image_rows": [request.is_image(config).nonzero(as_tuple=True)[0]],
    }


def _in_prompt_order(out: dict, is_image: torch.Tensor) -> torch.Tensor:
    """The bridge's two spans put back where the rows came from, the recordings' layout."""
    text, image = out["vl_text"][0], out["vl_image"][0]
    rows = text.new_empty(is_image.numel(), text.shape[-1])
    is_image = is_image.to(rows.device)
    rows[~is_image] = text
    rows[is_image] = image
    return rows


# ----------------------------------------------------------------------------
# vision_encoder
# ----------------------------------------------------------------------------


@_needs(*_VISION_KEYS)
class TestVisionEncoder:
    @pytest.fixture(scope="class")
    def built(self) -> Iterator[_Built]:
        yield from _build(VISION_ENCODER)

    def test_the_embeds_and_every_deepstack_feature_match_the_oracle(self, built: _Built):
        """The merged embeddings and the three DeepStack features, each at its own floor.

        The floors differ (measured 0.0199 for the embeds, 0.0026 / 0.0058 / 0.0120 for the
        features), so a pooled mean would let the embeds' floor cover a broken feature.
        """
        for index in range(_REQUESTS):
            request = _load(index, _VISION_KEYS)
            out = _forward(built.node, _staged(built.node, VISION_ATTN, _vision_inputs(request)))
            _within_bar(
                out["vision_embeds"][0], request.fp32["vision_embeds"], request.bf16["vision_embeds"],
                f"req {index} vision_embeds",
            )
            assert len(out["deepstack_embeds"]) == len(built.config.deepstack_visual_indexes)
            for tap, ours in enumerate(out["deepstack_embeds"]):
                _within_bar(
                    ours, request.fp32["deepstack_embeds"][tap], request.bf16["deepstack_embeds"][tap],
                    f"req {index} deepstack feature {tap}",
                )

    @pytest.mark.parametrize("shift", _TAP_SHIFTS)
    def test_taps_one_block_off_land_past_the_bar(self, built: _Built, shift: int):
        """The control: taps 5/11/17 moved to 4/10/16 or 6/12/18 land past the bar.

        Only the encoder's block-to-merger map moves. The merged embeds do not read the taps, so
        they must come back bit-identical: the control changed the taps and nothing else.
        """
        encoder = built.node.model
        taps = dict(encoder._merger_of)
        assert sorted(taps) == list(built.config.deepstack_visual_indexes)
        for index in _CONTROL_REQUESTS:
            request = _load(index, _VISION_KEYS)
            right = _forward(built.node, _staged(built.node, VISION_ATTN, _vision_inputs(request)))
            encoder._merger_of = {layer + shift: merger for layer, merger in taps.items()}
            try:
                wrong = _forward(built.node, _staged(built.node, VISION_ATTN, _vision_inputs(request)))
            finally:
                encoder._merger_of = taps
            assert torch.equal(wrong["vision_embeds"][0], right["vision_embeds"][0])
            for tap, ours in enumerate(wrong["deepstack_embeds"]):
                _past_bar(
                    ours, request.fp32["deepstack_embeds"][tap], request.bf16["deepstack_embeds"][tap],
                    f"req {index} deepstack feature {tap}, taps shifted {shift:+d}",
                )


# ----------------------------------------------------------------------------
# LLM
# ----------------------------------------------------------------------------


@_needs(*_VISION_KEYS, _LLM_KEY)
class TestLLM:
    @pytest.fixture(scope="class")
    def built(self) -> Iterator[_Built]:
        yield from _build(LLM)

    def test_the_backbone_features_match_the_oracle(self, built: _Built):
        """The LLM's output, the features the bridge reads, at the bar on all 20, in bf16 where it
        serves. Handed the bf16 run's vision outputs, the ratio reads the embedding, the scatter
        onto the image rows, M-RoPE, the DeepStack injection and the 16 causal layers.
        """
        for index in range(_REQUESTS):
            request = _load(index, (*_VISION_KEYS, _LLM_KEY))
            out = _forward(built.node, _staged(built.node, LLM_ATTN, _llm_inputs(request)))
            _within_bar(
                out["backbone_features"][0], request.fp32[_LLM_KEY], request.bf16[_LLM_KEY],
                f"req {index} backbone_features",
            )

    def test_deepstack_one_layer_late_lands_past_the_bar(self, built: _Built):
        """The control: feature ``i`` added after layer ``i + 1`` instead of layer ``i``.

        A zero feature goes in front of the three, so only the injection point moves. It is put on
        the staged tensors because ``prepare_inputs`` refuses a fourth tap.
        """
        for index in _CONTROL_REQUESTS:
            request = _load(index, (*_VISION_KEYS, _LLM_KEY))
            tensors = _staged(built.node, LLM_ATTN, _llm_inputs(request))
            features = tensors["deepstack_embeds"]
            tensors["deepstack_embeds"] = torch.cat([torch.zeros_like(features[:1]), features])
            _past_bar(
                _forward(built.node, tensors)["backbone_features"][0],
                request.fp32[_LLM_KEY], request.bf16[_LLM_KEY],
                f"req {index} backbone_features, DeepStack one layer late",
            )


# ----------------------------------------------------------------------------
# vl_bridge
# ----------------------------------------------------------------------------


@_needs(_LLM_KEY, _BRIDGE_KEY)
class TestVLBridge:
    @pytest.fixture(scope="class")
    def built(self) -> Iterator[_Built]:
        yield from _build(VL_BRIDGE)

    def test_the_conditioning_rows_match_the_oracle(self, built: _Built):
        """The bridge's two spans, back in prompt order, against the recorded ``vl_embeds``: the
        rows the DT cross-attends on every step."""
        for index in range(_REQUESTS):
            request = _load(index, (_LLM_KEY, _BRIDGE_KEY))
            out = _forward(built.node, _staged(built.node, BRIDGE_ATTN, _bridge_inputs(request, built.config)))
            _within_bar(
                _in_prompt_order(out, request.is_image(built.config)),
                request.fp32[_BRIDGE_KEY], request.bf16[_BRIDGE_KEY],
                f"req {index} vl_embeds",
            )

    def test_two_requests_in_one_segment_land_past_the_bar(self, built: _Built):
        """The control: two requests' rows packed into **one** attention segment.

        The bridge is the reference's one unmasked attention, so per-request segments are all that
        keeps a prompt from attending to its neighbour. The same packed rows run twice: in two
        segments each request must pass the bar, in one each must fail it (measured once at ~30x
        the floor). Only the segment layout differs.
        """
        a, b = (_load(index, (_LLM_KEY, _BRIDGE_KEY)) for index in _CONTROL_REQUESTS)
        inputs = [_bridge_inputs(request, built.config) for request in (a, b)]
        sizes = [request.input_ids.numel() for request in (a, b)]
        packed = {
            "backbone_features": [torch.cat([i["backbone_features"][0] for i in inputs])],
            "image_rows": [torch.cat([inputs[0]["image_rows"][0], inputs[1]["image_rows"][0] + sizes[0]])],
        }
        prepared = built.node.prepare_inputs(_WALK, _FWD, packed)
        steps = {
            "apart": built.node.declare_step(
                _WALK, ["a", "b"], [built.node.prepare_inputs(_WALK, _FWD, i) for i in inputs],
            ),
            "fused": built.node.declare_step(_WALK, ["ab"], [prepared]),
        }
        assert [s.span for s in steps["apart"].segments] == sizes
        assert [s.span for s in steps["fused"].segments] == [sum(sizes)]

        tensors = built.node.preprocess(_WALK, _ENGINE, [prepared])
        is_image = torch.cat([request.is_image(built.config) for request in (a, b)])
        rows = {}
        for layout, step in steps.items():
            _bind(built.node, BRIDGE_ATTN, step)
            rows[layout] = _in_prompt_order(_forward(built.node, tensors), is_image).split(sizes)
        for k, request in enumerate((a, b)):
            want, rounded = request.fp32[_BRIDGE_KEY], request.bf16[_BRIDGE_KEY]
            _within_bar(rows["apart"][k], want, rounded, f"req {request.index}, own segment")
            _past_bar(rows["fused"][k], want, rounded, f"req {request.index}, one segment with its neighbour")
