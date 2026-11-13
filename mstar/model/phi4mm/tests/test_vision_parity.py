"""GPU parity of the ``vision_encoder`` node against the reference
``Phi4MMImageEmbedding``, real weights, served bf16 on the ragged attention
resource.

The oracle (``$PHI4MM_ORACLE``) holds the
processor's outputs per case (``sdpa/<case>.pt``, ``record.py``) and the
reference's vision path from those inputs under several attention routes
(``vision/<route>/<case>.pt``, ``record_vision.py``):

- ``eager``: the remote code as the oracle venv runs it (no flash_attn), where
  a padded patch's query attends the valid keys; bitwise the ``sdpa`` recording
- ``fa2``: FlashAttention-2's semantics, the checkpoint's default and this
  port's: a padded patch's attention output is zero; same matmul kernels
- ``fa2_sdpa``: the same semantics on torch SDPA's kernels, the
  reference-against-itself spread
- ``fa2_fp32``: ``fa2`` in fp32

Compared at three points: the intake's pixels and token counts (exact), the
tower's features at the valid patches (``hidden_states[-2]``), and each image's
projected rows. Run with ``-s`` for the table.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F

from mstar.model.keys import VISION_ATTN
from mstar.model.multimodal import MEDIA_LAYOUT
from mstar.model.phi4mm import vision as vision_mod
from mstar.model.phi4mm.tests._paths import ORACLE, snapshot
from mstar.model.phi4mm.vision import (
    GLOBAL_PIXEL_VALUES,
    PIXEL_VALUES,
    VISION_OUTPUT,
    create_vision_submodule,
    image_intake,
    vision_resources,
)
from mstar.model.submodule_base import ModelInputsFromEngine

SNAPSHOT = snapshot(full=True) and str(snapshot(full=True) / "model.safetensors.index.json")
CASES = {
    "image_bagel": ["bagel"],
    "image_food_small": ["food_small"],
    "image_food_wide": ["food_wide"],
    "two_images": ["bagel", "food_small"],
    "image_audio": ["bagel"],
}
ROUTES = ("eager", "fa2", "fa2_sdpa", "fa2_fp32")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
    pytest.mark.skipif(SNAPSHOT is None, reason="the Phi-4-multimodal weights are not cached"),
    pytest.mark.skipif(
        ORACLE is None or not all((ORACLE / "vision" / r / f"{c}.pt").exists() for r in ROUTES for c in CASES),
        reason=f"no reference recordings under {ORACLE}/vision (record_vision.py)",
    ),
]


class Metrics(NamedTuple):
    max_abs: float
    rel: float
    cos: float

    def __str__(self):
        return f"max {self.max_abs:.3g}  rel {self.rel:.3g}  cos {self.cos:.6f}"


def metrics(got: torch.Tensor, want: torch.Tensor) -> Metrics:
    a, b = got.double().flatten(), want.double().flatten()
    d = a - b
    return Metrics(d.abs().max().item(), (d.norm() / b.norm()).item(), F.cosine_similarity(a, b, dim=0).item())


class SegmentSDPA:
    """Exact attention within each declared segment, for the fp32 run (the
    ragged kernels take bf16 / fp16 only)."""

    max_segments_per_step = 296

    def plan(self, step, ctx=None):
        self.lengths = [s.span for s in step.segments]

    def run(self, q, k, v, label=None):
        out, start = [], 0
        for n in self.lengths:
            sl = slice(start, start + n)
            out.append(F.scaled_dot_product_attention(
                q[sl].transpose(0, 1), k[sl].transpose(0, 1), v[sl].transpose(0, 1)).transpose(0, 1))
            start += n
        return torch.cat(out)


class Served(NamedTuple):
    node: object
    attn: object


def _served(dtype: torch.dtype) -> Served:
    from mstar.engine.resources.base import EngineResourceInfo, build_resource

    node = create_vision_submodule(str(Path(SNAPSHOT).parent), "cuda", dtype)
    if dtype == torch.float32:
        attn = SegmentSDPA()
    else:
        (spec,) = vision_resources(node.vision_config)
        attn = build_resource(spec, EngineResourceInfo(device=torch.device("cuda"), kv_dtype=dtype))
    node.bind_node_resources({VISION_ATTN: attn})
    return Served(node, attn)


@pytest.fixture(scope="module")
def bf16() -> Served:
    return _served(torch.bfloat16)


@pytest.fixture(scope="module")
def fp32():
    # the reference's fp32 run had TF32 off; mstar's torch config turns it on
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield _served(torch.float32)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


@lru_cache(maxsize=None)
def _load(route: str, case: str) -> dict:
    path = ORACLE / ("sdpa" if route == "oracle" else f"vision/{route}") / f"{case}.pt"
    return torch.load(path, weights_only=False, map_location="cpu")


def _intake(name: str, config):
    from mstar.model.media_limits import read_image

    # as `Model.load_image` hands it to the intake
    image = read_image(str(ORACLE / "media" / f"{name}.png"), rgb=True).float().div_(255.0)
    return image_intake(image, {}, config)


class StepOut(NamedTuple):
    rows: list[torch.Tensor]  # per request
    encoded: torch.Tensor     # the step's valid patches, packed


def _step(served: Served, requests: list[list[str]]) -> StepOut:
    node, attn = served
    from mstar.engine.resources import StepContext

    rids, infos, inputs = [], {}, []
    for i, names in enumerate(requests):
        prepared = [_intake(n, node.vision_config) for n in names]
        layout = {
            "offset": [0] * len(prepared), "tokens": [p.tokens[0] for p in prepared],
            "output": [VISION_OUTPUT] * len(prepared), "grid": [list(p.grids[0]) for p in prepared],
        }
        rid = f"r{i}"
        infos[rid] = SimpleNamespace(step_metadata={MEDIA_LAYOUT: layout})
        rids.append(rid)
        inputs.append(node.prepare_inputs("prefill_vision", infos[rid], {
            PIXEL_VALUES: [p.inputs[PIXEL_VALUES] for p in prepared],
            GLOBAL_PIXEL_VALUES: [p.inputs[GLOBAL_PIXEL_VALUES] for p in prepared],
        }))
    step = node.declare_step("prefill_vision", rids, inputs)
    attn.plan(step.get(VISION_ATTN), StepContext(request_ids=rids, graph_walk="prefill_vision", slot=0,
                                                  capture=False))
    engine = ModelInputsFromEngine(request_ids=rids, per_request_info=infos)
    with torch.inference_mode():
        kwargs = node.preprocess("prefill_vision", engine, inputs)
        out = node.forward_batched("prefill_vision", engine, **kwargs)
        tensors = {k: v for k, v in kwargs.items() if torch.is_tensor(v)}
        encoded = node.model.encode(*node.model.embed(**tensors))
    return StepOut([out[rid][VISION_OUTPUT][0].float().cpu() for rid in rids], encoded.float().cpu())


def _valid_features(record: dict) -> torch.Tensor:
    """The reference's ``hidden_states[-2]`` at the valid patches, in the node's
    pack order (each image's global view, then its crops; no padding crops)."""
    inputs = record["inputs"]
    mask = inputs["image_attention_mask"].bool()
    hidden = record["vision_hidden"][0].view(mask.shape[0], mask.shape[1], -1, record["vision_hidden"][0].shape[-1])
    out = []
    for i in range(mask.shape[0]):
        h, w = (inputs["image_sizes"][i] // 448).tolist()
        out.append(hidden[i, :1 + h * w][mask[i, :1 + h * w].flatten(1)])
    return torch.cat(out).float()


def _rows(record: dict) -> torch.Tensor:
    return torch.cat([e[0] for e in record["image_embeds"]]).float()


@pytest.mark.parametrize("case", list(CASES))
def test_intake_is_the_processor(case, bf16):
    """Pixels, patch masks and token counts, bit for bit, from the image file
    through the loader's decode."""
    inputs = _load("oracle", case)["inputs"]
    tokens = 0
    for i, name in enumerate(CASES[case]):
        prepared = _intake(name, bf16.node.vision_config)
        down, across, rows, cols = prepared.grids[0]
        ref = inputs["input_image_embeds"][i]
        assert inputs["image_sizes"][i].tolist() == [448 * down, 448 * across]
        assert torch.equal(prepared.inputs[GLOBAL_PIXEL_VALUES][0], ref[0])
        crops = prepared.inputs[PIXEL_VALUES].float().div(255).sub(0.5).div(0.5)
        assert torch.equal(crops, ref[1:1 + down * across])
        mask = inputs["image_attention_mask"][i, 1:1 + down * across]
        canvas = mask.view(down, across, 32, 32).permute(0, 2, 1, 3).reshape(32 * down, 32 * across)
        assert int(canvas[:, 0].sum()) == rows and int(canvas[0].sum()) == cols and int(canvas.sum()) == rows * cols
        tokens += prepared.tokens[0]
    assert tokens == int((inputs["input_ids"] == 200010).sum())
    # the device-side normalize rounds as the processor's host one does
    crops = prepared.inputs[PIXEL_VALUES].cuda()
    on_device = crops.float().div(torch.full((), 255.0, device="cuda")).sub_(0.5).div_(0.5)
    assert torch.equal(on_device.cpu(), ref[1:1 + down * across])


def _bound(got: Metrics, routes: list[Metrics]) -> bool:
    """Served is no farther from fp32 than the reference's own bf16 routes are."""
    return got.rel <= 1.25 * max(r.rel for r in routes) and got.cos >= min(r.cos for r in routes) - 1e-4


@pytest.mark.parametrize("case", list(CASES))
def test_features_and_rows_against_the_reference(case, bf16, capsys):
    """bf16 served against the FA2-semantics reference, both measured from the
    fp32 reference: served must be as close to it as the reference's own bf16
    kernel routes (``fa2``, ``fa2_sdpa``) are. Also printed: served against
    bf16 ``fa2``, beside ``fa2_sdpa`` against it (the spread)."""
    served = _step(bf16, [CASES[case]])
    fa2, other, truth = _load("fa2", case), _load("fa2_sdpa", case), _load("fa2_fp32", case)
    assert served.rows[0].shape == _rows(fa2).shape
    for point, served_t, pick in (("features", served.encoded, _valid_features), ("rows", served.rows[0], _rows)):
        got = metrics(served_t, pick(truth))
        routes = [metrics(pick(fa2), pick(truth)), metrics(pick(other), pick(truth))]
        with capsys.disabled():
            print(f"\n{case:17s} {point:8s} vs bf16 fa2: served {metrics(served_t, pick(fa2))}"
                  f" | fa2_sdpa {metrics(pick(other), pick(fa2))}")
            print(f"{'':17s} {point:8s} vs fp32 fa2: served {got} | fa2 {routes[0]} | fa2_sdpa {routes[1]}")
        assert _bound(got, routes), (point, got, routes)


@pytest.mark.parametrize("case", ["image_bagel", "image_food_wide"])
def test_fp32_is_the_reference_to_rounding(case, fp32, capsys):
    """Both sides in fp32 (exact attention here): the bf16 residue is rounding."""
    served = _step(fp32, [CASES[case]])
    ref = _load("fa2_fp32", case)
    hidden = metrics(served.encoded, _valid_features(ref))
    rows = metrics(served.rows[0], _rows(ref))
    with capsys.disabled():
        print(f"\n{case:17s} fp32 features {hidden}  | rows {rows}")
    assert hidden.rel < 1e-4 and rows.rel < 1e-4


@pytest.mark.parametrize("case", ["image_bagel", "image_food_small", "image_food_wide"])
def test_padded_patches_follow_flash_attention(case, bf16, capsys):
    """At the cells where the 2x2 pool mixes a padded patch into a kept row,
    the served rows are FA2's (as close to fp32 FA2 as the bf16 reference is),
    not eager attention's, which differs there by far more. Elsewhere the two
    semantics are bitwise equal."""
    served = _step(bf16, [CASES[case]]).rows[0]
    fa2, eager, other, truth = (_rows(_load(r, case)) for r in ("fa2", "oracle", "fa2_sdpa", "fa2_fp32"))
    mixed = (fa2 != eager).any(-1)
    assert 0 < int(mixed.sum()) < len(mixed)
    got = metrics(served[mixed], truth[mixed])
    routes = [metrics(fa2[mixed], truth[mixed]), metrics(other[mixed], truth[mixed])]
    served_eager = metrics(served[mixed], eager[mixed])
    with capsys.disabled():
        print(f"\n{case:17s} {int(mixed.sum())} mixed rows vs fp32 fa2: served {got} | fa2 {routes[0]}"
              f" | fa2_sdpa {routes[1]} | eager {metrics(eager[mixed], truth[mixed])}")
        print(f"{'':17s} served vs eager {served_eager}")
    assert _bound(got, routes), (got, routes)
    assert served_eager.rel >= 10 * max(r.rel for r in routes)
    # the padded patches' one feature, against every padded patch of the FA2 reference
    record = _load("fa2", case)
    padded = record["vision_hidden"][0].flatten(0, 1)[~record["inputs"]["image_attention_mask"].bool().flatten()]
    feature = bf16.node.model.masked_feature().float().cpu()
    assert (padded == padded[0]).all()
    assert metrics(feature, padded[0].float()).rel < 1e-2


@pytest.mark.parametrize("control", ["position_ids", "glb_sub"])
def test_negative_controls_diverge(control, bf16, monkeypatch, capsys):
    """The gate can fail: a partial crop's positions taken on the full grid
    (no NaViT bucketing), or the HD transform's views in ``glb_sub`` order
    rather than the checkpoint's ``sub_glb``."""
    case = "image_food_wide"
    truth = _rows(_load("fa2_fp32", case))
    routes = [metrics(_rows(_load(r, case)), truth) for r in ("fa2", "fa2_sdpa")]
    if control == "position_ids":
        def grid_ids(rows, cols, side):
            return (torch.arange(rows)[:, None] * side + torch.arange(cols)).flatten()

        monkeypatch.setattr(vision_mod, "navit_position_ids", grid_ids)
        vision_mod.image_plan.cache_clear()
    else:
        plan_of = vision_mod.image_plan
        glb = 1 + (16 * 17)  # glb_GN, then the global view's rows

        def glb_sub(grid, side):
            plan = plan_of(grid, side)
            pool = plan.pool_index
            return plan._replace(pool_index=torch.cat([pool[-glb + 1:], pool[-glb:-glb + 1], pool[:-glb]]))

        monkeypatch.setattr(vision_mod, "image_plan", glb_sub)
    try:
        got = metrics(_step(bf16, [CASES[case]]).rows[0], truth)
    finally:
        monkeypatch.undo()
        vision_mod.image_plan.cache_clear()
    with capsys.disabled():
        print(f"\ncontrol {control:12s} vs fp32 fa2: {got} | bf16 routes {routes[0]} / {routes[1]}")
    assert got.rel >= 3 * max(r.rel for r in routes)


def test_two_requests_in_one_step_are_each_alone(bf16, capsys):
    together = _step(bf16, [["bagel"], ["food_wide", "food_small"]]).rows
    alone = [_step(bf16, [["bagel"]]).rows[0], _step(bf16, [["food_wide", "food_small"]]).rows[0]]
    spread = metrics(_rows(_load("fa2_sdpa", "image_bagel")), _rows(_load("fa2", "image_bagel")))
    for got, want in zip(together, alone, strict=True):
        assert got.shape == want.shape
        m = metrics(got, want)
        with capsys.disabled():
            print(f"\npacked vs alone {m}")
        assert m.rel <= 0.1 * spread.rel
