"""Single-arm tiling device route and reference canvas checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mstar.model.praxis1 import submodules as port_sub
from mstar.model.praxis1.components.canvas import build_canvas, single_arm_needs_fit_resize


def _node(device: str = "cpu") -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            embodiment="single_arm", antialias=True,
            tile_height=224, tile_width=416,
        ),
        get_device=lambda: torch.device(device),
    )


@pytest.mark.parametrize(
    ("height", "width", "host_resize"),
    [(288, 512, True), (234, 416, False)],
)
def test_only_raw_antialiased_views_resize_before_device_transfer(
    monkeypatch: pytest.MonkeyPatch, height: int, width: int, host_resize: bool,
) -> None:
    events: list[str] = []

    class TrackedView:
        shape = (2, 3, height, width)

        def dim(self) -> int:
            return 4

        def cpu(self) -> "TrackedView":
            events.append("host")
            return self

        def to(self, device: torch.device) -> "TrackedView":
            assert device.type == "cuda"
            events.append("device")
            return self

    def tile(view: TrackedView, *_args, **_kwargs) -> TrackedView:
        events.append("tile")
        return view

    monkeypatch.setattr(port_sub, "tile_view_single_arm", tile)
    view = TrackedView()
    assert single_arm_needs_fit_resize(height, width, 416) is host_resize
    assert port_sub.Praxis1VaeEncoderSubmodule._tiles(_node("cuda"), [view]) == [view]
    assert events == (["host", "tile", "device"] if host_resize
                      else ["device", "tile", "device"])


@pytest.mark.parametrize(("height", "width"), [(288, 512), (234, 416)])
def test_single_arm_canvas_matches_reference_on_cpu(height: int, width: int) -> None:
    pytest.importorskip("praxis", reason="Praxis reference checkout is not on PYTHONPATH")
    from praxis.single_arm import tile_clip

    rng = np.random.default_rng(291)
    exterior = rng.integers(0, 256, size=(2, height, width, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(2, height, width, 3), dtype=np.uint8)
    views = [torch.from_numpy(x.copy()).permute(0, 3, 1, 2) for x in (exterior, wrist)]
    tiles = port_sub.Praxis1VaeEncoderSubmodule._tiles(_node(), views)
    actual = build_canvas(tiles)
    expected = tile_clip(exterior, wrist, 224, 416, device="cpu", antialias=True)
    assert torch.equal(actual, expected)
