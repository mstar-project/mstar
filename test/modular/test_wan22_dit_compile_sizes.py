"""CPU checks that the wan22 DiT compiles only for ``compile_dit_sizes``, and
that request sizes are bounded. A tiny stand-in transformer replaces the 5B
DiT: unlisted shapes never reach the compiled forward and match the eager
node exactly; a listed shape compiles at most once per CFG setting. No
weights, no GPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar.model.wan22.config import Wan22Config
from mstar.model.wan22.submodules import Wan22DitSubmodule
from mstar.model.wan22.wan22_model import Wan22Model

CHANNELS = 4
TEXT_LEN, TEXT_DIM = 3, 8
LISTED = ("64x32x5",)  # latent grid (2, 2, 4)


class _TinyTransformer(nn.Module):
    """Same call signature and output shape as the Wan 2.2 DiT."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(CHANNELS, CHANNELS, 1)
        self.text = nn.Linear(TEXT_DIM, CHANNELS)
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=gen) * 0.3)

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        text = self.text(encoder_hidden_states).mean(1)[:, :, None, None, None]
        t = timestep.float().mean(1)[:, None, None, None, None] * 1e-3
        return self.proj(hidden_states) + text + t


def _node(compile_dit: bool, sizes=LISTED) -> Wan22DitSubmodule:
    cfg = Wan22Config(in_channels=CHANNELS, compile_dit=compile_dit, compile_dit_sizes=sizes)
    return Wan22DitSubmodule(_TinyTransformer(), cfg)


def _step(node: Wan22DitSubmodule, width: int, height: int, frames: int, guidance: float) -> torch.Tensor:
    t_lat, h_lat, w_lat = (frames - 1) // 4 + 1, height // 16, width // 16
    gen = torch.Generator().manual_seed(width * 7 + height * 3 + frames)
    shape = (1, CHANNELS, t_lat, h_lat, w_lat)
    meta = {"num_inference_steps": 4, "guidance_scale": guidance}
    out = node.forward(
        "video_gen",
        engine_inputs=SimpleNamespace(single_request_info=SimpleNamespace(step_metadata=meta)),
        latents=torch.randn(shape, generator=gen),
        time_index=torch.tensor([1]),
        unipc_model_outputs=torch.randn((2, *shape), generator=gen),
        unipc_last_sample=torch.randn(shape, generator=gen),
        text_embeds_pos=torch.randn(1, TEXT_LEN, TEXT_DIM, generator=gen),
        text_embeds_neg=torch.randn(1, TEXT_LEN, TEXT_DIM, generator=gen),
    )
    return out["latents"][0]


def _stub_compile(monkeypatch) -> tuple[list[tuple], list[int]]:
    """Replace ``torch.compile`` with an eager stand-in recording each new input
    shape (what a ``dynamic=False`` graph recompiles on) and every call."""
    compiles, calls = [], []

    def compile(fn, **kwargs):
        def run(*, hidden_states, timestep, encoder_hidden_states):
            key = (tuple(hidden_states.shape), tuple(timestep.shape), tuple(encoder_hidden_states.shape))
            calls.append(1)
            if key not in compiles:
                compiles.append(key)
            return fn(hidden_states=hidden_states, timestep=timestep, encoder_hidden_states=encoder_hidden_states)

        return run

    monkeypatch.setattr(torch, "compile", compile)
    return compiles, calls


UNLISTED = [(96, 32, 5), (64, 64, 5), (64, 32, 9), (32, 32, 1)]


@pytest.mark.parametrize("guidance", [5.0, 1.0])
def test_unlisted_shapes_never_reach_the_compiled_forward(monkeypatch, guidance) -> None:
    compiles, calls = _stub_compile(monkeypatch)
    node, eager = _node(True), _node(False)
    for shape in UNLISTED:
        assert torch.equal(_step(node, *shape, guidance), _step(eager, *shape, guidance))
    assert calls == [] and compiles == []


def test_a_listed_shape_uses_the_compiled_forward(monkeypatch) -> None:
    compiles, calls = _stub_compile(monkeypatch)
    node, eager = _node(True), _node(False)
    assert torch.equal(_step(node, *(64, 32, 5), 5.0), _step(eager, *(64, 32, 5), 5.0))
    assert len(calls) == 1 and len(compiles) == 1


def test_cfg_on_and_off_at_a_listed_shape_stay_within_two_compiles(monkeypatch) -> None:
    compiles, calls = _stub_compile(monkeypatch)
    node = _node(True, sizes=("64x32x5", "32x64x9"))
    for _ in range(2):
        for shape in ((64, 32, 5), (32, 64, 9), *UNLISTED):
            for guidance in (5.0, 1.0):
                _step(node, *shape, guidance)
    assert len(calls) == 2 * 2 * 2
    assert len(compiles) == 2 * 2
    assert {key[0][0] for key in compiles} == {1, 2}  # CFG off and on


def test_a_swapped_in_transformer_runs_at_every_shape(monkeypatch) -> None:
    """The reference tests replace ``node.transformer`` with a recording wrapper;
    the saved eager forward belongs to the original and must not bypass it."""
    _stub_compile(monkeypatch)
    node = _node(True)
    original, seen = node.transformer, []

    class Recording(nn.Module):
        dtype = torch.float32

        def forward(self, **kwargs):
            seen.append(tuple(kwargs["hidden_states"].shape[2:]))
            return original.forward(**kwargs)

    node.transformer = Recording()
    for shape in ((64, 32, 5), *UNLISTED):
        _step(node, *shape, 5.0)
    assert len(seen) == 1 + len(UNLISTED)


def test_real_dynamo_compiles_nothing_for_unlisted_shapes(monkeypatch) -> None:
    """Real Dynamo guards with a counting backend: the eager forward of an
    unlisted shape adds no graph, and a listed shape adds one per CFG setting
    (the stand-in has no graph break, so one graph per compile)."""
    graphs = []

    def counting(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    real_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: real_compile(fn, **{**kw, "backend": counting}))
    torch._dynamo.reset()
    try:
        node, eager = _node(True), _node(False)
        for shape in UNLISTED:
            for guidance in (5.0, 1.0):
                assert torch.equal(_step(node, *shape, guidance), _step(eager, *shape, guidance))
        assert graphs == []
        for guidance in (5.0, 1.0, 5.0, 1.0):
            _step(node, *(64, 32, 5), guidance)
        assert len(graphs) == 2
        for shape in UNLISTED:
            _step(node, *shape, 5.0)
        assert len(graphs) == 2
    finally:
        torch._dynamo.reset()


def test_compile_dit_sizes_default_to_the_default_request_shape() -> None:
    assert Wan22Config().compile_dit_grids() == {(21, 44, 80)}
    model = Wan22Model(model_path_hf="unused", skip_weight_loading=True, compile_dit_sizes=["832x480x81"])
    assert model.config.compile_dit_grids() == {(21, 30, 52)}


@pytest.mark.parametrize("sizes", ["1280x704x81", ["1280x704"], ["1280x700x81"], ["1280x704x80"], ["0x704x81"]])
def test_compile_dit_sizes_must_be_aligned_wxhxf(sizes) -> None:
    with pytest.raises(ValueError, match="compile_dit_sizes"):
        Wan22Model(model_path_hf="unused", skip_weight_loading=True, compile_dit_sizes=sizes)


def _model(**kwargs) -> Wan22Model:
    return Wan22Model(model_path_hf="unused", skip_weight_loading=True, **kwargs)


@pytest.mark.parametrize("size", [{}, {"num_frames": 121}, {"height": 1088, "width": 1920, "num_frames": 121}])
def test_sizes_under_the_ceiling_pass(size) -> None:
    _model()._validate_generation_size(size)


def test_a_size_above_max_video_pixels_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_video_pixels=268435456"):
        _model()._validate_generation_size({"height": 1280, "width": 1280, "num_frames": 181})
    _model(max_video_pixels=2**30)._validate_generation_size({"height": 1280, "width": 1280, "num_frames": 181})
    with pytest.raises(ValueError, match="max_video_pixels"):
        _model(max_video_pixels=0)


@pytest.mark.parametrize("size", [
    {"height": 32, "width": 32800, "num_frames": 1},
    {"height": 32800, "width": 32, "num_frames": 1},
    {"height": 32, "width": 32, "num_frames": 4097},
])
def test_an_axis_past_the_rotary_table_is_rejected(size) -> None:
    with pytest.raises(ValueError, match="1024 rotary positions"):
        _model()._validate_generation_size(size)


@pytest.mark.parametrize("size", [
    {"height": 32, "width": 16416, "num_frames": 1},
    {"height": 16416, "width": 32, "num_frames": 1},
    {"height": 32, "width": 32768, "num_frames": 1},
])
def test_a_side_past_the_mp4_encoder_is_rejected_before_generating(size) -> None:
    """Under the rotary limit but past libx264's 16384-pixel side, postprocess
    would fail after the whole video is generated (a 500)."""
    with pytest.raises(ValueError, match="16384 pixels per side"):
        _model()._validate_generation_size(size)


def test_the_largest_side_the_mp4_encoder_takes_passes() -> None:
    _model()._validate_generation_size({"height": 32, "width": 16384, "num_frames": 1})
    _model()._validate_generation_size({"height": 16384, "width": 32, "num_frames": 1})


def test_the_config_examples_uncomment_into_one_model_kwargs_block() -> None:
    """configs/wan22.yaml ships a ``model_kwargs`` block and documents the other knobs
    as commented lines under it; uncommenting them keeps one valid block."""
    from pathlib import Path

    import yaml

    out, inside = [], False
    for line in (Path(__file__).resolve().parents[2] / "configs" / "wan22.yaml").read_text().splitlines():
        inside = line == "model_kwargs:" or (inside and line.startswith(("  ", "#")))
        out.append(line[2:] if inside and line.startswith("#   ") else line)
    text = "\n".join(out)
    assert text.count("\nmodel_kwargs:") == 1
    kwargs = yaml.safe_load(text)["model_kwargs"]
    assert set(kwargs) == {"vae_decode_fast_sizes", "vae_decode_fast", "compile_dit_sizes", "max_video_pixels",
                           "attention_backend", "dit_loop", "accelerator_graph", "capture_batch_sizes",
                           "max_batch_size"}
    _model(**kwargs)
