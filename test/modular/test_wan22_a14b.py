"""CPU contract tests for Wan2.2-T2V-A14B (``variant: t2v_a14b`` on ``wan22``).

The variant's config and request seam, its graph and serving configs, and the dit's
two experts: which expert and which guidance scale each step runs (diffusers'
``WanPipeline``: the high-noise ``transformer`` while ``t >= boundary_ratio * 1000``,
``transformer_2`` after), a batch straddling the boundary, the per-sample timestep,
and the loader's refusal of another variant's checkpoint. Stand-in transformers, no
weights, no GPU; the GPU parity against diffusers is test_wan22_a14b_equivalence.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar.model.components.diffusion.unipc import make_unipc_tables
from mstar.model.wan22.components.dit import Wan22DiT
from mstar.model.wan22.config import WAN22_VARIANT_T2V_A14B, Wan22Config
from mstar.model.wan22.submodules import DENOISE_LOOP_NAME, Wan22DitSubmodule
from mstar.model.wan22.wan22_model import Wan22Model
from mstar.model.wan22.weight_loader import check_transformer_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
A14B_CONFIGS = ("wan22_a14b.yaml", "wan22_a14b_cfg_parallel_2gpu.yaml", "wan22_a14b_sp2.yaml")
CHANNELS, TEXT_LEN, TEXT_DIM = 4, 3, 8


def _model() -> Wan22Model:
    return Wan22Model(model_path_hf="test/wan22", variant=WAN22_VARIANT_T2V_A14B, skip_weight_loading=True)


def _plan(model, model_kwargs=None, input_modalities=("text",)):
    from mstar.graph.base import TensorPointerInfo

    info = TensorPointerInfo(
        dims=[1], dtype=torch.float32, nbytes=4, address=0, stride=[1],
        uuid="uuid", source_session_id="test:0", source_entity="test",
    )
    signals = {"text_inputs": [info], "image_inputs": [info]}
    return model.walk_plan("default", list(input_modalities), ["video"], signals, model_kwargs or {})


# ----------------------------------------------------------------------
# config and request seam
# ----------------------------------------------------------------------


def test_a14b_config_is_the_checkpoints():
    cfg = Wan22Config.for_variant(WAN22_VARIANT_T2V_A14B)
    assert (cfg.num_attention_heads, cfg.attention_head_dim, cfg.num_layers, cfg.ffn_dim) == (40, 128, 40, 13824)
    assert (cfg.in_channels, cfg.out_channels, cfg.vae_scale_factor_spatial) == (16, 16, 8)
    assert cfg.spatial_alignment == (16, 16)
    assert cfg.is_moe and cfg.boundary_ratio == 0.875 and not cfg.expand_timesteps and not cfg.supports_i2v
    assert (cfg.guidance_scale, cfg.guidance_scale_2, cfg.default_num_inference_steps, cfg.flow_shift) == (
        4.0, 3.0, 40, 3.0,
    )
    assert (cfg.default_width, cfg.default_height, cfg.default_num_frames) == (1280, 720, 81)
    assert cfg.compile_dit_grids() == {(21, 90, 160)}
    # TI2V-5B is untouched by the variant table
    assert Wan22Config.for_variant() == Wan22Config()


def test_unknown_variant_is_refused():
    with pytest.raises(NotImplementedError, match="t2v_a14b"):
        Wan22Model(model_path_hf="test/wan22", variant="i2v_a14b")


def test_a14b_refuses_a_step_cache():
    """The calibrated MagCache table is TI2V-5B's, so A14B refuses a step cache rather than read it."""
    spec = {"method": "magcache", "threshold": 0.06}
    with pytest.raises(ValueError, match="step_cache is not supported on t2v_a14b"):
        Wan22Model(
            model_path_hf="test/wan22", variant=WAN22_VARIANT_T2V_A14B, skip_weight_loading=True, step_cache=spec,
        )
    assert Wan22Model(model_path_hf="test/wan22", skip_weight_loading=True, step_cache=spec).config.step_cache == spec


def test_a14b_accepts_its_benchmark_shapes():
    cfg = Wan22Config.for_variant(WAN22_VARIANT_T2V_A14B, compile_dit_sizes=["832x480x81", "1280x720x81"])
    assert cfg.compile_dit_grids() == {(21, 60, 104), (21, 90, 160)}
    model = _model()
    for width, height in ((832, 480), (1280, 720)):
        plan = _plan(model, {"width": width, "height": height, "num_frames": 81})
        assert plan[-1].step_metadata["height"] == height
    # 16-aligned is enough; 8 is not
    with pytest.raises(ValueError, match="multiple of 16"):
        _plan(model, {"width": 832, "height": 488})


def test_a14b_is_text_to_video_only():
    model = _model()
    assert set(model.get_graph_walk_graphs()) == {Wan22Model.ENCODE_TEXT_WALK, Wan22Model.VIDEO_GEN_WALK}
    with pytest.raises(ValueError, match="text-to-video only"):
        _plan(model, input_modalities=("text", "image"))
    with pytest.raises(ValueError, match="text-to-video only"):
        model._validate_request("a cat", ["text", "image"], ["video"], {"image_inputs": [torch.zeros(3, 16, 16)]})


@pytest.mark.parametrize("kwargs,expected", [
    ({}, (4.0, 3.0)),                                            # the checkpoint's pair
    ({"guidance_scale": 5.0}, (5.0, 5.0)),                       # diffusers: guidance_scale_2 follows it
    ({"guidance_scale": 5.0, "guidance_scale_2": 2.0}, (5.0, 2.0)),
    ({"guidance_scale_2": 1.0}, (4.0, 1.0)),
    ({"guidance_scale": None, "guidance_scale_2": None}, (4.0, 3.0)),  # null is absent
])
def test_a14b_guidance_pair(kwargs, expected):
    meta = _plan(_model(), kwargs)[-1].step_metadata
    assert (meta["guidance_scale"], meta["guidance_scale_2"]) == expected
    assert meta["num_inference_steps"] == 40 and "flow_shift" not in meta


def test_request_knobs_are_declared_and_validated():
    model = _model()
    assert {"guidance_scale", "guidance_scale_2", "flow_shift"} <= model.request_kwargs()
    assert "guidance_scale_2" not in Wan22Model(model_path_hf="test/wan22").request_kwargs()
    assert _plan(model, {"flow_shift": 12})[-1].step_metadata["flow_shift"] == 12.0
    for bad in ({"guidance_scale_2": "3"}, {"flow_shift": 0}, {"flow_shift": float("nan")}):
        with pytest.raises(ValueError, match="guidance_scale_2|flow_shift"):
            _plan(model, bad)


@pytest.mark.parametrize("name", A14B_CONFIGS)
def test_a14b_configs_place_every_node(name):
    import yaml

    from mstar.validation.placement import check_placement

    with open(CONFIGS / name) as f:
        config = yaml.safe_load(f)
    kwargs = config["model_kwargs"]
    assert kwargs["variant"] == WAN22_VARIANT_T2V_A14B
    assert kwargs["model_path_hf"] == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    model = Wan22Model(skip_weight_loading=True, **kwargs)
    assert check_placement(model, config) == []
    walks = {walk for wg in model.get_worker_graphs(str(CONFIGS / name)) for walk in wg.graph_walks}
    assert walks == {Wan22Model.ENCODE_TEXT_WALK, Wan22Model.VIDEO_GEN_WALK}


# ----------------------------------------------------------------------
# the two experts
# ----------------------------------------------------------------------


class _Expert(nn.Module):
    """A dit stand-in: ``scale`` times its text embedding, recording each call's timestep."""

    dtype = torch.float32

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.timesteps: list[torch.Tensor] = []

    def forward(self, hidden_states, timestep, encoder_hidden_states, shard=None):
        self.timesteps.append(timestep.clone())
        return self.scale * encoder_hidden_states[:, :1, :1, None, None].expand_as(hidden_states).clone()


def _node(**overrides) -> tuple[Wan22DitSubmodule, _Expert, _Expert]:
    cfg = Wan22Config.for_variant(WAN22_VARIANT_T2V_A14B, in_channels=CHANNELS, compile_dit=False, **overrides)
    high, low = _Expert(1.0), _Expert(10.0)
    node = Wan22DitSubmodule(high, cfg, transformer_2=low)
    node.get_device = lambda: torch.device("cpu")
    return node, high, low


def _info(rid: str, step: int, steps: int, meta: dict) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=rid, rid_handle=rid, graph_walk=Wan22Model.VIDEO_GEN_WALK, random_seed=0,
        step_metadata={"num_inference_steps": steps, "height": 64, "width": 64, "num_frames": 5, **meta},
        dynamic_loop_iter_counts={DENOISE_LOOP_NAME: step},
    )


def _run(node: Wan22DitSubmodule, infos: list) -> dict[str, torch.Tensor]:
    shape = (1, CHANNELS, 2, 8, 8)
    gen = torch.Generator().manual_seed(0)
    prepared = [
        node.prepare_inputs(Wan22Model.VIDEO_GEN_WALK, info, {
            "latents": [torch.randn(shape, generator=gen)],
            "unipc_model_outputs": [torch.randn((2, *shape), generator=gen)],
            "unipc_last_sample": [torch.randn(shape, generator=gen)],
            "text_embeds_pos": [torch.full((1, TEXT_LEN, TEXT_DIM), 2.0)],
            "text_embeds_neg": [torch.full((1, TEXT_LEN, TEXT_DIM), 1.0)],
        })
        for info in infos
    ]
    engine_inputs = SimpleNamespace(
        request_ids=[i.request_id for i in infos], per_request_info={i.request_id: i for i in infos},
    )
    stacked = node.preprocess(Wan22Model.VIDEO_GEN_WALK, engine_inputs, prepared)
    stacked.pop("bucket_key")
    key = node.bucket_key_for(infos[0])
    carried = {name: stacked.pop(name) for name in node.loop_back_names}
    timestep = stacked.pop("timestep")
    return node.model_output(engine_inputs, key, carried["latents"], timestep, **stacked)


@pytest.mark.parametrize("flow_shift", [None, 12.0])
def test_each_step_runs_the_expert_diffusers_picks(flow_shift):
    """Per step: the expert by ``t >= 875`` on the request's table, the combine at
    that expert's scale, and one timestep per sample, the step's."""
    steps = 10
    meta = {"guidance_scale": 4.0, "guidance_scale_2": 3.0}
    if flow_shift is not None:
        meta["flow_shift"] = flow_shift
    _, timesteps = make_unipc_tables(steps, 3.0 if flow_shift is None else flow_shift)
    expected_high = [bool(t >= 875) for t in timesteps]
    assert any(expected_high) and not all(expected_high)
    node, high, low = _node()
    for step in range(steps):
        before = (len(high.timesteps), len(low.timesteps))
        out = _run(node, [_info("r", step, steps, meta)])
        ran_high = len(high.timesteps) > before[0]
        assert ran_high == expected_high[step] and (len(low.timesteps) > before[1]) != ran_high, step
        expert = high if ran_high else low
        # cond 2s, uncond 1s: uncond + g * (cond - uncond)
        g = 4.0 if ran_high else 3.0
        assert torch.allclose(out, torch.full_like(out, expert.scale * (1 + g)))
        ts = expert.timesteps[-1]
        assert ts.shape == (2, 1) and torch.all(ts == timesteps[step])


def test_guidance_2_at_or_below_one_still_runs_both_branches():
    """diffusers decides CFG on ``guidance_scale`` alone, so a low-noise step at
    guidance_scale_2 1.0 runs both branches and combines at 1.0."""
    steps = 4
    node, _, low = _node()
    _, timesteps = make_unipc_tables(steps, 3.0)
    last = steps - 1
    assert timesteps[last] < 875
    out = _run(node, [_info("r", last, steps, {"guidance_scale": 4.0, "guidance_scale_2": 1.0})])
    assert low.timesteps[-1].shape[0] == 2
    assert torch.allclose(out, torch.full_like(out, 20.0))


def test_a_batch_straddling_the_boundary_matches_each_row_alone():
    steps = 10
    meta = {"guidance_scale": 4.0, "guidance_scale_2": 3.0}
    _, timesteps = make_unipc_tables(steps, 3.0)
    first_low = next(i for i, t in enumerate(timesteps) if t < 875)
    infos = [_info("a", first_low - 1, steps, meta), _info("b", first_low, steps, meta)]
    node, high, low = _node(dit_max_batch_size=2)
    together = _run(node, infos)
    assert len(high.timesteps) == len(low.timesteps) == 1
    alone = torch.cat([_run(_node()[0], [info]) for info in infos])
    assert torch.equal(together, alone)


def test_a14b_needs_both_experts():
    cfg = Wan22Config.for_variant(WAN22_VARIANT_T2V_A14B, compile_dit=False)
    with pytest.raises(ValueError, match="transformer_2"):
        Wan22DitSubmodule(_Expert(1.0), cfg)


def test_dit_takes_a_per_sample_timestep():
    """A ``[B, 1]`` timestep (A14B) is the per-frame path with every frame at the
    same value, to rounding."""
    cfg = Wan22Config(
        num_attention_heads=2, attention_head_dim=12, num_layers=2, ffn_dim=32,
        in_channels=CHANNELS, out_channels=CHANNELS, text_dim=TEXT_DIM, freq_dim=16, rope_max_seq_len=16,
    )
    torch.manual_seed(0)
    dit = Wan22DiT(cfg)
    for param in dit.parameters():
        nn.init.normal_(param, std=0.2)
    gen = torch.Generator().manual_seed(1)
    hidden = torch.randn(2, CHANNELS, 3, 4, 4, generator=gen)
    text = torch.randn(2, TEXT_LEN, TEXT_DIM, generator=gen)
    t = torch.tensor([[700.0], [300.0]])
    with torch.no_grad():
        per_sample = dit(hidden, t, text)
        per_frame = dit(hidden, t.expand(2, 3), text)
    torch.testing.assert_close(per_sample, per_frame, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="per-sample"):
        dit(hidden, t.expand(2, 2), text)


def test_loader_refuses_another_variants_checkpoint(tmp_path):
    a14b = Wan22Config.for_variant(WAN22_VARIANT_T2V_A14B)
    (tmp_path / "config.json").write_text(json.dumps({
        "num_attention_heads": 40, "attention_head_dim": 128, "num_layers": 40, "ffn_dim": 13824,
        "in_channels": 16, "out_channels": 16, "text_dim": 4096, "freq_dim": 256,
    }))
    check_transformer_config(a14b, tmp_path)
    with pytest.raises(ValueError, match="num_layers=40 \\(variant has 30\\)"):
        check_transformer_config(Wan22Config(), tmp_path)
