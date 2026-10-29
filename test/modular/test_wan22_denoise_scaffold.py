"""CPU checks for Wan2.2's denoise loop on the DiT scaffold (``Wan22DenoiseSubmodule``).

A tiny native ``Wan22DiT`` in float32 stands in for the 5B DiT, so the two loops
differ only in how the solver arithmetic is grouped: the scaffold loop's UniPC in
coefficient form must track the bit-exact legacy loop (``Wan22DitSubmodule``) to
float32 rounding over a whole T2V and I2V trajectory, rows batched at different
steps must equal each row run alone, and the step declaration and capture buckets
must follow the request shape. No weights, no GPU.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402
from mstar.model.wan22.components.dit import Wan22DiT  # noqa: E402
from mstar.model.wan22.components.unipc import (  # noqa: E402
    UniPCCoefficients,
    UniPCState,
    make_unipc_tables,
    unipc_coefficient_step,
    unipc_coefficients,
    unipc_convert_model_output,
    unipc_corrector_step,
    unipc_effective_order,
    unipc_predictor_step,
)
from mstar.model.wan22.config import DIT_ATTN, DIT_XATTN, TEXT_SPAN, VIDEO_SPAN, Wan22Config  # noqa: E402
from mstar.model.wan22.submodules import (  # noqa: E402
    DENOISE_LOOP_NAME,
    UNIPC_LAST_SAMPLE,
    UNIPC_MODEL_OUTPUTS,
    Wan22DenoiseSubmodule,
    Wan22DitSubmodule,
    Wan22Shape,
    Wan22VaeDecoderSubmodule,
)
from mstar.model.wan22.wan22_model import Wan22Model  # noqa: E402

T2V, I2V = Wan22Model.VIDEO_GEN_WALK, Wan22Model.VIDEO_GEN_I2V_WALK
# 64x64 pixels x 5 frames -> latent (2, 4, 4) -> 2 x 2 x 2 = 8 tokens per branch
META = {"height": 64, "width": 64, "num_frames": 5}
TEXT_LEN = 6


def _config(**kwargs) -> Wan22Config:
    return Wan22Config(
        num_attention_heads=2, attention_head_dim=12, num_layers=2, ffn_dim=32, in_channels=4, out_channels=4,
        text_dim=8, freq_dim=16, text_max_seq_len=TEXT_LEN, compile_dit=False, **kwargs,
    )


def _tiny_dit(config: Wan22Config) -> Wan22DiT:
    dit = Wan22DiT(config)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in dit.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)
    return dit.eval()


@pytest.fixture(scope="module")
def dit():
    return _tiny_dit(_config())


_HANDLES: dict[str, int] = {}


def _info(rid: str, k: int, walk: str = T2V, steps: int = 4, guidance: float = 5.0, seed: int = 7):
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=walk, fwd_index=k, random_seed=seed, max_tokens=0,
        rid_handle=_HANDLES.setdefault(rid, len(_HANDLES)),
        step_metadata={**META, "num_inference_steps": steps, "guidance_scale": guidance},
        dynamic_loop_iter_counts={DENOISE_LOOP_NAME: k},
    )


def _conditioning(seed: int = 1, i2v: bool = False) -> dict[str, list[torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    # bf16, as the UMT5 encoder emits them
    cond = {
        "text_embeds_pos": [torch.randn(1, TEXT_LEN, 8, generator=gen).to(torch.bfloat16)],
        "text_embeds_neg": [torch.randn(1, TEXT_LEN, 8, generator=gen).to(torch.bfloat16)],
    }
    if i2v:
        cond["image_latent"] = [torch.randn(1, 4, 1, 4, 4, generator=gen)]
    return cond


def _scaffold(dit, **kwargs) -> Wan22DenoiseSubmodule:
    return Wan22DenoiseSubmodule(dit, _config(), **kwargs)


def _run_legacy(dit, walk: str, steps: int, guidance: float) -> torch.Tensor:
    node = Wan22DitSubmodule(dit, _config())
    cond = _conditioning(i2v=walk == I2V)
    carried: dict[str, list[torch.Tensor]] = {}
    for k in range(steps):
        info = _info("legacy", k, walk, steps, guidance)
        inputs = node.prepare_inputs(walk, info, {**cond, **carried})
        engine_inputs = SimpleNamespace(single_request_info=info)
        out = node.forward(walk, engine_inputs=engine_inputs, **inputs.tensor_inputs)
        carried = out
    return carried["latents"][0][0]


def _step_scaffold(node, walk, rows: list[tuple[str, CurrentForwardPassInfo, dict]]) -> dict[str, dict]:
    """One batched step over ``(rid, info, inputs)`` rows; returns each row's outputs."""
    prepared = [node.prepare_inputs(walk, info, inputs) for _, info, inputs in rows]
    rids = [rid for rid, _, _ in rows]
    engine_inputs = ModelInputsFromEngine(request_ids=rids, per_request_info={})
    return node.forward_batched(walk, engine_inputs, **node.preprocess(walk, engine_inputs, prepared))


def _run_scaffold(node, walk: str, steps: int, guidance: float, rid: str = "scaffold") -> torch.Tensor:
    cond = _conditioning(i2v=walk == I2V)
    carried: dict[str, list[torch.Tensor]] = {}
    for k in range(steps):
        carried = _step_scaffold(node, walk, [(rid, _info(rid, k, walk, steps, guidance), {**cond, **carried})])[rid]
    return carried["latents"][0]


# ----------------------------------------------------------------------
# The solver in coefficient form
# ----------------------------------------------------------------------


@pytest.mark.parametrize("steps", [1, 2, 3, 5, 20, 50])
def test_coefficient_form_tracks_the_exact_port(steps):
    sigmas, _ = make_unipc_tables(steps, 5.0)
    coeffs = unipc_coefficients(sigmas, steps)
    gen = torch.Generator().manual_seed(steps)
    shape = (1, 4, 3, 5)
    x = torch.randn(shape, generator=gen)
    ring, last = torch.zeros(2, *shape), torch.zeros(shape)
    xc, ring_c, last_c = x.clone(), torch.zeros(1, 2, *shape), torch.zeros(1, *shape)
    for k in range(steps):
        eps = torch.randn(shape, generator=gen)
        m = unipc_convert_model_output(eps, x, sigmas, k)
        sample = x
        if k > 0:
            sample = unipc_corrector_step(
                UniPCState(ring, last), m, x, sigmas, k, unipc_effective_order(k - 1, steps),
            )
        ring = torch.stack([ring[1], m])
        x = unipc_predictor_step(UniPCState(ring, sample), sample, sigmas, k, unipc_effective_order(k, steps))
        last = sample

        mc = unipc_convert_model_output(eps, xc, sigmas, k)[None]
        weights = UniPCCoefficients(*(w[k].view(1, 1, 1, 1, 1) for w in coeffs))
        new, ring_c, last_c = unipc_coefficient_step(xc[None], mc, ring_c, last_c, weights)
        xc = new[0]
        torch.testing.assert_close(xc, x, rtol=0, atol=1e-5)
    torch.testing.assert_close(ring_c[0], ring, rtol=0, atol=1e-5)


def test_first_and_last_step_weights_are_exact():
    """Step 0 has no corrector and the last step lands on the x0 prediction: both
    are exact weights, so neither rounds anything."""
    sigmas, _ = make_unipc_tables(20, 5.0)
    c = unipc_coefficients(sigmas, 20)
    assert (c.corr_sample[0], c.corr_last[0], c.corr_prev[0], c.corr_prev2[0], c.corr_this[0]) == (1, 0, 0, 0, 0)
    assert (c.pred_sample[-1], c.pred_this[-1], c.pred_prev[-1]) == (0, 1, 0)
    assert torch.all(c.corr_sample[1:] == 0)
    # the order ramp: no second-order history before step 1, none on the last step
    assert c.pred_prev[0] == 0 and c.pred_prev[1] != 0 and c.corr_prev2[1] == 0 and c.corr_prev2[2] != 0


# ----------------------------------------------------------------------
# The loop against the legacy loop
# ----------------------------------------------------------------------


@pytest.mark.parametrize("walk", [T2V, I2V])
@pytest.mark.parametrize("guidance", [5.0, 1.0])
def test_scaffold_loop_tracks_the_legacy_loop(dit, walk, guidance):
    steps = 6
    legacy = _run_legacy(dit, walk, steps, guidance)
    scaffold = _run_scaffold(_scaffold(dit), walk, steps, guidance)
    torch.testing.assert_close(scaffold, legacy, rtol=0, atol=1e-4)
    if walk == I2V:
        # the conditioning frame is written back on the last step, as the reference does
        torch.testing.assert_close(scaffold[:, :1], _conditioning(i2v=True)["image_latent"][0][0], rtol=0, atol=0)


def test_rows_at_different_steps_batch_like_each_row_alone(dit):
    """Each row carries its own schedule and solver ring, so a batch of a fresh
    request and one midway through equals each run alone."""
    node = _scaffold(dit)
    cond_a, cond_b = _conditioning(seed=1), _conditioning(seed=2)
    # b runs two steps alone first
    carried_b: dict = {}
    for k in range(2):
        carried_b = _step_scaffold(node, T2V, [("b", _info("b", k, seed=3), {**cond_b, **carried_b})])["b"]

    alone_a = _step_scaffold(node, T2V, [("a", _info("a", 0), cond_a)])["a"]
    alone_b = _step_scaffold(node, T2V, [("b", _info("b", 2, seed=3), {**cond_b, **carried_b})])["b"]
    batched = _step_scaffold(node, T2V, [
        ("a", _info("a", 0), cond_a), ("b", _info("b", 2, seed=3), {**cond_b, **carried_b}),
    ])
    for rid, alone in (("a", alone_a), ("b", alone_b)):
        for name in ("latents", UNIPC_MODEL_OUTPUTS, UNIPC_LAST_SAMPLE):
            torch.testing.assert_close(batched[rid][name][0], alone[name][0], rtol=0, atol=1e-5)


def test_overshoot_iteration_is_vetoed(dit):
    node = _scaffold(dit)
    node.prepare_inputs(T2V, _info("o", 0, steps=2), _conditioning())
    carried = {name: [torch.zeros(1)] for name in node.loop_back_names}
    assert node.prepare_inputs(T2V, _info("o", 2, steps=2), {**_conditioning(), **carried}) is None


# ----------------------------------------------------------------------
# Batching, attention and capture
# ----------------------------------------------------------------------


@pytest.mark.parametrize("walk,guidance,rows", [(T2V, 5.0, 2), (T2V, 1.0, 1), (I2V, 5.0, 2)])
def test_shape_and_step_declaration_follow_the_request(walk, guidance, rows):
    node = Wan22DenoiseSubmodule(None, _config(), use_ragged_attention=True)
    shape = node.bucket_key_for(_info("r", 0, walk, guidance=guidance))
    assert shape == Wan22Shape(grid=(2, 4, 4), text_len=TEXT_LEN, cfg=rows == 2, i2v=walk == I2V, video_tokens=8)
    assert node.num_tokens(shape) == 8 * rows

    inp = SimpleNamespace(resource_step_info=shape)
    step = node.declare_step(walk, ["r", "s"], [inp, inp])
    video = [(s.request_id, s.label, s.span) for s in step.steps[DIT_ATTN].segments]
    assert video == [("r", VIDEO_SPAN, 8)] * rows + [("s", VIDEO_SPAN, 8)] * rows
    cross = step.steps[DIT_XATTN]
    assert cross.pairs == ((VIDEO_SPAN, TEXT_SPAN),)
    assert [(s.request_id, s.label) for s in cross.segments][2 * rows:] == [("r", TEXT_SPAN)] * rows + [
        ("s", TEXT_SPAN)] * rows
    assert step.cg_key_info == shape


def test_only_equal_shapes_batch(dit):
    node = _scaffold(dit)
    a = node.prepare_inputs(T2V, _info("a", 0), _conditioning())
    b = node.prepare_inputs(T2V, _info("b", 0, guidance=1.0), _conditioning())
    c = node.prepare_inputs(T2V, _info("c", 0), _conditioning())
    assert node.can_batch(None, [a, c]) and not node.can_batch(None, [a, b])


def test_model_captures_each_compiled_shape_for_t2v_and_i2v():
    model = object.__new__(Wan22Model)
    model.config = _config(compile_dit_sizes=("64x64x5", "128x64x5"))
    model.dit_loop, model.accelerator_graph = "scaffold", True
    buckets = model.capture_buckets()
    assert sorted((walk, shape.grid, shape.i2v, shape.cfg) for walk, shape in buckets) == [
        (T2V, (2, 4, 4), False, True), (T2V, (2, 4, 8), False, True),
        (I2V, (2, 4, 4), True, True), (I2V, (2, 4, 8), True, True),
    ]
    model.dit_loop = "legacy"
    assert model.capture_buckets() == []


def test_capture_rows_carry_every_input_a_served_row_does(dit):
    node = _scaffold(dit, capture_buckets=[(I2V, node_shape := Wan22Shape((2, 4, 4), TEXT_LEN, True, True, 8))])
    (config,) = node.get_accelerator_graph_configs(torch.device("cpu"))
    captured = config.single_request_inputs
    served = node.prepare_inputs(I2V, _info("cap", 0, I2V), _conditioning(i2v=True))
    assert captured.resource_step_info == served.resource_step_info == node_shape
    assert captured.input_seq_len == served.input_seq_len == 16
    assert set(captured.tensor_inputs) == set(served.tensor_inputs)
    for name, tensor in served.tensor_inputs.items():
        assert captured.tensor_inputs[name].shape == tensor.shape, name
        assert captured.tensor_inputs[name].dtype == tensor.dtype, name


def test_loop_back_edges_depend_on_the_loop():
    model = object.__new__(Wan22Model)
    model.config, model.dit_loop = _config(), "scaffold"
    assert model.loop_back_names() == ["latents", UNIPC_MODEL_OUTPUTS, UNIPC_LAST_SAMPLE]
    assert tuple(model.loop_back_names()) == Wan22DenoiseSubmodule(None, _config()).loop_back_names
    model.dit_loop = "legacy"
    assert "time_index" in model.loop_back_names()


def test_decoder_takes_the_scaffold_latents_without_a_batch_dim():
    decoder = object.__new__(Wan22VaeDecoderSubmodule)
    latents = torch.zeros(4, 2, 4, 4)
    out = decoder.prepare_inputs(T2V, _info("d", 0), {"latents": [latents]})
    assert out.tensor_inputs["latents"].shape == (1, 4, 2, 4, 4)
    legacy = decoder.prepare_inputs(T2V, _info("d", 0), {"latents": [latents[None]]})
    assert legacy.tensor_inputs["latents"].shape == (1, 4, 2, 4, 4)
