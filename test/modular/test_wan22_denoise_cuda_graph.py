"""Replay one captured Wan2.2 scaffold denoise step across whole trajectories.

The scaffold loop's step reads every per-step decision (the solver's order ramp, the
corrector at step 0, the last step's x0 landing, I2V's write-back) from per-row
weights, so one graph captured at a batch of two rows must reproduce the eager loop
at every step, with the rows at different steps. A host sync or a CPU-resident
scalar inside the step fails the capture outright. Tiny native DiT in the serving
dtypes (bf16 with fp32 islands), SDPA attention; skipped without a GPU.
"""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402
from mstar.model.wan22.components.dit import Wan22DiT  # noqa: E402
from mstar.model.wan22.config import Wan22Config  # noqa: E402
from mstar.model.wan22.submodules import DENOISE_LOOP_NAME, Wan22DenoiseSubmodule  # noqa: E402
from mstar.model.wan22.wan22_model import Wan22Model  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

STEPS = 6
TEXT_LEN = 8
META = {"height": 64, "width": 96, "num_frames": 9}  # latent (3, 4, 6)


def _node(device) -> Wan22DenoiseSubmodule:
    config = Wan22Config(
        num_attention_heads=2, attention_head_dim=64, num_layers=2, ffn_dim=256, in_channels=8, out_channels=8,
        text_dim=32, freq_dim=32, text_max_seq_len=TEXT_LEN, compile_dit=False,
    )
    dit = Wan22DiT(config)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in dit.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
    dit.cast_serving_dtypes()
    return Wan22DenoiseSubmodule(dit.to(device).eval(), config)


def _info(rid: str, handle: int, k: int, walk: str) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=walk, fwd_index=k, random_seed=handle + 3, max_tokens=0, rid_handle=handle,
        step_metadata={**META, "num_inference_steps": STEPS, "guidance_scale": 4.0},
        dynamic_loop_iter_counts={DENOISE_LOOP_NAME: k},
    )


def _conditioning(seed: int, device, i2v: bool) -> dict:
    gen = torch.Generator().manual_seed(seed)
    cond = {
        name: [torch.randn(1, TEXT_LEN, 32, generator=gen).to(device, torch.bfloat16)]
        for name in ("text_embeds_pos", "text_embeds_neg")
    }
    if i2v:
        cond["image_latent"] = [torch.randn(1, 8, 1, 4, 6, generator=gen).to(device)]
    return cond


@pytest.mark.parametrize("walk", [Wan22Model.VIDEO_GEN_WALK, Wan22Model.VIDEO_GEN_I2V_WALK])
def test_one_captured_step_replays_both_trajectories(walk):
    device = torch.device("cuda")
    node = _node(device)
    i2v = walk == Wan22Model.VIDEO_GEN_I2V_WALK
    # b starts three steps after a, so the batch mixes step 0 and the last step with the middle ones
    rows = {"a": (0, 0, _conditioning(1, device, i2v)), "b": (1, 3, _conditioning(2, device, i2v))}
    rids = list(rows)
    engine_inputs = ModelInputsFromEngine(request_ids=rids, per_request_info={})

    def batch(iteration: int, carried: dict) -> dict:
        prepared = []
        for rid, (handle, offset, cond) in rows.items():
            k = min(max(iteration - offset, 0), STEPS - 1)
            prepared.append(node.prepare_inputs(walk, _info(rid, handle, k, walk), {**cond, **carried.get(rid, {})}))
        return node.preprocess(walk, engine_inputs, prepared)

    # eager trajectories; a row past its last step or before its first repeats its edge step
    def run(step_fn) -> list[dict]:
        node.request_states.clear()
        carried, outs = {}, []
        for it in range(STEPS + 3):
            out = step_fn(batch(it, carried))
            for rid, (_, offset, _) in rows.items():
                k = it - offset
                if 0 <= k < STEPS:
                    carried[rid] = {name: [t.clone()] for name, t in ((n, v[0]) for n, v in out[rid].items())}
            outs.append({rid: {n: v[0].clone() for n, v in out[rid].items()} for rid in rids})
        return outs

    eager = run(lambda inputs: node.forward_batched(walk, engine_inputs, **inputs))

    node.request_states.clear()
    static = batch(0, {})
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        node.forward_batched(walk, engine_inputs, **static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = node.forward_batched(walk, engine_inputs, **static)

    def replay(inputs: dict) -> dict:
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                static[name].copy_(value)
        graph.replay()
        return static_out

    captured = run(replay)
    # GEMMs may pick other kernels under capture; a weight baked into the graph would
    # instead move the trajectory by O(1)
    for it, (e, c) in enumerate(zip(eager, captured, strict=True)):
        for rid in rids:
            for name in e[rid]:
                torch.testing.assert_close(c[rid][name], e[rid][name], rtol=0, atol=2e-2,
                                           msg=lambda m, it=it, rid=rid, name=name: f"iter {it} {rid} {name}: {m}")
