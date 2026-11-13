"""MiniMax-H3-Image's DiT step as one CUDA graph per row bucket, on one GPU: a tiny random DiT (bf16, the
reference's fp32 modules) captured through the engine's ``AcceleratorGraphRunner`` at a bucket's graph key, then
replayed the way the engine replays it for prompts of three lengths that bucket holds (padded, padded more, and
filling it), each against the same step run eager, bit for bit."""
import dataclasses

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
from mstar.engine.resources import StepContext, StepRunner
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.spec import resolve_spec_dependencies
from mstar.engine.state.adapters import node_handles, node_resources
from mstar.engine.state.state_op import bind_state_ops
from mstar.model.glm52._testing import per_request_outputs
from mstar.model.minimax_h3.components.transformer import H3DiT
from mstar.model.minimax_h3.config import DENOISE_LOOP, DIT_ATTN, DIT_NODE, DIT_SHARD
from mstar.model.minimax_h3.submodules import LATENTS, TEXT_EMBEDS
from mstar.model.minimax_h3.tests.tiny_checkpoint import CFG
from mstar.model.minimax_h3_image.config import H3ImageConfig
from mstar.model.minimax_h3_image.layout import H3ImageLayout
from mstar.model.minimax_h3_image.minimax_h3_image_model import GENERATE_WALK, H3ImageModel
from mstar.model.minimax_h3_image.submodules import H3ImageDenoiseSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine, engine_step

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")

DEVICE = torch.device("cuda:0")
TINY = H3ImageConfig(transformer=dataclasses.replace(CFG, audio_enabled=False, num_output_heads=3),
                     sigmas=(1.0, 0.75, 0.5, 0.0))
ROW_BUCKET = 32
# an 8 x 12 latent: 24 image rows, so a prompt of 9 to 40 rows pads to one 64-row bucket
GRID = (8, 12)
GRAPH_KEY = H3ImageLayout(text_len=40, latent_height=GRID[0], latent_width=GRID[1])


def _dit() -> H3DiT:
    torch.manual_seed(0)
    dit = H3DiT(TINY.transformer)
    with torch.no_grad():
        for param in dit.parameters():
            param.normal_(0, 0.2)
    dit = dit.to(DEVICE, torch.bfloat16)
    dit.keep_fp32()
    return dit.eval().requires_grad_(False)


class _Node:
    """The dit node over its resources as the engine builds and binds them, and (``capture=True``) a captured
    ``AcceleratorGraphRunner``; ``step`` drives one denoise step the engine's way (lease, pad, declare, admit, plan,
    preprocess, then forward or stage-and-replay, commit)."""

    def __init__(self, dit: H3DiT, capture: bool):
        model = H3ImageModel(model_path_hf="test/minimax_h3_image", skip_weight_loading=True)
        model.set_config(TINY)
        specs = resolve_spec_dependencies(node_resources(model))
        trivial = CommGroup.trivial()
        self.groups = JointGroups(tp_group=trivial, sp_group=trivial, instance=trivial)
        info = EngineResourceInfo(device=DEVICE, kv_dtype=torch.bfloat16, joint_comm_group=self.groups)
        self.resources = {
            key: build_resource(specs[key], dataclasses.replace(
                info, dependencies={dep: specs[dep] for dep in specs[key].depends_on()}))
            for key in (DIT_ATTN, DIT_SHARD)
        }
        for key, resource in self.resources.items():
            resource.bind_dependencies({dep: self.resources[dep] for dep in specs[key].depends_on()})
        self.runner = StepRunner(self.resources, node_resources={DIT_NODE: list(self.resources)})
        self.sub = H3ImageDenoiseSubmodule(dit, TINY, loop_name=DENOISE_LOOP, row_bucket=ROW_BUCKET,
                                           capture_buckets=[(GENERATE_WALK, GRAPH_KEY)] if capture else ())
        self.sub.bind_node_resources(self.resources)
        handles = node_handles(model, DIT_NODE, self.resources)
        bind_state_ops(handles)
        self.sub.bind_node_state(handles)
        self.cg = None
        if capture:
            self.cg = AcceleratorGraphRunner(
                submodule_name=DIT_NODE, submodule=self.sub, resources=self.resources, step_runner=self.runner,
                device=DEVICE, autocast_dtype=None, joint_comm_group=self.groups, num_slots=1, head_shards=1,
            )
            with torch.no_grad():
                self.cg.warmup_and_capture(None)

    def step(self, rid: int, text: torch.Tensor, latents: torch.Tensor, k: int):
        info = CurrentForwardPassInfo(
            request_id=str(rid), graph_walk=GENERATE_WALK, fwd_index=k, random_seed=0, max_tokens=1, rid_handle=rid,
            step_metadata={"text_len": text.shape[0], "latent_height": GRID[0], "latent_width": GRID[1]},
            dynamic_loop_iter_counts={DENOISE_LOOP: k},
        )
        infos = {rid: info}
        inputs = [self.sub.prepare_inputs(GENERATE_WALK, info, {TEXT_EMBEDS: [text], LATENTS: [latents]})]
        lease = None
        if self.cg is not None:
            lease = self.cg.lease_slot(GENERATE_WALK, 1, slot=0, num_tokens=inputs[0].input_seq_len,
                                       cg_key_info=self.sub.cg_key_info(GENERATE_WALK, infos))
            assert lease is not None, f"no graph for a {text.shape[0]}-row prompt"
        step_ids, meta = [rid], infos
        if lease is not None:
            inputs = self.cg.pad_inputs(lease, inputs)
            step_ids = self.cg.step_ids(lease, [rid])
            meta = self.cg.step_metadata(lease, [rid], infos)
        ctx = StepContext(request_ids=(rid,), graph_walk=GENERATE_WALK, slot=0, capture=False, slot_lease=lease)
        ctx.set_padded_rids(step_ids)
        step = engine_step(self.sub, graph_walk=GENERATE_WALK, request_ids=list(step_ids), inputs=inputs,
                           slot_lease=lease, per_request_info=meta)
        step.set_ctx(ctx)
        try:
            with torch.no_grad():
                assert self.runner.admit(step).ok
                self.runner.plan(step)
                step.begin_calls()
                try:
                    engine_inputs = ModelInputsFromEngine(
                        request_ids=list(step_ids), per_request_info=meta, resources=self.resources, step=step,
                        captured=lease is not None,
                    )
                    pre = self.sub.preprocess(GENERATE_WALK, engine_inputs, inputs)
                    if lease is None:
                        raw, out_ids = self.sub.forward_batched(GENERATE_WALK, engine_inputs, **pre), step_ids
                    else:
                        raw, out_ids = self.cg.run_forward(lease, pre), self.cg.slot_for(lease).dummy_rids
                    out = per_request_outputs(raw, out_ids)[out_ids[0]][LATENTS][0]
                    step.check_calls()
                finally:
                    step.end_calls()
                self.runner.commit(step)
        finally:
            if lease is not None:
                self.cg.release(lease, 1)
        torch.cuda.synchronize()
        return lease, out.clone()


def test_one_graph_replays_every_prompt_length_of_its_row_bucket_as_eager_runs_it():
    dit = _dit()
    g = torch.Generator().manual_seed(1)
    image_rows = GRAPH_KEY.num_video_rows
    cases = []
    for rid, text_len in enumerate((13, 29, 40)):
        text = torch.randn(text_len, TINY.transformer.hidden_size, generator=g).to(DEVICE, torch.bfloat16)
        latents = torch.randn(image_rows, TINY.transformer.patch_dim, generator=g).to(DEVICE)
        cases.append((rid, text, latents))

    eager = _Node(dit, capture=False)
    want = {(rid, k): eager.step(rid, text, latents, k)[1] for rid, text, latents in cases for k in (1, 2)}

    node = _Node(dit, capture=True)
    assert node.cg.any_graphs and node.cg.dropped_buckets == []
    buckets = set()
    for rid, text, latents in cases:
        for k in (1, 2):
            lease, got = node.step(rid, text, latents, k)
            buckets.add(lease.bucket)
            assert torch.isfinite(got).all()
            assert torch.equal(got, want[rid, k]), (text.shape[0], k, (got - want[rid, k]).abs().max())
    assert len(buckets) == 1, buckets
