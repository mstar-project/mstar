"""``GuidanceParallel``: one classifier-free guidance branch per rank of a lockstep
instance.

The mesh and the config are checked directly. The lockstep contract (invariant 11)
runs on two real gloo ranks in their own processes, each driving a fake two-row
denoise model through the engine's admit/plan/commit sweep: every rank must run the
same collectives in the same order, take the same admit decisions, including the
refusals, and end with the same latents, which must equal the two rows run on one
rank.
"""
import json
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.distributed.communication import CommGroup, GlobalParallelConfig, JointGroups  # noqa: E402
from mstar.distributed.guidance_parallel import GuidanceParallel  # noqa: E402
from mstar.engine.resources import AdmitOutcome, Resource, StepContext, StepRunner  # noqa: E402
from mstar.engine.resources.step import AdmitFailedReason  # noqa: E402
from mstar.model.base import WorkerGraph  # noqa: E402
from mstar.model.components.diffusion.denoise_loop import LATENTS, DenoiseLoopSubmodule  # noqa: E402
from mstar.model.components.diffusion.flow_match import FlowMatchConfig, FlowMatchSchedule, euler_step  # noqa: E402
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402
from mstar.validation import DeploymentError, validate_deployment  # noqa: E402
from mstar.worker.engine_manager import parallel_kwargs  # noqa: E402

LOOP = "denoise_loop"
WALK = "video_gen"
ATTN = "attn"
TOKENS = 6


@dataclass(frozen=True)
class _Shape:
    tokens: int
    guided: bool


class _TwoRowDenoise(DenoiseLoopSubmodule):
    """A velocity per branch that reads only that branch's conditioning, so a row
    computed on another rank equals the row computed here."""

    def __init__(self, guidance_parallel=None):
        super().__init__(loop_name=LOOP, attn_resource_key=ATTN, guidance_parallel=guidance_parallel)

    def bucket_key_for(self, fwd_info):
        md = fwd_info.step_metadata
        return _Shape(int(md["tokens"]), float(md["guidance_scale"]) > 1.0)

    def schedule_for(self, fwd_info, bucket_key):
        steps = int(fwd_info.step_metadata["num_inference_steps"])
        return FlowMatchSchedule.build(FlowMatchConfig(), steps, bucket_key.tokens)

    def seed_latents(self, fwd_info, bucket_key, generator):
        return torch.randn(bucket_key.tokens, 4, generator=generator)

    def request_inputs(self, fwd_info, inputs, bucket_key):
        if bucket_key.guided and self.guidance_parallel is None:
            return {"cond": inputs["cond"][0], "uncond": inputs["uncond"][0], "scale": inputs["scale"][0]}
        name = "uncond" if bucket_key.guided and not self.guidance_parallel.runs_cond else "cond"
        return {name: inputs[name][0], "scale": inputs["scale"][0]}

    def num_tokens(self, bucket_key):
        return bucket_key.tokens * self.branch_rows(bucket_key.guided)

    def denoise(self, engine_inputs, bucket_key, latents, timestep, sigma, sigma_next, scale, cond=None, uncond=None):
        batch = latents.shape[0]
        text = torch.stack([cond, uncond], dim=1).flatten(0, 1) if cond is not None and uncond is not None \
            else (cond if cond is not None else uncond)
        rows = self.branch_rows(bucket_key.guided)
        velocity = latents.repeat_interleave(rows, dim=0) * 0.25 + text[:, None, :]
        if bucket_key.guided:
            cond_v, uncond_v = self.guidance_branches(velocity, batch)
            velocity = uncond_v + scale.view(batch, 1, 1) * (cond_v - uncond_v)
        return euler_step(latents, velocity, sigma, sigma_next)


class _TokenPool(Resource):
    """A finite pool the step's tokens reserve while it runs: an over-subscribed step is
    refused, as a paged KV pool refuses one."""

    name = ATTN

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.reserved = 0

    @classmethod
    def build(cls, spec, device, comm_group, **engine_kwargs):
        raise NotImplementedError("built directly")

    def depends_on(self):
        return set()

    def admit(self, step, ctx):
        tokens = sum(segment.span for segment in step.segments)
        if self.reserved + tokens > self.capacity:
            return AdmitOutcome(ok=False, reason=AdmitFailedReason(message=f"{tokens} tokens over the pool"))
        self.reserved += tokens
        return AdmitOutcome(ok=True)

    def rollback_admit(self, step, ctx):
        self.reserved -= sum(segment.span for segment in step.segments)

    def plan(self, step, ctx):
        return None

    def commit(self, step, ctx):
        self.reserved -= sum(segment.span for segment in step.segments)


def _info(rid: str, handle: int, k: int, request) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=WALK, fwd_index=k, random_seed=request["seed"], max_tokens=0,
        rid_handle=handle,
        step_metadata={
            "tokens": request["tokens"], "guidance_scale": request["guidance"],
            "num_inference_steps": request["steps"],
        },
        dynamic_loop_iter_counts={LOOP: k},
    )


def _conditioning(request) -> dict:
    gen = torch.Generator().manual_seed(request["seed"] + 1)
    return {
        "cond": [torch.randn(4, generator=gen)],
        "uncond": [torch.randn(4, generator=gen)],
        "scale": [torch.tensor(float(request["guidance"]))],
    }


# (rids in the step, whether the pool is taken beforehand). The pool holds a two-request
# step of one rank's rows, so the taken pool refuses what would otherwise run.
SCRIPT = [
    (("a",), False), (("a", "b"), False), (("a", "b"), True), (("a", "b"), False),
    (("b",), False), (("c",), False), (("c", "d"), False), (("c",), True), (("c", "d"), False),
]
REQUESTS = {
    "a": {"seed": 11, "tokens": TOKENS, "guidance": 5.0, "steps": 4},
    "b": {"seed": 12, "tokens": TOKENS, "guidance": 5.0, "steps": 3},
    # guidance off: a step with nothing to exchange
    "c": {"seed": 13, "tokens": TOKENS, "guidance": 1.0, "steps": 2},
    "d": {"seed": 14, "tokens": TOKENS, "guidance": 1.0, "steps": 1},
}


def _run_script(submodule, capacity: int, log: list) -> dict[str, torch.Tensor]:
    """The sweep the engine runs per step: prepare, declare, admit, plan, forward, commit."""
    pool = _TokenPool(capacity)
    runner = StepRunner({ATTN: pool})
    handles = {rid: i for i, rid in enumerate(REQUESTS)}
    done = dict.fromkeys(REQUESTS, 0)
    latents: dict[str, torch.Tensor] = {}
    device = torch.device("cpu")
    submodule.get_device = lambda: device
    for rids, pool_taken in SCRIPT:
        rids = [rid for rid in rids if done[rid] < REQUESTS[rid]["steps"]]
        if not rids:
            continue
        if pool_taken:
            pool.reserved += capacity
        inputs = []
        for rid in rids:
            edges = _conditioning(REQUESTS[rid])
            if rid in latents:
                edges[LATENTS] = [latents[rid]]
            info = _info(rid, handles[rid], done[rid], REQUESTS[rid])
            inputs.append(submodule.prepare_inputs(WALK, info, edges))
        row_ids = [handles[rid] for rid in rids]
        step = submodule.declare_step(WALK, row_ids, inputs)
        step.set_ctx(StepContext(request_ids=row_ids, graph_walk=WALK, slot=0, capture=False))
        outcome = runner.admit(step)
        log.append(["admit", rids, bool(outcome.ok)])
        if pool_taken:
            pool.reserved -= capacity
        if not outcome.ok:
            continue
        runner.plan(step)
        engine_inputs = ModelInputsFromEngine(request_ids=row_ids, per_request_info={})
        kwargs = submodule.preprocess(WALK, engine_inputs, inputs)
        outputs = submodule.forward_batched(WALK, engine_inputs, **kwargs)
        runner.commit(step)
        for rid in rids:
            latents[rid] = outputs[handles[rid]][LATENTS][0]
            done[rid] += 1
    return latents


def _single_rank_latents() -> dict[str, torch.Tensor]:
    """Every row on one rank: what the placement has to reproduce."""
    # room for both branches of the widest step, so its refusals fall where the two ranks' do
    return _run_script(_TwoRowDenoise(), 4 * TOKENS, [])


def _rank_main(rank, store, out_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2, timeout=timedelta(seconds=60),
    )
    torch.set_num_threads(1)
    group = CommGroup(rank, rank, [0, 1])
    group.device_group = group.cpu_group = dist.group.WORLD
    collectives: list = []
    real = dist.all_gather_into_tensor

    def recording_all_gather(output, tensor, group=None, **kwargs):
        collectives.append(["all_gather", list(tensor.shape), str(tensor.dtype)])
        return real(output, tensor, group=group, **kwargs)

    dist.all_gather_into_tensor = recording_all_gather
    log: list = []
    latents = _run_script(_TwoRowDenoise(GuidanceParallel(group)), 2 * TOKENS, log)
    Path(out_dir, f"{rank}.json").write_text(json.dumps({
        "admits": log, "collectives": collectives,
        "latents": {rid: tensor.tolist() for rid, tensor in latents.items()},
    }))
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def two_ranks(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("cfg2")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(_rank_main, args=(str(out_dir / "store"), str(out_dir)), nprocs=2, join=True)
    return [json.loads((out_dir / f"{rank}.json").read_text()) for rank in range(2)]


def test_both_ranks_run_the_same_collectives_in_the_same_order(two_ranks):
    first, second = two_ranks
    assert first["collectives"] == second["collectives"]
    # one exchange per admitted guided step, at this rank's own rows, and none for guidance off
    assert [entry[1][0] for entry in first["collectives"]] == [1, 2, 2, 1]


def test_both_ranks_take_every_admit_decision_alike(two_ranks):
    first, second = two_ranks
    assert first["admits"] == second["admits"]
    decisions = [admitted for _, _, admitted in first["admits"]]
    assert False in decisions and True in decisions


def test_the_latents_stay_replicated_and_match_one_rank_running_both_rows(two_ranks):
    first, second = two_ranks
    assert first["latents"] == second["latents"]
    for rid, expected in _single_rank_latents().items():
        assert torch.equal(torch.tensor(first["latents"][rid], dtype=expected.dtype), expected), rid


def test_the_mesh_places_a_branch_per_rank_and_tiles_instances():
    wg = WorkerGraph(section=None, graph_walks={WALK}, ranks=[0, 1, 2, 3], cfg_size=2)
    assert wg._instance_ranks == [[0, 1], [2, 3]]
    assert wg._cfg_ranks == [[0, 1], [2, 3]]
    assert wg._tp_ranks == [[0], [1], [2], [3]]
    assert wg.tp_size == 2  # the conductor's lockstep unit


def test_a_guidance_branch_can_itself_be_tensor_parallel():
    wg = WorkerGraph(section=None, graph_walks={WALK}, ranks=[0, 1, 2, 3], tp_size=2, cfg_size=2)
    assert wg._instance_ranks == [[0, 1, 2, 3]]
    assert wg._tp_ranks == [[0, 1], [2, 3]]
    assert wg._cfg_ranks == [[0, 2], [1, 3]]
    assert wg.tp_size == 4


def _cfg2_view():
    class _Section:
        def get_nodes(self):
            return {"dit": None}

    wg = WorkerGraph(section=_Section(), graph_walks={WALK}, ranks=[0, 1], cfg_size=2)
    return GlobalParallelConfig({0: wg}, ["worker_0", "worker_1"]).per_worker_config


def test_rank_zero_of_the_guidance_group_leads_the_instance():
    per_worker = _cfg2_view()
    assert [per_worker[w].get_instance_rank_for_node("dit") for w in ("worker_0", "worker_1")] == [0, 1]
    assert all(view.get_instance_world_size_for_node("dit") == 2 for view in per_worker.values())
    assert per_worker["worker_0"].world_parallel_groups == [(0, 1)]
    assert per_worker["worker_0"].any_parallelism


def test_the_joint_group_addresses_the_guidance_axis():
    group = CommGroup(3, 1, [1, 3])
    joint = JointGroups(tp_group=CommGroup(3, 1, [2, 3]), sp_group=CommGroup.trivial(), cfg_group=group)
    assert (joint.rank, joint.world_size) == (3, 4)


def test_the_placement_is_handed_over_only_where_the_config_asks():
    per_worker = _cfg2_view()
    placed = parallel_kwargs("dit", per_worker["worker_1"], {})["guidance_parallel"]
    assert not placed.runs_cond
    assert parallel_kwargs("dit", per_worker["worker_0"], {})["guidance_parallel"].runs_cond
    # a node outside the group, and a deployment that never asked
    assert parallel_kwargs("vae_decoder", per_worker["worker_0"], {}) == {}
    wg = WorkerGraph(section=None, graph_walks={WALK}, ranks=[0])
    assert wg.cfg_size == 1 and wg._cfg_ranks == []


def test_the_placement_needs_exactly_two_ranks():
    with pytest.raises(ValueError, match="has 3"):
        GuidanceParallel(CommGroup(0, 0, [0, 1, 2]))


def _wan22(tmp_path, **group):
    from mstar.model.wan22.wan22_model import Wan22Model

    model = Wan22Model("Wan-AI/Wan2.2-TI2V-5B-Diffusers", skip_weight_loading=True, **group.pop("kwargs", {}))
    config = {
        "model": "wan22", "max_seq_len": 512,
        "node_groups": [
            {"node_names": ["text_encoder", "vae_encoder", "vae_decoder"], "ranks": [0]},
            {"node_names": ["dit"], "ranks": [0, 1], **group},
        ],
    }
    path = tmp_path / "wan22.yaml"
    path.write_text(yaml.safe_dump(config))
    return model, config, str(path)


def test_wan22_serves_the_placement_and_the_shipped_config_asks_for_it(tmp_path):
    model, config, path = _wan22(tmp_path, cfg_size=2)
    validate_deployment(model, config, path)
    shipped = yaml.safe_load(Path("configs/wan22_cfg2.yaml").read_text())
    assert [group.get("cfg_size", 1) for group in shipped["node_groups"]].count(2) == 1
    single = yaml.safe_load(Path("configs/wan22.yaml").read_text())
    assert all("cfg_size" not in group for group in single["node_groups"])


def test_the_legacy_loop_refuses_the_placement(tmp_path):
    model, config, path = _wan22(tmp_path, cfg_size=2, kwargs={"dit_loop": "legacy"})
    with pytest.raises(DeploymentError, match="cfg_enabled_nodes"):
        validate_deployment(model, config, path)


def test_cfg_size_places_two_branches(tmp_path):
    model, config, path = _wan22(tmp_path, cfg_size=3)
    with pytest.raises(DeploymentError, match="must be 1 or 2"):
        validate_deployment(model, config, path)
