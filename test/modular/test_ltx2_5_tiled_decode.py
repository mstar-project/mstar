"""LTX-2.5 spatial tile split against diffusers' own tiled decode."""

from __future__ import annotations

import math
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import yaml

from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.distributed.communication import CommGroup, GlobalParallelConfig
from mstar.engine.engine import Engine
from mstar.engine.resources.sequence_shard import SeqShardSpec, sequence_parallel_refusals
from mstar.graph.base import GraphEdge, GraphNode, NodeAndGraphWalk
from mstar.graph.runtime.base import RouteInput
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import WorkerGraph
from mstar.model.ltx2_5.config import LTX25Config
from mstar.model.ltx2_5.ltx2_5_model import GENERATE_VIDEO_WALK, LTX25Model
from mstar.model.ltx2_5.submodules import VIDEO_OUTPUT, LTXVideoDecoderSubmodule
from mstar.model.ltx2_5.tiled_decode import distributed_tiled_decode, spatial_tile_count
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.containers import ParallelList
from mstar.validation import DeploymentError, validate_deployment
from mstar.validation.placement import check_worker_graphs


class _ToyDecoder(torch.nn.Module):
    def forward(self, z, temb=None, causal=None):
        return z.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4) * 0.375


def _toy_vae():
    # Use the published diffusers assembler itself as the reference. Only its
    # expensive convolutional decoder is replaced with a deterministic kernel.
    from diffusers import AutoencoderKLLTX2Video

    class VAE(AutoencoderKLLTX2Video):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.decoder = _ToyDecoder()
            self.spatial_compression_ratio = 2
            self.tile_sample_min_height = self.tile_sample_min_width = 8
            self.tile_sample_stride_height = self.tile_sample_stride_width = 6

    return VAE()


def _small_real_vae():
    from diffusers import AutoencoderKLLTX2Video

    vae = AutoencoderKLLTX2Video(
        latent_channels=4, block_out_channels=(32, 32), decoder_block_out_channels=(32, 32),
        down_block_types=("LTX2VideoDownBlock3D", "LTX2VideoDownBlock3D"),
        layers_per_block=(1, 1, 1), decoder_layers_per_block=(1, 1, 1),
        spatio_temporal_scaling=(False, False), decoder_spatio_temporal_scaling=(False, False),
        decoder_inject_noise=(False, False, False),
        downsample_type=("spatial", "spatial"), upsample_type=("spatial", "spatial"),
        upsample_residual=(False, False), upsample_factor=(1, 1),
        patch_size=2, patch_size_t=1,
    )
    vae.tile_sample_min_height = vae.tile_sample_min_width = 8
    vae.tile_sample_stride_height = vae.tile_sample_stride_width = 6
    return vae.eval()


def _rank_check(rank: int, world_size: int, init_file: str, results, real_decoder: bool) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(123)
        group = CommGroup(rank, rank, list(range(world_size)))
        group.device_group = dist.group.WORLD
        vae = _small_real_vae() if real_decoder else _toy_vae()
        # 3x3 tiles, with the last row and column smaller than the others.
        shape = (1, 4, 2, 9, 9) if real_decoder else (1, 2, 3, 7, 7)
        z = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) / 97
        with torch.inference_mode():
            reference = vae.tiled_decode(z, None, return_dict=False)[0]
            got = distributed_tiled_decode(vae, z, group)
        results.put((rank, torch.equal(got, reference) if got is not None else False,
                     tuple(got.shape) if got is not None else None))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size,real_decoder", [(2, False), (4, False), (2, True)])
def test_split_spatial_tiles_match_stock_diffusers_assembler(world_size: int, real_decoder: bool, tmp_path: Path):
    pytest.importorskip("diffusers")
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    init_file = str(tmp_path / "gloo-init")
    procs = [ctx.Process(target=_rank_check, args=(rank, world_size, init_file, results, real_decoder))
             for rank in range(world_size)]
    for proc in procs:
        proc.start()
    try:
        observed = [results.get(timeout=60) for _ in procs]
        shape = (1, 3, 2, 18, 18) if real_decoder else (1, 2, 3, 14, 14)
        assert sorted(observed) == [(0, True, shape)] + [
            (rank, False, None) for rank in range(1, world_size)
        ]
    finally:
        for proc in procs:
            proc.join(timeout=20)
            if proc.is_alive():
                proc.terminate()
                proc.join()
        assert all(proc.exitcode == 0 for proc in procs)


def test_tile_count_covers_partial_edge_tiles():
    pytest.importorskip("diffusers")
    vae = _toy_vae()
    assert spatial_tile_count(vae, torch.zeros(1, 2, 3, 7, 7)) == 9


def test_non_sequence_sp_requires_explicit_capability_and_no_engine_shard():
    assert "neither splits tokens" in sequence_parallel_refusals([], "vae_decoder", 1, 2)[0]
    assert sequence_parallel_refusals([], "vae_decoder", 1, 2, sp_collective=True) == []
    shard = SeqShardSpec("vae_shard", {"vae_decoder"}, ("tiles",), 0)
    assert "also declares engine resources" in sequence_parallel_refusals(
        [shard], "vae_decoder", 1, 2, sp_collective=True,
    )[0]
    assert sequence_parallel_refusals([], "vae_decoder", 1, 1, sp_collective=True) == []
    assert "2 TP ranks" in sequence_parallel_refusals([], "vae_decoder", 2, 1, sp_collective=True)[0]
    assert "2 TP ranks" in sequence_parallel_refusals([], "vae_decoder", 2, 2, sp_collective=True)[0]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("fallback,batch", [
    ("untiled", 1), ("too_few_tiles", 1), ("framewise", 1), ("sliced", 2),
])
def test_sp_fallback_decodes_only_on_leader(rank, fallback, batch, monkeypatch):
    from mstar.model.ltx2_5 import submodules

    decoder = LTXVideoDecoderSubmodule.__new__(LTXVideoDecoderSubmodule)
    torch.nn.Module.__init__(decoder)
    calls = []
    x = torch.ones(batch, 1, 1, 3, 3)

    def decode(z, *args, **kwargs):
        calls.append(tuple(z.shape))
        return (z,)

    decoder.vae = SimpleNamespace(
        dtype=torch.float32, latents_mean=torch.zeros(1), latents_std=torch.ones(1),
        config=SimpleNamespace(scaling_factor=1.0), decode=decode,
        use_framewise_decoding=fallback == "framewise", use_slicing=fallback == "sliced",
        spatial_compression_ratio=2, tile_sample_min_height=4, tile_sample_min_width=4,
        tile_sample_stride_height=8, tile_sample_stride_width=8,
    )
    decoder.comm_group = SimpleNamespace(rank=rank, world_size=2)
    decoder.get_device = lambda: torch.device("cpu")
    decoder._tiled = lambda shape: fallback != "untiled"
    monkeypatch.setattr(submodules, "unpack_video", lambda latents, shape: latents)
    out = decoder.run_batch(x, shape=None)
    assert bool(out) is (rank == 0)
    assert calls == ([tuple(x.shape)] if rank == 0 else [])


def test_direct_too_few_tile_fallback_is_leader_only():
    z = torch.ones(1, 1, 1, 2, 2)
    calls = []
    vae = SimpleNamespace(
        spatial_compression_ratio=2,
        tile_sample_min_height=8, tile_sample_min_width=8,
        tile_sample_stride_height=8, tile_sample_stride_width=8,
        decode=lambda *args, **kwargs: (calls.append(1) or z,),
    )
    assert distributed_tiled_decode(vae, z, SimpleNamespace(rank=1, world_size=2)) is None
    assert calls == []
    assert torch.equal(distributed_tiled_decode(vae, z, SimpleNamespace(rank=0, world_size=2)), z)
    assert calls == [1]


def test_opt_in_graph_replicates_latents_but_only_rank_zero_emits():
    model = LTX25Model(model_path_hf="test/ltx2_5", skip_weight_loading=True)
    model.set_config(LTX25Config())
    path = "configs/ltx2_5_sp2_tiled_vae.yaml"
    graphs = model.get_worker_graphs(path)
    sharding = model.get_sharding_config(path)
    assert "vae_decoder" not in sharding.tp_enabled_nodes
    parallel = GlobalParallelConfig(dict(enumerate(graphs)), ["worker_0", "worker_1"])
    for rank in (0, 1):
        groups = parallel.per_worker_config[f"worker_{rank}"]
        vae = groups.get_joint_group_for_node("vae_decoder")
        dit = groups.get_joint_group_for_node("dit")
        assert vae.sp_group.world_size == dit.sp_group.world_size == 2
        assert vae.tp_group.world_size == dit.tp_group.world_size == 1
        assert model.submodule_comm_group("vae_decoder", vae) is vae.sp_group
        assert model.submodule_comm_group("dit", dit) is dit.tp_group
    node_to_workers = {
        NodeAndGraphWalk(node, walk): [f"worker_{rank}" for rank in graph.ranks]
        for graph in graphs
        for walk in graph.graph_walks
        for node in graph.section.get_nodes()
    }
    sharding.setup(node_to_workers)
    latents_zero = sharding.compute_fanout(
        "latents", GENERATE_VIDEO_WALK, "dit", "vae_decoder", [], GENERATE_VIDEO_WALK,
        source_tp_rank=0,
    )
    latents_one = sharding.compute_fanout(
        "latents", GENERATE_VIDEO_WALK, "dit", "vae_decoder", [], GENERATE_VIDEO_WALK,
        source_tp_rank=1,
    )
    assert {dest.worker for dest in latents_zero} == {"worker_0"}
    assert {dest.worker for dest in latents_one} == {"worker_1"}
    rank_zero = sharding.compute_fanout(
        "video_output", GENERATE_VIDEO_WALK, "vae_decoder", EMIT_TO_CLIENT, [], None,
        source_tp_rank=0,
    )
    rank_one = sharding.compute_fanout(
        "video_output", GENERATE_VIDEO_WALK, "vae_decoder", EMIT_TO_CLIENT, [], None,
        source_tp_rank=1,
    )
    assert len(rank_zero) == 1 and rank_zero[0].worker == ""
    assert rank_one == []


def test_sp_collective_refuses_tp_placement_at_startup(tmp_path):
    model = LTX25Model(model_path_hf="test/ltx2_5", skip_weight_loading=True)
    model.set_config(LTX25Config())
    config = yaml.safe_load(Path("configs/ltx2_5_sp2_tiled_vae.yaml").read_text())
    config["node_groups"][2].pop("sp_size")
    config["node_groups"][2]["tp_size"] = 2
    path = tmp_path / "bad-tp-vae.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(DeploymentError, match="not in tp_enabled_nodes"):
        validate_deployment(model, config, str(path))


def test_engine_rejects_sp_collective_on_tp_ranks():
    class _NeverRuns(NodeSubmodule):
        def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
            raise AssertionError("not stepped")

        def forward(self, graph_walk, engine_inputs, **kwargs):
            raise AssertionError("not stepped")

    graph = GraphNode(name="vae_decoder", input_names=[], outputs=[])
    worker_graph = WorkerGraph(
        section=graph, graph_walks={GENERATE_VIDEO_WALK}, ranks=[0, 1],
        tp_size=2, worker_graph_id=0,
    )
    sharding = ShardingConfig(
        groups=[ShardingGroup(nodes={"vae_decoder"}, tp_size=2)],
        tp_enabled_nodes={"vae_decoder"}, shard_dim={},
        sp_collective_nodes={"vae_decoder"},
    )
    model = LTX25Model(model_path_hf="test/ltx2_5", skip_weight_loading=True)
    model.set_config(LTX25Config())
    problems = check_worker_graphs(model, [worker_graph], sharding, [])
    assert any("non-sequence SP collective but also runs on 2 TP ranks" in p for p in problems)
    groups = GlobalParallelConfig({0: worker_graph}, ["worker_0", "worker_1"])
    with pytest.raises(ValueError, match="non-sequence SP collective but also runs on 2 TP ranks"):
        Engine(None, autocast_dtype=None).load_model(
            {"vae_decoder": _NeverRuns()}, [], groups.per_worker_config["worker_0"],
            torch.device("cpu"), transfer_engine_info=None,
            sp_collective_nodes={"vae_decoder"},
        )


@pytest.mark.parametrize("rank", [0, 1])
def test_follower_completes_worker_route_without_a_video_tensor(rank: int, tmp_path: Path):
    pytest.importorskip("mstar_rust")
    from rust_runtime import build_runtime

    from mstar.communication.tensors import SharedMemoryCommunicationManager
    worker = f"worker_{rank}"
    manager = SharedMemoryCommunicationManager(
        my_entity_id=worker, hostname="localhost", device="cpu",
        communicator=SimpleNamespace(send=lambda *args, **kwargs: None),
        shm_dir=str(tmp_path),
    )
    graph = GraphNode(name="vae_decoder", input_names=[], outputs=[
        GraphEdge(name=VIDEO_OUTPUT, next_node=EMIT_TO_CLIENT, output_modality="video"),
    ])
    sharding_group = ShardingGroup(nodes={"vae_decoder"}, tp_size=2)
    sharding_group._tp_rank = rank
    runtime = build_runtime(
        bookkeeping=manager.tensor_store.bookkeeping, my_worker_id=worker,
        my_worker_graphs=[WorkerGraph(
            section=graph, graph_walks={GENERATE_VIDEO_WALK}, ranks=[0, 1],
            sp_size=2, worker_graph_id=0,
        )],
        all_wg_ids_to_graph_walks={0: {GENERATE_VIDEO_WALK}},
        all_wg_ids_to_dyn_loops={0: set()},
        all_wg_ids_to_nodes={0: {"vae_decoder"}},
        node_to_partition={"vae_decoder": "default"},
        sharding_config=ShardingConfig(
            groups=[sharding_group],
            tp_enabled_nodes=set(), shard_dim={},
        ),
    )
    rid = runtime.add_request(
        "tile", "default", GENERATE_VIDEO_WALK, [0],
        ParallelList([0], [["worker_0", "worker_1"]]), {},
    )
    manager.register_request(rid, runtime.get_sharding_config(rid))

    decoder = LTXVideoDecoderSubmodule.__new__(LTXVideoDecoderSubmodule)
    torch.nn.Module.__init__(decoder)
    decoder.run_batch = lambda **kwargs: (
        {VIDEO_OUTPUT: torch.ones(1, 2, 4, 4, 3, dtype=torch.uint8)} if rank == 0 else {}
    )
    one = decoder.forward(GENERATE_VIDEO_WALK, SimpleNamespace(request_ids=[rid]))
    assert set(one) == ({VIDEO_OUTPUT} if rank == 0 else set())
    outputs = decoder.forward_batched(
        GENERATE_VIDEO_WALK, SimpleNamespace(request_ids=[rid]),
    )
    assert set(outputs) == {rid}
    if rank == 0:
        assert torch.equal(outputs[rid][VIDEO_OUTPUT][0], torch.ones(2, 4, 4, 3, dtype=torch.uint8))
    else:
        assert outputs[rid] == {}
    stored = manager.store_and_return_tensor_info_batch(
        [rid], outputs, [VIDEO_OUTPUT], node_name="vae_decoder", graph_walk=GENERATE_VIDEO_WALK,
    )
    manager.increment_ref_batch_uniform(stored.flat_uuids, 1)
    route = runtime.complete_and_route_batch(RouteInput(
        partition="default", graph_walk=GENERATE_VIDEO_WALK, node_name="vae_decoder",
        output_signals=[VIDEO_OUTPUT], wg_ids=ParallelList([rid], [0]),
        tensors=stored.flat_uuids, num_tensors=stored.num_tensors,
    ))
    assert stored.num_tensors == ([1] if rank == 0 else [0])
    assert route.register_uuids == (stored.flat_uuids if rank == 0 else [])
