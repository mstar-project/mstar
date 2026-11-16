"""LTX-2.5 spatial tile split against diffusers' own tiled decode."""

from __future__ import annotations

import math
import multiprocessing as mp
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from mstar.distributed.communication import CommGroup
from mstar.graph.base import NodeAndGraphWalk
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.ltx2_5.config import LTX25Config
from mstar.model.ltx2_5.ltx2_5_model import GENERATE_VIDEO_WALK, LTX25Model
from mstar.model.ltx2_5.tiled_decode import distributed_tiled_decode, spatial_tile_count


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
        results.put((rank, torch.equal(got, reference), tuple(got.shape)))
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
        assert sorted(observed) == [(rank, True, shape) for rank in range(world_size)]
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


def test_opt_in_graph_replicates_latents_but_only_rank_zero_emits():
    model = LTX25Model(model_path_hf="test/ltx2_5", skip_weight_loading=True)
    model.set_config(LTX25Config())
    path = "configs/ltx2_5_sp2_tiled_vae.yaml"
    graphs = model.get_worker_graphs(path)
    sharding = model.get_sharding_config(path)
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
