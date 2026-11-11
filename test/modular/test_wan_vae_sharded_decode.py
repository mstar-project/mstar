"""CPU checks for the height-sharded Wan VAE decode (``HeightShardedDecoder`` in
mstar/model/components/wan_vae.py): each simulated rank decodes its band of
rows and the gathered frames are the whole-clip decode's, for both decoder
families, even and uneven splits, a single image latent, and a clip too short
to split. Tiny random VAEs from test_wan_vae_decode; no weights, no GPU."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from test_wan_vae_decode import tiny_wan_vae  # noqa: E402

from mstar.model.components import wan_vae  # noqa: E402
from mstar.testing_utils.comm import run_ranks  # noqa: E402


def _latents(vae, frames: int, rows: int, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, vae.config.z_dim, frames, rows, 6, generator=gen)


def _sharded_decode(kind: str, z: torch.Tensor, world: int) -> list[torch.Tensor]:
    def rank_run(rank, group):
        vae = tiny_wan_vae(kind)
        decoder = wan_vae.HeightShardedDecoder(vae, group, compile=False)
        with torch.no_grad():
            return decoder.gather(decoder.decode(z))
    return run_ranks(world, rank_run)


@pytest.mark.parametrize("kind", ("wan21", "wan22"))
@pytest.mark.parametrize("frames, rows, world", [
    (5, 4, 2),   # even bands
    (5, 5, 2),   # uneven: 3 + 2 latent rows
    (5, 7, 3),   # three ranks, uneven
    (1, 4, 2),   # one latent: an image
    (5, 1, 2),   # fewer rows than ranks: each rank decodes it whole
])
def test_sharded_decode_is_the_whole_clip_decode(kind, frames, rows, world) -> None:
    vae = tiny_wan_vae(kind)
    z = _latents(vae, frames, rows)
    with torch.no_grad():
        ref = vae.decode(z).sample
    for out in _sharded_decode(kind, z, world):
        assert out.shape == ref.shape
        # same math; the convs only see other shapes, so allow float rounding
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_split_rows_puts_the_extra_rows_last() -> None:
    """Rank 0 leads and holds more beside the decode, so it gets the smaller band."""
    assert wan_vae.split_rows(30, 2) == (15, 15)
    assert wan_vae.split_rows(45, 2) == (22, 23)
    assert wan_vae.split_rows(7, 3) == (2, 2, 3)
    assert wan_vae.split_rows(1, 2) == (0, 1)


def test_a_sharded_vae_refuses_a_stock_decode() -> None:
    """Its layers expect a band; switching it back to the stock layout would
    hand them whole frames, so that is refused."""
    def rank_run(rank, group):
        decoder = wan_vae.HeightShardedDecoder(tiny_wan_vae("wan22"), group, compile=False)
        with pytest.raises(ValueError):
            decoder.use_stock_layout()
        with pytest.raises(ValueError):
            wan_vae.to_default_layout(decoder.vae)
        return wan_vae.is_height_sharded(decoder.vae)
    assert run_ranks(2, rank_run) == [True, True]


def test_the_warm_up_key_carries_the_row_split() -> None:
    """A compiled step's warm-up runs collectives, so it must be decided alike on
    every rank: the key carries the clip's split, shared by all of them, not
    only this rank's band shapes (which one rank can repeat and another not)."""
    def rank_run(rank, group):
        decoder = wan_vae.HeightShardedDecoder(tiny_wan_vae("wan22"), group, compile=False)
        vae = decoder.vae
        keys = []
        with torch.no_grad():
            for rows in (5, 4):  # rank 0's band is 2 rows both times; rank 1's 3 then 2
                decoder.decode(_latents(vae, 2, rows))
                keys.append(decoder.step.key_extra())
        return keys
    assert run_ranks(2, rank_run) == [[((2, 3),), ((2, 2),)]] * 2


@pytest.mark.parametrize("module", [
    torch.nn.Conv2d(4, 4, 3, padding=0),          # a 3-row kernel without padding would shrink each band
    torch.nn.Conv2d(4, 4, (1, 3), padding=(1, 1)),  # padding rows a 1-row kernel never reads
    torch.nn.Conv2d(4, 4, 3, padding=1, stride=2),
    torch.nn.Conv2d(4, 4, 3, padding=1, dilation=2),
    torch.nn.Conv2d(4, 4, 3, padding=1, padding_mode="reflect"),  # the halo pads with zeros
])
def test_a_layer_that_cannot_split_exactly_is_refused(module) -> None:
    module.__class__ = wan_vae.ChannelsLastConv2d
    with pytest.raises(ValueError):
        wan_vae.HeightShardedConv2d.adopt(module)


def test_a_row_crossing_layer_without_a_band_class_is_refused() -> None:
    """A layer the conversion doesn't know must work per row, or sharding is refused."""
    vae = tiny_wan_vae("wan22")
    wan_vae.to_channels_last(vae)
    vae.decoder.add_module("extra", torch.nn.Conv3d(4, 4, 3, padding=1))
    with pytest.raises(ValueError, match="reaches across rows"):
        wan_vae.to_height_sharded(vae, None)
    vae = tiny_wan_vae("wan22")
    wan_vae.to_channels_last(vae)
    vae.decoder.add_module("extra", torch.nn.Upsample(scale_factor=2, mode="bilinear"))
    with pytest.raises(ValueError, match="blends rows"):
        wan_vae.to_height_sharded(vae, None)


def _cosmos3_decoder(monkeypatch, group=None, fast=True):
    from mstar.model.cosmos3 import submodules
    from mstar.model.cosmos3.config import Cosmos3Config

    monkeypatch.setenv("COSMOS3_COMPILE_VAE", "0")
    monkeypatch.setattr(submodules, "wan_vae_decode_dtype", lambda: torch.float32)
    return submodules.Cosmos3VAEDecoderSubmodule(tiny_wan_vae("wan22"), Cosmos3Config(), fast=fast, tp_group=group)


def test_cosmos3_vae_decoder_on_two_ranks_decodes_like_one(monkeypatch) -> None:
    """The node's tp_group reaches the decode: two ranks give the one-rank node's
    uint8 frames (an uneven split, 5 latent rows)."""
    latents = _latents(tiny_wan_vae("wan22"), 3, 5)
    ref = _cosmos3_decoder(monkeypatch).forward("video_gen", None, latents.clone())["video_output"][0]
    outs = run_ranks(2, lambda rank, group: _cosmos3_decoder(monkeypatch, group).forward(
        "video_gen", None, latents.clone())["video_output"][0])
    for out in outs:
        assert out.shape == ref.shape
        # fp32 rounding can tip a value across a .5 boundary
        assert (out.int() - ref.int()).abs().max().item() <= 1


def test_cosmos3_vae_decoder_on_two_ranks_needs_the_fast_decode(monkeypatch) -> None:
    def rank_run(rank, group):
        with pytest.raises(ValueError, match="vae_decode_fast"):
            _cosmos3_decoder(monkeypatch, group, fast=False)
    run_ranks(2, rank_run)


# --- On two GPUs: NCCL, compiled ------------------------------------------------

def _gpu_worker(rank: int, store_path: str, out_dir: str) -> None:
    import datetime
    import os

    import torch.distributed as dist

    from mstar.distributed.communication import CommGroup

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{store_path}", rank=rank, world_size=2,
                            device_id=torch.device("cuda", rank), timeout=datetime.timedelta(seconds=90))
    group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
    group.device_group = dist.group.WORLD
    dev = torch.device("cuda", rank)
    results = {}
    # fp32: the compiled sharded decode against the compiled one-rank decode
    sharded = wan_vae.HeightShardedDecoder(tiny_wan_vae("wan22").to(dev), group, compile=True)
    whole = wan_vae.FastDecoder(tiny_wan_vae("wan22").to(dev), compile=True)
    with torch.no_grad():
        for frames, rows in ((5, 4), (5, 5), (1, 4)):  # even, uneven, an image
            z = _latents(tiny_wan_vae("wan22"), frames, rows).to(dev)
            results[(frames, rows)] = (sharded.gather(sharded.decode(z)).cpu(), whole.decode(z).cpu())
        # bf16, where a compiled step warms each new call up with its collectives: splits whose
        # bands repeat on one rank and not the other (2+3 then 2+2 then 2+3) must not hang
        bf16 = wan_vae.HeightShardedDecoder(tiny_wan_vae("wan22").to(dev, torch.bfloat16), group, compile=True)
        for i, rows in enumerate((5, 4, 5)):
            z = _latents(tiny_wan_vae("wan22"), 4, rows).to(dev, torch.bfloat16)
            results[("bf16", i)] = bf16.gather(bf16.decode(z)).float().cpu()
    torch.save(results, os.path.join(out_dir, f"{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_compiled_sharded_decode_on_two_gpus(tmp_path) -> None:
    import torch.multiprocessing as mp

    mp.spawn(_gpu_worker, args=(str(tmp_path / "store"), str(tmp_path)), nprocs=2)
    for rank in (0, 1):
        results = torch.load(tmp_path / f"{rank}.pt")
        for (frames, rows), (out, ref) in ((k, v) for k, v in results.items() if k[0] != "bf16"):
            assert out.shape == ref.shape, (frames, rows)
            torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
        for i in range(3):
            assert torch.isfinite(results[("bf16", i)]).all()
