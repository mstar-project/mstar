"""Split the independent spatial tiles of an LTX-2 video VAE decode over ranks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mstar.distributed.communication import CommGroup


def spatial_tile_count(vae, z: torch.Tensor) -> int:
    """Count the decoder calls that diffusers' spatial tiled decode would make."""
    ratio = vae.spatial_compression_ratio
    stride_h = vae.tile_sample_stride_height // ratio
    stride_w = vae.tile_sample_stride_width // ratio
    return len(range(0, z.shape[-2], stride_h)) * len(range(0, z.shape[-1], stride_w))


def distributed_tiled_decode(vae, z: torch.Tensor, group: CommGroup) -> torch.Tensor:
    """Decode each tile once and assemble it in diffusers' row-major blend order.

    All ranks receive the same decoded tiles and return the same full sample.
    The decoded tile sizes can differ at the right and bottom edges, so one
    all-gather carries their shapes and a second carries padded pixel data.
    """
    ratio = vae.spatial_compression_ratio
    tile_h = vae.tile_sample_min_height // ratio
    tile_w = vae.tile_sample_min_width // ratio
    stride_h = vae.tile_sample_stride_height // ratio
    stride_w = vae.tile_sample_stride_width // ratio
    positions = [
        (h, w)
        for h in range(0, z.shape[-2], stride_h)
        for w in range(0, z.shape[-1], stride_w)
    ]
    if len(positions) < group.world_size:
        return vae.decode(z, None, return_dict=False)[0]

    local_tiles = [
        (idx, vae.decoder(z[:, :, :, h:h + tile_h, w:w + tile_w], None, causal=None))
        for idx, (h, w) in enumerate(positions)
        if idx % group.world_size == group.rank
    ]
    max_owned = math.ceil(len(positions) / group.world_size)
    # [tile index, B, C, T, H, W]; -1 marks a padding slot.
    local_shapes = torch.full((max_owned, 6), -1, dtype=torch.int64, device=z.device)
    for slot, (idx, tile) in enumerate(local_tiles):
        local_shapes[slot] = torch.tensor((idx, *tile.shape), dtype=torch.int64, device=z.device)
    all_shapes = group.all_gather(local_shapes, dim=0).reshape(group.world_size, max_owned, 6).tolist()
    rank_sizes = [
        sum(math.prod(entry[1:]) for entry in rank_shapes if entry[0] >= 0)
        for rank_shapes in all_shapes
    ]
    max_elements = max(rank_sizes)
    local_data = torch.cat([tile.reshape(-1) for _, tile in local_tiles])
    padded = torch.zeros(max_elements, dtype=local_data.dtype, device=z.device)
    padded[:local_data.numel()] = local_data
    all_data = group.all_gather(padded, dim=0).reshape(group.world_size, max_elements)

    tiles = [None] * len(positions)
    for rank, rank_shapes in enumerate(all_shapes):
        offset = 0
        for idx, *shape in rank_shapes:
            if idx < 0:
                continue
            size = math.prod(shape)
            tiles[idx] = all_data[rank, offset:offset + size].view(*shape)
            offset += size
    assert all(tile is not None for tile in tiles)

    rows = []
    row_width = len(range(0, z.shape[-1], stride_w))
    for row_start in range(0, len(positions), row_width):
        rows.append(tiles[row_start:row_start + row_width])
    result_rows = []
    blend_h = vae.tile_sample_min_height - vae.tile_sample_stride_height
    blend_w = vae.tile_sample_min_width - vae.tile_sample_stride_width
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i:
                tile = vae.blend_v(rows[i - 1][j], tile, blend_h)
            if j:
                tile = vae.blend_h(row[j - 1], tile, blend_w)
            result_row.append(tile[:, :, :, :vae.tile_sample_stride_height, :vae.tile_sample_stride_width])
        result_rows.append(torch.cat(result_row, dim=4))
    return torch.cat(result_rows, dim=3)[:, :, :, :z.shape[-2] * ratio, :z.shape[-1] * ratio]
