import dataclasses

import torch

from mstar.model.waypoint.config import WaypointConfig

__all__ = ["describe_ring_memory", "ring_memory_bytes"]


def ring_memory_bytes(
    config: WaypointConfig, *, num_worlds: int = 1, dtype: torch.dtype = torch.bfloat16
) -> list[int]:
    """Per-layer ring bytes for ``config``, computed without allocating anything
    (so it can be called on a laptop while sizing a deployment).

    ``num_worlds`` is a straight multiplier: the world dimension is folded into
    the token axis, so N worlds is N times one world's slots and the per-slot
    arithmetic is untouched. That it is a multiplier is the whole reason
    ``num_worlds`` is a deployment sizing knob and the geometry is not."""
    per_slot = 2 * num_worlds * config.n_kv_heads * config.d_head * dtype.itemsize
    return [per_slot * config.kv_capacity(i) for i in range(config.n_layers)]


def _fmt_bytes(n: int) -> str:
    return f"{n / 2**20:.1f} MiB" if n < 2**30 else f"{n / 2**30:.2f} GiB"


def describe_ring_memory(
    config: WaypointConfig, *, num_worlds: int = 1, dtype: torch.dtype = torch.bfloat16
) -> str:
    """Geometry and footprint of every ring, grouped local vs global, with the
    counterfactual under the opposite ``full_global_ring`` setting.

    The compacted default is what a reviewer should see; the ``full_global_ring``
    line is the reference's allocation, 8/9ths of whose global storage is
    permanently unwritten.
    """
    per_layer = ring_memory_bytes(config, num_worlds=num_worlds, dtype=dtype)
    total = sum(per_layer)

    lines = [
        f"Waypoint ring KV  variant={config.variant}  worlds={num_worlds}  dtype={dtype}  "
        f"full_global_ring={config.full_global_ring}"
    ]
    groups = (
        ("local ", [i for i in range(config.n_layers) if not config.is_global_layer(i)]),
        ("global", sorted(config.global_layers)),
    )
    for name, indices in groups:
        if not indices:
            continue
        i = indices[0]
        lines.append(
            f"  {name} x{len(indices):2d}  "
            f"{config.ring_frames(i):3d} ring frames @ stride {config.pinned_dilation(i)} "
            f"({config.ring_buckets(i)} addressable) + 1 scratch  "
            f"= {config.kv_capacity(i):6d} tok  "
            f"= {_fmt_bytes(per_layer[i]):>9s}/layer  "
            f"= {_fmt_bytes(sum(per_layer[j] for j in indices)):>9s}"
        )
    lines.append(f"  total {_fmt_bytes(total)}  ({total} bytes)")

    other = dataclasses.replace(config, full_global_ring=not config.full_global_ring)
    other_total = sum(ring_memory_bytes(other, num_worlds=num_worlds, dtype=dtype))
    delta = other_total - total
    lines.append(
        f"  full_global_ring={other.full_global_ring} would use {_fmt_bytes(other_total)} "
        f"({'+' if delta > 0 else '-'}{_fmt_bytes(abs(delta))})"
    )
    return "\n".join(lines)
