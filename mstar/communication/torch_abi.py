"""Whether ``mstar_rust``'s torch wrapper runs on this process's torch, checked
once at startup so a mismatch is an error here and not a crash mid-request.

The wrapper reads torch's inline tensor layout (``rust/src/torch_abi/``), which
torch does not promise to keep. So a version no implementor lists is refused, and
a listed one has to agree with Python about one real tensor before anything
relies on it.
"""
from __future__ import annotations

import functools
import sys

import torch

from mstar.platforms import current_platform


def _mismatch(what: str, got, want) -> str:
    return f"{what}: shim says {got!r}, torch says {want!r}"


def _self_test(device: torch.device) -> list[str]:
    import mstar_rust as rs

    wrong: list[str] = []

    def check_facts(name: str, t: torch.Tensor) -> None:
        facts = rs.torch_abi_facts(t)
        expect = {
            "data": t.data_ptr(), "nbytes": t.numel() * t.element_size(),
            "device_type": 1 if t.is_cuda else 0, "ndim": t.dim(),
            "dims": list(t.shape), "contiguous": t.is_contiguous(), "plain": True,
        }
        wrong.extend(
            _mismatch(f"{name}.{field}", facts[field], want)
            for field, want in expect.items() if facts[field] != want
        )
        if facts["dtype"] != rs.torch_abi_dtype(t.dtype):
            wrong.append(_mismatch(f"{name}.dtype", facts["dtype"], str(t.dtype)))

    base = (torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4) * 1.5)[1:]
    check_facts("view", base)
    check_facts("permuted", base.permute(0, 2, 1))
    made = rs.torch_abi_empty(
        [3, 5], torch.int64, 1 if device.type == "cuda" else 0,
        device.index if device.index is not None else -1,
    )
    if type(made) is not torch.Tensor or made.shape != (3, 5) or made.dtype != torch.int64 \
            or made.device != device:
        wrong.append(_mismatch("empty", (type(made), made.shape, made.dtype, made.device),
                               (torch.Tensor, (3, 5), torch.int64, device)))
    check_facts("empty", made)
    return wrong


@functools.cache
def _checked(device: torch.device) -> str:
    import mstar_rust as rs

    described = rs.torch_abi_load(torch.__version__)
    wrong = _self_test(device)
    if wrong:
        raise RuntimeError(
            f"mstar_rust's torch wrapper ({described}) disagrees with torch "
            f"{torch.__version__} on {device}: {'; '.join(wrong)}. It reads torch's tensor "
            "layout, so this torch needs its own implementor in rust/src/torch_abi/."
        )
    return described


def require_torch_abi(device: str | torch.device = "cpu") -> bool:
    """Load the wrapper and self-test it on ``device``, raising if torch is not
    one it supports. Returns False on a platform with no wrapper at all (it is
    Linux-only, like the serving wheels), whose callers keep to torch calls."""
    import mstar_rust as rs

    if not hasattr(rs, "torch_abi_load"):
        raise ImportError(
            f"The mstar_rust build at {rs.__file__} predates the torch wrapper. Rebuild it with "
            "`maturin develop --release -m rust/Cargo.toml`."
        )
    if sys.platform != "linux":
        return False
    device = torch.device(device)
    # Only NVIDIA devices are tested: the wrapper never reads a ROCm or XPU tensor
    # (see ArenaShmCommunicationManager), but its CPU paths still run there.
    if device.type != "cuda" or not current_platform.is_cuda():
        device = torch.device("cpu")
    elif device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    _checked(device)
    return True
