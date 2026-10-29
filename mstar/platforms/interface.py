"""The hardware platforms M* runs on, and what each reports about itself."""

from __future__ import annotations

from collections.abc import Callable

import torch

from mstar.platforms.ptxas import blackwell_ptxas_env
from mstar.utils.cpu_list import parse_cpu_list


class Platform:
    """One accelerator stack; ``current_platform`` is the one this process runs.

    PyTorch's ROCm build drives AMD GPUs through ``torch.cuda``: their device
    type is ``"cuda"``, and ``torch.cuda.get_device_capability()`` returns the
    GFX version, (9, 4) on an MI300X, which compares like an NVIDIA Hopper.
    Code that branches on hardware asks the platform instead.
    """

    name = "unknown"

    def is_cuda(self) -> bool:
        """NVIDIA GPUs, through CUDA."""
        return False

    def is_rocm(self) -> bool:
        """AMD GPUs, through ROCm; torch still calls them ``"cuda"`` devices."""
        return False

    def is_xpu(self) -> bool:
        """Intel GPUs, through XPU."""
        return False

    def is_cpu(self) -> bool:
        """No accelerator in this process."""
        return False

    def arch(self, device=None) -> str | None:
        """The GPU's architecture, e.g. ``sm_90`` or ``gfx942``; None if unknown."""
        return None

    def cuda_capability(self, device=None) -> tuple[int, int] | None:
        """NVIDIA compute capability; None on any other platform."""
        return None

    def has_cuda_capability(self, major: int, minor: int = 0, device=None) -> bool:
        """An NVIDIA GPU of at least ``major.minor``; never true off NVIDIA."""
        capability = self.cuda_capability(device)
        return capability is not None and capability >= (major, minor)

    def inductor_overrides(self, num_tokens: int | None = None) -> dict[str, object]:
        """``torch._inductor.config`` values to patch in while the engine
        compiles a forward for ``num_tokens`` tokens, a capture bucket's count,
        or ``None`` when the forward serves any count.

        Meant for inductor codegen settings on this stack (say, which GEMM
        backends to autotune at small sizes), not for picking kernels. Empty
        keeps inductor's defaults.
        """
        return {}

    def process_env_defaults(self) -> dict[str, str | Callable[[str | None], str]]:
        """Environment variables for settings a runtime or library reads once
        at init. Workers apply them with ``os.environ.setdefault`` before their
        first device call, so a value the user exported wins.

        A callable adds to the user's value instead: it gets the current value
        (None when unset) and returns the one to set. The conductor applies the
        defaults and each worker again, so it must return a value that already
        has its addition unchanged.
        """
        return {}

    def local_cpus(self, device_index: int) -> frozenset[int] | None:
        """The CPUs on the device's NUMA node, as the host lists them; None
        where the platform or the host doesn't say."""
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def _pci_local_cpus(device_index: int) -> frozenset[int] | None:
    """The device's ``local_cpulist`` in sysfs, found by the PCI address torch
    reports. ROCm builds report it the same way, so this needs no NVML."""
    try:
        props = torch.cuda.get_device_properties(device_index)
        bdf = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        with open(f"/sys/bus/pci/devices/{bdf}/local_cpulist") as f:
            return parse_cpu_list(f.read()) or None
    except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
        return None


class CudaPlatform(Platform):
    name = "cuda"

    def is_cuda(self) -> bool:
        return True

    def arch(self, device=None) -> str:
        major, minor = self.cuda_capability(device)
        return f"sm_{major}{minor}"

    def cuda_capability(self, device=None) -> tuple[int, int]:
        return torch.cuda.get_device_capability(device)

    def local_cpus(self, device_index: int) -> frozenset[int] | None:
        return _pci_local_cpus(device_index)

    def process_env_defaults(self) -> dict[str, str | Callable[[str | None], str]]:
        # NCCL registering the buffers captured in CUDA graphs gave wrong Cosmos3 SP4 images
        return {"NCCL_GRAPH_REGISTER": "0", **blackwell_ptxas_env()}


class RocmPlatform(Platform):
    name = "rocm"

    def is_rocm(self) -> bool:
        return True

    def arch(self, device=None) -> str:
        # "gfx942:sramecc+:xnack-": what follows the colon are target features
        return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]

    def inductor_overrides(self, num_tokens: int | None = None) -> dict[str, object]:
        # none until one is measured to help on AMD GPUs
        return {}

    def process_env_defaults(self) -> dict[str, str]:
        # none until one is measured to help on AMD GPUs
        return {}

    def local_cpus(self, device_index: int) -> frozenset[int] | None:
        return _pci_local_cpus(device_index)


class XpuPlatform(Platform):
    name = "xpu"

    def is_xpu(self) -> bool:
        return True


class CpuPlatform(Platform):
    name = "cpu"

    def is_cpu(self) -> bool:
        return True
