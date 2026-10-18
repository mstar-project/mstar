"""The hardware platforms M* runs on, and what each reports about itself."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from mstar.platforms.ptxas import blackwell_ptxas_env


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

    def visible_devices_env(
        self, devices: str | None, environ: Mapping[str, str],
    ) -> dict[str, str]:
        """Env vars that limit this process and its children to ``devices``
        (``mstar serve --gpus``). With None, check the current env and keep it,
        raising ValueError if it picks devices inconsistently.

        Runs before the first device call, so like process_env_defaults it is
        asked of the platform torch was built for.
        """
        return {} if devices is None else {"CUDA_VISIBLE_DEVICES": devices}

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class CudaPlatform(Platform):
    name = "cuda"

    def is_cuda(self) -> bool:
        return True

    def arch(self, device=None) -> str:
        major, minor = self.cuda_capability(device)
        return f"sm_{major}{minor}"

    def cuda_capability(self, device=None) -> tuple[int, int]:
        return torch.cuda.get_device_capability(device)

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

    # captured buckets up to this many tokens autotune GEMMs over ATen (hipBLASLt) only
    ATEN_GEMM_MAX_TOKENS = 32

    def inductor_overrides(self, num_tokens: int | None = None) -> dict[str, object]:
        # At decode sizes inductor's autotuner can't reliably pick between its
        # Triton GEMM templates and hipBLASLt on MI300X. Its timings (12-17 us)
        # sit above the real kernel time in a graph (6-13 us) and move up to 30%
        # between compiles, so different compiles pick different kernels. On
        # orpheus a bad pick cost 10% of the decode step at concurrency 4.
        # hipBLASLt alone was the same or better at 1, 4, 16 and 64.
        if num_tokens is not None and num_tokens <= self.ATEN_GEMM_MAX_TOKENS:
            return {"max_autotune_gemm_backends": "ATEN"}
        return {}

    def process_env_defaults(self) -> dict[str, str]:
        return {
            # MIOpen's split-K implicit-GEMM conv solver is not deterministic. Turning
            # it off costs about 1% and makes same-seed outputs repeat.
            "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS_NHWC": "0",
        }

    def visible_devices_env(
        self, devices: str | None, environ: Mapping[str, str],
    ) -> dict[str, str]:
        # HIP (and torch on ROCm) prefer HIP_VISIBLE_DEVICES over
        # CUDA_VISIBLE_DEVICES, so set both or a stale value wins.
        if devices is not None:
            return {"CUDA_VISIBLE_DEVICES": devices, "HIP_VISIBLE_DEVICES": devices}
        hip = environ.get("HIP_VISIBLE_DEVICES")
        cuda = environ.get("CUDA_VISIBLE_DEVICES")
        if hip is not None and cuda is not None and hip != cuda:
            raise ValueError(
                f"HIP_VISIBLE_DEVICES={hip!r} and CUDA_VISIBLE_DEVICES={cuda!r} "
                "disagree and HIP uses the first. Unset one or pass --gpus"
            )
        return {}


class XpuPlatform(Platform):
    name = "xpu"

    def is_xpu(self) -> bool:
        return True


class CpuPlatform(Platform):
    name = "cpu"

    def is_cpu(self) -> bool:
        return True
