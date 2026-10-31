"""The hardware platforms M* runs on, and what each reports about itself."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import torch

from mstar.platforms import host_memory
from mstar.platforms.facts import DeviceFacts, device_facts
from mstar.platforms.ptxas import blackwell_ptxas_env
from mstar.utils.cpu_list import parse_cpu_list

logger = logging.getLogger(__name__)


class WorkerColocation:
    """What a platform does so that workers sharing one GPU (``rank_devices``)
    run their kernels side by side, when the deployment asks with ``mps: true``.
    The conductor calls ``start`` before it spawns the workers, puts
    ``client_env`` into each worker's environment, and calls ``stop`` once the
    workers have exited.
    """

    client_env: dict[str, str] = {}

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class HardwareQueueColocation(WorkerColocation):
    """AMD GPUs: nothing to start. Each process submits to hardware queues of
    its own, which the GPU schedules without a daemon, so two workers on one
    device can run their kernels side by side. Whether a given pair gains from
    sharing is the deployment's A/B."""

    def start(self) -> None:
        logger.info("mps: true on ROCm: nothing to start, the GPU schedules the workers' queues itself")


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

    def device_facts(self, device=None) -> DeviceFacts:
        """The GPU's architecture, product, compute units, caches and peak
        bandwidth and throughput, for kernel heuristics and the kernel harness
        (see ``mstar.platforms.facts``); never for model code. Unknown fields
        are None."""
        return device_facts(self, device)

    def inductor_overrides(
        self, num_tokens: int | None = None, rows: int | None = None,
        decode_rows: bool = False,
    ) -> dict[str, object]:
        """``torch._inductor.config`` values to patch in while the engine
        compiles a forward for ``num_tokens`` tokens in ``rows`` rows, a
        capture bucket's counts, or ``None`` when the forward serves any count.
        ``decode_rows`` marks a bucket of several tokens a row that still
        compiles as decode (a speculative verify step).

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

    def knob_defaults(self) -> dict[str, str]:
        """This platform's defaults for engine knobs (``mstar.utils.knobs``), by
        ``knobs:`` key, as strings in the env vars' format. A deployment config's
        ``knobs:`` block and an exported variable both override them."""
        return {}

    def tp_async_sched_default(self) -> bool:
        """Whether nodes with tp_size > 1 take TP async scheduling when neither
        the deployment config's ``tp_async_sched`` nor ``MSTAR_TP_ASYNC_SCHED``
        sets it."""
        return False

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

    def colocated_workers(self) -> WorkerColocation:
        """How workers sharing a GPU get their kernels to overlap, for a
        deployment with ``mps: true`` (see ``WorkerColocation``). Raises
        ValueError where this platform has no way."""
        raise ValueError(f"mps: true needs an NVIDIA or AMD GPU, not {self!r}")

    def local_cpus(self, device_index: int) -> frozenset[int] | None:
        """The CPUs on the device's NUMA node, as the host lists them; None
        where the platform or the host doesn't say."""
        return None

    def host_register(self, ptr: int, nbytes: int) -> bool:
        """Page-lock ``nbytes`` of mapped host memory at ``ptr`` for this
        process's device copies, which then run directly and asynchronously
        instead of through the runtime's staging buffer. A range already
        registered counts. False if this platform cannot or the runtime
        refused; the memory stays pageable and copies still work. Releases the
        GIL while the runtime pins the pages.
        """
        return False

    def host_unregister(self, ptr: int) -> bool:
        """Undo ``host_register`` of the range that starts at ``ptr``. Must run
        before the memory is unmapped. False if it was not registered."""
        return False

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

    def process_env_defaults(self) -> dict[str, str | Callable[[str | None], str]]:
        # NCCL registering the buffers captured in CUDA graphs gave wrong Cosmos3 SP4 images
        return {"NCCL_GRAPH_REGISTER": "0", **blackwell_ptxas_env()}

    def colocated_workers(self) -> WorkerColocation:
        # a CUDA MPS control daemon of the conductor's own; processes on one
        # GPU time-slice without it
        from mstar.conductor.mps import MpsDaemon

        return MpsDaemon()

    def local_cpus(self, device_index: int) -> frozenset[int] | None:
        return _pci_local_cpus(device_index)

    def host_register(self, ptr: int, nbytes: int) -> bool:
        return host_memory.CUDA_HOST.register(ptr, nbytes)

    def host_unregister(self, ptr: int) -> bool:
        return host_memory.CUDA_HOST.unregister(ptr)


class RocmPlatform(Platform):
    name = "rocm"

    def is_rocm(self) -> bool:
        return True

    def arch(self, device=None) -> str:
        # "gfx942:sramecc+:xnack-": what follows the colon are target features
        return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]

    # captured buckets up to this many tokens autotune GEMMs over ATen (hipBLASLt) only
    ATEN_GEMM_MAX_TOKENS = 32
    # and one-token-per-row (decode) buckets up to this many rows
    ATEN_GEMM_MAX_DECODE_ROWS = 128

    def inductor_overrides(
        self, num_tokens: int | None = None, rows: int | None = None,
        decode_rows: bool = False,
    ) -> dict[str, object]:
        # At decode sizes inductor's autotuner can't reliably pick between its
        # Triton GEMM templates and hipBLASLt on MI300X. Its timings (12-17 us)
        # sit above the real kernel time in a graph (6-13 us) and move up to 30%
        # between compiles, so different compiles pick different kernels. On
        # orpheus a bad pick cost 10% of the decode step at concurrency 4.
        # hipBLASLt alone was the same or better at 1, 4, 16 and 64. Decode
        # buckets of 48-128 rows also lose with Triton: it fuses silu*mul into
        # the down projection's template, 40% slower than hipBLASLt there (9B
        # c=64: +7.7% tok/s on ATen). Packed prefill buckets keep the Triton
        # templates, which win there.
        if num_tokens is None:
            return {}
        decode = rows is not None and (rows == num_tokens or decode_rows)
        limit = self.ATEN_GEMM_MAX_DECODE_ROWS if decode else self.ATEN_GEMM_MAX_TOKENS
        # a verify bucket's draft GEMMs run at its rows, its target's at a few
        # times that, so its rows bound the shapes these choices are about
        size = rows if decode_rows else num_tokens
        if size > limit:
            return {}
        overrides: dict[str, object] = {"max_autotune_gemm_backends": "ATEN"}
        from mstar.platforms.rocm_gemm import (
            DECODE_GEMV_MAX_ROWS,
            DECODE_LINEAR_MAX_ROWS,
            SKINNY_MAX_ROWS,
            ChainedGraphPass,
            decode_linear_pass,
            skinny_gemm_passes,
        )

        post = None
        if size <= DECODE_GEMV_MAX_ROWS and (passes := skinny_gemm_passes()) is not None:
            # linears of up to 16 rows on AITER's skinny GEMM or the Triton
            # weight stream, as the device's tuned config picks per shape
            overrides["pre_grad_custom_pass"], post = passes
        if (
            decode and SKINNY_MAX_ROWS < num_tokens and size <= DECODE_LINEAR_MAX_ROWS
            and (tuned := decode_linear_pass()) is not None
        ):
            # decode linears still on hipBLASLt take its tuned solution, where the device has one
            post = tuned if post is None else ChainedGraphPass(post, tuned)
        if post is not None:
            overrides["post_grad_custom_post_pass"] = post
        return overrides

    def process_env_defaults(self) -> dict[str, str]:
        return {
            # MIOpen's split-K implicit-GEMM conv solvers are not deterministic. Turning
            # them off costs about 1% and makes same-seed outputs repeat. The backward
            # one runs transposed convs (LTX-2.5's vocoder upsamplers).
            "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS_NHWC": "0",
            "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_GTC_XDLOPS_NHWC": "0",
        }

    def knob_defaults(self) -> dict[str, str]:
        # Each launch next to a replayed decode graph costs about 5 us of GPU
        # time on MI300X. Building the rope tables and scattering the
        # sampler's rows inside the graph drops about ten a step (Qwen3.5
        # 2B/9B: +1-3% tok/s at c=1-4).
        # Step threads stay on the process mask: on MI325X the pin gains no
        # tok/s at c=1 or c=64, raises 0.8B c=64 TTFT (57.5 -> 62.2 ms) and
        # graph-replay launch p95 (0.34-0.47 -> 1.3-3.3 ms).
        #
        # A DiT's eager norms (compile_exact_ops) exist to match cuBLAS/cuDNN
        # reference output bit for bit. hipBLASLt picks GEMM kernels per M, so
        # that match does not hold here and the bar is the reference's own
        # cross-seed spread; the eager norms only cost fusion (one graph break
        # each, 137 on FLUX.2 klein 9B).
        return {
            "ingraph_decode_rope": "1", "sampler_ingraph_scatter": "1", "step_thread_core": "off",
            "dit_exact_ops": "0",
        }

    def tp_async_sched_default(self) -> bool:
        # Qwen3.5 9B TP2 on MI325X, greedy output unchanged: 255-267 -> 306-314
        # tok/s at c=1 and 6505-6662 -> 8065-8362 at c=64.
        return True

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

    def colocated_workers(self) -> WorkerColocation:
        return HardwareQueueColocation()

    def local_cpus(self, device_index: int) -> frozenset[int] | None:
        return _pci_local_cpus(device_index)

    def host_register(self, ptr: int, nbytes: int) -> bool:
        return host_memory.HIP_HOST.register(ptr, nbytes)

    def host_unregister(self, ptr: int) -> bool:
        return host_memory.HIP_HOST.unregister(ptr)


class XpuPlatform(Platform):
    name = "xpu"

    def is_xpu(self) -> bool:
        return True


class CpuPlatform(Platform):
    name = "cpu"

    def is_cpu(self) -> bool:
        return True
