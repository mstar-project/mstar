"""The GPUs the simulator can pretend to be.

Peak FLOP/s are dense (no 2:4 sparsity): NVIDIA's datasheets quote tensor-core
rates "with sparsity", so those are halved here. Memory sizes are nvidia-smi's
``memory.total`` (torch's ``total_memory`` is ~0.5% lower: the driver's
reserve). Efficiency factors are what well-tuned kernels reach in practice
(cuBLAS on a large GEMM, a stream copy on a large buffer); a profile DB measured
on the real GPU (:mod:`mstar_sim.profile`) replaces the guesswork op by op.

Launch overheads are *device-side* gaps between consecutive kernels, the time
the simulated stream loses per kernel when the host keeps ahead of it. NVIDIA's
"Getting Started with CUDA Graphs" (developer blog, 2019) measures a ~2.9 us
kernel at 3.8 us per kernel launched back to back on a stream and 3.4 us inside
a graph: ~1 us per eager launch, ~0.5 us per graph node. Host-side launch cost
(~5 us per eager PyTorch op) is not charged here; the simulated host spends its
own time. A graph replay costs ``graph_launch_s`` once, then each node its
kernel plus ``graph_node_s``. ``min_kernel_s`` is the fixed time of the smallest kernel (block
scheduling, the first DRAM round trip, the tail): nsys shows a PyTorch
elementwise kernel on a few elements at ~1.5-2.5 us on A100/H100.
"""

from __future__ import annotations

from dataclasses import dataclass

TB = 1e12
MiB = 2**20


@dataclass(frozen=True)
class GPU:
    name: str  # as torch.cuda.get_device_name() reports it
    capability: tuple[int, int]
    sm_count: int
    memory_bytes: int
    hbm_bw: float  # DRAM bytes/s
    host_link_bw: float  # host<->device bytes/s per direction (PCIe)
    # Dense peak FLOP/s (OP/s for int8) by math mode; None where the GPU has no such units.
    fp64: float
    fp32: float  # CUDA cores, no tensor cores
    tf32: float
    bf16: float  # bf16 and fp16 tensor cores, fp32 accumulate
    fp8: float | None
    int8: float | None
    l2_bytes: int
    smem_per_sm: int  # shared memory per SM, bytes (CUDA programming guide's table by capability)
    compute_eff: float  # fraction of peak a large GEMM reaches
    memory_eff: float  # fraction of hbm_bw a large copy reaches
    launch_s: float = 1.0e-6  # device gap per eager stream launch
    graph_node_s: float = 0.5e-6  # device gap per kernel node in a CUDA-graph replay
    min_kernel_s: float = 2.0e-6  # fixed time of the smallest kernel
    graph_launch_s: float = 4.0e-6  # cudaGraphLaunch until the first node starts
    copy_latency_s: float = 2.0e-6  # fixed time of a host<->device cudaMemcpyAsync

    def peak(self, mode: str) -> float:
        """Peak FLOP/s for a math mode ("fp32", "tf32", "bf16", "fp8", "int8", "fp64")."""
        value = getattr(self, mode)
        return value if value is not None else self.bf16  # e.g. fp8 on Ampere: upcast to bf16


_PCIE5 = 64e9  # PCIe Gen5 x16, per direction
_PCIE4 = 32e9  # PCIe Gen4 x16, per direction

GPUS: dict[str, GPU] = {
    gpu.name: gpu
    for gpu in [
        # H100 datasheet (nvidia.com/en-us/data-center/h100), SXM5 column: FP64 34 TF, FP32 67 TF,
        # TF32 989 TF*, BF16/FP16 1979 TF*, FP8 3958 TF*, INT8 3958 TOPS* (*sparse), 80 GB HBM3
        # at 3.35 TB/s; GH100 whitepaper: 132 SMs. cuBLAS bf16 8192^3 reaches ~700-750 TF.
        GPU(
            "NVIDIA H100 80GB HBM3", (9, 0), 132, 81559 * MiB, 3.35 * TB, _PCIE5,
            fp64=34e12, fp32=67e12, tf32=494.7e12, bf16=989.4e12, fp8=1978.9e12, int8=1978.9e12,
            l2_bytes=50 * MiB, smem_per_sm=233472, compute_eff=0.75, memory_eff=0.85,
        ),
        # H100 PCIe (original datasheet PCIe column): FP64 26 TF, FP32 51 TF, TF32 756 TF*,
        # BF16 1513 TF*, FP8 3026 TF*, INT8 3026 TOPS*, 80 GB HBM2e at 2.0 TB/s, 114 SMs, 350 W.
        GPU(
            "NVIDIA H100 PCIe", (9, 0), 114, 81559 * MiB, 2.0 * TB, _PCIE5,
            fp64=26e12, fp32=51e12, tf32=378e12, bf16=756e12, fp8=1513e12, int8=1513e12,
            l2_bytes=50 * MiB, smem_per_sm=233472, compute_eff=0.7, memory_eff=0.85,
        ),
        # H200 datasheet (nvidia.com/en-us/data-center/h200): same compute as H100 SXM,
        # 141 GB HBM3e at 4.8 TB/s.
        GPU(
            "NVIDIA H200", (9, 0), 132, 143771 * MiB, 4.8 * TB, _PCIE5,
            fp64=34e12, fp32=67e12, tf32=494.7e12, bf16=989.4e12, fp8=1978.9e12, int8=1978.9e12,
            l2_bytes=50 * MiB, smem_per_sm=233472, compute_eff=0.75, memory_eff=0.85,
        ),
        # HGX B200 datasheet (nvidia.com/en-us/data-center/hgx), 8-GPU totals / 8: FP8 72 PF*,
        # INT8 72 POPS*, FP16/BF16 36 PF*, TF32 18 PF*, FP32 600 TF, FP64 296 TF, 1440 GB HBM3e,
        # 64 TB/s. Blackwell: 148 SMs, sm_100. Early cuBLAS reaches ~1.5-1.6 PF bf16.
        GPU(
            "NVIDIA B200", (10, 0), 148, 183359 * MiB, 8.0 * TB, _PCIE5,
            fp64=37e12, fp32=75e12, tf32=1125e12, bf16=2250e12, fp8=4500e12, int8=4500e12,
            l2_bytes=126 * MiB, smem_per_sm=233472, compute_eff=0.7, memory_eff=0.8,
        ),
        # A100 datasheet (nvidia.com/en-us/data-center/a100), 80GB SXM: FP64 9.7 TF (19.5 TC),
        # FP32 19.5 TF, TF32 156 TF, BF16/FP16 312 TF, INT8 624 TOPS, 80 GB HBM2e at 2039 GB/s;
        # GA100 whitepaper: 108 SMs. No FP8.
        GPU(
            "NVIDIA A100-SXM4-80GB", (8, 0), 108, 81920 * MiB, 2.039 * TB, _PCIE4,
            fp64=19.5e12, fp32=19.5e12, tf32=156e12, bf16=312e12, fp8=None, int8=624e12,
            l2_bytes=40 * MiB, smem_per_sm=167936, compute_eff=0.8, memory_eff=0.87,
        ),
        # L40S datasheet: FP32 91.6 TF, TF32 366 TF*, BF16/FP16 733 TF*, FP8 1466 TF*,
        # INT8 1466 TOPS*, 48 GB GDDR6 at 864 GB/s; AD102 with 142 SMs. FP64 is 1/64 of FP32.
        # Clocks throttle under sustained tensor load, hence the low compute_eff.
        GPU(
            "NVIDIA L40S", (8, 9), 142, 46068 * MiB, 0.864 * TB, _PCIE4,
            fp64=1.43e12, fp32=91.6e12, tf32=183e12, bf16=362e12, fp8=733e12, int8=733e12,
            l2_bytes=96 * MiB, smem_per_sm=102400, compute_eff=0.6, memory_eff=0.85,
        ),
        # Ada Lovelace whitepaper, RTX 4090 column: 128 SMs, FP32 82.6 TF, FP16/BF16 tensor with
        # FP32 accumulate 165.2 TF (GeForce runs fp32-accumulate at half rate), FP8 (fp32 acc)
        # 330.3 TF, TF32 82.6 TF, INT8 660.6 TOPS, 24 GB GDDR6X at 1008 GB/s.
        GPU(
            "NVIDIA GeForce RTX 4090", (8, 9), 128, 24564 * MiB, 1.008 * TB, _PCIE4,
            fp64=1.29e12, fp32=82.6e12, tf32=82.6e12, bf16=165.2e12, fp8=330.3e12, int8=660.6e12,
            l2_bytes=72 * MiB, smem_per_sm=102400, compute_eff=0.85, memory_eff=0.9,
        ),
    ]
}

ALIASES: dict[str, str] = {
    "h100": "NVIDIA H100 80GB HBM3",
    "h100-sxm": "NVIDIA H100 80GB HBM3",
    "h100-pcie": "NVIDIA H100 PCIe",
    "h200": "NVIDIA H200",
    "b200": "NVIDIA B200",
    "a100": "NVIDIA A100-SXM4-80GB",
    "a100-80gb": "NVIDIA A100-SXM4-80GB",
    "l40s": "NVIDIA L40S",
    "4090": "NVIDIA GeForce RTX 4090",
    "rtx4090": "NVIDIA GeForce RTX 4090",
}


def lookup(name: str) -> GPU:
    """The spec for an alias ("h100") or a torch device name ("NVIDIA H100 80GB HBM3"), any case."""
    key = name.strip().lower()
    found = ALIASES.get(key.replace("_", "-").replace(" ", "-")) or next((n for n in GPUS if n.lower() == key), None)
    if found is None:
        raise ValueError(f"unknown GPU {name!r}; known: {sorted(ALIASES)} or {sorted(GPUS)}")
    return GPUS[found]
