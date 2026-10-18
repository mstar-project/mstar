"""Fused tensor-parallel all-reduce + residual add + RMSNorm.

Replaces the all-reduce, add and norm after a row-parallel projection with one
kernel: FlashInfer's ``allreduce_fusion`` on NVIDIA (~4-6us vs ~11us unfused at
decode shapes), AITER's custom all-reduce with a fused RMSNorm on ROCm. Each is
an ``ARFusion`` strategy behind the ``mstar::allreduce_add_rmsnorm`` op, so
dynamo and CUDA-graph capture see one opaque node; the op reaches its strategy
through an int-keyed registry. A strategy also says which inputs it takes
(``accepts``); the rest keep ``CommGroup``'s all-reduce + separate norm.

Building a strategy is a collective over the TP group. Every choice of code
path is a function of shapes and world size, or is agreed across the group
before it is acted on, so all ranks run the same collectives.
"""
from __future__ import annotations

import importlib.metadata
import inspect
import logging
import re

import torch
import torch.distributed as dist

from mstar.platforms.dispatch import PlatformDispatch

logger = logging.getLogger(__name__)

# ws_id -> strategy
_FUSIONS: list[ARFusion] = []


class GraphCaptureScope:
    """One graph capture on a TP group, entered by every rank, pass or fail.

    The caller sets ``captured`` once its capture succeeded; after the scope
    exits, ``usable`` says whether to keep the graph. A strategy that finishes
    a capture across ranks (registering the buffers it recorded) does so on
    exit, so ``usable`` is the same on every rank.
    """

    def __init__(self):
        self.captured = False
        self.usable = False

    def __enter__(self) -> GraphCaptureScope:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.usable = exc_type is None and self.captured
        return False


class ARFusion:
    """A TP group's fused all-reduce + residual + RMSNorm kernel."""

    ws_id: int = -1

    def accepts(
        self, x: torch.Tensor, residual: torch.Tensor, weight_bias: float,
    ) -> bool:
        """Whether ``run`` serves these inputs. Depends on shapes, dtypes and
        layout only, so every rank of the group answers the same."""
        raise NotImplementedError

    def run(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        eps: float, weight_bias: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``r = allreduce(x) + residual``; returns ``(rmsnorm(r), r)``,
        scaling by ``weight + weight_bias``. Out of place."""
        raise NotImplementedError

    def graph_capture(self) -> GraphCaptureScope:
        return GraphCaptureScope()

    def close(self) -> None:
        pass


# --- NVIDIA: FlashInfer ------------------------------------------------------


def _all_ranks(ok: bool, cpu_group) -> bool:
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=cpu_group)
    return bool(flag.item())


class FlashInferARFusion(ARFusion):
    """FlashInfer's one-shot ``allreduce_fusion``, up to ``max_tokens`` rows."""

    # "auto" picks MNNVL whenever FlashInfer's capability vote passes, but mapping
    # its multicast buffer can still fail (CUDA_ERROR_INVALID_VALUE inside some
    # containers). The TRT-LLM backend uses IPC handles, not multicast, so it is
    # tried before giving up on fusion.
    BACKENDS = ("auto", "trtllm")

    def __init__(self, workspace, max_tokens: int):
        self._workspace = workspace
        self.max_tokens = max_tokens

    @classmethod
    def create(
        cls, rank: int, world_size: int, cpu_group, max_tokens: int,
        hidden: int, dtype: torch.dtype,
    ) -> FlashInferARFusion | None:
        try:
            import flashinfer.comm as fc
            from flashinfer.comm.comm_backend import TorchDistBackend
        except Exception:
            logger.exception("FlashInfer all-reduce fusion unavailable; "
                             "keeping NCCL all-reduce + separate norm")
            return None
        for backend in cls.BACKENDS:
            ws = None
            try:
                ws = fc.create_allreduce_fusion_workspace(
                    backend=backend, world_size=world_size, rank=rank,
                    max_token_num=max_tokens, hidden_dim=hidden, dtype=dtype,
                    comm_backend=TorchDistBackend(group=cpu_group), group=cpu_group,
                )
            except Exception:
                logger.warning("FlashInfer all-reduce fusion backend %r failed",
                               backend, exc_info=True)
            # Every rank must use the same backend, or the next collective
            # deadlocks: keep one only if it built on all of them.
            try:
                agreed = _all_ranks(ws is not None, cpu_group)
            except RuntimeError:
                # A rank failed between FlashInfer's own collectives, so this vote
                # met a different one; the gloo timeout closes the group on every
                # member, and every rank falls back here.
                logger.exception("TP ranks lost sync building the fusion workspace; "
                                 "keeping NCCL all-reduce + separate norm")
                if ws is not None:
                    ws.destroy()
                return None
            if agreed:
                logger.info("all-reduce fusion workspace: %s (backend %r), up to "
                            "%d tokens x %d", type(ws).__name__, backend,
                            max_tokens, hidden)
                return cls(ws, max_tokens)
            if ws is not None:
                ws.destroy()
        logger.warning("no FlashInfer all-reduce fusion backend works here; "
                       "keeping NCCL all-reduce + separate norm")
        return None

    def accepts(self, x, residual, weight_bias) -> bool:
        return (x.shape[0] <= self.max_tokens
                and x.is_contiguous() and residual.is_contiguous())

    def run(self, x, residual, weight, eps, weight_bias):
        import flashinfer.comm as fc

        norm_out = torch.empty_like(x)
        residual_out = torch.empty_like(residual)
        fc.allreduce_fusion(
            input=x, workspace=self._workspace,
            pattern=fc.AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=True, residual_in=residual, residual_out=residual_out,
            norm_out=norm_out, rms_gamma=weight, rms_eps=eps, fp32_acc=True,
            weight_bias=weight_bias,
            # Complete at the end: with an early trigger a PDL-launched consumer
            # can read the one-shot Lamport buffer before it is committed (see
            # flashinfer-ai/flashinfer#1223).
            trigger_completion_at_end=True,
        )
        return norm_out, residual_out


# --- ROCm: AITER --------------------------------------------------------------


def _agree(cpu_group, values: list[int]) -> list[int]:
    """Element-wise max of ``values`` over ``cpu_group`` (gloo, host only)."""
    t = torch.tensor(values, dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=cpu_group)
    return t.tolist()


def _aiter_version() -> str | None:
    try:
        return importlib.metadata.version("amd-aiter")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        from aiter._version import __version__
    except ImportError:
        return None
    return __version__


def _release(version: str) -> tuple[int, ...] | None:
    """``0.1.19.post1+rocm7.2`` -> ``(0, 1, 19, 1)``; ``0.1.25`` -> ``(0, 1, 25, 0)``."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?", version)
    if m is None:
        return None
    return tuple(int(g or 0) for g in m.groups())


def _round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


class AiterARFusion(ARFusion):
    """AITER's custom all-reduce with its one-stage fused RMSNorm.

    Runs only where vLLM measured the one-stage kernel ahead of RCCL + a
    separate norm (``use_1stage_fused_ar_rms``): at most 80 tokens, rows of
    16-byte packs with at most 1024 packs, and under 256 KiB on 4 GPUs or
    128 KiB on 8. Everything else keeps RCCL + norm; AITER's two-stage fused
    kernel is slower than that pair and is never called.

    Outside capture AITER copies the input into its pre-registered IPC pool. A
    capture inside ``graph_capture`` instead records the input's own address,
    and the scope's exit registers those addresses on every rank.
    """

    WORLD_SIZES = (2, 4, 8)
    MAX_TOKENS = 80
    # strict upper bound on the input's bytes, by world size
    MAX_BYTES = {4: 256 << 10, 8: 128 << 10}
    MAX_PACKS_PER_ROW = 1024
    # First release whose one-stage kernel waits for every peer to finish
    # reading before it returns, so a fast rank cannot overwrite its input pool
    # with the next call's input while a slow one still reads it. No signature
    # changed with it, so only the version tells.
    MIN_VERSION = (0, 1, 19, 1)

    def __init__(self, ca, cpu_group, hidden: int, dtype: torch.dtype,
                 max_tokens: int, register_in_capture: bool):
        self._ca = ca
        self._cpu_group = cpu_group
        self.hidden = hidden
        self.dtype = dtype
        self.max_tokens = max_tokens
        self._register_in_capture = register_in_capture
        self._capturing = False

    @classmethod
    def fused_token_bound(
        cls, world_size: int, hidden: int, dtype: torch.dtype, max_tokens: int,
    ) -> int:
        """Most rows the one-stage kernel takes; 0 if it takes none."""
        if world_size not in cls.WORLD_SIZES or dtype not in (torch.bfloat16, torch.float16):
            return 0
        elt = torch.empty((), dtype=dtype).element_size()
        pack = 16 // elt
        if hidden % pack or hidden // pack > cls.MAX_PACKS_PER_ROW:
            return 0
        tokens = min(cls.MAX_TOKENS, max_tokens)
        cap = cls.MAX_BYTES.get(world_size)
        if cap is not None:
            tokens = min(tokens, (cap - 1) // (hidden * elt))
        return max(tokens, 0)

    @classmethod
    def create(
        cls, rank: int, world_size: int, cpu_group, max_tokens: int,
        hidden: int, dtype: torch.dtype,
    ) -> AiterARFusion | None:
        bound = cls.fused_token_bound(world_size, hidden, dtype, max_tokens)
        if bound == 0:
            # shape facts, the same on every rank
            logger.info("AITER fused all-reduce + RMSNorm does not cover tp=%d "
                        "hidden=%d %s; keeping RCCL + separate norm",
                        world_size, hidden, dtype)
            return None

        # Building the workspace is a sequence of collectives, so a rank that
        # would refuse must say so before any rank starts.
        pool_cls = None
        try:
            from aiter.dist.device_communicators.custom_all_reduce import (
                CustomAllreduce,
                IPCBufferPool,
            )
            pool_cls = IPCBufferPool
            reason = cls._unsupported(CustomAllreduce, IPCBufferPool)
        except Exception as e:  # noqa: BLE001
            reason = f"cannot import AITER's custom all-reduce: {type(e).__name__}: {e}"
        # AITER keys each IPC exchange in the default store by a process-wide
        # pool counter; this group's members start its pool from one count.
        seq = pool_cls._pool_seq if reason is None else 0
        refused, seq = _agree(cpu_group, [int(reason is not None), seq])
        if refused:
            logger.warning("AITER fused all-reduce + RMSNorm unavailable (%s); "
                           "keeping RCCL + separate norm",
                           reason or "refused on another rank")
            return None
        pool_cls._pool_seq = seq

        elt = torch.empty((), dtype=dtype).element_size()
        # The meta buffer is meta_size() + 2 * max_size: from 2 MiB up it is
        # IPC-exportable at any size, below that only at whole 4 KiB pages.
        max_size = max(1 << 20, _round_up(bound * hidden * elt, 1 << 20))
        device = torch.device("cuda", torch.cuda.current_device())
        ca = None
        try:
            ca = CustomAllreduce(cpu_group, device, max_size=max_size)
            built = not ca.disabled
        except Exception:
            logger.exception("AITER CustomAllreduce construction failed")
            built = False
        if _agree(cpu_group, [int(not built)])[0]:
            logger.warning("AITER custom all-reduce unavailable on some rank; "
                           "keeping RCCL + separate norm")
            if ca is not None:
                ca.close()
            return None

        bad = cls._probe(ca, rank, world_size, hidden, dtype, bound, device)
        # registering captured inputs is off under expandable segments
        register = bool(getattr(ca, "enable_register_for_capturing", False)
                        and hasattr(ca, "_ops_get_graph_buffer_count"))
        failed, no_register = _agree(cpu_group, [int(bad is not None), int(not register)])
        if failed:
            logger.error("AITER fused all-reduce + RMSNorm gave wrong results "
                         "(tp=%d hidden=%d %s): %s; keeping RCCL + separate norm",
                         world_size, hidden, dtype, bad or "on another rank")
            ca.close()
            return None

        logger.info("all-reduce fusion workspace: AITER CustomAllreduce, up to "
                    "%d tokens x %d, max_size %d MiB, captured inputs %s",
                    bound, hidden, max_size >> 20,
                    "copied into the pool" if no_register else "registered")
        return cls(ca, cpu_group, hidden, dtype, bound, not no_register)

    @classmethod
    def _unsupported(cls, custom_allreduce, pool_cls) -> str | None:
        version = _aiter_version()
        release = _release(version) if version else None
        if release is None:
            return f"unknown AITER version {version!r}"
        if release < cls.MIN_VERSION:
            return f"AITER {version} predates the one-stage kernel's exit barrier"
        params = inspect.signature(custom_allreduce.fused_ar_rms).parameters
        if "gemma_norm" not in params or "use_1stage" not in params:
            return f"AITER {version} fused_ar_rms takes no gemma_norm/use_1stage"
        if not isinstance(getattr(pool_cls, "_pool_seq", None), int):
            return f"AITER {version} does not key its IPC pools by _pool_seq"
        return None

    @staticmethod
    def _probe(ca, rank, world_size, hidden, dtype, max_tokens, device) -> str | None:
        """Run the kernel at the smallest and largest fused shapes, both norm
        flavours, against a host reference. A build that skips this hidden size
        leaves the NaN the outputs were filled with. The inputs come from one
        seeded generator, so every rank knows every rank's input."""
        gen = torch.Generator().manual_seed(1234)
        try:
            for tokens in sorted({1, max_tokens}):
                xs = torch.randn(world_size, tokens, hidden, generator=gen).to(dtype)
                res = torch.randn(tokens, hidden, generator=gen).to(dtype)
                w = (torch.randn(hidden, generator=gen) * 0.1).to(dtype)
                r_ref = (xs.float().sum(0).to(dtype).float() + res.float()).to(dtype)
                v = r_ref.float()
                v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
                for gemma in (False, True):
                    out_ref = (v * (w.float() + float(gemma))).to(dtype)
                    out = torch.full(
                        (tokens, hidden), float("nan"), dtype=dtype, device=device,
                    )
                    r = torch.full_like(out, float("nan"))
                    ca.fused_ar_rms(
                        xs[rank].to(device), res.to(device), res_out=r, out=out,
                        w=w.to(device), eps=1e-6, registered=False,
                        use_1stage=True, gemma_norm=gemma,
                    )
                    for name, got, want in (("residual", r, r_ref), ("norm", out, out_ref)):
                        got, want = got.float().cpu(), want.float()
                        err = (got - want).abs().nan_to_num(float("inf"))
                        if (err > 5e-3 + 2e-2 * want.abs()).any():
                            return (f"{name} at tokens={tokens} gemma={gemma}: "
                                    f"max |err| {err.max().item():.3g}")
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"
        return None

    def accepts(self, x, residual, weight_bias) -> bool:
        return (0 < x.shape[0] <= self.max_tokens
                and x.dim() == 2 and x.shape[1] == self.hidden
                and x.dtype == self.dtype and residual.dtype == self.dtype
                and residual.shape == x.shape
                and weight_bias in (0.0, 1.0)
                and x.is_contiguous() and residual.is_contiguous())

    def run(self, x, residual, weight, eps, weight_bias):
        # AITER reads the weight in the activation dtype
        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)
        registered = self._capturing and torch.cuda.is_current_stream_capturing()
        return self._ca.fused_ar_rms(
            x, residual, w=weight, eps=eps, registered=registered,
            use_1stage=True, gemma_norm=weight_bias == 1.0,
        )

    def graph_capture(self) -> GraphCaptureScope:
        if not self._register_in_capture:
            return GraphCaptureScope()
        return _RegisteredCapture(self)

    def _finish_capture(self, captured: bool) -> bool:
        """Register the input addresses this capture recorded, on every rank.

        Registration pairs each rank's i-th recorded address with every peer's
        i-th, so it needs the same count everywhere. If the counts differ the
        pairing is lost: the graph is dropped on every rank, and later
        captures copy into the pre-registered pool instead.
        """
        count = self._ca._ops_get_graph_buffer_count(self._ca._ptr)
        hi, neg_lo, failed = _agree(
            self._cpu_group, [count, -count, int(not captured)],
        )
        if hi != -neg_lo:
            logger.error("graph capture recorded %d to %d all-reduce inputs across "
                         "ranks; dropping it, and copying into AITER's pool in "
                         "later captures", -neg_lo, hi)
            self._register_in_capture = False
            return False
        if hi:
            self._ca.register_graph_buffers()
        return not failed

    def close(self) -> None:
        self._ca.close()


class _RegisteredCapture(GraphCaptureScope):
    def __init__(self, fusion: AiterARFusion):
        super().__init__()
        self._fusion = fusion

    def __enter__(self) -> _RegisteredCapture:
        self._fusion._capturing = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._fusion._capturing = False
        if exc_type is not None:
            # unwinding; peers may never reach the agreement
            self.usable = False
            return False
        self.usable = self._fusion._finish_capture(self.captured)
        return False


_create = PlatformDispatch(
    "mstar::allreduce_add_rmsnorm workspace",
    rocm=lambda: AiterARFusion.create,
    default=lambda: FlashInferARFusion.create,
)


def create_fusion(
    rank: int, world_size: int, cpu_group, max_tokens: int, hidden: int,
    dtype: torch.dtype,
) -> ARFusion | None:
    """Collective over the TP group: every member calls it, in the same order.
    Returns the strategy behind ``allreduce_add_rmsnorm``, or None if none can
    be built here (the caller keeps the unfused path)."""
    fusion = _create(rank, world_size, cpu_group, max_tokens, hidden, dtype)
    if fusion is not None:
        fusion.ws_id = len(_FUSIONS)
        _FUSIONS.append(fusion)
    return fusion


@torch.library.custom_op("mstar::allreduce_add_rmsnorm", mutates_args=())
def allreduce_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    eps: float, weight_bias: float, ws_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``r = allreduce(x) + residual``; returns ``(rmsnorm(r), r)``, with the
    norm scaling by ``weight + weight_bias``."""
    return _FUSIONS[ws_id].run(x, residual, weight, eps, weight_bias)


@allreduce_add_rmsnorm.register_fake
def _(x, residual, weight, eps, weight_bias, ws_id):
    return torch.empty_like(x), torch.empty_like(residual)
