"""Fused TP all-reduce + residual add + RMSNorm.

One kernel instead of all-reduce, add and norm after a row-parallel linear.
FlashInfer's ``allreduce_fusion`` on NVIDIA (~4-6us vs ~11us unfused at decode
sizes), AITER's custom all-reduce with fused RMSNorm on ROCm. Each is an
``ARFusion`` behind the ``mstar::allreduce_add_rmsnorm`` op, so dynamo and graph
capture see one opaque node. Inputs a strategy doesn't accept go through
``CommGroup``'s plain all-reduce + norm.

Setting one up is collective over the TP group. Anything that picks a code path
either depends only on sizes and world size or is agreed across ranks first.
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
    """Wraps one graph capture. Every rank enters it, whether or not its
    capture works. The caller sets ``captured`` when the capture succeeded and
    reads ``usable`` after exit to decide whether to keep the graph. A strategy
    that registers captured buffers across ranks does it on exit, so ``usable``
    agrees on every rank."""

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
        """Whether ``run`` takes these inputs. Only looks at sizes, dtypes and
        layout, so all ranks give the same answer."""
        raise NotImplementedError

    def run(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        eps: float, weight_bias: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(rmsnorm(r), r)`` with ``r = allreduce(x) + residual``,
        scaled by ``weight + weight_bias``. Out of place."""
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
    """Element-wise max of ``values`` across ``cpu_group`` (gloo, on the host)."""
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
    """``0.1.19.post1+rocm7.2`` -> ``(0, 1, 19, 1)``, ``0.1.25`` -> ``(0, 1, 25, 0)``."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?", version)
    if m is None:
        return None
    return tuple(int(g or 0) for g in m.groups())


def _round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def _reference(xs: torch.Tensor, res: torch.Tensor, w: torch.Tensor,
               weight_bias: float, eps: float):
    """Expected ``(norm, residual)`` for the stacked per-rank inputs ``xs``,
    plus a tolerance for each. Allows about one ulp per rounding step, enough
    for a build that truncates (the harness measures those too)."""
    dtype = xs.dtype
    a = xs.float().sum(0)
    r = (a.to(dtype).float() + res.float()).to(dtype)
    v = r.float()
    scale = torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    w_eff = w.float() + weight_bias
    normed = (v * scale * w_eff).to(dtype)
    step = a.abs() + v.abs()
    rel = 2.0 ** -6  # ~2 bf16 ulps
    return (normed, r), (rel * (normed.float().abs() + scale * w_eff.abs() * step) + 1e-6,
                         rel * step + 1e-6)


def _compare(xs, res, w, gemma, out, r, tokens) -> str | None:
    """What's wrong with one startup-check case, or None."""
    want, tol = _reference(xs, res, w, float(gemma), 1e-6)
    for name, got, ref, bound in zip(("norm", "residual"), (out, r), want, tol, strict=True):
        err = (got.float().cpu() - ref.float()).abs().nan_to_num(float("inf"))
        if (err > bound).any():
            return f"{name} at tokens={tokens} gemma={gemma}: max |err| {err.max().item():.3g}"
    # AITER's release builds truncate to bf16 on gfx942, which shrinks the
    # residual stream a bit more every layer (~15% over 64 layers).
    got_r, ref_r = r.float().cpu().abs(), want[1].float().abs()
    toward_zero = int((got_r < ref_r).sum()) - int((got_r > ref_r).sum())
    if toward_zero > 0.05 * ref_r.numel():
        return (f"residual rounds toward zero ({toward_zero} of {ref_r.numel()} values), "
                f"needs an AITER build with OPUS_FP32_to_BF16_DEFAULT=0")
    return None


class AiterARFusion(ARFusion):
    """AITER's custom all-reduce with its one-stage fused RMSNorm.

    Used only where vLLM found the one-stage kernel faster than RCCL + norm
    (``use_1stage_fused_ar_rms``). That's up to 80 tokens, at most 1024 16-byte
    packs per row, and under 256 KiB on 4 GPUs or 128 KiB on 8. Everything else
    stays on RCCL + norm. The two-stage fused kernel is slower than that and
    is never used.

    Builds that truncate to bf16 (the release wheels on gfx942) fail the startup
    check, since they bias the residual toward zero. A build with
    ``OPUS_FP32_to_BF16_DEFAULT=0`` rounds to nearest and passes.

    Outside capture AITER copies the input into its pre-registered IPC pool.
    Inside ``graph_capture`` it records the input's own address instead, and
    the scope registers those addresses on all ranks when it exits.
    """

    WORLD_SIZES = (2, 4, 8)
    MAX_TOKENS = 80
    # input bytes must stay below this, per world size
    MAX_BYTES = {4: 256 << 10, 8: 128 << 10}
    MAX_PACKS_PER_ROW = 1024
    # Before this the one-stage kernel returned without waiting for peers to
    # finish reading, so a fast rank could overwrite its input pool with the
    # next call's data. No API changed with the fix, so we check the version.
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
        """Max rows the one-stage kernel takes here, 0 for none."""
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
            # depends only on sizes, same on every rank
            logger.info("AITER fused all-reduce + RMSNorm doesn't cover tp=%d "
                        "hidden=%d %s, using RCCL + separate norm",
                        world_size, hidden, dtype)
            return None

        # Building the workspace runs collectives, so every rank has to agree
        # to go ahead before any of them starts.
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
        # AITER names its store keys after a per-process pool counter. Line it
        # up across the group so all members use the same keys.
        seq = pool_cls._pool_seq if reason is None else 0
        refused, seq = _agree(cpu_group, [int(reason is not None), seq])
        if refused:
            logger.warning("AITER fused all-reduce + RMSNorm unavailable (%s), "
                           "using RCCL + separate norm",
                           reason or "refused on another rank")
            return None
        pool_cls._pool_seq = seq

        elt = torch.empty((), dtype=dtype).element_size()
        # AITER's meta buffer is meta_size() + 2 * max_size. Below 2 MiB it
        # only exports over IPC at whole 4 KiB pages, so stay above that.
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
            logger.warning("AITER custom all-reduce failed to build on some rank, "
                           "using RCCL + separate norm")
            if ca is not None:
                ca.close()
            return None

        bad = cls._check_kernel(ca, rank, world_size, hidden, dtype, bound, device)
        # AITER turns off registering captured inputs under expandable segments
        register = bool(getattr(ca, "enable_register_for_capturing", False)
                        and hasattr(ca, "_ops_get_graph_buffer_count"))
        failed, no_register = _agree(cpu_group, [int(bad is not None), int(not register)])
        if failed:
            logger.error("AITER fused all-reduce + RMSNorm failed its startup check "
                         "(tp=%d hidden=%d %s, %s), using RCCL + separate norm",
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
    def _check_kernel(ca, rank, world_size, hidden, dtype, max_tokens, device) -> str | None:
        """Startup check. Runs the kernel at 1 and ``max_tokens`` rows, gemma
        on and off, and compares with a host reference. Outputs start as NaN,
        so a build that skips this hidden size fails. All ranks draw inputs from
        the same seeded generator, so each one can compute the full sum.

        Every rank runs every case even after one fails: each call is a
        collective, and a rank that stops early leaves its peers spinning in
        the kernel."""
        gen = torch.Generator().manual_seed(1234)
        bad = None
        try:
            for tokens in sorted({1, max_tokens}):
                xs = torch.randn(world_size, tokens, hidden, generator=gen).to(dtype)
                res = torch.randn(tokens, hidden, generator=gen).to(dtype)
                w = (torch.randn(hidden, generator=gen) * 0.1).to(dtype)
                for gemma in (False, True):
                    out = torch.full(
                        (tokens, hidden), float("nan"), dtype=dtype, device=device,
                    )
                    r = torch.full_like(out, float("nan"))
                    ca.fused_ar_rms(
                        xs[rank].to(device), res.to(device), res_out=r, out=out,
                        w=w.to(device), eps=1e-6, registered=False,
                        use_1stage=True, gemma_norm=gemma,
                    )
                    if bad is None:
                        bad = _compare(xs, res, w, gemma, out, r, tokens)
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"
        if bad is not None:
            logger.warning("AITER fused all-reduce + RMSNorm startup check failed on "
                           "this rank (rank %d): %s", rank, bad)
        return bad

    def accepts(self, x, residual, weight_bias) -> bool:
        return (0 < x.shape[0] <= self.max_tokens
                and x.dim() == 2 and x.shape[1] == self.hidden
                and x.dtype == self.dtype and residual.dtype == self.dtype
                and residual.shape == x.shape
                and weight_bias in (0.0, 1.0)
                and x.is_contiguous() and residual.is_contiguous())

    def run(self, x, residual, weight, eps, weight_bias):
        # AITER reads the weight as the activation dtype
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
        """Registers the input addresses this capture recorded, on all ranks.

        AITER pairs each rank's i-th address with every peer's i-th, so the
        counts have to match. If they don't, the graph is dropped everywhere
        and later captures fall back to copying into the pool.
        """
        count = self._ca._ops_get_graph_buffer_count(self._ca._ptr)
        hi, neg_lo, failed = _agree(
            self._cpu_group, [count, -count, int(not captured)],
        )
        if hi != -neg_lo:
            logger.error("graph capture recorded %d to %d all-reduce inputs across "
                         "ranks, dropping it and copying into AITER's pool from now on",
                         -neg_lo, hi)
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
            # unwinding, peers may never get to the agreement
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
    """Collective over the TP group, every member calls it in the same order.
    Returns the strategy for ``allreduce_add_rmsnorm``, or None if there's none
    here and the caller should stay unfused."""
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
    """Returns ``(rmsnorm(r), r)`` with ``r = allreduce(x) + residual``,
    scaled by ``weight + weight_bias``."""
    return _FUSIONS[ws_id].run(x, residual, weight, eps, weight_bias)


@allreduce_add_rmsnorm.register_fake
def _(x, residual, weight, eps, weight_bias, ws_id):
    return torch.empty_like(x), torch.empty_like(residual)
