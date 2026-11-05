"""The ragged resource's ``auto`` backend for layouts of equal-length segments: the kernel per GPU,
head geometry and segment length comes from the ``ragged_attention`` op's tables.

FlashInfer's ragged prefill reads its layout from device buffers, so one captured graph replays any
layout of its bucket. A dense kernel (torch SDPA over ``[B, H, S, D]`` views of the packed rows) bakes
``B`` and ``S`` into the capture, so it can only serve a resource that declares
``RaggedAttentionConfig.uniform_segments``: every step lays out ``B`` segments of one length, and a
capture bucket's replays repeat the layout it was captured with. Where a table has the resource's
heads, rows by segment length name the kernel (``flashinfer`` or ``cudnn``); where none does, the
resource is FlashInfer's, as for any other layout.
"""

import functools
import logging
from collections.abc import Callable

import torch
import torch.nn.functional as F

from mstar.engine.resources.attn.ragged.flashinfer import FlashInferRaggedManager
from mstar.engine.resources.attn.ragged.wrappers import FlashInferRaggedPrefillWrapper, RaggedPrefillWrapper
from mstar.engine.resources.attn.workspace import Workspace
from mstar.kernels import Choice, Impl, KernelOp, Target, Tolerance, register
from mstar.kernels.registry import describe_rows
from mstar.kernels.table import Rows
from mstar.platforms import current_platform

logger = logging.getLogger(__name__)


def _flashinfer_unavailable(target: Target) -> str | None:
    from mstar.engine.resources.attn.dense import _flashinfer_ragged_unavailable_reason

    return _flashinfer_ragged_unavailable_reason(target.device)


def _cudnn_unavailable(target: Target) -> str | None:
    if target.device.type != "cuda" or not current_platform.is_cuda():
        return "torch's cuDNN attention runs on NVIDIA GPUs only"
    if not current_platform.has_cuda_capability(9, 0, target.device):
        return "torch enables cuDNN attention from compute capability 9.0"
    if target.dtype not in (torch.bfloat16, torch.float16):
        return f"cuDNN attention takes fp16 / bf16, not {target.dtype}"
    if target.shape["H"] != target.shape["H_kv"]:
        return "the dense path attends each query head to its own key head"
    if target.shape["D"] % 8 or target.shape["D"] > 128:
        return f"head dim {target.shape['D']} is outside cuDNN's 8..128 in steps of 8"
    return None


# bf16 attention sits ~0.3% of max |ref| from fp32 at the image DiTs' sizes on every kernel measured
_TOLERANCE = Tolerance(rtol=1e-2)

RAGGED_ATTENTION = register(KernelOp(
    name="ragged_attention",
    shape=("H", "H_kv", "D"),
    bucket="S",
    impls=(
        Impl("flashinfer", check=_flashinfer_unavailable, tolerance=_TOLERANCE,
             doc="FlashInfer's ragged prefill; a config's 'backend' picks its kernel, none keeps the resource's"),
        Impl("cudnn", check=_cudnn_unavailable, tolerance=_TOLERANCE,
             doc="torch SDPA on cuDNN over [B, H, S, D] views of the packed rows"),
    ),
    doc="cacheless self-attention over B segments of S tokens (H query and H_kv key heads of D, one rank's)",
))


def table_rows(num_qo_heads: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype,
               device: torch.device) -> Rows | None:
    """The rows for one rank's head geometry on ``device``, or None (no table has it)."""
    entry = RAGGED_ATTENTION.lookup(dtype, {"H": num_qo_heads, "H_kv": num_kv_heads, "D": head_dim}, device)
    return None if entry is None else entry.rows


class _CudnnDense:
    """Self-attention over ``B`` packed segments of ``S`` tokens through torch SDPA on cuDNN. The
    packed rows are viewed as ``[B, H, S, D]``, which cuDNN reads by stride, and its output comes back
    in the packed order."""

    def __init__(self, sm_scale: float):
        self.sm_scale = sm_scale
        self.layout: tuple[int, int, bool] | None = None

    def run(self, q, k, v, return_lse=False):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        bsz, seq, causal = self.layout
        rows, heads, dim = q.shape
        if rows != bsz * seq:
            raise ValueError(f"cuDNN ragged attention planned {bsz} x {seq} rows, got {rows}")
        qb, kb, vb = (t.reshape(bsz, seq, t.shape[1], dim).transpose(1, 2) for t in (q, k, v))
        if return_lse:
            out, lse = torch.ops.aten._scaled_dot_product_cudnn_attention(
                qb, kb, vb, None, True, 0.0, causal, False, scale=self.sm_scale,
            )[:2]
            lse = lse.reshape(bsz, heads, seq).transpose(1, 2).reshape(rows, heads)
            return out.transpose(1, 2).reshape(rows, heads, dim), lse
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            out = F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal, scale=self.sm_scale)
        return out.transpose(1, 2).reshape(rows, heads, dim)


@functools.cache
def _warn_uneven(label: str) -> None:
    logger.warning(
        "ragged attention %s declares uniform segments but planned segments of different lengths; "
        "running FlashInfer for them", label,
    )


class SelectedRaggedPrefillWrapper(RaggedPrefillWrapper):
    """Self-attention through the table's kernel for each plan's segment length, FlashInfer's
    ragged prefill being the fallback for everything a row doesn't send to cuDNN.

    A captured wrapper keeps the kernel of its first plan, the one its graph was captured with: the
    graph holds that kernel's launches. On cuDNN that also fixes the layout, so a later plan of
    another layout raises rather than replay a graph that attends the wrong rows."""

    def __init__(self, workspace: Callable[[], Workspace], num_qo_heads: int, num_kv_heads: int, head_dim: int, *,
                 rows: Rows, backend: str = "auto", **kwargs):
        super().__init__(num_qo_heads, num_kv_heads, head_dim, **kwargs)
        self._rows = rows
        # taken on the first FlashInfer plan: a wrapper that only runs cuDNN holds no workspace
        self._workspace = workspace
        self._fi_args = dict(num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
                             backend=backend, **kwargs)
        self._flashinfer: dict[str, FlashInferRaggedPrefillWrapper] = {}
        self._dense = _CudnnDense(self.sm_scale)
        self._active: _CudnnDense | FlashInferRaggedPrefillWrapper | None = None
        self._captured: Choice | None = None

    @property
    def impl(self) -> str | None:
        """The kernel the last plan picked: ``cudnn``, or ``flashinfer:<backend>``."""
        if self._active is None:
            return None
        if self._active is self._dense:
            return "cudnn"
        return next(f"flashinfer:{b}" for b, w in self._flashinfer.items() if w is self._active)

    def _flashinfer_for(self, choice: Choice) -> FlashInferRaggedPrefillWrapper:
        config = dict(choice.config or {})
        backend = config.get("backend", self._fi_args["backend"])
        wrapper = self._flashinfer.get(backend)
        if wrapper is None:
            wrapper = self._flashinfer[backend] = FlashInferRaggedPrefillWrapper(
                self._workspace(), **{**self._fi_args, "backend": backend},
            )
        return wrapper

    def _plan(self, cu_seqlens: torch.Tensor, kv_cu_seqlens: torch.Tensor, causal: bool) -> None:
        # a host list: the manager plans from pinned CPU offsets, and a handful of rows is cheaper in Python
        cu = cu_seqlens.tolist()
        lens = [b - a for a, b in zip(cu, cu[1:], strict=False)]
        n = len(lens)
        seq = max(lens, default=0)
        even = n > 0 and all(x == seq for x in lens)
        choice = self._captured or RAGGED_ATTENTION.row_for(self._rows, seq)
        if choice.impl == "cudnn" and not even:
            _warn_uneven(type(self).__name__)
            if self._captured is not None:
                raise ValueError(
                    f"{type(self).__name__}: captured on cuDNN for one layout, planned segments of "
                    f"lengths {lens}"
                )
            choice = Choice("flashinfer")
        self._num_segments = n
        if choice.impl == "cudnn":
            layout = (n, seq, causal)
            if self._captured is not None and self._dense.layout != layout:
                raise ValueError(
                    f"{type(self).__name__}: captured on cuDNN for {self._dense.layout[0]} segments of "
                    f"{self._dense.layout[1]} tokens, planned {n} of {seq}"
                )
            self._dense.layout = layout
            self._active = self._dense
        else:
            self._active = self._flashinfer_for(choice)
            self._active.plan(cu_seqlens, causal)
        if self.accelerator_graph:
            self._captured = choice

    def forget_plan(self) -> None:
        for wrapper in self._flashinfer.values():
            wrapper.forget_plan()

    def _plan_kernel(self, cu, kv_cu, total_tokens, causal) -> None:
        raise AssertionError("planned by the selected kernel's own wrapper")

    @torch.compiler.disable
    def run(self, q, k, v, return_lse=False):
        return self._active.run(q, k, v, return_lse=return_lse)


class SelectedRaggedManager(FlashInferRaggedManager):
    """FlashInfer's ragged self-attention manager whose wrappers take each plan's kernel from the
    table's rows."""

    wrapper_class = SelectedRaggedPrefillWrapper

    def __init__(self, device, dtype, config, key, rows: Rows):
        super().__init__(device=device, dtype=dtype, config=config, key=key)
        self._kwargs["rows"] = rows

    def _build_wrapper(self, label: str, slot: int, **graph_kwargs) -> SelectedRaggedPrefillWrapper:
        workspace = functools.partial(self._workspaces.get, label, slot)
        return self.wrapper_class(workspace, **self._kwargs, **graph_kwargs)


def build_auto(device: torch.device, dtype: torch.dtype, config, key: str):
    """The ``auto`` backend's manager for one rank's (sharded) config: the table's choice where the
    layout allows a dense kernel and a table has these heads on ``device``, else FlashInfer's."""
    rows = None
    if config.uniform_segments and config.flashinfer_backend == "auto":
        rows = table_rows(config.num_qo_heads, config.num_kv_heads, config.head_dim, dtype, device)
    if rows is None:
        return FlashInferRaggedManager(device=device, dtype=dtype, config=config, key=key)
    logger.info("ragged attention %s (%d/%d heads of %d): %s", key, config.num_qo_heads,
                config.num_kv_heads, config.head_dim, describe_rows(rows))
    return SelectedRaggedManager(device=device, dtype=dtype, config=config, key=key, rows=rows)
