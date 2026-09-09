"""GLM-5.3-Flash MLP/MoE: the glm52 blocks parameterized + the SwiGLU clamp.

Two deltas vs GLM-5.2, both from the reference (assembly spec section 2):

1. **SwiGLU clamp everywhere** — ``gate.clamp(max=swiglu_limit)`` (no min),
   ``up.clamp(-limit, limit)``, then ``silu(gate) * up`` — dense MLP,
   shared expert, and routed experts alike (``swiglu_limit = 10.0``).
   GLM-5.2 clamps nothing, so every dispatch path is forked here; the
   shared triton ``act_and_mul_kernel`` now takes an OPT-IN ``swiglu_limit``
   so the clamp threads into the fused ``fused_experts_fp8`` path too, which
   ``process_weights_after_loading`` selects for fp8 experts on CUDA.
2. **Router topk normalization carries ``+ 1e-20``** in the denominator
   (HF ``norm_topk_prob`` parity; glm52's gate omits the eps).
3. **Combine weights stay fp32 through the per-expert multiply** (HF
   ``Glm5NextTextExperts`` promotes the bf16 expert output by the fp32
   router weight and downcasts once at ``index_add_``; glm52 downcasts
   the weights before dispatch — a ~1-ulp deviation this port does not
   inherit).

Everything else — the noaux_tc sigmoid router shape (bias-added scores
select, raw sigmoid scores combine, groupless n_group=1), fp8-resident
expert containers, stacked per-expert loaders, TP slicing — is GLM-5.2's
``Glm52SparseMoeBlock`` carried in here verbatim (same [2048, 4096] expert
geometry, same [128, 128] block divisibility; 288 experts instead of 256 is
just a config number), so the package stands on its own on main.

Two dispatch paths, resolved in ``process_weights_after_loading``:

* ``_dispatch_clamped`` -- the clamped reference per-expert loop for bf16 AND
  fp8-resident experts (dequantize only the experts a batch hit). Like glm52's
  reference dispatch it hosts ``.nonzero()`` plus a per-hit-expert
  ``torch.where`` -- ~(1 + top_k) host syncs per MoE layer per decode step --
  so it is NOT capture-safe. It is the HF-faithful fallback and the default
  (``moe_quant_kernel="reference"``, the M1 eager parity anchor).
* ``_dispatch_fused`` -- the clamped fp8 fused kernel
  (``fused_experts_fp8(..., swiglu_limit=...)``, the opt-in clamp now in the
  shared ``act_and_mul_kernel``). No host sync, so cuda graphs capture; it is
  the fast production path, selected by ``moe_quant_kernel="auto"/"triton"`` on
  CUDA. NOT bit-exact vs the reference (fp8 GEMM vs the bf16 reference GEMM,
  the same gap glm52 accepts); the clamp is what makes it CORRECT.

Enabling the fused path is the ~13x M3 lever; schedule it TOGETHER with the
mHC Sinkhorn fusion (``mhc.py``) -- capture is the cheap mitigation for the
Sinkhorn launch storm, and this dispatch was what blocked capture.
"""
from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.distributed import ParallelGatedMLP
from mstar.model.components.moe import _down_proj_weight_loader, _gate_up_weight_loader
from mstar.model.glm5_next.config import Glm5NextModelConfig
from mstar.model.glm5_next.quantization import FP8_DTYPE, dequantize_fp8_block_weight

# HF Glm5NextTextTopkRouter: denominator = topk_weights.sum(...) + 1e-20.
_TOPK_NORM_EPS = 1e-20


def _fused_supports_swiglu_clamp(fused_fn) -> bool:
    """True iff the loaded fused fp8 kernel accepts ``swiglu_limit`` -- the
    SwiGLU clamp GLM-5.3 needs. Guards against an older ``fused_experts_fp8``
    (no clamp arg) silently serving the unclamped activations the reference
    forbids."""
    return "swiglu_limit" in inspect.signature(fused_fn).parameters


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _gate_up_fp8_loader(
    tp_rank: int, tp_size: int, full_inter: int, row_unit: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: str | int | None = None,
):
    """Route one expert's gate/up tensor into the stacked per-rank param.

    ``row_unit`` is 1 for the fp8 bytes and block_size[0] for the scale rows;
    the same slicing logic covers both because scales tile the row axis.
    Shape-driven: a full checkpoint tensor is TP-sliced here; a pre-sliced
    shard (the ``slice_spec`` fast read path — each rank reads only its
    bytes) is written as-is.
    """
    assert loaded_shard_id is not None
    kind, expert_str = str(loaded_shard_id).split(":")
    expert_idx = int(expert_str)
    rows = divide(divide(full_inter, tp_size), row_unit)
    full_rows = divide(full_inter, row_unit)
    if loaded_weight.dtype == FP8_DTYPE:
        loaded_weight = loaded_weight.view(torch.uint8)
    if loaded_weight.shape[0] == full_rows:
        start = tp_rank * rows
        loaded_weight = loaded_weight[start:start + rows, :]
    elif loaded_weight.shape[0] != rows:
        raise ValueError(
            f"expert gate/up tensor has {loaded_weight.shape[0]} rows; expected "
            f"the full {full_rows} or the per-rank {rows}"
        )
    if kind == "gate":
        param.data[expert_idx, :rows, :] = loaded_weight
    else:
        param.data[expert_idx, rows:2 * rows, :] = loaded_weight


def _down_fp8_loader(
    tp_rank: int, tp_size: int, full_inter: int, col_unit: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: str | int | None = None,
):
    """Column (contraction-dim) sharding twin of :func:`_gate_up_fp8_loader`."""
    assert loaded_shard_id is not None
    expert_idx = int(str(loaded_shard_id).split(":")[1])
    cols = divide(divide(full_inter, tp_size), col_unit)
    full_cols = divide(full_inter, col_unit)
    if loaded_weight.dtype == FP8_DTYPE:
        loaded_weight = loaded_weight.view(torch.uint8)
    if loaded_weight.shape[1] == full_cols:
        start = tp_rank * cols
        loaded_weight = loaded_weight[:, start:start + cols]
    elif loaded_weight.shape[1] != cols:
        raise ValueError(
            f"expert down tensor has {loaded_weight.shape[1]} cols; expected "
            f"the full {full_cols} or the per-rank {cols}"
        )
    param.data[expert_idx, :, :] = loaded_weight


class Glm5NextGatedMLP(ParallelGatedMLP):
    """``ParallelGatedMLP`` with the GLM-5.3 SwiGLU clamp.

    Used for the three dense layers and every shared expert; activation
    stays config ``hidden_act`` (silu). The clamp bounds pre-activation
    magnitudes, not the output — order is clamp, then silu, then product.
    """

    def __init__(self, *args, swiglu_limit: float = 10.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.split(self.intermediate_size_per_partition, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act(gate) * up)


class Glm5NextMoEGate(nn.Module):
    """DeepSeek-V3 noaux_tc sigmoid router, groupless (n_group=1), with the
    ``+ 1e-20`` topk-norm eps of HF's ``Glm5NextTextTopkRouter``.

    Bias-added scores drive expert *selection*; raw sigmoid scores drive the
    combine weights. Everything runs in fp32 (``moe_router_dtype``): sigmoid
    scores, the fp32 selection bias (checkpoint convention;
    ``restore_fp32_params`` re-widens it), and the normalization — only the
    returned combine weights are downcast by the caller.
    """

    def __init__(
        self,
        hidden_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        routed_scaling_factor: float,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.top_k = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob

        self.weight = nn.Parameter(torch.zeros(n_routed_experts, hidden_size))
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(n_routed_experts, dtype=torch.float32)
        )
        # fp32 copy of ``weight`` built once by ``finalize_weights`` (called
        # from the MoE block's process_weights_after_loading): the forward
        # would otherwise cast the (E, hidden) bf16 router to fp32 on every
        # layer of every step. Plain attribute, not a buffer, so
        # model.to(bf16) cannot downcast it. Bit-identical: the same fp32
        # values, computed once.
        self._weight_fp32: torch.Tensor | None = None

    def finalize_weights(self) -> None:
        """Cache the fp32 router weight; call after the weights are loaded."""
        self._weight_fp32 = self.weight.detach().float()

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = hidden_states.reshape(-1, self.hidden_size).float()
        w = self._weight_fp32
        if w is None or w.device != self.weight.device:
            w = self.weight.float()
        scores = F.linear(h, w).sigmoid()  # (T, E)

        biased = scores + self.e_score_correction_bias.unsqueeze(0)
        topk_ids = torch.topk(biased, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + _TOPK_NORM_EPS
            topk_weights = topk_weights / denominator
        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights, topk_ids


class Glm5NextSparseMoeBlock(nn.Module):
    """288 routed + 1 shared expert with the clamp on every path.

    The container/loader half is GLM-5.2's ``Glm52SparseMoeBlock`` (fp8
    bytes in uint8 containers with fp32 block scales, per-shard stacked
    loaders, ``_apply`` reattachment, TP slicing) brought in here so this
    package stands on its own on main; the dispatch half is GLM-5.3's: the
    eps'd gate, the clamped shared expert, and every routed path clamped.
    The routed partial and the shared partial each reduce themselves (the
    one-reduce fusion is a measured perf call, not a port default).
    """

    def __init__(
        self, config: Glm5NextModelConfig, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        self.tp_size = comm_group.world_size
        self.tp_rank = comm_group.rank
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.moe_intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        shard_inter = divide(config.moe_intermediate_size, self.tp_size)

        self.fp8_experts = (
            config.quantization_config is not None and config.moe_fp8_resident
        )
        self.quant_kernel = getattr(config, "moe_quant_kernel", "reference")
        # Resolved on the real device by process_weights_after_loading;
        # blocks used without the load hook (CPU tests) stay on reference.
        self._use_fused = False

        self.gate = Glm5NextMoEGate(
            hidden_size=config.hidden_size,
            n_routed_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            norm_topk_prob=config.norm_topk_prob,
        )

        self.experts = nn.Module()
        if self.fp8_experts:
            bo, bi = config.quantization_config.weight_block_size
            self.block_size = (bo, bi)
            hidden, E = config.hidden_size, config.n_routed_experts
            assert shard_inter % bo == 0 and shard_inter % bi == 0, (
                f"per-rank intermediate {shard_inter} must be a multiple of the "
                f"fp8 scale block {self.block_size} for clean TP slicing (full "
                f"model: 2048/8=256 per rank, 2 blocks of 128)"
            )
            # fp8 bytes in uint8 containers; e4m3 view happens at dispatch.
            self.experts.gate_up_proj_fp8 = nn.Parameter(
                torch.empty(E, 2 * shard_inter, hidden, dtype=torch.uint8),
                requires_grad=False,
            )
            self.experts.gate_up_proj_scale_inv = nn.Parameter(
                torch.empty(
                    E, 2 * (shard_inter // bo), _ceil_div(hidden, bi),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.experts.down_proj_fp8 = nn.Parameter(
                torch.empty(E, hidden, shard_inter, dtype=torch.uint8),
                requires_grad=False,
            )
            self.experts.down_proj_scale_inv = nn.Parameter(
                torch.empty(
                    E, _ceil_div(hidden, bo), shard_inter // bi,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        else:
            self.experts.gate_up_proj = nn.Parameter(
                torch.empty(
                    config.n_routed_experts, 2 * shard_inter, config.hidden_size,
                )
            )
            self.experts.down_proj = nn.Parameter(
                torch.empty(
                    config.n_routed_experts, config.hidden_size, shard_inter,
                )
            )
        self._attach_expert_weight_loaders()

        self.shared_expert = Glm5NextGatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            comm_group=comm_group,
            activation=config.hidden_act,
            bias=False,
            swiglu_limit=config.swiglu_limit,
        )

    def _attach_expert_weight_loaders(self) -> None:
        """Reattach per-shard loaders after ``_apply`` rebuilds parameters."""
        from functools import partial

        full_inter = self.moe_intermediate_size
        if self.fp8_experts:
            bo, bi = self.block_size
            self.experts.gate_up_proj_fp8.weight_loader = partial(
                _gate_up_fp8_loader, self.tp_rank, self.tp_size, full_inter, 1,
            )
            self.experts.gate_up_proj_scale_inv.weight_loader = partial(
                _gate_up_fp8_loader, self.tp_rank, self.tp_size, full_inter, bo,
            )
            self.experts.down_proj_fp8.weight_loader = partial(
                _down_fp8_loader, self.tp_rank, self.tp_size, full_inter, 1,
            )
            self.experts.down_proj_scale_inv.weight_loader = partial(
                _down_fp8_loader, self.tp_rank, self.tp_size, full_inter, bi,
            )
        else:
            self.experts.gate_up_proj.weight_loader = partial(
                _gate_up_weight_loader, self.tp_rank, self.tp_size, full_inter,
            )
            self.experts.down_proj.weight_loader = partial(
                _down_proj_weight_loader, self.tp_rank, self.tp_size, full_inter,
            )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_expert_weight_loaders()
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        flat = hidden_states.view(-1, self.hidden_size).contiguous()

        # Combine weights stay fp32 into the dispatch (module docstring
        # delta 3): the reference's per-expert multiply promotes and
        # index_add_ downcasts once; the fused kernel folds the same fp32
        # weights into GEMM-2's fp32 accumulator, so NEITHER path downcasts
        # topk_weights (glm52's fused branch does).
        topk_weights, topk_ids = self.gate(flat)
        if self._use_fused:
            routed = self._dispatch_fused(flat, topk_weights, topk_ids)
        else:
            routed = self._dispatch_clamped(flat, topk_weights, topk_ids)
        if self.tp_size > 1:
            self.comm_group.all_reduce(routed)
        shared = self.shared_expert(flat)
        return (routed + shared).view(input_shape)

    def _expert_weights(self, expert_idx, out_dtype):
        """One expert's (gate_up, down) weights, dequantized if fp8-resident."""
        if not self.fp8_experts:
            return (
                self.experts.gate_up_proj[expert_idx],
                self.experts.down_proj[expert_idx],
            )
        gate_up = dequantize_fp8_block_weight(
            self.experts.gate_up_proj_fp8[expert_idx],
            self.experts.gate_up_proj_scale_inv[expert_idx],
            block_size=self.block_size, out_dtype=out_dtype,
        )
        down = dequantize_fp8_block_weight(
            self.experts.down_proj_fp8[expert_idx],
            self.experts.down_proj_scale_inv[expert_idx],
            block_size=self.block_size, out_dtype=out_dtype,
        )
        return gate_up, down

    def _dispatch_clamped(
        self,
        flat: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Clamped per-expert loop over the experts this batch hit.

        Same loop shape as glm52's ``_dispatch_fp8_reference`` (the
        weighted ``index_add_`` keeps partial sums linear, so the TP
        all-reduce of the (T, hidden) result equals reduce-then-sum);
        returns this rank's PARTIAL — the caller reduces.
        ``topk_weights`` arrives fp32: the per-expert multiply promotes the
        expert output to fp32 and ``index_add_`` downcasts once — exactly
        the reference ``Glm5NextTextExperts.forward`` arithmetic.
        """
        final = torch.zeros_like(flat)

        with torch.no_grad():
            expert_mask = F.one_hot(topk_ids, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_t in expert_hit:
            e = expert_idx_t[0]
            top_k_pos, token_idx = torch.where(expert_mask[e])
            tokens = flat[token_idx]

            gate_up_w, down_w = self._expert_weights(e, flat.dtype)
            gate, up = torch.mm(tokens, gate_up_w.T).chunk(2, dim=-1)
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            out = torch.mm(F.silu(gate) * up, down_w.T)
            out = out * topk_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))

        return final

    def _dispatch_fused(
        self,
        flat: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Clamped fp8 fused dispatch -- the fast, capture-safe production path.

        Same fp8-resident experts as ``_dispatch_clamped``, but through the
        shared grouped-GEMM kernel with GLM-5.3's SwiGLU clamp threaded in
        (``swiglu_limit``: gate max ``L``, up ``[-L, L]`` before the
        activation). Same reduce semantics: ``reduce_results`` (default True)
        sums over top-k and returns this rank's PARTIAL ``(T, hidden)`` -- the
        caller does the TP all-reduce, matching ``_dispatch_clamped``.
        ``topk_weights`` stays fp32: the down-GEMM folds it into an fp32
        accumulator and downcasts once (module docstring delta 3), so this is
        the reference's fp32 combine, NOT glm52's bf16-pre-rounded weight.
        NOT bit-exact vs ``_dispatch_clamped`` (fp8 GEMM + on-the-fly
        activation quant vs the bf16 reference GEMM, exactly like glm52
        fused-vs-reference); the clamp is what makes it CORRECT, so it replaces
        the reference on the fast path while ``_dispatch_clamped`` stays the
        HF-faithful fallback.
        """
        from mstar.utils.fused_moe import fused_experts_fp8

        return fused_experts_fp8(
            flat,
            self.experts.gate_up_proj_fp8,
            self.experts.down_proj_fp8,
            self.experts.gate_up_proj_scale_inv,
            self.experts.down_proj_scale_inv,
            topk_weights,
            topk_ids,
            block_size=self.block_size,
            swiglu_limit=self.swiglu_limit,
        )

    def process_weights_after_loading(self, device) -> None:
        """Finalize the fp32 router copy and resolve reference-vs-fused.

        glm52 quant_kernel semantics (``"auto"`` probes, ``"triton"`` must not
        silently downgrade, ``"reference"`` keeps the bit-exact loop) with ONE
        extra guard: the shared ``fused_experts_fp8`` is usable here only if it
        accepts ``swiglu_limit`` -- a clampless kernel would serve subtly wrong
        activations (the reason ``"triton"`` used to be refused outright), so
        ``"triton"`` errors when a clamp-capable kernel is unavailable rather
        than downgrading correctness. The clamped reference loop stays the
        fallback -- and the default (``moe_quant_kernel="reference"``, the M1
        eager parity anchor) -- so ``_use_fused`` is False unless serving
        explicitly asks for the fused path on CUDA.
        """
        self.gate.finalize_weights()
        if not self.fp8_experts:
            self._use_fused = False
            return
        dev = torch.device(device) if device is not None else torch.device("cpu")
        kernel = self.quant_kernel
        fused_ok = dev.type == "cuda"
        if fused_ok:
            try:
                from mstar.utils.fused_moe import fused_experts_fp8
            except Exception:
                fused_ok = False
            else:
                fused_ok = _fused_supports_swiglu_clamp(fused_experts_fp8)
        if kernel == "triton" and not fused_ok:
            raise RuntimeError(
                "moe_quant_kernel='triton' requested but a SwiGLU-clamp-capable "
                "fused fp8 kernel is unavailable (needs CUDA + a fused_experts_fp8 "
                "that accepts swiglu_limit). Use 'reference' (default) for the "
                "clamped reference dispatch."
            )
        self._use_fused = kernel == "triton" or (kernel == "auto" and fused_ok)
