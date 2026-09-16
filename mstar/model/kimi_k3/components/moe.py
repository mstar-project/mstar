"""Latent MoE layer (spec E): noaux-tc sigmoid router, bf16 latent down/up projections,
routed experts in the latent space with fused per-rank expert parameters, and the fused
shared-expert MLP.

Expert storage (per rank, tensor-parallel on the intermediate dim, like
``ParallelSparseMoeBlock``): ``experts.gate_up_proj [E, 2 * inter_local, latent]`` (gate
rows then up rows) and ``experts.down_proj [E, latent, inter_local]``. The checkpoint's
per-expert ``w1`` (gate), ``w3`` (up), ``w2`` (down) tensors are routed into them by the
model's ``load_weights``; MXFP4-packed experts are dequantized by the loader for now
(the quantized-parameter path lands with the kernels).

Dispatch: the reference per-expert loop (any device) or, on CUDA, the Triton grouped GEMM
(``mstar.utils.fused_moe``) once it grows a SiTU epilogue; selected by ``dispatch``.
"""
from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.distributed.linear import RowParallelLinear
from mstar.model.components.moe import _down_proj_weight_loader, _gate_up_weight_loader
from mstar.model.kimi_k3.components.common import (
    KimiRMSNorm,
    ReplicatedLinear,
    replicated_loader,
    restore_kept_dtypes,
)
from mstar.model.kimi_k3.components.mlp import ParallelSiTUMLP
from mstar.model.kimi_k3.reference.moe import routed_experts_loop
from mstar.model.kimi_k3.reference.mxfp4 import MXFP4_GROUP, dequant_mxfp4
from mstar.model.kimi_k3.components.router_kernel import fused_route, fused_route_supported, gate_logits
from mstar.model.kimi_k3.reference.router import noaux_tc_route


def _mxfp4_gate_up_loader(tp_rank, tp_size, full_inter, param, loaded, loaded_shard_id=None):
    """Route one expert's packed ``w1``/``w3`` (``[inter, K/2]`` bytes or ``[inter, K/32]``
    scales) into the fused ``[E, 2*inter_local, ...]`` parameter."""
    kind, expert = loaded_shard_id.split(":")
    expert = int(expert)
    inter_local = full_inter // tp_size
    src = loaded.narrow(0, tp_rank * inter_local, inter_local)
    row0 = 0 if kind == "gate" else inter_local
    dst = param.data[expert].narrow(0, row0, inter_local)
    assert dst.shape == src.shape, (tuple(dst.shape), tuple(src.shape))
    dst.copy_(src)


def _mxfp4_down_loader(tp_rank, tp_size, full_inter, per_col, param, loaded, loaded_shard_id=None):
    """Route one expert's packed ``w2`` (``[K, inter/2]`` bytes or ``[K, inter/32]`` scales):
    the rank's slice is along the packed input dim (``per_col`` = 2 or 32 elements per column)."""
    expert = int(loaded_shard_id.split(":")[1])
    cols_local = (full_inter // tp_size) // per_col
    src = loaded.narrow(1, tp_rank * cols_local, cols_local)
    dst = param.data[expert]
    assert dst.shape == src.shape, (tuple(dst.shape), tuple(src.shape))
    dst.copy_(src)


class NoAuxTCRouter(nn.Module):
    """``gate.weight [E, hidden]`` (fp32 math) + ``gate.e_score_correction_bias [E]``."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, *, renormalize: bool = True,
                 routed_scaling_factor: float = 1.0, scoring: str = "sigmoid",
                 num_expert_group: int = 1, topk_group: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts, dtype=torch.float32))
        self.top_k = top_k
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring = scoring
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.weight.weight_loader = replicated_loader
        self.e_score_correction_bias.weight_loader = replicated_loader
        self.e_score_correction_bias._keep_dtype = torch.float32

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.weight.weight_loader = replicated_loader
        self.e_score_correction_bias.weight_loader = replicated_loader
        self.e_score_correction_bias._keep_dtype = torch.float32
        restore_kept_dtypes(self)
        self._weight_fp32_cache = None
        return result

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.is_cuda and fused_route_supported(self.top_k, self.num_expert_group, self.topk_group, self.scoring):
            # the fp32 gate logits (bf16-in / fp32-out GEMM, no cast launch; see gate_logits),
            # then one launch for scores, bias, top-k and weights
            logits = gate_logits(x.reshape(-1, x.shape[-1]), self.weight, self._gate_weight_fp32)
            return fused_route(
                logits, self.e_score_correction_bias, self.top_k, scoring=self.scoring,
                renormalize=self.renormalize, scale=self.routed_scaling_factor,
            )
        return noaux_tc_route(
            x, self.weight, self.e_score_correction_bias, self.top_k, scoring=self.scoring,
            renormalize=self.renormalize, routed_scaling_factor=self.routed_scaling_factor,
            num_expert_group=self.num_expert_group, topk_group=self.topk_group,
        )

    def _gate_weight_fp32(self) -> torch.Tensor:
        """The gate weight in fp32 (the reference routes in fp32), cast once after loading
        rather than on every call."""
        cached = getattr(self, "_weight_fp32_cache", None)
        # keyed by the parameter's in-place version so a (re)load or device move refreshes it
        key = (self.weight.data_ptr(), self.weight._version)
        if cached is None or getattr(self, "_weight_fp32_key", None) != key:
            cached = self.weight.detach().float().contiguous()
            self._weight_fp32_cache, self._weight_fp32_key = cached, key
        return cached


class KimiLatentMoE(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        latent_size: int,
        num_experts: int,
        top_k: int,
        moe_intermediate_size: int,
        num_shared_experts: int,
        situ_beta: float = 4.0,
        situ_linear_beta: float | None = 25.0,
        latent_norm: bool = True,
        norm_eps: float = 1e-5,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        num_expert_group: int = 1,
        topk_group: int = 1,
        comm_group: CommGroup | None = None,
        dispatch: str = "auto",
        quantized: bool = False,
    ):
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        tp_rank, tp_size = comm_group.rank, comm_group.world_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_intermediate_size = moe_intermediate_size
        self.inter_local = divide(moe_intermediate_size, tp_size)
        self.situ_beta, self.situ_linear_beta = situ_beta, situ_linear_beta
        self.dispatch = dispatch
        self.quantized = quantized
        if quantized:
            assert latent_size % MXFP4_GROUP == 0 and self.inter_local % MXFP4_GROUP == 0

        self.gate = NoAuxTCRouter(
            hidden_size, num_experts, top_k, renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor, num_expert_group=num_expert_group, topk_group=topk_group,
        )
        self.routed_expert_down_proj = ReplicatedLinear(hidden_size, latent_size)
        # row-parallel over the latent: each rank projects its slice of the (replicated,
        # normalized) latent and the partial rides the same all-reduce as the shared
        # experts' partial -- one collective on hidden, and 1/tp of the up-proj weights
        self.routed_expert_up_proj = RowParallelLinear(
            comm_group, latent_size, hidden_size, bias=False, input_is_parallel=False, reduce_results=False,
        )
        self.routed_expert_norm = KimiRMSNorm(latent_size, eps=norm_eps) if latent_norm else None
        self.experts = nn.Module()
        if quantized:
            u8 = torch.uint8
            il = self.inter_local
            self.experts.gate_up_packed = nn.Parameter(
                torch.empty(num_experts, 2 * il, latent_size // 2, dtype=u8), requires_grad=False)
            self.experts.gate_up_scale = nn.Parameter(
                torch.empty(num_experts, 2 * il, latent_size // MXFP4_GROUP, dtype=u8), requires_grad=False)
            self.experts.down_packed = nn.Parameter(
                torch.empty(num_experts, latent_size, il // 2, dtype=u8), requires_grad=False)
            self.experts.down_scale = nn.Parameter(
                torch.empty(num_experts, latent_size, il // MXFP4_GROUP, dtype=u8), requires_grad=False)
        else:
            self.experts.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * self.inter_local, latent_size))
            self.experts.down_proj = nn.Parameter(torch.empty(num_experts, latent_size, self.inter_local))
        self.shared_experts = None
        if num_shared_experts:
            self.shared_experts = ParallelSiTUMLP(
                hidden_size, moe_intermediate_size * num_shared_experts, comm_group=comm_group,
                situ_beta=situ_beta, situ_linear_beta=situ_linear_beta, reduce_results=False,
            )
        self._tp = (tp_rank, tp_size)
        # a converted-weight expert backend (FlashInfer CUTLASS or Marlin) once
        # prepare_experts_backend() ran; it owns the routed forward from then on
        self._backend = None
        self._attach_loaders()

    def _attach_loaders(self) -> None:
        tp_rank, tp_size = self._tp
        full = self.moe_intermediate_size
        if self.quantized:
            for prm in (self.experts.gate_up_packed, self.experts.gate_up_scale,
                        self.experts.down_packed, self.experts.down_scale):
                # the checkpoint's packed uint8 layout until a backend converted the experts;
                # afterwards the parameters hold the kernel layout (Marlin: int32 tiles and
                # E8M0 scales) and must keep *that* dtype through any later ``_apply``
                if self._backend is None:
                    prm._keep_dtype = torch.uint8
            self.experts.gate_up_packed.weight_loader = partial(_mxfp4_gate_up_loader, tp_rank, tp_size, full)
            self.experts.gate_up_scale.weight_loader = partial(_mxfp4_gate_up_loader, tp_rank, tp_size, full)
            self.experts.down_packed.weight_loader = partial(_mxfp4_down_loader, tp_rank, tp_size, full, 2)
            self.experts.down_scale.weight_loader = partial(
                _mxfp4_down_loader, tp_rank, tp_size, full, MXFP4_GROUP)
        else:
            self.experts.gate_up_proj.weight_loader = partial(_gate_up_weight_loader, tp_rank, tp_size, full)
            self.experts.down_proj.weight_loader = partial(_down_proj_weight_loader, tp_rank, tp_size, full)
        self.routed_expert_down_proj.weight.weight_loader = replicated_loader

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_loaders()
        restore_kept_dtypes(self.experts)
        if self._backend is not None:
            # a device move or dtype pass rebinds the parameters' storage: the backend must
            # read the parameters, not copies it kept from the conversion
            ex = self.experts
            self._backend.rebind(ex.gate_up_packed.data, ex.gate_up_scale.data, ex.down_packed.data, ex.down_scale.data)
        return result

    def _use_triton(self, z: torch.Tensor) -> bool:
        if self.dispatch == "reference":
            return False
        return z.is_cuda and z.dtype in (torch.bfloat16, torch.float16)

    EXPERT_BACKENDS = ("w4a16", "humming", "marlin")

    def prepare_experts_backend(self, backend: str, device: torch.device) -> None:
        """Convert the packed experts in place to a fused kernel's layout and route ``_routed``
        through it. ``w4a16``/``humming`` are FlashInfer's SM90 CUTLASS grouped GEMM (bf16 / FP8
        activations), ``marlin`` the Marlin MXFP4 MoE kernel (SM80+, bf16 activations, the
        fastest at decode batch sizes). Only for ``quantized`` modules on CUDA; irreversible
        for this module instance."""
        assert self.quantized, "the fused expert backends take MXFP4-packed experts"
        assert self._backend is None, "already converted"
        assert backend in self.EXPERT_BACKENDS, backend
        ex = self.experts
        packed = (ex.gate_up_packed, ex.gate_up_scale, ex.down_packed, ex.down_scale)
        if backend == "marlin":
            from mstar.utils.fused_moe.marlin import MarlinMXFP4Experts

            be = MarlinMXFP4Experts(
                situ_beta=self.situ_beta, situ_linear_beta=self.situ_linear_beta, device=device)
            converted = be.convert(*(prm.data for prm in packed))
            # the Marlin layouts have other shapes and dtypes: rebind the parameters so the
            # weights stay registered (device moves, state_dict) and the old ones are freed
            for prm, new in zip(packed, converted, strict=True):
                prm.data = new
                prm._keep_dtype = new.dtype
        else:
            from mstar.utils.fused_moe.flashinfer_cutlass import FlashInferMXFP4Experts

            be = FlashInferMXFP4Experts(
                mode=backend, situ_beta=self.situ_beta, situ_linear_beta=self.situ_linear_beta, device=device)
            be.convert(*(prm.data for prm in packed))
        self._backend = be

    def dequantized_experts(self) -> tuple[torch.Tensor, torch.Tensor]:
        """bf16 ``(w13 [E, 2*inter_local, latent], w2 [E, latent, inter_local])`` from the packed
        parameters (reference/CPU path; materializes the experts, so tests only)."""
        assert self._backend is None, "experts were converted to a fused kernel layout"
        e = self.num_experts
        w13 = torch.stack(
            [dequant_mxfp4(self.experts.gate_up_packed[i], self.experts.gate_up_scale[i]) for i in range(e)])
        w2 = torch.stack([dequant_mxfp4(self.experts.down_packed[i], self.experts.down_scale[i]) for i in range(e)])
        return w13, w2

    def _routed(self, z: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        if self._backend is not None:
            return self._backend(z, topk_idx, topk_weight)
        if self.quantized:
            if self._use_triton(z):
                from mstar.utils.fused_moe.mxfp4 import fused_experts_mxfp4

                return fused_experts_mxfp4(
                    z, self.experts.gate_up_packed, self.experts.gate_up_scale,
                    self.experts.down_packed, self.experts.down_scale, topk_weight, topk_idx,
                    self.situ_beta, self.situ_linear_beta,
                )
            w13, w2 = self.dequantized_experts()
            return routed_experts_loop(z, topk_idx, topk_weight, w13, w2, self.situ_beta, self.situ_linear_beta)
        if self._use_triton(z):
            from mstar.utils.fused_moe.mxfp4 import fused_experts_bf16_situ

            return fused_experts_bf16_situ(
                z, self.experts.gate_up_proj, self.experts.down_proj, topk_weight, topk_idx,
                self.situ_beta, self.situ_linear_beta,
            )
        return routed_experts_loop(
            z, topk_idx, topk_weight, self.experts.gate_up_proj, self.experts.down_proj,
            self.situ_beta, self.situ_linear_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        topk_idx, topk_weight = self.gate(x)
        z = self.routed_expert_down_proj(x)
        y = self._routed(z, topk_idx, topk_weight)
        if self.comm_group.world_size > 1:
            y = self.comm_group.all_reduce(y)  # partial sums over the intermediate shards
        if self.routed_expert_norm is not None:
            y = self.routed_expert_norm(y)
        y = self.routed_expert_up_proj(y)  # partial over the latent shards
        if self.shared_experts is not None:
            s_out = self.shared_experts(x)  # partial over the intermediate shards
            buf = self.comm_group.symm_buffer(y.shape, y.dtype, y.device)
            if buf is not None:  # the add lands in the all-reduce buffer: no copy launch
                torch.add(y, s_out, out=buf)
                return self.comm_group.all_reduce_symm_buffer(buf).view(shape)
            y = y + s_out
        if self.comm_group.world_size > 1:
            y = self.comm_group.all_reduce(y)
        return y.view(shape)


__all__ = ["KimiLatentMoE", "NoAuxTCRouter", "F"]
