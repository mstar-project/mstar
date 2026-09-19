"""Latent MoE layer (spec E): noaux-tc sigmoid router, bf16 latent down/up projections,
routed experts in the latent space with fused per-rank expert parameters, and the fused
shared-expert MLP.

Expert storage follows an ``ExpertSharding`` over the comm group: ``experts.gate_up_proj
[E_local, 2 * inter_local, latent]`` (gate rows then up rows) and ``experts.down_proj [E_local,
latent, inter_local]``, where tensor parallelism shards the intermediate dim (``inter_local``,
the ``ParallelSparseMoeBlock`` layout) and expert parallelism (``ep_size > 1``) gives each
rank a subset of whole experts (``E_local``); a rank's routed output is a partial sum in both
cases and the same all-reduce over the latent combines them. The checkpoint's per-expert
``w1`` (gate), ``w3`` (up), ``w2`` (down) tensors are routed into the fused parameters by the
model's ``load_weights`` (packed MXFP4 bytes and scales stay packed in ``quantized`` mode).

Routed dispatch: the reference per-expert loop (any device), the in-tree Triton grouped
GEMM, or a converted-weight backend (Marlin, FlashInfer CUTLASS) installed by
``prepare_experts_backend``; ``dispatch="reference"`` forces the loop.

The layer input is projected once: ``in_proj`` = ``routed_expert_down_proj`` (latent
down-projection) | ``shared_experts.gate_proj`` | ``shared_experts.up_proj`` (column-parallel), one
GEMM per rank instead of two launches (``MOE_IN_PROJ_PARAMS`` routes the checkpoint tensors by
segment). The latent down-projection is column-parallel too by default (``shard_latent``): each
rank projects ``latent / tp`` columns and the shards are all-gathered (a Lamport launch over
symmetric memory on one node, ``CommGroup.all_gather``), which cuts that projection's weight
bytes ``tp``-fold (51 MB to 6.4 MB per layer at TP8) for one small collective; replicated, every
rank streams the whole weight. The router keeps its own fp32-output GEMM.
"""
from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.linear import RowParallelLinear
from mstar.model.components.distributed.merged_linear import COLUMN, REPLICATED, MergedParallelLinear
from mstar.model.components.expert_sharding import ExpertSharding
from mstar.model.kimi_k3.components.common import (
    KimiRMSNorm,
    SiTUAndMul,
    replicated_loader,
    restore_kept_dtypes,
)
from mstar.model.kimi_k3.reference.moe import routed_experts_loop
from mstar.model.kimi_k3.reference.mxfp4 import MXFP4_GROUP, dequant_mxfp4
from mstar.model.kimi_k3.components.common import fused_decode_kernels
from mstar.utils.streams import Fork
from mstar.model.kimi_k3.components.router_kernel import fused_route, fused_route_supported, gate_logits
from mstar.model.kimi_k3.reference.router import noaux_tc_route


# stacked-shard rules for the merged input projection; the shared-expert rules must precede the
# generic ``.gate_proj -> .gate_up_proj`` rules of the dense MLPs in the model's rule list
MOE_IN_PROJ_PARAMS = [
    (".in_proj", ".routed_expert_down_proj", "routed_down"),
    ("block_sparse_moe.in_proj", "block_sparse_moe.shared_experts.gate_proj", "shared_gate"),
    ("block_sparse_moe.in_proj", "block_sparse_moe.shared_experts.up_proj", "shared_up"),
]


def _expert_gate_up_loader(sharding: ExpertSharding, param, loaded, loaded_shard_id=None):
    """Route one expert's ``w1``/``w3`` (bf16 ``[inter, K]``, packed ``[inter, K/2]`` bytes or
    ``[inter, K/32]`` scales) into the fused ``[E_local, 2*inter_local, ...]`` parameter;
    ``loaded_shard_id`` is ``"gate:<expert>"`` / ``"up:<expert>"`` with the global expert id."""
    kind, expert = loaded_shard_id.split(":")
    sharding.load_gate_up(param, loaded, kind, int(expert))


def _expert_down_loader(sharding: ExpertSharding, per_col: int, param, loaded, loaded_shard_id=None):
    """Route one expert's ``w2`` (``[K, inter]`` weights, ``[K, inter/2]`` bytes or ``[K, inter/32]``
    scales, ``per_col`` intermediate channels per stored column) into ``[E_local, K, ...]``."""
    sharding.load_down(param, loaded, int(loaded_shard_id.split(":")[1]), per_col)


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
        ep_size: int = 1,
        expert_sharding: ExpertSharding | None = None,
        shard_latent: bool = True,
    ):
        """``ep_size`` expert-parallel groups over the comm group (1: every rank holds every
        expert, sharded on the intermediate dim; ``comm_group.world_size``: each rank holds
        ``num_experts / world_size`` whole experts; in between: the hybrid). ``expert_sharding``
        overrides it with an explicit placement, for single-rank tests of one shard's partial.
        ``shard_latent`` splits the latent down-projection over the ranks (its output shards are
        all-gathered); False keeps it replicated."""
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        if expert_sharding is None:
            expert_sharding = ExpertSharding.from_group(comm_group, num_experts, moe_intermediate_size, ep_size)
        elif comm_group.world_size > 1 and (expert_sharding.world_size, expert_sharding.rank) != (
                comm_group.world_size, comm_group.rank):
            raise ValueError(f"expert sharding {expert_sharding} does not describe rank {comm_group.rank} "
                             f"of a group of {comm_group.world_size}")
        self.sharding = expert_sharding
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_experts = num_experts  # global: the router scores all of them
        self.top_k = top_k
        self.moe_intermediate_size = moe_intermediate_size
        self.local_experts = self.sharding.local_experts
        self.inter_local = self.sharding.inter_local
        self.situ_beta, self.situ_linear_beta = situ_beta, situ_linear_beta
        self.dispatch = dispatch
        self.quantized = quantized
        if quantized:
            assert latent_size % MXFP4_GROUP == 0 and self.inter_local % MXFP4_GROUP == 0

        self.gate = NoAuxTCRouter(
            hidden_size, num_experts, top_k, renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor, num_expert_group=num_expert_group, topk_group=topk_group,
        )
        # one GEMM on the layer input: the latent down-projection (column-parallel and all-gathered,
        # or replicated) and, when present, the shared experts' column-parallel gate | up (adjacent,
        # so SiTU reads them as one block)
        shared_inter = moe_intermediate_size * num_shared_experts
        tp = comm_group.world_size
        self.latent_sharded = bool(shard_latent) and tp > 1 and latent_size % tp == 0
        self.latent_local = latent_size // tp if self.latent_sharded else latent_size
        segments = [("routed_down", latent_size, COLUMN if self.latent_sharded else REPLICATED)]
        if num_shared_experts:
            segments += [("shared_gate", shared_inter, COLUMN), ("shared_up", shared_inter, COLUMN)]
        self.in_proj = MergedParallelLinear(comm_group, hidden_size, segments)
        if num_shared_experts:
            # SiTU reads gate | up as one [T, 2 * S_local] block: the two segments must be adjacent
            gate_off, up_off = self.in_proj.offsets["shared_gate"], self.in_proj.offsets["shared_up"]
            assert up_off == gate_off + self.in_proj.local_sizes["shared_gate"], (gate_off, up_off)
        # row-parallel over the latent: each rank projects its slice of the (replicated,
        # normalized) latent and the partial rides the same all-reduce as the shared
        # experts' partial -- one collective on hidden, and 1/tp of the up-proj weights
        self.routed_expert_up_proj = RowParallelLinear(
            comm_group, latent_size, hidden_size, bias=False, input_is_parallel=True, reduce_results=False,
        )
        self.routed_expert_norm = KimiRMSNorm(latent_size, eps=norm_eps) if latent_norm else None
        self.experts = nn.Module()
        e_local = self.local_experts
        if quantized:
            u8 = torch.uint8
            il = self.inter_local
            self.experts.gate_up_packed = nn.Parameter(
                torch.empty(e_local, 2 * il, latent_size // 2, dtype=u8), requires_grad=False)
            self.experts.gate_up_scale = nn.Parameter(
                torch.empty(e_local, 2 * il, latent_size // MXFP4_GROUP, dtype=u8), requires_grad=False)
            self.experts.down_packed = nn.Parameter(
                torch.empty(e_local, latent_size, il // 2, dtype=u8), requires_grad=False)
            self.experts.down_scale = nn.Parameter(
                torch.empty(e_local, latent_size, il // MXFP4_GROUP, dtype=u8), requires_grad=False)
        else:
            self.experts.gate_up_proj = nn.Parameter(torch.empty(e_local, 2 * self.inter_local, latent_size))
            self.experts.down_proj = nn.Parameter(torch.empty(e_local, latent_size, self.inter_local))
        self.shared_experts = None
        if num_shared_experts:
            # gate/up live in in_proj; the SiTU-GLU and the row-parallel down-projection (its
            # partial rides the layer's final all-reduce) stay under the checkpoint's module name
            self.shared_experts = nn.Module()
            self.shared_experts.act = SiTUAndMul(situ_beta, situ_linear_beta)
            self.shared_experts.down_proj = RowParallelLinear(
                comm_group=comm_group, input_size=shared_inter, output_size=hidden_size,
                bias=False, input_is_parallel=True, reduce_results=False,
            )
            self._shared_gate_up = (self.in_proj.offsets["shared_gate"], 2 * self.in_proj.local_sizes["shared_gate"])
        # a converted-weight expert backend (FlashInfer CUTLASS or Marlin) once
        # prepare_experts_backend() ran; it owns the routed forward from then on
        self._backend = None
        self._fork_in, self._fork_experts = Fork(), Fork()
        self._attach_loaders()

    def _attach_loaders(self) -> None:
        sh = self.sharding
        if self.quantized:
            for prm in (self.experts.gate_up_packed, self.experts.gate_up_scale,
                        self.experts.down_packed, self.experts.down_scale):
                # the checkpoint's packed uint8 layout until a backend converted the experts;
                # afterwards the parameters hold the kernel layout (Marlin: int32 tiles and
                # E8M0 scales) and must keep *that* dtype through any later ``_apply``
                if self._backend is None:
                    prm._keep_dtype = torch.uint8
            self.experts.gate_up_packed.weight_loader = partial(_expert_gate_up_loader, sh)
            self.experts.gate_up_scale.weight_loader = partial(_expert_gate_up_loader, sh)
            self.experts.down_packed.weight_loader = partial(_expert_down_loader, sh, 2)
            self.experts.down_scale.weight_loader = partial(_expert_down_loader, sh, MXFP4_GROUP)
        else:
            self.experts.gate_up_proj.weight_loader = partial(_expert_gate_up_loader, sh)
            self.experts.down_proj.weight_loader = partial(_expert_down_loader, sh, 1)

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
                situ_beta=self.situ_beta, situ_linear_beta=self.situ_linear_beta, device=device,
                sharding=self.sharding)
            converted = be.convert(*(prm.data for prm in packed))
            # the Marlin layouts have other shapes and dtypes: rebind the parameters so the
            # weights stay registered (device moves, state_dict) and the old ones are freed
            for prm, new in zip(packed, converted, strict=True):
                prm.data = new
                prm._keep_dtype = new.dtype
        else:
            from mstar.utils.fused_moe.flashinfer_cutlass import FlashInferMXFP4Experts

            be = FlashInferMXFP4Experts(
                mode=backend, situ_beta=self.situ_beta, situ_linear_beta=self.situ_linear_beta, device=device,
                sharding=self.sharding)
            be.convert(*(prm.data for prm in packed))
        self._backend = be

    def dequantized_experts(self) -> tuple[torch.Tensor, torch.Tensor]:
        """bf16 ``(w13 [E_local, 2*inter_local, latent], w2 [E_local, latent, inter_local])`` from the
        packed parameters (reference/CPU path; materializes the experts, so tests only)."""
        assert self._backend is None, "experts were converted to a fused kernel layout"
        e = self.local_experts
        w13 = torch.stack(
            [dequant_mxfp4(self.experts.gate_up_packed[i], self.experts.gate_up_scale[i]) for i in range(e)])
        w2 = torch.stack([dequant_mxfp4(self.experts.down_packed[i], self.experts.down_scale[i]) for i in range(e)])
        return w13, w2

    def _routed(
        self, z: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor, out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """This rank's partial sum of the routed experts, ``[T, latent]`` (the full sum when it
        holds every expert); ``topk_idx`` are global expert ids. The fused backends can write
        the result into ``out`` (the caller checks ``result is out``)."""
        if self._backend is not None:
            return self._backend(z, topk_idx, topk_weight, out=out)
        # local expert ids; assignments of other expert-parallel ranks carry the skipped id
        topk_idx = self.sharding.localize(topk_idx)
        partial_sum = self.sharding.is_partial
        if self.quantized:
            if self._use_triton(z):
                from mstar.utils.fused_moe.mxfp4 import fused_experts_mxfp4

                return fused_experts_mxfp4(
                    z, self.experts.gate_up_packed, self.experts.gate_up_scale,
                    self.experts.down_packed, self.experts.down_scale, topk_weight, topk_idx,
                    self.situ_beta, self.situ_linear_beta, partial=partial_sum,
                )
            w13, w2 = self.dequantized_experts()
            return routed_experts_loop(z, topk_idx, topk_weight, w13, w2, self.situ_beta, self.situ_linear_beta)
        if self._use_triton(z):
            from mstar.utils.fused_moe.mxfp4 import fused_experts_bf16_situ

            return fused_experts_bf16_situ(
                z, self.experts.gate_up_proj, self.experts.down_proj, topk_weight, topk_idx,
                self.situ_beta, self.situ_linear_beta, partial=partial_sum,
            )
        return routed_experts_loop(
            z, topk_idx, topk_weight, self.experts.gate_up_proj, self.experts.down_proj,
            self.situ_beta, self.situ_linear_beta,
        )

    def _latent(self, mixed: torch.Tensor) -> torch.Tensor:
        """The routed experts' input ``[T, latent]`` out of the merged projection's output: this
        rank's columns, all-gathered when the down-projection is sharded. The expert kernels read
        it as a plain row-major matrix (the all-gather writes one, from the slice as it is; a
        replicated view is one for a single row and a small copy otherwise)."""
        z = mixed.narrow(-1, 0, self.latent_local)
        if self.latent_sharded:
            # the Lamport gather reads the slice with its row stride; the NCCL fallback copies it itself
            return self.comm_group.all_gather(z if fused_decode_kernels() else z.contiguous(), dim=-1)
        return z.contiguous()

    def routed_down(self, x: torch.Tensor) -> torch.Tensor:
        """The latent input of the routed experts ``[T, latent]`` (tests; the forward takes the
        same view out of the merged projection)."""
        return self._latent(self.in_proj(x.reshape(-1, self.hidden_size)))

    def _routed_reduced(self, z: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        """The routed experts' output summed over the ranks: this rank's partial (over its
        intermediate shard and/or its experts) written straight into the symmetric all-reduce
        buffer when that path applies (no copy launch), then reduced."""
        buf = self.comm_group.symm_buffer(z.shape, z.dtype, z.device)
        y = self._routed(z, topk_idx, topk_weight, out=buf)
        if buf is not None:
            if y is not buf:
                buf.copy_(y)
            return self.comm_group.all_reduce_symm_buffer(buf)
        if self.comm_group.world_size > 1:
            y = self.comm_group.all_reduce(y)
        return y

    def _shared(self, mixed: torch.Tensor) -> torch.Tensor:
        """Shared experts' partial ``[T, hidden]`` from the merged projection's gate | up block (a
        strided view: the SiTU kernel takes the row stride)."""
        off, width = self._shared_gate_up
        return self.shared_experts.down_proj(self.shared_experts.act(mixed.narrow(-1, off, width)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        # the router and the merged projection read the same input: two streams under a capture
        (topk_idx, topk_weight), mixed = self._fork_in.run(lambda: self.gate(x), lambda: self.in_proj(x))
        z = self._latent(mixed)
        # the shared experts only need the projection: they run beside the routed path
        y, s_out = self._fork_experts.run(
            lambda: self._routed_reduced(z, topk_idx, topk_weight),
            lambda: self._shared(mixed) if self.shared_experts is not None else None,  # partial over the intermediate shards
        )
        if self.routed_expert_norm is not None:
            y = self.routed_expert_norm(y)
        # partial over the latent shards: this rank's columns of the latent, a view the GEMM reads with
        # its row stride (splitting inside the row-parallel linear copied them for more than one row)
        up = self.routed_expert_up_proj
        y = y.narrow(-1, up.tp_rank * up.input_size_per_partition, up.input_size_per_partition)
        if not fused_decode_kernels():
            y = y.contiguous()
        if self.shared_experts is not None:
            buf = self.comm_group.symm_buffer((y.shape[0], self.hidden_size), y.dtype, y.device)
            if buf is not None and fused_decode_kernels():
                # the up-projection accumulates onto the shared partial, straight into the all-reduce
                # buffer: the GEMM and the add in one launch, one rounding
                torch.addmm(s_out, y, up.weight.t(), out=buf)
                return self.comm_group.all_reduce_symm_buffer(buf).view(shape)
            y = up(y)
            if buf is not None:  # the add lands in the all-reduce buffer: no copy launch
                torch.add(y, s_out, out=buf)
                return self.comm_group.all_reduce_symm_buffer(buf).view(shape)
            y = y + s_out
        else:
            y = up(y)
        if self.comm_group.world_size > 1:
            y = self.comm_group.all_reduce(y)
        return y.view(shape)


__all__ = ["MOE_IN_PROJ_PARAMS", "KimiLatentMoE", "NoAuxTCRouter", "F"]
