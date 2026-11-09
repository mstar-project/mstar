"""Kimi K3's MLPs: the SiTU-GLU dense MLP and the latent MoE.

The latent MoE routes each token on its hidden state (896 experts, top 16, sigmoid noaux_tc),
projects it down into the routed experts' latent (``routed_expert_down_proj``, 7168 -> 3584),
runs the experts there, normalizes their sum (``routed_expert_norm``) and projects it back up
(``routed_expert_up_proj``); the two shared experts (one MLP twice as wide) read the hidden
state. The block is the shared ``ParallelSparseMoeBlock`` (tensor parallel over the experts'
intermediate dim) or ``ExpertParallelSparseMoeBlock`` (whole experts per rank) over the latent,
so ``<block>.experts`` is the routed experts layer the quantization API sees, and the router
is the shared ``NoAuxTcRouter``. After merceod's #257 (its ``KimiLatentMoE``) without its
fused-kernel and collective rewrites, which main has its own versions of or lacks.
"""
from __future__ import annotations

import logging

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine import side_stream
from mstar.model.components.distributed import ParallelGatedMLP, RowParallelLinear
from mstar.model.components.moe import (
    _MM_OUT_DTYPE,
    ExpertParallelSparseMoeBlock,
    NoAuxTcRouter,
    ParallelSparseMoeBlock,
)
from mstar.model.components.norm import RMSNorm
from mstar.model.kimi_k3.components import latent_front
from mstar.model.kimi_k3.config import KimiK3ModelConfig
from mstar.utils import small_m_gemm
from mstar.utils.fused_moe import mxfp8_cast
from mstar.utils.situ import SiTU

logger = logging.getLogger(__name__)

# the routed finalize folds into the routed all-reduce + RMSNorm up to this many tokens (decode; vLLM's K3 tail
# fusion stops at 128 too): the one-shot Lamport payload is [tokens, latent], its TRT-LLM workspace sized for it
FOLD_MAX_TOKENS = 128
# (TP group, latent, dtype) -> whether the fold serves (decided once, on the first block, by every rank)
_FOLD_VERDICT: dict[tuple, tuple[bool, str]] = {}


def _raise_if_device_fault(exc: Exception) -> None:
    """A device fault (an illegal address, ...) poisons the context: say so and stop the load, rather than log a
    fallback that the next CUDA call fails anyway."""
    fault = getattr(torch, "AcceleratorError", None)
    if (fault is not None and isinstance(exc, fault)) or "CUDA error" in str(exc):
        raise RuntimeError("the routed finalize fold's probe faulted the device, which no fallback survives; "
                           "serve with model_kwargs.routed_experts_finalize: kernel") from exc


class KimiSiTUMLP(ParallelGatedMLP):
    """``ParallelGatedMLP`` with SiTU-GLU, which bends the up half too: ``act(gate) * up`` does
    not express it."""

    def __init__(self, *args, situ: SiTU, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.situ = situ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.from_gate_up(self.gate_up_proj(x))

    def from_gate_up(self, gate_up: torch.Tensor) -> torch.Tensor:
        """``forward`` from ``gate_up_proj``'s output, computed by the caller."""
        gate, up = gate_up.split(self.intermediate_size_per_partition, dim=-1)
        return self.down_proj(self.situ(gate, up))


class _LatentMoE:
    """The latent MoE over a routed-experts block (``ParallelSparseMoeBlock`` or its expert
    parallel twin, the next base in the MRO): this rank's partial of the layer's output, summed
    in one all-reduce with the shared experts' partial. Two all-reduces a layer under TP/EP: the
    routed experts' latent sum must be whole before ``routed_expert_norm`` (nonlinear), so it
    cannot join the shared experts' partial; it is reduced in the norm's fused kernel."""

    def __init__(self, config: KimiK3ModelConfig, comm_group: CommGroup | None = None) -> None:
        comm_group = comm_group or CommGroup.trivial()
        latent = config.routed_expert_hidden_size
        super().__init__(
            hidden_size=latent,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            router=NoAuxTcRouter(
                hidden_size=config.hidden_size,
                n_routed_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                routed_scaling_factor=config.routed_scaling_factor,
                norm_topk_prob=config.moe_renormalize,
            ),
            comm_group=comm_group,
            activation=config.situ(),
        )
        self.model_hidden_size = config.hidden_size
        # FlashInfer's trtllm-gen MXFP4 x MXFP8 (SiTU fused) on B200/B300, else Marlin (QuantExperts)
        self.routed_experts_backend = config.routed_experts_backend
        self.routed_experts_autotune = config.routed_experts_autotune
        self.routed_experts_tune_tokens = config.routed_experts_tune_tokens or None
        # with the padding rows masked every call skips slots (the sentinel id): FlashInfer's FP4 MoE is tuned so,
        # unless the router runs inside it (the logits carry no sentinel; QuantExperts.experts_after_load)
        self.routed_experts_skip_invalid = config.moe_mask_padding
        # which branch runs on the engine's side stream up to shared_experts_overlap_max_tokens rows
        # (KimiK3ModelConfig.shared_experts_overlap; past them one stream and the packed GEMMs); under expert
        # parallelism the routed experts sync their routing over the group, a collective the side stream can't hold
        mode = config.shared_experts_overlap if config.num_shared_experts else "off"
        if mode == "routed" and config.moe_expert_parallel:
            mode = "shared"
        self.moe_overlap = mode
        self.shared_experts_overlap_max_tokens = config.shared_experts_overlap_max_tokens
        # the router inside FlashInfer's FP4 MoE where it can run there (QuantExperts.experts_after_load)
        self.routed_experts_routing = config.routed_experts_routing
        # "fused": the routed experts' top-k sum folds into the routed all-reduce + RMSNorm (_setup_finalize_fold)
        self.routed_experts_finalize = config.routed_experts_finalize
        self._fold_finalize = False
        # "flashinfer": the router, the latent and the shared experts' activation in one launch up to
        # latent_front.MAX_TOKENS rows (_setup_front, at load)
        self.latent_moe_front = config.latent_moe_front
        self._front = False
        # Replicated: one GEMM with no collective, which at decode beats an all-gather of a
        # column-parallel projection (51 MB a layer, read once).
        self.routed_expert_down_proj = nn.Linear(config.hidden_size, latent, bias=False)
        self.routed_expert_norm = (
            RMSNorm(latent, eps=config.rms_norm_eps) if config.latent_moe_use_norm else None)
        # Row-parallel over the (replicated, normed) latent: its partial joins the shared
        # experts' in the layer's one all-reduce.
        self.routed_expert_up_proj = RowParallelLinear(
            comm_group, latent, config.hidden_size, bias=False,
            input_is_parallel=False, reduce_results=False)
        self.shared_experts = None
        if config.num_shared_experts:
            self.shared_experts = KimiSiTUMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
                comm_group=comm_group, situ=config.situ(), reduce_results=False)
        # [router | routed_expert_down_proj | shared gate_up_proj], the three projections of the
        # block's input as one GEMM (``process_weights_after_loading``); their weights view it
        nn.Module.register_buffer(self, "_in_weight", None, persistent=False)
        # [routed_expert_up_proj | shared down_proj] along K: the two row-parallel partials of the
        # layer's output as one GEMM over [this rank's latent | the shared experts' activation]
        nn.Module.register_buffer(self, "_out_weight", None, persistent=False)

    def _in_parts(self) -> list[nn.Parameter] | None:
        """The weights packed into ``_in_weight``, where all three are plain tensors of one dtype: the router's,
        the latent down projection's and the shared experts' gate_up, in that order, so an overlapped step's main
        path takes the first two rows' worth as one contiguous slice and the shared experts the rest."""
        if self.shared_experts is None:
            return None
        parts = [self.gate.weight, self.routed_expert_down_proj.weight,
                 self.shared_experts.gate_up_proj.weight]
        mods = (self.routed_expert_down_proj, self.shared_experts.gate_up_proj)
        if (any(getattr(m, "quant_format", None) is not None for m in mods)
                or len({(w.dtype, w.device, w.shape[1]) for w in parts}) != 1
                or parts[0].dtype not in (torch.bfloat16, torch.float16, torch.float32)):
            return None
        return parts

    def _out_parts(self) -> list[nn.Parameter] | None:
        """The weights packed into ``_out_weight``: the latent up projection's and the shared experts'
        down projection's, where both are plain row-parallel weights of one dtype. Their parameters view it
        (column slices), which an overlapped step's separate GEMMs read as they are."""
        if self.shared_experts is None:
            return None
        mods = (self.routed_expert_up_proj, self.shared_experts.down_proj)
        parts = [m.weight for m in mods]
        if (any(type(m) is not RowParallelLinear or m.quant_format is not None or m.bias is not None
                for m in mods)
                or len({(w.dtype, w.device, w.shape[0]) for w in parts}) != 1
                or parts[0].dtype not in (torch.bfloat16, torch.float16, torch.float32)):
            return None
        return parts

    def process_weights_after_loading(self, device) -> None:
        """Pack the router, the latent down projection and the shared experts' gate_up, which all
        read the block's input, into one weight: at decode each is a few-microsecond GEMM whose
        cost is mostly its launch (B300 TP8 c=1: 92 layers x 3). Pack the latent up projection and
        the shared experts' down projection along K too: both are this rank's partial of the
        layer's output, summed by the one all-reduce, so one GEMM over the two activations side by
        side gives their sum (7168 x 1216 at TP8 for 3.0 + 3.65 us and an add, B300 bench).
        Their parameters become views of the packed weights, so the block holds one copy."""
        del device
        parts = self._in_parts()
        if parts is not None and self._in_weight is None:
            self._in_weight = torch.cat([w.detach() for w in parts]).contiguous()
            self._view_in_weight()
        parts = self._out_parts()
        if parts is not None and self._out_weight is None:
            self._out_weight = torch.cat([w.detach() for w in parts], dim=1).contiguous()
            self._view_out_weight()

    def _view_out_weight(self) -> None:
        start = 0
        for weight in self._out_parts() or ():
            cols = weight.shape[1]
            weight.data = self._out_weight[:, start:start + cols]
            start += cols

    def _view_in_weight(self) -> None:
        start = 0
        for weight in self._in_parts() or ():
            rows = weight.shape[0]
            weight.data = self._in_weight[start:start + rows]
            start += rows

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # `_apply` moved the parameters apart from the packed weights
        if self._in_weight is not None:
            self._view_in_weight()
        if self._out_weight is not None:
            self._view_out_weight()
        return result

    def _project_in(self, x: torch.Tensor, with_shared: bool = True, route: bool = True, mxfp8: bool = False):
        """``(routing weights, experts, z, gate_up, logits)``: the router's top-k from its fp32
        logits, the latent ``z`` and the shared experts' ``gate_up`` (None: the shared experts
        project their own), as the separate projections give them up to summation order, and the
        fp32 logits themselves (None off the packed GEMM). ``with_shared=False`` (an overlapped step)
        runs the packed GEMM over the router's and the latent's rows only; ``route=False`` leaves the
        top-k to the experts kernel (weights and experts None) where the packed GEMM gives the logits.
        ``mxfp8``: z as the experts kernel's MXFP8 ``(values, scales)``, rounded, quantized and gate_up
        cast in one kernel (``mxfp8_cast``; B300 K3.1 decode: three launches a layer before)."""
        w = self._in_weight
        if w is not None and not with_shared:
            w = w[:self.gate.n_routed_experts + self.routed_expert_down_proj.out_features]
        fp32 = x.dtype == torch.float32
        if w is None or x.dtype != w.dtype or not (fp32 or (x.is_cuda and _MM_OUT_DTYPE)):
            routing_weights, selected_experts, _ = self.gate(x)
            return routing_weights, selected_experts, self.routed_expert_down_proj(x), None, None
        e, latent = self.gate.n_routed_experts, self.routed_expert_down_proj.out_features
        # the router's logits stay fp32; z and gate_up round to bf16 as their own GEMMs' outputs do
        with torch.autocast(x.device.type, enabled=False):
            h = small_m_gemm.linear(x, w, out_dtype=torch.float32)
        logits = h[:, :e]
        routing_weights = selected_experts = None
        if route:
            routing_weights, selected_experts, _ = self.gate.route(logits)
        shared = h.shape[1] > e + latent
        if mxfp8 and not route and mxfp8_cast.supported(h, latent):
            z_q, z_sf, gate_up = torch.ops.mstar.split_cast_mxfp8(h, e, latent)
            return None, None, (z_q, z_sf), gate_up if shared else None, logits
        # one cast for both (B300 K3.1 decode: two kernels of ~1.2 us a layer before), and z apart only
        # where it isn't already: a single row's view is (Dynamo specializes a size of 1)
        zg = h[:, e:].to(x.dtype)
        z = zg[:, :latent].contiguous() if shared else zg
        gate_up = zg[:, latent:] if shared else None
        return routing_weights, selected_experts, z, gate_up, logits

    def _kernel_routes(self) -> bool:
        """The routed experts' kernel routes from the logits itself (``TrtllmFp4Experts.from_logits``:
        FlashInfer's DeepSeekV3 routing, set at load where it serves this router)."""
        return getattr(getattr(self, "_experts_kernel", None), "routing", None) is not None

    def _takes_mxfp8(self, kernel_routes: bool) -> bool:
        """The kernel-routed experts take z as MXFP8 (``TrtllmFp4Experts.takes_mxfp8``): ``_project_in``
        quantizes it in its cast."""
        return kernel_routes and getattr(self._experts_kernel, "takes_mxfp8", bool)()

    def forward(self, hidden_states: torch.Tensor, token_valid: torch.Tensor | None = None) -> torch.Tensor:
        """``token_valid`` (``(tokens,)`` bool): a padded step's real rows; the others skip the routed experts."""
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.model_hidden_size)
        norm = self.routed_expert_norm
        tp = self.comm_group if self.comm_group.world_size > 1 else None
        kernel_routes = self._kernel_routes()
        mode = self.moe_overlap
        # FlashInfer's front computes the shared experts' activation in the router's and the latent's launch, so
        # its steps run on one stream
        front = self._front and x.shape[0] <= latent_front.MAX_TOKENS and x.dtype == self._in_weight.dtype
        if front or x.shape[0] > self.shared_experts_overlap_max_tokens:
            # past the bucket the overlap pays off at: one stream, the packed GEMMs (both layouts view the same
            # weights, so a captured bucket bakes its own choice)
            mode = "off"
        if mode == "routed":
            # the router and the routed experts on the side stream, the shared experts on the main one; the
            # routed partial joins before its all-reduce + norm, so every collective stays on the main stream
            # (and the MoE finalizes on the side stream: no finalize fold, itself a collective, in this mode)
            y, token = side_stream.fork(self._routed_partial, x, token_valid)
            shared_out = self.shared_experts(x)
            side_stream.join(token)
            out = self._up_plus(self._reduce_routed(y, norm, tp), shared_out)
        else:
            token = router = None
            split = mode == "shared" and self.latent_moe_front == "split" and self._in_weight is not None
            if split:
                # the router's GEMM first on the side stream, beside the latent's on the main one
                router = side_stream.fork(self._router_logits, x)
            if mode == "shared":
                # the shared experts read only x and join at the up projection: on the side stream, beside
                # the router, the routed experts and their all-reduce + norm
                shared_out, token = side_stream.fork(self._shared_partial, x)
            act = None
            if front:
                routing_weights, selected_experts, z, act, logits = self._front_in(x, route=not kernel_routes)
                gate_up = None
            elif split:
                routing_weights, selected_experts, z, gate_up, logits = self._split_in(
                    x, router, route=not kernel_routes)
            else:
                routing_weights, selected_experts, z, gate_up, logits = self._project_in(
                    x, with_shared=mode == "off", route=not kernel_routes, mxfp8=self._takes_mxfp8(kernel_routes))
            # summed over the ranks: the norm needs every intermediate shard (or expert). With the
            # norm, this rank's partial goes to it unreduced and the norm's kernel reduces it (the
            # TP group's fused all-reduce + RMSNorm where it has one, else all-reduce then norm).
            if kernel_routes and self._fold_finalize and routing_weights is None:
                y = self._routed_folded(z, logits, tp)
            else:
                y = self._routed_experts(z, routing_weights, selected_experts,
                                         reduce_results=norm is None or tp is None, token_valid=token_valid,
                                         logits=logits if kernel_routes and routing_weights is None else None)
                if norm is not None:
                    y = norm(y, tp)
            if mode != "off":
                side_stream.join(token)
                out = self._up_plus(y, shared_out)
            elif self._out_weight is not None:
                out = self._project_out(x, y, gate_up, act)
            else:
                out = self.routed_expert_up_proj(y)
                if act is not None:
                    out = out + self.shared_experts.down_proj(act)
                elif self.shared_experts is not None:
                    out = out + (self.shared_experts(x) if gate_up is None
                                 else self.shared_experts.from_gate_up(gate_up))
        if self.comm_group.world_size > 1:
            out = self.comm_group.all_reduce(out)
        return out.view(shape)

    def _routed_partial(self, x: torch.Tensor, token_valid: torch.Tensor | None = None) -> torch.Tensor:
        """The router, the latent down projection (their packed GEMM) and the routed experts: this rank's
        unreduced partial of the experts' latent sum (``mode == "routed"``: on the side stream, which
        holds no collective)."""
        kernel_routes = self._kernel_routes()
        routing_weights, selected_experts, z, _, logits = self._project_in(
            x, with_shared=False, route=not kernel_routes, mxfp8=self._takes_mxfp8(kernel_routes))
        return self._routed_experts(z, routing_weights, selected_experts, reduce_results=False,
                                    token_valid=token_valid,
                                    logits=logits if kernel_routes and routing_weights is None else None)

    def _reduce_routed(self, y: torch.Tensor, norm, tp) -> torch.Tensor:
        """The routed partial's reduction over the ranks and its norm: the TP group's fused all-reduce + RMSNorm
        (``RMSNorm.forward(y, tp)``), or the all-reduce alone without the norm."""
        if norm is not None:
            return norm(y, tp)
        return self.comm_group.all_reduce(y) if tp is not None else y

    def _shared_partial(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's partial of the shared experts' output: gate_up, SiTU and down, the SiTU one fused
        kernel on CUDA (the side stream runs it outside the compiled region)."""
        shared = self.shared_experts
        gate_up = shared.gate_up_proj(x)
        if not gate_up.is_cuda:
            return shared.from_gate_up(gate_up)
        from mstar.utils.fused_moe.kernels import act_and_mul_triton

        act = gate_up.new_empty(gate_up.shape[0], shared.intermediate_size_per_partition)
        act_and_mul_triton(gate_up.contiguous(), act, activation=shared.situ)
        return shared.down_proj(act)

    def _up_plus(self, y: torch.Tensor, shared_out: torch.Tensor) -> torch.Tensor:
        """This rank's partial of the layer's output: the latent up projection of its slice of ``y``, with
        the shared experts' partial added in the GEMM (``addmm``) where the projection is a plain weight."""
        up = self.routed_expert_up_proj
        if type(up) is not RowParallelLinear or up.quant_format is not None or up.bias is not None:
            return up(y) + shared_out
        cols = up.input_size_per_partition
        rows = y[..., up.tp_rank * cols:(up.tp_rank + 1) * cols] if y.shape[-1] != cols else y
        return torch.addmm(shared_out, rows, up.weight.t())

    def _project_out(self, x: torch.Tensor, y: torch.Tensor, gate_up: torch.Tensor | None,
                     act: torch.Tensor | None = None) -> torch.Tensor:
        """This rank's partial of the layer's output: ``[y's rank slice | SiTU(gate, up)] @ _out_weight.T``,
        the up projection's and the shared experts' down projection's partials summed in one GEMM. ``act``: the
        shared experts' activation already computed (the front's)."""
        up = self.routed_expert_up_proj
        cols = up.input_size_per_partition
        rows = y[..., up.tp_rank * cols:(up.tp_rank + 1) * cols] if y.shape[-1] != cols else y
        shared = self.shared_experts
        if act is None:
            if gate_up is None:
                gate_up = shared.gate_up_proj(x)
            gate, up_half = gate_up.split(shared.intermediate_size_per_partition, dim=-1)
            act = shared.situ(gate, up_half)
        return small_m_gemm.linear(torch.cat([rows, act], dim=-1), self._out_weight)

    def _router_logits(self, x: torch.Tensor) -> torch.Tensor:
        """The router's fp32 logits from its rows of the packed input weight (``split``: on the side stream)."""
        with torch.autocast(x.device.type, enabled=False):
            return small_m_gemm.linear(x, self._in_weight[:self.gate.n_routed_experts], out_dtype=torch.float32)

    def _split_in(self, x: torch.Tensor, router, route: bool = True):
        """``_project_in``'s ``(routing weights, experts, z, None, logits)`` with the router's GEMM forked to the side
        stream (``router``: ``side_stream.fork``'s pair) and the latent's here, in ``x``'s dtype (its own small-M
        backend: FlashInfer's cute-dsl on B300, 9.7 us at one row); the experts kernel quantizes z itself."""
        e, latent = self.gate.n_routed_experts, self.routed_expert_down_proj.out_features
        z = small_m_gemm.linear(x, self._in_weight[e:e + latent])
        logits, token = router
        side_stream.join(token)
        routing_weights = selected_experts = None
        if route:
            routing_weights, selected_experts, _ = self.gate.route(logits)
        return routing_weights, selected_experts, z, None, logits

    def _front_in(self, x: torch.Tensor, route: bool = True):
        """``_project_in``'s ``(routing weights, experts, z, ·, logits)`` from FlashInfer's front, with the shared
        experts' SiTU activation in the gate_up slot: one launch for the router's fp32 logits, the bf16 latent and
        the activation (``latent_front``)."""
        logits, z, act = torch.ops.mstar.k31_latent_front(
            x.contiguous(), self._in_weight, self.gate.n_routed_experts, self.routed_expert_down_proj.out_features)
        routing_weights = selected_experts = None
        if route:
            routing_weights, selected_experts, _ = self.gate.route(logits)
        return routing_weights, selected_experts, z, act, logits

    @torch.compiler.disable
    def _routed_folded(self, z: torch.Tensor, logits: torch.Tensor, tp: CommGroup) -> torch.Tensor:
        """The routed experts, normed and reduced, with their top-k sum folded into the routed all-reduce +
        RMSNorm: the MoE stops before its finalize and ``allreduce_moe_finalize_rmsnorm`` (FlashInfer's
        kMoEFinalizeARResidualRMSNorm) sums, reduces and norms in one kernel. Past ``FOLD_MAX_TOKENS`` the MoE
        finalizes and the fused all-reduce + RMSNorm reduces, as without the fold."""
        norm, kernel = self.routed_expert_norm, self._experts_kernel
        if logits.shape[0] > FOLD_MAX_TOKENS:
            return norm(kernel.from_logits(z, logits), tp)
        permuted, weights, idx = kernel.from_logits(z, logits, finalize=False)
        y = tp.allreduce_moe_finalize_rmsnorm(permuted, idx, weights, norm.weight, norm.variance_epsilon,
                                              norm._weight_bias)
        if y is None:
            raise RuntimeError("the routed finalize fold lost its workspace")
        return y

    def experts_after_load(self) -> None:
        """The experts' kernel (``QuantExperts.experts_after_load``), then the finalize fold on top of it and the
        front."""
        super().experts_after_load()
        self._setup_finalize_fold()
        self._setup_front()

    def _setup_front(self) -> None:
        """``latent_moe_front: flashinfer``: check FlashInfer's front serves this block (``latent_front.unavailable``)
        and build its programs; a block it can't serve refuses the load rather than run another front than the
        config names. Per rank, no collective."""
        self._front = False
        if self.latent_moe_front != "flashinfer":
            return
        why = latent_front.unavailable(
            self._in_weight, self.gate.n_routed_experts, self.routed_expert_down_proj.out_features,
            self.shared_experts.situ if self.shared_experts is not None else None, self.comm_group.world_size)
        if why is not None:
            raise RuntimeError(f"latent_moe_front: flashinfer can't serve {type(self).__name__}: {why} "
                               "(latent_moe_front: gemm runs the packed GEMM)")
        latent_front.warm(self._in_weight)
        self._front = True

    def _setup_finalize_fold(self) -> None:
        """Fold the routed finalize into the routed all-reduce + RMSNorm where every rank's experts kernel routes
        from the logits (``from_logits`` can stop before its finalize), the TRT-LLM workspace builds and the probe
        agrees. Collective: every rank runs it for every block in the same order, whatever its own kernel; the
        first block decides for the rest (its workspace and probe), each block's ranks still agree on routing."""
        self._fold_finalize = False
        tp, norm = self.comm_group, self.routed_expert_norm
        if (tp.world_size == 1 or norm is None or self.routed_experts_finalize != "fused"
                or self.experts_packed is None):
            return
        if not tp.agree([int(self._kernel_routes())])[0]:
            return
        latent = self.routed_expert_down_proj.out_features
        key = (id(tp), latent, norm.weight.dtype)
        if key not in _FOLD_VERDICT:
            ok, detail = False, "no TRT-LLM all-reduce fusion workspace"
            if tp.enable_moe_finalize_fusion(latent, norm.weight.dtype, FOLD_MAX_TOKENS):
                tuned = self._tune_unfinalized()
                ok, detail = self._probe_fold()
                detail = f"{detail}; {tuned}" if ok and tuned else detail
            _FOLD_VERDICT[key] = (ok, detail)
            logger.info("%s routed finalize %s (%s)", type(self).__name__,
                        "folded into the routed all-reduce + RMSNorm" if ok else "in the MoE", detail)
        self._fold_finalize = _FOLD_VERDICT[key][0]

    def _tune_unfinalized(self) -> str:
        """Boot-tune ``from_logits(finalize=False)`` (its own entry in FlashInfer's tuner) at the decode buckets
        the fold serves: ``routed_experts_tune_tokens`` (default ``TUNE_TOKENS``) up to ``FOLD_MAX_TOKENS``, as
        ``experts_after_load`` tuned the finalized call. '' without boot tuning."""
        from mstar.utils.fused_moe import fp4_flashinfer

        tokens = [t for t in (self.routed_experts_tune_tokens or fp4_flashinfer.TUNE_TOKENS) if t <= FOLD_MAX_TOKENS]
        if not self.routed_experts_autotune or not tokens:
            return ""
        return "unfinalized call " + fp4_flashinfer.autotune(
            self, self._experts_kernel, self.routed_expert_norm.weight.device, tokens, finalize=False)

    @torch.no_grad()
    def _probe_fold(self) -> tuple[bool, str]:
        """``(ok, detail)``: the fold against the MoE's own finalize + the fused all-reduce + RMSNorm at 1, 32 and
        ``FOLD_MAX_TOKENS`` tokens (the same inputs on every rank), rel L2 below 2e-2 on every rank. Collective.

        It must not poison the context: a device fault is sticky, so no fallback can run after one. Each rank
        first runs its MoE both ways (local) and checks the unfinalized output against the fold kernel's
        contract, on the host (``moe_finalize_unfit``: shapes, dtypes, strides, workspace size) and on the
        device (every expanded index a permuted row or -1; one sync, at load). Then the ranks vote, and either
        all launch the collectives or none does (a rank staying out would leave the rest spinning in the
        one-shot exchange)."""
        tp, norm, kernel = self.comm_group, self.routed_expert_norm, self._experts_kernel
        device = norm.weight.device
        g = torch.Generator(device=device).manual_seed(0)
        worst = 0.0
        for m in (1, 32, FOLD_MAX_TOKENS):
            why = None
            try:
                z = torch.randn(m, self.routed_expert_down_proj.out_features, generator=g, device=device)
                z = z.to(norm.weight.dtype)
                logits = 2 * torch.randn(m, self.gate.n_routed_experts, generator=g, device=device)
                finalized = kernel.from_logits(z, logits)
                permuted, weights, idx = kernel.from_logits(z, logits, finalize=False)
                why = tp.moe_finalize_unfit(permuted, idx, weights, norm.weight)
                if why is None and bool(((idx < -1) | (idx >= permuted.shape[0])).any()):
                    why = f"an expanded index outside the {permuted.shape[0]} permuted rows"
            except Exception as exc:  # noqa: BLE001 - this rank says no; the vote keeps the ranks together
                _raise_if_device_fault(exc)
                why = f"{type(exc).__name__}: {str(exc)[:200]}"
            if not tp.agree([int(why is None)])[0]:
                return False, f"not launched at {m} tokens: {why or 'another rank'}"
            want = norm(finalized, tp).float()
            got = tp.allreduce_moe_finalize_rmsnorm(permuted, idx, weights, norm.weight, norm.variance_epsilon,
                                                    norm._weight_bias)
            try:
                err = (float("inf") if not bool(torch.isfinite(got).all())
                       else ((got.float() - want).norm() / want.norm().clamp_min(1e-12)).item())
            except Exception as exc:  # noqa: BLE001
                _raise_if_device_fault(exc)
                raise
            worst = max(worst, err)
        ok = bool(tp.agree([int(worst < 2e-2)])[0])
        return ok, (f"probe rel L2 <= {worst:.3g} at 1/32/{FOLD_MAX_TOKENS} tokens" if ok
                    else f"probe failed on some rank (here: rel L2 {worst:.3g})")

    @torch.compiler.disable
    def _routed_experts(self, z: torch.Tensor, routing_weights: torch.Tensor | None,
                        selected_experts: torch.Tensor | None, reduce_results: bool = True,
                        token_valid: torch.Tensor | None = None,
                        logits: torch.Tensor | None = None) -> torch.Tensor:
        """The routed experts (and their all-reduce, with ``reduce_results``), eager inside the
        compiled MoE region (the submodule's ``compile_regions``): their kernels are already fused
        (Marlin, the W4A16 or the fused bf16 MoE), and the bf16 path's tuned-config lookup takes
        Python ints Dynamo cannot trace symbolically (H200 job 1624). With ``logits`` the kernel
        routes them itself (``_kernel_routes``): no top-k, gather or renormalize launches here, and no
        padding mask (the kernel picks every row's experts; ``token_valid`` masks only ids the block chose)."""
        if logits is None:
            return self.experts_forward(z, routing_weights, selected_experts, reduce_results=reduce_results,
                                        token_valid=token_valid)
        out = self._experts_kernel.from_logits(z, logits)
        if reduce_results and self.comm_group.world_size > 1:
            out = self.comm_group.all_reduce(out)
        return out


class KimiLatentMoE(_LatentMoE, ParallelSparseMoeBlock):
    """Routed experts sharded over the TP group by intermediate dim (3072 / 8 = 384 at TP8)."""


class KimiLatentMoEExpertParallel(_LatentMoE, ExpertParallelSparseMoeBlock):
    """Routed experts as whole experts per rank (896 / 8 = 112 at EP8)."""


def build_mlp(config: KimiK3ModelConfig, layer_idx: int, comm_group: CommGroup | None = None):
    """Layer ``layer_idx``'s MLP: the dense SiTU MLP below ``first_k_dense_replace``, else the
    latent MoE."""
    if not config.is_moe_layer(layer_idx):
        return KimiSiTUMLP(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
            comm_group=comm_group, situ=config.situ())
    cls = KimiLatentMoEExpertParallel if config.moe_expert_parallel else KimiLatentMoE
    return cls(config, comm_group=comm_group)
