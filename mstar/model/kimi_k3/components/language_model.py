r"""The Kimi K3 text backbone for M*: embedding, decoder layers with attention residuals,
output mixing, final norm, LM head, and checkpoint loading.

Two execution paths share every parameter:

* ``forward(hidden, label)``: the paged path used in serving. The KDA layers read the
  recurrent-state resource and the MLA layers the paged latent cache through the
  cursors the model loop sets (one KDA slot index / one MLA cache layer per layer).
* ``forward_dense(input_ids, state)``: explicit per-request state for one sequence,
  used by tests and as an eager reference (no resources needed).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ColumnParallelLinear, VocabParallelEmbedding
from mstar.model.kimi_k3.components.attn_res import AttnResRead
from mstar.model.kimi_k3.components.common import KimiRMSNorm
from mstar.model.kimi_k3.components.decoder_layer import KimiK3DecoderLayer
from mstar.model.kimi_k3.components.kda import KDA_STACKED_PARAMS, ParallelKDAAttention
from mstar.model.kimi_k3.components.mla import ParallelMLAAttention
from mstar.model.kimi_k3.components.mlp import ParallelSiTUMLP
from mstar.model.kimi_k3.components.moe import KimiLatentMoE
from mstar.model.kimi_k3.config import KDA_STATE, MLA_ATTN, MLA_KV, KimiK3TextConfig
from mstar.model.kimi_k3.reference.mxfp4 import dequant_mxfp4

logger = logging.getLogger(__name__)

_EXPERT_RE = re.compile(r"^(.*\.block_sparse_moe)\.experts\.(\d+)\.(w1|w2|w3)\.(weight|weight_packed|weight_scale)$")
_SHARD_OF = {"w1": "gate", "w3": "up", "w2": "down"}


def _build_layer(
    cfg: KimiK3TextConfig, i: int, comm_group: CommGroup | None, quantized_experts: bool = False,
) -> KimiK3DecoderLayer:
    if cfg.is_kda_layer(i):
        attn = ParallelKDAAttention(
            hidden_size=cfg.hidden_size, num_heads=cfg.kda_num_heads, head_dim=cfg.kda_head_dim,
            conv_kernel_size=cfg.kda_conv_kernel_size, gate_lower_bound=cfg.kda_gate_lower_bound,
            norm_eps=cfg.rms_norm_eps, comm_group=comm_group, state_key=KDA_STATE,
        )
    else:
        attn = ParallelMLAAttention(
            hidden_size=cfg.hidden_size, num_heads=cfg.num_attention_heads, q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank, qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim, v_head_dim=cfg.v_head_dim,
            use_output_gate=cfg.mla_use_output_gate, norm_eps=cfg.mla_norm_eps,
            comm_group=comm_group, attn_key=MLA_ATTN, kv_key=MLA_KV,
        )
    if cfg.is_moe_layer(i):
        mlp = KimiLatentMoE(
            hidden_size=cfg.hidden_size, latent_size=cfg.routed_expert_hidden_size or cfg.hidden_size,
            num_experts=cfg.num_experts, top_k=cfg.num_experts_per_token,
            moe_intermediate_size=cfg.moe_intermediate_size, num_shared_experts=cfg.num_shared_experts,
            situ_beta=cfg.activation_situ_beta, situ_linear_beta=cfg.activation_situ_linear_beta,
            latent_norm=cfg.latent_moe_use_norm and cfg.use_latent_moe, norm_eps=cfg.rms_norm_eps,
            renormalize=cfg.moe_renormalize, routed_scaling_factor=cfg.routed_scaling_factor,
            num_expert_group=cfg.num_expert_group, topk_group=cfg.topk_group, comm_group=comm_group,
            quantized=quantized_experts,
        )
    else:
        mlp = ParallelSiTUMLP(
            cfg.hidden_size, cfg.intermediate_size, comm_group=comm_group,
            situ_beta=cfg.activation_situ_beta, situ_linear_beta=cfg.activation_situ_linear_beta,
        )
    return KimiK3DecoderLayer(
        layer_idx=i, hidden_size=cfg.hidden_size, self_attn=attn, mlp=mlp,
        is_kda=cfg.is_kda_layer(i), is_moe=cfg.is_moe_layer(i),
        attn_res_block_size=cfg.attn_res_block_size, norm_eps=cfg.rms_norm_eps,
    )


class KimiK3LanguageModel(nn.Module):
    def __init__(self, cfg: KimiK3TextConfig, comm_group: CommGroup | None = None, quantized_experts: bool = False):
        super().__init__()
        self.cfg = cfg
        self.quantized_experts = quantized_experts
        self.embed_tokens = VocabParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size, comm_group=comm_group, padding_idx=cfg.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [_build_layer(cfg, i, comm_group, quantized_experts) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = KimiRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if cfg.use_attn_res:
            self.output_attn_res = AttnResRead(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self._kda_slots = {i: cfg.kda_layer_slot(i) for i in cfg.kda_layers}
        self._mla_slots = {i: cfg.mla_layer_slot(i) for i in cfg.full_attn_layers}

    # ---------------------------------------------------------------- cursors
    def _set_cursors(self, layer: KimiK3DecoderLayer) -> None:
        i = layer.layer_idx
        if layer.is_kda:
            layer.self_attn.state.set_default_layer_idx(self._kda_slots[i])
        else:
            layer.self_attn.kv.set_default_layer_idx(self._mla_slots[i])
            layer.self_attn.attn.set_default_layer_idx(self._mla_slots[i])

    def bind_label(self, label: str) -> None:
        for layer in self.layers:
            if not layer.is_kda:
                layer.self_attn.kv.set_default_label(label)
                layer.self_attn.attn.set_default_label(label)
                break  # one shared resource: setting it once is enough
        for layer in self.layers:
            if layer.is_kda:
                layer.self_attn.state.set_default_label(label)
                break

    # ---------------------------------------------------------------- paths
    def _finish(self, prefix: torch.Tensor, blocks: torch.Tensor, pending: torch.Tensor | None) -> torch.Tensor:
        """Output read (with the last layer's residual add and the final norm folded in) or,
        without attention residuals, the final norm of the completed prefix."""
        if self.cfg.use_attn_res:
            return self.output_attn_res.read(prefix, blocks, out_norm=self.norm, add=pending)[0]
        return self.norm(prefix if pending is None else prefix + pending)

    def forward(self, hidden: torch.Tensor, *, label: str = "main") -> torch.Tensor:
        """Paged path over packed tokens ``hidden [T, H]``; returns the final-normed
        hidden states ``[T, H]``."""
        self.bind_label(label)
        prefix, pending = hidden, None
        blocks = hidden.new_zeros(hidden.shape[0], 0, hidden.shape[1])
        for layer in self.layers:
            self._set_cursors(layer)
            prefix, blocks, pending = layer(prefix, blocks, pending)
        return self._finish(prefix, blocks, pending)

    def forward_dense(
        self, input_ids: torch.Tensor, state: list | None = None,
    ) -> tuple[torch.Tensor, list]:
        """One sequence with explicit state (list with one entry per layer)."""
        state = state or [None] * len(self.layers)
        new_state: list = [None] * len(self.layers)
        hidden = self.embed_tokens(input_ids)
        prefix, pending = hidden, None
        blocks = hidden.new_zeros(hidden.shape[0], 0, hidden.shape[1])
        for i, layer in enumerate(self.layers):
            prefix, blocks, new_state[i], pending = layer.forward_dense(prefix, blocks, state[i], pending)
        return self._finish(prefix, blocks, pending), new_state


def select_kda_kernels(model: nn.Module, device) -> bool:
    """Install the best KDA kernels for ``device`` on every KDA layer; returns whether
    the decode path may be captured in a CUDA graph."""
    from mstar.model.kimi_k3.components.kda_kernels import default_kernels

    layers = [mod for mod in model.modules() if isinstance(mod, ParallelKDAAttention)]
    if not layers:
        return True
    # FlashKDA is specialised for head_dim 128 with a bounded gate; other shapes use fla
    fits = all(m.head_dim == 128 and m.gate_lower_bound is not None for m in layers)
    kernels = default_kernels(device, prefer="flashkda" if fits else "fla")
    for mod in layers:
        mod.set_kernels(kernels)
    return bool(getattr(kernels, "cuda_graph_safe", False))


MOE_BACKENDS = ("auto", "triton", "w4a16", "humming", "marlin")


def _retry(fn, attempts: int = 3, delay_s: float = 3.0):
    """Call ``fn()`` up to ``attempts`` times, sleeping ``delay_s`` between failures."""
    import time

    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(delay_s)


def _all_ranks_agree(ok: bool, comm_group) -> bool:
    """``ok`` AND-ed across the tensor-parallel group (a single rank or no group: ``ok`` itself)."""
    if comm_group is None or comm_group.world_size == 1 or not torch.cuda.is_available():
        return ok
    import torch.distributed as dist

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=comm_group.device_group)
    return bool(flag.item())


def marlin_supports(moe) -> bool:
    """Marlin's tiles need the latent width a multiple of 256 and the per-rank expert
    intermediate width a multiple of 128 (the released K3 shapes at TP1..8 all qualify)."""
    return moe.latent_size % 256 == 0 and moe.inter_local % 128 == 0


def prepare_moe_kernels(model: nn.Module, device, backend: str = "auto") -> str:
    """Pick the routed-expert kernels for MXFP4 experts on CUDA: ``marlin`` (the Marlin MoE
    kernel, fastest at decode batch sizes), FlashInfer's SM90 CUTLASS MoE (``w4a16`` bf16
    activations, ``humming`` FP8 activations), or the in-tree Triton kernel (``triton``, also
    the reference precision). ``auto`` takes ``marlin`` when its extension builds and the shapes
    fit, else ``w4a16``, else ``triton``. Returns the backend used."""
    from mstar.model.kimi_k3.components.moe import KimiLatentMoE

    assert backend in MOE_BACKENDS, backend
    dev = torch.device(device)
    moes = [m for m in model.modules() if isinstance(m, KimiLatentMoE)]
    if backend == "triton" or dev.type != "cuda" or not moes or not moes[0].quantized:
        return "triton"
    candidates = ["marlin", "w4a16"] if backend == "auto" else [backend]
    for mode in candidates:
        try:
            if mode == "marlin":
                if not all(marlin_supports(m) for m in moes):
                    raise RuntimeError("Marlin needs latent % 256 == 0 and inter/TP % 128 == 0")
                from mstar.utils.fused_moe.marlin import _load_ops

                # JIT-builds the extension once per machine; raises if it cannot. Retried because
                # concurrent loads (other ranks, other nodes sharing the cache dir) can trip on the
                # build lock for a moment.
                _retry(_load_ops, attempts=3, delay_s=3.0)
            else:
                import flashinfer.fused_moe  # noqa: F401
            ok = True
        except Exception as ex:
            if backend != "auto":
                raise
            logger.warning("Kimi K3 MoE backend %s unavailable (%s); trying the next", mode, ex)
            ok = False
        # every tensor-parallel rank must run the same kernels: one rank on a slower backend
        # paces the whole group and has a different memory footprint (an OOM on one GPU, 2026-09-15)
        if not _all_ranks_agree(ok, moes[0].comm_group if moes else None):
            if ok:
                logger.warning("Kimi K3 MoE backend %s unavailable on another rank; trying the next", mode)
            continue
        for m in moes:
            m.prepare_experts_backend(mode, dev)
        if mode != "marlin":
            # one layer's shapes stand for all: tune the CUTLASS tactics per decode bucket
            moes[0]._backend.autotune(top_k=moes[0].top_k)
        return mode
    return "triton"


class KimiK3ForCausalLM(nn.Module):
    def __init__(self, cfg: KimiK3TextConfig, comm_group: CommGroup | None = None, quantized_experts: bool = False):
        super().__init__()
        self.cfg = cfg
        self.quantized_experts = quantized_experts
        self.model = KimiK3LanguageModel(cfg, comm_group=comm_group, quantized_experts=quantized_experts)
        self.lm_head = ColumnParallelLinear(
            comm_group=comm_group or CommGroup.trivial(), input_size=cfg.hidden_size,
            output_size=cfg.vocab_size, bias=False, gather_output=True,
        )

    def forward_dense(self, input_ids: torch.Tensor, state: list | None = None) -> tuple[torch.Tensor, list]:
        hidden, state = self.model.forward_dense(input_ids, state)
        return self.lm_head(hidden), state

    @torch.no_grad()
    def generate_dense(self, prompt_ids: torch.Tensor, max_new_tokens: int, eos_id: int | None = None) -> list[int]:
        logits, state = self.forward_dense(prompt_ids)
        out: list[int] = []
        nxt = int(logits[-1].argmax())
        for _ in range(max_new_tokens):
            out.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
            logits, state = self.forward_dense(torch.tensor([nxt], device=prompt_ids.device), state)
            nxt = int(logits[-1].argmax())
        return out

    # ---------------------------------------------------------------- loading
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream ``(name, tensor)`` pairs from the checkpoint into the parameters.

        Handles the ``language_model.`` prefix, drops the vision tower and projector,
        routes per-expert ``w1/w2/w3`` into the fused expert parameters (dequantizing
        MXFP4 ``weight_packed``/``weight_scale`` pairs on the fly), and applies the
        stacked-shard rules for ``qkv_proj`` (KDA) and ``gate_up_proj`` (MLPs).
        """
        from mstar.model.loader.base import load_weights_into

        params = dict(self.named_parameters())
        loaded: set[str] = set()
        pending_packed: dict[str, torch.Tensor] = {}

        def route_expert(prefix: str, expert: int, w: str, tensor: torch.Tensor, kind: str = "weight") -> None:
            base = "down" if w == "w2" else "gate_up"
            if kind == "weight":
                target = f"{prefix}.experts.{base}_proj"
            else:
                target = f"{prefix}.experts.{base}_{'packed' if kind == 'weight_packed' else 'scale'}"
            param = params[target]
            param.weight_loader(param, tensor, f"{_SHARD_OF[w]}:{expert}")
            loaded.add(target)

        def plain_stream():
            """Route expert tensors as they stream (side effect) and yield the rest to the
            generic loader *immediately*: a full-size tensor must never outlive its own
            dispatch, or a rank accumulates the unsharded checkpoint on its GPU."""
            for name, tensor in weights:
                if name.startswith("language_model."):
                    name = name[len("language_model."):]
                if name.startswith(("vision_tower.", "mm_projector.")):
                    continue
                m = _EXPERT_RE.match(name)
                if m is None:
                    yield name, tensor
                    continue
                prefix, expert, w, kind = m.group(1), int(m.group(2)), m.group(3), m.group(4)
                if kind == "weight" or self.quantized_experts:
                    route_expert(prefix, expert, w, tensor, kind)
                    continue
                key = f"{prefix}.experts.{expert}.{w}"
                other = pending_packed.pop(key, None)
                if other is None:
                    pending_packed[key] = tensor
                    continue
                packed, scale = (tensor, other) if kind == "weight_packed" else (other, tensor)
                route_expert(prefix, expert, w, dequant_mxfp4(packed, scale, dtype=params[
                    f"{prefix}.experts.down_proj"].dtype))

        stacked = [
            *[_Rule(t, s, i) for t, s, i in KDA_STACKED_PARAMS],
            _Rule(".gate_up_proj", ".gate_proj", 0),
            _Rule(".gate_up_proj", ".up_proj", 1),
        ]
        loaded |= load_weights_into(self, plain_stream(), stacked_params=stacked, name_remapper=_remap_name)
        assert not pending_packed, f"unpaired MXFP4 tensors: {sorted(pending_packed)[:3]}"
        missing = sorted(set(params) - loaded)
        if missing:
            logger.warning("Kimi K3: %d parameters not loaded, e.g. %s", len(missing), missing[:5])
        return loaded


def _remap_name(name: str) -> str | None:
    # attention-residual parameters live under <prefix>_res_norm / <prefix>_res_proj in the
    # checkpoint and under <prefix>_res.norm / <prefix>_res.proj here
    name = name.replace("self_attention_res_norm.", "self_attention_res.norm.")
    name = name.replace("self_attention_res_proj.", "self_attention_res.proj.")
    name = name.replace("mlp_res_norm.", "mlp_res.norm.")
    name = name.replace("mlp_res_proj.", "mlp_res.proj.")
    name = name.replace("output_attn_res_norm.", "output_attn_res.norm.")
    name = name.replace("output_attn_res_proj.", "output_attn_res.proj.")
    return name


class _Rule:
    """Minimal stand-in for ``StackedParamRule`` (positional fields)."""

    def __init__(self, target_suffix: str, source_suffix: str, shard_id):
        self.target_suffix = target_suffix
        self.source_suffix = source_suffix
        self.shard_id = shard_id
