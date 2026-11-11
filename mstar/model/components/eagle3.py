"""EAGLE-3 (``model_kwargs.draft: {method: eagle3, model: RedHatAI/Qwen3-8B-speculator.eagle3}``): one decoder
layer that reads taps of three target layers and chains ``k`` one-token passes (``chain_drafts``), drafting over a
reduced vocabulary that a token map takes back to the target's.

A row at position ``p`` reads the token at ``p + 1`` (its embedding) and a hidden vector: ``fc`` of the target's taps
at ``p`` on a prompt step and over a verify block (`observe`), or the layer's own output at ``p - 1`` in a chained
pass. Its output, before the final norm, is the carry across steps. Its K/V is one more layer of the target's set,
and it attends through the target's attention, so it needs no lag. The layer's attention input is the token's
embedding and the hidden vector side by side (``2 * hidden`` wide), each RMS-normed first.

Two checkpoint formats are read: the ``speculators`` library's (``speculators_model_type: eagle3``, the layer in
``transformer_layer_config``) and the Llama-style one of the EAGLE authors (``draft_vocab_size`` beside a Llama config,
weights under ``midlayer.``). Variants that no reference pair uses are refused: more than one layer,
``norm_before_fc``, ``fc_norm``, ``norm_output`` and parallel drafting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.draft.drafter import (
    CarrySpec,
    DrafterNeeds,
    DraftInputs,
    ModelGeometry,
    ObserveInputs,
    PrefillInputs,
)
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep
from mstar.engine.resources.taps.config import TapSpec, TapStep
from mstar.model.components.chained_draft import at_last_rows, chain_drafts, kept_rows
from mstar.model.components.hosted_drafter import HostedDrafter, HostedDrafterDecl

logger = logging.getLogger(__name__)

LAYER_SET = "eagle3"
LABEL = "main"
CARRY_DTYPE = torch.float32
# a variant no supported checkpoint uses: refused rather than drafted wrong
_REFUSED_FLAGS = ("norm_before_fc", "fc_norm", "norm_output", "parallel_drafting", "use_parallel_drafting")


@dataclass(frozen=True)
class Eagle3Config:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    # the target's vocabulary (its embedding's rows) and the reduced one it drafts in
    vocab_size: int
    draft_vocab_size: int
    # the target's residual width, which fc reads three of
    target_hidden_size: int
    norm_before_residual: bool
    # target layers whose input it reads; None: the checkpoint names none (the target's default applies)
    taps: tuple[int, ...] | None
    # the k it was trained to chain, when the checkpoint says
    trained_depth: int | None
    # the Llama-style format names the layer's weights `midlayer.`
    layer_prefix: str = "layers.0."

    @classmethod
    def from_hf(cls, hf: dict) -> Eagle3Config:
        """The config of either format; ``ValueError`` names every refused variant."""
        if hf.get("speculators_model_type") == "eagle3":
            layer = dict(hf["transformer_layer_config"])
            methods = (hf.get("speculators_config") or {}).get("proposal_methods") or []
            depths = [m["speculative_tokens"] for m in methods if "speculative_tokens" in m]
            depth = max(depths) if depths else None
            taps = hf.get("eagle_aux_hidden_state_layer_ids")
            prefix, flags = "layers.0.", {**layer, **hf}
            # the speculators library's reference default
            residual_normed = hf.get("norm_before_residual", True)
        elif "draft_vocab_size" in hf:
            layer = dict(hf)
            depth = None
            taps = (hf.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids")
            prefix, flags = "midlayer.", hf
            residual_normed = hf.get("norm_before_residual", False)
        else:
            raise ValueError("not an EAGLE-3 checkpoint: no speculators_model_type eagle3 and no draft_vocab_size")
        problems = []
        if layer.get("num_hidden_layers", 1) != 1:
            problems.append(f"{layer['num_hidden_layers']} decoder layers (one is supported)")
        problems += [f"{flag}" for flag in _REFUSED_FLAGS if flags.get(flag)]
        if layer.get("attention_bias") or layer.get("mlp_bias"):
            problems.append("attention or MLP biases")
        if problems:
            raise ValueError("unsupported EAGLE-3 variant: " + ", ".join(problems))
        hidden = layer["hidden_size"]
        return cls(
            hidden_size=hidden, num_attention_heads=layer["num_attention_heads"],
            num_key_value_heads=layer["num_key_value_heads"],
            head_dim=layer.get("head_dim") or hidden // layer["num_attention_heads"],
            intermediate_size=layer["intermediate_size"], rms_norm_eps=layer.get("rms_norm_eps", 1e-6),
            rope_theta=float(layer.get("rope_theta", 10_000.0)), vocab_size=layer["vocab_size"],
            draft_vocab_size=hf.get("draft_vocab_size") or layer["vocab_size"],
            target_hidden_size=hf.get("target_hidden_size") or hidden,
            norm_before_residual=bool(residual_normed),
            taps=None if taps is None else tuple(int(t) for t in taps), trained_depth=depth, layer_prefix=prefix,
        )


def default_taps(target_layers: int) -> tuple[int, ...]:
    """The layers EAGLE-3 reads when its checkpoint names none: an early, the middle and a late one (the inputs of
    layers 2, L/2 and L-3, as its reference training and vLLM take them)."""
    return (2, target_layers // 2, target_layers - 3)


@dataclass(frozen=True)
class Eagle3(HostedDrafterDecl):
    """The EAGLE-3 drafter as its node declares it, from config alone."""

    config: Eagle3Config
    num_draft_tokens: int
    dtype: str
    taps: tuple[int, ...]
    # the target's resources it writes, attends and rotates through
    kv_key: str
    pos_key: str
    attn_key: str
    tap_key: str = "eagle3_taps"
    checkpoint: str | None = None
    cache_dir: str | None = None

    def __post_init__(self):
        depth = self.config.trained_depth
        if depth is not None and self.num_draft_tokens > depth:
            logger.warning("EAGLE-3 drafts %d tokens a step, past the %d its checkpoint was trained to chain: "
                           "acceptance past that depth is untested", self.num_draft_tokens, depth)

    @property
    def needs(self) -> DrafterNeeds:
        cfg = self.config
        geometry = ModelGeometry(
            num_layers=1, num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
            hidden_size=cfg.target_hidden_size, vocab_size=cfg.vocab_size, dtype=self.dtype,
            # it attends through the target's attention and rotates through its positions
            num_qo_heads=cfg.num_attention_heads, rope_theta=cfg.rope_theta,
        )
        return DrafterNeeds(
            taps=self.taps, tap_key=self.tap_key,
            kv=geometry.layer_set(self.kv_key, LAYER_SET),
            carry=CarrySpec(cfg.hidden_size, CARRY_DTYPE),
            # no chunk_boundary (UNDECLARED): the prefill reads the token sampled at each prompt's last row, so a
            # chunk's last row would need the next prompt token, which no input carries; the startup gate refuses
            # a chunking target
            geometry=geometry,
        )

    def node_resources(self, nodes: set[str]) -> list[NodeResourceSpec]:
        return [TapSpec(resource_key=self.tap_key, nodes=set(nodes), layers=self.taps,
                        hidden_size=self.config.target_hidden_size)]

    def prefill_steps(self) -> dict[str, ResourceStep]:
        return {self.tap_key: TapStep()}

    def verify_steps(self) -> dict[str, ResourceStep]:
        return {self.tap_key: TapStep()}

    def build(self, *, device, comm_group=None, dtype: torch.dtype | None = None, target: nn.Module | None = None):
        """The module with the checkpoint's weights (none: deterministic random ones, for tests). A checkpoint
        without an embedding borrows ``target``'s."""
        with torch.device("meta"):
            module = Eagle3Drafter(self, comm_group=comm_group)
        if dtype is not None:
            module = module.to(dtype)
        module.to_empty(device=device)
        # offsets from each draft id to its target id: none (the identity) until a checkpoint's are loaded
        module.d2t = torch.zeros(self.config.draft_vocab_size, dtype=torch.long, device=device)
        if self.checkpoint is None:
            gen = torch.Generator(device="cpu").manual_seed(0)
            for param in module.parameters():
                param.data.copy_((torch.randn(param.shape, generator=gen) * 0.02).to(param.dtype))
        else:
            from mstar.model.loader.iterators import iter_safetensors_shards
            from mstar.utils.hf_snapshot import resolve_snapshot_dir

            directory = resolve_snapshot_dir(
                self.checkpoint, cache_dir=self.cache_dir, allow_patterns=["*.json", "*.safetensors"],
            )
            module.load_weights(iter_safetensors_shards(directory, device=device), target=target)
        module.bind_token_map()
        return module.requires_grad_(False).eval()


class Eagle3Drafter(nn.Module, HostedDrafter):
    """The drafter a verifying node calls."""

    def __init__(self, decl: Eagle3, comm_group=None):
        super().__init__()
        from mstar.model.components.distributed.attention import ParallelAttention
        from mstar.model.components.distributed.embedding import VocabParallelEmbedding
        from mstar.model.components.distributed.mlp import ParallelGatedMLP
        from mstar.model.components.norm import RMSNorm

        cfg = decl.config
        self.decl = decl
        self.needs = decl.needs
        hidden = cfg.hidden_size
        # sharded by vocabulary as the target's is, so one borrowed from the target is this rank's shard of it
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=cfg.vocab_size, embedding_dim=hidden, comm_group=comm_group or CommGroup.trivial(),
        )
        self.fc = nn.Linear(len(decl.taps) * cfg.target_hidden_size, hidden, bias=False)
        self.input_layernorm = RMSNorm(hidden, eps=cfg.rms_norm_eps)
        self.hidden_norm = RMSNorm(hidden, eps=cfg.rms_norm_eps)
        self.self_attn = ParallelAttention(
            comm_group=comm_group, hidden_size=cfg.num_attention_heads * cfg.head_dim,
            num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
            rms_norm_eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta, input_hidden_size=2 * hidden,
            output_hidden_size=hidden,
            attn_key=decl.attn_key, kv_key=decl.kv_key, pos_key=decl.pos_key,
            # its one layer, the set's layer 0, which the pool places
            kv_layer_set=LAYER_SET,
        )
        self.post_attention_layernorm = RMSNorm(hidden, eps=cfg.rms_norm_eps)
        self.mlp = ParallelGatedMLP(comm_group=comm_group, hidden_size=hidden,
                                    intermediate_size=cfg.intermediate_size, activation="silu")
        self.norm = RMSNorm(hidden, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(hidden, cfg.draft_vocab_size, bias=False)
        self.register_buffer("d2t", torch.zeros(cfg.draft_vocab_size, dtype=torch.long), persistent=False)
        self.token_map = None
        self.taps = None

    def load_weights(self, weights, target: nn.Module | None = None) -> None:
        """The checkpoint's tensors (either format); ``t2d`` is derived, an embedding left out is the target's."""
        from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights

        prefix = self.decl.config.layer_prefix
        renamed = []
        for name, tensor in weights:
            if name == "d2t":
                self.d2t = tensor.long().to(self.d2t.device)
                continue
            if name == "t2d":
                continue
            renamed.append((name.removeprefix(prefix) if name.startswith(prefix) else name, tensor))
        loaded = load_hf_weights(self, renamed, stacked_params=LLAMA_STACKED_PARAMS)
        if "embed_tokens.weight" not in loaded:
            if target is None:
                raise RuntimeError("the EAGLE-3 checkpoint ships no embedding and no target lends one")
            # shared by reference, as DSpark shares the target's: both shard the vocabulary over the same ranks,
            # so the target's local shard is this rank's, and no second table comes out of KV capacity
            ours, theirs = self.embed_tokens.weight.shape, target.embed_tokens.weight.shape
            if ours != theirs:
                raise ValueError(f"the EAGLE-3 checkpoint's embedding is {tuple(ours)}, the target's {tuple(theirs)}")
            self.embed_tokens = target.embed_tokens
            loaded.add("embed_tokens.weight")
        missing = sorted(set(dict(self.named_parameters())) - loaded)
        if missing:
            raise RuntimeError(f"EAGLE-3 checkpoint left {len(missing)} parameter(s) unloaded, e.g. {missing[:5]}")

    def bind_token_map(self) -> None:
        from mstar.engine.resources.draft.token_map import TokenMap

        self.token_map = TokenMap.from_offsets(self.d2t, self.decl.config.vocab_size).to(self.d2t.device)

    def bind_resources(self, resources: dict) -> None:
        self.taps = resources.get(self.needs.tap_key)

    # ── the layer ──

    def _head(self, hidden: torch.Tensor, next_ids: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """The layer over every row, its K/V written for each: ``hidden`` (fc of the taps, or the last output) and
        the next token's embedding. Its output, before the final norm, at the rows ``keep`` indexes."""
        normed = self.hidden_norm(hidden)
        residual = normed if self.decl.config.norm_before_residual else hidden
        embeds = self.input_layernorm(self.embed_tokens(next_ids).to(hidden.dtype))
        attend = self.self_attn.attend
        attend.bind_step(LABEL)
        _set_layer(attend, 0)
        h = self.self_attn(torch.cat((embeds, normed), dim=-1))
        if keep is not None:
            h, residual = h.index_select(0, keep), residual.index_select(0, keep)
        h = residual + h
        return h + self.mlp(self.post_attention_layernorm(h))

    def _draft(self, out: torch.Tensor) -> torch.Tensor:
        """Each row's draft in the target's vocabulary."""
        return self.token_map.to_target(self.lm_head(self.norm(out)).argmax(-1))

    def _fused_taps(self) -> torch.Tensor:
        taps = self.taps.read()
        return self.fc(taps.reshape(taps.shape[0], -1).to(self.fc.weight.dtype))

    # ── prompt step ──

    def prefill(self, inputs: PrefillInputs) -> torch.Tensor:
        """The layer over a whole prompt's rows: each row reads the next prompt token, the last the token the step
        sampled. Returns each prompt's carry: the output at its last row."""
        input_ids, sampled = inputs.input_ids, inputs.sampled
        tokens = input_ids.shape[0]
        last, at = inputs.resources[self.decl.attn_key].prompt_last_rows(tokens, input_ids.device, LABEL)
        # each row's next prompt token; a prompt's last row the token sampled from it (a padding row's
        # writes go to the sink at ``tokens``)
        fill = at_last_rows(sampled.reshape(-1).to(input_ids.dtype), at, tokens)
        is_last = torch.zeros(tokens + 1, dtype=torch.bool, device=input_ids.device).index_fill(0, at, True)[:tokens]
        following = torch.where(is_last, fill, input_ids.roll(-1, 0))
        out = self._head(self._fused_taps(), following, keep=last)
        return out.to(CARRY_DTYPE)

    # ── verify step ──

    def propose(self, inputs: DraftInputs) -> torch.Tensor:
        return chain_drafts(inputs.carry.to(self.fc.weight.dtype), inputs.num_draft_tokens, self._head, self._draft)

    def observe(self, inputs: ObserveInputs) -> torch.Tensor:
        """The layer's K/V over the block from the target's taps there, and the new carry: its output at each row's
        last kept position."""
        result = inputs.result
        out = self._head(self._fused_taps(), result.tokens.reshape(-1), keep=kept_rows(inputs.block, result.accepted))
        return out.to(CARRY_DTYPE)


@torch.compiler.disable
def _set_layer(attend, idx: int) -> None:
    attend.set_layer_idx(idx)
