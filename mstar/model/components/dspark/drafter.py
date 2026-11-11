"""A DSpark drafter inside the node that verifies its drafts, on the engine's
drafter interface (`mstar.engine.resources.draft.drafter`).

Its KV is extra layers of the target's KV pool (the target's layers, then the
drafter's), so it shares the target's pages, length, reservation, take-back,
eviction, offload and prefix pages: at each position the target holds, the
drafter holds the context K/V of that position (its layers' K/V projections of
``hidden_norm(fc(taps))``). It rotates at the target's positions. A verify step
of ``k`` drafts:

    propose   the block [anchor, mask x k] through the drafter's layers at the
              verify block's positions, its K/V written into the block's slots,
              every row attending to the context and to the block's first k
              rows (non-causal, the last row hidden); rows 0..k-1 give the base
              logits, and the Markov head drafts left to right
    target    the target over [anchor, d1 .. dk], which writes the taps
    observe   the context K/V of the whole block from the taps, over the same
              slots; the take-back drops the rows the verify rejected with
              the target's

A prompt's prefill writes its context K/V from the taps the target's prefill
wrote, so a cached prompt prefix carries the drafter's context too.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.draft.drafter import (
    ChunkBoundary,
    DrafterNeeds,
    DraftInputs,
    ModelGeometry,
    ObserveInputs,
)
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep
from mstar.engine.resources.taps.config import TapSpec, TapStep
from mstar.model.components.dspark.config import DSparkConfig
from mstar.model.components.hosted_drafter import HostedDrafter, HostedDrafterDecl
from mstar.model.components.norm import RMSNorm

# the KV label the drafter's layers write and attend under: the target's
LABEL = "main"
# the layer set its layers join the target's KV pool with
LAYER_SET = "dspark"


@dataclass(frozen=True)
class DSpark(HostedDrafterDecl):
    """A DSpark drafter as its node declares it: the resources, steps and needs
    it adds to the target's. Plain data, built from config alone; ``build``
    makes the module on a worker."""

    config: DSparkConfig
    num_draft_tokens: int
    # the target's resources it writes and rotates through
    kv_key: str
    pos_key: str
    # its own: an attention resource with the block's mask, and the taps
    attn_key: str = "dspark_attn"
    tap_key: str = "dspark_taps"
    # the checkpoint (a Hugging Face id or a path) and its cache, for ``build``
    checkpoint: str | None = None
    cache_dir: str | None = None

    def __post_init__(self):
        if not 1 <= self.num_draft_tokens <= self.config.block_size:
            raise ValueError(
                f"the DSpark drafter was trained on a block of {self.config.block_size}: "
                f"num_draft_tokens must be 1..{self.config.block_size}, got {self.num_draft_tokens}"
            )

    @property
    def needs(self) -> DrafterNeeds:
        cfg = self.config
        geometry = ModelGeometry(
            num_layers=cfg.num_layers, num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
            hidden_size=cfg.target_hidden_size, vocab_size=cfg.vocab_size, dtype=cfg.dtype,
            # its attention plans over the target's pool, and it rotates through the target's positions
            num_qo_heads=cfg.num_attention_heads, rope_theta=cfg.rope_theta,
        )
        return DrafterNeeds(
            taps=cfg.taps, tap_key=self.tap_key, kv=geometry.layer_set(self.kv_key, LAYER_SET),
            # its context K/V at p is a function of the taps at p alone
            chunk_boundary=ChunkBoundary.NONE,
            geometry=geometry,
        )

    def node_resources(self, nodes: set[str]) -> list[NodeResourceSpec]:
        return [
            AttentionSpec(resource_key=self.attn_key, nodes=set(nodes), config=AttentionConfig(kv_cache=self.kv_key)),
            TapSpec(
                resource_key=self.tap_key, nodes=set(nodes), layers=self.config.taps,
                # the target's residual stream as it runs (the node's compute dtype)
                hidden_size=self.config.target_hidden_size,
            ),
        ]

    def prefill_steps(self) -> dict[str, ResourceStep]:
        """A prompt step's share: the target writes taps, the drafter its context
        K/V through the target's KV and position steps."""
        return {self.tap_key: TapStep()}

    def verify_steps(self) -> dict[str, ResourceStep]:
        """A verify step's share: the taps, and the drafter's attention over its
        block, its rows attending each other both ways with the trailing row hidden."""
        return {
            self.tap_key: TapStep(),
            self.attn_key: AttentionStep(verify=True, causal=False, hide_last=1),
        }

    def build(self, *, device, comm_group=None, dtype: torch.dtype | None = None, target: nn.Module | None = None):
        """The module, sharded like the target, with the checkpoint's weights
        (none: deterministic random weights, for tests). ``target`` lends its
        embedding or head to a checkpoint that ships none."""
        with torch.device("meta"):
            module = DSparkDrafter(self, comm_group=comm_group)
        if dtype is not None:
            module = module.to(dtype)
        module.to_empty(device=device)
        module.reset_buffers()
        if self.checkpoint is None:
            module.reset_parameters()
        else:
            from mstar.model.loader.iterators import iter_safetensors_shards
            from mstar.utils.hf_snapshot import resolve_snapshot_dir

            directory = resolve_snapshot_dir(
                self.checkpoint, cache_dir=self.cache_dir, allow_patterns=["*.json", "*.safetensors"],
            )
            module.load_weights(iter_safetensors_shards(directory, device=device), target=target)
        return module.requires_grad_(False).eval()


class DSparkDrafter(nn.Module, HostedDrafter):
    """The drafter (`Drafter`): ``propose`` and ``observe`` for a verify step,
    ``prefill`` for a prompt. Parameter names are the checkpoint's (``fc``,
    ``hidden_norm``, the backbone's ``embed_tokens`` / ``layers`` / ``norm`` /
    ``lm_head``, ``markov_head.*``)."""

    def __init__(self, decl: DSpark, comm_group=None):
        super().__init__()
        from mstar.model.components.qwen3_lm import Qwen3DenseLM, Qwen3LMConfig

        cfg = decl.config
        self.decl = decl
        self.needs = decl.needs
        self.backbone = Qwen3DenseLM(
            Qwen3LMConfig(
                hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_layers,
                num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim, intermediate_size=cfg.intermediate_size, vocab_size=cfg.vocab_size,
                rms_norm_eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta,
                max_position_embeddings=cfg.max_position_embeddings,
            ),
            attn_key=decl.attn_key, kv_key=decl.kv_key, pos_key=decl.pos_key, comm_group=comm_group,
            # its layers in the target's pool, placed there by the pool's layer set
            kv_layer_set=LAYER_SET,
        )
        self.fc = nn.Linear(len(cfg.taps) * cfg.target_hidden_size, cfg.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.markov_head = _MarkovHead(cfg.vocab_size, cfg.markov_rank)
        self.register_buffer("_mask", torch.full((1, 1), cfg.mask_token_id, dtype=torch.long), persistent=False)
        self.taps = None

    def bind_resources(self, resources: dict) -> None:
        # read only in steps that declared a TapStep: a prefill and a verify step
        self.taps = resources.get(self.decl.needs.tap_key)

    def reset_buffers(self) -> None:
        """The buffers ``to_empty`` leaves unset."""
        self._mask.fill_(self.decl.config.mask_token_id)

    def reset_parameters(self) -> None:
        """Deterministic random weights, for a drafter with no checkpoint."""
        gen = torch.Generator(device="cpu").manual_seed(0)
        for name, p in self.named_parameters():
            if name.endswith("norm.weight"):
                p.data.fill_(1.0)
            else:
                values = torch.randn(p.shape, generator=gen) / max(p.shape[-1], 1) ** 0.5
                p.data.copy_(values.to(p.dtype))

    def load_weights(self, weights, target: nn.Module | None = None) -> set[str]:
        """``(name, tensor)`` pairs under the checkpoint's names; a checkpoint's
        confidence head (adaptive verification, unused) is skipped. A head the
        checkpoint ties to its own embedding is that embedding; an embedding or
        head it doesn't ship at all is ``target``'s module."""
        from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights

        own, rest = {}, []
        for name, tensor in weights:
            if name.startswith(("fc.", "hidden_norm.", "markov_head.")):
                own[name] = tensor
            elif name.startswith(("mask_embedding", "t2d", "d2t")):
                # a learned mask row, or a reduced draft vocabulary, which this drafter doesn't apply
                raise RuntimeError(f"the DSpark checkpoint ships {name}, which this drafter doesn't implement")
            elif not name.startswith("confidence_head."):
                rest.append((name.removeprefix("model."), tensor))
        names = {name for name, _ in rest}
        tied = self.decl.config.tie_word_embeddings and "embed_tokens.weight" in names
        shared = set()
        for module in ("embed_tokens", "lm_head"):
            if f"{module}.weight" in names or (module == "lm_head" and tied):
                continue
            if target is None:
                raise RuntimeError(f"the DSpark checkpoint has no {module}.weight and no target lends one")
            if target.config.hidden_size != self.decl.config.hidden_size:
                raise RuntimeError(
                    f"the DSpark checkpoint has no {module}.weight, and the target's is "
                    f"{target.config.hidden_size} wide, not the drafter's {self.decl.config.hidden_size}"
                )
            setattr(self.backbone, module, getattr(target, module))
            shared.add(f"backbone.{module}.weight")
        loaded = load_hf_weights(self.backbone, rest, stacked_params=LLAMA_STACKED_PARAMS)
        loaded = {f"backbone.{name}" for name in loaded}
        if tied and "lm_head.weight" not in names:
            # both shard the vocabulary alike, so the local embedding shard is the head's
            self.backbone.lm_head.weight.data.copy_(self.backbone.embed_tokens.weight.data)
            loaded.add("backbone.lm_head.weight")
        params = dict(self.named_parameters())
        for name, tensor in own.items():
            if name not in params:
                raise RuntimeError(f"the DSpark checkpoint names {name}, which the drafter lacks")
            params[name].data.copy_(tensor.to(params[name].dtype))
            loaded.add(name)
        missing = sorted(set(params) - loaded - shared)
        if missing:
            raise RuntimeError(f"the DSpark checkpoint left {len(missing)} parameter(s) unloaded, e.g. {missing[:5]}")
        return loaded | shared

    # ── the node's forward ──

    def prefill(self, engine_inputs=None, input_ids=None, sampled=None) -> None:
        """The prompt's context K/V, from the taps the target's prefill wrote (not its ids)."""
        del engine_inputs, input_ids, sampled
        self._write_context()

    def propose(self, inputs: DraftInputs) -> torch.Tensor:
        """``[n, k]`` drafts from each row's anchor, one backbone pass over
        ``[anchor, mask x k]`` and the Markov head left to right (argmax)."""
        k = inputs.num_draft_tokens
        anchor = inputs.input_ids.reshape(-1).long()
        n = anchor.shape[0]
        block = torch.cat([anchor[:, None], self._mask.expand(n, k)], dim=1)
        lm = self.backbone
        hidden = lm(lm.embed(block.reshape(-1)), label=LABEL).view(n, k + 1, -1)[:, :k]
        base = lm.logits(hidden.reshape(n * k, -1)).float().view(n, k, -1)
        prev, drafts = anchor, []
        for i in range(k):
            prev = (base[:, i] + self.markov_head(prev)).argmax(-1)
            drafts.append(prev)
        return torch.stack(drafts, dim=1)

    def observe(self, inputs: ObserveInputs) -> None:
        """The whole block's context K/V over the slots ``propose`` wrote; the
        take-back drops the rejected rows with the target's."""
        del inputs
        self._write_context()

    def _write_context(self) -> None:
        taps = self.taps.read()
        context = self.hidden_norm(self.fc(taps.reshape(taps.shape[0], -1).to(self.fc.weight.dtype)))
        # the query half of each layer's projection goes unused; a K/V-only projection would save it
        self.backbone.write_kv(context, label=LABEL)


class _MarkovHead(nn.Module):
    """``markov_w2(markov_w1[previous token])``: a rank-``r`` transition bias on
    the base logits. Replicated under TP: it runs once per draft, and sharded
    it would add a gather each time."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(tokens)).float()
