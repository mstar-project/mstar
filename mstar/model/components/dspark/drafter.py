"""A DSpark drafter inside the node that verifies its drafts: one module for every DSpark
checkpoint, configured by ``DSparkConfig``.

Its state is the node's resources, declared by ``DSpark.node_resources``: a paged KV of its own
(``dspark_kv``, sized and evicted by the engine like any pool), positions on it and an
attention resource over it, and the target taps (``TapSpec``). Its KV runs level with the
target's: it holds, at each position the target has kept, the drafter's context K/V of that
position (its layers' K/V projections of ``hidden_norm(fc(taps))``). A verify step's KV step
is the verify block itself, so the verdict takes back the same rows from both caches. Per
verify step of ``L`` drafts (``DSpark.max_drafts``, every step):

    propose   the block [anchor, mask x L] through the backbone, every row attending to the
              context and to the block's first L rows (non-causal, the last row hidden), its
              K/V written into the block's slots; rows 0..L-1 give the base logits, the Markov
              head drafts left to right
    verify    the target over [anchor, d1 .. dL], which writes the taps (the engine's)
    observe   the context K/V of the whole verified block, from the taps, over the block's slots

so the next step reads, for the rows the verdict keeps, the target's own features, and the
rejected rows are rewound with the target's. A prefill observes the prompt: the target's
prefill tapped every row.

The block's rows sit at the verify block's positions (the anchor at the position its verify
row takes), as DSpark's training places them; the trailing row the verify block has one more
of is a placeholder no row attends to (``AttentionStep(hide_last=1)``). The whole step, drafts
and verify, replays as one captured graph per batch size and block.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn

from mstar.engine.resources import (
    DRAFT_TAPS,
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    Drafts,
    KVSpec,
    KVStep,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    PositionStep,
    TapSpec,
    TapStep,
)
from mstar.engine.resources.step import ResourceStep
from mstar.model.base import KVLabel, LabelLayout, Span
from mstar.model.components.dspark.config import DSparkConfig
from mstar.model.components.norm import RMSNorm

if TYPE_CHECKING:
    from mstar.distributed.communication import CommGroup
    from mstar.engine.resources.draft_verify.resource import DraftVerifyResource, VerdictTensors


class DSparkKeys(NamedTuple):
    """The resource keys a DSpark drafter's own state is declared under."""
    kv: str = "dspark_kv"
    attn: str = "dspark_attn"
    pos: str = "dspark_pos"


@dataclass(frozen=True)
class DSpark:
    """A DSpark drafter as its node declares it: the resources, request layouts, steps and
    policy fields it adds to the target's. Plain configuration, built wherever the model is
    (no weights); ``build`` makes the module on a worker."""
    config: DSparkConfig
    max_drafts: int
    # the drafter's KV pool (its head geometry; the engine sizes the pages)
    kv: PagedKVConfig
    keys: DSparkKeys = DSparkKeys()
    # draw each draft as the request samples (``DraftVerifyResource.draft_sample``); else the argmax
    sample: bool = False
    # the taps' dtype: the target's activations
    tap_dtype: torch.dtype = torch.bfloat16
    # the checkpoint (a Hugging Face id or a path) and the cache it resolves in, for ``build``
    checkpoint: str | None = None
    cache_dir: str | None = None
    # the prompt's input signal its prefill span names (the target's)
    prompt_signal: str = "text_inputs"
    # the dtype positions rotate in (``PositionConfig.rope_dtype``; None: the activations')
    rope_dtype: torch.dtype | None = None
    extra: dict = field(default_factory=dict)

    # Its prefill is the target's packed prefill and the context write from the taps, over the
    # same rows, so the target's prefill captures replay it.
    prefill_capturable = True

    def __post_init__(self):
        if not 1 <= self.max_drafts <= self.config.block_size:
            raise ValueError(
                f"a DSpark drafter of block_size {self.config.block_size} drafts 1..{self.config.block_size} "
                f"tokens a step; asked for {self.max_drafts}"
            )

    @classmethod
    def for_target(
        cls, config: DSparkConfig, max_drafts: int, *, max_seq_len: int, target_layers: int,
        target_hidden_size: int, sample: bool = False, tap_dtype: torch.dtype = torch.bfloat16,
        checkpoint: str | None = None, cache_dir: str | None = None, keys: DSparkKeys = DSparkKeys(),
    ) -> DSpark:
        """The declaration for a target of ``target_layers`` layers of ``target_hidden_size``:
        refuses a drafter that reads layers the target lacks or a width it doesn't have."""
        if max(config.taps) > target_layers:
            raise ValueError(f"the DSpark drafter reads target layers {config.taps}; the target has {target_layers}")
        if config.target_hidden_size != target_hidden_size:
            raise ValueError(
                f"the DSpark drafter reads a {config.target_hidden_size}-wide stream; "
                f"the target's is {target_hidden_size}"
            )
        if config.rope_scaling is not None:
            raise ValueError(
                f"a DSpark drafter with rope_scaling {config.rope_scaling} needs that scaling on the position "
                "resource (YaRN: the YaRN RoPE tables change)"
            )
        if config.sliding_window is not None and not config.causal:
            raise ValueError("a non-causal DSpark drafter over a sliding window needs its family's attention")
        kv = PagedKVConfig(
            num_layers=config.num_layers, num_kv_heads=config.num_key_value_heads,
            num_qo_heads=config.num_attention_heads, head_dim=config.head_dim, max_seq_len=max_seq_len,
        )
        return cls(config, max_drafts, kv, keys, sample, tap_dtype, checkpoint, cache_dir)

    # ── what the node declares ──

    def policy_fields(self) -> dict:
        """What the walk's ``DraftPolicy`` carries for it: the target layers it reads, and that
        its drafts need no verdict on the host (``device_rewind``)."""
        return {"taps": self.config.taps, "device_rewind": True}

    def node_resources(self, nodes: set[str]) -> list[NodeResourceSpec]:
        keys = self.keys
        cfg = self.config
        return [
            KVSpec(resource_key=keys.kv, nodes=set(nodes), config=self.kv),
            AttentionSpec(
                resource_key=keys.attn, nodes=set(nodes),
                config=AttentionConfig(kv_cache=keys.kv, sliding_window=cfg.sliding_window),
            ),
            PositionSpec(
                resource_key=keys.pos, nodes=set(nodes),
                config=PositionConfig(kv_cache=keys.kv, rope_theta=cfg.rope_theta, rope_dtype=self.rope_dtype),
            ),
            TapSpec(
                resource_key=DRAFT_TAPS, nodes=set(nodes), layers=cfg.taps,
                hidden_size=cfg.target_hidden_size, dtype=self.tap_dtype,
            ),
        ]

    def kv_labels(self, walk: str, loop: str) -> dict[str, dict[str, KVLabel]]:
        return {self.keys.kv: {"main": KVLabel(growth=((walk, loop),))}}

    def check_prompt(self, length: int) -> None:
        """Any prompt: the drafter's context is the target's."""
        del length

    def kv_layout(
        self, prompt_len: int, prefill_walk: str, loop: str, lookahead: int,
    ) -> dict[str, dict[str, LabelLayout]]:
        """The drafter's KV layout beside the target's, level with it: the prompt, one slot per
        kept token, and the target's lookahead."""
        span = Span("rows", prefill_walk, prompt_len, prompt_len, self.prompt_signal)
        return {self.keys.kv: {"main": LabelLayout(spans=(span,), growth=((loop, 1),), lookahead=lookahead)}}

    def prefill_steps(self, request_ids: Sequence[int], spans: Sequence[int]) -> dict[str, ResourceStep]:
        """A prefill writes the prompt's context: the drafter's KV and positions over the
        step's rows, and the taps the target writes."""
        del request_ids, spans
        keys = self.keys
        return {keys.kv: KVStep(), keys.pos: PositionStep(), DRAFT_TAPS: TapStep()}

    def verify_steps(self, drafts: int | None = None) -> dict[str, ResourceStep]:
        """The drafter's part of a verify step of ``drafts`` drafts, on the step's own block
        segments (``with_draft_verify`` adds the taps); a step of none only observes."""
        keys = self.keys
        steps: dict[str, ResourceStep] = {keys.kv: KVStep(), keys.pos: PositionStep()}
        if drafts is None or drafts:
            causal = self.config.causal
            steps[keys.attn] = AttentionStep(causal=causal, hide_last=0 if causal else 1)
        return steps

    def carry_width(self) -> int:
        """Tokens a verify step's input carries: the last one kept."""
        return 1

    def prefill_inputs(self, prompts: Sequence[torch.Tensor]) -> dict:
        """Nothing beyond the target's prefill: the drafter reads the taps."""
        del prompts
        return {}

    def build(
        self, *, device, comm_group: CommGroup | None = None, dtype: torch.dtype | None = None,
        moe_parallel=None, target: nn.Module | None = None, backbone: nn.Module | None = None,
    ) -> DSparkDrafter:
        """The module on its resources, sharded like the target, with the checkpoint's weights
        (``checkpoint``; none leaves them as built). ``target`` lends its embedding and head
        to a checkpoint that ships none; ``backbone`` is a family's block stack."""
        with torch.device("meta"):
            module = DSparkDrafter(self, comm_group=comm_group, moe_parallel=moe_parallel, backbone=backbone)
        if dtype is not None:
            module = module.to(dtype)
        module.to_empty(device=device)
        module.reset_buffers()
        if self.checkpoint is not None:
            from mstar.model.loader.iterators import iter_safetensors_shards
            from mstar.utils.hf_snapshot import resolve_snapshot_dir

            directory = resolve_snapshot_dir(
                self.checkpoint, cache_dir=self.cache_dir, allow_patterns=["*.json", "*.safetensors"],
            )
            module.load_weights(iter_safetensors_shards(directory, device=device), target=target)
        else:
            module.reset_parameters()
        return module.eval()


class DSparkDrafter(nn.Module):
    """The drafter, on the resources ``DSpark`` declares: a ``Drafter`` for
    ``draft_verify_step`` (``propose``, ``observe``, ``next_inputs``) and the node's prefill
    (``prefill``). Parameter names are the checkpoints' (``fc``, ``hidden_norm``, the
    backbone's ``embed_tokens`` / ``layers`` / ``norm`` / ``lm_head``, ``markov_head.*``). A
    checkpoint's ``confidence_head``, which scores drafts for adaptive verification, isn't
    loaded: every step drafts the whole block."""

    def __init__(self, decl: DSpark, comm_group: CommGroup | None = None, moe_parallel=None,
                 backbone: nn.Module | None = None):
        super().__init__()
        cfg = decl.config
        self.decl = decl
        if backbone is None:
            if cfg.backbone != "qwen3":
                raise ValueError(
                    f"the {cfg.source} DSpark drafter's layers are its target family's blocks: build it "
                    "with that family's backbone"
                )
            backbone = _qwen3_backbone(cfg, decl.keys, comm_group, moe_parallel)
        self.backbone = backbone
        self.fc = nn.Linear(len(cfg.taps) * cfg.target_hidden_size, cfg.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.markov_head = _MarkovHead(cfg.vocab_size, cfg.markov_rank)
        self.register_buffer("_mask", torch.full((1, 1), cfg.mask_token_id, dtype=torch.long), persistent=False)
        self.taps = None

    def bind_resources(self, resources: dict) -> None:
        self.taps = resources.get(DRAFT_TAPS)

    def reset_buffers(self) -> None:
        """The buffers ``to_empty`` leaves unset."""
        self._mask.fill_(self.decl.config.mask_token_id)

    def reset_parameters(self) -> None:
        """Random weights (no checkpoint): a drafter for tests."""
        gen = torch.Generator(device="cpu").manual_seed(0)
        for name, p in self.named_parameters():
            if name.endswith(("norm.weight", "_norm.weight")):
                p.data.fill_(1.0)
            else:
                values = torch.randn(p.shape, generator=gen) / max(p.shape[-1], 1) ** 0.5
                p.data.copy_(values.to(p.dtype))

    # ── weights ──

    def load_weights(self, weights, target: nn.Module | None = None) -> set[str]:
        """``(name, tensor)`` pairs under the checkpoints' names. A checkpoint that ships no
        embedding or head takes ``target``'s (they share the target's vocabulary)."""
        own, rest = {}, []
        for name, tensor in weights:
            if name.startswith(("fc.", "hidden_norm.", "markov_head.")):
                own[name] = tensor
            elif name.startswith(("t2d", "d2t", "mask_embedding", "confidence_head.")):
                continue
            else:
                rest.append((name.removeprefix("model."), tensor))
        names = {name for name, _ in rest}
        for name, module in (("embed_tokens.weight", "embed_tokens"), ("lm_head.weight", "lm_head")):
            if name not in names:
                if target is None:
                    raise RuntimeError(f"the DSpark checkpoint has no {name} and no target lends one")
                setattr(self.backbone, module, getattr(target, module))
                rest.append((name, getattr(target, module).weight.detach()))
        loaded = {f"backbone.{name}" for name in self.backbone.load_weights(rest)}
        params = dict(self.named_parameters())
        for name, tensor in own.items():
            if name not in params:
                raise RuntimeError(f"the DSpark checkpoint names {name}, which the drafter lacks")
            params[name].data.copy_(tensor.to(params[name].dtype))
            loaded.add(name)
        missing = sorted(set(params) - loaded)
        if missing:
            raise RuntimeError(f"the DSpark checkpoint left {len(missing)} parameter(s) unloaded, e.g. {missing[:5]}")
        return loaded

    # ── the node's forward ──

    def prefill(self, **kwargs) -> None:
        """The prompt's context, from the taps the target's prefill wrote (every row)."""
        del kwargs
        self._write_context()

    def propose(self, inputs: torch.Tensor, dv: DraftVerifyResource, request_ids: list[int]) -> Drafts:
        """The step's drafts from each row's anchor (``inputs[:, -1]``), one backbone pass over
        ``[anchor, mask x L]`` and the heads left to right."""
        L = self.decl.max_drafts
        n = inputs.shape[0]
        anchor = inputs.reshape(n, -1)[:, -1].to(torch.long)
        block = torch.cat([anchor[:, None], self._mask.expand(n, L)], 1)
        lm = self.backbone
        hidden = lm(lm.embed(block.reshape(-1)), label="main").view(n, L + 1, -1)[:, :L]
        base = lm.logits(hidden.reshape(n * L, -1)).float().view(n, L, -1)
        prev = anchor
        tokens, dists = [], []
        for i in range(L):
            markov = self.markov_head.embed(prev)
            logits = base[:, i] + self.markov_head.bias(markov)
            if self.decl.sample:
                draft, dist = dv.draft_sample(request_ids, logits)
            else:
                draft, dist = logits.argmax(-1), None
            draft = draft.to(torch.long)
            tokens.append(draft)
            dists.append(dist)
            prev = draft
        probs = torch.stack(dists, 1) if dists[0] is not None else None
        return Drafts(torch.stack(tokens, 1), probs)

    def observe(self, block: torch.Tensor, verdict: VerdictTensors) -> None:
        """The verified block's context K/V, every row; the verdict's rewind takes the
        rejected rows back with the target's."""
        del block, verdict
        self._write_context()

    @staticmethod
    def next_inputs(block: torch.Tensor, verdict: VerdictTensors) -> torch.Tensor:
        """``[n, 1]``: the target's next token, the next block's anchor."""
        return verdict.next_inputs.to(block.dtype)

    def _write_context(self) -> None:
        taps = self.taps.read()
        context = self.hidden_norm(self.fc(taps.reshape(taps.shape[0], -1).to(self.fc.weight.dtype)))
        lm = self.backbone
        lm.layers[0].self_attn.attend.bind_step("main")
        for layer_idx, layer in enumerate(lm.layers):
            layer.self_attn.attend.set_layer_idx(layer_idx)
            write_context = getattr(layer.self_attn, "write_context", None)
            if write_context is not None:
                write_context(context)  # a family block's own
            else:
                _write_qwen3_context(layer.self_attn, context)


class _MarkovHead(nn.Module):
    """The sequential head: ``markov_w2(markov_w1[previous token])``, a ``V x r`` and an
    ``r x V`` factor of a transition bias added to the base logits. Replicated: it runs once
    per draft, and sharded it would add a gather each time (vLLM's choice too)."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(tokens)

    def bias(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(embedding).float()


def _qwen3_backbone(cfg: DSparkConfig, keys: DSparkKeys, comm_group, moe_parallel) -> nn.Module:
    """The generic backbone: the shared Qwen3 decoder (dense, or routed experts) on the
    drafter's resources."""
    from mstar.model.components.qwen3_lm import Qwen3DenseLM, Qwen3LMConfig, Qwen3MoeLM

    lm_config = Qwen3LMConfig(
        hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, intermediate_size=cfg.intermediate_size, vocab_size=cfg.vocab_size,
        rms_norm_eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_position_embeddings, num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok, moe_intermediate_size=cfg.moe_intermediate_size,
    )
    resource_keys = dict(attn_key=keys.attn, kv_key=keys.kv, pos_key=keys.pos, comm_group=comm_group)
    if cfg.num_experts:
        return Qwen3MoeLM(lm_config, moe_parallel=moe_parallel, **resource_keys)
    return Qwen3DenseLM(lm_config, **resource_keys)


def _write_qwen3_context(attn, context: torch.Tensor) -> None:
    """One layer's context K/V, written into the step's slots: its K (q/k-normed, rotated at
    the step's positions) and V projections of the context features. The query half of the
    projection is thrown away; a K/V-only projection is the obvious saving."""
    q, k, v = attn._project_qkv(context)
    q, k = attn._apply_qk_norm(q, k)
    _, k = attn._apply_rope(q, k, attn.attend.label)
    attn.kv.write_kv(k, v)
