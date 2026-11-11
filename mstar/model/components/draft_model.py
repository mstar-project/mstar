"""A separate draft model (``model_kwargs.draft: {method: draft_model, model: Qwen/Qwen3-0.6B}``): a smaller
model of the target's family drafts ``k`` tokens a step, greedily, from its own layers, query heads and
weights. It reads none of the target's state but the tokens.

Its KV is a layer set of the target's pool (`DrafterNeeds.kv`): the same pages, one stored length. It
lags the target by a token on a verify step (`DrafterNeeds.lag`): its block is the target's shifted back
one row, ``[prev, bonus, d1 .. d(k-1)]`` at positions ``L - 1 .. L + k - 1``, run in one-row passes after
a first pass of two rows (``[prev, bonus]``), each attending through its own extent. ``prev``, the token
before the block, crosses steps as the drafter's carry. Its K/V at ``L - 1`` exists already unless the
target accepted every draft last step, so the engine sends row 0's write to the sink unless the step's
keep flag says otherwise (`DRAFT_KEEP_FIRST`). A prompt step writes every prompt row unlagged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.draft.drafter import (
    CarrySpec,
    ChunkBoundary,
    DrafterNeeds,
    DraftInputs,
    ModelGeometry,
    ObserveInputs,
    PrefillInputs,
)
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import Lag, ResourceStep
from mstar.model.components.hosted_drafter import HostedDrafter, HostedDrafterDecl
from mstar.model.components.qwen3_lm import Qwen3DenseLM, Qwen3LMConfig

LAYER_SET = "draft_model"
LABEL = "main"


def draft_model_refusals(hf: dict, target_vocab: int) -> list[str]:
    """Why a checkpoint can't draft as a separate draft model here (unsupported for now)."""
    problems = []
    if hf.get("model_type") != "qwen3":
        problems.append(f"draft models outside the Qwen3 dense family are not supported (model_type "
                        f"{hf.get('model_type')!r})")
    if hf.get("num_experts") or hf.get("num_local_experts"):
        problems.append("a mixture-of-experts draft model is not supported")
    if hf.get("use_sliding_window"):
        problems.append("a sliding-window draft model is not supported")
    if hf.get("vocab_size") != target_vocab:
        problems.append(f"a draft model with another vocabulary ({hf.get('vocab_size')} tokens; the target's "
                        f"{target_vocab}) is not supported")
    return problems


@dataclass(frozen=True)
class DraftModel(HostedDrafterDecl):
    """A separate draft model as its node declares it, from config alone."""

    config: Qwen3LMConfig
    num_draft_tokens: int
    # the checkpoint's weight dtype, as config.json names it
    dtype: str
    # the target's resources it writes and rotates through
    kv_key: str
    pos_key: str
    # its own: an attention binding with its query heads over its layer set
    attn_key: str = "draft_model_attn"
    checkpoint: str | None = None
    cache_dir: str | None = None

    @property
    def lag(self) -> Lag:
        """``[prev, bonus]``, then a draft a pass."""
        return Lag(1, (2,) + (1,) * (self.num_draft_tokens - 1))

    @property
    def needs(self) -> DrafterNeeds:
        cfg = self.config
        geometry = ModelGeometry(
            num_layers=cfg.num_hidden_layers, num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
            hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size, dtype=self.dtype,
            num_qo_heads=cfg.num_attention_heads, rope_theta=cfg.rope_theta,
        )
        return DrafterNeeds(
            kv=geometry.layer_set(self.kv_key, LAYER_SET),
            # the token before the next block, which its first pass re-reads
            carry=CarrySpec(1, torch.int64),
            # its K/V at p is a function of the tokens up to p
            chunk_boundary=ChunkBoundary.NONE,
            geometry=geometry,
            lag=self.lag,
        )

    def node_resources(self, nodes: set[str]) -> list[NodeResourceSpec]:
        return [AttentionSpec(
            resource_key=self.attn_key, nodes=set(nodes),
            config=AttentionConfig(kv_cache=self.kv_key, layer_set=LAYER_SET,
                                   num_qo_heads=self.config.num_attention_heads),
        )]

    def prefill_steps(self) -> dict[str, ResourceStep]:
        """A prompt step's share: its layers over the prompt, unlagged."""
        return {self.attn_key: AttentionStep(causal=True)}

    def verify_steps(self) -> dict[str, ResourceStep]:
        """A verify step's share: its passes over the block (the engine adds the lag)."""
        return {self.attn_key: AttentionStep(verify=True, causal=True)}

    def build(self, *, device, comm_group=None, dtype: torch.dtype | None = None, target: nn.Module | None = None):
        """The module with the checkpoint's weights (none: deterministic random weights, for tests)."""
        del target  # it shares nothing with the target's modules
        with torch.device("meta"):
            module = DraftModelDrafter(self, comm_group=comm_group)
        if dtype is not None:
            module = module.to(dtype)
        module.to_empty(device=device)
        if self.checkpoint is None:
            # norms at 1, the rest fan-in scaled: logits far enough apart that greedy drafts are no coin flips
            gen = torch.Generator(device="cpu").manual_seed(0)
            for name, param in module.named_parameters():
                if name.endswith("norm.weight"):
                    param.data.fill_(1.0)
                else:
                    init = torch.randn(param.shape, generator=gen) / max(param.shape[-1], 1) ** 0.5
                    param.data.copy_(init.to(param.dtype))
        else:
            from mstar.model.loader.iterators import iter_safetensors_shards
            from mstar.utils.hf_snapshot import resolve_snapshot_dir

            directory = resolve_snapshot_dir(
                self.checkpoint, cache_dir=self.cache_dir, allow_patterns=["*.json", "*.safetensors"],
            )
            module.lm.load_weights((name.removeprefix("model."), tensor)
                                   for name, tensor in iter_safetensors_shards(directory, device=device))
        return module.requires_grad_(False).eval()


class DraftModelDrafter(nn.Module, HostedDrafter):
    """The draft model a verifying node calls: its own stack, embedding and head, writing the layer set's
    KV through its binding."""

    def __init__(self, decl: DraftModel, comm_group=None):
        super().__init__()
        self.decl = decl
        self.needs = decl.needs
        self.k = decl.num_draft_tokens
        self.lm = Qwen3DenseLM(
            decl.config, attn_key=decl.attn_key, kv_key=decl.kv_key, pos_key=decl.pos_key, comm_group=comm_group,
            # its layers in the target's pool, placed there by the pool's layer set
            kv_layer_set=LAYER_SET,
        )
        self._passes: list = []

    def bind_resources(self, resources: dict) -> None:
        # what a pass selects: its KV writes, its positions and its attention's extent (`select_pass`)
        self._passes = [resources[self.decl.kv_key], resources[self.decl.pos_key], resources[self.decl.attn_key]]

    @torch.compiler.disable
    def _select(self, p: int | None) -> None:
        for resource in self._passes:
            resource.select_pass(p)

    def _run(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm(self.lm.embed(ids), label=LABEL)

    # ── prompt step ──

    def prefill(self, inputs: PrefillInputs) -> torch.Tensor:
        """Its K/V over every prompt row, unlagged; returns each prompt's carry: its last token, the ``prev``
        of the request's first verify step."""
        input_ids = inputs.input_ids
        self._run(input_ids)
        attn = inputs.resources[self.decl.attn_key]
        last = attn.select_last_hidden(input_ids.view(-1, 1), label=LABEL)
        return last.long()

    # ── verify step ──

    def propose(self, inputs: DraftInputs) -> torch.Tensor:
        n, k = inputs.input_ids.shape[0], self.k
        prev, bonus = inputs.carry.view(n).long(), inputs.input_ids.view(n).long()
        self._select(0)
        hidden = self._run(torch.stack([prev, bonus], dim=1).view(-1))
        draft = self.lm.logits(hidden.view(n, 2, -1)[:, 1]).argmax(-1)
        drafts = [draft]
        for p in range(1, k):
            self._select(p)
            draft = self.lm.logits(self._run(draft)).argmax(-1)
            drafts.append(draft)
        self._select(None)
        return torch.stack(drafts, dim=1)

    def observe(self, inputs: ObserveInputs) -> torch.Tensor:
        """The next step's ``prev``: the token before its first, the last one this step kept of the block."""
        block, n = inputs.block, inputs.block.shape[0]
        return block[torch.arange(n, device=block.device), inputs.result.accepted].view(n, 1)
