"""A drafter inside the node that verifies its drafts, whichever kind a deployment names
(``DraftConfig.method``: a draft model, a DSpark drafter), behind one surface, so a target
model integrates every kind the same way and names none of them.

The declaration (plain data, built wherever the model is) gives the node:

- ``policy_fields()``: what the walk's ``DraftPolicy`` carries for it (a DSpark drafter's taps);
- ``node_resources(nodes)``: its own resources beside the target's;
- ``kv_labels(walk, loop)``, ``kv_layout(prompt_len, prefill_walk, loop, lookahead)`` and
  ``check_prompt(len)``: its KV layout beside the target's;
- ``prefill_steps(request_ids, spans)`` and ``verify_steps(drafts)``: its parts of a step;
- ``carry_width()``: the tokens a verify step's input carries (a decode loop that starts from
  more than the prefill's sample takes them on a ``decode_input`` edge);
- ``prefill_inputs(prompts)``: what the prefill forward hands the module;
- ``prefill_capturable``: whether the target's packed prefill captures can replay its prefill
  (else the node prefills eagerly);
- ``build(device=, comm_group=, dtype=, moe_parallel=, target=)``: the module, on a worker.

The module is a ``Drafter`` for ``draft_verify_step`` and has ``prefill(tokens=, **inputs)``,
which returns the decode loop's first inputs when they are more than the prefill's sample.
"""

from __future__ import annotations

import json
from typing import NamedTuple

import torch

from mstar.engine.resources import DraftConfig, PagedKVConfig


class TargetInfo(NamedTuple):
    """What a drafter's declaration checks against and sizes from: the target's."""
    vocab_size: int
    num_layers: int
    hidden_size: int
    max_seq_len: int
    # the target checkpoint's directory, whose tokenizer a draft model must share
    checkpoint_dir: object = None
    dtype: torch.dtype = torch.bfloat16


def in_node_drafter(draft: DraftConfig | None, target: TargetInfo, *, cache_dir: str | None = None,
                    families: dict | None = None):
    """The declaration of ``draft``'s drafter for ``target``, or None for a method with none
    (prompt lookup). ``families``: model_type -> ``LMFamily``, the LMs a draft model may be."""
    if draft is None or draft.method not in DraftConfig.DRAFTERS:
        return None
    from mstar.utils.hf_snapshot import resolve_snapshot_dir

    directory = resolve_snapshot_dir(draft.model, cache_dir=cache_dir, allow_patterns=["*.json", "*.txt"])
    hf = json.loads((directory / "config.json").read_text())
    if draft.method == "dspark":
        from mstar.model.components.dspark import DSpark, DSparkConfig

        config = DSparkConfig.from_hf(hf)
        if config.vocab_size < target.vocab_size:
            raise ValueError(f"the DSpark drafter {draft.model} has a vocabulary of {config.vocab_size}, "
                             f"short of the target's {target.vocab_size}")
        return DSpark.for_target(
            config, draft.max_drafts, max_seq_len=target.max_seq_len, target_layers=target.num_layers,
            target_hidden_size=target.hidden_size, sample=draft.draft_sample, tap_dtype=target.dtype,
            checkpoint=draft.model, cache_dir=cache_dir,
        )
    return DraftModelDrafter.from_checkpoint(draft, target, hf, directory, cache_dir, families or {})


class DraftModelDrafter:
    """A draft model's declaration (``DraftModel``) with its checkpoint and LM family, which
    ``build`` loads; everything else is the ``DraftModel``'s."""

    def __init__(self, decl, family, lm_config, checkpoint: str, cache_dir: str | None):
        self.decl = decl
        self.family = family
        self.lm_config = lm_config
        self.checkpoint = checkpoint
        self.cache_dir = cache_dir

    @classmethod
    def from_checkpoint(cls, draft: DraftConfig, target: TargetInfo, hf: dict, directory, cache_dir, families):
        from mstar.model.components.draft_model import DraftModel

        if hf.get("model_type") not in families:
            raise ValueError(f"the draft model {draft.model} is a {hf.get('model_type')!r}; known: {sorted(families)}")
        family = families[hf["model_type"]]
        config = family.config(hf)
        if config.vocab_size != target.vocab_size or (
            target.checkpoint_dir is not None and _vocab(directory) != _vocab(target.checkpoint_dir)
        ):
            raise ValueError(
                f"the draft model {draft.model} tokenizes differently from the target; a draft model shares "
                "the target's tokenizer"
            )
        if config.max_position_embeddings < target.max_seq_len:
            raise ValueError(f"the draft model {draft.model} takes {config.max_position_embeddings} positions, "
                             f"short of max_seq_len {target.max_seq_len}")
        decl = DraftModel(
            max_drafts=draft.max_drafts, sample=draft.draft_sample, rope_theta=config.rope_theta,
            kv=PagedKVConfig(
                num_layers=config.num_hidden_layers, num_kv_heads=config.num_key_value_heads,
                num_qo_heads=config.num_attention_heads, head_dim=config.head_dim, max_seq_len=target.max_seq_len,
            ),
        )
        return cls(decl, family, config, draft.model, cache_dir)

    def __getattr__(self, name):
        # the declaration's surface (node_resources, kv_layout, ...) is the DraftModel's
        return getattr(self.decl, name)

    def build(self, *, device, comm_group=None, dtype=None, moe_parallel=None, target=None):
        """The draft model's LM on its own resources, sharded like the target."""
        import logging

        from mstar.model.components.draft_model import DraftLLMProposer
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.utils.hf_snapshot import resolve_snapshot_dir

        del target
        weights = resolve_snapshot_dir(
            self.checkpoint, cache_dir=self.cache_dir, allow_patterns=["*.json", "*.safetensors"],
        )
        keys = self.decl.keys
        with torch.device("meta"):
            lm = self.family.build(
                self.lm_config, comm_group=comm_group, moe_parallel=moe_parallel, keys=(keys.attn, keys.kv, keys.pos),
            )
        if dtype is not None:
            lm = lm.to(dtype)
        lm.to_empty(device=device)
        lm.load_weights(
            (name, tensor) for key, tensor in iter_safetensors_shards(weights, device=device)
            if (name := self.family.weight_name(key)) is not None
        )
        lm.eval()
        logging.getLogger(__name__).info("%s drafts %d tokens a step", self.checkpoint, self.decl.max_drafts)
        return DraftLLMProposer(self.decl, lm)


def _vocab(directory) -> dict | None:
    """A checkpoint's tokenizer vocabulary, None without a ``tokenizer.json``."""
    path = directory / "tokenizer.json"
    return json.loads(path.read_text())["model"]["vocab"] if path.is_file() else None
