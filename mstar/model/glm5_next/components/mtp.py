"""GLM-5.3-Flash MTP module — the layer-45 draft model (M2 wiring target).

The HF implementation has NO MTP class (``_keys_to_ignore_on_load_unexpected``
drops ``layers.45.`` wholesale); glm52's ``components/mtp.py`` plus the
checkpoint tensors are the specification. Checkpoint anatomy at
``model.language_model.layers.45.*`` (1,760 tensors): DeepSeek-V3 glue —
``enorm`` / ``hnorm`` [4096], ``eh_proj`` [4096, 8192] fusing
``cat(enorm(tok_embed), hnorm(prev_hidden))``, ``shared_head.norm`` — plus
one full decoder layer: NoPE MLA with its OWN full DSA indexer (the 12th)
and a 288+1-expert MoE. **No ``hc_attn_*``/``hc_ffn_*`` tensors** — the
draft layer is plain-residual, running on the collapsed single stream, NOT
the mHC layer. No embedding and no head under layer 45: drafts reuse the
trunk's ``embed_tokens``/``lm_head`` exactly like glm52.

Call contract (glm52 MTP-loop compatible): ``forward(token_embeds,
prev_hidden)`` returns ``(head_input,
raw_hidden)`` — the shared_head-normed state for the caller-owned
``lm_head``, and the raw layer output for chaining draft iterations
(``hnorm`` expects the UN-normalized stream; feeding normed hidden
double-norms the fusion and zeroed acceptance in glm52, 2026-08-09).

M2 open items this module deliberately does not decide: which trunk stream
``prev_hidden`` pairs against (post-``hc_head`` pre-final-norm vs post-norm
vs a single stream — port glm52's env-switch A/B, do not assume), the
``index_share_for_mtp_iteration`` selection reuse, and KV + **KDA
recurrent-state** rewind on rejection (the KDA store's snapshot/restore is
the primitive).
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.glm5_next.components.decoder_layer import Glm5NextPlainDecoderLayer
from mstar.model.glm5_next.components.language_model import build_rmsnorm
from mstar.model.glm5_next.config import Glm5NextModelConfig

# Glue sub-keys under ``layers.45.`` that map 1:1 onto module attributes;
# the loader's remap keeps its own copy (it must import without this
# package's engine-touching modules) — ``weight_loader.MTP_GLUE_PREFIXES``
# is asserted equal here so the two cannot drift.
MTP_GLUE_PREFIXES = ("enorm", "hnorm", "eh_proj", "shared_head")


def _assert_loader_glue_contract() -> None:
    from mstar.model.glm5_next import weight_loader

    assert weight_loader.MTP_GLUE_PREFIXES == MTP_GLUE_PREFIXES, (
        "MTP glue prefixes drifted between components/mtp.py and "
        f"weight_loader.py: {MTP_GLUE_PREFIXES} vs "
        f"{weight_loader.MTP_GLUE_PREFIXES}"
    )


class Glm5NextSharedHead(nn.Module):
    """Named to mirror the checkpoint's ``shared_head.norm`` key: the MTP
    module's final norm. The actual head weight is the trunk's ``lm_head``
    (not duplicated in the checkpoint), applied by the caller."""

    def __init__(self, config: Glm5NextModelConfig) -> None:
        super().__init__()
        self.norm = build_rmsnorm(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class Glm5NextMTPModule(nn.Module):
    """One draft iteration: fuse (token embedding, previous hidden) and run
    the layer-45 plain-residual decoder layer.

    The draft KV lives at plane ``kv_plane = len(full_attn_layer_indices)``
    (11 for the full model — plane indices are dense over full-attention
    layers, NOT layer indices), and the transformer layer sets it itself,
    so a glm52-style loop calling ``set_layer_idx(num_hidden_layers)``
    first is harmlessly overridden instead of silently addressing a
    nonexistent plane 45. The engine half owns allocating that plane and
    rewinding it (with the KDA state snapshots) on draft rejection.
    """

    def __init__(
        self, config: Glm5NextModelConfig, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        _assert_loader_glue_contract()
        self.kv_plane = len(config.full_attn_layer_indices)
        self.enorm = build_rmsnorm(config)
        self.hnorm = build_rmsnorm(config)
        self.eh_proj = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=False
        )
        self.transformer_layer = Glm5NextPlainDecoderLayer(
            config,
            layer_idx=config.num_hidden_layers,
            kv_plane=self.kv_plane,
            comm_group=comm_group,
        )
        self.shared_head = Glm5NextSharedHead(config)

    def fuse(
        self, token_embeds: torch.Tensor, prev_hidden: torch.Tensor
    ) -> torch.Tensor:
        return self.eh_proj(
            torch.cat([self.enorm(token_embeds), self.hnorm(prev_hidden)], dim=-1)
        )

    def forward(
        self,
        token_embeds: torch.Tensor,
        prev_hidden: torch.Tensor,
        dsa_ctx=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(head_input, raw_hidden)`` — see the module docstring.

        ``dsa_ctx`` exists for glm52-loop signature parity only; the
        glm5_next DSA engine path is a post-M1 follow-up and MTP v1 stays
        in the ctx <= index_topk identity regime.
        """
        if dsa_ctx is not None:
            raise NotImplementedError(
                "glm5_next MTP has no DSA engine path yet (identity regime "
                "only); pass dsa_ctx=None"
            )
        hidden_states = self.fuse(token_embeds, prev_hidden)
        hidden_states = self.transformer_layer(
            hidden_states
        )
        return self.shared_head(hidden_states), hidden_states
