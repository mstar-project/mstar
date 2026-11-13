"""The assembled DeepSeek-V4.1 text backbone: hyper-connected blocks, Engram, the LM head."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ColumnParallelLinear, VocabParallelEmbedding
from mstar.model.deepseek_v41.components.attention import DeepseekV41Attention, SharedSelection
from mstar.model.deepseek_v41.components.engram import Engram, NgramHasher
from mstar.model.deepseek_v41.components.linear import RMSNorm
from mstar.model.deepseek_v41.components.moe import DeepseekV41MoE
from mstar.model.deepseek_v41.components.rope import Rope
from mstar.model.deepseek_v41.config import ENGRAM, DeepseekV41Config


class HyperConnection(nn.Module):
    """One hyper-connection site: from the live streams, the weights that collapse them into a
    sublayer's input (``pre``), place its output (``post``) and mix the streams (``comb``, made
    doubly stochastic by Sinkhorn steps). In fp32, in the order the checkpoint's reference
    computes it: the streams are projected and then scaled by their RMS, not normalized first."""

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        self.hc, self.eps, self.rms_eps = config.hc_mult, config.hc_eps, config.rms_norm_eps
        self.iters = config.hc_sinkhorn_iters
        mix = (2 + self.hc) * self.hc
        self.fn = nn.Parameter(torch.empty(mix, self.hc * config.hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(mix, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def weights(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``streams (T, hc, hidden)`` -> ``pre (T, hc)``, ``post (T, hc)``, ``comb (T, hc, hc)``."""
        hc, eps = self.hc, self.eps
        flat = streams.flatten(1).float()
        mixes = F.linear(flat, self.fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_eps)
        pre = torch.sigmoid(mixes[..., :hc] * self.scale[0] + self.base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * self.scale[1] + self.base[hc:2 * hc])
        comb = (mixes[..., 2 * hc:] * self.scale[2] + self.base[2 * hc:]).unflatten(-1, (hc, hc))
        comb = comb.softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(self.iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb


def collapse(streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """``streams (T, hc, hidden)`` weighted by ``pre (T, hc)`` into one sublayer input."""
    return (pre.unsqueeze(-1) * streams.float()).sum(dim=1).to(streams.dtype)


def update(streams: torch.Tensor, out: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """A sublayer's output written back: placed by ``post``, the streams mixed by ``comb``, in
    fp32 and summed stream by stream, as the reference sums it."""
    mixed = torch.sum(comb.unsqueeze(-1) * streams.unsqueeze(-2), dim=1)
    return (post.unsqueeze(-1) * out.unsqueeze(-2) + mixed).to(out.dtype)


class DeepseekV41Block(nn.Module):
    """Attention and MoE between hyper-connection sites. A site's collapse weights are used
    one sublayer later: the attention collapses with the weights the block before it made at
    its MoE site, the MoE with the ones this block's attention site made."""

    def __init__(self, config: DeepseekV41Config, layer: int, comm_group: CommGroup | None = None,
                 moe_parallel=None) -> None:
        super().__init__()
        self.engram = Engram(config, layer, comm_group) if layer in config.engram_layer_ids else None
        self.attn = DeepseekV41Attention(config, layer, comm_group)
        self.ffn = DeepseekV41MoE(config, comm_group, moe_parallel)
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_attn = HyperConnection(config)
        self.hc_ffn = HyperConnection(config)

    def forward(self, streams: torch.Tensor, pre: torch.Tensor, rope: Rope,
                shared: SharedSelection) -> tuple[torch.Tensor, torch.Tensor]:
        """``streams (T, hc, hidden)`` and the collapse weights for this block's attention;
        returns the streams and the weights for the next block's."""
        attn_pre, post, comb = self.hc_attn.weights(streams)
        out = self.attn(self.attn_norm(collapse(streams, pre)), rope, shared)
        streams = update(streams, out, post, comb)
        ffn_pre, post, comb = self.hc_ffn.weights(streams)
        out = self.ffn(self.ffn_norm(collapse(streams, attn_pre)))
        return update(streams, out, post, comb), ffn_pre


class DeepseekV41LanguageModel(nn.Module):
    def __init__(self, config: DeepseekV41Config, token_map: torch.Tensor | None = None,
                 comm_group: CommGroup | None = None, moe_parallel=None) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.embed = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, comm_group=comm_group)
        self.layers = nn.ModuleList(
            DeepseekV41Block(config, layer, comm_group, moe_parallel) for layer in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = Rope(config)
        self.hasher = None
        if config.engram_layer_ids:
            if token_map is None:
                raise ValueError("Engram hashes compressed token ids: pass the tokenizer's token_map")
            self.hasher = NgramHasher(config, token_map)
        # ENGRAM, when the Engram tables are in host memory
        self._host_tables = None

    def bind_resources(self, resources: dict) -> None:
        self._host_tables = resources.get(ENGRAM)

    def forward(self, input_ids: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
        """``input_ids (T,)`` flat batch -> the final hidden ``(T, hidden)``, or only ``rows``."""
        hashes = self.hasher(input_ids) if self.hasher is not None else None
        if self._host_tables is not None:
            # every Engram layer's rows in one trip to the host, before the first layer is queued
            self._host_tables.fetch({
                layer.engram.host_table: hashes[:, layer.engram.index]
                for layer in self.layers if layer.engram is not None})
        hidden = self.embed(input_ids)
        streams = hidden.unsqueeze(1).repeat(1, self.hc_mult, 1)
        # the first block's attention reads the first stream alone
        pre = torch.zeros(hidden.shape[0], self.hc_mult, dtype=torch.float32, device=hidden.device)
        pre[:, 0] = 1.0
        shared = SharedSelection()
        for layer in self.layers:
            if layer.engram is not None:
                streams = layer.engram(streams, hashes[:, layer.engram.index])
            streams, pre = layer(streams, pre, self.rope, shared)
        hidden = collapse(streams, pre)
        if rows is not None:
            hidden = hidden.index_select(0, rows)
        return self.norm(hidden)


class DeepseekV41ForCausalLM(nn.Module):
    def __init__(self, config: DeepseekV41Config, token_map: torch.Tensor | None = None,
                 comm_group: CommGroup | None = None, moe_parallel=None) -> None:
        super().__init__()
        self.config = config
        self.comm_group = comm_group or CommGroup.trivial()
        self.model = DeepseekV41LanguageModel(config, token_map, comm_group, moe_parallel)
        self.lm_head = ColumnParallelLinear(
            self.comm_group, input_size=config.hidden_size, output_size=config.vocab_size, bias=False)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """fp32 logits, summed in fp32 from the head's stored weight."""
        local = F.linear(hidden.float(), self.lm_head.weight.float())
        if self.comm_group.world_size > 1:
            local = self.comm_group.all_gather(local, dim=-1)
        return local

    def forward(self, input_ids: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
        return self.logits(self.model(input_ids, rows))

    def load_weights(self, weights, sliced: bool = False, **kwargs) -> set[str]:
        from mstar.model.deepseek_v41.weight_loader import load_deepseek_v41_weights

        return load_deepseek_v41_weights(self, weights, sliced)
