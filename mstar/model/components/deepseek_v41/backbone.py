"""Standalone DeepSeek-V4.1 text backbone and causal LM for CPU verification and engine reuse."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.decoder_layer import eager_frame
from mstar.model.components.deepseek_v41.attention import DeepSeekV41Attention, SharedSelection
from mstar.model.components.deepseek_v41.config import DeepSeekV41Config
from mstar.model.components.deepseek_v41.engram import DeepSeekV41Engram
from mstar.model.components.deepseek_v41.engram_hash import EngramLayout, compute_engram_hashes
from mstar.model.components.deepseek_v41.layer import DeepSeekV41Layer
from mstar.model.components.deepseek_v41.moe import DeepSeekV41MoE
from mstar.model.components.deepseek_v41.quant import ReferenceRMSNorm
from mstar.model.components.deepseek_v41.rope import DeepSeekV41Rope
from mstar.model.components.distributed import ColumnParallelLinear, VocabParallelEmbedding


class DeepSeekV41TextCache:
    """Per-batch mutable KV, compressed cache, compressor partial, and Engram tail state."""

    def __init__(
        self,
        config: DeepSeekV41Config,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.windows: dict[int, torch.Tensor] = {
            layer_id: torch.zeros((batch_size, config.sliding_window, config.head_dim), device=device, dtype=dtype)
            for layer_id in range(config.num_hidden_layers)
        }
        self.compress_kv: dict[int, torch.Tensor] = {
            layer_id: torch.zeros(
                (batch_size, max(1, max_seq_len // config.compress_ratios[layer_id]), config.head_dim),
                device=device,
                dtype=dtype,
            )
            for layer_id in config.kv_source_layer_ids
        }
        self.compress_k: dict[int, torch.Tensor] = {
            layer_id: torch.zeros(
                (batch_size, max(1, max_seq_len // config.compress_ratios[layer_id]), config.index_head_dim),
                device=device,
                dtype=dtype,
            )
            for layer_id in config.kv_source_layer_ids
        }
        self.partials: dict[int, tuple[torch.Tensor, torch.Tensor]] = {
            layer_id: (
                torch.zeros(
                    (batch_size, config.compress_ratios[layer_id], config.head_dim),
                    device=device,
                    dtype=torch.float32,
                ),
                torch.full(
                    (batch_size, config.compress_ratios[layer_id], config.head_dim),
                    -torch.inf,
                    device=device,
                    dtype=torch.float32,
                ),
            )
            for layer_id in config.pooled_source_layer_ids
        }
        self.engram_tail: torch.Tensor | None = (
            torch.full(
                (batch_size, config.engram_max_ngram_size - 1),
                -1,
                device=device,
                dtype=torch.long,
            )
            if config.engram_layer_ids
            else None
        )
        self._current_layer: int | None = None

    @torch.compiler.disable
    def bind_layer(self, layer_id: int) -> None:
        self._current_layer = layer_id

    @torch.compiler.disable
    def read_layer(
        self,
        layer_id: int | None,
        src_layer: int | None,
        has_partial: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        lid = self._current_layer if layer_id is None else layer_id
        if lid is None:
            raise RuntimeError("cache.bind_layer(layer_id) must be called before layer forward")
        return (
            self.windows[lid],
            self.compress_kv[src_layer] if src_layer is not None else None,
            self.compress_k[src_layer] if src_layer is not None else None,
            self.partials.get(lid) if has_partial else None,
        )

    @torch.compiler.disable
    def write_layer(
        self,
        layer_id: int | None,
        src_layer: int | None,
        has_compressor: bool,
        next_win: torch.Tensor,
        next_ckv: torch.Tensor | None,
        next_ck: torch.Tensor | None,
        next_partial: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        lid = self._current_layer if layer_id is None else layer_id
        if lid is None:
            raise RuntimeError("cache.bind_layer(layer_id) must be called before layer forward")
        self.windows[lid] = next_win
        if src_layer is not None and has_compressor:
            assert next_ckv is not None and next_ck is not None
            self.compress_kv[src_layer] = next_ckv
            self.compress_k[src_layer] = next_ck
            if next_partial is not None:
                self.partials[lid] = next_partial
        if layer_id is None:
            self._current_layer = None


class DeepSeekV41TextBackbone(nn.Module):
    """DeepSeek-V4.1-Flash text transformer stack matching checkpoint parameter layout."""

    def __init__(
        self,
        config: DeepSeekV41Config,
        *,
        token_map: torch.Tensor | None = None,
        comm_group: CommGroup | None = None,
        engram_wkv_reference: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.comm_group = comm_group or CommGroup.trivial()
        self.hc_mult = config.hc_mult
        self.embed = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            comm_group=self.comm_group,
        )
        moe_cfg = config.moe_config()
        self.layers = nn.ModuleList(
            DeepSeekV41Layer(
                layer_id,
                config.hidden_size,
                DeepSeekV41Attention(config, layer_id, self.comm_group),
                DeepSeekV41MoE(layer_id, moe_cfg, self.comm_group),
                hc_mult=config.hc_mult,
                norm_eps=config.rms_norm_eps,
                hc_eps=config.hc_eps,
                hc_sinkhorn_iters=config.hc_sinkhorn_iters,
            )
            for layer_id in range(config.num_hidden_layers)
        )
        self.engrams = nn.ModuleDict(
            {
                str(layer_id): DeepSeekV41Engram(
                    num_embeddings,
                    config.engram_head_dim,
                    config.engram_hash_cols,
                    config.hidden_size,
                    config.hc_mult,
                    comm_group=self.comm_group,
                    norm_eps=config.rms_norm_eps,
                    wkv_reference=engram_wkv_reference,
                )
                for layer_id, num_embeddings in zip(
                    config.engram_layer_ids, config.engram_num_embeddings, strict=True
                )
            }
        )
        self.norm = ReferenceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ColumnParallelLinear(
            self.comm_group,
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
        )
        self.rope = DeepSeekV41Rope(config)
        self.engram_layout = EngramLayout.from_config(config, token_map) if config.engram_layer_ids else None
        self._engram_index = {layer_id: idx for idx, layer_id in enumerate(config.engram_layer_ids)}

    @property
    def embed_dtype(self) -> torch.dtype:
        return self.embed.weight.dtype

    def new_cache(
        self,
        batch_size: int,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> DeepSeekV41TextCache:
        dev = torch.device(device) if device is not None else self.embed.weight.device
        dt = dtype if dtype is not None else self.embed.weight.dtype
        return DeepSeekV41TextCache(
            self.config,
            batch_size,
            max_seq_len or self.config.max_seq_len,
            dev,
            dt,
        )

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        local = F.linear(hidden.float(), self.head.weight.float())
        if self.comm_group.world_size > 1:
            local = self.comm_group.all_gather(local, dim=-1)
        return local

    @eager_frame
    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        cache: DeepSeekV41TextCache | None = None,
        token_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run standalone ``[B, S]`` causal LM forward returning final normalized hidden states."""
        if cache is None:
            cache = self.new_cache(input_ids.shape[0], start_pos + input_ids.shape[1], input_ids.device)
        hashes = None
        if self.engram_layout is not None:
            hashes, next_tail = compute_engram_hashes(
                input_ids,
                self.engram_layout,
                prev_tail=cache.engram_tail,
                token_mask=token_mask,
            )
            cache.engram_tail = next_tail

        hidden = self.embed(input_ids)
        streams = hidden.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre = torch.zeros(hidden.shape[0], hidden.shape[1], self.hc_mult, dtype=torch.float32, device=hidden.device)
        pre[:, :, 0] = 1.0
        shared = SharedSelection()
        for layer_id, layer in enumerate(self.layers):
            cache.bind_layer(layer_id)
            if layer_id in self._engram_index and hashes is not None:
                idx = self._engram_index[layer_id]
                streams = self.engrams[str(layer_id)](streams, hashes[:, :, idx, :], token_mask=token_mask)
            streams, pre = layer(streams, start_pos, pre, image_mask, self.rope, cache, shared)
        final_hidden = DeepSeekV41Layer._collapse(streams, pre)
        return self.norm(final_hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        cache: DeepSeekV41TextCache | None = None,
        token_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.logits(
            self.forward_hidden(
                input_ids,
                start_pos=start_pos,
                cache=cache,
                token_mask=token_mask,
                image_mask=image_mask,
            )
        )

    def process_weights_after_loading(self, device: torch.device | str) -> None:
        for layer in self.layers:
            layer.ffn.process_weights_after_loading(device)
