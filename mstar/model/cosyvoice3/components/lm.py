# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
# Copyright (c) 2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li, Qihua, Shengqiang Li)
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: the Qwen2 speech-token LM
# rebuilt on M* components over a paged KV cache.
#
# Original file (FunAudioLLM/CosyVoice, cosyvoice/llm/llm.py) was released
# under Apache-2.0, with the full license text available at
# https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""CosyVoice3's speech-token LM on M* components.

A Qwen2 0.5B decoder (``CosyVoice-BlankEN``: RMSNorm, GQA with biased q/k/v,
SwiGLU, plain RoPE at theta 1e6) reads
``[sos | text tokens | task_id | prompt speech tokens]`` and predicts the next
speech token through its own head over ``speech_token_size + 200`` ids, the
200 above the codebook being control tokens (any of them ends the utterance).
``sos`` and ``task_id`` are rows of the speech embedding.

Ported from ``cosyvoice/llm/llm.py`` (``Qwen2LM``, ``CosyVoice3LM``) and the HF
Qwen2 implementation. Parameter paths drop the checkpoint's
``llm.model.model.`` prefix; the backbone's text head is unused and dropped.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.chatterbox.loader import load_component
from mstar.model.components import DecoderLayer, RMSNorm
from mstar.model.components.distributed import ParallelAttention, ParallelGatedMLP
from mstar.model.cosyvoice3.config import LM_ATTN, LM_KV, LM_POS, LMConfig
from mstar.model.loader import LLAMA_STACKED_PARAMS


class CosyVoice3LM(nn.Module):
    lora_unservable = {"llm_decoder": "run as F.linear over its float32 weight (logits), never its own forward"}

    def __init__(self, config: LMConfig, comm_group: CommGroup | None = None):
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        self.config = config
        self.embed_tokens = nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    self_attn=ParallelAttention(
                        comm_group=comm_group,
                        hidden_size=config.num_attention_heads * config.head_dim,
                        num_heads=config.num_attention_heads,
                        num_kv_heads=config.num_key_value_heads,
                        head_dim=config.head_dim,
                        qkv_bias=True,
                        rope_theta=config.rope_theta,
                        input_hidden_size=config.hidden_size,
                        attn_key=LM_ATTN,
                        kv_key=LM_KV,
                        pos_key=LM_POS,
                    ),
                    mlp=ParallelGatedMLP(
                        comm_group=comm_group,
                        hidden_size=config.hidden_size,
                        intermediate_size=config.intermediate_size,
                        activation="silu",
                    ),
                    input_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                    post_attention_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.speech_embedding = nn.Embedding(config.speech_vocab_size, config.hidden_size)
        self.llm_decoder = nn.Linear(config.hidden_size, config.speech_vocab_size, bias=False)

    # -- embeddings ---------------------------------------------------------

    def build_prefill_embeds(self, text_ids: torch.Tensor, prompt_speech_tokens: torch.Tensor) -> torch.Tensor:
        """One request's prefill ``[sos | text | task_id | prompt speech]``,
        ``text_ids`` being the prompt text and the text to speak, concatenated."""
        cfg = self.config
        device = text_ids.device
        # rows of the table, not a lookup: an id tensor would be a copy from the host per request
        table = self.speech_embedding.weight
        sos, task_id = table[cfg.sos_token : cfg.sos_token + 1], table[cfg.task_id_token : cfg.task_id_token + 1]
        text = self.embed_tokens(text_ids)
        prompt = self.speech_embedding(prompt_speech_tokens.to(device, torch.long))
        return torch.cat([sos, text, task_id, prompt], dim=0)

    def embed_speech(self, ids: torch.Tensor) -> torch.Tensor:
        return self.speech_embedding(ids)

    # -- transformer --------------------------------------------------------

    def hidden(self, input_embeds: torch.Tensor, *, label: str) -> torch.Tensor:
        """``[N, D] -> [N, D]`` over the packed step. The label and the layer
        index are cursors on the shared resources: bind once, advance per layer."""
        hidden = input_embeds
        self.layers[0].self_attn.attend.bind_step(label)
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.attend.set_layer_idx(layer_idx)
            hidden = layer(hidden)
        return self.norm(hidden)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """float32 logits over the speech vocabulary (the head is kept in float32;
        sampling a bfloat16 head shifts the nucleus). The engine runs the LM
        under bfloat16 autocast, which would recast this matmul."""
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return F.linear(hidden.float(), self.llm_decoder.weight.float())

    def keep_head_in_float32(self) -> None:
        self.llm_decoder.float()

    # -- weights ------------------------------------------------------------

    @staticmethod
    def _remap(name: str) -> str | None:
        if name.startswith("llm.model.lm_head."):
            return None
        if name.startswith("llm.model.model."):
            return name[len("llm.model.model.") :]
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream ``llm.pt`` (or ``llm.rl.pt``) into this module, exactly."""
        return load_component(
            self,
            weights,
            component="CosyVoice3 LM",
            name_remapper=self._remap,
            stacked_params=LLAMA_STACKED_PARAMS,
        )
