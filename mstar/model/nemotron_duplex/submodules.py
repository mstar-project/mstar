"""NodeSubmodules for NVIDIA NemotronLabs VoiceChat-11B (duplex).

Nodes:
    nano_llm         (KV_CACHE)  — Nemotron-H hybrid backbone      [implemented]
    conformer_encoder(STATELESS) — Fast-Conformer STT + RNN-T head [Phase 4 stub]
    eartts_talker    (KV_CACHE)  — Gemma3 talker → RVQ codes        [Phase 5 stub]
    audio_codec      (STATELESS) — RVQ codec → 22.05 kHz waveform   [Phase 5 stub]

Per the agreed phasing, the ``nano_llm`` text path is drafted first; the audio
stages carry correct interfaces + TODOs and are only wired once the LLM matches
the HF reference numerically.

Correctness note for ``nano_llm``: it runs eager (``disable_torch_compile``)
because Option-A Mamba state lives in ``per_request_states`` (None during
CUDA-graph capture). Batching + CUDA graphs come with the Option-B SSM pool.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.kv_store import PositionInfo
from mstar.model.nemotron_duplex.components.nemotron_h import MambaStateAccessor
from mstar.model.nemotron_duplex.config import NemotronDuplexConfig
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)

# Walks in which the nano LLM is (re)starting its recurrent state.
PREFILL_WALKS = {"prefill_text", "prefill_audio"}


class NemotronHLLMSubmodule(ARNodeSubmodule):
    """Nemotron-H backbone node.

    ``prefill_*`` embeds the prompt (or consumes fused ``combined_embeds`` from
    the encoder) and fills the attention KV cache + seeds Mamba state; ``decode``
    single-steps. Sampling is the engine's; EOS handling is in ``check_stop``.
    """

    # Option-A Mamba state is a Python-dict lookup keyed by request id — not
    # compile-/capture-safe. Keep this node eager until the Option-B pool lands.
    disable_torch_compile = True

    def __init__(self, language_model: nn.Module, config: NemotronDuplexConfig):
        super().__init__()
        self.language_model = language_model
        self.embeddings = language_model.embeddings
        self.lm_head = language_model.lm_head
        self.config = config

    def _mamba_state(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, seq_lens: list | None = None,
    ) -> MambaStateAccessor:
        return MambaStateAccessor(
            request_states=engine_inputs.per_request_states or {},
            request_ids=list(engine_inputs.request_ids),
            is_prefill=graph_walk in PREFILL_WALKS,
            seq_lens=seq_lens,
        )

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        pos_info: dict[str, PositionInfo] = {},  # noqa: B006 - matches base signature
        **kwargs,
    ) -> ARNodeInputs:
        # ``combined_embeds`` (fused audio+text) takes precedence over token ids
        # on the voicechat audio path; the text path supplies ``text_inputs``.
        if "combined_embeds" in inputs:
            emb = inputs["combined_embeds"][0]
            return ARNodeInputs(input_embeds=emb, input_seq_len=emb.shape[0])
        ids = inputs["text_inputs"][0]
        return ARNodeInputs(input_ids=ids, input_seq_len=ids.shape[0])

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        cache_manager = engine_inputs.cache_manager
        seq_lens = [inp.input_seq_len for inp in inputs]
        cache_manager.set_active_label("main")
        # NoPE: plan attention only — no plan_rope (Nemotron-H attention layers
        # apply no positional encoding).
        cache_manager.plan_attention(seq_lens=seq_lens, is_causal=True, label="main")

        out: dict[str, torch.Tensor | Any] = {"seq_lens": seq_lens}
        if inputs[0].input_embeds is not None:
            out["input_embeds"] = torch.cat([inp.input_embeds for inp in inputs], dim=0)
        else:
            out["input_ids"] = torch.cat([inp.input_ids for inp in inputs], dim=0)
        return out

    def can_batch(self, batch, model_inputs) -> bool:
        # Eager (disable_torch_compile) batched decode/prefill: attention batches via
        # the paged-KV pool, Mamba conv/ssm state via the per-request MambaStateAccessor.
        return len(model_inputs) > 1

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        seq_lens: list | None = None,
        **kwargs,
    ) -> NameToTensorList:
        cache_handle = engine_inputs.cache_manager
        if input_embeds is None:
            input_embeds = self.embeddings(input_ids)
        hidden = self.language_model(
            input_embeds,
            cache_handle=cache_handle,
            mamba_state=self._mamba_state(graph_walk, engine_inputs, seq_lens),
        )
        logits = self.lm_head(hidden[-1:])
        return {"logits": [logits]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        seq_lens: list | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """One fused forward advancing every request in the batch. Attention uses the
        planned paged-KV layout; Mamba conv/ssm state is stepped per request from the
        packed rows (see ``Mamba2Mixer.forward`` / ``MambaStateAccessor``). Per-request
        next-token logits are read at each request's last packed position."""
        import itertools

        cache_handle = engine_inputs.cache_manager
        if input_embeds is None:
            input_embeds = self.embeddings(input_ids)
        hidden = self.language_model(
            input_embeds,
            cache_handle=cache_handle,
            mamba_state=self._mamba_state(graph_walk, engine_inputs, seq_lens),
        )
        ends = list(itertools.accumulate(seq_lens))          # last index+1 per request
        result: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(engine_inputs.request_ids):
            last = hidden[ends[i] - 1: ends[i]]              # (1, H) request i's last token
            result[rid] = {"logits": [self.lm_head(last)]}
        return result

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # Rebind the sampled token for loop-back routing (metadata only).
        if "new_token" in outputs:
            outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        token = outputs["new_token"][0].item()
        ignore_eos = request_info.sampling_config["nano_llm"].ignore_eos
        at_max = (
            request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
            >= request_info.max_tokens
        )
        if (not ignore_eos and token == self.config.eos_token_id) or at_max:
            return {"decode_loop"}
        return set()

    def cleanup_request(self, request_id: str):
        # Drop this request's Mamba conv/ssm state (base clears request_states).
        super().cleanup_request(request_id)


# ---------------------------------------------------------------------------
# Audio stages — Phase 4/5 stubs (interfaces fixed; internals TODO)
# ---------------------------------------------------------------------------

class ConformerEncoderSubmodule(NodeSubmodule):
    """Fast-Conformer streaming STT encoder (audio in → fused embeds + transcript).

    STATELESS node. Holds the perception stack plus the RNN-T decoder/joint —
    all three checkpoint subtrees are loaded (weight loading verified). Produces
    ``combined_embeds`` for the nano LLM and a streaming ``transcript_token``.
    Forward is Phase 4 (audio-in).
    """

    def __init__(self, perception: nn.Module, rnnt_decoder: nn.Module, rnnt_joint: nn.Module,
                 config: NemotronDuplexConfig):
        super().__init__()
        self.perception = perception
        self.rnnt_decoder = rnnt_decoder
        self.rnnt_joint = rnnt_joint
        self.config = config

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        raise NotImplementedError("ConformerEncoderSubmodule.forward — Phase 4 (audio-in).")

    def forward(self, graph_walk, engine_inputs, **kwargs) -> NameToTensorList:
        raise NotImplementedError("ConformerEncoderSubmodule.forward — Phase 4 (audio-in).")


class EarTTSTalkerSubmodule(ARNodeSubmodule):
    """Gemma3-text talker: nano hidden/text → RVQ codec codes (autoregressive).

    KV_CACHE node. Emits ``num_quantizers`` codebooks per frame; the extra
    codebooks sample from a ``code_predictor`` aux config (see
    Model.get_aux_sampling_configs), mirroring Qwen3-Omni's Talker. Weights
    loaded (verified); forward is Phase 5 (audio-out).
    """

    def __init__(self, talker: nn.Module, config: NemotronDuplexConfig):
        super().__init__()
        self.talker = talker
        self.config = config

    def prepare_inputs(self, graph_walk, fwd_info, inputs, seen_token_mask=None, pos_info={}, **kwargs):  # noqa: B006
        raise NotImplementedError("EarTTSTalkerSubmodule.forward — Phase 5 (audio-out).")

    def preprocess(self, graph_walk, engine_inputs, inputs):
        raise NotImplementedError("EarTTSTalkerSubmodule.forward — Phase 5 (audio-out).")

    def forward(self, graph_walk, engine_inputs, **kwargs) -> NameToTensorList:
        raise NotImplementedError("EarTTSTalkerSubmodule.forward — Phase 5 (audio-out).")


class AudioCodecDecoderSubmodule(NodeSubmodule):
    """RVQ codec decoder: talker codes → 22.05 kHz PCM (sliding-window stream).

    Weights loaded (verified); forward is Phase 5 (audio-out).
    """

    def __init__(self, codec: nn.Module, config: NemotronDuplexConfig):
        super().__init__()
        self.codec = codec
        self.config = config

    def get_stateless_flavor(self) -> str:
        return "audio_codec"

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        raise NotImplementedError("AudioCodecDecoderSubmodule.forward — Phase 5 (audio-out).")

    def forward(self, graph_walk, engine_inputs, **kwargs) -> NameToTensorList:
        raise NotImplementedError("AudioCodecDecoderSubmodule.forward — Phase 5 (audio-out).")
