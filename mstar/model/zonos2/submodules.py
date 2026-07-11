"""Engine submodules for Zonos2 TTS.

* :class:`Zonos2LLMSubmodule` — the autoregressive multi-codebook decoder.
  It runs :class:`Zonos2ForCausalLM`, samples a full frame per step with the
  custom multi-codebook sampler (so the engine's single-token sampler is
  bypassed — ``forward`` returns ``new_token`` directly, not ``logits``),
  tracks per-request repetition history, and detects EOS across the delayed
  codebooks in ``check_stop``.

* :class:`Zonos2DACSubmodule` — a stateless audio-codec node that consumes
  streamed frames and emits PCM via the DAC vocoder.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.kv_cache_engine import BatchedCacheManager
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.model.zonos2.tts_sampling import TTSSamplingParams, sample_frame
from mstar.model.zonos2.vocoder import StreamingDacDecoder


class Zonos2LLMSubmodule(ARNodeSubmodule):
    """Autoregressive multi-codebook LLM wrapper.

    Dispatches prefill / decode the same way (embed frames -> transformer ->
    sample the last position's per-codebook logits). Returns ``new_token``:
    the sampled frame ``(1, n_codebooks + 1)``.
    """

    def __init__(
        self,
        model: nn.Module,
        n_codebooks: int,
        text_vocab: int,
        eoa_id: int,
        params: TTSSamplingParams,
    ):
        super().__init__()
        self.model = model
        self.n_codebooks = n_codebooks
        self.text_vocab = text_vocab
        self.eoa_id = eoa_id
        self.params = params

        # Per-request state.
        self._history: dict[str, torch.Tensor] = {}   # (N, n_codebooks) recent codes
        self._eos: dict[str, dict] = {}               # EOS countdown tracking
        self._generators: dict[str, torch.Generator] = {}

    # -- input plumbing ------------------------------------------------
    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        seen_token_mask=None,
        pos_info: dict = {},
        **kwargs,
    ) -> ARNodeInputs:
        ids = inputs["text_inputs"][0]  # (num_frames, n_codebooks + 1)
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
        cache_manager.plan_attention(seq_lens=seq_lens, is_causal=True, label="main")
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")
        input_ids = torch.cat([inp.input_ids for inp in inputs], dim=0).to(
            device=self.get_device(), dtype=torch.long
        )
        return {"input_ids": input_ids}

    # -- forward + sampling --------------------------------------------
    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        cache_handle: BatchedCacheManager = engine_inputs.cache_manager
        hidden = self.model(input_ids, cache_handle)          # (num_frames, hidden)
        logits = self.model.compute_logits(hidden[-1:])       # (1, C, V)

        rid = engine_inputs.request_ids[0]
        frame = sample_frame(
            logits,
            self.params,
            repetition_token_ids=self._rep_ids(rid),
            text_placeholder=self.text_vocab,
            generator=self._generator(rid, logits.device),
        )                                                     # (1, C + 1)
        self._append_history(rid, frame)
        return {"new_token": [frame]}

    # -- repetition history / RNG --------------------------------------
    def _rep_ids(self, rid: str) -> torch.Tensor | None:
        hist = self._history.get(rid)
        if (
            hist is None
            or self.params.repetition_window <= 0
            or self.params.repetition_penalty == 1.0
        ):
            return None
        window = hist[-self.params.repetition_window:]        # (w, C)
        ids = window.t().unsqueeze(0).contiguous()            # (1, C, w)
        rc = self.params.repetition_codebooks
        if 0 <= rc < self.n_codebooks:
            ids = ids.clone()
            ids[:, rc:, :] = -1  # codebooks past rc are excluded from the penalty
        return ids

    def _append_history(self, rid: str, frame: torch.Tensor) -> None:
        codes = frame[:, : self.n_codebooks]                 # (1, C)
        prev = self._history.get(rid)
        self._history[rid] = codes if prev is None else torch.cat([prev, codes], dim=0)

    def _generator(self, rid: str, device) -> torch.Generator | None:
        if self.params.seed is None:
            return None
        gen = self._generators.get(rid)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(self.params.seed))
            self._generators[rid] = gen
        return gen

    # -- graph routing + stop ------------------------------------------
    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # Feed the sampled frame back as the next decode input. Metadata-only
        # (no tensor value reads on the GPU thread).
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
        frame = outputs["new_token"][0].flatten().tolist()
        audio = frame[: self.n_codebooks]

        st = self._eos.setdefault(
            request_id, {"step": -1, "eos_frame": None, "countdown": 0}
        )
        st["step"] += 1
        # EOS detection matches the reference (zonos2 ``tts/sequence.py``): the
        # first frame in which *any* codebook emits eoa starts a delayed stop
        # countdown. The aligned end frame is shifted back by the highest eoa
        # codebook index (that codebook is delayed by its index under the
        # inter-codebook shear) and clamped at zero.
        if not self.params.ignore_eos and st["eos_frame"] is None:
            eos_cols = [i for i in range(self.n_codebooks) if audio[i] == self.eoa_id]
            if eos_cols:
                st["eos_frame"] = max(0, st["step"] - max(eos_cols))
                st["countdown"] = self.n_codebooks + 1
        if st["eos_frame"] is not None and st["countdown"] > 0:
            st["countdown"] -= 1

        finished = st["eos_frame"] is not None and st["countdown"] <= 0
        max_tokens = getattr(request_info, "max_tokens", None) or self.params.max_tokens
        if request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1 >= max_tokens:
            finished = True
        return {"decode_loop"} if finished else set()

    def cleanup_request(self, request_id: str):
        self._history.pop(request_id, None)
        self._eos.pop(request_id, None)
        self._generators.pop(request_id, None)


class Zonos2DACSubmodule(NodeSubmodule):
    """Stateless DAC vocoder node.

    Consumes streamed frames (per request) and emits int16 PCM chunks. Runs
    incrementally via :class:`StreamingDacDecoder`. On the final
    chunk (``request_id in engine_inputs.final_stream_rids``) it flushes the
    withheld crossfade tail; the trailing ``n_codebooks - 1`` shear-alignment
    frames carry no audio of their own and are dropped.
    """

    def __init__(self, decoder: StreamingDacDecoder, n_codebooks: int):
        super().__init__()
        self.decoder = decoder
        self.n_codebooks = n_codebooks
        self.frame_width = n_codebooks + 1
        # Marker parameter so ``get_device`` / ``.to(device)`` work (the DAC
        # model itself is loaded lazily inside the decoder).
        self._device_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def get_stateless_flavor(self) -> str:
        return "audio_codec"  # fp32, no autocast, no torch.compile

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        streamed = inputs.get("new_token", [])
        if not streamed or streamed[0] is None:
            frames = torch.empty(0, self.frame_width, dtype=torch.long)
        else:
            frames = streamed[0].reshape(-1, self.frame_width)
        return NodeInputs(tensor_inputs={"frames": frames})

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        return {"frames": inputs[0].tensor_inputs["frames"]}

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        frames: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        rid = engine_inputs.request_ids[0]
        audio_codes = frames[:, : self.n_codebooks]
        is_final = rid in engine_inputs.final_stream_rids
        pcm = self.decoder.add_frames(rid, audio_codes, is_final=is_final)
        return {"audio_chunk": [pcm]}

    def cleanup_request(self, request_id: str):
        self.decoder.reset(request_id)
