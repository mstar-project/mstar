"""The Kokoro synthesis node: one sentence chunk per loop iteration, batched
across requests.

``process_prompt`` hands every request all of its chunks at once (padded
``[n_chunks, T]`` phoneme ids, lengths, one style row per chunk, speed). Each
iteration of the ``chunk_loop`` synthesizes chunk ``i`` for every request in
the batch and emits its PCM to the client; ``check_stop`` ends a request's
loop after its last chunk.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import ExecutingBatch
from mstar.model.kokoro.components import KokoroTTS
from mstar.model.kokoro.config import (
    AUDIO_CHUNK,
    BOUNDARY_TOKEN_ID,
    CHUNK_LOOP,
    PHONEME_IDS,
    PHONEME_LENS,
    REF_STYLE,
    SPEED,
    KokoroModelConfig,
)
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)

PCM16_SCALE = 32767


class KokoroSynthSubmodule(NodeSubmodule):
    """Phonemes + style -> 24 kHz PCM16 for one chunk of every request."""

    # The forward reads the frame count on the host and shapes the decoder by
    # it, so the whole method is not compilable; the two static halves are
    # captured piecewise instead (see ``KokoroTTS``).
    disable_torch_compile = True
    # fp32 end to end, as the reference: the phase-sensitive vocoder does not
    # tolerate reduced precision.
    disable_autocast = True

    def __init__(self, model: KokoroTTS, config: KokoroModelConfig):
        super().__init__()
        self.model = model
        self.config = config

    # -- per request -------------------------------------------------------

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs | None:
        """Select this iteration's chunk; ``None`` once the request has none left."""
        del graph_walk, kwargs
        state = self.request_state(fwd_info.request_id)
        if "chunk_lengths" not in state:
            # One host read per request: the chunk lengths size every later slice.
            state.add("chunk_lengths", inputs[PHONEME_LENS][0].tolist())
        chunk_lengths: list[int] = state["chunk_lengths"]
        iteration = fwd_info.dynamic_loop_iter_counts.get(CHUNK_LOOP, 0)
        if iteration >= len(chunk_lengths):
            return None
        length = chunk_lengths[iteration]
        return NodeInputs(
            tensor_inputs={
                "input_ids": inputs[PHONEME_IDS][0][iteration, :length],
                "style": inputs[REF_STYLE][0][iteration],
                "speed": inputs[SPEED][0].reshape(()),
            },
            input_seq_len=length,
        )

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        del outputs
        num_chunks = len(self.request_state(request_id).get("chunk_lengths", ()))
        if request_info.dynamic_loop_iter_counts.get(CHUNK_LOOP, 0) + 1 >= num_chunks:
            return {CHUNK_LOOP}
        return set()

    # -- per batch -----------------------------------------------------------

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        del batch, model_inputs
        return True

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor]:
        """Pad the chunks of the batch to a common phoneme length."""
        del graph_walk, engine_inputs
        device = self.get_device()
        return {
            "input_ids": pad_sequence(
                [inp.tensor_inputs["input_ids"].to(device) for inp in inputs],
                batch_first=True,
                padding_value=BOUNDARY_TOKEN_ID,
            ),
            "lengths": torch.tensor([inp.input_seq_len for inp in inputs], dtype=torch.long, device=device),
            "style": torch.stack([inp.tensor_inputs["style"].to(device) for inp in inputs]),
            "speed": torch.stack([inp.tensor_inputs["speed"].to(device) for inp in inputs]),
        }

    def _synthesize(
        self, input_ids: torch.Tensor, lengths: torch.Tensor, style: torch.Tensor, speed: torch.Tensor
    ) -> list[torch.Tensor]:
        audio, frame_lengths, _ = self.model(input_ids, lengths, style, speed)
        pcm = (audio.clamp(-1, 1) * PCM16_SCALE).to(torch.int16)
        sizes = (frame_lengths * self.config.samples_per_frame).tolist()
        return [pcm[i, :n] for i, n in enumerate(sizes)]

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        style: torch.Tensor,
        speed: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        return {AUDIO_CHUNK: [self._synthesize(input_ids, lengths, style, speed)[0]]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        style: torch.Tensor,
        speed: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, NameToTensorList]:
        del graph_walk, kwargs
        chunks = self._synthesize(input_ids, lengths, style, speed)
        return {rid: {AUDIO_CHUNK: [pcm]} for rid, pcm in zip(engine_inputs.request_ids, chunks, strict=True)}
