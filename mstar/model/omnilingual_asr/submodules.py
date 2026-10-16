"""The Omnilingual ASR CTC node: waveform -> collapsed CTC token ids.

One stateless forward per request: the wav2vec 2.0 encoder, the CTC head, a
per-frame argmax over each request's valid frames and the collapse of
repeated frames, all on the device. The blank and the other control ids stay
in the emitted ids; ``OmnilingualASRModel.postprocess`` drops them when it
turns ids into text, as the reference's decoder does. So a request always
emits at least one id (one per distinct run of frames), even for silence.

Concurrent requests are batched (up to ``max_batch_size``). Inside a step
the rows are sorted by length and cut into sub-batches whose padded audio
(rows times the longest) stays within ``max_batch_seconds``; each sub-batch
is right-padded, with each row's length masking its padding (see
``components/wav2vec2.py``). Sorting keeps a short clip from being padded to
a long one, and the budget bounds a forward's memory.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import ExecutingBatch
from mstar.model.omnilingual_asr.components.wav2vec2 import Wav2Vec2CTC
from mstar.model.omnilingual_asr.config import SAMPLE_RATE, TOKEN_IDS, WAVEFORM
from mstar.model.omnilingual_asr.tokenizer import ctc_collapse
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule


@contextmanager
def _full_fp32_matmuls(enabled: bool):
    """The engine runs fp32 matmuls as TF32 (``set_float32_matmul_precision('high')``);
    an fp32 deployment of this model asks for full fp32, which is what the
    reference computes."""
    if not enabled:
        yield
        return
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


class OmniASRCTCSubmodule(NodeSubmodule):
    # Every request has its own length, so a compiled forward would recompile
    # per shape; the encoder runs eagerly.
    disable_torch_compile = True
    # The model runs in its own dtype end to end, as the reference does (only
    # the conv stack's LayerNorms upcast to fp32), not under autocast.
    disable_autocast = True

    def __init__(self, model: Wav2Vec2CTC, max_batch_size: int, max_batch_seconds: float):
        super().__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self._max_batch_size = max_batch_size
        self._max_batch_samples = int(max_batch_seconds * SAMPLE_RATE)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        del graph_walk, fwd_info, kwargs
        waveform = inputs[WAVEFORM][0]
        return NodeInputs(tensor_inputs={WAVEFORM: waveform}, input_seq_len=int(waveform.shape[-1]))

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self._max_batch_size

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        """Any set batches: ``forward_batched`` splits it to fit the audio budget."""
        del batch
        return len(model_inputs) > 1

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        del graph_walk, engine_inputs
        device = self.get_device()
        return {"waveforms": [inp.tensor_inputs[WAVEFORM].reshape(-1).to(device, self.dtype) for inp in inputs]}

    def sub_batches(self, lengths: list[int]) -> list[list[int]]:
        """Row indices, shortest first, cut so that each group's row count times
        its longest row stays within the audio budget (a row alone always fits)."""
        groups: list[list[int]] = []
        for row in sorted(range(len(lengths)), key=lengths.__getitem__):
            if groups and (len(groups[-1]) + 1) * lengths[row] <= self._max_batch_samples:
                groups[-1].append(row)
            else:
                groups.append([row])
        return groups

    def _transcribe(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        """Collapsed argmax ids per row, over the row's valid frames."""
        lengths = [w.shape[0] for w in waveforms]
        out: list[torch.Tensor | None] = [None] * len(waveforms)
        for group in self.sub_batches(lengths):
            longest = max(lengths[row] for row in group)
            batch = waveforms[group[0]].new_zeros(len(group), longest)
            for i, row in enumerate(group):
                batch[i, : lengths[row]] = waveforms[row]
            padded = any(lengths[row] < longest for row in group)
            seq_lens = torch.tensor([lengths[row] for row in group], device=batch.device) if padded else None
            with _full_fp32_matmuls(self.dtype == torch.float32 and batch.is_cuda):
                logits, frame_lens = self.model(batch, seq_lens)
            ids = logits.argmax(dim=-1)
            for i, (row, n) in enumerate(zip(group, frame_lens.tolist(), strict=True)):
                out[row] = ctc_collapse(ids[i, :n])
        return out

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        waveforms: list[torch.Tensor],
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        return {TOKEN_IDS: [self._transcribe(waveforms)[0]]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        waveforms: list[torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, NameToTensorList]:
        del graph_walk, kwargs
        rows = self._transcribe(waveforms)
        return {rid: {TOKEN_IDS: [ids]} for rid, ids in zip(engine_inputs.request_ids, rows, strict=True)}
