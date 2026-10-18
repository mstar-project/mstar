"""The transducer node: greedy RNN-T / TDT decoding over a batch of encoder
outputs. The encoder node is the shared ``FastConformerEncoderSubmodule``."""
from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    PiecewiseAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
)
from mstar.engine.engine import ExecutingBatch
from mstar.model.components.transducer import TransducerHead
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.utils.h2d import to_device_nonblocking
from mstar.utils.precision import ieee_fp32_matmul

# Encoder frames captured per greedy chunk: 41 s of audio, and Parakeet's 610 s
# request limit. A step costs the same whatever the length, so one bucket
# serves every shorter clip, padded.
FRAME_BUCKETS = (512, 8192)
CAPTURE_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
# rows x frames per capture: the static frames buffer, shared by every capture
MAX_BATCH_FRAMES = 64 * 512


def _region(bucket: int) -> str:
    return f"greedy_T{bucket}"


def _batch_sizes(bucket: int) -> list[int]:
    return [bs for bs in CAPTURE_BATCH_SIZES if bs * bucket <= MAX_BATCH_FRAMES]


class TransducerSubmodule(NodeSubmodule):
    """Runs the whole greedy loop for a batch in one forward and emits each
    request's token ids at once.

    The head is small (a 640-wide LSTM and joint) and its argmax decides every
    token, so it runs in fp32 outside the engine's autocast and TF32. Each
    chunk of the loop replays a graph captured per frame bucket, of a compiled
    greedy step: compiling fuses the step's many tiny kernels.
    """

    MAX_BATCH_SIZE = max(CAPTURE_BATCH_SIZES)
    disable_autocast = True
    # forward is a data-dependent Python loop; only the step it replays is
    # compiled (``__init__``)
    disable_torch_compile = True

    def __init__(self, head: TransducerHead, compile_step: bool = True):
        super().__init__()
        self.head = head
        self._frames: torch.Tensor | None = None
        # One compile for every capture rather than the region's own compile,
        # which would autotune each of the ten shapes: fixed shapes first, then
        # the dims that change go symbolic. fullgraph: a graph break would
        # otherwise fall back to eager kernels without a word.
        compile_step = compile_step and next(head.parameters()).is_cuda
        self._step = torch.compile(head.step, fullgraph=True) if compile_step else head.step

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        states = inputs["encoder_states"][0].to(device=self.get_device())
        tensor_inputs = {"encoder_states": states}
        if self.head.prompt_projector is not None:
            # moved to the device with the batch's, in preprocess
            tensor_inputs["prompt_ids"] = inputs["prompt_ids"][0].reshape(1)
        return NodeInputs(tensor_inputs=tensor_inputs, input_seq_len=states.shape[0])

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        # fp32 for the head, in one cast for the batch
        states = pad_sequence([inp.tensor_inputs["encoder_states"] for inp in inputs], batch_first=True).float()
        num_frames = [inp.input_seq_len for inp in inputs]
        out = {
            "encoder_states": states,
            "lengths": to_device_nonblocking(torch.tensor(num_frames), states.device),
            "num_frames": num_frames,
        }
        if self.head.prompt_projector is not None:
            prompt_ids = torch.cat([inp.tensor_inputs["prompt_ids"] for inp in inputs])
            out["prompt_ids"] = to_device_nonblocking(prompt_ids, states.device)
        return out

    # -- the greedy chunk, captured per frame bucket or eager ----------------

    def _capture(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        with ieee_fp32_matmul():
            return self.head.advance(inp.static_inputs, self._step)

    def _static_inputs(self, shape: PiecewiseCaptureShape, bucket: int, device) -> dict[str, torch.Tensor]:
        cfg = self.head.config
        rows, hidden = shape.bs, cfg.hidden_size
        if self._frames is None:
            # one buffer under every capture: they never replay at once
            self._frames = torch.zeros(MAX_BATCH_FRAMES * hidden, device=device)
        state = {
            name: torch.zeros(rows, dtype=torch.long, device=device)
            for name in ("lengths", "cursor", "symbols")
        }
        return {
            **state,
            "frames": self._frames[:rows * bucket * hidden].view(rows, bucket, hidden),
            "active": torch.zeros(rows, dtype=torch.bool, device=device),
            "dec_out": torch.zeros(rows, hidden, device=device),
            "h": torch.zeros(rows, cfg.num_layers, hidden, device=device),
            "c": torch.zeros(rows, cfg.num_layers, hidden, device=device),
        }

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs: Any,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """One region per frame bucket; the head holds no engine resources."""
        del autocast_dtype, tp_world_size, kwargs
        return {
            _region(bucket): PiecewiseBatchedConfig(
                capture_fn=self._capture,
                make_static_inputs=partial(self._static_inputs, bucket=bucket, device=device),
                seq_len=bucket,
                capture_batch_sizes=_batch_sizes(bucket),
            )
            for bucket in FRAME_BUCKETS
        }

    def _uncaptured(self, encoder_states, lengths, prompt_ids) -> list[list[int]]:
        def run_eager(state, first):
            return self.head.advance(state, self._step)

        return self.head.greedy_decode(encoder_states, lengths, prompt_ids, run_eager)

    def _replayed(self, runner, bucket: int, encoder_states, lengths, prompt_ids) -> list[list[int]]:
        b = encoder_states.shape[0]

        def run_chunk(state, first):
            return runner.run(static_inputs=state if first else {}, real_bs=b)

        return self.head.greedy_decode(encoder_states, lengths, prompt_ids, run_chunk, pad_frames_to=bucket)

    def _greedy(self, encoder_states, lengths, prompt_ids, runner, bucket: int | None) -> list[list[int]]:
        if runner is None:
            # no region: CUDA graphs off, none of its captures worked, or a clip past the widest bucket
            return self._uncaptured(encoder_states, lengths, prompt_ids)
        # a replay holds at most the bucket's widest capture: more rows go a part at a time
        part = max(_batch_sizes(bucket))
        tokens: list[list[int]] = []
        for start in range(0, encoder_states.shape[0], part):
            rows = slice(start, start + part)
            states, lens = encoder_states[rows], lengths[rows]
            prompts = None if prompt_ids is None else prompt_ids[rows]
            if runner.can_run(states.shape[0]):
                tokens += self._replayed(runner, bucket, states, lens, prompts)
            else:
                # the capture this part pads to failed at warmup
                tokens += self._uncaptured(states, lens, prompts)
        return tokens

    def _decode(self, encoder_states, lengths, prompt_ids, num_frames: list[int], runners: dict[str, Any]):
        # each row in its own frame bucket, as the encoder groups by mel bucket:
        # one long clip doesn't drag the short ones into a narrow capture
        groups: dict[int | None, list[int]] = {}
        for row, n in enumerate(num_frames):
            groups.setdefault(next((f for f in FRAME_BUCKETS if n <= f), None), []).append(row)
        tokens: list[list[int]] = [[] for _ in num_frames]
        device = encoder_states.device
        for bucket, rows in groups.items():
            states, lens, prompts = encoder_states, lengths, prompt_ids
            if len(groups) > 1:
                index = to_device_nonblocking(torch.tensor(rows), device)
                states, lens = encoder_states[index, :max(num_frames[r] for r in rows)], lengths[index]
                prompts = None if prompt_ids is None else prompt_ids[index]
            runner = runners.get(_region(bucket)) if bucket is not None else None
            for row, row_tokens in zip(rows, self._greedy(states, lens, prompts, runner, bucket), strict=True):
                tokens[row] = row_tokens
        # an empty transcript still sends one tensor; the blank renders as nothing
        blank = self.head.config.blank_token_id
        return [to_device_nonblocking(torch.tensor(t or [blank], dtype=torch.long), device) for t in tokens]

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        encoder_states: torch.Tensor,
        lengths: torch.Tensor,
        num_frames: list[int],
        prompt_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        tokens = self._decode(encoder_states, lengths, prompt_ids, num_frames, engine_inputs.piecewise_runners)
        return {"tokens": tokens}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        encoder_states: torch.Tensor,
        lengths: torch.Tensor,
        num_frames: list[int],
        prompt_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        tokens = self._decode(encoder_states, lengths, prompt_ids, num_frames, engine_inputs.piecewise_runners)
        return {rid: {"tokens": [t]} for rid, t in zip(engine_inputs.request_ids, tokens, strict=True)}
