"""The transducer node: greedy RNN-T / TDT decoding as a loop in the graph. The
encoder node is the shared ``FastConformerEncoderSubmodule``.

``prefill`` projects a batch's encoder output once (with Nemotron's language
prompt) into each request's ``frames``, which the decode loop re-injects every
iteration. A ``decode`` iteration reads its rows' state from the recurrent
pool, runs ``CHUNK`` greedy steps, writes the state back and emits the
iteration's tokens; the stop check ends a request's loop once its cursor has
passed its last frame.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.fp32_precision import fp32_precision_scope, node_fp32_precision
from mstar.engine.resources import Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.recurrent import RecurrentStatePool, RecurrentStep
from mstar.model.components.transducer import CHUNK, TransducerHead
from mstar.model.parakeet.config import DECODE_LOOP, DECODE_WALK, NUM_FRAMES, TRANSDUCER_STATE
from mstar.model.submodule_base import (
    BatchedModelOutput,
    HostRows,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.utils.h2d import to_device_nonblocking

# Encoder frames a decode iteration is captured over: 41 s of audio, and
# Parakeet's 610 s request limit. A step costs the same whatever the length,
# so one bucket serves every shorter clip, padded; a longer clip runs eager.
FRAME_BUCKETS = (512, 8192)
CAPTURE_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
# rows x frames per capture, which bounds each bucket's static frames buffer
MAX_BATCH_FRAMES = 64 * 512


def _batch_sizes(bucket: int) -> list[int]:
    return [bs for bs in CAPTURE_BATCH_SIZES if bs * bucket <= MAX_BATCH_FRAMES]


class TransducerSubmodule(NodeSubmodule):
    """Projects the frames at prefill; each decode iteration then replays, per
    frame bucket, the pool's gather, ``CHUNK`` compiled greedy steps and the
    scatter.

    The head is small (a 640-wide LSTM and joint) and its argmax decides every
    token, so it runs in fp32 outside the engine's autocast and TF32.
    """

    MAX_BATCH_SIZE = max(CAPTURE_BATCH_SIZES)
    disable_autocast = True
    # only the greedy step is compiled (``__init__``)
    disable_torch_compile = True
    # TF32 flips argmax near-ties
    float32_matmul_precision = "highest"

    def __init__(self, head: TransducerHead, compile_step: bool = True, frame_buckets=FRAME_BUCKETS):
        super().__init__()
        self.head = head
        self.frame_buckets = tuple(frame_buckets)
        # where a fresh slot starts, and every padding row; in full fp32, as
        # the engine runs the forward
        with fp32_precision_scope(node_fp32_precision(type(self).__name__, self)):
            self._start = head.initial_state()
        # the zeros frames are padded with, sliced per row
        self._pad: torch.Tensor | None = None
        # One compile for every capture rather than each config's own, which
        # would autotune each shape: fixed shapes first, then the dims that
        # change go symbolic. fullgraph: a graph break would otherwise fall
        # back to eager kernels without a word.
        compile_step = compile_step and next(head.parameters()).is_cuda
        self._step = torch.compile(head.step, fullgraph=True) if compile_step else head.step

    def _bucket(self, num_frames: int | None) -> int | None:
        """The narrowest capture that holds ``num_frames``; None past the widest."""
        if num_frames is None:
            return None
        return next((bucket for bucket in self.frame_buckets if num_frames <= bucket), None)

    # -- inputs ----------------------------------------------------------------

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        if graph_walk == DECODE_WALK:
            # the prefill's output, already on this device
            frames = inputs["frames"][0]
            counted = fwd_info.step_metadata.get(NUM_FRAMES)
            if counted is not None and frames.shape[0] != counted:
                raise ValueError(
                    f"the encoder heard {frames.shape[0]} frames of this clip, process_prompt counted {counted}"
                )
            return NodeInputs(
                tensor_inputs={"frames": frames}, input_seq_len=1, resource_step_info=self._bucket(frames.shape[0]),
            )
        states = inputs["encoder_states"][0].to(device=self.get_device())
        tensor_inputs = {"encoder_states": states}
        if self.head.prompt_projector is not None:
            # moved to the device with the batch's, in preprocess
            tensor_inputs["prompt_ids"] = inputs["prompt_ids"][0].reshape(1)
        return NodeInputs(tensor_inputs=tensor_inputs, input_seq_len=states.shape[0])

    def _padding(self, rows: int, like: torch.Tensor) -> torch.Tensor:
        if self._pad is None or self._pad.shape[0] < rows:
            self._pad = like.new_zeros(max(rows, *self.frame_buckets), like.shape[-1])
        return self._pad

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        if graph_walk == DECODE_WALK:
            frames = [inp.tensor_inputs["frames"] for inp in inputs]
            lengths = [f.shape[0] for f in frames]
            # a replay reads its capture's shape; eager pads to the longest row
            width = inputs[0].resource_step_info if engine_inputs.captured else max(1, *lengths)
            pad = self._padding(width, frames[0])
            parts = []
            for f, n in zip(frames, lengths, strict=True):
                parts += (f, pad[:width - n])
            return {
                "frames": torch.cat(parts).view(len(frames), width, -1),
                "lengths": to_device_nonblocking(torch.tensor(lengths), frames[0].device),
            }
        # fp32 for the head, in one cast for the batch
        states = pad_sequence([inp.tensor_inputs["encoder_states"] for inp in inputs], batch_first=True).float()
        out = {"encoder_states": states, "num_frames": [inp.input_seq_len for inp in inputs]}
        if self.head.prompt_projector is not None:
            prompt_ids = torch.cat([inp.tensor_inputs["prompt_ids"] for inp in inputs])
            out["prompt_ids"] = to_device_nonblocking(prompt_ids, states.device)
        return out

    # -- resources and captures --------------------------------------------------

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        if graph_walk != DECODE_WALK:
            # nothing to read or write yet: a request takes its slot at its first iteration
            return SubmoduleStep(steps={})
        buckets = {inp.resource_step_info for inp in inputs}
        return SubmoduleStep(
            segments=[Segment(rid, "main", 1) for rid in request_ids],
            steps={TRANSDUCER_STATE: RecurrentStep()},
            # the bucket from the frames, as cg_key_info has it from step_metadata
            cg_key_info=buckets.pop() if len(buckets) == 1 else None,
        )

    def cg_key_info(
        self, graph_walk: str,
        per_request_info: Mapping[str, CurrentForwardPassInfo],
        per_request_input_metadata=None,
        **kwargs: Any,
    ) -> int | None:
        if graph_walk != DECODE_WALK:
            return None
        buckets = {self._bucket(info.step_metadata.get(NUM_FRAMES)) for info in per_request_info.values()}
        return buckets.pop() if len(buckets) == 1 else None

    def split_batches_by_capture_key(self, graph_walk: str) -> bool:
        # the buckets can't share a replay, and an eager batch pads to its longest clip
        return graph_walk == DECODE_WALK

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        """One decode capture per frame bucket. The template row carries no
        frames, so a replay's padding rows copy none; preprocess pads every
        row to the bucket."""
        hidden = self.head.config.hidden_size
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk=DECODE_WALK,
                single_request_inputs=NodeInputs(
                    tensor_inputs={"frames": torch.zeros(0, hidden, device=device)},
                    input_seq_len=1,
                    resource_step_info=bucket,
                ),
                additional_key_info=bucket,
                capture_batch_sizes=_batch_sizes(bucket),
                # the step inside is compiled already, once for every capture
                compile=False,
            )
            for bucket in self.frame_buckets
        ]

    # -- forward ------------------------------------------------------------------

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs,
    ) -> BatchedModelOutput | dict[str, NameToTensorList]:
        if graph_walk != DECODE_WALK:
            return self._project(engine_inputs, **kwargs)
        return self._decode(engine_inputs, **kwargs)

    def _project(
        self,
        engine_inputs: ModelInputsFromEngine,
        encoder_states: torch.Tensor,
        num_frames: list[int],
        prompt_ids: torch.Tensor | None = None,
    ) -> dict[str, NameToTensorList]:
        frames = self.head.project_frames(encoder_states, prompt_ids)
        # each request's own copy: a view would keep the padded batch alive
        # until the request's last iteration
        return {
            rid: {"frames": [frames[i, :n].clone()]}
            for i, (rid, n) in enumerate(zip(engine_inputs.request_ids, num_frames, strict=True))
        }

    def _decode(
        self, engine_inputs: ModelInputsFromEngine, frames: torch.Tensor, lengths: torch.Tensor,
    ) -> BatchedModelOutput:
        pool: RecurrentStatePool = engine_inputs.resources[TRANSDUCER_STATE]
        # a fresh slot, and a padding row on the sink, start from the start state
        state = {name: pool.gather(name, start=start) for name, start in self._start.items()}
        tokens, active = self.head.advance(frames, lengths, state, CHUNK, self._step)
        for name, value in state.items():
            pool.scatter_(name, value)
        # tokens are -1 where a step emitted none. The clock only makes the
        # node a speculation target; no input reads it.
        return BatchedModelOutput(
            row_outputs={"tokens": tokens, "clock": active},
            check_stop_buffers={"active": active},
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs,
    ) -> NameToTensorList:
        out = BatchedModelOutput.coerce(self.forward_batched(graph_walk, engine_inputs, **kwargs))
        if out.row_outputs:
            return {name: [rows[:1]] for name, rows in out.row_outputs.items()}
        return out.per_rid_outputs[engine_inputs.request_ids[0]]

    # -- stopping -------------------------------------------------------------------

    def _stops(self, active: bool, info: CurrentForwardPassInfo) -> bool:
        if not active:
            return True
        # the loop cap, which never binds (`decode_iterations`)
        num_frames = info.step_metadata.get(NUM_FRAMES)
        done = info.dynamic_loop_iter_counts.get(DECODE_LOOP, 0) + 1
        return num_frames is not None and done >= self.head.config.decode_iterations(num_frames)

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "clock" not in outputs:
            return set()
        # the clock carries the row's active flag
        return {DECODE_LOOP} if self._stops(bool(outputs["clock"][0].reshape(-1)[0]), request_info) else set()

    def check_stop_batched(
        self,
        request_ids: list[str],
        request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows,
        step_walk: str | None = None,
    ) -> dict[str, set[str]] | None:
        active = host_rows.buffers.get("active")
        if not torch.is_tensor(active):
            return None
        values = active.tolist()
        row_of = {rid: i for i, rid in enumerate(host_rows.request_ids)}
        stops: dict[str, set[str]] = {}
        for rid in request_ids:
            i = row_of.get(rid)
            if i is not None and self._stops(values[i], request_infos[rid]):
                stops[rid] = {DECODE_LOOP}
        return stops
