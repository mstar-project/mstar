"""The transducer node: greedy RNN-T / TDT decoding as a loop in the graph. The
encoder node is the shared ``FastConformerEncoderSubmodule``.

The decode loop re-injects each request's encoder output every iteration. An
iteration reads its rows' state from the recurrent pool, runs ``CHUNK`` greedy
steps, each projecting the frames under the cursors (with Nemotron's language
prompt), writes the state back and emits the iteration's tokens; the stop
check ends a request's loop once its cursor has passed its last frame.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.fp32_precision import fp32_precision_scope, node_fp32_precision
from mstar.engine.resources import Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.recurrent import RecurrentAddressing, RecurrentStatePool, RecurrentStep
from mstar.model.components.transducer import CHUNK, TransducerHead
from mstar.model.parakeet.config import DECODE_LOOP, NUM_FRAMES, PROMPT_ID, TRANSDUCER_STATE, WALK
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
    """Each decode iteration replays, per frame bucket, the pool's gather,
    ``chunk`` compiled greedy steps and the scatter.

    The head is small (a 640-wide LSTM and joint) and its argmax decides every
    token, so it runs in fp32 outside the engine's autocast and TF32. The
    encoder output stays in the encoder's dtype until a step projects the
    frames it reads.
    """

    MAX_BATCH_SIZE = max(CAPTURE_BATCH_SIZES)
    disable_autocast = True
    # only the greedy step is compiled (``__init__``)
    disable_torch_compile = True
    # TF32 flips argmax near-ties
    float32_matmul_precision = "highest"

    def __init__(
        self, head: TransducerHead, compile_step: bool = True, frame_buckets=FRAME_BUCKETS,
        encoder_dtype: torch.dtype = torch.float32, chunk: int = CHUNK,
    ):
        super().__init__()
        self.head = head
        self.frame_buckets = tuple(frame_buckets)
        # steps an iteration; the model's loop cap counts CHUNK, so only tests
        # that drive the node by hand pass fewer
        self.chunk = chunk
        # the encoder output's dtype, which the captures' frames buffers take
        self.encoder_dtype = encoder_dtype
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
        states = inputs["encoder_states"][0]
        counted = fwd_info.step_metadata.get(NUM_FRAMES)
        if counted is not None and states.shape[0] != counted:
            raise ValueError(
                f"the encoder heard {states.shape[0]} frames of this clip, process_prompt counted {counted}"
            )
        # the prompt is known on the host, so it rides the step metadata
        kwargs = {PROMPT_ID: fwd_info.step_metadata[PROMPT_ID]} if self.head.prompt_projector is not None else {}
        return NodeInputs(
            tensor_inputs={"encoder_states": states}, kwargs=kwargs, input_seq_len=1,
            resource_step_info=self._bucket(states.shape[0]),
        )

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
        states = [inp.tensor_inputs["encoder_states"] for inp in inputs]
        lengths = [s.shape[0] for s in states]
        # a replay reads its capture's shape; eager pads to the longest row
        width = inputs[0].resource_step_info if engine_inputs.captured else max(1, *lengths)
        pad = self._padding(width, states[0])
        parts = []
        for s, n in zip(states, lengths, strict=True):
            parts += (s, pad[:width - n])
        device = states[0].device
        out = {
            "encoder_states": torch.cat(parts).view(len(states), width, -1),
            "lengths": to_device_nonblocking(torch.tensor(lengths), device),
        }
        if self.head.prompt_projector is not None:
            out["prompt_ids"] = to_device_nonblocking(torch.tensor([inp.kwargs[PROMPT_ID] for inp in inputs]), device)
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
        buckets = {self._bucket(info.step_metadata.get(NUM_FRAMES)) for info in per_request_info.values()}
        return buckets.pop() if len(buckets) == 1 else None

    def split_batches_by_capture_key(self, graph_walk: str) -> bool:
        # the buckets can't share a replay, and an eager batch pads to its longest clip
        del graph_walk
        return True

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        """One decode capture per frame bucket. The template row carries no
        frames, so a replay's padding rows copy none; preprocess pads every
        row to the bucket."""
        hidden = self.head.config.encoder_hidden_size
        kwargs = {PROMPT_ID: 0} if self.head.prompt_projector is not None else {}
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk=WALK,
                single_request_inputs=NodeInputs(
                    tensor_inputs={"encoder_states": torch.zeros(0, hidden, dtype=self.encoder_dtype, device=device)},
                    kwargs=kwargs,
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

    def _read_state(self, pool: RecurrentStatePool, rows: RecurrentAddressing) -> dict[str, torch.Tensor]:
        """The rows' state from their slots; a fresh slot, and a padding row on
        the sink, from the start state."""
        fresh = ~rows.has_state[:rows.num_rows]
        return {
            name: torch.where(fresh.view(-1, *(1,) * start.dim()), start, pool.gather(name, rows))
            for name, start in self._start.items()
        }

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs,
    ) -> BatchedModelOutput:
        return self._decode(engine_inputs, **kwargs)

    def _decode(
        self, engine_inputs: ModelInputsFromEngine, encoder_states: torch.Tensor, lengths: torch.Tensor,
        prompt_ids: torch.Tensor | None = None,
    ) -> BatchedModelOutput:
        pool: RecurrentStatePool = engine_inputs.resources[TRANSDUCER_STATE]
        rows = pool.addressing("main")
        assert encoder_states.shape[0] == rows.num_rows, (
            f"{encoder_states.shape[0]} rows of frames for a step that addresses {rows.num_rows}"
        )
        state = self._read_state(pool, rows)
        tokens, active = self.head.advance(encoder_states, lengths, state, self.chunk, self._step, prompt_ids)
        for name, value in state.items():
            pool.scatter_(name, rows, value)
        # tokens are -1 where a step emitted none
        return BatchedModelOutput(
            row_outputs={"tokens": tokens},
            # the tokens too: the stop check doesn't read them, but the
            # client's copy then comes from their host rows, not a copy per request
            check_stop_buffers={"active": active, "tokens": tokens},
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs,
    ) -> NameToTensorList:
        out = self.forward_batched(graph_walk, engine_inputs, **kwargs)
        # the active flag for the per-request stop check, beside the tokens
        return {name: [rows[:1]] for name, rows in {**out.row_outputs, **out.check_stop_buffers}.items()}

    # -- stopping -------------------------------------------------------------------

    def _stops(self, active: bool, info: CurrentForwardPassInfo) -> bool:
        if not active:
            return True
        # the loop cap, which never binds (`decode_iterations`)
        num_frames = info.step_metadata.get(NUM_FRAMES)
        done = info.dynamic_loop_iter_counts.get(DECODE_LOOP, 0) + 1
        return num_frames is not None and done >= self.head.config.decode_iterations(num_frames, self.chunk)

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "active" not in outputs:
            return set()
        return {DECODE_LOOP} if self._stops(bool(outputs["active"][0].reshape(-1)[0]), request_info) else set()

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
