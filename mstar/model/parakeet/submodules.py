"""The transducer node: greedy RNN-T / TDT decoding as a loop in the graph. The
encoder node is the shared ``FastConformerEncoderSubmodule``.

The decode loop re-injects each request's encoder output every iteration; the
first writes it into the request's recurrent state slot, beside the decoder
state. An iteration reads its rows' state from the pool, runs ``CHUNK`` greedy
steps, each reading the frames under the cursors from the slots and projecting
them (with Nemotron's language prompt), writes the state back and emits the
iteration's tokens; the stop check ends a request's loop once its cursor has
passed its last frame.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.engine.accelerator_graph_runner import compile_kwargs
from mstar.engine.engine import ExecutingBatch
from mstar.engine.fp32_precision import fp32_precision_scope, node_fp32_precision
from mstar.engine.resources import Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.recurrent import RecurrentStatePool, RecurrentStep
from mstar.model.components.transducer import CHUNK, TransducerHead
from mstar.model.parakeet.config import DECODE_LOOP, FRAMES, NUM_FRAMES, PROMPT_ID, TRANSDUCER_STATE, WALK
from mstar.model.submodule_base import (
    BatchedModelOutput,
    HostRows,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.platforms.compile import with_inductor_overrides
from mstar.utils.h2d import to_device_nonblocking

# Rows a decode iteration is captured for. The frames are read in the slots,
# so a capture serves clips of every length; a wider batch runs eager.
CAPTURE_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)


class TransducerSubmodule(NodeSubmodule):
    """Each decode iteration replays the pool's gather, ``chunk`` compiled
    greedy steps over the frames in the slots, and the scatter.

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
        self, head: TransducerHead, compile_step: bool = True,
        encoder_dtype: torch.dtype = torch.float32, chunk: int = CHUNK,
    ):
        super().__init__()
        self.head = head
        # steps an iteration; the model's loop cap counts CHUNK, so only tests
        # that drive the node by hand pass fewer
        self.chunk = chunk
        # the encoder output's dtype, for the captures' template row
        self.encoder_dtype = encoder_dtype
        # where a fresh slot starts, and every padding row; in full fp32, as
        # the engine runs the forward
        with fp32_precision_scope(node_fp32_precision(type(self).__name__, self)):
            self._start = head.initial_state()
        # One compile for every capture rather than each capture's own, which
        # would autotune each shape: fixed shapes first, then the rows go
        # symbolic. It tunes as the compile level (``-O``) says.
        # fullgraph: a graph break would otherwise fall back to eager kernels
        # without a word.
        compile_step = compile_step and next(head.parameters()).is_cuda
        self._step = with_inductor_overrides(torch.compile(
            head.step, **{**compile_kwargs(None), "fullgraph": True, "dynamic": None},
        )) if compile_step else head.step

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
        return NodeInputs(tensor_inputs={"encoder_states": states}, kwargs=kwargs, input_seq_len=1)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        states = [inp.tensor_inputs["encoder_states"] for inp in inputs]
        # a request's first iteration writes its frames into its slot; every
        # step reads them there, so the later iterations copy nothing
        engine_inputs.resources[TRANSDUCER_STATE].write_new_(FRAMES, states)
        device = states[0].device
        out = {"lengths": to_device_nonblocking(torch.tensor([s.shape[0] for s in states]), device)}
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
        return SubmoduleStep(
            segments=[Segment(rid, "main", 1) for rid in request_ids],
            steps={TRANSDUCER_STATE: RecurrentStep()},
        )

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        """One decode capture per batch size. The template row has no frames:
        a padding row is inactive from its first step."""
        hidden = self.head.config.encoder_hidden_size
        kwargs = {PROMPT_ID: 0} if self.head.prompt_projector is not None else {}
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk=WALK,
                single_request_inputs=NodeInputs(
                    tensor_inputs={"encoder_states": torch.zeros(0, hidden, dtype=self.encoder_dtype, device=device)},
                    kwargs=kwargs,
                    input_seq_len=1,
                ),
                capture_batch_sizes=list(CAPTURE_BATCH_SIZES),
                # the step inside is compiled already, once for every capture
                compile=False,
            ),
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
    ) -> BatchedModelOutput:
        return self._decode(engine_inputs, **kwargs)

    def _decode(
        self, engine_inputs: ModelInputsFromEngine, lengths: torch.Tensor, prompt_ids: torch.Tensor | None = None,
    ) -> BatchedModelOutput:
        pool: RecurrentStatePool = engine_inputs.resources[TRANSDUCER_STATE]
        # a fresh slot, and a padding row on the sink, start from the start state
        state = {name: pool.gather(name, start=start) for name, start in self._start.items()}
        frames = pool.reader(FRAMES)
        tokens, active = self.head.advance(frames, lengths, state, self.chunk, self._step, prompt_ids)
        for name, value in state.items():
            pool.scatter_(name, value)
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

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "active" not in outputs:
            return set()
        # the loop cap is the runtime's (`get_loop_caps`)
        return set() if bool(outputs["active"][0].reshape(-1)[0]) else {DECODE_LOOP}

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
            if i is not None and not values[i]:
                stops[rid] = {DECODE_LOOP}
        return stops
