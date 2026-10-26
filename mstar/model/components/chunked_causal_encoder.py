"""A chunked causal encoder node: one fixed-size chunk of one attachment a step.

For a tower whose every op is a causal conv or per token (a waveform conv
stack), so its output over a long attachment is the same computed a chunk at a
time. A tower that attends across chunks would need their keys kept, which
this does not do. The model sets ``MediaEncoder.chunk_tokens`` (C), and the media walks
run the node as the Loop ``encoder.loop``: iteration ``i`` encodes C tokens of
one attachment, every request's chunks batch into one captured shape, and no
step is longer than one chunk. ``get_loop_caps`` caps the loop at the
request's chunk count from its media layout, so it ends on its last chunk with
no stop check. The LLM gets every chunk's rows, in order, when it ends
(``Loop.accumulated_outputs``).

What a chunk boundary cuts, each causal conv's left context, lives in a
`RecurrentStatePool` the model declares under ``state_key``: one slot per
request, which the tower's layers bind (``bind_resources``) and read and write
through the step's addressing (``gather`` / ``scatter_``). An attachment's
last chunk releases the slot when its step commits (``RecurrentStep.release``),
so a request holds one only while an attachment is being encoded, and the next
attachment's first chunk takes a fresh slot, which ``gather`` reads as
zeros. The pool keeps its sink slot, which a replay's padding
rows address. A final partial chunk is zero-padded on the right: the tower is
causal, so its real tokens are exact.

The loop is not speculated: its node reads no loop-back edge, so the async
worker runs chunk ``i + 1`` only once chunk ``i`` has completed.

A model declares ``chunk_tokens``, the pool, and a tower whose
``forward(values)`` maps ``[rows, C * per_token]`` to ``[rows, C, hidden]``;
its intake hands over each attachment in the tower's dtype.
"""

from bisect import bisect_right
from collections.abc import Sequence
from functools import cached_property
from itertools import accumulate
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.model.multimodal import (
    MEDIA_LAYOUT,
    MediaEncoder,
    encoder_combined_walks,
    media_runs,
    media_walk_names,
)
from mstar.model.submodule_base import BatchedModelOutput, ModelInputsFromEngine, NodeInputs, NodeSubmodule


class ChunkedCausalEncoder(NodeSubmodule):
    CAPTURE_BATCH_SIZES = [1, 2, 4, 8]

    def __init__(
        self, tower: torch.nn.Module, per_token: int,
        encoder: MediaEncoder, encoders: Sequence[MediaEncoder], state_key: str,
    ):
        """``per_token``: input steps a token (an attachment's tensor is
        its tokens times this along its first dim). ``encoders``: the
        model's, which say the walks this node runs in. ``state_key``: the
        model's `RecurrentStatePool` for what crosses a chunk boundary."""
        super().__init__()
        if encoder.chunk_tokens is None or len(encoder.inputs) != 1:
            raise ValueError(f"{encoder.node!r} is not a chunked encoder of one input")
        self.tower = tower
        self.state_key = state_key
        self.per_token = per_token
        self.encoder = encoder
        self.chunk = encoder.chunk_tokens
        combined = encoder_combined_walks(encoders).get(encoder.node, {})
        self.walks = [*media_walk_names(encoders, encoder), *combined.keys()]

    @cached_property
    def dtype(self) -> torch.dtype:
        """What the tower takes its input in, and its capture's buffer is.
        First read once the engine has cast the node (``prepare_inputs``,
        ``get_accelerator_graph_configs``), not in ``__init__``, which runs
        before that cast."""
        return next(self.tower.parameters()).dtype

    def bind_node_resources(self, resources: dict[str, Any]) -> None:
        pool = resources.get(self.state_key)
        if pool is None:
            raise ValueError(f"{self.encoder.node} has no {self.state_key!r} pool to carry its chunks' state in")
        if pool.config.disable_sink_slot:
            # a replay's padding rows would address slot -1
            raise ValueError(f"{self.encoder.node}'s {self.state_key!r} pool needs its sink slot")
        super().bind_node_resources(resources)

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs: Any,
    ) -> NodeInputs:
        """This iteration's chunk: the samples of its tokens, a slice of the
        attachment wherever it is, and how many are real."""
        runs = media_runs(fwd_info.step_metadata.get(MEDIA_LAYOUT) or {}, self.encoder.output)
        attachments = inputs[self.encoder.inputs[0]]
        if len(attachments) != len(runs):
            raise ValueError(f"{self.encoder.node} got {len(attachments)} attachments for {len(runs)} runs")
        i = fwd_info.dynamic_loop_iter_counts.get(self.encoder.loop, 0)
        # each run's first iteration, and the loop's end
        firsts = list(accumulate(map(self.encoder.chunks, runs), initial=0))
        run = bisect_right(firsts, i) - 1
        if run == len(runs):
            raise ValueError(f"{self.encoder.loop} iteration {i} is past the prompt's {firsts[-1]} chunks")
        tokens, values = runs[run], attachments[run].reshape(-1)
        if values.shape[0] != tokens * self.per_token:
            raise ValueError(f"{self.encoder.node}: an attachment of {values.shape[0]} steps for {tokens} tokens")
        if values.dtype != self.dtype:
            # the capture's buffer is in the tower's dtype, and the copy into it would cast silently
            raise ValueError(f"{self.encoder.node}: an attachment in {values.dtype}, not the tower's {self.dtype}")
        start = (i - firsts[run]) * self.chunk
        n = min(self.chunk, tokens - start)
        return NodeInputs(
            input_seq_len=self.chunk,
            tensor_inputs={"values": values[start * self.per_token:(start + n) * self.per_token]},
            # the attachment's last chunk frees the slot
            kwargs={"tokens": n, "release": start + n == tokens},
        )

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs], **kwargs: Any,
    ) -> SubmoduleStep:
        rows = list(zip(request_ids, inputs, strict=True))
        return SubmoduleStep(
            segments=[Segment(rid, "main", inp.input_seq_len) for rid, inp in rows],
            steps={self.state_key: RecurrentStep(
                release=frozenset(rid for rid, inp in rows if inp.kwargs["release"]),
            )},
        )

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        return max(self.CAPTURE_BATCH_SIZES)

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        """The rows' chunks, a partial one zero-padded. Host chunks are staged
        in one pinned tensor and copied over at once, so the copy does not
        block the host; device ones (attachments a worker received, a
        replay's padding rows) are copied in on the device after it."""
        device = self.get_device()
        chunks = [inp.tensor_inputs["values"] for inp in inputs]
        stage = device.type != "cpu" and any(c.device.type == "cpu" for c in chunks)
        rows = torch.zeros(
            len(chunks), self.chunk * self.per_token, dtype=self.dtype,
            device="cpu" if stage else device, pin_memory=stage,
        )
        home = rows.device
        for row, chunk in zip(rows, chunks, strict=True):
            if chunk.device == home:
                row[:chunk.shape[0]].copy_(chunk, non_blocking=True)
        rows = rows.to(device, non_blocking=True)
        for row, chunk in zip(rows, chunks, strict=True):
            if chunk.device != home:
                row[:chunk.shape[0]].copy_(chunk, non_blocking=True)
        return {"values": rows}

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, values: torch.Tensor,
    ) -> NameToTensorList:
        return {self.encoder.output: [self.tower(values)]}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, values: torch.Tensor,
    ) -> BatchedModelOutput:
        # rows, not a dict keyed by request id: a compiled forward would
        # guard on the ids and recompile for every batch
        return BatchedModelOutput(row_outputs={self.encoder.output: self.tower(values)})

    def postprocess(
        self, request_id: str, request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]], inputs: NodeInputs | None = None, **kwargs: Any,
    ) -> None:
        # The real tokens of this row's chunk, as a copy: the loop holds every
        # chunk until it ends, and a view would keep the step's whole batch.
        (row,) = outputs[self.encoder.output]
        outputs[self.encoder.output] = [row[0, :inputs.kwargs["tokens"]].clone()]

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        """One full chunk a row, for every walk the node runs in. On the
        device: a replay's padding rows are clones of it, which ``preprocess``
        would otherwise pin and copy from the host every step."""
        chunk = NodeInputs(
            input_seq_len=self.chunk,
            tensor_inputs={"values": torch.zeros(self.chunk * self.per_token, dtype=self.dtype, device=device)},
            kwargs={"tokens": self.chunk, "release": False},
        )
        return [BatchedAcceleratorGraphConfig(
            capture_graph_walk=self.walks[0],
            replay_graph_walks=self.walks,
            single_request_inputs=chunk,
            capture_batch_sizes=self.CAPTURE_BATCH_SIZES,
            # [rows, samples]: the samples can equal a bucket's token count
            input_seq_dims={"values": 0},
        )]
