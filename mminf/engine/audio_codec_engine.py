import logging

import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.engine.cuda_graph_runner import CodecCudaGraphRunner, CodecGraphNotApplicableError
from mminf.utils.profiler import range_pop, range_push

logger = logging.getLogger(__name__)


class AudioCodecEngine(BaseEngine):
    """
    Wraps submodules for audio codec forward passes.
    """

    def __init__(
        self, enable_nvtx: bool = False,
        autocast_dtype=torch.bfloat16,
        **kwargs
    ):
        super().__init__(enable_nvtx=enable_nvtx)
        self.submodules: dict[str, torch.nn.Module] = {}
        self.device = None
        self.autocast_dtype = autocast_dtype

    def engine_type(self) -> EngineType:
        return EngineType.AUDIO_CODEC

    def has_autocast(self):
        return False

    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        device: torch.device,
        **kwargs
    ) -> None:
        # Audio codecs need float32 precision (reference runs without autocast).
        # Override bfloat16 cast from engine_manager.
        self.submodules = {
            name: mod.float() for name, mod in submodules.items()
        }
        self.device = device

    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        if self.enable_nvtx:
            range_push(f"engine.audio_codec.{batch.node_name}.{batch.graph_walk}.bs{len(batch.request_ids)}")

        submodule = self.submodules.get(batch.node_name)
        if submodule is None:
            output = NodeOutput(
                per_request_output_tensors={rid: {} for rid in batch.request_ids}
            )
            if self.enable_nvtx:
                range_pop()
            return output

        try:
            with torch.inference_mode():
                output = self._dispatch(batch, submodule)
                for rid, info in batch.per_request_info.items():
                    submodule.postprocess(
                        request_id=rid,
                        request_info=info,
                        outputs=output.per_request_output_tensors.get(rid, {})
                    )
                return output
        finally:
            if self.enable_nvtx:
                range_pop()

    def _dispatch(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Pick cuda_graph / batched / sequential, with eager fallback if
        the CUDA-graph runner rejects the batch (e.g. SNAC frame mismatch).
        """
        if self._can_use_cuda_graph(batch, submodule):
            try:
                return self._execute_with_cuda_graph(batch, submodule)
            except CodecGraphNotApplicableError as exc:
                logger.debug(
                    "%s: CUDA graph path declined for batch %s (%s); falling back",
                    batch.node_name, batch.request_ids, exc,
                )
        if hasattr(submodule, 'can_batch') and submodule.can_batch(batch):
            return self._execute_batched(batch, submodule)
        return self._execute_sequential(batch, submodule)

    def _can_use_cuda_graph(self, batch: NodeBatch, submodule) -> bool:
        runner: CodecCudaGraphRunner | None = getattr(
            submodule, 'cuda_graph_runner', None,
        )
        if runner is None:
            return False
        if not submodule.can_use_cuda_graphs(batch):
            return False
        return runner.can_run(
            batch_size=len(batch.request_ids),
            graph_walk=batch.graph_walk,
        )

    def _execute_with_cuda_graph(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Replay the captured graph. Runner handles preprocess + replay +
        per-rid split; we only wrap the result.
        """
        per_request_inputs = [
            batch.per_request_input_tensors.get(rid, {}) for rid in batch.request_ids
        ]
        if self.enable_nvtx:
            range_push("codec.cuda_graph.run")
        per_rid = submodule.cuda_graph_runner.run(
            graph_walk=batch.graph_walk,
            request_ids=batch.request_ids,
            per_request_inputs=per_request_inputs,
            per_request_info=batch.per_request_info,
            submodule=submodule,
        )
        if self.enable_nvtx:
            range_pop()
        return NodeOutput(per_request_output_tensors=per_rid)

    def _execute_batched(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Eager batched path: preprocess → forward_batched(packed) → per-rid."""
        per_request_inputs = [
            batch.per_request_input_tensors.get(rid, {}) for rid in batch.request_ids
        ]

        if self.enable_nvtx:
            range_push("codec.batched.preprocess", synchronize=True)
        packed = submodule.preprocess(
            graph_walk=batch.graph_walk,
            per_request_inputs=per_request_inputs,
            request_ids=batch.request_ids,
            per_request_info=batch.per_request_info,
        )
        if self.enable_nvtx:
            range_pop(synchronize=True)
        if not packed:
            # Submodule signaled non-batchable — fall back to sequential.
            return self._execute_sequential(batch, submodule)

        if self.enable_nvtx:
            range_push("codec.batched.forward")
        outputs = submodule.forward_batched(
            graph_walk=batch.graph_walk,
            request_ids=batch.request_ids,
            packed_inputs=packed,
            per_request_info=batch.per_request_info,
        )
        if self.enable_nvtx:
            range_pop()
        return NodeOutput(per_request_output_tensors=outputs)

    def _execute_sequential(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Execute each request individually."""
        outputs = {}
        for i, rid in enumerate(batch.request_ids):
            inputs = batch.per_request_input_tensors.get(rid, {})

            fwd_info = batch.per_request_info[rid]

            if hasattr(submodule, 'preprocess'):
                if self.enable_nvtx:
                    range_push(f"codec.preprocess.{i}", synchronize=True)
                preprocessed = submodule.preprocess(
                    batch.graph_walk,
                    per_request_inputs=[inputs],
                    request_ids=[rid],
                    per_request_info={
                        rid: fwd_info,
                    },
                )
                if self.enable_nvtx:
                    range_pop(synchronize=True)
                # Submodule may signal "skip this request" by returning an
                # empty dict (e.g. SNAC with <7 new_tokens or a heterogeneous
                # batch falling back through the sequential path).
                if not preprocessed:
                    outputs[rid] = {}
                    continue
                if self.enable_nvtx:
                    range_push(f"codec.forward.{i}")
                outputs[rid] = submodule(**preprocessed)
                if self.enable_nvtx:
                    range_pop()
            else:
                if self.enable_nvtx:
                    range_push(f"codec.forward.{i}")
                result = submodule(**{k: v[0] for k, v in inputs.items()})
                if self.enable_nvtx:
                    range_pop()
                if isinstance(result, dict):
                    outputs[rid] = result
                elif isinstance(result, torch.Tensor):
                    outputs[rid] = {"output": [result]}
                else:
                    outputs[rid] = {}
        return NodeOutput(per_request_output_tensors=outputs)

    def warmup(self) -> None:
        """Capture CUDA graphs for submodules that support it.

        Submodules that opt in expose:
          - ``cuda_graph_runner`` attribute (set to the captured runner here)
          - ``get_cuda_graph_configs(device)`` returning at least one config
          - ``cuda_graph_forward(**static_inputs)`` that the runner captures
        """
        if not torch.cuda.is_available() or self.device is None:
            return

        from mminf.engine.cuda_graph_runner import CodecCudaGraphRunner

        for node_name, submodule in self.submodules.items():
            if not hasattr(submodule, 'get_cuda_graph_configs'):
                continue

            runner = CodecCudaGraphRunner(
                submodule_name=node_name,
                submodule=submodule,
                device=self.device,
            )
            runner.enable_nvtx = self.enable_nvtx
            runner.warmup_and_capture()
            if runner.graphs:
                submodule.cuda_graph_runner = runner
                logger.info(
                    "AudioCodecEngine: CUDA graphs captured for %s (%d graphs)",
                    node_name, len(runner.graphs),
                )

    def add_request(self, request_id: str) -> None:
        pass  # stateless

    def remove_request(self, request_id: str) -> None:
        pass  # stateless
