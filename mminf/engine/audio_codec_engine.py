import logging

import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
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
        self._streaming_buffers: dict[str, dict[str, list[torch.Tensor]]] | None = None

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
                if hasattr(submodule, 'can_batch') and submodule.can_batch(batch):
                    output = self._execute_batched(batch, submodule)
                else:
                    output = self._execute_sequential(batch, submodule)
                for rid, info in batch.per_request_info.items():
                    submodule.postprocess(
                        request_info=info,
                        outputs=output.per_request_output_tensors.get(rid, {})
                    )
                return output
        finally:
            if self.enable_nvtx:
                range_pop()

    def _execute_batched(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Execute all requests in a single batched forward pass."""
        # Inject streaming buffers into per_request_info
        per_request_inputs = []
        for rid in batch.request_ids:
            inputs = batch.per_request_input_tensors.get(rid, {})
            fwd_info = batch.per_request_info[rid]
            if self._streaming_buffers and rid in self._streaming_buffers:
                fwd_info.step_metadata = {
                    **fwd_info.step_metadata,
                    "_streaming_buffer": self._streaming_buffers[rid],
                }
            per_request_inputs.append(inputs)

        if self.enable_nvtx:
            range_push("codec.batched.forward")
        outputs = submodule.forward_batched(
            graph_walk=batch.graph_walk,
            request_ids=batch.request_ids,
            per_request_inputs=per_request_inputs,
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
            if self._streaming_buffers and rid in self._streaming_buffers:
                fwd_info.step_metadata = {
                    **fwd_info.step_metadata,
                    "_streaming_buffer": self._streaming_buffers[rid],
                }

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
        """Capture CUDA graphs for submodules that support it."""
        if not torch.cuda.is_available() or self.device is None:
            return

        for node_name, submodule in self.submodules.items():
            if not hasattr(submodule, 'cuda_graph_runner'):
                continue

            # Import here to avoid circular deps; only SNAC uses this today
            from mminf.model.orpheus.submodules import SNACCudaGraphRunner

            runner = SNACCudaGraphRunner(
                snac_model=submodule.snac_model,
                config=submodule.config,
                device=self.device,
            )
            runner.warmup_and_capture()
            if runner.graphs:
                submodule.cuda_graph_runner = runner
                logger.info(
                    "AudioCodecEngine: CUDA graphs captured for %s (%d batch sizes)",
                    node_name, len(runner.graphs),
                )

    def add_request(self, request_id: str) -> None:
        pass  # stateless

    def remove_request(self, request_id: str) -> None:
        pass  # stateless
