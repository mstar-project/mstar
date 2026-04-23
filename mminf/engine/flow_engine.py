import logging

import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.utils.profiler import range_pop, range_push

logger = logging.getLogger(__name__)


class FlowEngine(BaseEngine):
    """
    Flow/diffusion engine. Executes a single denoising step per call.
    Loop iteration is handled by the graph system.

    For BAGEL's image_gen, each flow step is a full LLM forward. Multiple
    requests' latents can be concatenated into a single LLM forward when
    the submodule supports forward_batched() (leveraging BatchedCacheManager
    from Phase 1).
    """

    def __init__(self, enable_nvtx: bool = False, **kwargs):
        super().__init__(enable_nvtx=enable_nvtx)
        self.submodules: dict[str, torch.nn.Module] = {}
        self.device = None

    def engine_type(self) -> EngineType:
        return EngineType.FLOW

    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        device: torch.device,
        **kwargs
    ) -> None:
        self.submodules = submodules
        self.device = device

    def _execute_sequential(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Original per-request execution."""
        outputs = {}
        for rid in batch.request_ids:
            inputs = batch.per_request_input_tensors.get(rid, {})
            metadata = batch.per_request_info[rid]
            if hasattr(submodule, 'preprocess'):
                preprocessed = submodule.preprocess(
                    batch.graph_walk,
                    per_request_inputs=[inputs],
                    request_ids=[rid],
                    per_request_info={
                        rid: metadata
                    },
                )
                outputs[rid] = submodule(request_info=metadata, **preprocessed)
            else:
                result = submodule(**{k: v[0] for k, v in inputs.items()})
                if isinstance(result, dict):
                    outputs[rid] = result
                elif isinstance(result, torch.Tensor):
                    outputs[rid] = {"output": [result]}
                else:
                    outputs[rid] = {}
        return NodeOutput(per_request_output_tensors=outputs)

    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        if self.enable_nvtx:
            range_push(f"engine.flow.{batch.node_name}.{batch.graph_walk}.bs{len(batch.request_ids)}")

        submodule = self.submodules.get(batch.node_name)
        if submodule is None:
            output = NodeOutput(
                per_request_output_tensors={rid: {} for rid in batch.request_ids}
            )
            if self.enable_nvtx:
                range_pop()
            return output

        try:
            with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
                with torch.no_grad():
                    output = self._execute_sequential(batch, submodule)
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

    def add_request(self, request_id: str) -> None:
        pass

    def remove_request(self, request_id: str) -> None:
        pass
