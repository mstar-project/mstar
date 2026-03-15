import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.utils.profiler import range_pop, range_push


class EncoderDecoderEngine(BaseEngine):
    """
    Wraps torch.nn.Module submodules for stateless forward passes
    (ViT encoder, text embedding, VAE decoder).
    """

    def __init__(self, enable_nvtx: bool = False):
        super().__init__(enable_nvtx=enable_nvtx)
        self.submodules: dict[str, torch.nn.Module] = {}
        self.device = None

    def engine_type(self) -> EngineType:
        return EngineType.ENC_DEC

    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        model_config: dict,
        device: torch.device,
    ) -> None:
        self.submodules = submodules
        self.device = device

    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        if self.enable_nvtx:
            range_push(f"engine.enc_dec.{batch.node_name}.{batch.graph_walk}")

        submodule = self.submodules.get(batch.node_name)
        if submodule is None:
            # Dummy mode: return empty tensors matching expected output names
            output = NodeOutput(
                per_request_output_tensors={rid: {} for rid in batch.request_ids}
            )
            if self.enable_nvtx:
                range_pop()
            return output

        try:
            with torch.no_grad():
                outputs = {}
                for rid in batch.request_ids:
                    inputs = batch.per_request_input_tensors.get(rid, {})
                    metadata = batch.per_request_metadata.get(rid, {})
                    if hasattr(submodule, 'preprocess'):
                        # torch.nn.Module: preprocess (list → tensor) then forward
                        preprocessed = submodule.preprocess(batch.graph_walk, **inputs)
                        outputs[rid] = submodule(**preprocessed, **metadata)
                    else:
                        # Raw nn.Module: unwrap single tensors, run, re-wrap
                        result = submodule(**{k: v[0] for k, v in inputs.items()})
                        if isinstance(result, dict):
                            outputs[rid] = result
                        elif isinstance(result, torch.Tensor):
                            outputs[rid] = {"output": [result]}
                        else:
                            outputs[rid] = {}
                return NodeOutput(per_request_output_tensors=outputs)
        finally:
            if self.enable_nvtx:
                range_pop()

    def add_request(self, request_id: str) -> None:
        pass  # stateless

    def remove_request(self, request_id: str) -> None:
        pass  # stateless
