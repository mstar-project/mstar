import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput


class AudioCodecEngine(BaseEngine):
    """
    Wraps submodules for audio codec forward passes.
    Stateless — identical lifecycle to EncoderDecoderEngine.
    """

    def __init__(self):
        self.submodules: dict[str, torch.nn.Module] = {}
        self.device = None

    def engine_type(self) -> EngineType:
        return EngineType.AUDIO_CODEC

    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        model_config: dict,
        device: torch.device,
    ) -> None:
        self.submodules = submodules
        self.device = device

    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        submodule = self.submodules.get(batch.node_name)
        if submodule is None:
            return NodeOutput(
                per_request_output_tensors={
                    rid: {} for rid in batch.request_ids
                }
            )

        with torch.no_grad():
            outputs = {}
            for rid in batch.request_ids:
                inputs = batch.per_request_input_tensors.get(rid, {})
                if hasattr(submodule, 'preprocess'):
                    preprocessed = submodule.preprocess(batch.graph_walk, **inputs)
                    outputs[rid] = submodule(**preprocessed)
                else:
                    result = submodule(**{k: v[0] for k, v in inputs.items()})
                    if isinstance(result, dict):
                        outputs[rid] = result
                    elif isinstance(result, torch.Tensor):
                        outputs[rid] = {"output": [result]}
                    else:
                        outputs[rid] = {}
            return NodeOutput(per_request_output_tensors=outputs)

    def add_request(self, request_id: str) -> None:
        pass  # stateless

    def remove_request(self, request_id: str) -> None:
        pass  # stateless
