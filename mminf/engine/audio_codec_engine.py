import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.utils.profiler import range_pop, range_push


class AudioCodecEngine(BaseEngine):
    """
    Wraps submodules for audio codec forward passes.
    Stateless — identical lifecycle to EncoderDecoderEngine.
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

    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        model_config: dict,
        device: torch.device,
        **kwargs
    ) -> None:
        self.submodules = submodules
        self.device = device

    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        if self.enable_nvtx:
            range_push(f"engine.audio_codec.{batch.node_name}.{batch.graph_walk}")

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
                    outputs = {}
                    for rid in batch.request_ids:
                        inputs = batch.per_request_input_tensors.get(rid, {})
                        if hasattr(submodule, 'preprocess'):
                            preprocessed = submodule.preprocess(
                                batch.graph_walk,
                                per_request_inputs=[inputs],
                                request_ids=[rid],
                                per_request_info={
                                    rid: batch.per_request_info[rid]
                                },
                            )
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
        finally:
            if self.enable_nvtx:
                range_pop()

    def add_request(self, request_id: str) -> None:
        pass  # stateless

    def remove_request(self, request_id: str) -> None:
        pass  # stateless
