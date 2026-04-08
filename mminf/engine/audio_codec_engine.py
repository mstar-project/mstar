import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.utils.profiler import range_pop, range_push


class AudioCodecEngine(BaseEngine):
    """
    Wraps submodules for audio codec forward passes.

    Supports streaming mode: when ``set_streaming_buffers`` is called (via
    BaseEngine) with a reference to the worker-level streaming buffer dict,
    the engine exposes the buffer to submodules via
    ``per_request_info.step_metadata`` during ``execute_batch`` so submodules
    can read accumulated tokens.
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
        model_config: dict,
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
            with torch.inference_mode():
                outputs = {}
                for rid in batch.request_ids:
                    inputs = batch.per_request_input_tensors.get(rid, {})

                    fwd_info = batch.per_request_info[rid]
                    # Legacy path: inject streaming buffer into step_metadata
                    if self._streaming_buffers and rid in self._streaming_buffers:
                        fwd_info.step_metadata = {
                            **fwd_info.step_metadata,
                            "_streaming_buffer": self._streaming_buffers[rid],
                        }

                    if hasattr(submodule, 'preprocess'):
                        preprocessed = submodule.preprocess(
                            batch.graph_walk,
                            per_request_inputs=[inputs],
                            request_ids=[rid],
                            per_request_info={
                                rid: fwd_info,
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
