import torch

from mminf.engine.base import BaseEngine, EngineType, StageBatch, StageOutput


class FlowEngine(BaseEngine):
    """
    Flow/diffusion engine. Executes a single denoising step per call.
    Loop iteration is handled by the graph system.
    """

    def __init__(self):
        self.submodules: dict[str, torch.nn.Module] = {}
        self.device = None

    def engine_type(self) -> EngineType:
        return EngineType.FLOW

    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        model_config: dict,
        device: torch.device,
    ) -> None:
        self.submodules = submodules
        self.device = device

    def execute_batch(self, batch: StageBatch) -> StageOutput:
        submodule = self.submodules.get(batch.stage_name)
        if submodule is None:
            # Dummy mode
            return StageOutput(
                per_request_output_tensors={
                    rid: {} for rid in batch.request_ids
                }
            )

        with torch.no_grad():
            outputs = {}
            for rid in batch.request_ids:
                inputs = batch.per_request_input_tensors.get(rid, {})
                if hasattr(submodule, 'preprocess'):
                    preprocessed = submodule.preprocess(batch.phase, **inputs)
                    outputs[rid] = submodule(**preprocessed)
                else:
                    result = submodule(**{k: v[0] for k, v in inputs.items()})
                    if isinstance(result, dict):
                        outputs[rid] = result
                    elif isinstance(result, torch.Tensor):
                        outputs[rid] = {"output": [result]}
                    else:
                        outputs[rid] = {}
            return StageOutput(per_request_output_tensors=outputs)

    def add_request(self, request_id: str) -> None:
        pass

    def remove_request(self, request_id: str) -> None:
        pass
