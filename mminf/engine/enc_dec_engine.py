import torch

from mminf.engine.base import BaseEngine, EngineType, StageBatch, StageOutput


class EncoderDecoderEngine(BaseEngine):
    """
    Wraps an nn.Module for stateless forward passes
    (ViT encoder, text embedding, VAE decoder).
    """

    def __init__(self, model: torch.nn.Module | None = None):
        self.model = model
        self.device = None

    def engine_type(self) -> EngineType:
        return EngineType.ENC_DEC

    def load_model(self, model_config: dict, device: torch.device) -> None:
        self.device = device
        # Load model from config if self.model is None
        # For now, model must be passed in constructor or remains None (dummy mode)

    def execute_batch(self, batch: StageBatch) -> StageOutput:
        if self.model is None:
            # Dummy mode: return empty tensors matching expected output names
            return StageOutput(
                per_request_output_tensors={
                    rid: {} for rid in batch.request_ids
                }
            )

        with torch.no_grad():
            # Batch inputs, run model, split outputs per request
            outputs = {}
            for rid in batch.request_ids:
                inputs = batch.per_request_input_tensors.get(rid, {})
                result = self.model(**{k: v.unsqueeze(0) for k, v in inputs.items()})
                if isinstance(result, torch.Tensor):
                    outputs[rid] = {"output": result.squeeze(0)}
                elif isinstance(result, dict):
                    outputs[rid] = {k: v.squeeze(0) for k, v in result.items()}
                else:
                    outputs[rid] = {}
            return StageOutput(per_request_output_tensors=outputs)

    def add_request(self, request_id: str) -> None:
        pass  # stateless

    def remove_request(self, request_id: str) -> None:
        pass  # stateless
