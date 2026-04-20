import logging

import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.utils.profiler import range_pop, range_push

logger = logging.getLogger(__name__)


class EncoderDecoderEngine(BaseEngine):
    """
    Wraps torch.nn.Module submodules for stateless forward passes
    (ViT encoder, text embedding, VAE decoder).

    Supports batched execution when all inputs in a batch have the same
    shape — tensors are stacked along dim=0 for a single forward pass.
    Falls back to per-request sequential execution for variable-shape inputs.
    """

    def __init__(
        self,
        enable_nvtx: bool = False,
        autocast_dtype=torch.bfloat16,
        **kwargs
    ):
        super().__init__(enable_nvtx=enable_nvtx)
        self.submodules: dict[str, torch.nn.Module] = {}
        self.device = None
        self.autocast_dtype = autocast_dtype

    def engine_type(self) -> EngineType:
        return EngineType.ENC_DEC

    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        device: torch.device,
        **kwargs
    ) -> None:
        self.submodules = submodules
        self.device = device

    def _execute_batched(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Stack same-shaped inputs and run a single forward pass."""
        request_ids = batch.request_ids

        # Preprocess all requests
        all_preprocessed = submodule.preprocess(
            batch.graph_walk,
            per_request_inputs=batch.per_request_input_tensors,
            request_ids=batch.request_ids,
            per_request_info=batch.per_request_info,
        )

        # Single forward pass
        result = submodule(**all_preprocessed)

        # Split outputs back per-request
        outputs = {}
        if isinstance(result, dict):
            for rid_idx, rid in enumerate(request_ids):
                per_req = {}
                for name, tensor_list in result.items():
                    if isinstance(tensor_list, list):
                        per_req[name] = [t[rid_idx] for t in tensor_list]
                    elif isinstance(tensor_list, torch.Tensor):
                        per_req[name] = [tensor_list[rid_idx]]
                    else:
                        per_req[name] = tensor_list
                outputs[rid] = per_req
        else:
            # Fallback: return same output for all
            for rid in request_ids:
                outputs[rid] = result if isinstance(result, dict) else {}

        return NodeOutput(per_request_output_tensors=outputs)

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
                outputs[rid] = submodule(request_info=metadata,**preprocessed)
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
            range_push(f"engine.enc_dec.{batch.node_name}.{batch.graph_walk}.bs{len(batch.request_ids)}")

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
                    if submodule.can_batch(batch):
                        output = self._execute_batched(batch, submodule)
                    else:
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

    def warmup(self) -> None:
        """Apply torch.compile to stateless encoder/decoder submodules.

        ViT and VAE models are excellent torch.compile candidates since they
        have fixed computation graphs with no control flow.
        """
        if not torch.cuda.is_available():
            return

        return # TODO DEBUG

        for node_name, submodule in self.submodules.items():
            try:
                if hasattr(submodule, 'forward'):
                    submodule.forward = torch.compile(
                        submodule.forward,
                        fullgraph=False,
                    )
                    logger.info("EncDecEngine: torch.compile applied to %s", node_name)
            except Exception:
                logger.warning("EncDecEngine: torch.compile failed for %s, using eager mode",
                               node_name, exc_info=True)

    def add_request(self, request_id: str) -> None:
        pass  # stateless

    def remove_request(self, request_id: str) -> None:
        pass  # stateless
