import logging

import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
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
        self.submodules: dict[str, NodeSubmodule] = {}
        self.device = None
        self.autocast_dtype = autocast_dtype

    def engine_type(self) -> EngineType:
        return EngineType.ENC_DEC

    def load_model(
        self,
        submodules: dict[str, NodeSubmodule],
        device: torch.device,
        **kwargs
    ) -> None:
        self.submodules = submodules
        self.device = device

    def _execute_batched(
        self, batch: NodeBatch,
        inputs: list[NodeInputs],
        submodule: NodeSubmodule
    ) -> NodeOutput:
        """Stack same-shaped inputs and run a single forward pass."""
        request_ids = batch.request_ids
        engine_inputs = ModelInputsFromEngine(
            request_ids=request_ids,
            per_request_info=batch.per_request_info
        )
        preprocessed = submodule.preprocess(
            batch.graph_walk,
            engine_inputs=engine_inputs,
            inputs=inputs,
        )

        outputs = submodule.forward_batched(
            graph_walk=batch.graph_walk,
            engine_inputs=engine_inputs,
            **preprocessed
        )

        return NodeOutput(per_request_output_tensors=outputs)

    def _execute_sequential(
        self, batch: NodeBatch,
        inputs: list[NodeInputs],
        submodule: NodeSubmodule
    ) -> NodeOutput:
        """Original per-request execution."""
        outputs = {}
        for rid, node_inputs in zip(batch.request_ids, inputs, strict=True):
            engine_inputs = ModelInputsFromEngine(
                request_ids=[rid],
                per_request_info={
                    rid: batch.per_request_info[rid]
                }
            )
            preprocessed = submodule.preprocess(
                graph_walk=batch.graph_walk,
                engine_inputs=engine_inputs,
                inputs=[node_inputs]
            )
            outputs[rid] = submodule.forward(
                graph_walk=batch.graph_walk,
                engine_inputs=engine_inputs,
                **preprocessed
            )
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
                    if self.enable_nvtx:
                        range_push("engine.enc_dec.prepare_inputs", synchronize=True)
                    node_inputs: list[NodeInputs] = []
                    for rid in batch.request_ids:
                        node_inputs.append(
                            submodule.prepare_inputs(
                                graph_walk=batch.graph_walk,
                                fwd_info=batch.per_request_info[rid],
                                inputs=batch.per_request_input_tensors[rid],
                            )
                        )
                    if self.enable_nvtx:
                        range_pop(synchronize=True)

                    if submodule.can_batch(batch, node_inputs):
                        output = self._execute_batched(batch, node_inputs, submodule)
                    else:
                        output = self._execute_sequential(batch, node_inputs, submodule)
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

        for node_name, submodule in self.submodules.items():
            try:
                if hasattr(submodule, 'forward'):
                    submodule.forward = torch.compile(
                        submodule.forward,
                        fullgraph=False,
                    )
                    logger.info("EncDecEngine: torch.compile applied to %s.forward", node_name)
                if hasattr(submodule, 'forward_batched'):
                    submodule.forward_batched = torch.compile(
                        submodule.forward_batched,
                        fullgraph=False,
                    )
                    logger.info("EncDecEngine: torch.compile applied to %s.forward_batched", node_name)
            except Exception:
                logger.warning("EncDecEngine: torch.compile failed for %s, using eager mode",
                               node_name, exc_info=True)

    def add_request(self, request_id: str) -> None:
        pass  # stateless

    def remove_request(self, request_id: str) -> None:
        for submodule in self.submodules.values():
            submodule.cleanup_request(request_id)
