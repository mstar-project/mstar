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
        """Batched execution path for shape-homogeneous ENC_DEC submodules.

        Contract:
          * ``submodule.preprocess`` receives ``per_request_inputs`` as a
            ``list[NameToTensorList]`` ordered by ``batch.request_ids`` and
            returns a single packed dict (tensors stacked along dim 0).
          * ``submodule.forward_batched(graph_walk, request_ids,
            packed_inputs, per_request_info)`` runs the stacked forward
            and returns ``dict[rid -> NameToTensorList]`` directly (each
            per-rid slot typically slices the batch dim to ``[1, ...]`` so
            downstream submodules see consistent shapes vs the sequential
            path).

        Mirrors the audio_codec / AR batched conventions so submodules can
        share a pattern across engine types.
        """
        request_ids = batch.request_ids
        per_request_inputs = [
            batch.per_request_input_tensors[rid] for rid in request_ids
        ]

        packed = submodule.preprocess(
            graph_walk=batch.graph_walk,
            per_request_inputs=per_request_inputs,
            request_ids=request_ids,
            per_request_info=batch.per_request_info,
        )

        if not hasattr(submodule, 'forward_batched'):
            raise RuntimeError(
                f"{type(submodule).__name__}.can_batch returned True but "
                "the submodule does not implement forward_batched. Either "
                "add forward_batched or return False from can_batch."
            )

        per_rid_outputs = submodule.forward_batched(
            graph_walk=batch.graph_walk,
            request_ids=request_ids,
            packed_inputs=packed,
            per_request_info=batch.per_request_info,
        )
        return NodeOutput(per_request_output_tensors=per_rid_outputs)

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
        pass  # stateless
