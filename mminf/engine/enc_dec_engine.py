import logging

import torch

from mminf.engine.ar_engine import KVCacheConfig
from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.utils.flashinfer_utils import FlashInferAttentionNoCache
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
        kv_cache_config: dict[str, KVCacheConfig] = {},
        enable_nvtx: bool = False
    ):
        super().__init__(enable_nvtx=enable_nvtx)
        self.submodules: dict[str, torch.nn.Module] = {}
        self.attn_config = kv_cache_config
        self.attn_wrapper = {}
        self.device = None

    def engine_type(self) -> EngineType:
        return EngineType.ENC_DEC

    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        model_config: dict,
        device: torch.device,
    ) -> None:
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
        self.submodules = submodules
        self.device = device

        cfgs_override: dict[str, KVCacheConfig] = model_config.get(
            "kv_cache", None 
        )
        self.attn_config = {
            **self.attn_config,
            **cfgs_override
        }

        for node, cfg in self.attn_config.items():
            self.attn_wrapper[node] = FlashInferAttentionNoCache(
                device=device,
                workspace_buffer=self.workspace_buffer,
                head_dim=cfg.head_dim,
                max_seq_len=cfg.max_seq_len,
                num_qo_heads=cfg.num_qo_heads,
                num_kv_heads=cfg.num_kv_heads
            )

    def _can_batch_inputs(self, batch: NodeBatch, submodule) -> bool:
        """Check if all requests have same-shaped inputs for stacking."""
        if len(batch.request_ids) <= 1:
            return False
        
        # Some modules, like ViT, can batch via packing inputs even if the
        # inputs are not the same shape, so we push this check to the model
        return submodule.can_batch(batch)


    def _execute_batched(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Stack same-shaped inputs and run a single forward pass."""
        request_ids = batch.request_ids

        # Preprocess all requests
        all_preprocessed = submodule.preprocess(
            batch.graph_walk,
            per_request_inputs=[
                batch.per_request_input_tensors[rid] for rid in request_ids
            ],
            request_ids=batch.request_ids,
            per_request_metadata=batch.per_request_metadata,
            attn_wrapper=self.attn_wrapper.get(batch.node_name, None)
        )


        batched_output = submodule.forward_batched(
            graph_walk=batch.graph_walk,
            packed_inputs=all_preprocessed,
            request_ids=request_ids,
            per_request_metadata=batch.per_request_metadata,
        )

        return NodeOutput(per_request_output_tensors=batched_output)

    def _execute_sequential(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Original per-request execution."""
        outputs = {}
        for rid in batch.request_ids:
            inputs = batch.per_request_input_tensors.get(rid, {})
            metadata = batch.per_request_metadata.get(rid, {})
            if hasattr(submodule, 'preprocess'):
                preprocessed = submodule.preprocess(
                    batch.graph_walk,
                    per_request_inputs=[inputs],
                    request_ids=[rid],
                    per_request_metadata={
                        rid: batch.per_request_metadata.get(rid, {})
                    },
                    attn_wrapper=self.attn_wrapper.get(batch.node_name, None)
                )
                outputs[rid] = submodule(**preprocessed, **metadata)
                print("hi")
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
            range_push(f"engine.enc_dec.{batch.node_name}.{batch.graph_walk}")

        submodule = self.submodules.get(batch.node_name)
        if submodule is None:
            output = NodeOutput(
                per_request_output_tensors={rid: {} for rid in batch.request_ids}
            )
            if self.enable_nvtx:
                range_pop()
            return output

        print(batch.node_name, [rid for rid in batch.request_ids])
        try:
            with torch.no_grad():
                if self._can_batch_inputs(batch, submodule):
                    return self._execute_batched(batch, submodule)
                else:
                    return self._execute_sequential(batch, submodule)
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
                    logger.info("EncDecEngine: torch.compile applied to %s", node_name)
            except Exception:
                logger.warning("EncDecEngine: torch.compile failed for %s, using eager mode",
                               node_name, exc_info=True)

    def add_request(self, request_id: str) -> None:
        pass  # stateless

    def remove_request(self, request_id: str) -> None:
        pass  # stateless
