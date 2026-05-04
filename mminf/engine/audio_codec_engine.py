import logging

import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.engine.cuda_graph_runner import CodecCudaGraphRunner
from mminf.model.submodule_base import ARNodeInputs, ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mminf.utils.profiler import range_pop, range_push

logger = logging.getLogger(__name__)


class AudioCodecEngine(BaseEngine):
    """
    Wraps submodules for audio codec forward passes.
    """

    def __init__(
        self, enable_nvtx: bool = False,
        autocast_dtype=torch.bfloat16,
        **kwargs
    ):
        super().__init__(enable_nvtx=enable_nvtx)
        self.submodules: dict[str, NodeSubmodule] = {}
        self.device = None
        self.autocast_dtype = autocast_dtype
        self.cuda_graph_runners: dict[str, CodecCudaGraphRunner] = {}

        # Dedup set for "cuda graphs captured but not usable for this shape"
        # warnings — each unique miss shape is logged at most once.
        #TODO: Remove in production.
        self._logged_graph_misses: set[tuple] = set()

    def engine_type(self) -> EngineType:
        return EngineType.AUDIO_CODEC

    def has_autocast(self):
        return False

    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
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
            range_push(f"engine.audio_codec.{batch.node_name}.{batch.graph_walk}.bs{len(batch.request_ids)}")

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
                skipped_rids = []
                node_inputs: list[NodeInputs] = []
                for rid in batch.request_ids:
                    req_inputs = submodule.prepare_inputs(
                        graph_walk=batch.graph_walk,
                        fwd_info=batch.per_request_info[rid],
                        inputs=batch.per_request_input_tensors[rid],
                    )
                    if req_inputs is None:
                        skipped_rids.append(rid)
                    else:
                        node_inputs.append(req_inputs)
                skipped_rids = set(skipped_rids)

                # filter out skipped rids from the batch
                batch.request_ids = [rid for rid in batch.request_ids if rid not in skipped_rids]
                batch.per_request_info = {
                    rid: info for rid, info in batch.per_request_info.items() \
                        if rid not in skipped_rids
                    }
                output = self._dispatch(batch, node_inputs, submodule)
                for rid, info in batch.per_request_info.items():
                    if rid in skipped_rids:
                        continue
                    submodule.postprocess(
                        request_id=rid,
                        request_info=info,
                        outputs=output.per_request_output_tensors.get(rid, {})
                    )
                output.per_request_output_tensors.update({
                    rid: {} for rid in skipped_rids
                })
                return output
        finally:
            if self.enable_nvtx:
                range_pop()

    def _dispatch(
        self, batch: NodeBatch,
        inputs: list[NodeInputs],
        submodule: NodeSubmodule
    ) -> NodeOutput:
        """Pick cuda_graph / batched / sequential, with eager fallback if
        the CUDA-graph runner rejects the batch (e.g. SNAC frame mismatch).
        """
        if self._can_use_cuda_graph(batch, submodule, inputs):
            return self._execute_with_cuda_graph(batch, submodule, inputs)
        if submodule.can_batch(batch, inputs):
            return self._execute_batched(batch, submodule, inputs)
        return self._execute_sequential(batch, submodule, inputs)

    def _can_use_cuda_graph(
        self, batch: NodeBatch, submodule: NodeSubmodule,
        inputs: NodeInputs
    ) -> bool:
        runner = self.cuda_graph_runners.get(batch.node_name)
        if runner is None:
            return False
        bs = len(batch.request_ids)
        #TODO: Remove in production.
        if not submodule.can_use_cuda_graphs(batch, inputs):
            self._log_graph_miss(
                node_name=batch.node_name,
                graph_walk=batch.graph_walk,
                bs=bs,
                runner=runner,
                reason="submodule.can_use_cuda_graphs() returned False",
            )
            return False
        if not runner.can_run(batch_size=bs, graph_walk=batch.graph_walk):
            self._log_graph_miss(
                node_name=batch.node_name,
                graph_walk=batch.graph_walk,
                bs=bs,
                runner=runner,
                reason="no captured graph matches this (bs, graph_walk)",
            )
            return False
        return True

    def _log_graph_miss(
        self,
        node_name: str,
        graph_walk: str,
        bs: int,
        runner: CodecCudaGraphRunner,
        reason: str,
    ) -> None:
        """Warn (once per unique miss shape) when a runner exists but the
        current request can't use a captured graph.
        """
        if not runner.graphs:
            return
        miss_key = (node_name, graph_walk, bs, reason)
        if miss_key in self._logged_graph_misses:
            return
        self._logged_graph_misses.add(miss_key)

        captured_for_walk = sorted(
            {key[1] for key in runner.graphs.keys() if key[0] == graph_walk}
        )
        captured_walks = sorted({key[0] for key in runner.graphs.keys()})
        logger.warning(
            "[cuda-graph miss] node=%s graph_walk=%s requested=(bs=%d) "
            "reason='%s' captured_bs_for_walk=%s captured_walks=%s — falling back to eager.",
            node_name, graph_walk, bs,
            reason, captured_for_walk or "<none>", captured_walks,
        )

    def _execute_with_cuda_graph(
        self, batch: NodeBatch,
        submodule: NodeSubmodule,
        inputs: list[ARNodeInputs]
    ) -> NodeOutput:
        """Replay the captured graph. Runner handles preprocess + replay +
        per-rid split; we only wrap the result.
        """
        if self.enable_nvtx:
            range_push("codec.cuda_graph.run")
        runner = self.cuda_graph_runners[batch.node_name]
        per_rid = runner.run(
            graph_walk=batch.graph_walk,
            request_ids=batch.request_ids,
            inputs=inputs,
            per_request_info=batch.per_request_info,
            submodule=submodule,
        )
        if self.enable_nvtx:
            range_pop()
        return NodeOutput(per_request_output_tensors=per_rid)

    def _execute_batched(
        self, batch: NodeBatch,
        submodule: NodeSubmodule,
        inputs: list[NodeInputs]
    ) -> NodeOutput:
        """Eager batched path: preprocess → forward_batched(packed) → per-rid."""
        engine_inputs = ModelInputsFromEngine(
            request_ids=batch.request_ids,
            per_request_info=batch.per_request_info,
        )

        if self.enable_nvtx:
            range_push("codec.batched.preprocess", synchronize=False)
        packed = submodule.preprocess(
            graph_walk=batch.graph_walk,
            engine_inputs=engine_inputs,
            inputs=inputs
        )

        if self.enable_nvtx:
            range_push("codec.batched.forward")
        outputs = submodule.forward_batched(
            graph_walk=batch.graph_walk,
            engine_inputs=engine_inputs,
            **packed
        )
        if self.enable_nvtx:
            range_pop()
        return NodeOutput(per_request_output_tensors=outputs)

    def _execute_sequential(
        self, batch: NodeBatch,
        submodule: NodeSubmodule,
        inputs: list[NodeInputs]
    ) -> NodeOutput:
        """Execute each request individually."""
        outputs = {}
        for i, rid in enumerate(batch.request_ids):
            node_input = inputs[i]

            fwd_info = batch.per_request_info[rid]
            engine_inputs = ModelInputsFromEngine(
                request_ids=[rid],
                per_request_info={rid: fwd_info},
            )

            if self.enable_nvtx:
                range_push(f"codec.preprocess.{i}", synchronize=False)
            preprocessed = submodule.preprocess(
                batch.graph_walk,
                engine_inputs=engine_inputs,
                inputs=[node_input],
            )
            if self.enable_nvtx:
                range_pop(synchronize=False)

            if self.enable_nvtx:
                range_push(f"codec.forward.{i}")
            outputs[rid] = submodule.forward(
                batch.graph_walk,
                engine_inputs=engine_inputs,
                **preprocessed
            )
            if self.enable_nvtx:
                range_pop()
        return NodeOutput(per_request_output_tensors=outputs)

    def warmup(self) -> None:
        """Capture CUDA graphs for submodules that support it.

        Submodules that opt in expose:
          - ``cuda_graph_runner`` attribute (set to the captured runner here)
          - ``get_cuda_graph_configs(device)`` returning at least one config
          - ``cuda_graph_forward(**static_inputs)`` that the runner captures
        """
        if not torch.cuda.is_available() or self.device is None:
            return

        from mminf.engine.cuda_graph_runner import CodecCudaGraphRunner

        for node_name, submodule in self.submodules.items():
            if not hasattr(submodule, 'get_cuda_graph_configs'):
                continue

            runner = CodecCudaGraphRunner(
                submodule_name=node_name,
                submodule=submodule,
                device=self.device,
            )
            runner.enable_nvtx = self.enable_nvtx
            runner.warmup_and_capture()
            if runner.graphs:
                self.cuda_graph_runners[node_name] = runner
                logger.info(
                    "AudioCodecEngine: CUDA graphs captured for %s (%d graphs)",
                    node_name, len(runner.graphs),
                )

    def add_request(self, request_id: str) -> None:
        pass  # stateless

    def remove_request(self, request_id: str) -> None:
        for submodule in self.submodules.values():
            submodule.cleanup_request(request_id)
