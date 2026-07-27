import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.base import NodeBatch
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule


class Submodule(NodeSubmodule):
    """Identity node — the forward is a no-op so a measured step is (almost)
    pure runtime overhead: scheduler decision, batch assembly, engine
    entry/exit, output routing."""

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        return NodeInputs(
            tensor_inputs={
                "x": inputs["x"][0]
            }
        )

    def forward(
        self,
        engine_inputs: ModelInputsFromEngine,
        x: torch.Tensor,
        **kwargs
    ) -> NameToTensorList:
        return {
            "x": [x]
        }

    # ---- batching -----------------------------------------------------
    # Without these the stateless engine falls back to _execute_sequential,
    # which runs one forward per request. That hides exactly the effect a
    # B=16 sweep is meant to show (dispatch overhead amortizing over a
    # batch), so the no-op node opts into real batched execution.

    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict:
        # Keep the per-request tensors as a list; shapes are identical here but
        # not stacking keeps the node free of any real compute.
        return {"xs": [inp.tensor_inputs["x"] for inp in inputs]}

    def forward_batched(
        self,
        engine_inputs: ModelInputsFromEngine,
        xs: list[torch.Tensor],
        **kwargs
    ) -> dict[str, NameToTensorList]:
        return {
            rid: {"x": [x]}
            for rid, x in zip(engine_inputs.request_ids, xs, strict=True)
        }

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        iter = request_info.dynamic_loop_iter_counts.get("loop", 0)
        n_iter = request_info.step_metadata["steps"]
        return {"loop"} if iter + 1 >= n_iter else set()
