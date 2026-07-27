import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.base import NodeBatch
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule


class Submodule(NodeSubmodule):
    """Identity node — see ``dummy_loop.submodules.Submodule``. This variant
    renames its output (``x`` in, ``y`` out) so the conductor has to route the
    tensor back through a new graph walk on every step."""

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
            "y": [x]
        }

    # ---- batching (see dummy_loop.submodules for why) ------------------

    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict:
        return {"xs": [inp.tensor_inputs["x"] for inp in inputs]}

    def forward_batched(
        self,
        engine_inputs: ModelInputsFromEngine,
        xs: list[torch.Tensor],
        **kwargs
    ) -> dict[str, NameToTensorList]:
        return {
            rid: {"y": [x]}
            for rid, x in zip(engine_inputs.request_ids, xs, strict=True)
        }
