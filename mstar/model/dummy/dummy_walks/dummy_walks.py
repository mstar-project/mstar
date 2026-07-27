"""
Dummy walks: a model with no compute, for measuring the overhead of our system
in the worker, conductor, api server, etc.

One identity node, re-dispatched as a fresh graph walk K times. Every step
returns to the conductor (``get_partition_forward_pass_args`` decides the next
walk), so a step costs the full dispatch hop: worker → conductor → worker, on
top of everything ``dummy_loop`` measures.

``steps`` (the K above) is a per-request ``model_kwargs`` value:

    client.generate(text="", output_modalities=("tensor",),
                    model_kwargs={"steps": 100})

Note the node executes ``steps + 1`` times: ``steps`` under the "walk" walk
plus one final "last_walk" execution that emits to the client. Divide measured
time by the ``n=`` exec count that ``--log-stats`` reports rather than by
``steps``.
"""

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.base import EngineType
from mstar.engine.kv_store import KVCacheConfig
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.dummy.dummy_walks.submodules import Submodule

DEFAULT_DUMMY_SIZE = (512, 512)
MAX_STEPS = 1000
DEFAULT_STEPS = 50


def _resolve_steps(model_kwargs: dict | None) -> int:
    steps = int((model_kwargs or {}).get("steps", DEFAULT_STEPS))
    if not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps={steps} out of range; must be in [1, {MAX_STEPS}]")
    return steps


class DummyWalks(Model):
    def __init__(
        self,
        model_path_hf: str = "",
        cache_dir: str | None = None,
        **kwargs,
    ):
        # No weights to load; the ctor only exists because the conductor and
        # worker entry points always construct models with these kwargs.
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        # NameToTensorList maps a name to a *list* of tensors.
        return {
            "x": [torch.zeros(
                kwargs.get("tensor_size", DEFAULT_DUMMY_SIZE),
                dtype=torch.float32,
            )]
        }

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,  # text | image | video | audio | tensor
        request_kwargs: dict | None = None,
    ) -> bytes:
        return output.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        return []

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        return {
            # Intermediate step: the output is persisted at the conductor and
            # fed back in as the next walk's input.
            "walk": GraphNode(
                name="node",
                input_names=["x"],
                outputs=[
                    GraphEdge(
                        next_node=EMPTY_DESTINATION, name="y",
                        persist=True
                    )
                ],
            ),
            # Final step: hand the tensor to the client instead.
            "last_walk": GraphNode(
                name="node",
                input_names=["x"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT, name="y",
                        output_modality="tensor",
                    )
                ],
            ),
        }

    def get_node_engine_types(self) -> dict[str, EngineType]:
        # STATELESS, not KV_CACHE — see dummy_loop for why.
        return {"node": EngineType.STATELESS}

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        steps = _resolve_steps(model_kwargs)
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk="walk",
            is_prefill=False,
            kwargs={
                "steps": steps,
                "curr_step": 0
            }
        )
        edge = GraphEdge(next_node="node", name="x")
        edge.tensor_info = input_signals["x"]
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=[edge],
            unpersist_tensors=[input_signals["x"]],
            step_metadata={}
        )

    def get_submodule(
        self, node_name: str, device="cpu", **kwargs
    ) -> torch.nn.Module | None:
        assert node_name == "node"
        return Submodule()

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        if partition_metadata.graph_walk == "last_walk":
            return ForwardPassArgs(
                full_metadata=partition_metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True
            )

        partition_metadata.kwargs["curr_step"] += 1
        curr_step = partition_metadata.kwargs["curr_step"]
        steps = partition_metadata.kwargs["steps"]
        if curr_step == steps:
            partition_metadata.graph_walk = "last_walk"

        edge = GraphEdge(next_node="node", name="x")
        edge.tensor_info = persist_signals["y"]

        return ForwardPassArgs(
            full_metadata=partition_metadata,
            inputs=[edge],
            unpersist_tensors=[persist_signals["y"]],
        )
