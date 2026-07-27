"""
Dummy loop: a model with no compute, for measuring the overhead of our system
in the worker, conductor, api server, etc.

A single graph walk holding a ``Loop`` of K iterations over one identity node.
The loop body stays on the worker, so a step costs scheduler decision + batch
assembly + engine entry/exit + output routing, but *no* conductor round trip.
Compare against ``dummy_walks``, which pays that round trip every step.

``steps`` (the K above) is a per-request ``model_kwargs`` value:

    client.generate(text="", output_modalities=("tensor",),
                    model_kwargs={"steps": 100})
"""

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.base import EngineType
from mstar.engine.kv_store import KVCacheConfig
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.dummy.dummy_loop.submodules import Submodule

DEFAULT_DUMMY_SIZE = (512, 512)
MAX_ITERS = 1000
DEFAULT_STEPS = 50


def _resolve_steps(model_kwargs: dict | None) -> int:
    steps = int((model_kwargs or {}).get("steps", DEFAULT_STEPS))
    if not 1 <= steps <= MAX_ITERS:
        raise ValueError(
            f"steps={steps} out of range; the Loop is built with "
            f"max_iters={MAX_ITERS}, so steps must be in [1, {MAX_ITERS}]"
        )
    return steps


class DummyLoop(Model):
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
        # NameToTensorList maps a name to a *list* of tensors — the data
        # worker hands these straight to store_and_return_tensor_info.
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
        # The tensor comes back on the worker's device; view(uint8) needs a
        # contiguous last dim.
        return output.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        return []

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        return {
            "walk": Loop(
                name="loop",
                section=GraphNode(
                    name="node",
                    input_names=["x"],
                    outputs=[
                        GraphEdge(next_node="node", name="x")
                    ],
                ),
                max_iters=MAX_ITERS,
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="x",
                        output_modality="tensor"
                    )
                ]
            )
        }

    def get_node_engine_types(self) -> dict[str, EngineType]:
        # STATELESS, not KV_CACHE: the KV-cache engine indexes every node into
        # the KVManagement built from get_kv_cache_config(), so declaring
        # KV_CACHE with an empty config raises KeyError at load time.
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
            }
        )
        edge = GraphEdge(next_node="node", name="x")
        edge.tensor_info = input_signals["x"]
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=[edge],
            unpersist_tensors=[input_signals["x"]],
            # step_metadata reaches the worker, where Submodule.check_stop
            # reads it to end the loop after `steps` iterations.
            step_metadata={
                "steps": steps,
            }
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
        # One graph walk: the loop already ran every iteration on the worker.
        return ForwardPassArgs(
            full_metadata=partition_metadata,
            inputs=[],
            unpersist_tensors=[],
            request_done=True
        )
