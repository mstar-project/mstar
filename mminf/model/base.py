

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type
from uuid import uuid4

import torch
import yaml

from mminf.communication.tensors import NameToTensorList
from mminf.engine.ar_engine import KVCacheConfig
from mminf.engine.base import EngineType
from mminf.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Parallel, Sequential, TensorPointerInfo


class NodeSubmodule(torch.nn.Module):
    """
    Base class for node wrapper submodules.

    Separates preprocessing (variable-length list[Tensor] → fixed Tensor)
    from computation (Tensor → NameToTensorList), enabling torch.compile
    and CUDA graphs on the forward() path.

    Engine call pattern:
        preprocessed = submodule.preprocess(graph_walk, **inputs)  # list → tensors
        result = submodule(**preprocessed)                     # tensor → tensor (compilable)
    """

    def preprocess(self, graph_walk: str, **inputs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Convert variable-length list[Tensor] inputs to fixed tensors.
        NOT compiled — handles Python-level variability.

        Default: assert each input has exactly 1 tensor and unwrap it.
        Override for nodes that handle multiple tensors (e.g., stacking images).
        """
        return {k: v[0] for k, v in inputs.items()}

    @abstractmethod
    def forward(self, **kwargs) -> NameToTensorList:
        """
        Pure tensor → NameToTensorList computation.
        Compilable + CUDA-graphable.
        """
        ...


@dataclass
class WorkerGraph:
    section: GraphSection
    graph_walks: set[str] # e.g., prefill, decode, image_gen
    consumes_stream: bool = field(default=False)
    ranks: list[int] = field(default_factory=list)
    _group_id: int = field(default=-1) # used in going from config yaml to worker graphs
    worker_graph_id: str = field(default_factory=lambda: str(uuid4()))


def _combine_sections_sequential_or_parallel(
    section: GraphSection, other: GraphSection,
    comb_type: Type[Sequential] | Type[Parallel]
):
    if isinstance(section, comb_type) and isinstance(other, comb_type):
        section.sections.extend(other.sections)
        return section
    if isinstance(section, comb_type):
        section.sections.append(other)
        return section
    if isinstance(other, comb_type):
        other.sections.insert(0, section)
        return other
    return comb_type([section, other])


def _divide_into_worker_graphs(
    graph: GraphSection,
    graph_walk: str,
    node_to_group_idx: dict[str, int],
    node_groups: list[dict]
) -> list[WorkerGraph]:
    """
    Given a graph, break it into worker graphs
    """
    if isinstance(graph, GraphNode):
        return [WorkerGraph(
            section=graph,
            graph_walks=set([graph_walk]),
            consumes_stream=graph.consumes_stream,
            _group_id=node_to_group_idx[graph.name],
            ranks=node_groups[node_to_group_idx[graph.name]]["ranks"]
        )]

    if isinstance(graph, Sequential):
        worker_graphs = _divide_into_worker_graphs(
            graph.sections[0],
            graph_walk=graph_walk,
            node_to_group_idx=node_to_group_idx,
            node_groups=node_groups
        )

        for i in range(1, len(graph.sections)):
            # Go through it sequentially and merge adjacent sections
            # that are on the same device
            new_worker_graphs = _divide_into_worker_graphs(
                graph.sections[i],
                graph_walk=graph_walk,
                node_to_group_idx=node_to_group_idx,
                node_groups=node_groups
            )
            if new_worker_graphs[0]._group_id == worker_graphs[-1]._group_id and \
                    not new_worker_graphs[0].consumes_stream:
                worker_graphs[-1].section = _combine_sections_sequential_or_parallel(
                    worker_graphs[-1].section, new_worker_graphs.pop(0).section,
                    comb_type=Sequential
                )
            worker_graphs.extend(new_worker_graphs)
        return worker_graphs

    if isinstance(graph, Parallel):
        all_worker_graphs = [
            _divide_into_worker_graphs(
                s, graph_walk=graph_walk,
                node_to_group_idx=node_to_group_idx,
                node_groups=node_groups
            ) for s in graph.sections
        ]
        # parallel sections that are all on the same worker can be merged
        singleton_worker_graphs = [
            s[0] for s in all_worker_graphs if len(s) == 1 and not s[0].consumes_stream
        ]
        group_id_to_worker_graph = {}
        for s in singleton_worker_graphs:
            if s._group_id in group_id_to_worker_graph:
                existing = group_id_to_worker_graph[s._group_id]
                existing.section = _combine_sections_sequential_or_parallel(
                    existing.section, s.section,
                    comb_type=Parallel
                )
            else:
                group_id_to_worker_graph[s._group_id] = s

        return list(group_id_to_worker_graph.values()) + sum([
            s for s in all_worker_graphs if len(s) > 1 or s[0].consumes_stream
        ], start=[]) # remaining worker graphs

    if isinstance(graph, Loop):
        loop_section_worker_graphs = _divide_into_worker_graphs(
            graph.section,
            graph_walk=graph_walk,
            node_to_group_idx=node_to_group_idx,
            node_groups=node_groups
        )
        if len(loop_section_worker_graphs) == 1:
            # fully colocated case
            loop_section_worker_graphs[0].section = graph
            return loop_section_worker_graphs

        # in the disaggregated case, we need to wrap all worker graphs in a loop
        # with the external signals and loop-back signals pre-computed
        for s in loop_section_worker_graphs:
            s.section = Loop(
                section=s.section,
                n_iters=graph.n_iters,
                curr_iter=graph.curr_iter,
                external_inputs=graph.external_inputs,
                loop_back_signals=graph.loop_back_signals,
                outputs=graph.outputs
            )
        return loop_section_worker_graphs


@dataclass
class CurrentForwardMetadata:
    """
    Full-model forward pass-level metadata for running the current
    forward pass
    """
    input_modalities: list[str]
    output_modalities: list[str]
    graph_walk: str
    is_prefill: bool
    kwargs: dict = field(default_factory=dict)


@dataclass
class ForwardPassArgs:
    # full_metadata is at the conductor level
    full_metadata: CurrentForwardMetadata
    inputs: list[GraphEdge]

    # de_persist_tensors are tensors that will be used for the final time and
    # not go into future graph nodes
    unpersist_tensors: list[TensorPointerInfo]

    # e.g., saw EOS or max tokens. Is used to end the request
    request_done: bool =  False

    # step_metadata is at the engine / worker level; and
    # is passed into the fwd pass
    step_metadata: dict = field(default_factory=dict)


class Model(ABC):
    def _get_worker_graphs_for_graph_walk(
        self, graph_walk: str, graph: GraphSection,
        node_groups: list[dict],
    ):
        node_groups = [
            g for g in node_groups if (
                "graph_walks" not in g or graph_walk in g["graph_walks"]
            )
        ]
        node_to_group_idx: dict[str, int] = {}
        for i, group in enumerate(node_groups):
            node_to_group_idx.update({
                name: i for name in group["node_names"]
            })

        return _divide_into_worker_graphs(
            graph,
            graph_walk=graph_walk,
            node_to_group_idx=node_to_group_idx,
            node_groups=node_groups
        )

    def get_worker_graphs(self, config_path: str) -> list[WorkerGraph]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        node_groups = config.get("node_groups")
        if node_groups is None:
            raise KeyError("Config must define `node_groups`.")

        # TODO: merge identical worker graphs from different graph walks
        return sum([
            self._get_worker_graphs_for_graph_walk(graph_walk, graph, node_groups) \
                for graph_walk, graph in self.get_graph_walk_graphs().items()
        ], start=[])

    @abstractmethod
    def get_kv_cache_config(self) -> KVCacheConfig:
        pass

    @abstractmethod
    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        pass

    @abstractmethod
    def get_node_engine_types(self) -> dict[str, EngineType]:
        """Returns node_name -> EngineType enum."""
        pass

    @abstractmethod
    def get_initial_forward_pass_args(
        self, input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        pass

    @abstractmethod
    def get_forward_pass_args(
        self, metadata: CurrentForwardMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        new_tokens: dict[str, list[int]],
    ) -> ForwardPassArgs:
        """
        Called by the conductor.

        **Important**: this sets ForwardPassArgs.request_done, which is used to
        end the request.

        Also extracts per-request metadata that will get passed into the model
        forward pass at the engine level.

        TODO: description
        """
        pass

    @abstractmethod
    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        **kwargs,
    ) -> NameToTensorList:
        """Tokenize prompt and produce initial text tensors.

        Called by the API server data worker to convert raw text input
        into model-specific tensor format (e.g., tokenization). The
        output dict keys are model-specific and will be referenced by
        get_forward_pass_inputs via persist_signals.

        Args:
            prompt: Raw text input from the user, or None if no text.
            input_modalities: List of input modality types for this request.
            output_modalities: List of desired output modality types.
            **kwargs: Model-specific parameters (e.g., from model_kwargs).

        Returns:
            NameToTensorList with model-specific keys, e.g.:
            {"text_inputs": [tokenized_tensor], "system_prompt": [sys_tensor]}
        """
        pass

    @abstractmethod
    def postprocess(
        self, output: torch.Tensor,
        modality: str # text | image | video | audio
    ) -> bytes:
        """
        Given an output of a certain modality, encode and return as bytes.
        This will likely need to overridden with model-specific behavior.

        Modality to expected encoding type:
        - text: utf-8
        - image: png
        """
        return output.cpu().numpy().tobytes()

    @abstractmethod
    def get_submodule(self, node_name: str, device="cpu") -> torch.nn.Module | None:
        """
        Return the nn.Module for this node, or None for dummy mode.
        The engine calls this (via EngineManager) to get the submodule it
        will execute directly with engine-specific wrapping (KV cache,
        FlashInfer, etc.).
        """
        pass
