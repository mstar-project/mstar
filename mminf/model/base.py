from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Type
from uuid import uuid4

import torch
import yaml

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    StreamingConnectionState,
)
from mminf.engine.base import EngineType
from mminf.engine.kv_store import KVCacheConfig
from mminf.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Parallel,
    Sequential,
    TensorPointerInfo,
)
from mminf.utils.sampling import SamplingConfig

DECODE = "decode"
MAX_OUTPUT_TOKENS = 2048


@dataclass
class TensorAndMetadata:
    data: torch.Tensor
    metadata: dict = field(default_factory=dict)


@dataclass
class WorkerGraph:
    section: GraphSection
    graph_walks: set[str]  # e.g., prefill, decode, image_gen
    consumes_stream: bool = field(default=False)
    ranks: list[int] = field(default_factory=list)
    _group_id: int = field(default=-1)  # used in going from config yaml to worker graphs
    worker_graph_id: str = field(default_factory=lambda: str(uuid4()))


def _combine_sections_sequential_or_parallel(
    section: GraphSection, other: GraphSection, comb_type: Type[Sequential] | Type[Parallel]
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
    node_groups: list[dict],
    input_streams: set[str],
) -> list[WorkerGraph]:
    """
    Given a graph, break it into worker graphs
    """
    if isinstance(graph, GraphNode):
        # Mutate via the GraphNode helper so the ReadySignals instances built in
        # __post_init__ keep referencing the live set (they captured by reference).
        graph._register_streaming(input_streams.intersection(graph.input_names))

        return [
            WorkerGraph(
                section=graph,
                graph_walks=set([graph_walk]),
                consumes_stream=graph.consumes_stream,
                _group_id=node_to_group_idx[graph.name],
                ranks=node_groups[node_to_group_idx[graph.name]]["ranks"],
            )
        ]

    if isinstance(graph, Sequential):
        worker_graphs = _divide_into_worker_graphs(
            graph.sections[0],
            graph_walk=graph_walk,
            node_to_group_idx=node_to_group_idx,
            node_groups=node_groups,
            input_streams=input_streams,
        )

        for i in range(1, len(graph.sections)):
            # Go through it sequentially and merge adjacent sections
            # that are on the same device
            new_worker_graphs = _divide_into_worker_graphs(
                graph.sections[i],
                graph_walk=graph_walk,
                node_to_group_idx=node_to_group_idx,
                node_groups=node_groups,
                input_streams=input_streams,
            )
            if (
                new_worker_graphs[0]._group_id == worker_graphs[-1]._group_id
                and not new_worker_graphs[0].consumes_stream
            ):
                worker_graphs[-1].section = _combine_sections_sequential_or_parallel(
                    worker_graphs[-1].section, new_worker_graphs.pop(0).section, comb_type=Sequential
                )
            worker_graphs.extend(new_worker_graphs)
        return worker_graphs

    if isinstance(graph, Parallel):
        all_worker_graphs = [
            _divide_into_worker_graphs(
                s,
                graph_walk=graph_walk,
                node_to_group_idx=node_to_group_idx,
                node_groups=node_groups,
                input_streams=input_streams,
            )
            for s in graph.sections
        ]
        # parallel sections that are all on the same worker can be merged
        singleton_worker_graphs = [s[0] for s in all_worker_graphs if len(s) == 1 and not s[0].consumes_stream]
        group_id_to_worker_graph = {}
        for s in singleton_worker_graphs:
            if s._group_id in group_id_to_worker_graph:
                existing = group_id_to_worker_graph[s._group_id]
                existing.section = _combine_sections_sequential_or_parallel(
                    existing.section, s.section, comb_type=Parallel
                )
            else:
                group_id_to_worker_graph[s._group_id] = s

        return list(group_id_to_worker_graph.values()) + sum(
            [s for s in all_worker_graphs if len(s) > 1 or s[0].consumes_stream], start=[]
        )  # remaining worker graphs

    if isinstance(graph, Loop):
        loop_section_worker_graphs = _divide_into_worker_graphs(
            graph.section,
            graph_walk=graph_walk,
            node_to_group_idx=node_to_group_idx,
            node_groups=node_groups,
            input_streams=input_streams,
        )
        ext_inps = set([
            (name, dest) for name, dest in graph._external_inputs \
                if name not in input_streams
        ])
        for s in loop_section_worker_graphs:
            # ``accumulated_outputs`` must be propagated to the per-worker
            # reconstruction — otherwise ``Loop.cache_outputs`` sees an
            # empty ``_accumulated_output_names`` on the worker side, every
            # per-iter tensor silently falls on the floor, and nothing ever
            # reaches ``EMIT_TO_CLIENT`` (symptom: 4 "Deferring cleanup"
            # warnings + empty client response in the V-JEPA 2 rollout run).
            s.section = Loop(
                section=s.section,
                max_iters=graph.max_iters,
                curr_iter=graph.curr_iter,
                outputs=graph.outputs,
                name=graph.name,
                accumulated_outputs=graph.accumulated_outputs,
                _external_inputs=ext_inps,
                _loop_back_inputs=graph._loop_back_inputs,
            )
        return loop_section_worker_graphs


@dataclass
class ForwardPassArgs:
    # full_metadata is at the conductor level
    full_metadata: CurrentForwardConductorMetadata
    inputs: list[GraphEdge]

    # de_persist_tensors are tensors that will be used for the final time and
    # not go into future graph nodes
    unpersist_tensors: list[TensorPointerInfo]

    # e.g., saw EOS or max tokens. Is used to end the request
    request_done: bool = False

    # step_metadata is at the engine / worker level; and
    # is passed into the fwd pass
    step_metadata: dict = field(default_factory=dict)


class Model(ABC):
    def _get_worker_graphs_for_graph_walk(
        self,
        graph_walk: str,
        graph: GraphSection,
        node_groups: list[dict],
    ):
        node_groups = [g for g in node_groups if ("graph_walks" not in g or graph_walk in g["graph_walks"])]
        node_to_group_idx: dict[str, int] = {}
        for i, group in enumerate(node_groups):
            node_to_group_idx.update({name: i for name in group["node_names"]})

        partition = "default"
        for part in self.get_partitions():
            if graph_walk in part.graph_walks:
                partition = part.name
                break
        input_streams = set()
        for conn in self.get_partition_topology().connections:
            if conn.to_partition == partition:
                input_streams.add(conn.edge_name)

        return _divide_into_worker_graphs(
            graph,
            graph_walk=graph_walk,
            node_to_group_idx=node_to_group_idx,
            node_groups=node_groups,
            input_streams=input_streams,
        )

    def get_worker_graphs(self, config_path: str) -> list[WorkerGraph]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        node_groups = config.get("node_groups")
        if node_groups is None:
            raise KeyError("Config must define `node_groups`.")

        # TODO: merge identical worker graphs from different graph walks
        return sum(
            [
                self._get_worker_graphs_for_graph_walk(graph_walk, graph, node_groups)
                for graph_walk, graph in self.get_graph_walk_graphs().items()
            ],
            start=[],
        )

    @abstractmethod
    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        """Return per-node KV cache configs.

        Maps AR node name -> KVCacheConfig. Nodes not in the dict
        fall back to the first config (for models where all AR nodes
        share the same config, e.g., Bagel's LLM / LLM_cfg_text / LLM_cfg_img).
        """
        pass

    def get_sampling_config(
        self, node_name: str,
        model_kwargs: dict | None = None,
    )  -> SamplingConfig | None:
        return SamplingConfig()


    @abstractmethod
    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        pass

    @abstractmethod
    def get_node_engine_types(self) -> dict[str, EngineType]:
        """Returns node_name -> EngineType enum."""
        pass

    @abstractmethod
    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        pass

    @abstractmethod
    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Tokenize prompt and produce initial tensors for the request.

        Called by the API server data worker AFTER it has loaded raw
        multimodal tensors from file_paths (images, audio, video).
        The model may inspect the raw tensors dict and compute additional
        derived tensors (e.g., Qwen3-Omni computes ``pixel_values``,
        ``image_grid_thw``, ``audio_features``, ``audio_seqlens`` from the
        raw ``image_inputs`` / ``audio_inputs`` / ``video_inputs``).

        Args:
            prompt: Raw text input from the user, or None if no text.
            input_modalities: List of input modality types for this request.
            output_modalities: List of desired output modality types.
            tensors: Raw modality tensors already loaded by the data worker
                (``image_inputs``, ``audio_inputs``, ``video_inputs``).
                The model may read these to compute derived tensors.
                Models that don't need the raw tensors may ignore this.
            **kwargs: Model-specific parameters (e.g., from model_kwargs).

        Returns:
            NameToTensorList with tensors to MERGE into the request's
            tensor dict.  Typically includes ``text_inputs`` plus any
            model-specific derived tensors.  The returned dict is merged
            into the existing ``tensors`` dict via ``dict.update``.
        """
        pass

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        import torchvision

        img = torchvision.io.decode_image(filepath).to(device)  # uint8 CxHxW
        img = img.float() / 255.0

        return TensorAndMetadata(img)

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(filepath, sample_rate=16000, num_channels=1)
        audio = decoder.get_all_samples().data[0]
        return TensorAndMetadata(data=audio, metadata=dict(sample_rate=16000, num_channels=1))

    def load_video(self, filepath: str, device: str):
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(filepath, device=self.device)
        video = torch.stack([frame for frame in decoder]).float() / 255.0
        return TensorAndMetadata(data=video, metadata=asdict(decoder.metadata))

    @abstractmethod
    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,  # text | image | video | audio
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

    def get_max_output_tokens(self, **model_kwargs):
        return model_kwargs.get("max_output_tokens", MAX_OUTPUT_TOKENS)

    def get_autocast_dtype(self):
        return torch.bfloat16

    # ------------------------------------------------------------------
    # Partition API (optional, backward-compatible defaults)
    # ------------------------------------------------------------------

    def get_partition_topology(self):
        """Return a PartitionTopology describing async partitions and streaming connections.

        Default: single "default" partition with no connections.
        Override for models with async partitions (e.g., Orpheus LLM + SNAC).
        """
        from mminf.streaming.topology import PartitionTopology

        return PartitionTopology(partitions=["default"], connections=[])

    def get_partitions(self) -> list[PartitionDefinition]:
        """Return partition definitions.

        Default: single "default" partition containing all graph walks.
        Override for models with async partitions (e.g., Orpheus LLM + SNAC).
        """
        walks = set(self.get_graph_walk_graphs().keys())
        return [
            PartitionDefinition(
                name="default",
                graph_walks=walks,
                initial_walk=None,
                producer_partitions=[],
            )
        ]

    @abstractmethod
    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        new_tokens: dict[str, list[int]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> "ForwardPassArgs":
        """Return the next forward pass arguments for a specific partition.

        Called by the conductor after each completed forward pass to determine
        the next graph walk, inputs, and whether the request is done.

        ``incoming_connections`` contains streaming-specific state (token counts,
        producer_done) for consumer partitions. For single-partition models,
        this will be ``None`` or an empty list.
        """
        pass
