

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type
from uuid import uuid4

import torch
import yaml

from mminf.graph.base import GraphPointer, GraphSection, GraphStage, Loop, Parallel, Sequential, TensorPointerInfo


STREAM_OUT = "stream_out"
SPECIAL_DESTINATIONS = {STREAM_OUT}  # add RELAY etc. later


@dataclass
class Subgraph:
    section: GraphSection
    phases: set[str] # e.g., prefill, decode, image_gen 
    consumes_stream: bool = field(default=False)
    ranks: list[int] = field(default_factory=list)
    _group_id: int = field(default=-1) # used in going from config yaml to subgraphs
    subgraph_id: str = field(default_factory=lambda: str(uuid4()))


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


def _divide_into_subgraphs(
    graph: GraphSection,
    phase: str,
    stage_to_group_idx: dict[str, int],
    stage_groups: list[dict]
) -> list[Subgraph]:
    """
    Given a graph, break it into subgraphs 
    """
    if isinstance(graph, GraphStage):
        return [Subgraph(
            section=graph,
            phases=set([phase]),
            consumes_stream=graph.consumes_stream,
            _group_id=stage_to_group_idx[graph.name],
            ranks=stage_groups[stage_to_group_idx[graph.name]]["ranks"]
        )]
    
    if isinstance(graph, Sequential):
        subgraphs = _divide_into_subgraphs(
            graph.sections[0],
            phase=phase,
            stage_to_group_idx=stage_to_group_idx,
            stage_groups=stage_groups
        )

        for i in range(1, len(graph.sections)):
            # Go through it sequentially and merge adjacent sections
            # that are on the same device
            new_subgraphs = _divide_into_subgraphs(
                graph.sections[i],
                phase=phase,
                stage_to_group_idx=stage_to_group_idx,
                stage_groups=stage_groups
            )
            if new_subgraphs[0]._group_id == subgraphs[-1]._group_id and \
                    not new_subgraphs[0].consumes_stream:
                subgraphs[-1].section = _combine_sections_sequential_or_parallel(
                    subgraphs[-1].section, new_subgraphs.pop(0).section,
                    comb_type=Sequential
                )
            subgraphs.extend(new_subgraphs)
        return subgraphs
    
    if isinstance(graph, Parallel):
        all_subgraphs = [
            _divide_into_subgraphs(
                s, phase=phase,
                stage_to_group_idx=stage_to_group_idx,
                stage_groups=stage_groups
            ) for s in graph.sections
        ]
        # parallel sections that are all on the same worker can be merged
        singleton_subgraphs = [
            s[0] for s in all_subgraphs if len(s) == 1 and not s[0].consumes_stream
        ]
        group_id_to_subgraph = {}
        for s in singleton_subgraphs:
            if s._group_id in group_id_to_subgraph:
                group_id_to_subgraph[s._group_id] = _combine_sections_sequential_or_parallel(
                    group_id_to_subgraph[s._group_id], s.section,
                    comb_type=Parallel
                )
            else:
                group_id_to_subgraph[s._group_id] = s

        return list(group_id_to_subgraph.values()) + sum([
            s for s in all_subgraphs if len(s) > 1 or s[0].consumes_stream
        ], start=[]) # remaining subgraphs
    
    if isinstance(graph, Loop):
        loop_section_subgraphs = _divide_into_subgraphs(
            graph.section,
            phase=phase,
            stage_to_group_idx=stage_to_group_idx,
            stage_groups=stage_groups
        )
        if len(loop_section_subgraphs) == 1:
            # fully colocated case
            loop_section_subgraphs[0].section = graph
            return loop_section_subgraphs
        
        # in the disaggregated case, we need to wrap all subgraphs in a loop
        # with the external signals and loop-back signals pre-computed
        for s in loop_section_subgraphs:
            s.section = Loop(
                section=s.section,
                n_iters=graph.n_iters,
                curr_iter=graph.curr_iter,
                external_inputs=graph.external_inputs,
                loop_back_signals=graph.loop_back_signals,
                outputs=graph.outputs
            )
        return loop_section_subgraphs


@dataclass
class CurrentForwardMetadata:
    """
    Full-model forward pass-level metadata for running the current
    forward pass
    """
    input_modalities: list[str]
    output_modalities: list[str]
    phase: str
    is_prefill: bool
    kwargs: dict = field(default_factory=dict)


class Model(ABC):
    def _get_subgraphs_for_phase(
        self, phase_name: str, graph: GraphSection,
        stage_groups: list[dict],
    ):
        stage_groups = [
            g for g in stage_groups if (
                "phases" not in g or phase_name in g["phases"]
            )
        ]
        stage_to_group_idx: dict[str, int] = {}
        for i, group in enumerate(stage_groups):
            stage_to_group_idx.update({
                name: i for name in group["stage_names"]
            })

        return _divide_into_subgraphs(
            graph,
            phase=phase_name,
            stage_to_group_idx=stage_to_group_idx,
            stage_groups=stage_groups
        )
    
    def get_subgraphs(self, config_path: str) -> list[Subgraph]:
        with open(config_path, "r") as f:
            stage_groups = yaml.safe_load(f)["stage_groups"]
        
        # TODO: merge identical subgraphs from different phases
        return sum([
            self._get_subgraphs_for_phase(phase, graph, stage_groups) \
                for phase, graph in self.get_phase_graphs().items()
        ], start=[])
        

    @abstractmethod
    def get_phase_graphs(self) -> dict[str, GraphSection]:
        pass

    @abstractmethod
    def get_initial_forward_metadata(
        self, input_modalities: list[str],
        output_modalities: list[str]
    ) -> CurrentForwardMetadata:
        pass

    @abstractmethod
    def get_forward_pass_inputs(
        self, metadata: CurrentForwardMetadata,
        persist_signals: dict[str, TensorPointerInfo],
        prev_forward_metadata: CurrentForwardMetadata=None,
    ) -> list[GraphPointer]:
        """
        Called by the conductor.

        These are the external inputs that go from the conductor to the
        workers at the beginning of the current forward pass; all other signals
        are handled internally via IPC.
        """
        pass

    @abstractmethod
    def update_for_next_forward(
        self, metadata: CurrentForwardMetadata,
        new_tokens: list[int],
    ) -> CurrentForwardMetadata:
        """
        Called by the conductor at the end of a full model fwd pass.
        """
        # e.g., check for BOI token, check if image was generated and should
        # be added to the input modalities and input tensors, adds new token
        # to the input text, etc...
        # Returns the metadata for the new forward pass
        pass

    @abstractmethod
    def step(
        self, stage_name: str,
        phase: str,
        input_tensors: dict[str, torch.Tensor],
        state, # TODO: figure out state
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Called by the worker to execute a stage
        """
        pass
