from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import NodeBatch
from mminf.engine.cache_manager import BatchedCacheManager
from mminf.engine.kv_store import PositionInfo


@dataclass
class NodeInputs:
    tensor_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    # non-tensor kwargs
    kwargs: dict = field(default_factory=dict)


def _clone_or_none(tensor):
    return tensor.clone() if tensor is not None else None


class StackingMethod(Enum):
    NONE = "none"
    STACK = "stack"
    CAT = "cat"


@dataclass
class ARNodeInputs(NodeInputs):
    """
    Unlike in regular ModelInputs, for LLMInputs we expect either input_ids
    or input_embeds to be set (but typically not both), and we require
    input_seq_len to be set (for cache planning).

    The tensor_inputs and kwargs dicts are still available for additional
    inputs as needed; but the main LLM inputs should be provided in the given
    dedicated fields.
    """
    input_seq_len: int = 0
    input_ids: torch.Tensor | None = None
    input_embeds: torch.Tensor | None = None

    # Tensor for single cache label, dict for multi-label
    custom_pos_ids: torch.Tensor | dict[str, torch.Tensor] | None = None

    @classmethod
    def collate(cls, inputs_list: list["ARNodeInputs"], stacking_method=StackingMethod.NONE):
        out = defaultdict(list)

        for inp in inputs_list:
            # --- required field ---
            out["input_seq_len"].append(inp.input_seq_len)

            # --- usually mutually exclusive main inputs ---
            if inp.input_ids is not None:
                out["input_ids"].append(inp.input_ids)
            if inp.input_embeds is not None:
                out["input_embeds"].append(inp.input_embeds)

            # --- custom_pos_ids ---
            if inp.custom_pos_ids is not None:
                if isinstance(inp.custom_pos_ids, dict):
                    for k, v in inp.custom_pos_ids.items():
                        out.setdefault("custom_pos_ids", {}).setdefault(k, []).append(v)
                else:
                    out["custom_pos_ids"].append(inp.custom_pos_ids)

            # --- tensor_inputs ---
            for k, v in inp.tensor_inputs.items():
                out.setdefault("tensor_inputs", {}).setdefault(k, []).append(v)

            # --- kwargs ---
            for k, v in inp.kwargs.items():
                out.setdefault("kwargs", {}).setdefault(k, []).append(v)

        # --- optional stacking ---
        def maybe_stack(x, stacking_method):
            if stacking_method == StackingMethod.NONE:
                return x
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], torch.Tensor):
                try:
                    if stacking_method == StackingMethod.STACK:
                        return torch.stack(x)
                    else:
                        return torch.cat(x)
                except RuntimeError:
                    return x  # fallback if shapes mismatch
            return x

        for k in ["input_ids", "input_embeds", "custom_pos_ids"]:
            if k in out and isinstance(out[k], list):
                out[k] = maybe_stack(out[k], stacking_method)

        # nested dicts
        for parent in ["tensor_inputs", "custom_pos_ids", "kwargs"]:
            if parent in out and isinstance(out[parent], dict):
                for k, v in out[parent].items():
                    out[k] = maybe_stack(v, stacking_method)

        return dict(out)
    
    def clone(self):
        custom_pos_ids = self.custom_pos_ids
        if isinstance(custom_pos_ids, torch.Tensor):
            custom_pos_ids = _clone_or_none(custom_pos_ids)
        elif isinstance(custom_pos_ids, dict):
            custom_pos_ids = {
                label: _clone_or_none(tensor) for label, tensor in custom_pos_ids.items()
            }

        return ARNodeInputs(
            input_seq_len=self.input_seq_len,
            input_ids=_clone_or_none(self.input_ids),
            input_embeds=_clone_or_none(self.input_embeds),
            custom_pos_ids=custom_pos_ids,
            tensor_inputs={k: _clone_or_none(t) for k, t in self.tensor_inputs.items()},
            kwargs=self.kwargs.copy()
        )


@dataclass
class ModelInputsFromEngine:
    request_ids: list[str]
    per_request_info: dict[str, CurrentForwardPassInfo]
    cache_manager: BatchedCacheManager | None = None

    @property
    def single_request_info(self):
        assert len(self.per_request_info) == 1
        return self.per_request_info[self.request_ids[0]]


class NodeSubmodule(torch.nn.Module):
    """
    TODO
    """

    def get_device(self):
        return next(self.parameters()).device

    @abstractmethod
    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        pass

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]: # input name to tensor
        if len(inputs) > 1:
            raise NotImplementedError(
                f"Batching not implemented for submodule {self.__class__.__name__}"
            )
        return {
            **inputs[0].tensor_inputs,
            **inputs[0].kwargs
        }

    @abstractmethod
    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs # coming from preprocess output
    ) -> NameToTensorList:
        """
        Pure tensor → NameToTensorList computation.
        Compilable + CUDA-graphable.
        """
        pass

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs, # coming from preprocess output
    )  -> dict[str, NameToTensorList]: # request_id to tensors
        """
        TODO comment
        """
        raise NotImplementedError(
            f"Batching not implemented for submodule {self.__class__.__name__}"
            " - override forward_batched to implement, or ensure can_batch returns False"
        )

    def can_batch(
        self,
        batch: NodeBatch,
        model_inputs: list[NodeInputs],
    ):
        return False # batching disabled by default
    
    # Note: do not import CudaGraphConfig; it causes a circular import situation
    def get_cuda_graph_configs(self, device: torch.device) -> list[CudaGraphConfig]:
        """TODO: add cuda graph support for pi05.
        """
        return []

    def can_use_cuda_graphs(
        self, batch: NodeBatch,
        model_inputs: list[NodeInputs]
    ) -> bool:
        """Return True if this submodule supports CUDA graphs for ``batch``.

        Default: derives from ``get_cuda_graph_configs`` — if the submodule
        declared a capture for this batch's graph_walk, CUDA graphs are
        supported. Subclasses can override to reject on batch shape /
        metadata (e.g. codec submodules that need homogeneous frame counts).
        """
        if not hasattr(self, "_cached_cuda_graph_walks"):
            self._cached_cuda_graph_walks = {
                cfg.capture_graph_walk for cfg in self.get_cuda_graph_configs(device=torch.device("cpu"))
            }
        return batch.graph_walk in self._cached_cuda_graph_walks

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        """
        Performs any required postprocessing (after sampling from logits, if applicable)
        on the submodule outputs (e.g., checking for EOS to stop the decode loop, as this
        python-level control flow cannot happen in a cuda graph section).

        E.g., submodules that always emit a static set of keys for capture
        compatibility can override this to drop keys on a per-request basis
        (e.g. the Qwen3-Omni Thinker always emits ``thinker_states`` inside
        the graph, then drops it here for requests that don't need audio).

        This function modifies the `outputs` dict in-place and returns nothing.
        """
        return
    
    def cleanup_request(self, request_id: str):
        """Remove per-request state when a request completes."""
        return


class ARNodeSubmodule(NodeSubmodule):
    @abstractmethod
    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        pos_info: dict[str, PositionInfo] = {},
    ) -> ARNodeInputs:
        pass

    # We are setting preprocess to be abstract here when it was not abstract
    # in the base NodeSubmodule class because the default behavior for preprocess
    # there is not valid in the AR case (batching should typically be enabled, and 
    # preprocess should be implemented). This "making a method abstract in the
    # subclass but not base class" behavior is supported by Python's abc module.
    @abstractmethod
    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]: # input name to tensor
        pass

    def get_needed_cache_labels(
        self,
        graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo]
    ) -> list[str] | None:
        """Return cache labels this node needs, or None to retrieve all.

        Used by AREngine to skip redundant KV cache transfers.
        Override in subclasses that only need a subset of available labels.
        """
        return None
    
    def filter_batched_output(
        self,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[torch.Tensor]]:
        return outputs

    def unpack_packed_outputs(
        self,
        static_output: dict,
        request_ids: list[str],
        real_seq_lens: list[int],
        inputs: list[ARNodeInputs],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, dict[str, list[torch.Tensor]]]:
        """Per-rid slicing for packed sentinels emitted by the captured graph.

        Decode-style submodules emit per-rid entries inside the captured
        forward (one slice per request, fixed shape), so they don't need
        this. Prefill-style submodules pack a (total_tokens, ...) tensor
        whose per-request slice ends depend on real seq_lens — slicing has
        to happen post-replay, outside the captured region. Default
        no-ops; override and key off ``static_output`` sentinel names.
        """
        return {}
