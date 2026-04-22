from abc import abstractmethod
from dataclasses import dataclass, field
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
    input_seq_len: int
    input_ids: torch.Tensor | None = None
    input_embeds: torch.Tensor | None = None
    custom_pos_ids: torch.Tensor | dict[str, torch.Tensor] | None = None # it's a dict if it's per-label


@dataclass
class CudaGraphConfig:
    """Defines what computation a captured graph represents."""
    graph_walk: str  # "decode"
    dummy_capture_inputs: list[ARNodeInputs]

    # whether CFG is active for image generation
    requires_cfg: bool = False

    # cache labels used: ["main"] or ["main", "cfg_img"]
    labels: list[str]  = field(default_factory=lambda: ["main"])

    # whether to run torch.compile on the submodule before cuda graph capture
    compile: bool = True

    # Per-config override for the set of batch sizes to capture. None → use the
    # runner's default (AR engine default: DEFAULT_AR_CAPTURE_BATCH_SIZES;
    # CodecCudaGraphRunner picks its own default). Useful for codec-style
    # submodules where memory cost per size is high, or for AR walks where a
    # small subset is enough.
    capture_batch_sizes: list[int] | None = None


@dataclass
class ModelInputsFromEngine:
    request_ids: list[str]
    per_request_info: dict[str, CurrentForwardPassInfo]
    cache_manager: BatchedCacheManager | None = None


class NodeSubmodule(torch.nn.Module):
    """
    TODO
    """

    @property
    def device(self):
        return next(self.model.parameters()).device

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
                cfg.graph_walk for cfg in self.get_cuda_graph_configs(device=torch.device("cpu"))
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