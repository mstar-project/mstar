from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import NodeBatch
from mminf.engine.cache_manager import BatchedCacheManager
from mminf.engine.kv_store import PositionInfo
from mminf.utils.sampling import BaseSampler, CudaGraphableSampler

if TYPE_CHECKING:
    from mminf.engine.cuda_graph_config import CudaGraphConfig


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
    preallocated_buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    sampler: BaseSampler | None = None

    @property
    def single_request_info(self):
        """
        IMPORTANT: asserts that there is only one request
        """
        assert len(self.per_request_info) == 1
        return self.per_request_info[self.request_ids[0]]
    
    @property
    def first_request_info(self):
        """
        unlike single_request_info, does not assert that there is only one request
        """
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
    
    def max_batch_size(self, graph_walk: str):
        return None

    def get_stateless_flavor(self) -> str:
        """Flavor key picked up by ``EngineManager`` when this submodule's
        node is declared ``EngineType.STATELESS``. The flavor selects which
        ``StatelessEngineConfig`` factory drives engine construction
        (autocast, force_float32, torch.compile, piecewise runner).

        Default: ``"enc_dec"`` — the most common stateless flavor (encoders,
        vae decoders, etc.). Audio-codec submodules that need no autocast
        and float32 weights override this to return ``"audio_codec"``.
        """
        return "enc_dec"

    def get_autocast_dtype(self) -> torch.dtype | None:
        """Per-submodule autocast dtype override for the engine's forward
        wrap. The engine consults this on each ``execute_batch`` and uses
        the returned dtype instead of its own when non-``None``.

        Default: ``None`` (inherit the engine's autocast dtype). To turn
        autocast off for one specific submodule whose engine otherwise has
        it enabled, wrap the submodule's forward with
        ``torch.amp.autocast(enabled=False)`` — that path is engine-agnostic
        and doesn't need this surface.
        """
        return None

    # Note: do not import CudaGraphConfig; it causes a circular import situation
    def get_cuda_graph_configs(self, device: torch.device, tp_world_size: int = 1) -> list[CudaGraphConfig]:
        return []

    def can_use_cuda_graphs(
        self, batch: NodeBatch,
        model_inputs: list[NodeInputs]
    ) -> bool:
        """Return True if this submodule supports CUDA graphs for ``batch``.

        Default: derives from ``get_cuda_graph_configs`` — if any declared
        config can replay for this batch's graph_walk, CUDA graphs are
        supported. We check ``cfg.replay_graph_walks`` (not just
        ``cfg.capture_graph_walk``) so aliased walks — e.g. Qwen3-Omni's
        ``prefill_audio`` reusing the ``prefill_text`` capture, or
        ``prefill_vision`` reusing its own — are correctly admitted at the
        eligibility gate. The runner's ``_config_for`` already looks up by
        ``replay_graph_walks``; this keeps the gate consistent so aliased
        walks don't silently fall through to the eager path.

        ``replay_graph_walks`` is always a superset of ``{capture_graph_walk}``
        (see ``CudaGraphConfig.__init__``), so this never narrows what the
        previous code accepted — only widens it for configs that explicitly
        declared aliases.

        Subclasses can override to reject on batch shape / metadata (e.g.
        codec submodules that need homogeneous frame counts).
        """
        if not hasattr(self, "_cached_cuda_graph_walks"):
            walks: set[str] = set()
            for cfg in self.get_cuda_graph_configs(device=torch.device("cpu"), tp_world_size=1):
                walks.update(cfg.replay_graph_walks)
            self._cached_cuda_graph_walks = walks
        return batch.graph_walk in self._cached_cuda_graph_walks

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        """
        Metadata-only postprocessing on the submodule outputs.

        Runs on the GPU thread inside ``execute_batch``. **Must not read tensor
        values** — no ``.item()`` / ``.cpu()`` / ``.tolist()`` etc. — because
        any sync here blocks the GPU thread and forfeits the worker's async-
        scheduling overlap. Stop-condition decisions that need token values
        (e.g. EOS) belong in ``check_stop``.

        Typical uses:
          - rebind output names for graph routing (``outputs["text_inputs"] =
            outputs["new_token"]``);
          - drop keys on a per-request basis for static-capture submodules
            (e.g. Qwen3-Omni Thinker dropping ``thinker_states`` for requests
            that don't need audio).

        Modifies ``outputs`` in-place; returns nothing.
        """
        return

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        """
        Return the set of dynamic-loop names that should stop after this step.

        Runs on the worker's slow-postprocess path *after* ``execute_batch``
        returns — never inside ``execute_batch``. **Allowed** to read tensor
        values (``.item()`` / ``.cpu()``) because by this point the GPU
        thread is no longer blocked by it.

        Stops returned here are deferred by one step: they apply to the
        worker's *next* iter's fast postprocess. The current in-flight step
        (already submitted under the assumption that the rid continues)
        will run for that rid and its output discarded — the standard
        1-wasted-step cost for any stop signal.

        Default: no stops.
        """
        return set()

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

        Used by KVCacheEngine to skip redundant KV cache transfers.
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
