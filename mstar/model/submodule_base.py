from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.base import Resource
from mstar.engine.resources.step import SubmoduleStep

if TYPE_CHECKING:
    from mstar.engine.v1.cuda_graph_config import CudaGraphConfig, PiecewiseCudaGraphConfig
    from mstar.engine.v1.engine import ExecutingBatch
    from mstar.engine.v1.cuda_graph_runner import PiecewiseCudaGraphRunner


@dataclass
class NodeInputs:
    tensor_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    # non-tensor kwargs
    kwargs: dict = field(default_factory=dict)

    # Any additional information required for declare_step, e.g., like
    # if CFG is required for diffusion/flow submodules
    resource_step_info: Any | None = None

    # Tokens this row contributes to the batch. The engine sums it to pick a
    # capture bucket and to size padding, and a step declares its spans from
    # it. 0 for a submodule whose inputs aren't sequence-shaped.
    input_seq_len: int = 0

    def clone(self):
        """Copy with tensors cloned, so a capture template can be reused.

        Goes through the fields rather than naming them, so a subclass gets
        its own type back without restating this.
        """
        return replace(
            self,
            **{f.name: _clone_value(getattr(self, f.name)) for f in fields(self)},
        )


def _clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_value(item) for item in value)
    return value


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


@dataclass
class PerRequestState:
    """Engine-owned per-request state a submodule persists across forwards.

    Submodules stash whatever a request's later steps need (schedulers,
    conditioning latents, packing metadata) instead of keeping private
    ``dict[request_id, ...]`` attributes. The engine owns the lifecycle: it
    injects the batch's states via ``ModelInputsFromEngine.per_request_states``
    and drops a request's state when the request is removed — no submodule
    cleanup code required.

    ``tensors`` vs ``kwargs`` split by value kind: device tensors go in
    ``tensors``, everything else (numbers, dicts, scheduler objects) in
    ``kwargs``. ``disag_shared_keys`` is reserved for PD disaggregation —
    marked keys would travel with the request (tensors via the tensor
    manager, kwargs with the forward-pass info); no engine implements the
    transfer yet.
    """

    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    disag_shared_keys: set[str] = field(default_factory=set)

    def add(self, key: str, value) -> None:
        if isinstance(value, torch.Tensor):
            self.kwargs.pop(key, None)
            self.tensors[key] = value
        else:
            self.tensors.pop(key, None)
            self.kwargs[key] = value

    def add_all(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self.add(key, value)

    def remove(self, keys) -> None:
        for key in ([keys] if isinstance(keys, str) else keys):
            self.tensors.pop(key, None)
            self.kwargs.pop(key, None)

    def get(self, key: str, default=None):
        if key in self.tensors:
            return self.tensors[key]
        return self.kwargs.get(key, default)

    def __getitem__(self, key: str):
        if key in self.tensors:
            return self.tensors[key]
        return self.kwargs[key]

    def __contains__(self, key: str) -> bool:
        return key in self.tensors or key in self.kwargs


@dataclass
class ModelInputsFromEngine:
    request_ids: list[str]
    per_request_info: dict[str, CurrentForwardPassInfo]
    resources: dict[str, Resource] = field(default_factory=dict)

    # label -> warmed-up PiecewiseCudaGraphRunner for inner-loop capture. Owned
    # by the engine, spread in at execute time (like ``cache_manager`` /
    # ``sampler``). Empty when the submodule opts into no piecewise graphs or
    # capture failed. See ``NodeSubmodule.get_piecewise_cuda_graph_configs``.
    piecewise_runners: dict[str, "PiecewiseCudaGraphRunner"] = field(default_factory=dict)

    # The batch's per-request states, injected by the engine (None on paths
    # that don't carry them, e.g. CUDA-graph capture with synthetic requests).
    per_request_states: dict[str, PerRequestState] | None = None

    # This step's declaration, as ``declare_step`` returned it. A forward
    # that has to agree with its own declaration reads it here rather than
    # re-deriving it (cosmos3 declares its denoise attention against the dense
    # backend or the paged one depending on whether the step got a capture
    # slot, and the forward has to call the one that was planned). None for a
    # submodule that declares no step.
    step: SubmoduleStep | None = None

    # Whether this forward runs under a captured CUDA graph — either the
    # capture itself or a replay. What ``cache_manager.is_captured`` used to
    # carry: a submodule whose ``preprocess`` packs differently for the
    # fixed-shape graph (cosmos3 stacks its denoise inputs on a leading batch
    # dim) reads it here. The forward method itself is chosen by the config's
    # ``capture_forward_method``, so most submodules never need this.
    captured: bool = False

    @property
    @torch.compiler.disable
    def single_request_info(self):
        """
        IMPORTANT: asserts that there is only one request
        """
        assert len(self.per_request_info) == 1
        return self.per_request_info[self.request_ids[0]]

    @property
    @torch.compiler.disable
    def first_request_info(self):
        """
        unlike single_request_info, does not assert that there is only one request
        """
        return self.per_request_info[self.request_ids[0]]


class NodeSubmodule(torch.nn.Module):
    """Base class for a model's compute units: defines the prepare_inputs →
    preprocess → forward(_batched) contract the engines drive."""

    # Set True on a submodule whose forward does not benefit from (or is broken
    # by) torch.compile — e.g. a data-dependent denoise loop, or a one-shot
    # forward where the trace cost dwarfs the win. The KV-cache / stateless
    # engines skip compiling such submodules (CUDA-graph capture is unaffected).
    disable_torch_compile: bool = False

    # Set True on a submodule that must run in its own parameter dtype — e.g. a
    # numerically sensitive fp32 vocoder. The engine then neither casts its
    # params to the autocast dtype nor wraps its forward (or capture) in
    # autocast, and explicitly disables any ambient one.
    disable_autocast: bool = False

    def __init__(self):
        super().__init__()
        # Per-request state store. prepare_inputs-time code (no engine inputs
        # in scope) reaches it via ``request_state``; preprocess/forward read
        # the engine-injected ``ModelInputsFromEngine.per_request_states`` view
        # of the same objects. The engine removes a request's entry via
        # ``cleanup_request`` when the request is removed.
        self.request_states: dict[str, PerRequestState] = {}
        # Engine-built resources for this submodule's node (KV cache pool,
        # embedder, scratch caches), bound once at load. Empty until then
        # and on engines that build none.
        self.node_resources: dict[str, Any] = {}

    def bind_node_resources(self, resources: dict[str, Any]) -> None:
        """Receive the engine-built resources for this submodule's node, and
        pass them down to every layer that calls one.

        A layer body (attention, cross-attention) calls the resources
        directly — ``attn.run``, ``kv.write_kv``, ``pos.apply_qk`` — so it
        needs its own references. It resolves them here, once at load, by the
        labels the model declared in ``get_node_resources``; a layer that
        names a label this node doesn't have fails at bind rather than in the
        middle of a forward.
        """
        self.node_resources = resources
        for module in self.modules():
            bind = getattr(module, "bind_resources", None)
            if bind is not None and module is not self:
                bind(resources)

    def cg_key_info(
        self, graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> Any:
        """Which of this walk's capture buckets a batch belongs to.

        A walk can be captured more than once when the batch's *shape* is not
        the whole story — bagel captures decode twice, guidance on and off,
        because the two declare different segments over the same token count.
        The engine leases a slot before the step is declared, so it cannot read
        the answer off the step; it asks here instead.

        Must equal the ``additional_key_info`` on the config that captured the
        bucket, and the ``cg_key_info`` this batch's ``declare_step`` puts on
        its step. Disagreeing is not an error anywhere — it just misses the
        capture and runs eager — so derive both from one place.

        None (the default) means the walk has a single capture.
        """
        del graph_walk, per_request_info
        return None

    def request_state(self, request_id: str) -> PerRequestState:
        """The request's state, created on first access."""
        state = self.request_states.get(request_id)
        if state is None:
            state = self.request_states[request_id] = PerRequestState()
        return state

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

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
    ) -> SubmoduleStep | None:
        """Declare this batch's step for the runner to drive: which cache
        streams it touches, what spans they grow by, which plans back it,
        which streams fork, and what commits when it lands. The runner
        drives the declaration before ``preprocess`` and commits it after
        the forward, so a declaring submodule keeps no plan or advance
        calls of its own. None means the submodule still plans and
        advances through the facade itself.

        ``request_ids`` pairs positionally with ``inputs``. Under a captured
        graph the batch is padded to the bucket's shape, so it carries the
        padding rows' ids too — declare their segments like any other row."""
        return None

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
        """Batched form of ``forward``: maps a multi-request batch to
        per-request outputs. Override when ``can_batch`` returns True."""
        raise NotImplementedError(
            f"Batching not implemented for submodule {self.__class__.__name__}"
            " - override forward_batched to implement, or ensure can_batch returns False"
        )

    def can_batch(
        self,
        batch: ExecutingBatch,
        model_inputs: list[NodeInputs],
    ):
        return False # batching disabled by default

    def filter_batched_output(
        self,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[torch.Tensor]]:
        """Drop keys a real request shouldn't receive. A captured forward emits
        a fixed key set for graph compat, so the filtering happens here."""
        return outputs

    def unpack_packed_outputs(
        self,
        static_output: dict,
        request_ids: list[str],
        real_seq_lens: list[int],
        inputs: list[NodeInputs],
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

    def get_piecewise_cuda_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1,
    ) -> dict[str, PiecewiseCudaGraphConfig]:
        """Return the piecewise CUDA graph configs this submodule opts into.

        ``autocast_dtype`` is the engine's autocast dtype — passed so a config's
        ``make_static_inputs`` can allocate the hidden-state buffer in the dtype
        the captured region runs under (avoids a copy-time upcast at replay).

        A piecewise CUDA graph captures ONE inner callable of this submodule's
        forward (e.g. a transformer block loop) as a CUDA graph while the
        surrounding compute stays eager. The engine builds one
        ``PiecewiseCudaGraphRunner`` per returned label and threads the runners
        into ``ModelInputsFromEngine.piecewise_runners`` so the submodule's
        forward can look them up by label:

            runner = engine_inputs.piecewise_runners.get("block_loop")
            if runner is not None and runner.can_run(bs):
                out = runner.run(static_inputs={...}, request_ids=..., seq_lens=...)

        A config that sets ``uses_sampler=True`` has the engine's per-node
        sampler buffers wired into its runner, which passes a ``sampler`` into
        the config's ``capture_fn`` so sampling can live INSIDE the capture
        (params are read straight from buffers whose addresses are stable across
        replays) instead of being hoisted out as static inputs.

        Default: no piecewise graphs. Override to return
        ``{label: PiecewiseCudaGraphConfig}``; multiple labels capture multiple
        independent graphs (i.e., one per outer function to be graphed).
        """
        return {}

    def can_use_cuda_graphs(
        self, batch: ExecutingBatch,
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
        inputs: NodeInputs | None = None,
        **kwargs
    ):
        """
        Per-request postprocessing on the submodule outputs.

        Runs on the GPU thread inside ``execute_batch``, after the forward
        (eager, batched, or CUDA-graph replay). ``inputs`` is the request's
        ``prepare_inputs`` result for this step, so a submodule can finish a
        step the captured graph could not hold (e.g. combine guidance branches
        and run a Python multistep scheduler against the step's input latents).

        Keep it metadata-only where possible. **Avoid reading tensor values**
        — ``.item()`` / ``.cpu()`` sync here block the GPU thread and forfeit
        the worker's async-scheduling overlap; stop-condition decisions that
        need token values (e.g. EOS) belong in ``check_stop``. A captured-path
        tail that must read scheduler state is the sanctioned exception — it
        costs the same sync wherever it runs.

        Typical uses:
          - rebind output names for graph routing (``outputs["text_inputs"] =
            outputs["new_token"]``);
          - drop keys on a per-request basis for static-capture submodules
            (e.g. Qwen3-Omni Thinker dropping ``thinker_states`` for requests
            that don't need audio);
          - finish a captured step from ``inputs`` (Cosmos3 denoise tail).

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
        """Remove per-request state when a request completes. The engines call
        this on request removal; overrides with extra internal state should
        call super()."""
        self.request_states.pop(request_id, None)


class ARNodeSubmodule(NodeSubmodule):
    @abstractmethod
    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
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

