from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import Resource, SlotLease, SubmoduleStep

if TYPE_CHECKING:
    from mstar.engine.cuda_graph_config import CudaGraphConfig, PiecewiseCudaGraphConfig
    from mstar.engine.cuda_graph_runner import PiecewiseCudaGraphRunner
    from mstar.engine.engine import ExecutingBatch


@dataclass
class BatchedModelOutput:
    """A step's outputs, split by how they are addressed.

    ``per_rid_outputs`` is keyed by request id. ``packed_outputs`` holds whole
    batch tensors a captured graph emitted under a ``__name__`` key, which a
    submodule's ``unpack_packed_outputs`` cuts per request.

    ``check_stop_buffers`` is what the stop check reads. It defaults to the
    per-rid outputs, which costs a device-to-host copy per tensor per request;
    a submodule that can hand over one batch tensor instead — row i belonging
    to request i, as everywhere else here — turns that into one copy.

    ``row_outputs`` is the same idea for the outputs themselves. The rows have
    to be copied out of a captured graph's buffers before the next replay
    overwrites them; slicing per request first makes that one clone per
    request, which at a decode batch is one tiny launch per row per step.
    Handing over the batch tensor lets the engine take a single clone and give
    each request a view of it. Only ``Engine._collect_outputs`` reads this —
    it materialises the per-rid views, so nothing downstream ever sees it.

    Both dicts are always dicts: a ``None`` here would have every reader guard
    before touching them, and the readers are on the step's critical path.
    """

    per_rid_outputs: dict[str, NameToTensorList] = field(default_factory=dict)
    packed_outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    # None means "not provided", which is not the same as "provided empty"
    check_stop_buffers: dict[str, torch.Tensor | NameToTensorList] | None = None
    # name -> [bs, ...] tensor, row i belonging to request i
    row_outputs: dict[str, torch.Tensor] | None = None
    # The request each row of ``check_stop_buffers`` belongs to, in the order
    # the forward ran them. Stamped by the engine, which is the only place that
    # order is known for sure: the worker rewrites its own copy of the batch's
    # request list between the forward and the stop check (dropping requests
    # whose loops already stopped, and not necessarily in place), so slicing
    # the rows by position against that list hands one request another's
    # token. None when there are no row-addressed stop buffers.
    row_request_ids: tuple[str, ...] | None = None

    @classmethod
    def coerce(cls, output: BatchedModelOutput | dict[str, Any]) -> BatchedModelOutput:
        if isinstance(output, BatchedModelOutput):
            return output
        per_rid_outputs = {}
        packed_outputs = {}
        for k, v in output.items():
            if k.startswith("__") and k.endswith("__"):
                packed_outputs[k] = v
            else:
                per_rid_outputs[k] = v
        return cls(
            per_rid_outputs=per_rid_outputs,
            packed_outputs=packed_outputs,
        )

    def get(self, key: str, default=None):
        if key in self.per_rid_outputs:
            return self.per_rid_outputs[key]
        return self.packed_outputs.get(key, default)

    def pop(self, request_id: str, default=None):
        """Drop one request's outputs, e.g. when it stopped or failed.

        Only the per-rid side: a packed, row or check-stop batch tensor is
        addressed by row, and the caller takes the request out of the batch
        instead.
        """
        return self.per_rid_outputs.pop(request_id, default)

    def clone_check_stop_buffers(self):
        """A detached copy, since a captured graph's buffers are overwritten by
        the next replay and the stop check reads them after that has started.
        """
        if self.check_stop_buffers is None:
            return None

        def _clone(value):
            if isinstance(value, torch.Tensor):
                return value.clone()
            if isinstance(value, list):
                return [
                    x.clone() if isinstance(x, torch.Tensor) else x for x in value
                ]
            if isinstance(value, dict):
                return {k: _clone(v) for k, v in value.items()}
            # anything else is not ours to copy — pass it through rather than
            # dropping it, which would silently lose a stop signal
            return value

        return {k: _clone(v) for k, v in self.check_stop_buffers.items()}

    def clone_row_outputs(self, num_rows: int) -> dict[str, torch.Tensor]:
        """One clone per row-addressed batch tensor, narrowed to the real rows.

        Same reason ``clone_check_stop_buffers`` copies: a captured graph's
        static output buffer is overwritten by the next replay, and the rows
        are read after that has started. Narrowing first means a padded
        replay's dummy rows are never copied.
        """
        if not self.row_outputs:
            return {}
        cloned: dict[str, torch.Tensor] = {}
        for name, tensor in self.row_outputs.items():
            if not isinstance(tensor, torch.Tensor) or tensor.dim() == 0:
                # not row-addressable; a submodule that puts one here means
                # something else, so pass it through rather than dropping it
                cloned[name] = tensor
                continue
            rows = min(num_rows, tensor.shape[0])
            cloned[name] = tensor[:rows].clone()
        return cloned

    def get_check_stop_input(self):
        if self.check_stop_buffers is not None:
            return self.check_stop_buffers
        return self.per_rid_outputs

    def update(self, other: BatchedModelOutput | dict[str, Any]):
        # coerced, because a caller holding the old dict contract is still a
        # valid producer — that is what `coerce` is for everywhere else here
        other = self.coerce(other)
        self.per_rid_outputs.update(other.per_rid_outputs)
        self.packed_outputs.update(other.packed_outputs)
        # Row-addressed buffers describe one forward. Merging two of them
        # would leave rows from different forwards under one name, so the
        # merged output keeps them only when exactly one side had any; the
        # stop check then falls back to the per-rid outputs, which are always
        # there.
        if other.check_stop_buffers is not None:
            if self.check_stop_buffers is None:
                self.check_stop_buffers = dict(other.check_stop_buffers)
                self.row_request_ids = other.row_request_ids
            else:
                self.check_stop_buffers = None
                self.row_request_ids = None
        if other.row_outputs is not None:
            if self.row_outputs is None:
                self.row_outputs = {}
            self.row_outputs.update(other.row_outputs)


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


def _split_pos_ids(
    pos_ids: "torch.Tensor | dict[str, torch.Tensor]",
    input_seq_len: int, start: int, end: int,
):
    """Cut position ids on their trailing dim, per label where there are labels.

    The trailing dim is the sequence one wherever it matches the walk's token
    count, which covers both the plain ``(seq,)`` layout and the ``(3, seq)``
    one mRoPE uses; anything else is a layout this cannot read.
    """
    if isinstance(pos_ids, dict):
        return {
            label: _split_pos_ids(value, input_seq_len, start, end)
            for label, value in pos_ids.items()
        }
    if pos_ids.shape[-1] != input_seq_len:
        raise NotImplementedError(
            f"custom position ids of shape {tuple(pos_ids.shape)} do not end "
            f"in this walk's {input_seq_len} tokens, so the sequence dim "
            "cannot be told apart; override `split_inputs`"
        )
    return pos_ids[..., start:end]


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


class LazyRequestStates(Mapping):
    """The batch's ``PerRequestState``s, resolved on first read.

    Only a couple of submodules read these, but the engine builds the view for
    every step of every node; materialising the dict there is bs dict inserts
    (and, for a padded step, bs ``PerRequestState`` allocations) per step that
    nothing usually looks at.
    """

    __slots__ = ("_submodule", "_rids", "_members")

    def __init__(self, submodule: "NodeSubmodule", rids: "Sequence[str]"):
        self._submodule = submodule
        self._rids = rids
        self._members: set[str] | None = None

    def __getitem__(self, rid: str) -> "PerRequestState":
        # membership set built on first read, so a step nobody reads pays nothing
        if self._members is None:
            self._members = set(self._rids)
        if rid not in self._members:
            raise KeyError(rid)
        return self._submodule.request_state(rid)

    def __iter__(self):
        return iter(self._rids)

    def __len__(self) -> int:
        return len(self._rids)


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
    # Usually a ``LazyRequestStates`` view rather than a materialised dict.
    per_request_states: "Mapping[str, PerRequestState] | None" = None

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


class NodeSubmodule(torch.nn.Module, ABC):
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

    def split_inputs(
        self,
        graph_walk: str,
        fwd_info: "CurrentForwardPassInfo",
        inputs: NodeInputs,
        start: int,
        end: int,
    ) -> NodeInputs:
        """This walk's inputs restricted to tokens ``[start, end)``.

        ``inputs`` is the whole prompt, prepared once and sliced rather than
        re-derived, so anything read out of a resource while preparing is read
        once, before any part of it runs.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot serve part of a walk's tokens; "
            "it has to override `split_inputs` to say how its inputs are cut"
        )

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
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
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
        padding rows' ids too — declare their segments like any other row.

        ``slot_lease`` is the slot this step will replay on, or None for an
        eager step. A submodule whose declaration differs between the two
        (cosmos3 packs both guidance branches into one plan for the captured
        shape) must key off this, not off its own capture key: the key says
        the batch *could* be captured, the lease says it was.

        ``piecewise_leases`` names the regions of this node that hold a slot
        for this step. Such a region declares, plans and commits its own work,
        so a resource it owns must be left out of this declaration."""
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
    )  -> dict[str, NameToTensorList] | BatchedModelOutput: # request_id to tensors
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
    ) -> dict[str, NameToTensorList]:
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

        A config's ``capture_fn`` takes one ``PiecewiseCallInputs``, whose
        ``engine_inputs.resources`` carries the node's resources — so a
        resource call (sampling included) can live INSIDE the capture, reading
        params straight from buffers whose addresses are stable across replays
        instead of being hoisted out as static inputs. A region that touches a
        resource declares its work in the config's own ``declare_step``; the
        runner admits, plans and commits it per replay.

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
    def split_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: ARNodeInputs,
        start: int,
        end: int,
    ) -> ARNodeInputs:
        """This walk's inputs restricted to tokens ``[start, end)``.

        Only the sequence-shaped fields are cut. ``tensor_inputs``, ``kwargs``
        and ``resource_step_info`` are opaque here, so a submodule carrying any
        of them says for itself how they are cut, or refuses to be cut at all.
        """
        del graph_walk, fwd_info
        for name in ("tensor_inputs", "kwargs", "resource_step_info"):
            if getattr(inputs, name):
                raise NotImplementedError(
                    f"{type(self).__name__} carries {name} into this walk, "
                    "which cannot be cut without knowing what it holds; "
                    "override `split_inputs`"
                )
        cut = replace(
            inputs,
            input_seq_len=end - start,
            input_ids=(
                None if inputs.input_ids is None
                else inputs.input_ids[start:end]
            ),
            input_embeds=(
                None if inputs.input_embeds is None
                else inputs.input_embeds[start:end]
            ),
        )
        if inputs.custom_pos_ids is not None:
            cut.custom_pos_ids = _split_pos_ids(
                inputs.custom_pos_ids, inputs.input_seq_len, start, end,
            )
        return cut

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


