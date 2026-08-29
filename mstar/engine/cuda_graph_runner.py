import logging
import os
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import JointGroups
from mstar.engine.cuda_graph_config import (
    CudaGraphConfig,
    CudaGraphConfigType,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewiseConfigType,
    PiecewiseCudaGraphConfig,
)
from mstar.engine.resources import BucketKey, CGSlotSpec, Resource, SlotLease, StepContext, StepRunner
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)


DEFAULT_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]


def autocast_scope(dtype: torch.dtype | None):
    """A forward's autocast scope; ``None`` (``disable_autocast``) runs the
    submodule in its own dtype and shuts out any ambient autocast."""
    return torch.amp.autocast(
        "cuda", enabled=dtype is not None, dtype=dtype or torch.bfloat16
    )


def dummy_metadata(
    rids: list[str], graph_walk: str,
) -> dict[str, CurrentForwardPassInfo]:
    """Stand-in request info for padding rows, which have no real request."""
    return {
        rid: CurrentForwardPassInfo(
            request_id=rid,
            graph_walk=graph_walk,
            requires_cfg=False,
            fwd_index=0,
            random_seed=0,
            max_tokens=1,
        ) for rid in rids
    }


class DummyRowPool:
    """The padding rows captured replays pad onto.

    Ingested once and kept for the runner's lifetime: a re-ingest would hand
    the resources a fresh stream and orphan the pages the last capture left
    resident, and keeping them resident is what lets a step's padding tail
    allocate nothing.
    """

    def __init__(
        self, prefix: str, step_runner: StepRunner,
        resources: Mapping[str, Resource],
    ):
        self._prefix = prefix
        self._step_runner = step_runner
        self._resources = resources
        self._held: dict[str, list[str]] = {}

    def names(self, key: str, bs: int) -> list[str]:
        return [f"__cg_{self._prefix}_{key}_{i}__" for i in range(bs)]

    def ensure(self, key: str, bs: int) -> list[str]:
        """``bs`` rows for ``key``, ingesting any this pool hasn't opened yet."""
        held = self._held.setdefault(key, [])
        names = self.names(key, bs)
        for rid in names[len(held):]:
            self._step_runner.ingest_request(rid)
            held.append(rid)
        return names

    def reset(self, rids: list[str], free: bool=False) -> None:
        for rid in rids:
            for resource in self._resources.values():
                resource.reset_request(rid, free=free)


@dataclass
class CudaGraphSlot:
    """One captured graph + the buffers its replay reads and writes.

    Double-buffer: a bucket holds one of these per slot. Replay alternates
    between slots so plan(N+1) on the inactive slot's resources can run
    concurrently with replay(N) on the active slot — so a node with nothing
    to pre-plan keeps a single slot.
    """
    graph: torch.cuda.CUDAGraph
    # preprocess output as captured; tensor entries are the static buffers
    static_inputs: dict[str, Any]
    static_input_keys: tuple[str, ...]
    static_outputs: dict
    # padding rows address these; their streams stay resident between steps
    dummy_rids: list[str]
    dummy_metadata: dict[str, CurrentForwardPassInfo]
    config_idx: int


@dataclass
class CudaGraphBucket:
    """The captured slots for one (walk, cg_key_info, bs, num_tokens)."""
    config: CudaGraphConfig
    config_idx: int
    slots: list[CudaGraphSlot] = field(default_factory=list)


class CudaGraphRunner:
    # Double-buffer: capture two graphs per config key, when the node has a
    # resource that pre-plans. Replay alternates so plan(N+1) on the inactive
    # slot can run concurrent with replay(N) on the active slot. Override via
    # MSTAR_NUM_SLOTS=1 to disable double-buffer
    NUM_SLOTS = int(os.environ.get("MSTAR_NUM_SLOTS", "2"))
    CAPTURE_BATCH_SIZES = DEFAULT_CAPTURE_BATCH_SIZES
    NUM_WARMUP = 2

    def __init__(
        self,
        submodule_name: str,
        submodule: NodeSubmodule,
        resources: Mapping[str, Resource],
        step_runner: StepRunner,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
        joint_comm_group: JointGroups,
        enable_nvtx: bool=False,
    ):
        self._submodule_name = submodule_name
        self._submodule = submodule
        self._comm_group = joint_comm_group

        self._capture_configs: list[CudaGraphConfig] = submodule.get_cuda_graph_configs(
            device, joint_comm_group.world_size
        )
        self._resources = resources
        self._step_runner = step_runner
        self._device = device
        self._autocast_dtype = autocast_dtype
        self._enable_nvtx = enable_nvtx

        # A second slot only buys something when a resource can plan a step
        # ahead: the point is for that plan to write buffers the in-flight
        # replay isn't reading. With nothing to pre-plan, the slots would hold
        # identical graphs and double the capture time and memory.
        self._num_slots = self.NUM_SLOTS if any(
            resource.supports_preplan for resource in resources.values()
        ) else 1

        self._buckets: dict[BucketKey, CudaGraphBucket] = {}

        # Rotated globally, not per bucket: slot-keyed buffers are shared across
        # buckets, so per-bucket counters let consecutive steps collide on a slot.
        self._next_slot = 0

        self._memory_pool = None
        # set by prepare_for_capture; capture reuses it rather than re-deriving
        self._prepared_slot_specs: list[CGSlotSpec] | None = None

        # (config_idx, tensor_key) → max-bucket static buffer. Lazily populated
        # by _intern_static_buffer on the first capture. Smaller-bucket captures
        # slice the leading dim of the same buffer
        self._shared_static_buffers: dict[tuple[int, str], torch.Tensor] = {}

        # (config_idx, tensor_key) → the dim that carries the (bucket-varying)
        # seq length in that tensor's original layout. _intern_static_buffer
        # brings this dim to the front for storage (so smaller buckets reslice
        # along dim 0) and inverts the move on return. Written once, by the
        # first (largest) capture, and read by every later one — see there.
        self._static_buffer_seq_dims: dict[tuple[int, str], int] = {}

        # padding rows, keyed "<config_idx>_slot<slot>". Sized by the largest
        # bucket (captures run largest-first); smaller buckets take a prefix.
        self._dummy_rows = DummyRowPool(
            prefix=submodule_name, step_runner=step_runner, resources=resources,
        )

        # Sum of bytes that WOULD have been allocated by per-capture clones
        # (one full tensor per call). For logging purposes.
        self._capture_clone_bytes_naive = 0

        # config_idx -> its compiled forward, shared by every bucket of that
        # config so shapes captured twice (once per slot) compile once
        self._compiled_forwards: dict[int, Any] = {}

        # Plan-overlap stream. Lazily created the first time pre_plan
        # is called from Worker.plan_executor.
        self._plan_stream: torch.cuda.Stream | None = None

        # Per-bucket template rows for declare-only calls; see
        # `declare_inputs_for`.
        self._declare_inputs: dict[BucketKey, list[NodeInputs]] = {}

        self.max_bs = max(
            (max(self._batch_sizes(config)) for config in self._capture_configs),
            default=1,
        )

    @property
    def any_graphs(self):
        return bool(self._buckets)

    def _batch_sizes(self, config: CudaGraphConfig) -> list[int]:
        return config.capture_batch_sizes or self.CAPTURE_BATCH_SIZES

    # ── Capture ─────────────────────────────────────────────────────────

    def _get_slot_specs(self) -> list[CGSlotSpec]:
        specs = []
        for config_idx, config in enumerate(self._capture_configs):
            for bs in self._batch_sizes(config):
                for num_tokens in config.get_total_tokens(bs):
                    bucket = BucketKey(
                        graph_walk=config.capture_graph_walk,
                        cg_key_info=config.additional_key_info,
                        bs=bs,
                        num_tokens=num_tokens,
                    )
                    for slot in range(self._num_slots):
                        specs.append(CGSlotSpec(
                            bucket=bucket,
                            slot=slot,
                            config=config,
                            config_idx=config_idx,
                        ))
        return specs

    def _get_addtl_slot_specs(self, spec: CGSlotSpec) -> list[CGSlotSpec]:
        """The same capture, keyed under the config's other replay walks."""
        return [
            replace(spec, bucket=replace(spec.bucket, graph_walk=walk))
            for walk in spec.config.replay_graph_walks
            if walk != spec.config.capture_graph_walk
        ]

    def prepare_for_capture(self) -> list[CGSlotSpec]:
        """Claim the static buffers this runner's captures will read.

        Split from the capture so every runner claims first: nodes share
        resources, and a build driven by a later node would move buffers an
        earlier node's graphs already baked in.
        """
        if self._prepared_slot_specs is None:
            slot_specs = self._get_slot_specs()
            # reverse sort based on batch size, then total tokens (largest first)
            slot_specs.sort(key=lambda s: (s.bs, s.num_tokens), reverse=True)
            max_seq_len = max((
                spec.bucket.num_tokens for spec in slot_specs
            ), default=1)
            self._step_runner.build_cuda_graph_buffers(
                slot_specs, max_bs=self.max_bs, max_seq_len=max_seq_len,
                node_name=self._submodule_name,
            )
            self._prepared_slot_specs = slot_specs
        return self._prepared_slot_specs

    def warmup_and_capture(self):
        """Capture graphs for all configs and batch sizes."""
        if self._device is None or not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping graph capture for %s",
                            self._submodule_name)
            return

        self._memory_pool = torch.cuda.graphs.graph_pool_handle()
        mem_before = torch.cuda.memory_allocated(self._device)

        slot_specs = self.prepare_for_capture()

        for spec in slot_specs:
            self._comm_group.tp_group.barrier()
            self._comm_group.sp_group.barrier()

            try:
                slot = self._capture_one(spec)
            except Exception:
                logger.warning(
                    "Failed to capture CUDA graph for %s: %s",
                    self._submodule_name, spec, exc_info=True
                )
                continue

            for key_spec in [spec, *self._get_addtl_slot_specs(spec)]:
                self._register_slot(key_spec, slot)

            # pre-seed preplan inputs
            if self._num_slots > 1:
                # can preplan
                self.declare_inputs_for(SlotLease(
                    slot=slot,
                    bucket=spec.bucket
                ))
            logger.info(
                "Captured CUDA graph for %s: %s", self._submodule_name, spec
            )

        mem_after = torch.cuda.memory_allocated(self._device)
        self._log_memory(mem_before, mem_after)

    def _register_slot(self, spec: CGSlotSpec, slot: CudaGraphSlot) -> None:
        bucket = self._buckets.get(spec.bucket)
        if bucket is None:
            bucket = self._buckets[spec.bucket] = CudaGraphBucket(
                config=spec.config, config_idx=spec.config_idx,
            )
        # slots are captured in index order within a bucket, so append keeps
        # list position == slot index; a failed capture drops the whole bucket
        # slot rather than leaving a hole
        bucket.slots.append(slot)

    def _log_memory(self, mem_before: int, mem_after: int):
        shared_bytes = sum(
            t.numel() * t.element_size() for t in self._shared_static_buffers.values()
        )
        # Report both: the deterministic synthetic counter (clean before/after
        # for the buffer-reuse change in isolation) and the actual GPU delta
        # (covers FlashInfer wrappers + dummy KV state too, but is noisier).
        logger.info(
            "CudaGraphRunner[%s]: warmup_and_capture done. "
            "shared_static_buffers: %d entries, %.2f MB resident "
            "(would have been %.2f MB with per-capture clones — saved %.2f MB). "
            "Total cuda alloc delta during warmup: %.2f MB.",
            self._submodule_name,
            len(self._shared_static_buffers),
            shared_bytes / (1024 ** 2),
            self._capture_clone_bytes_naive / (1024 ** 2),
            (self._capture_clone_bytes_naive - shared_bytes) / (1024 ** 2),
            (mem_after - mem_before) / (1024 ** 2),
        )

    def _capture_one(self, spec: CGSlotSpec) -> CudaGraphSlot:
        walk = spec.bucket.graph_walk
        config = spec.config
        dummy_rids = self._dummy_rows.ensure(
            f"{spec.config_idx}_slot{spec.slot}", spec.bs
        )
        dummy_inputs = config.get_node_inputs(spec.bs, spec.num_tokens)
        engine_inputs = self._dummy_engine_inputs(dummy_rids, walk)

        step = self._submodule.declare_step(
            graph_walk=walk, request_ids=dummy_rids, inputs=dummy_inputs,
        )
        if step is not None:
            step = replace(step, _ctx=StepContext(
                request_ids=tuple(dummy_rids),
                graph_walk=walk,
                slot=spec.slot,
                capture=True,
                slot_lease=SlotLease(slot=spec.slot, bucket=spec.bucket),
            ))

        engine_inputs.step = step

        def prepare() -> dict[str, Any]:
            if step is not None:
                outcome = self._step_runner.admit(step)
                if not outcome.ok:
                    raise RuntimeError(
                        f"capture admit failed for {spec.bucket}: {outcome.reason}"
                    )
                self._step_runner.plan(step)
            return self._submodule.preprocess(
                graph_walk=walk,
                engine_inputs=engine_inputs,
                inputs=dummy_inputs,
            )

        try:
            static_inputs = prepare()
            # Intern before capture so every bucket of this config records the
            # same GPU addresses; replay writes real values into them.
            for key, value in list(static_inputs.items()):
                if isinstance(value, torch.Tensor):
                    static_inputs[key] = self._intern_static_buffer(
                        spec.config_idx, key, value, seq_len=spec.num_tokens,
                    )
            static_input_keys = tuple(
                key for key, value in static_inputs.items()
                if isinstance(value, torch.Tensor)
            )

            forward = self._forward_for(spec)

            def run_forward():
                return forward(
                    graph_walk=walk,
                    engine_inputs=engine_inputs,
                    **static_inputs,
                )

            torch.cuda.set_device(self._device)
            torch.cuda.synchronize()
            for _ in range(self.NUM_WARMUP):
                with autocast_scope(self._autocast_dtype):
                    run_forward()
                # back to a clean stream state so the re-plan below (and the
                # capture after it) sees the same shapes the first prepare did
                self._dummy_rows.reset(dummy_rids)
                prepare()
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with autocast_scope(self._autocast_dtype):
                with torch.cuda.graph(graph, pool=self._memory_pool):
                    output = run_forward()
            torch.cuda.synchronize()

            return self._build_slot_from_capture(
                output=output,
                graph=graph,
                static_inputs=static_inputs,
                static_input_keys=static_input_keys,
                dummy_rids=dummy_rids,
                dummy_metadata=engine_inputs.per_request_info,
                config_idx=spec.config_idx,
            )
        finally:
            # pages stay with the dummy streams: replay's padding rows address
            # the same ids, so their plan finds the storage already resident
            self._dummy_rows.reset(dummy_rids)

    def _forward_for(self, spec: CGSlotSpec):
        """The callable this bucket captures, compiled once per config.

        One wrapper per config rather than per capture: ``dynamic=False`` keys
        compiled code by shape, so every bucket of a config shares a cache and
        the second slot's identical shapes are free. A wrapper per capture
        would recompile — and re-autotune — all of them.
        """
        forward = getattr(self._submodule, spec.config.capture_forward_method)
        if not spec.config.compile:
            return forward
        if spec.config_idx not in self._compiled_forwards:
            self._compiled_forwards[spec.config_idx] = torch.compile(
                forward,
                mode="max-autotune-no-cudagraphs",
                fullgraph=False,
                dynamic=False,
            )
        return self._compiled_forwards[spec.config_idx]

    def _dummy_engine_inputs(
        self, dummy_rids: list[str], graph_walk: str,
    ) -> ModelInputsFromEngine:
        return ModelInputsFromEngine(
            request_ids=list(dummy_rids),
            per_request_info=dummy_metadata(dummy_rids, graph_walk),
            resources=dict(self._resources),
            captured=True,
        )

    @staticmethod
    def _seq_dim(value: torch.Tensor, seq_len: int) -> int:
        """Index of the first dim whose size matches ``seq_len``, else 0.

        Used to bring the (bucket-varying) seq dim to the front for shared-buffer
        interning: most inputs are seq-leading (returns 0), but mrope-style ids
        carry seq in a later dim (e.g. ``[3, seq]`` → 1)."""
        for dim, size in enumerate(value.shape):
            if size == seq_len:
                return dim
        return 0

    def _intern_static_buffer(
        self, config_idx: int, key: str, value: torch.Tensor,
        seq_len: int | None = None,
    ) -> torch.Tensor:
        """Return a slice view into the shared buffer for (config_idx, key).

        The buffer is allocated at the first (largest, since captures are sorted
        largest-first) bucket's shape; smaller buckets reslice its leading dim.
        If ``seq_len`` is given, the seq dim is moved to the front for storage and
        back on return, so the captured forward sees the original layout.
        """
        buf_key = (config_idx, key)
        if seq_len is None:
            seq_dim = 0
        elif buf_key in self._static_buffer_seq_dims:
            # settled by the first (largest) capture. `_seq_dim` matches on
            # size, so a smaller bucket's seq_len can collide with another axis
            # — and the buffer is shared, so its layout cannot vary anyway
            seq_dim = self._static_buffer_seq_dims[buf_key]
        else:
            seq_dim = self._seq_dim(value, seq_len)
            self._static_buffer_seq_dims[buf_key] = seq_dim
        stored = value.movedim(seq_dim, 0) if seq_dim != 0 else value
        shared = self._shared_static_buffers.get(buf_key)
        if shared is None:
            shared = torch.empty(stored.shape, dtype=stored.dtype, device=stored.device)
            self._shared_static_buffers[buf_key] = shared
        self._capture_clone_bytes_naive += stored.numel() * stored.element_size()
        leading = stored.shape[0]
        if leading > shared.shape[0] or stored.shape[1:] != shared.shape[1:]:
            raise RuntimeError(
                f"_intern_static_buffer: key={key!r} (config_idx={config_idx}) needs "
                f"{tuple(stored.shape)}, shared buffer is {tuple(shared.shape)}; "
                "captures must be largest-first with matching trailing dims"
            )
        sliced = shared[:leading]
        sliced.copy_(stored)
        return sliced.movedim(0, seq_dim) if seq_dim != 0 else sliced

    def _build_slot_from_capture(
        self, output, graph, static_inputs, static_input_keys,
        dummy_rids, dummy_metadata, config_idx,
    ) -> CudaGraphSlot:
        """Wrap one slot's capture artifacts into a CudaGraphSlot."""
        return CudaGraphSlot(
            graph=graph,
            static_inputs=static_inputs,
            static_input_keys=static_input_keys,
            static_outputs=output,
            dummy_rids=list(dummy_rids),
            dummy_metadata=dict(dummy_metadata),
            config_idx=config_idx,
        )

    # ── Replay ──────────────────────────────────────────────────────────

    def select_bucket(
        self, graph_walk: str, bs: int, num_tokens: int,
        cg_key_info: Any | None = None,
    ) -> BucketKey | None:
        """Tightest captured bucket that fits this batch, or None for eager.

        A walk may have several captures (e.g. one per image resolution, each a
        fixed shape with its own token count), so every matching bucket is
        considered rather than the first config declared.
        """
        best: BucketKey | None = None
        for key, bucket in self._buckets.items():
            if key.graph_walk != graph_walk or key.cg_key_info != cg_key_info:
                continue
            if key.bs < bs or key.num_tokens < num_tokens or not bucket.slots:
                continue
            if best is None or (key.num_tokens, key.bs) < (best.num_tokens, best.bs):
                best = key
        return best

    def select_batched_bucket(
        self, graph_walk: str, bs: int, cg_key_info: Any | None = None,
    ) -> BucketKey | None:
        """A bucket for a batch whose inputs aren't built yet.

        Only a batched capture can answer this: its token count is a property
        of the config (``bs`` rows of a fixed per-request length), so the
        bucket follows from the batch size alone. A packed capture's token
        count is a property of the *requests*, so it has to wait for
        ``prepare_inputs`` and go through ``select_bucket``.

        This is what lets pre-plan run before inputs exist. Lifting the
        restriction means making ``prepare_inputs`` safe to run ahead of the
        forward — see ``Engine.pre_plan_for_batch``.
        """
        best: BucketKey | None = None
        for key, bucket in self._buckets.items():
            if key.graph_walk != graph_walk or key.cg_key_info != cg_key_info:
                continue
            if key.bs < bs or not bucket.slots:
                continue
            if bucket.config.get_config_type() != CudaGraphConfigType.BASIC_BATCHED:
                continue
            # the config's own token count for this padded batch; a bucket
            # whose num_tokens says otherwise belongs to a different capture
            if key.num_tokens not in bucket.config.get_total_tokens(key.bs):
                continue
            if best is None or (key.bs, key.num_tokens) < (best.bs, best.num_tokens):
                best = key
        return best

    def max_batch_size_for(self, graph_walk: str) -> int | None:
        """Largest batch this walk was captured for, or None for no cap.

        A config that opts out of capping captures only an acceleration subset
        of batch sizes — larger batches run eager rather than being split — so
        it does not constrain the batch. With every config for the walk opting
        out, nothing here caps it.
        """
        capped = [
            max(self._batch_sizes(config)) for config in self._capture_configs
            if graph_walk in config.replay_graph_walks
            and config.caps_eager_batch_size
        ]
        return max(capped) if capped else None

    def can_run(
        self, graph_walk: str, bs: int, num_tokens: int,
        cg_key_info: Any | None = None,
    ) -> bool:
        return self.select_bucket(
            graph_walk, bs, num_tokens, cg_key_info
        ) is not None

    def lease_slot(
        self, graph_walk: str, bs: int, num_tokens: int | None = None,
        cg_key_info: Any | None = None,
        slot: int | None = None,
    ) -> SlotLease | None:
        """Pick the bucket and double-buffer slot an upcoming step replays on.

        ``num_tokens=None`` means the batch's inputs aren't built yet, so the
        bucket comes from the batched-capture search instead.

        ``slot=None`` advances the bucket's counter so the next lease lands on
        the other slot; a caller that already reserved one (pre-plan) passes it
        back so both submissions target the same slot.
        """
        key = (
            self.select_batched_bucket(graph_walk, bs, cg_key_info)
            if num_tokens is None
            else self.select_bucket(graph_walk, bs, num_tokens, cg_key_info)
        )
        if key is None:
            return None
        bucket = self._buckets[key]
        if slot is None:
            slot = self._next_slot
            self._next_slot = (self._next_slot + 1) % self._num_slots
        # Hand back the bucket the capture ran under, not the alias this walk
        # found it by. Resources key their per-slot state (attention wrappers,
        # position buffers) on `lease.bucket`, so a lease naming an aliased
        # walk would build them fresh at replay instead of reusing the ones the
        # graph recorded addresses for — see `_get_addtl_slot_specs`.
        return SlotLease(
            slot=slot % len(bucket.slots),
            bucket=replace(key, graph_walk=bucket.config.capture_graph_walk),
        )

    def slot_for(self, lease: SlotLease) -> CudaGraphSlot:
        return self._buckets[lease.bucket].slots[lease.slot]

    def config_for(self, lease: SlotLease) -> CudaGraphConfig:
        return self._buckets[lease.bucket].config

    def declare_inputs_for(self, lease: SlotLease) -> list[NodeInputs]:
        """Template rows for a declare-only call, cached per bucket.

        A bucket's shape is fixed, so these rows are constant; building them
        fresh cost ~130us of clones per pre-plan at bs=16, plus that many D2D
        copies. Only for callers that hand the rows to ``declare_step`` and
        nothing else — the list is shared, so a caller that mutates it or
        passes it to ``preprocess`` corrupts every later step on this bucket.
        """
        cached = self._declare_inputs.get(lease.bucket)
        if cached is None:
            cached = self._declare_inputs[lease.bucket] = (
                self._buckets[lease.bucket].config.get_node_inputs(
                    lease.bucket.bs, lease.bucket.num_tokens
                )
            )
        return cached

    def pad_inputs(
        self, lease: SlotLease, inputs: list[NodeInputs],
    ) -> list[NodeInputs]:
        """Real inputs plus the rows that bring the batch to capture shape.
        """
        num_pad = lease.bucket.bs - len(inputs)
        if num_pad <= 0:
            return list(inputs)
        budget = lease.bucket.num_tokens - sum(inp.input_seq_len for inp in inputs)
        padding = self._buckets[lease.bucket].config.get_node_inputs(
            num_pad, max(budget, 0)
        )
        return [*inputs, *padding]

    def step_ids(self, lease: SlotLease, request_ids: list[str]) -> list[str]:
        """The padded addressing for one step: real ids first, then the slot's
        padding ids. Plans, commits, and advances address real request state by
        its own id; only the padding rows run against the slot's own state."""
        dummy_rids = self.slot_for(lease).dummy_rids
        return [*request_ids, *dummy_rids[len(request_ids):lease.bucket.bs]]

    def step_metadata(
        self, lease: SlotLease, request_ids: list[str],
        per_request_info: Mapping[str, CurrentForwardPassInfo],
    ) -> dict[str, CurrentForwardPassInfo]:
        slot = self.slot_for(lease)
        meta = {rid: per_request_info[rid] for rid in request_ids}
        for rid in slot.dummy_rids[len(request_ids):lease.bucket.bs]:
            meta[rid] = slot.dummy_metadata[rid]
        return meta

    def _stage(self, lease: SlotLease, preprocessed: dict[str, Any]) -> None:
        """Copy this step's preprocess output into the captured static buffers.

        Each key is sliced along the dim ``_intern_static_buffer`` recorded as
        its seq dim, so mrope-style ``[.., seq, ..]`` inputs land on the right
        axis. Trailing slots beyond the real length keep capture-time contents;
        the plan's indptrs keep attention off them.
        """
        slot = self.slot_for(lease)
        for key in slot.static_input_keys:
            value = preprocessed.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            static_buf = slot.static_inputs[key]
            seq_dim = self._static_buffer_seq_dims.get((slot.config_idx, key), 0)
            static_buf.narrow(seq_dim, 0, value.shape[seq_dim]).copy_(value)

    def _replay(self, lease: SlotLease) -> dict:
        slot = self.slot_for(lease)
        slot.graph.replay()
        return slot.static_outputs

    def run_forward(
        self, lease: SlotLease, preprocessed: dict[str, Any],
        plan_done_event: torch.cuda.Event | None = None,
        launch_started_event: "threading.Event | None" = None,
    ) -> dict:
        """Stage this step's inputs into the slot and replay it.

        ``plan_done_event`` is the pre-plan's event: its plan wrote this slot's
        buffers on another stream, so the replay has to wait for those writes
        before it reads them.

        ``launch_started_event`` releases the submitting thread. It is set HERE
        rather than before the forward: staging copies each preprocessed tensor
        into its static buffer and holds the GIL while doing so, so releasing
        earlier hands the main thread a GIL the GPU thread still needs. Only
        `graph.replay()` below drops it in C++.
        """
        self._stage(lease, preprocessed)
        if plan_done_event is not None:
            torch.cuda.default_stream(self._device).wait_event(plan_done_event)
        if launch_started_event is not None:
            launch_started_event.set()
        return self._replay(lease)

    def release(self, lease: SlotLease, real_bs: int) -> None:
        """Return the padding rows to their at-rest state after a step.

        Their pages stay resident (``free=False``), so the next step's plan for
        this slot allocates nothing for the tail.
        """
        dummy_rids = self.slot_for(lease).dummy_rids
        self._dummy_rows.reset(dummy_rids[real_bs:lease.bucket.bs])

    def plan_stream(self) -> torch.cuda.Stream | None:
        """Dedicated stream for pre-planning.

        Pre-plan must not submit onto the default stream: whether its memcpys
        land before or after the GPU thread records the previous batch's
        completion event is timing-dependent, and landing after delays that
        event past pre-plan's own kernels. Its own stream keeps the two
        independent; a plan-done event gates the replay that reads the buffers.
        """
        if not torch.cuda.is_available():
            return None
        if self._plan_stream is None:
            self._plan_stream = torch.cuda.Stream(device=self._device)
        return self._plan_stream


# Piecewise buckets share the resources' (bucket, slot, label) buffer space
# with the full-forward captures. They never coexist on one node — a forward
# either replays whole or runs eager around captured regions — so the walk name
# below plus the region's label is enough to keep the keys apart.
PIECEWISE_WALK = "__piecewise__"


@dataclass
class PiecewiseGraphData:
    """One captured region, for one (bs, total_tokens) bucket."""
    graph: torch.cuda.CUDAGraph
    static_inputs: dict[str, torch.Tensor]
    static_outputs: dict[str, torch.Tensor]
    dummy_rids: list[str]
    shape: PiecewiseCaptureShape
    bucket: BucketKey


class PiecewiseOutput:
    """Dict-like view over a captured region's output buffers.

    The runner replays into persistent buffers sized for the padded bucket;
    only the leading ``real_len`` rows are meaningful. Indexing and ``get``
    return an owned CLONE of that leading slice — safe to keep past the next
    replay. ``get_view`` returns the same slice WITHOUT copying.
    """
    __slots__ = ("_outputs", "_real_len")

    def __init__(self, outputs: dict[str, torch.Tensor], real_len: int):
        self._outputs = outputs
        self._real_len = real_len

    def __contains__(self, key: str) -> bool:
        return key in self._outputs

    def keys(self):
        return self._outputs.keys()

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._outputs[key][:self._real_len].clone()

    def get(self, key: str, default=None):
        value = self._outputs.get(key)
        if value is None:
            return default
        return value[:self._real_len].clone()

    def get_view(self, key: str, default=None):
        """The leading ``real_len`` slice WITHOUT copying.

        The result aliases the runner-owned static output buffer and is
        OVERWRITTEN by the next ``run``. Read it within the same step; use
        ``get`` when you need something that outlives the step.
        """
        value = self._outputs.get(key)
        if value is None:
            return default
        return value[:self._real_len]


class PiecewiseCudaGraphRunner:
    """Captures one inner callable of a submodule's forward as a CUDA graph.

    Where ``CudaGraphRunner`` replays a whole ``forward_batched`` under engine
    control, this captures a SUB-REGION — a transformer block loop, say —
    while the surrounding preamble stays eager and the submodule invokes
    ``run`` itself. The config supplies the callable, its static buffers, and
    the region's own step declaration; the runner drives that step through the
    same admit → plan → commit cycle the engine uses, so the region's
    attention plan lands in the resources' per-(bucket, slot, label) buffers
    and the captured graph reads them at fixed addresses.
    """

    CAPTURE_BATCH_SIZES = DEFAULT_CAPTURE_BATCH_SIZES
    NUM_WARMUP = 2
    # A region can't overlap itself, and there is no pre-plan path into one, so
    # a single slot suffices.
    SLOT = 0

    def __init__(
        self,
        label: str,
        config: PiecewiseCudaGraphConfig,
        resources: Mapping[str, Resource],
        step_runner: StepRunner,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
        joint_comm_group: JointGroups | None = None,
        node_name: str | None = None,
    ):
        self._label = label
        self._config = config
        self._resources = resources
        self._step_runner = step_runner
        self._device = device
        self._autocast_dtype = autocast_dtype
        self._comm_group = joint_comm_group
        # scopes buffer allocation; `label` carries it but mangled with the region
        self._node_name = node_name
        self._prepared_shapes: list | None = None

        self._capture_batch_sizes = sorted(
            config.capture_batch_sizes or self.CAPTURE_BATCH_SIZES
        )
        self._graphs: dict[tuple[int, int], PiecewiseGraphData] = {}
        self._memory_pool = None
        self._dummy_rows = DummyRowPool(
            prefix=f"pw_{label}", step_runner=step_runner, resources=resources,
        )

    @property
    def any_graphs(self) -> bool:
        return bool(self._graphs)

    def _bucket(self, shape: PiecewiseCaptureShape) -> BucketKey:
        return BucketKey(
            graph_walk=PIECEWISE_WALK,
            bs=shape.bs,
            num_tokens=shape.total_tokens,
            cg_key_info=self._label,
        )

    # ── Capture ─────────────────────────────────────────────────────────

    def prepare_for_capture(self) -> list:
        """Claim this region's static buffers; see ``CudaGraphRunner``."""
        if self._prepared_shapes is None:
            shapes = self._config.get_capture_shapes(self._capture_batch_sizes)
            if shapes:
                self._step_runner.build_cuda_graph_buffers(
                    [
                        CGSlotSpec(
                            bucket=self._bucket(shape), slot=self.SLOT,
                            config=self._config,
                        )
                        for shape in shapes
                    ],
                    max_bs=max(shape.bs for shape in shapes),
                    max_seq_len=max(shape.total_tokens for shape in shapes),
                    node_name=self._node_name,
                )
            self._prepared_shapes = shapes
        return self._prepared_shapes

    def warmup_and_capture(self) -> None:
        if self._device is None or not torch.cuda.is_available():
            logger.warning(
                "CUDA not available, skipping piecewise capture for %s", self._label
            )
            return

        torch.cuda.set_device(self._device)
        self._memory_pool = torch.cuda.graphs.graph_pool_handle()

        shapes = self.prepare_for_capture()
        if not shapes:
            return

        # largest bucket first, matching the full-forward runner
        for shape in sorted(
            shapes, key=lambda s: (s.bs, s.total_tokens), reverse=True
        ):
            # keep ranks in lockstep: the region may hold collectives, and a
            # rank still in pre-capture setup would mismatch one already in the
            # warmup forward
            if self._comm_group is not None:
                self._comm_group.tp_group.barrier()
                self._comm_group.sp_group.barrier()
            try:
                self._capture_one(shape)
                logger.info(
                    "PiecewiseCudaGraphRunner[%s]: captured bs=%d total_tokens=%d",
                    self._label, shape.bs, shape.total_tokens,
                )
            except Exception:
                logger.warning(
                    "PiecewiseCudaGraphRunner[%s]: failed to capture bs=%d "
                    "total_tokens=%d", self._label, shape.bs, shape.total_tokens,
                    exc_info=True,
                )

    def _capture_one(self, shape: PiecewiseCaptureShape) -> None:
        dummy_rids = self._dummy_rows.ensure(
            f"{shape.bs}_{shape.total_tokens}", shape.bs
        )
        static_inputs = self._config.make_static_inputs(shape)
        step = self._declare(
            dummy_rids, shape.seq_lens, self._bucket(shape), capture=True,
        )
        call = self._call_inputs(static_inputs, dummy_rids)

        fn = self._config.capture_fn
        if self._config.compile:
            fn = torch.compile(
                fn, mode="max-autotune-no-cudagraphs", fullgraph=False, dynamic=False,
            )

        def run_fn():
            return fn(call)

        try:
            self._plan(step, shape)
            torch.cuda.synchronize()
            for _ in range(self.NUM_WARMUP):
                with autocast_scope(self._autocast_dtype):
                    run_fn()
                # back to a clean stream state so the re-plan below (and the
                # capture after it) sees the shapes the first plan did
                self._dummy_rows.reset(dummy_rids)
                self._plan(step, shape)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with autocast_scope(self._autocast_dtype):
                with torch.cuda.graph(graph, pool=self._memory_pool):
                    static_outputs = self._normalize_output(run_fn())
            torch.cuda.synchronize()
        finally:
            # pages stay with the dummy streams: replay's padding rows address
            # the same ids, so their plan finds the storage already resident
            self._dummy_rows.reset(dummy_rids)

        self._graphs[(shape.bs, shape.total_tokens)] = PiecewiseGraphData(
            graph=graph,
            static_inputs=static_inputs,
            static_outputs=static_outputs,
            dummy_rids=list(dummy_rids),
            shape=shape,
            bucket=self._bucket(shape),
        )

    def _call_inputs(
        self,
        static_inputs: dict[str, torch.Tensor],
        step_ids: list[str],
    ) -> PiecewiseCallInputs:
        """What the region is handed, over the padded capture batch.

        Built once, for the capture: the region's Python runs only there, so
        the rows are the runner's padding ids and the only thing a replay
        changes is the contents of the static buffers.
        """
        return PiecewiseCallInputs(
            static_inputs=static_inputs,
            engine_inputs=ModelInputsFromEngine(
                request_ids=list(step_ids),
                per_request_info=dummy_metadata(step_ids, PIECEWISE_WALK),
                resources=dict(self._resources),
            ),
            kwargs=self._config.forward_kwargs,
        )

    def _declare(
        self, request_ids: list[str], seq_lens: list[int],
        bucket: BucketKey, capture: bool,
    ):
        """The region's step over the padded batch, addressed at its slot."""
        if self._config.declare_step is None:
            return None
        step = self._config.declare_step(list(request_ids), list(seq_lens))
        if step is None:
            return None
        step.set_ctx(StepContext(
            request_ids=tuple(request_ids),
            graph_walk=PIECEWISE_WALK,
            slot=self.SLOT,
            capture=capture,
            slot_lease=SlotLease(slot=self.SLOT, bucket=bucket),
        ))
        return step

    def _plan(self, step, shape: PiecewiseCaptureShape) -> None:
        if step is None:
            return
        outcome = self._step_runner.admit(step)
        if not outcome.ok:
            raise RuntimeError(
                f"piecewise {self._label!r} admit failed for bs={shape.bs}, "
                f"total_tokens={shape.total_tokens}: {outcome.reason}"
            )
        self._step_runner.plan(step)

    @staticmethod
    def _normalize_output(out) -> dict[str, torch.Tensor]:
        """Coerce the captured callable's return into ``{name: Tensor}``.

        The contract is a dict; a bare tensor is accepted (under ``"x"``) so a
        single-output block loop can ``return x`` directly.
        """
        if isinstance(out, torch.Tensor):
            return {"x": out}
        if isinstance(out, dict):
            return out
        raise TypeError(
            f"piecewise capture_fn must return a Tensor or dict[str, Tensor], "
            f"got {type(out).__name__}"
        )

    # ── Replay ──────────────────────────────────────────────────────────

    def _padded_bs(self, batch_size: int) -> int | None:
        return next(
            (bs for bs in self._capture_batch_sizes if bs >= batch_size), None
        )

    def _resolve(
        self, batch_size: int, total_tokens: int | None,
    ) -> PiecewiseGraphData | None:
        """The captured bucket serving this call, or None for eager.

        BATCHED buckets are fixed by the padded batch size; PACKED ones take
        the smallest captured token bucket that fits.
        """
        padded_bs = self._padded_bs(batch_size)
        if padded_bs is None:
            return None
        if self._config.get_config_type() == PiecewiseConfigType.BATCHED:
            return self._graphs.get(
                (padded_bs, self._config.seq_len * padded_bs)
            )
        if total_tokens is None:
            return None
        candidates = sorted(
            tokens for (bs, tokens) in self._graphs
            if bs == padded_bs and tokens >= total_tokens
        )
        if not candidates:
            return None
        return self._graphs[(padded_bs, candidates[0])]

    def can_run(self, batch_size: int, total_tokens: int | None = None) -> bool:
        return self._resolve(batch_size, total_tokens) is not None

    def run(
        self,
        static_inputs: dict[str, torch.Tensor],
        request_ids: list[str] | None = None,
        seq_lens: list[int] | None = None,
        real_bs: int | None = None,
    ) -> PiecewiseOutput:
        """Replay the captured region for these real inputs.

        Copies each real input into the runner-owned buffer of the same name,
        declares and plans the region's step over the padded batch, replays,
        then commits and returns the padded buffers behind a real-length view.
        Under ``reuses_outer_plan`` the middle three are the outer step's job
        and this only stages and replays.

        Only the static buffers carry data into a replay: the region's Python
        ran once, at capture, so whatever it read off ``PiecewiseCallInputs``
        is baked into the graph.
        """
        if real_bs is None:
            if request_ids is not None:
                real_bs = len(request_ids)
            elif seq_lens is not None:
                real_bs = len(seq_lens)
            else:
                raise ValueError(
                    "piecewise run: pass real_bs, request_ids, or seq_lens to "
                    "determine the batch size"
                )

        is_packed = self._config.get_config_type() == PiecewiseConfigType.PACKED
        real_total_tokens = sum(seq_lens) if seq_lens is not None else None
        data = self._resolve(real_bs, real_total_tokens if is_packed else None)
        if data is None:
            raise RuntimeError(
                f"piecewise {self._label!r}: no captured graph for bs={real_bs}, "
                f"total_tokens={real_total_tokens}"
            )

        for name, value in static_inputs.items():
            buffer = data.static_inputs.get(name)
            if buffer is None or not isinstance(value, torch.Tensor):
                continue
            n = value.shape[0]
            buffer[:n].copy_(value)
            if n < buffer.shape[0]:
                # the padded tail is real compute for a BATCHED capture, so it
                # reads whatever is here; zero rather than last step's values
                buffer[n:].zero_()

        if self._config.reuses_outer_plan:
            # the outer step's plan is live and covers this region
            data.graph.replay()
            return PiecewiseOutput(
                data.static_outputs, real_total_tokens if is_packed else real_bs
            )

        step = None
        if request_ids is not None:
            step_ids = [
                *request_ids, *data.dummy_rids[real_bs:data.shape.bs]
            ]
            step = self._declare(
                step_ids,
                self._config.replay_seq_lens(data.shape, seq_lens, real_bs),
                data.bucket,
                capture=False,
            )
        try:
            self._plan(step, data.shape)
            data.graph.replay()
            if step is not None:
                self._step_runner.commit(step)
        finally:
            self._dummy_rows.reset(data.dummy_rids[real_bs:data.shape.bs])

        return PiecewiseOutput(
            data.static_outputs, real_total_tokens if is_packed else real_bs
        )
