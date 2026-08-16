from dataclasses import dataclass, field, replace
import logging
import os
from typing import Any, Mapping

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import CGSlotSpec, Resource
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext
from mstar.engine.v1.cuda_graph_config import CudaGraphConfig
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)


DEFAULT_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]


@dataclass
class CudaGraphSlot:
    """One captured graph + the buffers its replay reads and writes.

    Double-buffer: each bucket holds NUM_SLOTS of these. Replay alternates
    between slots so plan(N+1) on the inactive slot's resources can run
    concurrently with replay(N) on the active slot.
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
    # Cached at capture time: True iff any dummy_rid in static_outputs has a
    # key besides "logits". When False, output remap can skip the per-rid
    # collection loop entirely.
    has_non_logit_outputs: bool = False


@dataclass
class CudaGraphBucket:
    """The captured slots for one (walk, cg_key_info, bs, num_tokens)."""
    config: CudaGraphConfig
    config_idx: int
    slots: list[CudaGraphSlot] = field(default_factory=list)
    next_slot: int = 0


class CudaGraphRunner:
    # Double-buffer: capture two graphs per config key. Replay alternates so
    # plan(N+1) on the inactive slot can run concurrent with replay(N) on the
    # active slot. Override via MSTAR_NUM_SLOTS=1 to disable double-buffer
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
        autocast_dtype: torch.dtype,
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

        self._buckets: dict[BucketKey, CudaGraphBucket] = {}

        self._memory_pool = None

        # (config_idx, tensor_key) → max-bucket static buffer. Lazily populated
        # by _intern_static_buffer on the first capture. Smaller-bucket captures
        # slice the leading dim of the same buffer
        self._shared_static_buffers: dict[tuple[int, str], torch.Tensor] = {}

        # (config_idx, tensor_key) → the dim that carries the (bucket-varying)
        # seq length in that tensor's original layout. _intern_static_buffer
        # brings this dim to the front for storage (so smaller buckets reslice
        # along dim 0) and inverts the move on return
        self._static_buffer_seq_dims: dict[tuple[int, str], int] = {}

        # (config_idx, slot) → the padding rows' request ids, ingested once and
        # kept for the runner's lifetime so their pages stay resident. Sized by
        # the largest bucket (captures run largest-first); smaller buckets take
        # a prefix.
        self._dummy_rids: dict[tuple[int, int], list[str]] = {}

        # Sum of bytes that WOULD have been allocated by per-capture clones
        # (one full tensor per call). For logging purposes.
        self._capture_clone_bytes_naive = 0

        # Plan-overlap stream. Lazily created the first time pre_plan
        # is called from Worker.plan_executor.
        self._plan_stream: torch.cuda.Stream | None = None

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
                    for slot in range(self.NUM_SLOTS):
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

    def warmup_and_capture(self):
        """Capture graphs for all configs and batch sizes."""
        if self._device is None or not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping graph capture for %s",
                            self._submodule_name)
            return

        self._memory_pool = torch.cuda.graphs.graph_pool_handle()
        mem_before = torch.cuda.memory_allocated(self._device)

        slot_specs = self._get_slot_specs()
        # reverse sort based on batch size, then total tokens (largest first)
        slot_specs.sort(key=lambda s: (s.bs, s.num_tokens), reverse=True)

        max_seq_len = max((
            spec.bucket.num_tokens for spec in slot_specs
        ), default=1)
        self._step_runner.build_cuda_graph_buffers(
            slot_specs, max_bs=self.max_bs, max_seq_len=max_seq_len
        )

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
        dummy_rids = self._ensure_dummy_rids(spec.config_idx, spec.slot, spec.bs)
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

            forward = getattr(self._submodule, config.capture_forward_method)
            if config.compile:
                forward = torch.compile(
                    forward,
                    mode="max-autotune-no-cudagraphs",
                    fullgraph=False,
                    dynamic=False,
                )

            def run_forward():
                return forward(
                    graph_walk=walk,
                    engine_inputs=engine_inputs,
                    **static_inputs,
                )

            torch.cuda.set_device(self._device)
            torch.cuda.synchronize()
            for _ in range(self.NUM_WARMUP):
                with torch.amp.autocast("cuda", enabled=True, dtype=self._autocast_dtype):
                    run_forward()
                # back to a clean stream state so the re-plan below (and the
                # capture after it) sees the same shapes the first prepare did
                self._reset_dummy_rids(dummy_rids)
                prepare()
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.amp.autocast("cuda", enabled=True, dtype=self._autocast_dtype):
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
            self._reset_dummy_rids(dummy_rids)

    def _dummy_rid_names(
        self, config_idx: int, slot: int, bs: int,
    ) -> list[str]:
        return [
            f"__cg_{self._submodule_name}_{config_idx}_slot{slot}_{i}__"
            for i in range(bs)
        ]

    def _ensure_dummy_rids(
        self, config_idx: int, slot: int, bs: int,
    ) -> list[str]:
        """The padding rows for one (config, slot), ingested once.

        Re-ingesting would hand the resources a fresh stream and orphan the
        pages the previous capture left resident, so existing ids are reused.
        """
        held = self._dummy_rids.setdefault((config_idx, slot), [])
        names = self._dummy_rid_names(config_idx, slot, bs)
        for rid in names[len(held):]:
            self._step_runner.ingest_request(rid)
            held.append(rid)
        return names

    def _dummy_engine_inputs(
        self, dummy_rids: list[str], graph_walk: str,
    ) -> ModelInputsFromEngine:
        return ModelInputsFromEngine(
            request_ids=list(dummy_rids),
            per_request_info=self._dummy_metadata(dummy_rids, graph_walk),
            resources=dict(self._resources),
        )

    def _dummy_metadata(
        self, dummy_rids: list[str], graph_walk: str,
    ) -> dict[str, CurrentForwardPassInfo]:
        return {
            rid: CurrentForwardPassInfo(
                request_id=rid,
                graph_walk=graph_walk,
                requires_cfg=False,
                fwd_index=0,
                random_seed=0,
                max_tokens=1,
                sampling_config={},
            ) for rid in dummy_rids
        }

    def _reset_dummy_rids(self, dummy_rids: list[str], free: bool=False):
        for rid in dummy_rids:
            for resource in self._resources.values():
                resource.reset_request(rid, free=free)

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
        seq_dim = self._seq_dim(value, seq_len) if seq_len is not None else 0
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
        has_non_logit = False
        if isinstance(output, dict):
            for k, v in output.items():
                if k == "__batched_logits__":
                    continue
                if isinstance(v, dict) and any(
                    out_key != "logits" for out_key in v.keys()
                ):
                    has_non_logit = True
                    break
        return CudaGraphSlot(
            graph=graph,
            static_inputs=static_inputs,
            static_input_keys=static_input_keys,
            static_outputs=output,
            dummy_rids=list(dummy_rids),
            dummy_metadata=dict(dummy_metadata),
            config_idx=config_idx,
            has_non_logit_outputs=has_non_logit,
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

    def can_run(
        self, graph_walk: str, bs: int, num_tokens: int,
        cg_key_info: Any | None = None,
    ) -> bool:
        return self.select_bucket(
            graph_walk, bs, num_tokens, cg_key_info
        ) is not None

    def lease_slot(
        self, graph_walk: str, bs: int, num_tokens: int,
        cg_key_info: Any | None = None,
        slot: int | None = None,
    ) -> SlotLease | None:
        """Pick the bucket and double-buffer slot an upcoming step replays on.

        ``slot=None`` advances the bucket's counter so the next lease lands on
        the other slot; a caller that already reserved one (pre-plan) passes it
        back so both submissions target the same slot.
        """
        key = self.select_bucket(graph_walk, bs, num_tokens, cg_key_info)
        if key is None:
            return None
        bucket = self._buckets[key]
        if slot is None:
            slot = bucket.next_slot
            bucket.next_slot = (bucket.next_slot + 1) % len(bucket.slots)
        return SlotLease(slot=slot % len(bucket.slots), bucket=key)

    def slot_for(self, lease: SlotLease) -> CudaGraphSlot:
        return self._buckets[lease.bucket].slots[lease.slot]

    def config_for(self, lease: SlotLease) -> CudaGraphConfig:
        return self._buckets[lease.bucket].config

    def pad_inputs(
        self, lease: SlotLease, inputs: list[ARNodeInputs],
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
    ) -> dict:
        """Stage this step's inputs into the slot and replay it.

        ``plan_done_event`` is the pre-plan's event: its plan wrote this slot's
        buffers on another stream, so the replay has to wait for those writes
        before it reads them.
        """
        self._stage(lease, preprocessed)
        if plan_done_event is not None:
            torch.cuda.default_stream(self._device).wait_event(plan_done_event)
        return self._replay(lease)

    def release(self, lease: SlotLease, real_bs: int) -> None:
        """Return the padding rows to their at-rest state after a step.

        Their pages stay resident (``free=False``), so the next step's plan for
        this slot allocates nothing for the tail.
        """
        dummy_rids = self.slot_for(lease).dummy_rids
        self._reset_dummy_rids(dummy_rids[real_bs:lease.bucket.bs])

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
