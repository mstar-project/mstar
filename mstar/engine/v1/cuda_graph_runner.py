from dataclasses import dataclass
import logging
import os
from typing import Mapping

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import CGSlotSpec, Resource
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext, SubmoduleStep
from mstar.engine.v1.cuda_graph_config import CudaGraphConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeSubmodule

logger = logging.getLogger(__name__)


DEFAULT_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]


@dataclass
class CudaGraphSlot:
    """One captured graph + its private FlashInfer wrappers.

    Double-buffer: each (graph_walk, requires_cfg, bs, num_tokens)
    key holds two of these in ``CudaGraphData.slots``. Replay alternates
    between slots so plan(N+1) on the inactive slot's wrapper can run
    concurrently with replay(N) on the active slot.
    """
    graph: torch.cuda.CUDAGraph
    static_inputs: dict
    static_outputs: dict
    # Cached at capture time: True iff any dummy_rid in static_outputs has a
    # key besides "logits". When False, sample_and_remap can skip the
    # per-rid collection loop entirely.
    has_non_logit_outputs: bool = False


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

        self._slots: dict[BucketKey, CudaGraphSlot] = {}

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

        # Sum of bytes that WOULD have been allocated by per-capture clones
        # (one full tensor per call). For logging purposes.
        self._capture_clone_bytes_naive = 0

        # Plan-overlap stream. Lazily created the first time pre_plan
        # is called from Worker.plan_executor.
        self._plan_stream: "torch.cuda.Stream | None" = None

        self.max_bs = max(
            (max(self._batch_sizes(config)) for config in self._capture_configs),
            default=1,
        )

    @property
    def any_graphs(self):
        return bool(self._slots)

    def _batch_sizes(self, config: CudaGraphConfig) -> list[int]:
        return config.capture_batch_sizes or self.CAPTURE_BATCH_SIZES

    def _get_slot_specs(self) -> list[CGSlotSpec]:
        return [
            CGSlotSpec(
                bucket=BucketKey(
                    graph_walk=config.capture_graph_walk,
                    cg_key_info=config.additional_key_info,
                    bs=bs,
                    num_tokens=num_tokens,
                ),
                slot=slot,
                config=config,
                config_idx=config_idx,
            )
            for config_idx, config in enumerate(self._capture_configs)
            for bs in self._batch_sizes(config)
            for num_tokens in config.get_total_tokens(bs)
            for slot in range(self.NUM_SLOTS)
        ]

    def _get_addtl_slot_specs(self, spec: CGSlotSpec):
        return [
            CGSlotSpec(
                bucket=BucketKey(
                    graph_walk=walk,
                    cg_key_info=spec.bucket.cg_key_info,
                    bs=spec.bs,
                    num_tokens=spec.num_tokens,
                ),
                slot=spec.slot,
                config=spec.config,
                config_idx=spec.config_idx,
            ) for walk in spec.config.replay_graph_walks \
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
                self._slots[spec] = slot
                for addtl in self._get_addtl_slot_specs(spec):
                    self._slots[addtl] = slot

                logger.info(
                    "Captured CUDA graph for %s: %s",
                    self._submodule_name, spec
                )
            except Exception:
                logger.warning(
                    "Failed to capture CUDA graph for %s: %s",
                    self._submodule_name, spec
                )
        mem_after = torch.cuda.memory_allocated(self._device)
        self._log_memory(mem_before, mem_after)

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
        dummy_rids = self._make_dummy_rids(spec.config, spec.bs)
        dummy_inputs = spec.config.get_node_inputs(spec.bs, spec.num_tokens)
        engine_inputs = self._get_dummy_engine_inputs(dummy_rids, walk)
        try:
            for rid in dummy_rids:
                self._step_runner.ingest_request(rid)
            step = self._submodule.declare_step(
                graph_walk=walk, inputs=dummy_inputs
            )
            
            if step is not None:
                step._ctx = StepContext(
                    request_ids=dummy_rids,
                    graph_walk=walk,
                    slot=spec.slot,
                    capture=True,
                    slot_lease=SlotLease(
                        slot=spec.slot,
                        bucket=spec.bucket,
                        filler=tuple()
                    )
                )

            def prepare():
                if step is not None:
                    step._ctx.plan_results.clear()
                    self._step_runner.admit(step)
                    self._step_runner.plan(step)

                return self._submodule.preprocess(
                    graph_walk=walk,
                    engine_inputs=engine_inputs,
                    inputs=dummy_inputs,
                )
            static_inputs = prepare()

            forward = getattr(self._submodule, config.capture_forward_method)
            if config.compile:
                forward = torch.compile(
                    forward,
                    mode="max-autotune-no-cudagraphs",
                    fullgraph=False,
                    dynamic=False,
                )

            def run_forward(
                _forward=forward,
                _engine_inputs=engine_inputs,
                _kwargs=static_inputs,
            ):
                return _forward(
                    graph_walk=config.capture_graph_walk,
                    engine_inputs=_engine_inputs,
                    **_kwargs,
                )

            torch.cuda.set_device(self._device)
            torch.cuda.synchronize()
            for _ in range(2):
                with torch.amp.autocast("cuda", enabled=True, dtype=self._autocast_dtype):
                    run_forward()
                self._reset_dummy_rids(spec, free=False)
                prepare()
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.amp.autocast("cuda", enabled=True, dtype=self._autocast_dtype):
                with torch.cuda.graph(graph, pool=self._memory_pool):
                    output = run_forward()
            torch.cuda.synchronize()

            slot = self._build_slot_from_capture(
                output=output,
                graph=graph,
                static_inputs=static_inputs,
            )
        finally:
            for rids in dummy_rids:
                self._reset_dummy_rids(rids, free=True)
        return slot


    def _make_dummy_rids(
        self, config: CudaGraphConfig, bs: int, slot_idx: int = 0,
    ):
        addl_key = hash(config.additional_key_info) if config.additional_key_info else ""
        dummy_rids = [
            f"__cg_{config.capture_graph_walk}_{addl_key}_slot{slot_idx}_{i}__"
            for i in range(bs)
        ]

        # Add dummy requests with all needed labels
        for rid in dummy_rids:
            self._step_runner.ingest_request(rid)
        return dummy_rids

    def _get_dummy_engine_inputs(self, dummy_rids: list[str], graph_walk: str):
        dummy_metadata = {
            rid: CurrentForwardPassInfo(
                request_id=rid,
                graph_walk=graph_walk,
                fwd_index=0,
                random_seed=0,
                max_tokens=1,
                sampling_config={}
            ) for rid in dummy_rids
        }
        return ModelInputsFromEngine(
            resources=dummy_rids,
            per_request_info=dummy_metadata,
            resources=self._resources
        )

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
        self, output, graph, static_inputs,
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
            static_outputs=output,
            has_non_logit_outputs=has_non_logit,
        )