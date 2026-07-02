"""Stateless engine for submodules with no cross-batch GPU state.

A single configurable engine class handles encoder/decoder forwards, flow /
diffusion forwards, and audio codec forwards. They share the same execution
skeleton — ``prepare_inputs → can_batch? → forward[_batched] → postprocess`` —
and differ only in knobs (autocast dtype, whether ``torch.compile`` is applied,
whether the piecewise CUDA-graph runner is opt-in, etc.). Those knobs live on
``StatelessEngineConfig``.

Stateful engines (paged KV cache, FlashInfer planning, sampling, CFG) live in
``kv_cache_engine.py`` and do NOT use this class.
"""

import logging
import time
from dataclasses import dataclass

import torch

from mstar.distributed.communication import WorkerTPGroups
from mstar.engine.base import (
    BaseEngine,
    EngineType,
    NodeBatch,
    NodeOutput,
    PlannedBatch,
    PreparedBatch,
)
from mstar.engine.cuda_graph_runner import StatelessCudaGraphRunner
from mstar.model.submodule_base import (
    ARNodeInputs,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.utils.profiler import range_pop, range_push

logger = logging.getLogger(__name__)


@dataclass
class StatelessEngineConfig:
    """Knobs that distinguish stateless engine variants.

    ``engine_type`` is the routing key the worker / EngineManager keys off of.
    The remaining fields gate behaviors that vary between flavors:

    - ``autocast_dtype`` of ``None`` disables autocast entirely and switches
      the engine to ``torch.inference_mode()`` (audio codec needs float32).
    - ``force_float32_submodules`` casts every submodule to ``.float()`` in
      ``load_model``. Required when ``autocast_dtype is None`` and the
      reference run is numerically sensitive (audio codec).
    - ``cuda_graph_capable`` enables ``StatelessCudaGraphRunner`` capture during
      ``warmup``. Submodules still opt in by returning a non-empty list from
      ``get_cuda_graph_configs(device)``.
    - ``apply_torch_compile`` enables ``torch.compile(submodule.forward)``
      during warmup. Audio codec disables this because the one-shot compile
      cost is too high for short forwards.
    - ``enable_piecewise_runner`` enables ``PiecewiseCudaGraphRunner`` for
      submodules that return a config from ``get_piecewise_runner_config()``.
      Used by VJepa2's masked predictor.
    - ``name`` is the NVTX range prefix.
    """

    engine_type: EngineType = EngineType.STATELESS
    autocast_dtype: torch.dtype | None = torch.bfloat16
    force_float32_submodules: bool = False
    cuda_graph_capable: bool = True
    apply_torch_compile: bool = True
    enable_piecewise_runner: bool = False
    name: str = "stateless"


def make_enc_dec_config(autocast_dtype: torch.dtype | None) -> StatelessEngineConfig:
    return StatelessEngineConfig(
        engine_type=EngineType.STATELESS,
        autocast_dtype=autocast_dtype,
        cuda_graph_capable=True,
        apply_torch_compile=True,
        enable_piecewise_runner=True,
        name="enc_dec",
    )


def make_audio_codec_config(_autocast_dtype: torch.dtype | None = None) -> StatelessEngineConfig:
    return StatelessEngineConfig(
        engine_type=EngineType.STATELESS,
        autocast_dtype=None,
        force_float32_submodules=True,
        cuda_graph_capable=True,
        apply_torch_compile=False,
        enable_piecewise_runner=False,
        name="audio_codec",
    )


class StatelessEngine(BaseEngine):
    """Single engine class for all stateless submodules.

    Holds the submodules it was given, plus any ``StatelessCudaGraphRunner``s
    captured during warmup. ``execute_batch`` runs the universal
    prepare → dispatch → postprocess skeleton; the dispatch picks between
    CUDA-graph replay, batched forward, and per-request sequential.
    """

    def __init__(
        self,
        config: StatelessEngineConfig | None = None,
        enable_nvtx: bool = False,
        autocast_dtype: torch.dtype | None = None,
        enable_prof: bool = False,
        **kwargs,
    ):
        super().__init__(enable_nvtx=enable_nvtx, enable_profile=enable_prof)
        if config is None:
            config = StatelessEngineConfig()
        if autocast_dtype is not None and config.autocast_dtype is not None:
            # EngineManager passes a runtime autocast_dtype that overrides the
            # config default (e.g. a model that prefers float16 over bfloat16).
            # Audio codec sets autocast_dtype=None in its config to lock it out
            # — the runtime value MUST NOT override that.
            config = StatelessEngineConfig(
                engine_type=config.engine_type,
                autocast_dtype=autocast_dtype,
                force_float32_submodules=config.force_float32_submodules,
                cuda_graph_capable=config.cuda_graph_capable,
                apply_torch_compile=config.apply_torch_compile,
                enable_piecewise_runner=config.enable_piecewise_runner,
                name=config.name,
            )
        self.config = config
        self.submodules: dict[str, NodeSubmodule] = {}
        self.device: torch.device | None = None
        self.cuda_graph_runners: dict[str, StatelessCudaGraphRunner] = {}

        # Dedup set for "captured graphs exist but don't match this shape"
        # warnings — each unique miss is logged at most once.
        self._logged_graph_misses: set[tuple] = set()

    # ─── BaseEngine interface ─────────────────────────────────────────

    def engine_type(self) -> EngineType:
        return self.config.engine_type

    def has_autocast(self) -> bool:
        return self.config.autocast_dtype is not None

    def get_max_batch_size(self, node_name: str, graph_walk: str):
        submodule = self.submodules.get(node_name)
        if submodule is None:
            return None
        submod_max_bs = submodule.max_batch_size(graph_walk)
        runner = self.cuda_graph_runners.get(node_name)
        if runner is None:
            return submod_max_bs
        configs = [cfg for cfg in runner.capture_configs if graph_walk in cfg.replay_graph_walks]
        if not configs:
            return submod_max_bs
        max_graph_bs = max(
            max(cfg.capture_batch_sizes or runner.DEFAULT_CAPTURE_BATCH_SIZES) for cfg in configs
        )
        if submod_max_bs is not None:
            return min(max_graph_bs, submod_max_bs)
        return max_graph_bs

    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        tp_groups: WorkerTPGroups,
        device: torch.device,
        **kwargs,
    ) -> None:
        if self.config.force_float32_submodules:
            # Reference parity for numerically sensitive forwards (audio codec).
            self.submodules = {name: mod.float() for name, mod in submodules.items()}
        else:
            self.submodules = dict(submodules)
        self.device = device
        self.tp_groups = tp_groups

    def add_request(self, request_id: str, **kwargs) -> None:
        pass  # stateless

    def remove_request(self, request_id: str) -> None:
        for submodule in self.submodules.values():
            submodule.cleanup_request(request_id)

    def check_stop_for_batch(
        self, batch: NodeBatch, output: NodeOutput
    ) -> dict[str, set[str]]:
        """Delegate to each rid's ``submodule.check_stop``.

        Called by the worker on its slow-postprocess path so the ``.item()`` /
        ``.cpu()`` reads no longer block ``execute_batch`` on the GPU thread.
        """
        if batch.node_name not in self.submodules:
            return {}
        submodule = self.submodules[batch.node_name]
        result: dict[str, set[str]] = {}
        for rid in batch.request_ids:
            req_outputs = output.per_request_output_tensors.get(rid, {})
            if not req_outputs:
                continue
            req_info = batch.per_request_info.get(rid)
            if req_info is None:
                continue
            stops = submodule.check_stop(rid, req_info, req_outputs)
            if stops:
                result[rid] = stops
        return result

    # ─── Core execution ────────────────────────────────────────────────

    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        """Wrap the template with the NVTX range and inference context that
        all stateless forwards run under.
        """
        if self.enable_nvtx:
            range_push(
                f"engine.{self.config.name}.{batch.node_name}."
                f"{batch.graph_walk}.bs{len(batch.request_ids)}"
            )
        self.tp_groups.get_tp_config_for_node(batch.node_name).barrier()
        submodule = self.submodules.get(batch.node_name)
        per_submodule_dtype = submodule.get_autocast_dtype() if submodule is not None else None
        try:
            with self._inference_context(per_submodule_dtype):
                return super().execute_batch(batch)
        finally:
            if self.enable_nvtx:
                range_pop(synchronize=False)

    def prepare_batch(self, batch: NodeBatch) -> PreparedBatch:
        """Look up the submodule, run per-rid ``prepare_inputs``, and filter
        out rids the submodule vetoed (returned None).

        Mutates ``batch.request_ids`` / ``batch.per_request_info`` to drop
        skipped rids when any exist — downstream batched forwards rely on the
        batch only containing active rids. The template emits empty output
        slots for skipped rids so the worker still sees one entry per request.
        """
        submodule = self.submodules.get(batch.node_name)
        if submodule is None:
            # Marking every rid as skipped → template emits empty output.
            return PreparedBatch(
                batch=batch,
                submodule=None,
                skipped_rids=set(batch.request_ids),
            )

        node_inputs, skipped_rids = self._prepare_inputs(batch, submodule)
        if skipped_rids:
            batch.request_ids = [rid for rid in batch.request_ids if rid not in skipped_rids]
            batch.per_request_info = {
                rid: info
                for rid, info in batch.per_request_info.items()
                if rid not in skipped_rids
            }
        return PreparedBatch(
            batch=batch,
            submodule=submodule,
            node_inputs=node_inputs,
            skipped_rids=skipped_rids,
        )

    def execute_forward(self, planned: PlannedBatch) -> NodeOutput:
        if planned.submodule is None:
            return NodeOutput(per_request_output_tensors={})
        return self._dispatch(planned.batch, planned.node_inputs, planned.submodule)

    def postprocess_batch(self, planned: PlannedBatch, output: NodeOutput) -> None:
        submodule = planned.submodule
        if submodule is None:
            return
        for rid in planned.batch.request_ids:
            info = planned.batch.per_request_info.get(rid)
            if info is None:
                continue
            submodule.postprocess(
                request_id=rid,
                request_info=info,
                outputs=output.per_request_output_tensors.get(rid, {}),
            )

    # ─── Internals ─────────────────────────────────────────────────────

    def _inference_context(self, autocast_dtype_override: torch.dtype | None = None):
        """Pick the right grad/autocast context.

        - With autocast: ``torch.amp.autocast(...)`` + ``torch.no_grad()``.
        - Without autocast (audio codec): ``torch.inference_mode()`` — stricter
          than no_grad and matches the reference codec implementation.

        ``autocast_dtype_override`` (set when the current node's submodule
        declares ``get_autocast_dtype()``) wins over ``self.config.autocast_dtype``.
        ``None`` falls back to the engine-level default.
        """
        dtype = autocast_dtype_override if autocast_dtype_override is not None \
            else self.config.autocast_dtype
        if dtype is not None:
            return _ComposedContext(
                torch.amp.autocast(
                    "cuda", enabled=True, dtype=dtype
                ),
                torch.no_grad(),
            )
        return torch.inference_mode()

    def _prepare_inputs(
        self, batch: NodeBatch, submodule: NodeSubmodule
    ) -> tuple[list[NodeInputs], set[str]]:
        """Call ``prepare_inputs`` per rid; ``None`` means skip this rid."""
        if self.enable_nvtx:
            range_push(f"engine.{self.config.name}.prepare_inputs", synchronize=False)
        try:
            node_inputs: list[NodeInputs] = []
            skipped_rids: set[str] = set()
            for rid in batch.request_ids:
                req_inputs = submodule.prepare_inputs(
                    graph_walk=batch.graph_walk,
                    fwd_info=batch.per_request_info[rid],
                    inputs=batch.per_request_input_tensors[rid],
                )
                if req_inputs is None:
                    skipped_rids.add(rid)
                else:
                    node_inputs.append(req_inputs)
            return node_inputs, skipped_rids
        finally:
            if self.enable_nvtx:
                range_pop(synchronize=False)

    def _dispatch(
        self,
        batch: NodeBatch,
        inputs: list[NodeInputs],
        submodule: NodeSubmodule,
    ) -> NodeOutput:
        """Pick CUDA-graph replay / batched forward / sequential."""
        can_batch = submodule.can_batch(batch, inputs)
        runner = self.cuda_graph_runners.get(batch.node_name)
        if can_batch and runner is not None and self._can_use_cuda_graph(
            batch, submodule, inputs, runner
        ):
            return self._execute_with_cuda_graph(batch, submodule, inputs, runner)
        if can_batch:
            return self._execute_batched(batch, inputs, submodule)
        return self._execute_sequential(batch, inputs, submodule)

    def _can_use_cuda_graph(
        self,
        batch: NodeBatch,
        submodule: NodeSubmodule,
        inputs: list[NodeInputs],
        runner: StatelessCudaGraphRunner,
    ) -> bool:
        bs = len(batch.request_ids)
        # Audio codec submodules export an extra veto (e.g. SNAC's frame-count
        # check). Encoder/decoder submodules don't — absence means "no veto".
        can_use = getattr(submodule, "can_use_cuda_graphs", None)
        if can_use is not None and not can_use(batch, inputs):
            self._log_graph_miss(
                batch, bs, runner, "submodule.can_use_cuda_graphs() returned False"
            )
            return False
        if not runner.can_run(batch_size=bs, graph_walk=batch.graph_walk):
            self._log_graph_miss(
                batch, bs, runner, "no captured graph matches this (bs, graph_walk)"
            )
            return False
        return True

    def _log_graph_miss(
        self,
        batch: NodeBatch,
        bs: int,
        runner: StatelessCudaGraphRunner,
        reason: str,
    ) -> None:
        if not runner.graphs:
            return
        miss_key = (batch.node_name, batch.graph_walk, bs, reason)
        if miss_key in self._logged_graph_misses:
            return
        self._logged_graph_misses.add(miss_key)
        captured_for_walk = sorted(
            {key[1] for key in runner.graphs.keys() if key[0] == batch.graph_walk}
        )
        captured_walks = sorted({key[0] for key in runner.graphs.keys()})
        logger.warning(
            "[cuda-graph miss] node=%s graph_walk=%s requested=(bs=%d) "
            "reason='%s' captured_bs_for_walk=%s captured_walks=%s — falling back to eager.",
            batch.node_name,
            batch.graph_walk,
            bs,
            reason,
            captured_for_walk or "<none>",
            captured_walks,
        )

    def _execute_with_cuda_graph(
        self,
        batch: NodeBatch,
        submodule: NodeSubmodule,
        inputs: list[ARNodeInputs],
        runner: StatelessCudaGraphRunner,
    ) -> NodeOutput:
        if self.enable_nvtx:
            range_push(f"{self.config.name}.cuda_graph.run")
        per_rid = runner.run(
            graph_walk=batch.graph_walk,
            request_ids=batch.request_ids,
            inputs=inputs,
            per_request_info=batch.per_request_info,
            submodule=submodule,
            launch_started_event=batch.metadata.get("launch_started_event"),
            exec_timings=batch.exec_timings if self.enable_profile else None,
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)
        return NodeOutput(per_request_output_tensors=per_rid)

    def _execute_batched(
        self,
        batch: NodeBatch,
        inputs: list[NodeInputs],
        submodule: NodeSubmodule,
    ) -> NodeOutput:
        engine_inputs = ModelInputsFromEngine(
            request_ids=batch.request_ids,
            per_request_info=batch.per_request_info,
        )

        if self.enable_nvtx:
            range_push(f"{self.config.name}.batched.preprocess", synchronize=False)
        preprocessed = submodule.preprocess(
            graph_walk=batch.graph_walk,
            engine_inputs=engine_inputs,
            inputs=inputs,
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)

        if self.enable_nvtx:
            range_push(f"{self.config.name}.batched.forward")
        # Signal main thread we're about to enter CUDA launch code so it can
        # resume Python-heavy postprocess in parallel — PyTorch drops the GIL
        # inside the C++ kernel-launch path.
        launch_started_event = batch.metadata.get("launch_started_event")
        if launch_started_event is not None:
            launch_started_event.set()
        if self.enable_profile:
            batch.exec_timings.fwd_start = time.perf_counter()
        outputs = submodule.forward_batched(
            graph_walk=batch.graph_walk,
            engine_inputs=engine_inputs,
            **preprocessed,
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)
        return NodeOutput(per_request_output_tensors=outputs)

    def _execute_sequential(
        self,
        batch: NodeBatch,
        inputs: list[NodeInputs],
        submodule: NodeSubmodule,
    ) -> NodeOutput:
        outputs: dict[str, dict] = {}
        for i, rid in enumerate(batch.request_ids):
            node_input = inputs[i]
            fwd_info = batch.per_request_info[rid]
            engine_inputs = ModelInputsFromEngine(
                request_ids=[rid],
                per_request_info={rid: fwd_info},
            )

            if self.enable_nvtx:
                range_push(f"{self.config.name}.preprocess.{i}", synchronize=False)
            preprocessed = submodule.preprocess(
                graph_walk=batch.graph_walk,
                engine_inputs=engine_inputs,
                inputs=[node_input],
            )
            if self.enable_nvtx:
                range_pop(synchronize=False)

            if self.enable_nvtx:
                range_push(f"{self.config.name}.forward.{i}")
            launch_started_event = batch.metadata.get("launch_started_event")
            if launch_started_event is not None:
                launch_started_event.set()
            # Batch-level fwd_start: stamp once, on the first rid's launch.
            if self.enable_profile and batch.exec_timings.fwd_start is None:
                batch.exec_timings.fwd_start = time.perf_counter()
            outputs[rid] = submodule.forward(
                graph_walk=batch.graph_walk,
                engine_inputs=engine_inputs,
                **preprocessed,
            )
            if self.enable_nvtx:
                range_pop(synchronize=False)
        return NodeOutput(per_request_output_tensors=outputs)

    # ─── Warmup ────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Apply ``torch.compile`` and capture optional CUDA graphs.

        Each step is opt-in by the submodule (via ``get_cuda_graph_configs``
        and ``get_piecewise_runner_config``) so submodules that don't support
        a feature are skipped without error.
        """
        if not torch.cuda.is_available() or self.device is None:
            return

        for node_name, submodule in self.submodules.items():
            if self.config.apply_torch_compile:
                self._apply_torch_compile(node_name, submodule)
            if self.config.cuda_graph_capable:
                self._capture_codec_graphs(node_name, submodule)
            if self.config.enable_piecewise_runner:
                self._install_piecewise_runner(node_name, submodule)

    def _apply_torch_compile(self, node_name: str, submodule: NodeSubmodule) -> None:
        try:
            if hasattr(submodule, "forward"):
                submodule.forward = torch.compile(
                    submodule.forward,
                    fullgraph=False,
                    dynamic=False,
                )
                logger.info(
                    "StatelessEngine[%s]: torch.compile applied to %s.forward",
                    self.config.name,
                    node_name,
                )
            # forward_batched is intentionally left eager — Inductor would pay
            # a ~30s one-shot trace cost for dynamic varlen shapes on the
            # first call, which dwarfs the per-call win.
        except Exception:
            logger.warning(
                "StatelessEngine[%s]: torch.compile failed for %s, using eager mode",
                self.config.name,
                node_name,
                exc_info=True,
            )

    def _capture_codec_graphs(self, node_name: str, submodule: NodeSubmodule) -> None:
        if not hasattr(submodule, "get_cuda_graph_configs"):
            return
        codec_configs = submodule.get_cuda_graph_configs(self.device)
        if not codec_configs:
            return
        try:
            runner = StatelessCudaGraphRunner(
                submodule_name=node_name,
                submodule=submodule,
                device=self.device,
                tp_group=self.tp_groups.get_tp_config_for_node(node_name)
            )
            runner.enable_nvtx = self.enable_nvtx
            runner.warmup_and_capture()
            if runner.graphs:
                self.cuda_graph_runners[node_name] = runner
                logger.info(
                    "StatelessEngine[%s]: StatelessCudaGraphRunner installed for %s (%d graphs)",
                    self.config.name,
                    node_name,
                    len(runner.graphs),
                )
        except Exception:
            logger.warning(
                "StatelessEngine[%s]: StatelessCudaGraphRunner capture failed for %s, using eager mode",
                self.config.name,
                node_name,
                exc_info=True,
            )

    def _install_piecewise_runner(self, node_name: str, submodule: NodeSubmodule) -> None:
        getter = getattr(submodule, "get_piecewise_runner_config", None)
        if getter is None:
            return
        pcgr_config = getter()
        if pcgr_config is None:
            return
        try:
            from mstar.engine.cuda_graph_runner import (
                DEFAULT_AR_CAPTURE_BATCH_SIZES,
                PiecewiseCudaGraphRunner,
            )

            pcgr = PiecewiseCudaGraphRunner(
                fn_factory=pcgr_config["fn_factory"],
                embed_dim=pcgr_config["embed_dim"],
                capture_batch_sizes=pcgr_config.get(
                    "capture_batch_sizes", DEFAULT_AR_CAPTURE_BATCH_SIZES
                ),
                capture_seq_len=pcgr_config["capture_seq_len"],
                device=self.device,
                autocast_dtype=self.config.autocast_dtype,
                pos_buf_shapes=pcgr_config.get("pos_buf_shapes"),
                kv_cache_config=None,
                alloc_manager=None,
                buffer_manager=None,
                tp_group=self.tp_groups.get_tp_config_for_node(node_name),
            )
            pcgr.warmup_and_capture()
            if pcgr.graphs:
                submodule.set_piecewise_runner(pcgr)
                logger.info(
                    "StatelessEngine[%s]: PiecewiseCudaGraphRunner installed for %s (%d bs buckets)",
                    self.config.name,
                    node_name,
                    len(pcgr.graphs),
                )
        except Exception:
            logger.warning(
                "StatelessEngine[%s]: PiecewiseCudaGraphRunner capture failed for %s, using eager mode",
                self.config.name,
                node_name,
                exc_info=True,
            )


class _ComposedContext:
    """Stacks two context managers as a single ``with`` target.

    Used so ``_inference_context`` can return a single object that enters
    both autocast and no_grad in order.
    """

    def __init__(self, outer, inner):
        self._outer = outer
        self._inner = inner

    def __enter__(self):
        self._outer.__enter__()
        try:
            return self._inner.__enter__()
        except Exception:
            self._outer.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self._inner.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._outer.__exit__(exc_type, exc_val, exc_tb)
