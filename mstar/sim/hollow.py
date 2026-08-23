"""Hollow execution: the real serving stack with a fake engine.

Everything above the engine runs for real — the conductor, the workers, the
micro-scheduler, graph routing, streaming buffers, ZMQ — while
:class:`SimEngine` stands in for the GPU, returning correctly-shaped tensors
after the delay the stepdb says the step would have taken.

What this is for:

* **A drift gate.** The DES re-implements the worker's pipeline; hollow mode
  does not. Running both on the same workload and comparing per-(node, walk)
  step counts catches a scheduling divergence introduced by either side —
  the V1 gate in :mod:`mstar.sim.validate`.
* **A no-GPU smoke test** of conductor/worker/graph changes, which the repo
  lost when the DummyModel path was deleted.

What this is *not* for: measuring CPU overhead. SimEngine replaces the engine
wholesale, so ``prepare_inputs``, the FlashInfer plan, sampling, and the
CUDA-graph replay never execute — precisely the CPU terms that matter most.
Those come from instrumented real-GPU runs. Hollow mode runs in wall-clock
time, so it is also not a fast predictor; that is the DES's job.

Enable with ``MSTAR_HOLLOW=1`` (plus ``MSTAR_HOLLOW_DB`` to charge measured
delays rather than a flat default).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

from mstar.engine.base import (
    BaseEngine,
    EngineType,
    NodeBatch,
    NodeOutput,
    PlannedBatch,
    PreparedBatch,
    StopCheckResult,
)

logger = logging.getLogger(__name__)

ENV_ENABLE = "MSTAR_HOLLOW"
ENV_DB = "MSTAR_HOLLOW_DB"
ENV_GPU = "MSTAR_HOLLOW_GPU"
#: Delay charged when no stepdb row matches, so a hollow run still advances.
DEFAULT_STEP_S = 0.002


def hollow_enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "").lower() in ("1", "true", "yes")


class SimEngine(BaseEngine):
    """An engine that produces plausible tensors instead of running a model.

    Two contracts have to be honored or the worker misbehaves rather than
    simply running fast:

    * outputs must be real tensors — the worker stores them and counts
      tokens with ``tensor.numel()``;
    * stopping must be decided here, because the real stop check reads
      sampled token values that do not exist in a hollow run. The request's
      ``max_tokens`` is used instead, which is what a benchmark run with a
      pinned length produces anyway.
    """

    def __init__(
        self,
        autocast_dtype=None,
        enable_nvtx: bool = False,
        enable_profile: bool = False,
        model: Any = None,
        stepdb_path: str | None = None,
        gpu_name: str | None = None,
        **kwargs,
    ):
        super().__init__(enable_nvtx=enable_nvtx, enable_profile=enable_profile)
        self.autocast_dtype = autocast_dtype
        self.device = torch.device("cpu")
        self.model = model
        self._requests: dict[str, dict] = {}
        self._steps: dict[str, int] = {}

        # node -> output edge names, read from the model's own graph so the
        # fabricated outputs match what the graph will route.
        self._node_outputs: dict[str, list[str]] = {}
        if model is not None:
            self._index_graph_outputs(model)

        # Hollow mode runs on CPU but emulates a GPU deployment, so rows are
        # looked up under the *target* device, not this host's. With one
        # device in the table the choice is unambiguous; otherwise name it
        # explicitly, because silently picking one would emulate the wrong
        # hardware without saying so.
        self._db = None
        self._model_key = ""
        path = stepdb_path or os.environ.get(ENV_DB)
        if path and os.path.exists(path):
            from mstar.sim.stepdb import StepDB
            probe = StepDB(path)
            names = probe.gpu_names()
            probe.close()
            target = gpu_name or os.environ.get(ENV_GPU)
            if target is None and len(names) == 1:
                target = names[0]
            elif target is None and len(names) > 1:
                logger.warning(
                    "stepdb %s holds rows for %s; set %s to choose one. "
                    "Falling back to a flat %.0f ms per step.",
                    path, ", ".join(names), ENV_GPU, DEFAULT_STEP_S * 1e3,
                )
            if target is not None:
                self._db = StepDB(path, gpu_name=target)
                models = self._db.models()
                self._model_key = models[0] if models else ""

    def _index_graph_outputs(self, model: Any) -> None:
        try:
            walks = model.get_graph_walk_graphs()
        except Exception:
            return
        for section in walks.values():
            try:
                nodes = section.get_nodes()
            except Exception:
                continue
            for name, node in nodes.items():
                names = [
                    e.name for e in (getattr(node, "outputs", []) or [])
                ]
                self._node_outputs.setdefault(name, [])
                for n in names:
                    if n not in self._node_outputs[name]:
                        self._node_outputs[name].append(n)

    # ── BaseEngine surface ───────────────────────────────────────────────

    def engine_type(self) -> EngineType:
        return EngineType.KV_CACHE

    def has_autocast(self) -> bool:
        return False

    def load_model(self, submodules, parallel_groups, kv_cache_config, device, **kwargs):
        self.device = device
        logger.info(
            "SimEngine active on %s for nodes %s — no real computation will run",
            device, sorted(submodules) or "(none)",
        )

    def add_request(self, request_id: str, **kwargs) -> None:
        self._requests[request_id] = dict(kwargs)
        self._steps[request_id] = 0

    def remove_request(self, request_id: str) -> None:
        self._requests.pop(request_id, None)
        self._steps.pop(request_id, None)

    def prepare_batch(self, batch: NodeBatch) -> PreparedBatch:
        return PreparedBatch(batch=batch, submodule=None)

    def execute_forward(self, planned: PlannedBatch) -> NodeOutput:
        """Satisfies the ABC; unused because ``execute_batch`` is overridden.

        The template's four hooks exist so an engine can plug into the
        standard flow. A hollow engine has no flow to plug into — there is
        nothing to prepare, plan, or postprocess — so it replaces
        ``execute_batch`` outright and routes here only if something calls
        the hook directly.
        """
        return self.execute_batch(planned.batch)

    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        """Charge the modeled delay, then emit one token per request."""
        delay = self._delay_for(batch)
        if delay > 0:
            time.sleep(delay)

        outputs: dict[str, dict[str, list[torch.Tensor]]] = {}
        names = self._node_outputs.get(batch.node_name) or ["output"]
        for rid in batch.request_ids:
            self._steps[rid] = self._steps.get(rid, 0) + 1
            # One token per step: a [1] int tensor is what the worker's token
            # accounting (numel) and the graph's routing both expect.
            outputs[rid] = {
                name: [torch.zeros(1, dtype=torch.long, device=self.device)]
                for name in names
            }
        return NodeOutput(per_request_output_tensors=outputs)

    def _delay_for(self, batch: NodeBatch) -> float:
        if self._db is None:
            return DEFAULT_STEP_S
        from mstar.sim.stepdb import Coverage, StepKey

        bs = len(batch.request_ids)
        cost = self._db.lookup(
            StepKey(
                model=self._model_key, node=batch.node_name,
                graph_walk=batch.graph_walk, padded_bs=bs, padded_num_tokens=bs,
            ),
            0,
        )
        if cost.coverage & Coverage.MISSING:
            return DEFAULT_STEP_S
        return max(cost.gpu_s, cost.cpu_s)

    def check_stop_for_batch(
        self, batch: NodeBatch, output: NodeOutput
    ) -> StopCheckResult:
        """Stop at the request's token budget.

        The real check reads sampled token values for EOS; there are none
        here, so the budget stands in. A request without a budget would run
        to the loop's max_iters, which is the honest behavior rather than
        stopping arbitrarily.
        """
        stops: dict[str, set[str]] = {}
        for rid in batch.request_ids:
            info = batch.per_request_info.get(rid)
            budget = getattr(info, "max_tokens", None) if info else None
            if budget and self._steps.get(rid, 0) >= budget:
                loops = self._loop_names_for(batch.graph_walk)
                if loops:
                    stops[rid] = loops
        return StopCheckResult(stops=stops)

    def _loop_names_for(self, graph_walk: str) -> set[str]:
        if self.model is None:
            return set()
        try:
            section = self.model.get_graph_walk_graphs().get(graph_walk)
            return set(section.get_loops().keys()) if section else set()
        except Exception:
            return set()

    def warmup(self) -> None:
        return

    def shutdown(self) -> None:
        if self._db is not None:
            self._db.close()


def install(model: Any = None) -> None:
    """Point every engine factory at :class:`SimEngine`.

    Called by ``EngineManager.build`` when hollow mode is on. Replacing the
    factories (rather than the engine classes) keeps the substitution at the
    one place the manager already consults, so nothing else in the worker
    needs to know.
    """
    from mstar.worker import engine_manager as em

    def factory(autocast_dtype, enable_nvtx, enable_prof):
        return SimEngine(
            autocast_dtype=autocast_dtype,
            enable_nvtx=enable_nvtx,
            enable_profile=enable_prof,
            model=model,
        )


    em.ENGINE_TYPE_FACTORIES = {k: factory for k in em.ENGINE_TYPE_FACTORIES}
    em.STATELESS_FLAVOR_FACTORIES = {
        k: factory for k in em.STATELESS_FLAVOR_FACTORIES
    }
    logger.warning(
        "MSTAR_HOLLOW is set — engines are hollow, outputs are synthetic, "
        "and no GPU computation will run"
    )
