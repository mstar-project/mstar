"""Opt-in hooks for harness-launched server workers: tensor ownership observation, an eager control, a replay trace."""

import json
import os
import time
from pathlib import Path


def track_request_ids(runtime_class, identities):
    """Record wire IDs when the runtime assigns its recyclable worker-local handle."""
    original = runtime_class.add_request

    def added(self, request_id, *args, **kwargs):
        handle = original(self, request_id, *args, **kwargs)
        identities[handle] = request_id
        return handle

    runtime_class.add_request = added


if os.environ.get("PERSONAPLEX_OWNERSHIP_OBSERVE") == "1":
    import torch

    from mstar.communication.tensors import TensorCommunicationManager
    from mstar.graph.runtime.python import PythonGraphRuntime

    root = Path(os.environ["PERSONAPLEX_OWNERSHIP_ROOT"])
    wire_ids = {}
    track_request_ids(PythonGraphRuntime, wire_ids)

    from mstar.api_server.entrypoint import APIServer

    original_submit = APIServer.submit_request

    def submitted(self, *args, **kwargs):
        request_id = original_submit(self, *args, **kwargs)
        row = {"event": "request_identity", "entity": "api_server", "time": time.time(),
               "request_id": request_id, "submission_key": self.submission_key(request_id)}
        with (root / ("ownership-events-" + str(os.getpid()) + ".jsonl")).open("a") as output:
            output.write(json.dumps(row) + "\n")
        return request_id

    APIServer.submit_request = submitted

    def snapshot(manager, event, rid=None, **extra):
        store = manager.tensor_store
        rows = []
        ids = store.get_all_uuids(rid) if rid is not None else list(store._uuid_to_rid)
        records = getattr(store.bookkeeping, "_ref_info", {})
        for uuid in ids:
            rec = records.get(uuid)
            info = store.get_info(uuid)
            rows.append(
                dict(
                    uuid=uuid,
                    name=manager.uuid_to_edge_name.get(uuid),
                    refs=None if rec is None else rec.ref_cnt,
                    persist=None if rec is None else rec.persist,
                    dims=None if info is None else info.dims,
                )
            )
        row = dict(
            event=event,
            pid=os.getpid(),
            time=time.time(),
            rid=rid,
            request_id=wire_ids.get(rid, rid if isinstance(rid, str) else None),
            entity=manager.my_entity_id,
            tensors=rows,
            total_tensors=len(store._uuid_to_rid),
            **extra,
        )
        if torch.cuda.is_initialized():
            row.update(cuda_allocated=torch.cuda.memory_allocated(), cuda_reserved=torch.cuda.memory_reserved())
        with (root / ("ownership-events-" + str(os.getpid()) + ".jsonl")).open("a") as out:
            out.write(json.dumps(row) + "\n")

    original_clean = PythonGraphRuntime.cleanup_consumed_inputs

    def consumed(self, node_name, rids, wg_ids):
        result = original_clean(self, node_name, rids, wg_ids)
        if node_name == "MimiDecoder":
            for rid in rids:
                snapshot(self._tensor_manager, "after_decoder_inputs_consumed", rid)
        return result

    PythonGraphRuntime.cleanup_consumed_inputs = consumed

    if os.environ.get("PERSONAPLEX_LIVE_BACKEND") == "rust":
        from mstar.graph.runtime.rust import RustGraphRuntime

        track_request_ids(RustGraphRuntime, wire_ids)
        pending_decoder_rids = set()
        rust_consumed = RustGraphRuntime.cleanup_consumed_inputs

        def rust_cleanup(self, node_name, rids, wg_ids):
            freed = rust_consumed(self, node_name, rids, wg_ids)
            if node_name == "MimiDecoder":
                # Arena/Rust bookkeeping may already remove UUID->rid entries.
                # The caller immediately passes these frees to cleanup_collectable
                # (worker.py); retain the known request IDs across that boundary.
                pending_decoder_rids.update(rids)
            return freed

        RustGraphRuntime.cleanup_consumed_inputs = rust_cleanup

        def observe_collectable(cls):
            original_collect = cls.__dict__["cleanup_collectable"]

            def collected(self, collectable, registered):
                result = original_collect(self, collectable, registered)
                rids = set(pending_decoder_rids)
                pending_decoder_rids.clear()
                for rid in rids:
                    snapshot(self, "after_decoder_inputs_consumed", rid)
                return result

            cls.cleanup_collectable = collected

        # The SHM arena manager overrides cleanup_collectable without calling
        # super(), so wrapping only the base class never fires under the arena.
        from mstar.communication.arena import ArenaShmCommunicationManager

        for cls in (TensorCommunicationManager, ArenaShmCommunicationManager):
            observe_collectable(cls)

    original_force = TensorCommunicationManager.force_cleanup_request

    def forced(self, request_id):
        snapshot(self, "before_hard_cleanup", request_id)
        result = original_force(self, request_id)
        snapshot(self, "after_hard_cleanup", request_id)
        wire_ids.pop(request_id, None)
        return result

    TensorCommunicationManager.force_cleanup_request = forced

    from mstar.engine.engine import Engine

    original_remove = Engine.remove_request

    def engine_removed(self, request_id):
        result = original_remove(self, request_id)
        row = {
            "event": "after_engine_cleanup", "entity": "worker_engine", "pid": os.getpid(),
            "time": time.time(), "rid": request_id, "request_id": wire_ids.get(request_id),
            "remaining_node_states": [name for name, management in self._submodules.items()
                                      if request_id in management.submodule.request_states],
            "reclaimable_pages": {name: resource.reclaimable(request_id)
                                  for name, resource in self._resources.items()},
        }
        with (root / ("ownership-events-" + str(os.getpid()) + ".jsonl")).open("a") as output:
            output.write(json.dumps(row) + "\n")
        return result

    Engine.remove_request = engine_removed

if os.environ.get("PERSONAPLEX_EAGER_CONTROL") == "1":
    # A control server for graph-vs-eager comparisons: the LM declares no
    # capture configs, so every step runs eager. Test tooling, not a product knob.
    from mstar.model.personaplex.submodules import LMSubmodule

    LMSubmodule.get_accelerator_graph_configs = lambda self, device, tp_world_size=1: []
    print("PERSONAPLEX_EAGER_CONTROL: LM capture configs suppressed", flush=True)

if os.environ.get("PERSONAPLEX_REPLAY_TRACE") == "1":
    # Diagnostic evidence that captured graphs actually replay under load (a capture
    # at startup does not prove it): count replays per node and log every 100th.
    from collections import Counter

    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner

    replays = Counter()
    original_replay = AcceleratorGraphRunner._replay

    def counted_replay(self, lease):
        replays[self._submodule_name] += 1
        if replays[self._submodule_name] % 100 == 1:
            print(f"PERSONAPLEX_REPLAY_TRACE {self._submodule_name} replays={replays[self._submodule_name]} "
                  f"bucket={lease.bucket}", flush=True)
        return original_replay(self, lease)

    AcceleratorGraphRunner._replay = counted_replay
