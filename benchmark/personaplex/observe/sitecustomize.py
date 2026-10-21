"""Opt-in observational hooks for CPU tensor ownership in actual server workers."""

import json
import os
import time
from pathlib import Path

if os.environ.get("PERSONAPLEX_OWNERSHIP_OBSERVE") == "1":
    import torch

    from mstar.communication.tensors import TensorCommunicationManager
    from mstar.graph.runtime.python import PythonGraphRuntime

    root = Path(os.environ["PERSONAPLEX_OWNERSHIP_ROOT"])

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
        original_collect = TensorCommunicationManager.cleanup_collectable

        def collected(self, collectable, registered):
            result = original_collect(self, collectable, registered)
            rids = set(pending_decoder_rids)
            pending_decoder_rids.clear()
            for rid in rids:
                snapshot(self, "after_decoder_inputs_consumed", rid)
            return result

        TensorCommunicationManager.cleanup_collectable = collected

    original_force = TensorCommunicationManager.force_cleanup_request

    def forced(self, request_id):
        snapshot(self, "before_hard_cleanup", request_id)
        result = original_force(self, request_id)
        snapshot(self, "after_hard_cleanup", request_id)
        return result

    TensorCommunicationManager.force_cleanup_request = forced
