"""The LM's served resources (paged KV, FlashInfer attention, RoPE, samplers), driven without a server."""

import socket
from types import SimpleNamespace

import torch

from mstar.model.personaplex import config as cfg


class ResourceBackend:
    """One step at a time over ``request_ids``, each admitted, planned and committed like the engine does."""

    def __init__(self, model, sub, request_ids=()):
        from mstar.communication.tensors import LocalTransferEngine
        from mstar.engine.resources.base import EngineResourceInfo, build_resource
        from mstar.engine.resources.kv.config import KVReqConfig
        from mstar.engine.resources.kv.transfer import TransferEngineInfo
        from mstar.engine.resources.runner import StepRunner

        specs = model.get_node_resources()
        by_key = {s.resource_key: s for s in specs}
        transfer = TransferEngineInfo(
            "numerical-oracle", socket.gethostname(), LocalTransferEngine(socket.gethostname())
        )
        self.resources = {
            s.resource_key: build_resource(
                s,
                EngineResourceInfo(
                    device=torch.device("cuda:0"),
                    kv_dtype=torch.bfloat16,
                    transfer_engine_info=transfer,
                    dependencies={k: by_key[k] for k in s.depends_on()},
                ),
            )
            for s in specs
        }
        self.runner = StepRunner(self.resources)
        self.request_ids, self.sub = [], sub
        self.overrides = model.get_request_resource_configs({}, {"greedy": True})
        self.overrides[cfg.LM_KV] = KVReqConfig(needed_labels=["main"], prefix_cache=False)
        for rid in request_ids:
            self.add(rid)
        sub.bind_node_resources(self.resources)
        self.backend = sub._backend
        self.step_rids = []

    def begin(self, walk, request_ids=None):
        """Admit and plan one frame for ``request_ids`` (default: every request), in that row order."""
        from mstar.engine.resources.step import StepContext

        self.step_rids = list(request_ids or self.request_ids)
        inputs = [SimpleNamespace(input_seq_len=1)] * len(self.step_rids)
        self.step = self.sub.declare_step(walk, self.step_rids, inputs)
        self.step.set_ctx(StepContext(self.step_rids, walk, slot=0, capture=False))
        outcome = self.runner.admit(self.step)
        assert outcome.ok and outcome.ready, outcome
        self.runner.plan(self.step)
        self.backend.begin()

    def finish(self):
        self.runner.commit(self.step)

    def add(self, rid):
        self.runner.ingest_request(rid, self.overrides)
        self.request_ids.append(rid)

    def remove(self, rid):
        self.runner.remove_request(rid)
        self.request_ids.remove(rid)

    def close(self):
        for rid in list(self.request_ids):
            self.remove(rid)

    def sample_text(self, logits):
        return self.resources[cfg.TEXT_SAMPLER].sample(self.step_rids, logits.float())

    def sample_depth(self, logits, index):
        return self.resources[cfg.DEPTH_SAMPLER].sample(self.step_rids, logits.float())
