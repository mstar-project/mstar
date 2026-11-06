"""The LM's frames driven without a server: over dense attention, or the served resources (paged KV,
FlashInfer attention, RoPE, samplers and the token cache pool)."""

import socket
from types import SimpleNamespace

import torch

from mstar.model.personaplex import config as cfg
from mstar.model.personaplex.submodules import read_cache_rows, write_cache_rows


class DenseFrames:
    """One request's frames over dense attention (or any backend with ``advance``), holding its cache itself."""

    def __init__(self, sub, backend):
        self.sub, self.backend, self.rows = sub, backend, None

    def conditioning(self, frames):
        (frame,) = frames
        self.sub.lm.temporal.forward_embeddings(frame.embeds, self.backend, text_logits=False)
        self.backend.advance()
        self.rows = type(frame.rows)(*(field.to(self.sub.get_device()) for field in frame.rows))

    def generate(self, user, walk=cfg.GENERATE_WALK):
        del walk
        frame = self.sub.generate_frame(
            self.rows, user, self.backend, lambda x: x.argmax(-1), lambda x, _step: x.argmax(-1)
        )
        self.backend.advance()
        self.rows = frame.rows
        return frame


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

    def conditioning(self, frames, request_ids=None):
        """One conditioning step over ``frames``, one for each of ``request_ids``."""
        self.begin(cfg.CONDITIONING_WALK, request_ids)
        self.sub.lm.temporal.forward_embeddings(torch.cat([f.embeds for f in frames]), self.backend, text_logits=False)
        device = self.sub.get_device()
        rows = [torch.cat(parts).to(device) for parts in zip(*(f.rows for f in frames), strict=True)]
        write_cache_rows(self.resources[cfg.LM_CACHE], type(frames[0].rows)(*rows))
        self.finish()

    def generate(self, user, walk=cfg.GENERATE_WALK, request_ids=None, sample_text=None, sample_depth=None):
        """One generate step, reading and writing the rows' caches in the pool as the engine does."""
        self.begin(walk, request_ids)
        pool = self.resources[cfg.LM_CACHE]
        frame = self.sub.generate_frame(read_cache_rows(pool), user, self.backend,
                                        sample_text or self.sample_text, sample_depth or self.sample_depth)
        write_cache_rows(pool, frame.rows)
        self.finish()
        return frame

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
