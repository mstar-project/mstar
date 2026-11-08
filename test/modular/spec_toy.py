"""A toy target and scripted drafters for speculative decoding tests, on the
real `Engine` with real KV, position and sampler resources on the CPU.

The target's next token depends on its input token, its position, and the sum
of every token in its KV cache up to itself (read back through the KV
manager's pages), so a missed KV or position take-back changes what it emits.
The drafters are in-node test fixtures: an oracle that drafts the target's own
greedy continuation, one that drafts every token wrong, and one that drafts
every ``m``-th wrong.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources.draft.config import DraftConfig
from mstar.engine.resources.draft.drafter import (
    ChunkBoundary,
    DrafterNeeds,
    DraftInputs,
    VerifyNode,
)
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.model.base import KVLabel, PrefixKeys
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    DeviceLoopback,
    HostRows,
    TokenStop,
)

NODE = "LLM"
VOCAB = 64
PAGE = 4
STOP = 63
HEAD_DIM = 8


def next_token(token: torch.Tensor, position: torch.Tensor, kv_sum: torch.Tensor) -> torch.Tensor:
    return (3 * token + 5 * position + kv_sum) % VOCAB


def reference(prompt: list[int], max_tokens: int) -> list[int]:
    """Plain greedy decoding of the toy target, with no engine."""
    seq = list(prompt)
    out = []
    while len(out) < max_tokens:
        token = int(next_token(torch.tensor(seq[-1]), torch.tensor(len(seq) - 1), torch.tensor(sum(seq))))
        out.append(token)
        seq.append(token)
        if token == STOP:
            break
    return out


class _Drafter:
    """Drafts the target's greedy continuation from the KV cache, then spoils
    the drafts ``spoil`` says to."""

    kv_layers_unbound = False

    def __init__(self, k: int, spoil=lambda j: False):
        self.k = k
        self.spoil = spoil
        self.needs = DrafterNeeds(chunk_boundary=ChunkBoundary.NONE)

    def propose(self, inputs: DraftInputs) -> torch.Tensor:
        n, k = inputs.input_ids.shape[0], self.k
        rope, kv = inputs.resources["rope"], inputs.resources["kv"]
        position = rope.pos_ids("main").view(n, k + 1)[:, 0]
        kv_sum = _context_sums(kv, n) + inputs.input_ids
        token = inputs.input_ids
        drafts = []
        for j in range(k):
            token = next_token(token, position + j, kv_sum)
            kv_sum = kv_sum + token
            drafts.append((token + 1) % VOCAB if self.spoil(j) else token)
        return torch.stack(drafts, dim=1)


def oracle(k):
    return _Drafter(k)


def always_wrong(k):
    return _Drafter(k, spoil=lambda j: True)


def every(m):
    return lambda k: _Drafter(k, spoil=lambda j: j % m == m - 1)


def _views(kv):
    return kv._current_plan_outputs_for_toy


def _context_sums(kv, n: int) -> torch.Tensor:
    """Each row's sum of the tokens its KV cache held before this step."""
    sums = []
    for view in _views(kv)[:n]:
        before = view.length - view.to_compute
        sums.append(_read_ids(kv, view, before).sum() if before else torch.tensor(0))
    return torch.stack(sums).long()


def _read_ids(kv, view, length: int) -> torch.Tensor:
    pages = torch.tensor([view.page_idxs[p // PAGE] for p in range(length)])
    offsets = torch.tensor([p % PAGE for p in range(length)])
    tokens = kv.kv_cache.read_tokens(layer_idx=0, page_idx=pages, cache_idx=offsets)
    return tokens[:, 0, 0, 0].round().long()


class ToyLM(ARNodeSubmodule):
    """A prompt walk that samples one token, and a decode walk that verifies."""

    def __init__(self, k: int, drafter_factory=oracle):
        super().__init__()
        self.k = k
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.token_stop = TokenStop(
            loop="decode_loop", buffer="new_token", stop_ids=frozenset({STOP}), sampler="sampler",
            walk="decode", prefill_walks=frozenset({"prefill"}),
        )
        self.drafter = drafter_factory(k)
        # the verify walk reads each row's last token off the sampler's loop-back slot
        self.device_loopback = DeviceLoopback(walk="decode", signal="ids", sampler="sampler", buffer="new_token")

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        ids = inputs["ids"][0]
        if graph_walk == "prefill":
            # a sequence the prefix cache can cut
            return ARNodeInputs(input_ids=ids, input_seq_len=ids.numel())
        # a row's first verify step brings its prompt's token routed; every later one
        # reads it off the loop-back slot (`uniform_row_inputs`)
        if kwargs.get("has_loopback"):
            return self.uniform_row_inputs(graph_walk, has_loopback=True)
        return ARNodeInputs(input_ids=ids.reshape(-1)[-1:], input_seq_len=self.k + 1)

    def can_batch(self, batch, model_inputs):
        return True

    def resident_walks(self):
        return frozenset({"decode"})

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs):
        segments = tuple(Segment(rid, "main", inp.input_seq_len) for rid, inp in zip(request_ids, inputs, strict=True))
        one = tuple(Segment(rid, "main", 1) for rid in request_ids)
        return SubmoduleStep(
            steps={
                "kv": KVStep(segments=segments), "rope": PositionStep(segments=segments),
                "sampler": SamplerStep(segments=one, apply_penalty=False),
            },
            segments=list(segments),
        )

    def preprocess(self, graph_walk, engine_inputs, inputs):
        if graph_walk == "prefill":
            return {"ids": torch.cat([inp.input_ids.reshape(-1) for inp in inputs])}
        return {"ids": self.loopback_inputs(engine_inputs, inputs)["input_ids"]}

    def _target(self, engine_inputs, ids: torch.Tensor) -> torch.Tensor:
        """Write ``ids`` to the KV cache, then each row's logits: a one-hot at
        its next token, which reads the cache up to and including itself."""
        kv, rope = engine_inputs.resources["kv"], engine_inputs.resources["rope"]
        positions = rope.pos_ids("main")
        values = ids.float()[:, None, None].expand(-1, 1, HEAD_DIM)
        kv.write_kv(values, values)
        sums = []
        for view in _views(kv):
            seen = _read_ids(kv, view, view.length)
            before = view.length - view.to_compute
            sums.append(seen.cumsum(0)[before:])
        token = next_token(ids, positions[: ids.numel()], torch.cat(sums))
        return torch.nn.functional.one_hot(token, VOCAB).float() * 20.0

    def forward(self, graph_walk, engine_inputs, ids, **kwargs):
        return self.forward_batched(graph_walk, engine_inputs, ids, **kwargs)

    def forward_batched(self, graph_walk, engine_inputs, ids, **kwargs):
        rids = engine_inputs.request_ids
        kv = engine_inputs.resources["kv"]
        kv._current_plan_outputs_for_toy = engine_inputs.step.ctx.plan_results["kv"]["main"].views
        if graph_walk == "prefill":
            logits = self._target(engine_inputs, ids)
            ends = torch.tensor([v.to_compute for v in _views(kv)]).cumsum(0) - 1
            # main's sampler draws on an accelerator only; the prompt step is greedy here
            tokens = logits[ends].argmax(-1)
            return BatchedModelOutput(
                per_rid_outputs={rid: {"new_token": [tokens[i: i + 1]]} for i, rid in enumerate(rids)},
            )
        n = len(rids)
        result = engine_inputs.draft_verify_step(
            ids.view(n), lambda block: self._target(engine_inputs, block), self.drafter,
        )
        return engine_inputs.verify_output(result)


class ToyModel:
    """What `Engine.load_model` and validation read of a model."""

    def __init__(self, k: int, mode: str = "synchronous", stop=None, needs: DrafterNeeds | None = None):
        self.k = k
        self.needs = needs or DrafterNeeds(chunk_boundary=ChunkBoundary.NONE)
        self.config = DraftConfig.parse({"method": "dspark", "model": "toy", "num_draft_tokens": k, "mode": mode})
        self._stop = stop

    def verify_nodes(self):
        loopback = DeviceLoopback(walk="decode", signal="ids", sampler="sampler", buffer="new_token")
        return [VerifyNode(NODE, "decode", self.config, self.needs, token_stop=self._stop, device_loopback=loopback)]

    def checkpoint_path(self):
        return None

    def preprocess_fingerprint(self):
        return ""

    def get_graph_walk_graphs(self):
        return {walk: SimpleNamespace(get_nodes=lambda: [NODE]) for walk in ("prefill", "decode")}

    def kv_labels(self):
        # the prompt keys its pages; the decode walk, which verifies, keys nothing
        return {"kv": {"main": KVLabel(
            growth=(("decode", "decode_loop"),),
            keys=PrefixKeys(tensor="ids", walks=("prefill", "decode"), growth=True),
        )}}

    def chunking_policy_of(self):
        return {}

    def rebuild_policies(self):
        # no node rebuilds a dropped request (`Engine._check_rebuild_policies`)
        return {}

    def __getattr__(self, name):
        # anything else the engine asks of a model: nothing declared
        if name.startswith("get_") or name.endswith("_of"):
            return lambda *a, **kw: {}
        raise AttributeError(name)


def specs(k: int, max_num_pages: int = 256):
    return [
        KVSpec("kv", {NODE}, draft_slack=2 * (k + 1), config=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=HEAD_DIM, max_seq_len=256 + 2 * (k + 1),
            max_num_pages=max_num_pages, page_size=PAGE, prefix_cache_salt="toy",
        )),
        PositionSpec("rope", {NODE}, PositionConfig(kv_cache="kv")),
        SamplerSpec("sampler", {NODE}, vocab_size=VOCAB, enable_repetion_penalty=False, verifies=True),
    ]


def engine(k: int, drafter_factory=oracle, mode: str = "synchronous") -> Engine:
    sub = ToyLM(k, drafter_factory)
    eng = Engine(graph_runtime=None)
    eng.load_model(
        {NODE: sub}, specs(k), WorkerParallelGroups(num_workers=1, global_rank=0),
        torch.device("cpu"), transfer_engine_info=None, kv_cache_type=torch.float32,
        model=ToyModel(k, mode, sub.token_stop, sub.drafter.needs),
    )
    # what a capture's buffers would build: the sampler's slot masters, the loop-back slots
    eng._resources["sampler"].build_accelerator_graph_buffers([], 8, 256)
    return eng


def _info(rid, walk, max_tokens):
    return SimpleNamespace(
        rid=rid, graph_walk=walk, max_tokens=max_tokens, step_metadata={},
        resource_configs={"sampler": SamplingReqConfig(temperature=0)},
        update_publish_info=lambda published: None,
    )


def _batch(walk, ids: dict[int, torch.Tensor], max_tokens: dict[int, int]) -> ExecutingBatch:
    rids = list(ids)
    return ExecutingBatch(
        node_name=NODE,
        per_request_info={rid: _info(rid, walk, max_tokens[rid]) for rid in rids},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
        per_request_input_tensors={rid: {"ids": [row]} for rid, row in ids.items()},
    )


class Driver:
    """Plays the worker: a prompt step, then verify steps, each cut on the
    host. Synchronous: a step is built once the last one was post-processed."""

    def __init__(self, eng: Engine):
        self.eng = eng
        self.out: dict[int, list[int]] = {}

    def add(self, rid: int, prompt: list[int], max_tokens: int, temperature: float = 0.0, seed: int = 0):
        cfg = SamplingReqConfig(temperature=temperature)
        cfg.apply_conductor_config(seed=seed)
        self.eng.add_request(rid, {"kv": KVReqConfig(), "sampler": cfg})
        _take_sampler_row(self.eng, rid)
        self.out[rid] = []
        self._max = getattr(self, "_max", {})
        self._max[rid] = max_tokens
        batch = _batch("prefill", {rid: torch.tensor(prompt)}, self._max)
        self.eng.prepare_inputs(batch)
        outputs = self.eng.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        token = int(outputs.per_rid_outputs[rid]["new_token"][0][0])
        self.out[rid].append(token)
        return token

    def step(self, rids: list[int]) -> dict[int, bool]:
        """One verify step over ``rids``; the rows that stopped."""
        ids = {rid: torch.tensor([self.out[rid][-1]] * 1) for rid in rids}
        batch = _batch("decode", ids, self._max)
        for rid in rids:
            batch.per_request_info[rid].graph_walk = "decode"
        self.eng.prepare_inputs(batch)
        outputs = self.eng.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        host = HostRows(tuple(outputs.row_request_ids or rids), {
            name: t.cpu() for name, t in (outputs.check_stop_buffers or {}).items()
        })
        self.eng.cut_verify_step(batch, outputs, host)
        stopped = {}
        stop_col = host.buffers["verify_stopped"].tolist()
        for i, rid in enumerate(host.request_ids):
            run = outputs.per_rid_outputs[rid]["new_token"][0].reshape(-1).tolist()
            self.out[rid] += run
            stopped[rid] = bool(stop_col[i])
        return stopped

    def run(self, rids: list[int], max_steps: int = 200) -> dict[int, list[int]]:
        live = [rid for rid in rids if self.out[rid][-1] != STOP and len(self.out[rid]) < self._max[rid]]
        for _ in range(max_steps):
            if not live:
                break
            stopped = self.step(live)
            live = [rid for rid in live if not stopped[rid]]
        for rid in rids:
            if self.out[rid] and len(self.out[rid]) >= self._max[rid] or self.out[rid][-1] == STOP:
                self.eng.remove_request(rid)
        return self.out


def _views_of(kv, rid):
    """A stream's pages, as `_read_ids` reads a plan's view."""
    return SimpleNamespace(page_idxs=list(kv._streams[rid]["main"].page_indices))


def _take_sampler_row(eng, rid) -> None:
    """The sampler's admission (`SamplerResource.reserve`), as the scheduler asks it."""
    sampler = eng._resources["sampler"]
    if sampler.ask(rid, NODE, "prefill") is not None:
        sampler.reserve(rid, NODE, "prefill", None)
