"""The sampler on persistent rows, through a real ``Engine``: each request's
draws, last token and loop-back state follow it across joins, leaves, steps
of another walk, removals and handles that come back as new requests.

Every step's draws are checked against the graph sampler run on the same
rows, each at the settings and RNG offset its request's step count implies,
so a row that read another request's state, or lost its own on a check-in,
draws differently. (FlashInfer draws every row from row 0's seed and offset,
so a request's offset shows whenever it lands in row 0.)
"""
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flashinfer")
if not torch.cuda.is_available():
    pytest.skip("the rows' sampler is the graph sampler, which needs CUDA", allow_module_level=True)

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.distributed.communication import WorkerParallelGroups  # noqa: E402
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources import StepContext  # noqa: E402
from mstar.engine.resources.sampler.config import (  # noqa: E402
    LoopbackState,
    SamplerSpec,
    SamplerStep,
    SamplingReqConfig,
)
from mstar.engine.resources.sampler.utils import (  # noqa: E402
    _rng_offset_stride,
    canonical_sampling_values,
    sample_accelerator_graphable_gpu,
)
from mstar.engine.resources.step import SubmoduleStep  # noqa: E402
from mstar.engine.rows import RowWalk  # noqa: E402
from mstar.model.submodule_base import NodeInputs, NodeSubmodule  # noqa: E402

NODE, ROWS_WALK, OTHER_WALK, MAX_ROWS, HANDLES, VOCAB = "toy", "decode", "other", 8, 32, 64
ACC, ACC_DEFAULT = "acc", (7, -7)


class _Toy(NodeSubmodule):
    reads_device_loopback = True

    def __init__(self, penalty: bool = False):
        super().__init__()
        self.penalty = penalty
        # set: a step of the other walk writes this loop-back state, by slot
        self.other_acc: int | None = None
        # rid -> (token, state) its next step brings routed, as a request's
        # first decode step does
        self.routed: dict[int, tuple[int, tuple[int, int]]] = {}
        gen = torch.Generator().manual_seed(0)
        # one fixed distribution per handle, so a draw depends only on its RNG state
        self.logits = torch.randn(HANDLES, VOCAB, generator=gen).cuda()

    def persistent_row_walks(self):
        return {ROWS_WALK: RowWalk()}

    def max_batch_size(self, graph_walk):
        return MAX_ROWS

    def can_batch(self, batch, model_inputs):
        return True

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        return NodeInputs(tensor_inputs={}, input_seq_len=1)

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs):
        writes = frozenset({ACC}) if graph_walk == ROWS_WALK else frozenset()
        return SubmoduleStep(steps={"sampler": SamplerStep(loopback_writes=writes, apply_penalty=self.penalty)})

    def preprocess(self, graph_walk, engine_inputs, inputs):
        rids = list(engine_inputs.request_ids)
        routed = [i for i, rid in enumerate(rids) if rid in self.routed]
        if graph_walk == ROWS_WALK and routed:
            sampler = engine_inputs.resources["sampler"]
            ids, acc = sampler.loopback_tokens(rids), sampler.loopback_state(ACC, rids)
            sampler.overwrite_rows(
                routed,
                (ids, torch.tensor([self.routed[rids[i]][0] for i in routed], device="cuda")),
                (acc, torch.tensor([self.routed[rids[i]][1] for i in routed], device="cuda")),
            )
        self.routed = {}
        return {}

    def forward(self, graph_walk, engine_inputs, **kwargs):
        raise AssertionError("every step runs batched")

    def forward_batched(self, graph_walk, engine_inputs, **kwargs):
        rids = list(engine_inputs.request_ids)
        sampler = engine_inputs.resources["sampler"]
        if graph_walk != ROWS_WALK:
            if self.other_acc is not None:
                acc = torch.full((len(rids), 2), self.other_acc, dtype=torch.long, device="cuda")
                sampler.store_loopback_state(ACC, rids, acc)
            return {rid: {} for rid in rids}
        prev = sampler.loopback_tokens(rids).clone()
        acc = sampler.loopback_state(ACC, rids).clone()
        tokens = sampler.sample(rids, self.logits[torch.tensor(rids, device="cuda")])
        sampler.store_loopback_state(ACC, rids, acc + 1)
        return {
            rid: {"token": [tokens[i:i + 1]], "prev": [prev[i:i + 1]], "acc": [acc[i:i + 1]]}
            for i, rid in enumerate(rids)
        }


def _engine(penalty: bool = False, max_live: int | None = None) -> Engine:
    spec = SamplerSpec(
        resource_key="sampler", nodes={NODE}, vocab_size=VOCAB, enable_repetion_penalty=penalty,
        loopback_state={ACC: LoopbackState(width=2, dtype=torch.long, default=ACC_DEFAULT)},
    )
    engine = Engine(None, autocast_dtype=None, max_concurrent_requests=max_live)
    engine.load_model(
        submodules={NODE: _Toy(penalty)}, specs=[spec],
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=torch.device("cuda"), transfer_engine_info=None,
    )
    engine.warmup()
    return engine


def _config(rid: int, gen: int, penalty: float = 1.0) -> SamplingReqConfig:
    # a recycled handle is a new request, with a seed of its own
    return SamplingReqConfig(
        temperature=1.0, top_k=8 * (rid % 3), top_p=0.9, _seed=1000 * gen + rid,
        repetition_penalty=penalty,
    )


def _expected_tokens(engine, rids, live, seen: dict | None = None) -> list[int]:
    """The step's draws, from each row's request state; with ``seen``, each
    request's tokens drawn so far penalised."""
    values = [
        canonical_sampling_values(cfg.temperature, cfg.top_k, cfg.top_p, 0.0)
        for cfg, _, _ in (live[rid] for rid in rids)
    ]

    def column(i, dtype):
        return torch.tensor([v[i] for v in values], dtype=dtype, device="cuda")

    stride = _rng_offset_stride("cuda", VOCAB)
    penalty = {}
    if seen is not None:
        penalty = dict(
            apply_penalty=True,
            rep_penalty=torch.tensor([live[rid][0].repetition_penalty for rid in rids], device="cuda"),
            seen_tokens=torch.stack([_mask(seen[rid]) for rid in rids]),
        )
    tokens = sample_accelerator_graphable_gpu(
        engine._submodules[NODE].submodule.logits[torch.tensor(rids, device="cuda")],
        column(0, torch.float32), column(1, torch.int32), column(2, torch.float32),
        torch.tensor([live[rid][0]._seed for rid in rids], device="cuda"),
        torch.tensor([live[rid][1] * stride for rid in rids], device="cuda"),
        **penalty,
    )
    return tokens.tolist()


def _mask(tokens) -> torch.Tensor:
    mask = torch.zeros(VOCAB, dtype=torch.bool, device="cuda")
    mask[list(tokens)] = True
    return mask


def _step(engine, walk, rids):
    batch = ExecutingBatch(
        node_name=NODE,
        per_request_info={rid: CurrentForwardPassInfo(
            request_id=f"r{rid}", rid_handle=rid, graph_walk=walk,
            fwd_index=0, random_seed=0, max_tokens=1,
        ) for rid in rids},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
        per_request_input_tensors={rid: {} for rid in rids},
    )
    engine.prepare_inputs(batch)
    out = engine.exec_and_postprocess(batch)
    if walk != ROWS_WALK:
        return batch, {}
    return batch, {
        rid: (int(out.get(rid)["token"][0]), int(out.get(rid)["prev"][0]),
              tuple(out.get(rid)["acc"][0].reshape(-1).tolist()))
        for rid in rids
    }


@pytest.mark.parametrize("seed", range(3))
def test_each_request_draws_from_its_own_state_across_rows_and_removals(seed):
    rng = random.Random(seed)
    engine = _engine()
    space = engine._rows[NODE].space(ROWS_WALK)
    gen = [0] * HANDLES
    # rid -> (config, draws so far, last token drawn)
    live: dict[int, tuple[SamplingReqConfig, int, int | None]] = {}
    free = list(range(HANDLES))
    for _ in range(120):
        if live and rng.random() < 0.15:
            rid = rng.choice(list(live))
            engine.remove_request(rid)
            del live[rid]
            free.insert(0 if rng.random() < 0.5 else len(free), rid)
        while free and len(live) < 12 and rng.random() < 0.3:
            rid = free.pop(0)
            gen[rid] += 1
            cfg = _config(rid, gen[rid])
            engine.add_request(rid, {"sampler": cfg})
            live[rid] = (cfg, 0, None)
        if not live:
            continue
        walk = ROWS_WALK if rng.random() < 0.7 else OTHER_WALK
        rids = rng.sample(list(live), rng.randint(1, min(MAX_ROWS, len(live))))
        batch, seen = _step(engine, walk, rids)
        if walk != ROWS_WALK:
            assert not any(space.holds(rid) for rid in rids)
            continue
        order = list(batch.request_ids)
        assert order == list(space.batch.order)
        assert [seen[rid][0] for rid in order] == _expected_tokens(engine, order, live)
        for rid in rids:
            cfg, draws, last = live[rid]
            token, prev, acc = seen[rid]
            if last is not None:
                # a new request's first read is whatever its slot last held
                assert prev == last, rid
            assert acc == (ACC_DEFAULT[0] + draws, ACC_DEFAULT[1] + draws), rid
            live[rid] = (cfg, draws + 1, token)
        space.check_mirrors()
    # every slot a removed request held came back
    bufs = engine._resources["sampler"]._cg_buffers
    assert not bufs._orphaned
    assert len(bufs._rid_to_slot) == len(live)


@pytest.mark.parametrize("max_live", [None, 16])
@pytest.mark.parametrize("seed", range(3))
def test_a_value_routed_in_preprocess_is_what_the_forward_reads(seed, max_live):
    """A row whose token and loop-back state come routed (a request's first
    decode step) has them overwritten in preprocess (``overwrite_rows``); the
    forward fills the step's rows from their slots first, and must still read
    the routed values, and the next step what this one stored."""
    rng = random.Random(seed)
    engine = _engine(max_live=max_live)
    toy = engine._submodules[NODE].submodule
    gen = [0] * HANDLES
    # rid -> (token, state) its next step reads, once known
    expect: dict[int, tuple[int, tuple[int, int]]] = {}
    live, free = set(), list(range(HANDLES))
    for k in range(120):
        if live and rng.random() < 0.15:
            rid = rng.choice(sorted(live))
            engine.remove_request(rid)
            live.discard(rid)
            expect.pop(rid, None)
            free.append(rid)
        while free and len(live) < 12 and rng.random() < 0.3:
            rid = free.pop(0)
            gen[rid] += 1
            engine.add_request(rid, {"sampler": _config(rid, gen[rid])})
            live.add(rid)
        if not live:
            continue
        walk = ROWS_WALK if rng.random() < 0.7 else OTHER_WALK
        rids = rng.sample(sorted(live), rng.randint(1, min(MAX_ROWS, len(live))))
        if walk == ROWS_WALK:
            toy.routed = {
                rid: (1000 + k, (500 + k, -500 - k))
                for rid in rids if rid not in expect or rng.random() < 0.2
            }
        routed = dict(toy.routed)
        _, seen = _step(engine, walk, rids)
        for rid, (token, prev, acc) in seen.items():
            want = routed.get(rid, expect.get(rid))
            assert (prev, acc) == want, (k, rid)
            expect[rid] = (token, (acc[0] + 1, acc[1] + 1))


def test_a_removed_request_s_slot_waits_for_its_row_s_check_in():
    """A step that checks a row in is proposed, the row's request is removed,
    and a new request writes its own state before that step lands: the
    check-in must not land in the new request's slot."""
    engine = _engine()
    toy = engine._submodules[NODE].submodule
    for rid in (0, 1):
        engine.add_request(rid, {"sampler": _config(rid, 1)})
    _step(engine, ROWS_WALK, [0, 1])
    leaving = ExecutingBatch(
        node_name=NODE,
        per_request_info={1: CurrentForwardPassInfo(
            request_id="r1", rid_handle=1, graph_walk=ROWS_WALK,
            fwd_index=0, random_seed=0, max_tokens=1,
        )},
        step_context=StepContext(request_ids=(1,), graph_walk=ROWS_WALK, slot=0, capture=False),
        per_request_input_tensors={1: {}},
    )
    engine.prepare_inputs(leaving)
    engine._enter_rows(leaving)  # proposed with 0 checking in
    engine.remove_request(0)
    engine.add_request(2, {"sampler": _config(2, 1)})
    toy.other_acc = 100
    _step(engine, OTHER_WALK, [2])
    toy.other_acc = None
    engine.exec_and_postprocess(leaving)
    _, seen = _step(engine, ROWS_WALK, [1, 2])
    assert seen[2][2] == (100, 100)


@pytest.mark.parametrize("seed", range(2))
def test_each_request_is_penalised_for_its_own_draws_across_rows_and_removals(seed):
    """A row holds its request's seen-token mask while it is in the rows: each
    draw penalises exactly the tokens its request drew before, whichever rows
    it held, and the request's own mask has them once its row left."""
    rng = random.Random(seed)
    engine = _engine(penalty=True)
    space = engine._rows[NODE].space(ROWS_WALK)
    sampler = engine._resources["sampler"]
    gen = [0] * HANDLES
    live: dict[int, tuple[SamplingReqConfig, int, int | None]] = {}
    seen: dict[int, set[int]] = {}
    free = list(range(HANDLES))
    for _ in range(100):
        if live and rng.random() < 0.15:
            rid = rng.choice(list(live))
            engine.remove_request(rid)
            del live[rid], seen[rid]
            free.insert(0 if rng.random() < 0.5 else len(free), rid)
        while free and len(live) < 12 and rng.random() < 0.3:
            rid = free.pop(0)
            gen[rid] += 1
            cfg = _config(rid, gen[rid], penalty=4.0)
            engine.add_request(rid, {"sampler": cfg})
            live[rid], seen[rid] = (cfg, 0, None), set()
        if not live:
            continue
        walk = ROWS_WALK if rng.random() < 0.7 else OTHER_WALK
        rids = rng.sample(list(live), rng.randint(1, min(MAX_ROWS, len(live))))
        batch, out = _step(engine, walk, rids)
        if walk != ROWS_WALK:
            # checked in: the request's own mask is the live one again
            for rid in rids:
                assert torch.equal(sampler._seen_mask(rid), _mask(seen[rid])), rid
            continue
        order = list(batch.request_ids)
        assert [out[rid][0] for rid in order] == _expected_tokens(engine, order, live, seen)
        for rid in rids:
            cfg, draws, _ = live[rid]
            live[rid] = (cfg, draws + 1, out[rid][0])
            seen[rid].add(out[rid][0])
