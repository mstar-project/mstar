"""Whisper's decoder node on the shared TokenStepNode: HF generate's
token suppression, applied to the logits the sampler gets; the batched stop
check, where EOS ends the loop on a decode step and the request on a prefill;
the per-request stop check, which copies only the token to the host; and the
walk and loop names of Whisper's graph."""
from collections.abc import Collection
from types import SimpleNamespace

import torch
from torch import nn

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.step import StepContext
from mstar.engine.state.adapters import node_handles, node_resources
from mstar.model.submodule_base import END_PARTITION, HostRows, ModelInputsFromEngine, engine_step
from mstar.model.whisper.components.decoder import WhisperDecoderModel
from mstar.model.whisper.config import (
    ATTN,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    DECODER_NODE,
    DECODER_STATE,
    KV_CACHE,
    PREFILL_WALK,
    SAMPLER,
    WhisperModelConfig,
)
from mstar.model.whisper.submodules import WhisperDecoderSubmodule
from mstar.model.whisper.whisper_model import WhisperModel


def _node():
    config = WhisperModelConfig(eos_token_id=3, suppress_tokens=[1, 5], begin_suppress_tokens=[2, 3])
    return WhisperDecoderSubmodule(nn.Identity(), config)


class _StubPages:
    """A KV pool or the cross read: every label open, nothing kept, and a read returns its queries."""

    requires_kv_write = True

    def layer_view(self, i):
        return torch.empty(0)

    def opens_label(self, request_id, label):
        return True

    def write_kv(self, k, v, layer_idx, label):
        pass

    def run(self, q, **kwargs):
        return q


class _StubPlan(_StubPages):
    """The self-attention's plan and the position counter: each row's next position, advanced at commit."""

    def __init__(self):
        self.next: dict[int, int] = {}
        self.segments = ()

    def position(self, request_id, label):
        return self.next.get(request_id, 0)

    def pos_ids(self, label):
        return torch.tensor([p for s in self.segments for p in range(self.next.get(s.request_id, 0),
                                                                     self.next.get(s.request_id, 0) + s.span)])

    def select_last_hidden(self, x, label):
        return x[torch.tensor([s.span for s in self.segments]).cumsum(0) - 1]

    def commit(self):
        for s in self.segments:
            self.next[s.request_id] = self.position(s.request_id, "main") + s.span


class _StubSampler:
    """Draws the next of ``tokens`` for each row as ``dtype``, records the logits it was given, and keeps each row's
    last draw for a loop-back row to read. An eager draw returns int32 tokens, and a captured one int64."""

    has_slot_masters = True

    def __init__(self):
        self.tokens: list[int] = []
        self.logits: list[torch.Tensor] = []
        self.dtype = torch.int32
        self.last: dict[int, int] = {}

    def sample(self, rows, logits):
        self.logits.append(logits.clone())
        drawn = torch.tensor([self.tokens.pop(0) for _ in rows], dtype=self.dtype)
        self.last.update(zip(rows, drawn.tolist(), strict=True))
        return drawn

    def loopback_tokens(self, rows):
        return torch.tensor([self.last[row] for row in rows])


class _Driver:
    """A tiny decoder node with random weights, over stub resources and a real decoder state pool, stepped as the
    engine steps it: declare, admit and plan the pool, open the handles, prepare, preprocess, forward, check and
    commit."""

    def __init__(self, config: WhisperModelConfig):
        model = object.__new__(WhisperModel)
        model.config = config
        torch.manual_seed(0)
        decoder = WhisperDecoderModel(config)
        for param in decoder.parameters():  # fused projections start from torch.empty
            nn.init.normal_(param, std=0.05)
        self.node = WhisperDecoderSubmodule(decoder.eval(), config)
        self.plan, self.sampler = _StubPlan(), _StubSampler()
        (spec,) = [spec for spec in node_resources(model) if spec.resource_key == DECODER_STATE]
        self.pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
        pages = _StubPages()
        self.node.bind_node_state(node_handles(model, DECODER_NODE, {
            KV_CACHE: pages, CROSS_KV_CACHE: pages, CROSS_ATTN: pages, ATTN: self.plan, "rope": self.plan,
            DECODER_STATE: self.pool, SAMPLER: self.sampler,
        }))
        self.ingested: set[int] = set()

    def step(self, walk: str, rows: dict, looped: Collection[int] = ()) -> dict:
        """One step over ``rows``, request handle to its forward-pass info and routed inputs. A row in ``looped`` reads
        its token off the sampler, as one whose last step here kept it on the device."""
        rids = list(rows)
        inputs = [self.node.prepare_inputs(walk, info, signals, has_loopback=rid in looped)
                  for rid, (info, signals) in rows.items()]
        for rid in set(rids) - self.ingested:
            self.pool.ingest_request(rid)
            self.ingested.add(rid)
        step = engine_step(self.node, graph_walk=walk, request_ids=rids, inputs=inputs)
        ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False)
        step.set_ctx(ctx)
        self.plan.segments = step.segments
        state = step.steps[DECODER_STATE]
        assert self.pool.admit(state, ctx).ok, "the pool holds a slot for every row"
        self.pool.plan(state, ctx)
        engine_inputs = ModelInputsFromEngine(request_ids=rids, per_request_info={}, step=step)
        step.begin_calls()
        try:
            with torch.no_grad():
                self.batch = self.node.preprocess(walk, engine_inputs, inputs)
                out = self.node.forward_batched(walk, engine_inputs, **self.batch)
            step.check_calls()
            self.pool.commit(state, ctx)
            step.commit_calls()
        finally:
            step.end_calls()
        if KV_CACHE in step.state:
            self.plan.commit()
        return out


def _tiny_config(**overrides) -> WhisperModelConfig:
    return WhisperModelConfig(**{
        "d_model": 64, "decoder_layers": 2, "decoder_attention_heads": 4, "decoder_ffn_dim": 128,
        "max_source_positions": 8, **overrides,
    })


def test_forward_suppresses_before_sampling():
    """A prefill and a decode step through ``forward``: the sampler sees the
    begin-suppressed tokens masked only for the first token. (The context is
    added in ``preprocess``, outside the captured prefill.)"""
    driver = _Driver(_tiny_config(suppress_tokens=[1, 5], begin_suppress_tokens=[2, 3]))
    info = SimpleNamespace(step_metadata={"timestamps": False})
    driver.sampler.tokens = [9, 9]
    driver.step(PREFILL_WALK, {0: (info, {
        "text_inputs": [torch.tensor([9, 9, 9, 9])], "encoder_states": [torch.zeros(8, 64)],
    })})
    driver.step("decode", {0: (info, {"text_inputs": [torch.tensor([9])]})})
    first, later = driver.sampler.logits
    assert first.shape == later.shape == (1, driver.node.config.vocab_size)
    assert torch.isinf(first[0, [1, 2, 3, 5]]).all() and not torch.isinf(first[0, [0, 4, 6, 7]]).any()
    assert torch.isinf(later[0, [1, 5]]).all() and not torch.isinf(later[0, [0, 2, 3, 4, 6, 7]]).any()


def test_names_match_whispers_graph():
    from mstar.model.whisper.whisper_model import WhisperModel

    model = object.__new__(WhisperModel)
    model.config = WhisperModelConfig()
    walks = model.get_graph_walk_graphs()
    node = _node()
    assert {PREFILL_WALK, node.DECODE_WALK} <= set(walks)
    assert walks[node.DECODE_WALK].name == node.DECODE_LOOP


def test_batched_stop_check_ends_the_loop_or_the_prefill():
    """As the per-request check: EOS ends the loop on a decode step (a row at
    its token cap is the loop cap's to end), and a prefill walk's EOS, which
    runs before the loop exists, ends the request."""
    node = _node()
    infos = {
        rid: SimpleNamespace(graph_walk=walk, resource_configs={"sampler": SimpleNamespace(ignore_eos=False)},
                             dynamic_loop_iter_counts={"decode_loop": n}, max_tokens=10,
                             step_metadata={}, rid_handle=i)
        for i, (rid, walk, n) in enumerate((("eos", "decode", 0), ("limit", "decode", 9), ("go", "decode", 0),
                                            ("prefill_eos", "prefill", 0)))
    }
    rows = HostRows(request_ids=("go", "limit", "eos", "prefill_eos"),
                    buffers={"new_token": torch.tensor([[7], [7], [3], [3]])})
    assert node.check_stop_batched(["eos", "limit", "go", "prefill_eos"], infos, rows) == {
        "eos": {"decode_loop"}, "prefill_eos": {END_PARTITION},
    }


