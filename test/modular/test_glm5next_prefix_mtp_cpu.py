"""GLM-5.3 prefix reuse under MTP, through the real ``Engine`` on CPU: requests that start from
cached pages and a KDA snapshot decode what the same requests do with the cache shut, and what
greedy decode without MTP does.

The trunk's hidden states are compared too. A wrong restore moves them long before it flips a
greedy token of a random-weight model, and they are what a hit must leave untouched: the MTP
layer's own plane is not (its row i pairs with token i + 1, so the last cached row took another
request's next token), which only moves drafts.
"""
from __future__ import annotations

import dataclasses
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_glm5next_engine_cpu import _batch, _GreedySampler, _stubs  # noqa: F401 (the fixture)
from test_glm5next_mtp_cpu import _oracle

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine
from mstar.engine.resources import SamplingReqConfig
from mstar.engine.resources.kv.config import KVReqConfig
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import StepContext
from mstar.model.base import keyed_labels
from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
from mstar.model.glm5_next.config import KDA_STATE, KV_CACHE, SAMPLER, Glm5NextModelConfig
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel, process_weights_after_loading
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule

K = 3
# decode steps a request may run: the plain reference runs K + 1 for each of an MTP run's
LOOP_CAP = 64
SAMPLING = SamplingReqConfig(temperature=0.0, ignore_eos=True)


def _model(k: int, long_context: bool) -> Glm5NextModel:
    """``long_context``: DSA, where a verify step's rejected rows are rewritten; else dense MLA,
    where they stay as holes."""
    if long_context:
        cfg = Glm5NextModelConfig.reduced_long_context(max_seq_len=1024)
        cfg.prefill_window_tokens = 96  # cuts prompts inside a KDA chunk
    else:
        cfg = dataclasses.replace(Glm5NextModelConfig.reduced(), index_topk=960, max_seq_len=1024)
    cfg.mtp_num_draft_tokens = k
    model = Glm5NextModel("x", config_variant="reduced", tokenizer_mode="byte",
                          kda_conv_dtype=torch.float32, kda_max_requests=4, mtp_num_draft_tokens=k)
    model.config = cfg
    # no weights on disk: the deployment's salt roots the cache instead
    model.checkpoint_path = lambda: None
    return model


def _text(n: int, seed: int) -> str:
    """``n`` printable ASCII characters, a byte token each."""
    rng = random.Random(seed)
    return "".join(chr(rng.randrange(33, 127)) for _ in range(n))


class _Session:
    """Requests on one engine, stepped as the worker steps them. ``cache`` off allocates no
    snapshot slots, which keeps the node's prefix cache shut."""

    def __init__(self, k: int, long_context: bool, cache: bool, steps_per_decode: int = 1):
        self.model = _model(k, long_context)
        torch.manual_seed(0)
        lm = Glm5NextForCausalLM(self.model.config)
        # the trunk's parameters come first, and building the MTP layer draws too: reseeded,
        # every k draws the same trunk
        torch.manual_seed(0)
        for p in lm.parameters():
            if p.dtype.is_floating_point:
                # large enough that the greedy token depends on its context
                torch.nn.init.normal_(p, std=0.1)
        process_weights_after_loading(lm, torch.device("cpu"))
        self.sub = Glm5NextLLMSubmodule(lm.eval(), self.model.config)
        specs = self.model.get_node_resources()
        for spec in specs:
            if spec.resource_key == KV_CACHE:
                spec.config.prefix_cache_salt = "a salt"
            if spec.resource_key == KDA_STATE:
                spec.config.prefix_slots = 2 if cache else 0
        self.engine = Engine(None)
        self.engine.load_model(
            {"LLM": self.sub}, specs,
            parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
            device=torch.device("cpu"),
            transfer_engine_info=TransferEngineInfo(
                my_entity_id="e", my_session_id="s", transfer_engine=LocalTransferEngine("localhost"),
            ),
            kv_cache_type=torch.float32, model=self.model,
        )
        greedy = _GreedySampler()
        self.engine._resources[SAMPLER] = greedy
        self.engine._submodules["LLM"].resources[SAMPLER] = greedy
        self.sub.node_resources[SAMPLER] = greedy
        self.engine.warmup()
        self.steps_per_decode = steps_per_decode
        self.tokens: dict[str, list[int]] = {}
        self.emitted: list[int] = []  # tokens per request per decode step
        self.ran: dict[str, int] = {}  # tokens each prefill ran
        self.hidden: list[torch.Tensor] = []  # the trunk's output, each decode step
        self._next: dict[str, torch.Tensor] = {}
        self._iter: dict[str, int] = {}
        self._decoding = False
        lm.model.register_forward_hook(
            lambda module, args, out: self.hidden.append(out.detach().clone()) if self._decoding else None)

    def _step(self, walk: str, inputs: dict[str, torch.Tensor]) -> dict:
        infos = {
            rid: CurrentForwardPassInfo(
                request_id=f"wire-{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0, random_seed=0,
                max_tokens=1 << 20, resource_configs={SAMPLER: SAMPLING},
                dynamic_loop_iter_counts={"decode_loop": self._iter.get(rid, 0)},
            ) for rid in inputs
        }
        batch = _batch(
            node_name="LLM", per_request_info=infos,
            per_request_input_tensors={rid: {"text_inputs": [ids]} for rid, ids in inputs.items()},
            step_context=StepContext(request_ids=tuple(inputs), graph_walk=walk, slot=0, capture=False),
        )
        self.engine.prepare_inputs(batch)
        assert not batch.failed_requests, batch.failed_requests
        if walk == "prefill":
            self.ran.update(zip(inputs, (inp.input_seq_len for inp in batch.inputs), strict=True))
        self._decoding = walk == "decode"
        outputs = self.engine.exec_and_postprocess(batch)
        self._decoding = False
        assert batch.admit_error is None, batch.admit_error
        self.engine.finalize_batch(batch)
        self.engine.check_stop_for_batch(batch, outputs)
        return outputs.per_rid_outputs

    def prefill(self, **prompts: str) -> _Session:
        """Key each prompt as the preprocess worker and conductor do, and prefill them in one step."""
        page = self.engine._resources[KV_CACHE].config.tokens_per_page
        worker = SimpleNamespace(
            _keyed_labels=keyed_labels(self.model), _prefix_page_sizes={KV_CACHE: page}, model=self.model,
            _page_layout=lambda *args: PreprocessWorkerThread._page_layout(None, *args),
        )
        inputs = {}
        for rid, text in prompts.items():
            out = self.model.process_prompt(text, ["text"], ["text"])
            tensors, layouts = dict(out.new_input_tensors), out.metadata["kv_layout"]
            rows = {key: {label: layout.row() for label, layout in by.items()} for key, by in layouts.items()}
            sent = PreprocessWorkerThread._prefix_keys(worker, tensors, layouts, rows, None, {})
            kv = KVReqConfig()
            kv.apply_conductor_config(
                kv_layout=rows[KV_CACHE], loop_caps={"decode_loop": LOOP_CAP},
                **{name: by[KV_CACHE] for name, by in sent.items()},
            )
            self.engine.add_request(rid, {SAMPLER: SAMPLING, KV_CACHE: kv})
            inputs[rid] = tensors["text_inputs"][0]
        out = self._step("prefill", inputs)
        for rid in prompts:
            self.tokens[rid] = out[rid]["new_token"][0].reshape(-1).tolist()
            self._next[rid] = out[rid]["text_inputs"][0].reshape(-1)
            self._iter[rid] = 0
        return self

    def decode(self, steps: int) -> _Session:
        """Every live request, together."""
        for _ in range(steps * self.steps_per_decode):
            out = self._step("decode", dict(self._next))
            for rid in self._next:
                new = out[rid]["new_token"][0].reshape(-1).tolist()
                self.tokens[rid] += new
                self.emitted.append(len(new))
                self._next[rid] = out[rid]["text_inputs"][0].reshape(-1)
                self._iter[rid] += 1
        return self

    def remove(self, *rids: str) -> None:
        for rid in rids:
            self.engine.remove_request(rid)
            del self._next[rid]


# Each scenario returns the tokens its requests take from the cache.

def _diverges_at_a_page(s: _Session) -> dict[str, int]:
    """b leaves a's prompt exactly where the hit ends, beside a cold prompt: the last cached row of
    the MTP layer's plane was written with a's next token."""
    shared = _text(256, 1)
    s.prefill(a=shared + _text(40, 2)).decode(6).remove("a")
    s.prefill(b=shared + _text(44, 3), c=_text(60, 4)).decode(8)
    return {"b": 256}


def _ends_where_a_prompt_ended(s: _Session) -> dict[str, int]:
    """a ends on a page, so that page's last MTP row took its sampled token; b extends a and keeps
    the state there, and c starts from it."""
    a = _text(384, 5)
    s.prefill(a=a).decode(8).remove("a")
    s.prefill(b=a + _text(30, 6)).decode(4).remove("b")
    s.prefill(c=a + _text(50, 7)).decode(8)
    return {"b": 256, "c": 384}


def _resend(s: _Session) -> dict[str, int]:
    a = _text(310, 8)
    s.prefill(a=a).decode(6).remove("a")
    s.prefill(b=a).decode(8)
    return {"b": 256}


def _first_still_decoding(s: _Session) -> dict[str, int]:
    """a ends six tokens short of a page and its verify blocks have crossed into the next when b,
    which extends its prompt, arrives; both go on."""
    a = _text(250, 9)
    s.prefill(a=a).decode(8)
    s.prefill(b=a + _text(60, 10)).decode(5)
    return {"b": 128}


def _reused_snapshot_slot(s: _Session) -> dict[str, int]:
    """a's verify steps end a page while it decodes, and z's snapshot then takes the first free
    slot: b must restore z's KDA state and nothing a verify step left."""
    shared = _text(270, 11)
    s.prefill(a=_text(120, 12)).decode(6).remove("a")
    s.prefill(z=shared + _text(40, 13)).decode(5).remove("z")
    s.prefill(b=shared + _text(44, 14)).decode(8)
    return {"b": 256}


def _several_windows_past_the_hit(s: _Session) -> dict[str, int]:
    """The third prompt on a shared prefix restores the state the second kept at the hit's end, and
    runs several prefill windows past it, beside a cold prompt."""
    shared = _text(600, 15)
    s.prefill(a=shared + _text(100, 16)).decode(4).remove("a")
    s.prefill(b=shared + _text(150, 17)).decode(4).remove("b")
    s.prefill(c=shared + _text(300, 18), d=_text(100, 19)).decode(6)
    return {"c": 512}


CASES = [
    (_diverges_at_a_page, True, "own"), (_diverges_at_a_page, False, "own"),
    (_diverges_at_a_page, True, "oracle"),
    (_ends_where_a_prompt_ended, True, "oracle"), (_ends_where_a_prompt_ended, False, "oracle"),
    (_resend, True, "oracle"),
    (_first_still_decoding, True, "oracle"), (_first_still_decoding, False, "oracle"),
    (_reused_snapshot_slot, True, "oracle"), (_reused_snapshot_slot, False, "oracle"),
    (_several_windows_past_the_hit, True, "oracle"),
]


@pytest.mark.parametrize(
    ("scenario", "long_context", "drafts"), CASES,
    ids=[f"{s.__name__[1:]}-{'long_context' if lc else 'dense'}-{d}" for s, lc, d in CASES],
)
def test_requests_past_a_cached_prefix_decode_as_with_the_cache_shut(scenario, long_context, drafts):
    """``drafts``: the MTP layer's own, which read its plane's stale row, or an oracle's, wrong
    every fourth call, so a step takes anything from none of them to all."""
    plain = _Session(0, long_context, cache=False, steps_per_decode=K + 1)
    scenario(plain)

    def run(cache: bool) -> tuple[_Session, dict[str, int]]:
        session = _Session(K, long_context, cache)
        if drafts == "oracle":
            _oracle(session.sub, plain.tokens, wrong_every=4)
        return session, scenario(session)

    cold, _ = run(cache=False)
    warm, hits = run(cache=True)

    assert not cold.ran.keys() - warm.ran.keys()
    assert {rid: cold.ran[rid] - warm.ran[rid] for rid in warm.ran if cold.ran[rid] != warm.ran[rid]} == hits
    if drafts == "oracle":
        assert 1 in warm.emitted and K + 1 in warm.emitted, "the oracle's drafts were all taken or all refused"
    assert warm.tokens == cold.tokens, "a prefix hit changed the greedy output"
    for rid, tokens in warm.tokens.items():
        assert tokens == plain.tokens[rid][:len(tokens)], f"{rid}: MTP changed the greedy output"
    assert len(warm.hidden) == len(cold.hidden)
    for step, (hit, ref) in enumerate(zip(warm.hidden, cold.hidden, strict=True)):
        assert torch.allclose(hit, ref, rtol=0, atol=1e-4), (
            f"decode step {step}: the trunk's states moved by {(hit - ref).abs().max():.3g} after a hit")
