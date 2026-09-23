"""What a request generates is keyed the same way its prompt was.

In agent traffic one turn's output is the next turn's prompt, so a generated
page is worth as much as a prompt page — but the tokens only exist once they are
sampled, and the only host copy of them is the one the stop check already takes.
That copy is early enough: a step writes the id the step before it sampled, so by
the commit that fills a page's last slot, every id in that page is already on the
host and the page is indexed there and then.

The page where the prompt ends and the generation begins belongs to both. Its
key covers the prompt's trailing tokens and the first generated ones together,
which is why the request carries that tail: the manager sees sampled tokens, and
nothing else of the prompt.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig, KVStep
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext

PAGE_SIZE = 128
ROOT = b"a root"
NODE = "LLM"
WALK = "decode"
TENSOR = "text_inputs"
EOS = 2


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del kwargs
        return transfer_info == self.get_kv_transfer_info()

    def remove_request(self, request_id):
        del request_id

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(max_num_pages: int = 64) -> KVManager:
    kv = KVManager(
        cfg=KVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=8192,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT)
    return kv


def _pages(tokens: list[int]) -> list[list[int]]:
    return [
        tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)
    ]


def _ingest(kv: KVManager, rid: str, prompt: list[int], chains: bool = True):
    whole = len(prompt) // PAGE_SIZE
    kv.ingest_request(rid, KVReqConfig(
        prefix_keys={"main": chain(_pages(prompt))},
        prefix_tail={"main": prompt[whole * PAGE_SIZE:]},
        prefix_decode={"main": TENSOR} if chains else None,
    ))


def _step(kv: KVManager, rid: str, span: int) -> None:
    step = KVStep(segments=(Segment(rid, "main", span),))
    ctx = StepContext(
        request_ids=(rid,), graph_walk=WALK, slot=0, capture=False,
    )
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _sampled(kv: KVManager, rid: str, token: int) -> None:
    """The host copy of what a step sampled, taken after its stop check."""
    kv.extend_prefix_chain(
        rid, NODE, WALK, {TENSOR: [torch.tensor([token])]},
    )


def _prefill(kv: KVManager, rid: str, prompt: list[int], sampled: int = 9000):
    """The prompt, and the id the prefill step sampled from it."""
    _step(kv, rid, len(prompt))
    _sampled(kv, rid, sampled)


def _decode(kv: KVManager, rid: str, token: int) -> None:
    """One decode step: it writes the id sampled before it, and samples ``token``."""
    _step(kv, rid, 1)
    _sampled(kv, rid, token)


def _indexed(kv: KVManager) -> int:
    return len(kv._index.pages())


# ── the page the generation fills ───────────────────────────────────────


def test_a_generated_page_is_indexed_at_the_commit_that_fills_it():
    kv = _manager()
    prompt = list(range(PAGE_SIZE))
    _ingest(kv, "r0", prompt)
    _prefill(kv, "r0", prompt)
    assert _indexed(kv) == 1, "the prompt's own page was not indexed"

    seen = []
    for token in range(1, PAGE_SIZE + 2):
        _decode(kv, "r0", 9000 + token)
        seen.append((kv._streams["r0"]["main"].stored_len, _indexed(kv)))

    filled = [step for step, (stored, _) in enumerate(seen) if stored == 2 * PAGE_SIZE]
    assert len(filled) == 1, (
        "the stream passed two pages of tokens, so the step that filled the "
        "second one is not the step under test"
    )
    at = filled[0]
    assert seen[at - 1][1] == 1, "a page was indexed before its last slot was written"
    assert seen[at][1] == 2, (
        "the page was not indexed at the commit that filled it; every id in it "
        "was already on the host by then"
    )
    kv.assert_pages_conserved()


def test_a_generated_page_is_matched_by_the_request_that_asks_for_it_next():
    kv = _manager()
    prompt = list(range(100))
    generated = list(range(9000, 9000 + 28))
    _ingest(kv, "r0", prompt)
    _prefill(kv, "r0", prompt, sampled=generated[0])
    for token in generated[1:]:
        _decode(kv, "r0", token)
    # the step that writes the last generated id, which fills the page
    _decode(kv, "r0", 9999)

    # the next turn: the whole of the first turn is now the prompt
    _ingest(kv, "r1", prompt + generated + list(range(200, 260)))

    assert kv.resolve_cached_prefix("r1", NODE, WALK) == PAGE_SIZE, (
        "the page the first turn generated was not there for the second"
    )
    kv.assert_pages_conserved()


# ── nodes that cannot chain what they generate ──────────────────────────


def test_a_stop_on_the_boundary_leaves_the_page_one_slot_short():
    kv = _manager()
    prompt = list(range(100))
    _ingest(kv, "r0", prompt)
    _prefill(kv, "r0", prompt, sampled=9000)

    # the request stops here: its last two sampled ids, the one before the
    # stop and the stop itself, never reach a step that could write them
    for token in range(1, 28):
        _decode(kv, "r0", 9000 + token)
    _sampled(kv, "r0", EOS)

    stream = kv._streams["r0"]["main"]
    assert stream.chain.keyed_pages == 1, "the chain did not reach the page boundary"
    assert stream.stored_len == PAGE_SIZE - 1, (
        "the id that ended the request was written into a slot"
    )
    assert _indexed(kv) == 0, (
        "a page whose last slot holds nothing was offered to the index"
    )
    kv.assert_pages_conserved()


def test_a_node_whose_decode_ids_are_not_the_sampled_token_stops_at_its_prompt():
    kv = _manager()
    prompt = list(range(PAGE_SIZE))
    _ingest(kv, "r0", prompt, chains=False)
    _prefill(kv, "r0", prompt)
    after_prompt = _indexed(kv)

    for token in range(1, PAGE_SIZE + 2):
        _decode(kv, "r0", 9000 + token)

    assert after_prompt == 1, (
        "a node whose decode ids are not the token it sampled keyed a page of "
        "them anyway"
    )
    assert _indexed(kv) == 1, (
        "a node whose decode ids are not the token it sampled indexed pages "
        "it cannot name"
    )
    kv.assert_pages_conserved()


def test_a_request_that_opted_out_chains_nothing():
    kv = _manager()
    prompt = list(range(PAGE_SIZE))
    kv.ingest_request("r0", KVReqConfig(
        prefix_keys={"main": chain(_pages(prompt))},
        prefix_tail={"main": []},
        prefix_decode={"main": TENSOR},
        prefix_cache=False,
    ))

    _prefill(kv, "r0", prompt)
    for token in range(1, PAGE_SIZE + 2):
        _decode(kv, "r0", 9000 + token)

    assert _indexed(kv) == 0, (
        "a request that turned the cache off left its generated pages behind"
    )
    kv.assert_pages_conserved()
