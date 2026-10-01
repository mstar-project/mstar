"""A chain describes a stream only while the stream holds nothing else.

The keys cover the prompt's leading text and then grow one sampled token at a
time. A stream can hold more than that: Bagel writes an image into the same
stream after the text, and a request read in from another rank starts its decode
from a token that rank sampled. Keep keying past either and a page is filed
under a key that describes different tokens, and the next request with those
tokens attends it. Once the stream holds what the chain never saw, the chain
stops. A stream reading back its own publish, as a colocated decode does every
step, sampled that token itself and keeps its chain.

A decode step is launched before the token its predecessor sampled is read back,
so a step can commit the token it writes before the chain has counted it. That
is not a gap, and it must not stop the chain.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig, KVStep
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import (
    KVManager,
    KVSequenceInfo,
    PublishedKVInfo,
    RetentionPolicy,
)
from mstar.engine.resources.step import Segment, StepContext

PAGE_SIZE = 16
ROOT = b"a root"
NODE = "LLM"
WALK = "decode"
TENSOR = "text_inputs"
PROMPT = list(range(100, 136))  # two whole pages and a tail of four
IMAGE = 12


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache):
        del transfer_engine_info, kv_cache

    def get_kv_transfer_info(self):
        return None

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager() -> KVManager:
    kv = KVManager(
        cfg=KVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=64, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT)
    return kv


def _ingest(kv: KVManager, rid: str, prompt: list[int] = PROMPT) -> None:
    whole = len(prompt) // PAGE_SIZE
    kv.ingest_request(rid, KVReqConfig(
        prefix_keys={"main": chain([
            prompt[at:at + PAGE_SIZE] for at in range(0, len(prompt), PAGE_SIZE)
        ])},
        prefix_tail={"main": prompt[whole * PAGE_SIZE:]},
        prefix_decode={"main": TENSOR},
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
    kv.extend_prefix_chain(rid, NODE, WALK, {TENSOR: [torch.tensor([token])]})


def _indexed(kv: KVManager) -> int:
    return len(kv._index.pages())


# ── what the chain never saw ────────────────────────────────────────────


def test_an_image_after_the_text_leaves_the_chain_at_the_text():
    kv = _manager()
    _ingest(kv, "r0")
    _step(kv, "r0", len(PROMPT))
    # the image is the last thing prefilled, so its step samples the first token
    _step(kv, "r0", IMAGE)
    _sampled(kv, "r0", 9000)
    kv.assert_pages_conserved()

    for token in range(9001, 9031):
        _step(kv, "r0", 1)
        _sampled(kv, "r0", token)
    kv.assert_pages_conserved()

    assert _indexed(kv) == len(PROMPT) // PAGE_SIZE, (
        "a page holding the image was filed under a key over sampled text"
    )
    assert kv._streams["r0"]["main"].chain is None, (
        "the chain kept growing over a stream that holds tokens it never saw"
    )


def test_a_read_in_stream_keys_nothing_it_generates():
    kv = _manager()
    _ingest(kv, "r0")
    published = PublishedKVInfo.build_for_rank(0, 1, {"main": KVSequenceInfo(
        seq_len=len(PROMPT), latest_kv_transfer_info="peer",
        page_indices=list(range(3)),
    )})
    assert kv.admit_retrieve("r0", NODE, WALK, published).ok, (
        "the read-in was refused, so this stream never held the prompt"
    )
    kv.assert_pages_conserved()

    # the first token was sampled by the rank that prefilled, so the samples
    # that arrive here start with the second
    for token in range(9001, 9031):
        _step(kv, "r0", 1)
        _sampled(kv, "r0", token)
    kv.assert_pages_conserved()

    generated = kv._streams["r0"]["main"].page_indices[len(PROMPT) // PAGE_SIZE:]
    assert not set(kv._index.pages()) & set(generated), (
        "a generated page was keyed one token late and filed under the wrong "
        "tokens"
    )


def test_a_peers_publish_drops_the_partial_chain_even_when_nothing_is_read():
    kv = _manager()
    whole = PROMPT[:2 * PAGE_SIZE]
    _ingest(kv, "r0", whole)
    _step(kv, "r0", len(whole))
    _ingest(kv, "r1", whole)
    published = PublishedKVInfo.build_for_rank(0, 1, {"main": KVSequenceInfo(
        seq_len=len(whole), latest_kv_transfer_info="peer",
        page_indices=list(range(2)),
    )})

    assert kv.admit_retrieve("r1", NODE, WALK, published).ok
    kv.assert_pages_conserved()

    stream = kv._streams["r1"]["main"]
    assert stream.page_indices == kv._streams["r0"]["main"].page_indices, (
        "the peer's pages were read in rather than matched here"
    )
    assert stream.chain.unkeyed is None, (
        "a peer's publish this cache already held left the chain open, and the "
        "peer's first sampled token would be keyed one place late"
    )


# ── what is not a gap ───────────────────────────────────────────────────


def test_a_text_stream_keeps_its_chain_through_its_decode():
    kv = _manager()
    _ingest(kv, "r0")
    _step(kv, "r0", len(PROMPT))
    _sampled(kv, "r0", 9000)

    for token in range(9001, 9031):
        _step(kv, "r0", 1)
        _sampled(kv, "r0", token)
    kv.assert_pages_conserved()

    assert _indexed(kv) == (len(PROMPT) + 30) // PAGE_SIZE, (
        "a stream that holds nothing but its prompt and its samples lost its chain"
    )


def test_a_stream_reading_its_own_publish_keeps_keying_its_samples():
    kv = _manager()
    _ingest(kv, "r0")
    _step(kv, "r0", len(PROMPT))
    _sampled(kv, "r0", 9000)

    # a colocated decode is handed the record its own last step published
    for token in range(9001, 9031):
        assert kv.admit_retrieve("r0", NODE, WALK, kv.publish("r0")).ok
        _step(kv, "r0", 1)
        _sampled(kv, "r0", token)
    kv.assert_pages_conserved()

    assert _indexed(kv) == (len(PROMPT) + 30) // PAGE_SIZE, (
        "a stream reading back its own publish stopped keying what it sampled"
    )


def test_a_decode_step_that_commits_before_its_token_is_read_back_keeps_the_chain():
    kv = _manager()
    _ingest(kv, "r0")
    _step(kv, "r0", len(PROMPT))

    # every step commits the token its predecessor sampled before that
    # predecessor's sample is counted, as a speculated step does
    for token in range(9000, 9030):
        _step(kv, "r0", 1)
        _sampled(kv, "r0", token)
    kv.assert_pages_conserved()

    assert kv._streams["r0"]["main"].chain is not None, (
        "a step that ran ahead of the read-back was taken for a gap in the chain"
    )
    assert _indexed(kv) == (len(PROMPT) + 30) // PAGE_SIZE, (
        "the generated pages were never indexed"
    )


# ── what a window releases ──────────────────────────────────────────────


def test_a_stream_that_released_its_front_keys_nothing_further():
    kv = _manager()
    _ingest(kv, "r0")
    _step(kv, "r0", len(PROMPT))
    indexed = _indexed(kv)
    _sampled(kv, "r0", 9000)

    # what a sliding window leaves behind: the pages moved, the lengths did not
    kv._streams["r0"]["main"].released = PAGE_SIZE
    for token in range(9001, 9031):
        _step(kv, "r0", 1)
        _sampled(kv, "r0", token)
    kv.assert_pages_conserved()

    assert _indexed(kv) == indexed, (
        "a page was filed under a key for whatever used to be at its index"
    )
    assert kv._streams["r0"]["main"].chain is None, (
        "the chain carried on over a stream whose pages had moved under it"
    )


def test_a_windowed_stream_indexes_up_to_its_protected_prefix():
    kv = _manager()
    _ingest(kv, "r0")
    kv._streams["r0"]["main"].retention = RetentionPolicy(
        context_budget=4 * PAGE_SIZE, protected_prefix=PAGE_SIZE,
    )

    _step(kv, "r0", len(PROMPT))
    kv.assert_pages_conserved()

    assert _indexed(kv) == 1, (
        "a windowed stream filed pages its window is free to release"
    )


def test_a_stream_with_no_window_indexes_every_page_it_fills():
    kv = _manager()
    _ingest(kv, "r0")

    _step(kv, "r0", len(PROMPT))
    kv.assert_pages_conserved()

    assert _indexed(kv) == len(PROMPT) // PAGE_SIZE, (
        "a stream nobody windowed lost pages to the window rule"
    )


# ── a walk the keys never described ─────────────────────────────────────

WALKS = {"main": ("prefill_text", "decode")}


def _first_write(walk: str) -> KVManager:
    """Key 40 tokens, then commit 50 under ``walk`` before anything else."""
    kv = KVManager(
        cfg=KVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=16, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT, WALKS)
    prompt = list(range(40))
    kv.ingest_request("r0", KVReqConfig(
        prefix_keys={"main": chain([prompt[:16], prompt[16:32], prompt[32:]])},
        prefix_tail={"main": prompt[32:]},
    ))
    step = KVStep(segments=(Segment("r0", "main", 50),))
    ctx = StepContext(request_ids=("r0",), graph_walk=walk, slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)
    return kv


def test_an_image_walk_writing_first_files_nothing_and_ends_the_chain():
    kv = _first_write("prefill_vae")

    assert _indexed(kv) == 0, (
        "pages the image walk wrote were filed under keys chained over the "
        "prompt's text"
    )
    assert kv._streams["r0"]["main"].chain is None, (
        "the chain outlived a walk it never described"
    )
    kv.assert_pages_conserved()


def test_the_keyed_walk_writing_the_same_span_files_its_whole_pages():
    kv = _first_write("prefill_text")

    assert _indexed(kv) == 2, "the walk the model keyed filed nothing"
    kv.assert_pages_conserved()
