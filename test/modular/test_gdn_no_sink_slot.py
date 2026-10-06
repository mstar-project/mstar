"""Prefill when the pool runs without its sink slot.

With `RecurrentStateConfig.disable_sink_slot`, an unaddressed row carries -1
rather than a real slot. The decode and conv kernels read that sentinel
themselves, but SM90's chunked prefill does not take `state_indices` at all —
`GDNPrefillWrapper.run` gathers and scatters the packed state in torch, and
neither `index_select` nor `index_copy_` takes a negative index.

What has to hold: a padding row reads zeros like any fresh row, and writes
nowhere. Compaction would say it in one line, but a data-dependent shape cannot
be captured, so the wrapper redirects those rows onto a live row instead. The
tests below are mostly about that redirect staying inert — including under
capture, where the padding pattern at replay is not the one recorded.

The first test is CPU-only; the rest need a GPU for FlashInfer's kernel.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.linear_attn.config import (
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnVariant,
)
from mstar.engine.resources.linear_attn.gdn import GDNManager
from mstar.engine.resources.linear_attn.wrappers import GDNPrefillWrapper
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.utils.causal_conv1d import PAD_SLOT_ID

POOL, BACKEND = "gdn_state", "linear_attn"
# The chunked kernel's shapes, not arbitrary ones: K=V=128 with a k/v head
# ratio, as Qwen3.5 has.
NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM = 2, 4, 128
CONV_DIM = 2 * NUM_K_HEADS * HEAD_DIM + NUM_V_HEADS * HEAD_DIM
MAX_SLOTS = 8
BUCKET_TOKENS = 64


def pool_spec(disable_sink: bool) -> RecurrentStateSpec:
    return RecurrentStateSpec(
        POOL, {"llm"},
        RecurrentStateConfig(
            num_layers=1,
            blocks={
                "state": RecurrentBlockConfig(
                    shape=(NUM_V_HEADS, HEAD_DIM, HEAD_DIM), dtype=torch.float32,
                ),
                "conv": RecurrentBlockConfig(
                    shape=(CONV_DIM, 3), dtype=torch.float32,
                ),
            },
            max_slots=MAX_SLOTS,
            disable_sink_slot=disable_sink,
        ),
    )


def test_backend_takes_the_sink_setting_off_the_pool():
    """The wrapper's flag is static per deployment, which is what lets `run`
    branch on it under capture. It has to come off the pool all the same."""
    cpu = torch.device("cpu")
    for disable_sink in (False, True):
        spec = pool_spec(disable_sink)
        pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=cpu))
        backend = GDNManager.build(
            LinearAttnSpec(
                BACKEND, {"llm"},
                LinearAttnConfig(recurrent_state=POOL, variant=LinearAttnVariant.GDN),
            ),
            EngineResourceInfo(device=cpu, dependencies={POOL: spec}),
        )
        wrapper = backend._get_wrapper("main", is_decode=False)
        assert pool.pad_index == (PAD_SLOT_ID if disable_sink else 0)
        assert wrapper._has_sink_state is not disable_sink


# --- the gather/scatter itself --------------------------------------------

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlashInfer's chunked kernel needs a GPU"
)


def make_state(device) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(
        MAX_SLOTS, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.float32
    )


def make_inputs(total_tokens: int, device) -> dict:
    torch.manual_seed(1)
    kw = dict(device=device, dtype=torch.bfloat16)
    q = torch.randn(total_tokens, NUM_K_HEADS, HEAD_DIM, **kw)
    k = torch.randn(total_tokens, NUM_K_HEADS, HEAD_DIM, **kw)
    return dict(
        # the wrapper's kernel ignores its own l2norm flag, so normalise here
        # as `GDNManager.run` does
        q=torch.nn.functional.normalize(q.float(), dim=-1).to(q.dtype),
        k=torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype),
        v=torch.randn(total_tokens, NUM_V_HEADS, HEAD_DIM, **kw),
        a=torch.randn(total_tokens, NUM_V_HEADS, **kw),
        b=torch.randn(total_tokens, NUM_V_HEADS, **kw),
        a_log=torch.randn(NUM_V_HEADS, **kw),
        dt_bias=torch.randn(NUM_V_HEADS, **kw),
    )


def make_wrapper(device, has_sink: bool, cuda_graph: bool = False):
    return GDNPrefillWrapper(
        device=device,
        pad_slot_id=PAD_SLOT_ID,
        has_sink_state=has_sink,
        num_tokens=BUCKET_TOKENS if cuda_graph else None,
        bs=4 if cuda_graph else None,
        cuda_graph=cuda_graph,
    )


class Addressing:
    """The pool's per-(bucket, label) buffers, in miniature.

    `plan` holds on to the tensors it is handed rather than copying them, so a
    captured graph reads whatever storage was live at capture. The pool writes
    its addressing in place for exactly that reason, and a test that allocated
    per step would replay against the capture's slots and prove nothing.
    """

    def __init__(self, rows: int, device):
        self.slots = torch.zeros(rows, dtype=torch.int32, device=device)
        self.has_state = torch.zeros(rows, dtype=torch.bool, device=device)

    def write(self, slots, has_state):
        self.slots.copy_(torch.tensor(slots, dtype=torch.int32))
        self.has_state.copy_(torch.tensor(has_state, dtype=torch.bool))
        return self


def run_once(wrapper, state, spans, slots, has_state, inputs, addressing=None):
    addressing = addressing or Addressing(len(spans), state.device)
    addressing.write(slots, has_state)
    wrapper.plan(spans, addressing.slots, addressing.has_state)
    return wrapper.run(state=state, **inputs)


@requires_gpu
def test_padding_rows_read_zeros_and_write_nowhere():
    """A -1 row against the same batch run with a spare slot in its place.

    Both feed the kernel identical rows — a padding row starts fresh, and so
    does an unwritten spare — so the outputs have to match bit for bit. The
    difference is the scatter: the spare slot is written and the -1 is not.
    """
    device = torch.device("cuda")
    spans = [12, 7, 9, 5]
    slots = [3, -1, 1, -1]
    spares = [3, 6, 1, 7]  # what the -1 rows would have addressed
    has_state = [True, False, True, False]
    inputs = make_inputs(sum(spans), device)

    sink_off = make_state(device)
    out = run_once(
        make_wrapper(device, has_sink=False), sink_off, spans, slots, has_state, inputs
    )

    reference = make_state(device)
    ref_out = run_once(
        make_wrapper(device, has_sink=True), reference, spans, spares, has_state, inputs
    )

    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    for row, slot in enumerate(slots):
        if slot < 0:
            continue
        torch.testing.assert_close(
            sink_off[slot], reference[spares[row]], rtol=0, atol=0,
        )
    # every slot the batch does not address, including the two the padding
    # rows would have landed on
    untouched = set(range(MAX_SLOTS)) - {s for s in slots if s >= 0}
    pristine = make_state(device)
    for slot in sorted(untouched):
        torch.testing.assert_close(sink_off[slot], pristine[slot], rtol=0, atol=0)


@requires_gpu
def test_a_batch_of_nothing_but_padding_leaves_the_state_alone():
    """No live row to redirect onto, so the write folds back onto itself.

    Slot 0 is the one at risk: it is a real request's slot with the sink off,
    and it is where a clamped -1 lands.
    """
    device = torch.device("cuda")
    state = make_state(device)
    before = state.clone()
    spans = [6, 4]
    run_once(
        make_wrapper(device, has_sink=False), state, spans, [-1, -1], [False, False],
        make_inputs(sum(spans), device),
    )
    torch.testing.assert_close(state, before, rtol=0, atol=0)


@requires_gpu
def test_replay_takes_a_padding_pattern_the_capture_never_saw():
    """The case this is all for: a bucket captured on live rows and replayed
    on a layout whose trailing rows have gone to -1."""
    device = torch.device("cuda")
    wrapper = make_wrapper(device, has_sink=False, cuda_graph=True)
    spans = [8, 8, 8, 8]
    inputs = make_inputs(BUCKET_TOKENS, device)
    state = make_state(device)
    addressing = Addressing(len(spans), device)

    # capture against a full batch, as the runner does with its dummy rows
    run_once(wrapper, state, spans, [0, 1, 2, 3], [False] * 4, inputs, addressing)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            wrapper.run(state=state, **inputs)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(state=state, **inputs)

    # replay a narrower layout: two live rows, two padding. Planning writes the
    # wrapper's buffers in place, which is what the captured graph reads.
    state.copy_(make_state(device))
    before = state.clone()
    run_once(
        wrapper, state, spans, [2, 5, -1, -1], [False] * 4, inputs, addressing,
    )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(state).all()
    for slot in range(MAX_SLOTS):
        if slot in (2, 5):
            assert not torch.equal(state[slot], before[slot]), (
                f"live slot {slot} was not written by the replay"
            )
        else:
            torch.testing.assert_close(state[slot], before[slot], rtol=0, atol=0)
