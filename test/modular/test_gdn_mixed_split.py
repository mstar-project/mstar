"""`GDNMixedWrapper` computes what the chunked path computes, on two kernels.

A mixed step runs its leading run of one-token rows that carry state on the
decode kernel and every other row on the chunked one, both reading the step's
tokens in place. Against the chunked path over every row: the same conv and
delta-rule outputs, the same recurrent and conv state left in every real slot,
and nothing written for a padding row. The engine orders a mixed step decode
first; any other order is still computed right, a decode row after the run
taking the chunked kernel. With the sink slot off, padding rows carry
NO_SLOT and slot 0 is an ordinary request's, which a padding row must leave as
the chunked path does.
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
from mstar.engine.resources.linear_attn.wrappers import GDNMixedWrapper, GDNPrefillWrapper
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
)
from mstar.engine.resources.recurrent.pool import NO_SLOT, SINK_SLOT
from mstar.engine.resources.spec import CombinedSplit
from mstar.engine.resources.step import BucketKey, SlotLease
from mstar.utils.causal_conv1d import PAD_SLOT_ID

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlashInfer's GDN kernels need a GPU"
)

HEADS, DIM, CONV_DIM, WIDTH, SLOTS = 4, 128, 256, 4, 16

# (span, slot, has_state)
LAYOUTS = {
    # as the engine orders a mixed step: decode rows, then a continuing chunk,
    # a stateless one-token prompt, a fresh prompt, a final chunk of one token
    # and a tail padding row
    "decode_first": [
        (1, 3, True), (1, 5, True), (1, 9, True), (6, 7, True),
        (1, 11, False), (4, 12, False), (1, 13, True), (0, SINK_SLOT, False),
    ],
    # interleaved: the decode rows after the first prompt run on the chunked kernel
    "interleaved": [
        (1, 3, True), (1, 5, True), (6, 7, True), (1, 9, True),
        (1, 11, False), (4, 12, False), (0, SINK_SLOT, False),
    ],
    # every row a prompt row: no decode run
    "no_decode": [(3, 3, True), (1, 11, False), (5, 12, False), (0, SINK_SLOT, False)],
}

# with the sink off: the layouts above with padding rows on NO_SLOT, plus
# slot 0 on a request's row, where a redirected read could address it
SINK_OFF_LAYOUTS = {
    **{
        name: [(s, NO_SLOT if slot == SINK_SLOT else slot, h) for s, slot, h in layout]
        for name, layout in LAYOUTS.items()
    },
    "slot0_in_run": [
        (1, 0, True), (1, 5, True), (1, 9, True), (6, 7, True),
        (4, 12, False), (0, NO_SLOT, False),
    ],
    "slot0_prompt": [(1, 3, True), (1, 5, True), (6, 0, True), (4, 12, False), (0, NO_SLOT, False)],
    # no live prompt row
    "decode_only_slot0": [(1, 0, True), (1, 5, True), (1, 9, True), (0, NO_SLOT, False)],
}
MAX_PROMPTS = 5


def _cols(layout):
    spans = [s for s, _, _ in layout]
    return spans, [slot for _, slot, _ in layout], [h for _, _, h in layout]


def _inputs(tokens, device):
    torch.manual_seed(1)
    kw = dict(device=device, dtype=torch.bfloat16)
    norm = lambda t: torch.nn.functional.normalize(t.float(), dim=-1).to(t.dtype)  # noqa: E731
    return dict(
        q=norm(torch.randn(tokens, HEADS, DIM, **kw)),
        k=norm(torch.randn(tokens, HEADS, DIM, **kw)),
        v=torch.randn(tokens, HEADS, DIM, **kw),
        a=torch.randn(tokens, HEADS, **kw),
        b=torch.randn(tokens, HEADS, **kw),
        # fp32, as the model keeps them (the decode kernel requires it)
        a_log=torch.randn(HEADS, device=device) * 0.1,
        dt_bias=torch.randn(HEADS, device=device) * 0.1,
    )


def _pools(device):
    # one guard row in front of each pool: a kernel that writes slot -1 lands there
    torch.manual_seed(0)
    state = torch.randn(1 + SLOTS, HEADS, DIM, DIM, device=device) * 0.05
    conv = torch.randn(1 + SLOTS, CONV_DIM, WIDTH - 1, device=device, dtype=torch.bfloat16)
    return state[1:], conv[1:]


def _guard(pool):
    return pool.as_strided(pool.shape[1:], pool.stride()[1:], pool.storage_offset() - pool.stride(0))


def _reference(device, x, weight, inputs, layout, null_slot=SINK_SLOT):
    SPANS, SLOT_IDS, HAS_STATE = _cols(layout)
    state, conv = _pools(device)
    # capture-sized: only a captured step carries padding rows, and the eager
    # wrapper does not size its token mask for their borrowed tokens
    w = GDNPrefillWrapper(
        device=device, pad_slot_id=PAD_SLOT_ID, null_slot_id=null_slot,
        num_tokens=x.shape[0], bs=len(SPANS), accelerator_graph=True,
        has_sink_state=null_slot != NO_SLOT,
    )
    w.plan(
        SPANS,
        torch.tensor(SLOT_IDS, dtype=torch.int32, device=device),
        torch.tensor(HAS_STATE, dtype=torch.bool, device=device),
    )
    conv_out = w.run_conv(x, conv, weight)
    out = w.run(state=state, **inputs)
    return conv_out, out, state, conv


def _mixed(device, x, weight, inputs, accelerator_graph, layout, decode_rows=None, null_slot=SINK_SLOT):
    SPANS, SLOT_IDS, HAS_STATE = _cols(layout)
    state, conv = _pools(device)
    tokens = x.shape[0]
    if accelerator_graph:
        # a bucket wider than the step: its padded tail must not leak in
        bucket_tokens, bucket_bs = 32, 10
        pad_rows = bucket_bs - len(SPANS)
        spans = SPANS + [0] * pad_rows
        slots = SLOT_IDS + [null_slot] * pad_rows
        has = HAS_STATE + [False] * pad_rows
        pad = bucket_tokens - tokens
        x = torch.cat((x, x.new_zeros(pad, x.shape[1])))
        inputs = {
            k: torch.cat((v, v.new_zeros(pad, *v.shape[1:]))) if v.shape[0] == tokens else v
            for k, v in inputs.items()
        }
    else:
        bucket_tokens = bucket_bs = None
        spans, slots, has = SPANS, SLOT_IDS, HAS_STATE
    w = GDNMixedWrapper(
        device=device, pad_slot_id=PAD_SLOT_ID, null_slot_id=null_slot,
        num_tokens=bucket_tokens, bs=decode_rows or bucket_bs,
        accelerator_graph=accelerator_graph, max_prefill_rows=MAX_PROMPTS,
        has_sink_state=null_slot != NO_SLOT,
    )
    w.plan(spans, tuple(slots), tuple(has))
    conv_out = w.run_conv(x, conv, weight)[:tokens]
    out = w.run(state=state, **inputs)[:tokens]
    return conv_out, out, state, conv


def _close(a, b, what):
    err = (a.float() - b.float()).abs().max().item()
    scale = b.float().abs().max().item() + 1e-6
    assert err <= 2e-2 * scale, f"{what}: max err {err:.3g} vs scale {scale:.3g}"


def _step(device, layout):
    tokens = sum(s for s, _, _ in layout)
    torch.manual_seed(2)
    x = torch.randn(tokens, CONV_DIM, device=device, dtype=torch.bfloat16)
    weight = torch.randn(CONV_DIM, WIDTH, device=device, dtype=torch.bfloat16) * 0.3
    return x, weight, _inputs(tokens, device)


def _check(layout, mixed, ref, null_slot=SINK_SLOT):
    _, SLOT_IDS, _ = _cols(layout)
    real = [i for i, (s, _, _) in enumerate(layout) if s > 0]
    mix_conv, mix_out, mix_state, mix_cstate = mixed
    ref_conv, ref_out, ref_state, ref_cstate = ref
    _close(mix_conv, ref_conv, "conv output")
    _close(mix_out, ref_out, "delta-rule output")
    real_slots = [SLOT_IDS[i] for i in real]
    _close(mix_state[real_slots], ref_state[real_slots], "recurrent state")
    _close(mix_cstate[real_slots], ref_cstate[real_slots], "conv state")
    # slots no row addresses are left alone (with the sink off, slot 0 too)
    untouched = [s for s in range(SLOTS) if s not in real_slots and s != null_slot]
    fresh_state, fresh_conv = _pools(mix_state.device)
    assert torch.equal(mix_state[untouched], fresh_state[untouched])
    assert torch.equal(mix_cstate[untouched], fresh_conv[untouched])
    if null_slot == NO_SLOT:
        assert torch.equal(_guard(mix_state), _guard(fresh_state)), "a padding row wrote slot -1"
        assert torch.equal(_guard(mix_cstate), _guard(fresh_conv)), "a padding row wrote slot -1"


def _split_cases():
    for null_slot, layouts, sink in ((SINK_SLOT, LAYOUTS, "sink"), (NO_SLOT, SINK_OFF_LAYOUTS, "no_sink")):
        for name in layouts:
            for graph, mode in ((False, "eager"), (True, "bucket")):
                # an eager step drops its padding rows, so a layout that differs
                # from its sink-on twin only in them adds nothing eagerly
                if null_slot == NO_SLOT and name in LAYOUTS and not graph:
                    continue
                yield pytest.param(null_slot, layouts[name], graph, id=f"{sink}-{name}-{mode}")


@requires_gpu
@pytest.mark.parametrize("null_slot,layout,accelerator_graph", list(_split_cases()))
def test_mixed_split_matches_the_chunked_path(null_slot, layout, accelerator_graph):
    device = torch.device("cuda")
    x, weight, inputs = _step(device, layout)
    ref = _reference(device, x, weight, inputs, layout, null_slot=null_slot)
    mixed = _mixed(device, x, weight, inputs, accelerator_graph, layout, null_slot=null_slot)
    _check(layout, mixed, ref, null_slot=null_slot)


@requires_gpu
def test_a_replay_of_another_layout_matches_the_chunked_path():
    """Captured over one layout and replayed over another: every buffer the
    graph reads is planned, so nothing of the captured step survives."""
    device = torch.device("cuda")
    bucket_tokens, bucket_bs = 32, 10
    w = GDNMixedWrapper(
        device=device, pad_slot_id=PAD_SLOT_ID, null_slot_id=SINK_SLOT,
        num_tokens=bucket_tokens, bs=bucket_bs, accelerator_graph=True,
        max_prefill_rows=MAX_PROMPTS,
    )
    state, conv = _pools(device)
    x_buf = torch.zeros(bucket_tokens, CONV_DIM, device=device, dtype=torch.bfloat16)
    weight = torch.randn(CONV_DIM, WIDTH, device=device, dtype=torch.bfloat16) * 0.3
    in_bufs = {
        k: (v.new_zeros(bucket_tokens, *v.shape[1:]) if v.dim() > 1 else v.clone())
        for k, v in _inputs(bucket_tokens, device).items()
    }

    def plan(layout):
        spans, slots, has = _cols(layout)
        pad = bucket_bs - len(spans)
        w.plan(spans + [0] * pad, tuple(slots + [SINK_SLOT] * pad), tuple(has + [False] * pad))

    def stage(layout):
        x, _, inputs = _step(device, layout)
        n = x.shape[0]
        x_buf.zero_()
        x_buf[:n].copy_(x)
        for k, v in inputs.items():
            if v.dim() > 1:
                in_bufs[k].zero_()
                in_bufs[k][:n].copy_(v)
        return x, inputs, n

    def forward():
        return w.run_conv(x_buf, conv, weight), w.run(state=state, **in_bufs)

    captured = LAYOUTS["interleaved"]
    plan(captured)
    stage(captured)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        forward()  # warm the kernels outside the graph
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        conv_out, out = forward()

    layout = LAYOUTS["decode_first"]
    state.copy_(_pools(device)[0])
    conv.copy_(_pools(device)[1])
    plan(layout)
    x, inputs, n = stage(layout)
    graph.replay()
    ref = _reference(device, x, weight, inputs, layout)
    _check(layout, (conv_out[:n], out[:n], state, conv), ref)


@requires_gpu
def test_the_decode_halfs_width_changes_no_real_rows_result():
    """Sizing the decode half to the bucket rather than to the largest capture
    only drops rows that address the null slot: outputs and every slot but the
    sink, which padding rows write as scratch, are bit-identical."""
    device = torch.device("cuda")
    layout = LAYOUTS["decode_first"]
    x, weight, inputs = _step(device, layout)

    conv_out, out, state, cstate = _mixed(device, x, weight, inputs, True, layout)
    wide = _mixed(device, x, weight, inputs, True, layout, decode_rows=128)

    assert torch.equal(conv_out, wide[0]) and torch.equal(out, wide[1])
    kept = [s for s in range(SLOTS) if s != SINK_SLOT]
    assert torch.equal(state[kept], wide[2][kept])
    assert torch.equal(cstate[kept], wide[3][kept])


def test_too_many_prompt_rows_for_the_bucket_is_refused():
    w = GDNMixedWrapper(
        device=torch.device("cpu"), pad_slot_id=PAD_SLOT_ID, null_slot_id=SINK_SLOT,
        num_tokens=64, bs=8, accelerator_graph=True, max_prefill_rows=2,
    )
    with pytest.raises(RuntimeError, match="prompt rows"):
        w.plan([4, 4, 4, 1], (1, 2, 3, 4), (False, False, False, True))


def test_a_captured_splits_decode_half_runs_the_buckets_rows():
    """The decode half runs every row it is sized for, so it is sized to its
    bucket rather than to the largest capture, as the other wrappers' buffers
    are; a 16-row mixed step would otherwise run the decode kernel on 128."""
    cpu = torch.device("cpu")
    spec = RecurrentStateSpec(
        "gdn_state", {"llm"},
        RecurrentStateConfig(
            num_layers=1,
            blocks={
                "state": RecurrentBlockConfig(shape=(HEADS, DIM, DIM), dtype=torch.float32),
                # [q|k|v] with two k heads
                "conv": RecurrentBlockConfig(shape=((4 + HEADS) * DIM, WIDTH - 1), dtype=torch.float32),
            },
            max_slots=SLOTS,
        ),
    )
    manager = GDNManager.build(
        LinearAttnSpec(
            "linear_attn", {"llm"},
            LinearAttnConfig(
                recurrent_state="gdn_state", variant=LinearAttnVariant.GDN,
                splits=(CombinedSplit("mixed", ("prefill_text",)),),
            ),
        ),
        EngineResourceInfo(device=cpu, dependencies={"gdn_state": spec}),
    )
    manager.build_accelerator_graph_buffers([], max_bs=128, max_seq_len=2048)

    def lease(walk, bs, tokens):
        return SlotLease(slot=0, bucket=BucketKey(graph_walk=walk, bs=bs, num_tokens=tokens))

    mixed = manager._get_wrapper(
        "main", is_decode=False, lease=lease("mixed", 16, 512),
        split=manager._splits["mixed"],
    )
    decode = manager._get_wrapper("main", is_decode=True, lease=lease("decode", 16, 16))
    assert mixed._decode._bs == 16
    assert decode._bs == 128


@requires_gpu
@pytest.mark.parametrize("accelerator_graph", [False, True], ids=["eager", "bucket"])
def test_a_mixed_steps_prompt_rows_save_the_snapshots_the_chunked_path_saves(accelerator_graph):
    """A prompt row's snapshot (its state at a page boundary, for the prefix
    cache) is addressed by the row's place in the chunked half. With more
    decode rows than a checkpoint interval, an eager step's chunked half still
    checkpoints only the prompt rows, which its buffer is sized to."""
    from mstar.engine.resources.recurrent.pool import checkpoint_rows

    device = torch.device("cuda")
    every, decode, prompts = 64, 70, (150, 100)
    snap_slots = [1 + decode + len(prompts) + i for i in range(len(prompts))]
    slots = snap_slots[-1] + 1
    layout = [(1, 1 + i, True) for i in range(decode)] + [
        (n, 1 + decode + i, i == 0) for i, n in enumerate(prompts)
    ]
    # (row, offset, slot): each prompt row keeps its last boundary
    snapshots = [(decode + i, n // every * every, s) for i, (n, s) in enumerate(zip(prompts, snap_slots, strict=True))]
    spans, slot_ids, has = _cols(layout)
    tokens = sum(spans)
    torch.manual_seed(2)
    x = torch.randn(tokens, CONV_DIM, device=device, dtype=torch.bfloat16)
    weight = torch.randn(CONV_DIM, WIDTH, device=device, dtype=torch.bfloat16) * 0.3
    inputs = _inputs(tokens, device)

    def pools():
        torch.manual_seed(0)
        return (
            torch.randn(slots, HEADS, DIM, DIM, device=device) * 0.05,
            torch.randn(slots, CONV_DIM, WIDTH - 1, device=device, dtype=torch.bfloat16),
        )

    def checkpoints(n):
        return torch.empty(n, HEADS, DIM, DIM, dtype=torch.float32, device=device)

    ref_state, ref_conv = pools()
    ref = GDNPrefillWrapper(
        device=device, pad_slot_id=PAD_SLOT_ID, null_slot_id=SINK_SLOT,
        num_tokens=tokens, bs=len(spans), accelerator_graph=True,
    )
    ref.plan(
        spans, torch.tensor(slot_ids, dtype=torch.int32, device=device),
        torch.tensor(has, dtype=torch.bool, device=device),
        snapshots, every, checkpoints(checkpoint_rows(tokens, len(spans), every)),
    )
    ref_conv_out = ref.run_conv(x, ref_conv, weight)
    ref_out = ref.run(state=ref_state, **inputs)

    state, conv = pools()
    if accelerator_graph:
        bucket_tokens, bucket_bs = 384, 80
        pad = bucket_bs - len(spans)
        spans, slot_ids, has = spans + [0] * pad, slot_ids + [SINK_SLOT] * pad, has + [False] * pad
        grow = bucket_tokens - tokens
        x = torch.cat((x, x.new_zeros(grow, x.shape[1])))
        inputs = {
            k: torch.cat((v, v.new_zeros(grow, *v.shape[1:]))) if v.shape[0] == tokens else v
            for k, v in inputs.items()
        }
        buffer = checkpoints(checkpoint_rows(bucket_tokens, bucket_bs, every))
    else:
        bucket_tokens = bucket_bs = None
        # as `GDNManager` sizes an eager step's buffer
        buffer = checkpoints(sum(max(span, 1) // every for span in spans))
    w = GDNMixedWrapper(
        device=device, pad_slot_id=PAD_SLOT_ID, null_slot_id=SINK_SLOT,
        num_tokens=bucket_tokens, bs=bucket_bs,
        accelerator_graph=accelerator_graph, max_prefill_rows=MAX_PROMPTS,
    )
    w.plan(spans, tuple(slot_ids), tuple(has), snapshots, every, buffer)
    conv_out = w.run_conv(x, conv, weight)[:tokens]
    out = w.run(state=state, **inputs)[:tokens]

    _close(conv_out, ref_conv_out, "conv output")
    _close(out, ref_out, "delta-rule output")
    assert torch.equal(conv[snap_slots], ref_conv[snap_slots]), "a snapshot saved another conv window"
    _close(state[snap_slots], ref_state[snap_slots], "snapshot state")
