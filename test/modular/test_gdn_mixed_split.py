"""`GDNMixedWrapper` computes what the chunked path computes, on two kernels.

A mixed step runs its one-token rows that carry state on the decode kernel
and its prompt rows on the chunked one. Against the chunked path over every
row: the same conv and delta-rule outputs, the same recurrent and conv state
left in every real slot, and nothing written for a padding row.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.linear_attn.wrappers import GDNMixedWrapper, GDNPrefillWrapper
from mstar.engine.resources.recurrent.pool import SINK_SLOT
from mstar.platforms import current_platform
from mstar.utils.causal_conv1d import PAD_SLOT_ID

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or current_platform.is_rocm(),
    reason="FlashInfer's GDN kernels need an NVIDIA GPU (test/rocm/test_gdn_aiter.py covers AMD)",
)

HEADS, DIM, CONV_DIM, WIDTH, SLOTS = 4, 128, 256, 4, 16

# (span, slot, has_state): decode rows, a continuing chunk, a stateless
# one-token prompt, a fresh prompt and a tail padding row
LAYOUT = [
    (1, 3, True), (1, 5, True), (6, 7, True), (1, 9, True),
    (1, 11, False), (4, 12, False), (0, SINK_SLOT, False),
]
SPANS = [s for s, _, _ in LAYOUT]
SLOT_IDS = [slot for _, slot, _ in LAYOUT]
HAS_STATE = [h for _, _, h in LAYOUT]
REAL = [i for i, s in enumerate(SPANS) if s > 0]


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
    torch.manual_seed(0)
    state = torch.randn(SLOTS, HEADS, DIM, DIM, device=device) * 0.05
    conv = torch.randn(SLOTS, CONV_DIM, WIDTH - 1, device=device, dtype=torch.bfloat16)
    return state, conv


def _reference(device, x, weight, inputs):
    state, conv = _pools(device)
    # capture-sized: only a captured step carries padding rows, and the eager
    # wrapper does not size its token mask for their borrowed tokens
    w = GDNPrefillWrapper(
        device=device, pad_slot_id=PAD_SLOT_ID, null_slot_id=SINK_SLOT,
        num_tokens=x.shape[0], bs=len(SPANS), accelerator_graph=True,
    )
    w.plan(
        SPANS,
        torch.tensor(SLOT_IDS, dtype=torch.int32, device=device),
        torch.tensor(HAS_STATE, dtype=torch.bool, device=device),
    )
    conv_out = w.run_conv(x, conv, weight)
    out = w.run(state=state, **inputs)
    return conv_out, out, state, conv


def _mixed(device, x, weight, inputs, accelerator_graph):
    state, conv = _pools(device)
    tokens = x.shape[0]
    if accelerator_graph:
        # a bucket wider than the step: its padded tail must not leak in
        bucket_tokens, bucket_bs = 32, 8
        pad_rows = bucket_bs - len(SPANS)
        spans = SPANS + [0] * pad_rows
        slots = SLOT_IDS + [SINK_SLOT] * pad_rows
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
        device=device, pad_slot_id=PAD_SLOT_ID, null_slot_id=SINK_SLOT,
        num_tokens=bucket_tokens, bs=bucket_bs, accelerator_graph=accelerator_graph,
        max_prefill_rows=4,
    )
    w.plan(spans, tuple(slots), tuple(has))
    conv_out = w.run_conv(x, conv, weight)[:tokens]
    out = w.run(state=state, **inputs)[:tokens]
    return conv_out, out, state, conv


def _close(a, b, what):
    err = (a.float() - b.float()).abs().max().item()
    scale = b.float().abs().max().item() + 1e-6
    assert err <= 2e-2 * scale, f"{what}: max err {err:.3g} vs scale {scale:.3g}"


@requires_gpu
@pytest.mark.parametrize("accelerator_graph", [False, True], ids=["eager", "bucket"])
def test_mixed_split_matches_the_chunked_path(accelerator_graph):
    device = torch.device("cuda")
    tokens = sum(SPANS)
    torch.manual_seed(2)
    x = torch.randn(tokens, CONV_DIM, device=device, dtype=torch.bfloat16)
    weight = torch.randn(CONV_DIM, WIDTH, device=device, dtype=torch.bfloat16) * 0.3
    inputs = _inputs(tokens, device)

    ref_conv, ref_out, ref_state, ref_cstate = _reference(device, x, weight, inputs)
    mix_conv, mix_out, mix_state, mix_cstate = _mixed(device, x, weight, inputs, accelerator_graph)

    _close(mix_conv, ref_conv, "conv output")
    _close(mix_out, ref_out, "delta-rule output")
    real_slots = [SLOT_IDS[i] for i in REAL]
    _close(mix_state[real_slots], ref_state[real_slots], "recurrent state")
    _close(mix_cstate[real_slots], ref_cstate[real_slots], "conv state")
    # slots no row addresses are left alone
    untouched = [s for s in range(SLOTS) if s not in real_slots and s != SINK_SLOT]
    fresh_state, fresh_conv = _pools(device)
    assert torch.equal(mix_state[untouched], fresh_state[untouched])
    assert torch.equal(mix_cstate[untouched], fresh_conv[untouched])


def test_too_many_prompt_rows_for_the_bucket_is_refused():
    w = GDNMixedWrapper(
        device=torch.device("cpu"), pad_slot_id=PAD_SLOT_ID, null_slot_id=SINK_SLOT,
        num_tokens=64, bs=8, accelerator_graph=True, max_prefill_rows=2,
    )
    with pytest.raises(RuntimeError, match="prompt rows"):
        w.plan([4, 4, 4, 1], (1, 2, 3, 4), (False, False, False, True))
