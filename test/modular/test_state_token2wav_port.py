"""Token2wav on the state contract: what it declares builds the resources it wrote by hand.

The node kept one slot pool and three bounded KV caches, each spec written by hand. It now declares a Value per
region that writes its slot, a Value of host fields, and an unbounded array per cache, and these tests hold the
specs the engine builds from them to the hand-written ones, copied here from before the port. Two things differ,
and both are the port's: the slot pool is three pools over the same blocks, and the DiT's cache holds a request's
two guidance rows as the heads of one row, (1, 16) for (2, 8). The last tests show that row attends and stores the
same bytes.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mstar.engine.resources.kv.bounded import BoundedKVConfig, BoundedKVStep, StreamPosition
from mstar.engine.resources.kv.bounded.config import SinkWindow
from mstar.engine.resources.kv.bounded.manager import BoundedKVManager
from mstar.engine.resources.recurrent.config import RecurrentBlockConfig
from mstar.engine.resources.step import Segment, StepContext
from mstar.engine.state.adapters import build_specs
from mstar.model.components.token2wav.flow import HandleLayerKV
from mstar.model.components.token2wav.keys import Token2WavKeys
from mstar.model.components.token2wav.model import (
    CACHE_FAMILIES,
    DIT_COMPUTE_DTYPE,
    KEEP_RECENT,
    UP_RATE,
    CacheCapacity,
    slot_layout,
)
from mstar.model.components.token2wav.state import token2wav_kernel_settings, token2wav_state

KEYS = Token2WavKeys(
    codes="tts_code", state={"conv": "t2w_conv", "vocoder": "t2w_vocoder", "tail": "t2w_tail"}, request="t2w_request",
    kv={"enc1": "t2w_enc1_kv", "enc2": "t2w_enc2_kv", "dit": "t2w_dit_kv"},
    attn={"enc1": "t2w_enc1_attn", "enc2": "t2w_enc2_attn", "dit": "t2w_dit_attn"},
)
SECONDS = (3.0, 10.0)
SLOTS = 9
# what each hand-written cache kept of a stream whose voice holds n entries
RETENTION = {
    "enc1": lambda n: SinkWindow(n + KEEP_RECENT // UP_RATE, 0),
    "enc2": lambda n: SinkWindow(n, KEEP_RECENT),
    "dit": lambda n: SinkWindow(KEEP_RECENT, n),
}


def _declared() -> dict:
    specs = build_specs(token2wav_state(KEYS, "Token2Wav", SECONDS),
                        kernel_settings=token2wav_kernel_settings(KEYS, SLOTS))
    return {spec.resource_key: spec.config for spec in specs}


def _capacity() -> tuple[CacheCapacity, int]:
    return CacheCapacity(int(SECONDS[1] * 25) + 1), max(0, int(SECONDS[0] * 25) - 1)


def _hand_written_caches() -> dict[str, BoundedKVConfig]:
    """``token2wav_resources``' caches before the port."""
    capacity, min_tokens = _capacity()
    return {
        KEYS.kv[name]: BoundedKVConfig(
            num_layers=family.num_layers, num_heads=family.num_heads, head_dim=family.head_dim,
            rows_per_request=family.rows, max_source_len=family.rate * capacity.prompt_tokens,
            min_source_len=family.rate * min_tokens, retention=RETENTION[name], max_slots=SLOTS - 1,
            reverse_step_order=family.reverse_step_order,
            dtype=DIT_COMPUTE_DTYPE if name == "dit" else torch.float32,
        )
        for name, family in CACHE_FAMILIES.items()
    }


def test_each_cache_s_spec_is_the_hand_written_one_with_its_rows_as_heads():
    declared = _declared()
    for key, reference in _hand_written_caches().items():
        config = declared[key]
        lengths = range(reference.min_source_len, reference.max_source_len + 1)
        assert [config.retention_for(n) for n in lengths] == [reference.retention_for(n) for n in lengths], (
            f"{key} would keep another sink or window than upstream's truncation"
        )
        folded = replace(reference, rows_per_request=1, num_heads=reference.rows_per_request * reference.num_heads,
                         retention=config.retention)
        assert config == folded, f"{key} would differ from the hand-written cache past its rows riding as heads"
        assert config.slot_tokens == reference.slot_tokens, f"{key}'s slots would hold another number of entries"
    dit = declared[KEYS.kv["dit"]]
    assert (dit.rows_per_request, dit.num_heads) == (1, 16), "the DiT's two guidance rows ride as heads"


def test_each_family_keeps_what_its_cache_keeps():
    """The model sizes the flow's noise offset and the encoder's position band from its families' kept sizes, a copy
    of what the bounded KV keeps."""
    declared = _declared()
    for name, family in CACHE_FAMILIES.items():
        config = declared[KEYS.kv[name]]
        for n in range(config.min_source_len, config.max_source_len + 1):
            assert SinkWindow(*family.kept(n)) == config.retention_for(n), (
                f"{name}'s noise offset or band would count other kept entries than its cache holds at length {n}"
            )


def test_the_slot_values_hold_the_hand_written_pool_s_blocks():
    declared = _declared()
    blocks = {}
    for key in KEYS.state.values():
        assert (declared[key].num_layers, declared[key].max_slots) == (1, SLOTS)
        blocks.update(declared[key].blocks)
    reference = slot_layout(_capacity()[0], dit_dtype=DIT_COMPUTE_DTYPE)
    assert blocks == {name: RecurrentBlockConfig(shape=block.shape, dtype=block.dtype)
                      for name, block in reference.items()}, "a block would have moved or changed between the pools"


def test_the_float32_caches_attend_in_tf32_as_they_did():
    declared = _declared()
    assert not any(declared[key].ieee for key in KEYS.kv.values()), "enc1 and enc2 would move to IEEE fp32"


# The DiT's guidance rows as heads of one row.


ROWS, HEADS, DIM, SOURCE, T = 2, 8, 64, 6, 5


def _dit_like(rows: int, heads: int, device: str) -> BoundedKVManager:
    return BoundedKVManager(BoundedKVConfig(
        num_layers=1, num_heads=heads, head_dim=DIM, rows_per_request=rows, max_source_len=SOURCE,
        min_source_len=SOURCE, retention=lambda n: SinkWindow(4, n), max_slots=2, reverse_step_order=True,
    ), torch.device(device))


def _assert_rows_as_heads_attend_and_store_as_rows(device: str) -> None:
    torch.manual_seed(0)
    by_rows, as_heads = _dit_like(ROWS, HEADS, device), _dit_like(1, ROWS * HEADS, device)
    source = torch.randn(1, ROWS, HEADS, SOURCE, 2 * DIM, device=device)
    requests = (1, 2)
    ctx = StepContext(request_ids=requests, graph_walk="w", slot=0, capture=False)
    # enough steps for the window to wrap
    for written in range(0, 3 * T, T):
        step = BoundedKVStep(
            segments=tuple(Segment(rid, "main", T) for rid in requests),
            positions={rid: StreamPosition(SOURCE, written) for rid in requests},
        )
        q, k, v = (torch.randn(len(requests) * ROWS, HEADS, T, DIM, device=device) for _ in range(3))
        outs = []
        for kv, fold in ((by_rows, False), (as_heads, True)):
            assert kv.admit(step, ctx).ok
            kv.plan(step, ctx)
            if fold:
                out = kv.attend(0, *(x.view(len(requests), ROWS * HEADS, T, DIM) for x in (q, k, v)),
                                source.view(1, 1, ROWS * HEADS, SOURCE, 2 * DIM))
                out = out.view(len(requests), T, ROWS, HEADS, DIM).transpose(1, 2).reshape(-1, T, HEADS, DIM)
            else:
                out = kv.attend(0, q, k, v, source)
            kv.commit(step, ctx)
            outs.append(out)
        assert torch.equal(*outs), "a guidance row riding as heads would attend other keys"
        assert torch.equal(by_rows._cache.flatten(2, 3), as_heads._cache.flatten(2, 3)), (
            "a guidance row riding as heads would store its keys elsewhere"
        )


def test_guidance_rows_as_heads_attend_and_store_as_rows_do():
    _assert_rows_as_heads_attend_and_store_as_rows("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels' programs, which load a row's heads by address")
def test_guidance_rows_as_heads_attend_and_store_as_rows_do_in_the_kernels():
    _assert_rows_as_heads_attend_and_store_as_rows("cuda")


class _StubHandle:
    """Returns a fixed read, laid out as the kernels lay it out: a request's rows side by side as heads."""

    def __init__(self, out: torch.Tensor):
        self.out = out

    def add(self, layer, k, v):
        pass

    def read(self, layer, q, **_):
        return self.out


def test_a_lone_request_s_guidance_rows_come_back_contiguous():
    torch.manual_seed(0)
    # one request, where unfolding the rows can be a strided view
    rows = torch.randn(ROWS, T, HEADS, DIM)
    as_heads = rows.view(1, ROWS, T, HEADS, DIM).transpose(1, 2).reshape(1, T, ROWS * HEADS, DIM)
    q = torch.randn(ROWS, HEADS, T, DIM)
    out = HandleLayerKV(_StubHandle(as_heads), 0, ROWS, max_keys=0).attend(q, q, q)
    assert torch.equal(out, rows)
    assert out.is_contiguous(), (
        "strided rows would send the DiT's projection through another matmul, which rounds differently"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernels run on CUDA")
def test_the_fused_residual_reads_guidance_rows_side_by_side():
    from mstar.model.components.token2wav.kernels import res_adaln

    torch.manual_seed(0)
    requests, width = 3, 64
    x, y = (torch.randn(requests * ROWS, T, width, device="cuda") for _ in range(2))
    gate, shift, scale = (torch.randn(requests * ROWS, 1, width, device="cuda") for _ in range(3))
    # as the DiT's projection leaves the read: a request's rows side by side at each step
    side_by_side = y.view(requests, ROWS, T, width).transpose(1, 2).contiguous()
    want = res_adaln(x, y, gate, shift, scale)
    got = res_adaln(x, side_by_side, gate, shift, scale, y_rows=ROWS)
    assert all(torch.equal(g, w) for g, w in zip(got, want, strict=True)), (
        "a guidance row would add the other row's attention output"
    )
