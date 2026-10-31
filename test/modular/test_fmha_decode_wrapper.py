"""The fmha decode wrapper: which captures take it, what its plan stages, and on a
GPU its output against the paged decode kernel it stands in for."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mstar.engine.resources.attn import flashinfer as fi_mod
from mstar.engine.resources.attn import wrappers as wrappers_mod
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.attn.flashinfer import FlashInferManager
from mstar.engine.resources.attn.wrappers import FmhaDecodeWrapper, fmha_decode_applies
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.step import BucketKey, SlotLease

_CUDA = torch.device("cuda")
_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize(
    "capability, expected",
    [((10, 3), True), ((10, 0), True), ((9, 0), False), ((12, 0), False), (None, False)],
)
def test_guard_is_blackwell_only(monkeypatch, capability, expected):
    monkeypatch.setattr(
        wrappers_mod, "current_platform",
        SimpleNamespace(cuda_capability=lambda device=None: capability),
    )
    assert fmha_decode_applies(_CUDA) is expected


def test_guard_refuses_cpu():
    assert not fmha_decode_applies(torch.device("cpu"))


def _cpu_wrapper(monkeypatch, batch_size=3, max_seq_len=1024, page_size=128):
    pytest.importorskip("flashinfer")
    import flashinfer.utils as fi_utils

    monkeypatch.setattr(fi_utils, "get_device_sm_count", lambda device: 100)
    monkeypatch.setattr(
        fi_utils, "_get_trtllm_gen_multi_ctas_kv_counter_buffer",
        lambda *args: torch.zeros(4, dtype=torch.uint8),
    )
    return FmhaDecodeWrapper(
        workspace_buffer=torch.zeros(4, dtype=torch.uint8),
        num_qo_heads=16, num_kv_heads=4, head_dim=256, page_size=page_size,
        batch_size=batch_size, max_seq_len=max_seq_len, device=torch.device("cpu"),
    )


def test_plan_stages_each_row_pages_and_lengths(monkeypatch):
    wrapper = _cpu_wrapper(monkeypatch)
    wrapper.plan(
        paged_kv_indptr=torch.tensor([0, 2, 3, 6], dtype=torch.int32),
        paged_kv_indices=torch.tensor([7, 8, 0, 4, 5, 6], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([10, 1, 128], dtype=torch.int32),
        kv_lens=torch.tensor([138, 1, 384], dtype=torch.int32),
        window_left=5,
    )
    table = wrapper._table.numpy()
    assert table[0, :2].tolist() == [7, 8]
    assert table[1, 0] == 0
    assert table[2, :3].tolist() == [4, 5, 6]
    assert wrapper._lens.tolist() == [138, 1, 384]
    assert wrapper._window_left == 5


def test_plan_derives_lengths_when_not_given(monkeypatch):
    wrapper = _cpu_wrapper(monkeypatch, batch_size=2)
    wrapper.plan(
        paged_kv_indptr=torch.tensor([0, 2, 3], dtype=torch.int32),
        paged_kv_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([5, 9], dtype=torch.int32),
    )
    assert wrapper._lens.tolist() == [133, 9]


def test_table_is_wide_enough_for_the_longest_request(monkeypatch):
    wrapper = _cpu_wrapper(monkeypatch, batch_size=1, max_seq_len=1000)
    assert wrapper._max_seq_len == 1024
    assert wrapper._table.shape == (1, 1024 // 128 + 1)
    indices = np.arange(8, dtype=np.int32)
    wrapper.plan(
        paged_kv_indptr=torch.tensor([0, 8], dtype=torch.int32),
        paged_kv_indices=torch.from_numpy(indices),
        paged_kv_last_page_len=torch.tensor([128], dtype=torch.int32),
    )
    assert wrapper._table[0, :8].tolist() == indices.tolist()


def test_plan_refuses_another_batch_size(monkeypatch):
    wrapper = _cpu_wrapper(monkeypatch, batch_size=2)
    with pytest.raises(ValueError, match="captured for 2"):
        wrapper.plan(
            paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
            paged_kv_indices=torch.tensor([1], dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        )


def test_a_kernel_that_will_not_launch_says_no(monkeypatch):
    wrapper = _cpu_wrapper(monkeypatch)

    def missing(self, q, kv_cache_layer):
        raise RuntimeError("Missing TRTLLM-GEN kernel (decode)")

    monkeypatch.setattr(FmhaDecodeWrapper, "run", missing)
    assert not wrapper.has_kernel(256, 16)


class _Recorder:
    def __init__(self, has_kernel=True, **kwargs):
        self.kwargs = kwargs
        self._has = has_kernel

    def has_kernel(self, head_dim, num_qo_heads):
        return self._has


def _manager(monkeypatch, decode_fmha, has_kernel=True, applies=True, dtype=torch.bfloat16, window=None):
    built = {}

    def decode(**kwargs):
        built["paged"] = kwargs
        return SimpleNamespace(kind="paged")

    monkeypatch.setattr(fi_mod, "FlashInferDecodeWrapper", decode)
    monkeypatch.setattr(
        fi_mod, "FmhaDecodeWrapper", lambda **kw: _Recorder(has_kernel=has_kernel, **kw),
    )
    monkeypatch.setattr(fi_mod, "fmha_decode_applies", lambda device: applies)
    manager = FlashInferManager(
        kv_cache="kv", device=torch.device("cpu"), dtype=dtype,
        kv_config=PagedKVConfig(
            num_layers=1, num_kv_heads=4, head_dim=256, max_seq_len=1024, num_qo_heads=16,
        ),
        decode_fmha=decode_fmha,
        sliding_window=window,
    )
    monkeypatch.setattr(manager, "_workspaces", SimpleNamespace(get=lambda *a: torch.zeros(1)))
    return manager, built


def _decode_lease(bs=16):
    return SlotLease(bucket=BucketKey(graph_walk="decode", bs=bs, num_tokens=bs), slot=0)


def test_captured_decode_takes_fmha_when_asked(monkeypatch):
    manager, built = _manager(monkeypatch, decode_fmha=True)
    wrapper = manager._cg_wrapper(_decode_lease(), "main", 16, False)
    assert isinstance(wrapper, _Recorder)
    assert wrapper.kwargs["batch_size"] == 16 and wrapper.kwargs["max_seq_len"] == 1024
    assert not built


@pytest.mark.parametrize(
    "decode_fmha, has_kernel, applies",
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_captured_decode_keeps_the_paged_kernel_otherwise(monkeypatch, decode_fmha, has_kernel, applies):
    manager, built = _manager(monkeypatch, decode_fmha, has_kernel, applies)
    wrapper = manager._cg_wrapper(_decode_lease(), "main", 16, False)
    assert wrapper.kind == "paged" and built


@pytest.mark.parametrize("kwargs", [dict(dtype=torch.float16), dict(window=8)])
def test_other_dtypes_and_sliding_windows_keep_the_paged_kernel(monkeypatch, kwargs):
    manager, built = _manager(monkeypatch, decode_fmha=True, **kwargs)
    wrapper = manager._cg_wrapper(_decode_lease(), "main", 16, False)
    assert wrapper.kind == "paged" and built


def test_yaml_can_turn_it_off_and_it_defaults_off():
    assert not AttentionConfig(kv_cache="kv").decode_fmha
    spec = AttentionSpec(
        resource_key="attn", nodes={"LLM"}, config=AttentionConfig(kv_cache="kv", decode_fmha=True),
    )
    spec.apply_yaml_overrides(decode_fmha=False)
    assert not spec.config.decode_fmha


@_needs_gpu
@pytest.mark.parametrize("batch", [1, 16, 64])
def test_fmha_matches_the_paged_kernel(batch):
    from mstar.engine.resources.attn.wrappers import FlashInferDecodeWrapper

    torch.manual_seed(batch)
    heads, kv_heads, dim, page = 16, 4, 256, 128
    length = 640
    per_row = (length + page - 1) // page
    arena = torch.randn(
        per_row * batch + 4, 2, page, kv_heads, dim, dtype=torch.bfloat16, device=_CUDA,
    ) * 0.1
    q = torch.randn(batch, heads, dim, dtype=torch.bfloat16, device=_CUDA)
    indptr = torch.arange(batch + 1, dtype=torch.int32) * per_row
    indices = torch.randperm(per_row * batch).to(torch.int32)
    last = torch.full((batch,), length - (per_row - 1) * page, dtype=torch.int32)
    lens = torch.full((batch,), length, dtype=torch.int32)
    workspace = torch.zeros(128 << 20, dtype=torch.uint8, device=_CUDA)
    fmha = FmhaDecodeWrapper(
        workspace_buffer=workspace, num_qo_heads=heads, num_kv_heads=kv_heads, head_dim=dim,
        page_size=page, batch_size=batch, max_seq_len=32768, device=_CUDA,
    )
    assert fmha.has_kernel(dim, heads)
    fmha.plan(indptr, indices, last, kv_lens=lens)
    paged = FlashInferDecodeWrapper(
        workspace_buffer=torch.zeros(128 << 20, dtype=torch.uint8, device=_CUDA),
        num_qo_heads=heads, num_kv_heads=kv_heads, head_dim=dim, page_size=page,
        batch_size=batch, max_num_pages=per_row * batch + 4, accelerator_graph=True, device=_CUDA,
    )
    paged.plan(indptr, indices, last, kv_lens=lens)
    got = fmha.run(q, arena).float()
    ref = paged.run(q, arena).float()
    assert (got - ref).abs().max() < 2e-3
