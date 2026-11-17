"""Qwen3 with an EAGLE-3 drafter through a host pool, on the CPU: a request offloaded and reloaded, and a request
that reuses its prefix from the host.

An EAGLE-3 drafter's K/V at a page's last slot reads the next page's first token, so its pool names page ``i`` by
the key of page ``i + 1`` (`PagedKVConfig.names_ahead`). An offload files the pages it carries to the host under
those names (`KVManager._offload_keys`), and a later request's host lookup reads them under the same names
(`KVManager._fetch_from_host`). A verifying node keys no verify walk, so a request's prefix chain goes at its first
verify step: only an offload between its prompt step and that step files keys, and A is offloaded there.

The host pool is `CPUPagePool` on unpinned memory, copying synchronously. The EAGLE-3 pool has one storage (the
drafter's layers share the target's page geometry), so on a device its copies are main's whole-page copier, and what
names_ahead changes is host-side.
"""
from __future__ import annotations

from typing import NamedTuple
from unittest import mock

import pytest
import torch
from qwen3_cpu import (
    PAGE,
    Driver,
    cpu_specs,
    drafter_first_layer,
    engine,
    make_model,
    patch_cpu,
    prompt_kv,
    write_checkpoints,
)
from test_eagle3_config import _cpu_eagle3

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.kv import cpu_page_pool
from mstar.engine.resources.kv import manager as manager_mod

A = list(range(1, 15))
# shares A's first two pages and differs on its third: page 0 is named by page 1's key, which B shares, so B
# reuses page 0 alone
B = A[:8] + [20, 21, 22, 23]
MAX = 12


class _Landed:
    def query(self):
        return True

    def synchronize(self):
        pass


class _SyncHostPool(cpu_page_pool.CPUPagePool):
    """The host pool on unpinned memory, each copy done when it is queued."""

    def __init__(self, config, kv_cache, arena):
        empty = torch.empty

        def unpinned(*args, pin_memory=False, **kwargs):
            return empty(*args, **kwargs)

        with mock.patch.object(torch, "empty", unpinned):
            super().__init__(config, kv_cache, arena)

    def copy_out(self, gpu_kv_cache, copies):
        for host, device in self._paired(gpu_kv_cache):
            for dst, src in zip(copies.cpu, copies.gpu, strict=True):
                host[:, dst].copy_(device[:, src])

    write = copy_out

    def copy_in(self, gpu_kv_cache, copies):
        for host, device in self._paired(gpu_kv_cache):
            for dst, src in zip(copies.gpu, copies.cpu, strict=True):
                device[:, dst].copy_(host[:, src])

    def record(self):
        return _Landed()

    record_write = record

    def sync(self):
        pass


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    patch_cpu(monkeypatch)
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    monkeypatch.setattr(manager_mod, "CPUPagePool", _SyncHostPool)


class _Run(NamedTuple):
    # B's drafter K/V after its prompt step, the prompt tokens it reused, and the pages of those copied from the host
    b_kv: torch.Tensor | None
    b_reused: int | None
    b_from_host: int | None
    # A's drafter K/V once back on the device, and its reply
    a_kv: torch.Tensor
    reply: list[int]


def _run(tmp_path, offload: bool, b: list[int] | None) -> _Run:
    """A keyed and its prompt step run, then, with ``offload``, A moved to the host and the device index emptied,
    so nothing of A stays on the device; ``b`` (if any) prefilled meanwhile, its readiness checked until its host
    copy lands, as the worker's scan does; then A reloaded and decoded to its end."""
    model = make_model(write_checkpoints(tmp_path)[0], {**_cpu_eagle3(tmp_path), "mode": "synchronous"},
                       max_seq_len=512, accelerator_graph=False)
    specs = cpu_specs(model, prefix_cache=True)
    for spec in specs:
        if spec.resource_key == "kv_cache":
            spec.config.cpu_offload_pages = 64
    eng = engine(model, prefix_cache=True, specs=specs)
    kv, layer = eng._resources["kv_cache"], [drafter_first_layer(model)]
    assert kv.config.names_ahead == 1 and len(kv.kv_cache.storages) == 1
    driver = Driver(eng, model)
    driver.add(1, A, MAX, keyed=True)
    driver.prefill([1], {1: A})
    if offload:
        assert eng.offload_request("LLM", 1) > 0
        # A's two whole prompt pages, each under the next page's key
        assert [len(state.keys) for state in kv._cpu_pool.offloaded[1].values()] == [2]
        kv._evict(kv._index.num_pages())
    b_kv = b_reused = b_from_host = None
    if b is not None:
        before = kv.prefix_tokens.get("LLM", (0, 0))[1]
        driver.add(2, b, MAX, keyed=True)
        info = CurrentForwardPassInfo("2", "prefill_text", 0, 0, MAX)
        assert any(eng.check_ready("LLM", 2, info).outcome.ready for _ in range(4))
        driver.prefill([2], {2: b})
        b_reused = kv.prefix_tokens.get("LLM", (0, 0))[1] - before
        b_from_host = kv._streams[2]["main"].chain.from_host
        b_kv = prompt_kv(eng, 2, len(b), layer).float().cpu()
        driver.run([2])
    if offload:
        assert eng.reload_request("LLM", 1)
    driver.step([1])
    a_kv = prompt_kv(eng, 1, len(A), layer).float().cpu()
    driver.run([1])
    return _Run(b_kv, b_reused, b_from_host, a_kv, driver.out[1])


def test_an_eagle3_request_offloaded_and_reloaded_drafts_as_one_that_stayed(tmp_path):
    """Once back from the host, A's drafter K/V at every prompt position is the bytes it had on a run that never
    left the device, and its reply is the same."""
    stayed, moved = _run(tmp_path, False, None), _run(tmp_path, True, None)
    assert torch.equal(moved.a_kv, stayed.a_kv)
    assert moved.reply == stayed.reply


def test_a_prefix_reused_from_the_host_is_named_as_on_the_device(tmp_path):
    """B arrives while A is on the host: it reuses A's page 0 alone, copied from the host, as it does from the
    device while A stays, so the host files A's pages under the names the device index gives them and the lookup
    reads them under those. B's drafter K/V is the same either way, and A then comes back and replies as it
    would have."""
    stayed, moved = _run(tmp_path, False, B), _run(tmp_path, True, B)
    assert moved.b_reused == stayed.b_reused == PAGE, "page 0 alone"
    assert (stayed.b_from_host, moved.b_from_host) == (0, 1)
    assert torch.equal(moved.b_kv, stayed.b_kv)
    assert torch.equal(moved.a_kv, stayed.a_kv)
    assert moved.reply == stayed.reply
