"""An unguided Bagel row batched with a guided one holds no guidance pages."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext
from mstar.model.bagel.submodules import LLMSubmodule
from mstar.model.submodule_base import ARNodeInputs

PAGE_SIZE = 16


@pytest.fixture(autouse=True)
def _no_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", lambda *args, **kwargs: SimpleNamespace(
        remove_request=lambda rid: None,
    ))


def _llm() -> LLMSubmodule:
    return LLMSubmodule(
        language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=None), lm_head=None), llm2vae=None,
        vae2llm=None, time_embedder=None, latent_pos_embed=None, config=SimpleNamespace(cfg_prune=True),
        boi_token_id=1, eoi_token_id=2, bos_token_id=3, eos_token_id=4,
    )


def _manager() -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, max_num_pages=64, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    for rid in ("guided", "plain"):
        kv.ingest_request(rid, KVReqConfig(needed_labels=["main", "cfg_text", "cfg_img"]))
    return kv


INFO = {rid: SimpleNamespace(step_metadata={"requires_cfg": rid == "guided"}) for rid in ("guided", "plain")}


def _step(kv: KVManager, llm: LLMSubmodule, walk: str, rows: int):
    inputs = [ARNodeInputs(input_seq_len=rows, resource_step_info=guided) for guided in (True, False)]
    if walk == "prefill_vit":
        for inp in inputs:
            inp.custom_pos_ids = dict.fromkeys(("main", "cfg_text", "cfg_img"), torch.zeros(rows, dtype=torch.int32))
    step = llm.declare_step(walk, ["guided", "plain"], inputs, per_request_info=INFO).steps["kv"]
    ctx = StepContext(request_ids=("guided", "plain"), graph_walk=walk, slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def test_an_unguided_row_holds_no_guidance_pages():
    kv, llm = _manager(), _llm()

    _step(kv, llm, "prefill_text", 40)
    _step(kv, llm, "prefill_vit", 20)
    for _ in range(PAGE_SIZE + 1):
        _step(kv, llm, "decode", 1)

    assert set(kv._streams["plain"]) == {"main"}, "the unguided row took pages on a guidance label"
    assert kv._streams["plain"]["main"].stored_len == 40 + 20 + PAGE_SIZE + 1, "the unguided row's main lost writes"
    guided = kv._streams["guided"]
    assert guided["cfg_img"].stored_len == 40 + PAGE_SIZE + 1, "the guided row's cfg_img lost writes"
    assert guided["cfg_text"].stored_len == 40 + 20, "the guided row's cfg_text missed a fork"
    kv.assert_pages_conserved()


def test_a_pre_plan_over_the_template_rows_is_the_step_that_runs(monkeypatch):
    kv, llm = _manager(), _llm()
    kv._plan_cache_mode, kv._cg_max_seq_len = 2, 4096
    _step(kv, llm, "prefill_text", 40)
    dropped, clear_preplan = [], kv.clear_preplan

    def _clear_preplan(promoted=False):
        dropped.append(not promoted)
        clear_preplan(promoted)

    monkeypatch.setattr(kv, "clear_preplan", _clear_preplan)
    rows = [*INFO, "pad0"]
    template = [ARNodeInputs(input_seq_len=1, resource_step_info=True) for _ in rows]
    real = [ARNodeInputs(input_seq_len=1, resource_step_info=rid != "plain") for rid in rows]
    bucket = BucketKey(graph_walk="decode", bs=len(rows), num_tokens=len(rows))

    for t in range(PAGE_SIZE + 1):
        ctx = StepContext(request_ids=tuple(INFO), graph_walk="decode", slot=t % 2, capture=False,
                          slot_lease=SlotLease(slot=t % 2, bucket=bucket))
        ctx.set_padded_rids(rows)
        for ctx.is_preplan, inputs in ((True, template), (False, real)):
            step = llm.declare_step("decode", rows, inputs, per_request_info=INFO).steps["kv"]
            assert kv.admit(step, ctx).ok
            kv.plan(step, ctx)
        kv.commit(step, ctx)

    assert not any(dropped), "the pre-plan over the template rows was dropped for the step over the real rows"
    assert set(kv._streams["plain"]) == {"main"} and kv.plan_cache_stats["mismatch"] == 0
    kv.assert_pages_conserved()
