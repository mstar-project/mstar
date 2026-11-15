"""An unguided Bagel row batched with a guided one holds no guidance pages."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext
from mstar.engine.state.adapters import node_handles, node_resources
from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.bagel.config import BagelAutoEncoderConfig, BagelModelConfig, BagelViTConfig
from mstar.model.bagel.submodules import LLMSubmodule
from mstar.model.submodule_base import ARNodeInputs, engine_step

PAGE_SIZE = 16


@pytest.fixture(autouse=True)
def _no_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", lambda *args, **kwargs: SimpleNamespace(
        remove_request=lambda rid: None,
    ))


def _model(config: BagelModelConfig | None = None) -> BagelModel:
    model = BagelModel.__new__(BagelModel)
    model.config = config or BagelModelConfig(vae_config=BagelAutoEncoderConfig(), vit_config=BagelViTConfig())
    model._has_cfg_parallel = False
    return model


class _StubResource:
    """Stands in for any resource the LLM's handles read at bind."""

    def layer_view(self, i):
        return torch.empty(0)


def _handles(config: BagelModelConfig | None = None) -> dict:
    """The LLM node's handles, over stand-ins: they expand its steps, and the tests run the real resources."""
    model = _model(config)
    return node_handles(model, "LLM", {spec.resource_key: _StubResource() for spec in node_resources(model)})


def _llm() -> LLMSubmodule:
    llm = LLMSubmodule(
        language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=None), lm_head=None), llm2vae=None,
        vae2llm=None, time_embedder=None, latent_pos_embed=None, config=SimpleNamespace(),
        boi_token_id=1, eoi_token_id=2, bos_token_id=3, eos_token_id=4,
    )
    llm.bind_node_state(_handles())
    return llm


def _kv_step(llm: LLMSubmodule, walk: str, request_ids: list, inputs: list, **kwargs) -> KVStep:
    """The KV step the engine expands ``llm``'s declaration on ``walk`` into."""
    return engine_step(llm, graph_walk=walk, request_ids=request_ids, inputs=inputs, **kwargs).steps["kv"]


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
    step = _kv_step(llm, walk, ["guided", "plain"], inputs, per_request_info=INFO)
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
    kv._cg_max_seq_len = 4096
    _step(kv, llm, "prefill_text", 40)
    dropped, clear_preplan = [], kv.clear_preplan

    def _clear_preplan(promoted=False):
        dropped.append(not promoted)
        clear_preplan(promoted)

    monkeypatch.setattr(kv, "clear_preplan", _clear_preplan)
    rows = [*INFO, "pad0"]
    template = [ARNodeInputs(input_seq_len=1, resource_step_info=True) for _ in rows]
    real = [ARNodeInputs(input_seq_len=1, resource_step_info=rid != "plain") for rid in rows]
    # the guidance-on decode capture plans each row on main and cfg_img
    bucket = BucketKey(graph_walk="decode", bs=len(rows), num_tokens=2 * len(rows))

    for t in range(PAGE_SIZE + 1):
        ctx = StepContext(request_ids=tuple(INFO), graph_walk="decode", slot=t % 2, capture=False,
                          slot_lease=SlotLease(slot=t % 2, bucket=bucket))
        ctx.set_padded_rids(rows)
        for ctx.is_preplan, inputs in ((True, template), (False, real)):
            step = _kv_step(llm, "decode", rows, inputs, per_request_info=INFO)
            assert kv.admit(step, ctx).ok
            kv.plan(step, ctx)
        kv.commit(step, ctx)

    assert not any(dropped), "the pre-plan over the template rows was dropped for the step over the real rows"
    assert set(kv._streams["plain"]) == {"main"}
    kv.assert_pages_conserved()
