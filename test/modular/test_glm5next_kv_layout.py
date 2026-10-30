"""GLM-5.3-Flash lays out its one paged label, so the preprocess worker and the conductor hold
each request to the cache rows its prompt leaves (``docs/adding_models.rst``, Paged KV layouts).
Under MTP every verify step keeps its whole block, so a decode step counts k + 1 rows."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import yaml
from test_kv_layout_intake import _Comm, _preprocess, _ScriptDoneError

import mstar.conductor.conductor as conductor_mod
from mstar.conductor.conductor import Conductor
from mstar.model.glm5_next.config import KV_CACHE
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel
from mstar.utils.ipc_format import ConductorMessage, ConductorMessageType, NewRequestConductor, SetupDone
from mstar.validation.resources import _undeclared_pools, _unknown_prefix_walks

PROMPT = len("hello")  # _preprocess's prompt, a token per byte

_K = pytest.mark.parametrize("k", [0, 3], ids=["plain", "mtp"])


def _model(**kwargs) -> Glm5NextModel:
    return Glm5NextModel(model_path_hf="", tokenizer_mode="byte", **kwargs)


def _conduct(model: Glm5NextModel, model_kwargs: dict, tmp_path, monkeypatch) -> list:
    """What the real conductor sends for one request: a NEW_REQUEST, or its refusal."""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({
        "max_seq_len": model.config.max_seq_len, "node_groups": [{"node_names": ["LLM"], "ranks": [0]}],
    }))
    comm = _Comm()
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: comm)
    conductor = Conductor(model, str(path), socket_path_prefix=str(tmp_path / "s"))
    comm.batches.extend([
        [ConductorMessage(ConductorMessageType.SETUP_DONE, SetupDone(w)) for w in conductor.worker_ids],
        [ConductorMessage(ConductorMessageType.NEW_REQUEST, NewRequestConductor(
            request_id="r", initial_signals={}, initial_input_modalities=["text"],
            initial_output_modalities=["text"], input_metadata={}, model_kwargs=model_kwargs,
        ))],
    ])

    def poll():
        if not comm.batches:
            raise _ScriptDoneError

    conductor._poll_worker_liveness = poll
    with pytest.raises(_ScriptDoneError):
        conductor.run()
    return comm.sent


def _dispatched(sent: list):
    [new_request] = [msg.body for msg in sent if getattr(msg.message_type, "name", "") == "NEW_REQUEST"]
    return new_request


def _failed(sent: list) -> list[int]:
    return [msg.body.status for msg in sent if getattr(msg, "message_type", None) == "request_failed"]


@_K
def test_every_paged_label_is_declared_at_load(k):
    model = _model(mtp_num_draft_tokens=k)

    assert list(_undeclared_pools(model.get_node_resources(), model)) == []
    assert list(_unknown_prefix_walks(model)) == []


@_K
def test_a_decode_step_grows_the_label_by_its_block(k):
    model = _model(mtp_num_draft_tokens=k)

    layout = model.process_prompt("hello", ["text"], ["text"]).metadata["kv_layout"][KV_CACHE]["main"]

    assert [(span.walk, span.length) for span in layout.spans] == [("prefill", PROMPT)]
    assert layout.growth == (("decode_loop", k + 1),)


def test_a_request_decodes_one_step_fewer_than_its_max_tokens(tmp_path, monkeypatch):
    model = _model()

    new_request = _dispatched(_conduct(model, _preprocess(model, {"max_output_tokens": 30}), tmp_path, monkeypatch))

    # the prefill emits the first token
    assert new_request.loop_caps["decode_loop"] == 29
    assert new_request.request_info.max_tokens == 30


@_K
def test_a_request_that_asks_no_length_fills_the_cache_rows(tmp_path, monkeypatch, k):
    model = _model(mtp_num_draft_tokens=k)
    rows, block = model.config.kv_rows, k + 1

    new_request = _dispatched(_conduct(model, _preprocess(model, {}), tmp_path, monkeypatch))

    cap = new_request.loop_caps["decode_loop"]
    assert PROMPT + block * cap <= rows < PROMPT + block * (cap + 1)
    assert cap <= model.max_decode_steps()
    assert new_request.request_info.max_tokens == cap + 1


@_K
@pytest.mark.parametrize("max_tokens", [1024, 1600, 2048 - PROMPT])
def test_mtp_admits_what_the_context_holds(tmp_path, monkeypatch, k, max_tokens):
    """A request the context holds is dispatched with or without MTP: the rows its rejected
    drafts could take never refuse it first (1024 was past the rows, 1600 past the loop cap)."""
    model = _model(mtp_num_draft_tokens=k)

    sent = _conduct(model, _preprocess(model, {"max_output_tokens": max_tokens}), tmp_path, monkeypatch)

    assert _failed(sent) == []
    assert _dispatched(sent).loop_caps["decode_loop"] == max_tokens - 1


@_K
@pytest.mark.parametrize("past", [0, 1])
def test_max_tokens_past_the_rows_its_prompt_leaves_is_a_400(tmp_path, monkeypatch, past, k):
    model = _model(mtp_num_draft_tokens=k)
    # the prompt's rows and a block per decode step fill the cache; the last token is never stored
    fits = (model.config.kv_rows - PROMPT) // (k + 1) + 1

    sent = _conduct(model, _preprocess(model, {"max_output_tokens": fits + past}), tmp_path, monkeypatch)

    assert _failed(sent) == ([400] if past else []), f"max_tokens {fits + past}"
    if not past:
        _dispatched(sent)
