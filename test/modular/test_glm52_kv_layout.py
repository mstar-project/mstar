"""GLM-5.2 lays out its one paged label, so the preprocess worker and the conductor hold
each request to the context its prompt leaves (``docs/adding_models.rst``, Paged KV layouts)."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import yaml
from test_kv_layout_intake import _Comm, _preprocess, _ScriptDoneError

import mstar.conductor.conductor as conductor_mod
from mstar.conductor.conductor import Conductor
from mstar.model.glm52.config import KV_RESOURCE
from mstar.model.glm52.glm52_model import Glm52Model
from mstar.utils.ipc_format import ConductorMessage, ConductorMessageType, NewRequestConductor, SetupDone
from mstar.validation.resources import _undeclared_pools, _unknown_prefix_walks

PROMPT = len("hello")  # _preprocess's prompt, a token per byte


def _model(**kwargs) -> Glm52Model:
    return Glm52Model(model_path_hf="", tokenizer_mode="byte", **kwargs)


def _conduct(model: Glm52Model, model_kwargs: dict, tmp_path, monkeypatch) -> list:
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


@pytest.mark.parametrize("kwargs", [{}, {"dsa_long_context": True}], ids=["dense", "dsa"])
def test_every_paged_label_is_declared_at_load(kwargs):
    model = _model(**kwargs)

    assert list(_undeclared_pools(model.get_node_resources(), model)) == []
    assert list(_unknown_prefix_walks(model)) == []


def test_a_request_decodes_one_step_fewer_than_its_max_tokens(tmp_path, monkeypatch):
    model = _model()

    new_request = _dispatched(_conduct(model, _preprocess(model, {"max_output_tokens": 30}), tmp_path, monkeypatch))

    # the prefill emits the first token
    assert new_request.loop_caps["decode_loop"] == 29
    assert new_request.request_info.max_tokens == 30


# the reduced variant's window (index_topk 64) is smaller than its max_seq_len (512); under MTP
# the verify rows past the kept tokens (the layout's lookahead) take their share of it
_WINDOWS = pytest.mark.parametrize(
    "kwargs",
    [
        {}, {"config_variant": "reduced"},
        {"mtp_num_draft_tokens": 3}, {"config_variant": "reduced", "mtp_num_draft_tokens": 2},
    ],
    ids=["full", "reduced", "mtp3", "reduced_mtp2"],
)


@pytest.mark.parametrize("k", [0, 1, 3])
def test_mtp_lays_out_its_verify_rows_as_a_lookahead(k):
    model = _model(mtp_num_draft_tokens=k)

    [row] = _preprocess(model, {"max_output_tokens": 30})["kv_layout"][KV_RESOURCE].values()

    # a slot per kept token, and the lookahead last only under MTP
    assert row[1] == [["decode_loop", 1]]
    assert row[5:] == ([2 * k + 1] if k else [])


@pytest.mark.parametrize(
    "kwargs", [{}, {"dsa_long_context": True}, {"config_variant": "reduced"}], ids=["dense", "dsa", "reduced"],
)
def test_the_kv_pool_holds_exactly_the_window(kwargs):
    # the conductor sizes requests by the pool's max_seq_len, so it must be the window the
    # submodule's preprocess guard holds contexts to
    model = _model(**kwargs)
    [kv] = [spec for spec in model.get_node_resources() if spec.resource_key == KV_RESOURCE]
    assert kv.config.max_seq_len == model.context_limit()


@_WINDOWS
def test_a_request_that_asks_no_length_runs_to_the_window(tmp_path, monkeypatch, kwargs):
    model = _model(**kwargs)
    window = model.context_limit() - model.config.mtp_kv_lookahead

    new_request = _dispatched(_conduct(model, _preprocess(model, {}), tmp_path, monkeypatch))

    assert new_request.loop_caps["decode_loop"] == window - PROMPT
    assert new_request.request_info.max_tokens == window - PROMPT + 1


@_WINDOWS
@pytest.mark.parametrize("past", [0, 1])
def test_max_tokens_past_the_window_its_prompt_leaves_is_a_400(tmp_path, monkeypatch, past, kwargs):
    model = _model(**kwargs)
    # the prompt's rows, one per decode step and the lookahead fill the window; the last token
    # is never stored
    fits = model.context_limit() - model.config.mtp_kv_lookahead - PROMPT + 1

    sent = _conduct(model, _preprocess(model, {"max_output_tokens": fits + past}), tmp_path, monkeypatch)

    failed = [msg.body for msg in sent if getattr(msg, "message_type", None) == "request_failed"]
    assert [body.status for body in failed] == ([400] if past else []), f"max_tokens {fits + past}"
    if not past:
        _dispatched(sent)
