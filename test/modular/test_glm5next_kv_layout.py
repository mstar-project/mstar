"""GLM-5.3-Flash's paged KV layout, through the real conductor: each request is held to the
cache rows its prompt leaves, and under MTP a decode step counts its whole k + 1 block."""

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
def test_the_label_is_declared_and_grows_by_the_block(k):
    model = _model(mtp_num_draft_tokens=k)

    assert list(_undeclared_pools(model.get_node_resources(), model)) == []
    assert list(_unknown_prefix_walks(model)) == []
    layout = model.process_prompt("hello", ["text"], ["text"]).metadata["kv_layout"][KV_CACHE]["main"]
    assert [(span.walk, span.length) for span in layout.spans] == [("prefill", PROMPT)]
    assert layout.growth == (("decode_loop", k + 1),)


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
def test_the_most_tokens_the_context_holds_are_admitted(tmp_path, monkeypatch, k):
    """The rows rejected drafts take never refuse a request the context holds: the room is
    the window's after the prompt, under MTP two verify blocks short of it."""
    model = _model(mtp_num_draft_tokens=k)
    max_tokens = model.config.index_topk - PROMPT - (2 * (k + 1) - 1 if k else 0)

    sent = _conduct(model, _preprocess(model, {"max_output_tokens": max_tokens}), tmp_path, monkeypatch)

    assert _failed(sent) == []
    # the prefill emits the first token
    assert _dispatched(sent).loop_caps["decode_loop"] == max_tokens - 1


@pytest.mark.parametrize("past", [0, 1])
def test_mtp_refuses_a_max_tokens_the_window_cant_hold(past):
    """Under MTP decode stops two verify blocks short of the window, inside the rows the
    conductor holds a request to, so process_prompt refuses a max_tokens past that (a 400)
    rather than cut it short."""
    model = _model(mtp_num_draft_tokens=3)
    prompt = "x" * 1000
    room = model.config.index_topk - 2 * 4 + 1 - len(prompt)
    if past:
        with pytest.raises(ValueError, match=f"emits at most {room} tokens"):
            model.process_prompt(prompt, ["text"], ["text"], max_output_tokens=room + 1)
    else:
        model.process_prompt(prompt, ["text"], ["text"], max_output_tokens=room)


@pytest.mark.parametrize("long_context", [False, True], ids=["window", "long_context"])
def test_mtp_serves_the_max_tokens_decode_emits_with_no_draft_accepted(long_context):
    """The largest max_tokens process_prompt takes under MTP is what _mtp_check_stop lets a
    request emit when every draft is rejected, within the 2048-token window or, with
    dsa_long_context, the rows of a longer one."""
    import re

    from test_glm5next_mtp_cpu import _stop_after

    model = _model(mtp_num_draft_tokens=3, **({"dsa_long_context": True, "max_seq_len": 8192}
                                              if long_context else {}))
    prompt = "x" * 3000 if long_context else "x" * 1000
    with pytest.raises(ValueError, match="emits at most") as refused:
        model.process_prompt(prompt, ["text"], ["text"], max_output_tokens=10 ** 9)
    room = int(re.search(r"at most (-?\d+)", str(refused.value)).group(1))
    assert room > 0
    model.process_prompt(prompt, ["text"], ["text"], max_output_tokens=room)
    cfg = model.config
    emitted = lambda max_tokens: _stop_after(len(prompt), max_tokens, 1, index_topk=cfg.context_limit,  # noqa: E731
                                             kv_rows=cfg.kv_rows)[1]
    assert emitted(room) >= room and emitted(room + 1) < room + 1


@pytest.mark.parametrize("past", [0, 1])
def test_max_tokens_past_the_rows_its_prompt_leaves_is_a_400(tmp_path, monkeypatch, past):
    """Without MTP the conductor refuses it; under MTP the window binds first."""
    model = _model()
    # the prompt's rows and a row per decode step fill the cache; the last token is never stored
    fits = model.config.kv_rows - PROMPT + 1

    sent = _conduct(model, _preprocess(model, {"max_output_tokens": fits + past}), tmp_path, monkeypatch)

    assert _failed(sent) == ([400] if past else []), f"max_tokens {fits + past}"
    if not past:
        _dispatched(sent)
