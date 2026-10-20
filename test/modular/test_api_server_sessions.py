"""APIServer session plumbing: the session chunk, and the result loop.

``submit_request`` reports the session on the request's first chunk and carries
the flags to the data worker; the result loop is what releases a session when
its request finishes and what lifts the tombstone when the conductor says the
state is gone.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from types import SimpleNamespace

from api_server_helpers import make_test_api_server

from mstar.api_server.data_worker import _request_session
from mstar.api_server.request_types import (
    APIServerMessage,
    PreprocessInput,
    RequestComplete,
    RequestFailed,
    SessionTornDown,
)
from mstar.api_server.sessions import SessionRegistry, SessionRequest
from mstar.model.sessions import SessionResourceConfig, SessionsConfig


def _config(**kwargs):
    return SessionsConfig(
        resources={"kv_cache": SessionResourceConfig()}, **kwargs
    )


def _server(config=None, inbox=()):
    torn_down = []
    s = make_test_api_server(
        _liveness_interval_s=1e9,
        sessions=SessionRegistry(config, teardown=torn_down.append),
        _next_session_sweep=1e9,  # the loop's sweep is exercised on its own
    )
    s.torn_down = torn_down
    s.preprocessed = []
    s.inbox = list(inbox)
    s.communicator = SimpleNamespace(
        get_all_new_messages=lambda: s.inbox.pop(0) if s.inbox else [],
    )
    s.preprocess_worker = SimpleNamespace(
        new_request=s.preprocessed.append,
        get_profile_updates=lambda: [],
        get_result_chunks=lambda: [],
        has_pending_tensors=lambda rid: False,
        received_final_chunks=lambda rid, outs: False,
        finished_reading=lambda rid, drained=True: None,
        new_result_tensors=lambda body: None,
        discard_result_tensors=lambda body: None,
        abort_request=lambda rid: None,
    )
    return s


def _one_tick(s):
    """Run the result loop for a single pass."""
    s.running = True
    original = s.communicator.get_all_new_messages

    def _drain():
        s.running = False  # stop after this pass
        return original()

    s.communicator.get_all_new_messages = _drain
    s._process_messages()


# ── submit_request ──────────────────────────────────────────────────────────

def _submit(s, session):
    """Submit ``r0``; returns the key the engine's messages will name."""
    s.submit_request(
        text="hi", input_modalities=["text"], output_modalities=["text"],
        request_id="r0", session=session,
    )
    return s.submission_key("r0")


def _start(s, session_id="s"):
    """Start ``session_id`` with ``r0`` and submit it; returns its key."""
    session = s.sessions.resolve(
        start_session=True, resume_session=False, end_session=False,
        session_id=session_id, session_timeout_s=None, request_id="r0",
    )
    return _submit(s, session)


def test_the_session_is_reported_as_the_first_chunk():
    s = _server(_config())
    session = s.sessions.resolve(
        start_session=True, resume_session=False, end_session=False,
        session_id=None, session_timeout_s=None, request_id="r0",
    )

    key = _submit(s, session)

    [chunk] = s.pending_requests[key].chunks
    assert chunk.modality == "session"
    assert chunk.data.decode() == session.session_id
    assert chunk.metadata["session_id"] == session.session_id
    assert chunk.metadata["created"] is True
    # the client sees its id; downstream gets the internal key
    assert session.key != session.session_id
    assert s.preprocessed[0].session_id == session.key


def test_the_flags_reach_the_data_worker():
    s = _server(_config())
    s.sessions.resolve(
        start_session=True, resume_session=False, end_session=False,
        session_id="s", session_timeout_s=None, request_id="r0",
    )
    session = SessionRequest(
        session_id="s", key="s.k", resumed=True, end_session=True,
    )

    _submit(s, session)

    [preprocess_input] = s.preprocessed
    assert preprocess_input.session_id == "s.k"
    assert preprocess_input.resumed is True
    assert preprocess_input.end_session is True
    # end_session holds the tombstone from submit, not from completion
    assert s.sessions.snapshot()[0]["closing"] is True


def _preprocess_input(**session):
    return PreprocessInput(
        request_id="r0", text="hi", file_paths=None,
        input_modalities=["text"], output_modalities=["text"], model_kwargs={},
        **session,
    )


def test_the_data_worker_hands_the_model_the_session():
    # what `process_prompt` reads to tell a resuming turn from an opening one
    session = _request_session(_preprocess_input(
        session_id="s", resumed=True, end_session=True,
    ))

    assert session.session_id == "s"
    assert session.resumed is True
    assert session.started is False
    assert session.end_session is True


def test_the_data_worker_hands_the_model_no_session_without_one():
    assert _request_session(_preprocess_input()) is None


def test_an_opening_turn_reaches_the_model_as_one():
    session = _request_session(_preprocess_input(session_id="s"))

    assert session.started is True
    assert session.resumed is False


def test_a_sessionless_request_emits_no_session_chunk():
    s = _server(_config())

    key = _submit(s, None)

    assert s.pending_requests[key].chunks == []
    assert s.preprocessed[0].session_id is None


# ── the result loop ─────────────────────────────────────────────────────────

def test_the_session_tracks_the_submission_not_the_client_id():
    """The engine's messages name the submission key, so that is what the
    session must hold as its request in flight."""
    s = _server(_config())

    key = _start(s)

    assert key != "r0"
    assert s.sessions.snapshot()[0]["active_request_ids"] == [key]


def test_completion_releases_the_session_for_the_next_request():
    s = _server(_config())
    key = _start(s)
    s.inbox = [[APIServerMessage(
        message_type="request_complete",
        body=RequestComplete(
            request_id=key, final_outputs={"out": None},
            conductor_ingest_time=0.0, conductor_finish_time=1.0,
        ),
    )]]

    _one_tick(s)

    assert s.sessions.snapshot()[0]["active_request_ids"] == []
    assert s.torn_down == []


def test_a_failure_tears_the_session_down():
    s = _server(_config())
    key = _start(s)
    s.inbox = [[APIServerMessage(
        message_type="request_failed",
        body=RequestFailed(request_id=key, error_message="engine blew up"),
    )]]

    _one_tick(s)

    assert s.torn_down == [s.sessions._sessions["s"].key]
    assert s.sessions.snapshot()[0]["closing"] is True


def test_the_conductor_s_ack_lifts_the_tombstone():
    s = _server(_config())
    session = s.sessions.resolve(
        start_session=True, resume_session=False, end_session=False,
        session_id="s", session_timeout_s=None, request_id="r0",
    )
    s.sessions.finish_request("r0")
    s.sessions.delete("s")
    s.inbox = [[APIServerMessage(
        message_type="session_torn_down",
        body=SessionTornDown(session_id=session.key),
    )]]

    _one_tick(s)

    assert s.sessions.snapshot() == []


def test_an_abandoned_request_takes_its_session_with_it():
    s = _server(_config())
    _start(s)

    s.abort_request("r0")

    assert s.torn_down == [s.sessions._sessions["s"].key]
