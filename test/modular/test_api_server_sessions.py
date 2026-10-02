"""APIServer session plumbing: the session chunk, and the result loop.

``submit_request`` reports the session on the request's first chunk and carries
the flags to the data worker; the result loop is what releases a session when
its request finishes and what lifts the tombstone when the conductor says the
state is gone.
"""

from __future__ import annotations

import collections
import sys
import threading

sys.path.insert(0, ".")

from types import SimpleNamespace

from mstar.api_server.entrypoint import APIServer, PendingRequest
from mstar.api_server.request_types import (
    APIServerMessage,
    RequestComplete,
    RequestFailed,
    SessionTornDown,
)
from mstar.api_server.sessions import SessionRegistry, SessionRequest
from mstar.model.sessions import SessionResourceConfig, SessionsConfig
from mstar.profile.format import RequestProfile, RequestTiming


def _config(**kwargs):
    return SessionsConfig(
        resources={"kv_cache": SessionResourceConfig()}, **kwargs
    )


def _server(config=None, inbox=()):
    s = object.__new__(APIServer)
    s.pending_requests = {}
    s.recently_completed = collections.OrderedDict()
    s._recently_completed_ttl = 15.0
    s.request_lock = threading.Lock()
    s.running = True
    s.conductor_proc = None
    s.fatal_error = None
    s.on_fatal = None
    s._liveness_interval_s = 1e9
    s.log_stats = False
    s.torn_down = []
    s.sessions = SessionRegistry(config, teardown=s.torn_down.append)
    s._next_session_sweep = 1e9  # the loop's sweep is exercised on its own
    s._session_sweep_interval_s = 1.0
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


def _pending(s, rid):
    s.pending_requests[rid] = PendingRequest(
        streaming=True,
        input_modalities=["text"],
        output_modalities=["text"],
        profile=RequestProfile(rid=rid, timing=RequestTiming(recv_time=0.0)),
    )
    return s.pending_requests[rid]


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
    return s.submit_request(
        text="hi", input_modalities=["text"], output_modalities=["text"],
        request_id="r0", session=session,
    )


def test_the_session_is_reported_as_the_first_chunk():
    s = _server(_config())
    session = s.sessions.resolve(
        start_session=True, resume_session=False, end_session=False,
        session_id=None, session_timeout_s=None, request_id="r0",
    )

    _submit(s, session)

    [chunk] = s.pending_requests["r0"].chunks
    assert chunk.modality == "session"
    assert chunk.data.decode() == session.session_id
    assert chunk.metadata["session_id"] == session.session_id
    assert chunk.metadata["created"] is True


def test_the_flags_reach_the_data_worker():
    s = _server(_config())
    s.sessions.resolve(
        start_session=True, resume_session=False, end_session=False,
        session_id="s", session_timeout_s=None, request_id="r0",
    )
    session = SessionRequest(session_id="s", end_session=True)

    _submit(s, session)

    [preprocess_input] = s.preprocessed
    assert preprocess_input.session_id == "s"
    assert preprocess_input.end_session is True
    # end_session holds the tombstone from submit, not from completion
    assert s.sessions.snapshot()[0]["closing"] is True


def test_a_sessionless_request_emits_no_session_chunk():
    s = _server(_config())

    _submit(s, None)

    assert s.pending_requests["r0"].chunks == []
    assert s.preprocessed[0].session_id is None


# ── the session a model reads ───────────────────────────────────────────────
#
# A model takes its session off model_kwargs, the way it takes a keyed prefix:
# the server fills the names in, so a model written against the session fields
# needs no new signature. Only the server may set them.

def test_a_model_reads_the_validated_session_off_model_kwargs():
    s = _server(_config())
    session = s.sessions.resolve(
        start_session=True, resume_session=False, end_session=False,
        session_id="s", session_timeout_s=None, request_id="r0",
    )

    s.submit_request(
        text="hi", input_modalities=["text"], output_modalities=["text"],
        request_id="r0", model_kwargs={"voice": "tara"}, session=session,
    )

    [sent] = s.preprocessed
    assert sent.model_kwargs == {
        "voice": "tara", "session_id": "s",
        "resume_session": False, "end_session": False,
    }


def test_a_resumed_request_says_so_to_the_model():
    s = _server(_config())
    s.sessions.resolve(
        start_session=True, resume_session=False, end_session=False,
        session_id="s", session_timeout_s=None, request_id="r0",
    )
    s.sessions.finish_request("r0")
    session = s.sessions.resolve(
        start_session=False, resume_session=True, end_session=True,
        session_id="s", session_timeout_s=None, request_id="r1",
    )

    s.submit_request(
        text="hi", input_modalities=["text"], output_modalities=["text"],
        request_id="r1", session=session,
    )

    [sent] = s.preprocessed
    assert sent.model_kwargs["resume_session"] is True
    assert sent.model_kwargs["end_session"] is True


def test_a_client_cannot_name_a_session_through_model_kwargs():
    # it would otherwise be served another session's state
    s = _server(_config())

    s.submit_request(
        text="hi", input_modalities=["text"], output_modalities=["text"],
        request_id="r0", session=None,
        model_kwargs={"session_id": "someone-elses", "resume_session": True},
    )

    [sent] = s.preprocessed
    assert sent.model_kwargs == {}


def test_a_sessionless_request_keeps_its_kwargs_untouched():
    s = _server(_config())

    s.submit_request(
        text="hi", input_modalities=["text"], output_modalities=["text"],
        request_id="r0", model_kwargs={"voice": "tara"}, session=None,
    )

    assert s.preprocessed[0].model_kwargs == {"voice": "tara"}


# ── the result loop ─────────────────────────────────────────────────────────

def test_completion_releases_the_session_for_the_next_request():
    s = _server(_config())
    s.sessions.resolve(
        start_session=True, resume_session=False, end_session=False,
        session_id="s", session_timeout_s=None, request_id="r0",
    )
    _pending(s, "r0")
    s.inbox = [[APIServerMessage(
        message_type="request_complete",
        body=RequestComplete(
            request_id="r0", final_outputs={"out": None},
            conductor_ingest_time=0.0, conductor_finish_time=1.0,
        ),
    )]]

    _one_tick(s)

    assert s.sessions.snapshot()[0]["active_request_ids"] == []
    assert s.torn_down == []


def test_a_failure_tears_the_session_down():
    s = _server(_config())
    s.sessions.resolve(
        start_session=True, resume_session=False, end_session=False,
        session_id="s", session_timeout_s=None, request_id="r0",
    )
    _pending(s, "r0")
    s.inbox = [[APIServerMessage(
        message_type="request_failed",
        body=RequestFailed(request_id="r0", error_message="engine blew up"),
    )]]

    _one_tick(s)

    assert s.torn_down == ["s"]
    assert s.sessions.snapshot()[0]["closing"] is True


def test_the_conductor_s_ack_lifts_the_tombstone():
    s = _server(_config())
    s.sessions.resolve(
        start_session=True, resume_session=False, end_session=False,
        session_id="s", session_timeout_s=None, request_id="r0",
    )
    s.sessions.finish_request("r0")
    s.sessions.delete("s")
    s.inbox = [[APIServerMessage(
        message_type="session_torn_down", body=SessionTornDown(session_id="s"),
    )]]

    _one_tick(s)

    assert s.sessions.snapshot() == []


def test_an_abandoned_request_takes_its_session_with_it():
    s = _server(_config())
    s.sessions.resolve(
        start_session=True, resume_session=False, end_session=False,
        session_id="s", session_timeout_s=None, request_id="r0",
    )
    _pending(s, "r0")

    s.abort_request("r0")

    assert s.torn_down == ["s"]
