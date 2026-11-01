"""``/v1/images/generations`` with ``n > 1`` releases every image it submitted.

``create_images`` submits ``n`` engine requests up front and reads them in
order. When one fails, times out, or the handler is cancelled, nothing will
read the others: the ones still running must be aborted on the engine and the
finished ones dropped from ``pending_requests`` (where they would hold their
image bytes until restart). Drives the real handler and the real ``APIServer``
request lifecycle against a scripted engine.
"""

import asyncio
import base64
import time
from types import SimpleNamespace

import pytest
from api_server_helpers import make_test_api_server
from fastapi import HTTPException

from mstar.api_server.openai.serving_images import create_images
from mstar.api_server.request_types import ResultChunk

OK, FAIL, RUNNING, RAISE = "ok", "fail", "running", "raise"


class _ScriptedEngine:
    """Stands in for the preprocess worker. The i-th submitted image finishes
    with bytes (OK), finishes with an error (FAIL), never finishes (RUNNING),
    or makes the submit itself raise (RAISE)."""

    def __init__(self, api, plan):
        self.api = api
        self.plan = plan
        self.submitted = []
        self.seeds = []
        self.aborted = []
        self.reads_done = []

    def new_request(self, input):
        index = len(self.submitted)
        self.submitted.append(input.request_id)
        self.seeds.append((input.model_kwargs or {}).get("seed"))
        outcome = self.plan[index]
        if outcome == RAISE:
            raise RuntimeError("injected submit failure")
        if outcome == RUNNING:
            return
        # What the message thread does once a request completes and drains.
        with self.api.request_lock:
            req = self.api.pending_requests[input.request_id]
            req.chunks.append(ResultChunk(input.request_id, "image", input.request_id.encode()))
            if outcome == FAIL:
                req.error = "injected image failure"
            req.event.set()
            self.api.recently_completed[input.request_id] = time.time()

    def abort_request(self, request_id):
        self.aborted.append(request_id)

    def finished_reading(self, request_id, drained=True):
        self.reads_done.append(request_id)

    def has_pending_tensors(self, request_id):
        return False

    def received_final_chunks(self, request_id, final_outputs):
        return True

    def ids(self, *outcomes):
        plan = self.plan[: len(self.submitted)]
        return sorted(rid for rid, o in zip(self.submitted, plan, strict=True) if o in outcomes)


def _api(plan, timeout_seconds=5.0):
    api = make_test_api_server(timeout_seconds=timeout_seconds)
    engine = _ScriptedEngine(api, plan)
    api.preprocess_worker = engine
    return api, engine


_ADAPTER = SimpleNamespace(
    image_to_request=lambda req, upload_dir: SimpleNamespace(
        text="a lighthouse", file_paths=None, input_modalities=["text"], model_kwargs={"seed": 7},
    )
)


def _create(api, n, raw_request=None):
    return create_images(api, "model", _ADAPTER, SimpleNamespace(n=n), raw_request)


def _assert_nothing_left(api, engine, *, aborted, reads_done):
    """After one message-thread prune tick no API-side state is left, every
    still-running image was aborted once, and every finished one was handed
    back to the engine (READS_DONE) without an abort."""
    with api.request_lock:
        api._prune_recently_completed()
    assert api.pending_requests == {}
    assert not api.recently_completed
    assert sorted(engine.aborted) == aborted
    assert sorted(engine.reads_done) == reads_done


@pytest.mark.parametrize(
    "plan, error",
    [
        # the 2nd image fails at collection: the 3rd finished, the 4th still runs
        ([OK, FAIL, OK, RUNNING], HTTPException),
        # the 3rd submit raises; the first two are already on the engine
        ([RUNNING, RUNNING, RAISE, OK], RuntimeError),
    ],
    ids=["collect", "submit"],
)
def test_failed_image_releases_its_siblings(plan, error):
    api, engine = _api(plan)
    with pytest.raises(error) as exc:
        asyncio.run(_create(api, n=4))
    if error is HTTPException:
        assert (exc.value.status_code, exc.value.detail) == (500, "injected image failure")
    _assert_nothing_left(
        api, engine,
        aborted=engine.ids(RUNNING, RAISE),
        reads_done=engine.ids(OK, FAIL),
    )


def test_timed_out_image_releases_its_siblings():
    api, engine = _api([RUNNING, OK, RUNNING, RUNNING], timeout_seconds=0.2)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(_create(api, n=4))
    assert (exc.value.status_code, exc.value.detail) == (500, "Request timed out")
    _assert_nothing_left(api, engine, aborted=engine.ids(RUNNING), reads_done=engine.ids(OK))


def test_all_images_succeed_unchanged():
    api, engine = _api([OK, OK, OK, OK])
    released = []
    real_release = api.release_request
    api.release_request = lambda rid: (released.append(rid), real_release(rid))
    out = asyncio.run(_create(api, n=4))
    # images in submit order, seeds follow the seed + i contract, nothing released
    assert [base64.b64decode(d["b64_json"]).decode() for d in out["data"]] == engine.submitted
    assert engine.seeds == [7, 8, 9, 10]
    assert released == []
    _assert_nothing_left(api, engine, aborted=[], reads_done=sorted(engine.submitted))


class _DisconnectedRequest:
    async def is_disconnected(self):
        return True


@pytest.mark.parametrize("how", ["cancelled", "polled"])
def test_client_disconnect_releases_every_image(how):
    api, engine = _api([RUNNING, OK, RUNNING, RUNNING])

    async def disconnect():
        if how == "polled":
            return await _create(api, n=4, raw_request=_DisconnectedRequest())
        # the handler task is cancelled while it waits on the first image
        task = asyncio.ensure_future(_create(api, n=4))
        await asyncio.sleep(0.05)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(disconnect())
    _assert_nothing_left(api, engine, aborted=engine.ids(RUNNING), reads_done=engine.ids(OK))


async def _cancel_while_waiting(api, n):
    task = asyncio.ensure_future(_create(api, n=n))
    await asyncio.sleep(0.05)
    task.cancel()
    await task


@pytest.mark.parametrize(
    "plan, run, original",
    [
        # handler cancelled while all three run: nothing has been read
        ([RUNNING, RUNNING, RUNNING], _cancel_while_waiting, asyncio.CancelledError),
        # the 3rd submit raises; the first two are already on the engine
        ([RUNNING, RUNNING, RAISE], lambda api, n: _create(api, n=n), RuntimeError),
        # the 1st image fails at collect; the other three are still running
        ([FAIL, RUNNING, RUNNING, RUNNING], lambda api, n: _create(api, n=n), HTTPException),
    ],
    ids=["cancelled", "submit", "http"],
)
def test_failed_release_still_releases_the_rest(plan, run, original, caplog):
    """One release raising must not stop the others or replace the request's
    own exception."""
    api, engine = _api(plan)
    client_ids, attempted = [], []
    real_submit, real_release = api.submit_request, api.release_request

    def submit(**kwargs):
        client_ids.append(kwargs["request_id"])
        return real_submit(**kwargs)

    def release(request_id):
        attempted.append(request_id)
        if len(attempted) == 1:
            raise RuntimeError("release failed")
        real_release(request_id)

    api.submit_request, api.release_request = submit, release
    with caplog.at_level("WARNING"), pytest.raises(original) as exc:
        asyncio.run(run(api, len(plan)))
    if original is RuntimeError:
        assert str(exc.value) == "injected submit failure"
    if original is HTTPException:
        assert exc.value.detail == "injected image failure"

    unread = [rid for rid, o in zip(client_ids, plan, strict=False) if o != FAIL]
    assert attempted == unread
    failed, rest = attempted[0], attempted[1:]
    # the engine may key a submit by its own id; engine.submitted is in submit order
    engine_id = dict(zip(client_ids, engine.submitted, strict=True))
    # nothing but the image whose release raised is left; every other one was aborted
    assert set(api.pending_requests) <= {engine_id[failed]}
    assert {engine_id[rid] for rid in rest} <= set(engine.aborted)
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and failed in r.getMessage()]
    assert len(warnings) == 1
