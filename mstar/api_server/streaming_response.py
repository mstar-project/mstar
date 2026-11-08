"""Streaming-response helpers: a response that cleans up after a body
Starlette never started, and a body that reports a mid-stream error in band."""

import asyncio
import contextlib
import inspect
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from mstar.api_server.request_types import ResultChunk

logger = logging.getLogger(__name__)


async def errors_in_band(request_id: str, chunks):
    """``chunks``, with an ``HTTPException`` raised mid-stream (``--timeout``)
    ending it as the error chunk an engine error ends it with: the 200 is
    already sent, so raising would only cut the body off."""
    try:
        async for chunk in chunks:
            yield chunk
    except HTTPException as exc:
        yield ResultChunk(
            request_id=request_id, modality="error",
            data=str(exc.detail).encode("utf-8"), metadata={"status": exc.status_code},
        )


class ReleasingStreamingResponse(StreamingResponse):
    """A ``StreamingResponse`` that calls ``release`` when its body never started.

    Starlette sends the headers before it iterates the body, and leaves the
    body unstarted when that send raises: uvicorn 0.28.0 (the release that
    reports ASGI 2.4) raises from ``send`` once the client has left. A server
    that cancels the response before its first step would do the same; current
    uvicorn does neither, so there the body starts and aborts the request
    itself, and this is a guard. An async generator that never started never
    runs its ``finally``, so what the body releases on an early exit (its
    engine request, read-ahead pieces, uploaded files) would be held until the
    process restarts.
    """

    def __init__(self, content: AsyncGenerator, *, release: Callable[[], None | Awaitable[None]], **kwargs):
        super().__init__(content, **kwargs)
        self._release = release

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if inspect.getasyncgenstate(self.body_iterator) == inspect.AGEN_CREATED:
                try:
                    result = self._release()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    # logged, not raised: raising here would replace the send
                    # error the server needs to see with this one
                    logger.exception("releasing a streaming response whose body never started failed")


async def read_stream_head(items, raw_request=None):
    """Read the first item before headers, aborting the iterator on disconnect."""
    async def disconnected():
        while (await raw_request.receive())["type"] != "http.disconnect":
            pass

    first = asyncio.ensure_future(anext(items, None))
    gone = asyncio.ensure_future(disconnected()) if raw_request is not None else None
    try:
        await asyncio.wait({first, gone} - {None}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # A pending receive would consume StreamingResponse's disconnect.
        for task in (gone, first):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
    return None if first.cancelled() else first.result()
