"""Parsing large ``/generate/ws`` frames without holding the event loop.

``json.loads``, ``msgpack.unpackb`` and base64 decoding hold the GIL for the
whole call, so a thread doesn't free the loop; nor does pickling the frame
across to a process, one more whole-frame call. A large frame is therefore
handed to a worker process through a file, written from a thread in slices
short enough that the loop gets the GIL between them; the worker parses it,
saves each file's data under the upload dir (decoding base64 as it goes) and
sends back only the message, each saved file's ``data`` a ``SavedUpload``. A
small frame stays on the loop, where the round trip costs more than the parse.

Taking the frame in stays on the loop: uvicorn and websockets assemble, unmask,
decode and, with deflate on, inflate it there.

The pool's processes re-import the main module, as every spawned process does:
under ``mstar serve`` that is light, under the ``mstar-serve`` script it is the
server's entrypoint (torch included, a few hundred MB each).
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import signal
import uuid
import weakref
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import msgpack

from mstar.api_server.media_io import _decode_base64_payload
from mstar.utils.orphan import watch_parent

logger = logging.getLogger(__name__)

# A frame this long goes to the pool, whose round trip blocks the loop for a
# millisecond or so and adds to this frame's own wait, more with size. A text frame costs
# the loop about 6 ms per MiB (JSON parse and base64 decode), so from 1 MiB up the
# pool is cheaper for everyone. A binary frame costs it far less, since msgpack
# only copies its files' bytes: measured, the pool added 2, 5 and 18 ms to the
# frame's wait at 1, 4 and 16 MiB to save at most about 2 ms of lag, and at 32 MiB about
# 24 ms to save about 15, so a binary frame goes to the pool from 32 MiB.
OFF_LOOP_TEXT_BYTES = 1 << 20
OFF_LOOP_BINARY_BYTES = 32 << 20

# The slice the hand-off writes per GIL hold: about a millisecond of copying.
_SLICE = 1 << 20

# Large frames parse side by side up to this many at a time; more wait their turn.
_PROCESSES = 2

_pool: ProcessPoolExecutor | None = None

# In a pool process: the frame being parsed, whose files are removed if the
# process is stopped mid-frame (the pool stops its others when one dies).
_parsing: tuple[str, str] | None = None


class SavedUpload:
    """A file's ``data`` the pool already saved to ``path``. The file is
    removed with this object unless ``claim`` takes it first, so a message
    refused before its files are taken leaves nothing behind."""

    def __init__(self, path: str):
        self.path = path
        self._remove = weakref.finalize(self, _unlink, path)

    def claim(self) -> str:
        self._remove.detach()
        return self.path

    def __reduce__(self):
        # The pool sends it back as a path; the server makes the owning object.
        return SavedUpload, (self.path,)


def _unlink(path: str) -> None:
    Path(path).unlink(missing_ok=True)


def parse(payload: bytes | str):
    """A frame's message: msgpack for bytes, JSON for text."""
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return msgpack.unpackb(payload, raw=False)
    return json.loads(payload)


def _files(message) -> list[dict]:
    """The file entries a message carries: its own, or its append's."""
    if not isinstance(message, dict):
        return []
    append = message.get("append")
    if isinstance(append, dict):
        append = append.get("files")
    found = [f for f in (message.get("files"), append) if isinstance(f, list)]
    return [entry for files in found for entry in files if isinstance(entry, dict)]


def save_entry(name: str, data, upload_dir: Path, prefix: str | None = None) -> Path:
    """Save a file's ``data``: raw bytes, or base64 text, by the decoder the
    data-URL paths use (unpadded or wrapped is fine; a bad alphabet is the
    client's error). A failed write leaves nothing."""
    if isinstance(data, str):
        data = _decode_base64_payload(data)
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError(f"file {name!r} carries no data")
    path = Path(upload_dir) / f"{prefix or uuid.uuid4()}_{os.path.basename(name) or 'upload'}"
    try:
        path.write_bytes(data)
    except BaseException:
        _unlink(str(path))
        raise
    return path


def _parse_saving_files(frame: str, binary: bool, upload_dir: str):
    """In a pool process: the message in ``frame``, with each file's data saved
    under ``upload_dir`` beside it and replaced by a ``SavedUpload``. A file that
    can't be saved is left for the request to refuse, as it would any frame's.
    The saved names start with the frame's: until this returns, they are
    ``parse_frame``'s to remove."""
    global _parsing
    stem = Path(frame).stem
    _parsing = (upload_dir, stem)
    with open(frame, "rb") as f:
        payload = f.read()
    try:
        message = parse(payload if binary else payload.decode("utf-8"))
    except Exception as exc:
        # its text alone: JSONDecodeError would carry the whole frame back
        raise ValueError(str(exc)) from None
    del payload
    for i, entry in enumerate(_files(message)):
        name, data = entry.get("name"), entry.get("data")
        if not isinstance(name, str):
            continue
        if isinstance(data, str):
            try:
                data = _decode_base64_payload(data)
            except ValueError:
                # refused as this data was (with a generic text), and doesn't travel back
                entry["data"] = "!"
                continue
        try:
            path = save_entry(name, data, Path(upload_dir), f"{stem}-{i}")
        except (OSError, ValueError):
            continue
        upload = object.__new__(SavedUpload)  # no finalizer in this process
        upload.path = str(path)
        entry["data"] = upload
    _parsing = None
    return message


def _write_frame(payload: bytes | str, path: str) -> None:
    """Write ``payload`` to ``path`` a slice at a time, so the loop's thread
    gets the GIL between slices (``os.write`` releases it, and so does the
    interpreter between short statements)."""
    view = memoryview(payload) if not isinstance(payload, str) else None
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        for start in range(0, len(payload), _SLICE):
            chunk = view[start:start + _SLICE] if view is not None \
                else payload[start:start + _SLICE].encode("utf-8")
            while chunk:
                chunk = chunk[os.write(fd, chunk):]
    finally:
        os.close(fd)


def _worker_init() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C reaches the group; the server stops the pool
    signal.signal(signal.SIGTERM, _stopped)
    watch_parent("ws frame parser", "API server", signal.SIGTERM)


def _stopped(signum, frame) -> None:
    if _parsing is not None:
        _sweep(*_parsing)
    os._exit(0)


def start() -> None:
    """Start the pool with the server, so its processes are starting before the
    first frame (they take a few seconds, spawned off the server's main module)."""
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(
            _PROCESSES, mp_context=mp.get_context("spawn"), initializer=_worker_init,
        )
        for _ in range(_PROCESSES):
            _pool.submit(parse, b"\x80")


def stop() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


async def parse_frame(payload: bytes | str, upload_dir: Path):
    """A frame's message. A large frame parses in the pool, its files' data
    already saved under ``upload_dir`` (``SavedUpload``). Raises what the parse
    raises, and ``OSError`` when it could not be handed off, or a pool process
    died: that fails every frame the pool held, on any socket, and a new pool
    takes the next."""
    global _pool
    binary = not isinstance(payload, str)
    if len(payload) < (OFF_LOOP_BINARY_BYTES if binary else OFF_LOOP_TEXT_BYTES):
        return parse(payload)
    if _pool is None:
        logger.warning("No frame parser pool: parsing a %d-byte frame on the loop", len(payload))
        return parse(payload)
    loop = asyncio.get_running_loop()
    stem = str(uuid.uuid4())
    frame = str(Path(upload_dir) / f"{stem}.wsframe")
    pool = None
    try:
        await loop.run_in_executor(None, _write_frame, payload, frame)
        pool = _pool
        future = pool.submit(_parse_saving_files, frame, binary, str(upload_dir))
        # Whatever ends the parse (it raised, its result couldn't be sent, its
        # process died), files no message came back to own go with it, also
        # when this frame's socket stopped waiting long before.
        future.add_done_callback(
            lambda f: f.cancelled() or f.exception() is None or _sweep(upload_dir, stem),
        )
        return await asyncio.wrap_future(future)
    except BrokenProcessPool:
        if _pool is pool:
            logger.error("A /generate/ws frame parser process died; starting a new pool")
            pool.shutdown(wait=False, cancel_futures=True)
            _pool = None
            start()
        raise OSError("the frame's parser process died") from None
    finally:
        _unlink(frame)


def _sweep(upload_dir: Path, stem: str) -> None:
    for path in Path(upload_dir).glob(f"{stem}-*"):
        _unlink(str(path))
