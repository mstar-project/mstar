"""Client-visible text for an exception raised while serving a request."""

import logging
import os
import re
import secrets
from dataclasses import dataclass

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# An absolute path: a "/" that does not continue a word, a number, a URL or
# another path, or the path of a file:// URL. Ends at whitespace, a quote, a
# bracket or a separator such as ":" or ",".
_ABS_PATH = re.compile(r"(?:\bfile://|(?<![\w.:/)\]]))/[^\s'\"`,;:()\[\]{}<>|]+")
# A quoted absolute path, taken whole: its directories may hold spaces or brackets.
_QUOTED_PATH = re.compile(r"(['\"`])((?:file://)?/[^'\"`\n]*)\1")


def _base_name(path: str) -> str:
    return os.path.basename(path) or "<path>"


def scrub_paths(text: str) -> str:
    """Replace each absolute file path in ``text`` with its base name, and a
    directory path (one ending in "/") with ``<path>``."""
    text = _QUOTED_PATH.sub(lambda m: m.group(1) + _base_name(m.group(2)) + m.group(1), text)
    return _ABS_PATH.sub(lambda m: _base_name(m.group(0)), text)


def client_error_text(
    exc: BaseException, what: str, *,
    log: logging.Logger = logger, type_name: bool = True, traceback: bool = True,
) -> tuple[str, str]:
    """Return ``(text, error_id)``: what a client may see for ``exc``.

    An ``HTTPException`` keeps its detail. A ``ValueError`` or ``TypeError``
    is a refusal raised on purpose, so it
    keeps its message (prefixed with the type name when ``type_name``), with
    absolute paths cut to their base names. Anything else becomes
    ``internal error (error id <id>)``. Either way the full error and its
    traceback (left out when ``traceback`` is False, for an error already
    logged where it happened) go to ``log`` as ``<what> [error id <id>]``.
    """
    error_id = secrets.token_hex(6)
    log.error(
        "%s [error id %s]: %s: %s", what, error_id, type(exc).__name__, exc,
        exc_info=exc if traceback else None,
    )
    if isinstance(exc, HTTPException):
        return scrub_paths(str(exc.detail)), error_id
    if isinstance(exc, (ValueError, TypeError)):
        text = scrub_paths(str(exc))
        if type_name:
            text = f"{type(exc).__name__}: {text}"
        return text, error_id
    return f"internal error (error id {error_id})", error_id


@dataclass(frozen=True)
class RequestError:
    """A request failure carried from its owning worker to the API.

    ``fatal`` marks a sticky device fault: every later CUDA call on this worker
    fails too, so ``StepPipeline._stop_if_fatal`` stops the worker on it.
    """

    message: str
    status: int = 500
    fatal: bool = False

    @classmethod
    def from_exception(
        cls, exc: BaseException, what: str, *,
        log=logger, traceback=True, whole_step=False,
    ):
        """The failure for ``exc``. A ``ValueError`` or ``TypeError`` is the
        request's fault (400) when one request's own stage raised it. With
        ``whole_step`` (a step or the main loop failed, so the origin is
        unknown) it keeps its message but is a 500. An ``HTTPException`` keeps
        its status either way."""
        from torch import AcceleratorError

        message, _ = client_error_text(exc, what, log=log, traceback=traceback)
        if isinstance(exc, HTTPException):
            status = exc.status_code
        elif isinstance(exc, (ValueError, TypeError)) and not whole_step:
            status = 400
        else:
            status = 500
        return cls(message, status, fatal=isinstance(exc, AcceleratorError))
