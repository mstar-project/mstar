"""Client-visible text for an exception raised while serving a request."""

import logging
import os
import re
import secrets

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
    refusal: bool | None = None,
) -> tuple[str, str]:
    """Return ``(text, error_id)``: what a client may see for ``exc``.

    A ``ValueError`` or ``TypeError`` is a refusal raised on purpose, so it
    keeps its message (prefixed with the type name when ``type_name``), with
    absolute paths cut to their base names. Anything else becomes
    ``internal error (error id <id>)``. Either way the full error and its
    traceback (left out when ``traceback`` is False, for an error already
    logged where it happened) go to ``log`` as ``<what> [error id <id>]``.
    ``refusal`` overrides that type check, for a caller that knows whether
    the request was refused.
    """
    error_id = secrets.token_hex(6)
    log.error(
        "%s [error id %s]: %s: %s", what, error_id, type(exc).__name__, exc,
        exc_info=exc if traceback else None,
    )
    if refusal is None:
        refusal = isinstance(exc, (ValueError, TypeError))
    if refusal:
        text = scrub_paths(str(exc))
        if type_name:
            text = f"{type(exc).__name__}: {text}"
        return text, error_id
    return f"internal error (error id {error_id})", error_id


def exception_status_text(exc: BaseException, what: str, *, log: logging.Logger = logger) -> tuple[int, str]:
    """``(status, text)`` an OpenAI route answers ``exc`` with: an exception that
    carries its own ``status_code`` and ``detail`` (an ``HTTPException``) keeps
    both; a ``ValueError`` is a refusal, a 400 (or its own status) with server
    paths cut to base names, logged as one INFO line; anything else goes through
    :func:`client_error_text` (a ``TypeError`` a 400 that keeps its traceback,
    any other a 500 showing only an error id). ``what`` names the route."""
    status, detail = getattr(exc, "status_code", None), getattr(exc, "detail", None)
    if isinstance(status, int) and detail is not None:
        return status, str(detail)
    if isinstance(exc, ValueError):
        log.info("%s request refused: %s", what, exc)
        return (status if isinstance(status, int) else 400), scrub_paths(str(exc))
    text, _ = client_error_text(exc, f"{what} request failed", log=log, type_name=False)
    if not isinstance(status, int):
        status = 400 if isinstance(exc, TypeError) else 500
    return status, text
