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
