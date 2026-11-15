"""Default IPC socket prefixes for the server entry points.

Every process of a deployment binds fixed names under its launch directory
(``conductor.ipc``, ``api_server.ipc``, ``worker_0.ipc``, ...), and libzmq
replaces an existing socket file on ``bind``. Two servers on one prefix
therefore take over each other's inboxes. The launch token keeps restarts on
the same base prefix in separate directories.
"""

import getpass
import hashlib
import os
import re
import tempfile

DEPLOYMENT_TOKEN_ENV = "MSTAR_DEPLOYMENT_TOKEN"
MAX_UNIX_SOCKET_PATH_BYTES = 107


def launch_socket_prefix(base: str, token: str | None) -> str:
    """Keep a restarted launch's IPC sockets away from surviving workers.

    The token is shared by this launch's processes and differs on restart.
    Hashing it keeps launcher-provided values out of paths and leaves room
    for the longest entity name under the Unix socket path limit.
    """
    if not token:
        return base
    return os.path.join(base, hashlib.sha1(token.encode()).hexdigest()[:10])


def check_ipc_socket_path(prefix: str, entity_id: str) -> None:
    """Fail before binding when a configured base leaves too little path space."""
    path = os.path.join(prefix, f"{entity_id}.ipc")
    length = len(os.fsencode(path))
    if length > MAX_UNIX_SOCKET_PATH_BYTES:
        raise ValueError(
            f"IPC socket path for {entity_id!r} is {length} bytes, above the "
            f"Linux limit of {MAX_UNIX_SOCKET_PATH_BYTES}; shorten "
            "ipc_socket_path_prefix"
        )


def _user() -> str:
    """The user name, or the uid when there is none or it is not a short
    ASCII name: a socket path holds at most 107 bytes, and the longest
    entity name under the prefix needs about 50 of them."""
    try:
        name = getpass.getuser()
    except (KeyError, OSError):
        # No passwd entry for this uid, as in containers run with an arbitrary uid.
        name = ""
    return name if re.fullmatch(r"[A-Za-z0-9._-]{1,32}", name) else str(os.getuid())


def per_port_socket_prefix(port: int) -> str:
    """``/tmp/mstar_<user>_<port>/`` for an HTTP server: the port is already
    unique to one server on the host."""
    return f"/tmp/mstar_{_user()}_{port}/"


def per_launch_socket_prefix() -> str:
    """A new ``/tmp/mstar_<user>_<random>`` directory (mode 0700) for a server
    with no port of its own; the caller removes it at exit."""
    return tempfile.mkdtemp(prefix=f"mstar_{_user()}_", dir="/tmp")
