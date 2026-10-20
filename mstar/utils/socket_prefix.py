"""Default IPC socket prefixes for the server entry points.

Every process of a deployment binds fixed names under its prefix
(``conductor.ipc``, ``api_server.ipc``, ``worker_0.ipc``, ...), and libzmq
replaces an existing socket file on ``bind``. Two servers on one prefix
therefore take over each other's inboxes, so each default here is unique to
one server on the host.
"""

import getpass
import os
import re
import tempfile


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


def default_upload_dir() -> str:
    """``/tmp/mstar_uploads_<user>/``: ``ensure_private_dir`` refuses an upload
    dir owned by another user, so one shared default would stop the second user
    on a host from starting. The Rust frontend's twin is
    ``runtime_dir::default_upload_dir``."""
    return f"/tmp/mstar_uploads_{_user()}/"


def per_launch_socket_prefix() -> str:
    """A new ``/tmp/mstar_<user>_<random>`` directory (mode 0700) for a server
    with no port of its own; the caller removes it at exit."""
    return tempfile.mkdtemp(prefix=f"mstar_{_user()}_", dir="/tmp")
