"""The server's runtime directories (IPC sockets, uploaded media) and the files
written into them. ``rust/server/src/runtime_dir.rs`` is the Rust frontend's
twin; keep the checks and the error text in step."""

import os
import stat
from pathlib import Path


def ensure_private_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` with mode 0700 if it is missing, then require a real
    directory owned by this user and not writable by group or others.

    The default locations sit in a shared ``/tmp`` under predictable names, so
    an existing directory is only accepted after this check; anything else
    raises ``PermissionError`` naming the path. Only the final component has
    to be ours; missing parents are created 0700 too, so a parent that is
    itself a runtime dir (a socket dir holding the upload dir) passes its own
    check under any umask.
    """
    # Path() drops a trailing slash, without which lstat would follow a symlink.
    path = Path(path)
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        missing = [path]
        while not os.path.lexists(missing[-1].parent):
            missing.append(missing[-1].parent)
        for p in reversed(missing):
            try:
                os.mkdir(p, 0o700)
            except FileExistsError:
                pass
        st = os.lstat(path)
    uid = os.geteuid()
    if stat.S_ISLNK(st.st_mode):
        problem = "it is a symlink"
    elif not stat.S_ISDIR(st.st_mode):
        problem = "it is not a directory"
    elif st.st_uid != uid:
        problem = f"it is owned by uid {st.st_uid}"
    elif st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        problem = f"its mode is {stat.S_IMODE(st.st_mode):o}"
    else:
        return path
    raise PermissionError(
        f"refusing {path}: {problem}. It must be a directory owned by this user "
        f"(uid {uid}) and not writable by group or others; remove it, or pass "
        "another path with --socket-path-prefix / --upload-dir."
    )


def write_private_file(path: str | os.PathLike, data: bytes) -> None:
    """Write ``data`` to a new file at ``path``, mode 0600; an existing file or
    symlink there raises ``FileExistsError`` instead of being written through."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
