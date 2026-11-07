"""The server's runtime directories (IPC sockets, uploaded media) and the files
written into them. ``rust/server/src/runtime_dir.rs`` is the Rust frontend's
twin; keep the checks and the error text in step."""

import errno
import os
import stat
from pathlib import Path

_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def ensure_private_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` with mode 0700 if it is missing, then require a real
    directory owned by this user and leave it at mode 0700.

    The default locations sit in a shared ``/tmp`` under predictable names, so
    an existing directory is only accepted after this check. A symlink, a
    non-directory and a directory owned by anyone else raise
    ``PermissionError`` naming the path; a real directory of our own is
    tightened to 0700 instead of being refused, because the dirs earlier
    releases created with ``makedirs`` are 0775 under Ubuntu's default umask
    002 and refusing those would stop every upgrade. Tightening it also keeps
    the sockets and files written inside from being reachable by the group
    under that umask. Only the final component has to be ours; missing parents
    are created 0700 too, so a parent that is itself a runtime dir (a socket
    dir holding the upload dir) passes its own check under any umask.

    The check and the ``chmod`` both run on one descriptor opened
    ``O_DIRECTORY | O_NOFOLLOW``, so the mode is only ever changed on the very
    object whose owner was read: there is no window in which the name could be
    pointed at something else. ``O_NOFOLLOW`` covers the final component only,
    as ``lstat`` did; a parent that untrusted accounts can write is still
    theirs to redirect, which is why only trusted accounts should be able to
    write one.
    """
    # Path() drops a trailing slash, without which the open would follow a symlink.
    path = Path(path)
    try:
        fd = os.open(path, _OPEN_FLAGS)
    except FileNotFoundError:
        missing = [path]
        while not os.path.lexists(missing[-1].parent):
            missing.append(missing[-1].parent)
        for p in reversed(missing):
            try:
                os.mkdir(p, 0o700)
            except FileExistsError:
                pass
            except OSError as e:
                # not the raw OSError: its text carries the pathname, and the
                # twin's does not, so a caller that forwards it would differ
                raise _refusal(path, f"it could not be created ({e.strerror})") from e
        fd = _open_or_refuse(path)
    except OSError as e:
        raise _refusal(path, _why_unopenable(path, e)) from e
    try:
        st = os.fstat(fd)
        uid = os.geteuid()
        if st.st_uid != uid:
            raise _refusal(path, f"it is owned by uid {st.st_uid}")
        if stat.S_IMODE(st.st_mode) != 0o700:
            os.fchmod(fd, 0o700)
    finally:
        os.close(fd)
    return path


def _open_or_refuse(path: Path) -> int:
    try:
        return os.open(path, _OPEN_FLAGS)
    except OSError as e:
        raise _refusal(path, _why_unopenable(path, e)) from e


def _why_unopenable(path: Path, exc: OSError) -> str:
    """Why the open failed, for the refusal text. A second look at the path is
    racy, so it only ever words a refusal already decided on the open itself.
    ``O_DIRECTORY | O_NOFOLLOW`` reports a symlink as ENOTDIR, so the ``lstat``
    comes first to tell the two apart."""
    try:
        st = os.lstat(path)
    except OSError:
        st = None
    if (st is not None and stat.S_ISLNK(st.st_mode)) or exc.errno == errno.ELOOP:
        return "it is a symlink"
    if (st is not None and not stat.S_ISDIR(st.st_mode)) or exc.errno == errno.ENOTDIR:
        return "it is not a directory"
    return f"it could not be opened ({exc.strerror})"


def _refusal(path: Path, problem: str) -> PermissionError:
    return PermissionError(
        f"refusing {path}: {problem}. It must be a directory owned by this user "
        f"(uid {os.geteuid()}); remove it, or pass another path with "
        "--socket-path-prefix / --upload-dir."
    )


def write_private_file(path: str | os.PathLike, data: bytes) -> None:
    """Write ``data`` to a new file at ``path``, mode 0600; an existing file or
    symlink there raises ``FileExistsError`` instead of being written through."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
