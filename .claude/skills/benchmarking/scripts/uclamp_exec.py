#!/usr/bin/env python3
"""Run a command under a uclamp clock floor: ``uclamp_exec.py 1024 -- vllm serve ...``.

Sets uclamp's ``util_min`` (0 to 1024) on this process, then execs the command in
its place, so the command and every thread and process it starts inherit the
floor. Needs no privileges. It gives another engine the floor M*'s ``knobs:
uclamp_min`` gives M*'s worker and API server; it mirrors
``mstar.utils.threads.hold_clock`` with the standard library alone, so it runs
under any engine's Python. Unlike M*, it exits rather than run without the
floor, since a comparison is only fair when both sides got it.
"""

import ctypes
import os
import platform
import sys

_SCHED_SETATTR = {"x86_64": 314, "aarch64": 274}
# keep the policy, keep its parameters, set util_min
_FLAGS = 0x08 | 0x10 | 0x20
_SYSTEM_CAP = "/proc/sys/kernel/sched_util_clamp_min"


class _SchedAttr(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32), ("policy", ctypes.c_uint32), ("flags", ctypes.c_uint64),
        ("nice", ctypes.c_int32), ("priority", ctypes.c_uint32),
        ("runtime", ctypes.c_uint64), ("deadline", ctypes.c_uint64), ("period", ctypes.c_uint64),
        ("util_min", ctypes.c_uint32), ("util_max", ctypes.c_uint32),
    ]


def main(argv: list[str]) -> None:
    if len(argv) < 3 or argv[1] != "--" or not argv[0].isdigit() or not 0 <= int(argv[0]) <= 1024:
        sys.exit("usage: uclamp_exec.py <0-1024> -- <command ...>")
    floor = int(argv[0])
    try:
        with open(_SYSTEM_CAP) as f:
            cap = int(f.read())
    except (OSError, ValueError) as e:
        sys.exit(f"uclamp_exec.py: this kernel has no uclamp ({e})")
    if cap < floor:
        sys.exit(f"uclamp_exec.py: {_SYSTEM_CAP} caps the floor at {cap}")
    number = _SCHED_SETATTR.get(platform.machine())
    if number is None:
        sys.exit(f"uclamp_exec.py: no sched_setattr number for {platform.machine()}")
    attr = _SchedAttr(size=ctypes.sizeof(_SchedAttr), util_min=floor, flags=_FLAGS)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(number, 0, ctypes.byref(attr), 0) != 0:
        sys.exit(f"uclamp_exec.py: sched_setattr failed: {os.strerror(ctypes.get_errno())}")
    os.execvp(argv[2], argv[2:])


if __name__ == "__main__":
    main(sys.argv[1:])
