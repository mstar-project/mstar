"""Executed real-frame history for a Praxis session.

The reference keeps 21 frames after the first executed chunk and appends 24
after each later chunk. A cold call has no executed chunk yet, but we retain
its current observation as the first frame of that future 21-frame window.
"""

from __future__ import annotations

FIRST_ROWS = 21
LATER_ROWS = 24
STATE_NAME = "executed_observations"
BOUNDARY_STATE_NAME = "boundary_poses"


def window_size(max_context_chunks: int) -> int:
    if type(max_context_chunks) is not int or max_context_chunks < 1:
        raise ValueError("max_context_chunks must be a positive integer")
    return FIRST_ROWS + LATER_ROWS * (max_context_chunks - 1)


def append_count(held: int, executed_rows: int) -> int:
    """How many new rows to append, excluding overlap with the cold seed.

    ``held`` is the conductor's resident count before this request, after
    episode reset or eviction. The caller must reset the episode before
    admission when the client cannot supply the required executed stack.
    """
    if type(held) is not int or held < 0:
        raise ValueError(f"invalid held history count {held!r}")
    if held == 0:
        # Admission can evict an otherwise warm session. The frontend may
        # still report its last reply's span; the state view is authoritative.
        return 1  # current real frame becomes the window's first anchor
    if held == 1:
        if executed_rows != FIRST_ROWS:
            raise ValueError("first warm call requires 21 executed rows")
        return FIRST_ROWS - 1  # seed is the first row of the overlap
    if executed_rows != LATER_ROWS:
        raise ValueError("later warm calls require 24 executed rows")
    return LATER_ROWS


def clean_depth(held: int, max_context_chunks: int) -> int:
    """The number of complete 21/24-frame chunks in the retained window."""
    if held < FIRST_ROWS:
        return 0
    return min(1 + (held - FIRST_ROWS) // LATER_ROWS, max_context_chunks)


def boundary_offsets(depth: int) -> tuple[int, ...]:
    """Frame positions of the clean chunks' first poses."""
    return tuple(0 if k == 0 else FIRST_ROWS + LATER_ROWS * (k - 1) for k in range(depth))
