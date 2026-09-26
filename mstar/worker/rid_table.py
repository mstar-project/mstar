"""Worker-local integer handles for request ids.

Messages carry the request id string; everything inside a worker keys on a
small int instead. The only translations are at the process boundary: a handle
is minted on NEW_REQUEST and looked up on every other inbound message, and the
string goes back on the wire at each send site.

Handles are RECYCLED once a request is removed, so anything keyed by one must
be purged on the same REMOVE_REQUEST that releases it. A leaked string key was
harmless -- the string never recurred -- but a leaked handle silently attaches
to whichever request gets that handle next. Known handle-keyed state:
MicroScheduler.{failed_rids, admit_errors, held_until, backlog,
pending_tp_follow_count} (purged by clear_rid), Worker.{_last_active,
_pending_removes, _pending_loop_stops, streaming_buffers}, the engine's and
tensor manager's per-request state, and WorkerProfileInfo.
"""


class RidTable:
    __slots__ = ("_names", "_handles", "_free")

    def __init__(self):
        self._names: list[str | None] = []
        self._handles: dict[str, int] = {}
        self._free: list[int] = []

    def intern(self, request_id: str) -> int:
        """Handle for ``request_id``, minting one if it has none.

        A request spanning several partitions on this worker gets one
        NEW_REQUEST per partition; they all share the first one's handle.
        """
        handle = self._handles.get(request_id)
        if handle is not None:
            return handle
        if self._free:
            handle = self._free.pop()
            self._names[handle] = request_id
        else:
            handle = len(self._names)
            self._names.append(request_id)
        self._handles[request_id] = handle
        return handle

    def handle(self, request_id: str) -> int | None:
        """Handle for an inbound message's request id, or None if this worker
        does not know it (never admitted, or already removed -- a benign race,
        not an error)."""
        return self._handles.get(request_id)

    def name(self, handle: int) -> str:
        """The wire identity, for a message about to leave this process."""
        name = self._names[handle] if 0 <= handle < len(self._names) else None
        if name is None:
            raise KeyError(f"no live request has handle {handle}")
        return name

    def release(self, handle: int) -> None:
        """Free ``handle`` for reuse. Must run after every handle-keyed map has
        been purged of it."""
        name = self._names[handle] if 0 <= handle < len(self._names) else None
        if name is None:
            return
        self._names[handle] = None
        del self._handles[name]
        self._free.append(handle)

    def __len__(self) -> int:
        return len(self._handles)
