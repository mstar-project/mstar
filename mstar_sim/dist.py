"""``torch.distributed`` across emulated GPUs.

Every emulated GPU is its own process, as on a real node. Process groups that
would be NCCL are gloo (the control traffic mstar sends over CPU groups is
real). Ranks meet at each collective over simulated-GPU tensors, as NCCL's do
(:func:`_meet`): each rank's description of the call is gathered on its group,
and a rank whose call doesn't pair with its peers' (skipped, reordered,
resized) aborts with every rank's, where NCCL would hang or corrupt data. A
captured graph meets once per group at each replay, on its collectives over that
group's ranks. The data moves (over gloo) only if every rank holds values (a vote,
lengths); otherwise its outputs are never computed. Point to point has no group to meet
on: a send's descriptor travels one way to its receive, ahead of a payload if the sender
holds values (:func:`_send`, :func:`_receive`). Either way it is charged its NCCL time
(:func:`mstar_sim.cost.collective_seconds`, measured tables where they exist) on the
current stream, on every rank alike, so timing stays symmetric, as SGLang's simulator and
Maya assume.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import itertools
import operator
import os
import resource
import sys
import threading
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist

from mstar_sim import clock, cost, cuda, device


class _Done:
    """The ``Work`` of a collective that finished at launch (host side)."""

    def wait(self, timeout=None):
        return True

    def is_completed(self):
        return True

    def is_success(self):
        return True

    def get_future(self):
        fut = torch.futures.Future()
        fut.set_result(None)
        return fut


def _sim_tensors(args, kwargs) -> list[torch.Tensor]:
    flat = device.tree_flatten((args, kwargs))[0]
    return [t for t in flat if isinstance(t, device.SimTensor)]


# --- the meet. A rank's descriptor of a collective is one int64 row:
# (primitive, numel, dtype, op, root, values, elements sent to each rank, received from
# each). The first five must be equal on every rank, and what rank i sends rank j is what j
# receives from i. The sixth is bits, ANDed across ranks: HELD, whether the data moves (every
# rank holds values), and COMPUTED, whether what it writes is (every rank's were).
_PRIMITIVES = ("AllReduce", "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllToAll",
               "AllReduceFusion", "Graph", "Barrier", "PointToPoint")
_DTYPES = tuple(dict.fromkeys(v for v in vars(torch).values() if isinstance(v, torch.dtype)))
_VALUES = 5  # the index of the values field
HELD, COMPUTED = 1, 2


def row(primitive: str, numel: int = 0, dtype: torch.dtype | None = None, op: int = 0, root: int = 0,
        values: int = 0, send: list[int] | None = None, recv: list[int] | None = None, size: int = 1):
    """A collective's descriptor (``send``/``recv``: elements to and from each of the ``size`` ranks)."""
    return [_PRIMITIVES.index(primitive), numel, _DTYPES.index(dtype) if dtype is not None else -1, op, root,
            values, *(send or [0] * size), *(recv or [0] * size)]


def _describe(r: list[int]) -> str:
    name = _PRIMITIVES[r[0]]
    if name == "Graph":
        return f"a graph replay of {r[1]} collectives (sequence hash {r[3] & (2**64 - 1):016x})"
    size = (len(r) - _VALUES - 1) // 2
    text = f"{name}(numel={r[1]}, dtype={_DTYPES[r[2]] if r[2] >= 0 else None}, op={r[3]}, root={r[4]}"
    if name == "AllToAll":
        text += f", sends={r[_VALUES + 1:_VALUES + 1 + size]}, receives={r[_VALUES + 1 + size:]}"
    return text + ")"


def _pairs(rows: list[list[int]]) -> bool:
    size = len(rows)
    send = [r[_VALUES + 1:_VALUES + 1 + size] for r in rows]
    recv = [r[_VALUES + 1 + size:] for r in rows]
    return (all(r[:_VALUES] == rows[0][:_VALUES] for r in rows)
            and all(send[i][j] == recv[j][i] for i in range(size) for j in range(size)))


def _abort() -> None:
    """Abort, as NCCL's watchdog does, after the stacks ``faulthandler`` prints: without a
    core file, which would hold only the emulator."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.abort()


@contextmanager
def _waiting():
    """The host waits for its peers: not the emulator's work, and the device's clock stands still."""
    with clock.computing():
        began = time.perf_counter()
        yield
        waited = time.perf_counter() - began
        clock.STATS["host_compute"] -= waited
        clock.STATS["host_wait"] += waited


def _watched(wait, where: str, mine: list[int], replay: bool = False) -> None:
    """``wait``, a call into gloo for a peer. One that never comes within the group's timeout
    aborts an eager call, as NCCL's watchdog does, and hangs a replay, whose collectives
    nothing watches. ``kill -USR1`` a peer for its stack."""
    try:
        wait()
    except RuntimeError as e:
        print(f"{where}: {_describe(mine)} met no peer ({e}). "
              + ("A replayed graph's collectives are unwatched: this rank hangs, as on a GPU." if replay else
                 "NCCL's watchdog aborts here (or the emulator's host-side meet hung: README Limits)."),
              file=sys.stderr, flush=True)
        if replay:
            threading.Event().wait()
        _abort()


def _meet(group, mine: list[int], replay: bool = False) -> list[list[int]]:
    """Meet the group's other ranks at a collective (``replay``: a graph's replay) and return
    every rank's descriptor. Blocks until they all come, as the GPU's kernel does
    (:func:`_watched`). A mismatch aborts (NCCL's behaviour there is undefined: a hang or
    corrupt data, and a raise would be caught and served past)."""
    ranks = dist.get_process_group_ranks(group)
    me = dist.get_rank()
    seq = group._get_sequence_number_for_group()  # torch's count of the group's gloo ops
    where = f"mstar_sim: rank {me}, torch sequence number {seq} (meets included) on the group of ranks {ranks}"
    gathered = torch.empty(len(ranks) * len(mine), dtype=torch.int64, device="cpu")  # whatever the default
    with _waiting():
        # straight to the backend: torch's public function logs a failure through the group's
        # registry entry, which the fused-AR channel hasn't, and turns the timeout into a ValueError
        _watched(lambda: group._allgather_base(gathered, torch.tensor(mine, dtype=torch.int64, device="cpu")).wait(),
                 where, mine, replay)
    rows = gathered.view(len(ranks), -1).tolist()
    if not _pairs(rows):
        lines = "".join(f"\n  rank {r}: {_describe(d)}{'  <- this rank' if r == me else ''}"
                        for r, d in zip(ranks, rows, strict=True))
        why = ("the ranks took different capture paths (AGENTS.md invariant 11): NCCL pairs the collectives"
               " inside one by one, so it may run on, hang or corrupt data"
               if any(_PRIMITIVES[d[0]] == "Graph" for d in rows) else "NCCL would hang or corrupt data")
        print(f"{where}: the ranks' collectives don't pair ({why}):{lines}", file=sys.stderr, flush=True)
        _abort()
    return rows


def collective(family: str, group, nbytes: int, desc: list[int] | None) -> int:
    """A collective on ``group`` (None: the default one) described by ``desc`` (:func:`row`;
    None for point to point, which has no group to meet on): charged its NCCL time (cost
    ``family``) on the current stream, then met with the group's other ranks, or recorded on
    the graph being captured, whose replays meet. Returns the values bits every other rank set
    (all of them with none to meet; none when captured)."""
    size = dist.get_world_size(group) if dist.is_initialized() else 1
    tape = clock.capturing()
    if not clock.FAST:
        clock.charge(cost.collective_seconds(family, size, nbytes, graph=tape is not None))
    if desc is None or size == 1:
        return -1 if tape is None else 0
    group = group or dist.group.WORLD
    if tape is not None:
        tape.collectives.append((group, desc[:_VALUES]))
        return 0
    return functools.reduce(operator.and_, (r[_VALUES] for r in _meet(group, desc)))


# --- point to point. A send's descriptor (:func:`row`, its ``op`` the tag its payload follows
# on) is the call's own message: on the call's tag and to the call's peer, so it pairs as the
# call would, whoever names whom. Only the sender decides whether a payload follows, so nothing
# flows back. Payload tags are fresh, above any a caller uses, so a receive finds its payload
# whichever of its rank's receives finishes first.
_PAYLOAD_TAGS = itertools.count(1 << 30)


class _Posted(_Done):
    """The ``Work`` of a point-to-point call. ``finish``, which returns the rank (in the group)
    it heard from, runs on its own thread, as gloo's progress does: a receive has to post for
    its payload once the descriptor comes, and its peer's ``wait`` can't hang on this rank's.
    What it raises is raised at the wait."""

    def __init__(self, finish):
        def run():
            try:
                self._source = finish()
            except Exception as e:  # noqa: BLE001 - for the caller's thread
                self._error = e

        self._source = self._error = None
        self._thread = threading.Thread(target=run, name="mstar_sim-point-to-point", daemon=True)
        self._thread.start()

    def wait(self, timeout=None):
        with _waiting():
            self._thread.join()
        if self._error is not None:
            raise self._error
        return True

    def is_completed(self):
        return not self._thread.is_alive()

    def is_success(self):
        return self._error is None

    def get_future(self):
        self.wait()
        return super().get_future()

    def _source_rank(self):
        return self._source


def _send(a: dict, group, tensor: torch.Tensor, mine: int) -> _Posted:
    """Post ``tensor``'s descriptor, and its data if this rank holds it."""
    tag = next(_PAYLOAD_TAGS)
    desc = row("PointToPoint", tensor.numel(), tensor.dtype, op=tag, values=mine)
    to = {"dst": a["dst"], "group": group, "group_dst": a["group_dst"]}
    posted = [dist.isend(torch.tensor(desc, dtype=torch.int64, device="cpu"), tag=a["tag"], **to)]
    if mine & HELD:
        posted.append(dist.isend(tensor._data, tag=tag, **to))
    where = f"mstar_sim: rank {dist.get_rank()}"

    def finish():
        for work in posted:
            _watched(work.wait, where, desc)

    return _Posted(finish)


def _receive(a: dict, group, tensor: torch.Tensor, mine: int) -> _Posted:
    """Post for a sender's descriptor, then take what follows it: into ``tensor``, or dropped
    if this rank holds no values. What it leaves is computed only if both ranks' were."""
    desc = row("PointToPoint", tensor.numel(), tensor.dtype, values=mine)
    frame = torch.empty(len(desc), dtype=torch.int64, device="cpu")
    into = tensor._data if mine & HELD else torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
    posted = dist.irecv(frame, a["src"], group, a["tag"], a["group_src"])
    where = f"mstar_sim: rank {dist.get_rank()}"

    def finish():
        _watched(posted.wait, where, desc)
        theirs = frame.tolist()
        if theirs[1:3] != desc[1:3]:
            print(f"{where}: the ranks' sends don't pair (NCCL would hang or corrupt data): {_describe(desc)} "
                  f"received {_describe(theirs)}", file=sys.stderr, flush=True)
            _abort()
        source, values = posted._source_rank(), theirs[_VALUES] & mine
        if theirs[_VALUES] & HELD:
            _watched(lambda: dist.recv(into, group=group, group_src=source, tag=theirs[3]), where, desc)
        if mine & HELD:
            device.set_never_computed(into, not (values & HELD and values & COMPUTED))
        return source

    return _Posted(finish)


def seal(tape: clock.Tape) -> None:
    """At the end of a capture, fold the tape's collectives into what each replay meets on:
    one row per group (in first-use order) with the count and a hash of the captured
    collectives over its ranks, on it or on another group of the same ranks (a device group
    and its fused-AR channel). Every rank of a group holds each of those groups, so the rows
    agree unless the ranks captured differently: they disagree for a wrong bucket, or for a
    reorder that keeps each group's first use; an inversion of first uses meets on different
    groups and hangs, as on a GPU. A replay is one host call, so nothing sits between them.
    Groups of other ranks meet on their own, in first-use order, as NCCL's communicators do."""
    ranks = {group: tuple(sorted(dist.get_process_group_ranks(group))) for group, _ in tape.collectives}
    over: dict[tuple[int, ...], list[list[int]]] = {}
    for group, desc in tape.collectives:
        over.setdefault(ranks[group], []).append(desc)
    meets = []
    for group, key in ranks.items():
        digest = hashlib.blake2b(repr(over[key]).encode(), digest_size=8).digest()
        meets.append((group, row("Graph", len(over[key]), op=int.from_bytes(digest, "little", signed=True),
                                 size=len(key))))
    tape.collectives = meets


def meet_replay(tape: clock.Tape) -> None:
    """A replay of a graph that captured collectives: meet on each row :func:`seal` left."""
    with clock.computing():
        for group, mine in tape.collectives:
            _meet(group, mine, replay=True)


# each collective writes its first argument, but for these
_ONLY_READS = {"send", "isend"}
# and these at their root or at the other ranks
_ONLY_READS_AT = {"broadcast": "root", "reduce": "others"}
# and all but these leave nothing of what was there (a broadcast's other ranks read nothing)
_READS_WHAT_IT_WRITES = {"all_reduce", "reduce"}
# these return a ``Work`` whether or not they take ``async_op``
_RETURN_WORK = {"isend", "irecv"}


def _splits(t: torch.Tensor, sizes, size: int) -> list[int]:
    """Elements of ``t`` to or from each rank, by an all-to-all's split sizes (rows)."""
    if not sizes:
        return [t.numel() // size] * size
    per_row = t.numel() // t.shape[0] if t.dim() and t.shape[0] else 0
    return [int(s) * per_row for s in sizes]


def _collective(name: str, family: str, primitive: str | None):
    real = getattr(dist, name)
    sig = inspect.signature(real)
    first = next(iter(sig.parameters))

    @functools.wraps(real)
    def call(*args, **kwargs):
        tensors = _sim_tensors(args, kwargs)
        if not tensors:
            return real(*args, **kwargs)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        group = a.get("group")
        if dist.distributed_c10d._rank_not_in_group(group):
            return real(*args, **kwargs)  # nothing runs on this rank: torch's own answer
        if name in ("recv", "irecv") and a["src"] is None and a["group_src"] is None:
            # gloo, under the stand-in, would serve it
            raise device.gpu_error(NotImplementedError("ProcessGroupNCCL does not support recvAnysource"))
        nbytes = max(t.numel() * t.element_size() for t in tensors)
        outputs = [] if name in _ONLY_READS else _sim_tensors((a[first],), {})
        root = next((a[k] for k in ("src", "dst", "group_src", "group_dst") if a.get(k) is not None), -1)
        written = outputs
        if name in _ONLY_READS_AT:
            by_group = a.get("group_src") is not None or a.get("group_dst") is not None
            side = "root" if root == (dist.get_rank(group) if by_group else dist.get_rank()) else "others"
            if _ONLY_READS_AT[name] == side:
                written = []
        covered = [] if name in _READS_WHAT_IT_WRITES else [t for t in written if device._covers(t._data)]
        # what it writes is computed if all it reads was, on every rank
        mine = (HELD * all(t._data.device.type != "meta" for t in tensors)
                | COMPUTED * (not any(device.never_computed(t._data) for t in tensors
                                      if not any(t is c for c in covered))))
        desc = None  # point to point (mstar sends none)
        if primitive is not None:
            size = dist.get_world_size(group) if dist.is_initialized() else 1
            send = recv = None
            if primitive == "AllToAll":
                send = _splits(a["input"], a["input_split_sizes"], size)
                recv = _splits(a["output"], a["output_split_sizes"], size)
            op = a.get("op")
            desc = row(primitive, 0 if send else sum(t.numel() for t in outputs), (outputs or tensors)[0].dtype,
                       op=int(op) if op is not None else 0, root=root, values=mine, send=send, recv=recv, size=size)
        values = collective(family, group, nbytes, desc) & mine
        if primitive is None and clock.capturing() is None:
            with clock.computing():
                work = (_send if name in _ONLY_READS else _receive)(a, group, a[first], mine)
            if name in _RETURN_WORK:
                return work
            work.wait()
            return None if name == "send" else dist.get_global_rank(group or dist.group.WORLD, work._source_rank())
        moved = bool(values & HELD)
        work = _Done() if kwargs.get("async_op") or name in _RETURN_WORK else None
        if name == "recv":  # captured, it hears from nobody yet: its rank is the one it names
            work = (a["src"] if a["group_src"] is None
                    else dist.get_global_rank(group or dist.group.WORLD, a["group_src"]))
        if moved:
            # every rank holds values (a vote, lengths): send them for real, so
            # what ranks agree on device is what they agreed on
            if outputs and not written:  # a root, or a reduce's others, which gloo's leaves partial sums in
                a[first] = a[first]._data.clone()
                args, kwargs = bound.args, bound.kwargs
            work = real(*device.tree_map(device._unwrap, args), **device.tree_map(device._unwrap, kwargs))
        # what didn't move, or a captured one at each replay, wasn't computed
        never = not (moved and values & COMPUTED)
        kept = [t._data for t in written if t._data.device.type != "meta"]

        def flag():
            for t in kept:
                device.set_never_computed(t, never)

        flag()
        clock.on_replay(flag)
        return work

    return call


def _barrier(group=None, async_op=False, device_ids=None):
    """A barrier meets as a collective (a meet is a barrier): NCCL's, a one-element all-reduce
    on a device group, pairs against its peers' collectives. mstar's barriers are on device
    groups; the fused-AR stand-in's, on a host group, is gloo's own (``__wrapped__``)."""
    for index in device_ids or ():
        cuda.device_index(index)
    if dist.get_world_size(group) > 1:
        _meet(group or dist.group.WORLD, row("Barrier", size=dist.get_world_size(group)))
    return _Done() if async_op else None


# name -> (cost family, NCCL primitive, or None for point to point)
_COLLECTIVES = {
    "all_reduce": ("all_reduce", "AllReduce"),
    "all_gather_into_tensor": ("all_gather", "AllGather"),
    "all_gather": ("all_gather", "AllGather"),
    "reduce_scatter_tensor": ("reduce_scatter", "ReduceScatter"),
    "reduce_scatter": ("reduce_scatter", "ReduceScatter"),
    "all_to_all_single": ("alltoall", "AllToAll"),
    "broadcast": ("all_gather", "Broadcast"),
    "reduce": ("all_reduce", "Reduce"),
    "send": ("all_gather", None),
    "recv": ("all_gather", None),
    "isend": ("all_gather", None),
    "irecv": ("all_gather", None),
}


def _gloo(backend):
    if backend is None:
        return None
    text = str(backend)
    if "nccl" in text:
        return "gloo"
    return backend


def install() -> None:
    init = dist.init_process_group

    @functools.wraps(init)
    def init_process_group(*args, **kwargs):
        if args:
            args = (_gloo(args[0]), *args[1:])
        if "backend" in kwargs:
            kwargs["backend"] = _gloo(kwargs["backend"])
        cuda.device_index(kwargs.pop("device_id", None))  # refused as CUDA would; no device to bind a gloo group to
        return init(*args, **kwargs)

    dist.init_process_group = init_process_group
    new_group = dist.new_group

    @functools.wraps(new_group)
    def new_group_gloo(*args, **kwargs):
        if "backend" in kwargs:
            kwargs["backend"] = _gloo(kwargs["backend"])
        return new_group(*args, **kwargs)

    dist.new_group = new_group_gloo
    default_backend = dist.get_default_backend_for_device
    dist.get_default_backend_for_device = (
        lambda d: "gloo" if str(getattr(d, "type", d)).startswith("cuda") else default_backend(d)
    )
    for name, (family, primitive) in _COLLECTIVES.items():
        if hasattr(dist, name):
            setattr(dist, name, _collective(name, family, primitive))
    dist.barrier = functools.wraps(dist.barrier)(_barrier)
