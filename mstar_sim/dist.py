"""``torch.distributed`` across emulated GPUs.

Every emulated GPU is its own process, as on a real node. Process groups that
would be NCCL are gloo (the control traffic mstar sends over CPU groups is
real). Ranks meet at each collective over simulated-GPU tensors, as NCCL's do
(:func:`_meet`): each rank's description of the call is gathered on its group,
and a rank whose call doesn't pair with its peers' (skipped, reordered,
resized) aborts with every rank's, where NCCL would hang or corrupt data. A
captured graph meets once per group at each replay, on its collectives over that
group's ranks. The data moves (over gloo) only if every rank holds values (a vote,
lengths); otherwise its outputs are never computed. Either way it is charged its
NCCL time (:func:`mstar_sim.cost.collective_seconds`, measured tables where
they exist) on the current stream, on every rank alike, so timing stays
symmetric, as SGLang's simulator and Maya assume.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import os
import resource
import sys
import threading
import time

import torch
import torch.distributed as dist

from mstar_sim import clock, cost, device

_all_gather = dist.all_gather_into_tensor  # the real one: a meet's own traffic


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
# (primitive, numel, dtype, op, root, holds values, elements sent to each rank, received
# from each). The first five must be equal on every rank, and what rank i sends rank j is
# what j receives from i; the AND of the sixth says whether the data moves.
_PRIMITIVES = ("AllReduce", "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllToAll",
               "AllReduceFusion", "Graph", "Barrier")
_DTYPES = tuple(dict.fromkeys(v for v in vars(torch).values() if isinstance(v, torch.dtype)))
_HELD = 5


def row(primitive: str, numel: int = 0, dtype: torch.dtype | None = None, op: int = 0, root: int = 0,
        held: bool = False, send: list[int] | None = None, recv: list[int] | None = None, size: int = 1):
    """A collective's descriptor (``send``/``recv``: elements to and from each of the ``size`` ranks)."""
    return [_PRIMITIVES.index(primitive), numel, _DTYPES.index(dtype) if dtype is not None else -1, op, root,
            int(held), *(send or [0] * size), *(recv or [0] * size)]


def _describe(r: list[int]) -> str:
    name = _PRIMITIVES[r[0]]
    if name == "Graph":
        return f"a graph replay of {r[1]} collectives (sequence hash {r[3] & (2**64 - 1):016x})"
    size = (len(r) - _HELD - 1) // 2
    text = f"{name}(numel={r[1]}, dtype={_DTYPES[r[2]] if r[2] >= 0 else None}, op={r[3]}, root={r[4]}"
    if name == "AllToAll":
        text += f", sends={r[_HELD + 1:_HELD + 1 + size]}, receives={r[_HELD + 1 + size:]}"
    return text + ")"


def _pairs(rows: list[list[int]]) -> bool:
    size = len(rows)
    send = [r[_HELD + 1:_HELD + 1 + size] for r in rows]
    recv = [r[_HELD + 1 + size:] for r in rows]
    return (all(r[:_HELD] == rows[0][:_HELD] for r in rows)
            and all(send[i][j] == recv[j][i] for i in range(size) for j in range(size)))


def _abort() -> None:
    """Abort, as NCCL's watchdog does, after the stacks ``faulthandler`` prints: without a
    core file, which would hold only the emulator."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.abort()


def _meet(group, mine: list[int], replay: bool = False) -> list[list[int]]:
    """Meet the group's other ranks at a collective (``replay``: a graph's replay) and return
    every rank's descriptor. Blocks until they all come, as the GPU's kernel does. A
    mismatch aborts (NCCL's behaviour there is undefined: a hang or corrupt data, and a
    raise would be caught and served past). A peer that never comes within the group's
    timeout aborts an eager call, as NCCL's watchdog does, and hangs a replay, whose
    collectives nothing watches. ``kill -USR1`` a peer for its stack."""
    ranks = dist.get_process_group_ranks(group)
    me = dist.get_rank()
    seq = group._get_sequence_number_for_group()  # torch's count of the group's gloo ops
    where = f"mstar_sim: rank {me}, torch sequence number {seq} (meets included) on the group of ranks {ranks}"
    with clock.computing():  # the device's clock stands still while ranks meet
        gathered = torch.empty(len(ranks) * len(mine), dtype=torch.int64)
        began = time.perf_counter()
        try:
            _all_gather(gathered, torch.tensor(mine, dtype=torch.int64), group=group)
            waited = time.perf_counter() - began  # for its peers: a wait, not the emulator's work
            clock.STATS["host_compute"] -= waited
            clock.STATS["host_wait"] += waited
        except RuntimeError as e:
            print(f"{where}: {_describe(mine)} met no peer ({e}). "
                  + ("A replayed graph's collectives are unwatched: this rank hangs, as on a GPU." if replay else
                     "NCCL's watchdog aborts here (or the emulator's host-side meet hung: README Limits)."),
                  file=sys.stderr, flush=True)
            if replay:
                threading.Event().wait()
            _abort()
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


def collective(family: str, group, nbytes: int, desc: list[int] | None) -> bool:
    """A collective on ``group`` (None: the default one) described by ``desc`` (:func:`row`;
    None for point to point, which has no group to meet on): charged its NCCL time (cost
    ``family``) on the current stream, then met with the group's other ranks, or recorded on
    the graph being captured, whose replays meet. Returns whether its data can move: every
    other rank holds its values, and it isn't captured."""
    size = dist.get_world_size(group) if dist.is_initialized() else 1
    tape = clock.capturing()
    if not clock.FAST:
        clock.charge(cost.collective_seconds(family, size, nbytes, graph=tape is not None))
    if desc is None or size == 1:
        return tape is None
    group = group or dist.group.WORLD
    if tape is not None:
        tape.collectives.append((group, desc[:_HELD]))
        return False
    return all(r[_HELD] for r in _meet(group, desc))


def seal(tape: clock.Tape) -> None:
    """At the end of a capture, fold the tape's collectives into what each replay meets on:
    one row per group (in first-use order) with the count and a hash of the captured
    collectives over its ranks, on it or on another group of the same ranks (a device group
    and its fused-AR channel). Every rank of a group holds each of those groups, so the rows
    agree unless the ranks captured differently (a wrong bucket, an order inversion across
    those groups); a replay is one host call, so nothing sits between them. Groups of other
    ranks meet on their own, in first-use order, as NCCL's communicators do."""
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
# and all but these leave nothing of what was there
_READS_WHAT_IT_WRITES = {"all_reduce", "reduce", "broadcast"}


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
        nbytes = max(t.numel() * t.element_size() for t in tensors)
        held = all(t._data.device.type != "meta" for t in tensors)
        written = [] if name in _ONLY_READS else _sim_tensors((a[first],), {})
        desc = None  # point to point (mstar sends none)
        if primitive is not None:
            size = dist.get_world_size(group) if dist.is_initialized() else 1
            send = recv = None
            if primitive == "AllToAll":
                send = _splits(a["input"], a["input_split_sizes"], size)
                recv = _splits(a["output"], a["output_split_sizes"], size)
            root = next((a[k] for k in ("src", "dst", "group_src", "group_dst") if a.get(k) is not None), -1)
            op = a.get("op")
            desc = row(primitive, 0 if send else sum(t.numel() for t in written), (written or tensors)[0].dtype,
                       op=int(op) if op is not None else 0, root=root, held=held, send=send, recv=recv, size=size)
        moved = collective(family, group, nbytes, desc) and held
        work = _Done() if kwargs.get("async_op") else None
        if moved:
            # every rank holds values (a vote, lengths): send them for real, so
            # what ranks agree on device is what they agreed on
            work = real(*device.tree_map(device._unwrap, args), **device.tree_map(device._unwrap, kwargs))
        covered = [] if name in _READS_WHAT_IT_WRITES else [t for t in written if device._covers(t._data)]
        # what it wrote is computed if all it read here was (the fact doesn't cross ranks); what
        # didn't move, or a captured one at each replay, wasn't
        never = not moved or any(device.never_computed(t._data) for t in tensors
                                 if not any(t is c for c in covered))
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
    on a device group, pairs against its peers' collectives; on a host group it waits as
    gloo's does, aborting at the group's timeout where gloo's raises."""
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
        kwargs.pop("device_id", None)  # no device to bind a gloo group to
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
