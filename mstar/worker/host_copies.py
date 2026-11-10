"""Host copies of a step's outputs: the pinned staging buffers they go through
and the side stream that copies them without waiting on the next step.

The default stream has GPU(N+1) queued behind GPU(N)'s outputs once N+1 is
launched, so syncing on it would also drain GPU(N+1) and erase the overlap. The
side stream waits on ``output.completion_event`` (recorded after GPU(N)) and then
runs an isolated D2H, so the main thread only blocks on the copy.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.model.submodule_base import BatchedModelOutput, HostRows
from mstar.utils.device import DeviceCapability

# pinned host-row buffers in flight at once (see HostCopies.stage_host_rows)
_HOST_ROWS_RING = 4
_HOST_SPANS_PURPOSE = tuple(f"output_spans{slot}" for slot in range(_HOST_ROWS_RING))


def _host_column(host_rows: HostRows | None, name: str, width: int = 1) -> torch.Tensor | None:
    """Host buffer ``name``'s first ``width`` values per row, a view; None without one."""
    buf = None if host_rows is None else host_rows.buffers.get(name)
    if not torch.is_tensor(buf) or buf.dim() == 0 or buf.numel() == 0:
        return None
    rows = buf.reshape(buf.shape[0], -1)
    return rows[:, 0] if width == 1 else rows[:, :width]


class _LazyPerRid(Mapping):
    """``rows_to_per_rid`` on demand: the per-rid host dicts are built
    the first time a consumer reads them. The batched stop check, the row store
    and the inline emit read the rows directly, so in the common decode step
    nobody does, and the per-request dicts are never built."""

    __slots__ = ("_host", "_request_ids", "_built")

    def __init__(self, host: dict, request_ids: list[int]):
        self._host = host
        self._request_ids = request_ids
        self._built: dict | None = None

    def _dict(self) -> dict:
        if self._built is None:
            self._built = rows_to_per_rid(self._host, self._request_ids)
        return self._built

    def __getitem__(self, key):
        return self._dict()[key]

    def __iter__(self):
        return iter(self._dict())

    def __len__(self) -> int:
        return len(self._request_ids)

    # the built dict's own views, rather than Mapping's per-key lookups
    def items(self):
        return self._dict().items()

    def keys(self):
        return self._dict().keys()

    def values(self):
        return self._dict().values()


class _LazyHostOutputs(Mapping):
    """Per-rid host copies of a step's outputs, copied by output name on first
    read (``copy_names(names) -> {name: {rid: host value}}``, one sync per call).

    The stop check reads at most the sampled token, and a loop that stops on
    its iteration count reads nothing, so copying every output down each step
    paid for transfers nobody read. Outputs that leave by a host transport are
    copied by the send instead; ``landed`` hands it what was copied here.
    """

    __slots__ = ("_source", "_copy_names", "_landed")

    def __init__(self, source: dict, copy_names: Callable[[list[str]], dict]):
        self._source = source
        self._copy_names = copy_names
        self._landed: dict[str, dict] = {}

    def _copy(self, names) -> None:
        missing = [n for n in dict.fromkeys(names) if n not in self._landed]
        if missing:
            self._landed.update(self._copy_names(missing))

    def _host(self, name: str) -> dict:
        self._copy((name,))
        return self._landed[name]

    def landed(self, names=()) -> dict[int, NameToTensorList]:
        """What was copied, ``names`` first copied too, in one batched copy
        and one sync for a host transport, which would otherwise copy each
        tensor on its own."""
        self._copy(names)
        out: dict[int, NameToTensorList] = {}
        for name, by_rid in self._landed.items():
            for rid, value in by_rid.items():
                out.setdefault(rid, {})[name] = value
        return out

    def get(self, rid, default=None):
        row = self._source.get(rid)
        if row is None:
            return default
        return _LazyHostRow(self, rid, row) if isinstance(row, dict) else row

    def __getitem__(self, rid):
        row = self._source[rid]
        return _LazyHostRow(self, rid, row) if isinstance(row, dict) else row

    def __iter__(self):
        return iter(self._source)

    def __len__(self) -> int:
        return len(self._source)


class _LazyHostRow(Mapping):
    """One request's view of ``_LazyHostOutputs``: reading a name copies it."""

    __slots__ = ("_parent", "_rid", "_row")

    def __init__(self, parent: _LazyHostOutputs, rid, row: dict):
        self._parent, self._rid, self._row = parent, rid, row

    def __getitem__(self, name):
        value = self._row[name]
        # only a list of tensors has anything to copy
        return self._parent._host(name)[self._rid] if isinstance(value, list) else value

    def __contains__(self, name) -> bool:
        return name in self._row

    def __iter__(self):
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)


def rows_to_per_rid(
    host: dict, request_ids: list[int],
) -> dict[int, NameToTensorList]:
    """Slice row-addressed buffers into the per-rid form ``check_stop``
    reads: row i goes to ``request_ids[i]``."""
    # One ``split`` per buffer makes every row view in a single call.
    rows = {
        name: buf.split(1)
        for name, buf in host.items()
        if torch.is_tensor(buf) and buf.shape
    }
    out: dict[int, NameToTensorList] = {}
    for i, rid in enumerate(request_ids):
        out[rid] = {
            name: [views[i]] for name, views in rows.items() if i < len(views)
        }
    return out


def crop_to_spans(outputs: "BatchedModelOutput") -> dict[int, list[str]]:
    """Cut each row of a spanned output to its span along the last dim
    (``NodeSubmodule.spanned_outputs``). Called once the step's completion
    event has fired, so the spans staged behind the step are on the host.
    The cut is a view, and it replaces the tensor in its list in place, so
    a second name that ``postprocess`` or ``unpack_packed_outputs`` gave
    the same list is cut with it. Two names the forward returned on its own
    get their own lists from the engine and keep their own widths.

    Returns rid -> the outputs whose span is empty, which the caller leaves
    out of the store as it does a withheld output.
    """
    spans = outputs.host_spans or outputs.output_spans
    per_rid = outputs.per_rid_outputs
    cut = False
    empty: dict[int, list[str]] = {}
    for name, span in spans.items():
        # columns, not a [start, end] list per row
        starts, ends = span.t().tolist()
        for rid, start, end in zip(outputs.span_request_ids, starts, ends, strict=True):
            row = per_rid.get(rid)
            tensors = row.get(name) if row is not None else None
            if not tensors:
                continue
            if end <= start:
                empty.setdefault(rid, []).append(name)
            for i, tensor in enumerate(tensors):
                if start > 0 or end < tensor.shape[-1]:
                    tensors[i] = tensor[..., max(start, 0):max(end, 0)]
                    cut = True
    if cut:
        # the views no longer describe the stored rows
        outputs.row_views = None
    return empty


class HostCopies:
    def __init__(self, device: torch.device, device_capability: DeviceCapability):
        self.device = device
        self.device_capability = device_capability
        self._host_rows_slot = 0
        # Lazy-initialized: workers without CUDA never touch it.
        self._d2h_stream: torch.Stream | None = None
        self._pinned_d2h_buffers: dict[
            tuple[str, torch.dtype, int], list[torch.Tensor]
        ] = defaultdict(list)

    def stage_host_rows(self, outputs: "BatchedModelOutput") -> None:
        """Start the device-to-host copy of the step's row-addressed stop
        buffers here on the gpu thread, on the execution stream right behind
        the clone that made them, so the completion event recorded next
        covers it. The main thread's stop check then finds the rows landed
        once it has waited on that event: no side stream, no second sync, and
        one fewer GIL release on its critical path.

        A small ring of pinned buffers: step N's rows are read by the main
        thread while this thread already stages N+1's (the pipeline is one
        step deep, the ring leaves slack).

        The step's output spans (``BatchedModelOutput.output_spans``) take the
        same path, so cutting the outputs to them costs no wait of its own.
        """
        buffers = outputs.check_stop_buffers
        rows = bool(buffers) and outputs.row_request_ids is not None
        if not rows and not outputs.output_spans:
            return
        slot = self._host_rows_slot = (self._host_rows_slot + 1) % _HOST_ROWS_RING
        if rows:
            outputs.host_rows = HostRows(
                tuple(outputs.row_request_ids),
                self._copy_to_pinned(buffers, f"host_rows{slot}"),
            )
        if outputs.output_spans:
            outputs.host_spans = self._copy_to_pinned(
                outputs.output_spans, _HOST_SPANS_PURPOSE[slot],
            )

    def _copy_to_pinned(self, buffers: dict, purpose: str) -> dict:
        """Queue a non-blocking copy of each device tensor in ``buffers`` into
        a pinned buffer of ``purpose``; anything else passes through."""
        host: dict[str, torch.Tensor] = {}
        # a tensor under two names is copied once
        landed: dict[int, torch.Tensor] = {}
        for index, (name, tensor) in enumerate(buffers.items()):
            if not (
                torch.is_tensor(tensor)
                and self.device_capability.is_accelerator
                and tensor.device.type == self.device.type
            ):
                host[name] = tensor
                continue
            if id(tensor) in landed:
                host[name] = landed[id(tensor)]
                continue
            buf = self._get_pinned_d2h_buffer(
                purpose, tensor.shape, tensor.dtype, index,
            )
            buf.copy_(tensor, non_blocking=True)
            host[name] = landed[id(tensor)] = buf
        return host

    def _get_pinned_d2h_buffer(
        self,
        purpose: str,
        shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        index: int = 0,
    ) -> torch.Tensor:
        numel = torch.Size(shape).numel()
        key = (purpose, dtype, self._pinned_size(numel))
        buffers = self._pinned_d2h_buffers[key]
        while len(buffers) <= index:
            buffers.append(
                torch.empty(key[2], dtype=dtype, device="cpu", pin_memory=True)
            )
        return buffers[index][:numel].view(shape)

    @staticmethod
    def _pinned_size(numel: int) -> int:
        return 1 << max(numel - 1, 0).bit_length()

    def _d2h_batched(
        self, buffers: dict, side: torch.Stream,
    ) -> dict[str, torch.Tensor]:
        """One device-to-host copy per named buffer, landed before this returns.

        Replaces a copy per tensor per request (48 at decode batch 16). Row i
        belongs to the i-th request in forward order
        (``BatchedModelOutput.row_request_ids``), not the batch's current
        request list, which has already dropped requests stopped a step ago.
        Padded rows past the real ones are not read. The host buffers are
        reused by the next step.
        """
        host: dict[str, torch.Tensor] = {}
        # a tensor under two names is copied once
        landed: dict[int, torch.Tensor] = {}
        with side:
            for index, (name, tensor) in enumerate(buffers.items()):
                if not (
                    torch.is_tensor(tensor)
                    and tensor.device.type == self.device.type
                ):
                    host[name] = tensor
                    continue
                if id(tensor) in landed:
                    host[name] = landed[id(tensor)]
                    continue
                buf = self._get_pinned_d2h_buffer(
                    "check_stop_batched", tensor.shape, tensor.dtype, index,
                )
                buf.copy_(tensor, non_blocking=True)
                host[name] = landed[id(tensor)] = buf
        side.synchronize()
        return host

    def prematerialize_for_check_stop(
        self,
        outputs: "BatchedModelOutput",
        completion_event: torch.Event | None,
        request_ids: list[int] | None = None,
    ) -> tuple[Mapping[int, Mapping[str, list[torch.Tensor]]], HostRows | None]:
        """Side-stream D→H of the CUDA tensors in ``outputs`` so the subsequent
        ``check_stop`` reads (typically ``.item()`` on the sampled token)
        don't trigger a default-stream sync. With same-thread async,
        GPU(N+1)'s kernels are already queued on default stream behind
        N's outputs by the time we get here — a default-stream sync would
        block waiting for N+1 to finish, defeating the overlap.

        Returns per-rid host outputs, plus ``HostRows`` when the submodule
        handed over row-addressed buffers, for a batched stop check. Without
        those, the outputs are a ``_LazyHostOutputs``, which copies an output
        name for every rid the first time it is read. Skipped (returns
        ``outputs`` unchanged) when there's no completion event (CPU execution)
        or when CUDA is unavailable.

        The host tensors are views of pinned buffers the next step reuses. Their
        one other reader, a ``needs_cpu_tensor`` transport, gets them through
        the store and sends in ``_register_outputs`` later in the same
        ``postprocess``, which then drops them from the store.
        """
        source = outputs.get_check_stop_input()
        # Rows are in forward order, which the engine stamped on the output.
        row_rids = (
            list(outputs.row_request_ids)
            if outputs.row_request_ids is not None else request_ids
        )
        landed = outputs.host_rows
        if landed is not None and completion_event is not None:
            # staged by the gpu thread behind the step; the caller has waited
            # on the completion event, so the rows are here
            return _LazyPerRid(landed.buffers, list(landed.request_ids)), landed
        if (
            completion_event is None
            or not self.device_capability.is_available
        ):
            if outputs.check_stop_buffers is not None and row_rids is not None:
                # host tensors already (a CPU device): only the re-keying
                host = outputs.check_stop_buffers
                return (
                    rows_to_per_rid(host, row_rids),
                    HostRows(tuple(row_rids), host),
                )
            return source, None
        if not source:
            return source, None

        if self._d2h_stream is None:
            self._d2h_stream = torch.Stream(device=self.device)
        side = self._d2h_stream
        side.wait_event(completion_event)

        if outputs.check_stop_buffers is not None and row_rids is not None:
            host = self._d2h_batched(outputs.check_stop_buffers, side)
            return (
                _LazyPerRid(host, row_rids),
                HostRows(tuple(row_rids), host),
            )

        # Shared by every name copied this step, so two names never land in
        # one pinned buffer.
        buffer_indices: dict[tuple[str, torch.dtype, int], int] = defaultdict(int)

        def copy_names(names: list[str]) -> dict[str, dict]:
            host: dict[str, dict] = {name: {} for name in names}
            with side:
                for name in names:
                    for rid, name_to_list in source.items():
                        if not isinstance(name_to_list, dict) or name not in name_to_list:
                            continue
                        tensors = name_to_list[name]
                        if not isinstance(tensors, list):
                            host[name][rid] = tensors
                            continue
                        new_list = []
                        for t in tensors:
                            if torch.is_tensor(t) and t.device.type == self.device.type:
                                # by size class, not shape: a per-shape count gives
                                # two shapes in one class the same buffer
                                key = ("check_stop", t.dtype, self._pinned_size(t.numel()))
                                idx = buffer_indices[key]
                                buffer_indices[key] += 1
                                cpu_t = self._get_pinned_d2h_buffer(
                                    "check_stop", t.shape, t.dtype, idx,
                                )
                                cpu_t.copy_(t, non_blocking=True)
                                new_list.append(cpu_t)
                            else:
                                new_list.append(t)
                        host[name][rid] = new_list
            side.synchronize()
            return host

        return _LazyHostOutputs(source, copy_names), None
