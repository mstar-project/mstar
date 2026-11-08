"""What layers and submodules hold for declared state: a handle per array or sampler, over today's resources.

A call that touches state runs behind a graph break, so it is plain Python in eager steps and at capture, it never runs
on a replay, and its checks live in it.
"""

from collections.abc import Mapping
from typing import Any

import torch

from mstar.engine.resources.step import StepHandle
from mstar.engine.state.contract import Array, ArrayStep, Bidirectional, GatedDeltaNet, Sampler, is_value
from mstar.engine.state.host import HostFields, is_host_value


class ArrayHandle(StepHandle):
    """A node's calls on one declared array that keeps entries, or on a Value.

    ``layer`` is always the model's layer id as declared: the int, or a 0-d int64 CPU tensor holding it. A compiled
    caller passes the tensor, so that its frame doesn't guard on the layer and one frame serves every layer. Each row's
    label comes from the step's segments. A handle holds each layer's state views, built once at bind, and the
    resources hold the step's plan.

    ``queried`` is the array whose segments lay out the queries of a cross read.
    """

    def __init__(
        self, array: Array, resources: Mapping[str, Any], positions_key: str | None,
        host_fields: HostFields | None = None, *, queried: Array | None = None,
    ):
        self.array = array
        self.host_fields = host_fields
        self.queried = queried
        # the state ops whose fused kernel can write this array's pages (`bind_state_ops`)
        self.fused_ops: frozenset[str] = frozenset()
        self.name = array.name
        self.label = self._query_label = "main"
        self.reads = {key: resources[key] for key in array.reads}
        self.first_read = next(iter(self.reads.values()), None)
        self.positions_key = positions_key
        self.positions = None if positions_key is None else resources[positions_key]
        rope = None if array.positions is None else array.positions.rope
        self.rope_dtype = None if rope is None else rope.dtype

        self._pool_layer = {layer: i for i, layer in enumerate(sorted(array.layers))}
        self._attn = {
            layer: self.reads[key]
            for key, read in array.reads.items() for layer in read.layers or array.layers
        }
        # each read's `AttentionStep.causal`, built once since `state_steps` runs on every step, replays too
        self.causal_reads = {key: not isinstance(read.mask, Bidirectional) for key, read in array.reads.items()}
        self._kv = None if is_value(array) or not array.max_len else resources[array.name]
        self._views = {} if self._kv is None else {
            layer: self._kv.layer_view(i) for layer, i in self._pool_layer.items()
        }
        self._pool = resources[array.name] if is_value(array) and array.entry.tensors else None
        self._linear = next(
            (resources[key] for key, update in array.updates.items() if isinstance(update, GatedDeltaNet)), None,
        )
        self._blocks = {} if self._pool is None else {
            field: {layer: self._pool.block(field, i) for layer, i in self._pool_layer.items()}
            for field in array.entry.tensors
        }

        # This step's calls, checked in eager steps and at capture.
        # `_held` is the k and v an add left for `_held_layer`'s read, on a kernel that takes them there.
        # `_kept` holds the layers whose kept add has run, `_scratch` those with a scratch add, `_read` those read,
        # and `_updated` the update calls run. `_grows` says whether `check_step` needs an add, for a step that lists
        # a row on the pages at span above 0.
        self._open = False
        self._keep = True
        self._capturing = False
        self._grows = False
        self._held: tuple[torch.Tensor, torch.Tensor] | None = None
        self._held_layer: int | None = None
        self._kept: set[int] = set()
        self._scratch: set[int] = set()
        self._read: set[int] = set()
        self._updated: set[tuple[str, int]] = set()

    def _refusal(self, cause: str) -> ValueError:
        return ValueError(f"Array {self.name!r}: {cause}")

    def _no_layout(self, missing: str) -> ValueError:
        return self._refusal(f"no layout yet: {missing}")

    def _layer_id(self, layer: int | torch.Tensor) -> int:
        if not isinstance(layer, torch.Tensor):
            return layer
        # A buffer would move to the GPU, and `int()` would then sync at capture.
        if layer.ndim or layer.dtype != torch.int64 or layer.device.type != "cpu":
            raise self._refusal(f"a layer tensor must be 0-d int64 on the CPU, and this one is {layer.dtype} of shape "
                                f"{tuple(layer.shape)} on {layer.device}")
        return int(layer)

    def begin_step(self, step) -> None:
        if self._open:
            raise self._refusal("a step began while another is open")
        self._open = True
        declared: ArrayStep = step.state[self.name]
        segments = declared.segments or step.segments
        if segments:
            self.label = segments[0].label
        # on a replay too, whose `eager_add` adds still run
        self._keep = declared.keep
        self._capturing = step.ctx.capture
        for calls in (self._kept, self._scratch, self._read, self._updated):
            calls.clear()
        # before the replay's return, since a captured step's writes reach this on a replay
        if declared.host:
            self._check_host_writes(declared, step.ctx)
        if step.ctx.replays:
            return
        self._check_step(declared, segments or (), step.ctx)
        # a capture opens the handles once around `preprocess`, where `eager_add` adds run, and again around the forward
        self._grows = self._kv is not None and not (declared.eager_add and step.ctx.capture) and any(
            segment.span and (segment.request_id, segment.label) not in declared.placeholders
            for segment in segments or ()
        )
        if self.queried is None:
            return
        queries = step.state[self.queried.name].segments or step.segments or ()
        if queries:
            self._query_label = queries[0].label
        if [s.request_id for s in queries] != [s.request_id for s in segments or ()]:
            raise self._refusal(f"its segments and those of {self.queried.name!r}, which lay out its queries, must "
                                "name the same requests in the same order, with one segment to one")

    def check_step(self, step) -> None:
        if step.ctx.replays or not self._keep:
            return
        only_scratch = self._scratch - self._kept
        if only_scratch:
            raise self._refusal(f"layers {sorted(only_scratch)} end on a scratch add in a step that keeps its adds, "
                                "and the step's commit would keep that add")
        if self._kept and self._kept != self.array.layers:
            raise self._refusal(f"layers {sorted(self.array.layers - self._kept)} have no add in a step that keeps "
                                "adds on others, and every layer stays the same length")
        if self._grows and not self._kept:
            raise self._refusal("a step that keeps lists rows at span above 0 and adds nothing, so the pages "
                                "would keep entries no add wrote; list those rows at span 0")

    def commit_step(self, step) -> None:
        if self.host_fields is not None:
            self.host_fields.commit(self.array, step.state[self.name], step.ctx.request_ids)

    def end_step(self, step) -> None:
        self._open = False
        self._held = self._held_layer = None

    def _check_host_writes(self, declared: ArrayStep, ctx) -> None:
        if self.host_fields is None:
            raise self._refusal("a host write, and the array declares no host fields")
        if not declared.keep:
            raise self._refusal("a host write on a step that keeps nothing, which has no commit to apply it at")
        for (row, label, layer), written in declared.host.items():
            where = f"a host write on row {row}, label {label!r}, layer {layer}"
            if not ctx.is_real_row(row):
                raise self._refusal(f"{where}, and that row isn't one of the step's requests")
            if label not in self.array.labels or layer not in self.array.layers:
                raise self._refusal(f"{where}, and the array declares labels {sorted(self.array.labels)} and layers "
                                    f"{sorted(self.array.layers)}")
            if not self.host_fields.is_open(row):
                raise self._refusal(f"{where}, and that request has no open fields")
            for field, value in written.items():
                kind = self.array.entry.host.get(field)
                if kind is None:
                    raise self._refusal(f"{where}: field {field!r} isn't declared, of {sorted(self.array.entry.host)}")
                if not is_host_value(value, kind):
                    raise self._refusal(f"{where}: field {field!r} takes {kind}, and got {value!r}")

    def _check_step(self, declared: ArrayStep, segments, ctx) -> None:
        declared_labels = self.array.labels
        labels = {segment.label for segment in segments}
        if labels - declared_labels:
            raise self._refusal(f"a step touches labels {sorted(labels - declared_labels)}, and the array declares "
                                f"{sorted(declared_labels)}")
        if len(labels) > 1:
            raise self._no_layout(f"a step touches labels {sorted(labels)}, and a plan holds one per call")
        forked = {label for fork in declared.forks for label in (fork.src, fork.dst)}
        if forked - declared_labels:
            raise self._refusal(f"a fork names labels {sorted(forked - declared_labels)}, and the array declares "
                                f"{sorted(declared_labels)}")
        if not declared.keep and self.array.updates:
            raise self._refusal("keep=False on an array with updates, whose kernels write in place, until saved "
                                "copies exist")
        if self.positions is None and (declared.advance is not None or declared.position_ids is not None):
            raise self._refusal("a step sets advance or position_ids, and the array declares no positions")
        if self._kv is None and (declared.placeholders or declared.eager_add):
            raise self._refusal("a step sets placeholders or eager_add, and the array has no paged entries to add to")
        if declared.forks and not self.array.max_len:
            raise self._refusal("a step forks an array with max_len 0, which keeps nothing to fork")
        # A capture's dummy rows open only the default label.
        if self._kv is None or ctx.capture:
            return
        for segment in segments:
            if (segment.request_id, segment.label) not in declared.placeholders and not self._kv.opens_label(
                segment.request_id, segment.label,
            ):
                raise self._refusal(f"a step touches label {segment.label!r} of request {segment.request_id}, "
                                    "which it didn't open")

    def _check_open(self, call: str) -> None:
        if not self._open:
            raise self._refusal(f"{call} in a step that doesn't list the array")

    def _record_add(self, layer: int, keep: bool | None) -> None:
        self._check_open("add")
        if layer not in self.array.layers:
            raise self._refusal(f"add on layer {layer}, which the array doesn't declare")
        if layer in self._read:
            raise self._refusal(f"an add on layer {layer} after a read in the same layer")
        if layer in self._kept:
            raise self._refusal(f"an add on layer {layer} after its kept add")
        if not self.array.max_len:
            return
        if keep and not self._keep:
            raise self._refusal(f"keep=True on layer {layer} in a step with keep=False, whose commit keeps no add")
        (self._kept if self._keep and keep is not False else self._scratch).add(layer)

    def _record_read(self, layer: int) -> None:
        self._check_open("read")
        if layer not in self.array.layers:
            raise self._refusal(f"read on layer {layer}, which the array doesn't declare")
        if self._held_layer not in (None, layer):
            raise self._refusal(f"a read on layer {layer} would take the add of layer {self._held_layer}")
        self._read.add(layer)

    def _record_update(self, field: str, layer: int) -> None:
        self._check_open(f"the update of {field}")
        if (field, layer) in self._updated:
            raise self._refusal(f"a second update of {field} on layer {layer} in one step")
        self._updated.add((field, layer))

    def block(self, field: str, layer: int) -> torch.Tensor:
        """``layer``'s view of the entry field ``field``, for an update's op."""
        blocks = self._blocks.get(field)
        if blocks is None or layer not in blocks:
            raise self._refusal(f"no field {field!r} on layer {layer}, of layers {sorted(self._pool_layer)}")
        return blocks[layer]

    def _reads_pages(self, layer: int) -> bool:
        """Whether this layer's add goes to the pages, rather than waiting for a read whose kernel takes k and v. A
        cross read attends the pages, and its resource has no ``requires_kv_write``."""
        if self.queried is not None:
            return True
        attn = self._attn.get(layer)
        return attn is None or attn.requires_kv_write

    def _holds(self, layer: int) -> bool:
        return not self._reads_pages(layer)

    def _take_hold(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        held = self._held
        if held is None:
            raise self._refusal(f"a read on layer {layer} with no add before it")
        self._held = None
        return held

    @torch.compiler.disable
    def add(self, layer: int | torch.Tensor, *, keep: bool | None = None, **fields: torch.Tensor) -> None:
        """Append each row's span at its next log positions. A kept add happens once per layer per step, as the
        layer's last add. ``keep`` False is scratch: it can repeat, and lasts until the next add or the end of the step.
        None takes the step's ``keep``. The resources keep or drop a step's adds as a whole, so a step that keeps needs
        each layer to end on a kept add, and a step with ``keep`` False refuses ``keep=True``."""
        layer = self._layer_id(layer)
        self._record_add(layer, keep)
        if self._holds(layer):
            self._held, self._held_layer = (fields["k"], fields["v"]), layer
            return
        self._kv.write_kv(fields["k"], fields["v"], layer_idx=self._pool_layer[layer], label=self.label)

    @torch.compiler.disable
    def read(self, layer: int | torch.Tensor, q: torch.Tensor | None = None, **mask_args: Any) -> torch.Tensor:
        """Attend ``q`` through the read declared for ``layer``: kept entries from earlier steps, plus this layer's
        latest add, at the positions its mask reaches."""
        if q is None:
            raise self._no_layout("a read with no op")
        if mask_args:
            raise self._no_layout(f"mask arguments {sorted(mask_args)}, which no mask defines")
        layer = self._layer_id(layer)
        self._record_read(layer)
        if self.queried is not None:
            # the cross resource keys its plans by the query side's label
            return self._attn[layer].run(q, label=self._query_label, kv_cache_layer=self._views[layer])
        k, v = self._take_hold(layer) if self._holds(layer) else (None, None)
        return self._attn[layer].run(
            q, label=self.label, kv_cache_layer=self._views[layer], k=k, v=v, layer_idx=self._pool_layer[layer],
        )

    @torch.compiler.disable
    def write(self, layer: int | torch.Tensor, *, at: torch.Tensor | None = None, **fields: torch.Tensor) -> None:
        """Overwrite entries in place, on a rewritable array. ``at`` gives each token's log position, and is omitted
        for a Value."""
        if not self.array.rewritable:
            raise self._refusal("write on an append-only array")
        if at is not None:
            raise self._no_layout("write at positions, on anything but a Value")
        layer = self._layer_id(layer)
        self._check_open("write")
        for field, value in fields.items():
            self.block(field, layer)
            # stored K-last, as `_state_block` lays it out
            value = value.transpose(-1, -2) if field == "state" else value
            self._pool.scatter_(field, value, self._pool_layer[layer], label=self.label)


class SamplerHandle(StepHandle):
    """A node's calls on one declared sampler. Its rows are the step's, padding included."""

    def __init__(self, sampler: Sampler, resource: Any):
        self.sampler = sampler
        self.name = sampler.name
        self.resource = resource
        self.rows = ()
        self._open = False
        self._drawn = 0

    def _refusal(self, cause: str) -> ValueError:
        return ValueError(f"Sampler {self.name!r}: {cause}")

    def begin_step(self, step) -> None:
        if self._open:
            raise self._refusal("a step began while another is open")
        self._open = True
        self.rows = step.ctx.padded_request_ids
        self._drawn = 0

    def check_step(self, step) -> None:
        # a step that keeps no row's sample may skip the kernel: an eager prefill short of its last chunk
        if step.ctx.replays or (not self._drawn and step.state[self.name].kept_rows == frozenset()):
            return
        if self._drawn != self.sampler.draws:
            raise self._refusal(f"a step listing it sampled {self._drawn} times, and draws is {self.sampler.draws}")

    def commit_step(self, step) -> None:
        """Nothing: the resource holds a sampler's state."""

    def end_step(self, step) -> None:
        self._open = False


@torch.compiler.disable
def causal_conv(
    h: ArrayHandle, layer: int | torch.Tensor, x: torch.Tensor, *, weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """The declared `CausalConv` over ``x``, which reads and writes ``layer``'s ``conv`` field in place."""
    layer = h._layer_id(layer)
    conv = h.block("conv", layer)
    h._record_update("conv", layer)
    return h._linear.run_conv(x, conv_layer=conv, weight=weight, bias=bias, label=h.label)


@torch.compiler.disable
def gated_delta_net(
    h: ArrayHandle, layer: int | torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor,
    b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
) -> torch.Tensor:
    """The declared `GatedDeltaNet`, which reads and writes ``layer``'s ``state`` field in place."""
    layer = h._layer_id(layer)
    state = h.block("state", layer)
    h._record_update("state", layer)
    return h._linear.run(q, k, v, a, b, state_layer=state, a_log=a_log, dt_bias=dt_bias, label=h.label)


def host(h: ArrayHandle, request: int, label: str, layer: int | torch.Tensor) -> Mapping[str, object]:
    """One entry's host fields on the host, as the last committed step left them."""
    layer = h._layer_id(layer)
    fields = None if h.host_fields is None else h.host_fields.fields(request, h.name, label, layer)
    if fields is None:
        raise h._refusal(f"no host fields for request {request}, label {label!r}, layer {layer}")
    return fields


def position(h: ArrayHandle, rid_handle: int, label: str) -> int:
    """The next position on ``label`` of the request at worker handle ``rid_handle``, on the host."""
    return h.positions.position(rid_handle, label)


def positions(h: ArrayHandle) -> torch.Tensor:
    """This step's position ids in segment order, on the device. Under a graph lease the buffer runs to the bucket's
    length, past the step's tokens."""
    h._check_open("positions")
    return h.positions.pos_ids(h.label)


def rope(h: ArrayHandle, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``q`` and ``k`` rotated at this step's positions with the declared `Rope`."""
    h._check_open("rope")
    return h.positions.apply_qk(q, k, label=h.label, rope_dtype=h.rope_dtype or q.dtype)


@torch.compiler.disable
def sample(s: SamplerHandle, logits: torch.Tensor) -> torch.Tensor:
    """One token per row of ``logits``, a fresh draw that advances each kept row's sampling state."""
    if not s._open:
        raise s._refusal("sample in a step that doesn't list the sampler")
    s._drawn += 1
    if s._drawn > s.sampler.draws:
        raise s._refusal(f"sample number {s._drawn} in one step, and draws is {s.sampler.draws}")
    return s.resource.sample(list(s.rows), logits=logits)


# Eager as a whole, as `AttentionManager.select_last_hidden` explains.
@torch.compiler.disable
def at_last_token(h: ArrayHandle, x: torch.Tensor) -> torch.Tensor:
    """``x`` at each row's last token in this step, in row order."""
    h._check_open("at_last_token")
    if h.queried is not None:
        raise h._refusal(f"at_last_token lays x out by this array's segments, and its read's queries are laid out by "
                         f"{h.queried.name!r}; call it on that array")
    return h.first_read.select_last_hidden(x, label=h.label)


def last_sampled(s: SamplerHandle) -> torch.Tensor:
    """Each row's last sampled token, on the device, for a sampler the node names in its `DeviceLoopback`."""
    if not s._open:
        raise s._refusal("last_sampled in a step that doesn't list the sampler")
    return s.resource.loopback_tokens(s.rows)


def kernel_setting(h: ArrayHandle, read: str, field: str) -> Any:
    """The value of a `kernel_settings` field in effect, YAML included, until the engine picks kernels itself."""
    resource = h.reads[read]
    if resource is None:
        raise h._refusal(f"no {field} for read {read!r}, which runs SDPA and has no kernel")
    return getattr(resource, field)
