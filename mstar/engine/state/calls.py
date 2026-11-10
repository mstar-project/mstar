"""What layers and submodules hold for declared state: a handle per array or sampler, over today's resources.

A call that touches state runs behind a graph break, so it is plain Python in eager steps and at capture, it never runs
on a replay, and its checks live in it.
"""

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

import torch

from mstar.engine.resources.step import StepHandle
from mstar.engine.state.contract import (
    FUSED_DRAW_MAX_VOCAB,
    Array,
    ArrayStep,
    CausalConv,
    FusedDrawArgs,
    GatedDeltaNet,
    Op,
    Sampler,
)


class ArrayHandle(StepHandle):
    """A node's calls on one declared array. Each kind of storage has a handle that implements the calls it runs, and
    this class refuses the rest as having no layout yet.

    ``layer`` is always the model's layer id as declared: the int, or a 0-d int64 CPU tensor holding it. A compiled
    caller passes the tensor, so that its frame doesn't guard on the layer and one frame serves every layer. Each row's
    label comes from the step's segments. A handle holds each layer's state views, built once at bind, and the
    resources hold the step's plan.
    """

    # whether a step that lists the array has anything to do for it
    needs_step = True

    def __init__(self, array: Array, resources: Mapping[str, Any], positions_key: str | None):
        self.array = array
        # the state ops whose fused kernel can write this array's pages (`bind_state_ops`)
        self.fused_ops: frozenset[str] = frozenset()
        self.name = array.name
        self.label = "main"
        self.positions_key = positions_key
        self.positions = None if positions_key is None else resources[positions_key]
        rope = None if array.positions is None else array.positions.rope
        self.rope_dtype = None if rope is None else rope.dtype

        self._pool_layer = {layer: i for i, layer in enumerate(sorted(array.layers))}

        # This step's calls, checked in eager steps and at capture.
        # `_held` is the k and v an add left for `_held_layer`'s read, on a kernel that takes them there.
        # `_kept` holds the layers whose kept add has run, `_scratch` those with a scratch add, `_read` those read,
        # and `_updated` the update calls run.
        self._open = False
        self._keep = True
        self._capturing = False
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

    def _unsupported(self, call: str) -> ValueError:
        return self._no_layout(f"{call} on a {type(self).__name__}")

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

    def commit_step(self, step) -> None:
        """Nothing: the resources hold what a step keeps of the array."""

    def end_step(self, step) -> None:
        self._open = False
        self._held = self._held_layer = None

    def _check_host_writes(self, declared: ArrayStep, ctx) -> None:
        raise self._refusal("a host write, and the array declares no host fields")

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
        if self.positions is None and (declared.advance is not None or declared.position_ids is not None):
            raise self._refusal("a step sets advance or position_ids, and the array declares no positions")
        self._check_storage(declared, segments, ctx)

    @abstractmethod
    def _check_storage(self, declared: ArrayStep, segments, ctx) -> None:
        """Refuse what ``declared`` asks of this array's storage that it can't do."""
        if declared.placeholders or declared.eager_add:
            raise self._refusal("a step sets placeholders or eager_add, and the array has no paged entries to add to")

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
        # not `_check_open`, whose message would be formatted on every update of every layer
        if not self._open:
            raise self._refusal(f"the update of {field} in a step that doesn't list the array")
        if (field, layer) in self._updated:
            raise self._refusal(f"a second update of {field} on layer {layer} in one step")
        self._updated.add((field, layer))

    def _take_hold(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        held = self._held
        if held is None:
            raise self._refusal(f"a read on layer {layer} with no add before it")
        self._held = None
        return held

    def add(self, layer: int | torch.Tensor, *, keep: bool | None = None, **fields: torch.Tensor) -> None:
        """Append each row's span at its next log positions. A kept add happens once per layer per step, as the
        layer's last add. ``keep`` False is scratch: it can repeat, and lasts until the next add or the end of the step.
        None takes the step's ``keep``. The resources keep or drop a step's adds as a whole, so a step that keeps needs
        each layer to end on a kept add, and a step with ``keep`` False refuses ``keep=True``."""
        raise self._unsupported("add")

    def read(self, layer: int | torch.Tensor, q: torch.Tensor | None = None, **mask_args: Any) -> torch.Tensor:
        """Attend ``q`` through the read declared for ``layer``: kept entries from earlier steps, plus this layer's
        latest add, at the positions its mask reaches."""
        raise self._unsupported("read")

    def write(self, layer: int | torch.Tensor, *, at: torch.Tensor | None = None, **fields: torch.Tensor) -> None:
        """Overwrite entries in place, on a rewritable array. ``at`` gives each token's log position, and is omitted
        for a Value."""
        if not self.array.rewritable:
            raise self._refusal("write on an append-only array")
        raise self._unsupported("write")

    def set_start(self, start: Mapping[str, torch.Tensor]) -> None:
        """What a row of this Value reads while it has nothing written, has span 0 or pads a step: ``start``'s tensor
        for each field it names, of one entry's shape on this rank, and zeros for the rest. The tensors are made at load
        and never written, since a captured read keeps their address."""
        raise self._unsupported("set_start")

    def block(self, field: str, layer: int) -> torch.Tensor:
        """``layer``'s view of the entry field ``field``, for an update's op."""
        raise self._unsupported("block")

    def at_last_token(self, x: torch.Tensor) -> torch.Tensor:
        raise self._unsupported("at_last_token")

    def _kernel_setting(self, read: str, field: str) -> Any:
        raise self._unsupported(f"kernel_setting of {read!r}")

    def _update(self, kind: type[Op], field: str, layer: int) -> tuple[Any, torch.Tensor]:
        """Record the update of type ``kind`` on ``layer``'s ``field``, and return the resource that runs it and the
        field's view."""
        raise self._unsupported(f"an update of type {kind.__name__}")

    def _host(self, request: int, label: str, layer: int) -> Mapping[str, object]:
        raise self._unsupported("host")


class SamplerHandle(StepHandle):
    """A node's calls on one declared sampler. Its rows are the step's, padding included."""

    def __init__(self, sampler: Sampler, resource: Any):
        self.sampler = sampler
        self.name = sampler.name
        self.resource = resource
        self.rows = ()
        self._open = False
        self._drawn = 0
        self._fused = False

    def _refusal(self, cause: str) -> ValueError:
        return ValueError(f"Sampler {self.name!r}: {cause}")

    def begin_step(self, step) -> None:
        if self._open:
            raise self._refusal("a step began while another is open")
        self._open = True
        self.rows = step.ctx.padded_request_ids
        self._drawn = 0
        self._fused = False

    def check_step(self, step) -> None:
        # a step that keeps no row's sample may skip the kernel: an eager prefill short of its last chunk
        if step.ctx.replays or (not self._drawn and step.state[self.name].kept_rows == frozenset()):
            return
        if self._drawn != self.sampler.draws:
            counted = (f"counted {self._drawn} draws by sample and drew" if self._fused
                       else f"sampled {self._drawn} times")
            raise self._refusal(f"a step listing it {counted}, and draws is {self.sampler.draws}")

    def commit_step(self, step) -> None:
        """Nothing: the resource holds a sampler's state."""

    def end_step(self, step) -> None:
        self._open = False
        self._fused = False


@torch.compiler.disable
def causal_conv(
    h: ArrayHandle, layer: int | torch.Tensor, x: torch.Tensor, *, weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """The declared `CausalConv` over ``x``, which reads and writes ``layer``'s ``conv`` field in place."""
    run, conv = h._update(CausalConv, "conv", h._layer_id(layer))
    return run.run_conv(x, conv_layer=conv, weight=weight, bias=bias, label=h.label)


@torch.compiler.disable
def gated_delta_net(
    h: ArrayHandle, layer: int | torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor,
    b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
) -> torch.Tensor:
    """The declared `GatedDeltaNet`, which reads and writes ``layer``'s ``state`` field in place."""
    run, state = h._update(GatedDeltaNet, "state", h._layer_id(layer))
    return run.run(q, k, v, a, b, state_layer=state, a_log=a_log, dt_bias=dt_bias, label=h.label)


def host(h: ArrayHandle, request: int, label: str, layer: int | torch.Tensor) -> Mapping[str, object]:
    """One entry's host fields on the host, as the last committed step left them."""
    return h._host(request, label, h._layer_id(layer))


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


@torch.compiler.disable
def fused_draw_args(s: SamplerHandle) -> FusedDrawArgs | None:
    """Not frozen yet, like the rest of the fused draw. Each row's sampling parameters, padding rows included, for a
    kernel in the forward that draws as `sample` would. None when `sample` would draw any other way this step, or the
    declared vocabulary is past what `draw_row` reproduces, and the model then calls `sample`."""
    if not s._open:
        raise s._refusal("fused_draw_args in a step that doesn't list the sampler")
    args = None if s.sampler.vocab_size > FUSED_DRAW_MAX_VOCAB else s.resource.fused_draw_args()
    s._fused = args is not None
    return args


@torch.compiler.disable
def drew(s: SamplerHandle, n: int) -> None:
    """Not frozen yet, like the rest of the fused draw. Count ``n`` draws a kernel made from `fused_draw_args`, and
    advance each row's sampling state as ``n`` `sample` calls would. Called after the launch, so the advance is ordered
    after the kernel's reads."""
    if not s._fused:
        raise s._refusal("drew, and no fused_draw_args this step returned arguments")
    if not (type(n) is int and n >= 1 and s._drawn + n <= s.sampler.draws):
        raise s._refusal(f"drew({n!r}) after {s._drawn} draws, and draws is {s.sampler.draws}")
    s.resource.advance_offsets(n)
    s._drawn += n


# Eager as a whole, as `AttentionManager.select_last_hidden` explains.
@torch.compiler.disable
def at_last_token(h: ArrayHandle, x: torch.Tensor) -> torch.Tensor:
    """``x`` at each row's last token in this step, in row order."""
    return h.at_last_token(x)


def last_sampled(s: SamplerHandle) -> torch.Tensor:
    """Each row's last sampled token, on the device, for a sampler the node names in its `DeviceLoopback`."""
    if not s._open:
        raise s._refusal("last_sampled in a step that doesn't list the sampler")
    return s.resource.loopback_tokens(s.rows)


def kernel_setting(h: ArrayHandle, read: str, field: str) -> Any:
    """The value of a `kernel_settings` field in effect, YAML included, until the engine picks kernels itself."""
    return h._kernel_setting(read, field)
