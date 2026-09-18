"""The real resources of a generated model, and the steps that drive them.

Tier 0 drives the resource lifecycle against stubs. It covers the order and
the scope of the calls, and no behavior of a resource. This module drives the
real resources. It uses no stub:

* ``KVManager``, built through ``KVSpec.resource_class.build``. This is the
  call that ``EngineManager.build`` makes.
* ``PositionManager``, when the model declares it. It names the KV cache in
  ``depends_on``. The runner therefore resolves the dependency, sorts the two
  resources, and gives the position resource the plan output of the cache.
* ``StepRunner``, with the per-node resource map. Each sweep is then scoped
  the way the worker scopes it.
* ``LocalTransferEngine``, the engine of a deployment on one node.

The cache is on the CPU, and its dtype is ``int32``. Every value that it holds
is therefore exact. No code here runs an attention kernel. A step reads and
writes the pages that the plan gives it, and the checksum is the payload.

The step of one node
--------------------

``run_step`` follows the cycle of ``Engine.step``, for one node and a batch of
requests:

1. ``admit`` reserves the pages for the spans of this step
2. ``plan`` gives one ``SequenceView`` for each (request, label) stream. A
   view holds the pages, the resident length, and the tokens this step adds.
3. the step reads the tokens that its own stream already holds. The checksum
   of the step covers them.
4. the step writes one value for each new token, into every layer
5. ``commit`` records the spans. ``publish`` gives what another device needs
   to retrieve the stream.

Step 3 is the reason for this module. A step that covers its identity alone
reads no page. A resource can then do three things that no oracle sees:

* give one page to two requests
* address the wrong page
* keep a page across a free

With step 3, each of those changes a number.

The value of a token depends on the layer, and on the position of the token
inside its step. The V half of a slot is the K half plus one. A write into
the wrong layer, at the wrong offset, or into the wrong half, therefore also
changes a number.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field

import torch

from fuzzer.tier1.executor import NodeInputs
from fuzzer.tier1.values import checksum, prefix_checksum, token_value
from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.resources import (
    BucketKey,
    KVConfig,
    KVSpec,
    KVStep,
    PositionConfig,
    PositionSpec,
    PositionStep,
    Segment,
    SlotLease,
    StepContext,
    StepRunner,
    SubmoduleStep,
    resolve_spec_dependencies,
)
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.transfer import TransferEngineInfo

__all__ = ["CACHE_NAME", "KV_KEY", "POS_KEY", "ResourceRig", "StepOutcome"]

KV_KEY = "kv"
POS_KEY = "pos"
WALK = "walk"
# The output name under which a step's cache value is computed. It is not a
# name of the graph, so it can never collide with one.
CACHE_NAME = "<kv>"


def _where(exc: BaseException) -> str:
    """The error, and the deepest frame of mstar that raised it."""
    frames = [
        frame for frame in traceback.extract_tb(exc.__traceback__)
        if "/fuzzer/" not in frame.filename
    ]
    site = "" if not frames else f"{frames[-1].filename.split('/')[-1]}:{frames[-1].lineno}"
    return f"{type(exc).__name__}({exc}) at {site}"


@dataclass
class StepOutcome:
    """What one step of one node gave back."""

    # request id -> output name -> value. Empty when the admit failed.
    values: dict[str, dict[str, str]] = field(default_factory=dict)
    admitted: bool = True
    # The pages the whole step needed, for the oracle on a refused admit.
    pages_short: int = 0
    # What ``publish`` reported for each (request, label), against what the
    # manager holds. The machine holds the invariant.
    publish_disagrees: list[str] = field(default_factory=list)
    # What ``plan`` or ``commit`` raised after ``admit`` accepted the step.
    # ``admit`` is the gate, so no call after it may raise.
    raised: str | None = None
    # The position ids the step planned, against the ids the length of each
    # stream implies. The machine holds the invariant.
    position_disagrees: list[str] = field(default_factory=list)
    # Whether this step promoted a plan that was staged a step ahead.
    promoted: bool = False
    # What the step read back out of the slots it had just written, against
    # what it wrote. The machine holds the invariant.
    readback_ok: bool = True


class ResourceRig:
    """The resources of one generated model, and the runner over them."""

    def __init__(self, spec: dict, node_names: list[str]) -> None:
        self.kv_spec = spec["kv"]
        self.labels: dict[str, str] = self.kv_spec["labels"]
        self.spans: dict[str, int] = self.kv_spec["spans"]
        self.num_layers = self.kv_spec["num_layers"]
        # node -> the fork its step carries, as (target label, when).
        self.forks: dict[str, tuple[str, str]] = {
            fork["node"]: (fork["to"], fork["when"])
            for fork in self.kv_spec.get("forks", [])
        }
        # The shape of a captured bucket, or None for an eager model.
        self.capture: dict | None = self.kv_spec.get("capture")
        self.preplan_enabled = bool(self.kv_spec.get("preplan"))
        self.slot = 0
        # The step that is staged a step ahead, as (node, request ids).
        self.staged: tuple[str, tuple[str, ...]] | None = None
        self.staged_lease: SlotLease | None = None
        self.staged_rows: tuple[str, ...] = ()
        self.before_stage: dict | None = None
        self.staged_raised: str | None = None
        self.dummy_rids: list[str] = []
        # How many padding rows addressed a page that a live stream holds.
        self.padding_hits: list[str] = []

        config = KVConfig(
            num_layers=self.num_layers,
            num_kv_heads=self.kv_spec["num_kv_heads"],
            head_dim=self.kv_spec["head_dim"],
            max_seq_len=self.kv_spec["max_seq_len"],
            max_num_pages=self.kv_spec["max_num_pages"],
            page_size=self.kv_spec["page_size"],
        )
        nodes = set(node_names)
        specs = [KVSpec(resource_key=KV_KEY, nodes=nodes, config=config)]
        self.with_positions = bool(self.kv_spec.get("positions"))
        if self.with_positions:
            specs.append(PositionSpec(
                resource_key=POS_KEY, nodes=nodes,
                config=PositionConfig(kv_cache=KV_KEY),
            ))
        # The same calls that the engine manager makes. They include the
        # dependency resolution. The position resource names the cache, and
        # the runner sorts the two so that the cache plans first.
        by_key = resolve_spec_dependencies(specs)
        resources = {
            spec.resource_key: spec.resource_class.build(
                spec, self._engine_info(spec, by_key),
            )
            for spec in specs
        }
        self.kv = resources[KV_KEY]
        self.runner = StepRunner(
            resources,
            node_resources={name: list(resources) for name in node_names},
        )
        self.config = config
        self.live: set[str] = set()
        if self.capture is not None:
            self._open_capture()

    def _open_capture(self) -> None:
        """Make the state that a captured replay addresses.

        The engine does three things before a bucket replays:

        * it asks every resource for its static buffers
        * it opens one row of padding state for each slot of the bucket
        * it runs the bucket one time at capture, with those rows full

        The third one gives a padding row its labels. A replay later declares
        a zero-span segment for that row, and ``KVManager.plan`` then reads
        the stream that capture left.
        """
        bucket_size = self.capture["bs"]
        self.runner.build_cuda_graph_buffers(
            [], bucket_size, self.capture["num_tokens"],
        )
        self.dummy_rids = [f"__cg_fuzz_{index}__" for index in range(bucket_size)]
        for rid in self.dummy_rids:
            self.runner.ingest_request(rid)
        # Capture runs the bucket full, with real spans, so every label of
        # every padding row exists from here on.
        for label in sorted(set(self.labels.values())):
            segments = tuple(Segment(rid, label, 1) for rid in self.dummy_rids)
            step = SubmoduleStep(steps={KV_KEY: KVStep(segments=segments)})
            step.set_ctx(StepContext(
                request_ids=tuple(self.dummy_rids), graph_walk=WALK,
                slot=0, capture=True,
            ))
            self.runner.admit(step)
            self.runner.plan(step)
            self.runner.commit(step)
        # Capture is over: the rows keep their labels and give back their
        # pages, as ``DummyRequestPool.release_all`` leaves them.
        for rid in self.dummy_rids:
            for resource in self.runner.resources.values():
                resource.reset_request(rid, free=True)

    @staticmethod
    def _engine_info(spec, by_key: dict) -> EngineResourceInfo:
        """What the engine gives a resource at build time.

        The transfer engine is the one of a deployment on one node. On a CPU
        cache it becomes ``LocalOnlyKVTransferEngine``. Thus ``publish`` stays
        real.
        """
        return EngineResourceInfo(
            device=torch.device("cpu"),
            kv_dtype=torch.int32,
            transfer_engine_info=TransferEngineInfo(
                my_entity_id="fuzzer",
                my_session_id="fuzzer",
                transfer_engine=LocalTransferEngine("fuzzer"),
            ),
            dependencies={key: by_key[key] for key in spec.depends_on()},
        )

    # -- the lifecycle of a request -----------------------------------------

    def ingest(self, request_id: str) -> None:
        if request_id in self.live:
            return
        self.runner.ingest_request(request_id)
        self.live.add(request_id)

    def remove(self, request_id: str) -> None:
        if request_id not in self.live:
            return
        self.runner.remove_request(request_id)
        self.live.discard(request_id)

    # -- one step ------------------------------------------------------------

    def _lease(self) -> SlotLease | None:
        """The slot a step replays on, or None for an eager step.

        A replay uses the two slots one after the other. The plan of the next
        step then writes the buffers that the replay in flight does not read.
        Pre-planning needs that, so the engine pre-plans under a lease only.
        """
        if self.capture is None:
            return None
        if self.capture["slots"] > 1:
            self.slot = 1 - self.slot
        return SlotLease(slot=self.slot, bucket=BucketKey(
            graph_walk=WALK, bs=self.capture["bs"],
            num_tokens=self.capture["num_tokens"],
        ))

    def _build_step(
        self, node_name: str, request_ids: list[str],
        lease: SlotLease | None, is_preplan: bool,
    ) -> SubmoduleStep:
        """The step of one node for a batch, with the rows it addresses."""
        label = self.labels[node_name]
        span = self.spans[node_name]
        padded = list(request_ids)
        if lease is not None:
            padded = request_ids + self.dummy_rids[len(request_ids):lease.bucket.bs]

        context = StepContext(
            request_ids=tuple(request_ids), graph_walk=WALK,
            slot=0 if lease is None else lease.slot,
            capture=False,
        )
        context.slot_lease = lease
        context.is_preplan = is_preplan
        if lease is not None:
            context.set_padded_rids(tuple(padded))

        segments = tuple(
            Segment(rid, label, span) for rid in request_ids
        ) + tuple(
            # A padding row declares its segment like any other row, and its
            # span is zero so it allocates nothing. ``SubmoduleStep`` says so.
            Segment(rid, label, 0) for rid in padded[len(request_ids):]
        )
        pre_forks, post_forks = self._forks_of(node_name, label)
        steps = {KV_KEY: KVStep(
            segments=segments, pre_forks=pre_forks, post_forks=post_forks,
        )}
        if self.with_positions:
            steps[POS_KEY] = PositionStep(segments=segments)
        step = SubmoduleStep(steps=steps)
        step.set_ctx(context)
        return step

    # -- pre-planning --------------------------------------------------------

    def stage_step(self, node_name: str, request_ids: list[str]) -> str | None:
        """Admit and plan a step ahead of the step that still runs.

        The engine pre-plans under a lease only. A staged step must write the
        plan buffers of a slot that the replay in flight does not read.

        Gives back the reason it did not stage, or None when it staged.
        """
        if self.capture is None or not self.preplan_enabled:
            return "the model does not pre-plan"
        if self.staged is not None:
            return "a step is already staged"
        lease = self._lease()
        step = self._build_step(node_name, request_ids, lease, is_preplan=True)
        # The rows this step addresses: the batch, and the padding rows of the
        # slot it leased.
        self.staged_rows = tuple(step.ctx.padded_request_ids)
        self.before_stage = self.snapshot(self.staged_rows)
        self.staged_raised = None
        try:
            outcome = self.runner.pre_admit(step)
            if not outcome.ok:
                # ``Engine._pre_plan_for_batch`` drops the stage and lets the
                # inline path re-drive the step.
                self.clear_preplan()
                return "pre_admit refused the step"
            self.runner.pre_plan(step)
        except Exception as exc:  # noqa: BLE001 - the contract is the finding
            self.staged_raised = _where(exc)
            self.clear_preplan()
            return "pre_plan raised"
        self.staged = (node_name, tuple(request_ids))
        self.staged_lease = lease
        return None

    def clear_preplan(self) -> None:
        """Drop a staged step, as ``Engine.reset_pre_plan_for_batch`` does."""
        for resource in self.runner.resources.values():
            resource.clear_preplan()
        self.staged = None
        self.staged_lease = None

    def snapshot(self, rows: tuple[str, ...] | None = None) -> dict:
        """The state that an abandoned pre-plan must give back.

        ``rows`` keeps only the request ids that one step addresses. Another
        request can change between the stage and the drop, and that change is
        not state that the stage left.
        """
        keep = None if rows is None else set(rows)
        streams = {
            (rid, label): (
                stream.stored_len, stream.generation, stream.step_in_flight,
                len(stream.page_indices),
            )
            for rid, per_label in self.kv._streams.items()
            if keep is None or rid in keep
            for label, stream in per_label.items()
        }
        positions: dict[tuple[str, str], int] = {}
        if self.with_positions:
            counters = self.runner.resources[POS_KEY]._counters
            positions = {
                (rid, label): value
                for rid, per_label in counters.items()
                if keep is None or rid in keep
                for label, value in per_label.items()
            }
        return {"streams": streams, "positions": positions}

    def run_step(
        self, node_name: str, batch: list[NodeInputs], pass_indices: dict[str, int],
    ) -> StepOutcome:
        """Drive one node's step for a batch of requests."""
        label = self.labels[node_name]
        span = self.spans[node_name]
        request_ids = [item.request_id for item in batch]

        # A staged step that does not describe this batch is stale. The
        # engine drops such a step before it drives the real one.
        promoted = self.staged == (node_name, tuple(request_ids))
        if self.staged is not None and not promoted:
            self.clear_preplan()
        lease = self.staged_lease if promoted else self._lease()
        step = self._build_step(node_name, request_ids, lease, is_preplan=False)
        self.staged = None
        self.staged_lease = None

        outcome = self.runner.admit(step)
        if not outcome.ok:
            reason = outcome.reason
            return StepOutcome(
                admitted=False,
                pages_short=getattr(reason, "pages_short", 0),
            )

        try:
            results = self.runner.plan(step)
        except Exception as exc:  # noqa: BLE001 - the contract is the finding
            return StepOutcome(admitted=True, raised=_where(exc))
        plans = results[KV_KEY]
        views = plans[label].views
        by_request = {view.request_id: view for view in views}

        # (3) read the prefix that each stream already holds, and make it part
        # of the checksum of the step.
        values: dict[str, dict[str, str]] = {}
        # The value that the step writes into the cache. It is separate from
        # the values of the output names, so which name the graph happens to
        # declare first decides nothing.
        written: dict[str, str] = {}
        for item in batch:
            view = by_request[item.request_id]
            resident = view.length - view.to_compute
            digest = prefix_checksum(self._read_stream(view, resident))
            pass_index = pass_indices[item.request_id]
            written[item.request_id] = checksum(
                item.request_id, pass_index, item.node_name, CACHE_NAME,
                item.run_index, item.inputs, extra=(digest,),
            )
            values[item.request_id] = {
                name: checksum(
                    item.request_id, pass_index, item.node_name, name,
                    item.run_index, item.inputs, extra=(digest,),
                )
                for name in item.output_names
            }

        # (4) write one value for each new token, in the packed order of the
        # plan, into every layer.
        readback_ok = self._write(views, written, span, label)
        if lease is not None:
            self._replay_padding(label, lease)

        # (5) record the spans, then publish, as the worker does after a step
        try:
            self.runner.commit(step)
        except Exception as exc:  # noqa: BLE001
            return StepOutcome(admitted=True, raised=_where(exc))
        return StepOutcome(
            values=values,
            readback_ok=readback_ok,
            publish_disagrees=self._publish(request_ids, node_name),
            position_disagrees=self._check_positions(results, views, label),
            promoted=promoted,
        )

    def _replay_padding(self, label: str, lease: SlotLease) -> None:
        """Write the rows of the bucket that hold no token.

        A captured replay writes every row of its bucket. It does not write
        only the rows that carry a request. ``KVPlanState.copy_`` points the
        other rows at ``SINK_PAGE``. Without that, those rows land in the
        cache of another request.

        The engine writes them inside the captured graph, which never calls
        ``write_kv``. This method therefore writes the cache directly, through
        the addressing that the plan made.
        """
        state = self.kv._current_plan_states[label]
        capture_len = lease.bucket.num_tokens
        real = state.total_tokens
        if real >= capture_len:
            return
        pages = state.token_to_page[real:capture_len]
        slots = state.token_to_cache[real:capture_len]

        live = {
            page for pages_of in self.held_pages().values() for page in pages_of
        }
        hit = sorted({int(page) for page in pages} & live)
        if hit:
            self.padding_hits.append(
                f"the padding of a step of label {label!r} addresses pages "
                f"{hit}, which live streams hold"
            )
        poison = torch.full(
            (capture_len - real, self.config.num_kv_heads, self.config.head_dim),
            -1, dtype=torch.int32,
        )
        for layer in range(self.num_layers):
            self.kv.kv_cache.write_tokens(
                layer_idx=layer, k=poison, v=poison,
                page_idx=pages, cache_idx=slots,
            )

    def _forks_of(
        self, node_name: str, label: str,
    ) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
        """The forks the step of one node carries.

        A pre-fork copies in ``plan``, before this step's spans land. A
        post-fork copies in ``commit``, after them.
        """
        fork = self.forks.get(node_name)
        if fork is None:
            return (), ()
        target, when = fork
        pair = ((label, target),)
        return (pair, ()) if when == "pre" else ((), pair)

    def _check_positions(
        self, results: dict, views: list, label: str,
    ) -> list[str]:
        """Compare the ids of a step against the length of each stream.

        A step writes the tokens ``resident`` to ``resident + span`` of its
        stream. Those numbers are its positions. The ids come back in the
        packed order of the KV plan. That order is what joins the two
        resources.
        """
        if not self.with_positions:
            return []
        ids = results[POS_KEY][label].tolist()
        # Under a lease the buffer has the size of the bucket. The tail after
        # the tokens of this step keeps what the step before it left.
        total = sum(view.to_compute for view in views)
        ids = ids[:total]
        disagrees: list[str] = []
        at = 0
        for view in views:
            resident = view.length - view.to_compute
            wanted = list(range(resident, resident + view.to_compute))
            got = ids[at:at + view.to_compute]
            at += view.to_compute
            if got != wanted:
                disagrees.append(
                    f"{view.request_id}/{label}: positions {got}, the stream "
                    f"of {resident} tokens implies {wanted}"
                )
        if at != len(ids):
            disagrees.append(
                f"the step planned {len(ids)} position ids for {at} tokens"
            )
        return disagrees

    def _publish(self, request_ids: list[str], node_name: str) -> list[str]:
        """Publish each request of the step, and check what it reported.

        A second device reads ``publish`` to retrieve a stream. A length or a
        page list that does not match the stream therefore makes that device
        copy the wrong bytes.
        """
        published = self.runner.publish(request_ids, node_name)
        lengths = self.stream_lengths()
        held = self.held_pages()
        disagrees: list[str] = []
        for request_id, per_key in published.items():
            info = per_key.get(KV_KEY)
            if info is None:
                continue
            for label, sequence in info.get(rank=0).items():
                stored = lengths.get((request_id, label))
                if sequence.seq_len != stored:
                    disagrees.append(
                        f"{request_id}/{label}: published {sequence.seq_len} "
                        f"tokens, the stream holds {stored}"
                    )
                pages = held.get((request_id, label), [])
                if list(sequence.page_indices) != pages:
                    disagrees.append(
                        f"{request_id}/{label}: published pages "
                        f"{list(sequence.page_indices)}, the stream holds {pages}"
                    )
        return disagrees

    def _write(
        self, views: list, written: dict[str, str], span: int, label: str,
    ) -> bool:
        """Write the value of each step into its new slots, then read it back.

        The engine packs the tokens of a step in the order of the views, so
        the tensor follows that order. A value that comes back different is a
        write and a read that do not agree on an address.
        """
        if span == 0:
            return True
        ok = True
        heads = self.config.num_kv_heads
        head_dim = self.config.head_dim
        for layer in range(self.num_layers):
            wanted: list[int] = []
            for view in views:
                # A padding row carries no token, so it adds nothing to the
                # packed order and has no value of its own.
                if view.to_compute == 0:
                    continue
                # One value for the whole step, then one number for each token
                # of it, so the offset inside the step is part of the number.
                value = written[view.request_id]
                wanted.extend(
                    token_value(value, index, layer) for index in range(view.to_compute)
                )
            keys = torch.tensor(wanted, dtype=torch.int32).view(-1, 1, 1)
            keys = keys.expand(len(wanted), heads, head_dim).contiguous()
            self.kv.write_kv(keys, keys + 1, layer_idx=layer, label=label)
            # The slots this step planned, read back through the same
            # addressing: [num_tokens, 2, heads, head_dim], K at 0 and V at 1.
            got = self.kv.read_kv(layer_idx=layer, plan_label=label)
            ok = ok and bool(
                torch.equal(got[:, 0], keys) and torch.equal(got[:, 1], keys + 1)
            )
        return ok

    def _read_stream(self, view, resident: int) -> list[int]:
        """The numbers that one stream already holds, every layer, in order.

        The addressing is the addressing of the cache: token ``i`` of a stream
        lives at offset ``start + i`` of the pages that the plan named.
        """
        if resident <= 0:
            return []
        page_size = self.config.page_size
        offsets = [view.start + index for index in range(resident)]
        pages = torch.tensor(
            [view.page_idxs[offset // page_size] for offset in offsets],
            dtype=torch.long,
        )
        slots = torch.tensor(
            [offset % page_size for offset in offsets], dtype=torch.long,
        )
        out: list[int] = []
        for layer in range(self.num_layers):
            got = self.kv.kv_cache.read_tokens(
                layer_idx=layer, page_idx=pages, cache_idx=slots,
            )
            # K of the first head and the first dimension carries the number;
            # the rest of the slot repeats it.
            out.extend(int(value) for value in got[:, 0, 0, 0])
        return out

    # -- what the oracles read -----------------------------------------------

    def free_pages(self) -> int:
        return self.kv._arena.num_free

    def held_pages(self) -> dict[tuple[str, str], list[int]]:
        """Every page that a live stream holds, by (request, label)."""
        return {
            (request_id, label): list(stream.page_indices)
            for request_id, streams in self.kv._streams.items()
            for label, stream in streams.items()
        }

    def stream_lengths(self) -> dict[tuple[str, str], int]:
        return {
            (request_id, label): stream.stored_len
            for request_id, streams in self.kv._streams.items()
            for label, stream in streams.items()
        }
