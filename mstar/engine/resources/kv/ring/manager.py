"""The paged ``KVManager`` reserves pages per step and hands them back;
this one reserves nothing. Every layer's ring is allocated once at
``build`` and reused for the life of the process,.
"""

import torch
from torch import Tensor

from mstar.engine.resources.base import (
    AttentionResource,
    CGSlotSpec,
    EngineResourceInfo,
)
from mstar.engine.resources.kv.config import KVSpec, RingKVConfig, RingKVStep
from mstar.engine.resources.kv.ring.cache import LayerRingCache
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.step import (
    ADMIT_OK,
    AdmitOutcome,
    AdmitRuntimeError,
    ResourceStep,
    StepContext,
)

__all__ = ["RingKVManager"]


class RingKVManager(AttentionResource):
    """Per-layer ring caches holding ``num_worlds`` worlds, one per request."""

    def __init__(
        self,
        config: RingKVConfig,
        name: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.name = name
        self.device = torch.device(device)
        self.dtype = dtype

        self.layers = [
            LayerRingCache(
                num_worlds=config.num_worlds,
                n_kv_heads=config.num_kv_heads,
                ring_frames=layer.ring_frames,
                ring_buckets=layer.ring_buckets,
                d_head=config.head_dim,
                tokens_per_frame=config.tokens_per_frame,
                pinned_dilation=layer.pinned_dilation,
                dtype=dtype,
                device=self.device,
            )
            for layer in config.layers
        ]

        self._worlds: dict[str, int] = {}
        self._free_worlds: set[int] = set(range(config.num_worlds))
        self._known_rids: set[str] = set()
        self._last_frames: dict[str, int] = {}
        self._static_world_idx = torch.zeros(1, dtype=torch.int64, device=self.device)

    @classmethod
    def build(cls, spec: KVSpec, info: EngineResourceInfo) -> "RingKVManager":
        config = spec.config
        if not isinstance(config, RingKVConfig):
            raise TypeError(
                f"{cls.__name__} needs a RingKVConfig; got {type(config).__name__}. "
                "KVSpec dispatches on the config, so this means the spec was "
                "built by hand with the wrong one."
            )
        if info.joint_comm_group is not None:
            config.shard(info.joint_comm_group.world_size)
        return cls(
            config=config,
            name=spec.resource_key,
            device=info.device,
            dtype=info.kv_dtype,
        )

    # ---- Model-facing API -------------------------------------------------

    @property
    def tokens_per_frame(self) -> int:
        return self.config.tokens_per_frame

    @property
    def num_worlds(self) -> int:
        """How many requests can hold a world here at once."""
        return self.config.num_worlds

    def capacity(self, layer_idx: int) -> int:
        """Token slots ONE world owns in ``layer_idx``'s ring, scratch frame
        included. ``total_slots`` is the whole buffer."""
        return self.layers[layer_idx].capacity

    def total_slots(self, layer_idx: int) -> int:
        """Token slots in ``layer_idx``'s buffer across every world -- the
        length of the KV view ``upsert`` returns and of its visibility row."""
        return self.layers[layer_idx].total_slots

    def world_of(self, rid: str) -> int | None:
        """``rid``'s world index, or None if it holds none. Host-side
        introspection; the forward reads the staged tensor, never this."""
        return self._worlds.get(rid)

    def upsert(
        self, k: Tensor, v: Tensor, layer_idx: int, frame_pos: Tensor, *, commit: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Write one frame's K/V for ``layer_idx`` and return what to attend to.

        ``k``/``v`` are ``[1, H_kv, tokens_per_frame, D]``. ``k`` is already
        RoPE'd and RMS-normed and ``v`` is post value-residual lerp: the cache
        stores post-RoPE keys, so replayed history is never re-rotated.
        Returns ``(k_all, v_all, visible)``, the first two spanning
        the whole buffer, every resident world and the third a
        ``[total_slots]`` bool row that is False everywhere outside the calling
        request's own world.

        ``frame_pos`` and ``commit`` are both arguments rather than resource
        state, for the same reason. ``frame_pos`` is the ``[]`` int64 ring clock
        -- not a slot id -- and alone determines the slot written and the
        visibility row;
        """
        # `layer_idx` is Python-level (it indexes a list of differently-shaped
        # rings), so indexing on it is graph-safe.
        kv = torch.stack([k, v], dim=0)
        return self.layers[layer_idx].upsert(
            kv, frame_pos, commit, self._static_world_idx
        )

    def _reset_world(self, rid: str) -> None:
        """Zero ``rid``'s world and drop its clock, leaving its claim in place."""
        world_idx = self._worlds.get(rid)
        if world_idx is None:
            return
        for layer in self.layers:
            layer.reset(world_idx)
        self._last_frames.pop(rid, None)

    def _release_world(self, rid: str) -> None:
        """Hand ``rid``'s world back to the pool. Zero it first."""
        self._reset_world(rid)
        world_idx = self._worlds.pop(rid, None)
        if world_idx is not None:
            self._free_worlds.add(world_idx)

    @torch.no_grad()
    def get_state(self, rid: str) -> dict:
        """Snapshot one request's world. Cloned, so the caller can hold it 
        across further rollout steps that mutate the rings in place.
        """
        world_idx = self._require_world(rid, "get_state")
        layers = []
        for layer in self.layers:
            lo, hi = layer.world_span(world_idx)
            layers.append((
                layer.kv[:, :, :, lo:hi].detach().clone(),
                layer.written[lo:hi].detach().clone(),
            ))
        return {"layers": layers}

    @torch.no_grad()
    def load_state(self, rid: str, state: dict) -> None:
        """Restore one world's contents into the existing allocation."""

        world_idx = self._require_world(rid, "load_state")
        layers = state["layers"]
        if len(layers) != len(self.layers):
            raise ValueError(
                f"state has {len(layers)} layers, ring has {len(self.layers)}."
            )
        for i, (layer, (kv, written)) in enumerate(zip(self.layers, layers, strict=True)):
            lo, hi = layer.world_span(world_idx)
            span = layer.kv[:, :, :, lo:hi]
            if tuple(kv.shape) != tuple(span.shape):
                raise ValueError(
                    f"layer {i} state shape {tuple(kv.shape)} != one world's ring "
                    f"shape {tuple(span.shape)}."
                )
            span.copy_(kv)
            layer.written[lo:hi].copy_(written)
        self._last_frames.pop(rid, None)

    def _require_world(self, rid: str, what: str) -> int:
        world_idx = self._worlds.get(rid)
        if world_idx is None:
            raise KeyError(
                f"ring KV {self.name!r} has no world for request {rid!r}; "
                f"{what} is per request and a request that never admitted owns "
                "no span to read or write."
            )
        return world_idx

    # ---- Introspection ----------------------------------------------------

    def memory_bytes(self) -> int:
        """Total resident ring bytes across all layers and all worlds."""
        return sum(layer.memory_bytes for layer in self.layers)

    # ---- Resource lifecycle -----------------------------------------------

    def _step_frames(self, step: ResourceStep) -> dict[str, int]:
        """This step's declared ring clock per request id."""

        if not isinstance(step, RingKVStep):
            raise TypeError(
                f"ring KV {self.name!r} was declared a {type(step).__name__}; it "
                "needs a RingKVStep, the step that carries `frames`. A KVStep "
                "admits and commits without ever checking the ring clock."
            )
        return dict(step.frames)

    def ingest_request(self, rid: str, overrides: ResourceReqConfig | None = None) -> None:
        """Register ``rid``. Deliberately does not claim a world."""

        del overrides  # no per-request tunables: the ring geometry is fixed
        self._known_rids.add(rid)

    def admit(self, step: ResourceStep, ctx: StepContext) -> AdmitOutcome:
        """Claim a world for this step's request, or refuse it terminally.
        Does not support eviction for now
        """
        # `request_ids`, not `padded_request_ids` as the paged manager uses: a
        # padding row is a dummy rid a replay pads a bucket out to, and it must
        # not be able to take a world from the real request in the same batch.
        rids = list(ctx.request_ids)

        if len({*rids}) > 1:
            return AdmitOutcome(
                ok=False, ready=False,
                reason=AdmitRuntimeError(
                    f"ring KV {self.name!r} was handed a batch of {len(set(rids))} "
                    f"requests ({sorted(set(rids))}); one step advances one world, "
                    "so max_batch_size must be 1. This is a step-batch limit, not "
                    "a ring limit -- the ring holds "
                    f"{self.num_worlds} worlds and they take turns across steps."
                ),
            )

        frames = self._step_frames(step)

        # Check every rid before claiming any world: a refused admit must not
        # leave a world half-claimed by the first rid of a batch it rejected.
        wanted = {rid for rid in rids if rid not in self._worlds}
        for rid in sorted(wanted):
            if rid not in self._known_rids:
                return AdmitOutcome(
                    ok=False, ready=False,
                    reason=AdmitRuntimeError(
                        f"ring KV {self.name!r} was asked to admit request {rid!r}, "
                        "which was never ingested. Worlds are handed back by "
                        "`remove_request`, which only ever runs for a request the "
                        "engine opened -- so a world claimed here would never "
                        "return to the pool and the node would lose capacity with "
                        "nothing raised."
                    ),
                )
        if len(wanted) > len(self._free_worlds):
            return AdmitOutcome(
                ok=False, ready=False,
                reason=AdmitRuntimeError(
                    f"ring KV {self.name!r} holds all {self.num_worlds} of its "
                    f"worlds ({sorted(self._worlds)}); request(s) {sorted(wanted)} "
                    "cannot be served concurrently. The rings are a fixed "
                    "physical buffer and there is nothing to evict -- raise "
                    "`resources.kv.num_worlds` (and `max_concurrent_requests` "
                    "with it) to serve more."
                ),
            )

        for rid in rids:
            frame = frames.get(rid)
            if frame is None:
                return AdmitOutcome(
                    ok=False, ready=False,
                    reason=AdmitRuntimeError(
                        f"ring KV {self.name!r} was handed a step declaring no ring "
                        f"clock for request {rid!r} (it names {sorted(frames)}). "
                        "Every admitted request needs one: the continuity check is "
                        "the only thing standing between a stalled clock and a "
                        "world that rewrites its own history."
                    ),
                )
            last = self._last_frames.get(rid)
            if last is not None and frame != last + 1:
                return AdmitOutcome(
                    ok=False, ready=False,
                    reason=AdmitRuntimeError(
                        f"ring KV {self.name!r} last committed frame "
                        f"{last} for request {rid!r}, so the next one must be "
                        f"{last + 1}; this step declares frame {frame}. "
                        "The ring clock advances by exactly one per committed frame "
                        "-- a skipped or repeated frame selects the wrong ring slot "
                        "and hides the wrong one, which rewrites history rather than "
                        "raising."
                    ),
                )

        for rid in rids:
            if rid not in self._worlds:
                world_idx = min(self._free_worlds)
                self._free_worlds.remove(world_idx)
                self._worlds[rid] = world_idx
        return ADMIT_OK

    def plan(self, step: ResourceStep, ctx: StepContext) -> None:
        """Stage this step's world index. Its ring addresses stay in the graph."""

        del step
        rids = {*ctx.request_ids}
        if len(rids) != 1:
            raise ValueError(
                f"ring KV {self.name!r} can stage one world index per step and "
                f"this step names {sorted(rids)}. `admit` refuses a mixed batch "
                "for the same reason; reaching here means it was bypassed."
            )
        rid = rids.pop()
        world_idx = self._require_world(rid, "plan")
        # `fill_`, not a `copy_` from a fresh host tensor: same in-place write
        # through the address the graph baked, without allocating a staging
        # tensor 24 times a second.
        self._static_world_idx.fill_(world_idx)

    def commit(self, step: ResourceStep, ctx: StepContext) -> None:
        """Record the frame each request just committed. Metadata only."""
        frames = self._step_frames(step)
        for rid in ctx.request_ids:
            frame = frames.get(rid)
            if frame is not None:
                self._last_frames[rid] = frame

    def reset_request(self, rid: str, free: bool = False) -> None:
        """Zero ``rid``'s world and release its claim.
        ``free`` is ignored, there is no physical allocation to hand back.
        """
        del free
        self._release_world(rid)

    def remove_request(self, rid: str) -> None:
        """The request is gone: drop its world, its claim, and its registration."""
        self._release_world(rid)
        self._known_rids.discard(rid)

    # `supports_preplan` stays the inherited False.

    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int
    ) -> None:
        """No-op: every buffer a replay touches was allocated at ``build``,
        ``_static_world_idx`` included.
        """
        del slots, max_bs, max_seq_len

    def post_warmup_validate(self) -> None:
        """Capture must leave every world exactly as it found it."""

        if self._worlds:
            raise RuntimeError(
                f"ring KV {self.name!r} is still claimed by {sorted(self._worlds)} "
                "after CUDA graph capture; a capture dummy rid was never reset, "
                "and the world it holds is gone from the pool for good."
            )
        if len(self._free_worlds) != self.num_worlds:
            raise RuntimeError(
                f"ring KV {self.name!r} has {len(self._free_worlds)} of "
                f"{self.num_worlds} worlds free after CUDA graph capture; a world "
                "was zeroed and never returned to the pool, so the node has "
                "silently lost concurrency."
            )
        if self._last_frames:
            raise RuntimeError(
                f"ring KV {self.name!r} recorded committed frames "
                f"({sorted(self._last_frames.items())}) during CUDA graph capture; "
                "capture drives admit and plan but must never commit, and the "
                "first real request will be refused at admit unless it happens to "
                "declare the very next frame."
            )
        for i, layer in enumerate(self.layers):
            history = layer.written.view(layer.num_worlds, layer.capacity)[:, : layer.ring_len]
            if bool(layer.kv.any()) or bool(history.any()):
                raise RuntimeError(
                    f"ring KV {self.name!r} layer {i} still holds capture-time "
                    "frames after warmup; the first rollout would attend to them "
                    "as real history."
                )

    def cleanup(self) -> None:
        return
