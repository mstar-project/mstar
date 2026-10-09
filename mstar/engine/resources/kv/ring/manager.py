"""The paged ``KVManager`` reserves pages per step and hands them back;
this one reserves nothing. Every layer's ring is allocated once at
``build`` and reused for the life of the process,.
"""

from typing import NamedTuple

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

__all__ = ["RingKVManager", "RingPlan"]


class RingPlan(NamedTuple):
    """Host facts downstream resources need to stage this fixed ring view, one
    entry per row of the step batch, in ``ctx.request_ids`` order."""

    request_ids: tuple[str, ...]
    session_idx: tuple[int, ...]
    frame_pos: tuple[int, ...]


class RingKVManager(AttentionResource):
    """Per-layer ring caches holding ``num_sessions`` sessions, one per request."""

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

        # ``total_sessions`` == ``num_sessions`` + 1: the extra session is the shared
        # scratch that ``plan`` parks a replay's padding tail on. It is never in
        # ``_free_sessions``, so no request is admitted to it.
        self.layers = [
            LayerRingCache(
                num_sessions=config.total_sessions,
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

        self._sessions: dict[str, int] = {}
        self._free_sessions: set[int] = set(range(config.num_sessions))
        # The one session past the resident pool; padding rows write here and it is
        # never claimed, so a padding write never reaches a real session's history.
        self._padding_session: int = config.num_sessions
        self._known_rids: set[str] = set()
        self._last_frames: dict[str, int] = {}
        # Sized to num_sessions, the largest B any step can carry (B_max <=
        # num_sessions is enforced at config load).
        self._static_session_idx = torch.zeros(
            config.num_sessions, dtype=torch.int64, device=self.device
        )

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
    def num_sessions(self) -> int:
        """How many requests can hold a session here at once."""
        return self.config.num_sessions

    def capacity(self, layer_idx: int) -> int:
        """Token slots ONE session owns in ``layer_idx``'s ring, scratch frame
        included. ``total_slots`` is the whole buffer."""
        return self.layers[layer_idx].capacity

    def total_slots(self, layer_idx: int) -> int:
        """Token slots in ``layer_idx``'s buffer across every session -- the
        length of the KV view ``upsert`` returns and of its visibility row."""
        return self.layers[layer_idx].total_slots

    def session_of(self, rid: str) -> int | None:
        """``rid``'s session index, or None if it holds none. Host-side
        introspection; the forward reads the staged tensor, never this."""
        return self._sessions.get(rid)

    def upsert(
        self,
        k: Tensor,
        v: Tensor,
        layer_idx: int,
        frame_pos: Tensor,
        *,
        commit: bool,
        build_visibility: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Write one step's K/V for ``layer_idx`` and return what to attend to.

        ``k``/``v`` are ``[B, H_kv, tokens_per_frame, D]``, one frame per
        resident session in the step batch. ``k`` is already RoPE'd and
        RMS-normed and ``v`` is post value-residual lerp: the cache stores
        post-RoPE keys, so replayed history is never re-rotated.
        Returns ``(k_all, v_all, visible)``, the first two spanning
        the whole buffer, every resident session and the third a
        ``[total_slots]`` bool row that is False everywhere outside the calling
        request's own session.

        ``frame_pos`` and ``commit`` are both arguments rather than resource
        state, for the same reason. ``frame_pos`` is the ``[B]`` int64 ring
        clock -- not a slot id -- and alone determines the slot written and the
        visibility row;
        """
        # `layer_idx` is Python-level (it indexes a list of differently-shaped
        # rings), so indexing on it is graph-safe.
        B = k.size(0)
        # [2, H_kv, B, T, D] contiguous, then folded into the ring's [2, 1,
        # H_kv, B*T, D] layout -- the one copy this already needed.
        kv = torch.stack([k.transpose(0, 1), v.transpose(0, 1)], dim=0)
        kv = kv.view(2, 1, k.size(1), B * k.size(2), k.size(3))
        return self.layers[layer_idx].upsert(
            kv, frame_pos, commit, self._static_session_idx[:B],
            build_visibility=build_visibility,
        )

    def _reset_session(self, rid: str) -> None:
        """Zero ``rid``'s session and drop its clock, leaving its claim in place."""
        session_idx = self._sessions.get(rid)
        if session_idx is None:
            return
        for layer in self.layers:
            layer.reset(session_idx)
        self._last_frames.pop(rid, None)

    def _release_session(self, rid: str) -> None:
        """Hand ``rid``'s session back to the pool. Zero it first."""
        self._reset_session(rid)
        session_idx = self._sessions.pop(rid, None)
        if session_idx is not None:
            self._free_sessions.add(session_idx)

    @torch.no_grad()
    def get_state(self, rid: str) -> dict:
        """Snapshot one request's session. Cloned, so the caller can hold it
        across further rollout steps that mutate the rings in place.
        """
        session_idx = self._require_session(rid, "get_state")
        layers = []
        for layer in self.layers:
            lo, hi = layer.session_span(session_idx)
            layers.append((
                layer.kv[:, :, :, lo:hi].detach().clone(),
                layer.written[lo:hi].detach().clone(),
            ))
        return {"layers": layers}

    @torch.no_grad()
    def load_state(self, rid: str, state: dict) -> None:
        """Restore one session's contents into the existing allocation."""

        session_idx = self._require_session(rid, "load_state")
        layers = state["layers"]
        if len(layers) != len(self.layers):
            raise ValueError(
                f"state has {len(layers)} layers, ring has {len(self.layers)}."
            )
        for i, (layer, (kv, written)) in enumerate(zip(self.layers, layers, strict=True)):
            lo, hi = layer.session_span(session_idx)
            span = layer.kv[:, :, :, lo:hi]
            if tuple(kv.shape) != tuple(span.shape):
                raise ValueError(
                    f"layer {i} state shape {tuple(kv.shape)} != one session's ring "
                    f"shape {tuple(span.shape)}."
                )
            span.copy_(kv)
            layer.written[lo:hi].copy_(written)
        self._last_frames.pop(rid, None)

    def _require_session(self, rid: str, what: str) -> int:
        session_idx = self._sessions.get(rid)
        if session_idx is None:
            raise KeyError(
                f"ring KV {self.name!r} has no session for request {rid!r}; "
                f"{what} is per request and a request that never admitted owns "
                "no span to read or write."
            )
        return session_idx

    # ---- Introspection ----------------------------------------------------

    def memory_bytes(self) -> int:
        """Total resident ring bytes across all layers and all sessions."""
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
        """Register ``rid``. Deliberately does not claim a session."""

        del overrides  # no per-request tunables: the ring geometry is fixed
        self._known_rids.add(rid)

    def admit(self, step: ResourceStep, ctx: StepContext) -> AdmitOutcome:
        """Claim a session for this step's request, or refuse it terminally.
        Does not support eviction for now
        """
        # `request_ids`, not `padded_request_ids` as the paged manager uses: a
        # padding row is a dummy rid a replay pads a bucket out to, and it must
        # not be able to take a session from the real request in the same batch.
        rids = list(ctx.request_ids)

        if len(set(rids)) != len(rids):
            return AdmitOutcome(
                ok=False, ready=False,
                reason=AdmitRuntimeError(
                    f"ring KV {self.name!r} was handed a batch naming "
                    f"{sorted({rid for rid in rids if rids.count(rid) > 1})} more "
                    "than once; a step batches distinct sessions, one row per request."
                ),
            )

        frames = self._step_frames(step)

        # Check every rid before claiming any session: a refused admit must not
        # leave a session half-claimed by the first rid of a batch it rejected.
        wanted = {rid for rid in rids if rid not in self._sessions}
        for rid in sorted(wanted):
            if rid not in self._known_rids:
                return AdmitOutcome(
                    ok=False, ready=False,
                    reason=AdmitRuntimeError(
                        f"ring KV {self.name!r} was asked to admit request {rid!r}, "
                        "which was never ingested. Worlds are handed back by "
                        "`remove_request`, which only ever runs for a request the "
                        "engine opened -- so a session claimed here would never "
                        "return to the pool and the node would lose capacity with "
                        "nothing raised."
                    ),
                )
        if len(wanted) > len(self._free_sessions):
            return AdmitOutcome(
                ok=False, ready=False,
                reason=AdmitRuntimeError(
                    f"ring KV {self.name!r} holds all {self.num_sessions} of its "
                    f"sessions ({sorted(self._sessions)}); request(s) {sorted(wanted)} "
                    "cannot be served concurrently. The rings are a fixed "
                    "physical buffer and there is nothing to evict -- raise "
                    "`resources.kv.num_sessions` (and `max_concurrent_requests` "
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
                        "session that rewrites its own history."
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
            if rid not in self._sessions:
                session_idx = min(self._free_sessions)
                self._free_sessions.remove(session_idx)
                self._sessions[rid] = session_idx
        return ADMIT_OK

    def plan(self, step: ResourceStep, ctx: StepContext) -> RingPlan:
        """Stage the padded batch's session indices, one per row. Ring addresses
        stay in the graph.

        A replay pads the batch to its capture bucket with dummy rids that hold
        no session. Those rows still write and attend (their output is dropped
        downstream), so every one is parked on ``_padding_session`` -- the shared
        scratch session outside the resident pool. The flex mask scopes each row to
        its own session, so a padding row's read and its commit both stay on that
        session and never reach a real request's history; no request is ever
        admitted to it, so the tail it leaves behind is never read.
        """
        frames = self._step_frames(step)
        real_rids = tuple(ctx.request_ids)
        padded_rids = tuple(ctx.padded_request_ids)
        session_idx = []
        frame_pos = []
        for b, rid in enumerate(real_rids):
            session = self._require_session(rid, "plan")
            # `fill_`, not a `copy_` from a fresh host tensor: same in-place
            # write through the address the graph baked, without allocating a
            # staging tensor 24 times a second.
            self._static_session_idx[b].fill_(session)
            session_idx.append(session)
            frame_pos.append(frames[rid])
        for b in range(len(real_rids), len(padded_rids)):
            self._static_session_idx[b].fill_(self._padding_session)
            session_idx.append(self._padding_session)
            # Frame 0's visibility is scratch-only, so the padding row attends
            # exactly one (garbage, dropped) block and never an unwritten slot.
            frame_pos.append(0)
        return RingPlan(padded_rids, tuple(session_idx), tuple(frame_pos))

    def commit(self, step: ResourceStep, ctx: StepContext) -> None:
        """Record the frame each request just committed. Metadata only."""
        frames = self._step_frames(step)
        for rid in ctx.request_ids:
            frame = frames.get(rid)
            if frame is not None:
                self._last_frames[rid] = frame

    def reset_request(self, rid: str, free: bool = False) -> None:
        """Zero ``rid``'s session and release its claim.
        ``free`` is ignored, there is no physical allocation to hand back.
        """
        del free
        self._release_session(rid)

    def remove_request(self, rid: str) -> None:
        """The request is gone: drop its session, its claim, and its registration."""
        self._release_session(rid)
        self._known_rids.discard(rid)

    # `supports_preplan` stays the inherited False.

    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int
    ) -> None:
        """No-op: every buffer a replay touches was allocated at ``build``,
        ``_static_session_idx`` included.
        """
        del slots, max_bs, max_seq_len

    def post_warmup_validate(self) -> None:
        """Capture must leave every session exactly as it found it."""

        if self._sessions:
            raise RuntimeError(
                f"ring KV {self.name!r} is still claimed by {sorted(self._sessions)} "
                "after CUDA graph capture; a capture dummy rid was never reset, "
                "and the session it holds is gone from the pool for good."
            )
        if len(self._free_sessions) != self.num_sessions:
            raise RuntimeError(
                f"ring KV {self.name!r} has {len(self._free_sessions)} of "
                f"{self.num_sessions} sessions free after CUDA graph capture; a session "
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
            history = layer.written.view(layer.num_sessions, layer.capacity)[:, : layer.ring_len]
            if bool(layer.kv.any()) or bool(history.any()):
                raise RuntimeError(
                    f"ring KV {self.name!r} layer {i} still holds capture-time "
                    "frames after warmup; the first rollout would attend to them "
                    "as real history."
                )

    def cleanup(self) -> None:
        return
