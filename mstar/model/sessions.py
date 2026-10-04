"""Session declarations models hand to the runtime.

A session is a named, longer-lived context a client resumes across requests:
the resources named here keep their state when a request is torn down and only
release it when the session ends (or its TTL expires).
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Hard ceiling on what a deployment may configure, so a typo in a YAML cannot
# park state on a GPU for a day.
ABSOLUTE_MAX_TIMEOUT_S = 24 * 3600


class SessionOverflowPolicy(Enum):
    """What happens when a session's held state exceeds its budget.

    A backstop, not a way to trim: a bounded rolling context is the resource's
    own business (a KV stream's ``RetentionPolicy``), not this budget's.
    """

    # Drop everything the session holds; the next request starts from scratch.
    CLEAR = "clear"
    # Drop everything and fail the session's next request, so a client is told
    # rather than silently served from a truncated context.
    ERROR = "error"


class SessionCapacityPolicy(Enum):
    """What a deployment at ``max_concurrent_sessions`` does with a new one."""

    # Nothing: it lives until the client ends the session or its TTL expires,
    # and a new session past the cap is refused.
    KEEP = "keep"
    # The least recently used idle session is torn down to make room. A session
    # with a request in flight is never evicted, so a full deployment of
    # in-flight sessions still refuses a new one.
    EVICT = "evict"


class SessionTTLMode(Enum):
    # Expire a session ``timeout_s`` after its last request finished.
    IDLE = "idle"
    # Expire it ``timeout_s`` after it was started, however busy it is.
    ABSOLUTE = "absolute"


@dataclass
class RequestSession:
    """The session one request belongs to, as the server validated it.

    Handed to the model with the request's initial forward-pass args and
    carried on every forward pass after it, so a model and its submodules read
    the session from the request rather than from the client's knobs.
    """

    session_id: str
    # Continuing a session that already holds state, rather than opening one.
    resumed: bool = False
    # The session ends when this request finishes.
    end_session: bool = False


@dataclass
class SessionResourceConfig:
    """One resource's session behaviour."""

    # Resource-native units: pages for a KV cache, slots for recurrent state.
    # None means unbounded (bounded in practice by the resource's own pool).
    max_state: int | None = None
    overflow_policy: SessionOverflowPolicy = SessionOverflowPolicy.CLEAR

    def __post_init__(self):
        if isinstance(self.overflow_policy, str):
            self.overflow_policy = SessionOverflowPolicy(self.overflow_policy)
        if self.max_state is not None and self.max_state <= 0:
            raise ValueError(
                f"max_state must be positive, got {self.max_state}"
            )


@dataclass
class SessionsConfig:
    """A model's session support. ``None`` from the model means unsupported."""

    # resource_key -> how that resource holds state across the session. A
    # resource not named here is cleared with the request, as before.
    resources: dict[str, SessionResourceConfig] = field(default_factory=dict)
    max_concurrent_sessions: int = 8
    default_timeout_s: float = 300.0
    max_timeout_s: float = 3600.0
    ttl_mode: SessionTTLMode = SessionTTLMode.IDLE
    # What a deployment at its cap does; see SessionCapacityPolicy.
    capacity_policy: SessionCapacityPolicy = SessionCapacityPolicy.KEEP
    # Sessions are one-request-at-a-time for now; the field names the
    # assumption the conductor and the resources rely on.
    # TODO: bidirectional streaming needs an `interruptible` flag here, so a
    # resume may land on a session that already has a request in flight. It
    # needs the runtime to route the second request's inputs into the first,
    # which nothing does yet; two requests sharing a session's resource state
    # would corrupt it.
    max_requests_in_flight: int = 1

    def __post_init__(self):
        if isinstance(self.ttl_mode, str):
            self.ttl_mode = SessionTTLMode(self.ttl_mode)
        if isinstance(self.capacity_policy, str):
            self.capacity_policy = SessionCapacityPolicy(self.capacity_policy)
        self.resources = {
            key: (
                cfg if isinstance(cfg, SessionResourceConfig)
                else SessionResourceConfig(**cfg)
            )
            for key, cfg in self.resources.items()
        }
        if self.max_concurrent_sessions < 1:
            raise ValueError(
                "max_concurrent_sessions must be at least 1, got "
                f"{self.max_concurrent_sessions}"
            )
        if self.max_timeout_s > ABSOLUTE_MAX_TIMEOUT_S:
            raise ValueError(
                f"max_timeout_s {self.max_timeout_s} exceeds the "
                f"{ABSOLUTE_MAX_TIMEOUT_S}s ceiling"
            )
        if not 0 < self.default_timeout_s <= self.max_timeout_s:
            raise ValueError(
                f"default_timeout_s {self.default_timeout_s} must be in "
                f"(0, max_timeout_s={self.max_timeout_s}]"
            )
        if self.max_requests_in_flight != 1:
            raise ValueError(
                "sessions currently allow exactly one in-flight request"
            )

    def resolve_timeout_s(self, requested: float | None) -> float:
        """The TTL a request's ``session_timeout_s`` resolves to.

        Raises ``ValueError`` for a request that asks for longer than the
        deployment allows, rather than silently clamping it.
        """
        if requested is None:
            return self.default_timeout_s
        if requested <= 0:
            raise ValueError("session_timeout_s must be positive")
        if requested > self.max_timeout_s:
            raise ValueError(
                f"session_timeout_s {requested} exceeds this deployment's "
                f"maximum of {self.max_timeout_s}"
            )
        return float(requested)


def apply_sessions_yaml_overrides(
    config: SessionsConfig | None, model_config: Mapping[str, Any],
) -> SessionsConfig | None:
    """Apply a deployment's ``sessions:`` block to the model's declaration.

    A deployment may tune the caps and the per-resource budgets, and may
    disable sessions outright with ``sessions: {enabled: false}``; it may not
    invent a session resource the model never declared.
    """
    overrides = dict(model_config.get("sessions") or {})
    if not overrides:
        return config
    if not overrides.pop("enabled", True):
        return None
    if config is None:
        raise ValueError(
            "serving config has a `sessions:` block, but this model does not "
            "support sessions (Model.get_sessions_config returned None)"
        )

    resources = overrides.pop("resources", None)
    known = {
        "max_concurrent_sessions", "default_timeout_s", "max_timeout_s",
        "ttl_mode", "capacity_policy",
    }
    unknown = sorted(overrides.keys() - known)
    if unknown:
        raise ValueError(
            f"unknown key(s) {unknown} in the serving config's `sessions:` "
            f"block; it takes {sorted(known)}, `resources` and `enabled`"
        )

    merged = SessionsConfig(
        resources=dict(config.resources),
        max_concurrent_sessions=overrides.get(
            "max_concurrent_sessions", config.max_concurrent_sessions
        ),
        default_timeout_s=overrides.get(
            "default_timeout_s", config.default_timeout_s
        ),
        max_timeout_s=overrides.get("max_timeout_s", config.max_timeout_s),
        ttl_mode=overrides.get("ttl_mode", config.ttl_mode),
        capacity_policy=overrides.get(
            "capacity_policy", config.capacity_policy
        ),
    )
    for key, kwargs in (resources or {}).items():
        if key not in merged.resources:
            raise ValueError(
                f"serving config configures session state for resource "
                f"{key!r}, which this model does not hold across a session; "
                f"it declares {sorted(merged.resources)}"
            )
        merged.resources[key] = SessionResourceConfig(**{
            **vars(merged.resources[key]), **kwargs,
        })
    logger.info("Sessions config after YAML overrides: %s", merged)
    return merged
