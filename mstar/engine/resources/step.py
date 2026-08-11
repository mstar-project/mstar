
from dataclasses import dataclass, field


class ResourceStep:
    ...



@dataclass(frozen=True)
class Segment:
    """One step's addition to a request's cache stream.

    A request contributes one segment per label active for it in a step;
    the batch's ordered segment list defines the layout of per-token
    arrays. ``span`` may be 0: a zero-span segment reads its stream
    without extending it (admission reserves nothing, commit is a no-op).
    """
    request_id: str
    label: str
    span: int


@dataclass
class KVStep(ResourceStep):
    segments: list[Segment] = field(default_factory=list)


@dataclass
class SubmoduleStep:
    kv_steps: dict[str, KVStep]
    # As per Atindra's comment on my PR comment, maybe we can have the
    # submodule return a dataclass like this