"""What the worker tells the engine of a walk a deployment asks to repeat
(``step_repeats:``)."""
from __future__ import annotations

from typing import NamedTuple


class WalkAsk(NamedTuple):
    """What the worker tells the engine of a walk asked to repeat, before capture."""
    # the signals one iteration hands the next on the device
    loop_back: frozenset[str]
    # the most iterations a device run takes; 0 replays it from the host
    device_limit: int = 0
