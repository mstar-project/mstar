"""Stand-ins the Cosmos3 DiT tests share."""

from mstar.engine.state.adapters import HostValueHandle
from mstar.engine.state.host_fields import HostFields
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
from mstar.model.cosmos3.submodules import PROGRESS


class StubSampler:
    """A request's sampler reduced to its schedule: ``update`` is the whole step
    and nothing rides the loop-back edges. Pair it with ``sub._solver_state = ()``."""

    def __init__(self, timesteps, update=lambda sample, velocity: sample - velocity):
        self.timesteps = timesteps
        self.update = update

    def initial_state(self, sample):
        return {}

    def step(self, model_output, sample, step_index, solver, noise=None):
        return self.update(sample, model_output), {}


def bind_progress(sub, *rid_handles) -> HostFields:
    """Bind ``sub`` the ``PROGRESS`` handle the engine binds a DiT, with host fields open for ``rid_handles``, as each
    request's ingest opens them."""
    declared = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True).declare_state()
    (progress,) = [array for array in declared if array.name == PROGRESS]
    fields = HostFields()
    sub.progress = HostValueHandle(progress, fields)
    for rid in rid_handles:
        fields.open(rid)
    return fields
