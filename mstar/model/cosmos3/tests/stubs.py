"""Stand-ins the Cosmos3 DiT tests share."""


class StubSampler:
    """A request's sampler reduced to its schedule: ``update`` is the whole step
    and nothing rides the loop-back edges. Pair it with ``sub._solver_state = ()``."""

    def __init__(self, timesteps, update=lambda sample, velocity: sample - velocity):
        self.timesteps = timesteps
        self.update = update

    def initial_state(self, sample):
        return {}

    def step(self, model_output, sample, step_index, solver, generator=None):
        return self.update(sample, model_output), {}
