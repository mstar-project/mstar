"""The registry of the tier 1 machines.

``MACHINES`` maps the name of a machine to its class. The command line and the
tests read this map. Add a new machine here to make it available.
"""

from fuzzer.common.machine import StateMachine
from fuzzer.tier1.kv_race import KvRaceMachine
from fuzzer.tier1.kv_run import KvRunMachine
from fuzzer.tier1.model_run import ModelRunMachine

MACHINES: dict[str, type[StateMachine]] = {
    machine.name: machine
    for machine in (
        ModelRunMachine,
        KvRunMachine,
        KvRaceMachine,
    )
}

TIER = "tier1"

__all__ = ["MACHINES", "TIER"]
