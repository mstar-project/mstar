"""The registry of the tier 0 machines.

``MACHINES`` maps the name of a machine to its class. The command line and the
tests read this map. Add a new machine here to make it available.
"""

from fuzzer.common.machine import StateMachine
from fuzzer.tier0.alloc_concurrent import AllocConcurrentMachine
from fuzzer.tier0.graph_io import GraphIOMachine
from fuzzer.tier0.micro_scheduler import MicroSchedulerMachine
from fuzzer.tier0.page_allocator import PageAllocatorMachine
from fuzzer.tier0.step_runner import StepRunnerMachine
from fuzzer.tier0.tensor_store import TensorStoreMachine

MACHINES: dict[str, type[StateMachine]] = {
    machine.name: machine
    for machine in (
        PageAllocatorMachine,
        AllocConcurrentMachine,
        TensorStoreMachine,
        GraphIOMachine,
        MicroSchedulerMachine,
        StepRunnerMachine,
    )
}

TIER = "tier0"

__all__ = ["MACHINES", "TIER"]
