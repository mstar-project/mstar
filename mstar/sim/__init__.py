"""M*-Sim: performance prediction for mstar deployments.

Three layers, usable independently:

* ``stepdb``      — measured per-step GPU/CPU costs, keyed by the padded shape
                    the engine actually executes.
* ``step_profiler`` — drives synthetic batches through the *real* engines to
                    fill the stepdb.
* ``des``         — a virtual-time discrete-event simulator that imports
                    mstar's own graph, scheduler, and conductor code and
                    prices every step from the stepdb.

The guiding rule: reuse mstar's semantics by importing them, and measure
costs rather than modeling them.
"""

from mstar.sim.stepdb import Coverage, StepCost, StepDB, StepKey, StepSample

__all__ = ["StepDB", "StepKey", "StepCost", "StepSample", "Coverage"]
