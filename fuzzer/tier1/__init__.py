"""Tier 1: fuzzing of the engine over a generated synthetic model.

Tier 0 fuzzes the data structures one at a time. Tier 1 makes the model an
input. It runs that model, and it compares every value that reaches the client
against an interpreter of the same model.

Three machines exist:

* ``model_run`` runs a generated model over the real graph layer of mstar
  (``mstar/graph/``).
* ``kv_run`` runs the same generated model over the real resources as well
  (``mstar/engine/resources/``): a real ``KVManager``, a real
  ``PositionManager`` and a real ``StepRunner``. The value of each step covers
  the cache pages that the step reads. This machine also drives three paths
  that tier 0 cannot reach. Those paths are a step that is planned one step
  ahead, the padded addressing of a capture bucket, and a fork between two
  streams.
* ``kv_race`` drives those same resources on several threads. The
  interleaving lives in the case, so a race replays and shrinks.

Every machine runs on the CPU. None of them needs a GPU, weights or a kernel.

``README.md`` holds the design of the whole tier. It also holds the two lanes
that have no machine yet. Lane 1a runs over the real conductor and the real
worker. Lane 1b runs on a device.

Do not import a machine in this file. Without that rule, the imports become
circular.
"""
