# tier 2 — fault injection (not implemented yet)

Tier 2 is tier 1 plus a seeded layer of faults. Most of the places to inject a
fault already exist:

* **Allocation failure** — a resource returns `AllocationFailed` or
  `RequestOffloading` on a seeded schedule. This drives
  `_handle_admit_failure` and `_handle_allocation_failure`
  (`mstar/worker/worker.py:1405,1431`) and the re-check of the backlog. Commit
  `bf6fbc56` ("guard speculation None on allocation-failed plan_future
  cleanup") corrected a bug in this area, in combination with speculation.
* **An abort at each step boundary** — call `Conductor._abort_request`
  (`mstar/conductor/conductor.py:976`) at each point in the life of a request.
  Then run the quiesce oracle. PR #234 (teardown against reader drain) is an
  example of a bug in the order of the abort operations.
* **Changed messages** — `Worker._process_message_list`
  (`mstar/worker/worker.py:718`) is one single point of control. Change the
  order inside the limits that the protocol permits. Also duplicate a message,
  or delay it. Use any message type in `mstar/utils/ipc_format.py` whose order
  the protocol does not guarantee.
* **Errors from a worker** — these must propagate as `FailRequests`.
* **Eviction storms** — force cycles of `offload` and `reload` while requests
  are in flight.

Determinism is the difficult part. Record each fault decision as a pair of the
logical step and the action. Do not record the time from the clock. If you
record the time, no case replays and no case shrinks.

## Exhaustive interleaving checks

Random fuzzing is weak against races; exhaustive state exploration is strong
against them, and the repository already has an example.
`test/modular/tp_async_sim.py` is an explicit-state model checker. It does a
depth-first search over every interleaving of a small scripted workload. It has
named invariants: I1 lockstep, I2 KV symmetry, I3 atomic enqueue and I4
liveness. It disproved the simple reading of cancel (`B2_RETRACT`) before that
reading could ship.

Take the world, action and invariant structure of that file and make a reusable
checker. Then write a model for each of the other protocols:

* the order of abort, teardown and drain
* the handoff between plan and replay in the double buffer
* the producer-done signal of a stream
* the publish and retrieve steps of a disaggregated prefill and decode

This covers more than the random fault layer does, and it runs on the CPU in
CI.

Division of the work: **use random fuzzing for data and for lifecycles. Use an
exhaustive depth-first search for protocols.**
