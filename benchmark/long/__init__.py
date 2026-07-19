"""Long-running soak / large-tensor stress harness (client half).

Drives a *mixture* of request types at a Poisson arrival rate with a cap on
in-flight requests, for a fixed wall-clock duration, and reports rolling
moving-average metrics (tok/s, audio-s/s, e2e) plus failure / timeout /
client-side backpressure signals over time.

This is the client half of the two SHM-arena stress runs:
  (a) multi-hour soak at mixed concurrency with heterogeneous tensor sizes, and
  (b) large-tensor stress driven toward the arena cap.

The server half (a launch wrapper that samples SHM arena ``stats()`` — segment
count, ``largest_free_block / free``, pinned bytes — over the same time axis) is
separate; both write timestamped JSONL so the two series line up.
"""
