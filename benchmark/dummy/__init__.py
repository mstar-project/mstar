"""Load drivers for the no-op ``dummy_loop`` / ``dummy_walks`` models.

These models do no GPU work, so end-to-end time is dominated by the runtime
itself. Driving them with a known step count K isolates the per-step dispatch
overhead; see ``benchmark/dummy/sweep.py`` for the JCT = a + b·K fit.
"""
