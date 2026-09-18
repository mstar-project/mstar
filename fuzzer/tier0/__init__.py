"""Tier 0: state-machine fuzzing of the pure-Python internals of mstar.

This tier uses no torch kernel, no GPU and no weights. It drives the data
structures that decide which request runs, which memory it runs on, and when
that memory goes back. One case costs a few milliseconds. For this reason,
tier 0 is the tier that runs in CI.

Each machine module imports ``fuzzer.tier0._stubs`` before it imports mstar.
Do not import a machine in this file. Without that rule, the imports become
circular.
"""
