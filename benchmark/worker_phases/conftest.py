"""Put this directory on sys.path so ``pytest benchmark/worker_phases`` works
from the repo root, not only from inside the directory.

The module under test is a plain script next to its test rather than a package
(it is meant to be run as ``python server.py ...``), so there is no import path
for pytest to find it by from elsewhere.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
