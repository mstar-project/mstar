"""Eval suites by task. Add a suite by subclassing ``Suite`` and listing it here."""

from benchmark.eval.suites.asr import ASRSuite
from benchmark.eval.suites.base import Suite
from benchmark.eval.suites.duplex import DuplexSuite
from benchmark.eval.suites.e2e import E2ESuite
from benchmark.eval.suites.tts import TTSSuite

SUITES: dict[str, type[Suite]] = {s.name: s for s in (TTSSuite, ASRSuite, DuplexSuite, E2ESuite)}

__all__ = ["SUITES", "Suite"]
