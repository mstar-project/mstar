"""Instrument a disposable M* source copy for one live Praxis stage trace.

This modifies only the copy supplied by the diagnostic server launcher. It
must never run on the working tree or a production server.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HOOK = r'''
"""Diagnostic Praxis node trace, installed only in a disposable server copy."""

import functools
import inspect
import itertools
import os
from pathlib import Path

import torch

_counter = itertools.count()
_output = Path(os.environ["MSTAR_PRAXIS_STAGE_TRACE_DIR"])
_output.mkdir(parents=True, exist_ok=True)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu(item) for item in value]
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    return repr(value)


def install_trace(namespace):
    names = (
        "Praxis1TextEncoderSubmodule",
        "Praxis1VaeEncoderSubmodule",
        "Praxis1BackboneSubmodule",
        "Praxis1ActionExpertSubmodule",
        "Praxis1ActionHeadSubmodule",
    )
    for name in names:
        cls = namespace[name]
        original = cls.forward
        signature = inspect.signature(original)

        @functools.wraps(original)
        def traced(self, *args, _original=original, _signature=signature, _name=name, **kwargs):
            bound = _signature.bind_partial(self, *args, **kwargs)
            # Only tensor inputs and small scalar settings. Exclude engine
            # handles and model weights from diagnostic artifacts.
            inputs = {
                key: _cpu(value)
                for key, value in bound.arguments.items()
                if key not in ("self", "engine_inputs", "graph_walk", "kwargs")
            }
            result = _original(self, *args, **kwargs)
            artifact = {"stage": _name, "inputs": inputs, "outputs": _cpu(result)}
            path = _output / f"{os.getpid()}-{next(_counter):03d}-{_name}.pt"
            torch.save(artifact, path)
            return result

        cls.forward = traced
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="disposable source copy")
    args = parser.parse_args()
    source = args.source.resolve()
    if ".git" in source.parts or (source / ".git").exists():
        raise SystemExit("refusing to instrument a Git working tree")
    module_dir = source / "mstar" / "model" / "praxis1"
    module = module_dir / "submodules.py"
    hook = module_dir / "_diagnostic_stage_trace.py"
    if not module.is_file() or hook.exists():
        raise SystemExit("unexpected disposable M* source layout")
    hook.write_text(HOOK)
    with module.open("a") as out:
        out.write("\nfrom ._diagnostic_stage_trace import install_trace as _install_trace\n")
        out.write("_install_trace(globals())\n")
    print(f"instrumented disposable source {source}")


if __name__ == "__main__":
    main()
