"""Put this directory on ``PYTHONPATH`` and every Python process started with it (the
server, the conductor, each worker) serves Qwen3.5's LLM node eagerly: no captured
graphs, no ``torch.compile``. For the captured-vs-eager checks in ``benchmark/lora``
only; nothing in mstar reads it.

The patch is applied when ``mstar.model.qwen3_5.submodules`` is first imported, so a
process that never loads the model pays nothing. Check the server log for
``torch.compile disabled for LLM`` to confirm it took.
"""

import importlib.abc
import importlib.machinery
import sys

_TARGET = "mstar.model.qwen3_5.submodules"


def _eager(module) -> None:
    cls = module.LLMSubmodule
    cls.get_accelerator_graph_configs = lambda self, *args, **kwargs: []
    cls.disable_torch_compile = True


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def patched(module):
            exec_module(module)
            _eager(module)

        spec.loader.exec_module = patched
        return spec


sys.meta_path.insert(0, _PatchOnImport())
