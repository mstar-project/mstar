"""CUDA extensions JIT-built on the emulated GPU (``torch.utils.cpp_extension.load``).

On a GPU, ``load`` compiles CUDA sources with nvcc and their ``TORCH_LIBRARY`` ops
register, capturable like any kernel. Here there is no nvcc: a build with CUDA sources
registers each op its sources define (``m.def("<schema>")`` in a ``TORCH_LIBRARY`` or
``TORCH_LIBRARY_FRAGMENT`` block with a literal namespace) as a custom op whose body is
one kernel of the simulated GPU, priced by its arguments' traffic, that writes its
mutable arguments (``Tensor!``) and returns nothing. What it can't read that way raises
as a failed build does, so the caller takes its fallback: a schema inferred from C++
(``m.def("name", &fn)``), an op that returns tensors, a Python-module build. A build
without CUDA sources is the real one.
"""

from __future__ import annotations

import inspect
import re

import torch

from mstar_sim import clock, triton_launch

_BLOCK = re.compile(r"TORCH_LIBRARY(?:_FRAGMENT)?\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\{")
_LITERALS = re.compile(r'\s*((?:"(?:[^"\\]|\\.)*"\s*)+)')


def _failed(source: str, why: str):
    raise RuntimeError(f"Error building extension: mstar_sim: {source}: {why} (no nvcc here)")


def _defs(source: str) -> list[tuple[str, str, str]]:
    """``(namespace, schema, source)`` of each op the source defines."""
    with open(source) as f:
        text = re.sub(r"//[^\n]*|/\*.*?\*/", "", f.read(), flags=re.S)
    found = []
    for block in _BLOCK.finditer(text):
        ns, lib = block.groups()
        if ns == "TORCH_EXTENSION_NAME":
            _failed(source, "a TORCH_EXTENSION_NAME namespace")
        depth, end = 1, block.end()  # to the brace that closes the block
        while depth and end < len(text):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            end += 1
        body = text[block.end():end]
        for call in re.finditer(rf"\b{lib}\s*\.\s*def\s*\(", body):
            literals = _LITERALS.match(body, call.end())
            schema = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', literals.group(1))) if literals else ""
            if "(" not in schema:
                _failed(source, f"{ns}'s m.def({schema or '...'!r}, ...) infers its schema from C++")
            found.append((ns, schema, source))
    return found


def _register(ns: str, schema: str, source: str) -> None:
    op = schema[:schema.index("(")].strip()
    parsed = torch._C.parse_schema(f"{ns}::{schema}")
    if parsed.returns:
        _failed(source, f"{ns}::{op} returns tensors")
    if f"{ns}::{op}" in torch._library.custom_ops.OPDEFS:
        return  # built already in this process
    mutated = tuple(a.name for a in parsed.arguments if a.alias_info is not None and a.alias_info.is_write)

    def body(*args, **kwargs):
        return None  # computes nothing: what it writes is never computed

    body.__signature__ = inspect.Signature(
        [inspect.Parameter(a.name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for a in parsed.arguments])
    kernel = clock.kernel_fn(f"{ns}.{op}", cost=triton_launch.traffic, writes_args=mutated)(body)
    torch.library.custom_op(f"{ns}::{op}", kernel, mutates_args=mutated, schema=schema[schema.index("("):])


def install() -> None:
    from torch.utils import cpp_extension

    real = cpp_extension.load
    signature = inspect.signature(real)

    def load(*args, **kwargs):
        a = signature.bind(*args, **kwargs)
        a.apply_defaults()
        sources = [str(s) for s in a.arguments["sources"]] if not isinstance(
            a.arguments["sources"], str) else [a.arguments["sources"]]
        if not (a.arguments["with_cuda"] or any(s.endswith((".cu", ".cuh")) for s in sources)):
            return real(*args, **kwargs)
        if a.arguments["is_python_module"] or a.arguments["is_standalone"]:
            _failed(a.arguments["name"], "a CUDA extension built as a Python module or an executable")
        defs = [d for s in sources if s.endswith((".cu", ".cuh", ".cpp", ".cc", ".cxx")) for d in _defs(s)]
        if not defs:
            _failed(a.arguments["name"], "no TORCH_LIBRARY op")
        for ns, schema, source in defs:
            _register(ns, schema, source)
        return None

    cpp_extension.load = load
