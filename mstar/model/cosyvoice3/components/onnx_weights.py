"""Weights of the ONNX-only front-end models, loaded into PyTorch modules.

CosyVoice3 ships its speech tokenizer (``speech_tokenizer_v3.onnx``) and its
speaker encoder (``campplus.onnx``) as ONNX graphs only. Both are architectures
Chatterbox already implements: the tokenizer is ``S3Tokenizer`` with twelve
blocks, the speaker encoder is ``CAMPPlus``. The exporter renamed most weights
(``onnx::MatMul_3998``), so they are matched by the order the graph uses them
against the order the PyTorch module calls its layers, with every shape
checked. CAMPPlus was exported with each batch norm that follows a convolution
folded into it; such a convolution loads its folded weight and its batch norm
becomes the identity plus the folded bias.

``read_onnx`` decodes just the protobuf fields needed (node inputs and op
types, initializers, ``Constant`` values, ``epsilon``), so no ``onnx``
dependency is needed.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

# TensorProto.DataType -> numpy dtype
_DTYPES = {1: np.float32, 6: np.int32, 7: np.int64, 10: np.float16, 11: np.float64}


@dataclass
class OnnxNode:
    op_type: str
    inputs: list[str]
    outputs: list[str]
    floats: dict[str, float] = field(default_factory=dict)


@dataclass
class OnnxGraph:
    nodes: list[OnnxNode]
    # initializers and Constant-node values, by name
    tensors: dict[str, np.ndarray]
    initializers: frozenset[str]


def _varint(buf: memoryview, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7


def _fields(buf: memoryview, start: int, end: int):
    """``(field_number, wire_type, value)`` of one message; length-delimited
    values come back as ``(start, end)`` offsets into ``buf``."""
    pos = start
    while pos < end:
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value, pos = bytes(buf[pos : pos + 8]), pos + 8
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value, pos = (pos, pos + length), pos + length
        elif wire == 5:
            value, pos = bytes(buf[pos : pos + 4]), pos + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def _string(buf: memoryview, span: tuple[int, int]) -> str:
    return bytes(buf[span[0] : span[1]]).decode("utf-8")


def _packed_varints(buf: memoryview, span: tuple[int, int]) -> list[int]:
    out, pos = [], span[0]
    while pos < span[1]:
        value, pos = _varint(buf, pos)
        out.append(value)
    return out


def _tensor(buf: memoryview, span: tuple[int, int]) -> tuple[str, np.ndarray]:
    dims: list[int] = []
    dtype = name = None
    raw = floats = None
    for number, wire, value in _fields(buf, *span):
        if number == 1:
            dims.extend(_packed_varints(buf, value) if wire == 2 else [value])
        elif number == 2:
            dtype = value
        elif number == 4 and wire == 2:
            floats = np.frombuffer(buf[value[0] : value[1]], dtype=np.float32)
        elif number == 8:
            name = _string(buf, value)
        elif number == 9:
            raw = buf[value[0] : value[1]]
        elif number in (13, 14):
            raise ValueError("ONNX tensors with external data are not supported")
    if dtype not in _DTYPES:
        raise ValueError(f"ONNX tensor {name!r} has unsupported data type {dtype}")
    if raw is not None:
        array = np.frombuffer(raw, dtype=_DTYPES[dtype])
    elif floats is not None:
        array = floats
    else:
        array = np.zeros(0, dtype=_DTYPES[dtype])
    return name or "", array.reshape(dims).copy()


def _node(buf: memoryview, span: tuple[int, int]) -> tuple[OnnxNode, list[tuple[str, np.ndarray]]]:
    node = OnnxNode(op_type="", inputs=[], outputs=[])
    constants = []
    for number, _wire, value in _fields(buf, *span):
        if number == 1:
            node.inputs.append(_string(buf, value))
        elif number == 2:
            node.outputs.append(_string(buf, value))
        elif number == 4:
            node.op_type = _string(buf, value)
        elif number == 5:
            attr_name, attr_float, attr_tensor = "", None, None
            for a_number, _a_wire, a_value in _fields(buf, *value):
                if a_number == 1:
                    attr_name = _string(buf, a_value)
                elif a_number == 2:
                    attr_float = struct.unpack("<f", a_value)[0]
                elif a_number == 5:
                    attr_tensor = _tensor(buf, a_value)[1]
            if attr_float is not None:
                node.floats[attr_name] = attr_float
            if attr_tensor is not None and attr_name == "value":
                constants.append(attr_tensor)
    if node.op_type == "Constant" and constants:
        return node, [(node.outputs[0], constants[0])]
    return node, []


def read_onnx(path: str | Path) -> OnnxGraph:
    """The nodes, initializers and constants of an ONNX model file."""
    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        buf = memoryview(mapped)
        try:
            graph_span = next(v for n, w, v in _fields(buf, 0, len(buf)) if n == 7 and w == 2)
            nodes, tensors, initializers = [], {}, set()
            for number, _wire, value in _fields(buf, *graph_span):
                if number == 1:
                    node, constants = _node(buf, value)
                    nodes.append(node)
                    tensors.update(constants)
                elif number == 5:
                    name, array = _tensor(buf, value)
                    tensors[name] = array
                    initializers.add(name)
        finally:
            buf.release()
    return OnnxGraph(nodes=nodes, tensors=tensors, initializers=frozenset(initializers))


def _call_order(module: nn.Module, kinds: tuple[type, ...] | None, run) -> list[nn.Module]:
    """Modules (with parameters, or of ``kinds``) in the order ``run()`` calls them."""
    calls: list[nn.Module] = []
    hooks = [
        sub.register_forward_pre_hook(lambda mod, *_: calls.append(mod))
        for sub in module.modules()
        if (isinstance(sub, kinds) if kinds else any(True for _ in sub.parameters(recurse=False)))
    ]
    try:
        with torch.no_grad():
            run()
    finally:
        for hook in hooks:
            hook.remove()
    return calls


@torch.no_grad()
def load_speech_tokenizer(tokenizer: nn.Module, path: str | Path) -> int:
    """``speech_tokenizer_v3.onnx`` -> Chatterbox's ``S3Tokenizer``; returns the
    number of tensors loaded. Linear weights are stored transposed (MatMul)."""
    graph = read_onnx(path)
    uses: list[tuple[str, str]] = []
    seen: set[str] = set()
    for node in graph.nodes:
        for name in node.inputs:
            if name in graph.initializers and name not in seen:
                seen.add(name)
                uses.append((node.op_type, name))
    device = next(tokenizer.parameters()).device
    calls = _call_order(tokenizer, None, lambda: tokenizer(torch.zeros(16000, device=device)))
    params = [p for mod in calls for p in mod.parameters(recurse=False)]
    if len(params) != len(uses) or len(params) != len(list(tokenizer.parameters())):
        raise RuntimeError(
            f"speech tokenizer: the ONNX graph uses {len(uses)} weights, the module "
            f"calls {len(params)} of its {len(list(tokenizer.parameters()))} parameters"
        )
    for param, (op, name) in zip(params, uses, strict=True):
        value = torch.from_numpy(graph.tensors[name])
        if op == "MatMul":
            value = value.t()
        if tuple(value.shape) != tuple(param.shape):
            raise RuntimeError(
                f"speech tokenizer: {name} has shape {tuple(value.shape)}, expected {tuple(param.shape)}"
            )
        param.copy_(value)
    return len(params)


_CONV = (nn.Conv1d, nn.Conv2d)
_BN = (nn.BatchNorm1d, nn.BatchNorm2d)


@torch.no_grad()
def load_campplus(campplus: nn.Module, path: str | Path, feat_dim: int = 80) -> int:
    """``campplus.onnx`` -> Chatterbox's ``CAMPPlus``; returns the number of
    convolutions and batch norms loaded."""
    graph = read_onnx(path)
    items = []
    for node in graph.nodes:
        if node.op_type == "Conv":
            weight = graph.tensors[node.inputs[1]]
            bias = graph.tensors.get(node.inputs[2]) if len(node.inputs) > 2 else None
            items.append(("conv", weight, bias, None))
        elif node.op_type == "BatchNormalization":
            stats = [graph.tensors.get(name) for name in node.inputs[1:5]]
            items.append(("bn", stats, None, node.floats.get("epsilon", 1e-5)))
    device = next(campplus.parameters()).device
    calls = _call_order(campplus, _CONV + _BN, lambda: campplus(torch.zeros(1, 200, feat_dim, device=device)))
    i = j = 0
    while i < len(calls):
        if j >= len(items):
            raise RuntimeError("campplus: the ONNX graph has fewer layers than the module")
        mod, (kind, a, b, eps) = calls[i], items[j]
        if isinstance(mod, _CONV):
            if kind != "conv" or tuple(a.shape) != tuple(mod.weight.shape):
                raise RuntimeError(f"campplus: layer {i} ({mod}) does not match ONNX node {j} ({kind})")
            mod.weight.copy_(torch.from_numpy(a))
            if mod.bias is not None:
                mod.bias.copy_(torch.from_numpy(b))
            elif b is not None:
                bn = calls[i + 1] if i + 1 < len(calls) else None
                if not isinstance(bn, _BN) or not bn.affine:
                    raise RuntimeError(f"campplus: ONNX folded a bias into layer {i} but no batch norm follows it")
                bn.running_mean.zero_()
                bn.running_var.fill_(1.0 - bn.eps)
                bn.weight.fill_(1.0)
                bn.bias.copy_(torch.from_numpy(b))
                i += 1
        else:
            if kind != "bn" or abs(eps - mod.eps) > 1e-9:
                raise RuntimeError(f"campplus: layer {i} ({mod}) does not match ONNX node {j} ({kind})")
            scale, bias, mean, var = a
            mod.running_mean.copy_(torch.from_numpy(mean))
            mod.running_var.copy_(torch.from_numpy(var))
            if mod.affine:
                mod.weight.copy_(torch.from_numpy(scale))
                mod.bias.copy_(torch.from_numpy(bias))
            elif not (np.allclose(scale, 1.0) and np.allclose(bias, 0.0)):
                raise RuntimeError(f"campplus: layer {i} has no affine part but the ONNX one scales")
        i += 1
        j += 1
    if j != len(items):
        raise RuntimeError(f"campplus: {len(items) - j} ONNX layers were left over")
    return j
