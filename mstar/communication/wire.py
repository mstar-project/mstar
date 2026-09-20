"""Typed msgpack encoding for everything that crosses a communicator edge.
Msgpack is readable by both Python and Rust, allowing, e.g., the sender
to be fully Rust and the receiver to be Python.

The encoding is **type-driven**: the decoder knows the declared type of every
field, so the wire carries values, not type tags, except where a field is
polymorphic (see ``_POLYMORPHIC``).

Field plans are **compiled once per class** into a tuple of closures; resulving
annotations per message is about 17x slower than pickle.

Notes:
- ``dict`` keyed by a tuple (``graph_timings``) becomes a list of entries.
- ``set`` becomes a list and is rebuilt on decode.
- ``PublishedInfo`` / ``ResourceReqConfig`` / ``MessageBody`` are abstract, so
  those fields carry ``[tag, payload]``.
"""
import dataclasses
import enum
import pickle
import typing
from typing import Any

import msgpack
import torch

# Registry for the abstract field types. A subclass has to be registered
# before it can cross the wire; registration is by stable name, not module
# path, so moving a class does not break the format.
_TAG_TO_TYPE: dict[str, type] = {}
_TYPE_TO_TAG: dict[type, str] = {}
_POLYMORPHIC: tuple[type, ...] = ()

#: cls -> ((field, encode, decode), ...); built on first use.
_PLANS: dict[type, tuple] = {}


class WireError(Exception):
    pass


def wire_type(tag: str):
    """Register a concrete class under ``tag`` for polymorphic fields."""
    def deco(cls):
        if tag in _TAG_TO_TYPE and _TAG_TO_TYPE[tag] is not cls:
            raise ValueError(f"wire tag {tag!r} already registered to {_TAG_TO_TYPE[tag]}")
        _TAG_TO_TYPE[tag] = cls
        _TYPE_TO_TAG[cls] = tag
        return cls
    return deco


def register(tag: str, cls: type) -> None:
    """Register a class whose definition we do not own."""
    wire_type(tag)(cls)


def _set_polymorphic(*bases: type) -> None:
    global _POLYMORPHIC
    _POLYMORPHIC = bases
    _PLANS.clear()


# -- dtype ------------------------------------------------------------------

def _dtype_name(dt):
    return str(dt).removeprefix("torch.") if isinstance(dt, torch.dtype) else dt


def _dtype_from_name(name):
    if not isinstance(name, str):
        return name
    dt = getattr(torch, name, None)
    return dt if isinstance(dt, torch.dtype) else name


def _identity(v):
    return v


# -- plan compilation -------------------------------------------------------

def _plan(cls) -> tuple:
    plan = _PLANS.get(cls)
    if plan is None:
        hints = typing.get_type_hints(cls)
        plan = tuple(
            (f.name, _encoder(hints.get(f.name, Any)), _decoder(hints.get(f.name, Any)))
            for f in dataclasses.fields(cls)
        )
        _PLANS[cls] = plan
    return plan


def _is_tagged(hint) -> bool:
    return isinstance(hint, type) and (
        hint in _POLYMORPHIC
        or (not dataclasses.is_dataclass(hint) and any(
            issubclass(hint, b) if isinstance(b, type) else False for b in _POLYMORPHIC))
    )


def _encoder(hint):
    """A closure that encodes one value of declared type ``hint``."""
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if origin is typing.Union or origin is type(int | str):
        branches = [a for a in args if a is not type(None)]
        if len(branches) == 1:
            inner = _encoder(branches[0])
            return lambda v: None if v is None else inner(v)
        encs = [(b, _encoder(b)) for b in branches]

        def enc_union(v):
            if v is None:
                return None
            for b, e in encs:
                bo = typing.get_origin(b) or b
                if isinstance(bo, type) and isinstance(v, bo):
                    return e(v)
            return _dynamic(v)
        return enc_union

    if origin in (list, tuple, set, frozenset):
        inner = _encoder(args[0]) if args else _dynamic
        return lambda v: [inner(x) for x in v]

    if origin is dict:
        kt, vt = args if args else (Any, Any)
        venc = _encoder(vt)
        if kt is str:
            return lambda v: {k: venc(x) for k, x in v.items()}
        kenc = _encoder(kt)
        return lambda v: [[kenc(k), venc(x)] for k, x in v.items()]

    if hint is torch.dtype:
        return _dtype_name

    if isinstance(hint, type):
        if issubclass(hint, enum.Enum):
            return lambda v: v.value
        if _is_tagged(hint):
            return _encode_tagged
        if dataclasses.is_dataclass(hint):
            return _encode_dataclass
        if hint in (str, int, float, bool, bytes):
            return _identity

    return _dynamic


def _decoder(hint):
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if origin is typing.Union or origin is type(int | str):
        branches = [a for a in args if a is not type(None)]
        if len(branches) == 1:
            inner = _decoder(branches[0])
            return lambda v: None if v is None else inner(v)
        decs = [_decoder(b) for b in branches]

        def dec_union(v):
            if v is None:
                return None
            for d in decs:
                try:
                    return d(v)
                except Exception:  # noqa: BLE001 - try the next branch
                    continue
            return v
        return dec_union

    if origin is list:
        inner = _decoder(args[0]) if args else _identity
        return lambda v: [inner(x) for x in v]
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            inner = _decoder(args[0])
            return lambda v: tuple(inner(x) for x in v)
        decs = [_decoder(a) for a in args]
        return lambda v: tuple(d(x) for d, x in zip(decs, v, strict=False))
    if origin in (set, frozenset):
        inner = _decoder(args[0]) if args else _identity
        return lambda v: origin(inner(x) for x in v)

    if origin is dict:
        kt, vt = args if args else (Any, Any)
        vdec = _decoder(vt)
        if kt is str:
            return lambda v: {k: vdec(x) for k, x in v.items()}
        kdec = _decoder(kt)
        return lambda v: {kdec(k): vdec(x) for k, x in v}

    if hint is torch.dtype:
        return _dtype_from_name

    if isinstance(hint, type):
        if issubclass(hint, enum.Enum):
            return hint
        if _is_tagged(hint):
            return _decode_tagged
        if dataclasses.is_dataclass(hint):
            return lambda v: _decode_dataclass(v, hint)

    return _identity


# -- dataclass encode / decode ----------------------------------------------

def _encode_dataclass(obj) -> dict:
    out = {}
    for name, enc, _ in _plan(type(obj)):
        v = getattr(obj, name)
        if v is not None:  # absent means default; keeps frames small
            out[name] = enc(v)
    return out


def _decode_dataclass(raw: dict, cls: type):
    kwargs = {}
    for name, _, dec in _plan(cls):
        if name in raw:
            kwargs[name] = dec(raw[name])
    return cls(**kwargs)


def _encode_tagged(obj):
    tag = _TYPE_TO_TAG.get(type(obj))
    if tag is None:
        raise WireError(
            f"{type(obj).__name__} crosses the wire under an abstract type "
            f"but is not registered; call wire.register(tag, cls)"
        )
    return [tag, _encode_dataclass(obj)]


def _decode_tagged(raw):
    tag, payload = raw
    cls = _TAG_TO_TYPE.get(tag)
    if cls is None:
        raise WireError(f"unknown wire tag {tag!r}")
    return _decode_dataclass(payload, cls)


def _dynamic(value):
    """A value whose declared type says nothing useful (bare ``dict``, ``Any``).
    Allow msgpack natives and anything losslessly reducible."""
    if isinstance(value, (str, int, float, bool, bytes)) or value is None:
        return value
    if isinstance(value, torch.dtype):
        return _dtype_name(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_dynamic(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: _dynamic(v) for k, v in value.items()}
        return [[_dynamic(k), _dynamic(v)] for k, v in value.items()]
    if dataclasses.is_dataclass(value):
        return _encode_tagged(value)
    raise WireError(f"cannot encode {type(value).__name__} without a declared type")


# -- public -----------------------------------------------------------------

#: Frame tag for a payload that is not a registered message. The codec has to
#: be total — a communicator carries whatever a caller hands it, and a codec
#: that raises on some payloads turns into a silent hang at the receiver
#: (the send dies, the peer waits forever). Registered messages, which are
#: the only ones a Rust endpoint decodes, never take this path.
OPAQUE = "__py__"


def encode(msg) -> bytes:
    tag = _TYPE_TO_TAG.get(type(msg))
    if tag is None:
        return msgpack.packb([OPAQUE, pickle.dumps(msg)], use_bin_type=True)
    return msgpack.packb([tag, _encode_dataclass(msg)], use_bin_type=True)


def decode(data: bytes):
    tag, payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
    if tag == OPAQUE:
        return pickle.loads(payload)
    cls = _TAG_TO_TYPE.get(tag)
    if cls is None:
        raise WireError(f"unknown wire message tag {tag!r}")
    return _decode_dataclass(payload, cls)
