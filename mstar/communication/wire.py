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
- A value whose declared type says nothing useful (bare ``dict``, ``Any``,
  ``model_kwargs``) goes as plain msgpack only when that round-trips exactly
  -- scalars, lists, str-keyed dicts -- and is pickled into an ExtType
  otherwise. Such fields hold whatever a model or a client put there, so
  converting them by hand is lossy (tuples come back as lists, int keys as
  pairs) or raises mid-send; pickle is what they crossed the wire as before.
  A Rust sender only ever splices these as opaque, already-encoded bytes.
"""
import dataclasses
import enum
import importlib
import logging
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

#: cls -> ((field, encode, decode, omit_if_none), ...); built on first use.
_PLANS: dict[type, tuple] = {}


logger = logging.getLogger(__name__)

#: Tag prefix for a dataclass reached through a loosely-typed field and never
#: explicitly registered. Self-describing so the decoder can find it without
#: maintenance -- the curated tags in wire_types.py stay short.
AUTO_PREFIX = "auto:"


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


#: msgpack ext code for a pickled value from a loosely-typed field.
PICKLED_EXT = 80


def _pickled(value) -> msgpack.ExtType:
    return msgpack.ExtType(PICKLED_EXT, pickle.dumps(value))


# -- plan compilation -------------------------------------------------------

def _plan(cls) -> tuple:
    plan = _PLANS.get(cls)
    if plan is None:
        hints = typing.get_type_hints(cls)
        plan = tuple(
            (
                f.name,
                _encoder(hints.get(f.name, Any)),
                _decoder(hints.get(f.name, Any)),
                # Omitting None keeps frames small, but only a field with a
                # default can be reconstructed from its absence. A REQUIRED
                # field that is legitimately None has to go on the wire.
                f.default is not dataclasses.MISSING
                or f.default_factory is not dataclasses.MISSING,
            )
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


def _checked(kinds: tuple[type, ...], name: str):
    """A plain-type decoder that refuses a value of the wrong kind. Only used
    for union branches: see ``_decoder``'s ``strict``."""
    def dec(v):
        if v is not None and not isinstance(v, kinds):
            raise WireError(f"expected {name}, got {type(v).__name__}")
        return v
    return dec


# What a union branch of each plain type accepts off the wire. Lenient where
# Python is: an int arrives for a float field, and bool is an int.
_PLAIN_CHECKS = {
    int: _checked((int,), "int"),
    float: _checked((int, float), "float"),
    bool: _checked((bool,), "bool"),
    str: _checked((str,), "str"),
    bytes: _checked((bytes,), "bytes"),
}


def _array(v):
    if not isinstance(v, list):
        raise WireError(f"expected an array, got {type(v).__name__}")
    return v


def _decoder(hint, strict: bool = False):
    """A closure that decodes one value of declared type ``hint``.

    ``strict`` is for union branches, which are tried in order until one does
    not raise. Unchecked, a plain type passes anything through and a
    container iterates whatever it is handed, so the first such branch
    claimed every value: ``int | tuple[str, str]`` decoded a tuple as a list
    (and as a dict key, raised unhashable), and ``tuple[str, str]`` decoded
    the string "ab" as ('a', 'b'). Strict branches check the value's kind
    first. Outside unions nothing is checked, so a field that round-tripped
    before still does.
    """
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    arr = _array if strict else _identity

    if origin is typing.Union or origin is type(int | str):
        branches = [a for a in args if a is not type(None)]
        if len(branches) == 1:
            inner = _decoder(branches[0], strict)
            return lambda v: None if v is None else inner(v)
        decs = [_decoder(b, strict=True) for b in branches]

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
        inner = _decoder(args[0], strict) if args else _dynamic_decode
        return lambda v: [inner(x) for x in arr(v)]
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            inner = _decoder(args[0], strict)
            return lambda v: tuple(inner(x) for x in arr(v))
        decs = [_decoder(a, strict) for a in args]
        if strict:
            def dec_tuple(v):
                if len(_array(v)) != len(decs):
                    raise WireError(
                        f"expected {len(decs)} elements, got {len(v)}"
                    )
                return tuple(d(x) for d, x in zip(decs, v, strict=True))
            return dec_tuple
        return lambda v: tuple(d(x) for d, x in zip(decs, v, strict=False))
    if origin in (set, frozenset):
        inner = _decoder(args[0], strict) if args else _dynamic_decode
        return lambda v: origin(inner(x) for x in arr(v))

    if origin is dict:
        kt, vt = args if args else (Any, Any)
        vdec = _decoder(vt, strict)
        if kt is str:
            def dec_str_map(v):
                if strict and not isinstance(v, dict):
                    raise WireError(f"expected a map, got {type(v).__name__}")
                return {k: vdec(x) for k, x in v.items()}
            return dec_str_map
        kdec = _decoder(kt, strict)
        return lambda v: {kdec(k): vdec(x) for k, x in arr(v)}

    if hint is torch.dtype:
        return _dtype_from_name

    if isinstance(hint, type):
        if issubclass(hint, enum.Enum):
            return hint
        if _is_tagged(hint):
            return _decode_tagged
        if dataclasses.is_dataclass(hint):
            return lambda v: _decode_dataclass(v, hint)
        if strict and hint in _PLAIN_CHECKS:
            return _PLAIN_CHECKS[hint]

    return _dynamic_decode


# -- dataclass encode / decode ----------------------------------------------

def _encode_dataclass(obj) -> dict:
    out = {}
    for name, enc, _, omit_if_none in _plan(type(obj)):
        v = getattr(obj, name)
        if v is None:
            if not omit_if_none:
                out[name] = None
            continue
        out[name] = enc(v)
    return out


def _decode_dataclass(raw: dict, cls: type):
    kwargs = {}
    for name, _, dec, _omit in _plan(cls):
        if name in raw:
            v = raw[name]
            kwargs[name] = None if v is None else dec(v)
    return cls(**kwargs)


def _encode_tagged(obj):
    cls = type(obj)
    tag = _TYPE_TO_TAG.get(cls)
    if tag is None:
        # Reached through a loosely-typed field (bare dict, Any, an abstract
        # base whose subclass lives in a module wire_types does not import).
        # Enumerating those by hand is a standing trap -- one gets added and
        # the failure shows up at runtime on whichever path first carries it --
        # so fall back to a self-describing tag instead of raising.
        if not cls.__module__.startswith("mstar."):
            raise WireError(
                f"{cls.__module__}.{cls.__qualname__} is not an mstar type; "
                f"the wire only resolves mstar classes automatically"
            )
        tag = f"{AUTO_PREFIX}{cls.__module__}:{cls.__qualname__}"
        register(tag, cls)
        logger.debug("auto-registered %s for the wire as %r", cls.__name__, tag)
    return [tag, _encode_dataclass(obj)]


def _resolve_auto_tag(tag: str) -> type:
    module_name, _, qualname = tag.removeprefix(AUTO_PREFIX).partition(":")
    if not module_name.startswith("mstar."):
        raise WireError(f"refusing to import {module_name!r} from the wire")
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not dataclasses.is_dataclass(obj):
        raise WireError(f"wire tag {tag!r} does not name a dataclass")
    return obj


def _decode_tagged(raw):
    tag, payload = raw
    cls = _TAG_TO_TYPE.get(tag)
    if cls is None:
        if not tag.startswith(AUTO_PREFIX):
            raise WireError(f"unknown wire tag {tag!r}")
        cls = _resolve_auto_tag(tag)
        register(tag, cls)
    return _decode_dataclass(payload, cls)


def _dynamic_decode(raw):
    """Mirror of ``_dynamic``: plain values come back as they are, and a
    pickled one is unpickled."""
    if isinstance(raw, msgpack.ExtType):
        if raw.code != PICKLED_EXT:
            raise WireError(f"unknown msgpack ext code {raw.code}")
        return pickle.loads(raw.data)
    if isinstance(raw, list):
        return [_dynamic_decode(v) for v in raw]
    if isinstance(raw, dict):
        return {k: _dynamic_decode(v) for k, v in raw.items()}
    return raw


def _is_plain(value) -> bool:
    """Whether msgpack alone round-trips ``value`` exactly."""
    if value is None or type(value) in (str, int, float, bool, bytes):
        return True
    if type(value) is list:
        return all(_is_plain(v) for v in value)
    if type(value) is dict:
        return all(
            type(k) is str and _is_plain(v) for k, v in value.items()
        )
    return False


def _dynamic(value):
    """A value whose declared type says nothing useful (bare ``dict``, ``Any``).
    Plain msgpack when that is exact, pickled otherwise -- see the module
    docstring."""
    if _is_plain(value):
        return value
    return _pickled(value)


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
