"""The encode/decode seam shared by both ZMQ transports.

Kept apart from either communicator so they can both import it, and mirroring
the Rust ``Codec`` trait::

    pub trait Codec<M> {
        fn encode(msg: &M) -> Result<Vec<u8>, CommError>;
        fn decode(bytes: &[u8]) -> Option<M>;
    }

The transport never looks inside the bytes, so moving an edge to another
encoding is a codec change, never a transport change.
"""
import logging
import os
import pickle
from collections.abc import Iterable

logger = logging.getLogger(__name__)


class Codec:
    @staticmethod
    def encode(msg) -> bytes:
        raise NotImplementedError

    @staticmethod
    def decode(data: bytes):
        raise NotImplementedError


class PickleCodec(Codec):
    """The old wire. Python-only, so an edge using it cannot terminate in a
    Rust process; kept for bisecting a wire problem."""

    encode = staticmethod(pickle.dumps)
    decode = staticmethod(pickle.loads)


class MsgpackCodec(Codec):
    """Raw msgpack, no schema: for edges whose payloads are already plain
    dicts and whose other end is not Python — the Rust ``mstar-server``
    frontend bridge. ``default=str`` is lossy, so this is NOT a general
    replacement for pickle; use :class:`WireCodec` for message dataclasses.
    """

    @staticmethod
    def encode(msg) -> bytes:
        import msgpack

        return msgpack.packb(msg, default=str)

    @staticmethod
    def decode(data: bytes):
        import msgpack

        return msgpack.unpackb(data, raw=False)


class WireCodec(Codec):
    """Typed msgpack (``mstar.communication.wire``): language-neutral and
    lossless for the message dataclasses. The default."""

    @staticmethod
    def encode(msg) -> bytes:
        from mstar.communication import wire

        return wire.encode(msg)

    @staticmethod
    def decode(data: bytes):
        from mstar.communication import wire

        return wire.decode(data)


def decode_each(codec: type[Codec], frames: Iterable[bytes], who: str) -> list:
    """Decode a drained batch one frame at a time.

    A frame that fails to decode is logged and skipped. Raising instead would
    lose every frame already drained with it -- control messages for other
    requests included -- to one bad payload.
    """
    out = []
    for frame in frames:
        try:
            out.append(codec.decode(frame))
        except Exception:
            logger.exception(
                "%s: dropping a %d-byte frame that failed to decode",
                who, len(frame),
            )
    return out


def default_codec() -> type[Codec]:
    """``MSTAR_WIRE_CODEC=pickle`` falls back to pickle.

    Both ends of an edge must agree, so this is all-or-nothing for a mesh —
    a single process set differently will not be able to talk to its peers.
    """
    if os.getenv("MSTAR_WIRE_CODEC", "msgpack").lower() == "pickle":
        return PickleCodec
    import mstar.communication.wire_types  # noqa: F401  (registers the tags)

    return WireCodec
