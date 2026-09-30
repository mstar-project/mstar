"""Page keys for cross-request prefix reuse.

A key commits to the whole prefix below it: the previous key, this page's token
ids, and where each input on the page that is not a token sits, with its digest.
One match confirms every page under it, and a new root leaves every old key
unreachable.

The encoding is canonical because one process builds these keys and another
matches them: fixed-width little-endian ints, and a length prefix on every
variable-length field so that no two different spans encode alike.
"""

import hashlib
from collections.abc import Mapping, Sequence
from typing import NamedTuple

# 4 bytes a token, not 8: a full page of 128 then hashes 512 bytes
_TOKEN_BYTES = 4
_LEN_BYTES = 4
_SLOT_BYTES = 4


def _field(payload: bytes) -> bytes:
    return len(payload).to_bytes(_LEN_BYTES, "little") + payload


class PageItem(NamedTuple):
    """An input that is not a token, as much of it as one page holds."""
    # its first slot on this page
    start: int
    # its slots on earlier pages
    offset: int
    # its slots in all
    length: int
    digest: bytes


def _item(item: PageItem) -> bytes:
    return b"".join(
        slot.to_bytes(_SLOT_BYTES, "little")
        for slot in (item.start, item.offset, item.length)
    ) + _field(item.digest)


def fingerprint(*fields: object) -> bytes:
    """SHA-256 over ``fields``, in the encoding page keys use.

    Used for process constants, so changing one makes every older key unreachable.
    """
    hasher = hashlib.sha256()
    for field in fields:
        payload = field if isinstance(field, bytes) else str(field).encode()
        hasher.update(_field(payload))
    return hasher.digest()


def page_key(
    prev: bytes, tokens: Sequence[int], items: Sequence[PageItem] = (),
) -> bytes:
    """Key one page from its parent, its ids and its items.

    The ids fill every slot the items leave, in order, so an item's ``start``
    is what tells ``[t1 t2 IMG]`` from ``[t1 IMG t2]``. Its whole ``length``
    and ``digest`` sit on every page it touches, because attention inside the
    item is not causal: a page's KV depends on the item's slots after it too.
    """
    hasher = hashlib.sha256()
    hasher.update(_field(prev))
    hasher.update(_field(b"".join(
        token.to_bytes(_TOKEN_BYTES, "little") for token in tokens
    )))
    hasher.update(_field(b"".join(_item(item) for item in items)))
    return hasher.digest()


def chain(
    pages_of_tokens: Sequence[Sequence[int]],
    items_by_page: Mapping[int, Sequence[PageItem]] | None = None,
    root: bytes = b"",
) -> list[bytes]:
    """Key every page of one stream, folding each key into the next."""
    items_by_page = items_by_page or {}
    keys = []
    prev = root
    for index, tokens in enumerate(pages_of_tokens):
        prev = page_key(prev, tokens, items_by_page.get(index, ()))
        keys.append(prev)
    return keys
