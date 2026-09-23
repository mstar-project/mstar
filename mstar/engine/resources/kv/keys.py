"""Page keys for cross-request prefix reuse.

A key commits to the whole prefix below it: the previous key, this page's token
ids, and a digest for each input in the span that is not a token. One match
confirms every page under it, and a new root leaves every old key unreachable.

The encoding is canonical because one process builds these keys and another
matches them: fixed-width little-endian ints, and a length prefix on every
variable-length field so that no two different spans encode alike.
"""

import hashlib
from collections.abc import Mapping, Sequence

# 4 bytes a token, not 8: a full page of 128 then hashes 512 bytes
_TOKEN_BYTES = 4
_LEN_BYTES = 4


def _field(payload: bytes) -> bytes:
    return len(payload).to_bytes(_LEN_BYTES, "little") + payload


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
    prev: bytes, tokens: Sequence[int], digests: Sequence[bytes] = (),
) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(_field(prev))
    hasher.update(_field(b"".join(
        token.to_bytes(_TOKEN_BYTES, "little") for token in tokens
    )))
    hasher.update(_field(b"".join(_field(digest) for digest in digests)))
    return hasher.digest()


def chain(
    pages_of_tokens: Sequence[Sequence[int]],
    digests_by_page: Mapping[int, Sequence[bytes]] | None = None,
    root: bytes = b"",
) -> list[bytes]:
    """Key every page of one stream, folding each key into the next.

    ``digests_by_page`` is keyed by the page holding an item's first
    placeholder, so an item spanning pages is hashed once.
    """
    digests_by_page = digests_by_page or {}
    keys = []
    prev = root
    for index, tokens in enumerate(pages_of_tokens):
        prev = page_key(prev, tokens, digests_by_page.get(index, ()))
        keys.append(prev)
    return keys
