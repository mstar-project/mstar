"""Client request IDs and deployment-local submission IDs."""

import hashlib


def client_id_seed(request_id: str) -> int:
    """The stable 32-bit fallback seed for a client-visible request ID."""
    digest = hashlib.md5(request_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")
