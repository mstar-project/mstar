"""A key names one prefix and no other, in whichever process computes it.

The preprocess worker chains a prompt's keys and the KV manager matches them,
in two processes that share nothing but this encoding, so a key that comes out
differently on either side turns every lookup into a miss. The expensive
failure is the other one: with sloppy field boundaries two different prefixes
would hash alike, and a request would be served another request's KV.
"""

from __future__ import annotations

import subprocess
import sys

sys.path.insert(0, ".")

from mstar.engine.resources.kv.keys import chain, page_key

PAGES = ([11, 12, 13], [14, 15, 16], [17, 18])


# ── the chain ───────────────────────────────────────────────────────────


def test_a_chain_is_a_page_by_page_fold():
    fold = []
    prev = b""
    for tokens in PAGES:
        prev = page_key(prev, tokens)
        fold.append(prev)

    assert chain(PAGES) == fold, "the chain took a different route than the fold"


def test_a_changed_token_changes_its_page_and_every_page_above_it():
    keys = chain(PAGES)

    changed = chain(([11, 12, 13], [14, 15, 99], [17, 18]))

    assert changed[0] == keys[0], "a page below the change was renamed"
    assert all(
        a != b for a, b in zip(changed[1:], keys[1:], strict=True)
    ), "a page at or above the change kept its key"


def test_a_changed_digest_changes_its_page_and_every_page_above_it():
    keys = chain(PAGES, {1: [b"image-a"]})

    changed = chain(PAGES, {1: [b"image-b"]})

    assert changed[0] == keys[0], "a page below the digest was renamed"
    assert all(
        a != b for a, b in zip(changed[1:], keys[1:], strict=True)
    ), "a page at or above the digest kept its key"


def test_a_changed_root_changes_every_key():
    keys = chain(PAGES)

    changed = chain(PAGES, root=b"another deployment")

    assert all(
        a != b for a, b in zip(changed, keys, strict=True)
    ), "a new root left an old key reachable"


# ── field boundaries ────────────────────────────────────────────────────


def test_a_digest_is_not_a_token_that_encodes_the_same_bytes():
    assert page_key(b"", [1, 2]) != page_key(b"", [1], [(2).to_bytes(4, "little")]), (
        "a token and a digest of its bytes hash alike"
    )


def test_two_digests_are_not_one_digest_of_their_bytes():
    assert page_key(b"", [], [b"ab", b"c"]) != page_key(b"", [], [b"abc"]), (
        "the split between two digests is not in the key"
    )


# ── across processes ────────────────────────────────────────────────────


def test_the_encoding_is_the_same_in_another_process():
    here = chain(PAGES, {1: [b"image-a"]}, root=b"root")

    there = subprocess.run(
        [
            sys.executable, "-c",
            "from mstar.engine.resources.kv.keys import chain;"
            "print(chain(([11,12,13],[14,15,16],[17,18]), {1: [b'image-a']}, b'root')"
            "[-1].hex())",
        ],
        capture_output=True, text=True, check=True, cwd=".",
    ).stdout.strip()

    assert there == here[-1].hex(), "the same prefix keyed differently in a second process"
