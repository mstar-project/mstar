"""The Omnilingual ASR character vocabulary and CTC decoding.

The tokenizer is a SentencePiece model whose pieces are single characters
(the space is a literal ``' '`` piece, not ``▁``), plus ``<s>``, ``<pad>``,
``</s>`` (control) and ``<unk>``. The reference decodes with fairseq2's
SentencePiece decoder, which concatenates pieces, renders control pieces as
nothing and ``<unk>`` as `` ⁇ ``, and adds or strips no whitespace. So a
lookup table reproduces it, and the ``.model`` file is read here directly
(its protobuf wire format, pieces and types only) rather than through the
``sentencepiece`` package.

CTC decoding is the reference's ``_apply_model_wav2vec2asr``: argmax per
frame, drop each frame equal to the one before it, then decode, which drops
the blank (a control piece) along with the other specials.
"""

from __future__ import annotations

from pathlib import Path

import torch

# sentencepiece ModelProto.SentencePiece.Type
_NORMAL, _UNKNOWN, _CONTROL, _USER_DEFINED, _UNUSED, _BYTE = 1, 2, 3, 4, 5, 6
UNK_SURFACE = " \u2047 "


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes):
    """``(field number, wire type, value)`` for each field of one message."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = buf[i : i + 8], i + 8
        elif wire == 2:
            size, i = _varint(buf, i)
            value, i = buf[i : i + size], i + size
        elif wire == 5:
            value, i = buf[i : i + 4], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def read_sentencepiece_pieces(data: bytes) -> list[tuple[str, int]]:
    """``(piece, type)`` per id from a serialized SentencePiece ``ModelProto``."""
    pieces = []
    for field, wire, value in _fields(data):
        if field == 1 and wire == 2:  # repeated SentencePiece pieces = 1
            piece, piece_type = "", _NORMAL
            for f, _, v in _fields(value):
                if f == 1:
                    piece = v.decode("utf-8")
                elif f == 3:
                    piece_type = v
            pieces.append((piece, piece_type))
    if not pieces:
        raise ValueError("no pieces found; not a SentencePiece model file")
    return pieces


class CharTokenizer:
    """id -> text for a single-character SentencePiece vocabulary."""

    def __init__(self, pieces: list[tuple[str, int]]):
        surfaces = []
        for piece, piece_type in pieces:
            if piece_type == _CONTROL:
                surfaces.append("")
            elif piece_type == _UNKNOWN:
                surfaces.append(UNK_SURFACE)
            elif piece_type == _BYTE or "\u2581" in piece:
                # neither occurs in the Omnilingual vocabularies; a model that
                # has them needs SentencePiece's own decoder
                raise ValueError(f"unsupported SentencePiece piece {piece!r} (type {piece_type})")
            else:
                surfaces.append(piece)
        self.surfaces = surfaces

    @classmethod
    def from_file(cls, path: str | Path) -> CharTokenizer:
        return cls(read_sentencepiece_pieces(Path(path).read_bytes()))

    @property
    def vocab_size(self) -> int:
        return len(self.surfaces)

    def decode(self, ids) -> str:
        surfaces = self.surfaces
        return "".join(surfaces[i] for i in ids)


def ctc_collapse(ids: torch.Tensor) -> torch.Tensor:
    """Keep the first frame and every frame that differs from the one before
    it (the blank is dropped later, at decode)."""
    if ids.numel() == 0:
        return ids
    keep = torch.ones_like(ids, dtype=torch.bool)
    keep[1:] = ids[1:] != ids[:-1]
    return ids[keep]


def ctc_greedy_ids(logits: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Collapsed argmax ids over a request's ``num_frames`` valid frames."""
    return ctc_collapse(logits[:num_frames].argmax(dim=-1))
