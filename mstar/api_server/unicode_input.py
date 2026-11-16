"""Validate decoded client text before it reaches a model tokenizer."""

from __future__ import annotations

import re
from collections.abc import Mapping

from pydantic import BaseModel

_SURROGATE = re.compile(r"[\ud800-\udfff]")
_ERROR = "text is not valid UTF-8 (contains a lone surrogate)"


def reject_lone_surrogates(value: object) -> None:
    """Reject surrogate code points in nested JSON or request-model values.

    A valid JSON escape pair decodes to one Unicode scalar. A lone escape
    decodes to a Python ``str`` that cannot be encoded as UTF-8. Walk the
    parsed value iteratively so deep client input cannot recurse here.
    """
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            # Large inline base64 fields are ASCII, so avoid scanning them
            # a second time after the JSON parser has already read them.
            if not item.isascii() and _SURROGATE.search(item):
                raise ValueError(_ERROR)
        elif isinstance(item, BaseModel):
            if id(item) not in seen:
                seen.add(id(item))
                pending.append(vars(item))
                if item.model_extra:
                    pending.append(item.model_extra)
        elif isinstance(item, Mapping):
            if id(item) not in seen:
                seen.add(id(item))
                for key, child in item.items():
                    pending.extend((key, child))
        elif isinstance(item, (list, tuple)):
            if id(item) not in seen:
                seen.add(id(item))
                pending.extend(item)
