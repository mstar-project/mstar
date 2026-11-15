"""Quote rejected strings the same way on both frontends, independent of Python's UCD."""

from bisect import bisect_right

from mstar.api_server.unicode_printable_ranges import UNPRINTABLE_RANGES

_UNPRINTABLE_STARTS = tuple(low for low, _ in UNPRINTABLE_RANGES)


def _is_printable(c: str) -> bool:
    code = ord(c)
    after = bisect_right(_UNPRINTABLE_STARTS, code)
    return after == 0 or code > UNPRINTABLE_RANGES[after - 1][1]


def pinned_repr(value: str) -> str:
    """Python 3.12 ``repr`` for a string, with UCD 15.0.0 printability pinned."""
    quote = '"' if "'" in value and '"' not in value else "'"
    out = [quote]
    for c in value:
        if c == "\\":
            out.append("\\\\")
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        elif c == quote:
            out.extend(("\\", c))
        elif not _is_printable(c):
            code = ord(c)
            if code <= 0xFF:
                out.append(f"\\x{code:02x}")
            elif code <= 0xFFFF:
                out.append(f"\\u{code:04x}")
            else:
                out.append(f"\\U{code:08x}")
        else:
            out.append(c)
    out.append(quote)
    return "".join(out)
