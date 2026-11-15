"""Check both frontends' pinned refusal-text tables.

The checked-in tables came from CPython 3.12 (UCD 15.0.0). The full comparison
against ``repr`` runs only on that source UCD; other versions read the same data.
"""

import re
import runpy
import sys
import unicodedata
from pathlib import Path

import pytest

from mstar.api_server.pinned_repr import pinned_repr
from mstar.api_server.unicode_printable_ranges import UNPRINTABLE_RANGES

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "rust/server/src/unicode_printable.rs"
GENERATOR = ROOT / "scripts/gen_unicode_printable.py"
SOURCE_UCD_VERSION = "15.0.0"

if not TABLE.is_file() or not GENERATOR.is_file():
    pytest.skip("needs the checkout's Rust source (wheel tests copy test/ alone)", allow_module_level=True)


def test_rust_printability_table_matches_source_python():
    source = TABLE.read_text()
    ranges = [(int(low, 16), int(high, 16))
              for low, high in re.findall(r"\(0x([0-9a-f]+), 0x([0-9a-f]+)\)", source)]
    assert len(ranges) > 100
    assert all(low <= high for low, high in ranges)
    assert all(ranges[i][1] + 1 < ranges[i + 1][0]
               for i in range(len(ranges) - 1))
    assert f"UCD {SOURCE_UCD_VERSION}" in source
    assert ranges == list(UNPRINTABLE_RANGES)

    generator = runpy.run_path(str(GENERATOR))
    python_table = ROOT / "mstar/api_server/unicode_printable_ranges.py"
    assert python_table.read_text() == generator["render_python"](ranges)

    # Only the source UCD can reproduce the checked-in table byte for byte.
    if unicodedata.unidata_version != SOURCE_UCD_VERSION:
        return

    index = 0
    mismatches = []
    for code in range(sys.maxunicode + 1):
        while index < len(ranges) and code > ranges[index][1]:
            index += 1
        rust_unprintable = index < len(ranges) and ranges[index][0] <= code
        if rust_unprintable != (not chr(code).isprintable()):
            mismatches.append(code)
    assert not mismatches, (
        f"Rust printability differs at {[hex(c) for c in mismatches[:8]]}; "
        "regenerate with `python scripts/gen_unicode_printable.py`"
    )

    assert source == generator["render"](generator["unprintable_ranges"]())
    assert all(pinned_repr(chr(code)) == repr(chr(code)) for code in range(sys.maxunicode + 1))


def test_an_unassigned_codepoint_stays_pinned_across_python_versions():
    source = TABLE.read_text()
    ranges = [(int(low, 16), int(high, 16))
              for low, high in re.findall(r"\(0x([0-9a-f]+), 0x([0-9a-f]+)\)", source)]
    code = 0x2FFC
    assert any(low <= code <= high for low, high in ranges)
    if unicodedata.unidata_version == "15.1.0":
        assert chr(code).isprintable()
    assert pinned_repr(chr(code)) == "'\\u2ffc'"
