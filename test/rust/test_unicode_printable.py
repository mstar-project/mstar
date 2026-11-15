"""Keep Rust refusal text on this interpreter's ``str.isprintable`` table.

The checked-in table came from CPython 3.12 (UCD 15.0.0); a Python upgrade
that changes printability must regenerate it before the frontend can match.
"""

import re
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "rust/server/src/unicode_printable.rs"
GENERATOR = ROOT / "scripts/gen_unicode_printable.py"


def test_rust_printability_table_matches_python():
    source = TABLE.read_text()
    ranges = [(int(low, 16), int(high, 16))
              for low, high in re.findall(r"\(0x([0-9a-f]+), 0x([0-9a-f]+)\)", source)]
    assert len(ranges) > 100
    assert all(low <= high for low, high in ranges)
    assert all(ranges[i][1] + 1 < ranges[i + 1][0]
               for i in range(len(ranges) - 1))

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

    generator = runpy.run_path(str(GENERATOR))
    assert source == generator["render"](generator["unprintable_ranges"]())
