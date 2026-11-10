"""Install hints name a distribution we publish and extras it defines.

The import package is ``mstar``, but the distribution is ``mstar-ai``: the bare name on
PyPI is another project (pyproject.toml), so a hint that puts an extra on it installs
that project instead.
"""

import re
import subprocess
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def published() -> dict[str, set[str]]:
    """Each distribution's extras: mstar-ai and the alias packages that forward to it."""
    projects = [REPO_ROOT / "pyproject.toml", *sorted(REPO_ROOT.glob("packaging/aliases/*/pyproject.toml"))]
    extras = {}
    for path in projects:
        project = tomllib.loads(path.read_text())["project"]
        extras[project["name"]] = set(project.get("optional-dependencies", {}))
    return extras


class InstallHintTests(unittest.TestCase):
    def test_hints_name_a_published_distribution_and_its_extras(self):
        extras = published()
        names = "|".join(sorted(map(re.escape, extras), key=len, reverse=True))
        # `name[extra,...]` for a published name or the bare import name
        hint = re.compile(rf"(?<![\w.-])({names}|mstar)\[([a-z0-9_]+(?:,[a-z0-9_]+)*)\]")
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        wrong = []
        for rel in tracked:
            try:
                lines = (REPO_ROOT / rel).read_text().splitlines()
            except (UnicodeDecodeError, OSError):  # binary, or removed in the working tree
                continue
            wrong += [
                f"{rel}:{number}: {name}[{asked}]"
                for number, line in enumerate(lines, 1)
                for name, asked in hint.findall(line)
                if not set(asked.split(",")) <= extras.get(name, set())
            ]
        self.assertEqual(wrong, [], "say mstar-ai[<extra>], with an extra pyproject.toml defines")


if __name__ == "__main__":
    unittest.main()
