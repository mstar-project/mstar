"""The alias packages (packaging/aliases/*) forward to mstar-ai: same version,
and every mstar-ai extra under the same name, so `pip install
"mstar-serve[flux2_klein]"` installs the extra instead of the bare engine."""

import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALIASES = sorted((REPO_ROOT / "packaging/aliases").glob("*/pyproject.toml"))


def project(path):
    return tomllib.loads(path.read_text())["project"]


class AliasPackagesTest(unittest.TestCase):
    def setUp(self):
        self.engine = project(REPO_ROOT / "pyproject.toml")

    def test_there_are_aliases(self):
        self.assertEqual([p.parent.name for p in ALIASES], ["mstar-project", "mstar-serve"])

    def test_each_alias_carries_mstar_ai_version(self):
        for path in ALIASES:
            with self.subTest(alias=path.parent.name):
                self.assertEqual(project(path)["version"], self.engine["version"])

    def test_each_alias_forwards_every_extra_under_its_name(self):
        expected = {name: [f"mstar-ai[{name}]"] for name in self.engine["optional-dependencies"]}
        for path in ALIASES:
            with self.subTest(alias=path.parent.name):
                self.assertEqual(project(path)["optional-dependencies"], expected)

    def test_each_alias_installs_mstar_ai_and_links_the_repo(self):
        for path in ALIASES:
            with self.subTest(alias=path.parent.name):
                alias = project(path)
                self.assertEqual(alias["dependencies"], ["mstar-ai"])
                self.assertEqual(alias["urls"]["Homepage"], "https://github.com/mstar-project/mstar")


if __name__ == "__main__":
    unittest.main()
