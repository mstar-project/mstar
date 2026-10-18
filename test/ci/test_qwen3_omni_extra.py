"""The qwen3_omni extra covers what qwen-omni-utils imports without declaring."""

import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class Qwen3OmniExtraTests(unittest.TestCase):
    def test_audioread_ships_with_qwen_omni_utils(self):
        # qwen_omni_utils.v2_5 imports audioread at module load (a video's audio
        # track); the Qwen3-Omni video loader imports that package
        extras = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
        for name in ("qwen3_omni", "all"):
            names = {req.split(";")[0].split("[")[0].split("=")[0].split(">")[0].split("<")[0].strip()
                     for req in extras[name]}
            self.assertIn("qwen-omni-utils", names, name)
            self.assertIn("audioread", names, name)


if __name__ == "__main__":
    unittest.main()
