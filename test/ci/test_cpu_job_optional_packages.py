"""The CPU Core job installs the optional packages its tests skip without."""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def install_commands():
    """The commands of the CPU Core job's install step, continuation lines joined."""
    text = CI_YML.read_text()
    job = text[text.index("\n  cpu-tests:\n"):]
    step = job[job.index("- name: Install mstar with CPU PyTorch\n"):]
    body = step[step.index("run: |\n") + len("run: |\n"):]
    body = re.split(r"\n\s*- name:", body, maxsplit=1)[0].replace("\\\n", " ")
    return [" ".join(line.split()) for line in body.splitlines()
            if line.strip() and not line.strip().startswith("#")]


class CpuJobOptionalPackageTests(unittest.TestCase):
    def setUp(self):
        self.commands = install_commands()
        self.pip_check = self.commands.index("python -m pip check")

    def only_install_of(self, package):
        found = [i for i, c in enumerate(self.commands) if package in c]
        self.assertEqual(len(found), 1, f"{package} installs in the CPU Core job: {found}")
        return found[0]

    def test_qwen_tts_goes_in_without_its_dependencies_after_pip_check(self):
        # qwen-tts pins transformers==4.57.3, so pip check fails once it is in
        index = self.only_install_of("qwen-tts")
        self.assertRegex(self.commands[index], r'^python -m pip install --no-deps "qwen-tts==[\d.]+"$')
        self.assertGreater(index, self.pip_check)

    def test_audioseal_goes_in_before_pip_check(self):
        index = self.only_install_of("audioseal")
        self.assertEqual(self.commands[index], 'python -m pip install "audioseal>=0.2,<0.3"')
        self.assertLess(index, self.pip_check)


if __name__ == "__main__":
    unittest.main()
