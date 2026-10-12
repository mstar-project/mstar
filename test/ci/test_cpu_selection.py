"""Check the optional Mac subset without requiring PyTorch or Docker."""

import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR = REPO_ROOT / "scripts/ci/select_cpu_tests.py"
SELECT_TESTS = runpy.run_path(str(SELECTOR))["select_tests"]


def manifest_lines(name):
    return [line.strip() for line in (REPO_ROOT / name).read_text().splitlines() if line.strip()]


class CpuSelectionTests(unittest.TestCase):
    def test_real_manifests_exclude_only_documented_incompatible_modules(self):
        core = manifest_lines("test/cpu-core.txt")
        excluded = manifest_lines("test/cpu-mac-exclude.txt")
        self.assertEqual(set(excluded), {
            "test/modular/test_waypoint_components.py",
            "test/modular/test_waypoint_dit.py",
            "test/modular/test_flex_attention_resource.py",
            "test/modular/test_waypoint_benchmark_videos.py",
        })
        selected = SELECT_TESTS(core, excluded)
        self.assertEqual(len(selected), len(core) - 4)
        for path in core:
            self.assertTrue((REPO_ROOT / path).is_file(), path)

    def test_new_core_modules_are_included_in_manifest_order(self):
        self.assertEqual(
            SELECT_TESTS(["first.py", "excluded.py", "last.py", "new.py"], ["excluded.py"]),
            ["first.py", "last.py", "new.py"],
        )

    def test_manifest_reader_ignores_blank_lines_and_surrounding_whitespace(self):
        read_manifest = runpy.run_path(str(SELECTOR))["read_manifest"]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "tests.txt"
            manifest.write_text("\n first.py \n\nlast.py\n \n")
            self.assertEqual(read_manifest(manifest), ["first.py", "last.py"])

    def test_unknown_exclusion_fails(self):
        with self.assertRaisesRegex(ValueError, "absent from core manifest: unknown.py"):
            SELECT_TESTS(["known.py"], ["unknown.py"])

    def test_duplicate_paths_fail(self):
        for core, excluded in (
            (["first.py", "first.py"], []),
            (["first.py", "last.py"], ["first.py", "first.py"]),
        ):
            with self.subTest(core=core, excluded=excluded):
                with self.assertRaisesRegex(ValueError, "Duplicate paths"):
                    SELECT_TESTS(core, excluded)

    def test_empty_selection_fails(self):
        for core, excluded in (([], []), (["only.py"], ["only.py"])):
            with self.subTest(core=core):
                with self.assertRaisesRegex(ValueError, "selection is empty"):
                    SELECT_TESTS(core, excluded)

    def test_cli_from_other_directory_separates_paths_and_explanation(self):
        with tempfile.TemporaryDirectory() as other_directory:
            result = subprocess.run(
                [sys.executable, str(SELECTOR)], cwd=other_directory,
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        core = manifest_lines("test/cpu-core.txt")
        excluded = manifest_lines("test/cpu-mac-exclude.txt")
        self.assertEqual(result.stdout.splitlines(), SELECT_TESTS(core, excluded))
        self.assertIn("requires AVX2", result.stderr)
        self.assertIn(f"Selected {len(core) - len(excluded)} of {len(core)}", result.stderr)
        for path in excluded:
            self.assertIn(path, result.stderr)


if __name__ == "__main__":
    unittest.main()
