"""Small local-runner checks requiring only Python and Git, with no Docker daemon."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO_ROOT / "scripts" / "ci" / "snapshot.py"


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo with spaces"
        self.root.mkdir()
        self.archive = Path(self.temp.name) / "snapshot.tar"
        self.git("init", "-q")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def run_snapshot(self):
        return subprocess.run(
            [sys.executable, str(SNAPSHOT), str(self.root), str(self.archive)],
            capture_output=True, text=True, check=False,
        )

    def test_current_source_and_symlinks_without_local_artifacts(self):
        self.write(".gitignore", "ignored.txt\n.env\n")
        self.write("staged.py", "original\n")
        self.write("modified.py", "original\n")
        self.write("deleted.py", "original\n")
        self.write(".sample.env", "EXAMPLE=1\n")
        # Ignore-like paths that are explicitly tracked still belong to the source.
        self.write("build/source.py", "tracked source\n")
        self.git("add", ".")
        self.write("staged.py", "staged change\n")
        self.git("add", "staged.py")
        self.write("modified.py", "unstaged change\n")
        (self.root / "deleted.py").unlink()
        self.write("new file.py", "new source\n")
        executable = self.write("run.sh", "#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        external = Path(self.temp.name) / "outside-secret"
        external.write_text("must not copy this content\n")
        (self.root / "external-link").symlink_to(external)
        (self.root / "broken-link").symlink_to("missing-file")
        for name in (
            "ignored.txt", ".env", ".env.local", ".venv/lib/package.py",
            "venv/lib/package.py", "build/generated.py", "dist/package.whl",
            "rust/target/release/stale", ".ci-results/previous.log",
        ):
            self.write(name, "must not copy\n")

        result = self.run_snapshot()
        self.assertEqual(result.returncode, 0, result.stderr)
        with tarfile.open(self.archive) as archive:
            self.assertEqual(set(archive.getnames()), {
                ".gitignore", ".sample.env", "build/source.py", "staged.py", "modified.py",
                "new file.py", "run.sh", "external-link", "broken-link",
            })
            self.assertEqual(archive.extractfile("staged.py").read(), b"staged change\n")
            self.assertEqual(archive.extractfile("modified.py").read(), b"unstaged change\n")
            self.assertEqual(archive.extractfile("new file.py").read(), b"new source\n")
            self.assertEqual(archive.getmember("run.sh").mode & 0o111, 0o111)
            self.assertTrue(archive.getmember("external-link").issym())
            self.assertEqual(archive.getmember("external-link").linkname, str(external))
            self.assertTrue(archive.getmember("broken-link").issym())

    def test_submodule_is_rejected_instead_of_silently_omitted(self):
        self.git("update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},vendor")
        result = self.run_snapshot()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not yet support submodules: vendor", result.stderr)

    def test_replaced_directory_symlink_cannot_copy_external_files(self):
        tracked = self.write("package/module.py", "original\n")
        self.git("add", ".")
        tracked.unlink()
        tracked.parent.rmdir()
        external = Path(self.temp.name) / "external"
        external.mkdir()
        (external / "module.py").write_text("outside source\n")
        tracked.parent.symlink_to(external, target_is_directory=True)
        result = self.run_snapshot()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("directory symlink", result.stderr)


class LocalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        temporary = Path(self.temp.name)
        self.root = temporary / "repo with spaces"
        scripts = self.root / "scripts" / "ci"
        scripts.mkdir(parents=True)
        for name in ("mac.sh", "snapshot.py"):
            shutil.copy2(REPO_ROOT / "scripts" / "ci" / name, scripts / name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True, capture_output=True)
        subprocess.run([
            "git", "-C", str(self.root), "-c", "user.name=Local CI Test", "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture",
        ], check=True, capture_output=True)
        self.calls = temporary / "docker-calls.jsonl"
        fake_bin = temporary / "bin"
        fake_bin.mkdir()
        docker = fake_bin / "docker"
        docker.write_text(f"#!{sys.executable}\n" + """
import json
import os
import sys
import time
from pathlib import Path
with open(os.environ["CI_FAKE_DOCKER_LOG"], "a") as output:
    output.write(json.dumps(sys.argv[1:]) + "\\n")
state = Path(os.environ["CI_FAKE_DOCKER_STATE"])
if sys.argv[1] == "rm":
    if state.exists() and json.loads(state.read_text())["name"] == sys.argv[-1]:
        state.unlink()
    sys.exit(0)
if sys.argv[1] == os.environ.get("CI_FAKE_DOCKER_BLOCK_COMMAND"):
    name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else None
    state.write_text(json.dumps({"name": name}))
    print("container check started", flush=True)
    while state.exists():
        time.sleep(0.02)
    sys.exit(137)
if sys.argv[1] == os.environ.get("CI_FAKE_DOCKER_FAIL_COMMAND"):
    phase = "image build" if sys.argv[1] == "build" else "container check"
    print(f"intentional {phase} failure", flush=True)
    sys.exit(17)
if sys.argv[1] == "run":
    print(f"container check passed: {sys.argv[-1]}", flush=True)
if sys.argv[1] == "info":
    print(os.environ["CI_FAKE_DOCKER_SECURITY"])
""")
        docker.chmod(0o755)
        uname = fake_bin / "uname"
        uname.write_text(f"#!{sys.executable}\n" + """
import os
import sys
print(os.environ["CI_FAKE_HOST_SYSTEM" if sys.argv[1] == "-s" else "CI_FAKE_HOST_ARCH"])
""")
        uname.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
            "CI_FAKE_DOCKER_LOG": str(self.calls),
            "CI_FAKE_DOCKER_STATE": str(temporary / "container-state.json"),
            "CI_FAKE_DOCKER_FAIL_COMMAND": "run",
            "CI_FAKE_DOCKER_SECURITY": "[]",
            "CI_PLATFORM": "linux/arm64",
            "CI_BUILD_JOBS": "2",
            "CI_FAKE_HOST_SYSTEM": "Darwin",
            "CI_FAKE_HOST_ARCH": "arm64",
        }

    def run_local(self, *args, cwd=None):
        return subprocess.run(
            ["bash", str(self.root / "scripts" / "ci" / "mac.sh"), *args],
            cwd=cwd or self.root, env=self.env, capture_output=True, text=True, check=False,
        )

    def test_help_and_invalid_job_do_not_contact_docker(self):
        help_result = self.run_local("--help")
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("Mac checks; full x86 CPU suite not run", help_result.stdout)
        self.assertIn("cpu-smoke", help_result.stdout)
        self.assertEqual(self.run_local("not-a-job").returncode, 2)
        self.assertEqual(self.run_local("cpu-tests").returncode, 2)
        self.assertFalse(self.calls.exists())
        self.assertFalse((self.root / ".ci-results").exists())

    def test_unsupported_hosts_are_rejected_before_docker(self):
        for system, architecture in (("Linux", "aarch64"), ("Darwin", "x86_64"), ("MINGW64_NT", "x86_64")):
            with self.subTest(system=system, architecture=architecture):
                self.env["CI_FAKE_HOST_SYSTEM"] = system
                self.env["CI_FAKE_HOST_ARCH"] = architecture
                result = self.run_local("lint")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Apple Silicon macOS", result.stderr)
                self.assertFalse(self.calls.exists())
                self.assertFalse((self.root / ".ci-results").exists())

    def test_emulated_platform_is_rejected_before_docker(self):
        self.env["CI_PLATFORM"] = "linux/amd64"
        result = self.run_local("lint")
        self.assertEqual(result.returncode, 2)
        self.assertIn("linux/arm64", result.stderr)
        self.assertFalse(self.calls.exists())
        self.assertFalse((self.root / ".ci-results").exists())

    def test_invalid_environment_is_rejected_before_docker(self):
        for name, value in (("CI_PLATFORM", "windows/amd64"), ("CI_BUILD_JOBS", "0")):
            with self.subTest(name=name):
                original = self.env[name]
                self.env[name] = value
                result = self.run_local("lint")
                self.env[name] = original
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn(name, result.stderr)
                self.assertFalse(self.calls.exists())
                self.assertFalse((self.root / ".ci-results").exists())

    def test_image_build_failure_preserves_logs_and_removes_snapshot(self):
        self.env["CI_FAKE_DOCKER_FAIL_COMMAND"] = "build"
        result = self.run_local("all")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual([call[0] for call in calls], ["info", "build"])
        reports = list((self.root / ".ci-results").iterdir())
        self.assertEqual(len(reports), 1, "Temporary source snapshot was not cleaned")
        self.assertFalse((reports[0] / "source.tar").exists())
        self.assertIn("intentional image build failure", (reports[0] / "lint" / "image-build.log").read_text())
        self.assertIn("lint: image build failed", (reports[0] / "summary.txt").read_text())
        self.assertNotIn("All selected Mac checks passed", result.stdout)

    def test_user_namespace_runtimes_are_rejected_before_build(self):
        for option in ("name=rootless", "name=userns"):
            with self.subTest(option=option):
                self.env["CI_FAKE_DOCKER_SECURITY"] = json.dumps(["name=seccomp,profile=builtin", option])
                result = self.run_local("lint")
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("rootful Docker", result.stderr)
                calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
                self.assertTrue(all(call[0] == "info" for call in calls))
                self.assertFalse((self.root / ".ci-results").exists())

    def test_all_jobs_succeed_when_launched_outside_checkout(self):
        self.env["CI_FAKE_DOCKER_FAIL_COMMAND"] = ""
        del self.env["CI_PLATFORM"]
        result = self.run_local(cwd=Path(self.temp.name))
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        jobs = ("lint", "cpu-smoke", "dynamo", "rust")
        self.assertEqual([call[-1] for call in calls if call[0] == "run"], list(jobs))
        reports = list((self.root / ".ci-results").iterdir())
        self.assertEqual(len(reports), 1, "Temporary source snapshot was not cleaned")
        summary = (reports[0] / "summary.txt").read_text()
        self.assertIn("Mac checks; full x86 CPU suite not run", summary)
        self.assertIn("full x86 CPU suite not run", result.stdout)
        for call in calls:
            if call[0] in ("build", "run"):
                self.assertEqual(call[call.index("--platform") + 1], "linux/arm64")
        for job in jobs:
            self.assertIn(f"{job}: passed", summary)
            self.assertIn(f"container check passed: {job}", (reports[0] / job / "check.log").read_text())
        self.assertIn("All selected Mac checks passed", result.stdout)
        self.assertFalse((Path(self.temp.name) / ".ci-results").exists())

    def check_cancellation(self, signum):
        self.env["CI_FAKE_DOCKER_FAIL_COMMAND"] = ""
        self.env["CI_FAKE_DOCKER_BLOCK_COMMAND"] = "run"
        state = Path(self.env["CI_FAKE_DOCKER_STATE"])
        with tempfile.TemporaryFile(mode="w+") as output:
            process = subprocess.Popen(
                ["bash", str(self.root / "scripts" / "ci" / "mac.sh"), "all"],
                cwd=self.root, env=self.env, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 5
                while not state.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(state.exists(), "Container did not start")
                process.send_signal(signum)
                process.wait(timeout=5)
                self.assertNotEqual(process.returncode, 0)
                self.assertFalse(state.exists(), "Cancelled job left its container running")
                calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
                self.assertEqual(len([call for call in calls if call[0] == "run"]), 1)
                reports = list((self.root / ".ci-results").iterdir())
                self.assertEqual(len(reports), 1, "Temporary source snapshot was not cleaned")
                self.assertIn("container check started", (reports[0] / "lint" / "check.log").read_text())
                output.seek(0)
                self.assertNotIn("All selected Mac checks passed", output.read())
            finally:
                # Also stop fake Docker if the runner regresses and leaves it alive.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)

    def test_sigterm_stops_owned_container_and_preserves_logs(self):
        self.check_cancellation(signal.SIGTERM)

    def test_sigint_stops_owned_container_and_preserves_logs(self):
        self.check_cancellation(signal.SIGINT)

    def test_container_failure_preserves_logs_and_removes_snapshot(self):
        result = self.run_local("all")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        runs = [call for call in calls if call[0] == "run"]
        self.assertEqual(len(runs), 1, "Later jobs must not run after a failure")
        snapshot_mount = next(arg for arg in runs[0] if "target=/source.tar" in arg)
        source = next(field.removeprefix("source=") for field in snapshot_mount.split(",")
                      if field.startswith("source="))
        self.assertFalse(Path(source).parent.exists(), "Temporary snapshot directory was not cleaned")
        reports = list((self.root / ".ci-results").iterdir())
        self.assertEqual(len(reports), 1)
        self.assertIn("intentional container check failure", (reports[0] / "lint" / "check.log").read_text())
        self.assertIn("FAILED", (reports[0] / "summary.txt").read_text())
        self.assertNotIn("All selected Mac checks passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
