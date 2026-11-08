"""Check live-pr-test's gpu.py: the Slurm node and Eras Cloud providers."""

import argparse
import base64
import contextlib
import hashlib
import io
import json
import os
import runpy
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GPU = runpy.run_path(str(REPO_ROOT / ".claude/skills/live-pr-test/scripts/gpu.py"))
SLURM_CFG = {"slurm": {"host": "root@node", "job_prefix": "live-pr", "workdir": "/workspace/live-pr"}}


def fake_slurm(replies, calls=None):
    """Stands in for gpu.py's slurm_run: each command gets the first reply whose key it contains, and
    an exception is a failed command (empty stdout with check=False)."""
    def slurm_run(cfg, command, check=True):
        if calls is not None:
            calls.append(command)
        reply = next(r for k, r in replies.items() if k in command)
        if isinstance(reply, Exception):
            if not check:
                return ""
            raise reply
        return reply
    return slurm_run


class TestSlurm(unittest.TestCase):
    def setUp(self):
        self.globals = GPU["status"].__globals__
        self.saved = self.globals["slurm_run"]

    def tearDown(self):
        self.globals["slurm_run"] = self.saved

    def test_a_failed_ssh_is_not_a_job_that_ended(self):
        e = {"provider": "slurm", "instance_id": "42", "run": True}
        self.globals["slurm_run"] = fake_slurm({"squeue": GPU["ApiError"](255, "ssh: connect to host node: timed out")})
        with self.assertRaises(GPU["ApiError"], msg="a flaky ssh would end a running live test"):
            GPU["status"](e, SLURM_CFG)
        self.globals["slurm_run"] = fake_slurm(
            {"squeue": GPU["ApiError"](1, "slurm: squeue: slurm_load_jobs error: Invalid job id specified")})
        self.assertEqual(GPU["status"](e, SLURM_CFG)[0], "ended")
        self.globals["slurm_run"] = fake_slurm({"squeue": "RUNNING\n"})
        self.assertEqual(GPU["status"](e, SLURM_CFG)[0], "running")

    def test_terminate_leaves_nothing_of_the_run_on_the_node(self):
        calls, saved = [], {k: self.globals[k] for k in ("load_config", "STATE", "LEDGER")}
        self.addCleanup(self.globals.update, saved)
        state = Path(tempfile.mkdtemp())
        self.globals.update(slurm_run=fake_slurm({"scancel": ""}, calls), load_config=lambda: SLURM_CFG,
                            STATE=state, LEDGER=state / "instances.json")
        jobdir = "/workspace/live-pr/live-pr-7-1-1"
        GPU["terminate"]("slurm:42", {"provider": "slurm", "instance_id": "42", "dir": jobdir, "launched_at": 0,
                                      "usd_per_hour": 0.0})
        self.assertTrue(any("scancel 42" in c and f"rm -rf {jobdir}" in c for c in calls),
                        f"the job's directory (the Anthropic key, venvs, builds) would stay on the node: {calls}")

    def test_no_launch_past_max_running_from_any_session(self):
        key, state, calls = Path(tempfile.mkdtemp()) / "key.pub", Path(tempfile.mkdtemp()), []
        key.write_text("ssh-ed25519 AAAA test")
        saved = {k: self.globals[k] for k in ("eras", "STATE", "LEDGER")}
        self.addCleanup(self.globals.update, saved)
        self.globals.update(
            slurm_run=fake_slurm({"sinfo": "node1 gpu:8 gpu:0 idle\n", "sbatch": "42\n",
                                  "squeue": "live-pr-3-9-1 41 RUNNING\njoe-live-pr-5-x 40 RUNNING\n"}, calls),
            eras=lambda *a, **k: {"items": [{"id": "i1", "name": "live-pr-4-8-1", "status": "active"},
                                            {"id": "i2", "name": "live-pr-2-1-1", "status": "ended"}]},
            STATE=state, LEDGER=state / "instances.json")
        args = argparse.Namespace(offer="slurm:node1:1", pr=7, hours=1, tag="1-1", run=None, input=[], cache="ro",
                                  allow_over_budget=False, dry_run=False)

        def launch(max_running):
            cfg = {**SLURM_CFG, "ssh_public_key": str(key), "eras": {}, "budget": {
                "max_usd_per_hour": 0, "max_hours": 4, "max_concurrent_instances": 9, "max_running": max_running}}
            with unittest.mock.patch.dict(os.environ, {"ERAS_API_KEY": "k"}), \
                    unittest.mock.patch.object(subprocess, "Popen"), contextlib.redirect_stdout(io.StringIO()):
                GPU["cmd_launch"](args, cfg)
            return [c for c in calls if c.startswith("sbatch")]

        # Up: one of the bot's Slurm jobs and one live Eras instance, from other runs.
        with self.assertRaises(SystemExit, msg="every live test would hold its own box, past budget.max_running"), \
                contextlib.redirect_stderr(io.StringIO()):
            launch(2)
        self.assertEqual(len(launch(3)), 1)

        def eras_down(*a, **k):
            raise GPU["ApiError"](-1, "GET /instances: gave up after retries")
        self.globals["eras"] = eras_down
        self.assertEqual(len(launch(2)), 2, "an Eras outage would stop launches on the Slurm node too")

    def test_the_job_uses_the_shared_hf_cache(self):
        key = Path(tempfile.mkdtemp()) / "key.pub"
        key.write_text("ssh-ed25519 AAAA test")
        cfg = {**SLURM_CFG, "ssh_public_key": str(key), "budget": {"max_usd_per_hour": 0, "max_hours": 4},
               "slurm": {**SLURM_CFG["slurm"], "hf_home": "/workspace/shared/hf"}}
        self.globals["slurm_run"] = fake_slurm({"sinfo": "node1 gpu:8 gpu:0 idle\n"})
        args = argparse.Namespace(offer="slurm:node1:1", pr=7, hours=1, tag="1-1", run="job.sh",
                                  input=[], allow_over_budget=False, dry_run=True)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            GPU["cmd_launch"](args, cfg)
        self.assertIn("HF_HOME=/workspace/shared/hf", out.getvalue(),
                      "each run would download the weights again into its own directory")


class TestEras(unittest.TestCase):
    def test_terminate_deletes_only_its_own_ssh_key_once_no_live_instance_binds_it(self):
        """A deleted key leaves the authorized_keys of every box that binds it (the spec's
        SecretBindingRequest.ssh_key_ids), and DELETE /ssh-keys refuses nothing in use."""
        calls, state = [], Path(tempfile.mkdtemp())
        old = "2026-01-01T00:00:00Z"
        keys = [{"id": k, "name": n, "key_type": "ssh-ed25519", "fingerprint": f"SHA256:{k}", "created_at": old,
                 "created_by": None} for k, n in (("k1", "live-pr-7-1-1"), ("k2", "someone's laptop"),
                                                  ("k3", "live-pr-6-1-1"))]
        names = {"i-1": "live-pr-7-1-1", "i-2": "live-pr-9-1-1", "i-3": "live-pr-8-1-1"}
        binds = {"i-1": ["k1"], "i-2": ["k1"], "i-3": ["k3"]}  # i-2 reuses the key launch 7 registered
        status = {"i-1": "releasing", "i-2": "active", "i-3": "active"}

        def eras(method, path, body=None, params=None):
            calls.append((method, path))
            if method == "DELETE" and refused:
                raise GPU["ApiError"](403, {"code": "permission_denied", "message": "Not allowed."})
            if path == "/instances":  # ended instances aren't listed
                return {"items": [{"id": i, "name": names[i], "status": status[i], "mode": "standalone",
                                   "standalone": {"current_run": {"status": "running", "bindings": {
                                       "secrets": {"authorized-keys": {"ssh_key_ids": ids}}, "volumes": {}}}}}
                                  for i, ids in binds.items()], "next_page_token": None, "total": len(binds)}
            flt = json.loads((params or {}).get("filter") or "null")
            return {"items": [k for k in keys if not flt or k["name"] == flt["value"]], "next_page_token": None,
                    "total": len(keys)}

        globs = GPU["terminate"].__globals__
        saved = {k: globs[k] for k in ("eras", "STATE", "LEDGER")}
        self.addCleanup(globs.update, saved)
        globs.update(eras=eras, STATE=state, LEDGER=state / "instances.json")

        def terminate(iid, name):
            calls.clear()
            e = {"provider": "eras", "instance_id": iid, "name": name, "launched_at": 0, "usd_per_hour": 1.0}
            GPU["save_ledger"]({f"eras:{iid}": e})
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                GPU["terminate"](f"eras:{iid}", e)
            return [c[1] for c in calls if c[0] == "DELETE"]

        refused = False
        self.assertEqual(terminate("i-1", names["i-1"]), [],
                         "the box still binding the key would lose ssh access")
        status["i-2"] = "releasing"
        self.assertEqual(terminate("i-1", names["i-1"]), ["/ssh-keys/k1"],
                         "a key whose other binders are going away would never be deleted")
        del binds["i-3"]
        self.assertEqual(terminate("i-3", names["i-3"]), [],
                         "a launch would delete a key it reused, which another launch or a person registered")
        # A delete that fails: the release stands, and the watchdog stops.
        refused = True
        terminate("i-1", names["i-1"])
        self.assertEqual(GPU["ledger"](), {}, "the watchdog would retry a released instance forever")

    def eras_launch(self, write=None, max_running=None, held=("v1",), refused_by="preview", deleted=False):
        """gpu.py launch on a fake Eras whose cache volumes, v1 and v2, look free, but the preview (or the
        create, `refused_by`) refuses those `held` (as the spec has it), as when another box took them in
        between; the calls it made. With `deleted`, the organization has the launch's ssh key already, and
        the terminate of the launch that registered it deletes it between the lookup and the create."""
        key, state, calls = Path(tempfile.mkdtemp()) / "key.pub", Path(tempfile.mkdtemp()), []
        key.write_text("ssh-ed25519 AAAA test")
        digest = hashlib.sha256(base64.b64decode("AAAA")).digest()
        fingerprint = "SHA256:" + base64.b64encode(digest).decode().rstrip("=")
        keys = [{"id": "k0", "name": "live-pr-1-old", "fingerprint": fingerprint}] if deleted else []
        tpl = {"endpoints": {"ssh": {"port": 22}}, "pod": {"containers": [
            {"resources": {"limits": {"nvidia.com/gpu": "1"}}, "volumeMounts": []}], "volumes": []}}

        def eras(method, path, body=None, params=None):
            calls.append((method, path, json.loads(json.dumps(body))))  # as sent, not as changed later
            bound = method == "POST" and path.startswith("/instances") and \
                (body["run"]["bindings"]["volumes"].get("cache") or {}).get("volume_id") in held
            if path == "/instances" and bound and refused_by == "create":
                raise GPU["ApiError"](409, {"code": "volume_bound", "message": "Another instance binds the volume.",
                                            "timestamp": "2026-11-08T00:00:00Z"})
            if path == "/ssh-keys":
                if method == "POST":
                    keys.append({"id": f"k{len(calls)}", "name": body["name"], "fingerprint": fingerprint})
                return keys[-1] if method == "POST" else {"items": keys}
            if path == "/instances" and method == "POST" and not {k["id"] for k in keys} >= set(
                    body["run"]["bindings"]["secrets"]["authorized-keys"]["ssh_key_ids"]):
                raise GPU["ApiError"](404, {"code": "not_found", "message": "No such SSH key.",
                                            "timestamp": "2026-11-08T00:00:00Z"})
            if path == "/instances/preview":
                if deleted:
                    keys.clear()
                bound = bound and refused_by == "preview"
                return {"run": {"problems": []}, "is_available": not bound,
                        "unavailable_reason": "volume_bound" if bound else None,
                        "unavailable_detail": "Another instance binds the volume." if bound else None,
                        "available_until": None, "quote": {"line_items": []}}
            replies = {"/instances": {"id": "i-1"}, "/volumes/v1": {"attachment": None},
                       "/volumes/v2": {"attachment": None}}
            return replies.get(path, tpl)

        saved = {k: GPU["cmd_launch"].__globals__[k] for k in ("eras", "find_offer", "STATE", "LEDGER")}
        self.addCleanup(GPU["cmd_launch"].__globals__.update, saved)
        GPU["cmd_launch"].__globals__.update(
            eras=eras, STATE=state, LEDGER=state / "instances.json", find_offer=lambda cfg, offer: {
                "id": "eras:l1", "provider": "eras", "gpu": "H200", "count": 1, "usd_per_hour": 3.0, "region": "x",
                "storage": True})
        cfg = {"ssh_public_key": str(key), "eras": {"cache_volume_ids": ["v1", "v2"]},
               "budget": {"max_usd_per_hour": 30, "max_hours": 4, "max_running": max_running}}
        args = argparse.Namespace(offer="eras:l1", pr=0, hours=1, tag="c", run=None, input=[], cache="ro",
                                  cache_write=write, allow_over_budget=False, dry_run=False)
        with unittest.mock.patch.object(subprocess, "Popen"), contextlib.redirect_stdout(io.StringIO()), \
                unittest.mock.patch.dict(os.environ, {"ERAS_API_KEY": "k"}), contextlib.redirect_stderr(io.StringIO()):
            try:
                GPU["cmd_launch"](args, cfg)
            except SystemExit as e:
                calls.append(("exit", e.code, None))
        return calls

    def test_a_cache_refresh_never_launches_without_its_volume(self):
        calls = self.eras_launch(write="v1")
        self.assertEqual([c for c in calls if c[1] == "/instances" and c[0] == "POST"], [],
                         "the refresh would fill a box's local disk and report success, writing nothing to the volume")
        for held, bound in ((["v1"], ["v2"]), (["v1", "v2"], [])):
            created = [c[2] for c in self.eras_launch(held=held) if c[1] == "/instances" and c[0] == "POST"]
            self.assertEqual([[v["volume_id"] for k, v in b["run"]["bindings"]["volumes"].items() if k == "cache"]
                              for b in created], [bound],
                             "a live test would give up on Eras rather than take the next free volume or start cold")

    def test_each_launch_attempt_has_its_own_idempotency_key(self):
        keys = [c[2]["idempotency_key"] for c in self.eras_launch(held=["v1", "v2"], refused_by="create")
                if c[:2] == ("POST", "/instances")]
        self.assertEqual(len(set(keys)), 3, "Eras refuses a key reused with another body (idempotency_key_reused), "
                                            "so the next volume or a cold box would never launch")

    def test_a_launch_whose_ssh_key_was_deleted_registers_it_again(self):
        calls = self.eras_launch(deleted=True, held=())
        bound = [c[2]["run"]["bindings"]["secrets"]["authorized-keys"]["ssh_key_ids"] for c in calls
                 if c[:2] == ("POST", "/instances")]
        self.assertEqual([ids == ["k0"] for ids in bound], [True, False],
                         "a launch reusing a key another launch deleted mid-launch would fail")
        self.assertNotIn("exit", [c[0] for c in calls])

    def test_a_launch_past_the_caps_registers_nothing(self):
        calls = self.eras_launch(max_running=0)
        self.assertEqual([c[:2] for c in calls if c[0] in ("POST", "exit")], [("exit", 4)],
                         "a refused launch would leave an ssh key in the organization")


if __name__ == "__main__":
    unittest.main()
