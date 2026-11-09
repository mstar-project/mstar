"""Check live-pr-test's gpu.py: the Eras Cloud provider."""

import argparse
import base64
import contextlib
import functools
import hashlib
import io
import json
import os
import runpy
import subprocess
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GPU = runpy.run_path(str(REPO_ROOT / ".claude/skills/live-pr-test/scripts/gpu.py"))


class TestEras(unittest.TestCase):
    def eras_launch(self, write=None, max_running=None, held=("v1",), refused_by="preview", dry_run=False, run=None,
                    inputs=(), gone=()):
        """gpu.py launch on a fake Eras whose cache volumes, v1 and v2, look free, but the preview (or the
        create, `refused_by`) refuses those `held` (as the spec has it), as when another box took them in
        between, and those `gone` no longer exist; the calls it made."""
        key, state, calls = Path(tempfile.mkdtemp()) / "key.pub", Path(tempfile.mkdtemp()), []
        key.write_text("ssh-ed25519 AAAA test")
        digest = hashlib.sha256(base64.b64decode("AAAA")).digest()
        fingerprint = "SHA256:" + base64.b64encode(digest).decode().rstrip("=")
        keys = []
        tpl = {"endpoints": {"ssh": {"port": 22}}, "pod": {"containers": [
            {"resources": {"limits": {"nvidia.com/gpu": "1"}}, "volumeMounts": []}], "volumes": []}}

        def eras(method, path, body=None, params=None, retry=True):
            calls.append((method, path, json.loads(json.dumps(body))))  # as sent, not as changed later
            if path.removeprefix("/volumes/") in gone:
                raise GPU["ApiError"](404, {"code": "not_found"})
            bound = method == "POST" and path.startswith("/instances") and \
                (body["run"]["bindings"]["volumes"].get("cache") or {}).get("volume_id") in held
            if path == "/instances" and bound and refused_by == "create":
                raise GPU["ApiError"](409, {"code": "volume_bound", "message": "Another instance binds the volume.",
                                            "timestamp": "2026-11-08T00:00:00Z"})
            if path == "/ssh-keys":
                if method == "POST":
                    keys.append({"id": f"k{len(calls)}", "name": body["name"], "fingerprint": fingerprint})
                return keys[-1] if method == "POST" else {"items": keys}
            if path == "/instances/preview":
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
        args = argparse.Namespace(offer="eras:l1", pr=0, hours=1, tag="c", run=run, input=list(inputs), cache="ro",
                                  cache_write=write, allow_over_budget=False, dry_run=dry_run)
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

    def test_a_deleted_cache_volume_starts_cold_but_stops_a_refresh(self):
        for gone, bound in ((["v1"], ["v2"]), (["v1", "v2"], [])):
            created = [c[2] for c in self.eras_launch(held=(), gone=gone) if c[:2] == ("POST", "/instances")]
            self.assertEqual([[v["volume_id"] for k, v in b["run"]["bindings"]["volumes"].items() if k == "cache"]
                              for b in created], [bound], "one stale id would fail every Eras launch")
        with self.assertRaises(GPU["ApiError"], msg="a refresh of a missing volume would fill a box's own disk"):
            self.eras_launch(write="v1", gone=["v1"])

    def test_a_box_makes_sshd_host_keys_before_starting_it(self):
        created = next(c[2] for c in self.eras_launch(held=()) if c[:2] == ("POST", "/instances"))
        script = created["run"]["pod"]["containers"][0]["args"][0]
        self.assertIn("ssh-keygen -A\n", script.split("sshd -D")[0],
                      "sshd would exit with no host keys, and the Pod with it, about 49 s after every launch")

    def test_a_runs_files_reach_the_pod_byte_for_byte_with_an_emoji(self):
        d = Path(tempfile.mkdtemp())
        box, body = d / "box.sh", d / "pr-body.md"
        box.write_text("echo hello\n")
        body.write_text("Generated with Claude Code \N{ROBOT FACE}\n", encoding="utf-8")
        created = next(c[2] for c in self.eras_launch(held=(), run=str(box), inputs=[str(body)])
                       if c[:2] == ("POST", "/instances"))
        files = created["run"]["config_maps"]["live-pr"]
        self.assertNotIn("data", files, "Eras refuses a ConfigMap whose text data holds a 4-byte character")
        self.assertEqual(base64.b64decode(files["binaryData"]["pr-body.md"]), body.read_bytes(),
                         "the box would read a changed PR description")

    def test_a_dry_run_writes_nothing_to_eras(self):
        self.assertEqual([c[:2] for c in self.eras_launch(dry_run=True) if c[0] != "GET"], [],
                         "a dry run would register the ssh key with the organization")
        self.assertIn(("POST", "/ssh-keys"), [c[:2] for c in self.eras_launch(held=())],
                      "a real launch would have no key to sign in with")

    def test_each_launch_attempt_has_its_own_idempotency_key(self):
        keys = [c[2]["idempotency_key"] for c in self.eras_launch(held=["v1", "v2"], refused_by="create")
                if c[:2] == ("POST", "/instances")]
        self.assertEqual(len(set(keys)), 3, "Eras refuses a key reused with another body (idempotency_key_reused), "
                                            "so the next volume or a cold box would never launch")

    def test_a_create_whose_answer_was_lost_is_not_sent_again(self):
        posts, globs = [], GPU["eras"].__globals__
        self.addCleanup(globs.update, {"_eras_org": globs["_eras_org"]})
        globs["_eras_org"] = "org"

        def urlopen(req, timeout=None):
            if req.full_url.endswith("/instances/preview"):
                return io.BytesIO(json.dumps({"run": {"problems": []}, "is_available": True, "available_until": None,
                                              "quote": {"line_items": []}}).encode())
            posts.append(req.full_url)
            raise TimeoutError("timed out")

        body = {"listing_id": "l1", "name": "live-pr-7-t", "run": {"bindings": {
            "volumes": {}, "secrets": {"authorized-keys": {"ssh_key_ids": ["k1"]}}}}}
        with unittest.mock.patch.dict(os.environ, {"ERAS_API_KEY": "k"}), unittest.mock.patch("time.sleep"), \
                unittest.mock.patch("urllib.request.urlopen", urlopen), \
                self.assertRaises(GPU["ApiError"]) as caught:
            GPU["eras_create"](body)
        self.assertEqual(len(posts), 1, "the retry loop would buy a box for each resend it couldn't confirm")
        self.assertEqual(caught.exception.status, -1)

    def test_a_pending_entry_releases_the_live_instance_with_its_name(self):
        calls, state = [], Path(tempfile.mkdtemp())

        def eras(method, path, body=None, params=None, retry=True):
            calls.append((method, path))
            items = [{"id": "gone", "name": "live-pr-7-t", "status": "releasing"},
                     {"id": "other", "name": "live-pr-8-t", "status": "active"},
                     {"id": "i-9", "name": "live-pr-7-t", "status": "active"}]
            return {"items": items if path == "/instances" else [], "next_page_token": None}

        globs = GPU["terminate"].__globals__
        self.addCleanup(globs.update, {k: globs[k] for k in ("eras", "STATE", "LEDGER")})
        globs.update(eras=eras, STATE=state, LEDGER=state / "instances.json")
        e = {"provider": "eras", "instance_id": None, "name": "live-pr-7-t", "launched_at": 0, "usd_per_hour": 1.0,
             "region": "x"}
        GPU["save_ledger"]({"pending:live-pr-7-t": e})
        with contextlib.redirect_stdout(io.StringIO()):
            GPU["terminate"]("pending:live-pr-7-t", e)
        self.assertEqual([p for m, p in calls if m == "POST"], ["/instances/i-9/release"])
        self.assertEqual(GPU["ledger"](), {})

    def test_extend_cant_outrun_a_deadline_fixed_at_launch(self):
        globs = GPU["cmd_extend"].__globals__
        self.addCleanup(globs.update, {k: globs[k] for k in ("STATE", "LEDGER")})
        state = Path(tempfile.mkdtemp())
        globs.update(STATE=state, LEDGER=state / "instances.json")
        soon = time.time() + 600
        with GPU["ledger_txn"]() as d:
            d.update({f"{p}:i": {"provider": p, "deadline": soon, "launched_at": time.time(), "usd_per_hour": 1.0}
                      for p in ("eras", "runpod")})
        cfg = {"budget": {"max_hours": 4}}
        # --allow-over-budget lifts the caps, not what was fixed at launch
        with self.assertRaises(SystemExit, msg="the watchdog would move but the box would end at its old time"), \
                contextlib.redirect_stderr(io.StringIO()):
            GPU["cmd_extend"](argparse.Namespace(instance="eras:i", hours=1, allow_over_budget=True), cfg)
        extend = functools.partial(argparse.Namespace, allow_over_budget=False)
        with contextlib.redirect_stdout(io.StringIO()):
            GPU["cmd_extend"](extend(instance="runpod:i", hours=1), cfg)
            GPU["cmd_extend"](extend(instance="eras:i", hours=0), cfg)  # how a lost API key still terminates
        self.assertLess(GPU["ledger"]()["eras:i"]["deadline"], soon)
        self.assertGreater(GPU["ledger"]()["runpod:i"]["deadline"], soon)

    def test_max_running_counts_this_tools_live_instances_on_every_page(self):
        globs = GPU["running"].__globals__
        self.addCleanup(globs.update, {"eras": globs["eras"]})
        globs["eras"] = lambda *a, params=None, **k: (
            {"items": [{"id": "i1", "name": "live-pr-4-8-1", "status": "active"},
                       {"id": "i3", "name": "someone-else", "status": "active"}]} if params.get("page_token")
            else {"items": [{"id": "i2", "name": "live-pr-2-1-1", "status": "ended"}], "next_page_token": "p2"})
        with unittest.mock.patch.dict(os.environ, {"ERAS_API_KEY": "k"}):
            up = GPU["running"]({"eras": {}})
        self.assertEqual([line.split()[0] for line in up], ["eras:i1"],
                         "every live test would hold its own box, past budget.max_running")

    def test_an_eras_ssh_key_on_a_later_page_is_reused(self):
        key = base64.b64encode(b"key").decode()
        fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(b"key").digest()).decode().rstrip("=")
        posts = []

        def eras(method, path, body=None, params=None):
            if method == "POST":
                posts.append(path)
                return {"id": "new"}
            return {"items": [{"id": "k2", "fingerprint": fingerprint}]} if params.get("page_token") \
                else {"items": [{"id": "k1", "fingerprint": "other"}], "next_page_token": "p2"}
        globs = GPU["eras_ssh_key_id"].__globals__
        self.addCleanup(globs.update, {"eras": globs["eras"]})
        globs["eras"] = eras
        self.assertEqual(GPU["eras_ssh_key_id"](f"ssh-ed25519 {key} me", "live-pr-1"), "k2",
                         "a key on a later page would be registered again on every launch")
        self.assertEqual(posts, [])

    def test_a_launch_past_the_caps_registers_nothing(self):
        calls = self.eras_launch(max_running=0)
        self.assertEqual([c[:2] for c in calls if c[0] in ("POST", "exit")], [("exit", 4)],
                         "a refused launch would leave an ssh key in the organization")


if __name__ == "__main__":
    unittest.main()
