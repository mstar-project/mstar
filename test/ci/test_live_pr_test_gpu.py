"""Check live-pr-test's gpu.py: the Slurm node and Eras Cloud providers."""

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
SLURM_CFG = {"slurm": [{"name": "node1", "host": "root@node", "job_prefix": "live-pr",
                        "workdir": "/workspace/live-pr"}]}


def fake_slurm(replies, calls=None):
    """Stands in for gpu.py's slurm_run: each command gets the first reply whose key it contains, and
    an exception is a failed command (empty stdout with check=False)."""
    def slurm_run(sl, command, check=True):
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
        e = {"provider": "slurm", "instance_id": "42", "run": True, "region": "node1"}
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
                                      "usd_per_hour": 0.0, "region": "node1"})
        self.assertTrue(any("scancel 42" in c and f"rm -rf {jobdir}" in c for c in calls),
                        f"the job's directory (the Anthropic key, venvs, builds) would stay on the node: {calls}")

    def test_the_same_job_id_on_two_nodes_is_two_instances(self):
        key, state = Path(tempfile.mkdtemp()) / "key.pub", Path(tempfile.mkdtemp())
        key.write_text("ssh-ed25519 AAAA test")
        saved = {k: self.globals[k] for k in ("STATE", "LEDGER")}
        self.addCleanup(self.globals.update, saved)
        self.globals.update(slurm_run=fake_slurm({"sinfo": "n gpu:8 gpu:0 idle\n", "sbatch": "42\n"}),
                            STATE=state, LEDGER=state / "instances.json")
        nodes = [{**SLURM_CFG["slurm"][0], "name": name} for name in ("node1", "node2")]
        cfg = {"slurm": nodes, "ssh_public_key": str(key), "budget": {
            "max_usd_per_hour": 0, "max_hours": 4, "max_concurrent_instances": 9}}
        for name in ("node1", "node2"):
            args = argparse.Namespace(offer=f"slurm:{name}:1", pr=7, hours=1, tag=name, run=None, input=[],
                                      cache="ro", allow_over_budget=False, dry_run=False)
            with unittest.mock.patch.object(subprocess, "Popen"), contextlib.redirect_stdout(io.StringIO()):
                GPU["cmd_launch"](args, cfg)
        self.assertEqual(sorted(GPU["ledger"]()), ["slurm:node1:42", "slurm:node2:42"],
                         "the second node's job would overwrite the first's entry, and its deadline watchdog")

    def test_nodes_on_one_host_keep_their_own_known_hosts(self):
        self.addCleanup(self.globals.update, {"STATE": self.globals["STATE"]})
        self.globals["STATE"] = Path(tempfile.mkdtemp())
        files = [next(a for a in GPU["slurm_ssh"]({"name": name, "host": "root@7.tcp.ngrok.io", "port": port,
                                                  "host_key": f"ssh-ed25519 {name}"}) if "UserKnownHostsFile" in a)
                 for name, port in (("node1", 21), ("node2", 22))]
        self.assertNotEqual(*files, "one node's pinned key would overwrite the other's, and ssh would refuse it")

    def test_a_bad_slurm_config_is_refused_with_a_reason(self):
        for slurm in ({"host": "h", "name": "n"}, [{"host": "h"}], [{"host": "h", "name": "a:b"}],
                      [{"host": "h", "name": "n", "enabled": False}]):
            with self.subTest(slurm=slurm), self.assertRaisesRegex(GPU["ApiError"], "slurm"):
                GPU["slurm_nodes"]({"slurm": slurm})

    def test_the_watchdog_outlives_a_node_gone_from_the_config(self):
        saved = {k: self.globals[k] for k in ("STATE", "LEDGER", "load_config")}
        self.addCleanup(self.globals.update, saved)
        state = Path(tempfile.mkdtemp())
        self.globals.update(STATE=state, LEDGER=state / "instances.json", load_config=lambda: {"slurm": []})
        with GPU["ledger_txn"]() as d:
            d["slurm:gone:42"] = {"provider": "slurm", "instance_id": "42", "region": "gone", "deadline": 0}

        with unittest.mock.patch("time.sleep", side_effect=InterruptedError), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaises(InterruptedError, msg="the watchdog would exit, and the job would run on"):
            GPU["cmd_watchdog"](argparse.Namespace(instance="slurm:gone:42"), {})

    def test_extend_cant_outrun_a_deadline_fixed_at_launch(self):
        saved = {k: self.globals[k] for k in ("STATE", "LEDGER")}
        self.addCleanup(self.globals.update, saved)
        state = Path(tempfile.mkdtemp())
        self.globals.update(STATE=state, LEDGER=state / "instances.json")
        soon = time.time() + 600
        with GPU["ledger_txn"]() as d:
            d.update({f"{p}:i": {"provider": p, "deadline": soon, "launched_at": time.time(), "usd_per_hour": 1.0}
                      for p in ("eras", "slurm", "runpod")})
        cfg = {"budget": {"max_hours": 4}}
        for p in ("eras", "slurm"):  # --allow-over-budget lifts the caps, not what was fixed at launch
            with self.assertRaises(SystemExit, msg="the watchdog would move but the box would end at its old time"), \
                    contextlib.redirect_stderr(io.StringIO()):
                GPU["cmd_extend"](argparse.Namespace(instance=f"{p}:i", hours=1, allow_over_budget=True), cfg)
        extend = functools.partial(argparse.Namespace, allow_over_budget=False)
        with contextlib.redirect_stdout(io.StringIO()):
            GPU["cmd_extend"](extend(instance="runpod:i", hours=1), cfg)
            GPU["cmd_extend"](extend(instance="eras:i", hours=0), cfg)  # how a lost API key still terminates
        self.assertLess(GPU["ledger"]()["eras:i"]["deadline"], soon)
        self.assertGreater(GPU["ledger"]()["runpod:i"]["deadline"], soon)

    def test_no_launch_past_max_running_from_any_session(self):
        key, state, calls = Path(tempfile.mkdtemp()) / "key.pub", Path(tempfile.mkdtemp()), []
        key.write_text("ssh-ed25519 AAAA test")
        saved = {k: self.globals[k] for k in ("eras", "STATE", "LEDGER")}
        self.addCleanup(self.globals.update, saved)
        self.globals.update(
            slurm_run=fake_slurm({"sinfo": "node1 gpu:8 gpu:0 idle\n", "sbatch": "42\n",
                                  "squeue": "live-pr-3-9-1 41 RUNNING\njoe-live-pr-5-x 40 RUNNING\n"}, calls),
            eras=lambda *a, params=None, **k: (
                {"items": [{"id": "i1", "name": "live-pr-4-8-1", "status": "active"}]} if params.get("page_token")
                else {"items": [{"id": "i2", "name": "live-pr-2-1-1", "status": "ended"}], "next_page_token": "p2"}),
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
        self.addCleanup(self.globals.update, {"eras": self.globals["eras"]})
        self.globals["eras"] = eras
        self.assertEqual(GPU["eras_ssh_key_id"](f"ssh-ed25519 {key} me", "live-pr-1"), "k2",
                         "a key on a later page would be registered again on every launch")
        self.assertEqual(posts, [])

    def test_the_job_uses_the_shared_hf_cache(self):
        key = Path(tempfile.mkdtemp()) / "key.pub"
        key.write_text("ssh-ed25519 AAAA test")
        cfg = {**SLURM_CFG, "ssh_public_key": str(key), "budget": {"max_usd_per_hour": 0, "max_hours": 4},
               "slurm": [{**SLURM_CFG["slurm"][0], "hf_home": "/workspace/shared/hf"}]}
        self.globals["slurm_run"] = fake_slurm({"sinfo": "node1 gpu:8 gpu:0 idle\n"})
        args = argparse.Namespace(offer="slurm:node1:1", pr=7, hours=1, tag="1-1", run="job.sh",
                                  input=[], allow_over_budget=False, dry_run=True)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            GPU["cmd_launch"](args, cfg)
        self.assertIn("HF_HOME=/workspace/shared/hf", out.getvalue(),
                      "each run would download the weights again into its own directory")
        self.assertIn("-J live-pr-7-1-1 ", out.getvalue(), "gpu_review's cleanup finds the job by its --tag name")
        args.tag = None
        with contextlib.redirect_stdout(io.StringIO()) as out:
            GPU["cmd_launch"](args, cfg)
        self.assertRegex(out.getvalue(), r"-J live-pr-7-\d{4}-\d{6}-[0-9a-f]{4} ",
                         "a lost create's job is found by this name, so two launches must not share it")

    def test_offers_cover_every_node_and_the_freest_comes_first(self):
        """Node a has 2 GPUs free, b has 6, and c doesn't answer: offers lists b, then a, says why c is
        missing, and launching the first offer starts the job on b."""
        key, state, calls = Path(tempfile.mkdtemp()) / "key.pub", Path(tempfile.mkdtemp()), []
        key.write_text("ssh-ed25519 AAAA test")
        saved = {k: self.globals[k] for k in ("STATE", "LEDGER")}
        self.addCleanup(self.globals.update, saved)
        nodes = {"a": fake_slurm({"sinfo": "a1 gpu:8 gpu:6 mixed\n", "sbatch": "41\n"}),
                 "b": fake_slurm({"sinfo": "b1 gpu:8 gpu:2 mixed\n", "sbatch": "42\n"}),
                 "c": fake_slurm({"": GPU["ApiError"](255, "slurm: sinfo: ssh: connect to host c: refused")})}

        def slurm_run(sl, command, check=True):
            calls.append((sl["name"], command))
            return nodes[sl["name"]](sl, command, check)
        self.globals.update(slurm_run=slurm_run, STATE=state, LEDGER=state / "instances.json")
        cfg = {"slurm": [{"name": n, "host": f"root@{n}"} for n in nodes], "ssh_public_key": str(key),
               "eras": {"enabled": False}, "runpod": {"enabled": False}, "lambda": {"enabled": False},
               "budget": {"max_usd_per_hour": 0, "max_hours": 4}}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            GPU["cmd_offers"](argparse.Namespace(gpus=2, min_vram=80, all_regions=False, allow_arm=False, json=True),
                              cfg)
        listing = json.loads(out.getvalue())
        self.assertEqual([o["id"] for o in listing["offers"]], ["slurm:b:2", "slurm:a:2"],
                         "a live test would queue behind people's jobs on the busier node")
        self.assertEqual([e for e in listing["errors"] if e.startswith("slurm")],
                         ["slurm c: HTTP 255: slurm: sinfo: ssh: connect to host c: refused"],
                         "a node that is down would stop the others' offers, or vanish without a reason")
        args = argparse.Namespace(offer=listing["offers"][0]["id"], pr=7, hours=1, tag="1-1", run=None, input=[],
                                  cache="ro", allow_over_budget=False, dry_run=False)
        with unittest.mock.patch.object(subprocess, "Popen"), contextlib.redirect_stdout(io.StringIO()):
            GPU["cmd_launch"](args, cfg)
        self.assertEqual([n for n, c in calls if c.startswith("sbatch")], ["b"])

        # No offers: an unknown node makes that an error, in both modes; nodes merely full don't.
        full = fake_slurm({"sinfo": "x gpu:8 gpu:8 mixed\n"})
        for down, code in ((True, 1), (False, None)):
            nodes.update(a=full, b=full, **({} if down else {"c": full}))
            for as_json in (True, False):
                query = argparse.Namespace(gpus=2, min_vram=80, all_regions=False, allow_arm=False, json=as_json)
                with self.subTest(down=down, as_json=as_json), contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    try:
                        GPU["cmd_offers"](query, cfg)
                        exit_code = None
                    except SystemExit as e:
                        exit_code = e.code
                    self.assertEqual(exit_code, code, "a down node would read as 'no availability'")

    def test_a_lost_sbatch_is_found_or_ended_by_name(self):
        key, state, calls = Path(tempfile.mkdtemp()) / "key.pub", Path(tempfile.mkdtemp()), []
        key.write_text("ssh-ed25519 AAAA test")
        saved = {k: self.globals[k] for k in ("STATE", "LEDGER", "load_config")}
        self.addCleanup(self.globals.update, saved)
        cfg = {**SLURM_CFG, "ssh_public_key": str(key), "budget": {
            "max_usd_per_hour": 0, "max_hours": 4, "max_concurrent_instances": 9}}
        self.globals.update(STATE=state, LEDGER=state / "instances.json", load_config=lambda: cfg)
        args = argparse.Namespace(offer="slurm:node1:1", pr=7, hours=1, tag="t", run=None, input=[], cache="ro",
                                  allow_over_budget=False, dry_run=False)

        def launch(sbatch, squeue):
            calls.clear()
            self.globals["slurm_run"] = fake_slurm(
                {"sinfo": "node1 gpu:8 gpu:0 idle\n", "sbatch": sbatch, "squeue": squeue, "scancel": ""}, calls)
            with unittest.mock.patch.object(subprocess, "Popen"), unittest.mock.patch("time.sleep"), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    GPU["cmd_launch"](args, cfg)
                except SystemExit:
                    pass

        lost = GPU["ApiError"](255, "slurm: sbatch: ssh: connection reset")
        launch(lost, "42\n")
        self.assertEqual(list(GPU["ledger"]()), ["slurm:node1:42"], "the job would run on with no watchdog")
        self.assertEqual(GPU["ledger"]()["slurm:node1:42"]["instance_id"], "42")

        GPU["save_ledger"]({})
        with self.assertRaises(GPU["ApiError"]), unittest.mock.patch.object(subprocess, "Popen"), \
                contextlib.redirect_stdout(io.StringIO()):  # sbatch's own refusal made nothing
            launch(GPU["ApiError"](1, "slurm: sbatch: error: invalid partition"), "")
        self.assertEqual(GPU["ledger"](), {})

        launch(subprocess.TimeoutExpired("sbatch", 120), "")
        pending = "pending:live-pr-7-t"
        e = GPU["ledger"]()[pending]
        self.assertIsNone(e["instance_id"])
        self.assertIn("answer lost", GPU["status"](e, cfg)[0], "squeue -j None would call it ended")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            GPU["reachable"](pending)  # srun --jobid=None otherwise
        with contextlib.redirect_stdout(io.StringIO()):
            GPU["terminate"](pending, e)
        self.assertTrue(any("scancel --name=live-pr-7-t" in c for c in calls),
                        f"a job submitted with its answer lost would run to its --time: {calls}")
        self.assertEqual(GPU["ledger"](), {})


class TestEras(unittest.TestCase):
    def eras_launch(self, write=None, max_running=None, held=("v1",), refused_by="preview", dry_run=False, run=None,
                    inputs=()):
        """gpu.py launch on a fake Eras whose cache volumes, v1 and v2, look free, but the preview (or the
        create, `refused_by`) refuses those `held` (as the spec has it), as when another box took them in
        between; the calls it made."""
        key, state, calls = Path(tempfile.mkdtemp()) / "key.pub", Path(tempfile.mkdtemp()), []
        key.write_text("ssh-ed25519 AAAA test")
        digest = hashlib.sha256(base64.b64decode("AAAA")).digest()
        fingerprint = "SHA256:" + base64.b64encode(digest).decode().rstrip("=")
        keys = []
        tpl = {"endpoints": {"ssh": {"port": 22}}, "pod": {"containers": [
            {"resources": {"limits": {"nvidia.com/gpu": "1"}}, "volumeMounts": []}], "volumes": []}}

        def eras(method, path, body=None, params=None, retry=True):
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

    def test_a_launch_past_the_caps_registers_nothing(self):
        calls = self.eras_launch(max_running=0)
        self.assertEqual([c[:2] for c in calls if c[0] in ("POST", "exit")], [("exit", 4)],
                         "a refused launch would leave an ssh key in the organization")


if __name__ == "__main__":
    unittest.main()
