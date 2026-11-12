"""Check live-pr-test's gpu.py: the Eras Cloud, RunPod, Lambda and DigitalOcean providers, and what they share."""

import argparse
import base64
import contextlib
import datetime
import functools
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / ".claude/skills/live-pr-test/scripts"))
import gpu as gpu_module  # noqa: E402

GPU = vars(gpu_module)  # patched in place, with each test's cleanup restoring it


def keys(*names):
    """The environment with only these providers' API keys set, so no test reaches a real provider."""
    return unittest.mock.patch.dict(os.environ, {cls.key_env: "k" if cls.name in names else ""
                                                 for cls in GPU["PROVIDERS"].values()})


class TestEras(unittest.TestCase):
    def eras_launch(self, write=None, max_running=None, held=("v1",), refused_by="preview", dry_run=False, run=None,
                    inputs=(), gone=(), boxes=1):
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

        saved = {k: GPU[k] for k in ("eras", "find_offer", "STATE", "LEDGER")}
        self.addCleanup(GPU.update, saved)
        GPU.update(
            eras=eras, STATE=state, LEDGER=state / "instances.json", find_offer=lambda cfg, offer: {
                "id": "eras:l1", "provider": "eras", "gpu": "H200", "count": 1, "usd_per_hour": 3.0, "region": "x",
                "storage": True})
        cfg = {"ssh_public_key": str(key), "eras": {"cache_volume_ids": ["v1", "v2"]},
               "budget": {"max_usd_per_hour": 30, "max_hours": 4, "max_running": max_running}}
        args = argparse.Namespace(boxes=boxes, offer="eras:l1", pr=0, hours=1, tag="c", run=run, input=list(inputs),
                                  cache="ro", cache_write=write, allow_over_budget=False, dry_run=dry_run)
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
        posts, globs = [], GPU
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

        globs = GPU
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
        globs = GPU
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

    def test_max_running_counts_this_tools_live_instances_on_every_page_of_every_provider(self):
        globs = GPU
        self.addCleanup(globs.update, {k: globs[k] for k in ("eras", "runpod")})
        globs["eras"] = lambda *a, params=None, **k: (
            {"items": [{"id": "i1", "name": "live-pr-4-8-1", "status": "active"},
                       {"id": "i3", "name": "someone-else", "status": "active"}]} if params.get("page_token")
            else {"items": [{"id": "i2", "name": "live-pr-2-1-1", "status": "ended"}], "next_page_token": "p2"})
        globs["runpod"] = lambda *a, **k: {"pods": [{"id": "p1", "name": "live-pr-5-x", "status": "RUNNING"}]}
        with keys("eras", "runpod"):
            up = GPU["running"]({"eras": {}})
        self.assertEqual([line.split()[0] for line in up], ["eras:i1", "runpod:p1"],
                         "every live test would hold its own box, past budget.max_running")

        def down(*a, **k):
            raise GPU["ApiError"](503, "down")
        globs["runpod"] = down
        with keys("eras", "runpod"), self.assertRaises(GPU["ApiError"], msg="a provider it can't list would count 0"):
            GPU["running"]({"eras": {}})

    def test_offers_are_of_the_vendor_asked_for_and_eras_offers_no_amd(self):
        globs = GPU
        self.addCleanup(globs.update, {"eras": globs["eras"]})
        listing = [{"id": f"l{i}", "hourly_price": 3.0, "launches_available": 1,
                    "access_method": {"is_available": True},
                    "hardware": {"gpu": {"model": model, "count": 1, "memory_per_device": "192Gi"}}}
                   for i, model in enumerate(("H200", "MI300X", "A100"))]
        globs["eras"] = lambda *a, **k: {"items": listing}

        def offers(vendor):
            printed = io.StringIO()
            args = argparse.Namespace(gpus=1, vendor=vendor, min_vram=80, all_regions=False, allow_arm=False, json=True)
            with keys("eras"), contextlib.redirect_stdout(printed):
                GPU["cmd_offers"](args, {"budget": {"max_usd_per_hour": 30}})
            return [(o["id"], o["vendor"]) for o in json.loads(printed.getvalue())["offers"]]

        self.assertEqual(offers("nvidia"), [("eras:l0", "nvidia")], "an A100 or an AMD box would pass for Hopper")
        self.assertEqual(offers("amd"), [], "an AMD box would get the Eras template, which asks for nvidia.com/gpu")

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
        globs = GPU
        self.addCleanup(globs.update, {"eras": globs["eras"]})
        globs["eras"] = eras
        self.assertEqual(GPU["eras_ssh_key_id"](f"ssh-ed25519 {key} me", "live-pr-1"), "k2",
                         "a key on a later page would be registered again on every launch")
        self.assertEqual(posts, [])

    def test_an_error_is_cut_at_whole_lines(self):
        text = "x" * 300 + "\n" + "y" * 300 + "\nlast line\n"
        self.assertEqual(GPU["tail_lines"](text, 400), "y" * 300 + "\nlast line",
                         "a provider's error would be cut mid-word")

    def test_a_launch_past_the_caps_registers_nothing(self):
        calls = self.eras_launch(max_running=0)
        self.assertEqual([c[:2] for c in calls if c[0] in ("POST", "exit")], [("exit", 4)],
                         "a refused launch would leave an ssh key in the organization")
        self.assertIn(("POST", "/instances"), [c[:2] for c in self.eras_launch(max_running=1, held=())],
                      "a launch under budget.max_running would be refused")
        self.assertIn(("exit", 4), [c[:2] for c in self.eras_launch(max_running=1, held=(), boxes=2)],
                      "the first of a run's two boxes would start with room for one, and the second be refused")
        self.assertEqual([c[:2] for c in self.eras_launch(max_running=0, held=(), boxes=0) if c[0] != "GET"],
                         [("exit", 1)], "--boxes 0 would launch past budget.max_running")


def do_size(slug, model, count, vram, price, regions=("nyc2", "tor1"), available=True):
    """A GPU size as DigitalOcean's GET /v2/sizes lists it (the spec's size and gpu_info)."""
    return {"slug": slug, "memory": 245760 * count, "vcpus": 20 * count, "disk": 720, "transfer": 15.0,
            "price_monthly": round(price * 730, 2), "price_hourly": price, "regions": list(regions),
            "available": available, "description": f"{model} GPU - {count}X",
            "disk_info": [{"type": "local", "size": {"amount": 720, "unit": "gib"}},
                          {"type": "scratch", "size": {"amount": 5120, "unit": "gib"}}],
            "gpu_info": {"count": count, "vram": {"amount": vram, "unit": "gib"}, "model": model}}


DO_SIZES = [  # two pages, as the API pages them
    {"sizes": [{"slug": "s-1vcpu-1gb", "memory": 1024, "vcpus": 1, "disk": 25, "transfer": 1.0, "price_monthly": 6,
                "price_hourly": 0.00893, "regions": ["nyc1"], "available": True, "description": "Basic"},
               do_size("gpu-h100x1-80gb", "nvidia_h100", 1, 80, 3.39),
               do_size("gpu-h100x8-640gb", "nvidia_h100", 8, 80, 23.92),
               do_size("gpu-h200x1-141gb", "nvidia_h200", 1, 141, 3.44, regions=()),  # "available", in no region
               do_size("gpu-l40sx1-48gb", "nvidia_l40s", 1, 48, 1.57)],
     "links": {"pages": {"last": "https://api.digitalocean.com/v2/sizes?page=2",
                         "next": "https://api.digitalocean.com/v2/sizes?page=2"}}, "meta": {"total": 10}},
    {"sizes": [do_size("gpu-mi300x1-192gb", "amd_mi300x", 1, 192, 1.99, regions=()),  # likewise
               do_size("gpu-h200x8-1128gb-contracted", "nvidia_h200", 8, 141, 27.52),
               do_size("gpu-h100x1-80gb-spot", "nvidia_h100", 1, 80, 1.99),
               do_size("gpu-b300x1-288gb-lc-spot", "nvidia_b300", 1, 288, 6.17),
               do_size("gpu-mi300x8-1536gb", "amd_mi300x", 8, 192, 15.92)],
     "links": {"pages": {"first": "https://api.digitalocean.com/v2/sizes?page=1",
                         "prev": "https://api.digitalocean.com/v2/sizes?page=1"}}, "meta": {"total": 10}},
]


def ago(hours):
    """A time `hours` ago as DigitalOcean writes created_at."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - hours * 3600))


class TestDigitalOcean(unittest.TestCase):
    def setUp(self):
        self.calls, self.keys = [], []
        self.droplets = [{"id": 3164444, "name": "live-pr-7-1-1-amd", "status": "active", "created_at": ago(1),
                          "networks": {"v4": [{"ip_address": "10.0.0.2", "type": "private"},
                                              {"ip_address": "104.236.32.182", "type": "public"}]}},
                         {"id": 3164445, "name": "someone-else", "status": "active", "created_at": ago(1)},
                         {"id": 3164446, "name": "live-pr-6-x", "status": "archive", "created_at": ago(1)}]
        saved = {k: GPU[k] for k in ("digitalocean", "STATE", "LEDGER", "find_offer")}
        self.addCleanup(GPU.update, saved)
        state = Path(tempfile.mkdtemp())
        GPU.update(digitalocean=self.api, STATE=state, LEDGER=state / "instances.json")
        env = keys("digitalocean")
        env.start()
        self.addCleanup(env.stop)

    def api(self, method, path, body=None, params=None, retry=True):
        self.calls.append((method, path, body, params))
        if path == "/sizes":
            return DO_SIZES[params["page"] - 1]
        if path == "/droplets" and method == "GET":
            return {"droplets": self.droplets, "links": {}, "meta": {"total": len(self.droplets)}}
        if path == "/droplets":
            return {"droplet": {"id": 3164447, "name": body["name"], "status": "new"}, "links": {"actions": []}}
        if method == "DELETE":
            self.droplets = [d for d in self.droplets if f"/droplets/{d['id']}" != path]
            return {}
        if path.startswith("/droplets/"):
            found = [d for d in self.droplets if str(d["id"]) == path.split("/")[-1]]
            if not found:
                raise GPU["ApiError"](404, {"id": "not_found", "message": "The resource you requested could not be "
                                                                          "found."})
            return {"droplet": found[0]}
        if path.startswith("/account/keys/"):
            if not self.keys:
                raise GPU["ApiError"](404, {"id": "not_found", "message": "not found"})
            return {"ssh_key": self.keys[0]}
        if path == "/account/keys":
            self.keys.append({"id": 512189, "fingerprint": "f", "public_key": body["public_key"], "name": body["name"]})
            return {"ssh_key": self.keys[-1]}
        raise AssertionError(f"unexpected {method} {path}")

    def offers(self, vendor, gpus=1, all_regions=False):
        printed = io.StringIO()
        args = argparse.Namespace(gpus=gpus, vendor=vendor, min_vram=80, all_regions=all_regions, allow_arm=False,
                                  json=True)
        with contextlib.redirect_stdout(printed):
            GPU["cmd_offers"](args, {"budget": {"max_usd_per_hour": 30}})
        return {o["id"]: (o["vendor"], o["gpu"], o["count"], o["usd_per_hour"], o["availability"])
                for o in json.loads(printed.getvalue())["offers"]}

    def test_offers_are_its_self_serve_gpu_sizes_by_vendor(self):
        self.assertEqual(self.offers("nvidia"), {
            "digitalocean:gpu-h100x1-80gb": ("nvidia", "NVIDIA H100", 1, 3.39, "available"),
            "digitalocean:gpu-h100x8-640gb": ("nvidia", "NVIDIA H100", 8, 23.92, "available")},
            "a size under 80 GB, sold only under contract or as spot, or in no region would be offered")
        self.assertEqual(self.offers("amd"), {
            "digitalocean:gpu-mi300x8-1536gb": ("amd", "AMD MI300X", 8, 15.92, "available")},
            "a size listed available but in no region would be offered, and its create refused")
        self.assertEqual(list(self.offers("amd", gpus=2)), ["digitalocean:gpu-mi300x8-1536gb"])
        self.assertEqual(self.offers("nvidia", all_regions=True)["digitalocean:gpu-h200x1-141gb"][-1], "none",
                         "--all-regions would hide a size a person may want to see")
        with self.assertRaises(SystemExit, msg="a size whose regions emptied since the listing would be created"):
            GPU["find_offer"]({}, "digitalocean:gpu-mi300x1-192gb")

    def launch(self, run, inputs=(), dry_run=False, offer="digitalocean:gpu-mi300x1-192gb", gpu="AMD MI300X",
               count=1, cfg=None):
        """cmd_launch of `offer`; the create's body, or None when nothing was created."""
        key = Path(tempfile.mkdtemp()) / "key.pub"
        key.write_text("ssh-ed25519 AAAA test")
        GPU["find_offer"] = lambda cfg, oid: {"id": offer, "provider": "digitalocean", "gpu": gpu, "count": count,
                                              "usd_per_hour": 1.99, "region": "ric1", "storage": True}
        args = argparse.Namespace(boxes=1, offer=offer, pr=7, hours=1, tag="1-1-amd", run=run, input=list(inputs),
                                  allow_over_budget=False, dry_run=dry_run)
        cfg = {"ssh_public_key": str(key), "budget": {"max_usd_per_hour": 30, "max_hours": 4, "max_running": 4},
               **(cfg or {})}
        with unittest.mock.patch.object(subprocess, "Popen"), contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            GPU["cmd_launch"](args, cfg)
        self.printed = out.getvalue()
        return next((c[2] for c in self.calls if c[:2] == ("POST", "/droplets")), None)

    def test_a_run_is_written_and_started_by_cloud_init(self):
        d = Path(tempfile.mkdtemp())
        box, body = d / "box.sh", d / "pr-body.md"
        box.write_text("echo hello\n")
        body.write_text("Generated with Claude Code \N{ROBOT FACE}\n", encoding="utf-8")
        cfg = {"run_secret_env": "BOX_KEY"}
        with unittest.mock.patch.dict(os.environ, {"BOX_KEY": "sk-ant-box"}):
            self.launch(str(box), [str(body)], dry_run=True, cfg=cfg)
            self.assertEqual([c[:2] for c in self.calls if c[0] != "GET"], [], "a dry run would register the key")
            self.assertNotIn("sk-ant-box", self.printed, "a dry run would print the box's key")
            created = self.launch(str(box), [str(body)], cfg=cfg)
        self.assertEqual(created["image"], "gpu-amd-base", "an AMD box would boot an image with no ROCm")
        self.assertEqual(created["region"], "ric1", "the create would land where the size isn't, and be refused")
        self.assertEqual(created["ssh_keys"], ["f"], "the box would get a root password, not the runner's key")
        header, config = created["user_data"].split("\n", 1)
        self.assertEqual(header, "#cloud-config")
        config = json.loads(config)
        files = {f["path"]: gzip.decompress(base64.b64decode(f["content"])) for f in config["write_files"]}
        self.assertEqual(files, {"/run/live-pr/box.sh": box.read_bytes(), "/run/live-pr/pr-body.md": body.read_bytes(),
                                 "/run/live-pr-key/value": b"sk-ant-box"},
                         "the box would read changed inputs, or have no Anthropic key")
        self.assertEqual({f["permissions"] for f in config["write_files"]}, {"0600"})
        self.assertEqual(config["runcmd"], [["systemd-run", "--unit", "live-pr", "bash", "/run/live-pr/box.sh"]],
                         "the droplet would boot and bill with nothing running")
        self.assertEqual(list(GPU["ledger"]()), ["digitalocean:3164447"])

        self.calls.clear()
        GPU["save_ledger"]({})
        created = self.launch(str(box), offer="digitalocean:gpu-h100x8-640gb", gpu="NVIDIA H100", count=8)
        self.assertEqual(created["image"], "gpu-h100x8-base")
        self.assertNotIn(("POST", "/account/keys"), [c[:2] for c in self.calls], "the key would be registered twice")
        self.assertNotIn("/run/live-pr-key/value", created["user_data"])
        big = d / "big.md"
        big.write_bytes(os.urandom(60 * 1024))
        self.calls.clear()
        GPU["save_ledger"]({})
        with self.assertRaises(SystemExit, msg="DigitalOcean would refuse the create, or cut the box's files"):
            self.launch(str(box), [str(big)])
        self.assertIsNone(next((c for c in self.calls if c[:2] == ("POST", "/droplets")), None))

    def test_this_tools_gpu_droplets_count_and_a_deleted_one_has_ended(self):
        with keys("digitalocean"):
            self.assertEqual(GPU["running"]({}), ["digitalocean:3164444  live-pr-7-1-1-amd  active"],
                             "the bot's AMD boxes would run past budget.max_running")
        listed = [c[3] for c in self.calls if c[:2] == ("GET", "/droplets")]
        self.assertTrue(listed and all(params.get("type") == "gpus" for params in listed),
                        "GPU droplets aren't in DigitalOcean's default listing")
        e = {"provider": "digitalocean", "instance_id": "3164444", "region": "any"}
        self.assertEqual(GPU["status"](e, {}), ("active", "104.236.32.182", 22))
        self.assertEqual(GPU["status"]({**e, "instance_id": "9"}, {})[0], "ended",
                         "the bot would wait on a deleted droplet until its timeout")

    def test_reap_releases_only_this_tools_boxes_past_their_deadline(self):
        self.droplets = [{"id": 1, "name": "live-pr-7-1-1-amd", "status": "active", "created_at": ago(6)},  # runaway
                         {"id": 2, "name": "live-pr-8-2-1-amd", "status": "active", "created_at": ago(4.5)},
                         {"id": 3, "name": "someone-else", "status": "active", "created_at": ago(30)},
                         {"id": 4, "name": "live-pr-9-3-1-amd", "status": "active", "created_at": ago(0.5)}]
        past = {"launched_at": time.time() - 1800, "deadline": time.time() - 60, "usd_per_hour": 1.99}
        GPU["save_ledger"]({"digitalocean:4": {"provider": "digitalocean", "instance_id": "4",
                                               "name": "live-pr-9-3-1-amd", **past},
                            # its Pod's deadline ends it; reap must leave it alone even past the ledger's
                            "eras:e1": {"provider": "eras", "instance_id": "e1", "name": "live-pr-5-x", **past}})
        self.addCleanup(GPU.update, {"eras": GPU["eras"]})
        eras_calls = []
        GPU["eras"] = lambda method, path, *a, **k: eras_calls.append((method, path)) or {
            "items": [{"id": "e1", "name": "live-pr-5-x", "status": "active"}]}
        with keys("digitalocean", "eras"), contextlib.redirect_stderr(io.StringIO()):
            created = self.launch(None, cfg={"budget": {"max_usd_per_hour": 30, "max_hours": 4, "max_running": 4,
                                                        "max_concurrent_instances": 3}})
        self.assertEqual(sorted(d["name"] for d in self.droplets), ["live-pr-8-2-1-amd", "someone-else"],
                         "a box whose runner died would bill until someone noticed, or a live or foreign one would go")
        self.assertIsNotNone(created, "runaway boxes would hold budget.max_running")
        self.assertEqual([c for c in eras_calls if c[0] != "GET"], [],
                         "an Eras box, which its Pod's deadline ends, would be released early")


def storage_free_offers(provider, calls, keys=()):
    """The fake API of `provider` (runpod or lambda) whose team storage is not to be asked about: every call it
    makes is in `calls`."""
    def runpod(method, path, body=None, params=None, retry=True):
        calls.append((method, path, body))
        if path == "/catalog/gpus":
            return {"gpus": [{"id": "NVIDIA H100 80GB HBM3", "memory": 80, "price": {"secure": 2.69}, "dataCenters": [
                {"id": "US-CA-2", "availability": "HIGH"}, {"id": "EU-RO-1", "availability": "NONE"}]}]}
        if path == "/pods" and method == "POST":
            return {"id": "p1"}
        raise AssertionError(f"unexpected {method} {path}")

    def lambda_(method, path, body=None, retry=True):
        calls.append((method, path, body))
        return {"/ssh-keys": list(keys), "/instance-operations/launch": {"instance_ids": ["i-1"]},
                "/instance-types": {"gpu_1x_h100": {"regions_with_capacity_available": [{"name": "us-west-1"}],
                                                    "instance_type": {"gpu_description": "H100 (80 GB SXM5)",
                                                                      "specs": {"gpus": 1},
                                                                      "price_cents_per_hour": 329}}}}[path]
    return {"runpod": runpod, "lambda": lambda_}[provider]


class TestStorageFreeProviders(unittest.TestCase):
    """RunPod and Lambda for a box that keeps nothing (`storage: false`), as the review bot uses them."""

    def provider(self, name, cfg, keys=()):
        self.calls = []
        func = {"runpod": "runpod", "lambda": "lambda_"}[name]
        self.addCleanup(GPU.update, {func: GPU[func]})
        GPU[func] = storage_free_offers(name, self.calls, keys)
        return GPU[{"runpod": "RunPod", "lambda": "Lambda"}[name]]({name: {"storage": False, **cfg}})

    def test_offers_cover_every_region_with_capacity_and_ask_for_no_storage(self):
        for name, region in (("runpod", "US-CA-2"), ("lambda", "us-west-1")):
            with self.subTest(name):
                offers = self.provider(name, {}).offers(1, 0, False)
                self.assertEqual([(o["region"], o["storage"]) for o in offers], [(region, True)],
                                 "a bot box would wait for a region with the team's storage")
                self.assertEqual([c for c in self.calls if "volume" in c[1] or "filesystem" in c[1]], [],
                                 "the bot's account has no volumes to ask about")

    def test_lambda_registers_its_key_by_fingerprint_once_and_mounts_nothing(self):
        pubkey = "ssh-ed25519 " + base64.b64encode(b"key").decode()
        name = "live-pr-" + hashlib.sha256(b"key").hexdigest()[:12]
        for listed in ([], [{"name": name}]):
            with self.subTest(registered=bool(listed)):
                p = self.provider("lambda", {"filesystems": ["team"]}, listed)
                body, _ = p.plan({"id": "lambda:gpu_1x_h100:us-west-1", "region": "us-west-1"}, "live-pr-7-t", pubkey,
                                 1, argparse.Namespace(dry_run=False, run=None))
                self.assertNotIn("file_system_names", body, "a bot box would mount the team's filesystem")
                self.assertEqual(body["ssh_key_names"], [name])
                self.assertEqual([c for c in self.calls if c[:2] == ("POST", "/ssh-keys")],
                                 [] if listed else [("POST", "/ssh-keys", {"name": name, "public_key": pubkey})],
                                 "the key would be registered again on every launch, or never")

    def test_a_pod_mounts_nothing_and_a_lambda_run_goes_in_user_data(self):
        d = Path(tempfile.mkdtemp())
        box = d / "box.sh"
        box.write_text("echo hello\n")
        cfg = {"run_secret_env": "BOX_KEY"}
        args = argparse.Namespace(dry_run=False, run=str(box), input=[])
        with unittest.mock.patch.dict(os.environ, {"BOX_KEY": "sk-ant-box"}):
            body, _ = self.provider("runpod", {"image": "img", "hf_home": "/h", **cfg}).plan(
                {"gpu": "H", "count": 1, "region": "US-CA-2"}, "live-pr-7-t", "ssh-ed25519 AAAA", 1, args)
            self.assertNotIn("mounts", body, "a pod would mount a volume the bot's account lacks")
            self.assertNotIn("sk-ant-box", json.dumps(body), "the key would reach RunPod's API")
            self.assertNotIn("user_data", body)
            lam = self.provider("lambda", {"ssh_key_name": "k"})
            lam.cfg.update(cfg)
            body, _ = lam.plan({"id": "lambda:gpu_1x_h100:us-west-1", "region": "us-west-1"}, "live-pr-7-t",
                               "ssh-ed25519 AAAA", 1, args)
        files = {f["path"]: gzip.decompress(base64.b64decode(f["content"]))
                 for f in json.loads(body["user_data"].split("\n", 1)[1])["write_files"]}
        self.assertEqual(files, {"/run/live-pr/box.sh": b"echo hello\n", "/run/live-pr-key/value": b"sk-ant-box"},
                         "a Lambda box would have no script to run, or no Anthropic key")

    def test_a_pods_run_is_copied_over_ssh_and_a_pod_that_cant_take_it_is_ended(self):
        d = Path(tempfile.mkdtemp())
        box, extra = d / "box.sh", d / "pr-body.md"
        box.write_text("echo hello\n")
        extra.write_text("body\n")
        state, key = Path(tempfile.mkdtemp()), d / "key.pub"
        key.write_text("ssh-ed25519 AAAA test")
        saved = {k: GPU[k] for k in ("runpod", "find_offer", "STATE", "LEDGER", "cmd_wait", "terminate")}
        self.addCleanup(GPU.update, saved)
        sent, ended, created = [], [], []

        def runpod(method, path, body=None, params=None, retry=True):
            if method == "POST":
                created.append(body)
            return {"id": "p1"}

        def fake_run(command, **kwargs):
            sent.append((command, Path(command[-2]).read_text() if "value" in command[-2:][0] else None))
            return subprocess.CompletedProcess(command, 1 if self.ssh_fails else 0, "", "boom")

        def cmd_wait(args, cfg):
            if self.api_down:
                raise GPU["ApiError"](503, "x")
            e = GPU["ledger"]()[args.instance]
            with GPU["ledger_txn"]() as ledger:
                ledger[args.instance].update(host="1.2.3.4", port=2222)
            self.assertEqual(e["instance_id"], "p1")

        GPU.update(runpod=runpod, STATE=state, LEDGER=state / "instances.json", cmd_wait=cmd_wait,
                   find_offer=lambda cfg, offer: {"id": "runpod:H100:dc:1", "provider": "runpod", "gpu": "H100",
                                                  "count": 1, "usd_per_hour": 3.0, "region": "dc", "storage": True},
                   terminate=lambda k, e: ended.append(k))
        cfg = {"ssh_public_key": str(key), "run_secret_env": "BOX_KEY", "runpod": {"image": "i", "hf_home": "/h",
                                                                                       "storage": False},
               "budget": {"max_usd_per_hour": 30, "max_hours": 4, "max_running": None}}
        args = argparse.Namespace(boxes=1, offer="runpod:H100:dc:1", pr=7, hours=1, tag="t", run=str(box),
                                  input=[str(extra)], allow_over_budget=False, dry_run=False)
        for self.ssh_fails, self.api_down in ((False, False), (True, False), (False, True)):
            sent.clear(), ended.clear(), created.clear()
            with self.subTest(ssh_fails=self.ssh_fails, api_down=self.api_down), \
                    unittest.mock.patch.object(subprocess, "Popen"), \
                    unittest.mock.patch.object(subprocess, "run", fake_run), \
                    unittest.mock.patch.dict(os.environ, {"RUNPOD_API_KEY": "k", "BOX_KEY": "sk-ant-box"}), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                GPU["save_ledger"]({})
                try:
                    GPU["cmd_launch"](args, cfg)
                except (SystemExit, GPU["ApiError"]):
                    pass
            self.assertNotIn("sk-ant-box", json.dumps(created), "the key would reach RunPod's API")
            self.assertEqual(ended, ["runpod:p1"] if self.ssh_fails or self.api_down else [],
                             "a pod without its run would bill idle, or a started one would be ended")
            if not (self.ssh_fails or self.api_down):
                self.assertEqual(sent[-1][0][-1].split()[:3], ["setsid", "nohup", "bash"], "the run would not start")
                self.assertEqual(sent[-2][1], "sk-ant-box", "the box would have no Anthropic key")
                self.assertIn(str(extra), sent[1][0], "the box would miss its inputs")

    def test_a_deleted_box_has_ended_and_creation_times_are_read(self):
        created = "2026-11-12T10:00:00Z"
        replies = {"runpod": {"pods": [{"id": "p1", "name": "live-pr-1-t", "status": "RUNNING", "createdAt": created}]},
                   "lambda": [{"id": "i-1", "name": "live-pr-1-t", "status": "active", "first_healthy": created},
                              {"id": "i-2", "name": "live-pr-2-t", "status": "booting"}]}
        for name, cls in (("runpod", "RunPod"), ("lambda", "Lambda")):
            with self.subTest(name):
                func = {"runpod": "runpod", "lambda": "lambda_"}[name]
                self.addCleanup(GPU.update, {func: GPU[func], "runpod_list": GPU["runpod_list"]})

                def api(method, path, body=None, params=None, retry=True, name=name):
                    if path.startswith(("/pods/", "/instances/")):
                        raise GPU["ApiError"](404, {"message": "gone"})
                    return replies[name]
                GPU[func] = api
                GPU["runpod_list"] = lambda path, field, params=None: replies["runpod"]["pods"]
                p = GPU[cls]({})
                self.assertEqual(p.status({"instance_id": "x"})[0], "ended",
                                 "the bot would wait on a deleted box until its timeout")
                epoch = datetime.datetime(2026, 11, 12, 10, tzinfo=datetime.timezone.utc).timestamp()
                self.assertEqual([i.created for i in p.instances()][0], epoch,
                                 "the hourly reap could never tell a runaway box from a new one")


if __name__ == "__main__":
    unittest.main()
