#!/usr/bin/env python3
"""Provision GPUs for live PR tests (NVIDIA Hopper or newer, AMD MI300X or newer): Eras Cloud, RunPod, Lambda
or DigitalOcean.

usage:
  gpu.py check                                   keys, persistent storage, budget
  gpu.py offers --gpus N [--vendor nvidia|amd] [--min-vram GB] [--all-regions] [--allow-arm] [--json]
  gpu.py launch <offer-id> --pr N [--hours H] [--tag T] [--run SCRIPT [--input FILE ...]] [--boxes N]
               [--cache ro|none | --cache-write ID] [--allow-over-budget] [--dry-run]
  gpu.py wait <instance>                         until ssh answers; prints how to reach it
  gpu.py ssh <instance> [command ...]            run a command (or a shell) on it
  gpu.py push <instance> <local> <remote>        scp -r to the box
  gpu.py pull <instance> <remote> <local>        scp -r from the box
  gpu.py list [--all]                            instances launched here (--all: this tool's, by any session)
  gpu.py extend <instance> --hours H [--allow-over-budget]
                                                 move the watchdog deadline to now + H (Eras: earlier only)
  gpu.py terminate <instance>
  gpu.py reap                                    release this tool's runaway boxes (below), from any session

Offers only cover regions where the team has persistent storage (a RunPod global volume
mounts anywhere; a RunPod network volume or a Lambda filesystem pins one region), since the
HF cache lives there; --all-regions lists the rest for
a "no availability" report. Prices are USD/hr for the whole box.

An Eras instance runs sshd as root in a Pod that Kubernetes stops at the deadline
(`activeDeadlineSeconds`, `restartPolicy: Never`); a run that ends releases it.

A DigitalOcean GPU Droplet is a VM, root over ssh, NVIDIA or AMD by size, billed until deleted:
nothing on DigitalOcean's side ends it.

`launch --run SCRIPT` makes the script the work itself rather than holding the GPUs for ssh, with
the --input files in /run/live-pr. On Eras it is the Pod's command, with sshd up beside it for
copying results back, and the instance is released 15 minutes after it exits, so the GPUs go when
the script ends, whether or not whoever launched it is still there. On DigitalOcean cloud-init
writes the files and starts the script at boot (in the droplet's user_data, 64 KiB at most); the
droplet stays up until terminated.

Every launch is recorded in a ledger and gets a detached watchdog that terminates it at
its deadline (budget.max_hours by default), even if the session that launched it is gone.
`reap`, which every launch under budget.max_running runs first, is the backstop for a box whose
watchdog's machine is gone too: on a provider whose instances have no deadline of their own, it
releases this tool's instances past their ledger deadline, or past budget.max_hours plus an hour
from their creation when this machine's ledger doesn't have them.

`launch` exits 4 when budget.max_running refuses it, 1 on any other error.

Keys come from RUNPOD_API_KEY / LAMBDA_API_KEY / ERAS_API_KEY / DIGITALOCEAN_TOKEN. Settings from config.local.json
next to the skill (see config.example.json). Stdlib only, so it runs from any python3.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import datetime
import fcntl
import functools
import gzip
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.client import HTTPException  # the module name is taken by http() below
from pathlib import Path
from typing import NamedTuple

SKILL_DIR = Path(__file__).resolve().parent.parent
STATE = Path(os.environ.get("LIVE_PR_STATE", "~/.local/state/mstar-live-pr-test")).expanduser()
LEDGER = STATE / "instances.json"
RUNPOD = "https://api.runpod.io/v2"
RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"
LAMBDA = "https://cloud.lambda.ai/api/v1"
ERAS = "https://api.erascloud.ai"
DIGITALOCEAN = "https://api.digitalocean.com/v2"
# The GPUs this tool rents, by vendor: NVIDIA Hopper or newer (\b keeps out the H20, Hopper silicon with a
# fraction of the compute), and AMD MI300X or newer.
SUPPORTED = {"nvidia": re.compile(r"\b(H100|H200|GH200|B200|B300)\b", re.I),
             "amd": re.compile(r"\bMI3[0-5]\d?X\b", re.I)}
JOB_PREFIX = "live-pr-"  # what the names of this tool's instances start with


def vendor_of(gpu: str) -> str | None:
    """The vendor of a GPU this tool rents, by the GPU's name; None for any other GPU."""
    return next((vendor for vendor, names in SUPPORTED.items() if names.search(gpu)), None)


def die(msg: str, code: int = 1) -> None:
    print(f"gpu.py: {msg}", file=sys.stderr)
    sys.exit(code)


def load_config() -> dict:
    path = SKILL_DIR / "config.local.json"
    if not path.exists():
        die(f"no {path}; copy config.example.json and fill it in")
    return json.loads(path.read_text())


# ---------------------------------------------------------------- HTTP

class ApiError(Exception):
    def __init__(self, status: int, body: dict | str):
        self.status, self.body = status, body
        super().__init__(f"HTTP {status}: {body}")

    def code(self) -> str:  # the machine-readable error code, if any: Lambda's under "error", Eras's at the top
        if not isinstance(self.body, dict):
            return ""
        return self.body.get("code") or (self.body.get("error") or {}).get("code", "")


def http(method: str, url: str, key_env: str, body: dict | None = None, params: dict | None = None,
         retry: bool = True, key_header: str = "Authorization", key_prefix: str = "Bearer "):
    """``retry=False`` is for a call that must not run twice (a create): it is retried only on a
    429, which the provider refused before acting on it. Any other failure, a 5xx, a dropped
    connection or a timeout, may have come after the provider acted, so it raises at once with
    ``status`` -1 or the 5xx, and the caller finds out what happened."""
    key = os.environ.get(key_env)
    if not key:
        raise ApiError(0, f"{key_env} is not set")
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    last = ApiError(-1, f"{method} {url}: gave up after retries")
    for attempt in range(6):
        req = urllib.request.Request(url, data=data, method=method, headers={
            key_header: f"{key_prefix}{key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "mstar-live-pr-test/1",  # Cloudflare in front of Lambda rejects urllib's default
        })
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            try:
                return json.loads(raw) if raw else {}
            except ValueError:  # a proxy's error page: the request may still have been acted on
                raise ApiError(-1, f"{method} {url}: a reply that is not JSON: {raw[:200]!r}") from None
        except urllib.error.HTTPError as e:
            try:
                raw = e.read().decode(errors="replace")
            except (OSError, HTTPException):  # the error body itself was cut off
                raw = ""
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = raw[:500]
            if e.code == 429 or (retry and e.code >= 500):
                last = ApiError(e.code, parsed)
                time.sleep(float(e.headers.get("Retry-After") or 2 ** attempt * 3))
                continue
            raise ApiError(e.code, parsed) from None
        # every transport failure: URLError, a read timeout, a reset or dropped connection
        except (OSError, HTTPException) as e:
            if not retry:
                raise ApiError(-1, f"{method} {url}: {e!r}") from None
            last = ApiError(-1, f"{method} {url}: {e!r}")
            time.sleep(2 ** attempt * 3)
    raise last


def runpod(method, path, body=None, params=None, retry=True):
    return http(method, RUNPOD + path, "RUNPOD_API_KEY", body, params, retry)


def runpod_list(path: str, field: str, params: dict | None = None) -> list:
    out, params = [], dict(params or {})
    while True:
        page = runpod("GET", path, params=params)
        if isinstance(page, list):  # tolerate an unwrapped list (v2 is beta)
            return out + page
        out += page.get(field) or page.get("data") or page.get("items") or []
        nxt = (page.get("pagination") or {}).get("nextCursor")
        if not nxt or not (page.get("pagination") or {}).get("hasNextPage"):
            return out
        params["cursor"] = nxt


def lambda_(method, path, body=None, retry=True):
    return http(method, LAMBDA + path, "LAMBDA_API_KEY", body, retry=retry).get("data")


_eras_org: str | None = None


def eras(method, path, body=None, params=None, retry=True):
    """Eras Cloud, under the key's default organization."""
    global _eras_org
    def call(m, url, b=None, q=None, again=True):
        return http(m, ERAS + url, "ERAS_API_KEY", b, q, again, "x-api-key", "")
    _eras_org = _eras_org or call("GET", "/v1/me")["default_organization"]["id"]
    return call(method, f"/v1/org/{_eras_org}{path}", body, params, retry)


def digitalocean(method, path, body=None, params=None, retry=True):
    return http(method, DIGITALOCEAN + path, "DIGITALOCEAN_TOKEN", body, params, retry)


def digitalocean_list(path: str, field: str, params: dict | None = None) -> list:
    """Every item of a DigitalOcean list, across all pages."""
    items, page = [], 1
    while True:
        reply = digitalocean("GET", path, params={**(params or {}), "per_page": 200, "page": page})
        items += reply.get(field) or []
        if not ((reply.get("links") or {}).get("pages") or {}).get("next"):
            return items
        page += 1


def tail_lines(text: str, limit: int) -> str:
    """The last whole lines of `text` that fit in `limit` characters; one longer line is clipped."""
    text = text.strip()
    return text if len(text) <= limit else text[-limit:].partition("\n")[2] or text[-limit:]


# ---------------------------------------------------------------- storage

GLOBAL = "*"  # runpod_volumes() key for a global volume, which mounts in any data center


def _volume(v: dict, kind: str) -> dict:
    return {"id": v.get("id"), "name": v.get("name"), "size": v.get("size"), "kind": kind,
            "dataCenter": v.get("dataCenter") or v.get("dataCenterId")}


def runpod_volumes(cfg: dict) -> dict[str, dict]:
    """data center -> network volume to mount there, plus GLOBAL -> a global volume.

    Network volumes live in one data center. Global volumes (S3-backed, usage-billed; what the
    console attaches as an S3 /workspace) mount anywhere.
    """
    rp = cfg["runpod"]
    want = set(rp.get("volumes") or rp.get("network_volumes") or [])
    vols = [_volume(v, "network") for v in runpod_list("/network-volumes", "networkVolumes")]
    vols += [_volume(v, "global") for v in runpod_list("/global-volumes", "globalVolumes")]
    listed = {v["id"] for v in vols} | {v["name"] for v in vols}
    for vid in want - listed:  # look up configured ids the listings miss
        for kind, path in (("network", "/network-volumes/"), ("global", "/global-volumes/")):
            try:
                vols.append(_volume(runpod("GET", path + urllib.parse.quote(vid)), kind))
                break
            except ApiError:
                pass  # `check` reports why
    by_dc: dict[str, dict] = {}
    for v in vols:
        if want and not {v["id"], v["name"]} & want:
            continue
        key = GLOBAL if v["kind"] == "global" else v["dataCenter"]
        if key:
            by_dc.setdefault(key, v)
    return by_dc


def runpod_volume_for(vols: dict[str, dict], dc: str) -> dict | None:
    """A network volume in the pod's own data center (faster) if there is one, else the global one."""
    return vols.get(dc) or vols.get(GLOBAL)


def lambda_filesystems(cfg: dict) -> dict[str, dict]:
    """region -> filesystem to attach there."""
    want = set(cfg["lambda"].get("filesystems") or [])
    by_region: dict[str, dict] = {}
    for fs in lambda_("GET", "/filesystems") or []:
        if want and fs["name"] not in want:
            continue
        by_region.setdefault(fs["region"]["name"], fs)
    return by_region


# ---------------------------------------------------------------- offers

def lambda_vram(desc: str) -> int | None:
    m = re.search(r"(\d+)\s*GB", desc)
    return int(m.group(1)) if m else None  # unknown: don't filter on it


GIB = re.compile(r"(\d+(?:\.\d+)?)\s*([GT])i?", re.I)


# ---------------------------------------------------------------- providers

class Instance(NamedTuple):
    """One of a provider's instances, as its listing shows it; `created` in epoch seconds, None where it doesn't
    say."""
    id: str
    name: str
    status: str
    created: float | None = None


class Provider:
    """One source of GPUs: what the CLI and the ledger below do differently per provider.

    `region` is the offer's or instance's region."""

    name = ""
    key_env = ""  # the API key this provider needs
    user = "root"  # who ssh logs in as
    fixed_deadline = False  # its deadline was fixed at launch, so a later one wouldn't move it
    vendors = ("nvidia",)  # the GPU vendors its images run
    takes_run = False  # whether `launch --run` works here

    def __init__(self, cfg: dict, region: str | None = None):
        self.cfg, self.region = cfg, region

    def enabled(self) -> bool:
        return self.cfg.get(self.name, {}).get("enabled", True) and bool(os.environ.get(self.key_env))

    def offers(self, gpus, min_vram, all_regions, allow_arm=False) -> list[dict]:
        """The GPUs this provider offers now."""
        raise NotImplementedError

    def hf_home(self) -> str:
        """Where the box keeps the Hugging Face cache."""
        raise NotImplementedError

    def status(self, e: dict) -> tuple[str, str | None, int]:
        """(the instance's status, its ssh host if it has one yet, its ssh port)."""
        raise NotImplementedError

    def release(self, iid: str, e: dict) -> None:
        """Release the instance."""
        raise NotImplementedError

    @classmethod
    def pool(cls, cfg: dict, offer_id: str) -> list[dict]:
        """The offers a saved offer id is looked up among."""
        raise NotImplementedError

    def plan(self, o: dict, name: str, pubkey: str, hours: float, args) -> tuple[dict, object]:
        """(the request a dry run prints, the call that sends it and returns the instance id, once). Writes
        nothing on a dry run."""
        raise NotImplementedError

    def instances(self) -> list[Instance]:
        """Every instance on the account, but those ended or being released."""
        raise NotImplementedError

    def storage(self) -> dict | None:
        """Region -> the team's persistent storage there, or None for a provider that needs none."""
        return None

    def named(self, name: str) -> list[str]:
        """The ids of the account's instances with this name (one create makes one, the name is unique)."""
        return [i.id for i in self.instances() if i.name == name]

    def adopt(self, name: str) -> str | None:
        """The instance a create whose answer was lost made, found by its name, or None after 2 min."""
        for _ in range(12):  # a new instance can take a while to show in the listing
            if ids := self.named(name):
                return ids[0]
            time.sleep(10)
        return None

    def release_ids(self, e: dict) -> list[str]:
        """What to release: the instance, or for a lost create whatever has its name, waiting as long as
        launch would for it to be listed."""
        if e["instance_id"]:
            return [e["instance_id"]]
        ids = [i for i in [self.adopt(e["name"])] if i]
        if not ids:
            print(f"no instance named {e['name']} was listed in 2 min; nothing to terminate", file=sys.stderr)
        return ids


class RunPod(Provider):
    name, key_env = "runpod", "RUNPOD_API_KEY"

    @classmethod
    def pool(cls, cfg, offer_id):
        _, gpu, dc, count = offer_id.split(":")
        return cls(cfg).offers(int(count), 0, True)

    def offers(self, gpus, min_vram, all_regions, allow_arm=False):
        vols = runpod_volumes(self.cfg)
        cat = runpod("GET", "/catalog/gpus", params={
            "include": "AVAILABILITY", "product": "POD", "count": gpus, "cloud": "SECURE"})
        offers = []
        for g in cat.get("gpus") or []:
            if vendor_of(g["id"]) not in self.vendors or (g.get("memory") or 0) < min_vram:
                continue
            per_gpu = (g.get("price") or {}).get("secure")
            if not per_gpu:
                continue
            # dataCenters lists only DCs that can serve `count` right now
            for dc in g.get("dataCenters") or []:
                if dc.get("availability", "NONE") == "NONE":
                    continue
                storage = runpod_volume_for(vols, dc["id"]) is not None
                if storage or all_regions:
                    offers.append({
                        "id": f"runpod:{g['id'].replace(' ', '_')}:{dc['id']}:{gpus}",
                        "provider": "runpod", "gpu": g["id"], "count": gpus, "vram_gb": g["memory"],
                        "region": dc["id"], "usd_per_hour": round(per_gpu * gpus, 2),
                        "availability": dc.get("availability"), "storage": storage,
                    })
        return offers

    def plan(self, o, name, pubkey, hours, args):
        rp = self.cfg["runpod"]
        vol = runpod_volume_for(runpod_volumes(self.cfg), o["region"])
        kind = "global" if vol["kind"] == "global" else "network"
        body = {
            "name": name, "cloud": "SECURE",
            "gpu": {"id": o["gpu"], "count": o["count"], "minCudaVersion": rp.get("min_cuda", "12.8")},
            "image": rp["image"], "disk": rp.get("disk_gb", 200),
            # Both names: the API spec and the SSH docs disagree on which one the image reads.
            "env": {"PUBLIC_KEY": pubkey, "SSH_PUBLIC_KEY": pubkey, "HF_HOME": rp["hf_home"]},
            "ports": ["22/tcp"], "startSsh": True,
            "mounts": {kind: [{"volumeId": vol["id"], "path": "/workspace"}]},
            "dataCenterIds": [o["region"]],
        }
        return body, lambda: runpod("POST", "/pods", body, retry=False)["id"]

    def hf_home(self):
        return self.cfg["runpod"]["hf_home"]

    def status(self, e):
        p = runpod("GET", f"/pods/{e['instance_id']}")
        direct = (p.get("ssh") or {}).get("direct") or {}
        return p.get("status", "?"), direct.get("host"), int(direct.get("port") or 22)

    def instances(self):
        return [Instance(p["id"], p.get("name"), p.get("status")) for p in runpod_list("/pods", "pods")]

    def release(self, iid, e):
        runpod("DELETE", f"/pods/{iid}")

    def storage(self):
        return runpod_volumes(self.cfg)

    def explain_no_storage(self):
        rp = self.cfg["runpod"]
        for vid in rp.get("volumes") or rp.get("network_volumes") or []:
            for path in ("/network-volumes/", "/global-volumes/"):
                try:
                    runpod("GET", path + urllib.parse.quote(vid))
                except ApiError as e:
                    print(f"  {path}{vid}: {str(e)[:160]}")
        try:
            q = {"query": "query { myself { id email } }"}
            me = (http("POST", RUNPOD_GRAPHQL, "RUNPOD_API_KEY", q).get("data") or {}).get("myself") or {}
            print(f"  key's account: {me.get('email')} ({me.get('id')}). Volumes are per account: a team's"
                  " need a key created while the team is selected.")
        except ApiError as e:
            print(f"  key's account: unknown ({e})")


class Lambda(Provider):
    name, key_env, user = "lambda", "LAMBDA_API_KEY", "ubuntu"

    @classmethod
    def pool(cls, cfg, offer_id):
        return cls(cfg).offers(1, 0, True, allow_arm=True)

    def offers(self, gpus, min_vram, all_regions, allow_arm=False):
        fss = lambda_filesystems(self.cfg)
        offers = []
        for name, entry in (lambda_("GET", "/instance-types") or {}).items():
            it = entry["instance_type"]
            desc = it.get("gpu_description", "")
            n = it.get("specs", {}).get("gpus", 0)
            vram = lambda_vram(desc)
            if vendor_of(desc) not in self.vendors or n < gpus or (vram is not None and vram < min_vram):
                continue
            # GH200 is arm64; the torch/flashinfer wheels this repo installs are x86_64-first.
            if it.get("architecture", "x86_64") != "x86_64" and not allow_arm:
                continue
            for region in entry.get("regions_with_capacity_available") or []:
                storage = region["name"] in fss
                if storage or all_regions:
                    offers.append({
                        "id": f"lambda:{name}:{region['name']}",
                        "provider": "lambda", "gpu": desc, "count": n, "vram_gb": vram or "?",
                        "region": region["name"], "usd_per_hour": round(it["price_cents_per_hour"] / 100, 2),
                        "availability": "available", "storage": storage,
                        "arch": it.get("architecture", ""),
                    })
        return offers

    def plan(self, o, name, pubkey, hours, args):
        lm = self.cfg["lambda"]
        fs = lambda_filesystems(self.cfg)[o["region"]]
        body = {
            "name": name, "region_name": o["region"], "instance_type_name": o["id"].split(":")[1],
            "ssh_key_names": [lm["ssh_key_name"]], "file_system_names": [fs["name"]],
            "image": {"family": lm.get("image_family", "lambda-stack-24-04")},
        }
        return body, lambda: lambda_("POST", "/instance-operations/launch", body, retry=False)["instance_ids"][0]

    def hf_home(self):
        fs = lambda_filesystems(self.cfg)[self.region]
        mount = fs.get("mount_point") or f"/lambda/nfs/{fs['name']}"
        return self.cfg["lambda"]["hf_home"].format(fs=fs["name"], mount=mount)

    def status(self, e):
        i = lambda_("GET", f"/instances/{e['instance_id']}") or {}
        return i.get("status", "?"), i.get("ip"), 22

    def instances(self):
        return [Instance(i["id"], i.get("name"), i.get("status")) for i in lambda_("GET", "/instances") or []
                if i.get("status") not in ("terminating", "terminated")]

    def release(self, iid, e):
        lambda_("POST", "/instance-operations/terminate", {"instance_ids": [iid]})

    def storage(self):
        return lambda_filesystems(self.cfg)

    def explain_no_storage(self):
        pass


class Eras(Provider):
    name, key_env = "eras", "ERAS_API_KEY"
    fixed_deadline = takes_run = True

    @classmethod
    def pool(cls, cfg, offer_id):
        return cls(cfg).offers(1, 0, False)

    def offers(self, gpus, min_vram, all_regions, allow_arm=False):
        # The model is checked here rather than in the filter: Eras names models "H200", "B300", ...
        flt = json.dumps({"field": "hardware.gpu.count", "op": ">=", "value": gpus})
        page, offers = eras("GET", "/compute-listings", params={"page_size": 100, "filter": flt}), []
        for x in page.get("items") or []:
            g = (x.get("hardware") or {}).get("gpu") or {}
            m = GIB.match(str(g.get("memory_per_device") or ""))
            vram = int(float(m.group(1)) * (1024 if m.group(2).upper() == "T" else 1)) if m else None
            if vendor_of(str(g.get("model") or "")) not in self.vendors or (vram is not None and vram < min_vram):
                continue
            if x.get("hourly_price") is None:
                continue
            now = (x.get("access_method") or {}).get("is_available") and (x.get("launches_available") or 0) > 0
            if now or all_regions:
                # Local NVMe only: stage_model pulls weights from the Hub (the default hf_home is on it).
                offers.append({
                    "id": f"eras:{x['id']}", "provider": "eras", "gpu": g["model"], "count": g["count"],
                    "vram_gb": vram or "?", "region": x.get("name", "eras"),
                    "usd_per_hour": round(x["hourly_price"], 2),
                    "availability": "now" if now else "none", "storage": True,
                })
        return offers

    def plan(self, o, name, pubkey, hours, args):
        # tried in turn: each free cache volume, then none (cold); a refresh only its own
        free = eras_cache_volumes(self.cfg, args.cache, args.cache_write)
        candidates = free if args.cache_write else [*free, None]
        key_id = "<registered at launch>" if args.dry_run else eras_ssh_key_id(pubkey, name)
        body_for = functools.partial(eras_launch_body, self.cfg, o, name, key_id, hours, args.run, args.input,
                                     rw=bool(args.cache_write))
        first = body_for(candidates[0])

        def create():
            for i, vid in enumerate(candidates):
                try:
                    return eras_create(body_for(vid) if i else first)
                except ApiError as err:  # another box took the cache volume since: the next free one, then cold
                    if not volume_refused(err) or vid is None:
                        raise
                    if args.cache_write:  # a refresh without its volume would write nothing that lasts
                        die(f"cache volume {vid} is held or being deleted: {err}")
                    print(f"gpu.py: cache volume {vid} is held or being deleted ({str(err)[:200]}); "
                          "trying the next free one, else cold", file=sys.stderr)
        return first, create

    def hf_home(self):
        return self.cfg.get("eras", {}).get("hf_home", "/workspace/hf")

    def status(self, e):
        i = eras("GET", f"/instances/{e['instance_id']}")
        ssh = next((p for p in (i.get("standalone") or {}).get("endpoints") or []
                    if p.get("name") == "ssh" and p.get("ready")), None)
        run = (i.get("standalone") or {}).get("current_run") or {}
        st = i.get("status", "?")
        if e.get("run") and run.get("status") in ("succeeded", "failed", "stopped"):
            failure = run.get("failure") or {}  # why the platform says it failed, e.g. launch_refused
            st = f"run {run['status']}" + (f" ({failure.get('code')}: {str(failure.get('message') or '')[:300]})"
                                           if failure else "")
        return st, ssh and ssh.get("host"), int((ssh or {}).get("port") or 22)

    def instances(self):
        return [Instance(i["id"], i.get("name"), i.get("status")) for i in eras_items("/instances", 200)
                if i.get("status") not in ("ended", "releasing")]

    def release(self, iid, e):
        eras("POST", f"/instances/{iid}/release", {})


class DigitalOcean(Provider):
    """GPU Droplets: one offer per size, created wherever DigitalOcean has it (no region asked for)."""
    name, key_env = "digitalocean", "DIGITALOCEAN_TOKEN"
    vendors = ("nvidia", "amd")
    takes_run = True

    @classmethod
    def pool(cls, cfg, offer_id):
        return cls(cfg).offers(1, 0, True)

    def offers(self, gpus, min_vram, all_regions, allow_arm=False):
        offers = []
        for s in digitalocean_list("/sizes", "sizes"):
            g = s.get("gpu_info") or {}
            gpu = str(g.get("model") or "").replace("_", " ").upper()  # nvidia_h100 -> NVIDIA H100
            vram = (g.get("vram") or {}).get("amount") or 0
            # Contracted sizes are sold under a contract, never created self-serve.
            if vendor_of(gpu) not in self.vendors or (g.get("count") or 0) < gpus or vram < min_vram \
                    or s["slug"].endswith("-contracted") or not (s.get("available") or all_regions):
                continue
            offers.append({
                "id": f"digitalocean:{s['slug']}", "provider": "digitalocean", "gpu": gpu, "count": g["count"],
                "vram_gb": vram, "region": "any", "usd_per_hour": round(s["price_hourly"], 2),
                "availability": "available" if s.get("available") else "none", "storage": True,
            })
        return offers

    def plan(self, o, name, pubkey, hours, args):
        # DigitalOcean's AI/ML-ready images: ROCm for AMD; for NVIDIA one for every 1-GPU size, one for 8.
        image = ("gpu-amd-base" if vendor_of(o["gpu"]) == "amd" else
                 "gpu-h100x8-base" if o["count"] > 1 else "gpu-h100x1-base")
        key = "<registered at launch>" if args.dry_run else digitalocean_ssh_key(pubkey, name)
        body = {"name": name, "size": o["id"].split(":", 1)[1], "image": image, "ssh_keys": [key]}
        if args.run:
            secret = self.cfg.get("digitalocean", {}).get("run_secret_env")
            if secret and not args.dry_run and not os.environ.get(secret):
                die(f"digitalocean.run_secret_env names {secret}, which is not set; the box would have no key")
            body["user_data"] = cloud_config(args.run, args.input, secret and (
                "<the value of $" + secret + ">" if args.dry_run else os.environ[secret]))
            if len(body["user_data"].encode()) > USER_DATA_MAX:
                die(f"the --run script and inputs take {len(body['user_data'].encode())} bytes of user_data, over "
                    f"DigitalOcean's {USER_DATA_MAX}; send smaller inputs")
        return body, lambda: str(digitalocean("POST", "/droplets", body, retry=False)["droplet"]["id"])

    def hf_home(self):
        return self.cfg.get("digitalocean", {}).get("hf_home", "/workspace/hf")

    def status(self, e):
        try:
            d = digitalocean("GET", f"/droplets/{e['instance_id']}")["droplet"]
        except ApiError as err:
            if err.status == 404:  # deleted, by terminate or by hand
                return "ended", None, 22
            raise
        host = next((n.get("ip_address") for n in (d.get("networks") or {}).get("v4") or []
                     if n.get("type") == "public"), None)
        return "ended" if d.get("status") == "archive" else d.get("status", "?"), host, 22

    def instances(self):
        # GPU droplets are listed only with type=gpus.
        return [Instance(str(d["id"]), d.get("name"), d.get("status"),
                         datetime.datetime.fromisoformat(d["created_at"].replace("Z", "+00:00")).timestamp())
                for d in digitalocean_list("/droplets", "droplets", {"type": "gpus"}) if d.get("status") != "archive"]

    def release(self, iid, e):
        digitalocean("DELETE", f"/droplets/{iid}")


USER_DATA_MAX = 64 * 1024  # what DigitalOcean takes in a droplet's user_data


def cloud_config(run: str, inputs: list[str], secret: str | None) -> str:
    """The user_data that runs `run` at boot as `launch --run` does on Eras: the script and its inputs in
    /run/live-pr, a secret at /run/live-pr-key/value as Eras mounts run_secret_id, and the script started
    as a unit of its own so cloud-init isn't left waiting on it. JSON is YAML, which cloud-init reads."""
    def entry(path, data):
        return {"path": path, "encoding": "gz+b64", "permissions": "0600",
                "content": base64.b64encode(gzip.compress(data, mtime=0)).decode()}
    files = [entry(f"/run/live-pr/{Path(f).name}", Path(f).read_bytes()) for f in [run, *inputs]]
    if secret:
        files.append(entry("/run/live-pr-key/value", secret.encode()))
    return "#cloud-config\n" + json.dumps({"write_files": files, "runcmd": [
        ["systemd-run", "--unit", "live-pr", "bash", f"/run/live-pr/{Path(run).name}"]]})


def digitalocean_ssh_key(pubkey: str, name: str) -> str:
    """The team's fingerprint for this public key, registering it the first time, under the launch's name.
    A droplet created with no key gets a root password instead."""
    blob = base64.b64decode(pubkey.split()[1])
    fingerprint = ":".join(f"{b:02x}" for b in hashlib.md5(blob, usedforsecurity=False).digest())
    try:
        return digitalocean("GET", f"/account/keys/{fingerprint}")["ssh_key"]["fingerprint"]
    except ApiError as err:
        if err.status != 404:
            raise
    return digitalocean("POST", "/account/keys", {"name": name, "public_key": pubkey})["ssh_key"]["fingerprint"]


PROVIDERS = {cls.name: cls for cls in (Eras, RunPod, Lambda, DigitalOcean)}


def provider_of(cfg: dict, record: dict) -> Provider:
    """The provider of an offer or a ledger entry."""
    return PROVIDERS[record["provider"]](cfg, record.get("region"))


def has_provider(e: dict) -> bool:
    """Whether a ledger entry's provider is one gpu.py has. One it no longer has (a removed provider's
    entry) is out of gpu.py's reach: it counts toward no cap, and `terminate` drops it without releasing anything."""
    return e["provider"] in PROVIDERS


def job_name(pr: int, tag: str) -> str:
    return f"{JOB_PREFIX}{pr}-{tag}"


def cmd_offers(args, cfg):
    offers, errors, answered, failed = [], [], 0, []
    for p in (cls(cfg) for cls in PROVIDERS.values()):  # one that can't be reached is skipped, with its reason
        if not p.enabled():
            errors.append(f"{p.name}: skipped (disabled, or no API key)")
            continue
        try:
            offers += p.offers(args.gpus, args.min_vram, args.all_regions, allow_arm=args.allow_arm)
            answered += 1
        except ApiError as e:
            errors.append(f"{p.name}: {e}")
            failed.append(p.name)
    offers = [{**o, "vendor": vendor_of(o["gpu"])} for o in offers]
    offers = [o for o in offers if o["vendor"] == args.vendor]
    offers.sort(key=lambda o: (o["availability"] == "none", o["usd_per_hour"], o["count"]))  # what can start now first
    cap = cfg["budget"]["max_usd_per_hour"]
    # No offers with a provider that errored, or none answering, says nothing about capacity there:
    # that is not "no availability".
    unknown = not offers and (failed or not answered)
    if args.json:
        print(json.dumps({"offers": offers, "errors": errors, "max_usd_per_hour": cap}, indent=1))
        if unknown:
            sys.exit(1)
        return
    for e in errors:
        print("#", e)
    if unknown:
        die(f"availability unknown: {', '.join(failed) or 'no provider'} didn't answer (see above), "
            "and no provider that did has an offer; fix the keys or the errors")
    if all(o["availability"] == "none" for o in offers):
        print(f"no availability: no {args.vendor} box with >= {args.gpus} GPUs"
              + ("" if args.all_regions else " in a region with team storage (try --all-regions)"))
        if not offers:
            return
    print(f"{'offer':58} {'$/hr':>7}  {'gpus':>4}  {'vram':>5}  avail   storage")
    for o in offers:
        flag = "" if o["usd_per_hour"] <= cap else f"  over ${cap}/hr cap"
        print(f"{o['id']:58} {o['usd_per_hour']:7.2f}  {o['count']:4}  {o['vram_gb']:4}G  "
              f"{str(o['availability']):7} {'yes' if o['storage'] else 'no '}{flag}")


# ---------------------------------------------------------------- ledger

def ledger() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {}


def save_ledger(d: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(LEDGER)


@contextlib.contextmanager
def ledger_txn():
    """Read-modify-write of the ledger under an exclusive lock. Several sessions
    (parallel PR tests, their watchdogs) share one ledger; without the lock one
    save drops another's entry, and that instance loses its budget accounting."""
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "instances.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        d = ledger()
        yield d
        save_ledger(d)


def committed_usd(e: dict) -> float:
    """What an instance costs if it runs to its deadline."""
    return e["usd_per_hour"] * (e["deadline"] - e["launched_at"]) / 3600


def entry(key: str, unknown_provider_ok: bool = False) -> dict:
    d = ledger()
    if key not in d:
        die(f"unknown instance {key!r}; known: {', '.join(d) or 'none'}")
    if not (unknown_provider_ok or has_provider(d[key])):
        die(f"{key} is a {d[key]['provider']!r} instance, a provider gpu.py doesn't have, so there is nothing to "
            f"reach, extend or release; `gpu.py terminate {key}` drops it from the ledger")
    return d[key]


# ---------------------------------------------------------------- launch

def find_offer(cfg, offer_id: str) -> dict:
    provider = offer_id.split(":", 1)[0]
    if provider not in PROVIDERS:
        die(f"bad offer id {offer_id!r}")
    for o in PROVIDERS[provider].pool(cfg, offer_id):
        if o["id"] == offer_id:
            return o
    die(f"{offer_id} is no longer available; rerun offers")


def check_hours(cfg, hours: float, allow: bool) -> None:
    if hours > cfg["budget"]["max_hours"] and not allow:
        die(f"{hours} h is over the {cfg['budget']['max_hours']} h cap; ask before passing --allow-over-budget")


def check_total(cfg, committed: float, allow: bool) -> None:
    """``committed``: what the live instances cost if they all run to their deadlines."""
    max_total = cfg["budget"].get("max_total_usd")
    if max_total is not None and committed > max_total and not allow:
        die(f"running to their deadlines, the live instances would cost ~${committed:.0f}, "
            f"over budget.max_total_usd ${max_total}; ask before passing --allow-over-budget")


def cmd_launch(args, cfg):
    o = find_offer(cfg, args.offer)
    if not o["storage"]:
        die(f"{o['region']} has no team storage; the HF cache would be cold. Pick a region with storage.")
    cap, hours = cfg["budget"]["max_usd_per_hour"], args.hours or cfg["budget"]["max_hours"]
    if o["usd_per_hour"] > cap and not args.allow_over_budget:
        die(f"${o['usd_per_hour']}/hr is over the ${cap}/hr cap; ask before passing --allow-over-budget")
    check_hours(cfg, hours, args.allow_over_budget)
    pubkey = Path(cfg["ssh_public_key"]).expanduser().read_text().strip()
    p = provider_of(cfg, o)
    if args.run and not p.takes_run:
        die(f"{p.name} can't take --run; launch without it and start the work over ssh")
    # unique per launch, so a create whose answer was lost is found again by this name. A --tag is the
    # caller's own unique name: what it looks the instance up by, and what makes an Eras launch idempotent.
    name = job_name(args.pr, args.tag or f"{time.strftime('%m%d-%H%M%S')}-{secrets.token_hex(2)}")
    max_instances = cfg["budget"].get("max_concurrent_instances", 1)
    max_running = cfg["budget"].get("max_running")

    # The caps are checked and the launch recorded under one lock, so parallel sessions can't each
    # pass the check and launch together; and the body is built after the caps, since building an
    # Eras one registers an ssh key. A dry run checks nothing, records nothing and writes nothing.
    pending, lost = f"pending:{name}", None
    with contextlib.nullcontext({}) if args.dry_run else ledger_txn() as d:
        if not args.dry_run:
            live = {k: e for k, e in d.items() if has_provider(e)}
            if len(live) >= max_instances and not args.allow_over_budget:
                die(f"{len(live)} instance(s) already up ({', '.join(live)}); budget.max_concurrent_instances is "
                    f"{max_instances}. Wait for one to finish, or ask before passing --allow-over-budget")
            if max_running is not None and not args.allow_over_budget:
                try:
                    for line in reap(cfg):
                        print(f"gpu.py: {line}", file=sys.stderr)
                    up = running(cfg)
                except ApiError as err:
                    die(f"can't count this tool's instances for budget.max_running: {err}")
                if len(up) + args.boxes > max_running:
                    die(f"{len(up)} of this tool's instances are up, from every session, and this launch needs room "
                        f"for {args.boxes}; budget.max_running is {max_running}. Wait for one to end, or ask before "
                        "passing --allow-over-budget", code=4)
            check_total(cfg, sum(committed_usd(e) for e in live.values()) + o["usd_per_hour"] * hours,
                        args.allow_over_budget)
        body, create = p.plan(o, name, pubkey, hours, args)
        if args.dry_run:
            print(json.dumps(body, indent=1).replace(pubkey, "<public key>"))
            return
        try:
            iid = create()
        except Exception as e:  # noqa: BLE001 - anything after the send may have made the instance
            if isinstance(e, ApiError) and e.status == 429:
                die(f"rate-limited on create ({e}); nothing was made, try again in a minute")
            if isinstance(e, ApiError) and 0 < e.status < 500:  # refused: nothing was made
                if e.status == 409 or re.search(r"capacity|no longer any instances|not available",
                                                f"{e.code()} {e.body}", re.I):
                    die(f"no availability: {o['id']} was taken before launch ({e}); rerun offers")
                raise  # a bad config, a bad image
            # A 5xx, a lost connection, a reply without an id: the provider may have made it. Until
            # it is found by name, the ledger holds its cost, and a watchdog its deadline.
            lost, iid = e, None
        key = pending if lost else f"{p.name}:{iid}"
        now = time.time()
        d[key] = fields = {"provider": o["provider"], "instance_id": iid, "name": name, "offer": o["id"], "pr": args.pr,
              "run": bool(args.run),
              "usd_per_hour": o["usd_per_hour"], "gpu": o["gpu"], "count": o["count"], "region": o["region"],
              "launched_at": now, "deadline": now + hours * 3600,
              "user": p.user, "hf_home": p.hf_home()}
    arm_watchdog(key)  # a pending entry's watchdog terminates by name at the deadline
    if lost:
        try:
            iid = p.adopt(name)
        except ApiError as err:
            die(f"the create's answer was lost ({lost}), and looking for {name} failed ({err}): "
                f"`gpu.py terminate {pending}` terminates it by name once the API answers")
        if iid is None:
            die(f"the create failed ({lost}), and no instance named {name} showed up in 2 min. Its watchdog "
                f"still terminates one by name at the deadline; `gpu.py terminate {pending}` does it now")
        print(f"the create's answer was lost ({lost}); {p.name}:{iid} is the instance named {name}",
              file=sys.stderr)
        with ledger_txn() as d:
            key = f"{p.name}:{iid}"
            # recorded even if pending was terminated meanwhile: the pod was just found, so it gets a watchdog
            # the entry as it is now (an extend meanwhile), or as it was if it was terminated meanwhile
            d[key] = {**(d.pop(pending, None) or fields), "instance_id": iid}  # pending's watchdog then exits
        arm_watchdog(key)
    print(f"launched {key} ({o['count']}x {o['gpu']}, {o['region']}, ${o['usd_per_hour']}/hr); "
          f"watchdog terminates it at {time.strftime('%H:%M', time.localtime(now + hours * 3600))} ({hours} h)")
    print(key)


def arm_watchdog(key: str) -> None:
    log = STATE / f"{key.replace(':', '_')}.watchdog.log"
    with open(log, "a") as fh:
        subprocess.Popen([sys.executable, __file__, "watchdog", key], stdout=fh, stderr=fh,
                         stdin=subprocess.DEVNULL, start_new_session=True)


ERAS_GPU_NOTEBOOK = "ed6d2a30-36aa-5d50-b911-1f6b551f2081"  # Eras Cloud Notebook (NVIDIA GPU): sshd as root


def eras_ssh_key_id(pubkey: str, name: str) -> str:
    """The organization's id for this public key, registering it the first time, under the launch's
    name. It stays registered for later launches."""
    blob = base64.b64decode(pubkey.split()[1])
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
    for k in eras_items("/ssh-keys", 100):
        if k.get("fingerprint") == fingerprint:
            return k["id"]
    return eras("POST", "/ssh-keys", {"name": name, "public_key": pubkey})["id"]


def eras_launch_body(cfg: dict, o: dict, name: str, ssh_key_id: str, hours: float, run: str | None = None,
                     inputs: list[str] = (), volume: str | None = None, rw: bool = False) -> dict:
    """The GPU notebook template's container and image, with sshd as root, every GPU of the
    listing and a big /dev/shm. Kubernetes stops the Pod at the deadline and never restarts
    it, and a run that ends releases the instance, so the box goes even if nothing that
    launched it is left to. With `run`, the script is the Pod's work and its inputs are a
    ConfigMap at /run/live-pr; the instance stays 15 minutes after it exits, for copying back.
    A cache volume (weights, compile caches; see eras_cache_volumes) mounts at /cache, read-only
    unless `rw`."""
    er = cfg.get("eras", {})
    tpl = eras("GET", f"/templates/{er.get('template_id', ERAS_GPU_NOTEBOOK)}")
    pod = tpl["pod"]
    box = next(c for c in pod["containers"] if (c.get("resources") or {}).get("limits", {}).get("nvidia.com/gpu"))
    box["resources"]["limits"]["nvidia.com/gpu"] = str(o["count"])
    box.pop("readinessProbe", None)  # it probes Jupyter, which this Pod doesn't run
    box["ports"] = [p for p in box.get("ports") or [] if p.get("name") == "ssh"]
    box["env"] = [v for v in box.get("env") or [] if not v["name"].startswith("ERAS_JUPYTER")]
    sshd = "/usr/sbin/sshd -D -e -o StrictModes=no -o 'AuthorizedKeysFile /etc/ssh/keys/authorized_keys'"
    # These args replace the template's, and with them its entrypoint, which makes sshd's host keys.
    box["args"] = ["set -e\ninstall -d -m 0700 /root/.ssh /run/sshd\nssh-keygen -A\n"
                   + (f"{sshd} &\nexec bash /run/live-pr/{Path(run).name}\n" if run else f"exec {sshd}\n")]
    pod["restartPolicy"] = "Never"
    pod["activeDeadlineSeconds"] = int(hours * 3600)
    for v in pod.get("volumes") or []:
        if v["name"] == "shm":
            v["emptyDir"]["sizeLimit"] = er.get("shm", "64Gi")  # the most Eras allows
    config_maps = None
    if run:
        pod["volumes"].append({"name": "live-pr", "configMap": {"name": "live-pr"}})
        box["volumeMounts"].append({"mountPath": "/run/live-pr", "name": "live-pr", "readOnly": True})
        # Base64, since Eras refuses a Pod whose ConfigMap text holds a 4-byte UTF-8 character (an emoji, say).
        config_maps = {"live-pr": {"binaryData": {Path(f).name: base64.b64encode(Path(f).read_bytes()).decode()
                                                  for f in [run, *inputs]}}}
        if er.get("run_secret_id"):  # a saved Eras secret (an API key, say), at /run/live-pr-key/value
            pod["volumes"].append({"name": "live-pr-key", "secret": {"secretName": "live-pr-key",
                                                                      "items": [{"key": "value", "path": "value"}]}})
            box["volumeMounts"].append({"mountPath": "/run/live-pr-key", "name": "live-pr-key", "readOnly": True})
    claims = {}
    if volume:
        claims["cache"] = {"volume_id": volume}
        pod["volumes"].append({"name": "cache", "persistentVolumeClaim": {"claimName": "cache"}})
        box["volumeMounts"].append({"mountPath": "/cache", "name": "cache", "readOnly": not rw})
    keep = 900 if run else 0
    return {
        "listing_id": o["id"].split(":", 1)[1], "name": name,
        "run": {"pod": pod, "endpoints": {"ssh": tpl["endpoints"]["ssh"]}, "config_maps": config_maps,
                "bindings": {"secrets": {"authorized-keys": {"ssh_key_ids": [ssh_key_id]},
                                         **({"live-pr-key": {"secret_id": er["run_secret_id"]}}
                                            if run and er.get("run_secret_id") else {})},
                             "volumes": {"workspace": "emptyDir", **claims}},
                "release_when": {"after_success_seconds": keep, "after_failure_seconds": keep}},
    }


def eras_cache_volumes(cfg: dict, cache: str, write: str | None) -> list[str]:
    """The cache volumes a launch tries to bind, in order. An instance binds one volume, so
    eras.cache_volume_ids are replicas, one per box: "ro" tries those no box holds and that can be
    read (a deleted id is skipped), then none (the box starts cold). `write` (a cache refresh) takes
    only that volume, and fails if it can't be read or a box holds it."""
    ids = [write] if write else [] if cache == "none" else cfg.get("eras", {}).get("cache_volume_ids") or []
    free = []
    for vid in ids:
        try:
            att = eras("GET", f"/volumes/{vid}").get("attachment") or {}
        except ApiError as err:
            if write:
                raise
            print(f"gpu.py: skipping cache volume {vid}, which can't be read: {err}", file=sys.stderr)
            continue
        if not att.get("instance_id") or att.get("detached_at"):
            free.append(vid)
        elif write:
            die(f"cache volume {vid} is held by instance {att['instance_id']}; wait for it to end")
    if ids and not free:
        print("gpu.py: no cache volume is free; launching cold", file=sys.stderr)
    return free


# The error codes of a launch refused over a volume another instance holds, or one being deleted.
VOLUME_HELD = ("volume_bound", "volume_deleting", "volume_binding_conflict", "volume_binding_deleting",
               "resource_in_use")


def volume_refused(err: ApiError) -> bool:
    """Whether Eras refused a launch over a volume binding, by its error code, never by its wording."""
    return err.code() in VOLUME_HELD


def eras_create(body: dict) -> str:
    """Preview, accept the quoted prices, create; the instance id. Eras refuses an idempotency key
    reused with another request, so each attempt (listing, cache volume) has its own. The create is
    sent again only after a 429, with the same key; any other failure may have come after Eras acted,
    so it goes to launch, which finds the instance by name."""
    preview = eras("POST", "/instances/preview", body)
    problems = (preview.get("run") or {}).get("problems") or []
    if problems:
        raise ApiError(409, {"code": problems[0].get("code"), "problems": problems})
    if not preview.get("is_available"):
        reason, detail = preview.get("unavailable_reason"), preview.get("unavailable_detail")
        if reason in VOLUME_HELD:  # another box binds the cache volume, or it is being deleted
            raise ApiError(409, {"code": reason, "message": detail})
        raise ApiError(409, f"capacity: {detail or reason}")
    for w in (preview.get("run") or {}).get("warnings") or []:
        print(f"gpu.py: eras preview: {w.get('message')}", file=sys.stderr)
    body = dict(body, accepted_prices={i["usage_type"]: i["unit_price"] for i in preview["quote"]["line_items"]},
                accepted_available_until=preview.get("available_until"))
    attempt = json.dumps([body["listing_id"], (body["run"]["bindings"]["volumes"].get("cache") or {}).get("volume_id")])
    body["idempotency_key"] = f"{body['name']}-{hashlib.sha256(attempt.encode()).hexdigest()[:8]}"
    return eras("POST", "/instances", body, retry=False)["id"]


# ---------------------------------------------------------------- status / ssh

def status(e: dict, cfg: dict) -> tuple[str, str | None, int]:
    """(status, host, ssh port)."""
    if not has_provider(e):
        return (f"{e['provider']!r} is not a provider gpu.py has, so it can't reach or release it; end it on "
                "the provider if it's still up, then terminate drops it"), None, 0
    if not e["instance_id"]:
        return f"create's answer lost; terminate it to end {e['name']}", None, 0
    return provider_of(cfg, e).status(e)


def ssh_base(e: dict, cfg: dict, scp: bool = False) -> list[str]:
    known = STATE / f"{e['provider']}_{e['instance_id']}.known_hosts"
    return ["scp" if scp else "ssh", "-P" if scp else "-p", str(e["port"]),
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known}",
            "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=15",
            # the private half of ssh_public_key, when ssh wouldn't try it by default
            *(["-i", str(Path(cfg["identity_file"]).expanduser())] if cfg.get("identity_file") else [])]


def remote(e: dict, cfg: dict, command: list[str] | str) -> list[str]:
    """The ssh command line that runs `command` on the box (a shell, when it is empty)."""
    cmd = command if isinstance(command, str) else " ".join(command)
    return ssh_base(e, cfg) + [f"{e['user']}@{e['host']}"] + ([cmd] if cmd else [])


def cmd_wait(args, cfg):
    e = entry(args.instance)
    if not e["instance_id"]:
        die(f"{args.instance} is a create whose answer was lost; `gpu.py terminate {args.instance}` and launch again")
    t0 = time.time()
    while True:
        st, host, port = status(e, cfg)
        if st.lower() in ("exited", "error", "terminated", "terminating", "unhealthy", "preempted",
                          "ended", "releasing"):
            die(f"{args.instance} is {st}")
        if host:
            e["host"], e["port"] = host, port
            probe = remote(e, cfg, "nvidia-smi -L || amd-smi list")
            probe[1:1] = ["-o", "BatchMode=yes"]
            r = subprocess.run(probe, capture_output=True, text=True, check=False)
            if r.returncode == 0:
                with ledger_txn() as d:
                    if args.instance in d:
                        d[args.instance].update(host=host, port=port)
                print(r.stdout.strip())
                print(f"ready after {int(time.time() - t0)}s: {' '.join(remote(e, cfg, ''))}")
                print(f"hf_home: {e['hf_home']}")
                return
        if time.time() - t0 > args.timeout:
            die(f"{args.instance} not reachable after {args.timeout}s (status {st}, host {host})")
        time.sleep(10)


def reachable(key: str) -> dict:
    e = entry(key)
    if not e["instance_id"]:
        die(f"{key} is a create whose answer was lost; `gpu.py terminate {key}` and launch again")
    if not e.get("host"):
        die(f"run `gpu.py wait {key}` first")
    return e


def cmd_ssh(args, cfg):
    e = reachable(args.instance)
    os.execvp("ssh", remote(e, cfg, args.command))


def cmd_copy(args, cfg):
    e = reachable(args.instance)
    target = f"{e['user']}@{e['host']}:{args.remote}"
    src, dst = (args.local, target) if args.cmd == "push" else (target, args.local)
    sys.exit(subprocess.run(ssh_base(e, cfg, scp=True) + ["-r", src, dst], check=False).returncode)


# ---------------------------------------------------------------- list / extend / terminate

def terminate(key: str, e: dict) -> None:
    if not has_provider(e):
        with ledger_txn() as d:
            d.pop(key, None)
        print(f"dropped {key} from the ledger: gpu.py has no {e['provider']!r} provider, so nothing was released")
        return
    p = provider_of({}, e)  # releasing needs only the API key
    for iid in p.release_ids(e):
        try:
            p.release(iid, e)
        except ApiError as err:
            if err.status != 404 and "does-not-exist" not in err.code():
                raise
    with ledger_txn() as d:
        d.pop(key, None)
    hours = (time.time() - e["launched_at"]) / 3600
    print(f"terminated {key} after {hours:.2f} h, ~${hours * e['usd_per_hour']:.2f}")


def eras_items(path: str, page_size: int) -> list[dict]:
    """Every item of an Eras list, across all pages."""
    items, token = [], None
    while True:
        page = eras("GET", path, params={"page_size": page_size, **({"page_token": token} if token else {})})
        items += page.get("items") or []
        if not (token := page.get("next_page_token")):
            return items


def cmd_terminate(args, cfg):
    terminate(args.instance, entry(args.instance, unknown_provider_ok=True))


def cmd_extend(args, cfg):
    entry(args.instance)
    # a later deadline keeps to launch's caps, on the whole run from launch to the new deadline
    with ledger_txn() as d:
        e = d[args.instance]
        deadline = time.time() + args.hours * 3600
        if deadline > max(e["deadline"], time.time()):  # an earlier deadline only saves money, whatever was approved
            if PROVIDERS[e["provider"]].fixed_deadline:
                die(f"{args.instance} can't run later than its current deadline: the Eras Pod's deadline was fixed "
                    "at launch, so it would still end then. Launch a new one for more time")
            check_hours(cfg, (deadline - e["launched_at"]) / 3600, args.allow_over_budget)
            check_total(cfg, sum(committed_usd(x) for k, x in d.items() if k != args.instance and has_provider(x))
                        + committed_usd({**e, "deadline": deadline}), args.allow_over_budget)
        e["deadline"] = deadline
    print(f"{args.instance} deadline now {time.strftime('%H:%M', time.localtime(e['deadline']))}")


def enabled_providers(cfg: dict) -> list[Provider]:
    return [p for p in (cls(cfg) for cls in PROVIDERS.values()) if p.enabled()]


def running(cfg: dict) -> list[str]:
    """This tool's instances up now on every enabled provider, from every session and machine (the
    ledger has only this machine's), one line each. "This tool's" goes by name, so every live-pr-*
    instance on those accounts counts. budget.max_running reads this list, then launches: a soft cap,
    since two machines can both see the last free slot. A provider that can't be listed raises, so the
    cap fails closed."""
    return [f"{p.name}:{i.id}  {i.name}  {i.status}" for p in enabled_providers(cfg) for i in p.instances()
            if str(i.name).startswith(JOB_PREFIX)]


def reap(cfg: dict) -> list[str]:
    """Release this tool's instances, from every session and machine, that outlived their deadline: their
    ledger entry's, else budget.max_hours plus an hour from their creation. Only on enabled providers whose
    instances have no deadline of their own and whose listing dates them. What it released, one line each."""
    d, grace, released = ledger(), (cfg["budget"]["max_hours"] + 1) * 3600, []
    for p in enabled_providers(cfg):
        if p.fixed_deadline:
            continue
        for i in p.instances():
            e = d.get(f"{p.name}:{i.id}")
            deadline = e["deadline"] if e else i.created and i.created + grace
            if str(i.name).startswith(JOB_PREFIX) and deadline and time.time() > deadline:
                p.release(i.id, e or {})
                released.append(f"released {p.name}:{i.id}  {i.name}, past its deadline")
    return released


def release_by_name(cfg: dict, names: list[str]) -> None:
    """End whatever this tool launched with these names (job_name) on any enabled provider, for a launch
    cancelled before its instance was recorded. Every provider is tried; the first error is raised after."""
    failed = None
    for p in enabled_providers(cfg):
        try:
            for iid in [i.id for i in p.instances() if i.name in names]:
                p.release(iid, {})
        except ApiError as err:
            print(f"{p.name}: {err}", file=sys.stderr)
            failed = failed or err
    if failed:
        raise failed


def cmd_list(args, cfg):
    d = ledger()
    if not d:
        print("no instances launched by this tool are up")
    for key, e in d.items():
        try:
            st = status(e, cfg)[0]
        except ApiError as err:
            st = f"? ({err.status})"
        hrs = (time.time() - e["launched_at"]) / 3600
        left = (e["deadline"] - time.time()) / 60
        print(f"{key}  pr {e['pr']}  {e['count']}x {e['gpu']} {e['region']}  {st}  "
              f"{hrs:.2f} h, ~${hrs * e['usd_per_hour']:.2f}  deadline in {left:.0f} min")
    if args.all:
        print("--- this tool's on every provider, from every session ---")
        try:
            up = running(cfg)
        except ApiError as err:
            up = [f"? {err}"]
        for line in up:
            print(line)


def cmd_reap(args, cfg):
    for line in reap(cfg) or ["nothing past its deadline"]:
        print(line)


def cmd_watchdog(args, cfg):
    """Detached: terminate the instance at its deadline unless it is terminated first."""
    while True:
        e = ledger().get(args.instance)
        if e is None:
            return  # terminated normally
        if not has_provider(e):
            print(f"{time.ctime()}: gpu.py has no {e['provider']!r} provider; nothing to terminate", flush=True)
            return
        if time.time() >= e["deadline"]:
            print(f"{time.ctime()}: deadline reached for {args.instance}; terminating", flush=True)
            try:
                terminate(args.instance, e)
                return
            except Exception as err:  # keep trying: an unterminated box keeps billing
                print(f"{time.ctime()}: terminate failed: {err}", flush=True)
        time.sleep(60)


def cmd_check(args, cfg):
    print(f"budget: ${cfg['budget']['max_usd_per_hour']}/hr, {cfg['budget']['max_hours']} h")
    key_ok = Path(cfg["ssh_public_key"]).expanduser().exists()
    print(f"ssh key: {cfg['ssh_public_key']} ({'ok' if key_ok else 'MISSING'})")
    for p in (cls(cfg) for cls in PROVIDERS.values()):
        if not p.enabled():
            print(f"{p.name}: skipped (disabled or no API key)")
            continue
        try:
            stores = p.storage()
        except ApiError as e:
            print(f"{p.name}: {e}")
            continue
        if stores is None:  # no team storage to check
            continue
        where = ", ".join(
            f"any data center (global volume {s['name']!r}, {s['id']})" if r == GLOBAL
            else f"{r} ({s.get('name') or s.get('id')})" for r, s in stores.items())
        print(f"{p.name}: storage in {where or 'NO REGIONS'}")
        if not stores:
            p.explain_no_storage()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("offers")
    p.add_argument("--gpus", type=int, required=True)
    p.add_argument("--vendor", choices=sorted(SUPPORTED), default="nvidia")
    p.add_argument("--min-vram", type=int, default=80)
    p.add_argument("--all-regions", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--allow-arm", action="store_true", help="include arm64 boxes (Lambda GH200)")
    p = sub.add_parser("launch")
    p.add_argument("offer")
    p.add_argument("--pr", type=int, required=True)
    p.add_argument("--hours", type=float)
    p.add_argument("--tag", help="names the instance live-pr-<pr>-<tag> (and makes an Eras launch idempotent); "
                                "by default the time and 4 hex digits")
    p.add_argument("--run", help="Eras, DigitalOcean: a script that is the work itself (on Eras the GPUs go when it "
                                "ends)")
    p.add_argument("--input", action="append", default=[],
                   help="a file for the --run script, in /run/live-pr. No secrets here: the files stay in the "
                        "run's record; use eras.run_secret_id or digitalocean.run_secret_env")
    cache = p.add_mutually_exclusive_group()
    cache.add_argument("--cache", choices=("ro", "none"), default="ro",
                       help="Eras: mount a free one of eras.cache_volume_ids read-only at /cache (default), or none")
    cache.add_argument("--cache-write", metavar="ID",
                       help="Eras: mount this cache volume read-write at /cache instead, to fill it; dies if held")
    p.add_argument("--boxes", type=int, default=1,
                   help="budget.max_running must leave room for this many, this one and those launched after it "
                        "for the same job")
    p.add_argument("--allow-over-budget", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("wait")
    p.add_argument("instance")
    p.add_argument("--timeout", type=int, default=1200)
    p = sub.add_parser("ssh")
    p.add_argument("instance")
    p.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("push", "pull"):
        p = sub.add_parser(name)
        p.add_argument("instance")
        for arg in ("local", "remote") if name == "push" else ("remote", "local"):
            p.add_argument(arg)
    p = sub.add_parser("list")
    p.add_argument("--all", action="store_true")
    p = sub.add_parser("extend")
    p.add_argument("instance")
    p.add_argument("--hours", type=float, required=True)
    p.add_argument("--allow-over-budget", action="store_true")
    p = sub.add_parser("terminate")
    p.add_argument("instance")
    sub.add_parser("reap")
    p = sub.add_parser("watchdog")
    p.add_argument("instance")
    args = ap.parse_args()
    cfg = load_config()
    handler = {"check": cmd_check, "offers": cmd_offers, "launch": cmd_launch, "wait": cmd_wait,
               "ssh": cmd_ssh, "push": cmd_copy, "pull": cmd_copy, "list": cmd_list, "extend": cmd_extend,
               "terminate": cmd_terminate, "reap": cmd_reap, "watchdog": cmd_watchdog}[args.cmd]
    try:
        handler(args, cfg)
    except ApiError as e:
        die(str(e))


if __name__ == "__main__":
    main()
