#!/usr/bin/env python3
"""Provision Hopper-or-newer GPUs for live PR tests: a Slurm node, Eras Cloud, RunPod or Lambda.

usage:
  gpu.py check                                   keys, persistent storage, budget
  gpu.py offers --gpus N [--min-vram GB] [--all-regions] [--allow-arm] [--json]
  gpu.py launch <offer-id> --pr N [--hours H] [--tag T] [--run SCRIPT [--input FILE ...]]
               [--cache ro|none | --cache-write ID] [--allow-over-budget] [--dry-run]
  gpu.py wait <instance>                         until ssh answers; prints how to reach it
  gpu.py ssh <instance> [command ...]            run a command (or a shell) on it
  gpu.py push <instance> <local> <remote>        scp -r to the box
  gpu.py pull <instance> <remote> <local>        scp -r from the box
  gpu.py list [--all]                            instances launched here (--all: by any session; all RunPod, Lambda)
  gpu.py extend <instance> --hours H             move the watchdog deadline to now + H
  gpu.py terminate <instance>

Offers only cover regions where the team has persistent storage (a RunPod global volume
mounts anywhere; a RunPod network volume or a Lambda filesystem pins one region), since the
HF cache lives there; --all-regions lists the rest for
a "no availability" report. Prices are USD/hr for the whole box.

A Slurm node (`slurm` in the config) is offered first when it has the GPUs free now: an
"instance" there is a batch job that holds them, every command runs inside it with
`srun --overlap`, and Slurm's `--time` ends it at the deadline. Serve with `mserve` there,
never `--gpus`. An Eras instance runs sshd as root in a Pod that Kubernetes stops at the
deadline (`activeDeadlineSeconds`, `restartPolicy: Never`); a run that ends releases it.

`launch --run SCRIPT` makes the script the work itself rather than holding the GPUs for ssh.
The Slurm job runs Slurm's copy of it, in the job's own directory under slurm.workdir, which
holds the --input files; terminate (and so the watchdog) deletes that directory, so pull what
you need first. On Eras it is the Pod's command, with the inputs in /run/live-pr and sshd up
beside it for copying results back; the instance is released 15 minutes after it exits.
Either way the GPUs go when the script ends, whether or not whoever launched it is still there.

Every launch is recorded in a ledger and gets a detached watchdog that terminates it at
its deadline (budget.max_hours by default), even if the session that launched it is gone.

`launch` exits 4 when budget.max_running refuses it, 1 on any other error.

Keys come from RUNPOD_API_KEY / LAMBDA_API_KEY / ERAS_API_KEY; the Slurm node is reached by
ssh. Settings from config.local.json next to the skill (see config.example.json). Stdlib
only, so it runs from any python3.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import functools
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
STATE = Path(os.environ.get("LIVE_PR_STATE", "~/.local/state/mstar-live-pr-test")).expanduser()
LEDGER = STATE / "instances.json"
RUNPOD = "https://api.runpod.io/v2"
RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"
LAMBDA = "https://cloud.lambda.ai/api/v1"
ERAS = "https://api.erascloud.ai"
# Hopper or newer. \b keeps out the H20 (Hopper silicon, a fraction of the compute).
HOPPER_PLUS = re.compile(r"\b(H100|H200|GH200|B200|B300)\b", re.I)


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
         key_header: str = "Authorization", key_prefix: str = "Bearer "):
    key = os.environ.get(key_env)
    if not key:
        raise ApiError(0, f"{key_env} is not set")
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
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
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = raw[:500]
            if e.code == 429 or e.code >= 500:
                time.sleep(float(e.headers.get("Retry-After") or 2 ** attempt * 3))
                continue
            raise ApiError(e.code, parsed) from None
        except urllib.error.URLError:
            time.sleep(2 ** attempt * 3)
    raise ApiError(-1, f"{method} {url}: gave up after retries")


def runpod(method, path, body=None, params=None):
    return http(method, RUNPOD + path, "RUNPOD_API_KEY", body, params)


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


def lambda_(method, path, body=None):
    return http(method, LAMBDA + path, "LAMBDA_API_KEY", body).get("data")


_eras_org: str | None = None


def eras(method, path, body=None, params=None):
    """Eras Cloud, under the key's default organization."""
    global _eras_org
    def call(m, url, b=None, q=None):
        return http(m, ERAS + url, "ERAS_API_KEY", b, q, "x-api-key", "")
    _eras_org = _eras_org or call("GET", "/v1/me")["default_organization"]["id"]
    return call(method, f"/v1/org/{_eras_org}{path}", body, params)


def slurm_run(cfg: dict, command: str, check: bool = True) -> str:
    """One shell command on the Slurm login host; stdout."""
    r = subprocess.run(slurm_ssh(cfg) + [command], capture_output=True, text=True, check=False, timeout=120)
    if check and r.returncode != 0:
        raise ApiError(r.returncode, f"slurm: {command.split(maxsplit=1)[0]}: {(r.stderr or r.stdout).strip()[-400:]}")
    return r.stdout


def slurm_cancel(cfg: dict, job: str, jobdir: str | None) -> None:
    """Cancel a job (scancel's arguments) and, once it has gone, remove its directory: on a shared
    node nothing of a run stays but what was copied back (weights live in slurm.hf_home)."""
    rm = f"sleep 60; rm -rf {shlex.quote(jobdir)}" if jobdir else ":"
    slurm_run(cfg, f"scancel {job}; (nohup sh -c {shlex.quote(rm)} > /dev/null 2>&1 < /dev/null &)")


def slurm_ssh(cfg: dict, scp: bool = False) -> list[str]:
    sl = cfg["slurm"]
    known = STATE / f"slurm_{re.sub(r'[^A-Za-z0-9.]', '_', sl['host'])}.known_hosts"
    STATE.mkdir(parents=True, exist_ok=True)
    if sl.get("host_key"):  # pinned: the node's ngrok port is reused across machines
        host = sl["host"].split("@")[-1]
        hostport = host if sl.get("port", 22) == 22 else f"[{host}]:{sl['port']}"
        known.write_text(f"{hostport} {sl['host_key']}\n")
    cmd = ["scp" if scp else "ssh", "-P" if scp else "-p", str(sl.get("port", 22)),
           "-o", f"StrictHostKeyChecking={'yes' if sl.get('host_key') else 'accept-new'}",
           "-o", f"UserKnownHostsFile={known}",
           "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes"]
    if sl.get("identity_file"):
        cmd += ["-i", str(Path(sl["identity_file"]).expanduser())]
    return cmd if scp else cmd + [sl["host"]]


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

def runpod_offers(cfg, gpus, min_vram, all_regions) -> list[dict]:
    vols = runpod_volumes(cfg)
    cat = runpod("GET", "/catalog/gpus", params={
        "include": "AVAILABILITY", "product": "POD", "count": gpus, "cloud": "SECURE"})
    offers = []
    for g in cat.get("gpus") or []:
        if not HOPPER_PLUS.search(g["id"]) or (g.get("memory") or 0) < min_vram:
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


def lambda_vram(desc: str) -> int | None:
    m = re.search(r"(\d+)\s*GB", desc)
    return int(m.group(1)) if m else None  # unknown: don't filter on it


def lambda_offers(cfg, gpus, min_vram, all_regions, allow_arm=False) -> list[dict]:
    fss = lambda_filesystems(cfg)
    offers = []
    for name, entry in (lambda_("GET", "/instance-types") or {}).items():
        it = entry["instance_type"]
        desc = it.get("gpu_description", "")
        n = it.get("specs", {}).get("gpus", 0)
        vram = lambda_vram(desc)
        if not HOPPER_PLUS.search(desc) or n < gpus or (vram is not None and vram < min_vram):
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


GIB = re.compile(r"(\d+(?:\.\d+)?)\s*([GT])i?", re.I)


def eras_offers(cfg, gpus, min_vram, all_regions) -> list[dict]:
    # Hopper+ is checked here rather than in the filter: Eras names models "H200", "B300", ...
    flt = json.dumps({"field": "hardware.gpu.count", "op": ">=", "value": gpus})
    page, offers = eras("GET", "/compute-listings", params={"page_size": 100, "filter": flt}), []
    for x in page.get("items") or []:
        g = (x.get("hardware") or {}).get("gpu") or {}
        m = GIB.match(str(g.get("memory_per_device") or ""))
        vram = int(float(m.group(1)) * (1024 if m.group(2).upper() == "T" else 1)) if m else None
        if not HOPPER_PLUS.search(str(g.get("model") or "")) or (vram is not None and vram < min_vram):
            continue
        if x.get("hourly_price") is None:
            continue
        now = (x.get("access_method") or {}).get("is_available") and (x.get("launches_available") or 0) > 0
        if now or all_regions:
            # Local NVMe only: stage_model pulls weights from the Hub (the default hf_home is on it).
            offers.append({
                "id": f"eras:{x['id']}", "provider": "eras", "gpu": g["model"], "count": g["count"],
                "vram_gb": vram or "?", "region": x.get("name", "eras"), "usd_per_hour": round(x["hourly_price"], 2),
                "availability": "now" if now else "none", "storage": True,
            })
    return offers


def gres_gpus(gres: str) -> int:
    """The GPU count in "gpu:8(S:0-1)" or "gpu:(null):5(IDX:0-3,6)": drop the parentheses, then it is last."""
    m = re.search(r"gpu:(?:[^:,]*:)?(\d+)", re.sub(r"\([^)]*\)", "", gres))
    return int(m[1]) if m else 0


def slurm_offers(cfg, gpus, min_vram, all_regions) -> list[dict]:
    """A node with `gpus` free now. Never more than slurm.max_gpus: the node is shared."""
    sl = cfg["slurm"]
    if gpus > sl.get("max_gpus", 4):
        return []
    offers = []
    partition = shlex.quote(sl.get("partition", "main"))
    out = slurm_run(cfg, f"sinfo -h -N -p {partition} -O 'NodeHost:40,Gres:40,GresUsed:60,StateLong:20'")
    for line in out.splitlines():
        host, gres, used, state = (line.split() + ["", "", "", ""])[:4]
        total, busy = gres_gpus(gres), gres_gpus(used)
        free = 0 if any(s in state for s in ("drain", "down", "fail")) else total - busy
        if free >= gpus or (all_regions and total >= gpus):
            offers.append({
                "id": f"slurm:{host}:{gpus}", "provider": "slurm", "gpu": sl.get("gpu", "?"), "count": gpus,
                "vram_gb": sl.get("vram_gb", "?"), "region": host, "usd_per_hour": 0.0,
                "availability": f"{free} free" if free >= gpus else f"busy ({free} free)", "storage": True,
            })
    return offers


PROVIDERS = {"slurm": slurm_offers, "eras": eras_offers, "runpod": runpod_offers, "lambda": lambda_offers}


def enabled(cfg, provider) -> bool:
    if provider == "slurm":
        return bool(cfg.get("slurm", {}).get("host")) and cfg["slurm"].get("enabled", True)
    key = {"runpod": "RUNPOD_API_KEY", "lambda": "LAMBDA_API_KEY", "eras": "ERAS_API_KEY"}[provider]
    return cfg.get(provider, {}).get("enabled", True) and bool(os.environ.get(key))


def cmd_offers(args, cfg):
    offers, errors = [], []
    for provider, fn in PROVIDERS.items():
        if not enabled(cfg, provider):
            errors.append(f"{provider}: skipped (disabled, or no API key or host)")
            continue
        try:
            offers += fn(cfg, args.gpus, args.min_vram, args.all_regions,
                         **({"allow_arm": args.allow_arm} if provider == "lambda" else {}))
        except (ApiError, subprocess.TimeoutExpired) as e:
            errors.append(f"{provider}: {e}")
    # What can start now first, then the cheapest (a Slurm node costs nothing extra).
    offers.sort(key=lambda o: (o["availability"] in ("none",) or o["availability"].startswith("busy"),
                               o["usd_per_hour"], o["count"]))
    cap = cfg["budget"]["max_usd_per_hour"]
    if args.json:
        print(json.dumps({"offers": offers, "errors": errors, "max_usd_per_hour": cap}, indent=1))
        return
    for e in errors:
        print("#", e)
    if not any(o["availability"] not in ("none",) and not o["availability"].startswith("busy") for o in offers):
        print(f"no availability: no Hopper+ box with >= {args.gpus} GPUs"
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


def entry(key: str) -> dict:
    d = ledger()
    if key not in d:
        die(f"unknown instance {key!r}; known: {', '.join(d) or 'none'}")
    return d[key]


# ---------------------------------------------------------------- launch

def find_offer(cfg, offer_id: str) -> dict:
    provider = offer_id.split(":", 1)[0]
    if provider == "runpod":
        _, gpu, dc, count = offer_id.split(":")
        pool = runpod_offers(cfg, int(count), 0, True)
    elif provider == "lambda":
        pool = lambda_offers(cfg, 1, 0, True, allow_arm=True)
    elif provider == "slurm":
        pool = slurm_offers(cfg, int(offer_id.rsplit(":", 1)[1]), 0, False)
    elif provider == "eras":
        pool = eras_offers(cfg, 1, 0, False)
    else:
        die(f"bad offer id {offer_id!r}")
    for o in pool:
        if o["id"] == offer_id:
            return o
    die(f"{offer_id} is no longer available; rerun offers")


def cmd_launch(args, cfg):
    o = find_offer(cfg, args.offer)
    if not o["storage"]:
        die(f"{o['region']} has no team storage; the HF cache would be cold. Pick a region with storage.")
    cap, hours = cfg["budget"]["max_usd_per_hour"], args.hours or cfg["budget"]["max_hours"]
    if o["usd_per_hour"] > cap and not args.allow_over_budget:
        die(f"${o['usd_per_hour']}/hr is over the ${cap}/hr cap; ask before passing --allow-over-budget")
    if hours > cfg["budget"]["max_hours"] and not args.allow_over_budget:
        die(f"{hours} h is over the {cfg['budget']['max_hours']} h cap; ask before passing --allow-over-budget")
    pubkey = Path(cfg["ssh_public_key"]).expanduser().read_text().strip()
    name = job_name(cfg, o["provider"], args.pr, args.tag or time.strftime('%m%d-%H%M'))
    max_instances = cfg["budget"].get("max_concurrent_instances", 1)
    max_total = cfg["budget"].get("max_total_usd")
    max_running = cfg["budget"].get("max_running")

    # The caps are checked and the launch recorded under one lock, so parallel sessions can't each
    # pass the check and launch together; and the body is built after the caps, since building an
    # Eras one registers an ssh key. A dry run checks nothing and records nothing.
    with contextlib.nullcontext({}) if args.dry_run else ledger_txn() as d:
        if not args.dry_run:
            if len(d) >= max_instances and not args.allow_over_budget:
                die(f"{len(d)} instance(s) already up ({', '.join(d)}); budget.max_concurrent_instances is "
                    f"{max_instances}. Wait for one to finish, or ask before passing --allow-over-budget")
            if max_running is not None and not args.allow_over_budget:
                up, errors = running(cfg)
                if o["provider"] in errors:
                    die(f"can't count this tool's {o['provider']} jobs for budget.max_running: {errors[o['provider']]}")
                if len(up) >= max_running:
                    die(f"{len(up)} of this tool's jobs are up, from every session; budget.max_running is "
                        f"{max_running}. Wait for one to end, or ask before passing --allow-over-budget", code=4)
            committed = sum(committed_usd(e) for e in d.values()) + o["usd_per_hour"] * hours
            if max_total is not None and committed > max_total and not args.allow_over_budget:
                die(f"running to their deadlines, the live instances plus this one would cost ~${committed:.0f}, "
                    f"over budget.max_total_usd ${max_total}; ask before passing --allow-over-budget")
        if o["provider"] == "slurm":
            sl, n = cfg["slurm"], o["count"]
            jobdir = job_dir(cfg, name)
            body = " ".join(["sbatch", "--parsable", "-p", shlex.quote(sl.get("partition", "main")),
                             f"--gres=gpu:{n}", f"-c {sl.get('cpus_per_gpu', 24) * n}",
                             f"--mem={sl.get('mem_gb_per_gpu', 180) * n}G", f"--time={max(1, int(hours * 60))}",
                             f"--nice={sl.get('nice', 100)}", "--no-requeue", f"-J {shlex.quote(name)}",
                             shlex.quote(f"--export=ALL,HF_HOME={hf_home(cfg, 'slurm', None)}"),
                             *([f"--chdir={shlex.quote(jobdir)}", f"-o {shlex.quote(jobdir)}/job.log",
                                shlex.quote(f"{jobdir}/{Path(args.run).name}")] if args.run
                               else ["-o /dev/null", "--wrap 'sleep infinity'"])])
        elif o["provider"] == "eras":  # tried in turn: each free cache volume, then none (cold); a refresh only its own
            free = eras_cache_volumes(cfg, args.cache, args.cache_write)
            candidates = free if args.cache_write else [*free, None]
            eras_body = functools.partial(eras_launch_body, cfg, o, name, pubkey, hours, args.run, args.input,
                                          rw=bool(args.cache_write))
            body = eras_body(candidates[0])
        elif o["provider"] == "runpod":
            rp = cfg["runpod"]
            vol = runpod_volume_for(runpod_volumes(cfg), o["region"])
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
        else:
            lm = cfg["lambda"]
            fs = lambda_filesystems(cfg)[o["region"]]
            body = {
                "name": name, "region_name": o["region"], "instance_type_name": o["id"].split(":")[1],
                "ssh_key_names": [lm["ssh_key_name"]], "file_system_names": [fs["name"]],
                "image": {"family": lm.get("image_family", "lambda-stack-24-04")},
            }
        if args.dry_run:
            print(body if isinstance(body, str) else json.dumps(body, indent=1).replace(pubkey, "<public key>"))
            return
        try:
            if o["provider"] == "slurm":
                if args.run:  # the job's directory, script and inputs exist before it can start
                    slurm_run(cfg, f"umask 077 && mkdir -p {shlex.quote(jobdir)}")
                    for f in [args.run, *args.input]:  # scp keeps the mode, so a 0600 key file stays 0600
                        subprocess.run(slurm_ssh(cfg, scp=True) + ["-p", f, f"{sl['host']}:{jobdir}/"], check=True,
                                       capture_output=True, timeout=300)
                iid = slurm_run(cfg, body).strip().split(";")[0]
            elif o["provider"] == "eras":
                for i, vid in enumerate(candidates):
                    try:
                        iid = eras_create(eras_body(vid) if i else body)
                        break
                    except ApiError as err:  # another box took the cache volume since: the next free one, then cold
                        if not volume_refused(err) or vid is None:
                            raise
                        if args.cache_write:  # a refresh without its volume would write nothing that lasts
                            die(f"cache volume {vid} is held or being deleted: {err}")
                        print(f"gpu.py: cache volume {vid} is held or being deleted ({str(err)[:200]}); "
                              "trying the next free one, else cold", file=sys.stderr)
            elif o["provider"] == "runpod":
                iid = runpod("POST", "/pods", body)["id"]
            else:
                iid = lambda_("POST", "/instance-operations/launch", body)["instance_ids"][0]
        except ApiError as e:
            if e.status in (400, 409) or "capacity" in e.code():
                die(f"no availability: {o['id']} was taken before launch ({e}); rerun offers")
            raise
        key = f"{o['provider']}:{iid}"
        now = time.time()
        d[key] = {"provider": o["provider"], "instance_id": iid, "name": name, "offer": o["id"], "pr": args.pr,
              "run": bool(args.run), "dir": jobdir if o["provider"] == "slurm" and args.run else None,
              "usd_per_hour": o["usd_per_hour"], "gpu": o["gpu"], "count": o["count"], "region": o["region"],
              "launched_at": now, "deadline": now + hours * 3600,
              "user": "ubuntu" if o["provider"] == "lambda" else "root",
              "hf_home": hf_home(cfg, o["provider"], o["region"])}
    log = STATE / f"{key.replace(':', '_')}.watchdog.log"
    with open(log, "a") as fh:
        subprocess.Popen([sys.executable, __file__, "watchdog", key], stdout=fh, stderr=fh,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"launched {key} ({o['count']}x {o['gpu']}, {o['region']}, ${o['usd_per_hour']}/hr); "
          f"watchdog terminates it at {time.strftime('%H:%M', time.localtime(now + hours * 3600))} ({hours} h)")
    print(key)


def job_prefix(cfg: dict, provider: str) -> str:
    """What the names of this tool's jobs and instances start with: slurm.job_prefix on a shared node."""
    return (cfg.get("slurm", {}).get("job_prefix", "live-pr") if provider == "slurm" else "live-pr") + "-"


def job_name(cfg: dict, provider: str, pr: int, tag: str) -> str:
    return f"{job_prefix(cfg, provider)}{pr}-{tag}"


def job_dir(cfg: dict, name: str) -> str:
    """A Slurm job's directory: its working directory, holding a --run script and its inputs."""
    return f"{cfg['slurm'].get('workdir', '/workspace/live-pr')}/{name}"


ERAS_GPU_NOTEBOOK = "ed6d2a30-36aa-5d50-b911-1f6b551f2081"  # Eras Cloud Notebook (NVIDIA GPU): sshd as root


def eras_ssh_key_id(pubkey: str, name: str) -> str:
    """The organization's id for this public key. A key it doesn't know yet is registered under the
    instance's name, and a terminate after that instance ends deletes it (eras_sweep_keys)."""
    import base64
    import hashlib
    blob = base64.b64decode(pubkey.split()[1])
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
    for k in eras("GET", "/ssh-keys", params={"page_size": 100}).get("items") or []:
        if k.get("fingerprint") == fingerprint:
            return k["id"]
    return eras("POST", "/ssh-keys", {"name": name, "public_key": pubkey})["id"]


def eras_launch_body(cfg: dict, o: dict, name: str, pubkey: str, hours: float, run: str | None = None,
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
    box["args"] = ["set -e\ninstall -d -m 0700 /root/.ssh /run/sshd\n"
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
        config_maps = {"live-pr": {"data": {Path(f).name: Path(f).read_text() for f in [run, *inputs]}}}
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
                "bindings": {"secrets": {"authorized-keys": {"ssh_key_ids": [eras_ssh_key_id(pubkey, name)]},
                                         **({"live-pr-key": {"secret_id": er["run_secret_id"]}}
                                            if run and er.get("run_secret_id") else {})},
                             "volumes": {"workspace": "emptyDir", **claims}},
                "release_when": {"after_success_seconds": keep, "after_failure_seconds": keep}},
    }


def eras_cache_volumes(cfg: dict, cache: str, write: str | None) -> list[str]:
    """The cache volumes a launch tries to bind, in order. An instance binds one volume, so
    eras.cache_volume_ids are replicas, one per box: "ro" tries those no box holds, then none (the
    box starts cold). `write` (a cache refresh) takes only that volume, and dies if a box holds it."""
    ids = [write] if write else [] if cache == "none" else cfg.get("eras", {}).get("cache_volume_ids") or []
    free = []
    for vid in ids:
        att = eras("GET", f"/volumes/{vid}").get("attachment") or {}
        if not att.get("instance_id") or att.get("detached_at"):
            free.append(vid)
        elif write:
            die(f"cache volume {vid} is held by instance {att['instance_id']}; wait for it to end")
    if ids and not free:
        print("gpu.py: every cache volume is held by another box; launching cold", file=sys.stderr)
    return free


# The error codes of a launch refused over a volume another instance holds, or one being deleted.
VOLUME_HELD = ("volume_bound", "volume_deleting", "volume_binding_conflict", "volume_binding_deleting",
               "resource_in_use")


def volume_refused(err: ApiError) -> bool:
    """Whether Eras refused a launch over a volume binding, by its error code, never by its wording."""
    return err.code() in VOLUME_HELD


def eras_create(body: dict) -> str:
    """Preview, accept the quoted prices, create; the instance id. Eras refuses an idempotency key
    reused with another request, so each attempt (listing, cache volume, ssh key) has its own, and
    the same attempt sent again, after a timeout or by a rerun with the same --tag, keeps it."""
    import hashlib
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
    binds = body["run"]["bindings"]
    attempt = json.dumps([body["listing_id"], (binds["volumes"].get("cache") or {}).get("volume_id"),
                          binds["secrets"]["authorized-keys"]["ssh_key_ids"]])
    body = dict(body, accepted_prices={i["usage_type"]: i["unit_price"] for i in preview["quote"]["line_items"]},
                accepted_available_until=preview.get("available_until"),
                idempotency_key=f"{body['name']}-{hashlib.sha256(attempt.encode()).hexdigest()[:8]}")
    return eras("POST", "/instances", body)["id"]


def hf_home(cfg, provider, region) -> str:
    if provider == "slurm":
        return cfg["slurm"].get("hf_home", "/workspace/shared/hf")
    if provider == "eras":
        return cfg.get("eras", {}).get("hf_home", "/workspace/hf")
    if provider == "runpod":
        return cfg["runpod"]["hf_home"]
    fs = lambda_filesystems(cfg)[region]
    return cfg["lambda"]["hf_home"].format(fs=fs["name"], mount=fs.get("mount_point") or f"/lambda/nfs/{fs['name']}")


# ---------------------------------------------------------------- status / ssh

def status(e: dict, cfg: dict | None = None) -> tuple[str, str | None, int]:
    """(status, host, ssh port)."""
    if e["provider"] == "slurm":
        try:  # "ended" only on squeue's word: a failed ssh says nothing about the job
            st = slurm_run(cfg or load_config(), f"squeue -h -j {e['instance_id']} -o %T").strip()
        except ApiError as err:
            if "Invalid job id" not in str(err):
                raise
            st = ""
        return (st or "ended").lower(), ("slurm" if st == "RUNNING" or (st == "" and e.get("run")) else None), 0
    if e["provider"] == "eras":
        i = eras("GET", f"/instances/{e['instance_id']}")
        ssh = next((p for p in (i.get("standalone") or {}).get("endpoints") or []
                    if p.get("name") == "ssh" and p.get("ready")), None)
        run = ((i.get("standalone") or {}).get("current_run") or {}).get("status")
        st = f"run {run}" if e.get("run") and run in ("succeeded", "failed", "stopped") else i.get("status", "?")
        return st, ssh and ssh.get("host"), int((ssh or {}).get("port") or 22)
    if e["provider"] == "runpod":
        p = runpod("GET", f"/pods/{e['instance_id']}")
        direct = (p.get("ssh") or {}).get("direct") or {}
        return p.get("status", "?"), direct.get("host"), int(direct.get("port") or 22)
    i = lambda_("GET", f"/instances/{e['instance_id']}") or {}
    return i.get("status", "?"), i.get("ip"), 22


def ssh_base(e: dict, cfg: dict, scp: bool = False) -> list[str]:
    known = STATE / f"{e['provider']}_{e['instance_id']}.known_hosts"
    return ["scp" if scp else "ssh", "-P" if scp else "-p", str(e["port"]),
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known}",
            "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=15",
            # the private half of ssh_public_key, when ssh wouldn't try it by default
            *(["-i", str(Path(cfg["identity_file"]).expanduser())] if cfg.get("identity_file") else [])]


def remote(e: dict, cfg: dict, command: list[str] | str, tty: bool = False) -> list[str]:
    """The ssh command line that runs `command` on the box (inside the job, on Slurm)."""
    cmd = command if isinstance(command, str) else " ".join(command)
    if e["provider"] == "slurm":
        shell = f"bash -lc {shlex.quote(cmd)}" if cmd else "--pty bash -l"
        inner = f"srun --jobid={e['instance_id']} --overlap {shell}"
        return slurm_ssh(cfg)[:1] + (["-t"] if tty or not cmd else []) + slurm_ssh(cfg)[1:] + [inner]
    return ssh_base(e, cfg) + [f"{e['user']}@{e['host']}"] + ([cmd] if cmd else [])


def cmd_wait(args, cfg):
    e = entry(args.instance)
    t0 = time.time()
    while True:
        st, host, port = status(e, cfg)
        if st.lower() in ("exited", "error", "terminated", "terminating", "unhealthy", "preempted",
                          "ended", "releasing", "cancelled", "failed", "timeout", "completed"):
            die(f"{args.instance} is {st}")
        if host:
            e["host"], e["port"] = host, port
            probe = remote(e, cfg, "nvidia-smi -L")
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
    if not e.get("host") and e["provider"] != "slurm":  # a Slurm job is reached through the login host
        die(f"run `gpu.py wait {key}` first")
    return e


def cmd_ssh(args, cfg):
    e = reachable(args.instance)
    os.execvp("ssh", remote(e, cfg, args.command))


def cmd_copy(args, cfg):
    e = reachable(args.instance)
    if e["provider"] == "slurm":  # the job's files are the login host's files; relative to its directory
        rel = args.remote if args.remote.startswith(("/", "~")) or not e.get("dir") else f"{e['dir']}/{args.remote}"
        target, scp = f"{cfg['slurm']['host']}:{rel}", slurm_ssh(cfg, scp=True)
    else:
        target, scp = f"{e['user']}@{e['host']}:{args.remote}", ssh_base(e, cfg, scp=True)
    src, dst = (args.local, target) if args.cmd == "push" else (target, args.local)
    sys.exit(subprocess.run(scp + ["-r", src, dst], check=False).returncode)


# ---------------------------------------------------------------- list / extend / terminate

def terminate(key: str, e: dict) -> None:
    try:
        if e["provider"] == "slurm":
            slurm_cancel(load_config(), e["instance_id"], e.get("dir"))
        elif e["provider"] == "eras":
            eras("POST", f"/instances/{e['instance_id']}/release", {})
        elif e["provider"] == "runpod":
            runpod("DELETE", f"/pods/{e['instance_id']}")
        else:
            lambda_("POST", "/instance-operations/terminate", {"instance_ids": [e["instance_id"]]})
    except ApiError as err:
        if err.status != 404 and "does-not-exist" not in err.code():
            raise
    with ledger_txn() as d:
        d.pop(key, None)
    if e["provider"] == "eras":
        eras_sweep_keys()
    hours = (time.time() - e["launched_at"]) / 3600
    print(f"terminated {key} after {hours:.2f} h, ~${hours * e['usd_per_hour']:.2f}")


def eras_sweep_keys() -> None:
    """Best effort: delete the ssh keys launches registered (each named after its instance) whose
    instance has ended or is gone. A key can't go while its instance releases, so each terminate
    deletes those earlier ones left. One under 15 minutes old may be a launch's in progress, its
    instance not created yet."""
    try:
        live = {i.get("name") for i in eras("GET", "/instances", params={"page_size": 100}).get("items") or []
                if i.get("status") != "ended"}
        keys = eras("GET", "/ssh-keys", params={"page_size": 100}).get("items") or []
    except ApiError as err:
        print(f"gpu.py: ssh keys not swept: {err}", file=sys.stderr)
        return
    recent = time.time() - 900
    for k in keys:
        created = datetime.datetime.fromisoformat(k["created_at"].replace("Z", "+00:00")).timestamp()
        if str(k.get("name")).startswith(job_prefix({}, "eras")) and k["name"] not in live and created < recent:
            try:  # one a later launch reuses (same public key) stays while that instance holds it
                eras("DELETE", f"/ssh-keys/{k['id']}")
            except ApiError as err:
                print(f"gpu.py: ssh key {k['name']} not deleted: {err}", file=sys.stderr)


def cmd_terminate(args, cfg):
    terminate(args.instance, entry(args.instance))


def cmd_extend(args, cfg):
    entry(args.instance)
    with ledger_txn() as d:
        e = d[args.instance]
        e["deadline"] = time.time() + args.hours * 3600
    print(f"{args.instance} deadline now {time.strftime('%H:%M', time.localtime(e['deadline']))}")


def running(cfg: dict) -> tuple[list[str], dict[str, str]]:
    """This tool's jobs and instances up now on the Slurm node and Eras, from every session and
    machine (the ledger has only this machine's), one line each; and why a provider couldn't be listed."""
    up, errors = [], {}
    if enabled(cfg, "slurm"):
        try:
            out = slurm_run(cfg, "squeue -h -o '%j %i %T %M/%l'")
            up += [f"slurm: {ln}" for ln in out.splitlines() if ln.startswith(job_prefix(cfg, "slurm"))]
        except (ApiError, subprocess.TimeoutExpired) as err:
            errors["slurm"] = str(err)
    if enabled(cfg, "eras"):
        try:
            items = eras("GET", "/instances", params={"page_size": 100}).get("items") or []
        except (ApiError, subprocess.TimeoutExpired) as err:
            errors["eras"], items = str(err), []
        for i in items:
            if str(i.get("name")).startswith(job_prefix(cfg, "eras")) and i.get("status") not in ("ended", "releasing"):
                up.append(f"eras:{i['id']}  {i.get('name')}  {i.get('status')}")
    return up, errors


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
        print("--- this tool's on Slurm and Eras, from every session; all of RunPod and Lambda ---")
        if enabled(cfg, "runpod"):
            for p in runpod_list("/pods", "pods"):
                print(f"runpod:{p['id']}  {p.get('name')}  {p.get('status')}  ${p.get('cost', '?')}/hr")
        up, errors = running(cfg)
        for line in up + [f"{p}: {err}" for p, err in errors.items()]:
            print(line)
        if enabled(cfg, "lambda"):
            for i in lambda_("GET", "/instances") or []:
                print(f"lambda:{i['id']}  {i.get('name')}  {i.get('status')}  {i['instance_type']['name']}")


def cmd_watchdog(args, cfg):
    """Detached: terminate the instance at its deadline unless it is terminated first."""
    while True:
        e = ledger().get(args.instance)
        if e is None:
            return  # terminated normally
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
    for provider, fn in (("runpod", runpod_volumes), ("lambda", lambda_filesystems)):
        if not enabled(cfg, provider):
            print(f"{provider}: skipped (disabled or no API key)")
            continue
        try:
            stores = fn(cfg)
        except ApiError as e:
            print(f"{provider}: {e}")
            continue
        where = ", ".join(
            f"any data center (global volume {s['name']!r}, {s['id']})" if r == GLOBAL
            else f"{r} ({s.get('name') or s.get('id')})" for r, s in stores.items())
        print(f"{provider}: storage in {where or 'NO REGIONS'}")
        if provider == "runpod" and not stores:
            rp = cfg["runpod"]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("offers")
    p.add_argument("--gpus", type=int, required=True)
    p.add_argument("--min-vram", type=int, default=80)
    p.add_argument("--all-regions", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--allow-arm", action="store_true", help="include arm64 boxes (Lambda GH200)")
    p = sub.add_parser("launch")
    p.add_argument("offer")
    p.add_argument("--pr", type=int, required=True)
    p.add_argument("--hours", type=float)
    p.add_argument("--tag", help="names the instance <prefix>-<pr>-<tag> (and makes an Eras launch idempotent)")
    p.add_argument("--run", help="a script that is the work itself; the GPUs go when it ends")
    p.add_argument("--input", action="append", default=[],
                   help="a file for the --run script: in its working directory on Slurm, in /run/live-pr on Eras")
    cache = p.add_mutually_exclusive_group()
    cache.add_argument("--cache", choices=("ro", "none"), default="ro",
                       help="Eras: mount a free one of eras.cache_volume_ids read-only at /cache (default), or none")
    cache.add_argument("--cache-write", metavar="ID",
                       help="Eras: mount this cache volume read-write at /cache instead, to fill it; dies if it is held")
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
    p = sub.add_parser("terminate")
    p.add_argument("instance")
    p = sub.add_parser("watchdog")
    p.add_argument("instance")
    args = ap.parse_args()
    cfg = load_config()
    handler = {"check": cmd_check, "offers": cmd_offers, "launch": cmd_launch, "wait": cmd_wait,
               "ssh": cmd_ssh, "push": cmd_copy, "pull": cmd_copy, "list": cmd_list, "extend": cmd_extend,
               "terminate": cmd_terminate, "watchdog": cmd_watchdog}[args.cmd]
    try:
        handler(args, cfg)
    except ApiError as e:
        die(str(e))


if __name__ == "__main__":
    main()
