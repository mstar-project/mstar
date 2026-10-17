#!/usr/bin/env python3
"""Provision Hopper-or-newer GPU boxes on RunPod and Lambda for live PR tests.

usage:
  gpu.py check                                   keys, persistent storage, budget
  gpu.py offers --gpus N [--min-vram GB] [--all-regions] [--allow-arm] [--json]
  gpu.py launch <offer-id> --pr N [--hours H] [--allow-over-budget] [--dry-run]
  gpu.py wait <instance>                         until ssh answers; prints how to reach it
  gpu.py ssh <instance> [command ...]            run a command (or a shell) on it
  gpu.py push <instance> <local> <remote>        scp -r to the box
  gpu.py pull <instance> <remote> <local>        scp -r from the box
  gpu.py list [--all]                            instances launched by this tool (--all: whole accounts)
  gpu.py extend <instance> --hours H             move the watchdog deadline to now + H
  gpu.py terminate <instance>

Offers only cover regions where the team has persistent storage (a RunPod global volume
mounts anywhere; a RunPod network volume or a Lambda filesystem pins one region), since the
HF cache lives there; --all-regions lists the rest for
a "no availability" report. Prices are USD/hr for the whole box.

Every launch is recorded in a ledger and gets a detached watchdog that terminates it at
its deadline (budget.max_hours by default), even if the session that launched it is gone.

Keys come from RUNPOD_API_KEY / LAMBDA_API_KEY; settings from config.local.json next to
the skill (see config.example.json). Stdlib only, so it runs from any python3.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
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
# Hopper or newer. \b keeps out the H20 (Hopper silicon, a fraction of the compute).
HOPPER_PLUS = re.compile(r"\b(H100|H200|GH200|B200|B300)\b", re.I)


def die(msg: str) -> None:
    print(f"gpu.py: {msg}", file=sys.stderr)
    sys.exit(1)


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

    def code(self) -> str:  # Lambda's machine-readable error code, if any
        return self.body.get("error", {}).get("code", "") if isinstance(self.body, dict) else ""


def http(method: str, url: str, key_env: str, body: dict | None = None, params: dict | None = None):
    key = os.environ.get(key_env)
    if not key:
        raise ApiError(0, f"{key_env} is not set")
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(6):
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {key}",
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


def enabled(cfg, provider) -> bool:
    key = {"runpod": "RUNPOD_API_KEY", "lambda": "LAMBDA_API_KEY"}[provider]
    return cfg.get(provider, {}).get("enabled", True) and bool(os.environ.get(key))


def cmd_offers(args, cfg):
    offers, errors = [], []
    for provider, fn in (("runpod", runpod_offers), ("lambda", lambda_offers)):
        if not enabled(cfg, provider):
            errors.append(f"{provider}: skipped (disabled or no API key)")
            continue
        try:
            offers += fn(cfg, args.gpus, args.min_vram, args.all_regions,
                         **({"allow_arm": args.allow_arm} if provider == "lambda" else {}))
        except ApiError as e:
            errors.append(f"{provider}: {e}")
    offers.sort(key=lambda o: (o["usd_per_hour"], o["count"]))
    cap = cfg["budget"]["max_usd_per_hour"]
    if args.json:
        print(json.dumps({"offers": offers, "errors": errors, "max_usd_per_hour": cap}, indent=1))
        return
    for e in errors:
        print("#", e)
    if not offers:
        print(f"no availability: no Hopper+ box with >= {args.gpus} GPUs"
              + ("" if args.all_regions else " in a region with team storage (try --all-regions)"))
        return
    print(f"{'offer':58} {'$/hr':>7}  {'gpus':>4}  {'vram':>5}  avail   storage")
    for o in offers:
        flag = "" if o["usd_per_hour"] <= cap else f"  over ${cap}/hr cap"
        print(f"{o['id']:58} {o['usd_per_hour']:7.2f}  {o['count']:4}  {o['vram_gb']:4}G  "
              f"{o['availability']:7} {'yes' if o['storage'] else 'no '}{flag}")


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
    name = f"live-pr-{args.pr}-{time.strftime('%m%d-%H%M')}"
    max_instances = cfg["budget"].get("max_concurrent_instances", 1)
    max_total = cfg["budget"].get("max_total_usd")

    if o["provider"] == "runpod":
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
        print(json.dumps(body, indent=1).replace(pubkey, "<public key>"))
        return

    # The caps are checked and the launch recorded under one lock, so parallel
    # sessions can't each pass the check and launch together.
    with ledger_txn() as d:
        if len(d) >= max_instances and not args.allow_over_budget:
            die(f"{len(d)} instance(s) already up ({', '.join(d)}); budget.max_concurrent_instances is "
                f"{max_instances}. Wait for one to finish, or ask before passing --allow-over-budget")
        committed = sum(committed_usd(e) for e in d.values()) + o["usd_per_hour"] * hours
        if max_total is not None and committed > max_total and not args.allow_over_budget:
            die(f"running to their deadlines, the live instances plus this one would cost ~${committed:.0f}, "
                f"over budget.max_total_usd ${max_total}; ask before passing --allow-over-budget")
        try:
            if o["provider"] == "runpod":
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
              "usd_per_hour": o["usd_per_hour"], "gpu": o["gpu"], "count": o["count"], "region": o["region"],
              "launched_at": now, "deadline": now + hours * 3600,
              "user": "root" if o["provider"] == "runpod" else "ubuntu",
              "hf_home": hf_home(cfg, o["provider"], o["region"])}
    log = STATE / f"{key.replace(':', '_')}.watchdog.log"
    with open(log, "a") as fh:
        subprocess.Popen([sys.executable, __file__, "watchdog", key], stdout=fh, stderr=fh,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"launched {key} ({o['count']}x {o['gpu']}, {o['region']}, ${o['usd_per_hour']}/hr); "
          f"watchdog terminates it at {time.strftime('%H:%M', time.localtime(now + hours * 3600))} ({hours} h)")
    print(key)


def hf_home(cfg, provider, region) -> str:
    if provider == "runpod":
        return cfg["runpod"]["hf_home"]
    fs = lambda_filesystems(cfg)[region]
    return cfg["lambda"]["hf_home"].format(fs=fs["name"], mount=fs.get("mount_point") or f"/lambda/nfs/{fs['name']}")


# ---------------------------------------------------------------- status / ssh

def status(e: dict) -> tuple[str, str | None, int]:
    """(status, host, ssh port)."""
    if e["provider"] == "runpod":
        p = runpod("GET", f"/pods/{e['instance_id']}")
        direct = (p.get("ssh") or {}).get("direct") or {}
        return p.get("status", "?"), direct.get("host"), int(direct.get("port") or 22)
    i = lambda_("GET", f"/instances/{e['instance_id']}") or {}
    return i.get("status", "?"), i.get("ip"), 22


def ssh_base(e: dict, scp: bool = False) -> list[str]:
    known = STATE / f"{e['provider']}_{e['instance_id']}.known_hosts"
    return ["scp" if scp else "ssh", "-P" if scp else "-p", str(e["port"]),
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known}",
            "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=15"]


def cmd_wait(args, cfg):
    e = entry(args.instance)
    t0 = time.time()
    while True:
        st, host, port = status(e)
        if st.lower() in ("exited", "error", "terminated", "terminating", "unhealthy", "preempted"):
            die(f"{args.instance} is {st}")
        if host:
            e["host"], e["port"] = host, port
            probe = ssh_base(e) + ["-o", "BatchMode=yes", f"{e['user']}@{host}", "nvidia-smi -L"]
            r = subprocess.run(probe, capture_output=True, text=True, check=False)
            if r.returncode == 0:
                with ledger_txn() as d:
                    if args.instance in d:
                        d[args.instance].update(host=host, port=port)
                print(r.stdout.strip())
                print(f"ready after {int(time.time() - t0)}s: {' '.join(ssh_base(e))} {e['user']}@{host}")
                print(f"hf_home: {e['hf_home']}")
                return
        if time.time() - t0 > args.timeout:
            die(f"{args.instance} not reachable after {args.timeout}s (status {st}, host {host})")
        time.sleep(10)


def reachable(key: str) -> dict:
    e = entry(key)
    if not e.get("host"):
        die(f"run `gpu.py wait {key}` first")
    return e


def cmd_ssh(args, cfg):
    e = reachable(args.instance)
    os.execvp("ssh", ssh_base(e) + [f"{e['user']}@{e['host']}"] + args.command)


def cmd_copy(args, cfg):
    e = reachable(args.instance)
    target = f"{e['user']}@{e['host']}:{args.remote}"
    src, dst = (args.local, target) if args.cmd == "push" else (target, args.local)
    sys.exit(subprocess.run(ssh_base(e, scp=True) + ["-r", src, dst], check=False).returncode)


# ---------------------------------------------------------------- list / extend / terminate

def terminate(key: str, e: dict) -> None:
    try:
        if e["provider"] == "runpod":
            runpod("DELETE", f"/pods/{e['instance_id']}")
        else:
            lambda_("POST", "/instance-operations/terminate", {"instance_ids": [e["instance_id"]]})
    except ApiError as err:
        if err.status != 404 and "does-not-exist" not in err.code():
            raise
    with ledger_txn() as d:
        d.pop(key, None)
    hours = (time.time() - e["launched_at"]) / 3600
    print(f"terminated {key} after {hours:.2f} h, ~${hours * e['usd_per_hour']:.2f}")


def cmd_terminate(args, cfg):
    terminate(args.instance, entry(args.instance))


def cmd_extend(args, cfg):
    entry(args.instance)
    with ledger_txn() as d:
        e = d[args.instance]
        e["deadline"] = time.time() + args.hours * 3600
    print(f"{args.instance} deadline now {time.strftime('%H:%M', time.localtime(e['deadline']))}")


def cmd_list(args, cfg):
    d = ledger()
    if not d:
        print("no instances launched by this tool are up")
    for key, e in d.items():
        try:
            st = status(e)[0]
        except ApiError as err:
            st = f"? ({err.status})"
        hrs = (time.time() - e["launched_at"]) / 3600
        left = (e["deadline"] - time.time()) / 60
        print(f"{key}  pr {e['pr']}  {e['count']}x {e['gpu']} {e['region']}  {st}  "
              f"{hrs:.2f} h, ~${hrs * e['usd_per_hour']:.2f}  deadline in {left:.0f} min")
    if args.all:
        print("--- every instance on the accounts ---")
        if enabled(cfg, "runpod"):
            for p in runpod_list("/pods", "pods"):
                print(f"runpod:{p['id']}  {p.get('name')}  {p.get('status')}  ${p.get('cost', '?')}/hr")
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
