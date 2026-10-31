"""How many requests of L tokens a paged KV pool that keeps a window holds, with and without it.

Builds the KV manager as a server would: the model's own KV spec with the serving
config's ``resources:`` block applied (``apply_yaml_overrides``), sharded to one rank of
its node's tp * sp heads, on the GPU. It then admits requests one after another as a
server does and grows each one: a prefill of L - 128 tokens in one step, then 128 one-token
decode steps. It stops at the first request admission makes wait, or whose step is refused.
At one rank a request waits until it can be allotted its pages (`ask` / `reserve`); above
one rank no pool allots, so only a refused step stops it. Every request stays resident.
A one-step prefill holds all of a prompt's sliding K/V until it commits, so a server that
chunks a prompt's prefill can hold more requests of long prompts than this counts.
"without" is the same pool keeping every layer's tokens, with the same max_num_pages, so
the same bytes.

Only the checkpoint's metadata (configs, tokenizer) is read, from the registry's repo.
``--model-path`` reads another repo or a local directory instead:

    python benchmark/kv_capacity.py --config configs/muse_glimmer.yaml \\
        --model-path /path/to/checkpoint --out glimmer.json
"""
import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

# run as a file, import the tree this file is in
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mstar.communication.tensors import LocalTransferEngine  # noqa: E402
from mstar.engine.resources.kv.config import (  # noqa: E402
    KVReqConfig,
    KVSpec,
    KVStep,
    PagedKVConfig,
    PrefixSpan,
    StreamLayout,
)
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.kv.transfer import TransferEngineInfo  # noqa: E402
from mstar.engine.resources.spec import apply_yaml_overrides  # noqa: E402
from mstar.engine.resources.step import Segment, StepContext  # noqa: E402

DECODE = 128


def kv_configs(config, model_path=None):
    """The serving config's sliding KV pool for one rank, with and without its sliding layers."""
    from mstar.model.registry import get_model_class, model_init_kwargs

    serving = yaml.safe_load(Path(config).read_text())
    kwargs = {**model_init_kwargs(serving["model"]), **(serving.get("model_kwargs") or {})}
    if model_path:
        kwargs["model_path_hf"] = model_path
    specs = get_model_class(serving["model"])(**kwargs).get_node_resources()
    apply_yaml_overrides(specs, serving)
    sliding = [s for s in specs if isinstance(s, KVSpec) and isinstance(s.config, PagedKVConfig)
               and len(s.config.groups) > 1]
    if len(sliding) != 1:
        raise ValueError(f"{config} declares {len(sliding)} paged KV pools with full and sliding layers, not one")
    (spec,) = sliding
    cfg = spec.config
    # A copy declares what the pool did and keeps every token, since the pool derives its window
    # from its attentions; max_num_pages counts the same bytes either way.
    without = dataclasses.replace(cfg)
    # one rank's share of the heads, as the KV manager shards it
    groups = [g for g in serving["node_groups"] if spec.nodes & set(g["node_names"])]
    if not groups:
        raise ValueError(f"{config}: no node group holds {sorted(spec.nodes)}")
    shards = {g.get("tp_size", 1) * g.get("sp_size", 1) for g in groups}
    if len(shards) != 1:
        raise ValueError(f"{config}: the node groups holding {sorted(spec.nodes)} "
                         f"({[g['node_names'] for g in groups]}) shard its heads {sorted(shards)} ways, not one way")
    (shards,) = shards
    cfg.shard(shards)
    without.shard(shards)
    return cfg, without, int(serving.get("max_seq_len", cfg.max_seq_len)), shards


def manager(cfg, device="cuda"):
    return KVManager(
        cfg=cfg, name="kv", joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
        device=torch.device(device), dtype=torch.bfloat16,
    )


def step(m, rid, span):
    kv = KVStep(segments=(Segment(rid, "main", span),))
    ctx = StepContext(request_ids=(rid,), graph_walk="walk", slot=0, capture=False)
    if not m.admit(kv, ctx).ok:
        return False
    m.plan(kv, ctx)
    m.commit(kv, ctx)
    return True


def admitted(m, rid, prompt, decode, allots):
    """Whether admission lets ``rid`` in, with ``allots`` asking for its prompt's pages as a
    pool at one rank does."""
    layout = StreamLayout(spans=(PrefixSpan(prompt, prompt, "walk"),), growth=(("decode", 1),))
    m.ingest_request(rid, KVReqConfig(kv_layout={"main": layout}, loop_caps={"decode": decode}))
    if not allots:
        return True
    # a request that can never fit fails its ask
    return m.ask(rid, "node", "walk").ok and m.reserve(rid, "node", "walk", None) is None


def fill(cfg, length, shards=1, decode=DECODE, device="cuda"):
    m = manager(cfg, device)
    n = 0
    while True:
        rid = f"r{n}"
        if not (
            admitted(m, rid, length - decode, decode, allots=shards == 1)
            and step(m, rid, length - decode) and all(step(m, rid, 1) for _ in range(decode))
        ):
            m.remove_request(rid)
            break
        n += 1
    counts = m.page_counts()
    return {"requests": n, "used_pages": counts.used, "total_pages": counts.total,
            "kv_bytes": m.kv_cache.tensor.nbytes, "page_size": cfg.tokens_per_page,
            "groups": [(len(g.layers), g.page_size, None if g.retention is None else g.retention.context_budget)
                       for g in cfg.groups]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="serving config, e.g. configs/muse_glimmer.yaml")
    ap.add_argument("--out", help="JSON results, rewritten after each cell (required unless --one)")
    ap.add_argument("--lengths", default="8192,32768,131072",
                    help="request lengths; those past the config's max_seq_len are skipped")
    ap.add_argument("--model-path", help="read the checkpoint's configs from this repo or directory")
    ap.add_argument("--one", help="length,arm: run one cell in this process and print its JSON")
    a = ap.parse_args()
    if a.one:
        length, arm = a.one.split(",")
        with_cfg, without_cfg, _, shards = kv_configs(a.config, a.model_path)
        cfg = with_cfg if arm == "with" else without_cfg
        print("RESULT " + json.dumps(fill(cfg, int(length), shards)), flush=True)
        return
    if not a.out:
        ap.error("--out is required")
    _, _, max_seq_len, _ = kv_configs(a.config, a.model_path)
    lengths = [n for n in map(int, a.lengths.split(",")) if n <= max_seq_len]
    # a process per cell: a built KV manager's pool isn't returned to the GPU when it's dropped
    results = {}
    for length in lengths:
        for arm in ("without", "with"):
            t0 = time.time()
            cmd = [sys.executable, str(Path(__file__).resolve()), "--config", a.config, "--one", f"{length},{arm}"]
            if a.model_path:
                cmd += ["--model-path", a.model_path]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode:
                sys.stderr.write(proc.stderr)
                raise SystemExit(f"cell {length} {arm} exited {proc.returncode}")
            (line,) = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
            r = json.loads(line[len("RESULT "):])
            r["seconds"] = round(time.time() - t0, 1)
            results[f"{length} {arm}"] = r
            print(length, arm, r, flush=True)
            Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
