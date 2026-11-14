"""DeepSeek-V4.1's Engram tables off the device: what a decode step pays to read its rows. Not a
CI test: it needs a GPU, and ``--dir`` on a disk with room for sparse files.

    python test/deepseek_v41/gpu_engram.py --mode prepare --dir <scratch dir>
    python test/deepseek_v41/gpu_engram.py --mode roundtrip --dir <scratch dir> --out roundtrip.json
    python test/deepseek_v41/gpu_engram.py --mode uva --dir <scratch dir> --out uva.json

``prepare`` (no GPU): two sparse files with the layout of the checkpoint's Engram shards (202.8 GB
apparent), a pool of their rows read into the page cache, and a written file for ``uva``.
``roundtrip``: the host table resource as served with ``engram_tables: host``: per decode step
of 1 to 64 rows (48 table rows a token), the ids copied to the host, the rows gathered there and
copied to the device, timed until ``fetch`` returns and until the rows have landed; next to the
same rows gathered from tables on the device.
``uva``: the alternative with no trip through the host: the tables' memory registered with CUDA,
so a kernel on the device reads host rows by address. The same gathers, their bytes checked
against the host's, eagerly and replayed from a captured graph; over anonymous memory and over a
read-only mapping of a file, which is how the resource maps the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch

from mstar.model.deepseek_v41.components.engram import host_tables
from mstar.model.deepseek_v41.config import DeepseekV41Config

# (engram layer, shard file, bytes before the shard's data) of the checkpoint's two tables
SHARDS = ((1, "model-00047-of-00048.safetensors", 664), (14, "model-00048-of-00048.safetensors", 672))
SIZES = (1, 8, 32, 64)
POOL = 200_000
UVA_FILE = "uva-rows.bin"
REGISTER_READ_ONLY = 0x08  # cudaHostRegisterReadOnly


def make_checkpoint(directory: str, config: DeepseekV41Config) -> None:
    """The two Engram shards as sparse safetensors files of the checkpoint's layout."""
    index = {}
    for (layer, name, base), rows in zip(SHARDS, config.engram_num_embeddings, strict=True):
        stem = f"layers.{layer}.engram.embed"
        split = rows * config.engram_head_dim
        end = split + rows * (config.engram_head_dim // config.fp8_block)
        header = json.dumps({
            f"{stem}.weight": {"dtype": "F8_E4M3", "shape": [rows, config.engram_head_dim], "data_offsets": [0, split]},
            f"{stem}.scale": {"dtype": "F8_E8M0", "shape": [rows, config.engram_head_dim // config.fp8_block],
                              "data_offsets": [split, end]},
        }).encode().ljust(base - 8)
        path = os.path.join(directory, name)
        with open(path, "r+b" if os.path.exists(path) else "w+b") as f:
            f.write(len(header).to_bytes(8, "little") + header)
            f.truncate(base + end)
        index[f"{stem}.weight"] = index[f"{stem}.scale"] = name
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": index}, f)


def _manager(directory: str, config: DeepseekV41Config, device):
    from mstar.engine.resources.base import EngineResourceInfo, build_resource
    from mstar.engine.resources.host_table.config import HostTableConfig, HostTableSpec

    # sparse files: a prefault would fill 203 GB of holes with zero pages
    return build_resource(
        HostTableSpec(resource_key="engram", nodes={"LLM"}, config=HostTableConfig(
            checkpoint=directory, tables=host_tables(config), prefault=False)),
        EngineResourceInfo(device=device))


def _pools(config: DeepseekV41Config) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {name: torch.randint(0, table.rows, (POOL,), generator=g) for name, table in host_tables(config).items()}


def _stats(laps: list[float]) -> dict:
    laps = sorted(laps)
    return {"ms": round(statistics.median(laps), 4), "p10": round(laps[len(laps) // 10], 4),
            "p90": round(laps[-1 - len(laps) // 10], 4)}


def mode_prepare(args) -> dict:
    from mstar.engine.resources.host_table.config import HostTableStep
    from mstar.engine.resources.step import StepContext

    config = DeepseekV41Config()
    make_checkpoint(args.dir, config)
    manager = _manager(args.dir, config, torch.device("cpu"))
    t0 = time.perf_counter()
    manager.plan(HostTableStep(), StepContext(request_ids=(), graph_walk="decode", slot=0, capture=False))
    manager.fetch(_pools(config))
    pool_s = time.perf_counter() - t0
    path = Path(args.dir) / UVA_FILE
    nbytes = args.uva_gib << 30
    if not path.exists() or path.stat().st_size != nbytes:
        block = np.random.default_rng(0).integers(0, 256, size=64 << 20, dtype=np.uint8).tobytes()
        with open(path, "wb") as f:
            for _ in range(nbytes // len(block)):
                f.write(block)
    return {"prepare": {"pool_rows_a_table": POOL, "pool_read_s": round(pool_s, 2), "uva_file_gib": args.uva_gib}}


def mode_roundtrip(args) -> dict:
    from mstar.engine.resources.host_table.config import HostTableStep
    from mstar.engine.resources.step import StepContext

    device = torch.device(args.device)
    config = DeepseekV41Config()
    tables = host_tables(config)
    manager = _manager(args.dir, config, device)
    step = HostTableStep()
    ctx = [StepContext(request_ids=(), graph_walk="decode", slot=slot, capture=False) for slot in (0, 1)]
    pools = {name: pool.to(device) for name, pool in _pools(config).items()}
    manager.plan(step, ctx[0])
    manager.fetch(pools)  # the pool's pages mapped in this process
    manager.commit(step, ctx[0])
    cols = config.engram_hash_cols
    g = torch.Generator(device=device).manual_seed(1)
    row_bytes = sum(part.width for table in tables.values() for part in table.parts)
    # tables on the device, as the default placement holds a rank's share: the pool's rows, by position
    on_device = {name: [torch.randint(0, 256, (POOL, part.width), dtype=torch.uint8, device=device, generator=g)
                        for part in table.parts] for name, table in tables.items()}
    rows: dict[str, dict] = {}
    for tokens in SIZES:
        for threads in (1, 2, 4):
            torch.set_num_threads(threads)
            returned, landed = [], []
            for trial in range(args.trials + 5):
                picks = {name: torch.randperm(POOL, device=device, generator=g)[:tokens * cols] for name in pools}
                ids = {name: pools[name][pick].view(tokens, cols) for name, pick in picks.items()}
                torch.cuda.synchronize(device)
                manager.plan(step, ctx[trial % 2])
                t0 = time.perf_counter()
                manager.fetch(ids)
                t1 = time.perf_counter()
                torch.cuda.synchronize(device)
                t2 = time.perf_counter()
                manager.commit(step, ctx[trial % 2])
                if trial >= 5:
                    returned.append((t1 - t0) * 1e3)
                    landed.append((t2 - t0) * 1e3)
            rows[f"host/rows{tokens}/threads{threads}"] = {
                "fetch_returns": _stats(returned), "rows_on_device": _stats(landed),
                "table_rows": tokens * cols * len(tables), "bytes": tokens * cols * row_bytes}
            print(f"host/rows{tokens}/threads{threads}: " + json.dumps(rows[f"host/rows{tokens}/threads{threads}"]),
                  flush=True)
        laps = []
        for trial in range(args.trials + 5):
            picks = {name: torch.randperm(POOL, device=device, generator=g)[:tokens * cols] for name in pools}
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            got = [part[picks[name]] for name, parts in on_device.items() for part in parts]
            torch.cuda.synchronize(device)
            if trial >= 5:
                laps.append((time.perf_counter() - t0) * 1e3)
        del got
        rows[f"device/rows{tokens}"] = {"gather": _stats(laps)}
        print(f"device/rows{tokens}: " + json.dumps(rows[f"device/rows{tokens}"]), flush=True)
    return rows


class _Registered:
    """Host memory registered with CUDA, as a device array: what ``torch.as_tensor`` takes for a
    tensor on the device whose storage is the host's pages."""

    def __init__(self, pointer: int, shape: tuple[int, ...]):
        self.__cuda_array_interface__ = {"shape": shape, "typestr": "|u1", "data": (pointer, False), "version": 2}


def _uva_case(name: str, owner: np.ndarray, flags: int, width: int, device, args) -> dict:
    cudart = torch.cuda.cudart()
    pointer, nbytes = owner.ctypes.data, owner.nbytes
    t0 = time.perf_counter()
    result = cudart.cudaHostRegister(pointer, nbytes, flags)
    if getattr(result, "value", result) != 0:
        return {"error": f"cudaHostRegister(flags={flags}) failed: {result}"}
    row: dict = {"register_s": round(time.perf_counter() - t0, 3), "gib": round(nbytes / 2**30, 2), "flags": flags}
    try:
        table_rows = nbytes // width
        view = torch.as_tensor(_Registered(pointer, (table_rows, width)), device=device)
        host = torch.from_numpy(owner[:table_rows * width].reshape(table_rows, width))
        g = torch.Generator(device=device).manual_seed(2)
        for tokens in SIZES:
            n = tokens * 48
            laps, equal = [], True
            for trial in range(args.trials + 5):
                ids = torch.randint(0, table_rows, (n,), device=device, generator=g)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                got = view.index_select(0, ids)
                torch.cuda.synchronize(device)
                if trial >= 5:
                    laps.append((time.perf_counter() - t0) * 1e3)
                if trial % 10 == 0:
                    equal = equal and bool(torch.equal(got.cpu(), host[ids.cpu()]))
            row[f"rows{tokens}"] = dict(_stats(laps), table_rows=n, bytes=n * width, bytes_equal_host=equal)
        # the same gather recorded once and replayed over new ids
        ids = torch.randint(0, table_rows, (64 * 48,), device=device, generator=g)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = view.index_select(0, ids)
        laps, equal = [], True
        for _ in range(args.trials):
            ids.copy_(torch.randint(0, table_rows, (64 * 48,), device=device, generator=g))
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize(device)
            laps.append((time.perf_counter() - t0) * 1e3)
            equal = equal and bool(torch.equal(out.cpu(), host[ids.cpu()]))
        row["replayed/rows64"] = dict(_stats(laps), bytes_equal_host=equal)
        del view, out, graph
    except Exception as error:  # noqa: BLE001 - a registration CUDA will not read through is a result
        row["error"] = f"{type(error).__name__}: {str(error)[-300:]}"
    finally:
        cudart.cudaHostUnregister(pointer)
    print(f"{name}: " + json.dumps(row), flush=True)
    return row


def mode_uva(args) -> dict:
    device = torch.device(args.device)
    torch.zeros(1, device=device)  # the context the registration belongs to
    width = 264  # an Engram row: 256 values and their 8 scales
    rows: dict[str, dict] = {}
    nbytes = args.uva_gib << 30
    # anonymous memory, filled before it is registered
    anon = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    owner = np.frombuffer(anon, dtype=np.uint8)
    block = np.random.default_rng(1).integers(0, 256, size=64 << 20, dtype=np.uint8)
    for start in range(0, nbytes, block.size):
        owner[start:start + block.size] = block
    rows["anonymous"] = _uva_case("anonymous", owner, 0, width, device, args)
    del owner
    anon.close()
    # a read-only mapping of a file in the page cache, as the host table resource maps a checkpoint
    with open(Path(args.dir) / UVA_FILE, "rb") as f:
        mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    mapped.madvise(mmap.MADV_POPULATE_READ) if hasattr(mmap, "MADV_POPULATE_READ") else None
    owner = np.frombuffer(mapped, dtype=np.uint8)
    for flags, name in ((REGISTER_READ_ONLY, "file-read-only/read-only-flag"), (0, "file-read-only/default-flag")):
        rows[name] = _uva_case(name, owner, flags, width, device, args)
        if "error" not in rows[name]:
            break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("prepare", "roundtrip", "uva"), required=True)
    ap.add_argument("--dir", required=True, help="scratch directory for the sparse tables")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--uva-gib", type=int, default=4)
    ap.add_argument("--out")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    rows = {"prepare": mode_prepare, "roundtrip": mode_roundtrip, "uva": mode_uva}[args.mode](args)
    cuda = args.mode != "prepare"
    meta = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device if cuda else "cpu",
            "gpu": torch.cuda.get_device_name(torch.device(args.device)) if cuda else None,
            "load_average": os.getloadavg()[0]}
    if args.out:
        Path(args.out).write_text(json.dumps({"mode": args.mode, "meta": meta, "rows": rows}, indent=1))
    else:
        print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
