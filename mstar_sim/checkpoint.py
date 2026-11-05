"""A shapes-only copy of a checkpoint, for simulating a model without its weights.

``synthesize(repo_or_dir, out_dir)`` reads a checkpoint's safetensors *headers*
(an HF repo id, fetched by ranges without the large weights' data, or a local
directory) and writes each file as its header, verbatim, over a sparse
(unallocated) data region holding only the bytes of its small tensors (under
``SMALL_WEIGHT`` elements: norms, scales, a rotary table's inputs), next to
copies of the configs, tokenizer and other small files. Weights in other formats
are copied shapes-only too: a ``torch.load`` file saved again with its large
tensors value-less, an ONNX model over ``MAX_ASSET_BYTES`` byte for byte over a hole
where its large initializers' data was. Every model's own loader
reads it unchanged; the simulator reads the small tensors for real and the rest
from the headers only (``mstar_sim.patches.header_only_weights``), so a model of
any size costs a few MB of disk.

CLI::

    python -m mstar_sim.checkpoint Qwen/Qwen3.5-27B /path/out
    python -m mstar_sim.checkpoint Qwen/Qwen3.5-27B --hf-home ~/sim-hf   # as the HF cache
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import math
import mmap
import os
import shutil
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

MANIFEST = "mstar_sim_checkpoint.json"
VERSION = 4
# tensors this small are read for real (a norm, an fp8 scale, a LoRA alpha, a test's
# fixture): cheap, and host code reads their values; only larger weights go value-less
SMALL_WEIGHT = 1 << 16
# small tensors' byte ranges closer than this are fetched from the Hub as one
_MERGE_GAP = 1 << 20
# any other file this small is copied whole, an ONNX model the host may run (Smart Turn) included
MAX_ASSET_BYTES = 50 * 2**20
# weight formats that duplicate the safetensors; never copied next to them
_DUPLICATE_WEIGHTS = ("*.bin", "*.h5", "*.msgpack", "*.gguf", "*.pt", "*.pth")
# a weight-format file this small is an asset (Chatterbox's conds.pt), not a weight copy
SMALL_WEIGHT_FILE = 8 * 2**20
_WEIGHT_FILES = (".pt", ".pth", ".bin", ".onnx")


def _hidden(rel: str) -> bool:
    return any(c.startswith(".") for c in rel.split("/"))


class _LocalSource:
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.revision = None
        self.files = {
            p.relative_to(self.root).as_posix(): p.stat().st_size for p in sorted(self.root.rglob("*"))
            if p.is_file() and not _hidden(p.relative_to(self.root).as_posix())
        }

    def fetch(self, rel: str) -> Path:
        return self.root / rel

    def ranges(self, rel: str, spans: list[tuple[int, int]]) -> list[bytes]:
        out = []
        with open(self.root / rel, "rb") as f:
            for start, stop in spans:
                f.seek(start)
                out.append(f.read(stop - start))
        return out


class _HubSource:
    def __init__(self, repo: str):
        from huggingface_hub import HfApi

        self.api = HfApi()
        self.name = repo
        info = self.api.model_info(repo, files_metadata=True)
        self.revision = info.sha
        self.files = {s.rfilename: s.size or 0 for s in info.siblings if not _hidden(s.rfilename)}

    def fetch(self, rel: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(self.name, rel, revision=self.revision))

    def ranges(self, rel: str, spans: list[tuple[int, int]]) -> list[bytes]:
        """HTTP range reads of the file, over a thread pool."""
        from huggingface_hub import get_session, hf_hub_url
        from huggingface_hub.utils import build_hf_headers, hf_raise_for_status

        url, headers = hf_hub_url(self.name, rel, revision=self.revision), build_hf_headers()

        def read(span):
            reply = get_session().get(url, headers={**headers, "Range": f"bytes={span[0]}-{span[1] - 1}"})
            hf_raise_for_status(reply)
            if reply.status_code != 206 or len(reply.content) != span[1] - span[0]:
                raise ValueError(f"{url}: bytes {span[0]}-{span[1] - 1} came back as {reply.status_code} with "
                                 f"{len(reply.content)} bytes: does the endpoint (HF_ENDPOINT) honour Range?")
            return reply.content

        with ThreadPoolExecutor(16) as pool:
            return list(pool.map(read, spans))


def _write_over_hole(path: Path, size: int, pieces) -> None:
    """A file of ``size`` bytes, unallocated but for ``pieces`` [(offset, bytes)]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for offset, data in pieces:
            f.seek(offset)
            f.write(data)
        f.truncate(size)


def _write_sparse(src, rel: str, path: Path) -> int:
    """``rel`` as its header, verbatim (every tensor at its own offset), over a hole holding
    only its small tensors' bytes; returns its parameter count."""
    (n,) = struct.unpack("<Q", src.ranges(rel, [(0, 8)])[0])
    blob = src.ranges(rel, [(0, 8 + n)])[0]
    header = json.loads(blob[8:])
    header.pop("__metadata__", None)
    small = sorted(tuple(v["data_offsets"]) for v in header.values()  # an empty one has no bytes to keep
                   if 0 < math.prod(v["shape"]) < SMALL_WEIGHT)
    spans: list[list[int]] = []  # the small tensors' ranges, merged where they lie close
    for start, stop in small:
        if spans and start - spans[-1][1] < _MERGE_GAP:
            spans[-1][1] = max(spans[-1][1], stop)
        else:
            spans.append([start, stop])
    base = 8 + n
    chunks = src.ranges(rel, [(base + start, base + stop) for start, stop in spans])

    def pieces():
        yield 0, blob
        span = 0
        for start, stop in small:  # only the small tensors: a large one between stays a hole
            while stop > spans[span][1]:
                span += 1
            yield base + start, chunks[span][start - spans[span][0]:stop - spans[span][0]]

    _write_over_hole(path, base + max((v["data_offsets"][1] for v in header.values()), default=0), pieces())
    return sum(math.prod(v["shape"]) for v in header.values())


def _weight_copy(rel: str, files) -> bool:
    """A weight format beside the safetensors (``pytorch_model.bin`` next to
    ``model.safetensors``), not an asset (a codec's own ``model.pt``)."""
    name, where = os.path.basename(rel), os.path.dirname(rel)
    return (name.startswith(("pytorch_model", "diffusion_pytorch_model", "model.", "consolidated")) or any(
        fnmatch.fnmatch(name, pat) for pat in ("*.h5", "*.msgpack", "*.gguf", "*.bin.index.json"))) and any(
        os.path.dirname(f) == where and f.endswith(".safetensors") for f in files)


def _shapes_only_torch_file(path: Path, dest: Path) -> int | None:
    """A ``torch.load``-able asset (a vocoder's ``hift.pt``, Kokoro's weights) saved again with
    every tensor but the small ones value-less: a few KB, and ``torch.load`` gives back the same
    structure. Returns its parameter count; None if it isn't a plain tensor file."""
    import torch

    try:
        try:
            obj = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        except RuntimeError:  # a legacy (non-zip) file can't be mapped: read it
            obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - not a plain tensor file: keep nothing, say so
        return None
    from torch.utils._pytree import tree_flatten, tree_map

    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tree_map(lambda x: (x.clone() if x.numel() < SMALL_WEIGHT else torch.empty_like(x, device="meta"))
                        if isinstance(x, torch.Tensor) else x, obj), dest)
    return sum(x.numel() for x in tree_flatten(obj)[0] if isinstance(x, torch.Tensor))


def _varint(buf, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7


def _fields(buf, start: int, end: int):
    """A protobuf message's ``(field, wire type, value)``; a length-delimited value is its
    ``(start, end)`` in ``buf``."""
    pos = start
    while pos < end:
        key, pos = _varint(buf, pos)
        wire = key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire in (1, 5):
            value, pos = None, pos + (8 if wire == 1 else 4)
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value, pos = (pos, pos + length), pos + length
        else:
            raise ValueError(f"protobuf wire type {wire}")
        yield key >> 3, wire, value


def _shapes_only_onnx(path: Path, dest: Path) -> int | None:
    """An ONNX model (CosyVoice3's speech tokenizer, whose weights mstar reads into torch
    modules) copied byte for byte over a hole where each large initializer's data was: its
    graph and small tensors as they are. Returns its parameter count; None if not ONNX."""
    holes, params = [], 0
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        try:
            graph = next(v for n, w, v in _fields(buf, 0, len(buf)) if n == 7 and w == 2)
            for n, w, span in _fields(buf, *graph):
                if n != 5 or w != 2:  # not an initializer (a TensorProto)
                    continue
                dims, data = [], None
                for field, wire, value in _fields(buf, *span):
                    if field == 1 and wire == 0:
                        dims.append(value)
                    elif field == 1:  # packed
                        pos, end = value
                        while pos < end:
                            dim, pos = _varint(buf, pos)
                            dims.append(dim)
                    elif field in (4, 9) and wire == 2:  # float_data or raw_data
                        data = value
                params += math.prod(dims)
                if data is not None and math.prod(dims) >= SMALL_WEIGHT:
                    holes.append(data)
        except (StopIteration, ValueError, IndexError):
            return None
        kept = zip([0, *(stop for _, stop in holes)], [*(start for start, _ in holes), len(buf)], strict=True)
        _write_over_hole(dest, len(buf), ((start, buf[start:stop]) for start, stop in kept))
    return params


def _build(src, out: Path) -> dict:
    params, notes, kept = 0, [], False  # kept: a weight file, though a small one counts no parameters
    for rel, size in src.files.items():
        if rel.endswith(".safetensors"):
            params += _write_sparse(src, rel, out / rel)
            kept = True
        elif (size > (MAX_ASSET_BYTES if rel.endswith(".onnx") else SMALL_WEIGHT_FILE)
              and rel.endswith(_WEIGHT_FILES) and not _weight_copy(rel, src.files)):
            shapes_only = _shapes_only_onnx if rel.endswith(".onnx") else _shapes_only_torch_file
            count = shapes_only(src.fetch(rel), out / rel)
            if count is None:
                notes.append(f"skipped {rel}: not a plain tensor file or ONNX model")
            elif rel.endswith(".onnx"):
                notes.append(f"{rel}: its large weights read as zeros, so a runtime would run it on them")
            params += count or 0
            kept = kept or count is not None
        elif size > MAX_ASSET_BYTES or (size > SMALL_WEIGHT_FILE and any(
                fnmatch.fnmatch(os.path.basename(rel), pat) for pat in _DUPLICATE_WEIGHTS)):
            notes.append(f"skipped {rel} ({size / 2**20:.0f} MB)")
        else:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src.fetch(rel), out / rel)
            kept = kept or rel.endswith(_WEIGHT_FILES)
    if not kept:
        raise ValueError(" ".join(["the source has no weights a copy can keep (safetensors, torch or ONNX files).",
                                   *notes]))
    return {"param_count": params, "notes": notes, "revision": src.revision}


def synthesize(repo_or_dir: str, out_dir: str | os.PathLike) -> dict:
    """Write a shapes-only copy of ``repo_or_dir`` to ``out_dir`` and return its
    manifest; an existing copy of the same source is reused."""
    local = Path(repo_or_dir).is_dir()
    source = str(Path(repo_or_dir).resolve()) if local else repo_or_dir
    out = Path(out_dir)
    try:
        cached = json.loads((out / MANIFEST).read_text())
    except (OSError, ValueError):
        cached = None
    if cached and cached.get("version") == VERSION and cached.get("source") == source:
        return cached
    if out.exists() and cached is None and any(out.iterdir()):
        raise ValueError(f"{out} exists and is not a mstar_sim checkpoint; refusing to overwrite it")
    src = _LocalSource(repo_or_dir) if local else _HubSource(repo_or_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{out.name}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        manifest = {"version": VERSION, "source": source, **_build(src, tmp)}
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    for n in manifest["notes"]:
        log.info("%s", n)
    return manifest


def into_hf_cache(repo: str, hf_home: str | os.PathLike) -> dict:
    """Write a shapes-only copy of Hub repo ``repo`` into the Hugging Face cache
    at ``hf_home`` (``hub/models--org--name/snapshots/<rev>``, ``refs/main``), so
    code that loads it by name (``from_pretrained``, ``snapshot_download``,
    ``hf_hub_download``) finds it with ``HF_HOME=<hf_home> HF_HUB_OFFLINE=1``."""
    from huggingface_hub import HfApi

    revision = HfApi().model_info(repo).sha
    repo_dir = Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}"
    manifest = synthesize(repo, repo_dir / "snapshots" / revision)
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(revision)
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m mstar_sim.checkpoint", description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="HF repo id or local checkpoint directory")
    ap.add_argument("out_dir", nargs="?", help="where the copy goes (or --hf-home)")
    ap.add_argument("--hf-home", help="write into this Hugging Face cache instead, for code that loads "
                                      "the model by name (then run with HF_HOME=<dir> HF_HUB_OFFLINE=1)")
    a = ap.parse_args(argv)
    if bool(a.out_dir) == bool(a.hf_home):
        ap.error("give either out_dir or --hf-home")
    if a.hf_home and Path(a.source).is_dir():
        ap.error("--hf-home takes a Hub repo id (the cache is keyed by its revision), not a local directory")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = into_hf_cache(a.source, a.hf_home) if a.hf_home else synthesize(a.source, a.out_dir)
    json.dump(manifest, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
