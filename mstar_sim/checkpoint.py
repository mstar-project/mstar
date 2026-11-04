"""A shapes-only copy of a checkpoint, for simulating a model without its weights.

``synthesize(repo_or_dir, out_dir)`` reads a checkpoint's safetensors *headers*
(an HF repo id, fetched without any weight data, or a local directory) and
writes each file as its header over a sparse (unallocated) data region, next to
copies of the configs, tokenizer and other small files. Every model's own
loader reads it unchanged; the simulator reads only the headers
(``mstar_sim.patches.header_only_weights``), so a model of any size costs a few
MB of disk.

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
import os
import shutil
import struct
import sys
from pathlib import Path

log = logging.getLogger(__name__)

MANIFEST = "mstar_sim_checkpoint.json"
VERSION = 2
MAX_ASSET_BYTES = 50 * 2**20
# weight formats that duplicate the safetensors; never copied next to them
_DUPLICATE_WEIGHTS = ("*.bin", "*.h5", "*.msgpack", "*.gguf", "*.pt", "*.pth")
# a weight-format file this small is an asset (Chatterbox's conds.pt), not a weight copy
SMALL_WEIGHT_FILE = 8 * 2**20
_ITEMSIZE = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "BF16": 2, "F16": 2,
             "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


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

    def header(self, rel: str) -> dict[str, tuple[str, list[int]]]:
        with open(self.root / rel, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        return {k: (v["dtype"], v["shape"]) for k, v in header.items()}


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

    def header(self, rel: str) -> dict[str, tuple[str, list[int]]]:
        meta = self.api.parse_safetensors_file_metadata(self.name, rel, revision=self.revision)
        return {k: (t.dtype, list(t.shape)) for k, t in meta.tensors.items()}


def _write_sparse(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> int:
    """A safetensors file whose data region is a hole: valid header, no bytes."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = math.prod(shape) * _ITEMSIZE[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    header["__metadata__"] = {"format": "pt", "mstar_sim": "shapes_only"}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.truncate(8 + len(blob) + offset)
    return sum(math.prod(shape) for _, shape in tensors.values())


def _weight_copy(rel: str, files) -> bool:
    """A weight format beside the safetensors (``pytorch_model.bin`` next to
    ``model.safetensors``), not an asset (a codec's own ``model.pt``)."""
    name, where = os.path.basename(rel), os.path.dirname(rel)
    return (name.startswith(("pytorch_model", "diffusion_pytorch_model", "model.", "consolidated")) or any(
        fnmatch.fnmatch(name, pat) for pat in ("*.h5", "*.msgpack", "*.gguf", "*.bin.index.json"))) and any(
        os.path.dirname(f) == where and f.endswith(".safetensors") for f in files)


def _shapes_only_torch_file(path: Path, dest: Path) -> bool:
    """A ``torch.load``-able asset (a vocoder's ``hift.pt``) saved again with every
    tensor value-less: a few KB, and ``torch.load`` gives back the same structure."""
    import torch

    try:
        try:
            obj = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        except RuntimeError:  # a legacy (non-zip) file can't be mapped: read it
            obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - not a plain tensor file: keep nothing, say so
        return False
    from torch.utils._pytree import tree_map

    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tree_map(lambda x: torch.empty_like(x, device="meta") if isinstance(x, torch.Tensor) else x, obj),
               dest)
    return True


def _build(src, out: Path) -> dict:
    params, notes = 0, []
    for rel, size in src.files.items():
        if rel.endswith(".safetensors"):
            params += _write_sparse(out / rel, src.header(rel))
        elif size > SMALL_WEIGHT_FILE and rel.endswith((".pt", ".pth", ".bin")) and not _weight_copy(rel, src.files):
            if not _shapes_only_torch_file(src.fetch(rel), out / rel):
                notes.append(f"skipped {rel}: not a plain tensor file")
        elif size > MAX_ASSET_BYTES or (size > SMALL_WEIGHT_FILE and any(
                fnmatch.fnmatch(os.path.basename(rel), pat) for pat in _DUPLICATE_WEIGHTS)):
            notes.append(f"skipped {rel} ({size / 2**20:.0f} MB)")
        else:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src.fetch(rel), out / rel)
    if not params:
        raise ValueError("the source has no safetensors files")
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
