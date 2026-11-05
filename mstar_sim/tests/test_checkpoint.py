"""``mstar_sim.checkpoint.synthesize``: a shapes-only copy a loader reads unchanged."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mstar_sim.checkpoint import main, synthesize
from mstar_sim.tests.test_device import run_sim


def test_copy_keeps_every_tensor_header_and_only_small_tensors_data(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text('{"num_hidden_layers": 2}')
    (src / "tokenizer.json").write_text('{"fake": true}')
    with open(src / "pytorch_model.bin", "wb") as f:  # a duplicate of the weights: skipped
        f.truncate(16 * 2**20)
    (src / "conds.pt").write_bytes(b"a small asset with no safetensors twin")
    # large torch assets are kept shapes-only: a component's own model.pt (no safetensors beside
    # it, so not a copy of the weights), and one saved in torch's legacy (non-zip) format
    (src / "codec").mkdir()
    torch.save({"w": torch.zeros(3 << 20), "scale": torch.tensor([0.5])}, src / "codec" / "model.pt")
    torch.save({"w": torch.zeros(3 << 20)}, src / "hift.pt", _use_new_zipfile_serialization=False)
    shards = {
        "model-00001-of-00002.safetensors": {"embed.weight": torch.zeros(4096, 1024, dtype=torch.bfloat16)},
        "model-00002-of-00002.safetensors": {"layers.0.q.weight": torch.zeros(1024, 1024, dtype=torch.float8_e4m3fn),
                                             "layers.0.norm.weight": torch.arange(1024.0)},
    }
    for name, tensors in shards.items():
        save_file(tensors, str(src / name))
    index = {"weight_map": {k: f for f, ts in shards.items() for k in ts}}
    (src / "model.safetensors.index.json").write_text(json.dumps(index))

    out = tmp_path / "out"
    manifest = synthesize(str(src), out)
    assert manifest["param_count"] == 4096 * 1024 + 1024 * 1024 + 1024
    for name, tensors in shards.items():
        with safe_open(str(out / name), framework="pt") as f:
            assert {k: (f.get_slice(k).get_shape(), f.get_slice(k).get_dtype()) for k in f.keys()} == {
                k: (list(t.shape), {torch.bfloat16: "BF16", torch.float8_e4m3fn: "F8_E4M3",
                                    torch.float32: "F32"}[t.dtype]) for k, t in tensors.items()}
        assert (out / name).stat().st_blocks * 512 < 64 << 10  # sparse: no large tensor's data on disk
    with safe_open(str(out / "model-00002-of-00002.safetensors"), framework="pt") as f:  # a small one's is kept
        assert torch.equal(f.get_tensor("layers.0.norm.weight"), torch.arange(1024.0))
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        assert (out / name).read_text() == (src / name).read_text()
    assert not (out / "pytorch_model.bin").exists()
    assert (out / "conds.pt").read_bytes() == (src / "conds.pt").read_bytes()
    for asset in ("codec/model.pt", "hift.pt"):
        assert torch.load(out / asset, weights_only=True)["w"].shape == (3 << 20,), asset
    assert torch.load(out / "codec/model.pt", weights_only=True)["scale"].tolist() == [0.5]

    stamp = (out / "config.json").stat().st_mtime_ns
    synthesize(str(src), out)  # the same source: reused
    assert (out / "config.json").stat().st_mtime_ns == stamp


def test_the_emulated_loader_reads_what_the_copy_kept(tmp_path):
    """mstar's loader under the emulator reads a small tensor's kept bytes, and a shard of a
    large one (which the copy left a hole) as meta, not as the hole's zeros; it refuses a copy
    made before small tensors kept their bytes."""
    src = tmp_path / "src"
    src.mkdir()
    save_file({"scale": torch.arange(float(1 << 17)).view(128, 1024), "norm": torch.arange(1024.0)},
              str(src / "model.safetensors"))
    synthesize(str(src), tmp_path / "out")
    save_file({"norm": torch.zeros(1024)}, str(tmp_path / "old.safetensors"), metadata={"mstar_sim": "shapes_only"})
    out = run_sim(f"""
        import json, os, struct, torch
        from mstar.model.loader import iterators

        def read(path, key, spec=None):
            fd = os.open(path, os.O_RDONLY)
            (n,) = struct.unpack("<Q", os.pread(fd, 8, 0))
            return iterators._read_tensor(fd, iterators._plan_read(json.loads(os.pread(fd, n, 8))[key], 8 + n, spec))

        copy = "{tmp_path / "out" / "model.safetensors"}"
        assert torch.equal(read(copy, "norm"), torch.arange(1024.0))
        assert read(copy, "scale", iterators.TensorSlice(0, 0, 32)).is_meta  # a shard: small, of a large tensor
        try:
            read("{tmp_path / "old.safetensors"}", "norm")
            raise SystemExit("an old copy's zeros were read as values")
        except ValueError:
            pass
        print("ok")
    """)
    assert out.strip() == "ok"


def test_hf_cache_copy_refuses_a_local_directory(tmp_path):
    """The cache is keyed by a Hub revision, which a local directory doesn't have."""
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--hf-home", str(tmp_path / "hf")])
