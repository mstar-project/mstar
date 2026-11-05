"""``mstar_sim.checkpoint.synthesize``: a shapes-only copy a loader reads unchanged."""

import json
import types

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mstar_sim.checkpoint import SMALL_WEIGHT_FILE, main, synthesize
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
    assert manifest["param_count"] == 4096 * 1024 + 1024 * 1024 + 1024 + 2 * (3 << 20) + 1  # torch assets too
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


def _protobuf(*fields) -> bytes:
    """A protobuf message of ``(number, value)`` fields: an int is a varint, bytes are
    length-delimited."""
    def varint(n):
        out = b""
        while True:
            out += bytes([n & 0x7F | (0x80 if n > 0x7F else 0)])
            n >>= 7
            if not n:
                return out
    return b"".join(varint(n << 3) + varint(v) if isinstance(v, int) else varint(n << 3 | 2) + varint(len(v)) + v
                    for n, v in fields)


def test_a_checkpoint_of_torch_and_onnx_weights_copies_shapes_only(tmp_path):
    """Weights with no safetensors (Kokoro's ``.pth``, CosyVoice3's speech tokenizer ``.onnx``,
    which mstar reads into torch modules) are copied shapes-only: the ONNX file byte for byte
    over a hole where its large initializer's data was, which mstar's reader reads unchanged."""
    from mstar.model.cosyvoice3.components.onnx_weights import read_onnx

    src = tmp_path / "src"
    src.mkdir()
    torch.save({"w": torch.zeros(3 << 20), "bias": torch.ones(4)}, src / "kokoro.pth")
    big = (4096, 4096)  # float32: 64 MiB of data, over the asset size the copy skips

    def initializer(name, dims, raw):
        return _protobuf(*[(1, d) for d in dims], (2, 1), (8, name.encode()), (9, raw))

    graph = _protobuf((1, _protobuf((1, b"x"), (1, b"w"), (2, b"y"), (4, b"MatMul"))),
                      (5, initializer("w", big, bytes(4 * 4096 * 4096))),
                      (5, initializer("scale", (2,), np.array([0.5, 2.0], dtype=np.float32).tobytes())))
    (src / "speech_tokenizer.onnx").write_bytes(_protobuf((1, 8), (7, graph)))

    manifest = synthesize(str(src), tmp_path / "out")
    assert manifest["param_count"] == (3 << 20) + 4 + 4096 * 4096 + 2
    assert torch.load(tmp_path / "out" / "kokoro.pth", weights_only=True)["w"].shape == (3 << 20,)
    copy = tmp_path / "out" / "speech_tokenizer.onnx"
    assert copy.stat().st_size == (src / "speech_tokenizer.onnx").stat().st_size
    assert copy.stat().st_blocks * 512 < 1 << 20  # sparse: the large initializer's data is a hole
    onnx = read_onnx(copy)
    assert onnx.tensors["w"].shape == big and onnx.tensors["scale"].tolist() == [0.5, 2.0]
    assert [node.op_type for node in onnx.nodes] == ["MatMul"]
    assert any("speech_tokenizer.onnx" in note for note in manifest["notes"])  # holes read as zeros: said so


def test_an_onnx_model_the_host_could_run_stays_real(tmp_path):
    """An ``.onnx`` under ``MAX_ASSET_BYTES`` (Smart Turn, which onnxruntime runs on the host) is
    copied as it is, as any asset."""
    src = tmp_path / "src"
    src.mkdir()
    payload = bytes(range(256)) * (SMALL_WEIGHT_FILE // 256 + 4096)  # past the small file size
    (src / "smart_turn.onnx").write_bytes(_protobuf((1, 8), (7, _protobuf((5, _protobuf(
        (1, len(payload)), (2, 2), (8, b"w"), (9, payload)))))))
    save_file({"w": torch.zeros(4)}, str(src / "weights.safetensors"))
    manifest = synthesize(str(src), tmp_path / "out")
    assert (tmp_path / "out" / "smart_turn.onnx").read_bytes() == (src / "smart_turn.onnx").read_bytes()
    assert manifest["notes"] == []


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


def test_hub_range_reads_refuse_a_reply_that_ignored_the_range(monkeypatch):
    """A mirror that ignores ``Range`` answers 200 with the whole file, whose first bytes
    would be written as the range's."""
    import huggingface_hub

    from mstar_sim import checkpoint

    reply = types.SimpleNamespace(status_code=200, content=bytes(64), raise_for_status=lambda: None)
    monkeypatch.setattr(huggingface_hub, "get_session", lambda: types.SimpleNamespace(get=lambda *a, **k: reply))
    hub = object.__new__(checkpoint._HubSource)
    hub.name, hub.revision = "org/model", "main"
    with pytest.raises(ValueError):
        hub.ranges("model.safetensors", [(8, 16)])


def test_hf_cache_copy_refuses_a_local_directory(tmp_path):
    """The cache is keyed by a Hub revision, which a local directory doesn't have."""
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--hf-home", str(tmp_path / "hf")])
