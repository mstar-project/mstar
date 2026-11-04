"""``mstar_sim.checkpoint.synthesize``: a shapes-only copy a loader reads unchanged."""

import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mstar_sim.checkpoint import synthesize


def test_copy_keeps_every_tensor_header_but_no_data(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text('{"num_hidden_layers": 2}')
    (src / "tokenizer.json").write_text('{"fake": true}')
    with open(src / "pytorch_model.bin", "wb") as f:  # a duplicate of the weights: skipped
        f.truncate(16 * 2**20)
    (src / "conds.pt").write_bytes(b"a small asset with no safetensors twin")
    shards = {
        "model-00001-of-00002.safetensors": {"embed.weight": torch.zeros(4096, 1024, dtype=torch.bfloat16)},
        "model-00002-of-00002.safetensors": {"layers.0.q.weight": torch.zeros(1024, 1024, dtype=torch.float8_e4m3fn),
                                             "layers.0.norm.weight": torch.zeros(1024)},
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
        assert (out / name).stat().st_blocks * 512 < 64 << 10  # sparse: no data on disk
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        assert (out / name).read_text() == (src / name).read_text()
    assert not (out / "pytorch_model.bin").exists()
    assert (out / "conds.pt").read_bytes() == (src / "conds.pt").read_bytes()

    stamp = (out / "config.json").stat().st_mtime_ns
    synthesize(str(src), out)  # the same source: reused
    assert (out / "config.json").stat().st_mtime_ns == stamp
