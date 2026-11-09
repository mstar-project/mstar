"""``mstar.model.kimi_k3.truncate`` on a tiny checkpoint in the release's format (``language_model.``
names, MXFP4-packed experts, A_log padded, a vision tensor, the release's config.json shrunk):
the MXFP4 decode, the cut, and the result loading through ``KimiK31Model.get_submodule`` and
running in the real Engine against the reference. The GPU parity job
(``test/kimi_k3/gpu_truncated_parity.py``) runs the same path on the release's own layers."""
import copy
import json
import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mstar.model.kimi_k3 import truncate
from mstar.model.kimi_k3.config import KimiK3ModelConfig
from mstar.model.kimi_k3.tests import engine_helpers as H
from mstar.model.kimi_k3.tests.reference import random_checkpoint, reference_logits

FIXTURES = Path(__file__).parent / "fixtures"
TINY = dict(latent=64, experts=8)


def quant_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-to-nearest MXFP4 (tests only): a power-of-two scale per 32 along K with the group's
    largest magnitude at most 6, codes packed low nibble first."""
    n, k = w.shape
    g = w.float().view(n, k // 32, 32)
    exp = torch.ceil(torch.log2(g.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 6.0))
    scaled = (g / torch.exp2(exp)).view(n, k)
    mag = (scaled.abs().unsqueeze(-1) - truncate.E2M1[:8]).abs().argmin(-1)
    codes = (mag + 8 * (scaled < 0).long()).to(torch.uint8)
    return codes[:, 0::2] | (codes[:, 1::2] << 4), (exp.view(n, -1) + 127).to(torch.uint8)


def _tiny_hf_config(moe_intermediate: int = 32) -> dict:
    hf = copy.deepcopy(json.loads((FIXTURES / "k31_config.json").read_text()))
    t = hf["text_config"]
    t.update(hidden_size=64, num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=4,
             q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
             intermediate_size=96, num_experts=TINY["experts"], num_experts_per_token=2,
             moe_intermediate_size=moe_intermediate, routed_expert_hidden_size=TINY["latent"], attn_res_block_size=2,
             vocab_size=256, eos_token_id=250, bos_token_id=252, pad_token_id=255)
    t["linear_attn_config"].update(num_heads=4, head_dim=16, kda_layers=[1, 2, 3], full_attn_layers=[4, 5])
    t["dsa_config"].update(dsa_layers=[4, 5], index_num_heads=4, index_head_dim=16, index_topk=64)
    return hf


def make_tiny_release(dst: Path, moe_intermediate: int = 32) -> dict[str, torch.Tensor]:
    """The tiny release in ``dst``; returns its text weights as the reference reads them (names
    without ``language_model.``), the experts as the bf16 their packing stands for."""
    hf = _tiny_hf_config(moe_intermediate)
    cfg = KimiK3ModelConfig.from_hf_config(hf, strict=False)
    sd = random_checkpoint(cfg, seed=5, dtype=torch.bfloat16, a_log_pad=4)
    tensors, unpacked = {"vision_tower.patch_embed.weight": torch.zeros(4, 4)}, {}
    for name, t in sd.items():
        m = re.fullmatch(r"(.*\.experts\.\d+\.w[123])\.weight", name)
        if m:
            packed, scale = quant_mxfp4(t)
            tensors[f"language_model.{m.group(1)}.weight_packed"] = packed
            tensors[f"language_model.{m.group(1)}.weight_scale"] = scale
            unpacked[name] = truncate.dequant_mxfp4(packed, scale)
        else:
            tensors[f"language_model.{name}"] = t.contiguous()
            unpacked[name] = t
    names = sorted(tensors)
    shards = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
              "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    for shard, keys in shards.items():
        save_file({k: tensors[k] for k in keys}, str(dst / shard))
    weight_map = {k: shard for shard, keys in shards.items() for k in keys}
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (dst / "config.json").write_text(json.dumps(hf))
    return unpacked


def test_mxfp4_decode_is_the_e2m1_grid_times_the_scale():
    codes = torch.arange(16, dtype=torch.uint8).repeat(4).view(2, 32)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scale = torch.tensor([[127], [129]], dtype=torch.uint8)  # 1 and 4
    out = truncate.dequant_mxfp4(packed, scale)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), truncate.E2M1[codes.long()] * torch.tensor([[1.0], [4.0]]))
    w = torch.randn(8, 64)
    deq = truncate.dequant_mxfp4(*quant_mxfp4(w)).float()
    # the top of the grid is 2 scale units apart and the scale is at most amax / 3
    assert (deq - w).abs().max() <= w.abs().max() / 3 + 1e-6


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    H.install_cpu_stubs(monkeypatch)
    torch.set_num_threads(2)


def test_truncated_release_loads_and_matches_the_reference(tmp_path):
    src, dst = tmp_path / "release", tmp_path / "trunc"
    src.mkdir()
    unpacked = make_tiny_release(src)
    truncate.main(["--src", str(src), "--dst", str(dst), "--layers", "4", "--experts", "4"])

    hf = json.loads((dst / "config.json").read_text())
    t = hf["text_config"]
    assert t["num_hidden_layers"] == 4 and t["num_experts"] == 4 and "quantization_config" not in t
    assert t["linear_attn_config"]["full_attn_layers"] == [4] and t["dsa_config"]["dsa_layers"] == [4]
    names = set(json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"])
    assert not any("vision_tower" in n or ".layers.4." in n or ".experts.4." in n for n in names)

    from mstar.model.kimi_k3.kimi_k3_model import KimiK31Model

    model = KimiK31Model(str(dst), strict_config=False, kda_max_requests=4, tokenizer_mode="byte",
                         kda_conv_dtype=torch.float32)
    cfg = model.config
    assert (cfg.num_hidden_layers, cfg.num_experts, cfg.full_attn_layers) == (4, 4, (3,))
    sub = model.get_submodule("LLM", device="cpu", autocast_dtype=torch.float32)
    lm = sub.language_model
    # the decoded experts, exactly
    w1 = unpacked["model.layers.1.block_sparse_moe.experts.2.w1.weight"].float()
    torch.testing.assert_close(lm.model.layers[1].block_sparse_moe.experts.gate_up_proj[2, :32], w1,
                               atol=0, rtol=0)

    from mstar.communication.tensors import LocalTransferEngine
    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.engine.engine import Engine
    from mstar.engine.resources.kv.transfer import TransferEngineInfo

    engine = Engine(None)
    engine.load_model({"LLM": sub}, model.get_node_resources(),
                      parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
                      device=torch.device("cpu"),
                      transfer_engine_info=TransferEngineInfo(
                          my_entity_id="e", my_session_id="s", transfer_engine=LocalTransferEngine("localhost")),
                      kv_cache_type=torch.float32)
    greedy = H.GreedySampler()
    engine._resources["sampler"] = engine._submodules["LLM"].resources["sampler"] = greedy
    sub.node_resources["sampler"] = greedy
    engine.warmup()
    prompt = torch.tensor([11, 40, 7, 99, 3, 56])
    H.add_request(engine, "a")
    H.run_step(engine, "prefill", {"a": prompt})

    from safetensors import safe_open

    sd = {}
    for shard in sorted(set(json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"].values())):
        with safe_open(str(dst / shard), framework="pt") as f:
            sd.update({k[len("language_model."):]: f.get_tensor(k).float() for k in f.keys()})
    ref = reference_logits(sd, prompt, cfg)
    torch.testing.assert_close(greedy.logits[-1][-1], ref[-1], atol=5e-5, rtol=0)


def test_keep_mxfp4_cuts_without_decoding(tmp_path):
    """``--keep-mxfp4``: the kept experts stay packed with the release's quantization_config, and
    load as stored through the quantization scope."""
    src, dst = tmp_path / "release", tmp_path / "trunc"
    src.mkdir()
    make_tiny_release(src)
    truncate.main(["--src", str(src), "--dst", str(dst), "--layers", "4", "--experts", "4", "--keep-mxfp4"])
    t = json.loads((dst / "config.json").read_text())["text_config"]
    assert t["num_experts"] == 4 and t["quantization_config"]["format"] == "mxfp4-pack-quantized"
    names = set(json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"])
    assert "language_model.model.layers.1.block_sparse_moe.experts.3.w1.weight_packed" in names
    assert not any(".experts.4." in n or n.endswith("experts.0.w1.weight") for n in names)

    from mstar.model.kimi_k3.kimi_k3_model import KimiK31Model
    from mstar.quantization.scope import load_submodule

    model = KimiK31Model(str(dst), strict_config=False, tokenizer_mode="byte", kda_conv_dtype=torch.float32)
    sub = load_submodule(model, "LLM", {}, "cpu", autocast_dtype=torch.float32)
    block = sub.language_model.model.layers[1].block_sparse_moe
    assert block.experts_packed.name == "mxfp4" and block.experts.gate_up_proj_packed.shape[0] == 4
