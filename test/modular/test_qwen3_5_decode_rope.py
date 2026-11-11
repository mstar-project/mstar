"""A captured Qwen3.5 decode step, and captured text prefill or mixed steps with ``ingraph_decode_rope``
on, build their rotary tables as plain RoPE over the 1D positions; eager steps and media walks keep the
woven build.

A text row's three MRoPE grids are one position, so its tables are the three-axis tables
(`compute_3d_cos_sin`) without the weave. Inductor fuses the weave into one kernel, which Triton 3.7.1
miscompiles under some tuned launch configs (measured on that kernel alone). This module checks the plain
tables; it doesn't reproduce the miscompile.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from mstar.model.qwen3_5.components.rope import compute_3d_cos_sin, compute_inv_freq, compute_text_cos_sin


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rotary_dim, theta, section", [(64, 1e7, [11, 11, 10]), (32, 1e4, [6, 5, 5])])
def test_text_tables_are_the_three_axis_tables_bit_for_bit(dtype, rotary_dim, theta, section):
    inv = compute_inv_freq(rotary_dim, theta)
    # decode positions, including past an image's position jump (a row's offset absorbs it)
    pos = torch.cat([torch.arange(0, 300), torch.tensor([4321, 4322, 70000, 70001])])
    want = compute_3d_cos_sin(pos.unsqueeze(0).expand(3, -1), inv, section, target_dtype=dtype)
    got = compute_text_cos_sin(pos, inv, target_dtype=dtype)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def _cuda_reason() -> str | None:
    if not torch.cuda.is_available():
        return "needs CUDA"
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return "needs flashinfer"
    return None


NODE = "LLM"
_STOP = 96
_LINEAR, _FULL = "linear_attention", "full_attention"
# the smallest text stack FlashInfer's and the GDN kernels take: head dims of 128, bf16 weights on an fp32 state
_TEXT = dict(
    hidden_size=256, num_hidden_layers=4, layer_types=[_LINEAR, _LINEAR, _LINEAR, _FULL],
    num_attention_heads=2, num_key_value_heads=1, head_dim=128, intermediate_size=512,
    linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=128, linear_value_head_dim=128,
    linear_conv_kernel_dim=4, vocab_size=512, rms_norm_eps=1e-6, max_position_embeddings=512,
    attn_output_gate=True, tie_word_embeddings=False, eos_token_id=_STOP, mamba_ssm_dtype="float32", dtype="bfloat16",
    rope_parameters={"rope_theta": 10_000.0, "partial_rotary_factor": 0.25, "mrope_section": [6, 5, 5],
                     "mrope_interleaved": True},
)


class _Tokenizer:
    """What a text-only model reads of a tokenizer outside ``process_prompt``."""
    eos_token_id = _STOP
    unk_token_id = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        pass

    def remove_request(self, request_id):
        pass

    def cleanup(self):
        pass


def _captured_engine(monkeypatch, root):
    """A tiny random Qwen3.5 on the real engine and device kernels, bf16 under autocast, its graphs captured."""
    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.engine.engine import Engine
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.kv.config import KVSpec
    from mstar.engine.state.adapters import node_resources
    from mstar.model.qwen3_5 import qwen3_5_model
    from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    monkeypatch.setattr(qwen3_5_model, "AutoTokenizer", _Tokenizer)
    monkeypatch.setattr(LLMSubmodule, "DECODE_CAPTURE_BATCH_SIZES", [8])
    monkeypatch.setattr(LLMSubmodule, "PREFILL_CAPTURE_BATCH_SIZES", [1, 8])
    (root / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
        "vision_start_token_id": 90, "vision_end_token_id": 91, "text_config": _TEXT,
    }))
    monkeypatch.setattr(LLMSubmodule, "PREFILL_TOKEN_BUCKETS", [64, 256])
    monkeypatch.setattr(LLMSubmodule, "MIXED_TOKEN_BUCKETS", [64, 256])
    model = qwen3_5_model.Qwen3_5DenseModel(model_path_hf=str(root))

    lm = model._build(lambda: Qwen3_5ForCausalLM(model.config, None), torch.float32, "cpu")
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in lm.named_parameters():
            if name.endswith("norm.weight"):
                p.fill_(1.0)
            else:
                p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float64) / max(p.shape[-1], 1) ** 0.5)
            if name.endswith(("A_log", "dt_bias")):
                p.mul_(0.3)
    lm = lm.eval().to(device="cuda", dtype=torch.bfloat16)
    sub = LLMSubmodule(lm, model.config, model.vision_config, model.encoders)

    specs = node_resources(model)
    for spec in specs:
        if isinstance(spec, KVSpec):
            spec.config.page_size = 16
            spec.config.max_num_pages = 512
    eng = Engine(graph_runtime=None, autocast_dtype=torch.bfloat16)
    eng.load_model(
        {NODE: sub}, specs, WorkerParallelGroups(num_workers=1, global_rank=0),
        torch.device("cuda", torch.cuda.current_device()),
        transfer_engine_info=None, kv_cache_type=torch.bfloat16, model=model,
    )
    eng.warmup()
    return eng, model


class _Driver:
    """Plays the worker, synchronously: a text prompt step, then decode steps, each built after the
    last was post-processed."""

    def __init__(self, eng):
        self.eng = eng
        self.out: dict[int, list[int]] = {}
        self.max: dict[int, int] = {}
        self.sampling: dict = {}

    def add(self, rid: int, prompt: list[int], max_tokens: int) -> None:
        from mstar.engine.resources.kv.config import KVReqConfig
        from mstar.engine.resources.sampler.config import SamplingReqConfig
        from mstar.model.base import LabelLayout, Span

        kv = KVReqConfig()
        kv.apply_conductor_config(
            kv_layout={"main": LabelLayout(
                spans=(Span("ids", "prefill_text", len(prompt), len(prompt), "text_inputs"),),
                growth=(("decode_loop", 1),),
            ).row()},
            loop_caps={"decode_loop": max_tokens},
        )
        sampling = SamplingReqConfig(temperature=0.0, top_k=0, top_p=1.0, ignore_eos=True)
        sampling.apply_conductor_config(seed=rid)
        self.sampling[rid] = sampling
        self.eng.add_request(rid, {"kv_cache": kv, "sampler": sampling})
        sampler = self.eng._resources["sampler"]
        if sampler.ask(rid, NODE, "prefill_text") is not None:
            sampler.reserve(rid, NODE, "prefill_text", None)
        self.out[rid], self.max[rid] = [], max_tokens

    def _run(self, walk: str, ids: dict[int, torch.Tensor], metadata: dict) -> None:
        from mstar.engine.engine import ExecutingBatch
        from mstar.engine.resources.step import StepContext

        rids = list(ids)
        batch = ExecutingBatch(
            node_name=NODE,
            per_request_info={rid: SimpleNamespace(
                rid=rid, request_id=rid, rid_handle=rid, graph_walk=walk, max_tokens=self.max[rid],
                step_metadata=dict(metadata), resource_configs={"sampler": self.sampling[rid]},
                update_publish_info=lambda published: None, dynamic_loop_iter_counts={},
            ) for rid in rids},
            step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
            per_request_input_tensors={rid: {"text_inputs": [row]} for rid, row in ids.items()},
        )
        self.eng.prepare_inputs(batch)
        out = self.eng.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        tokens = out.check_stop_buffers["new_token"].reshape(-1).tolist()
        for i, rid in enumerate(rids):
            self.out[rid].append(int(tokens[i]))

    def prefill(self, prompts: dict[int, list[int]]) -> None:
        device = self.eng._device
        self._run("prefill_text", {rid: torch.tensor(p, device=device) for rid, p in prompts.items()},
                  {"is_prefill": True, "last_prefill": True})

    def step(self, rids: list[int]) -> None:
        device = self.eng._device
        self._run("decode", {rid: torch.tensor([self.out[rid][-1]], device=device) for rid in rids}, {})


@pytest.mark.skipif(_cuda_reason() is not None, reason=str(_cuda_reason()))
def test_captured_decode_keys_rotate_every_grid_as_a_prefill_of_the_same_tokens(tmp_path, monkeypatch):
    """Six rows decoded with the bs=8 capture: each reply key's rotated dims are what a fresh prefill of
    the same history writes, H dims included (an H grid of zeros moves them by a turn per position)."""
    eng, model = _captured_engine(monkeypatch, tmp_path)
    driver = _Driver(eng)
    gen = torch.Generator().manual_seed(0)
    prompts = {rid: torch.randint(1, 400, (20,), generator=gen).tolist() for rid in range(1, 7)}
    for rid, prompt in prompts.items():
        driver.add(rid, prompt, 40)
    driver.prefill(prompts)
    for _ in range(12):
        driver.step(list(prompts))
    kv = eng._resources["kv_cache"]
    tensor = kv.kv_cache.tensor
    rotary = model.config.rotary_dim

    def keys(rid: int, n: int) -> torch.Tensor:
        pages = [int(x) for x in kv._streams[rid]["main"].page_indices]
        return tensor[0, pages, 0].float().reshape(-1, *tensor.shape[4:])[:n, :, :rotary]

    for rid, prompt in prompts.items():
        n = int(kv._streams[rid]["main"].stored_len)
        history = (prompt + driver.out[rid])[:n]
        driver.add(100 + rid, history, 4)
        driver.prefill({100 + rid: history})
        decoded, fresh = keys(rid, n)[20:], keys(100 + rid, n)[20:]
        rel = float((decoded - fresh).norm() / fresh.norm())
        assert rel < 0.05, f"request {rid}: its decoded keys' rotated dims are {rel:.3f} off a fresh prefill's"
