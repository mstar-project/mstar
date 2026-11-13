"""Test support for DeepSeek-V4.1: the checkpoint's own ``inference/model.py`` as a CPU oracle.

The reference runs its numerics through tilelang CUDA kernels (``inference/kernel.py``).
``reference_kernels`` is that module's interface in torch, written from the kernels' source, so
``load_reference`` can import ``model.py`` unchanged on a machine with no GPU. ``write_snapshot``
writes random weights in the checkpoint's own names, dtypes and packing, which both the reference
(through its ``convert.py``) and M*'s loader read.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

from mstar.model.deepseek_v41.config import DeepseekV41Config
from mstar.utils.mx_round import pack_e2m1, pow2_ceil, quant_fp4, quant_fp8, round_e2m1, unpack_e2m1

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
FP4X2 = torch.float4_e2m1fn_x2


# -- inference/kernel.py in torch ------------------------------------------------------------
# The rounding itself is ``mstar.utils.mx_round``, the same functions the model runs, so a
# misreading of the kernels' rounding shows only against the real kernels on a GPU.

def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    q, scale = quant_fp8(x, block_size, pow2_scale=scale_fmt is not None)
    if inplace:
        x.copy_((q.float().unflatten(-1, (-1, block_size)) * scale.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return q, scale.to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=E8M0):
    assert scale_dtype in (E8M0, FP8)
    q, scale = quant_fp4(x, block_size, e4m3_scale=scale_dtype == FP8)
    if inplace:
        x.copy_((q.unflatten(-1, (-1, block_size)) * scale.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return pack_e2m1(q).view(FP4X2), scale.to(scale_dtype)


def _block_gemm(a, a_s, b, b_s, act_block: int, weight_block: int) -> torch.Tensor:
    """``sum_k (A_k @ B_k^T) * a_s[:, k] * b_s[:, k]`` over ``weight_block``-wide K blocks, in
    fp32; ``b_s`` is already one row per output row."""
    k = a.size(-1)
    m, n, blocks = a.numel() // k, b.size(0), k // weight_block
    c = torch.einsum("mkb,nkb->mnk", a.reshape(m, blocks, weight_block), b.reshape(n, blocks, weight_block))
    a_s = a_s.float().reshape(m, -1).repeat_interleave(act_block // weight_block, dim=1)
    c = (c * a_s[:, None, :] * b_s[None, :, :]).sum(-1)
    return c.reshape(*a.shape[:-1], n).to(torch.get_default_dtype())


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    del scale_dtype
    rows = b_s.float().repeat_interleave(block_size, dim=0)[: b.size(0)]
    return _block_gemm(a.float(), a_s, b.float(), rows, block_size, block_size)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    del scale_dtype
    return _block_gemm(a.float(), a_s, unpack_e2m1(b), b_s.float(), act_block_size, 32)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """The kernel's online softmax over 64-entry blocks of each row's index list: scores and
    sums in fp32, each block's probabilities rounded to the inputs' dtype before they weight
    the values, and the sink added to the denominator only."""
    b, m, h, d = q.shape
    acc = torch.zeros(b, m, h, d, dtype=torch.float32, device=q.device)
    total = torch.zeros(b, m, h, dtype=torch.float32, device=q.device)
    peak = torch.full((b, m, h), -1e30, dtype=torch.float32, device=q.device)
    batch = torch.arange(b, device=q.device)[:, None, None]
    for start in range(0, topk_idxs.size(-1), 64):
        idx = topk_idxs[:, :, start:start + 64]
        valid = idx != -1
        rows = kv[batch, idx.clamp_min(0).long()].float() * valid[..., None]
        scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), rows)
        scores = torch.where(valid[:, :, None, :], scores * softmax_scale, -torch.inf)
        last, peak = peak, torch.maximum(peak, scores.amax(-1))
        decay = torch.exp(last - peak)
        prob = torch.exp(scores - peak[..., None])
        total = total * decay + prob.sum(-1)
        acc = acc * decay[..., None] + torch.einsum("bmhk,bmkd->bmhd", prob.to(q.dtype).float(), rows)
    total = total + torch.exp(attn_sink.float() - peak)
    return (acc / total[..., None]).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def reference_kernels() -> types.ModuleType:
    module = types.ModuleType("kernel")
    for fn in (act_quant, fp4_act_quant, fp4_gemm, fp8_gemm, hc_split_sinkhorn, sparse_attn):
        setattr(module, fn.__name__, fn)
    return module


# -- the reference itself --------------------------------------------------------------------

_REFERENCE_MODULES = ("engram", "image_processor", "vision", "model", "convert")


def load_reference(snapshot_dir: str | Path) -> types.SimpleNamespace:
    """The checkpoint snapshot's ``inference/`` modules (``model``, ``engram``, ``convert``...),
    with ``kernel`` replaced by ``reference_kernels``. They import each other by bare names
    (``kernel``, ``vision``...) that other packages in one test process may own, so those names
    are put back once the imports are done."""
    inference = Path(snapshot_dir) / "inference"
    names = ("kernel", *_REFERENCE_MODULES)
    saved = {name: sys.modules.get(name) for name in names}
    loaded: dict[str, types.ModuleType] = {"kernel": reference_kernels()}
    try:
        sys.modules["kernel"] = loaded["kernel"]
        for name in _REFERENCE_MODULES:
            spec = importlib.util.spec_from_file_location(name, inference / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = loaded[name] = module
            spec.loader.exec_module(module)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return types.SimpleNamespace(**loaded)


# -- a checkpoint snapshot with random weights -------------------------------------------------

def synthetic_tokenizer(vocab_size: int):
    """A word-level tokenizer of ``vocab_size`` tokens, some of which normalize alike (case,
    accents, surrounding spaces), so the Engram compressed vocabulary is smaller than it."""
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    words = ["<bos>", "<eos>", "<pad>"]
    stems = [f"w{i}" for i in range(vocab_size)]
    for i in range(vocab_size - len(words)):
        stem = stems[i // 4]
        words.append((stem, stem.upper(), " " + stem, stem + "\u0301")[i % 4] if i % 7 else f"u{i}")
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<pad>"))
    tokenizer.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>")


def to_hf_config(config: DeepseekV41Config) -> dict:
    """``config`` as the checkpoint's config.json spells it (``from_hf_config``'s inverse)."""
    text = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    for knob in ("max_seq_len", "max_output_tokens", "moe_fp8_resident", "fp8_block", "bos_token_id",
                 "eos_token_ids", "pad_token_id"):
        text.pop(knob)
    text["rope_scaling"] = {
        "rope_type": "yarn", "factor": text.pop("rope_factor"), "beta_fast": text.pop("rope_beta_fast"),
        "beta_slow": text.pop("rope_beta_slow"),
        "original_max_position_embeddings": text.pop("rope_original_max_position_embeddings"),
    }
    return {
        "model_type": "deepseek_v41", "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_ids[0], "pad_token_id": config.pad_token_id,
        "quantization_config": {
            "quant_method": "fp8", "activation_scheme": "dynamic", "scale_fmt": "ue8m0",
            "weight_block_size": [config.fp8_block, config.fp8_block], "expert_dtype": "fp4"},
        "text_config": text,
    }


def _fp8_pair(w: torch.Tensor, rows: int, cols: int) -> dict[str, torch.Tensor]:
    """``w`` as E4M3 with a power-of-two scale per ``rows`` x ``cols`` block."""
    out, inp = w.shape
    blocks = w.view(out // rows, rows, inp // cols, cols)
    scale = pow2_ceil(blocks.abs().amax(dim=(1, 3)).clamp_min(1e-4) / 448.0)
    q = (blocks / scale[:, None, :, None]).reshape(out, inp).to(FP8)
    return {"weight": q, "scale": scale.to(E8M0)}


def _fp4_pair(w: torch.Tensor, block: int) -> dict[str, torch.Tensor]:
    """``w`` as packed E2M1 (int8, as the checkpoint stores it) with a power-of-two scale per
    row and ``block`` inputs."""
    groups = w.unflatten(-1, (-1, block))
    scale = pow2_ceil(groups.abs().amax(-1).clamp_min(6 * 2.0 ** -20) / 6.0)
    q = round_e2m1((groups / scale.unsqueeze(-1)).clamp(-6.0, 6.0)).flatten(-2)
    return {"weight": pack_e2m1(q).view(torch.int8), "scale": scale.to(E8M0)}


def random_checkpoint(config: DeepseekV41Config, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random weights under the checkpoint's own names, dtypes and packing (text backbone)."""
    g = torch.Generator().manual_seed(seed)
    c, bf16, b = config, torch.bfloat16, config.fp8_block
    hidden, width, inter, hc = c.hidden_size, c.head_dim, c.moe_intermediate_size, c.hc_mult

    def randn(*shape, std=1.0):
        return torch.randn(*shape, generator=g) * std

    def linear(out, inp, gain=1.0):
        return randn(out, inp, std=gain * inp ** -0.5)

    def norm(dim):
        return (1 + randn(dim, std=0.1)).to(bf16)

    out: dict[str, torch.Tensor] = {}

    def fp8(name, w, rows=b, cols=b):
        for leaf, t in _fp8_pair(w, rows, cols).items():
            out[f"{name}.{leaf}"] = t

    def fp4(name, w):
        for leaf, t in _fp4_pair(w, b).items():
            out[f"{name}.{leaf}"] = t

    out["embed.weight"] = randn(c.vocab_size, hidden).to(bf16)
    out["head.weight"] = linear(c.vocab_size, hidden, 4.0).to(bf16)
    out["norm.weight"] = norm(hidden)
    mix = (2 + hc) * hc
    for i in range(c.num_hidden_layers):
        p, heads = f"layers.{i}", c.num_attention_heads
        for site in ("attn", "ffn"):
            out[f"{p}.hc_{site}_fn"] = randn(mix, hc * hidden, std=0.5 * (hc * hidden) ** -0.5)
            out[f"{p}.hc_{site}_base"] = randn(mix, std=0.5)
            out[f"{p}.hc_{site}_scale"] = 1 + randn(3, std=0.2)
            out[f"{p}.{site}_norm.weight"] = norm(hidden)
        a = f"{p}.attn"
        out[f"{a}.attn_sink"] = randn(heads)
        fp8(f"{a}.wq_a", linear(c.q_lora_rank, hidden))
        fp8(f"{a}.wq_b", linear(heads * width, c.q_lora_rank, 2.0))
        fp8(f"{a}.wkv", linear(width, hidden))
        fp8(f"{a}.wo_a", linear(c.o_groups * c.o_lora_rank, heads * width // c.o_groups))
        fp8(f"{a}.wo_b", linear(hidden, c.o_groups * c.o_lora_rank))
        out[f"{a}.q_norm.weight"], out[f"{a}.kv_norm.weight"] = norm(c.q_lora_rank), norm(width)
        if i in c.kv_source_layer_ids:
            out[f"{a}.compressor.wkv.weight"] = linear(width, hidden).to(bf16)
            out[f"{a}.compressor.norm.weight"] = norm(width)
            if c.compress_ratios[i] > 1:
                out[f"{a}.compressor.wgate.weight"] = linear(width, hidden, 3.0).to(bf16)
            out[f"{a}.indexer.wk.weight"] = linear(c.index_head_dim, width).to(bf16)
            out[f"{a}.indexer.k_norm.weight"] = norm(c.index_head_dim)
        if i in c.index_source_layer_ids:
            fp8(f"{a}.indexer.wq_b", linear(c.index_n_heads * c.index_head_dim, c.q_lora_rank))
            out[f"{a}.indexer.weights_proj.weight"] = linear(c.index_n_heads, hidden).to(bf16)
        f = f"{p}.ffn"
        out[f"{f}.gate.weight"] = linear(c.n_routed_experts, hidden, 2.0).to(bf16)
        out[f"{f}.gate.bias"] = randn(c.n_routed_experts, std=0.1)
        for e in [*range(c.n_routed_experts), None]:
            quant, name = (fp4, f"{f}.experts.{e}") if e is not None else (fp8, f"{f}.shared_experts")
            quant(f"{name}.w1", linear(inter, hidden, 2.0))
            quant(f"{name}.w3", linear(inter, hidden, 2.0))
            quant(f"{name}.w2", linear(hidden, inter))
        if i in c.engram_layer_ids:
            e, rows = f"{p}.engram", c.engram_num_embeddings[c.engram_layer_ids.index(i)]
            for leaf, t in _fp8_pair(randn(rows, c.engram_head_dim), 1, b).items():
                out[f"{e}.embed.{leaf}"] = t
            fp8(f"{e}.wkv", linear(hidden * (hc + 1), c.engram_hash_cols * c.engram_head_dim))
            out[f"{e}.q_weight"] = (1 + randn(hc, hidden, std=0.1)).to(bf16)
            out[f"{e}.k_weight"] = (1 + randn(hc, hidden, std=0.1)).to(bf16)
    return out


def write_snapshot(path: str | Path, config: DeepseekV41Config, seed: int = 0):
    """A checkpoint directory M* and the reference both load: config.json, a tokenizer, and
    ``random_checkpoint`` as one safetensors file. Returns the config with the tokenizer's
    compressed vocabulary size, and the tokenizer."""
    from safetensors.torch import save_file

    from mstar.model.deepseek_v41.components.engram import compressed_token_map

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tokenizer = synthetic_tokenizer(config.vocab_size)
    config = dataclasses.replace(
        config, engram_compressed_vocab_size=compressed_token_map(tokenizer)[1])
    tokenizer.save_pretrained(str(path))
    (path / "config.json").write_text(json.dumps(to_hf_config(config)))
    save_file(random_checkpoint(config, seed), str(path / "model.safetensors"))
    return config, tokenizer


def reference_model(ref: types.SimpleNamespace, snapshot: str | Path, config: DeepseekV41Config,
                    tokenizer, workdir: str | Path, max_batch_size: int = 1):
    """The reference ``Transformer`` on ``snapshot``'s weights, loaded through its own
    ``convert.py``. It is built in torch's current default dtype, as the reference runs
    (bf16 as shipped). Under another default, its bf16-declared weights are widened to it."""
    from safetensors.torch import load_model

    c = config
    args = ref.model.ModelArgs(
        max_batch_size=max_batch_size, max_seq_len=c.max_seq_len, temperature=0.0,
        vocab_size=c.vocab_size, dim=c.hidden_size, moe_inter_dim=c.moe_intermediate_size,
        n_layers=c.num_hidden_layers, n_mtp_layers=c.num_nextn_predict_layers,
        n_heads=c.num_attention_heads, n_routed_experts=c.n_routed_experts,
        n_shared_experts=c.n_shared_experts, n_activated_experts=c.num_experts_per_tok,
        score_func=c.scoring_func, norm_topk_prob=c.norm_topk_prob, route_scale=c.routed_scaling_factor,
        swiglu_limit=c.swiglu_limit, q_lora_rank=c.q_lora_rank, head_dim=c.head_dim,
        rope_head_dim=c.qk_rope_head_dim, norm_eps=c.rms_norm_eps, o_groups=c.o_groups,
        o_lora_rank=c.o_lora_rank, window_size=c.sliding_window, compress_ratios=c.compress_ratios,
        kv_source_layers=c.kv_source_layer_ids, index_source_layers=c.index_source_layer_ids,
        compress_rope_theta=c.compress_rope_theta,
        original_seq_len=c.rope_original_max_position_embeddings, rope_theta=c.rope_theta,
        rope_factor=c.rope_factor, beta_fast=c.rope_beta_fast, beta_slow=c.rope_beta_slow,
        index_n_heads=c.index_n_heads, index_head_dim=c.index_head_dim, index_topk=c.index_topk,
        candidate_source_layer=c.candidate_source_layer_id,
        candidate_topk_blocks=c.candidate_topk_blocks, candidate_block_size=c.candidate_block_size,
        hc_mult=c.hc_mult, hc_sinkhorn_iters=c.hc_sinkhorn_iters, hc_eps=c.hc_eps,
        engram_layer_ids=c.engram_layer_ids, engram_num_embeddings=c.engram_num_embeddings,
        engram_max_ngram_size=c.engram_max_ngram_size, engram_vocab_size=c.engram_vocab_size,
        engram_n_heads=c.engram_n_heads, engram_head_dim=c.engram_head_dim,
        engram_pad_id=c.engram_pad_token_id,
        engram_compressed_vocab_size=c.engram_compressed_vocab_size,
    )
    threads = torch.get_num_threads()
    try:
        ref.convert.main(str(snapshot), str(workdir), 1, "fp4")  # sets torch's thread count
    finally:
        torch.set_num_threads(threads)
    model = ref.model.Transformer(args, tokenizer)
    load_model(model, str(Path(workdir) / "model0-mp1.safetensors"))
    default = torch.get_default_dtype()
    for param in model.parameters():
        if param.dtype == torch.bfloat16 and default != torch.bfloat16:
            param.data = param.data.to(default)
    return model.eval()


# -- the model in a CPU engine ----------------------------------------------------------------

class RecordingSampler:
    """Argmax in place of the kernel sampler, keeping the step's logits. Built on
    ``StubResource`` by ``cpu_engine``."""

    logits: torch.Tensor | None = None

    def sample(self, request_ids, logits, **kwargs):
        self.logits = logits.detach().float().clone()
        return logits.argmax(-1)


class StubTransfer:
    """No KV transfer between workers: the engine here is one process."""

    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del transfer_info, kwargs
        return False

    def remove_request(self, request_id):
        del request_id


def patch_cpu_engine(monkeypatch) -> None:
    """What a test's engine on CPU needs patched: no KV transfer, and no capture on a GPU box."""
    from mstar.engine.resources.kv import manager as kv_manager

    monkeypatch.setattr(kv_manager, "KVTransferManager", StubTransfer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def cpu_engine(path: str | Path, dtype: torch.dtype, **model_kwargs):
    """M*'s model on the snapshot ``path``, loaded as a server loads it, in a CPU ``Engine``
    whose sampler is a ``RecordingSampler``. Returns ``(engine, sampler, model)``."""
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.engine.engine import Engine
    from mstar.engine.resources.kv.transfer import TransferEngineInfo
    from mstar.model.deepseek_v41.config import SAMPLER
    from mstar.model.deepseek_v41.deepseek_v41_model import DeepseekV41Model
    from mstar.testing_utils.resource import StubResource

    model_kwargs = {"max_seq_len": 256, "max_requests": 4, "moe_fp8_resident": False, **model_kwargs}
    model = DeepseekV41Model(str(path), **model_kwargs)
    sub = model.get_submodule("LLM", device="cpu", autocast_dtype=dtype)
    engine = Engine(None, autocast_dtype=None)
    engine.load_model(
        {"LLM": sub}, model.get_node_resources(),
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=torch.device("cpu"),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="e", my_session_id="s", transfer_engine=LocalTransferEngine("localhost")),
        kv_cache_type=dtype,
    )
    sampler = type("Sampler", (RecordingSampler, StubResource), {})()
    engine._resources[SAMPLER] = sampler
    engine._submodules["LLM"].resources[SAMPLER] = sampler
    sub.node_resources[SAMPLER] = sampler
    engine.warmup()
    return engine, sampler, model


def add_request(engine, rid: str) -> None:
    from mstar.engine.resources import SamplingReqConfig
    from mstar.model.deepseek_v41.config import SAMPLER

    engine.add_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0)})


def run_step(engine, sampler, walk: str, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """One step of ``walk`` over ``inputs`` (token ids per request); each request's logits."""
    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.engine.engine import ExecutingBatch
    from mstar.engine.resources import SamplingReqConfig
    from mstar.engine.resources.step import StepContext
    from mstar.model.deepseek_v41.config import SAMPLER
    from mstar.model.submodule_base import InputMetadata

    rids = list(inputs)
    info = {
        rid: CurrentForwardPassInfo(
            request_id=f"wire-{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0, random_seed=0,
            max_tokens=1 << 20, resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0)})
        for rid in rids
    }
    batch = ExecutingBatch(
        node_name="LLM", per_request_info=info,
        per_request_input_tensors={rid: {"text_inputs": [ids]} for rid, ids in inputs.items()},
        per_request_input_metadata={rid: InputMetadata(dynamic_loop_iter_counts={}) for rid in rids},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
    )
    engine.prepare_inputs(batch)
    assert not batch.failed_requests, batch.failed_requests
    engine.exec_and_postprocess(batch)
    assert batch.admit_error is None, batch.admit_error
    engine.finalize_batch(batch)
    return {rid: sampler.logits[i] for i, rid in enumerate(rids)}


def run_request(engine, sampler, ids: torch.Tensor, prompt: int, rid: str = "r") -> torch.Tensor:
    """Logits after a ``prompt``-token prefill and after each later id fed as one decode step."""
    add_request(engine, rid)
    out = [run_step(engine, sampler, "prefill", {rid: ids[:prompt]})[rid]]
    for token in ids[prompt:]:
        out.append(run_step(engine, sampler, "decode", {rid: token.view(1)})[rid])
    engine.remove_request(rid)
    return torch.stack(out)


def random_ids(config: DeepseekV41Config, n: int, seed: int) -> torch.Tensor:
    return torch.randint(3, config.vocab_size, (n,), generator=torch.Generator().manual_seed(seed))


def logit_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    """The largest logit difference, in units of the logits' own spread."""
    return ((a - b).abs().max() / b.std()).item()
