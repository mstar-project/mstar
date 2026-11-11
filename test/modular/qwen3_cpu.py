"""A tiny random-weight Qwen3 target, with or without a tiny random DSpark
drafter, on the real `Engine` and its real KV, position, sampler and tap
resources, on the CPU.

Main has no CPU kernels for three things a forward needs, so the harness
stands in for them, and nothing else:

- ``CPUPagedAttention``: a reference paged attention over the KV pool's pages
  (causal, or DSpark's non-causal block with its last rows hidden), declared
  in place of each ``AttentionSpec``;
- a CPU kernel for the RoPE op (``mstar::rope_apply_qk_inplace``), rotate-half
  as HF's Qwen3;
- the prefill's first token: main's sampler draws on CUDA or XPU only, so a
  CPU draw (argmax at temperature 0, else a seeded draw) replaces
  ``sample_tokens``. Every later token comes from the sampler's ``verify``,
  which runs on the CPU, or, without a drafter, from this draw.

Everything runs in float64, so a verify block and a one-token decode step
agree to far below any gap between logits.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionSpec, AttentionStep
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVSpec
from mstar.engine.resources.sampler import utils as sampler_utils
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.resources.step import StepContext
from mstar.model.submodule_base import HostRows
from mstar.testing_utils.kv import conductor_kv_config

NODE = "LLM"
# the scaffold's text-prompt walk
PREFILL = "prefill_text"
PAGE = 4
VOCAB = 97
STOP = 96
MASK = 95
DTYPE = torch.float64

TARGET = dict(
    architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=32, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=48, vocab_size=VOCAB,
    rms_norm_eps=1e-6, rope_theta=10_000.0, max_position_embeddings=512, tie_word_embeddings=False,
    eos_token_id=STOP, torch_dtype="float32",
)
DSPARK = dict(
    architectures=["Qwen3DSparkModel"], model_type="qwen3", hidden_size=32, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=48, vocab_size=VOCAB,
    rms_norm_eps=1e-6, rope_parameters={"rope_theta": 10_000.0, "rope_type": "default"},
    max_position_embeddings=512, block_size=4, markov_rank=8, mask_token_id=MASK,
    # taps enter target layers 1 and 2 (the outputs of layers 0 and 1)
    target_layer_ids=[0, 1], dtype="float32", use_sliding_window=False,
)


def write_checkpoints(root: Path, target: dict | None = None, dspark: dict | None = None) -> tuple[Path, Path]:
    """Config-only checkpoint directories for the target and the drafter."""
    paths = []
    for name, config in (("target", {**TARGET, **(target or {})}), ("dspark", {**DSPARK, **(dspark or {})})):
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(config))
        paths.append(path)
    (paths[0] / "generation_config.json").write_text(json.dumps({"do_sample": False}))
    return paths[0], paths[1]


def draft_block(dspark: Path, k: int, mode: str = "synchronous") -> dict:
    return {"method": "dspark", "model": str(dspark), "num_draft_tokens": k, "mode": mode}


def set_context(target: Path, max_seq_len: int = 256) -> None:
    """Sets ``target``'s ``max_position_embeddings``. Spawned ranks share one checkpoint, so the
    file is replaced whole, and only when the value changes: a rank reading it while another
    rewrote it in place could read it empty."""
    path = target / "config.json"
    config = json.loads(path.read_text())
    if config.get("max_position_embeddings") == max_seq_len:
        return
    tmp = path.with_name(f"config.json.{os.getpid()}")
    tmp.write_text(json.dumps({**config, "max_position_embeddings": max_seq_len}))
    os.replace(tmp, path)


def make_model(target: Path, draft: dict | None = None, max_seq_len: int = 256, **kwargs):
    """The model over ``target``, whose context (what bounds a request) is ``max_seq_len``."""
    from mstar.model.qwen3.qwen3_model import Qwen3Model

    set_context(target, max_seq_len)
    kwargs.setdefault("accelerator_graph", False)
    return Qwen3Model(model_path_hf=str(target), draft=draft, **kwargs)


# ── stand-ins for the CUDA-only kernels ──


class CPUPagedAttention(AttentionManager):
    """Reference paged attention, read straight off the KV pool's pages."""

    # it holds no request's state, so a rebuild policy (`Engine._check_rebuild_policies`) takes it
    supports_drop = True

    def __init__(self, kv_key: str):
        self._kv_key = kv_key
        self._views: dict[str, list] = {}
        self._step: AttentionStep | None = None

    @classmethod
    def build(cls, spec, info):
        return cls(spec.config.kv_cache)

    def depends_on(self):
        return {self._kv_key}

    def plan(self, step: AttentionStep, ctx: StepContext):
        self.reset_default_cursors()
        self._step = step
        self._views = {label: list(out.views) for label, out in ctx.plan_results[self._kv_key].items()}

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        del k, v, layer_idx
        step = self._step
        out, row = [], 0
        for view in self._views[label or self._default_label]:
            n = view.to_compute
            keys = view.length - step.hide_last
            pos = torch.arange(keys)
            pages = torch.tensor(view.page_idxs)[pos // PAGE]
            kv = kv_cache_layer[pages, :, pos % PAGE]  # [keys, 2, heads, dim]
            kk, vv = kv[:, 0], kv[:, 1]
            rep = q.shape[1] // kk.shape[1]
            kk, vv = kk.repeat_interleave(rep, 1), vv.repeat_interleave(rep, 1)
            qq = q[row: row + n]
            mask = None
            if step.causal:
                at = torch.arange(view.length - n, view.length)
                mask = pos[None, :] <= at[:, None]
            o = F.scaled_dot_product_attention(
                qq.transpose(0, 1), kk.transpose(0, 1), vv.transpose(0, 1), attn_mask=mask,
            )
            out.append(o.transpose(0, 1))
            row += n
        return torch.cat(out)

    def qo_indptr_buf(self, label="main"):
        counts = torch.tensor([v.to_compute for v in self._views[label]])
        return torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])


@dataclasses.dataclass
class CPUAttentionSpec(AttentionSpec):
    @property
    def resource_class(self):
        return CPUPagedAttention


def _rope_cpu(q, k, pos_ids, cos_sin_cache, rotary_dim, interleave, rope_scale, rope_theta,
              low_freq_factor=None, high_freq_factor=None, old_context_len=None):
    assert not interleave and cos_sin_cache is None
    d = rotary_dim or q.shape[-1]
    half = d // 2
    inv = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float64) * 2 / d))
    angle = pos_ids.to(torch.float64)[:, None] * inv[None]
    cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]
    for x in (q, k):
        x1, x2 = x[..., :half].double(), x[..., half:d].double()
        x[..., :d] = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


_ROPE_REGISTERED = False


def register_cpu_rope() -> None:
    global _ROPE_REGISTERED
    if not _ROPE_REGISTERED:
        import mstar.engine.resources.position.rope  # noqa: F401  (defines the op)

        torch.library.register_kernel("mstar::rope_apply_qk_inplace", "cpu", _rope_cpu)
        _ROPE_REGISTERED = True


_device_sample_tokens = sampler_utils.sample_tokens


def _cpu_draw(logits, temperature=0.6, top_k=0, top_p=1.0, repetition_penalty=1.0, seen_token_mask=None,
              any_greedy=None, top_k_zero_count=None, seed=None, rand_offset=None, **kwargs):
    """The prefill's draw on the CPU: argmax at temperature 0, else a draw seeded by
    the row's seed and offset. A device's logits go to the real sampler."""
    if logits.device.type != "cpu":
        return _device_sample_tokens(
            logits, temperature, top_k, top_p, repetition_penalty, seen_token_mask, any_greedy,
            top_k_zero_count, seed, rand_offset, **kwargs,
        )
    n = logits.shape[0]
    temperature = torch.as_tensor(temperature, dtype=torch.float64).expand(n)
    out = logits.argmax(-1)
    for i in range(n):
        if temperature[i] > 0:
            draw_seed = 0 if seed is None else int(seed[i]) * 1_000_003 + int(rand_offset[i])
            g = torch.Generator().manual_seed(draw_seed)
            probs = torch.softmax(logits[i].double() / temperature[i], -1)
            out[i] = torch.multinomial(probs, 1, generator=g)[0]
    return out


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        pass

    def remove_request(self, request_id):
        pass

    def cleanup(self):
        pass


def patch_cpu(monkeypatch) -> None:
    """The three stand-ins, and no KV transfer engine."""
    register_cpu_rope()
    monkeypatch.setattr(sampler_utils, "sample_tokens", _cpu_draw)
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


# ── the engine ──


def cpu_specs(model, max_num_pages: int = 512, prefix_cache: bool = False):
    specs = []
    for spec in model.get_node_resources():
        if isinstance(spec, AttentionSpec):
            spec = CPUAttentionSpec(resource_key=spec.resource_key, nodes=spec.nodes, config=spec.config)
        if isinstance(spec, KVSpec):
            spec.config.page_size = PAGE
            spec.config.max_num_pages = max_num_pages
            if prefix_cache:
                spec.config.prefix_cache_salt = "qwen3"
        specs.append(spec)
    return specs


def init_random(module: torch.nn.Module, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    for name, p in module.named_parameters():
        if name.endswith("norm.weight"):
            p.data.fill_(1.0)
        else:
            p.data.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float64) / max(p.shape[-1], 1) ** 0.5)


def build_submodule(model, seed: int = 0, weights=None, device="cpu", dtype=DTYPE):
    """The node: the target's weights from ``weights`` (HF names), else random from
    ``seed``; the drafter's its own deterministic random ones."""
    lm = model.build_lm(device, dtype=dtype)
    if weights is None:
        init_random(lm, seed)
    else:
        lm.load_weights((name.removeprefix("model."), tensor) for name, tensor in weights)
    lm.eval()
    if model.drafter_decl is not None:
        # no checkpoint weights: the drafter's own deterministic random ones
        model.drafter_decl = dataclasses.replace(model.drafter_decl, checkpoint=None)
    return model.make_submodule(lm, device, dtype=dtype)


def copy_weights(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    """``src``'s parameters into ``dst``'s, by name, in ``dst``'s device and dtype: a
    float64 reference of a bfloat16 model computes on exactly its weights."""
    have = dict(src.named_parameters())
    for name, p in dst.named_parameters():
        p.data.copy_(have[name].detach().to(device=p.device, dtype=p.dtype))


def engine(model, seed: int = 0, prefix_cache: bool = False, specs=None, weights=None,
           slot_masters: bool = True, device: str = "cpu", dtype=DTYPE, warmup: bool = False,
           prepare=None) -> Engine:
    """The node on a real Engine. On the CPU with the stand-ins and no autocast (the
    norms would round through bfloat16, `mstar::rms_norm`); elsewhere with the
    model's own resources, under ``dtype`` autocast, captured when ``warmup``.
    ``prepare(submodule)`` edits the weights before the engine sees them."""
    sub = build_submodule(model, seed, weights, device=device, dtype=dtype)
    if prepare is not None:
        prepare(sub)
    cpu = torch.device(device).type == "cpu"
    eng = Engine(graph_runtime=None, autocast_dtype=None if cpu else dtype)
    if specs is None:
        specs = cpu_specs(model, prefix_cache=prefix_cache) if cpu else device_specs(model, prefix_cache)
    eng.load_model(
        {NODE: sub}, specs, WorkerParallelGroups(num_workers=1, global_rank=0), _indexed(device),
        transfer_engine_info=None, kv_cache_type=dtype, model=model,
    )
    if warmup:
        eng.warmup()
    elif slot_masters:
        # what a capture's buffers would build: the sampler's slot masters, the loop-back slots
        eng._resources["sampler"].build_accelerator_graph_buffers([], 8, 256)
    return eng


def device_specs(model, prefix_cache: bool = False, page_size: int = 16, max_num_pages: int = 512):
    """The model's own resources, with small pages so short prompts cross them."""
    specs = model.get_node_resources()
    for spec in specs:
        if isinstance(spec, KVSpec):
            spec.config.page_size = page_size
            spec.config.max_num_pages = max_num_pages
            if prefix_cache:
                spec.config.prefix_cache_salt = "qwen3"
    return specs


def _info(rid, walk, max_tokens, sampling):
    return SimpleNamespace(
        rid=rid, graph_walk=walk, max_tokens=max_tokens, step_metadata={},
        resource_configs={"sampler": sampling}, update_publish_info=lambda published: None,
    )


class Driver:
    """Plays the worker, synchronously: a prompt step, then decode steps, each
    built after the last was post-processed (and, verifying, cut)."""

    def __init__(self, eng: Engine, model):
        self.eng = eng
        self.model = model
        self.out: dict[int, list[int]] = {}
        self.max: dict[int, int] = {}
        self.sampling: dict[int, SamplingReqConfig] = {}
        self.verifies = model.draft is not None
        self.device = eng._device
        (self._kv_spec,) = [spec for spec in model.get_node_resources() if spec.resource_key == "kv_cache"]

    def add(self, rid: int, prompt: list[int], max_tokens: int, temperature: float = 0.0, seed: int = 0,
            keyed: bool = False) -> None:
        from mstar.engine.resources.kv.keys import chain

        # the layout `process_prompt` gives a text prompt of these ids, as the conductor applies it
        layout = self.model.kv_layout({}, len(prompt))["kv_cache"]["main"]
        whole = len(prompt) // PAGE
        kv = conductor_kv_config(
            self._kv_spec, {"main": layout.row()},
            loop_caps=self.model.get_loop_caps(max_output_tokens=max_tokens),
            **({"prefix_keys": {"main": chain([prompt[at: at + PAGE] for at in range(0, whole * PAGE, PAGE)])},
                "prefix_tail": {"main": prompt[whole * PAGE:]}} if keyed else {}),
        )
        sampling = SamplingReqConfig(temperature=temperature, top_k=0, top_p=1.0)
        sampling.apply_conductor_config(seed=seed)
        self.sampling[rid] = sampling
        self.eng.add_request(rid, {"kv_cache": kv, "sampler": sampling})
        sampler = self.eng._resources["sampler"]
        if sampler.ask(rid, NODE, PREFILL) is not None:
            sampler.reserve(rid, NODE, PREFILL, None)
        self.out[rid], self.max[rid] = [], max_tokens

    def prefill(self, rids: list[int], prompts: dict[int, list[int]], prefix: bool = False) -> None:
        if prefix:
            from mstar.model.submodule_base import ARNodeInputs

            for rid in rids:
                inputs = ARNodeInputs(input_seq_len=len(prompts[rid]))
                hit = self.eng._runner.resolve_cached_prefix(rid, NODE, PREFILL, inputs)
                assert hit is not None and hit.tokens > 0
                self.eng._runner.apply_cached_prefix(rid, NODE, PREFILL, inputs, hit)
                self.prefix_hit = hit.tokens
        batch = self._batch(PREFILL, {rid: torch.tensor(prompts[rid], device=self.device) for rid in rids})
        self.eng.prepare_inputs(batch)
        out = self.eng.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        for rid in rids:
            self.out[rid].append(int(out.per_rid_outputs[rid]["new_token"][0].reshape(-1)[0]))

    def _batch(self, walk, ids):
        rids = list(ids)
        return ExecutingBatch(
            node_name=NODE,
            per_request_info={rid: _info(rid, walk, self.max[rid], self.sampling[rid]) for rid in rids},
            step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
            per_request_input_tensors={rid: {"text_inputs": [row]} for rid, row in ids.items()},
        )

    def step(self, rids: list[int]) -> dict[int, bool]:
        """One decode step over ``rids``; which rows stopped."""
        batch = self._batch("decode", {rid: torch.tensor([self.out[rid][-1]], device=self.device) for rid in rids})
        self.eng.prepare_inputs(batch)
        outputs = self.eng.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        stopped = {}
        if self.verifies:
            host = HostRows(tuple(outputs.row_request_ids or rids), {
                name: t.cpu() for name, t in (outputs.check_stop_buffers or {}).items()
            })
            self.eng.cut_verify_step(batch, outputs, host)
            stop_col = host.buffers["verify_stopped"].tolist()
            for i, rid in enumerate(host.request_ids):
                self.out[rid] += outputs.per_rid_outputs[rid]["new_token"][0].reshape(-1).cpu().tolist()
                stopped[rid] = bool(stop_col[i])
            self.accepted = getattr(self, "accepted", []) + host.buffers["verify_accepted"].tolist()[: len(rids)]
            return stopped
        tokens = outputs.check_stop_buffers["new_token"].reshape(-1).cpu().tolist()
        for i, rid in enumerate(rids):
            self.out[rid].append(tokens[i])
            stopped[rid] = tokens[i] == STOP or len(self.out[rid]) >= self.max[rid]
        return stopped

    def run(self, rids: list[int], max_steps: int = 400) -> dict[int, list[int]]:
        live = [rid for rid in rids if self.out[rid][-1] != STOP and len(self.out[rid]) < self.max[rid]]
        for _ in range(max_steps):
            if not live:
                break
            stopped = self.step(live)
            live = [rid for rid in live if not stopped[rid]]
        assert not live, "rows still running"
        for rid in rids:
            self.eng.remove_request(rid)
        return {rid: self.out[rid] for rid in rids}


def decode(model, prompts: dict[int, list[int]], max_tokens: int, seed: int = 0, temperature: float = 0.0,
           batched: bool = True, weights=None, eng: Engine | None = None, **engine_kw) -> tuple[dict, Driver]:
    eng = eng or engine(model, seed, weights=weights, **engine_kw)
    driver = Driver(eng, model)
    for rid, prompt in prompts.items():
        driver.add(rid, prompt, max_tokens, temperature=temperature, seed=rid)
    groups = [list(prompts)] if batched else [[rid] for rid in prompts]
    for group in groups:
        driver.prefill(group, prompts)
        driver.run(group)
    return driver.out, driver


def forced_drafts(reference: dict[int, list[int]], spoil=lambda j: False, record: list | None = None):
    """A `DSparkDrafter.propose` that runs the real one (its block's KV writes and
    attention included), records ``(rows, anchor positions, its drafts)`` into
    ``record``, then proposes each
    row's ``reference`` continuation instead, spoiling draft ``j`` where ``spoil(j)``:
    so a random drafter still exercises every accepted count."""
    from mstar.model.components.dspark.drafter import DSparkDrafter

    real = DSparkDrafter.propose

    def propose(self, inputs):
        drafts = real(self, inputs)
        n, k = drafts.shape
        anchors = inputs.resources["rope"].pos_ids("main").view(n, k + 1)[:, 0].tolist()
        if record is not None:
            record.append((list(inputs.request_ids), anchors, drafts.clone()))
        rows = []
        for rid, at in zip(inputs.request_ids, anchors, strict=True):
            seq = reference[rid]
            row = [seq[at + 1 + j] if at + 1 + j < len(seq) else 0 for j in range(k)]
            rows.append([(t + 1) % VOCAB if spoil(j) else t for j, t in enumerate(row)])
        return torch.tensor(rows, dtype=torch.long)

    return propose


class _DenseAttention:
    """Causal attention over one call's own K/V: the reference recomputes the whole sequence."""

    requires_kv_write = False
    default_label = "main"

    def set_default_label(self, label):
        pass

    def set_default_layer_idx(self, idx):
        pass

    def layer_view(self):
        return None

    def run(self, q, kv_cache_layer=None, k=None, v=None):
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        out = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True)
        return out.transpose(0, 1)


class _DensePositions:
    def __init__(self):
        self.n = 0

    def fusable_rope_theta(self, head_dim):
        return None

    def apply_qk(self, q, k, label=None, rope_theta=10_000.0, **kwargs):
        _rope_cpu(q, k, torch.arange(q.shape[0]), None, None, False, 1.0, rope_theta)
        return q, k


def _dense_target(model, seed: int, like=None):
    lm = model.build_lm("cpu", dtype=DTYPE)
    init_random(lm, seed)
    if like is not None:
        copy_weights(lm, like.model)
    attn, pos = _DenseAttention(), _DensePositions()
    for layer in lm.layers:
        layer.self_attn.bind_resources({"attn": attn, "kv_cache": attn, "rope": pos})
    return lm.eval()


def reference_greedy(model, prompt: list[int], max_tokens: int, seed: int = 0, like=None,
                     margins: list | None = None) -> list[int]:
    """Plain greedy decoding of the target with no engine: the whole sequence
    recomputed each token, dense causal attention, the same weights (``like``'s,
    when given: a node on another device or dtype). ``margins`` collects each
    token's top-2 log-prob margin."""
    lm = _dense_target(model, seed, like)
    seq, out = list(prompt), []
    with torch.no_grad():
        while len(out) < max_tokens:
            logp = lm.logits(lm(lm.embed(torch.tensor(seq)), label="main"))[-1].log_softmax(-1)
            top = logp.topk(2)
            token = int(top.indices[0])
            if margins is not None:
                margins.append(float(top.values[0] - top.values[1]))
            out.append(token)
            seq.append(token)
            if token == STOP:
                break
    return out


class _Taps:
    """What ``run_layers`` hands a drafter's taps, kept for the reference."""

    def __init__(self, layers):
        self.columns = {layer: i for i, layer in enumerate(layers)}
        self.rows = {}

    def write(self, layer, hidden):
        self.rows[layer] = hidden.clone()


class _BlockAttention:
    """The drafter's block attention in the reference: every query of the block
    over its layer's context K/V and the block's first ``k`` rows, no mask."""

    requires_kv_write = False
    default_label = "main"

    def __init__(self, context, k, first=0):
        self.context, self.k, self.layer, self._first = context, k, 0, first

    def layer_set(self, name):
        # where the drafter's layer set sits, as the pool would say: its layers follow the target's
        return SimpleNamespace(first=self._first)

    def set_default_label(self, label):
        pass

    def set_default_layer_idx(self, idx):
        self.layer = idx

    def layer_view(self):
        return self.layer

    def run(self, q, kv_cache_layer=None, k=None, v=None):
        ck, cv = self.context[kv_cache_layer]
        keys, values = torch.cat([ck, k[: self.k]]), torch.cat([cv, v[: self.k]])
        rep = q.shape[1] // keys.shape[1]
        keys, values = keys.repeat_interleave(rep, 1), values.repeat_interleave(rep, 1)
        out = F.scaled_dot_product_attention(q.transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1))
        return out.transpose(0, 1)


class _AtPositions:
    def __init__(self, start):
        self.start = start

    def fusable_rope_theta(self, head_dim):
        return None

    def apply_qk(self, q, k, label=None, rope_theta=10_000.0, **kwargs):
        _rope_cpu(q, k, torch.arange(self.start, self.start + q.shape[0]), None, None, False, 1.0, rope_theta)
        return q, k


def bind_share(sub, k: int):
    """What the engine binds into a verifying node at load (`NodeSubmodule.bind_draft`), for a
    test that drives the node without one: its decode walk verifies ``k`` drafts."""
    from mstar.engine.resources.draft.node import DrafterShare

    given = dict(verify_walk=sub.DECODE_WALK, sampler_key="sampler", num_draft_tokens=k,
                 prompt_walks=sub.token_stop.prefill_walks)
    fields = {name: given.get(name, frozenset() if name.endswith("writes") else {})
              for name in DrafterShare._fields if name not in DrafterShare._field_defaults}
    sub.bind_draft(DrafterShare(**fields))
    return sub


def drafter_first_layer(model) -> int:
    """The drafter's first layer in the KV pool the model declares, as the engine binds it."""
    (decl,) = model.verify_nodes()
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    return specs[decl.needs.kv.kv_cache].config.layer_sets[decl.needs.kv.name].first


def reference_drafts(model, seq: list[int], at: int, k: int, seed: int = 0, like=None,
                     margins: list | None = None) -> list[int]:
    """DSpark's ``k`` drafts for the anchor ``seq[at]``, recomputed densely with no
    engine: its context is ``hidden_norm(fc(taps))`` of the target over
    ``seq[:at]``, each drafter layer's K/V of that context (rotated at
    positions 0..at-1), and the block [anchor, mask x k] at positions
    at..at+k attending to the context and its first k rows; rows 0..k-1 give
    the base logits, and the Markov head drafts left to right."""
    sub = build_submodule(model, seed)
    if like is not None:
        copy_weights(sub, like)
    lm, drafter, decl = sub.model, sub.drafter, model.drafter_decl
    # no engine binds the drafter here: its first layer from the pool the model declares
    first = drafter_first_layer(model)
    with torch.no_grad():
        dense, pos = _DenseAttention(), _DensePositions()
        for layer in lm.layers:
            layer.self_attn.bind_resources({"attn": dense, "kv_cache": dense, "rope": pos})
        taps = _Taps(decl.config.taps)
        lm(lm.embed(torch.tensor(seq[:at])), label="main", taps=taps)
        stacked = torch.stack([taps.rows[layer] for layer in decl.config.taps], dim=1)
        context = drafter.hidden_norm(drafter.fc(stacked.reshape(at, -1)))
        per_layer = {}
        for i, layer in enumerate(drafter.backbone.layers):
            attn = layer.self_attn
            q, key, value = attn._project_qkv(context)
            q, key = attn._apply_qk_norm(q, key)
            _rope_cpu(q, key, torch.arange(at), None, None, False, 1.0, decl.config.rope_theta)
            per_layer[first + i] = (key, value)
        block_attn = _BlockAttention(per_layer, k, first)
        positions = _AtPositions(at)
        for layer in drafter.backbone.layers:
            layer.self_attn.bind_resources({"dspark_attn": block_attn, "kv_cache": block_attn, "rope": positions})
        block = torch.tensor([seq[at]] + [decl.config.mask_token_id] * k)
        hidden = drafter.backbone(drafter.backbone.embed(block), label="main")[:k]
        base = drafter.backbone.logits(hidden).float()
        prev, drafts = torch.tensor(seq[at]), []
        for i in range(k):
            logits = base[i] + drafter.markov_head(prev[None])[0]
            top = logits.log_softmax(-1).topk(2)
            if margins is not None:
                margins.append(float(top.values[0] - top.values[1]))
            prev = top.indices[0]
            drafts.append(int(prev))
    return drafts


def _indexed(device) -> torch.device:
    """The device as a worker passes it: a GPU with its index (capture sets the device by index)."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device
