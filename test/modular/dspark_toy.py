"""The draft-verify toy (``draft_verify_toy``) with a DSpark drafter: its trunk taps two of its
layers for a randomly initialized ``DSparkDrafter`` (``mstar.model.components.dspark``) on KV,
positions and attention of its own, which drafts each verify step's block in one pass.

The trunk is the target and the drafter the real component; on CPU the drafter attends through
the toy's dense ``ToyAttention`` (non-causal, the block's last row hidden), on a GPU through
FlashInfer. ``reference_drafts`` recomputes what the drafter should draft at a step from the
request's kept tokens alone, in float64, with no cache: a dense target pass for the taps, the
drafter's context K/V and its block, so a test can check the engine's drafts against it.
"""

from __future__ import annotations

import dataclasses

import draft_verify_toy as toy
import torch

from mstar.engine.resources import (
    AttentionSpec,
    AttentionStep,
    DraftPolicy,
    KVReqConfig,
    PagedKVConfig,
    PositionStep,
    SamplerStep,
    SamplingReqConfig,
    Segment,
    SubmoduleStep,
    draft_verify_step,
    with_draft_verify,
)
from mstar.engine.resources.base import RewindKind
from mstar.model.components.dspark import DSpark, DSparkConfig, DSparkDrafter, DSparkKeys
from mstar.model.submodule_base import BatchedModelOutput

KEYS = DSparkKeys()
DKV, DATTN, DPOS = KEYS
TAPS = (1, toy.LAYERS)
MASK = toy.VOCAB - 1


def config(block_size: int = 7, causal: bool = False, taps=TAPS) -> DSparkConfig:
    return DSparkConfig(
        hidden_size=toy.HIDDEN, num_layers=2, num_attention_heads=toy.HEADS, num_key_value_heads=toy.HEADS,
        head_dim=toy.HEAD_DIM, vocab_size=toy.VOCAB, taps=tuple(taps), target_hidden_size=toy.HIDDEN,
        block_size=block_size, mask_token_id=MASK, markov_rank=16, intermediate_size=2 * toy.HIDDEN,
        rope_theta=10_000.0, max_position_embeddings=toy.MAX_POS, causal=causal, source="toy",
    )


def declaration(k: int, max_num_pages: int = 256, kv_heads: int = toy.HEADS, sample: bool = False,
                causal: bool = False, dtype: torch.dtype = torch.float32,
                rope_dtype: torch.dtype | None = torch.float32) -> DSpark:
    cfg = config(causal=causal)
    kv = PagedKVConfig(
        num_layers=cfg.num_layers, num_kv_heads=kv_heads, head_dim=cfg.head_dim,
        max_seq_len=toy.MAX_POS, max_num_pages=max_num_pages, page_size=toy.PAGE,
    )
    # float32 rotation on CPU, where the reference is float64; FlashInfer rotates 16-bit only
    return DSpark(cfg, k, kv, KEYS, sample=sample, tap_dtype=dtype, rope_dtype=rope_dtype)


class ToyDSparkSubmodule(toy.ToySubmodule):
    """The toy whose verify steps take their drafts from a ``DSparkDrafter``."""

    def __init__(self, k: int = 3, drafter_seed: int = 5, sample: bool = False, causal: bool = False,
                 capture_batch_sizes=(1, 2, 4), max_num_pages: int = 256, kv_heads: int = toy.HEADS,
                 rope_dtype: torch.dtype | None = torch.float32, device_rewind: bool = False):
        self.decl = declaration(k, max_num_pages, kv_heads=kv_heads, sample=sample, causal=causal,
                                rope_dtype=rope_dtype)
        super().__init__(k=k, capture_batch_sizes=capture_batch_sizes, taps=self.decl.config.taps)
        # each verdict taken back on the device in the step after it (``DraftPolicy.device_rewind``)
        self.device_rewind = device_rewind
        self.drafter = DSparkDrafter(self.decl)
        self.drafter.reset_parameters()
        gen = torch.Generator().manual_seed(drafter_seed)
        with torch.no_grad():
            for p in self.drafter.parameters():
                p.mul_(1.0 + 0.5 * torch.randn(p.shape, generator=gen))
        # what the drafter proposed at each verify step, for a test to check; ``oracle`` and
        # ``kept`` override some of its drafts with the greedy stream's (``_Logged``)
        self.draft_log: list | None = None
        self.oracle: dict | None = None
        self.kept: dict = {}

    def get_draft_policy(self, graph_walk):
        if graph_walk != "decode":
            return None
        return DraftPolicy(
            max_drafts=self.k, stop_ids=frozenset({toy.STOP}), taps=self.decl.config.taps,
            device_rewind=self.device_rewind,
        )

    def declare_step(self, graph_walk, request_ids, inputs, slot_lease=None, **kwargs):
        steps = {toy.KV: toy.KVStep(), toy.ATTN: AttentionStep(causal=True), toy.POS: PositionStep()}
        spans = [inp.input_seq_len for inp in inputs]
        if graph_walk == "decode":
            steps.update(self.decl.verify_steps(spans[0] - 1 if spans else None))
        else:
            steps[toy.SAMPLER] = SamplerStep(apply_penalty=False)
            steps.update(self.decl.prefill_steps(request_ids, spans))
        step = SubmoduleStep(
            steps=steps,
            segments=[Segment(rid, "main", span) for rid, span in zip(request_ids, spans, strict=True)],
        )
        if graph_walk != "decode":
            return step
        return with_draft_verify(step, self.get_draft_policy(graph_walk), toy.DV, sampler=toy.SAMPLER)

    def forward_batched(self, graph_walk, engine_inputs, input_ids, last=None, **kwargs):
        res = engine_inputs.resources
        pos = res[toy.POS].pos_ids("main")
        rids = list(engine_inputs.request_ids)
        if graph_walk != "decode":
            logits = self._trunk(input_ids, pos[: input_ids.shape[0]]).index_select(0, last)
            tokens = res[toy.SAMPLER].sample(rids, logits)
            self.drafter.prefill()
            return BatchedModelOutput(row_outputs={"new_token": tokens, "text_inputs": tokens})
        target = lambda ids: self._skewed(self._trunk(ids, pos[: ids.shape[0]]))  # noqa: E731
        drafter = self.drafter
        if self.draft_log is not None:
            drafter = _Logged(self.drafter, self.draft_log, self.oracle, self.kept)
        carry = draft_verify_step(res[toy.DV], rids, input_ids, target, drafter)
        return BatchedModelOutput(row_outputs={"text_inputs": carry})


class _Logged:
    """The drafter, its drafts kept as it proposes. With ``oracle`` (request ->
    its greedy stream, prompt included, and ``kept`` -> how many tokens each has kept), each
    row's first ``kept % (drafts + 1)`` drafts are replaced by the stream's, after logging, so
    the target keeps some and rejects the rest (eager steps only)."""

    def __init__(self, drafter, log, oracle=None, kept=None):
        self.drafter, self.log, self.oracle, self.kept = drafter, log, oracle, kept

    def propose(self, inputs, dv, request_ids):
        drafts = self.drafter.propose(inputs, dv, request_ids)
        self.log.append((inputs.reshape(inputs.shape[0], -1)[:, -1].clone(), drafts.tokens.clone()))
        if self.oracle is None:
            return drafts
        tokens = drafts.tokens.clone()
        width = tokens.shape[1]
        for row, rid in enumerate(request_ids):
            if rid not in self.oracle:
                continue
            kept = self.kept[rid]
            right = kept % (width + 1)
            tokens[row, :right] = torch.tensor(self.oracle[rid][kept:kept + right])
        return drafts._replace(tokens=tokens)

    def __getattr__(self, name):
        return getattr(self.drafter, name)


def specs(sub: ToyDSparkSubmodule, attn: str = "toy", **kwargs):
    drafter = sub.decl.node_resources({toy.NODE})
    if attn == "toy":
        drafter = [
            toy.ToyAttentionSpec(resource_key=DATTN, nodes={toy.NODE}, kv=DKV) if isinstance(spec, AttentionSpec)
            else spec
            for spec in drafter
        ]
    elif "fi_backend" in kwargs:
        for spec in drafter:
            if isinstance(spec, AttentionSpec):
                spec.config = dataclasses.replace(spec.config, flashinfer_backend=kwargs["fi_backend"])
    return toy.specs(attn, **kwargs) + drafter


def engine(sub: ToyDSparkSubmodule, device, attn: str = "toy", warmup: bool = False, dtype=torch.float32,
           kv_dtype: torch.dtype | None = None, comm=None, **spec_kwargs):
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.engine.engine import Engine
    from mstar.engine.resources.kv.transfer import TransferEngineInfo

    sub.to(device=device, dtype=dtype)
    kv_dtype = kv_dtype or dtype
    sub.attn_dtype = kv_dtype
    eng = Engine(None, autocast_dtype=None)
    eng.load_model(
        submodules={toy.NODE: sub}, specs=specs(sub, attn, **spec_kwargs),
        parallel_groups=toy._Group(comm), device=device,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
        kv_cache_type=kv_dtype, model=toy.ToyModel(),
    )
    if warmup:
        eng.warmup()
    return eng


def settle_all(eng, request_ids) -> None:
    """Take back on the host every verdict of these requests still waiting there (under device
    rewind the last committed step's wait until the request's next step would take them back
    on the device), so their streams' lengths are the kept tokens'."""
    dv = eng._resources[toy.DV]
    device, dv.device_rewind = dv.device_rewind, False
    try:
        verdicts = dv.settle(list(request_ids))
    finally:
        dv.device_rewind = device
    if verdicts is not None:
        for resource in eng._resources.values():
            if resource.rewind_kind is RewindKind.HOST:
                resource.rewind(verdicts)


def dspark_layout(decl: DSpark, prompt: int, max_tokens: int, lookahead: int) -> KVReqConfig:
    """The drafter's KV config as the conductor builds it from ``DSpark.kv_layout``."""
    layouts = decl.kv_layout(prompt, "prefill", "decode_loop", lookahead)[DKV]
    rows = {label: layout.row() for label, layout in layouts.items()}
    config = KVReqConfig()
    config.apply_conductor_config(kv_layout=rows, loop_caps={"decode_loop": max_tokens - 1})
    return config


class Driver(toy.Driver):
    """``draft_verify_toy.Driver`` with the drafter's KV layout beside the target's."""

    def add_batch(self, prompts, max_tokens, ignore_eos=False, seeds=None, **sampling):
        lookahead = 2 * self.sub.k + 1
        for rid, prompt in prompts.items():
            config = SamplingReqConfig(**{
                "temperature": 0.0, "ignore_eos": ignore_eos, "_seed": (seeds or {}).get(rid, 0), **sampling,
            })
            self.eng.add_request(rid, {
                toy.KV: toy.layout(len(prompt), max_tokens, lookahead),
                DKV: dspark_layout(self.sub.decl, len(prompt), max_tokens, lookahead),
                toy.SAMPLER: config,
            })
            self.sampling[rid] = config
            self.max_tokens[rid] = max_tokens
            self.emitted[rid] = []
        out = self._step("prefill", {rid: torch.tensor(prompt) for rid, prompt in prompts.items()})
        for rid in prompts:
            self.emitted[rid] += out[rid]["new_token"][0].reshape(-1).tolist()
            self.next[rid] = out[rid]["text_inputs"][0]


# ── a drafter recomputed from scratch, float64, no cache ──


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    angles = torch.outer(positions.double(), inv)
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
    a, b = x[..., : d // 2], x[..., d // 2:]
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


def _streams(ids: list[int], seed: int = toy.SEED) -> torch.Tensor:
    """The float64 trunk's stream entering each block and after the last: ``[len, LAYERS + 1, H]``."""
    sub = toy.ToySubmodule(k=0, seed=seed).double()
    with torch.no_grad():
        x = sub.embed(torch.tensor(ids)) + sub.pos_emb(torch.arange(len(ids)))
        out = [x]
        for block in sub.blocks:
            q, k, v = block.qkv(x).view(len(ids), 3, toy.HEADS, toy.HEAD_DIM).unbind(1)
            scores = torch.einsum("qhd,khd->hqk", q, k) * toy.HEAD_DIM ** -0.5
            mask = torch.ones(len(ids), len(ids), dtype=torch.bool).tril()
            attn = scores.masked_fill(~mask, float("-inf")).softmax(-1)
            o = torch.einsum("hqk,khd->qhd", attn, v).reshape(len(ids), toy.HIDDEN)
            x = block.mlp(x + block.attn_scale * block.o(o))
            out.append(x)
    return torch.stack(out, 1)


def reference_drafts(drafter: DSparkDrafter, kept: list[int], drafts: int) -> tuple[list[int], list[float]]:
    """What ``drafter`` should draft at a verify step of ``drafts`` drafts whose anchor is the
    last of ``kept``, and the margins between its top two logits at each draft (where a float32
    engine may part from it): the context is every kept token before the anchor, its taps from
    a dense float64 target pass."""
    cfg = drafter.decl.config
    with torch.no_grad():
        d = _double(drafter)
        lm = d.backbone
        anchor_pos = len(kept) - 1
        taps = _streams(kept[:anchor_pos])[:, list(cfg.taps)] if anchor_pos else None
        block = torch.tensor([kept[-1]] + [cfg.mask_token_id] * drafts)
        h = lm.embed_tokens(block)
        positions = torch.arange(anchor_pos, anchor_pos + drafts + 1)
        context = d.hidden_norm(d.fc(taps.reshape(anchor_pos, -1))) if anchor_pos else None
        for layer in lm.layers:
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q, k, v = attn._project_qkv(x)
            q, k = attn._apply_qk_norm(q, k)
            q, k = _rope(q, positions, cfg.rope_theta), _rope(k, positions, cfg.rope_theta)
            keys, values = k[:drafts], v[:drafts]  # the block's last row is hidden
            if context is not None:
                _, ck, cv = attn._project_qkv(context)
                _, ck = attn._apply_qk_norm(ck, ck)
                ck = _rope(ck, torch.arange(anchor_pos), cfg.rope_theta)
                keys, values = torch.cat([ck, keys]), torch.cat([cv, values])
            scores = torch.einsum("qhd,khd->hqk", q, keys) * cfg.head_dim ** -0.5
            out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values).reshape(drafts + 1, -1)
            h = h + attn.o_proj(out)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        hidden = lm.norm(h)[:drafts]
        base = lm.logits(hidden)
        prev, tokens, margins = kept[-1], [], []
        for i in range(drafts):
            logits = base[i] + d.markov_head.markov_w2(d.markov_head.markov_w1(torch.tensor(prev)))
            top = logits.topk(2)
            tokens.append(int(top.indices[0]))
            margins.append(float(top.values[0] - top.values[1]))
            prev = tokens[-1]
    return tokens, margins


def _double(drafter: DSparkDrafter) -> DSparkDrafter:
    """A float64 copy of the weights, on no resources."""
    out = DSparkDrafter(drafter.decl)
    out.load_state_dict({name: value.detach().cpu() for name, value in drafter.state_dict().items()})
    return out.double()
