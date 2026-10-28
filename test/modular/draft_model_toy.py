"""The draft-verify toy (``draft_verify_toy``) with a draft model: its trunk verifies the drafts
of a separate tiny LLM, which runs on KV, positions and attention of its own through
``mstar.model.components.draft_model``.

The drafter is the trunk with its weights moved by noise (``drafter_seed``), so it is right
about some tokens and wrong about others; with no noise it is the target itself and drafts
every token right, as long as its KV and positions are right. On CPU the
drafter attends through the toy's dense ``ToyAttention``; on a GPU through FlashInfer.
"""

from __future__ import annotations

import copy

import draft_verify_toy as toy
import torch
from torch import nn

from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig
from mstar.engine.resources import (
    AttentionSpec,
    AttentionStep,
    KVReqConfig,
    KVStep,
    PositionStep,
    SamplerStep,
    SamplingReqConfig,
    Segment,
    SubmoduleStep,
    draft_verify_step,
    with_draft_verify,
)
from mstar.engine.resources.convenience import AttentionCallable
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.model.components.draft_model import DraftKeys, DraftLLMProposer, DraftModel
from mstar.model.submodule_base import ARNodeInputs, BatchedModelOutput

KEYS = DraftKeys()
DKV, DATTN, DPOS = KEYS
# the drafter's weights: the target's, each moved by this fraction of its own spread
NOISE = 0.3


def drafter_trunk(noise_seed: int | None, noise: float = NOISE) -> toy.ToySubmodule:
    """The toy trunk under the drafter's weights: the target's, with noise drawn from
    ``noise_seed`` (None: none, the target itself)."""
    trunk = toy.ToySubmodule(k=0)
    if noise_seed is not None:
        gen = torch.Generator().manual_seed(noise_seed)
        with torch.no_grad():
            for p in trunk.parameters():
                p.add_(torch.randn(p.shape, generator=gen) * noise * p.std())
    return trunk


class ToyLM(nn.Module):
    """The toy trunk as a standalone LM on the drafter's resources (``DraftLM``): learned
    position embeddings read at the drafter's planned positions, so a missed rewind of its
    KV or positions changes what it drafts."""

    def __init__(self, noise_seed: int | None = None, noise: float = NOISE, keys: DraftKeys = KEYS):
        super().__init__()
        trunk = drafter_trunk(noise_seed, noise)
        self.embed_tokens = copy.deepcopy(trunk.embed)
        self.pos_emb = copy.deepcopy(trunk.pos_emb)
        self.blocks = copy.deepcopy(trunk.blocks)
        self.lm_head = copy.deepcopy(trunk.lm_head)
        self.keys = keys
        self.attn_dtype: torch.dtype | None = None
        self.attend = None
        self.pos = None

    def bind_resources(self, resources):
        self.attend = AttentionCallable(kv=resources[self.keys.kv], attn=resources[self.keys.attn])
        self.pos = resources[self.keys.pos]

    def embed(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(self, input_embeds, *, label):
        positions = self.pos.pos_ids(label)[: input_embeds.shape[0]]
        x = input_embeds + self.pos_emb(positions)
        self.attend.bind_step(label)
        for i, block in enumerate(self.blocks):
            self.attend.set_layer_idx(i)
            x = block(x, self.attend, self.attn_dtype)
        return x

    def logits(self, hidden):
        return self.lm_head(hidden)


def draft_model(k: int, sample: bool = False, max_num_pages: int = 256) -> DraftModel:
    return DraftModel(
        max_drafts=k, keys=KEYS, sample=sample,
        kv=PagedKVConfig(
            num_layers=toy.LAYERS, num_kv_heads=toy.HEADS, head_dim=toy.HEAD_DIM,
            max_seq_len=toy.MAX_POS, max_num_pages=max_num_pages, page_size=toy.PAGE,
        ),
    )


class ToyDraftModelSubmodule(toy.ToySubmodule):
    """The toy whose verify steps take their drafts from a ``DraftLLMProposer``. ``corrupt``
    swaps the drafter's draft at every position divisible by it after it drafts, so the
    target rejects there while the drafter's own rows stay what it drafted."""

    def __init__(self, k: int = 3, drafter_seed: int | None = 7, sample: bool = False, corrupt: int = 0,
                 noise: float = NOISE, capture_batch_sizes=(1, 2, 4), max_num_pages: int = 256):
        super().__init__(k=k, capture_batch_sizes=capture_batch_sizes)
        self.decl = draft_model(k, sample=sample, max_num_pages=max_num_pages)
        self.drafter = DraftLLMProposer(self.decl, ToyLM(drafter_seed, noise))
        self.draft_corrupt = corrupt

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        ids = inputs["text_inputs"][0].reshape(-1).to(torch.long)
        if graph_walk == "decode":
            # the last two kept tokens: the drafter's block starts one position early
            return ARNodeInputs(input_ids=ids[-2:], input_seq_len=self.block)
        return ARNodeInputs(input_ids=ids, input_seq_len=ids.shape[0])

    def declare_step(self, graph_walk, request_ids, inputs, slot_lease=None, **kwargs):
        steps = {toy.KV: KVStep(), toy.ATTN: AttentionStep(causal=True), toy.POS: PositionStep()}
        spans = [inp.input_seq_len for inp in inputs]
        if graph_walk == "decode":
            steps.update(self.decl.verify_steps())
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

    def preprocess(self, graph_walk, engine_inputs, inputs):
        device = self.embed.weight.device
        if graph_walk == "decode":
            return {"input_ids": torch.stack([inp.input_ids for inp in inputs]).to(device)}
        out = super().preprocess(graph_walk, engine_inputs, inputs)
        out["draft_ids"] = torch.cat([inp.input_ids[:-1] for inp in inputs]).to(device)
        out["prev"] = torch.stack([inp.input_ids[-1] for inp in inputs]).to(device)
        return out

    def forward_batched(self, graph_walk, engine_inputs, input_ids, last=None, draft_ids=None,
                        prev=None, **kwargs):
        res = engine_inputs.resources
        pos = res[toy.POS].pos_ids("main")
        rids = list(engine_inputs.request_ids)
        if graph_walk != "decode":
            logits = self._trunk(input_ids, pos[: input_ids.shape[0]]).index_select(0, last)
            tokens = res[toy.SAMPLER].sample(rids, logits)
            self.drafter.prefill(draft_ids)
            carry = torch.stack([prev, tokens.to(prev.dtype)], 1)
            return BatchedModelOutput(row_outputs={"new_token": tokens, "text_inputs": carry})
        n = input_ids.shape[0]
        positions = pos[: n * self.block]
        target = lambda ids: self._trunk(ids, positions)  # noqa: E731
        if not self.draft_corrupt:
            carry = draft_verify_step(res[toy.DV], rids, input_ids, target, self.drafter)
            return BatchedModelOutput(row_outputs={"text_inputs": carry})
        dv = res[toy.DV]
        drafts = self.drafter.propose(input_ids, dv, rids)
        at = positions.view(n, self.block)[:, 1:]
        tokens = torch.where(at % self.draft_corrupt == 0, (drafts.tokens + 1) % toy.VOCAB, drafts.tokens)
        block = dv.block_ids(input_ids[:, -1], dv.agree(tokens))
        verdict = dv.verify(rids, block, target(block.reshape(-1)), drafts.probs)
        return BatchedModelOutput(row_outputs={"text_inputs": self.drafter.next_inputs(block, verdict)})

    def get_accelerator_graph_configs(self, device, tp_world_size=1):
        return [BatchedAcceleratorGraphConfig(
            capture_graph_walk="decode",
            single_request_inputs=ARNodeInputs(
                input_ids=torch.zeros(2, dtype=torch.long, device=device), input_seq_len=self.block,
            ),
            capture_batch_sizes=self.capture_batch_sizes,
        )]


def specs(sub: ToyDraftModelSubmodule, attn: str = "toy", **kwargs):
    drafter = sub.decl.node_resources({toy.NODE})
    if attn == "toy":
        drafter = [
            toy.ToyAttentionSpec(resource_key=DATTN, nodes={toy.NODE}, kv=DKV)
            if isinstance(spec, AttentionSpec) else spec
            for spec in drafter
        ]
    return toy.specs(attn, **kwargs) + drafter


def engine(sub: ToyDraftModelSubmodule, device, attn: str = "toy", warmup: bool = False,
           dtype=torch.float32, kv_dtype: torch.dtype | None = None, comm=None, **spec_kwargs):
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.engine.engine import Engine
    from mstar.engine.resources.kv.transfer import TransferEngineInfo

    sub.to(device=device, dtype=dtype)
    kv_dtype = kv_dtype or dtype
    sub.attn_dtype = sub.drafter.lm.attn_dtype = kv_dtype
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


def draft_layout(decl: DraftModel, prompt: int, max_tokens: int, lookahead: int) -> KVReqConfig:
    """The drafter's KV config as the conductor builds it from ``DraftModel.kv_layout``."""
    rows = {
        label: layout.row()
        for label, layout in decl.kv_layout(prompt, "prefill", "decode_loop", lookahead)[DKV].items()
    }
    config = KVReqConfig()
    config.apply_conductor_config(kv_layout=rows, loop_caps={"decode_loop": max_tokens - 1})
    return config


class Driver(toy.Driver):
    """``draft_verify_toy.Driver`` with the drafter's KV layout beside the target's."""

    def __init__(self, eng, sub: ToyDraftModelSubmodule, draft_lookahead: int | None = None,
                 plan_ahead: bool = False):
        super().__init__(eng, sub, plan_ahead=plan_ahead)
        self.draft_lookahead = draft_lookahead

    def add_batch(self, prompts, max_tokens, ignore_eos=False, seeds=None, **sampling):
        lookahead = 2 * self.sub.k + 1
        draft_lookahead = lookahead if self.draft_lookahead is None else self.draft_lookahead
        for rid, prompt in prompts.items():
            config = SamplingReqConfig(**{
                "temperature": 0.0, "ignore_eos": ignore_eos, "_seed": (seeds or {}).get(rid, 0), **sampling,
            })
            self.eng.add_request(rid, {
                toy.KV: toy.layout(len(prompt), max_tokens, lookahead),
                DKV: draft_layout(self.sub.decl, len(prompt), max_tokens, draft_lookahead),
                toy.SAMPLER: config,
            })
            self.sampling[rid] = config
            self.max_tokens[rid] = max_tokens
            self.emitted[rid] = []
        out = self._step("prefill", {rid: torch.tensor(prompt) for rid, prompt in prompts.items()})
        for rid in prompts:
            self.emitted[rid] += out[rid]["new_token"][0].reshape(-1).tolist()
            self.next[rid] = out[rid]["text_inputs"][0]


def drafter_reference(prompt: list[int], k: int, drafter_seed: int | None, max_tokens: int,
                      noise: float = NOISE) -> list[int]:
    """How many drafts each verify step of an ``ignore_eos`` reply of ``max_tokens`` accepts,
    from a float64 target and a float64 drafter recomputed from scratch for every draft: no
    KV, so what a correct rewind gives."""
    lm = drafter_trunk(drafter_seed, noise).double()
    # past the cap: a step's verdict counts the drafts the target agrees with, cut or not
    tokens = list(prompt) + toy.reference(prompt, max_tokens + k + 1, ignore_eos=True)
    kept, accepted = len(prompt) + 1, []
    while kept - len(prompt) < max_tokens:
        drafts: list[int] = []
        for _ in range(k):
            drafts.append(int(toy.next_logits(tokens[:kept] + drafts, sub=lm).argmax()))
        good = 0
        while good < k and drafts[good] == tokens[kept + good]:
            good += 1
        accepted.append(good)
        kept += good + 1
    return accepted
