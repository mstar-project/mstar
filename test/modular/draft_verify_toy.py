"""A tiny autoregressive model with a chained draft head, for the draft-verify engine tests.

The trunk is a few attention layers over the paged KV resource, with learned position
embeddings read at the position resource's planned positions, so a missed rewind of either
changes the tokens it emits. The drafter is MTP-shaped but stateless: the draft for each row
of a block is the trunk's lm head over the previous draft's embedding at its position, with the
attention skipped, so the target agrees with it often but not always. ``corrupt`` drafts a
wrong token at every position divisible by it, and ``skew`` at some others on ranks above 0, so
a test controls where verify steps reject; ``oracle``, one request's whole stream by position,
drafts every token right. With ``sampled_drafts`` the head samples its drafts instead
(``draft_sample``) and hands verify the distributions they came from.

``ToyAttention`` is a dense reference over the KV resource's pages for CPU runs; on a GPU the
toy runs the FlashInfer attention resource, which a graph can capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import (
    DRAFT_VERIFY,
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
    DraftPolicy,
    DraftVerifySpec,
    KVReqConfig,
    KVSpec,
    KVStep,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    PositionStep,
    SamplerSpec,
    SamplerStep,
    SamplingReqConfig,
    Segment,
    StepContext,
    SubmoduleStep,
    with_draft_verify,
)
from mstar.engine.resources.base import AttentionResource, RewindKind
from mstar.engine.resources.convenience import AttentionCallable
from mstar.engine.resources.kv.config import PrefixSpan, StreamLayout
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule, BatchedModelOutput

VOCAB, HIDDEN, HEADS, HEAD_DIM, LAYERS, MAX_POS, PAGE = 64, 128, 2, 64, 2, 512, 4
KV, ATTN, POS, SAMPLER, DV = "kv", "attn", "pos", "sampler", DRAFT_VERIFY
NODE = "llm"
STOP = 15  # the toy's stop token, which some of its streams reach mid-block
# weights whose reference streams keep their top two logits 0.015 or more apart, so a run
# with a 16-bit KV still picks the float64 argmax at every step
SEED = 36


class ToyAttention(AttentionResource):
    """Dense causal attention over the rows the KV resource holds, planned from its views."""

    rewind_kind = RewindKind.NONE
    requires_kv_write = True

    def __init__(self, kv: str):
        self._kv = kv
        self._views = None

    @classmethod
    def build(cls, spec, info):
        return cls(spec.kv)

    def depends_on(self):
        return {self._kv}

    def plan(self, step, ctx):
        self.reset_default_cursors()
        self._views = ctx.plan_results[self._kv]["main"].views

    @torch.compiler.disable
    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        del label, k, v, layer_idx
        page_size = kv_cache_layer.shape[2]
        outs, row = [], 0
        for view in self._views:
            pages = torch.as_tensor(view.page_idxs, dtype=torch.long)
            rows = kv_cache_layer[pages].transpose(0, 1).reshape(
                2, len(view.page_idxs) * page_size, *kv_cache_layer.shape[3:])[:, :view.length]
            start = view.length - view.to_compute
            for j in range(view.to_compute):
                keys, vals = rows[0, :start + j + 1].float(), rows[1, :start + j + 1].float()
                scores = torch.einsum("hd,thd->ht", q[row + j].float(), keys) * HEAD_DIM ** -0.5
                outs.append(torch.einsum("ht,thd->hd", scores.softmax(-1), vals).to(q.dtype))
            row += view.to_compute
        return torch.stack(outs)


@dataclass
class ToyAttentionSpec(NodeResourceSpec):
    kv: str = KV

    def depends_on(self):
        return {self.kv}

    @property
    def resource_class(self):
        return ToyAttention


class GreedySampler(SamplerResource):
    """The sampler resource with its eager draw replaced by an argmax, for CPU runs: the
    real one samples on CUDA or XPU only. Ingest, the refusals and ``verify`` are its own."""

    @torch.compiler.disable
    def sample(self, request_ids, logits, **kwargs):
        tokens = logits.argmax(-1)
        group = self._comm_group
        if group is not None and group.world_size > 1:
            tokens = group.broadcast(tokens, src=0)
        return tokens


@dataclass
class GreedySamplerSpec(SamplerSpec):
    @property
    def resource_class(self):
        return GreedySampler


class _Block(nn.Module):
    def __init__(self, attn_scale: float):
        super().__init__()
        self.qkv = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.o = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.up = nn.Linear(HIDDEN, 2 * HIDDEN, bias=False)
        self.down = nn.Linear(2 * HIDDEN, HIDDEN, bias=False)
        self.attn_scale = attn_scale

    def mlp(self, x):
        return x + self.down(torch.tanh(self.up(x)))

    def forward(self, x, attend, attn_dtype=None):
        dtype = attn_dtype or x.dtype
        q, k, v = (t.to(dtype).contiguous() for t in self.qkv(x).view(x.shape[0], 3, HEADS, HEAD_DIM).unbind(1))
        out = attend(q, k, v).reshape(x.shape[0], HIDDEN).to(x.dtype)
        return self.mlp(x + self.attn_scale * self.o(out))


class ToySubmodule(ARNodeSubmodule):
    """``k`` drafts per verify step; ``k=0`` decodes one token a step through the sampler."""

    def __init__(self, k: int = 3, corrupt: int = 0, skew: int = 0, attn_scale: float = 0.6,
                 seed: int = SEED, capture_batch_sizes=(1, 2, 4), sampled_drafts: bool = False):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.pos_emb = nn.Embedding(MAX_POS, HIDDEN)
        self.blocks = nn.ModuleList(_Block(attn_scale) for _ in range(LAYERS))
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        for name, p in self.named_parameters():
            std = 1.0 if name.startswith(("embed", "pos_emb")) else 0.4
            p.data = torch.randn(p.shape, generator=gen) * std / (1 if "embed" in name else p.shape[-1] ** 0.5)
        self.k, self.corrupt, self.skew = k, corrupt, skew
        self.sampled_drafts = sampled_drafts
        self.rank = 0
        self.logit_skew = 0.0
        self.oracle: torch.Tensor | None = None
        # the KV's dtype, which the attention runs in (FlashInfer: 16-bit); None: the model's
        self.attn_dtype: torch.dtype | None = None
        self.capture_batch_sizes = list(capture_batch_sizes)
        self.attend = None

    @property
    def block(self) -> int:
        return self.k + 1

    def bind_node_resources(self, resources):
        super().bind_node_resources(resources)
        self.attend = AttentionCallable(kv=resources[KV], attn=resources[ATTN])

    def get_draft_policy(self, graph_walk):
        if graph_walk != "decode" or not self.k:
            return None
        return DraftPolicy(max_drafts=self.k, stop_ids=frozenset({STOP}))

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        ids = inputs["text_inputs"][0].reshape(-1).to(torch.long)
        if graph_walk == "decode":
            return ARNodeInputs(input_ids=ids[-1:], input_seq_len=self.block)
        return ARNodeInputs(input_ids=ids, input_seq_len=ids.shape[0])

    def can_batch(self, batch, model_inputs):
        return True

    def declare_step(self, graph_walk, request_ids, inputs, slot_lease=None, **kwargs):
        steps = {KV: KVStep(), ATTN: AttentionStep(causal=True), POS: PositionStep()}
        policy = self.get_draft_policy(graph_walk)
        if policy is None:
            steps[SAMPLER] = SamplerStep(apply_penalty=False)
        step = SubmoduleStep(
            steps=steps,
            segments=[Segment(rid, "main", inp.input_seq_len) for rid, inp in zip(request_ids, inputs, strict=True)],
        )
        return step if policy is None else with_draft_verify(step, policy, DV)

    def preprocess(self, graph_walk, engine_inputs, inputs):
        device = self.embed.weight.device
        out = {"input_ids": torch.cat([inp.input_ids.reshape(-1) for inp in inputs]).to(device)}
        if graph_walk == "prefill":
            ends = torch.tensor([inp.input_seq_len for inp in inputs]).cumsum(0) - 1
            out["last"] = ends.to(device)
        return out

    def _trunk(self, ids, positions):
        x = self.embed(ids) + self.pos_emb(positions)
        self.attend.bind_step("main")
        for i, block in enumerate(self.blocks):
            self.attend.set_layer_idx(i)
            x = block(x, self.attend, self.attn_dtype)
        return self.lm_head(x).float()

    def _skewed(self, logits):
        """``logits`` as a rank above 0 reads them, when the test skews ranks: as a
        non-bitwise all-reduce rounds differently."""
        if not (self.logit_skew and self.rank > 0):
            return logits
        wobble = torch.sin(torch.arange(logits.numel(), device=logits.device)).view_as(logits)
        return logits + self.logit_skew * wobble

    def _draft(self, inputs, positions, dv=None, request_ids=None):
        """``[n, k]``: each draft from the one before it, the trunk without attention; and the
        distributions sampled drafts came from (None for the argmax head)."""
        prev, drafts, dists = inputs, [], []
        for i in range(self.k):
            x = self.embed(prev) + self.pos_emb(positions[:, i])
            for block in self.blocks:
                x = block.mlp(x)
            if self.sampled_drafts:
                draft, dist = dv.draft_sample(request_ids, self._skewed(self.lm_head(x).float()))
                dists.append(dist)
                drafts.append(draft)
                prev = draft
                continue
            draft = self.lm_head(x).argmax(-1)
            at = positions[:, i + 1]
            if self.oracle is not None:
                draft = self.oracle.to(at.device)[at]
            if self.corrupt:
                draft = torch.where(at % self.corrupt == 0, (draft + 1) % VOCAB, draft)
            if self.skew and self.rank > 0:
                draft = torch.where(at % self.skew == 1, (draft + 2) % VOCAB, draft)
            drafts.append(draft)
            prev = draft
        probs = torch.stack(dists, 1) if dists and dists[0] is not None else None
        return torch.stack(drafts, 1), probs

    def forward(self, graph_walk, engine_inputs, **kwargs):
        raise NotImplementedError("the toy runs batched")

    def forward_batched(self, graph_walk, engine_inputs, input_ids, last=None, **kwargs):
        res = engine_inputs.resources
        pos = res[POS].pos_ids("main")
        n = input_ids.shape[0]
        if graph_walk == "decode" and self.k:
            dv = res[DV]
            positions = pos[: n * self.block].view(n, self.block)
            rids = list(engine_inputs.request_ids)
            drafts, draft_probs = self._draft(input_ids, positions, dv, rids)
            block = dv.block_ids(input_ids, dv.agree(drafts))
            logits = self._skewed(self._trunk(block.view(-1), positions.reshape(-1)))
            verdict = dv.verify(rids, block, logits, draft_probs)
            return BatchedModelOutput(row_outputs={"text_inputs": verdict.next_inputs})
        logits = self._trunk(input_ids, pos[:n])
        if last is not None:
            logits = logits.index_select(0, last)
        tokens = res[SAMPLER].sample(list(engine_inputs.request_ids), logits)
        return BatchedModelOutput(row_outputs={"new_token": tokens, "text_inputs": tokens})

    def get_accelerator_graph_configs(self, device, tp_world_size=1):
        return [BatchedAcceleratorGraphConfig(
            capture_graph_walk="decode",
            single_request_inputs=ARNodeInputs(
                input_ids=torch.zeros(1, dtype=torch.long, device=device), input_seq_len=self.block,
            ),
            capture_batch_sizes=self.capture_batch_sizes,
        )]


def specs(attn: str = "toy", max_num_pages: int = 256, draft_verify: bool = True,
          kv_heads: int = HEADS, sampled_verify: bool = False,
          argmax_prefill: bool | None = None) -> list[NodeResourceSpec]:
    # ``kv_heads`` before sharding: a lockstep test gives each rank every head.
    # ``argmax_prefill``: the prefill's token is the argmax, so it draws nothing (on CPU always,
    # where the real draw has no kernel).
    kv_config = PagedKVConfig(
        num_layers=LAYERS, num_kv_heads=kv_heads, head_dim=HEAD_DIM,
        max_seq_len=MAX_POS, max_num_pages=max_num_pages, page_size=PAGE,
    )
    out = [
        KVSpec(resource_key=KV, nodes={NODE}, config=kv_config),
        PositionSpec(resource_key=POS, nodes={NODE}, config=PositionConfig(kv_cache=KV)),
        (GreedySamplerSpec if (attn == "toy" if argmax_prefill is None else argmax_prefill) else SamplerSpec)(
            resource_key=SAMPLER, nodes={NODE}, vocab_size=VOCAB,
            enable_repetion_penalty=False, draft_verify=draft_verify,
            enable_sampled_verify=sampled_verify,
        ),
    ]
    if attn == "toy":
        out.append(ToyAttentionSpec(resource_key=ATTN, nodes={NODE}))
    else:
        out.append(AttentionSpec(resource_key=ATTN, nodes={NODE}, config=AttentionConfig(
            kv_cache=KV, backend=AttnBackend(attn))))
    if draft_verify:
        out.append(DraftVerifySpec(resource_key=DV, nodes={NODE}, sampler=SAMPLER))
    return out


class _Group:
    """A rank's view of a node's lockstep instance, as the engine reads it."""

    def __init__(self, comm=None):
        from mstar.distributed.communication import CommGroup, JointGroups

        self._comm = comm or JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial())

    def get_joint_group_for_node(self, node):
        return self._comm

    def get_instance_world_size_for_node(self, node):
        return self._comm.world_size

    def get_instance_rank_for_node(self, node):
        return self._comm.rank

    def resource_needs_remote_transfer(self, nodes, local):
        return False


class ToyModel:
    """What the engine reads off a model at load."""

    def checkpoint_path(self):
        return None

    def preprocess_fingerprint(self):
        return "toy"

    def kv_labels(self):
        return {}

    def get_combined_graph_walks(self):
        return {}

    def get_graph_walk_graphs(self):
        graph = SimpleNamespace(get_nodes=lambda: [NODE])
        return {"prefill": graph, "decode": graph}


def engine(sub: ToySubmodule, device, attn: str = "toy", comm=None, warmup: bool = False,
           dtype=torch.float32, kv_dtype: torch.dtype | None = None, bare: bool = False,
           **spec_kwargs) -> Engine:
    """The toy's engine; ``bare`` loads it without the model, as a worker that has none does."""
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.engine.resources.kv.transfer import TransferEngineInfo

    sub.to(device=device, dtype=dtype)
    kv_dtype = kv_dtype or dtype
    sub.attn_dtype = kv_dtype
    eng = Engine(None, autocast_dtype=None)
    eng.load_model(
        submodules={NODE: sub}, specs=specs(attn, draft_verify=bool(sub.k), **spec_kwargs),
        parallel_groups=_Group(comm), device=device,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
        kv_cache_type=kv_dtype, model=None if bare else ToyModel(),
    )
    if warmup:
        eng.warmup()
    return eng


def layout(prompt: int, max_tokens: int, lookahead: int) -> KVReqConfig:
    return KVReqConfig(
        kv_layout={"main": StreamLayout(
            spans=(PrefixSpan(prompt, prompt, "prefill"),), growth=(("decode_loop", 1),),
            lookahead=lookahead,
        )},
        loop_caps={"decode_loop": max_tokens - 1},
    )


class Driver:
    """The worker's part, for one node: ingest, prefill, then decode steps until every
    request stops, each step through the engine's own exec and stop check."""

    def __init__(self, eng: Engine, sub: ToySubmodule, plan_ahead: bool = False):
        self.eng, self.sub = eng, sub
        self.max_tokens: dict[int, int] = {}
        self.sampling: dict[int, SamplingReqConfig] = {}
        self.emitted: dict[int, list[int]] = {}
        self.next: dict[int, torch.Tensor] = {}
        self.leases: list = []
        # pre-plan each decode step as the worker's plan thread does: after the step before it
        # committed, before anything reads its verdict, on a slot of its own
        self.plan_ahead = plan_ahead
        self.planned_ahead = 0

    def add(self, rid: int, prompt: list[int], max_tokens: int, ignore_eos: bool = False,
            seed: int = 0, **sampling):
        self.add_batch({rid: prompt}, max_tokens, ignore_eos, {rid: seed}, **sampling)

    def add_batch(self, prompts: dict[int, list[int]], max_tokens: int, ignore_eos: bool = False,
                  seeds: dict[int, int] | None = None, **sampling):
        """``add`` for several requests, their prefills in one step; ``seeds`` by request."""
        lookahead = 2 * self.sub.k + 1 if self.sub.k else 0
        for rid, prompt in prompts.items():
            config = SamplingReqConfig(**{
                "temperature": 0.0, "ignore_eos": ignore_eos, "_seed": (seeds or {}).get(rid, 0), **sampling,
            })
            self.eng.add_request(rid, {KV: layout(len(prompt), max_tokens, lookahead), SAMPLER: config})
            self.sampling[rid] = config
            self.max_tokens[rid] = max_tokens
            self.emitted[rid] = []
        out = self._step("prefill", {rid: torch.tensor(prompt) for rid, prompt in prompts.items()})
        for rid in prompts:
            self.emitted[rid] += out[rid]["new_token"][0].reshape(-1).tolist()
            self.next[rid] = out[rid]["text_inputs"][0]

    def _step(self, walk: str, inputs: dict[int, torch.Tensor]):
        rids = list(inputs)
        batch = ExecutingBatch(
            node_name=NODE,
            per_request_info={rid: CurrentForwardPassInfo(
                request_id=f"r{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0,
                random_seed=0, max_tokens=self.max_tokens[rid],
                resource_configs={SAMPLER: self.sampling[rid]},
            ) for rid in rids},
            step_context=StepContext(
                request_ids=tuple(rids), graph_walk=walk, slot=None if self.plan_ahead else 0, capture=False,
            ),
            per_request_input_tensors={rid: {"text_inputs": [inputs[rid]]} for rid in rids},
        )
        if self.plan_ahead and walk == "decode" and self.eng.reserve_replay_slot(batch) is not None:
            self.planned_ahead += bool(self.eng.pre_plan_for_batch(batch))
        self.eng.prepare_inputs(batch)
        outputs = self.eng.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        self.leases.append(batch.step_context.slot_lease)
        self.stops = self.eng.check_stop_for_batch(batch, outputs.per_rid_outputs)
        if walk == "decode" and not self.sub.k:
            # plain decode: the toy's own stop, as a model's check_stop would say it
            self.stops = {
                rid: {"decode_loop"} for rid in rids
                if outputs.get(rid)["new_token"][0].item() == STOP
                or len(self.emitted[rid]) + 1 >= self.max_tokens[rid]
            }
        return outputs.per_rid_outputs

    def decode(self, rids: list[int]) -> None:
        """One decode step of ``rids``; requests that stop leave ``self.live``."""
        out = self._step("decode", {rid: self.next[rid] for rid in rids})
        for rid in rids:
            self.emitted[rid] += out[rid]["new_token"][0].reshape(-1).tolist()
            self.next[rid] = out[rid]["text_inputs"][0]

    def run(self, rids: list[int], max_steps: int = 200, batched: bool = True) -> dict[int, list[int]]:
        # plain decode never runs a request whose prefill already ended it; a verify step of
        # one emits nothing and stops, which is the engine's to say
        live = [rid for rid in rids if self.sub.k or not self._ended(rid)]
        for _ in range(max_steps):
            if not live:
                break
            groups = [live] if batched else [[rid] for rid in live]
            stopped = set()
            for group in groups:
                self.decode(group)
                stopped |= set(self.stops)
            live = [rid for rid in live if rid not in stopped]
        assert not live, f"requests {live} never stopped"
        return {rid: self.emitted[rid] for rid in rids}

    def _ended(self, rid: int) -> bool:
        tokens = self.emitted[rid]
        return len(tokens) >= self.max_tokens[rid] or (tokens[-1] == STOP and not self.sampling[rid].ignore_eos)


def reference(prompt: list[int], max_tokens: int, seed: int = SEED, attn_scale: float = 0.6,
              ignore_eos: bool = False, margins: list | None = None) -> list[int]:
    """Plain greedy decode of the toy's trunk on CPU in float64, no engine: the stream every
    engine run must reproduce. Stops after a stop token or at max_tokens. ``margins`` gets
    each step's gap between the top two logits: a bf16 run can only match where it is wide."""
    sub = ToySubmodule(k=0, seed=seed, attn_scale=attn_scale).double()
    ids = list(prompt)
    out = []
    with torch.no_grad():
        while len(out) < max_tokens:
            logits = next_logits(ids, sub=sub)
            if margins is not None:
                top = logits.topk(2).values
                margins.append(float(top[0] - top[1]))
            token = int(logits.argmax())
            out.append(token)
            ids.append(token)
            if token == STOP and not ignore_eos:
                break
    return out


def next_logits(ids: list[int], seed: int = SEED, attn_scale: float = 0.6, sub=None) -> torch.Tensor:
    """The toy trunk's float64 logits for the token after ``ids``, no engine."""
    if sub is None:
        sub = ToySubmodule(k=0, seed=seed, attn_scale=attn_scale).double()
    with torch.no_grad():
        x = sub.embed(torch.tensor(ids)) + sub.pos_emb(torch.arange(len(ids)))
        for block in sub.blocks:
            q, k, v = block.qkv(x).view(len(ids), 3, HEADS, HEAD_DIM).unbind(1)
            scores = torch.einsum("qhd,khd->hqk", q, k) * HEAD_DIM ** -0.5
            mask = torch.ones(len(ids), len(ids), dtype=torch.bool).tril()
            attn = scores.masked_fill(~mask, float("-inf")).softmax(-1)
            out_ = torch.einsum("hqk,khd->qhd", attn, v).reshape(len(ids), HIDDEN)
            x = block.mlp(x + block.attn_scale * block.o(out_))
        return sub.lm_head(x[-1])
