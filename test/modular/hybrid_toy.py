"""A tiny hybrid model, two GDN layers around one attention layer (as Qwen3.5
interleaves them), verifying drafts on the real `Engine` with real recurrent,
linear-attention, KV, position and sampler resources, on the CPU through
`gdn_cpu`'s and `qwen3_cpu`'s kernel stand-ins.

Its drafter proposes a reference continuation, spoiled where a test says, so a
step accepts any count the test chooses; the output must still be plain greedy
decoding's, which a dense, engine-free run of the same weights gives
(`reference_greedy`): every GDN state recomputed from zeros over the whole
sequence, so a rejected token left in a slot would show.

Everything runs in float32 (the GDN pool's state dtype), so a parity check
uses the margin rule: where two runs differ, the reference's top-2 margin there
must be a near-tie.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from spec_toy import Driver, _batch
from torch import nn

from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine
from mstar.engine.resources.attn.config import AttentionConfig, AttentionStep
from mstar.engine.resources.draft.config import DRAFT_METHODS, DraftConfig, DraftMethod
from mstar.engine.resources.draft.drafter import (
    ChunkBoundary,
    Drafter,
    DrafterDecl,
    DrafterNeeds,
    VerifyNode,
    drafter_resources,
)
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.linear_attn.config import (
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
)
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.recurrent.config import (
    DeltaNetGeometry,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.model.base import KVLabel, PrefixKeys
from mstar.model.components.attention import Attention
from mstar.model.components.linear_attn import GatedDeltaNet, GDNProjLayout
from mstar.model.components.norm import RMSNorm
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    DeviceLoopback,
    TokenStop,
)

NODE = "LLM"
VOCAB = 64
STOP = 63
PAGE = 4
# the stack: kind of each layer, in order
LAYERS = ("gdn", "attn", "gdn")


@dataclass(frozen=True)
class Shapes:
    """Where the toy runs, and in what shapes."""
    hidden: int = 32
    geometry: DeltaNetGeometry = DeltaNetGeometry(
        num_k_heads=1, num_v_heads=2, head_k_dim=8, head_v_dim=8, conv_kernel_size=4,
    )
    attn_heads: int = 2
    attn_head_dim: int = 16
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    page: int = PAGE


CPU_SHAPES = Shapes()
# what FlashInfer's and FA3's kernels take: head dims of 128, a bf16 model on an fp32 state
CUDA_SHAPES = Shapes(
    hidden=256, geometry=DeltaNetGeometry(num_k_heads=1, num_v_heads=2, head_k_dim=128, head_v_dim=128,
                                          conv_kernel_size=4),
    attn_heads=2, attn_head_dim=128, dtype=torch.bfloat16, device="cuda", page=16,
)


class _GDNMix:
    """The pool's blocks handed to the linear-attention resource by layer index. Main's shared `GatedDeltaNet`
    reaches its state only through a state handle, and the toy declares specs, not state (`declare_state`), so
    its layer calls the resource the way the shared layer did before the state contract."""

    def __init__(self, pool, attn):
        self.pool, self.attn, self._layer = pool, attn, 0

    @torch.compiler.disable
    def bind_step(self, label: str) -> None:
        self.attn.set_default_label(label)

    @torch.compiler.disable
    def set_layer_idx(self, layer: int) -> None:
        """The layer's index among the recurrent layers, not the stack's."""
        self._layer = layer
        self.attn.set_default_layer_idx(layer)

    @torch.compiler.disable
    def conv(self, x, weight, bias=None):
        return self.attn.run_conv(x, conv_layer=self.pool.block("conv", self._layer), weight=weight, bias=bias)

    @torch.compiler.disable
    def __call__(self, q, k, v, a, b, a_log, dt_bias):
        return self.attn.run(
            q, k, v, a, b, state_layer=self.pool.block("state", self._layer), a_log=a_log, dt_bias=dt_bias,
        )


class ToyGatedDeltaNet(GatedDeltaNet):
    """Main's layer, its weights and projections, run on the toy's resources (`_GDNMix`)."""

    def __init__(self, *, linear_attn_key: str, state_key: str, **kwargs):
        super().__init__(state_key=state_key, **kwargs)
        self._linear_attn_key = linear_attn_key
        self.mix = None

    def bind_resources(self, resources: dict) -> None:
        self.mix = _GDNMix(resources.get(self._state_key), resources.get(self._linear_attn_key))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        project = self._project_split if self.layout is GDNProjLayout.SPLIT else self._project_fused
        qkv, z, a, b = project(hidden_states)
        qkv = self.mix.conv(qkv, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias)
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(num_tokens, self.num_k_heads, self.head_k_dim)
        k = k.view(num_tokens, self.num_k_heads, self.head_k_dim)
        v = v.view(num_tokens, self.num_v_heads, self.head_v_dim)
        core = self.mix(q, k, v, a, b, self.A_log, self.dt_bias)
        return self.out_proj(self.norm(core, z).reshape(num_tokens, self.value_dim))


class Block(nn.Module):
    """A pre-norm residual mixer: ``x + mixer(norm(x))``."""

    def __init__(self, kind: str, shapes: Shapes):
        super().__init__()
        self.kind = kind
        self.norm = RMSNorm(shapes.hidden)
        if kind == "gdn":
            g = shapes.geometry
            self.mixer = ToyGatedDeltaNet(
                hidden_size=shapes.hidden, num_k_heads=g.num_k_heads, num_v_heads=g.num_v_heads,
                head_k_dim=g.head_k_dim, head_v_dim=g.head_v_dim, conv_kernel_size=g.conv_kernel_size,
                linear_attn_key="linear_attn", state_key="gdn_state",
            )
        else:
            self.mixer = Attention(
                hidden_size=shapes.hidden, num_heads=shapes.attn_heads, num_kv_heads=1,
                head_dim=shapes.attn_head_dim, rope_theta=10_000.0, attn_key="attn", kv_key="kv", pos_key="rope",
            )


class TinyHybrid(nn.Module):
    def __init__(self, seed: int = 0, shapes: Shapes = CPU_SHAPES):
        super().__init__()
        hidden = shapes.hidden
        self.embed = nn.Embedding(VOCAB, hidden)
        self.blocks = nn.ModuleList(Block(kind, shapes) for kind in LAYERS)
        self.norm = RMSNorm(hidden)
        self.head = nn.Linear(hidden, VOCAB, bias=False)
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name.endswith("norm.weight"):
                    p.fill_(1.0)
                elif name.endswith(("A_log", "dt_bias")):
                    p.copy_(torch.randn(p.shape, generator=gen) * 0.3)
                else:
                    p.copy_(torch.randn(p.shape, generator=gen) / max(p.shape[-1], 1) ** 0.5)
        # GatedDeltaNet keeps its gates and norm fp32 through the cast
        self.to(device=shapes.device, dtype=shapes.dtype)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Logits of every token."""
        x = self.embed(ids)
        gdn = attn = 0
        for block in self.blocks:
            mixer = block.mixer
            if block.kind == "gdn":
                mixer.mix.bind_step("main")
                mixer.mix.set_layer_idx(gdn)
                gdn += 1
            else:
                mixer.attend.bind_step("main")
                mixer.attend.set_layer_idx(attn)
                attn += 1
            x = x + mixer(block.norm(x))
        return self.head(self.norm(x))


class ForcedDrafter(Drafter):
    """Drafts each row's ``reference`` continuation from its anchor's position,
    spoiling draft ``j`` where ``spoil(j)``."""

    needs = DrafterNeeds(chunk_boundary=ChunkBoundary.NONE)

    def __init__(self, reference: dict[int, list[int]], spoil=lambda j: False):
        self.reference, self.spoil = reference, spoil

    def propose(self, inputs) -> torch.Tensor:
        k, n = inputs.num_draft_tokens, len(inputs.request_ids)
        anchors = inputs.resources["rope"].pos_ids("main").view(n, k + 1)[:, 0].tolist()
        rows = []
        for rid, at in zip(inputs.request_ids, anchors, strict=True):
            seq = self.reference[rid]
            row = [seq[at + 1 + j] if at + 1 + j < len(seq) else 0 for j in range(k)]
            rows.append([(t + 1) % VOCAB if self.spoil(j) else t for j, t in enumerate(row)])
        return torch.tensor(rows, dtype=torch.long)


class TableDrafter(Drafter):
    """Drafts from a device table keyed by the anchor's position and token, built
    from reference continuations: pure tensor ops, so a captured verify step
    replays it (a host-side drafter would run once, at capture)."""

    needs = DrafterNeeds(chunk_boundary=ChunkBoundary.NONE)

    def __init__(self, reference: dict[int, list[int]], k: int, spoil=lambda j: False, device="cpu",
                 positions: int = 256, prompts: dict[int, list[int]] | None = None):
        """``prompts``: where each reference's anchors start (its prompt's
        positions are never anchors, and would collide with another row's)."""
        table = torch.zeros(positions, VOCAB, k, dtype=torch.long)
        owner: dict[tuple[int, int], list[int]] = {}
        for rid, seq in reference.items():
            start = len(prompts[rid]) if prompts is not None else 0
            for at in range(start, len(seq) - 1):
                row = [seq[at + 1 + j] if at + 1 + j < len(seq) else 0 for j in range(k)]
                drafts = [(t + 1) % VOCAB if spoil(j) else t for j, t in enumerate(row)]
                if prompts is not None and owner.setdefault((at, seq[at]), drafts) != drafts:
                    raise ValueError(f"two references draft differently from token {seq[at]} at {at}")
                table[at, seq[at]] = torch.tensor(drafts)
        self.table = table.to(device)

    def propose(self, inputs) -> torch.Tensor:
        k = inputs.num_draft_tokens
        anchor = inputs.input_ids.reshape(-1).long()
        n = anchor.shape[0]
        at = inputs.resources["rope"].pos_ids("main")[: n * (k + 1)].view(n, k + 1)[:, 0].long()
        return self.table[at.clamp(max=self.table.shape[0] - 1), anchor]


class HybridLM(ARNodeSubmodule):
    """Verifies ``k`` drafts a decode step, or with no drafter decodes plainly,
    one token a step: the same build's spec-off run."""
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4]
    PREFILL_TOKEN_BUCKETS = [16, 32]

    def __init__(self, net: TinyHybrid, k: int, drafter, accelerator_graph: bool = False):
        super().__init__()
        self.net, self.k, self.drafter = net, k, drafter
        self.verifies = drafter is not None
        self.block = k + 1 if self.verifies else 1
        self.accelerator_graph = accelerator_graph
        self.token_stop = TokenStop(
            loop="decode_loop", buffer="new_token", stop_ids=frozenset({STOP}), sampler="sampler",
            walk="decode", prefill_walks=frozenset({"prefill"}),
        )
        self.device_loopback = DeviceLoopback(walk="decode", signal="ids", sampler="sampler", buffer="new_token")

    def prepare_inputs(self, graph_walk, fwd_info, inputs, has_loopback=False, **kwargs):
        if (row := self.uniform_row_inputs(graph_walk, has_loopback)) is not None:
            return row
        # the toy's driver routes host tensors
        ids = inputs["ids"][0].reshape(-1).to(self.net.embed.weight.device)
        if graph_walk == "prefill":
            return ARNodeInputs(input_ids=ids, input_seq_len=ids.numel())
        return ARNodeInputs(input_ids=ids[-1:], input_seq_len=self.block)

    def can_batch(self, batch, model_inputs):
        return True

    def resident_walks(self) -> frozenset[str]:
        # decode's preprocess reads only the loop-back tokens; tables need a capture
        return frozenset({"decode"}) if self.accelerator_graph else frozenset()

    def get_accelerator_graph_configs(self, device, tp_world_size: int = 1):
        from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig, PackedAcceleratorGraphConfig

        if not self.accelerator_graph:
            return []

        def ids(n: int) -> ARNodeInputs:
            return ARNodeInputs(input_ids=torch.zeros(n, dtype=torch.long, device=device), input_seq_len=n)

        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device), input_seq_len=self.block,
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill", capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                make_node_input=ids, capture_batch_sizes=[1, 2],
            ),
        ]

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs):
        verify = graph_walk == "decode" and self.verifies
        segments = [Segment(rid, "main", inp.input_seq_len) for rid, inp in zip(request_ids, inputs, strict=True)]
        one = [Segment(rid, "main", 1) for rid in request_ids] if verify else None
        return SubmoduleStep(
            steps={
                "kv": KVStep(), "rope": PositionStep(), "attn": AttentionStep(causal=True, verify=verify),
                "gdn_state": RecurrentStep(), "linear_attn": LinearAttnStep(),
                "sampler": SamplerStep(segments=one, apply_penalty=False),
            },
            segments=segments,
        )

    def preprocess(self, graph_walk, engine_inputs, inputs):
        if graph_walk == "prefill":
            return {"ids": torch.cat([inp.input_ids for inp in inputs])}
        return {"ids": self.loopback_inputs(engine_inputs, inputs)["input_ids"]}

    def forward(self, graph_walk, engine_inputs, ids, **kwargs):
        return self.forward_batched(graph_walk, engine_inputs, ids, **kwargs)

    def forward_batched(self, graph_walk, engine_inputs, ids, **kwargs):
        rids = engine_inputs.request_ids
        if graph_walk == "prefill":
            hidden = self.net(ids)
            # each prompt's last row, by the attention resource's plan (a captured step's included)
            last = engine_inputs.resources["attn"].select_last_hidden(hidden, label="main")
            # the prompt step is greedy here
            tokens = last.argmax(-1)
            return BatchedModelOutput(
                per_rid_outputs={rid: {"new_token": [tokens[i: i + 1]]} for i, rid in enumerate(rids)},
            )
        n = len(rids)
        if not self.verifies:
            tokens = self._draw(engine_inputs, self.net(ids.view(n)))
            return BatchedModelOutput(row_outputs={"new_token": tokens}, check_stop_buffers={"new_token": tokens})
        result = self.draft_verify_step(engine_inputs, ids.view(n), self.net)
        return self.verify_output(result)

    @torch.compiler.disable
    def _draw(self, engine_inputs, logits: torch.Tensor) -> torch.Tensor:
        return engine_inputs.resources["sampler"].sample(list(engine_inputs.request_ids), logits=logits)


# the toy drafters load no checkpoint: a method of the toy's own, as a model serving another drafter passes
SCRIPTED = {**DRAFT_METHODS, "scripted": DraftMethod("scripted", needs_model=False)}


class HybridModel:
    """What `Engine.load_model` and validation read of a model."""

    def __init__(self, k: int, mode: str = "synchronous", stop=None, verifies: bool = True, prefix: bool = False):
        self.config = DraftConfig.parse({"method": "scripted", "num_draft_tokens": k, "mode": mode}, SCRIPTED)
        self._stop, self._verifies, self._prefix = stop, verifies, prefix

    def verify_nodes(self):
        if not self._verifies:
            return []
        loopback = DeviceLoopback(walk="decode", signal="ids", sampler="sampler", buffer="new_token")
        return [VerifyNode(NODE, "decode", self.config, ForcedDrafter.needs, token_stop=self._stop,
                           device_loopback=loopback)]

    def checkpoint_path(self):
        return None

    def declare_state(self):
        # the toy builds its resources from specs, as a model off the state contract does
        return []

    def preprocess_fingerprint(self):
        return ""

    def get_graph_walk_graphs(self):
        return {walk: SimpleNamespace(get_nodes=lambda: [NODE]) for walk in ("prefill", "decode")}

    def kv_labels(self):
        # with a prefix cache, the prompt keys its pages (a verifying decode walk keys none)
        keys = PrefixKeys(tensor="ids", walks=("prefill", "decode"), growth=True) if self._prefix else None
        return {"kv": {"main": KVLabel(growth=(("decode", "decode_loop"),), keys=keys)}}

    def chunking_policy_of(self):
        return {}

    def rebuild_policies(self):
        # no node rebuilds a dropped request (`Engine._check_rebuild_policies`)
        return {}

    def __getattr__(self, name):
        if name.startswith("get_") or name.endswith("_of"):
            return lambda *a, **kw: {}
        raise AttributeError(name)


class _HybridDecl(DrafterDecl):
    """The toy's drafters as the node declares them: no resources of their own."""
    needs = DrafterNeeds(chunk_boundary=ChunkBoundary.NONE)

    def node_resources(self, nodes):
        return []


def specs(k: int, shapes: Shapes = CPU_SHAPES, verifies: bool = True, prefix: bool = False, offload: bool = False):
    """The node's specs as a model declares them, through `drafter_resources` when it verifies,
    which holds the KV slack past the 256-token limit and makes the sampler a verifying one."""
    from mstar.engine.resources.attn.config import AttentionSpec

    attn_layers = sum(kind == "attn" for kind in LAYERS)
    if shapes.device == "cpu":
        from qwen3_cpu import CPUAttentionSpec as AttentionSpec  # noqa: N814
    declared = [
        KVSpec("kv", {NODE}, config=PagedKVConfig(
            num_layers=attn_layers, num_kv_heads=1, num_qo_heads=shapes.attn_heads, head_dim=shapes.attn_head_dim,
            max_seq_len=256, max_num_pages=256, page_size=shapes.page,
            prefix_cache_salt="toy" if prefix else None,
            # host pages a request's KV offloads to (`Engine.offload_request`)
            cpu_offload_pages=256 if offload else 0,
        )),
        AttentionSpec("attn", {NODE}, AttentionConfig(kv_cache="kv")),
        PositionSpec("rope", {NODE}, PositionConfig(kv_cache="kv")),
        RecurrentStateSpec("gdn_state", {NODE}, RecurrentStateConfig(
            num_layers=sum(kind == "gdn" for kind in LAYERS),
            blocks=shapes.geometry.to_blocks(
                state_dtype=torch.float32, conv_dtype=torch.float32 if shapes.device == "cpu" else shapes.dtype,
            ),
            max_slots=16,
            # prompt snapshots at the KV cache's page boundaries, which a prefix hit restores
            prefix_slots=8 if prefix else 0, kv_cache="kv" if prefix else None,
        )),
        LinearAttnSpec("linear_attn", {NODE}, LinearAttnConfig(recurrent_state="gdn_state",
                                                               variant=LinearAttnVariant.GDN)),
        SamplerSpec("sampler", {NODE}, vocab_size=VOCAB, enable_repetition_penalty=False),
    ]
    if not verifies:
        return declared
    config = DraftConfig.parse({"method": "scripted", "num_draft_tokens": k}, SCRIPTED)
    return drafter_resources(declared, _HybridDecl(), {NODE}, config)


def engine(k: int, drafter, mode: str = "synchronous", seed: int = 0, shapes: Shapes = CPU_SHAPES,
           captured: bool = False, prefix: bool = False, offload: bool = False) -> Engine:
    """The toy on the real Engine; ``drafter=None`` decodes plainly (``k`` unused).
    ``prefix``: a prefix cache over the KV pages and the pool's prompt snapshots
    (requests must then come in through `add_keyed`)."""
    verifies = drafter is not None
    sub = HybridLM(TinyHybrid(seed, shapes), k, drafter, accelerator_graph=captured)
    cpu = shapes.device == "cpu"
    eng = Engine(graph_runtime=None, autocast_dtype=None if cpu else shapes.dtype)
    eng.load_model(
        {NODE: sub}, specs(k, shapes, verifies, prefix, offload), WorkerParallelGroups(num_workers=1, global_rank=0),
        _indexed(shapes.device), transfer_engine_info=None, kv_cache_type=shapes.dtype,
        model=HybridModel(k, mode, sub.token_stop, verifies, prefix),
    )
    if captured:
        eng.warmup()
    else:
        eng._resources["sampler"].build_accelerator_graph_buffers([], 8, 256)
    return eng


def add_keyed(driver: Driver, rid: int, prompt: list[int], max_tokens: int, k: int, page: int) -> int:
    """A request whose prompt pages are keyed (as a model's ``process_prompt``
    declares them), looked up in the prefix cache as the worker does, then its
    prompt step. Returns the prefix tokens it hit."""
    from spec_toy import _take_sampler_row

    from mstar.engine.resources.kv.config import KVReqConfig, PrefixSpan, StreamLayout
    from mstar.engine.resources.kv.keys import chain
    from mstar.engine.resources.sampler.config import SamplingReqConfig
    from mstar.model.submodule_base import ARNodeInputs

    eng = driver.eng
    whole = len(prompt) // page
    kv = KVReqConfig(
        prefix_keys={"main": chain([prompt[at: at + page] for at in range(0, whole * page, page)])},
        prefix_tail={"main": prompt[whole * page:]},
        kv_layout={"main": StreamLayout(
            spans=(PrefixSpan(len(prompt), len(prompt), "prefill"),), growth=(("decode_loop", 1),),
            scratch=2 * (k + 1), keyed_spans=1,
        )},
        loop_caps={"decode_loop": max_tokens},
    )
    sampling = SamplingReqConfig(temperature=0)
    sampling.apply_conductor_config(seed=0)
    eng.add_request(rid, {"kv": kv, "sampler": sampling})
    _take_sampler_row(eng, rid)
    inputs = ARNodeInputs(input_seq_len=len(prompt))
    # every request asks: it also tells the pool which streams the cache keys
    prefix = eng._runner.resolve_cached_prefix(rid, NODE, "prefill", inputs)
    if prefix is not None:
        eng._runner.apply_cached_prefix(rid, NODE, "prefill", inputs, prefix)
    driver.out[rid] = []
    driver._max = {**getattr(driver, "_max", {}), rid: max_tokens}
    batch = _batch("prefill", {rid: torch.tensor(prompt)}, driver._max)
    eng.prepare_inputs(batch)
    out = eng.exec_and_postprocess(batch)
    assert batch.admit_error is None, batch.admit_error
    driver.out[rid].append(int(out.per_rid_outputs[rid]["new_token"][0][0]))
    return 0 if prefix is None else prefix.tokens


class PlainDriver(Driver):
    """`spec_toy.Driver` for a plain run: one token a decode step."""

    def step(self, rids):
        batch = _batch("decode", {rid: torch.tensor([self.out[rid][-1]]) for rid in rids}, self._max)
        self.eng.prepare_inputs(batch)
        outputs = self.eng.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        tokens = outputs.check_stop_buffers["new_token"].reshape(-1).cpu().tolist()
        stopped = {}
        for i, rid in enumerate(rids):
            self.out[rid].append(tokens[i])
            stopped[rid] = tokens[i] == STOP or len(self.out[rid]) >= self._max[rid]
        return stopped


# ── the reference: no engine, every state from zeros ──


class _DenseGDN:
    """The linear-attention resource for a whole sequence from zero state."""

    def __init__(self):
        self.default_label = "main"

    def set_default_label(self, label):
        pass

    def set_default_layer_idx(self, idx):
        pass

    def run_conv(self, x, conv_layer, weight, bias=None, activation="silu", label=None):
        width = weight.shape[1] - 1
        seq = torch.cat([x.new_zeros(width, x.shape[1]), x])
        y = sum(weight[:, j] * seq[j: j + x.shape[0]] for j in range(width + 1))
        return F.silu(y + (0 if bias is None else bias))

    def run(self, q, k, v, a, b, state_layer, a_log, dt_bias, label=None):
        import gdn_cpu

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        alpha = torch.exp(-torch.exp(a_log) * F.softplus(a + dt_bias))
        beta = torch.sigmoid(b)
        state = q.new_zeros(v.shape[1], v.shape[2], k.shape[2])
        out, _ = gdn_cpu.delta_rule(q, k, v, alpha, beta, state, k.shape[-1] ** -0.5)
        return out


class _NoPool:
    def block(self, name, layer):
        return None


def dense_net(seed: int = 0, like: TinyHybrid | None = None) -> TinyHybrid:
    """`TinyHybrid` with no engine, every state recomputed from zeros, on the
    CPU in float32; ``like``'s weights when given (a device model's)."""
    from qwen3_cpu import _DenseAttention, _DensePositions, copy_weights

    net = TinyHybrid(seed).eval()
    if like is not None:
        net = TinyHybrid(seed, Shapes(**{**CUDA_SHAPES.__dict__, "dtype": torch.float32, "device": "cpu"})).eval()
        copy_weights(net, like)
    gdn, attn, pos = _DenseGDN(), _DenseAttention(), _DensePositions()
    for block in net.blocks:
        if block.kind == "gdn":
            block.mixer.bind_resources({"linear_attn": gdn, "gdn_state": _NoPool()})
        else:
            block.mixer.bind_resources({"attn": attn, "kv": attn, "rope": pos})
    return net


@torch.no_grad()
def margin_at(seq: list[int], like: TinyHybrid | None = None) -> float:
    """The dense reference's top-2 log-prob margin for the token after ``seq``."""
    top = dense_net(like=like)(torch.tensor(seq))[-1].log_softmax(-1).topk(2).values
    return float(top[0] - top[1])


def reference_greedy(prompt: list[int], max_tokens: int, seed: int = 0, margins: list | None = None,
                     like: TinyHybrid | None = None) -> list[int]:
    """Plain greedy decoding of `dense_net`."""
    net = dense_net(seed, like)
    seq, out = list(prompt), []
    with torch.no_grad():
        while len(out) < max_tokens:
            logp = net(torch.tensor(seq))[-1].log_softmax(-1)
            top = logp.topk(2)
            if margins is not None:
                margins.append(float(top.values[0] - top.values[1]))
            token = int(top.indices[0])
            out.append(token)
            seq.append(token)
            if token == STOP:
                break
    return out


def _indexed(device) -> torch.device:
    """The device as a worker passes it: a GPU with its index (capture sets the device by index)."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device
