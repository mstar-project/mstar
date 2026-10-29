"""Every model serving per-request LoRA through ``module_lora_spec``, on CPU.

Per model (``lora_module_helpers.STACKS``): its node, bound by the generic path and
addressed by its own ``declare_step``, computes per row what it computes with that row's
adapter merged into the checkpoint (loaded by the model's own loader), in a batch of two
adapters and the base model, at simulated TP; its declaration binds at the released
sizes; Hub adapters are served or refused by name. Then Qwen3.5's own cases: the vision
node and the tied head. Only kernels holding no adapter are stubbed.
"""

from __future__ import annotations

import dataclasses
import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

pytest.importorskip("safetensors")
import lora_module_helpers as H  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.engine.resources.lora import LORA_RESOURCE_KEY, LoraReqConfig, LoraSpec  # noqa: E402
from mstar.engine.resources.lora.checkpoint import normalize_lora_key, parse_lora_key  # noqa: E402
from mstar.engine.resources.lora.serving import lora_spec  # noqa: E402
from mstar.engine.resources.lora.store import inspect_keys  # noqa: E402
from mstar.engine.resources.spec import instance_key  # noqa: E402
from mstar.engine.resources.step import PIECEWISE_WALK, BucketKey, Segment, SlotLease, StepContext  # noqa: E402
from mstar.model.components.lora.capable import make_lora_capable  # noqa: E402
from mstar.model.components.lora.modules import _check  # noqa: E402
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM  # noqa: E402
from mstar.model.qwen3_5.components.vision import Qwen3_5VisionModel  # noqa: E402
from mstar.model.qwen3_5.config import VISION_ATTN, Qwen3_5Config, Qwen3_5VisionConfig  # noqa: E402
from mstar.model.qwen3_5.lora import qwen3_5_lora_spec  # noqa: E402
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel  # noqa: E402
from mstar.model.qwen3_5.submodules import VISION_BLOCK_LOOP, VisionEncoderSubmodule  # noqa: E402
from mstar.model.submodule_base import declared_step  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
# the vision node's pool, as the engine keys it (`LoraSpec.for_node`)
VISION_POOL = instance_key(LORA_RESOURCE_KEY, "vision_encoder")
# two adapters, the base model, and a1 again (sharing its slot)
ROWS = [("r0", "a1", 3), ("r1", None, 2), ("r2", "a2", 4), ("r3", "a1", 2)]
# ranks set up one at a time: the Qwen3.5 build sets the process-wide default dtype
_SETUP = threading.Lock()


def _bound(module) -> bool:
    return getattr(module, "lora_delta", None) is not None or getattr(module, "lora_deferred", None) is not None


def _unbound(stack, c, weights):
    sub = stack.build(c, CommGroup.trivial(), weights)
    sub.bind_node_resources({})
    return sub


@pytest.mark.parametrize("name,world", [(n, w) for n, f in H.STACKS.items() for w in f().worlds])
def test_mixed_batch_matches_each_rows_merged_adapter(name, world, tmp_path):
    stack = H.STACKS[name]()
    c = stack.config()
    p1, r1 = H.make_adapter(tmp_path / "a1.safetensors", stack.a1(c), 4, 8.0, 100, stack.embedding, stack.a1_rename)
    p2, r2 = H.make_adapter(tmp_path / "a2.safetensors", stack.a2(c), 2, None, 200, rename=stack.a2_rename)
    spec = H.serve(stack.lora_spec(c), {"a1": p1, "a2": p2})
    base = stack.weights(c)
    rows = [stack.inputs(c, n, i) for i, (_, _, n) in enumerate(ROWS)]
    rids = [rid for rid, _, _ in ROWS]

    def rank_run(rank, group):
        with _SETUP:
            sub = stack.build(c, group, base)
            lora_pool = H.pool(spec, rank, world)
            sub.bind_node_resources({spec.resource_key: lora_pool})
            assert sum(map(_bound, sub.modules())) == len(spec.module_map.bindings)
            H.admit_declared(lora_pool, stack.declare(sub, rids, rows), ROWS)
        with torch.no_grad(), H.compute(lora_pool):
            return stack.run(sub, rows)

    results = H.run_ranks(world, rank_run)
    plain = _unbound(stack, c, base)
    # control: unbound, the node declares no LoRA rows and holds no delta
    assert spec.resource_key not in stack.declare(plain, rids, rows).steps
    assert not any(map(_bound, plain.modules()))
    adapted = {a: _unbound(stack, c, H.merged(base, r, stack.embedding)) for a, r in (("a1", r1), ("a2", r2))}
    for i, (_, adapter, _) in enumerate(ROWS):
        with torch.no_grad():
            want = stack.run(adapted.get(adapter, plain), [rows[i]])[0]
            unadapted = stack.run(plain, [rows[i]])[0]
        for out in results:  # every rank holds the whole output
            for got, expected in zip(out[i], want, strict=True):
                torch.testing.assert_close(got, expected, rtol=1e-9, atol=1e-9)
        if adapter is not None:  # control: the adapter moves its row far past the tolerance
            assert (want[0] - unadapted[0]).abs().max() > 1e-3


# --- the released sizes ------------------------------------------------------------------------

QWEN3_5_SIZES = json.loads((FIXTURES / "qwen3_5_configs.json").read_text())["sizes"]


def _released(name: str, size: str | None = None):
    """The model's LoRA spec and config at its released size (Qwen3.5: ``size``), and per
    node the spec reaches, in the spec's order, ``(TP group -> the root it binds in,
    whether that root is TP)``."""
    if name != "qwen3_5":
        model, root = H.STACKS[name]().released()
        return lora_spec(model.get_node_resources()), model.config, [(lambda g: root(model.config, g), True)]
    entry = QWEN3_5_SIZES[size]
    text = Qwen3_5Config(**entry["text_config"], tie_word_embeddings=entry["tie_word_embeddings"])
    vision = Qwen3_5VisionConfig(**entry["vision_config"])
    # the model's own declaration; its vision attention reads only these two processor fields
    processor = SimpleNamespace(size={"shortest_edge": 65536}, patch_size=vision.patch_size)
    model = H.model_object(Qwen3_5DenseModel, text, vision_config=vision, _image_processor=processor)
    return lora_spec(model.get_node_resources()), text, [
        (lambda g: Qwen3_5ForCausalLM(text, g), True), (lambda g: Qwen3_5VisionModel(vision), False),
    ]


def _shipped() -> list[tuple[str, int]]:
    """Each Qwen3.5 size the fixture holds, at every LLM TP degree a serving config ships."""
    out = set()
    for path in (Path(__file__).resolve().parents[2] / "configs").glob("qwen3_5_*.yaml"):
        config = yaml.safe_load(path.read_text())
        size = config["model"].removeprefix("qwen3_5_").upper()
        llm = next(g for g in config["node_groups"] if "LLM" in g["node_names"])
        if size in QWEN3_5_SIZES:
            out.add((size, int(llm.get("tp_size", 1))))
    return sorted(out)


@pytest.mark.parametrize("name,size,world", [
    *((n, None, w) for n, f in H.STACKS.items() if f().released for w in (1, 2, 4)),
    *(("qwen3_5", size, w) for size, w in _shipped()),
])
def test_every_placed_module_binds_at_the_released_size(name, size, world):
    """What generic binding checks on each rank: the module is the declared target, and
    its rank-local shape is the pool's shard of it (whole, on a root that is not TP)."""
    spec, config, roots = _released(name, size)
    nodes = list(dict.fromkeys(t.node or next(iter(spec.nodes)) for t in spec.targets.values()))
    for rank in sorted({0, world - 1}):
        group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(world)))
        for node, (build, tp) in zip(nodes, roots, strict=True):
            with torch.device("meta"):
                root = build(group if tp else CommGroup.trivial())
            for path, ref in spec.module_map.part(node).bindings.items():
                target = spec.targets[ref.target]
                _check(make_lora_capable(root.get_submodule(path)), target, target.shard(world if tp else 1),
                       path, spec.resource_key)
            if isinstance(root, Qwen3_5ForCausalLM):
                # the delta-net's slices are this rank's [q|k|v|z|a+pad|b]
                q, k, v, z, a, pad, b = root.model.layers[config.linear_layer_indices[0]].self_attn.in_proj_blocks
                assert spec.targets["in_proj_fused"].shard(world).out_slices == (q, k, v, z, a + pad, b)


# --- adapters on the Hub ---------------------------------------------------------------------------

# headers (key -> shape) read off the Hub, a few per key layout and size; `refused` is a
# substring of the refusal an adapter must get, `served_without` what it is refused for
HUB = {m: json.loads((FIXTURES / f"{m}_lora_hub_keys.json").read_text())["adapters"] for m in ("qwen3_5", "orpheus")}
HUB["qwen3_omni_thinker"] = {"moe-experts": {  # packed tensors, run by the fused kernel
    "refused": "experts.*not a LoRA-capable linear",
    "keys": {
        "base_model.model.thinker.model.layers.1.self_attn.q_proj.lora_A.weight": [2, 2048],
        "base_model.model.thinker.model.layers.1.self_attn.q_proj.lora_B.weight": [4096, 2],
        "base_model.model.thinker.model.layers.0.mlp.experts.1.gate_proj.lora_A.weight": [2, 2048],
        "base_model.model.thinker.model.layers.0.mlp.experts.1.gate_proj.lora_B.weight": [768, 2],
    },
}}


def _validate(model: str, size: str | None, repo: str, keys: dict) -> LoraSpec:
    """What registering an adapter checks (``check_adapter_file``), from its header."""
    spec = (_released(model, size)[0] if size != "tiny"
            else H.STACKS[model]().lora_spec(H.tiny()))
    spec.apply_yaml_overrides(max_rank=512)
    spec.check_adapter(repo, inspect_keys(spec, sorted(keys.items())))
    return spec


@pytest.mark.parametrize("model,repo", [(m, r) for m, entries in HUB.items() for r in sorted(entries)])
def test_hub_adapter_is_served_or_refused_by_name(model, repo):
    entry = HUB[model][repo]
    keys = entry["keys"]
    if "refused" in entry:
        with pytest.raises(ValueError, match=entry["refused"]):
            _validate(model, entry.get("size"), repo, keys)
        if "served_without" not in entry:
            return
        keys = {k: v for k, v in keys.items() if entry["served_without"] not in k}
    spec = _validate(model, entry.get("size"), repo, keys)
    # no module is silently dropped: each is placed on one of the model's targets
    for module in {parse_lora_key(normalize_lora_key(k))[0] for k in keys}:
        ref = spec.module_map(module)
        assert ref is not None and ref.target in spec.targets, module


@pytest.mark.parametrize("model,repo,other", [
    ("qwen3_5", "lucabaroni/qwen3.5-9b-rlvr-reward-hacking", "4B"),
    ("orpheus", "Tachyeon/orpheus-3b-sft-lora", "tiny"),
])
def test_control_an_adapter_checked_against_another_size_is_refused(model, repo, other):
    """Mechanism off for the header's shape check: refused, not loaded until a request fails."""
    layer0 = {k: v for k, v in HUB[model][repo]["keys"].items() if ".layers.0." in k}
    with pytest.raises(ValueError, match="does not fit"):
        _validate(model, other, repo, layer0)


# --- Qwen3.5: the vision node --------------------------------------------------------------------

V_HIDDEN, V_INTER, PATCH, TEMPORAL, CHANNELS = 8, 16, 2, 2, 3
PATCH_NUMEL = CHANNELS * TEMPORAL * PATCH * PATCH
GRIDS = [(1, 4, 4), (1, 2, 4)]  # two images in one prompt: 16 and 8 patches
PATCHES = sum(t * h * w for t, h, w in GRIDS)
VISION_SHAPES = {  # below `model.visual.`: (out, in)
    **{f"blocks.{i}.{m}": shape for i in range(2) for m, shape in (
        ("attn.qkv", (3 * V_HIDDEN, V_HIDDEN)), ("attn.proj", (V_HIDDEN, V_HIDDEN)),
        ("mlp.linear_fc1", (V_INTER, V_HIDDEN)), ("mlp.linear_fc2", (V_HIDDEN, V_INTER)),
    )},
    "merger.linear_fc1": (4 * V_HIDDEN, 4 * V_HIDDEN),  # a row folds 2x2 patches
    "merger.linear_fc2": (H.HIDDEN, 4 * V_HIDDEN),
    "patch_embed.proj": (V_HIDDEN, PATCH_NUMEL),
}


def vision_config() -> Qwen3_5VisionConfig:
    return Qwen3_5VisionConfig(
        depth=2, hidden_size=V_HIDDEN, intermediate_size=V_INTER, num_heads=2, in_channels=CHANNELS,
        patch_size=PATCH, temporal_patch_size=TEMPORAL, spatial_merge_size=2, num_position_embeddings=16,
        out_hidden_size=H.HIDDEN,
    )


def tower(ref=None) -> Qwen3_5VisionModel:
    """The tower, random; with ``ref``'s adapter merged when given."""
    torch.manual_seed(0)
    torch.set_default_dtype(H.DTYPE)
    try:
        model = Qwen3_5VisionModel(vision_config())
    finally:
        torch.set_default_dtype(torch.float32)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) / math.sqrt(max(p.shape[-1], 1)))
        for module, (a, b, s) in (ref or {}).items():
            params[module + ".weight"] += (s * (b @ a)).view(params[module + ".weight"].shape)
    return model.requires_grad_(False).eval()


def write_vision_adapter(directory, rank=3, alpha=6.0, seed=31):
    """A vision-and-text adapter as unsloth saves one (``model.visual.*``, the patch
    embed's factors conv-shaped, a text ``q_proj`` the LLM's spec takes). Returns
    ``(path, {module: (a, b, scaling)})`` with 2-D factors."""
    directory.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    sd, ref = {}, {}
    for module, (out, inp) in VISION_SHAPES.items():
        a = (torch.randn(rank, inp, generator=g) / math.sqrt(inp)).float()
        b = (torch.randn(out, rank, generator=g) / math.sqrt(rank)).float()
        conv = module == "patch_embed.proj"
        key = f"base_model.model.model.visual.{module}"
        sd[f"{key}.lora_A.weight"] = a.view(rank, CHANNELS, TEMPORAL, PATCH, PATCH) if conv else a
        sd[f"{key}.lora_B.weight"] = b.view(out, rank, 1, 1, 1) if conv else b
        ref[module] = (a.to(H.DTYPE), b.to(H.DTYPE), alpha / rank)
    text = f"base_model.model.{H.TEXT}layers.1.self_attn.q_proj"
    sd[f"{text}.lora_A.weight"] = torch.zeros(rank, H.HIDDEN)
    sd[f"{text}.lora_B.weight"] = torch.zeros(H.PROJ_SHAPES["self_attn.q_proj"][0], rank)
    save_file(sd, str(directory / "adapter_model.safetensors"))
    (directory / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": alpha}))
    return str(directory / "adapter_model.safetensors"), ref


class _FrameAttend:
    """Full attention within each frame; rows past the frames (a bucket's tail) are 0."""

    def __init__(self):
        self.lengths: list[int] = []

    def run(self, q, k, v):
        out, at = torch.zeros_like(q), 0
        for n in self.lengths:
            s = slice(at, at + n)
            out[s] = F.scaled_dot_product_attention(*(x[s].transpose(0, 1) for x in (q, k, v))).transpose(0, 1)
            at += n
        return out


def _vision_node(model, resources=None) -> VisionEncoderSubmodule:
    sub = VisionEncoderSubmodule(model, vision_config())
    sub._frames = _FrameAttend()
    sub.bind_node_resources({VISION_ATTN: sub._frames, **(resources or {})})
    return sub


def _prompt(sub):
    pixels = torch.randn(PATCHES, PATCH_NUMEL, generator=torch.Generator().manual_seed(5), dtype=H.DTYPE)
    return sub.prepare_inputs("encode", None, {
        "image_grid_thw": [torch.tensor([grid]) for grid in GRIDS], "pixel_values": [pixels],
    })


def _engine(rid):
    return SimpleNamespace(request_ids=[rid], piecewise_runners={}, step=None)


def _encode(sub, rid="r"):
    inp = _prompt(sub)
    sub._frames.lengths = list(inp.tensor_inputs["seq_lengths"])
    kwargs = sub.preprocess("encode", _engine(rid), [inp])
    with torch.no_grad():
        return sub.forward_batched("encode", _engine(rid), **kwargs)[rid]["vision_embeds"][0]


def _vision_pool(paths, max_slots=2):
    """The vision node's pool, from the model's one spec as the engine builds it."""
    spec = qwen3_5_lora_spec(H.tiny_config(), vision_config()).for_node("vision_encoder")
    spec.apply_yaml_overrides(max_slots=max_slots, max_rank=4, adapters=dict(paths))
    spec.dtype, spec.backend = H.DTYPE, "torch"
    return H.pool(H.pin_digests(spec))


def _admit(lora_pool, step, rid, adapter, walk="encode", lease=None):
    lora_pool.ingest_request(rid, None if adapter is None else LoraReqConfig(adapter=adapter))
    # idle as the engine sets it: an eager step of a base row (a region never is)
    ctx = StepContext(request_ids=[rid], graph_walk=walk, slot=0, capture=False, slot_lease=lease,
                      idle=adapter is None and lease is None)
    lora = step.steps[VISION_POOL]
    assert lora_pool.admit(lora, ctx).ok
    lora_pool.plan(lora, ctx)
    return lora, ctx


def _region_step(region_cfg, resources, request_ids, seq_lens):
    """The block loop's step as its runner, built over the node's resources, declares it:
    the region's own, plus the rows it adds for a ``segment_rows`` region, a request's
    merged into one."""
    from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner
    from mstar.engine.resources.runner import StepRunner

    runner = PiecewiseAcceleratorGraphRunner(
        label=VISION_BLOCK_LOOP, config=region_cfg, resources=resources, step_runner=StepRunner(resources),
        device=torch.device("cpu"), autocast_dtype=H.DTYPE, num_slots=1,
    )
    return runner._declared_step(request_ids, seq_lens)


def _encode_leased(sub, lora_pool):
    """As the leased path runs it: the outer step plans the adapter rows only (attention
    is the region's), the block loop's region plans its own into its bucket's buffer as
    its capture recorded it, and the merger after the region reads the outer plan again."""
    region_cfg = sub.get_piecewise_accelerator_graph_configs(torch.device("cpu"), H.DTYPE)[VISION_BLOCK_LOOP]
    assert region_cfg.segment_rows
    resources = {VISION_POOL: lora_pool}
    bucket = 32
    lease = SlotLease(slot=0, bucket=BucketKey(PIECEWISE_WALK, 1, bucket, VISION_BLOCK_LOOP))
    # the capture, at startup: the region over the bucket's whole shape, on the zero adapter
    lora_pool.ingest_capture_row("cap")
    cap = _region_step(region_cfg, resources, ["cap"], [bucket]).steps[VISION_POOL]
    cap_ctx = StepContext(request_ids=["cap"], graph_walk=PIECEWISE_WALK, slot=0, capture=True, slot_lease=lease)
    assert lora_pool.admit(cap, cap_ctx).ok
    lora_pool.plan(cap, cap_ctx)
    recorded = lora_pool._current  # the views the captured graph reads

    inp = _prompt(sub)
    outer = declared_step(sub, "encode", ["r"], [inp], piecewise_leases={VISION_BLOCK_LOOP: lease})
    assert set(outer.steps) == {VISION_POOL}
    _admit(lora_pool, outer, "r", "v")
    kwargs = sub.preprocess("encode", _engine("r"), [inp])
    with torch.no_grad():
        hidden, cos, sin = sub.model.embed(
            kwargs["pixel_values"], kwargs["indices"], kwargs["weights"], kwargs["position_ids"],
        )
        seq_lens = list(kwargs["seq_lengths"])
        region = _region_step(region_cfg, resources, ["r"], seq_lens)
        assert len(seq_lens) > 1 and region.steps[VISION_POOL].segments == (Segment("r", "main", PATCHES),)
        # control: a region that does not declare segment_rows gets no adapter rows
        unrowed = dataclasses.replace(region_cfg, segment_rows=False)
        assert VISION_POOL not in _region_step(unrowed, resources, ["r"], seq_lens).steps
        lora, ctx = _admit(lora_pool, region, "r", "v", walk=PIECEWISE_WALK, lease=lease)
        lora_pool._current = recorded  # the replay reads its capture's views of the planned buffer
        sub._frames.lengths = seq_lens
        pad = (0, 0, 0, bucket - PATCHES)
        encoded = sub.model.encode(F.pad(hidden, pad), F.pad(cos, pad), F.pad(sin, pad))[:PATCHES]
        lora_pool.commit(lora, ctx)
        return sub.model.merger(encoded)


@pytest.mark.parametrize("mode,adapter", [("eager", "v"), ("eager", None), ("leased", "v")])
def test_qwen3_5_vision_rows_match_the_merged_tower(tmp_path, mode, adapter):
    """Every vision target: the blocks, the merger (a row per four folded patches) and the
    conv patch embed, eagerly and through the piecewise block loop."""
    path, ref = write_vision_adapter(tmp_path / "v")
    lora_pool = _vision_pool({"v": path})
    sub = _vision_node(tower(), {VISION_POOL: lora_pool})
    if mode == "eager":
        step = declared_step(sub, "encode", ["r"], [_prompt(sub)])
        assert set(step.steps) == {VISION_ATTN, VISION_POOL}
        _admit(lora_pool, step, "r", adapter)
        got = _encode(sub)
    else:
        got = _encode_leased(sub, lora_pool)
    want = _encode(_vision_node(tower(ref if adapter else None)))
    assert got.shape == (PATCHES // 4, H.HIDDEN)
    # the tower ropes q and k in fp32 (as upstream), so merged and unmerged round apart
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    if adapter is not None:  # control: the adapter moves the output far past the tolerance
        assert (want - _encode(_vision_node(tower()))).abs().max() > 1e-3


def test_qwen3_5_vision_replays_the_block_loop_only_when_leased(tmp_path):
    """With adapters, a leased step still declares the engine's LoRA rows; it must replay
    the captured block loop, while an unleased one runs the eager tower on the attention
    plan it declared."""
    path, _ = write_vision_adapter(tmp_path / "v")
    sub = _vision_node(tower(), {VISION_POOL: _vision_pool({"v": path})})
    lease = SlotLease(slot=0, bucket=BucketKey(PIECEWISE_WALK, 1, 32, VISION_BLOCK_LOOP))
    replays = []

    def run(**kwargs):
        replays.append(kwargs)
        return SimpleNamespace(get_view=lambda name: kwargs["static_inputs"][name])

    runner = SimpleNamespace(can_run=lambda bs, tokens: True, run=run)
    rows = torch.zeros(PATCHES, 1)
    for leases, replayed in (({VISION_BLOCK_LOOP: lease}, True), (None, False)):
        step = declared_step(sub, "encode", ["r"], [_prompt(sub)], piecewise_leases=leases)
        engine = SimpleNamespace(request_ids=["r"], piecewise_runners={VISION_BLOCK_LOOP: runner}, step=step)
        assert (sub._replay_block_loop(engine, rows, rows, rows, (PATCHES,)) is not None) is replayed
    assert len(replays) == 1


def test_qwen3_5_text_only_adapter_holds_no_vision_slot_nor_a_wait(tmp_path):
    """An adapter with nothing for the tower is rank 0 on the vision pool. A row on it
    held back for want of a slot before its adapter was read leaves the wait list once a
    read finds it empty (else it would age as the oldest waiter and drain a slot), and
    rows on it are admitted with the pool full."""
    path, _ = write_vision_adapter(tmp_path / "v")
    text, _ = H.write_peft_adapter(tmp_path / "t", 2, ("self_attn.q_proj",), text_only=True)
    lora_pool = _vision_pool({"v": path, "t": text}, max_slots=1)
    sub = _vision_node(tower(), {VISION_POOL: lora_pool})

    def encode(rid, adapter):
        lora_pool.ingest_request(rid, LoraReqConfig(adapter=adapter))
        assert lora_pool.admit_retrieve(rid, "vision_encoder", "encode", None).ready
        ctx = StepContext(request_ids=[rid], graph_walk="encode", slot=0, capture=False)
        assert lora_pool.admit(declared_step(sub, "encode", [rid], [_prompt(sub)]).steps[VISION_POOL], ctx).ok
        return lora_pool.slot_of(rid)

    assert encode("r0", "v") == 1  # the one slot, held
    lora_pool.ingest_request("w", LoraReqConfig(adapter="t"))
    assert not lora_pool.admit_retrieve("w", "vision_encoder", "encode", None).ready
    assert "w" in lora_pool._waiting_since
    lora_pool.remove_request("r0")
    assert encode("r1", "t") == 0  # its admit reads the adapter: nothing for the tower
    assert lora_pool.admit_retrieve("w", "vision_encoder", "encode", None).ready
    assert "w" not in lora_pool._waiting_since
    assert encode("r2", "v") == 1 and encode("r3", "t") == 0  # ready with the pool full
    assert lora_pool.slot_state(1).adapter == "v" and lora_pool.slot_state(1).refcount == 1


# --- Qwen3.5: the tied head ------------------------------------------------------------------------

def test_qwen3_5_tied_head_and_embedding_keep_each_delta_on_its_own_side(tmp_path):
    """0.8B / 2B / 4B tie the head to the embedding. PEFT adapts the two modules apart:
    the embedding takes only its delta, the head only its own, the shared table neither."""
    embed = f"{H.TEXT}embed_tokens"
    path, ref = H.make_adapter(tmp_path / "h.safetensors", {"lm_head": (64, H.HIDDEN), embed: (H.HIDDEN, 64)},
                               4, 8.0, 21, embedding=(embed,))
    base = {k: v for k, v in H.hf_text_weights().items() if k != "lm_head.weight"}
    model = H.build_stack(base, config=dataclasses.replace(H.tiny_config(), tie_word_embeddings=True))
    assert model.lm_head.weight is model.model.embed_tokens.weight
    lora_pool = H.pool(H.lora_spec({"h": path}))
    H.bind(model, H._Rows(), {LORA_RESOURCE_KEY: lora_pool})
    H.plan([lora_pool], [("r0", "h", 1), ("r1", "h", 1), ("r2", "h", 1)])  # three decode rows
    ids, hidden = torch.tensor([1, 2, 60]), H.rand(3, H.HIDDEN, seed=4)
    with torch.no_grad(), H.compute(lora_pool):
        embeds, logits = model.model.embed(ids), model.lm_head(hidden)
    table = base[embed + ".weight"]
    torch.testing.assert_close(embeds, H.merged(base, {embed: ref[embed]}, (embed,))[embed + ".weight"][ids])
    a, b, s = ref["lm_head"]
    torch.testing.assert_close(logits, hidden @ table.t() + s * (hidden @ a.t()) @ b.t())
    assert torch.equal(model.lm_head.weight, table)
