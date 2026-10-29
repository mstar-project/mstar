"""The fused depth-loop kernels (mstar.platforms.depth_decoder_ops) and DepthDecoder's fused loop on AMD GPUs.

The kernels are checked against their torch references and against the kernels the per-position forwards run. The
loop is checked teacher-forced against a float64 evaluation of the same weights: its logits must sit as close to it
as the per-position forward's do (the bf16 noise), at the real shapes of both code predictors, with random weights
and with the cached checkpoints. Captured replays must equal eager runs, and the loop must launch far fewer kernels.
"""

import functools
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")

DEV = "cuda"


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def _ulps_bf16(a, b):
    """Largest difference in bf16 units in the last place of b."""
    a, b = a.float(), b.float()
    ulp = torch.where(b == 0, torch.full_like(b, 2.0 ** -133), (b.abs().log2().floor() - 7).exp2())
    return ((a - b).abs() / ulp).max().item()


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

def _qkv_and_cache(rows, heads, kv_heads, d, positions, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    qkv = torch.randn(rows, (heads + 2 * kv_heads) * d, device=DEV, generator=g).to(torch.bfloat16) * 3
    wq = (torch.rand(d, device=DEV, generator=g) + 0.5).to(torch.bfloat16)
    wk = (torch.rand(d, device=DEV, generator=g) + 0.5).to(torch.bfloat16)
    # a [rows, 2, positions, kv_heads, d] slab like the code predictors' scratch, K and V as views
    kv = torch.randn(rows, 2, positions, kv_heads, d, device=DEV, generator=g).to(torch.bfloat16)
    return qkv, wq, wk, kv


@pytest.mark.parametrize("rows", [1, 5, 32])
@pytest.mark.parametrize("cache_pos", [0, 1, 7, 15])
def test_depth_attention_kernel(rows, cache_pos):
    from mstar.platforms.depth_decoder_ops import depth_attention, depth_attention_reference
    from mstar.utils.attention import decode_attn_nhd, fused_qk_norm_rope

    heads, kv_heads, d, positions, eps, theta = 16, 8, 128, 16, 1e-6, 1e6
    qkv, wq, wk, kv = _qkv_and_cache(rows, heads, kv_heads, d, positions, rows * 100 + cache_pos)
    kv_ref, kv_old = kv.clone(), kv.clone()
    out = depth_attention(qkv, wq, wk, kv[:, 0], kv[:, 1], cache_pos, heads, kv_heads, eps, theta)
    want = depth_attention_reference(qkv, wq, wk, kv_ref[:, 0], kv_ref[:, 1], cache_pos, heads, kv_heads, eps, theta)

    # what the Omni per-position forward runs: in-place qk-norm + rope, slice writes, decode attention
    q, k, v = qkv.clone().split((heads * d, kv_heads * d, kv_heads * d), dim=-1)
    pos = torch.full((rows,), cache_pos, device=DEV)
    q = fused_qk_norm_rope(q.reshape(rows, heads, d), wq, pos, eps, theta)
    k = fused_qk_norm_rope(k.reshape(rows, kv_heads, d), wk, pos, eps, theta)
    kv_old[:, 0, cache_pos] = k
    kv_old[:, 1, cache_pos] = v.reshape(rows, kv_heads, d)
    old = decode_attn_nhd(q.unsqueeze(1), kv_old[:, 0], kv_old[:, 1], cache_pos + 1).reshape(rows, -1)

    # the new K slot as the old qk-norm/rope kernel writes it (FMA contraction may move a value by an ulp),
    # V and every other slot untouched
    assert _ulps_bf16(kv[:, 0, cache_pos], kv_old[:, 0, cache_pos]) <= 1
    assert _ulps_bf16(kv[:, 0, cache_pos], kv_ref[:, 0, cache_pos]) <= 1
    keep = torch.ones(positions, dtype=torch.bool, device=DEV)
    keep[cache_pos] = False
    assert torch.equal(kv[:, 1], kv_old[:, 1]) and torch.equal(kv[:, 0, keep], kv_old[:, 0, keep])
    print(f"rows={rows} pos={cache_pos}: out vs ref {_rel(out, want):.2e} ({_ulps_bf16(out, want):.0f} ulp), "
          f"vs old kernels {_rel(out, old):.2e} ({_ulps_bf16(out, old):.0f} ulp)")
    assert _rel(out, want) < 4e-3
    assert _ulps_bf16(out, old) <= 2


@pytest.mark.parametrize("width", [1024, 2048])
@pytest.mark.parametrize("sum_dtype", [torch.float32, torch.bfloat16])
def test_codec_embed_accumulate_kernel(width, sum_dtype):
    from mstar.platforms.depth_decoder_ops import codec_embed_accumulate, codec_embed_accumulate_reference

    rows, vocab = 27, 2048
    table = torch.randn(vocab, width, device=DEV).to(torch.bfloat16)
    start = torch.randn(rows, width, device=DEV).to(sum_dtype)
    for accumulate in (False, True):
        codes = torch.randint(0, vocab, (rows,), device=DEV)
        bufs = [torch.empty(rows, width, device=DEV, dtype=torch.bfloat16), start.clone(),
                torch.zeros(rows, 16, device=DEV, dtype=torch.long)]
        ref = [b.clone() for b in bufs]
        codec_embed_accumulate(codes, table, *bufs, 5, accumulate)
        codec_embed_accumulate_reference(codes, table, *ref, 5, accumulate)
        for got, want in zip(bufs, ref, strict=True):
            assert torch.equal(got, want)


def test_silu_and_mul_kernel():
    from mstar.platforms.depth_decoder_ops import silu_and_mul, silu_and_mul_reference

    x = (torch.randn(27, 6144, device=DEV) * 4).to(torch.bfloat16)
    assert _ulps_bf16(silu_and_mul(x), silu_and_mul_reference(x)) <= 1


def test_decode_linear_rows():
    from mstar.platforms.depth_decoder_ops import decode_linear

    w = (torch.randn(4096, 1024, device=DEV) / 32).to(torch.bfloat16)
    for rows in (1, 2, 3, 32):
        x = torch.randn(rows, 1024, device=DEV).to(torch.bfloat16)
        assert _rel(decode_linear(x, w), x.double() @ w.double().T) < 4e-3


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def _randomize(module, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            noise = torch.randn(p.shape, device=DEV, generator=g)
            p.copy_(1.0 + 0.1 * noise if "norm" in name else noise / math.sqrt(p.shape[-1]))


@functools.cache
def _omni_random():
    from mstar.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor
    from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig

    with torch.device("meta"):
        cp = Qwen3OmniCodePredictor(Qwen3OmniModelConfig())
    cp = cp.to(torch.bfloat16).to_empty(device=DEV)
    _randomize(cp, 0)
    cp.consolidate_stacked_weights()
    table = (torch.randn(3072, 1024, device=DEV) / 32).to(torch.bfloat16)
    return cp.eval(), torch.nn.Embedding.from_pretrained(table), torch.float32


@functools.cache
def _tts_random():
    from mstar.model.qwen3_tts.components.talker import Qwen3TTSCodePredictor
    from mstar.model.qwen3_tts.config import Qwen3TTSModelConfig, Qwen3TTSTalkerConfig

    with torch.device("meta"):
        cp = Qwen3TTSCodePredictor(Qwen3TTSModelConfig(talker=Qwen3TTSTalkerConfig(hidden_size=2048)))
    cp = cp.to(torch.bfloat16).to_empty(device=DEV)
    _randomize(cp, 1)
    cp.consolidate_stacked_weights()
    table = (torch.randn(3072, 2048, device=DEV) / 45).to(torch.bfloat16)
    return cp.eval(), torch.nn.Embedding.from_pretrained(table), None


def _snapshot(repo):
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snapshots = root / f"models--{repo.replace('/', '--')}" / "snapshots"
    for snap in sorted(snapshots.iterdir()) if snapshots.is_dir() else []:
        if any(snap.glob("*.safetensors")) or any(snap.glob("*.safetensors.index.json")):
            return snap
    return None


@functools.cache
def _tts_checkpoint():
    from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel

    snap = _snapshot("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    if snap is None:
        pytest.skip("Qwen3-TTS 1.7B is not in the local Hugging Face cache")
    talker = Qwen3TTSModel(model_path_hf=str(snap)).get_submodule("Talker", device="cuda:0",
                                                                  autocast_dtype=torch.bfloat16)
    return talker.code_predictor, talker.model.model.codec_embedding, None


@functools.cache
def _omni_checkpoint():
    from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights
    from mstar.model.loader.iterators import iter_safetensors_shards
    from mstar.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor
    from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    snap = _snapshot("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    if snap is None:
        pytest.skip("Qwen3-Omni is not in the local Hugging Face cache")
    config = Qwen3OmniModelConfig.from_pretrained(snap)
    with torch.device("meta"):
        cp = Qwen3OmniCodePredictor(config)
    cp = cp.to(torch.bfloat16).to_empty(device=DEV)
    prefix = "talker.code_predictor."
    load_hf_weights(cp, ((k.removeprefix(prefix), v) for k, v in iter_safetensors_shards(
        str(snap), device=DEV, prefix=prefix)), stacked_params=LLAMA_STACKED_PARAMS,
        name_remapper=Qwen3OmniModel._thinker_remap)
    cp.consolidate_stacked_weights()
    tables = [v for _, v in iter_safetensors_shards(str(snap), device=DEV, prefix="talker.model.codec_embedding")]
    return cp.eval(), torch.nn.Embedding.from_pretrained(tables[0].to(torch.bfloat16)), torch.float32


class _Forced:
    """Samples the given codes whatever the logits (teacher forcing), keeping the logits."""

    def __init__(self, codes):
        self.codes, self.logits = codes, []

    def __call__(self, logits):
        self.logits.append(logits.float())
        return self.codes[:, len(self.logits)].contiguous()


def _rms(x, norm):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon) * norm.weight.double()


def _rope(x, pos, theta):
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, device=DEV, dtype=torch.float64) / d)
    cos, sin = (pos * inv).cos(), (pos * inv).sin()
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def _fp64_logits(cp, last_hidden, group0, codes):
    """Every group's logits in float64, teacher-forced with ``codes``."""
    groups = cp.num_code_groups
    tables = [group0.weight] + [t.weight for t in cp.model.codec_embedding]
    inputs = [last_hidden] + [tables[g][codes[:, g]] for g in range(groups - 1)]
    keys = [[] for _ in cp.model.layers]
    values = [[] for _ in cp.model.layers]
    logits = []
    for pos, x in enumerate(inputs):
        x = x.double()
        if cp.input_projection is not None:
            x = F.linear(x, cp.input_projection.weight.double(), cp.input_projection.bias.double())
        for i, layer in enumerate(cp.model.layers):
            attn, mlp = layer.self_attn, layer.mlp
            heads, kv_heads, d = attn.num_heads, attn.num_kv_heads, attn.head_dim
            q, k, v = F.linear(_rms(x, layer.input_layernorm), attn.qkv_proj.weight.double()).split(
                (heads * d, kv_heads * d, kv_heads * d), dim=-1)
            q = _rope(_rms(q.reshape(-1, heads, d), attn.q_norm), pos, attn.rope_theta)
            keys[i].append(_rope(_rms(k.reshape(-1, kv_heads, d), attn.k_norm), pos, attn.rope_theta))
            values[i].append(v.reshape(-1, kv_heads, d))
            kk = torch.stack(keys[i], 2).repeat_interleave(heads // kv_heads, 1)
            vv = torch.stack(values[i], 2).repeat_interleave(heads // kv_heads, 1)
            out = torch.softmax(torch.einsum("bhd,bhnd->bhn", q, kk) / math.sqrt(d), -1)
            x = x + F.linear(torch.einsum("bhn,bhnd->bhd", out, vv).reshape(len(x), -1), attn.o_proj.weight.double())
            gate, up = F.linear(_rms(x, layer.post_attention_layernorm), mlp.gate_up_proj.weight.double()).chunk(2, -1)
            x = x + F.linear(F.silu(gate) * up, mlp.down_proj.weight.double())
        if pos:
            logits.append(F.linear(_rms(x, cp.model.norm), cp.lm_head_weight[pos - 1].double()))
    return torch.stack(logits)


def _loop(fn, cp, group0, last_hidden, codes, sum_dtype):
    from mstar.model.components.depth_decoder import DepthDecoder

    sampler = _Forced(codes)
    kv = cp.depth_kv_cache(len(codes), 32)
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
        out_codes, embed_sum = getattr(DepthDecoder, fn)(cp, last_hidden.clone(), codes[:, 0].contiguous(), group0,
                                                         sampler, kv, sum_dtype or group0.weight.dtype)
    assert torch.equal(out_codes, codes)
    return torch.stack(sampler.logits), embed_sum


@pytest.mark.parametrize("build", [_omni_random, _tts_random, _omni_checkpoint, _tts_checkpoint],
                         ids=["omni-random", "tts-random", "omni-checkpoint", "tts-checkpoint"])
@pytest.mark.parametrize("rows", [1, 27])
def test_fused_loop_is_as_close_to_float64_as_the_per_position_forward(build, rows):
    cp, group0, sum_dtype = build()
    g = torch.Generator(device=DEV).manual_seed(rows)
    width = cp.input_projection.in_features if cp.input_projection is not None else group0.weight.shape[1]
    last_hidden = torch.randn(rows, width, device=DEV, generator=g).to(torch.bfloat16)
    codes = torch.cat([torch.randint(0, group0.num_embeddings, (rows, 1), device=DEV, generator=g),
                       torch.randint(0, cp.lm_head_weight.shape[1], (rows, cp.num_code_groups - 1), device=DEV,
                                     generator=g)], dim=1)
    fused, fused_sum = _loop("_fused_loop", cp, group0, last_hidden, codes, sum_dtype)
    old, old_sum = _loop("_reference_loop", cp, group0, last_hidden, codes, sum_dtype)
    with torch.no_grad():
        exact = _fp64_logits(cp, last_hidden, group0, codes)
    assert torch.equal(fused_sum, old_sum)
    err_fused = [_rel(fused[i], exact[i]) for i in range(len(exact))]
    err_old = [_rel(old[i], exact[i]) for i in range(len(exact))]
    top_fused = (fused.argmax(-1) == exact.argmax(-1)).float().mean().item()
    top_old = (old.argmax(-1) == exact.argmax(-1)).float().mean().item()
    print(f"{build.__name__} rows={rows}: logits rel err vs fp64 per group, fused "
          f"{' '.join(f'{e:.1e}' for e in err_fused)} | per-position {' '.join(f'{e:.1e}' for e in err_old)}; "
          f"argmax agreement fused {top_fused:.3f} per-position {top_old:.3f}; fused vs per-position "
          f"{_rel(fused, old):.1e}")
    assert max(err_fused) < 5e-2
    assert sum(err_fused) <= 1.25 * sum(err_old) + 1e-3


@pytest.mark.parametrize("build", [_omni_random, _tts_random], ids=["omni", "tts"])
def test_fused_loop_replays_from_a_captured_graph(build):
    from mstar.model.components.depth_decoder import DepthDecoder

    cp, group0, sum_dtype = build()
    rows = 8
    width = cp.input_projection.in_features if cp.input_projection is not None else group0.weight.shape[1]
    last_hidden = torch.randn(rows, width, device=DEV).to(torch.bfloat16)
    layer0 = torch.randint(0, group0.num_embeddings, (rows,), device=DEV)
    kv = cp.depth_kv_cache(rows, 32)
    sum_dtype = sum_dtype or group0.weight.dtype

    def run():
        return DepthDecoder._fused_loop(cp, last_hidden, layer0, group0, lambda x: x.argmax(-1), kv, sum_dtype)

    with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
        eager = [t.clone() for t in run()]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        last_hidden.copy_(torch.randn_like(last_hidden))
        graph.replay()
        again = [t.clone() for t in run()]
        assert all(torch.equal(a, b) for a, b in zip(captured, again, strict=True))
        last_hidden.copy_(last_hidden.flip(0).clone())
        layer0.copy_(layer0.flip(0).clone())
        graph.replay()
        flipped = [t.clone() for t in captured]
    assert not all(torch.equal(a, b) for a, b in zip(eager, again, strict=True))
    assert torch.equal(flipped[0][:, 0], layer0)


@pytest.mark.parametrize("build", [_omni_random, _tts_random], ids=["omni", "tts"])
def test_fused_loop_kernel_count(build):
    """Kernels one frame launches, eager, with a greedy sampler (argmax), and the frame's time replayed from a
    captured graph at 27 rows padded to the 32-row bucket. The per-position forward runs uncompiled here, so its
    numbers are an upper bound for the served path, where Inductor fuses parts of it."""
    from torch.profiler import ProfilerActivity, profile

    from mstar.model.components.depth_decoder import DepthDecoder

    cp, group0, sum_dtype = build()
    rows = 32
    width = cp.input_projection.in_features if cp.input_projection is not None else group0.weight.shape[1]
    last_hidden = torch.randn(rows, width, device=DEV).to(torch.bfloat16)
    layer0 = torch.randint(0, group0.num_embeddings, (rows,), device=DEV)
    kv = cp.depth_kv_cache(rows, 32)
    sum_dtype = sum_dtype or group0.weight.dtype
    counts, times = {}, {}
    for fn in ("_fused_loop", "_reference_loop"):
        def run(loop=getattr(DepthDecoder, fn)):
            return loop(cp, last_hidden, layer0, group0, lambda x: x.argmax(-1), kv, sum_dtype)

        with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
            run()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                run()
                torch.cuda.synchronize()
            counts[fn] = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(5):
                graph.replay()
            start.record()
            for _ in range(50):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            times[fn] = start.elapsed_time(end) / 50
    print(f"{build.__name__}: kernels per frame (argmax sampler) {counts}; captured frame ms at 32 rows "
          f"{ {k: round(v, 3) for k, v in times.items()} }")
    assert counts["_fused_loop"] < 0.65 * counts["_reference_loop"]
    assert times["_fused_loop"] < times["_reference_loop"]
