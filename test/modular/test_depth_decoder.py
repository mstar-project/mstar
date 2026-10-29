"""The fused depth loop (``DepthDecoder._fused_loop``) computes what each code predictor's own per-position forward
computes. On the CPU every fused op runs its torch reference, so this pins the loop's structure: the residual stream
carried through the fused add-norms, the K/V positions, the projection of every depth input, the per-group heads and
the gather/sum/scatter of the codes. The Triton helpers of the per-position forwards are swapped for torch
equivalents."""
import pytest
import torch
import torch.nn.functional as F

from mstar import platforms
from mstar.model.components.depth_decoder import DepthDecoder
from mstar.model.qwen3_omni.components import talker as omni_talker
from mstar.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor
from mstar.model.qwen3_omni.config import CodePredictorConfig, Qwen3OmniModelConfig, TalkerConfig
from mstar.model.qwen3_tts.components import talker as tts_talker
from mstar.model.qwen3_tts.components.talker import Qwen3TTSCodePredictor
from mstar.model.qwen3_tts.config import (
    Qwen3TTSCodePredictorConfig,
    Qwen3TTSModelConfig,
    Qwen3TTSTalkerConfig,
)
from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform
from mstar.platforms import depth_decoder_ops as ops

GROUPS = 4


def _rope(x, pos, theta):
    d = x.shape[-1]
    inv = theta ** (-torch.arange(d // 2, dtype=torch.float32) * 2.0 / d)
    angle = pos.float().reshape(-1, 1, 1) * inv
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat((x1 * angle.cos() - x2 * angle.sin(), x2 * angle.cos() + x1 * angle.sin()), dim=-1)


def _qk_norm_rope(x, w_norm, pos, eps, rope_theta):
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w_norm.float()
    x.copy_(_rope(x32, pos, rope_theta))
    return x


def _rope_pos_ids(q, k, pos, theta):
    q.copy_(_rope(q.float(), pos, theta))
    k.copy_(_rope(k.float(), pos, theta))
    return q, k


def _decode_attn(q, k_cache, v_cache, cache_len, sm_scale=None):
    k, v = k_cache[:, :cache_len], v_cache[:, :cache_len]
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=sm_scale, enable_gqa=True,
    )
    return out.transpose(1, 2)


def _randomize(module: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            noise = torch.randn(p.shape, generator=g)
            p.copy_(1.0 + 0.1 * noise if "norm" in name else noise / p.shape[-1] ** 0.5)


@pytest.fixture
def omni(monkeypatch):
    monkeypatch.setattr(omni_talker, "fused_qk_norm_rope", _qk_norm_rope)
    monkeypatch.setattr(omni_talker, "decode_attn_nhd", _decode_attn)
    cfg = Qwen3OmniModelConfig(
        talker=TalkerConfig(num_code_groups=GROUPS),
        code_predictor=CodePredictorConfig(vocab_size=16, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                                           num_attention_heads=4, num_key_value_heads=2, head_dim=8),
    )
    cp = Qwen3OmniCodePredictor(cfg)
    _randomize(cp, 0)
    cp.consolidate_stacked_weights()
    return cp.eval(), torch.nn.Embedding(24, 32)


def _tts_config(talker_hidden: int) -> Qwen3TTSModelConfig:
    cp = Qwen3TTSCodePredictorConfig(num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                     hidden_size=16, intermediate_size=40, head_dim=8, vocab_size=20,
                                     num_code_groups=GROUPS)
    talker = Qwen3TTSTalkerConfig(hidden_size=talker_hidden, vocab_size=64, num_code_groups=GROUPS, code_predictor=cp)
    return Qwen3TTSModelConfig(talker=talker)


@pytest.fixture(params=[32, 16], ids=["projected", "same-width"])
def tts(request, monkeypatch):
    monkeypatch.setattr(tts_talker, "apply_rope_pos_ids", _rope_pos_ids)
    monkeypatch.setattr(tts_talker, "decode_attn_nhd", _decode_attn)
    cfg = _tts_config(request.param)
    cp = Qwen3TTSCodePredictor(cfg)
    _randomize(cp, 1)
    cp.consolidate_stacked_weights()
    assert (cp.input_projection is None) == (request.param == 16)
    return cp.eval(), torch.nn.Embedding(64, request.param)


class _Greedy:
    """Argmax sampling that keeps every group's logits."""

    def __init__(self):
        self.logits = []

    def __call__(self, logits):
        self.logits.append(logits.clone())
        return logits.argmax(-1)


def _run(loop, cp: DepthDecoder, group0: torch.nn.Embedding, rows: int, sum_dtype: torch.dtype):
    g = torch.Generator().manual_seed(rows)
    width = cp.input_projection.in_features if cp.input_projection is not None else cp.model.norm.weight.shape[0]
    last_hidden = torch.randn(rows, width, generator=g)
    layer0 = torch.randint(0, group0.num_embeddings, (rows,), generator=g)
    sampler = _Greedy()
    cp._depth_kv = None
    with torch.no_grad():
        codes, embed_sum = loop(cp, last_hidden.clone(), layer0, group0, sampler, 8, sum_dtype)
    return codes, embed_sum, torch.stack(sampler.logits), cp.depth_kv_cache(rows, 8).clone()


@pytest.mark.parametrize("rows", [1, 3])
def test_fused_loop_matches_the_omni_per_position_forward(omni, rows):
    cp, group0 = omni
    want = _run(DepthDecoder._reference_loop, cp, group0, rows, torch.float32)
    got = _run(DepthDecoder._fused_loop, cp, group0, rows, torch.float32)
    assert torch.equal(got[0], want[0])
    for g, w in zip(got[1:], want[1:], strict=True):
        torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rows", [1, 3])
def test_fused_loop_matches_the_tts_per_position_forward(tts, rows):
    cp, group0 = tts
    want = _run(DepthDecoder._reference_loop, cp, group0, rows, torch.float32)
    got = _run(DepthDecoder._fused_loop, cp, group0, rows, torch.float32)
    assert torch.equal(got[0], want[0])
    for g, w in zip(got[1:], want[1:], strict=True):
        torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-5)


def test_loop_returns_every_code_and_the_sum_of_their_embeddings(omni):
    cp, group0 = omni
    codes, embed_sum, logits, _ = _run(DepthDecoder._fused_loop, cp, group0, 2, torch.float32)
    assert codes.dtype == torch.long and codes.shape == (2, GROUPS)
    assert embed_sum.dtype == torch.float32 and embed_sum.shape == (2, 32)
    assert logits.shape == (GROUPS - 1, 2, 16)
    assert torch.equal(codes[:, 1:], logits.argmax(-1).T)
    embeds = [group0.weight[codes[:, 0]]] + [cp.model.codec_embedding[g - 1].weight[codes[:, g]]
                                            for g in range(1, GROUPS)]
    torch.testing.assert_close(embed_sum, torch.stack(embeds).sum(0))


def test_run_depth_loop_keeps_the_per_position_forward_off_amd_gpus(omni, monkeypatch):
    """The CPU platform takes the reference loop, which runs the model's own forward per position."""
    cp, group0 = omni
    calls = []
    forward = cp.forward_depth_unrolled
    monkeypatch.setattr(cp, "forward_depth_unrolled", lambda *a, **k: calls.append(k["cache_pos"]) or forward(*a, **k))
    with torch.no_grad():
        cp.run_depth_loop(torch.randn(2, 32), torch.tensor([1, 2]), group0, lambda x: x.argmax(-1), 4)
    assert calls == list(range(GROUPS))


def test_depth_kv_cache_is_allocated_once_and_sliced(omni):
    cp, _ = omni
    cp._depth_kv = None
    kv = cp.depth_kv_cache(3, 8)
    assert kv.shape == (3, 3, 2, GROUPS, 2, 8)
    assert cp.depth_kv_cache(5, 8).data_ptr() == kv.data_ptr()


def test_codec_embed_accumulate_reference():
    table = torch.randn(10, 6, dtype=torch.bfloat16)
    codes = torch.tensor([3, 7])
    embed = torch.empty(2, 6, dtype=torch.bfloat16)
    total = torch.empty(2, 6, dtype=torch.float32)
    all_codes = torch.zeros(2, 4, dtype=torch.long)
    ops.codec_embed_accumulate(codes, table, embed, total, all_codes, 0, False)
    assert torch.equal(embed, table[codes]) and torch.equal(total, table[codes].float())
    more = torch.tensor([1, 1])
    ops.codec_embed_accumulate(more, table, embed, total, all_codes, 2, True)
    assert torch.equal(embed, table[more])
    assert torch.equal(total, table[codes].float() + table[more].float())
    assert all_codes.tolist() == [[3, 0, 1, 0], [7, 0, 1, 0]]


def test_silu_and_mul_reference_rounds_once():
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    gate, up = x.float().chunk(2, dim=-1)
    assert torch.equal(ops.silu_and_mul(x), (F.silu(gate) * up).to(torch.bfloat16))


@pytest.mark.parametrize("cache_pos", [0, 2, 5])
def test_depth_attention_reference_is_per_head_attention(cache_pos):
    torch.manual_seed(cache_pos)
    rows, heads, kv_heads, d, positions = 2, 4, 2, 8, 6
    qkv = torch.randn(rows, (heads + 2 * kv_heads) * d)
    wq, wk = torch.rand(d) + 0.5, torch.rand(d) + 0.5
    k_cache, v_cache = torch.randn(rows, positions, kv_heads, d), torch.randn(rows, positions, kv_heads, d)
    k_before, v_before = k_cache.clone(), v_cache.clone()
    out = ops.depth_attention(qkv, wq, wk, k_cache, v_cache, cache_pos, heads, kv_heads, 1e-6, 1e4)

    q, k, v = qkv.split((heads * d, kv_heads * d, kv_heads * d), dim=-1)
    pos = torch.full((rows,), cache_pos)
    norm = lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w  # noqa: E731
    q = _rope(norm(q.reshape(rows, heads, d), wq), pos, 1e4)
    k = _rope(norm(k.reshape(rows, kv_heads, d), wk), pos, 1e4)
    torch.testing.assert_close(k_cache[:, cache_pos], k)
    torch.testing.assert_close(v_cache[:, cache_pos], v.reshape(rows, kv_heads, d))
    keep = torch.ones(positions, dtype=torch.bool)
    keep[cache_pos] = False
    assert torch.equal(k_cache[:, keep], k_before[:, keep]) and torch.equal(v_cache[:, keep], v_before[:, keep])
    for h in range(heads):
        keys = k_cache[:, : cache_pos + 1, h // 2]
        weights = torch.softmax((keys @ q[:, h].unsqueeze(-1)).squeeze(-1) / d ** 0.5, dim=-1)
        want = (weights.unsqueeze(-1) * v_cache[:, : cache_pos + 1, h // 2]).sum(1)
        torch.testing.assert_close(out[:, h * d:(h + 1) * d], want, rtol=1e-5, atol=1e-5)


def _omni_on(platform, monkeypatch):
    """A tiny Omni code predictor built while ``current_platform`` is ``platform``."""
    monkeypatch.setattr(platforms, "resolve_platform", lambda: platform)
    for dispatch in _DISPATCHES:
        dispatch.reset()
    cfg = Qwen3OmniModelConfig(
        talker=TalkerConfig(num_code_groups=GROUPS),
        code_predictor=CodePredictorConfig(vocab_size=16, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                           num_attention_heads=4, num_key_value_heads=2, head_dim=8),
    )
    cp = Qwen3OmniCodePredictor(cfg)
    _randomize(cp, 3)
    cp.consolidate_stacked_weights()
    return cp.eval(), torch.nn.Embedding(24, 32)


_DISPATCHES = (ops._depth_attention_impl, ops._codec_embed_accumulate_impl, ops._silu_and_mul_impl,
               ops.decode_linear)


@pytest.fixture
def reset_dispatches():
    yield
    for dispatch in _DISPATCHES:
        dispatch.reset()


@pytest.mark.parametrize("platform", [RocmPlatform(), CudaPlatform(), CpuPlatform()], ids=lambda p: p.name)
def test_the_fused_loop_and_its_kernels_run_on_amd_gpus_only(monkeypatch, reset_dispatches, platform):
    cp, _ = _omni_on(platform, monkeypatch)
    rocm = platform.is_rocm()
    assert cp.run_depth_loop.__func__ is (DepthDecoder._fused_loop if rocm else DepthDecoder._reference_loop)
    assert cp.run_depth_loop.__self__ is cp
    assert ops._depth_attention_impl._resolve() is (
        ops._depth_attention_triton if rocm else ops.depth_attention_reference)
    assert ops._codec_embed_accumulate_impl._resolve() is (
        ops._codec_embed_accumulate_triton if rocm else ops.codec_embed_accumulate_reference)
    assert ops._silu_and_mul_impl._resolve() is (ops._silu_and_mul_triton if rocm else ops.silu_and_mul_reference)
    if not rocm:
        assert ops.decode_linear._resolve() is F.linear


class _GraphBreakingSampler:
    """Greedy, out of dynamo like the engine's samplers, so every group's sampling is a graph break."""

    @torch.compiler.disable
    def sample(self, logits):
        return logits.argmax(-1)


@pytest.mark.parametrize("platform", [RocmPlatform(), CpuPlatform()], ids=["fused", "per-position"])
def test_a_compiled_caller_reaches_the_loop_without_recompiling(monkeypatch, reset_dispatches, platform):
    """The Talkers call the loop from torch.compile'd forwards and capture them under the fail_on_recompile stance.
    Nothing between the caller and the loop may be compiled around the bound-method sampler, which is new on every
    call (on CPU tensors the fused loop runs its op references)."""
    monkeypatch.setattr(omni_talker, "fused_qk_norm_rope", _qk_norm_rope)
    monkeypatch.setattr(omni_talker, "decode_attn_nhd", _decode_attn)
    cp, group0 = _omni_on(platform, monkeypatch)
    sampler = _GraphBreakingSampler()

    def forward(last_hidden, layer0):
        hidden = last_hidden * 1.0
        return cp.run_depth_loop(hidden, layer0, group0, sampler.sample, 4, sum_dtype=torch.float32)

    torch._dynamo.reset()
    compiled = torch.compile(forward, backend="eager", fullgraph=False, dynamic=False)
    x, codes = torch.randn(3, 32), torch.tensor([1, 5, 7])
    with torch.no_grad():
        want = forward(x, codes)
        for _ in range(2):
            compiled(x, codes)
        with torch.compiler.set_stance("fail_on_recompile"):
            got = compiled(x, codes)
    torch._dynamo.reset()
    assert torch.equal(got[0], want[0])
    torch.testing.assert_close(got[1], want[1])
