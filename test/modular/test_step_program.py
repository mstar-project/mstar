"""StepProgram for the code-predictor depth loop, built from the Qwen3-Omni Talker's declarations.

CPU only: the Talker's code predictor is built on the meta device (its
parameter names and shapes are what the program binds), the program is checked
against them, lowered to the resident kernel's phase table, interpreted in
float64 against an independent forward of the same weights, and the executor
choice falls back to the accelerator graph off ROCm.
"""

from __future__ import annotations

import math
import types
from collections import Counter

import pytest
import torch

from mstar.engine import persistent_executor as pe
from mstar.engine import step_program as sp
from mstar.platforms.facts import DeviceFacts


@pytest.fixture(scope="module")
def talker():
    from mstar.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor
    from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
    from mstar.model.qwen3_omni.submodules import TalkerSubmodule

    cfg = Qwen3OmniModelConfig()
    with torch.device("meta"):
        code_predictor = Qwen3OmniCodePredictor(cfg)
    # the smallest real submodule: what declare_step and the program builder read
    sub = TalkerSubmodule.__new__(TalkerSubmodule)
    torch.nn.Module.__init__(sub)
    sub.code_predictor = code_predictor
    sub.config = cfg
    sub.cp_cfg = cfg.code_predictor
    sub.num_codes = cfg.code_predictor.num_code_groups
    return sub


@pytest.fixture(scope="module")
def program(talker):
    from mstar.model.qwen3_omni.config import CODE_PRED_SAMPLER

    return sp.program_for_talker(talker, CODE_PRED_SAMPLER)


def test_sampler_is_one_the_decode_step_declares(talker, program):
    step = talker.declare_step("talker_decode", ["r0", "r1"], [types.SimpleNamespace(input_seq_len=1)] * 2)
    assert set(program.resources) <= set(step.steps)
    assert program.resources == {"code_sampler": "sampler"}


def test_phases_follow_the_loop(program):
    s = program.shape
    assert (s.layers, s.positions, s.hidden, s.heads, s.kv_heads, s.head_dim, s.intermediate, s.vocab) == (
        5, 16, 1024, 16, 8, 128, 3072, 2048)
    kinds = Counter(op.kind for op in program.ops)
    sampled = s.positions - 1
    # position 0 stops its last layer after the K/V append
    full_layers = s.positions * s.layers - 1
    assert kinds[sp.OpKind.QKV] == kinds[sp.OpKind.ATTN] == s.positions * s.layers
    assert kinds[sp.OpKind.OPROJ] == kinds[sp.OpKind.GATEUP] == kinds[sp.OpKind.DOWN] == full_layers
    assert kinds[sp.OpKind.LMHEAD] == kinds[sp.OpKind.SAMPLE] == sampled
    assert kinds[sp.OpKind.PRO] == kinds[sp.OpKind.INPUT] == 1
    assert len(program.ops) == program.barriers() == 429
    assert sp.check(program) == []
    cache_only = [op for op in program.ops if op.kind == sp.OpKind.ATTN and op.aux == 0]
    assert [(op.position, op.layer) for op in cache_only] == [(0, s.layers - 1)]


def test_every_operand_is_a_submodule_tensor_of_the_right_shape(talker, program):
    tensors = {**dict(talker.named_parameters()), **dict(talker.named_buffers())}
    assert [n for n in sp.required_params(program) if n not in tensors] == []
    s = program.shape
    for op in program.ops:
        if op.kind not in sp.GEMM_OPS:
            continue
        w = tensors[op.weight]
        if op.weight_index >= 0:
            w = w[op.weight_index]
        n, k = w.shape
        assert op.ldw == k, op
        if op.kind == sp.OpKind.GATEUP:
            assert (op.tiles * sp.TILE_N * 2, op.chunks * sp.CHUNK_K // 2) == (n, k), op
        else:
            assert (op.tiles * sp.TILE_N, op.chunks * sp.CHUNK_K) == (n, k), op
        if op.norm is not None:
            assert tensors[op.norm].shape == (s.hidden,)
    # one frame streams every layer once per position, minus the q columns and the MLP position 0 skips
    layer_bytes = 2 * sum(tensors[f"code_predictor.model.layers.0.{n}"].numel() for n in (
        "self_attn.qkv_proj.weight", "self_attn.o_proj.weight", "mlp.gate_up_proj.weight", "mlp.down_proj.weight"))
    head_bytes = 2 * s.vocab * s.hidden
    q_bytes = 2 * s.heads * s.head_dim * s.hidden
    skipped = layer_bytes - (tensors["code_predictor.model.layers.0.self_attn.qkv_proj.weight"].numel() * 2 - q_bytes)
    assert program.gemm_bytes() == s.positions * s.layers * layer_bytes - skipped + (s.positions - 1) * head_bytes


def test_lowered_table(program):
    names = sp.required_params(program)
    params = {n: torch.zeros(15, 2, 3) if n.endswith("lm_head_weight") else torch.zeros(4) for n in names}
    table = sp.lower(program, params)
    assert table.dtype == torch.int64 and tuple(table.shape) == (len(program.ops), len(sp.PHASE_FIELDS))
    assert table[:, 0].tolist() == [int(op.kind) for op in program.ops]
    gemm = torch.tensor([op.kind in sp.GEMM_OPS for op in program.ops])
    assert bool((table[gemm, sp.PHASE_FIELDS.index("w")] != 0).all())
    heads = [i for i, op in enumerate(program.ops) if op.kind == sp.OpKind.LMHEAD]
    addrs = table[heads, sp.PHASE_FIELDS.index("w")]
    heads_w = params["code_predictor.lm_head_weight"]
    stride = heads_w[1].data_ptr() - heads_w[0].data_ptr()
    assert (addrs[1:] - addrs[:-1]).tolist() == [stride] * (len(heads) - 1)


def _weights(shape: sp.DepthDecoderShape, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    s = shape

    def lin(n, k):
        return (torch.randn(n, k, generator=g) / math.sqrt(k)).to(torch.bfloat16)

    def norm(n):
        return (1 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)

    w = {}
    for li in range(s.layers):
        p = f"model.layers.{li}."
        w[p + "self_attn.qkv_proj.weight"] = lin((s.heads + 2 * s.kv_heads) * s.head_dim, s.hidden)
        w[p + "self_attn.o_proj.weight"] = lin(s.hidden, s.heads * s.head_dim)
        w[p + "self_attn.q_norm.weight"] = norm(s.head_dim)
        w[p + "self_attn.k_norm.weight"] = norm(s.head_dim)
        w[p + "input_layernorm.weight"] = norm(s.hidden)
        w[p + "post_attention_layernorm.weight"] = norm(s.hidden)
        w[p + "mlp.gate_up_proj.weight"] = lin(2 * s.intermediate, s.hidden)
        w[p + "mlp.down_proj.weight"] = lin(s.hidden, s.intermediate)
    w["model.norm.weight"] = norm(s.hidden)
    w["lm_head_weight"] = torch.stack([lin(s.vocab, s.hidden) for _ in range(s.positions - 1)])
    codebook = (0.05 * torch.randn(s.positions - 1, s.vocab, s.hidden, generator=g)).to(torch.bfloat16)
    return w, codebook


def _model_style_logits(shape, w, codebook, last_hidden, embed0, forced):
    """The depth loop written the way the model writes it: one position at a time
    through every layer with a growing K/V list, float64."""
    s, f = shape, torch.float64
    W = {k: v.to(f) for k, v in w.items()}
    cb = codebook.to(f)
    rows = last_hidden.shape[0]
    half = s.head_dim // 2
    inv = 1.0 / (s.rope_theta ** (torch.arange(0, half, dtype=f) * 2 / s.head_dim))

    def rms(x, wt):
        return x * torch.rsqrt((x ** 2).mean(-1, keepdim=True) + s.eps) * wt

    def rope(x, p):
        cos, sin = torch.cat([(p * inv).cos()] * 2), torch.cat([(p * inv).sin()] * 2)
        return x * cos + torch.cat([-x[..., half:], x[..., :half]], -1) * sin

    keys = [[] for _ in range(s.layers)]
    vals = [[] for _ in range(s.layers)]
    logits = []
    qd, kd = s.heads * s.head_dim, s.kv_heads * s.head_dim
    for p in range(s.positions):
        h = last_hidden.to(f) if p == 0 else embed0.to(f) if p == 1 else cb[p - 2][forced[:, p - 1]]
        for li in range(s.layers):
            pre = f"model.layers.{li}."
            qkv = rms(h, W[pre + "input_layernorm.weight"]) @ W[pre + "self_attn.qkv_proj.weight"].T
            q = rope(rms(qkv[:, :qd].view(rows, s.heads, s.head_dim), W[pre + "self_attn.q_norm.weight"]), p)
            k = rope(rms(qkv[:, qd:qd + kd].view(rows, s.kv_heads, s.head_dim), W[pre + "self_attn.k_norm.weight"]), p)
            keys[li].append(k)
            vals[li].append(qkv[:, qd + kd:].view(rows, s.kv_heads, s.head_dim))
            kk, vv = torch.stack(keys[li], 1), torch.stack(vals[li], 1)
            qg = q.view(rows, s.kv_heads, s.heads // s.kv_heads, s.head_dim)
            att = (torch.einsum("bhgd,bjhd->bhgj", qg, kk) / math.sqrt(s.head_dim)).softmax(-1)
            h = h + torch.einsum("bhgj,bjhd->bhgd", att, vv).reshape(rows, qd) @ W[pre + "self_attn.o_proj.weight"].T
            gu = rms(h, W[pre + "post_attention_layernorm.weight"]) @ W[pre + "mlp.gate_up_proj.weight"].T
            g, u = gu[:, :s.intermediate], gu[:, s.intermediate:]
            h = h + (g * torch.sigmoid(g) * u) @ W[pre + "mlp.down_proj.weight"].T
        if p >= 1:
            logits.append(rms(h, W["model.norm.weight"]) @ W["lm_head_weight"][p - 1].T)
    return torch.stack(logits)


def test_interpret_is_the_model_loop():
    torch.manual_seed(0)
    shape = sp.DepthDecoderShape(layers=5, positions=16, hidden=1024, heads=16, kv_heads=8, head_dim=128,
                                 intermediate=3072, vocab=2048, eps=1e-6, rope_theta=1e6)
    program = sp.depth_decoder_program(shape, "code_sampler")
    w, codebook = _weights(shape)
    last_hidden = torch.randn(1, shape.hidden).to(torch.bfloat16)
    embed0 = (0.05 * torch.randn(1, shape.hidden)).to(torch.bfloat16)
    forced = torch.randint(0, shape.vocab, (1, shape.positions))
    state = sp.interpret(program, w, codebook, last_hidden, embed0, forced)
    expected = _model_style_logits(shape, w, codebook, last_hidden, embed0, forced)
    # not 1e-15: every executor reads the rope angles from one fp32 table
    assert float((state["logits"] - expected).norm() / expected.norm()) < 1e-6
    emb = embed0.double() + sum(codebook.double()[g - 1][forced[:, g]] for g in range(1, shape.positions))
    assert torch.equal(state["emb_sum"], emb)
    # rounding where the resident kernel rounds stays within bf16's reach
    rounded = sp.interpret(program, w, codebook, last_hidden, embed0, forced, kernel_rounding=True)
    assert float((rounded["logits"] - expected).norm() / expected.norm()) < 3e-2


def test_check_names_a_norm_read_before_its_sum_exists(program):
    broken = sp.StepProgram(shape=program.shape, ops=program.ops[1:], resources=program.resources,
                            ss_slots=program.ss_slots)
    assert any("before it is written" in p for p in sp.check(broken))


def test_executor_falls_back_to_the_graph_off_rocm(program):
    graph = pe.GraphExecutor(lambda inputs: None)
    config = pe.PersistentStepConfig(walk="talker_decode")
    cpu = DeviceFacts(platform="cpu")
    assert "platform cpu" in pe.PersistentExecutor.unsupported_reason(program, cpu, 27, config)
    chosen = pe.select_executor(program, cpu, 27, config, graph, make_persistent=lambda: pytest.fail("built"))
    assert chosen is graph
    assert pe.select_executor(program, cpu, 27, None, graph, make_persistent=lambda: pytest.fail("built")) is graph


def test_executor_limits_on_rocm(program):
    mi325 = DeviceFacts(platform="rocm", arch="gfx942", product="mi325x", compute_units=304, xcds=8)
    config = pe.PersistentStepConfig(walk="talker_decode")
    assert pe.PersistentExecutor.unsupported_reason(program, mi325, 27, config) is None
    assert "rows per group" in pe.PersistentExecutor.unsupported_reason(program, mi325, 200, config)
    assert "one group per XCD" in pe.PersistentExecutor.unsupported_reason(
        program, mi325, 27, pe.PersistentStepConfig(walk="talker_decode", groups=4))
    assert pe.PersistentExecutor.unsupported_reason(
        program, mi325, 27, pe.PersistentStepConfig(walk="talker_decode", groups=4, scope="agent")) is None
    mi210 = DeviceFacts(platform="rocm", arch="gfx90a", xcds=1)
    assert "arch gfx90a" in pe.PersistentExecutor.unsupported_reason(program, mi210, 27, config)
