"""Ripple on CPU, no weights: the layer's math against a plain transliteration of
``modeling_ricursive.py`` with stand-in attention, KV and RoPE resources, then the
config, the checkpoint's routing into the fused projections, and the chat prompt."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import RowParallelLinear
from mstar.model.registry import get_model_class
from mstar.model.ripple.config import ATTN_FULL, ATTN_SLIDING, KV_CACHE, ROPE, RippleConfig
from mstar.model.ripple.language_model import RippleForCausalLM

LAYER_TYPES = (["sliding_attention"] * 4 + ["full_attention"]) * 2
RAW = {
    "model_type": "ricursive", "hidden_act": "silu", "use_valres": True, "use_gated_attention": True,
    "gate_full_precision": True, "tie_word_embeddings": False,
    "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": len(LAYER_TYPES),
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "layer_types": LAYER_TYPES,
    "sliding_window": 5, "max_position_embeddings": 128, "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
    "embed_scale": 4.0, "ffn_clamp_threshold": 0.5, "eos_token_id": [1, 2],
    "auto_map": {"AutoConfig": "configuration_ricursive.RicursiveConfig"},
}
SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
# concatenates ``content``, as the checkpoint's template does
TEMPLATE = (
    "{%- for m in messages -%}{{- '<|im_start|> ' + m.role + ' ' + m.content + ' <|im_end|> ' -}}{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|> assistant ' -}}{%- endif -%}"
)


def _rope(x, theta):
    dim = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
    freqs = torch.arange(x.shape[0]).float()[:, None] * inv
    emb = torch.cat([freqs, freqs], -1)
    rot = torch.cat([-x[..., dim // 2:], x[..., :dim // 2]], -1)
    return x * emb.cos()[:, None] + rot * emb.sin()[:, None]


def _attend(q, k, v, window):
    """Causal GQA over one sequence: a query sees the last ``window`` positions, its own included."""
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    scores = torch.einsum("qhd,khd->hqk", q, k) * q.shape[-1] ** -0.5
    i = torch.arange(q.shape[0])
    visible = i[None, :] <= i[:, None]
    if window is not None:
        visible &= i[None, :] > i[:, None] - window
    return torch.einsum("hqk,khd->qhd", scores.masked_fill(~visible, float("-inf")).softmax(-1), v)


class _Attn:
    requires_kv_write = False

    def __init__(self, window):
        self.window = window

    def set_default_label(self, label):
        pass

    def set_default_layer_idx(self, idx):
        pass

    def run(self, q, kv_cache_layer=None, k=None, v=None):
        return _attend(q, k, v, self.window)


class _KV(_Attn):
    default_label = "main"

    def __init__(self):
        super().__init__(None)

    def layer_view(self):
        return None


class _Pos:
    def __init__(self, theta):
        self.theta = theta

    def apply_qk(self, q, k, label):
        return _rope(q, self.theta), _rope(k, self.theta)


def _norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _reference_logits(model: RippleForCausalLM, ids: torch.Tensor, cfg: RippleConfig):
    """``modeling_ricursive.py``'s forward, written as its modules are, from our weights."""
    lm, eps = model.model, cfg.rms_norm_eps
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    h = lm.embed_tokens.weight[ids] * cfg.embed_scale
    embeds = h
    for layer, sliding in zip(lm.layers, cfg.is_sliding, strict=True):
        attn = layer.self_attn
        q_w, k_w, g_w = attn.qkg_proj.weight.split([nh * hd, nkv * hd, nh * hd])
        x = _norm(h, layer.input_layernorm.weight, eps)
        blend = torch.sigmoid(attn.value_res_lambda[0])
        q = _norm((x @ q_w.T).view(-1, nh, hd), attn.q_norm.weight, eps)
        k = _norm((x @ k_w.T).view(-1, nkv, hd), attn.k_norm.weight, eps)
        v = (((1 - blend) * embeds + blend * x) @ attn.v_proj.weight.T).view(-1, nkv, hd)
        if sliding:
            q, k = _rope(q, cfg.rope_theta), _rope(k, cfg.rope_theta)
        a = _attend(q, k, v, cfg.sliding_window if sliding else None).reshape(-1, nh * hd)
        a = (a * torch.sigmoid(x @ g_w.T)) @ attn.o_proj.weight.T
        h = h + _norm(a, layer.post_attention_layernorm.weight, eps)
        x = _norm(h, layer.pre_feedforward_layernorm.weight, eps)
        gate, up = (x @ layer.mlp.gate_up_proj.weight.T).chunk(2, -1)
        clamp = cfg.ffn_clamp_threshold
        m = (F.silu(gate.clamp(max=clamp)) * up.clamp(-clamp, clamp)) @ layer.mlp.down_proj.weight.T
        h = h + _norm(m, layer.post_feedforward_layernorm.weight, eps)
    return _norm(h, lm.norm.weight, eps) @ model.lm_head.weight.T


class _SumOfOneGroup(CommGroup):
    """Two ranks whose partial sums are this one's and zeros: the reduce is the
    identity, and each call is counted, the norm-fused ones apart."""

    def __init__(self):
        super().__init__(my_global_rank=0, my_group_rank=0, group_members=[0, 1])
        self.reduces = 0
        self.norm_reduces = 0

    def all_reduce(self, input_):
        self.reduces += 1
        return input_

    def allreduce_rmsnorm(self, *args, **kwargs):
        self.norm_reduces += 1
        return super().allreduce_rmsnorm(*args, **kwargs)


@pytest.mark.parametrize("tp", [1, 2])
def test_the_text_stack_matches_the_reference_layer(model_dir, tp):
    torch.manual_seed(0)
    cfg = RippleConfig.from_dict(RAW)
    model = RippleForCausalLM(cfg).double()
    group = _SumOfOneGroup() if tp == 2 else None
    if group is not None:
        # every reduce goes through the group: the post-norms' and any the
        # row-parallel projections would make themselves
        model.model.comm_group = group
        for m in model.modules():
            if isinstance(m, RowParallelLinear):
                m.comm_group, m.tp_size = group, group.world_size
    # the stand-ins take the window and theta the model declares, so a wrong one fails against the reference
    ripple = get_model_class("ripple_65b")(str(model_dir))
    specs = {spec.resource_key: spec for spec in ripple.get_node_resources()}
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    resources = {
        KV_CACHE: _KV(), ATTN_SLIDING: _Attn(specs[ATTN_SLIDING].config.sliding_window),
        ATTN_FULL: _Attn(specs[ATTN_FULL].config.sliding_window), ROPE: _Pos(specs[ROPE].config.rope_theta),
    }
    for module in model.modules():
        if hasattr(module, "bind_resources"):
            module.bind_resources(resources)
    # long enough that the 5-token window cuts the sliding layers off
    ids = torch.randint(0, cfg.vocab_size, (13,))
    ours = model.logits(model(model.embed(ids), label="main"))
    ref = _reference_logits(model, ids, cfg)
    # the gate's sigmoid and the norms compute in fp32; the rest is fp64
    assert torch.allclose(ours, ref, atol=1e-4, rtol=1e-4), (ours - ref).abs().max()
    if group is not None:
        # once a projection: in its post-norm, where the fused kernel runs, not also in the linear
        assert group.norm_reduces == group.reduces == 2 * cfg.num_hidden_layers


@pytest.mark.parametrize("change", [
    {"use_valres": False}, {"use_gated_attention": False}, {"gate_full_precision": False},
    {"tie_word_embeddings": True}, {"embed_scale": None}, {"ffn_clamp_threshold": None},
    {"quantization_config": {"quant_method": "modelopt"}}, {"layer_types": ["full_attention"] * 10},
], ids=lambda change: next(iter(change)))
def test_a_checkpoint_the_layer_would_miscompute_is_refused(change):
    with pytest.raises(ValueError):
        RippleConfig.from_dict({**RAW, **change})


@pytest.fixture
def model_dir(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {t: i for i, t in enumerate(SPECIALS)}
    for word in ("system", "user", "assistant", "be", "brief", "hi", "[UNK]"):
        vocab[word] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens(SPECIALS)
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|im_end|>",
    }))
    (tmp_path / "chat_template.jinja").write_text(TEMPLATE)
    (tmp_path / "config.json").write_text(json.dumps(RAW))
    return tmp_path


def test_a_chat_through_the_adapter_keeps_every_turn(model_dir, monkeypatch):
    from mstar.api_server.openai.adapters import get_adapter
    from mstar.api_server.openai.protocol import ChatCompletionRequest

    # the checkpoint's auto_map must not stop a server at transformers' prompt
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("prompted to run the checkpoint's code"))
    model = get_model_class("ripple_65b")(str(model_dir))
    request = get_adapter("ripple_65b").chat_to_request(ChatCompletionRequest(model="ripple_65b", messages=[
        {"role": "developer", "content": "be brief"}, {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "be"}, {"role": "user", "content": "brief"},
    ]), model_dir)
    prompt = model.process_prompt(
        request.text, request.input_modalities, request.output_modalities, prompt_parts=request.prompt_parts,
    )
    assert model.tokenizer.decode(prompt.new_input_tensors["text_inputs"][0]) == (
        "<|im_start|> system be brief <|im_end|> <|im_start|> user hi <|im_end|> "
        "<|im_start|> assistant be <|im_end|> <|im_start|> user brief <|im_end|> <|im_start|> assistant"
    )
    assert set(model.text_output().stop_token_ids) == {1, 2}
    # the template has no thinking switch, so a request's enable_thinking is logged as ignored, not dropped silently
    assert "enable_thinking" not in model.request_kwargs()


@pytest.mark.parametrize("rank", [0, 1])
def test_the_checkpoint_loads_into_this_rank_s_heads(model_dir, rank):
    cfg = RippleConfig.from_dict(RAW)
    q, kv = cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
    hidden, inter = cfg.hidden_size, cfg.intermediate_size
    weights = {
        "model.embed_tokens.weight": (cfg.vocab_size, hidden), "lm_head.weight": (cfg.vocab_size, hidden),
        "model.norm.weight": (hidden,),
    }
    for i in range(cfg.num_hidden_layers):
        weights |= {f"model.layers.{i}.{name}": shape for name, shape in {
            "input_layernorm.weight": (hidden,), "post_attention_layernorm.weight": (hidden,),
            "pre_feedforward_layernorm.weight": (hidden,), "post_feedforward_layernorm.weight": (hidden,),
            "self_attn.q_proj.weight": (q, hidden), "self_attn.k_proj.weight": (kv, hidden),
            "self_attn.v_proj.weight": (kv, hidden), "self_attn.g_proj.weight": (q, hidden),
            "self_attn.o_proj.weight": (hidden, q), "self_attn.q_norm.weight": (cfg.head_dim,),
            "self_attn.k_norm.weight": (cfg.head_dim,), "self_attn.value_res_lambda": (1,),
            "mlp.gate_proj.weight": (inter, hidden), "mlp.up_proj.weight": (inter, hidden),
            "mlp.down_proj.weight": (hidden, inter),
        }.items()}
    tensors = {name: torch.randn(shape) for name, shape in weights.items()}
    save_file(tensors, str(model_dir / "model.safetensors"))

    group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
    model = get_model_class("ripple_65b")(str(model_dir))
    attn = model._create_submodule("LLM", "cpu", torch.float32, group).model.model.layers[0].self_attn

    def rows(name, width):
        half = width // 2
        return tensors[f"model.layers.0.self_attn.{name}.weight"][rank * half:(rank + 1) * half]

    assert torch.equal(attn.qkg_proj.weight, torch.cat([rows("q_proj", q), rows("k_proj", kv), rows("g_proj", q)]))
    assert torch.equal(attn.v_proj.weight, rows("v_proj", kv))
