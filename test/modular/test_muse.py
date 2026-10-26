"""Muse Glimmer on CPU, no weights: config, layer math, vision index helpers,
weight routing, and the prompt's pad expansion.

The text stack runs against a plain transliteration of the reference's layer
(sandwich norms, gated attention, NoPE global layers, softcapped logits) with
stand-in attention, KV and RoPE resources, which checks the fused-residual
loop and the norm placement. The vision helpers are checked against
``F.grid_sample`` and against the reference's own index algorithms, ported
here verbatim.
"""

from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F

from mstar.distributed.communication import CommGroup
from mstar.model.components import MoeParallelConfig
from mstar.model.loader.base import _apply_stacked
from mstar.model.muse.components.text import (
    GatedQKVParallelLinear,
    MuseGlimmerForCausalLM,
)
from mstar.model.muse.components.vision import (
    _interpolation,
    _merge_order,
    tower_inputs,
)
from mstar.model.muse.config import (
    ATTN_FULL,
    ATTN_SLIDING,
    KV_CACHE,
    ROPE,
    MuseGlimmerConfig,
    MuseGlimmerVisionConfig,
)
from mstar.model.muse.weight_loader import (
    _STACKED_PARAMS,
    text_name_remapper,
    vision_name_remapper,
)

N_LAYERS = 8


def unit_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """The reference's weightless RMSNorm, in fp32 (the test's own copy)."""
    xf = x.float()
    return (xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)).to(x.dtype)
SLIDING = ["sliding_attention"] * 3 + ["full_attention"]


def _raw_config(layers: int = N_LAYERS) -> dict:
    types = (SLIDING * layers)[:layers]
    return {
        "image_token_id": 9, "video_token_id": 8,
        "out_hidden_size": 64, "projector_hidden_size": 32,
        "text_config": {
            "num_hidden_layers": layers, "hidden_size": 64, "intermediate_size": 96,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "vocab_size": 64, "layer_types": types,
            "layer_rope_theta": [5e5 if t == "sliding_attention" else 0 for t in types],
            "sliding_window": 5, "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5, "post_norm_eps": 1e-8, "qk_scale_factor": 3.87,
            "output_multiplier": 0.25, "final_logit_softcapping": 20.0,
            "bos_token_id": 0, "eos_token_id": 1,
            "rope_parameters": {"rope_theta": 5e5, "rope_type": "default"},
        },
        "vision_config": {
            "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2,
            "num_hidden_layers": 4, "patch_size": 14, "patch_temporal": 2, "merge_size": 2,
            "pos_emb_height": 4, "pos_emb_width": 4,
            "layer_types": ["window_attention"] * 3 + ["full_attention"],
            "rope_parameters": {"rope_theta": 10000.0},
        },
    }


@pytest.fixture
def ckpt_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_raw_config()))
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "eos_token_id": [1, 7], "temperature": 1.0, "top_p": 0.95, "top_k": 64,
    }))
    return tmp_path


# --- config -------------------------------------------------------------------


def _write(tmp_path, raw):
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return tmp_path


@pytest.mark.parametrize("key", ["num_experts", "n_routed_experts", "moe_intermediate_size"])
def test_a_mixture_of_experts_config_is_refused(tmp_path, key):
    raw = _raw_config()
    raw["text_config"][key] = 8
    with pytest.raises(ValueError, match="MoE"):
        MuseGlimmerConfig.from_hf(_write(tmp_path, raw))


@pytest.mark.parametrize("change", ["rotated full layer", "other theta"])
def test_rope_off_the_sliding_layers_or_at_another_theta_is_refused(tmp_path, change):
    raw = _raw_config()
    thetas = raw["text_config"]["layer_rope_theta"]
    if change == "rotated full layer":
        thetas[3] = 5e5
    else:
        thetas[0] = 1e4
    with pytest.raises(ValueError, match="rope theta"):
        MuseGlimmerConfig.from_hf(_write(tmp_path, raw))


@pytest.mark.parametrize("change, match", [
    ({"projector_hidden_act": "silu"}, "adapter's activation"),
    # the 30B's name for it
    ({"text_config": {"hidden_activation": "gelu"}}, "MLP's activation"),
    ({"text_config": {"layer_types": ["sliding_attention"] * N_LAYERS, "layer_rope_theta": [5e5] * N_LAYERS}},
     "sliding and full attention layers, both"),
    ({"text_config": {"rope_parameters": {"rope_theta": 5e5, "rope_type": "yarn", "factor": 8.0}}},
     "rope is a default one"),
    ({"vision_config": {"layer_types": ["window_attention"] * 3 + ["sliding_attention"]}},
     "window and full attention layers"),
    ({"text_config": {"num_hidden_layers": N_LAYERS + 1}}, "layer_types names"),
], ids=["projector act", "mlp act", "all sliding", "scaled rope", "vision layer kind", "layer count"])
def test_a_checkpoint_fact_the_port_builds_in_is_checked(tmp_path, change, match):
    raw = _raw_config()
    for key, value in change.items():
        raw[key] = {**raw[key], **value} if isinstance(value, dict) else value
    path = _write(tmp_path, raw)
    with pytest.raises(ValueError, match=match):
        MuseGlimmerVisionConfig.from_hf(path)
        MuseGlimmerConfig.from_hf(path)


def test_the_root_adapter_and_image_token_fields_are_required(tmp_path):
    for key in ("out_hidden_size", "projector_hidden_size", "image_token_id"):
        raw = _raw_config()
        del raw[key]
        with pytest.raises(KeyError, match=key):
            (MuseGlimmerVisionConfig if key != "image_token_id" else MuseGlimmerConfig).from_hf(
                _write(tmp_path, raw)
            )


def test_without_a_softcap_or_multiplier_the_logits_are_the_head_s(tmp_path):
    raw = _raw_config()
    for key in ("final_logit_softcapping", "output_multiplier", "qk_scale_factor"):
        del raw["text_config"][key]
    cfg = MuseGlimmerConfig.from_hf(_write(tmp_path, raw))
    assert cfg.attn_scale == pytest.approx(16 ** -0.5)
    lm = MuseGlimmerForCausalLM(cfg)
    torch.nn.init.normal_(lm.lm_head.weight)
    x = torch.randn(3, 64)
    torch.testing.assert_close(lm.logits(x), lm.lm_head(x).float())


def test_config_reads_the_checkpoint(ckpt_dir):
    cfg = MuseGlimmerConfig.from_hf(ckpt_dir)
    assert cfg.is_sliding == [True, True, True, False] * 2
    assert cfg.sliding_window == 5
    assert cfg.attn_scale == pytest.approx(3.87 / 4)
    # the tokenizer's eos and generation_config's <|eot|>
    assert cfg.stop_token_ids == {1, 7}
    vision = MuseGlimmerVisionConfig.from_hf(ckpt_dir)
    assert (vision.head_dim, vision.window_patches, vision.text_hidden_size) == (8, 4, 64)


# --- text stack against the reference layer -------------------------------------


def _rope(x, pos, theta):
    dim = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
    freqs = pos[:, None].float() * inv
    emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos()[:, None], emb.sin()[:, None]
    half = dim // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * cos + rot * sin


def _attend(q, k, v, scale, window_left):
    """Causal GQA over one sequence, optionally windowed."""
    n = q.shape[0]
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    i = torch.arange(n)
    visible = i[None, :] <= i[:, None]
    if window_left >= 0:
        visible &= i[None, :] >= i[:, None] - window_left
    scores = scores.masked_fill(~visible, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float()).to(q.dtype)


class _Attn:
    requires_kv_write = False

    def __init__(self, scale, window_left):
        self.scale, self.window_left = scale, window_left

    def set_default_label(self, label):
        pass

    def set_default_layer_idx(self, idx):
        pass

    def run(self, q, kv_cache_layer=None, k=None, v=None):
        return _attend(q, k, v, self.scale, self.window_left)


class _KV:
    default_label = "main"

    def set_default_label(self, label):
        pass

    def set_default_layer_idx(self, idx):
        pass

    def layer_view(self):
        return None


class _Pos:
    def __init__(self, theta):
        self.theta = theta

    def apply_qk(self, q, k, label):
        pos = torch.arange(q.shape[0])
        return _rope(q, pos, self.theta), _rope(k, pos, self.theta)


def _gemma(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1 + w.float())).to(x.dtype)


def _reference_logits(model: MuseGlimmerForCausalLM, ids: torch.Tensor, cfg: MuseGlimmerConfig):
    """The reference's forward, written as its modules are, from our weights."""
    lm = model.model
    h = unit_rms(lm.embed_tokens.weight[ids], cfg.rms_norm_eps)
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    for i, layer in enumerate(lm.layers):
        w = layer.self_attn.qkvg_proj.weight
        q_w, g_w, k_w, v_w = w.split([nh * hd, nh * hd, nkv * hd, nkv * hd])
        x = _gemma(h, layer.input_layernorm.weight, cfg.rms_norm_eps)
        q = unit_rms((x @ q_w.T).view(-1, nh, hd), cfg.rms_norm_eps) * cfg.qk_scale_factor
        k = unit_rms((x @ k_w.T).view(-1, nkv, hd), cfg.rms_norm_eps)
        v = (x @ v_w.T).view(-1, nkv, hd)
        if cfg.layer_rope_theta[i]:
            pos = torch.arange(ids.shape[0])
            q, k = _rope(q, pos, cfg.rope_theta), _rope(k, pos, cfg.rope_theta)
        window = cfg.sliding_window - 1 if cfg.is_sliding[i] else -1
        a = _attend(q, k, v, hd ** -0.5, window).reshape(-1, nh * hd)
        a = (a * torch.sigmoid(x @ g_w.T)) @ layer.self_attn.o_proj.weight.T
        h = h + _gemma(a, layer.post_attention_layernorm.weight, cfg.post_norm_eps)
        x = _gemma(h, layer.pre_feedforward_layernorm.weight, cfg.rms_norm_eps)
        gate, up = (x @ layer.mlp.gate_up_proj.weight.T).chunk(2, -1)
        m = (F.silu(gate) * up) @ layer.mlp.down_proj.weight.T
        h = h + _gemma(m, layer.post_feedforward_layernorm.weight, cfg.post_norm_eps)
    hf = h.float()
    h = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * lm.norm.weight).to(h.dtype)
    logits = (h @ model.lm_head.weight.T) * cfg.output_multiplier
    return cfg.final_logit_softcapping * torch.tanh(logits.float() / cfg.final_logit_softcapping)


def test_the_text_stack_matches_the_reference_layer(ckpt_dir):
    torch.manual_seed(0)
    cfg = MuseGlimmerConfig.from_hf(ckpt_dir)
    model = MuseGlimmerForCausalLM(cfg).double()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    resources = {
        KV_CACHE: _KV(),
        ATTN_SLIDING: _Attn(cfg.attn_scale, cfg.sliding_window - 1),
        ATTN_FULL: _Attn(cfg.attn_scale, -1),
        ROPE: _Pos(cfg.rope_theta),
    }
    for module in model.modules():
        bind = getattr(module, "bind_resources", None)
        if callable(bind):
            bind(resources)
    # long enough that the 5-token window cuts the sliding layers off
    ids = torch.randint(0, cfg.vocab_size, (13,))
    hidden = model.model(model.model.embed(ids), label="main")
    ours = model.logits(hidden)
    ref = _reference_logits(model, ids, cfg)
    # the norms compute in fp32 on both sides; the rest is fp64
    assert torch.allclose(ours, ref, atol=1e-4, rtol=1e-4), (ours - ref).abs().max()


# --- vision index helpers ----------------------------------------------------------


def _vision_config():
    return MuseGlimmerVisionConfig(
        hidden_size=16, intermediate_size=32, num_attention_heads=2, num_hidden_layers=4,
        patch_size=14, patch_temporal=2, merge_size=2, pos_emb_height=4, pos_emb_width=4,
        layer_types=["window_attention"] * 3 + ["full_attention"],
        out_hidden_size=64, projector_hidden_size=32, text_hidden_size=64,
    )


GRIDS = [[(1, 6, 10)], [(1, 4, 4), (2, 8, 2)], [(1, 12, 8)]]


@pytest.mark.parametrize("grid", GRIDS, ids=str)
def test_the_position_resample_matches_grid_sample(grid):
    side = 4
    table = torch.randn(side * side, 3, dtype=torch.float64)
    idx, w = _interpolation(grid, side)
    ours = (table[idx] * w[:, :, None].double()).sum(1)
    refs = []
    for t, h, wd in grid:
        img = table.T.reshape(1, 3, side, side)
        ys = (torch.arange(h, dtype=torch.float64) + 0.5) / h * 2 - 1
        xs = (torch.arange(wd, dtype=torch.float64) + 0.5) / wd * 2 - 1
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        sample = F.grid_sample(
            img, torch.stack([gx, gy], -1)[None], mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )
        refs.append(sample[0].reshape(3, -1).T.repeat(t, 1))
    assert torch.allclose(ours, torch.cat(refs), atol=1e-6)


def _reference_window_index(grid, window):
    """transformers' `get_vision_window_index` at spatial_merge_size 1."""
    window_index, cu, offset = [], [0], 0
    for t, h, w in grid:
        index = torch.arange(t * h * w).reshape(t, h, w)
        pad_h, pad_w = window - h % window, window - w % window
        nh, nw = (h + pad_h) // window, (w + pad_w) // window
        padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        padded = padded.reshape(t, nh, window, nw, window).permute(0, 1, 3, 2, 4)
        padded = padded.reshape(t, nh * nw, window, window)
        lens = (padded != -100).sum([2, 3]).reshape(-1)
        flat = padded.reshape(-1)
        window_index.append(flat[flat != -100] + offset)
        cu.extend((lens.cumsum(0) + cu[-1]).tolist())
        offset += t * h * w
    cu = torch.unique_consecutive(torch.tensor(cu))
    return torch.cat(window_index), tuple(torch.diff(cu).tolist())


def _reference_shuffle_index(grid, merge):
    """transformers' `get_vision_pixel_shuffle_index`."""
    out, offset = [], 0
    for t, h, w in grid:
        perm = torch.arange(h * w).view(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).reshape(-1)
        if t > 1:
            perm = (perm[None] + (torch.arange(t) * h * w)[:, None]).reshape(-1)
        out.append(perm + offset)
        offset += t * h * w
    return torch.cat(out)


@pytest.mark.parametrize("grid", GRIDS, ids=str)
def test_window_and_output_order_match_the_reference(grid):
    config = _vision_config()
    ti = tower_inputs(grid, config)
    ref_index, ref_lengths = _reference_window_index(grid, config.window_patches)
    assert torch.equal(ti.window_index, ref_index)
    assert ti.window_lengths == ref_lengths
    assert ti.frame_lengths == tuple(h * w for t, h, w in grid for _ in range(t))
    # the reference un-permutes with argsort, then shuffles
    x = torch.arange(len(ref_index))[ref_index]
    assert torch.equal(x[ti.output_index], x[torch.argsort(ref_index)][_reference_shuffle_index(grid, 2)])
    assert torch.equal(_merge_order(grid, 2), _reference_shuffle_index(grid, 2))
    # (w + 1, h + 1) in window order
    h0, w0 = grid[0][1], grid[0][2]
    first = ti.rope_positions[ti.window_index.argsort()][: h0 * w0]
    assert torch.equal(first[w0 + 1], torch.tensor([2, 2]))


def test_the_tower_s_indices_are_built_on_the_cpu():
    """`prepare_inputs` runs on the worker's main thread, where a tensor
    built on the device waits for the step in flight; `preprocess` moves them."""
    from types import SimpleNamespace

    from mstar.model.multimodal import MEDIA_LAYOUT
    from mstar.model.muse.submodules import VisionEncoderSubmodule

    encoder = object.__new__(VisionEncoderSubmodule)
    encoder.config = _vision_config()
    encoder.get_device = lambda: torch.device("meta")
    encoder.output = "vision_embeds"
    layout = {"offset": [1], "tokens": [6], "output": ["vision_embeds"], "grid": [[1, 4, 6]]}
    fwd_info = SimpleNamespace(step_metadata={MEDIA_LAYOUT: layout})
    out = encoder.prepare_inputs("prefill_vision", fwd_info, {"pixel_values": [torch.zeros(24, 1176)]})
    assert {t.device.type for t in out.tensor_inputs.values()} == {"cpu"}


def test_the_vision_rope_frequencies_survive_a_cast_to_bf16():
    from mstar.model.muse.components.vision import MuseGlimmerVisionModel

    rope = MuseGlimmerVisionModel(_vision_config()).rotary_emb
    exact = rope.inv_freq.clone()
    rope.to(torch.bfloat16)
    assert rope.inv_freq.dtype == torch.float32
    assert torch.equal(rope.inv_freq, exact)


# --- weights ------------------------------------------------------------------------


def _checkpoint_text_names(layers: int):
    names = ["lm_head.weight", "model.language_model.embed_tokens.weight", "model.language_model.norm.weight"]
    for i in range(layers):
        p = f"model.language_model.layers.{i}."
        names += [p + s + ".weight" for s in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        )]
    return names


def test_every_text_parameter_has_a_checkpoint_tensor(ckpt_dir):
    with torch.device("meta"):
        model = MuseGlimmerForCausalLM(MuseGlimmerConfig.from_hf(ckpt_dir))
    targets = {}
    for name in _checkpoint_text_names(N_LAYERS):
        target, shard = _apply_stacked(text_name_remapper(name), _STACKED_PARAMS)
        targets.setdefault(target, set()).add(shard)
    params = {n for n, _ in model.named_parameters()}
    assert params == set(targets)
    # the attention gate does not land in the MLP's gate_up
    assert targets["model.layers.0.self_attn.qkvg_proj.weight"] == {"q", "gate", "k", "v"}
    assert targets["model.layers.0.mlp.gate_up_proj.weight"] == {0, 1}


def test_every_vision_parameter_has_a_checkpoint_tensor():
    from mstar.model.muse.components.vision import MuseGlimmerVisionModel

    config = _vision_config()
    with torch.device("meta"):
        tower = MuseGlimmerVisionModel(config)
    names = [
        "model.vision_tower.patch_embedder.patch_embedding.weight",
        "model.vision_tower.patch_embedder.position_embedding_table.weight",
        "model.vision_adapter.fc1.weight", "model.vision_adapter.fc2.weight",
        "model.vision_projection.weight",
    ] + [f"model.vision_tower.{n}.{p}" for n in ("ln_pre", "ln_post") for p in ("weight", "bias")]
    for i in range(config.num_hidden_layers):
        for mod in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.proj",
                    "mlp.fc1", "mlp.fc2", "norm1", "norm2"):
            names += [f"model.vision_tower.layers.{i}.{mod}.{p}" for p in ("weight", "bias")]
    assert {vision_name_remapper(n) for n in names} == {n for n, _ in tower.named_parameters()}
    assert vision_name_remapper("model.language_model.norm.weight") is None


def test_a_checkpoint_tensor_with_no_parameter_here_is_refused(tmp_path):
    """A weighted norm or a bias this port lacks would otherwise be dropped,
    and the tower would run without it."""
    from safetensors.torch import save_file

    from mstar.model.muse.components.vision import MuseGlimmerVisionModel
    from mstar.model.muse.weight_loader import load_vision_weights

    tower = MuseGlimmerVisionModel(_vision_config())
    prefixes = {
        "adapter_fc1.": "model.vision_adapter.fc1.", "adapter_fc2.": "model.vision_adapter.fc2.",
        "projection.": "model.vision_projection.",
    }

    def checkpoint_name(ours: str) -> str:
        for prefix, theirs in prefixes.items():
            if ours.startswith(prefix):
                return theirs + ours[len(prefix):]
        return "model.vision_tower." + ours

    tensors = {checkpoint_name(n): p.detach().clone() for n, p in tower.named_parameters()}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    load_vision_weights(tower, tmp_path)
    tensors["model.vision_tower.layers.0.attn.q_norm.weight"] = torch.ones(8)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="no parameter here"):
        load_vision_weights(tower, tmp_path)


@pytest.mark.parametrize("rank", [0, 1])
def test_the_fused_gate_is_sharded_with_its_queries(rank):
    heads, kv_heads, hd, hidden = 4, 2, 3, 5
    layer = GatedQKVParallelLinear(
        CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1]),
        hidden_size=hidden, head_size=hd, total_num_heads=heads, total_num_kv_heads=kv_heads,
    )
    sizes = {"q": heads, "gate": heads, "k": kv_heads, "v": kv_heads}
    full = {s: torch.randn(n * hd, hidden) for s, n in sizes.items()}
    for shard, w in full.items():
        layer.weight_loader(layer.weight, w, shard)
    q, g, k, v = layer.weight.split([2 * hd, 2 * hd, hd, hd])
    rows = slice(rank * 2 * hd, (rank + 1) * 2 * hd)
    assert torch.equal(q, full["q"][rows])
    assert torch.equal(g, full["gate"][rows])
    assert torch.equal(k, full["k"][rank * hd:(rank + 1) * hd])
    assert torch.equal(v, full["v"][rank * hd:(rank + 1) * hd])


# --- prompt ----------------------------------------------------------------------------

SPECIALS = [
    "<|begin_of_text|>", "<|end_of_text|>", "<|start|>", "<|message|>", "<|eot|>",
    "<|image_start|>", "<|image_end|>", "<|eom|>", "<|video|>", "<|patch|>",
]
TEMPLATE = (
    "{{- bos_token -}}<|start|>system<|message|>sys {{ reasoning_strength | default('high') }}<|eot|>"
    "{%- for m in messages -%}<|start|>{{ m.role }}<|message|>"
    "{%- for p in m.content -%}{%- if p.type == 'image' -%}<|patch|>"
    "{%- else -%}{{ p.text }}{%- endif -%}{%- endfor -%}<|eot|>{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|start|>assistant{%- endif -%}"
)


@pytest.fixture
def model_dir(ckpt_dir):
    return _write_tokenizer(ckpt_dir, SPECIALS, TEMPLATE)


def _write_tokenizer(ckpt_dir, specials, template):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {t: i for i, t in enumerate(specials)}
    for word in ("system", "user", "assistant", "sys", "high", "low", "Describe", "this", "and", "[UNK]"):
        vocab[word] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens(specials)
    # as the checkpoint's: the post-processor prepends BOS by itself
    tok.post_processor = processors.TemplateProcessing(
        single="<|begin_of_text|> $A", special_tokens=[("<|begin_of_text|>", 0)],
    )
    tok.save(str(ckpt_dir / "tokenizer.json"))
    (ckpt_dir / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>",
    }))
    (ckpt_dir / "chat_template.jinja").write_text(template)
    return ckpt_dir


@pytest.mark.parametrize("processor, match", [
    ({"image_processor": {"merge_size": 3}}, "processor_config.json's merge_size"),
    ({"image_processor": {"patch_size": 16}}, "processor_config.json's patch_size"),
    # flat, not under "image_processor"
    ({"temporal_patch_size": 1}, "processor_config.json's temporal_patch_size"),
    # a renamed token cap would fall back to 4096
    ({"image_processor": {"max_tokens": 1024}}, "processor_config.json's max_tokens"),
    # no processor change: a tokenizer without <|image_start|>
    (None, r"tokenizer has no \['<\|nope\|>'\]"),
], ids=str)
def test_a_checkpoint_intake_would_misread_is_refused(model_dir, monkeypatch, processor, match):
    from mstar.model.muse import glimmer_model

    if processor is None:
        monkeypatch.setattr(glimmer_model, "IMAGE_START", "<|nope|>")
    else:
        (model_dir / "processor_config.json").write_text(json.dumps(processor))
    with pytest.raises(ValueError, match=match):
        glimmer_model.MuseGlimmerModel(str(model_dir))


def test_a_processor_config_naming_its_classes_is_read(model_dir):
    """``processor_class`` and ``auto_map`` name the transformers classes, not
    a setting intake builds in."""
    from mstar.model.muse import glimmer_model

    (model_dir / "processor_config.json").write_text(json.dumps({
        "processor_class": "MuseGlimmerProcessor", "auto_map": {"AutoProcessor": "x.MuseGlimmerProcessor"},
        "max_image_tokens": 1024,
    }))
    assert glimmer_model.MuseGlimmerModel(str(model_dir))._processor["max_tokens"] == 1024


# `enable_thinking: false`, which the chat clients send, is the template's "low"
@pytest.mark.parametrize("low", [{"reasoning_effort": "low"}, {"enable_thinking": False}], ids=str)
def test_each_image_expands_to_its_tokens_with_one_bos(model_dir, low):
    """The template's one ``<|patch|>`` becomes ``<|image_start|>``, a pad per
    image token and ``<|image_end|>``, as the reference processor writes it."""
    from mstar.model.multimodal import MEDIA_LAYOUT
    from mstar.model.muse.glimmer_model import MuseGlimmerModel

    model = MuseGlimmerModel(str(model_dir))
    image = torch.rand(3, 30, 50)
    out = model.process_prompt(
        "Describe this", ["text", "image", "text"], ["text"],
        tensors={"image_inputs": [image]},
        prompt_parts=None, **low,
    )
    layout = out.metadata[MEDIA_LAYOUT]
    out = out.new_input_tensors
    (ids,) = out["text_inputs"]
    ids = ids.tolist()
    assert ids[0] == 0 and ids[1] != 0, "one BOS, not the template's plus the post-processor's"
    assert model.tokenizer.convert_tokens_to_ids("low") in ids
    ((t, h, w),) = layout["grid"]
    assert (t, h % 2, w % 2) == (1, 0, 0)
    assert "image_grid_thw" not in out
    assert out["pixel_values"][0].shape == (h * w, 2 * 3 * 14 * 14)
    (offset,), (n,) = layout["offset"], layout["tokens"]
    assert n == h * w // 4 and ids.count(9) == n
    assert ids[offset - 1:offset + n + 1] == [5] + [9] * n + [6]

    signals = {k: list(v) for k, v in out.items()}
    (step,) = model._prefill_schedule(["text", "image", "text"], signals, layout)
    assert step.walk == "prefill_vision" and step.metadata == {MEDIA_LAYOUT: layout}


def test_a_spelled_out_patch_is_a_bad_request(model_dir):
    from mstar.model.muse.glimmer_model import MuseGlimmerModel

    model = MuseGlimmerModel(str(model_dir))
    with pytest.raises(ValueError, match="placement mismatch"):
        model.process_prompt(
            "Describe <|patch|>", ["text", "image"], ["text"],
            tensors={"image_inputs": [torch.rand(3, 30, 50)]},
        )


def test_the_checkpoint_sampling_defaults_apply_under_the_request(model_dir):
    from mstar.model.muse.config import SAMPLER
    from mstar.model.muse.glimmer_model import MuseGlimmerModel

    model = MuseGlimmerModel(str(model_dir))
    # the sampler has no min-p filter, so a min_p it passed on would be refused at ingest
    cfg = model.get_request_resource_configs({}, {"temperature": 0.0, "min_p": 0.1})[SAMPLER]
    assert (cfg.temperature, cfg.top_p, cfg.top_k) == (0.0, 0.95, 64)
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    cfg.validate(specs[SAMPLER])
    assert set(specs) == {KV_CACHE, ATTN_SLIDING, ATTN_FULL, ROPE, "sampler", "vision_attn"}
    assert set(model.get_graph_walk_graphs()) == {"prefill_text", "prefill_vision", "decode"}


def test_smart_resize_matches_the_reference_example():
    from mstar.model.muse.processing import smart_resize

    # transformers' integration test: COCO 39769 (640x480) -> grid [1, 34, 46]
    h, w = smart_resize(480, 640, 28, 4096)
    assert (h // 14, w // 14) == (34, 46)
    # the cap holds for any aspect ratio, a 1 x 89M-pixel image's too
    for size in ((4000, 3000), (1, 89_000_000), (89_000_000, 1)):
        assert math.prod(smart_resize(*size, 28, 4096)) // 28 ** 2 <= 4096


# --- Muse Spark Open ---------------------------------------------------------------


def _spark_raw(layers: int = 4) -> dict:
    raw = _raw_config(layers)
    raw["model_type"] = "muse_spark"
    raw["audio_token_id"] = 7
    raw["vision_config"]["model_type"] = "muse_spark_vision"
    raw["vision_config"]["rope_parameters"] = {"rope_theta": 10000.0, "rope_type": "axial"}
    raw["text_config"].update({
        "num_local_experts": 8, "num_experts_per_tok": 2, "moe_hidden_size": 32,
        "moe_intermediate_size": 48, "routed_scaling_factor": 1.0,
        "residual_gate_temperature": 0.3, "mlp_layer_types": ["sparse"] * layers,
        "num_shared_experts": 0,
    })
    raw["audio_config"] = {
        "patch_sizes": [4, 2], "hidden_sizes": [8, 16], "depths": [1, 1], "num_hidden_layers": 2,
        "out_hidden_size": 64, "conv_kernel_size": 3, "mlp_ratio": 2.0, "rms_norm_eps": 1e-6,
        "hidden_act": "gelu", "initializer_range": 0.02, "model_type": "muse_spark_audio",
    }
    return raw


def test_the_spark_config_reads_the_checkpoint(tmp_path):
    from mstar.model.muse.config import MuseSparkConfig

    config = MuseSparkConfig.from_hf(_write(tmp_path, _spark_raw()))
    assert config.audio_token_id == 7
    assert (config.num_local_experts, config.num_experts_per_tok, config.moe_hidden_size) == (8, 2, 32)
    assert MuseGlimmerVisionConfig.from_hf(tmp_path).pos_table_xy
    # the dense Glimmer entry refuses it rather than dropping the experts
    with pytest.raises(ValueError, match="dense Muse Glimmer"):
        MuseGlimmerConfig.from_hf(tmp_path)


@pytest.mark.parametrize(("change", "match"), [
    ({"mlp_layer_types": ["dense", "sparse", "sparse", "sparse"]}, "dense"),
    ({"num_shared_experts": 1}, "shared experts"),
    ({"n_group": 4}, "expert groups"),
    # the fused experts run SiLU whatever the config says
    ({"hidden_act": "gelu"}, "silu"),
    ({"output_multiplier": None}, "output_multiplier"),
])
def test_a_spark_config_this_port_would_run_wrong_is_refused(tmp_path, change, match):
    from mstar.model.muse.config import MuseSparkConfig

    raw = _spark_raw()
    raw["text_config"].update(change)
    with pytest.raises(ValueError, match=match):
        MuseSparkConfig.from_hf(_write(tmp_path, raw))


def _checkpoint_spark_text_names(layers: int):
    """The released index's text keys (model.safetensors.index.json)."""
    names = ["lm_head.weight", "model.language_model.embed_tokens.weight", "model.language_model.norm.weight"]
    for i in range(layers):
        p = f"model.language_model.layers.{i}."
        names += [p + s for s in (
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.gate_proj.weight", "self_attn.o_proj.weight",
            "input_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
            "post_attention_residual_gate.gate", "post_feedforward_residual_gate.gate",
            "mlp.gate.weight", "mlp.gate.e_score_correction_bias",
            "mlp.pre_expert_proj.weight", "mlp.pre_expert_norm.weight", "mlp.post_expert_proj.weight",
            "mlp.experts.gate_up_proj", "mlp.experts.down_proj", "mlp.experts.post_expert_norm.weight",
        )]
    return names


EP = MoeParallelConfig(parallel="ep")


def test_every_spark_text_parameter_has_a_checkpoint_tensor(tmp_path):
    from mstar.model.muse.components.spark import MuseSparkForCausalLM
    from mstar.model.muse.config import MuseSparkConfig

    with torch.device("meta"):
        model = MuseSparkForCausalLM(MuseSparkConfig.from_hf(_write(tmp_path, _spark_raw())), moe_parallel=EP)
    targets = {}
    for name in _checkpoint_spark_text_names(4):
        target, shard = _apply_stacked(text_name_remapper(name), _STACKED_PARAMS)
        targets.setdefault(target, set()).add(shard)
    assert {n for n, _ in model.named_parameters()} == set(targets)
    assert targets["model.layers.0.self_attn.qkvg_proj.weight"] == {"q", "gate", "k", "v"}
    # the router's fp32 weight is held as bf16 parts, and its bias stays fp32 through a cast
    gate = model.model.layers[0].mlp.gate
    model.to(torch.bfloat16)
    assert gate.weight.shape == (3, 8, 64) and gate.e_score_correction_bias.dtype == torch.float32


def test_each_tp_rank_holds_its_share_of_whole_experts(tmp_path):
    from types import SimpleNamespace

    from mstar.model.muse.components.spark import spark_moe
    from mstar.model.muse.config import MuseSparkConfig

    config = MuseSparkConfig.from_hf(_write(tmp_path, _spark_raw()))
    with torch.device("meta"):
        moe = spark_moe(config, SimpleNamespace(rank=1, world_size=2), EP)
    assert (moe.expert_start, moe.num_local_experts) == (4, 4)
    # whole experts: the intermediate dim is not split
    assert moe.experts.gate_up_proj.shape == (4, 2 * 48, 32)
    assert moe.experts.down_proj.shape == (4, 32, 48)


@pytest.mark.parametrize("ep_config", [None, {"parallel": "tp"}], ids=["default", "tp"])
def test_spark_refuses_a_tp_split_of_its_experts_at_startup(ep_config):
    """The per-expert norm needs each expert's whole output row, which a TP
    split of the intermediate dim leaves partial on every rank; the conductor
    refuses such a config before any worker loads the weights."""
    from mstar.model.muse.spark_model import MuseSparkModel

    group = {"node_names": ["LLM"], "ranks": [0, 1], "tp_size": 2}
    model = object.__new__(MuseSparkModel)
    model.validate_config_yaml({"node_groups": [{**group, "ep_config": {"parallel": "ep"}}]}, "config.yaml")
    if ep_config is not None:
        group["ep_config"] = ep_config
    with pytest.raises(ValueError, match="parallel: ep"):
        model.validate_config_yaml({"node_groups": [group]}, "config.yaml")


def _spark_moe(tmp_path):
    from mstar.model.muse.components.spark import spark_moe
    from mstar.model.muse.config import MuseSparkConfig

    torch.manual_seed(0)
    moe = spark_moe(MuseSparkConfig.from_hf(_write(tmp_path, _spark_raw())), CommGroup.trivial(), EP)
    for p in moe.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return moe.double()


def test_a_replay_s_padding_rows_skip_spark_s_experts(tmp_path):
    """Under ``mask_padding`` the shared block routes rows past the real
    tokens to no expert: the real rows are as without padding, and the
    padding's are the latent projection of nothing."""
    moe = _spark_moe(tmp_path)
    h = torch.randn(7, 64, dtype=torch.float64)
    valid = torch.arange(7) < 5
    out = moe(h, token_valid=valid)
    torch.testing.assert_close(out[:5], moe(h[:5]))
    assert not out[5:].any()


def _hf_spark_text_model(model):
    """transformers' own MuseSparkTextModel, its eager attention and experts,
    carrying ``model``'s weights."""
    modeling = pytest.importorskip("transformers.models.muse_spark.modeling_muse_spark")
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig

    config = MuseSparkTextConfig(
        **_spark_raw()["text_config"], attn_implementation="eager", experts_implementation="eager",
    )
    ref = modeling.MuseSparkTextModel(config).double().eval()
    cfg = model.config
    q, kv = cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
    state = {}
    latent = {"pre_experts.proj.": "pre_expert_proj.", "pre_experts.norm.": "pre_expert_norm.",
              "post_experts.": "post_expert_proj.", "mlp.expert_norm.": "mlp.experts.post_expert_norm."}
    for name, p in model.model.named_parameters():
        for ours, theirs in latent.items():
            name = name.replace(ours, theirs)
        if name.endswith("qkvg_proj.weight"):
            prefix = name[:-len("qkvg_proj.weight")]
            for part, w in zip(("q", "gate", "k", "v"), p.split([q, q, kv, kv]), strict=True):
                state[f"{prefix}{part}_proj.weight"] = w
        elif name.endswith("mlp.gate.weight"):
            # the router weight its bf16 parts hold
            state[name] = p.double().sum(0)
        else:
            state[name] = p
    ref.load_state_dict(state)
    return ref


def test_the_spark_text_stack_matches_transformers(tmp_path):
    """The whole text stack against transformers' MuseSparkTextModel on the
    same weights: attention with its output norm, the gated residuals, the
    router, the experts' latent and the per-expert norm."""
    from mstar.model.muse.components.spark import MuseSparkForCausalLM
    from mstar.model.muse.config import MuseSparkConfig

    torch.manual_seed(0)
    cfg = MuseSparkConfig.from_hf(_write(tmp_path, _spark_raw()))
    model = MuseSparkForCausalLM(cfg, moe_parallel=EP).double()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    ref = _hf_spark_text_model(model)
    resources = {
        KV_CACHE: _KV(),
        ATTN_SLIDING: _Attn(cfg.attn_scale, cfg.sliding_window - 1),
        ATTN_FULL: _Attn(cfg.attn_scale, -1),
        ROPE: _Pos(cfg.rope_theta),
    }
    for module in model.modules():
        bind = getattr(module, "bind_resources", None)
        if callable(bind):
            bind(resources)
    # long enough that the 5-token window cuts the sliding layers off
    ids = torch.randint(0, cfg.vocab_size, (13,))
    ours = model.model(model.model.embed(ids), label="main")
    with torch.no_grad():
        theirs = ref(input_ids=ids[None], use_cache=False).last_hidden_state[0]
    # the router's logits and the norms are fp32 on both sides, the rest fp64
    assert torch.allclose(ours.double(), theirs, atol=1e-4, rtol=1e-4), (ours.double() - theirs).abs().max()


def test_spark_s_patch_embedding_matches_transformers_on_non_square_grids(tmp_path):
    """Spark reads its position table through a (t, w, h) grid, as transformers'
    MuseSparkVisionPatchEmbedder does; on a non-square grid that differs from
    Glimmer's (t, h, w)."""
    modeling = pytest.importorskip("transformers.models.muse_spark.modeling_muse_spark")
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkVisionConfig

    from mstar.model.muse.components.vision import MuseGlimmerVisionModel

    raw = _spark_raw()["vision_config"]
    config = MuseGlimmerVisionConfig.from_hf(_write(tmp_path, _spark_raw()))
    torch.manual_seed(0)
    tower = MuseGlimmerVisionModel(config).double()
    hf_config = MuseSparkVisionConfig(**{k: v for k, v in raw.items() if k != "model_type"})
    ref = modeling.MuseSparkVisionPatchEmbedder(hf_config).double()
    ref.load_state_dict(tower.patch_embedder.state_dict())
    grid = [(1, 6, 10), (2, 8, 2)]
    pixels = torch.randn(sum(math.prod(g) for g in grid), config.patch_dim, dtype=torch.float64)
    ti = tower_inputs(grid, config)
    ours, _, _ = tower.embed(pixels, ti.interp_indices, ti.interp_weights, ti.window_index, ti.rope_positions)
    with torch.no_grad():
        theirs = tower.ln_pre(ref(pixels, torch.tensor(grid)))[ti.window_index]
    torch.testing.assert_close(ours, theirs)


# --- Muse Spark audio ----------------------------------------------------------------


AUDIO_SPECIALS = ["<|audio|>", "<|soft_audio_start|>", "<|soft_audio_end|>"]
SPARK_TEMPLATE = (
    "{{- bos_token -}}<|start|>system<|message|>sys {{ reasoning_effort | default('high') }}<|eot|>"
    "{%- for m in messages -%}<|start|>{{ m.role }}<|message|>"
    "{%- for p in m.content -%}{%- if p.type == 'image' -%}<|patch|>"
    "{%- elif p.type == 'audio' -%}<|audio|>"
    "{%- else -%}{{ p.text }}{%- endif -%}{%- endfor -%}<|eot|>{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|start|>assistant{%- endif -%}"
)


def _spark_dir(tmp_path, raw):
    specials = SPECIALS + AUDIO_SPECIALS
    raw["audio_token_id"] = specials.index("<|audio|>")
    _write(tmp_path, raw)
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 4]}))
    return _write_tokenizer(tmp_path, specials, SPARK_TEMPLATE)


@pytest.fixture
def spark_dir(tmp_path):
    return _spark_dir(tmp_path, _spark_raw())


def _glimmer_dir(tmp_path, raw):
    return _write_tokenizer(_write(tmp_path, raw), SPECIALS, TEMPLATE)


def _spark_layers(raw: dict) -> dict:
    """Spark's layer pattern: every fourth layer full, from layer 1."""
    tc = raw["text_config"]
    types = ["full_attention" if i % 4 == 1 else "sliding_attention" for i in range(tc["num_hidden_layers"])]
    tc.update(layer_types=types, layer_rope_theta=[5e5 if t == "sliding_attention" else 0 for t in types])
    return raw


@pytest.mark.parametrize("raw, write_dir, name", [
    (_raw_config(52), _glimmer_dir, "muse_glimmer"),
    (_spark_layers(_spark_raw(62)), _spark_dir, "muse_spark"),
], ids=["glimmer", "spark"])
def test_the_kv_pool_keeps_only_the_window_of_the_sliding_layers(raw, write_dir, name, tmp_path, monkeypatch):
    """At each checkpoint's layers (Glimmer's 39 sliding and 13 full, Spark's 46
    and 16), the cache keeps only the window of the config's sliding layers (by
    stack index), and each layer of the stack attends through the attention over
    its own group of the cache."""
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from mstar.engine.resources.attn.base import AttentionManager
    from mstar.engine.resources.attn.config import check_window_readers
    from mstar.engine.resources.base import EngineResourceInfo
    from mstar.model.muse.components.spark import MuseSparkForCausalLM
    from mstar.model.registry import get_model_class

    model = get_model_class(name)(str(write_dir(tmp_path, raw)))
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    types = raw["text_config"]["layer_types"]
    assert specs[KV_CACHE].config.groups[-1].layers == tuple(
        i for i, t in enumerate(types) if t == "sliding_attention"
    )
    check_window_readers(specs)
    # the managers' plans need no kernels to tell which layers they read
    wrapper = MagicMock(side_effect=lambda *a, **kw: MagicMock())
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(
        BatchPrefillWithPagedKVCacheWrapper=wrapper, BatchDecodeWithPagedKVCacheWrapper=wrapper,
    ))
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    info = EngineResourceInfo(device=torch.device("cpu"), dependencies={KV_CACHE: specs[KV_CACHE]})
    attn = {key: AttentionManager.build(specs[key], info) for key in (ATTN_SLIDING, ATTN_FULL)}
    lm = (MuseSparkForCausalLM(model.config, moe_parallel=EP) if name == "muse_spark"
          else MuseGlimmerForCausalLM(model.config))
    # the LLM node's resources, handed to every module that binds them as bind_node_resources does
    resources = {KV_CACHE: MagicMock(), ROPE: MagicMock(), **attn}
    for module in lm.modules():
        if hasattr(module, "bind_resources"):
            module.bind_resources(resources)
    for i, layer in enumerate(lm.model.layers):
        # what the stack's forward sets; the attention refuses a layer of the other group
        layer.self_attn.attend.set_layer_idx(i)


def test_each_audio_clip_expands_between_the_soft_audio_sentinels(spark_dir):
    """A clip takes ``ceil(samples / samples_per_token)`` pads (8 samples a
    token here), padded to whole tokens; the walk runs the encoders the
    prompt needs."""
    from mstar.model.multimodal import MEDIA_LAYOUT
    from mstar.model.muse.spark_model import MuseSparkModel

    model = MuseSparkModel(str(spark_dir))
    audio_pad, start, end = (model.tokenizer.convert_tokens_to_ids(t) for t in AUDIO_SPECIALS)
    out = model.process_prompt(
        "Describe this", ["audio", "text"], ["text"],
        tensors={"audio_inputs": [torch.randn(20)]}, reasoning_effort="low",
    )
    layout = out.metadata[MEDIA_LAYOUT]
    tensors = out.new_input_tensors
    (ids,) = tensors["text_inputs"]
    (offset,), (n,) = layout["offset"], layout["tokens"]
    assert n == 3 and layout["output"] == ["audio_embeds"]
    assert ids.tolist()[offset - 1:offset + n + 1] == [start] + [audio_pad] * 3 + [end]
    (clip,) = tensors["audio_values"]
    # in the tower's dtype, which its first op casts the samples to anyway
    assert clip.shape == (24,) and clip.dtype == torch.bfloat16 and not clip[20:].any()

    signals = {k: list(v) for k, v in tensors.items()}
    (step,) = model._prefill_schedule(["audio", "text"], signals, layout)
    assert step.walk == "prefill_audio"
    # the audio loop runs the clip's one chunk; a prompt with no audio has no such loop
    assert model.get_loop_caps(**out.metadata)["audio_encoder_chunks"] == 1
    assert "audio_encoder_chunks" not in model.get_loop_caps(
        **model.process_prompt("Describe this", ["text"], ["text"]).metadata,
    )
    with_image = model.process_prompt(
        "Describe this", ["image", "audio", "text"], ["text"],
        tensors={"image_inputs": [torch.rand(3, 30, 50)], "audio_inputs": [torch.randn(8)]},
    )
    signals = {k: list(v) for k, v in with_image.new_input_tensors.items()}
    (step,) = model._prefill_schedule(["image", "audio", "text"], signals, with_image.metadata[MEDIA_LAYOUT])
    assert step.walk == "prefill_vision_audio"


def test_a_request_s_audio_past_max_audio_seconds_is_refused(spark_dir):
    """``max_audio_seconds`` caps a request's clips together, as a 400."""
    from mstar.model.muse.spark_model import MuseSparkModel

    probe = MuseSparkModel(str(spark_dir)).audio_config
    # four tokens of 8 samples
    model = MuseSparkModel(str(spark_dir), max_audio_seconds=4.5 * probe.samples_per_token / probe.sampling_rate)
    clip = torch.randn(2 * probe.samples_per_token)
    model.process_prompt("Hi", ["audio", "audio", "text"], ["text"], tensors={"audio_inputs": [clip, clip]})
    with pytest.raises(ValueError, match="at most 4 audio tokens"):
        model.process_prompt(
            "Hi", ["audio", "audio", "audio", "text"], ["text"], tensors={"audio_inputs": [clip, clip, clip]},
        )


def test_spark_s_audio_node_refuses_a_dtype_its_clips_do_not_arrive_in(spark_dir):
    from mstar.model.muse.spark_model import MuseSparkModel

    with pytest.raises(ValueError, match="audio tower runs in"):
        MuseSparkModel(str(spark_dir))._create_submodule("audio_encoder", "cpu", torch.float32, None)


@pytest.mark.parametrize(("setting", "encoder_ranks", "refused"), [
    (True, [0], True), (["LLM"], [0], True), (True, [2], False), (False, [0], False), (["vision_encoder"], [0], False),
])
def test_spark_refuses_tp_async_scheduling_beside_an_encoder(spark_dir, monkeypatch, setting, encoder_ranks, refused):
    """TP async scheduling of the LLM is refused at startup while an encoder
    shares one of its ranks."""
    from mstar.model.muse.spark_model import MuseSparkModel

    monkeypatch.delenv("MSTAR_TP_ASYNC_SCHED", raising=False)
    config = {
        "tp_async_sched": setting,
        "node_groups": [
            {"node_names": ["vision_encoder", "audio_encoder"], "ranks": encoder_ranks},
            {"node_names": ["LLM"], "ranks": [0, 1], "tp_size": 2, "ep_config": {"parallel": "ep"}},
        ],
    }
    model = MuseSparkModel(str(spark_dir))
    if refused:
        with pytest.raises(ValueError, match="TP async scheduling"):
            model.validate_config_yaml(config, "config.yaml")
    else:
        model.validate_config_yaml(config, "config.yaml")


# the checkpoint's processor_config.json, at the test tower's 8 samples a token
FEATURE_EXTRACTOR = {
    "feature_extractor_type": "MuseSparkFeatureExtractor", "feature_size": 1, "padding_side": "right",
    "padding_value": 0.0, "return_attention_mask": True, "samples_per_token": 8, "sampling_rate": 24000,
}


def _audio_tower(tmp_path):
    from mstar.model.muse.components.audio import MuseSparkAudioTower
    from mstar.model.muse.config import MuseSparkAudioConfig

    torch.manual_seed(0)
    (tmp_path / "processor_config.json").write_text(json.dumps({"feature_extractor": FEATURE_EXTRACTOR}))
    config = MuseSparkAudioConfig.from_hf(_write(tmp_path, _spark_raw()))
    assert config.samples_per_token == 8 and config.sampling_rate == 24_000
    tower = MuseSparkAudioTower(config)
    for p in tower.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return tower


@pytest.mark.parametrize("change, match", [
    (lambda raw, fe: raw.pop("audio_config"), "no audio_config"),
    (lambda raw, fe: raw["audio_config"].update(causal=False), r"audio_config's \['causal'\]"),
    (lambda raw, fe: fe.update(padding_side="left"), "feature_extractor's padding_side"),
    (lambda raw, fe: fe.update(samples_per_token=1920), "feature_extractor's samples_per_token"),
    (lambda raw, fe: fe.update(do_normalize=True), "feature_extractor's do_normalize"),
], ids=["no_audio", "unknown_key", "left_padded", "token_width", "normalized"])
def test_an_audio_config_intake_would_misread_is_refused(tmp_path, change, match):
    from mstar.model.muse.config import MuseSparkAudioConfig

    raw, fe = _spark_raw(), dict(FEATURE_EXTRACTOR)
    change(raw, fe)
    (tmp_path / "processor_config.json").write_text(json.dumps({"feature_extractor": fe}))
    with pytest.raises(ValueError, match=match):
        MuseSparkAudioConfig.from_hf(_write(tmp_path, raw))


def _encode(tower, clips: dict[str, list[torch.Tensor]], chunk: int):
    """Each request's rows from the audio node's Loop at ``chunk`` tokens,
    driven as the runtime drives it: each request's chunk count from
    ``get_loop_caps``, the iteration map, and steps that batch every request
    still running (first, continuing and partial chunks) over the pool.
    On CUDA, preprocess must not wait on the device."""
    from types import SimpleNamespace

    from mstar.engine.resources.base import EngineResourceInfo
    from mstar.engine.resources.recurrent.config import RecurrentStateConfig, RecurrentStateSpec
    from mstar.engine.resources.recurrent.pool import RecurrentStatePool
    from mstar.engine.resources.step import StepContext
    from mstar.model.components.chunked_causal_encoder import ChunkedCausalEncoder
    from mstar.model.multimodal import MEDIA_LAYOUT, MediaEncoder
    from mstar.model.multimodal_decoder import MediaIntake
    from mstar.model.muse.components.audio import conv_state_blocks
    from mstar.model.muse.config import AUDIO_CONV
    from mstar.model.muse.spark_model import MuseSparkModel

    device = next(tower.parameters()).device
    encoder = MediaEncoder("audio", "audio_encoder", ("audio_values",), "audio_embeds", (7,), chunk_tokens=chunk)
    node = ChunkedCausalEncoder(tower, tower.config.samples_per_token, encoder, (encoder,), AUDIO_CONV)
    pool = RecurrentStatePool.build(
        RecurrentStateSpec(AUDIO_CONV, {"audio_encoder"}, RecurrentStateConfig(
            num_layers=1, blocks=conv_state_blocks(tower.config, torch.float32), max_slots=len(clips) + 1,
        )),
        EngineResourceInfo(device=device),
    )
    node.bind_node_resources({AUDIO_CONV: pool})
    model = object.__new__(MuseSparkModel)
    model.media = (MediaIntake("audio", encoder, 7, None),)

    layouts = {
        rid: {"tokens": [c.numel() // 8 for c in cs], "output": ["audio_embeds"] * len(cs)}
        for rid, cs in clips.items()
    }
    caps = {rid: model.get_loop_caps(**{MEDIA_LAYOUT: layout})[encoder.loop] for rid, layout in layouts.items()}
    out = {rid: [] for rid in clips}
    for i in range(max(caps.values())):
        rids = [rid for rid in clips if i < caps[rid]]
        rows = [
            node.prepare_inputs("prefill_audio", SimpleNamespace(
                dynamic_loop_iter_counts={encoder.loop: i}, step_metadata={MEDIA_LAYOUT: layouts[rid]},
            ), {"audio_values": clips[rid]})
            for rid in rids
        ]
        step = node.declare_step("prefill_audio", rids, rows).get(AUDIO_CONV)
        ctx = StepContext(request_ids=rids, graph_walk="prefill_audio", slot=0, capture=False)
        assert pool.admit(step, ctx).ok
        pool.plan(step, ctx)
        if device.type == "cuda":
            torch.cuda.set_sync_debug_mode("error")
        try:
            values = node.preprocess("prefill_audio", None, rows)
        finally:
            if device.type == "cuda":
                torch.cuda.set_sync_debug_mode("default")
        with torch.no_grad():
            embeds = node.forward_batched("prefill_audio", None, **values).row_outputs["audio_embeds"]
        pool.commit(step, ctx)
        for j, (rid, row) in enumerate(zip(rids, rows, strict=True)):
            outputs = {"audio_embeds": [embeds[j:j + 1]]}  # the engine's row view
            node.postprocess(rid, None, outputs, inputs=row)
            out[rid] += outputs["audio_embeds"]
    # each clip's last chunk gave its slot back
    assert pool.num_free_slots == pool.config.usable_slots
    return {rid: torch.cat(chunks) for rid, chunks in out.items()}, caps


@pytest.mark.parametrize("device", [
    "cpu",
    # the norms' CUDA kernel: a lone clip's conv output reached it with a row stride of 1
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")),
])
def test_the_audio_loop_s_chunks_are_each_clip_s_whole_tokens(tmp_path, device, fp32_matmul):
    """Two requests' clips (2 + 5 tokens, and 3) in chunks of 2 tokens, on the
    device a worker receives them on: each request's rows, in order, are its
    clips' tokens each run as one chunk (within fp32 summation order)."""
    import copy

    # fp32_matmul: TF32 (the engine's setting) rounds the chunks' GEMMs and the whole clip's differently
    tower = _audio_tower(tmp_path).to(device)
    clips = {
        rid: [torch.randn(n, device=device) for n in samples]
        for rid, samples in {"a": (16, 40), "b": (24,)}.items()
    }
    chunked, caps = _encode(copy.deepcopy(tower), clips, chunk=2)
    assert caps == {"a": 4, "b": 2}
    whole, caps = _encode(tower, clips, chunk=5)
    assert caps == {"a": 2, "b": 1}
    for rid in clips:
        torch.testing.assert_close(chunked[rid], whole[rid], rtol=1e-5, atol=1e-5)


def test_host_chunks_cross_in_one_pinned_copy_beside_a_replay_s_padding_rows(tmp_path, monkeypatch):
    """A replay's padding rows are already on the device; the step's host
    chunks still cross in one pinned tensor, not a pageable copy a row.
    ``meta`` stands in for the device, so this runs without CUDA."""
    from mstar.model.components.chunked_causal_encoder import ChunkedCausalEncoder
    from mstar.model.multimodal import MediaEncoder
    from mstar.model.muse.config import AUDIO_CONV
    from mstar.model.submodule_base import NodeInputs

    encoder = MediaEncoder("audio", "audio_encoder", ("audio_values",), "audio_embeds", (7,), chunk_tokens=2)
    node = ChunkedCausalEncoder(_audio_tower(tmp_path), 8, encoder, (encoder,), AUDIO_CONV)
    host = [torch.randn(16), torch.randn(8)]  # a full chunk and a partial one
    staged = torch.stack([host[0], F.pad(host[1], (0, 8)), torch.zeros(16)])
    rows = [NodeInputs(tensor_inputs={"values": v}) for v in (*host, torch.zeros(16, device="meta"))]

    pinned, host_copies = [], []
    zeros, copy_ = torch.zeros, torch.Tensor.copy_

    def zeros_recorded(*args, pin_memory=False, **kwargs):  # pinning needs a CUDA build
        out = zeros(*args, **kwargs)
        if pin_memory:
            pinned.append(out)
        return out

    def copy_recorded(dst, src, *args, **kwargs):
        if src.device.type == "cpu":
            host_copies.append(dst.untyped_storage().data_ptr())
        return copy_(dst, src, *args, **kwargs)

    monkeypatch.setattr(node, "get_device", lambda: torch.device("meta"))
    monkeypatch.setattr(torch, "zeros", zeros_recorded)
    monkeypatch.setattr(torch.Tensor, "copy_", copy_recorded)
    values = node.preprocess("prefill_audio", None, rows)["values"]
    monkeypatch.undo()

    assert values.device.type == "meta" and len(pinned) == 1
    assert host_copies == [pinned[0].untyped_storage().data_ptr()] * len(host)
    torch.testing.assert_close(pinned[0], staged)


def test_the_audio_node_refuses_a_pool_without_its_sink(tmp_path):
    """A replay's padding rows read and write the pool's sink slot; without
    one they would index slot -1."""
    from mstar.engine.resources.base import EngineResourceInfo
    from mstar.engine.resources.recurrent.config import RecurrentStateConfig, RecurrentStateSpec
    from mstar.engine.resources.recurrent.pool import RecurrentStatePool
    from mstar.model.components.chunked_causal_encoder import ChunkedCausalEncoder
    from mstar.model.multimodal import MediaEncoder
    from mstar.model.muse.components.audio import conv_state_blocks
    from mstar.model.muse.config import AUDIO_CONV

    tower = _audio_tower(tmp_path)
    encoder = MediaEncoder("audio", "audio_encoder", ("audio_values",), "audio_embeds", (7,), chunk_tokens=2)
    node = ChunkedCausalEncoder(tower, 8, encoder, (encoder,), AUDIO_CONV)
    pool = RecurrentStatePool.build(
        RecurrentStateSpec(AUDIO_CONV, {"audio_encoder"}, RecurrentStateConfig(
            num_layers=1, blocks=conv_state_blocks(tower.config, torch.float32), max_slots=2,
            disable_sink_slot=True,
        )),
        EngineResourceInfo(device=torch.device("cpu")),
    )
    with pytest.raises(ValueError, match="needs its sink slot"):
        node.bind_node_resources({AUDIO_CONV: pool})


def test_the_audio_tower_matches_transformers(tmp_path):
    """The tower on a whole clip as one chunk against transformers'
    MuseSparkAudioModel on the same weights, then the text embedding's
    weightless norm the reference applies to audio features."""
    modeling = pytest.importorskip("transformers.models.muse_spark.modeling_muse_spark")
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkAudioConfig

    tower = _audio_tower(tmp_path)
    ref = modeling.MuseSparkAudioModel(MuseSparkAudioConfig(**_spark_raw()["audio_config"])).eval()
    ref.load_state_dict(tower.state_dict())
    clip = torch.randn(1, 40)
    with torch.no_grad():
        theirs = unit_rms(ref(input_values=clip).last_hidden_state, tower.config.text_rms_norm_eps)
    ours, _ = _encode(tower, {"r": [clip[0]]}, chunk=5)
    torch.testing.assert_close(ours["r"], theirs[0], rtol=1e-5, atol=1e-5)


def test_every_audio_parameter_has_a_checkpoint_tensor(tmp_path):
    from mstar.model.muse.components.audio import MuseSparkAudioTower
    from mstar.model.muse.config import MuseSparkAudioConfig
    from mstar.model.muse.weight_loader import audio_name_remapper

    with torch.device("meta"):
        tower = MuseSparkAudioTower(MuseSparkAudioConfig.from_hf(_write(tmp_path, _spark_raw())))
    names = ["norm.weight", "proj.weight", "proj.bias", "post_proj_norm.weight"]
    for i in range(2):
        names += [f"layers.{i}.patch_embedder.{s}" for s in ("proj.weight", "proj.bias", "norm.weight")]
        names += [f"layers.{i}.conv_blocks.0.{s}" for s in (
            "conv.weight", "conv.bias", "norm.weight", "mlp.fc1.weight", "mlp.fc1.bias",
            "mlp.fc2.weight", "mlp.fc2.bias",
        )]
    # the released index's names, under model.audio_tower.
    assert {n for n, _ in tower.named_parameters()} == {audio_name_remapper("model.audio_tower." + n) for n in names}
    assert tower.layers[0].conv_blocks[0].conv.weight.shape == (8, 1, 3)

