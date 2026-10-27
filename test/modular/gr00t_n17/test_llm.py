"""GR00T N1.7's backbone LLM, the Qwen3-VL decoder stack run without a KV cache:
checked against an inline reference layer, with weights loaded the way a checkpoint's are
(separate q/k/v and gate/up, fused by ``load_gr00t_n17_weights``), and the ``LLM`` node.
"""

import dataclasses
import itertools
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.step import Segment
from mstar.model.gr00t_n17.components.llm import Gr00tN17LanguageModel
from mstar.model.gr00t_n17.components.rope import mrope_position_ids
from mstar.model.gr00t_n17.config import LLM_ATTN, Gr00tN17Config, Gr00tN17TextConfig
from mstar.model.gr00t_n17.submodules import Gr00tN17LLMSubmodule
from mstar.model.gr00t_n17.weight_loader import SKIPPED, load_gr00t_n17_weights
from mstar.model.qwen3_5.config import Qwen3_5VisionConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

from .helpers import SegmentAttention, relative_gap

_LLM = "backbone.model.model.language_model."
_LAYERS = 3


def _config(num_layers: int = _LAYERS) -> Gr00tN17Config:
    """A tiny text stack: 4 query heads over 2 KV heads, so a group is two queries wide and
    repeat and repeat_interleave disagree. ``mrope_section`` sums to ``head_dim // 2``."""
    text = Gr00tN17TextConfig(
        hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_hidden_layers=num_layers, rms_norm_eps=1e-6, rope_theta=10000.0,
        vocab_size=96, mrope_section=(4, 2, 2), mrope_interleaved=True,
        image_token_id=5, vision_start_token_id=6,
    )
    # the LLM reads `text` only; the rest of the config belongs to the other nodes
    return Gr00tN17Config(
        text=text, vision=None, dit=None, bridge=None, deepstack_visual_indexes=(), backbone_embedding_dim=0,
    )


def _checkpoint(text: Gr00tN17TextConfig, seed: int) -> dict[str, torch.Tensor]:
    """The tensors a Qwen3-VL text stack stores, under GR00T's checkpoint names; random
    everywhere, norms included, so a misplaced norm cannot hide behind an identity weight."""
    generator = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator) * 0.5

    q_dim = text.num_attention_heads * text.head_dim
    kv_dim = text.num_key_value_heads * text.head_dim
    weights = {f"{_LLM}embed_tokens.weight": rnd(text.vocab_size, text.hidden_size)}
    for i in range(text.num_hidden_layers):
        weights |= {
            f"{_LLM}layers.{i}.self_attn.q_proj.weight": rnd(q_dim, text.hidden_size),
            f"{_LLM}layers.{i}.self_attn.k_proj.weight": rnd(kv_dim, text.hidden_size),
            f"{_LLM}layers.{i}.self_attn.v_proj.weight": rnd(kv_dim, text.hidden_size),
            f"{_LLM}layers.{i}.self_attn.q_norm.weight": rnd(text.head_dim),
            f"{_LLM}layers.{i}.self_attn.k_norm.weight": rnd(text.head_dim),
            f"{_LLM}layers.{i}.self_attn.o_proj.weight": rnd(text.hidden_size, q_dim),
            f"{_LLM}layers.{i}.mlp.gate_proj.weight": rnd(text.intermediate_size, text.hidden_size),
            f"{_LLM}layers.{i}.mlp.up_proj.weight": rnd(text.intermediate_size, text.hidden_size),
            f"{_LLM}layers.{i}.mlp.down_proj.weight": rnd(text.hidden_size, text.intermediate_size),
            f"{_LLM}layers.{i}.input_layernorm.weight": rnd(text.hidden_size),
            f"{_LLM}layers.{i}.post_attention_layernorm.weight": rnd(text.hidden_size),
        }
    return weights


def _loaded_model(config: Gr00tN17Config, weights: dict[str, torch.Tensor], tmp_path) -> Gr00tN17LanguageModel:
    """Our module with those tensors in it, through the loader that fuses q/k/v and gate/up."""
    save_file(weights, tmp_path / "model.safetensors")
    model = Gr00tN17LanguageModel(config)
    load_gr00t_n17_weights(model, "LLM", tmp_path)
    return model


# ----------------------------------------------------------------------------
# Inline reference: one Qwen3-VL decoder layer
# (transformers `models/qwen3_vl/modeling_qwen3_vl.py`), with q, k and v as three
# separate linears and the gated MLP's gate and up as two, the way the checkpoint
# stores them. Pre-norm, two residuals, qk-norm over head_dim before the rotation,
# and GQA by repeating each KV head over the queries it serves.
# ----------------------------------------------------------------------------


class RefRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _ref_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half over ``[tokens, heads, head_dim]``; cos/sin broadcast over the heads."""
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


# Deliberate deviations, each a mistake that would otherwise look like the real layer.
_WRONG_ROPE_FIRST = "rope before the qk-norm"
_WRONG_KV_SWAP = "k and v swapped"
_WRONG_KV_REPEAT = "kv heads repeated instead of interleaved"
_WRONG_NO_RESIDUAL = "the mlp residual dropped"
DEVIATIONS = (_WRONG_ROPE_FIRST, _WRONG_KV_SWAP, _WRONG_KV_REPEAT, _WRONG_NO_RESIDUAL)


class RefAttention(nn.Module):
    def __init__(self, config: Gr00tN17TextConfig, deviation: str | None = None):
        super().__init__()
        self.config = config
        self.deviation = deviation
        q_dim = config.num_attention_heads * config.head_dim
        kv_dim = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.q_norm = RefRMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RefRMSNorm(config.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        config = self.config
        tokens = x.shape[0]
        q = self.q_proj(x).view(tokens, config.num_attention_heads, config.head_dim)
        k = self.k_proj(x).view(tokens, config.num_key_value_heads, config.head_dim)
        v = self.v_proj(x).view(tokens, config.num_key_value_heads, config.head_dim)
        if self.deviation == _WRONG_KV_SWAP:
            k, v = v, k
        if self.deviation == _WRONG_ROPE_FIRST:
            q, k = self.q_norm(_ref_rope(q, cos, sin)), self.k_norm(_ref_rope(k, cos, sin))
        else:
            q, k = _ref_rope(self.q_norm(q), cos, sin), _ref_rope(self.k_norm(k), cos, sin)
        group = config.num_attention_heads // config.num_key_value_heads
        if self.deviation == _WRONG_KV_REPEAT:
            k, v = k.repeat(1, group, 1), v.repeat(1, group, 1)
        else:
            k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True,
        ).transpose(0, 1)
        return self.o_proj(out.reshape(tokens, -1))


class RefDecoderLayer(nn.Module):
    def __init__(self, config: Gr00tN17TextConfig, deviation: str | None = None):
        super().__init__()
        self.deviation = deviation
        self.input_layernorm = RefRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = RefAttention(config, deviation)
        self.post_attention_layernorm = RefRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        normed = self.post_attention_layernorm(x)
        mlp = self.down_proj(F.silu(self.gate_proj(normed)) * self.up_proj(normed))
        return mlp if self.deviation == _WRONG_NO_RESIDUAL else x + mlp


@torch.no_grad()
def _reference_stack(
    text: Gr00tN17TextConfig, weights: dict[str, torch.Tensor], deviation: str | None = None,
) -> list[RefDecoderLayer]:
    """The same checkpoint tensors, one per separate projection."""
    layers = []
    for i in range(text.num_hidden_layers):
        layer = RefDecoderLayer(text, deviation)
        for name, param in layer.named_parameters():
            name = name.replace("gate_proj", "mlp.gate_proj").replace("up_proj", "mlp.up_proj")
            name = name.replace("down_proj", "mlp.down_proj")
            param.copy_(weights[f"{_LLM}layers.{i}.{name}"])
        layers.append(layer)
    return layers


def _bind(model: Gr00tN17LanguageModel, resources: dict) -> None:
    """Bind the way the engine does: on the node, fanned out to the layers underneath it."""
    node = nn.Module()
    node.model = model
    NodeSubmodule.bind_node_resources(node, resources)


@torch.no_grad()
def _prepare(model: Gr00tN17LanguageModel, tokens: int, seed: int):
    """One prompt's embeddings and rotation, with the attention resource bound the way the
    engine binds it: one segment covering the whole step."""
    ids = torch.randint(0, 96, (tokens,), generator=torch.Generator().manual_seed(seed))
    positions = torch.arange(tokens).float().expand(3, -1).contiguous()
    cos, sin = model.build_cos_sin(positions)
    _bind(model, {LLM_ATTN: SegmentAttention((tokens,), causal=True, gqa=True)})
    return model.embed(ids), cos, sin


@torch.no_grad()
def _run(model: Gr00tN17LanguageModel, layers: list[RefDecoderLayer], tokens: int, seed: int):
    """Both stacks over the same prompt, from token ids through ``embed`` to the last layer."""
    hidden, cos, sin = _prepare(model, tokens, seed)
    ours = model(hidden, cos, sin)
    for layer in layers:
        hidden = layer(hidden, cos, sin)
    return ours, hidden


# ----------------------------------------------------------------------------
# The layer against the reference
# ----------------------------------------------------------------------------

# Our q/k/v are one GEMM where the reference has three, and gate/up one where it has two, so
# the two stacks differ by the BLAS's rounding rather than by their maths. Measured over
# 6 weight seeds x 7 token counts (1 to 300) x intra-op thread counts 1, 2 and 4, the worst
# max|ours - ref| / max|ref| was 1.021e-06, about 8 fp32 ulps; at 11 tokens or fewer it is
# exactly 0, because the kernel's reduction over K does not depend on the output width until
# the output is wide enough to re-block.
_MEASURED_GAP = 1.021e-06
# Four times that, so a BLAS blocking differently from this host's still passes. It stays far
# under a wiring mistake: the closest deviation below misses by 8.3e+04 times this bar.
_BAR = 4 * _MEASURED_GAP


@pytest.mark.parametrize("tokens", [1, 11, 64, 300])
def test_our_stack_is_the_reference_decoder_layer(tmp_path, tokens):
    config = _config()
    weights = _checkpoint(config.text, seed=0)
    ours, theirs = _run(
        _loaded_model(config, weights, tmp_path), _reference_stack(config.text, weights), tokens, seed=1,
    )
    assert ours.shape == (tokens, config.text.hidden_size)
    assert relative_gap(ours, theirs) <= _BAR


@pytest.mark.parametrize("deviation", DEVIATIONS)
def test_a_layer_assembled_wrongly_misses_by_orders_of_magnitude(tmp_path, deviation):
    """The bar above only means something if a plausible mistake cannot slip under it.

    Measured relative gaps, against a bar of 4.1e-06: rope before the qk-norm 3.4e-01,
    k and v swapped 1.1e+00, kv heads repeated instead of interleaved 9.3e-01, the mlp
    residual dropped 1.0e+00 -- the smallest of them 8.3e+04 times the bar.
    """
    config = _config()
    weights = _checkpoint(config.text, seed=0)
    ours, theirs = _run(
        _loaded_model(config, weights, tmp_path),
        _reference_stack(config.text, weights, deviation),
        tokens=64, seed=1,
    )
    assert relative_gap(ours, theirs) > 1000 * _BAR


def test_the_fused_projection_holds_q_then_k_then_v_one_block_per_head(tmp_path):
    """What the forward's ``split`` assumes, and the half of GQA this module owns: which rows
    of the fused weight are a query head and which are a KV head. The other half -- which
    group a query head lands in -- is the kernel's, and the ``repeat`` deviation above is
    what pins it."""
    config = _config()
    text = config.text
    weights = _checkpoint(text, seed=2)
    attn = _loaded_model(config, weights, tmp_path).layers[0].self_attn

    q_dim = text.num_attention_heads * text.head_dim
    kv_dim = text.num_key_value_heads * text.head_dim
    assert attn.qkv_proj.output_sizes == [q_dim, kv_dim, kv_dim]
    assert (attn.num_heads, attn.num_kv_heads) == (text.num_attention_heads, text.num_key_value_heads)
    fused = attn.qkv_proj.weight
    for offset, part in ((0, "q"), (q_dim, "k"), (q_dim + kv_dim, "v")):
        checkpoint = weights[f"{_LLM}layers.0.self_attn.{part}_proj.weight"]
        assert torch.equal(fused[offset:offset + checkpoint.shape[0]], checkpoint)


# ----------------------------------------------------------------------------
# DeepStack injection
# ----------------------------------------------------------------------------

# Two image blocks with text between them. A single contiguous block still lines up with
# itself over most of its length after a roll or a reversal, so it hides a row-ordering
# mistake; two blocks do not.
_IMAGE_ROWS = torch.tensor([5, 6, 7, 8, 30, 31, 32, 33])
# Fewer features than layers, as the real stack has (three vision taps, `select_layer`
# layers), which also leaves the off-by-one deviations below a layer to land on.
_FEATURES = 3
_DEEPSTACK_LAYERS = 4
# the reversal (a list read backwards) and a 3-cycle; the closest of all five orders missed by
# 1.5e-02 (3.8e+03x the bar)
_PERMUTATIONS = [(2, 1, 0), (1, 2, 0)]


@torch.no_grad()
def _ref_forward(
    layers: list[RefDecoderLayer], hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    deepstack: list[torch.Tensor], rows, shift: int = 0, order=None,
) -> torch.Tensor:
    """The reference stack with ``Qwen3VLTextModel``'s DeepStack injection: after layer
    ``i + shift``, feature ``order[i]`` is added to ``rows`` the way ``_deepstack_process``
    does it, ``hidden_states[visual_pos_masks, :] += visual_embeds``.

    ``shift`` and ``order`` are the deviations, and default to the real thing. Injection
    "after layer -1" is reachable only at ``shift=-1``, which is one of them.
    """
    order = tuple(range(len(deepstack))) if order is None else order

    def inject(hidden: torch.Tensor, after_layer: int) -> torch.Tensor:
        for i in range(len(deepstack)):
            if i + shift == after_layer:
                hidden = hidden.clone()
                hidden[rows] += deepstack[order[i]]
        return hidden

    hidden = inject(hidden, -1)
    for i, layer in enumerate(layers):
        hidden = inject(layer(hidden, cos, sin), i)
    return hidden


class _Case(NamedTuple):
    """One DeepStack forward: our module, the reference stack, and the step both run on."""

    model: Gr00tN17LanguageModel
    layers: list[RefDecoderLayer]
    hidden: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    deepstack: list[torch.Tensor]
    rows: torch.Tensor
    # `visual_pos_masks`: the reference names the image rows with a boolean mask, we pass
    # indices. The two agree because the indices are unique and ascending, which is how the
    # vision encoder packs them; a deviation below passes indices the mask cannot express.
    mask: torch.Tensor

    def ours(self) -> torch.Tensor:
        return self.model(self.hidden, self.cos, self.sin, self.deepstack, self.rows)

    def theirs(self, rows=None, shift: int = 0, order=None) -> torch.Tensor:
        rows = self.mask if rows is None else rows
        return _ref_forward(self.layers, self.hidden, self.cos, self.sin, self.deepstack, rows, shift, order)


@torch.no_grad()
def _case(tmp_path, num_layers: int = _DEEPSTACK_LAYERS, features: int = _FEATURES, tokens: int = 64) -> _Case:
    config = _config(num_layers=num_layers)
    weights = _checkpoint(config.text, seed=0)
    model = _loaded_model(config, weights, tmp_path)
    hidden, cos, sin = _prepare(model, tokens, seed=1)
    generator = torch.Generator().manual_seed(10)
    # at the scale of the hidden states, and distinct, so permuting them is a real change
    deepstack = [
        torch.randn(len(_IMAGE_ROWS), config.text.hidden_size, generator=generator) * 0.5
        for _ in range(features)
    ]
    mask = torch.zeros(tokens, dtype=torch.bool)
    mask[_IMAGE_ROWS] = True
    return _Case(model, _reference_stack(config.text, weights), hidden, cos, sin, deepstack, _IMAGE_ROWS, mask)


def test_deepstack_features_land_on_the_image_rows_after_their_layer(tmp_path):
    """The layer-parity bar, for the same reason (fused vs separate GEMMs); the injection adds
    the same fp32 numbers on both sides. Worst measured 1.3e-06 over 6 x 3 seeds."""
    case = _case(tmp_path)
    ours = case.ours()
    assert ours.shape == case.hidden.shape
    assert relative_gap(ours, case.theirs()) <= _BAR


@pytest.mark.parametrize("shift", [-1, 1], ids=["a layer early", "a layer late"])
def test_a_feature_added_after_the_wrong_layer_misses_by_orders_of_magnitude(tmp_path, shift):
    """Moving an injection one layer cannot slip under the bar: measured a layer early
    6.8e-01, a layer late 4.9e-02, the smaller 1.2e+04 times the bar."""
    case = _case(tmp_path)
    assert relative_gap(case.ours(), case.theirs(shift=shift)) > 1000 * _BAR


@pytest.mark.parametrize("order", _PERMUTATIONS)
def test_features_applied_in_the_wrong_order_miss_by_orders_of_magnitude(tmp_path, order):
    """Each feature belongs to one layer; equal features would permute to themselves, so
    their distinctness is asserted rather than assumed."""
    case = _case(tmp_path)
    for first, second in itertools.combinations(case.deepstack, 2):
        assert not torch.equal(first, second)
    assert relative_gap(case.ours(), case.theirs(order=order)) > 1000 * _BAR


@pytest.mark.parametrize(
    "permute", [lambda rows: rows.flip(0), lambda rows: rows.roll(1)], ids=["reversed", "rolled by one"],
)
def test_row_k_of_a_feature_lands_on_image_row_k(tmp_path, permute):
    """The vision encoder packs its rows in image-token order and we hand the LLM that order
    as indices, so the mapping is ours to get right. Measured 6.4e-02 reversed and 8.4e-02
    rolled, both over 1.5e+04 times the bar.
    """
    case = _case(tmp_path)
    assert relative_gap(case.ours(), case.theirs(rows=permute(case.rows))) > 1000 * _BAR


def test_only_the_image_rows_move(tmp_path):
    """``index_add`` copies the untouched rows, so they are ``torch.equal``. One layer and one
    feature, so the injection is last: with a layer after it, attention mixes every row."""
    case = _case(tmp_path, num_layers=1, features=1)
    base = case.model(case.hidden, case.cos, case.sin)
    moved = case.ours()
    assert torch.equal(base[~case.mask], moved[~case.mask])
    assert (base[case.mask] != moved[case.mask]).any(dim=1).all()


def test_no_features_is_the_forward_without_the_argument(tmp_path):
    """The DT and the bridge call this with no DeepStack at all; that path must not have
    picked up a clone, a reordering or any other arithmetic."""
    case = _case(tmp_path)
    plain = case.model(case.hidden, case.cos, case.sin)
    assert torch.equal(case.model(case.hidden, case.cos, case.sin, ()), plain)
    assert torch.equal(case.model(case.hidden, case.cos, case.sin, (), case.rows), plain)


def test_the_guards_say_which_call_was_wrong():
    """Both are caller mistakes that would otherwise be silent: features with nowhere to go
    would be dropped, and a fourth feature for a three-layer stack would never be added."""
    model = Gr00tN17LanguageModel(_config(num_layers=_LAYERS))
    hidden = torch.zeros(64, model.config.hidden_size)
    cos, sin = model.build_cos_sin(torch.zeros(3, 64))
    feature = torch.zeros(len(_IMAGE_ROWS), model.config.hidden_size)

    with pytest.raises(ValueError, match="no image rows"):
        model(hidden, cos, sin, [feature])
    with pytest.raises(ValueError, match=rf"4 DeepStack feature\(s\) for {_LAYERS} layers"):
        model(hidden, cos, sin, [feature] * 4, _IMAGE_ROWS)


# ----------------------------------------------------------------------------
# The attention resource
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("resources", [{}, {"vision_attn": object()}], ids=["nothing bound", "the wrong key"])
def test_an_unbound_attention_names_the_resource_it_wanted(resources):
    """A config that declares the resource under another name is a typo, and the message has
    to say which name the layer looked for."""
    model = Gr00tN17LanguageModel(_config(num_layers=1))
    _bind(model, resources)
    with pytest.raises(RuntimeError, match=LLM_ATTN):
        model(torch.zeros(2, model.config.hidden_size), *model.build_cos_sin(torch.zeros(3, 2)))


def test_binding_the_node_reaches_every_layers_attention():
    model = Gr00tN17LanguageModel(_config())
    attention = SegmentAttention((2,), causal=True, gqa=True)
    _bind(model, {LLM_ATTN: attention})
    assert [layer.self_attn.ragged_attn for layer in model.layers] == [attention] * _LAYERS


# ----------------------------------------------------------------------------
# Shapes and parameters
# ----------------------------------------------------------------------------


def test_build_cos_sin_turns_three_grids_of_positions_into_one_row_per_token():
    model = Gr00tN17LanguageModel(_config())
    tokens = 7
    # the three grids start together and then diverge, as an image block makes them
    positions = torch.arange(tokens).float().expand(3, -1) + torch.tensor([[0.0], [2.0], [5.0]])
    cos, sin = model.build_cos_sin(positions)
    assert cos.shape == sin.shape == (tokens, model.config.head_dim)
    # position 0 on every grid is the identity rotation, which pins tokens as the row axis
    zero_cos, zero_sin = model.build_cos_sin(torch.zeros(3, tokens))
    assert torch.equal(zero_cos, torch.ones_like(zero_cos))
    assert torch.equal(zero_sin, torch.zeros_like(zero_sin))


def test_the_parameters_are_the_checkpoints_tensors_after_fusing(tmp_path):
    """Every text tensor lands in a parameter and every parameter gets one, q/k/v and gate/up
    fused on the way in; the set equality catches a parameter with no tensor, such as the final
    norm GR00T does not run."""
    config = _config()
    weights = _checkpoint(config.text, seed=3)
    model = _loaded_model(config, weights, tmp_path)

    fused = {"q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj",
             "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj"}
    expected = set()
    for name in weights:
        name = name[len(_LLM):]
        part = name.split(".")[-2]
        expected.add(name.replace(part, fused[part]) if part in fused else name)
    assert {name for name, _ in model.named_parameters()} == expected

    # the reference reads the last layer's output, so neither the final norm nor the head runs
    assert {f"{_LLM}norm.weight", "backbone.model.lm_head.weight"} == SKIPPED
    assert not any(name.startswith(("norm.", "lm_head.")) for name, _ in model.named_parameters())


# ----------------------------------------------------------------------------
# The node: `Gr00tN17LLMSubmodule`, driven the way the engine drives it
# ----------------------------------------------------------------------------

_WALK = "prefill"
_FWD = CurrentForwardPassInfo(request_id="r0", graph_walk=_WALK, fwd_index=0, random_seed=0, max_tokens=0)
_ENGINE = ModelInputsFromEngine(request_ids=["r0"], per_request_info={})
_IMAGE_TOKEN = 5  # `_config`'s `image_token_id`
_MERGE = 2  # the spatial merge the image rows were packed at
# what `prepare_inputs` stages, which is also `forward`'s signature
_STAGED = {"input_ids", "image_rows", "position_ids", "vision_embeds", "deepstack_embeds"}

# Text-span lengths and `(t, h, w)` image grids. The two-image prompt's image rows are two
# non-contiguous blocks, so neither side of the scatter can be reversed or rolled back onto
# itself; a single block would still line up with itself over most of its length.
_PROMPTS = {
    "no images": [9],
    "one image": [3, (1, 4, 6), 2],
    "two images with text between": [2, (1, 4, 4), 3, (1, 2, 6), 2],
}


def _node_config(num_layers: int = _LAYERS) -> Gr00tN17Config:
    """``_config`` plus the one vision field the node reads: the merge its image rows were
    packed at, which is also what the M-RoPE grids are measured in."""
    vision = Qwen3_5VisionConfig(
        depth=1, hidden_size=16, intermediate_size=32, num_heads=1, in_channels=3,
        patch_size=2, temporal_patch_size=2, spatial_merge_size=_MERGE,
        num_position_embeddings=16, out_hidden_size=64,
    )
    # the node checks the feature count against this, so it has to say three
    return dataclasses.replace(
        _config(num_layers), vision=vision, deepstack_visual_indexes=(5, 11, 17),
    )


def _node(tmp_path, num_layers: int = _LAYERS) -> Gr00tN17LLMSubmodule:
    config = _node_config(num_layers)
    return Gr00tN17LLMSubmodule(_loaded_model(config, _checkpoint(config.text, seed=0), tmp_path), config)


def _prompt(spec: list, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """A spec of text-span lengths and image grids -> the walk's ``input_ids`` and its grids."""
    generator = torch.Generator().manual_seed(seed)
    tokens, grids = [], []
    for item in spec:
        if isinstance(item, int):
            text = torch.randint(0, 96, (item,), generator=generator)
            # the image token is what names the image rows, so the text must not carry it
            tokens += text.masked_fill(text == _IMAGE_TOKEN, 0).tolist()
        else:
            t, h, w = item
            tokens += [_IMAGE_TOKEN] * (t * (h // _MERGE) * (w // _MERGE))
            grids.append(item)
    return torch.tensor(tokens), torch.tensor(grids, dtype=torch.long).reshape(-1, 3)


def _walk_inputs(config: Gr00tN17Config, ids: torch.Tensor, grid: torch.Tensor, seed: int = 11) -> dict:
    """What the walk carries into the node: the prompt, its grids, and the vision encoder's two
    outputs -- one embedding and ``_FEATURES`` DeepStack features, a row each per image token."""
    rows = int((ids.reshape(-1) == _IMAGE_TOKEN).sum())
    generator = torch.Generator().manual_seed(seed)

    def rnd() -> torch.Tensor:
        return torch.randn(rows, config.text.hidden_size, generator=generator) * 0.5

    return {
        "input_ids": [ids], "image_grid_thw": [grid],
        "vision_embeds": [rnd()], "deepstack_embeds": [rnd() for _ in range(_FEATURES)],
    }


class _Run(NamedTuple):
    """One pass of the node, with what the walk handed it and what came back."""

    node: Gr00tN17LLMSubmodule
    ids: torch.Tensor
    grid: torch.Tensor
    walk: dict
    prepared: NodeInputs
    tensors: dict
    out: dict

    @property
    def rows(self) -> torch.Tensor:
        return self.prepared.tensor_inputs["image_rows"]

    @property
    def features(self) -> torch.Tensor:
        return self.out["backbone_features"][0]

    @torch.no_grad()
    def by_hand(self, scatter_rows=None, vision=None) -> torch.Tensor:
        """``forward`` written out: embed, scatter the vision rows on, rotate, run the stack.

        Derived from the prompt rather than from ``prepared``, so it owes the node nothing.
        ``scatter_rows`` and ``vision`` are the scatter's deviations and default to the real
        thing; DeepStack lands on the true image rows either way, leaving the scatter alone.
        """
        model = self.node.model
        ids = self.ids.reshape(-1)
        rows = (ids == _IMAGE_TOKEN).nonzero(as_tuple=True)[0]
        vision = self.walk["vision_embeds"][0] if vision is None else vision
        hidden = model.embed(ids).index_copy(0, rows if scatter_rows is None else scatter_rows, vision)
        positions = mrope_position_ids(ids, self.grid, _IMAGE_TOKEN, _MERGE)
        cos, sin = model.build_cos_sin(positions, hidden.dtype)
        return model(hidden, cos, sin, self.walk["deepstack_embeds"], rows)


@torch.no_grad()
def _drive(tmp_path, spec: list) -> _Run:
    """prepare_inputs -> declare_step's resource -> preprocess -> forward, as the engine runs it."""
    node = _node(tmp_path)
    ids, grid = _prompt(spec)
    walk = _walk_inputs(node.config, ids, grid)
    prepared = node.prepare_inputs(_WALK, _FWD, walk)
    # the engine plans the resource from the declaration, so the stand-in is planned from it too
    step = node.declare_step(_WALK, ["r0"], [prepared])
    spans = tuple(seg.span for seg in step.segments)
    node.bind_node_resources({LLM_ATTN: SegmentAttention(spans, causal=True, gqa=True)})
    tensors = node.preprocess(_WALK, _ENGINE, [prepared])
    return _Run(node, ids, grid, walk, prepared, tensors, node.forward(_WALK, _ENGINE, **tensors))


@pytest.mark.parametrize("spec", list(_PROMPTS.values()), ids=list(_PROMPTS))
def test_the_step_is_one_causal_segment_over_the_whole_prompt(tmp_path, spec):
    """One causal segment over every prompt token (the vision node cuts one non-causal segment
    per frame); both live only in the declaration, so this is where they are said."""
    node = _node(tmp_path)
    ids, grid = _prompt(spec)
    prepared = node.prepare_inputs(_WALK, _FWD, _walk_inputs(node.config, ids, grid))
    step = node.declare_step(_WALK, ["r0"], [prepared])

    assert step.segments == [Segment("r0", "main", len(ids))]
    assert list(step.steps) == [LLM_ATTN]
    assert step.steps[LLM_ATTN].causal is True
    # what the resource is planned from: `SubmoduleStep` fills it in from `segments`
    assert step.steps[LLM_ATTN].segments == step.segments


@pytest.mark.parametrize("spec", list(_PROMPTS.values()), ids=list(_PROMPTS))
def test_prepare_inputs_derives_the_rows_and_the_positions_from_the_prompt(tmp_path, spec):
    """Everything the forward needs beyond the walk's inputs, each against the prompt it came
    from; the positions against ``mrope_position_ids`` called directly."""
    node = _node(tmp_path)
    ids, grid = _prompt(spec)
    walk = _walk_inputs(node.config, ids, grid)
    prepared = node.prepare_inputs(_WALK, _FWD, walk)
    tensors = prepared.tensor_inputs

    # what the engine sizes the batch and the segment by
    assert prepared.input_seq_len == len(ids)
    assert set(tensors) == _STAGED
    assert torch.equal(tensors["input_ids"], ids)
    expected_rows = [i for i, token in enumerate(ids.tolist()) if token == _IMAGE_TOKEN]
    assert tensors["image_rows"].tolist() == expected_rows
    assert torch.equal(tensors["position_ids"], mrope_position_ids(ids, grid, _IMAGE_TOKEN, _MERGE))
    assert torch.equal(tensors["vision_embeds"], walk["vision_embeds"][0])

    # [taps, rows, hidden]: one tensor to stage, which the forward unbinds back to the three
    assert tensors["deepstack_embeds"].shape == (_FEATURES, len(expected_rows), node.config.text.hidden_size)
    unbound = tensors["deepstack_embeds"].unbind(0)
    assert all(torch.equal(ours, theirs) for ours, theirs in zip(unbound, walk["deepstack_embeds"], strict=True))
    # a prompt with no images has three empty features, which permute to themselves
    assert not expected_rows or all(
        not torch.equal(first, second) for first, second in itertools.combinations(unbound, 2)
    )


def test_a_prompt_that_arrives_as_a_row_is_the_same_one_request(tmp_path):
    """``input_ids`` reaches the node as the tokenizer left it, so a ``[1, tokens]`` row is one
    prompt and not a batch of one -- and ``input_seq_len`` is its token count either way."""
    node = _node(tmp_path)
    ids, grid = _prompt(_PROMPTS["one image"])
    flat = node.prepare_inputs(_WALK, _FWD, _walk_inputs(node.config, ids, grid))
    rowed = node.prepare_inputs(_WALK, _FWD, _walk_inputs(node.config, ids.unsqueeze(0), grid))

    assert rowed.input_seq_len == flat.input_seq_len == len(ids)
    assert all(torch.equal(rowed.tensor_inputs[name], flat.tensor_inputs[name]) for name in _STAGED)


@pytest.mark.parametrize("spec", list(_PROMPTS.values()), ids=list(_PROMPTS))
def test_driving_the_node_is_the_forwards_own_pieces_called_by_hand(tmp_path, spec):
    """prepare_inputs -> preprocess -> forward is embed, the scatter, the rotation and the stack
    on the same tensors, in the same order, so ``torch.equal``. The image rows ride beside the
    features, for the bridge to split on."""
    run = _drive(tmp_path, spec)
    assert list(run.out) == ["backbone_features", "image_rows"]
    assert len(run.out["backbone_features"]) == 1
    assert torch.equal(run.out["image_rows"][0], run.tensors["image_rows"])
    assert run.features.shape == (len(run.ids), run.node.config.text.hidden_size)
    # preprocess stages the tensors and nothing else; there are no non-tensor kwargs here
    assert set(run.tensors) == _STAGED
    assert torch.equal(run.features, run.by_hand())


_SCATTERS = {
    "the embeddings reversed": lambda run: {"vision": run.walk["vision_embeds"][0].flip(0)},
    "the embeddings rolled by one": lambda run: {"vision": run.walk["vision_embeds"][0].roll(1, 0)},
    "the rows reversed": lambda run: {"scatter_rows": run.rows.flip(0)},
    "the rows rolled by one": lambda run: {"scatter_rows": run.rows.roll(1)},
}


@pytest.mark.parametrize("permute", list(_SCATTERS.values()), ids=list(_SCATTERS))
def test_row_k_of_the_vision_embeds_lands_on_image_row_k(tmp_path, permute):
    """The node hands the vision rows over as indices, so the pairing is its to get right.
    Measured gaps for the four permutations: 1.0e+00, 8.3e-01, 1.0e+00, 1.0e+00 (>= 2.0e+05x the bar)."""
    run = _drive(tmp_path, _PROMPTS["two images with text between"])
    # the unpermuted scatter is exact, which is what makes the gap below mean something
    assert torch.equal(run.features, run.by_hand())
    assert relative_gap(run.features, run.by_hand(**permute(run))) > 1000 * _BAR


def test_the_node_takes_one_request_a_step(tmp_path):
    """One request per forward: a second raises (the interpreter's unpacking message, hence
    the loose match) rather than being dropped or appended to the first prompt."""
    node = _node(tmp_path)
    ids, grid = _prompt(_PROMPTS["one image"])
    prepared = node.prepare_inputs(_WALK, _FWD, _walk_inputs(node.config, ids, grid))
    two = ModelInputsFromEngine(request_ids=["r0", "r1"], per_request_info={})

    assert node.max_batch_size(_WALK) == 1
    with pytest.raises(ValueError, match="expected 1"):
        node.preprocess(_WALK, two, [prepared, prepared])
