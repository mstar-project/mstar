"""CPU parity of the Cosmos3-Edge reasoner against the Hugging Face
``Cosmos3EdgeForConditionalGeneration`` reference (transformers >= 5.17).

The reference runs out-of-process (a transformers 5 env; see
``notes/ref_dump_edge_reasoner.py`` in the workspace) and dumps, for the
model card's reasoning example in fp32: the processor's ``input_ids`` /
``pixel_values`` / ``image_grid_thw``, the 3D mRoPE ``position_ids``, the
projected vision tokens, the prefill logits at the last position and eight
greedy tokens. This test rebuilds every stage with M*'s own code on CPU (fp32,
real weights) and compares:

* preprocessing: identical pixel patches, grid and token ids;
* positions: identical mRoPE ids;
* vision tower + projector: vision tokens within fp32 tolerance;
* text tower + lm_head over the paged-cache path (an SDPA stand-in for the
  kv/attn resources): last-position logits within tolerance, the same
  greedy tokens.

Needs ``COSMOS3_EDGE_REF`` (the dump) and the snapshot (``COSMOS3_EDGE_DIR``
or the shared HF cache); skipped otherwise. Loading the 4B text tower in fp32
takes ~16 GB of host memory.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from mstar.model.cosmos3 import constants as C
from mstar.model.cosmos3.submodules import COND_LABEL, SAMPLER
from mstar.model.cosmos3.tests.test_edge import EDGE_DIR
from mstar.model.cosmos3.tests.test_engine_cache import _SdpaAttention, _SdpaKV, _SdpaResources, declaring
from mstar.testing_utils.resource import StubResource

REF = os.environ.get("COSMOS3_EDGE_REF")
needs_ref = pytest.mark.skipif(
    EDGE_DIR is None or not REF or not Path(REF).exists(),
    reason="set COSMOS3_EDGE_REF to the HF reasoner dump and COSMOS3_EDGE_DIR to the snapshot",
)


class _StreamKV(_SdpaKV):
    """`_SdpaKV` with a stream per request: its K/V by request and layer."""

    def plan(self, step, ctx):
        self.segments = tuple(step.segments)

    def write_kv(self, k, v, layer_idx=None, label=None, tokens=None):
        layer = self.layer_view(layer_idx)
        offset = 0
        for seg in self.segments:
            self.pending[(seg.request_id, layer)] = (k[offset:offset + seg.span], v[offset:offset + seg.span])
            offset += seg.span


class _CausalSdpa(_SdpaAttention):
    """`_SdpaAttention` over a `_StreamKV`: each request's queries attend causally over what it kept and this step's
    K/V."""

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        layer = self._default_layer_idx if layer_idx is None else layer_idx
        outs, offset = [], 0
        for seg in self._kv.segments:
            rows = slice(offset, offset + seg.span)
            offset += seg.span
            k_i, v_i = k[rows], v[rows]
            kept = self._kv.committed.get((seg.request_id, layer))
            if kept is not None:
                k_i, v_i = torch.cat([kept[0], k_i]), torch.cat([kept[1], v_i])
            # the step's queries are the stream's last positions, which sdpa's is_causal would align to its first
            mask = torch.ones(seg.span, k_i.shape[0], dtype=torch.bool).tril(diagonal=k_i.shape[0] - seg.span)
            out = F.scaled_dot_product_attention(
                q[rows].unsqueeze(0).transpose(1, 2), k_i.unsqueeze(0).transpose(1, 2),
                v_i.unsqueeze(0).transpose(1, 2), attn_mask=mask, enable_gqa=True,
            )
            outs.append(out.transpose(1, 2).squeeze(0))
        return torch.cat(outs)

    def select_last_hidden(self, x, label="main"):
        return x[torch.tensor([seg.span for seg in self._kv.segments]).cumsum(0) - 1]


class _CountedPositions(StubResource):
    """The positions stand-in: one counter per request, a step's positions read off it, moved as the step declares."""

    def __init__(self):
        self.counters: dict[str, int] = {}
        self.planned = None

    def position(self, rid, label):
        return self.counters.get(rid, 0)

    def pos_ids(self, label):
        return self.planned

    def plan(self, step, ctx):
        self.planned = torch.cat([
            torch.arange(self.position(seg.request_id, seg.label), self.position(seg.request_id, seg.label) + seg.span)
            for seg in step.segments
        ])

    def commit(self, step, ctx):
        advance = step.advance or tuple(seg.span for seg in step.segments)
        for seg, n in zip(step.segments, advance, strict=True):
            self.counters[seg.request_id] = self.position(seg.request_id, seg.label) + n


class _GreedySampler(StubResource):
    """The sampler stand-in. It keeps no slot masters, so a decode step takes the ids it is routed."""

    has_slot_masters = False

    def sample(self, request_ids, logits, **kwargs):
        return logits.argmax(dim=-1)


class _StreamResources(_SdpaResources):
    """`_SdpaResources` as the reasoner needs them: a stream per request, causal reads over what it kept, counted
    positions and a greedy sampler."""

    def __init__(self):
        super().__init__()
        self.kv = _StreamKV()
        self.attn = _CausalSdpa(self.kv)
        self.positions = _CountedPositions()
        self.sampler = _GreedySampler()

    def as_dict(self):
        return {**super().as_dict(), SAMPLER: self.sampler}


@needs_ref
def test_reasoner_prompt_and_vision_match_reference() -> None:
    import torchvision
    from transformers import AutoTokenizer

    from mstar.model.components.mrope import layout_position_ids
    from mstar.model.cosmos3.components.reasoner import (
        expand_placeholders,
        preprocess_image,
        render_chat,
        vision_layout,
    )
    from mstar.model.cosmos3.components.vision import Cosmos3VisionModel
    from mstar.model.cosmos3.config import Cosmos3Config
    from mstar.model.cosmos3.loader import load_vision_encoder_weights
    from mstar.model.multimodal import PromptPart

    rec = torch.load(REF)
    cfg = Cosmos3Config.from_pretrained(EDGE_DIR)
    tok = AutoTokenizer.from_pretrained(str(EDGE_DIR))
    import json

    prompt = json.load(open(EDGE_DIR / "assets" / "example_reasoning_prompt.json"))["prompt"]
    image = torchvision.io.decode_image(str(EDGE_DIR / "assets" / "example_reasoning_input.png")).float() / 255.0

    # Preprocessing: same patches and grid as the HF processor.
    pv, grid = preprocess_image(image, cfg.reasoner.image_processor)
    assert list(grid.thw) == rec["image_grid_thw"][0].tolist()
    assert torch.allclose(pv, rec["pixel_values"], atol=1e-6), (pv - rec["pixel_values"]).abs().max()

    # Prompt: same token ids, same mRoPE positions.
    parts = [PromptPart("image", None, 0), PromptPart("text", prompt)]
    rendered = render_chat(tok, parts, cfg.reasoner, enable_thinking=False)
    text = expand_placeholders(rendered, tok, cfg.reasoner, [grid], [])
    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
    assert ids.tolist() == rec["input_ids"].tolist()
    layout = vision_layout(ids, cfg.reasoner, [grid], [], "vision_embeds")
    pos, next_pos = layout_position_ids(layout, cfg.reasoner.vision.spatial_merge_size, len(ids))
    assert torch.equal(pos, rec["position_ids"])
    assert next_pos == int(rec["position_ids"].max()) + 1
    assert next_pos == len(ids) + int(rec["rope_delta"])

    # Vision tower + projector.
    with torch.device("meta"):
        vision = Cosmos3VisionModel(cfg.reasoner)
    vision = vision.to_empty(device="cpu").float()
    load_vision_encoder_weights(vision, EDGE_DIR, device="cpu")
    vision.eval()
    with torch.no_grad():
        tokens = vision(pv, [grid.thw])
    ref = rec["vision_tokens"]
    assert tokens.shape == ref.shape
    err = (tokens - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert err <= 2e-3 * max(scale, 1.0), f"vision tokens max abs diff {err:.3e} (ref scale {scale:.2f})"


@needs_ref
def test_reasoner_logits_and_greedy_tokens_match_reference() -> None:
    from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
    from mstar.model.cosmos3.config import Cosmos3Config
    from mstar.model.cosmos3.loader import load_transformer_weights

    rec = torch.load(REF)
    cfg = Cosmos3Config.from_pretrained(EDGE_DIR)
    with torch.device("meta"):
        model = Cosmos3OmniTransformer(cfg)
    model = model.to_empty(device="cpu").float()
    load_transformer_weights(model, EDGE_DIR, device="cpu")
    model.eval()
    res = _StreamResources()
    res.bind(declaring(cfg), model)

    ids = rec["input_ids"]
    pos = rec["position_ids"]
    vision_tokens = rec["vision_tokens"]
    r = cfg.reasoner
    with torch.no_grad():
        embeds = model.embed_tokens(ids)
        mask = (ids == r.image_token_id) | (ids == r.video_token_id)
        embeds = embeds.masked_scatter(mask.unsqueeze(-1), vision_tokens.to(embeds.dtype))
        prompt = [("r", COND_LABEL, ids.numel())]
        hidden = res.step(C.PREFILL_WALK, prompt, lambda: model.text_forward(embeds, pos))
        logits = model.lm_head(hidden[-1:])[0]
    ref_logits = rec["prefill_last_logits"]
    err = (logits - ref_logits).abs().max().item()
    assert err <= 5e-2, f"prefill logits max abs diff {err:.3e}"
    assert int(logits.argmax()) == int(rec["greedy_tokens"][0])

    # Greedy decode of the remaining reference tokens over the cached prefix.
    next_pos = int(pos.max()) + 1
    token = int(logits.argmax())
    produced = [token]
    with torch.no_grad():
        for _ in range(len(rec["greedy_tokens"]) - 1):
            step_pos = torch.full((3, 1), next_pos, dtype=torch.long)
            embeds = model.embed_tokens(torch.tensor([token]))
            run = functools.partial(model.text_forward, embeds, step_pos)
            h = res.step(C.PREFILL_WALK, [("r", COND_LABEL, 1)], run)
            token = int(model.lm_head(h)[0].argmax())
            produced.append(token)
            next_pos += 1
    assert produced == rec["greedy_tokens"].tolist(), (produced, rec["greedy_tokens"].tolist())


def run_served_reasoner(model, cfg, encoder, ids, grid, vision_tokens, steps, rids=("a", "b")):
    """``steps`` greedy tokens of each of ``rids`` (identical requests packed
    per step) through ``Cosmos3ReasonerSubmodule``'s served path: prepare_inputs
    -> the step it declares, admitted and planned -> preprocess -> forward_batched
    -> commit, the vision prefill and then decode, with the node's handles over
    stand-in resources."""
    import types

    from mstar.engine.resources import StepContext, StepRunner
    from mstar.engine.state.adapters import node_handles, state_setup
    from mstar.model.cosmos3.components.reasoner import vision_layout
    from mstar.model.cosmos3.constants import REASONER_DECODE_WALK, REASONER_PREFILL_VISION_WALK
    from mstar.model.cosmos3.cosmos3_model import REASONER_NODE
    from mstar.model.cosmos3.submodules import Cosmos3ReasonerSubmodule
    from mstar.model.multimodal import MEDIA_LAYOUT
    from mstar.model.submodule_base import ModelInputsFromEngine, engine_step

    sub = Cosmos3ReasonerSubmodule(transformer=model, config=cfg, encoders=(encoder,))
    resources = _StreamResources().as_dict()
    cosmos3 = declaring(cfg)
    setup, problems = state_setup({}, cosmos3.declare_state(), cosmos3.kernel_settings())
    assert not problems, problems
    sub.bind_node_resources(resources)
    sub.bind_node_state(node_handles(cosmos3, REASONER_NODE, resources, setup=setup))
    runner = StepRunner(resources)
    rids = list(rids)
    layout = vision_layout(ids, cfg.reasoner, [grid], [], encoder.output)

    def step(walk, per_request_inputs):
        fwd = {
            rid: types.SimpleNamespace(request_id=rid, rid_handle=rid, step_metadata={MEDIA_LAYOUT: layout})
            for rid in rids
        }
        inputs = [sub.prepare_inputs(walk, fwd[rid], per_request_inputs[rid]) for rid in rids]
        declared = engine_step(sub, graph_walk=walk, request_ids=rids, inputs=inputs)
        assert [seg.label for seg in declared.segments] == [COND_LABEL] * len(rids)
        declared.set_ctx(StepContext(tuple(rids), walk, 0, False))
        assert runner.admit(declared).ok
        runner.plan(declared)
        declared.begin_calls()
        try:
            ei = ModelInputsFromEngine(request_ids=rids, per_request_info=fwd, step=declared)
            pre = sub.preprocess(walk, ei, inputs)
            with torch.no_grad():
                tokens = sub.forward_batched(walk, ei, **pre).row_outputs["new_token"]
            declared.check_calls()
            runner.commit(declared)
        finally:
            declared.end_calls()
        return tokens

    tokens = step(REASONER_PREFILL_VISION_WALK, {
        rid: {"text_inputs": [ids], "vision_embeds": [vision_tokens]} for rid in rids
    })
    produced = {rid: [int(tokens[i])] for i, rid in enumerate(rids)}
    for _ in range(steps - 1):
        tokens = step(REASONER_DECODE_WALK, {rid: {"text_inputs": [tokens[i:i + 1]]} for i, rid in enumerate(rids)})
        for i, rid in enumerate(rids):
            produced[rid].append(int(tokens[i]))
    return produced


@needs_ref
def test_reasoner_submodule_batched_path_matches_reference() -> None:
    """The served code path with two identical requests packed per step: both
    must reproduce the reference greedy tokens, which also checks the vision
    prompt's positions and decoding on from ``max(position) + 1``."""
    from mstar.model.cosmos3.components.reasoner import MediaGrid
    from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
    from mstar.model.cosmos3.loader import load_transformer_weights

    rec = torch.load(REF)
    served = Cosmos3Model(model_path_hf=str(EDGE_DIR), skip_weight_loading=True)
    cfg = served.config
    with torch.device("meta"):
        model = Cosmos3OmniTransformer(cfg)
    model = model.to_empty(device="cpu").float()
    load_transformer_weights(model, EDGE_DIR, device="cpu")
    model.eval()
    grid = MediaGrid(*(int(x) for x in rec["image_grid_thw"][0]))
    ref = rec["greedy_tokens"].tolist()
    produced = run_served_reasoner(
        model, cfg, served._vision_encoder(), rec["input_ids"], grid, rec["vision_tokens"], len(ref),
    )
    for rid, tokens in produced.items():
        assert tokens == ref, (rid, tokens, ref)
