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

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from mstar.model.cosmos3.tests.test_edge import EDGE_DIR

REF = os.environ.get("COSMOS3_EDGE_REF")
needs_ref = pytest.mark.skipif(
    EDGE_DIR is None or not REF or not Path(REF).exists(),
    reason="set COSMOS3_EDGE_REF to the HF reasoner dump and COSMOS3_EDGE_DIR to the snapshot",
)


class _SdpaKV:
    """kv + attn stand-in for one request: appends this step's K/V to the
    per-layer history and attends causally over [history | step]."""

    requires_kv_write = True

    def __init__(self):
        self._label = "main"
        self._layer = 0
        self.history: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._pending = None

    @property
    def default_label(self):
        return self._label

    def set_default_label(self, label):
        self._label = label

    def set_default_layer_idx(self, i):
        self._layer = i

    def layer_view(self, layer_idx=None):
        return self._layer if layer_idx is None else layer_idx

    def write_kv(self, k, v, layer_idx=None, label=None):
        self._pending = (k, v)

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        k, v = self._pending
        self._pending = None
        prev = self.history.get(kv_cache_layer)
        if prev is not None:
            k = torch.cat([prev[0], k])
            v = torch.cat([prev[1], v])
        self.history[kv_cache_layer] = (k, v)
        n_q = q.shape[0]
        n_k = k.shape[0]
        # Causal over the whole stream; queries are the last n_q positions.
        mask = torch.ones(n_q, n_k, dtype=torch.bool).tril(diagonal=n_k - n_q)
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2), v.unsqueeze(0).transpose(1, 2),
            attn_mask=mask, enable_gqa=True,
        )
        return out.transpose(1, 2).squeeze(0)


def _bind(module, res):
    for child in module.modules():
        bind = getattr(child, "bind_resources", None)
        if bind is not None:
            bind({"kv": res, "attn": res})


@needs_ref
def test_reasoner_prompt_and_vision_match_reference() -> None:
    import torchvision
    from transformers import AutoTokenizer

    from mstar.model.cosmos3.components.reasoner import (
        expand_placeholders,
        preprocess_image,
        prompt_position_ids,
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
    pos, next_pos = prompt_position_ids(layout, cfg.reasoner.vision.spatial_merge_size, len(ids))
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
    res = _SdpaKV()
    _bind(model, res)

    ids = rec["input_ids"]
    pos = rec["position_ids"]
    vision_tokens = rec["vision_tokens"]
    r = cfg.reasoner
    with torch.no_grad():
        embeds = model.embed_tokens(ids)
        mask = (ids == r.image_token_id) | (ids == r.video_token_id)
        embeds = embeds.masked_scatter(mask.unsqueeze(-1), vision_tokens.to(embeds.dtype))
        hidden = model.text_forward(embeds, pos, "main")
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
            h = model.text_forward(model.embed_tokens(torch.tensor([token])), step_pos, "main")
            token = int(model.lm_head(h)[0].argmax())
            produced.append(token)
            next_pos += 1
    assert produced == rec["greedy_tokens"].tolist(), (produced, rec["greedy_tokens"].tolist())


class _FakeSampler:
    """Greedy stand-in for the sampler resource. It keeps no slot masters, so
    a decode step takes the ids it is routed."""

    has_slot_masters = False

    def sample(self, request_ids, logits):
        return logits.argmax(dim=-1)


class _PackedSdpaKV(_SdpaKV):
    """Multi-request variant: the step's packed K/V splits into per-request
    streams by the declared spans (``plan``), each with its own history."""

    def __init__(self):
        super().__init__()
        self.spans: list[int] = []
        self.rids: list[str] = []
        self.history = {}  # (rid, layer) -> (k, v)

    def plan(self, rids, spans):
        self.rids, self.spans = list(rids), list(spans)

    def select_last_hidden(self, hidden):
        return hidden[torch.tensor(self.spans).cumsum(0) - 1]

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        k_all, v_all = self._pending
        self._pending = None
        outs, off = [], 0
        for rid, span in zip(self.rids, self.spans, strict=True):
            q_i, k_i, v_i = q[off:off + span], k_all[off:off + span], v_all[off:off + span]
            off += span
            prev = self.history.get((rid, kv_cache_layer))
            if prev is not None:
                k_i = torch.cat([prev[0], k_i])
                v_i = torch.cat([prev[1], v_i])
            self.history[(rid, kv_cache_layer)] = (k_i, v_i)
            n_q, n_k = q_i.shape[0], k_i.shape[0]
            mask = torch.ones(n_q, n_k, dtype=torch.bool).tril(diagonal=n_k - n_q)
            o = F.scaled_dot_product_attention(
                q_i.unsqueeze(0).transpose(1, 2), k_i.unsqueeze(0).transpose(1, 2), v_i.unsqueeze(0).transpose(1, 2),
                attn_mask=mask, enable_gqa=True,
            )
            outs.append(o.transpose(1, 2).squeeze(0))
        return torch.cat(outs)


class _Positions:
    """Position resource stand-in: one counter per request, each step's
    planned positions read off it, moved as the step declares."""

    def __init__(self):
        self.counters: dict[str, int] = {}
        self.planned = None

    def position(self, rid, label):
        return self.counters.get(rid, 0)

    def plan(self, step):
        self.planned = torch.cat([
            torch.arange(self.position(seg.request_id, seg.label), self.position(seg.request_id, seg.label) + seg.span)
            for seg in step.segments
        ])

    def pos_ids(self, label):
        return self.planned

    def commit(self, step, rope):
        advance = step.steps[rope].advance or tuple(seg.span for seg in step.segments)
        for seg, n in zip(step.segments, advance, strict=True):
            self.counters[seg.request_id] = self.position(seg.request_id, seg.label) + n


def run_served_reasoner(model, cfg, encoder, ids, grid, vision_tokens, steps, rids=("a", "b")):
    """``steps`` greedy tokens of each of ``rids`` (identical requests packed
    per step) through ``Cosmos3ReasonerSubmodule``'s served path: prepare_inputs
    -> declare_step -> preprocess -> forward_batched, the vision prefill and
    then decode, over stand-in kv/attn/sampler/position resources."""
    import types

    from mstar.model.cosmos3.components.reasoner import vision_layout
    from mstar.model.cosmos3.constants import REASONER_DECODE_WALK, REASONER_PREFILL_VISION_WALK
    from mstar.model.cosmos3.submodules import ATTN, KV_CACHE, SAMPLER, Cosmos3ReasonerSubmodule
    from mstar.model.keys import ROPE
    from mstar.model.multimodal import MEDIA_LAYOUT
    from mstar.model.submodule_base import ModelInputsFromEngine

    sub = Cosmos3ReasonerSubmodule(transformer=model, config=cfg, encoders=(encoder,))
    kv, pos = _PackedSdpaKV(), _Positions()
    resources = {KV_CACHE: kv, ATTN: kv, SAMPLER: _FakeSampler(), ROPE: pos}
    sub.bind_node_resources(resources)
    rids = list(rids)
    layout = vision_layout(ids, cfg.reasoner, [grid], [], encoder.output)

    def step(walk, per_request_inputs):
        fwd = {
            rid: types.SimpleNamespace(request_id=rid, rid_handle=rid, step_metadata={MEDIA_LAYOUT: layout})
            for rid in rids
        }
        inputs = [sub.prepare_inputs(walk, fwd[rid], per_request_inputs[rid]) for rid in rids]
        declared = sub.declare_step(walk, rids, inputs)
        assert [seg.label for seg in declared.segments] == ["main"] * len(rids)
        kv.plan(rids, [seg.span for seg in declared.segments])
        pos.plan(declared)
        ei = ModelInputsFromEngine(request_ids=rids, per_request_info=fwd, resources=resources)
        pre = sub.preprocess(walk, ei, inputs)
        with torch.no_grad():
            tokens = sub.forward_batched(walk, ei, **pre).row_outputs["new_token"]
        pos.commit(declared, ROPE)
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
