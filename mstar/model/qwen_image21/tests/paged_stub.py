"""A CPU stand-in for the paged KV cache and its attention, and a tiny dit node driven over it, for tests.

The stand-in keeps each ``(request, label)`` stream's committed K/V per layer and attends a step's
rows over them with SDPA: a causal step's queries see everything before them plus a bottom-right
triangle over their own keys (FlashInfer's causal mask over a paged prefix), a non-causal step's see
all of it. Driven either row by row (``step``) or from a declared ``SubmoduleStep`` (``begin`` /
``bind_step`` / ``end``), whose segments, causality, commit, combined labels and post-forks it follows.

``dit_node`` builds the dit node over a tiny transformer for a 2-reference edit under CFG, and
``drive`` / ``denoise`` run the model's walk plan through it as the engine does.
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import torch
import torch.nn.functional as F

from mstar.engine.resources.kv.plan import group_by_plan_label
from mstar.model.qwen_image21.config import QwenImage21Config, QwenImage21TransformerConfig

CFG = QwenImage21TransformerConfig(
    in_channels=8, out_channels=8, num_layers=3, attention_head_dim=16, num_attention_heads=4, context_in_dim=24,
    mlp_ratio=3, axes_dims_rope=(4, 6, 6),
)
# text, (h, w) image, text, (h, w) image, text; then the target grid
LAYOUT = (5, (4, 6), 3, (6, 4), 7)
TARGET = (4, 4)
IMAGE_PAD_ID = 999
DIT_CONFIG = QwenImage21Config(transformer=CFG, text=replace(QwenImage21Config().text, image_token_id=IMAGE_PAD_ID))
STEPS, SCALE = 3, 4.5


class Row(NamedTuple):
    stream: tuple  # (request, label)
    length: int


class PagedAttendStub:
    def __init__(self, kv_key: str = "kv", attn_key: str = "attn"):
        self.kv_key, self.attn_key = kv_key, attn_key
        self.cache: dict[tuple, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        self.rows: list[Row] = []
        self.causal = False
        self.commit = False
        self.layer = 0
        self._declared = None

    def step(self, rows: list[Row], causal: bool, commit: bool) -> "PagedAttendStub":
        self.rows, self.causal, self.commit = list(rows), causal, commit
        return self

    # ------------------------------------------------------------ declared steps
    def begin(self, declared) -> None:
        self._declared = declared
        kv, attn = declared.steps[self.kv_key], declared.steps[self.attn_key]
        # a cfg rank's KV step holds every label; its attention reads its own
        assert {tuple(s) for s in attn.segments} <= {tuple(s) for s in kv.segments}
        self.causal, self.commit = attn.causal, kv.commit

    def bind_step(self, label: str) -> None:
        # the rows in the KV plan's packed order (a combined label's sources label-major)
        kv = self._declared.steps[self.kv_key]
        grouped = group_by_plan_label(tuple(kv.segments), kv.combined_labels)
        self.rows = [Row((s.request_id, s.label), s.span) for s in grouped[label]]

    def end(self, declared) -> None:
        kv = declared.steps[self.kv_key]
        for src, dst in kv.post_forks:
            for rid in kv.fork_rids if kv.fork_rids is not None else {s.request_id for s in kv.segments}:
                self.fork((rid, src), (rid, dst))
        self._declared = None

    def fork(self, src: tuple, dst: tuple) -> None:
        self.cache[dst] = {layer: (k.clone(), v.clone()) for layer, (k, v) in self.cache.get(src, {}).items()}

    def set_layer_idx(self, layer_idx: int) -> None:
        self.layer = layer_idx

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out, offset = [], 0
        for row in self.rows:
            qs, ks, vs = (t[offset:offset + row.length] for t in (q, k, v))
            offset += row.length
            stored = self.cache.setdefault(row.stream, {}).get(self.layer)
            if stored is not None:
                ks, vs = torch.cat([stored[0], ks]), torch.cat([stored[1], vs])
            if self.commit:
                self.cache[row.stream][self.layer] = (ks, vs)
            mask = None
            if self.causal:
                n, total = qs.shape[0], ks.shape[0]
                mask = torch.ones(n, total, dtype=torch.bool, device=qs.device).tril(total - n)
            attn = F.scaled_dot_product_attention(
                qs.transpose(0, 1)[None], ks.transpose(0, 1)[None], vs.transpose(0, 1)[None], attn_mask=mask,
            )
            out.append(attn[0].transpose(0, 1))
        assert offset == q.shape[0], (offset, q.shape)
        return torch.cat(out)


# ----------------------------------------------------------------------
# the dit node: the model's walk plan run through it as the engine runs it
# ----------------------------------------------------------------------

def dit_request(dtype: torch.dtype):
    """A 2-reference edit under CFG: token ids past the system turn, the text embeds (the negative
    prompt shares every row through the last image, as a causal encoder gives it), the latents."""
    g = torch.Generator().manual_seed(3)
    pos = []
    for part in LAYOUT:
        pos += list(range(10, 10 + part)) if isinstance(part, int) else [IMAGE_PAD_ID] * (part[0] * part[1] // 4)
    neg = pos[:-LAYOUT[-1]] + [7, 8, 9, 6]
    pos_vl = torch.randn(len(pos), CFG.context_in_dim, generator=g).to(dtype)
    neg_vl = torch.cat([pos_vl[:len(neg) - 4], torch.randn(4, CFG.context_in_dim, generator=g).to(dtype)])
    refs = [torch.randn(h * w, CFG.in_channels, generator=g).to(dtype) for h, w in LAYOUT[1::2]]
    return torch.tensor(pos), torch.tensor(neg), pos_vl, neg_vl, refs


def dit_node(transformer, dtype: torch.dtype):
    """The dit node over ``transformer`` with a ``PagedAttendStub`` as its attention, the model's plan
    for ``dit_request``'s edit, and that edit's text embeds and reference latents."""
    from mstar.model.qwen_image21.layout import prompt_spans
    from mstar.model.qwen_image21.qwen_image21_model import QwenImage21Model
    from mstar.model.qwen_image21.submodules import QwenImage21DenoiseSubmodule

    pos, neg, pos_vl, neg_vl, refs = dit_request(dtype)
    layout = prompt_spans(pos, IMAGE_PAD_ID, list(LAYOUT[1::2]), TARGET, neg)
    model = QwenImage21Model(skip_weight_loading=True)
    model.set_config(DIT_CONFIG)
    kwargs = {"layout": layout.to_metadata(), "prompt_lengths": [], "drop": 0, "vision_grids": [],
              "num_inference_steps": STEPS, "guidance_scale": SCALE, "negative_prompt": "x"}
    plan = model.walk_plan("default", ["text", "image"], ["image"], {}, kwargs)
    sub = QwenImage21DenoiseSubmodule(transformer, DIT_CONFIG, loop_name="denoise_loop")
    attend = PagedAttendStub()
    sub.attend = lambda *_: attend
    return sub, attend, plan, torch.cat([pos_vl, neg_vl]), torch.cat(refs)


def info(rid, step, seed=5, **metadata):
    from mstar.conductor.request_info import CurrentForwardPassInfo

    return CurrentForwardPassInfo(request_id=rid, graph_walk=step.walk, fwd_index=0, random_seed=seed, max_tokens=1,
                                  rid_handle=ord(rid), step_metadata={**step.step_metadata, **metadata})


def drive(sub, attend, rid, plan, text_embeds, ref_latents, cut=None):
    """Run ``plan`` (the model's walk plan) through the dit as the engine does: prepare, declare, the
    stand-in admits it, preprocess, forward, commit. ``cut`` maps a span to the tokens the cache
    served of it (None: all of them), as the engine's prefix cut does."""
    from mstar.model.qwen_image21.submodules import SPAN, TEXT_EMBEDS
    from mstar.model.submodule_base import ModelInputsFromEngine

    for step in plan:
        if step.walk.startswith("encode"):
            continue
        if step.walk == "image_gen":
            return denoise(sub, attend, {rid: info(rid, step)})[rid]
        fwd = info(rid, step)
        engine = ModelInputsFromEngine(request_ids=[rid], per_request_info={rid: fwd})
        inputs = {TEXT_EMBEDS: [text_embeds]} if step.walk == "prefix_text" else {"ref_latents": [ref_latents]}
        inp = sub.prepare_inputs(step.walk, fwd, inputs)
        served = (cut or {}).get(step.step_metadata[SPAN], 0)
        if served is None:
            continue
        if served:
            inp = sub.split_inputs(step.walk, fwd, inp, served, inp.input_seq_len)
        declared = sub.declare_step(step.walk, [rid], [inp], per_request_info={rid: fwd})
        attend.begin(declared)
        sub.forward(step.walk, engine, **sub.preprocess(step.walk, engine, [inp]))
        attend.end(declared)
    raise AssertionError("the plan has no image_gen step")


def denoise(sub, attend, infos):
    """The ``image_gen`` loop for the requests of ``infos`` (rid -> its info), batched in one step each
    iteration as the engine batches rows of one step key; each request's final latents."""
    from mstar.model.components.diffusion.denoise_loop import LATENTS
    from mstar.model.submodule_base import ModelInputsFromEngine

    rids = list(infos)
    latents = dict.fromkeys(rids)
    engine = ModelInputsFromEngine(request_ids=rids, per_request_info=infos)
    for k in range(infos[rids[0]].step_metadata["num_inference_steps"]):
        inputs = []
        for rid in rids:
            infos[rid].dynamic_loop_iter_counts = {"denoise_loop": k}
            inputs.append(sub.prepare_inputs("image_gen", infos[rid], {} if k == 0 else {LATENTS: [latents[rid]]}))
        declared = sub.declare_step("image_gen", rids, inputs, per_request_info=infos)
        attend.begin(declared)
        out = sub.forward_batched("image_gen", engine, **sub.preprocess("image_gen", engine, inputs))
        latents = {rid: out[rid][LATENTS][0] for rid in rids}
    return latents
