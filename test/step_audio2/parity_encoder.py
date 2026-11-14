#!/usr/bin/env python3
"""Step-Audio2 audio encoder + adapter parity against the reference oracle (GPU).

For the s2t clip (one clip) and s2t_long (a 25 s clip and a short one, so the key-mask rule's
``T_pad`` offset is exercised on a clip shorter than the batch), every encoder block's output
and the adapter's kept tokens are compared, in four configurations:

* **served**: bf16 through the ``RaggedKeyPrefixAttentionSpec`` resource (FlashInfer), against
  the bf16 oracle and against the fp32 reference. The yardstick is the bf16 reference's own
  distance from the fp32 reference: the adapter tokens must be no further from fp32 than
  ``TOKEN_SPREAD`` times that, each block no further than ``BLOCK_SPREAD`` times. Past block
  20 the stream carries outlier activations and any bf16 rounding moves the per-block number
  a lot: equally valid bf16 routes (FA3, the reference's own arithmetic on fp32 attention)
  land 0.85-1.8x the reference's distance there, so the block bound is looser.
* **reference shapes**: bf16 with a test-only attention that does the reference's arithmetic
  (``modeling_step_audio_2.py:191-206``: q and k each scaled by ``d ** -0.25`` in bf16, a bf16
  score matmul plus the additive mask, softmax in fp32), per clip, over every post-conv frame
  of the padded batch as the reference computes them. Every block must be bit-exact, and
  for one clip the adapter tokens too: that proves the port, and leaves the served path's
  residue to the attention kernel and to computing only the frames the kept tokens need
  (which changes GEMM shapes).
* **fp32**: the port in fp32 on that same reference-route attention, against the fp32
  reference: the port's arithmetic without bf16 rounding; must agree to ``FP32_TOL``.
* **served at padded widths**: the served path's convolutions at widths padded to the
  submodule's ``CONV_WIDTH_QUANTUM``, held to the served bounds; and the fp32 port so padded
  (reference-route attention) to ``FP32_TOL`` from the fp32 reference, which shows the padding
  changes no output, only the kernels' widths.
* **negative control**: the key mask off, as ``mask_probe.py`` runs the reference: each clip
  alone, every post-conv frame computed and a key. Must move the kept adapter tokens by
  more than ``CONTROL_FACTOR`` times the served error (the reference moves them 11-30%).

Run: ``python test/step_audio2/parity_encoder.py``; exits nonzero on any failed check.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from _common import AUDIO_CASES, CKPT, DEVICE, Verdict, diff, oracle, setup_torch

from mstar.engine.resources import KeyPrefixAttentionStep, RaggedAttentionConfig, RaggedKeyPrefixAttentionSpec, Segment
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.step import StepContext
from mstar.model.step_audio2.components.audio import ClipLayout, StepAudio2Audio, request_layout
from mstar.model.step_audio2.config import StepAudio2Config
from mstar.model.step_audio2.submodules import AudioEncoderSubmodule
from mstar.model.step_audio2.weight_loader import load_audio_weights

ATTN = "audio_attn"
TOKEN_SPREAD = 1.5
BLOCK_SPREAD = 2.5
FP32_TOL = 1e-4
CONTROL_FACTOR = 3.0
QUANTUM = AudioEncoderSubmodule.CONV_WIDTH_QUANTUM


class ReferenceRouteAttention:
    """Test-only stand-in for the key-prefix resource: the reference's own attention
    arithmetic, clip by clip, in the activations' dtype."""

    def __init__(self, layout: list[ClipLayout]):
        self.layout = layout

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        n, h, d = q.shape
        scale = d ** -0.25
        out, off = [], 0
        for clip in self.layout:
            t = clip.frames
            qs = q[off:off + t].permute(1, 0, 2)[None] * scale
            ks = k[off:off + t].permute(1, 2, 0)[None] * scale
            vs = v[off:off + t].permute(1, 0, 2)[None]
            mask = torch.zeros(t, device=q.device, dtype=q.dtype)
            mask[clip.key_len:] = -1.0e10
            qk = (qs @ ks + mask).float()
            w = F.softmax(qk, dim=-1).to(q.dtype)
            out.append((w @ vs).permute(0, 2, 1, 3)[0])
            off += t
        return torch.cat(out)


def reference_shapes(layout: list[ClipLayout]) -> list[ClipLayout]:
    """Every clip over the padded batch's post-conv frames, as the reference runs them."""
    padded = max(c.mel_frames for c in layout)
    frames = (padded - 1) // 2 + 1
    return [c._replace(conv_frames=padded, frames=frames) for c in layout]


def unmasked(layout: list[ClipLayout]) -> list[ClipLayout]:
    """Each clip alone with every post-conv frame a key: the reference with ``wav_lens = T``."""
    out = []
    for c in layout:
        frames = (c.mel_frames - 1) // 2 + 1
        out.append(c._replace(conv_frames=c.mel_frames, frames=frames, key_len=frames))
    return out


def key_prefix_resource(config, layout: list[ClipLayout]):
    spec = RaggedKeyPrefixAttentionSpec(
        resource_key=ATTN, nodes={"audio_encoder"},
        config=RaggedAttentionConfig(
            num_qo_heads=config.n_audio_head, num_kv_heads=config.n_audio_head, head_dim=config.head_dim,
            max_segments_per_request=8,
        ),
    )
    resource = build_resource(spec, EngineResourceInfo(device=DEVICE, kv_dtype=torch.bfloat16))
    step = KeyPrefixAttentionStep(
        segments=tuple(Segment(request_id="r0", label="main", span=c.frames) for c in layout),
        key_lens=tuple(c.key_len for c in layout),
    )
    resource.plan(step, StepContext(request_ids=["r0"], graph_walk="encode", slot=0, capture=False))
    return resource


def build(config, dtype) -> StepAudio2Audio:
    with torch.device(DEVICE):
        model = StepAudio2Audio(config, attn_key=ATTN)
    load_audio_weights(model, str(CKPT), DEVICE)
    return model.to(dtype).eval()


@torch.inference_mode()
def run(model: StepAudio2Audio, mels, layout, attention, quantum: int = 1) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Each block's output (packed) and the adapter's tokens."""
    for block in model.encoder.blocks:
        block.self_attn.bind_resources({ATTN: attention})
    blocks = []
    hooks = [b.register_forward_hook(lambda _m, _i, o: blocks.append(o.clone())) for b in model.encoder.blocks]
    try:
        tokens = model([m.to(DEVICE) for m in mels], layout, quantum)
    finally:
        for hk in hooks:
            hk.remove()
    return blocks, tokens


def per_clip(packed: torch.Tensor, layout: list[ClipLayout]) -> list[torch.Tensor]:
    return list(packed.split([c.frames for c in layout]))


def want_blocks(rec: dict, i: int, layout) -> list[torch.Tensor]:
    return [rec["encoder_blocks"][i][c, : clip.frames] for c, clip in enumerate(layout)]


def want_tokens(rec: dict, layout) -> torch.Tensor:
    return torch.cat([rec["adapter_out"][c, : clip.tokens] for c, clip in enumerate(layout)])


def compare(got_blocks, got_tokens, rec, layout) -> tuple[list[float], float, list[float]]:
    """Per block rel error (all clips), adapter rel error, per block max abs error."""
    rels, maxes = [], []
    for i, packed in enumerate(got_blocks):
        d = diff(packed, torch.cat(want_blocks(rec, i, layout)))
        rels.append(d.rel)
        maxes.append(d.max_abs)
    return rels, diff(got_tokens, want_tokens(rec, layout)).rel, maxes


def main() -> None:
    setup_torch()
    verdict = Verdict()
    config = StepAudio2Config.from_hf(CKPT).audio
    bf16 = build(config, torch.bfloat16)
    fp32 = build(config, torch.float32)
    for case in AUDIO_CASES:
        ref16, ref32 = oracle(f"llm_{case}"), oracle(f"fp32_{case}")
        mels = ref16["mels"]
        layout = request_layout([m.shape[-1] for m in mels])
        print(f"\n== {case}: clips {[(c.mel_frames, c.frames, c.key_len, c.tokens) for c in layout]}"
              " (mel frames, post-conv frames, keys, tokens)")
        assert [c.tokens for c in layout] == ref16["feat_lens"].tolist()

        # the yardstick: bf16 reference vs fp32 reference
        spread = [diff(torch.cat(want_blocks(ref16, i, layout)), torch.cat(want_blocks(ref32, i, layout))).rel
                  for i in range(config.n_audio_layer)]
        spread_tok = diff(want_tokens(ref16, layout), want_tokens(ref32, layout)).rel

        served = run(bf16, mels, layout, key_prefix_resource(config, layout))
        s_rel16, s_tok16, s_max16 = compare(*served, ref16, layout)
        s_rel32, s_tok32, _ = compare(*served, ref32, layout)
        ref_shapes = reference_shapes(layout)
        route_blocks, route_tokens = run(bf16, mels, ref_shapes, ReferenceRouteAttention(ref_shapes))
        r_rel16, r_tok16, r_max16 = compare(route_blocks, route_tokens, ref16, ref_shapes)
        exact32 = run(fp32, mels, layout, ReferenceRouteAttention(layout))
        f_rel32, f_tok32, _ = compare(*exact32, ref32, layout)
        padded = run(bf16, mels, layout, key_prefix_resource(config, layout), QUANTUM)
        q_rel32, q_tok32, _ = compare(*padded, ref32, layout)
        padded32 = run(fp32, mels, layout, ReferenceRouteAttention(layout), QUANTUM)
        fq_rel32, fq_tok32, _ = compare(*padded32, ref32, layout)
        off = unmasked(layout)
        _, control = run(bf16, mels, off, key_prefix_resource(config, off))
        c_tok16 = diff(control, want_tokens(ref16, layout)).rel
        print("block | ref16~ref32 | served~ref16 (max abs) | served~ref32 | ref-shapes~ref16 (max abs) | "
              "fp32~ref32")
        for i in range(config.n_audio_layer):
            print(f"{i:5d} | {spread[i]:.3e}   | {s_rel16[i]:.3e} ({s_max16[i]:.2e})   | {s_rel32[i]:.3e}    | "
                  f"{r_rel16[i]:.3e} ({r_max16[i]:.2e})       | {f_rel32[i]:.2e}")
        print(f"adapt | {spread_tok:.3e}   | {s_tok16:.3e}               | {s_tok32:.3e}    | {r_tok16:.3e}"
              f"                   | {f_tok32:.2e}")
        print(f"no-mask control: adapter tokens rel {c_tok16:.3e} from the bf16 oracle")
        block_ratio = max(a / b for a, b in zip(s_rel32, spread, strict=True))
        verdict.check(s_tok32 <= TOKEN_SPREAD * spread_tok,
                      f"{case}: served adapter tokens are within {TOKEN_SPREAD}x the bf16 reference's distance "
                      f"from fp32 ({s_tok32:.3e} vs {spread_tok:.3e})")
        verdict.check(block_ratio <= BLOCK_SPREAD,
                      f"{case}: every served block is within {BLOCK_SPREAD}x the bf16 reference's distance from "
                      f"fp32 (worst ratio {block_ratio:.2f})")
        verdict.check(max(f_rel32 + [f_tok32]) < FP32_TOL,
                      f"{case}: fp32 port matches the fp32 reference (worst rel {max(f_rel32 + [f_tok32]):.2e})")
        q_ratio = max(a / b for a, b in zip(q_rel32, spread, strict=True))
        verdict.check(q_tok32 <= TOKEN_SPREAD * spread_tok and q_ratio <= BLOCK_SPREAD,
                      f"{case}: served at widths padded to {QUANTUM} mel frames, the adapter tokens and every block "
                      f"are within the served bounds ({q_tok32:.3e}, worst block ratio {q_ratio:.2f})")
        verdict.check(max(fq_rel32 + [fq_tok32]) < FP32_TOL,
                      f"{case}: the fp32 port at padded widths matches the fp32 reference "
                      f"(worst rel {max(fq_rel32 + [fq_tok32]):.2e})")
        verdict.check(c_tok16 > CONTROL_FACTOR * s_tok16,
                      f"{case}: negative control (key mask off) diverges: tokens rel {c_tok16:.3e} vs served "
                      f"{s_tok16:.3e}")
        blocks_exact = all(m == 0.0 for m in r_max16)
        if len(layout) == 1:
            verdict.check(blocks_exact and r_tok16 == 0.0,
                          f"{case}: reference shapes and arithmetic are bit-exact against the bf16 oracle")
        else:
            # the adapter's linears run on the clips packed, the reference's on a padded batch
            verdict.check(blocks_exact, f"{case}: under reference shapes every block is bit-exact against the "
                          f"bf16 oracle (adapter tokens rel {r_tok16:.2e}: packed clips vs a padded batch)")
    verdict.finish()


if __name__ == "__main__":
    main()
