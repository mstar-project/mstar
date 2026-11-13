#!/usr/bin/env python3
"""Step-Audio2 LLM parity against the reference oracle (GPU).

The ``Qwen3DenseLM`` (Qwen2 flags) runs on the engine's own resources, built from their
specs and driven declare -> admit -> plan -> forward -> commit one step at a time as the
engine does: paged KV (FlashInfer), causal attention and RoPE positions. For each of the
four oracle cases it is fed the reference's prefill embeddings (``inputs_embeds``, so the
LLM is compared alone) and compared on every layer's hidden states and the last position's
logits. Then 32 greedy tokens, two ways:

* teacher forced on the reference's ``greedy``: at every step whose reference top-2 margin
  is above ``CONFIDENT_MARGIN`` the port must pick the reference's token;
* free running: its tokens must agree with one of the reference's own greedy runs (bf16
  default, bf16 math SDPA, fp32; ``greedy_spread.py``) for at least as long as those agree
  with each other. On s2s the bf16 reference and the fp32 one part at token 12, a
  near-tie, so 32 identical tokens is not a property even the reference has.

Yardsticks: the reference against itself under the math SDPA backend (``llm_s2t_math``), and
both against the fp32 reference (``fp32_<case>``): the port's bf16 distance from fp32 must be
within ``SPREAD`` times the bf16 reference's own, per layer and on the logits.

The port's arithmetic without bf16 rounding: the fp32 port on test-only dense causal SDPA and
HF's rotate-half RoPE and RMSNorm in fp32 (the engine's kernels are bf16) against the fp32 reference, which must
agree to ``FP32_TOL`` on every layer.

Negative control: RoPE at theta 1e4 rather than the checkpoint's 1e6 (the Llama-2 default, a
plausible convention slip), which must move the last logits by more than ``CONTROL_FACTOR``
times the served path's error.

The audio cases also run end to end, the prefill built from the M* audio tower's own
embeddings (``parity_encoder.py``'s served path), and report their greedy tokens.

Run: ``python test/step_audio2/parity_llm.py``; exits nonzero on any failed check.
"""
from __future__ import annotations

import gc

import torch
from _common import CASES, CKPT, DEVICE, Verdict, diff, oracle, setup_torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    KVSpec,
    KVStep,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    PositionStep,
    Segment,
    StepContext,
    StepRunner,
    SubmoduleStep,
    resolve_spec_dependencies,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.step_audio2.config import AUDIO_START, StepAudio2Config
from mstar.model.step_audio2.weight_loader import load_llm_weights

KV, ATTN, POS = "llm_kv", "llm_attn", "llm_pos"
SPREAD = 1.5
CONTROL_FACTOR = 3.0
# a teacher-forced step must pick the reference's token where its top-2 margin is above this
# (logits; the served prefill's logits are within ~0.25 of the reference's)
CONFIDENT_MARGIN = 0.5
FP32_TOL = 1e-4
GREEDY = 32


class LLMDriver:
    """One request at a time through the LLM's resources."""

    def __init__(self, lm: Qwen3DenseLM, config, max_seq_len: int = 4096):
        specs = [
            KVSpec(resource_key=KV, nodes={"llm"}, config=PagedKVConfig(
                num_layers=config.num_hidden_layers, num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim, max_seq_len=max_seq_len, num_qo_heads=config.num_attention_heads,
                max_num_pages=512, prefix_cache=False,
            )),
            AttentionSpec(resource_key=ATTN, nodes={"llm"}, config=AttentionConfig(kv_cache=KV)),
            PositionSpec(
                resource_key=POS, nodes={"llm"},
                config=PositionConfig(kv_cache=KV, rope_theta=config.rope_theta),
            ),
        ]
        by_key = resolve_spec_dependencies(specs)
        groups = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial(), instance=CommGroup.trivial())
        transfer = TransferEngineInfo("h", "h", LocalTransferEngine("h"))
        self.resources = {
            spec.resource_key: build_resource(spec, EngineResourceInfo(
                device=DEVICE, joint_comm_group=groups, transfer_engine_info=transfer, kv_dtype=torch.bfloat16,
                dependencies={key: by_key[key] for key in spec.depends_on()},
            ))
            for spec in specs
        }
        self.lm = lm
        self.runner = StepRunner(self.resources)

    def bind(self) -> None:
        for module in self.lm.modules():
            if hasattr(module, "bind_resources") and module is not self.lm:
                module.bind_resources(self.resources)

    def step(self, rid: str, embeds: torch.Tensor, capture_layers: bool = False):
        """One step of ``embeds [n, hidden]``: the post-norm hidden ``[n, hidden]`` and, with
        ``capture_layers``, the residual stream after every layer (HF's ``hidden_states[1:-1]``)."""
        step = SubmoduleStep(
            segments=[Segment(request_id=rid, label="main", span=embeds.shape[0])],
            steps={KV: KVStep(), ATTN: AttentionStep(causal=True), POS: PositionStep()},
        )
        step.set_ctx(StepContext(request_ids=(rid,), graph_walk="prefill", slot=0, capture=False))
        outcome = self.runner.admit(step)
        assert outcome.ok, f"admit failed: {outcome.reason}"
        self.runner.plan(step)
        layers = []
        originals = []
        if capture_layers:
            for layer in self.lm.layers:
                original = layer.forward_mlp_residual

                def spy(h, residual, next_norm, comm_group=None, _original=original):
                    out = _original(h, residual, next_norm, comm_group)
                    layers.append(out[1].clone())  # the stream is updated in place later
                    return out

                layer.forward_mlp_residual = spy
                originals.append(layer)
        try:
            hidden = self.lm(embeds, label="main")
        finally:
            for layer in originals:
                del layer.forward_mlp_residual
        self.runner.commit(step)
        return hidden, layers

    @torch.inference_mode()
    def prefill_and_decode(
        self, rid: str, embeds: torch.Tensor, steps: int, capture_layers: bool = True,
        forced: list[int] | None = None,
    ):
        """Prefill, then ``steps - 1`` decode steps, each fed the previous argmax or, with
        ``forced``, the reference's token (teacher forcing). Returns the prefill's hidden
        states, and every step's logits and argmax."""
        self.bind()
        self.runner.ingest_request(rid, {})
        try:
            hidden, layers = self.step(rid, embeds, capture_layers)
            step_logits = [self.lm.logits(hidden[-1:]).float()[0]]
            tokens = [int(step_logits[-1].argmax())]
            for i in range(steps - 1):
                fed = tokens[-1] if forced is None else forced[i]
                h, _ = self.step(rid, self.lm.embed(torch.tensor([fed], device=DEVICE)))
                step_logits.append(self.lm.logits(h[-1:]).float()[0])
                tokens.append(int(step_logits[-1].argmax()))
        finally:
            self.runner.remove_request(rid)
        return hidden, layers, step_logits, tokens


class DenseCausalAttention:
    """Test-only stand-in for the paged KV + attention pair, one request: causal SDPA over
    the keys written so far, GQA by repetition, in the activations' dtype (fp32 here)."""

    requires_kv_write = True

    def __init__(self):
        self._label, self._layer, self._kv, self._pending = "main", 0, {}, None

    @property
    def default_label(self):
        return self._label

    def set_default_label(self, label):
        self._label = label

    def set_default_layer_idx(self, idx):
        self._layer = idx

    def layer_view(self):
        return self._layer

    def write_kv(self, k, v):
        self._pending = (k, v)

    def run(self, q, kv_cache_layer, k=None, v=None):
        ks, vs = self._kv.setdefault(kv_cache_layer, ([], []))
        ks.append(self._pending[0])
        vs.append(self._pending[1])
        k_all, v_all = torch.cat(ks), torch.cat(vs)
        rep = q.shape[1] // k_all.shape[1]
        k_all, v_all = k_all.repeat_interleave(rep, dim=1), v_all.repeat_interleave(rep, dim=1)
        n_new, n_all = q.shape[0], k_all.shape[0]
        mask = torch.ones(n_new, n_all, dtype=torch.bool, device=q.device).tril(diagonal=n_all - n_new)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), k_all.transpose(0, 1), v_all.transpose(0, 1), attn_mask=mask)
        return out.transpose(0, 1)


class RotateHalfRope:
    """Test-only RoPE as HF's Qwen2 computes it: fp32 frequencies, rotate-half."""

    def __init__(self):
        self.next_pos = 0

    def fusable_rope_theta(self, head_dim):
        return None

    def apply_qk(self, q, k, label, rope_theta, **kwargs):
        n, _, d = q.shape
        pos = torch.arange(self.next_pos, self.next_pos + n, dtype=torch.float32, device=q.device)
        self.next_pos += n
        inv = 1.0 / (rope_theta ** (torch.arange(0, d, 2, dtype=torch.int64, device=q.device).float() / d))
        emb = torch.cat([torch.outer(pos, inv)] * 2, dim=-1)
        cos, sin = emb.cos()[:, None, :].to(q.dtype), emb.sin()[:, None, :].to(q.dtype)

        def rot(x):
            return torch.cat([-x[..., d // 2:], x[..., : d // 2]], dim=-1)

        return q * cos + rot(q) * sin, k * cos + rot(k) * sin


def _rms(x, weight, eps, gemma_mode=False):
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


def _add_rms(h, residual, weight, eps, gemma_mode=False):
    r = h + residual
    return _rms(r, weight, eps), r


@torch.inference_mode()
def fp32_prefill(lm32: Qwen3DenseLM, embeds: torch.Tensor):
    """The fp32 port's prefill on the test-only dense attention and RoPE, with RMSNorm in
    fp32 (the engine's kernel takes bf16 only, and would round): every layer's residual
    stream and the post-norm hidden."""
    from unittest import mock

    from mstar.model.components import norm

    with mock.patch.object(norm, "rms_norm", _rms), mock.patch.object(norm, "add_rms_norm", _add_rms):
        return _fp32_prefill(lm32, embeds)


def _fp32_prefill(lm32: Qwen3DenseLM, embeds: torch.Tensor):
    attn, rope = DenseCausalAttention(), RotateHalfRope()
    for module in lm32.modules():
        if hasattr(module, "bind_resources") and module is not lm32:
            module.bind_resources({ATTN: attn, KV: attn, POS: rope})
    layers = []
    for layer in lm32.layers:
        original = layer.forward_mlp_residual

        def spy(h, residual, next_norm, comm_group=None, _original=original):
            out = _original(h, residual, next_norm, comm_group)
            layers.append(out[1].clone())
            return out

        layer.forward_mlp_residual = spy
    try:
        hidden = lm32(embeds, label="main")
    finally:
        for layer in lm32.layers:
            del layer.forward_mlp_residual
    return [embeds, *layers[:-1], hidden], lm32.logits(hidden[-1:])[0]


def with_rope_theta(lm: Qwen3DenseLM, theta: float, fn, *args):
    """``fn(*args)`` with every layer's RoPE at ``theta`` (the layers pass theirs to the position
    resource, so that is where a wrong one would come from)."""
    saved = [layer.self_attn.rope_theta for layer in lm.layers]
    for layer in lm.layers:
        layer.self_attn.rope_theta = theta
    try:
        return fn(*args)
    finally:
        for layer, value in zip(lm.layers, saved, strict=True):
            layer.self_attn.rope_theta = value


def agreement(a: list[int], b: list[int]) -> int:
    """Leading tokens two greedy runs share."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def audio_embeds(rec: dict, lm: Qwen3DenseLM, tokens: torch.Tensor) -> torch.Tensor:
    """The prompt's embeddings with each ``<audio_start>``'s following run overwritten by
    the audio tower's rows, clip by clip (``modeling_step_audio_2.py:353-357``)."""
    ids = rec["input_ids"][0].to(DEVICE)
    embeds = lm.embed(ids).clone()
    starts = (ids == AUDIO_START).nonzero().flatten().tolist()
    lens = rec["feat_lens"].tolist()
    off = 0
    for s, n in zip(starts, lens, strict=True):
        embeds[s + 1:s + 1 + n] = tokens[off:off + n].to(embeds.dtype)
        off += n
    return embeds


def m_star_audio_tokens(rec: dict) -> torch.Tensor:
    import parity_encoder as pe

    from mstar.model.step_audio2.components.audio import request_layout

    config = StepAudio2Config.from_hf(CKPT).audio
    model = pe.build(config, torch.bfloat16)
    layout = request_layout([m.shape[-1] for m in rec["mels"]])
    _, tokens = pe.run(model, rec["mels"], layout, pe.key_prefix_resource(config, layout))
    del model
    return tokens


def main() -> None:
    setup_torch()
    verdict = Verdict()
    config = StepAudio2Config.from_hf(CKPT).llm
    with torch.device(DEVICE):
        lm = Qwen3DenseLM(config, attn_key=ATTN, kv_key=KV, pos_key=POS)
    load_llm_weights(lm, str(CKPT), DEVICE)
    lm = lm.to(torch.bfloat16).eval()
    driver = LLMDriver(lm, config)
    with torch.device(DEVICE):
        lm32 = Qwen3DenseLM(config, attn_key=ATTN, kv_key=KV, pos_key=POS)
    load_llm_weights(lm32, str(CKPT), DEVICE)
    lm32 = lm32.float().eval()
    math = oracle("llm_s2t_math")
    greedy_spread = oracle("greedy_spread")

    for case in CASES:
        ref16, ref32 = oracle(f"llm_{case}"), oracle(f"fp32_{case}")
        embeds = ref16["inputs_embeds"][0].to(DEVICE, torch.bfloat16)
        hidden, layers, step_logits, tokens = driver.prefill_and_decode(case, embeds, GREEDY)
        logits = step_logits[0]
        got = [embeds, *layers[:-1], hidden]
        n_layers = len(ref16["hidden_states"])
        assert len(got) == n_layers, (len(got), n_layers)
        print(f"\n== {case}: {embeds.shape[0]} prompt tokens")
        header = "layer | ref16~ref32 | served~ref16 | served~ref32"
        if case == "s2t":
            header += " | ref16~math"
        print(header)
        ratios = []
        for i in range(n_layers):
            spread = diff(ref16["hidden_states"][i], ref32["hidden_states"][i]).rel
            s16 = diff(got[i], ref16["hidden_states"][i]).rel
            s32 = diff(got[i], ref32["hidden_states"][i]).rel
            line = f"{i:5d} | {spread:.3e}   | {s16:.3e}    | {s32:.3e}"
            if case == "s2t":
                line += f"    | {diff(ref16['hidden_states'][i], math['hidden_states'][i]).rel:.3e}"
            print(line)
            if i:
                ratios.append(s32 / spread)
        l_spread = diff(ref16["last_logits"][0], ref32["last_logits"][0])
        l16, l32 = diff(logits, ref16["last_logits"][0]), diff(logits, ref32["last_logits"][0])
        print(f"logit | {l_spread.rel:.3e}   | {l16.rel:.3e}    | {l32.rel:.3e}   (max abs {l16.max_abs:.3f} vs ref16)")
        if case == "s2t":
            print(f"ref16~math logits {diff(ref16['last_logits'][0], math['last_logits'][0])}")
        verdict.check(max(ratios) <= SPREAD,
                      f"{case}: every layer within {SPREAD}x the bf16 reference's distance from fp32 "
                      f"(worst ratio {max(ratios):.2f})")
        verdict.check(l32.rel <= SPREAD * l_spread.rel,
                      f"{case}: last logits within {SPREAD}x ({l32.rel:.3e} vs {l_spread.rel:.3e})")
        got32, logits32 = fp32_prefill(lm32, ref32["inputs_embeds"][0].to(DEVICE))
        worst32 = max(diff(g, w).rel for g, w in zip(got32, ref32["hidden_states"], strict=True))
        worst32 = max(worst32, diff(logits32, ref32["last_logits"][0]).rel)
        verdict.check(worst32 < FP32_TOL, f"{case}: the fp32 port (test-only dense attention and RoPE) matches the "
                      f"fp32 reference, every layer and the logits (worst rel {worst32:.2e})")
        want = ref16["greedy"].tolist()
        variants = {"bf16": want, "math": greedy_spread[case]["math"], "fp32": greedy_spread[case]["fp32"]}
        names = list(variants)
        ref_floor = min(agreement(variants[a], variants[b]) for a in names for b in names if a < b)
        best = max(names, key=lambda n: agreement(tokens, variants[n]))
        free = agreement(tokens, variants[best])
        verdict.check(free >= min(ref_floor, len(want)),
                      f"{case}: free-running greedy agrees with the {best} reference for {free} tokens (the "
                      f"reference runs agree with each other for {ref_floor} of {len(want)}; with bf16 for "
                      f"{agreement(tokens, want)})")
        _, _, _, forced = driver.prefill_and_decode(f"{case}-forced", embeds, len(want), False, want)
        margins = greedy_spread[case]["margins"]
        missed = [i for i in range(len(want)) if margins[i] > CONFIDENT_MARGIN and forced[i] != want[i]]
        flips = [(i, round(margins[i], 3)) for i in range(len(want)) if forced[i] != want[i]]
        verdict.check(not missed, f"{case}: teacher forced, every confident step picks the reference's token "
                      f"(steps that flip, with the reference's top-2 margin: {flips})")
        control = with_rope_theta(lm, 1e4, driver.prefill_and_decode, f"{case}-theta", embeds, 1, False)[2][0]
        c = diff(control, ref16["last_logits"][0])
        verdict.check(c.rel > CONTROL_FACTOR * l16.rel,
                      f"{case}: negative control (rope_theta 1e4) diverges: logits rel {c.rel:.3e} vs served "
                      f"{l16.rel:.3e}")
        if "mels" in ref16 and ref16["mels"]:
            with torch.inference_mode():
                e2e = audio_embeds(ref16, lm, m_star_audio_tokens(ref16))
            gc.collect()
            _, _, e_logits, e_tokens = driver.prefill_and_decode(f"{case}-e2e", e2e, GREEDY, capture_layers=False)
            shared = {n: agreement(e_tokens, v) for n, v in variants.items()}
            print(f"{case} end to end (M* audio tower): prompt embeds rel {diff(e2e, embeds).rel:.3e}, logits rel "
                  f"{diff(e_logits[0], ref16['last_logits'][0]).rel:.3e}, greedy tokens shared with each reference "
                  f"run: {shared}")
    verdict.finish()


if __name__ == "__main__":
    main()
