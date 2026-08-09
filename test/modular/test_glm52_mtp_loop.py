"""M3 draft-loop integration tests (CPU, reduced config).

THE property: at temp 0, MTP-on emits the bit-identical token stream to
MTP-off — greedy verify guarantees it by construction, and these tests
execute the guarantee through the real submodule protocol
(prepare_inputs → preprocess → forward_batched → postprocess → check_stop)
with ``ReferenceCacheHandle`` standing in for the engine. Draft quality is
irrelevant to the property (random weights draft ~nothing); acceptance
length is a box measurement, not a CPU test.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Same import-time stubs as test_glm52_mtp.py — construction-level tests on
# machines without CUDA wheels.
if "flashinfer" not in sys.modules:
    def _cpu_rmsnorm(x, weight, eps=1e-6):
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        return (normed * weight.float()).to(x.dtype)

    _fi = types.ModuleType("flashinfer")
    _fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    sys.modules["flashinfer"] = _fi

from mstar.model.glm52._testing import ReferenceCacheHandle  # noqa: E402
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.config import Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402


class _ArgmaxSampler:
    """Greedy stand-in for the engine sampler (temp 0, no penalties)."""

    def sample(self, request_ids, logits, apply_penalty=True):
        return logits.argmax(dim=-1)


class _EngineInputs:
    def __init__(self, cache_manager, sampler):
        self.cache_manager = cache_manager
        self.sampler = sampler


def _mtp_cfg(k: int) -> Glm52ModelConfig:
    cfg = Glm52ModelConfig.reduced()
    cfg.num_hidden_layers = 4  # MTP position lands FULL (4 = offset-1 + freq)
    cfg.mtp_num_draft_tokens = k
    return cfg


def _fwd_info(max_tokens: int, ignore_eos: bool) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="r0",
        max_tokens=max_tokens,
        sampling_config={"LLM": SimpleNamespace(ignore_eos=ignore_eos)},
        dynamic_loop_iter_counts={},
    )


def _drive(sub, handle, prompt, fwd_info, max_steps=64):
    """Run prefill + decode through the submodule protocol until stop."""
    ei = _EngineInputs(handle, _ArgmaxSampler())
    emitted = []
    walk, text_inputs, decode_step = "prefill", prompt, 0
    for _ in range(max_steps):
        ar = sub.prepare_inputs(walk, fwd_info, {"text_inputs": [text_inputs]})
        kw = sub.preprocess(walk, ei, [ar])
        out = sub.forward_batched(walk, ei, **kw)["r0"]
        sub.postprocess("r0", fwd_info, out)
        emitted.append(out["new_token"][0])
        if walk == "decode":
            # Match the engine's flag-off accounting: after decode step j
            # (0-based), generated = j + 2 = tokens emitted so far.
            fwd_info.dynamic_loop_iter_counts["decode_loop"] = decode_step
            decode_step += 1
        if sub.check_stop("r0", fwd_info, out):
            break
        walk, text_inputs = "decode", out["text_inputs"][0]
    return torch.cat(emitted)


def _run_pair(k, max_tokens, ignore_eos, seed=0, eos_ids=None):
    """One model, two runs: flag toggled off then on. Same weights by
    construction — toggling beats reseeding because the MTP module's
    parameters would otherwise shift the trunk's RNG draws."""
    torch.manual_seed(seed)
    cfg = _mtp_cfg(k)
    if eos_ids is not None:
        cfg.eos_token_ids = eos_ids
    model = Glm52ForCausalLM(cfg)
    prompt = torch.arange(5, dtype=torch.long) + 3

    streams, handles = [], []
    for mode_k in (0, k):
        cfg.mtp_num_draft_tokens = mode_k
        sub = Glm52LLMSubmodule(model, cfg)
        handle = ReferenceCacheHandle(["r0"])
        streams.append(_drive(
            sub, handle, prompt, _fwd_info(max_tokens, ignore_eos)))
        handles.append(handle)
    cfg.mtp_num_draft_tokens = k
    return streams, handles


def test_mtp_stream_matches_baseline_bitwise():
    (base, spec), _ = _run_pair(k=2, max_tokens=24, ignore_eos=True)
    assert torch.equal(base, spec), (
        f"MTP-on diverged from baseline: {base.tolist()} vs {spec.tolist()}")


def test_mtp_stream_matches_with_k3():
    (base, spec), _ = _run_pair(k=3, max_tokens=17, ignore_eos=True, seed=1)
    assert torch.equal(base, spec)


def test_mtp_stops_exactly_at_max_tokens():
    (base, spec), _ = _run_pair(k=2, max_tokens=7, ignore_eos=True)
    assert base.shape[0] == 7  # engine flag-off accounting
    assert spec.shape[0] == 7  # in-step budget truncation
    assert torch.equal(base, spec)


def test_mtp_eos_truncation_matches_baseline():
    # Data-driven EOS: pick a token the baseline emits mid-stream, declare
    # it a stop id, and rerun both modes — they must stop at exactly its
    # first occurrence, even when it lands inside an accepted draft run.
    (base, _), _ = _run_pair(k=2, max_tokens=24, ignore_eos=True)
    eos = int(base[len(base) // 2])
    (base2, spec2), _ = _run_pair(
        k=2, max_tokens=24, ignore_eos=False, eos_ids=(eos,))
    first = (base2 == eos).nonzero()[0, 0]
    assert base2.shape[0] == int(first) + 1
    assert torch.equal(base2, spec2)


def test_mtp_trunk_kv_and_plane_bookkeeping():
    (base, spec), (h_off, h_on) = _run_pair(k=2, max_tokens=16, ignore_eos=True)
    assert torch.equal(base, spec)
    # Committed trunk KV must trail the baseline's (accepted rows are
    # recomputed identically; rejected tails sit above the counter).
    n = min(h_off._states["r0"].position_id_start,
            h_on._states["r0"].position_id_start)
    for layer in range(4):
        off_rows = h_off.committed_rows("r0", layer)
        on_rows = h_on.committed_rows("r0", layer)
        for p in range(n):
            for c in (0, 1):
                # equal_nan: the unloaded fixture model has uninitialized
                # (torch.empty) params, so NaNs appear — identically in
                # both runs, which is exactly the assertion.
                assert torch.allclose(
                    off_rows[p][c].float(), on_rows[p][c].float(),
                    atol=1e-5, rtol=1e-4, equal_nan=True,
                ), f"layer {layer} pos {p} KV diverged"
    # The MTP plane exists only in the flag-on run, holds exactly counter
    # entries (the shift-by-one alignment), and the baseline never touched
    # layer 4.
    assert ("r0", 4) not in h_off._store
    on_plane = h_on.committed_rows("r0", 4)
    assert len(on_plane) == h_on._states["r0"].position_id_start
