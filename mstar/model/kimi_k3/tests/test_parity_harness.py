"""The GPU parity harness (``test/kimi_k3/gpu_truncated_parity.py``) on CPU: what it reads as a
step's logits must be that step's, also when the step is a captured replay."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from mstar.model.kimi_k3 import truncate
from mstar.model.kimi_k3.tests.reference import reference_logits
from mstar.model.kimi_k3.tests.test_truncate import TINY, make_tiny_release

ROOT = Path(__file__).resolve().parents[4]


def _harness():
    spec = importlib.util.spec_from_file_location("kimi_gpu_parity", ROOT / "test/kimi_k3/gpu_truncated_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ReplayList(list):
    """A sampler's logits list as a captured replay leaves it: the sampler's Python ran at capture
    only, so nothing past the first entry is appended."""

    def append(self, item):
        if not self:
            super().append(item)


def test_generate_reads_each_steps_logits_even_when_the_sampler_runs_no_python(tmp_path):
    """GPU job 1461: decode replayed captured graphs, the harness read logits from the sampler's
    Python list, which stopped at the prefill's, and every dense case failed (error ~10, the
    control's size; argmax agreeing on the prefill row only). The logits now come back as the step's
    row output and must be the sampled token's argmax; with the list frozen they match the reference."""
    harness = _harness()
    harness._cpu_stubs()
    (tmp_path / "release").mkdir()
    ref_sd = make_tiny_release(tmp_path / "release")
    # the harness's input: a cut of the release with its experts decoded to bf16 (here every layer)
    truncate.main(["--src", str(tmp_path / "release"), "--dst", str(tmp_path / "cut"), "--layers", "5",
                   "--experts", str(TINY["experts"])])
    engine, model = harness.build(str(tmp_path / "cut"), "cpu", dtype=torch.float32)
    engine._resources["sampler"].logits = _ReplayList()
    prompt = torch.randint(0, model.config.vocab_size, (9,), generator=torch.Generator().manual_seed(3))
    run = harness.generate(engine, {"a": prompt}, steps=4)["a"]
    seq = torch.cat([prompt, torch.tensor(run["tokens"][:-1])])
    ref = reference_logits({k: v.float() for k, v in ref_sd.items()}, seq, model.config,
                           logits_from=len(prompt) - 1)
    torch.testing.assert_close(run["logits"], ref, atol=2e-4, rtol=0)
