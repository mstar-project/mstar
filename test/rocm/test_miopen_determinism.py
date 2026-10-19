"""A transposed conv repeats bit for bit under the platform's env defaults.

The conv is LTX-2.5's first vocoder upsampler. MIOpen's find put it on the split-K
backward-data solver on MI300X, which gave a different result on every call. It runs in
a fresh process that applies the defaults before the first device call, as the
conductor does for its workers.
"""

import json
import os
import subprocess
import sys

import pytest

from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")

CHILD = r"""
import json
from mstar.platforms import apply_process_env_defaults
apply_process_env_defaults()
import torch

torch.manual_seed(0)
conv = torch.nn.ConvTranspose1d(1536, 768, 11, stride=5, padding=3).to("cuda", torch.bfloat16)
x = torch.randn(1, 1536, 501).to("cuda", torch.bfloat16)
with torch.no_grad():
    outs = [conv(x) for _ in range(10)]
print(json.dumps({"distinct": len({o.cpu().view(torch.int16).numpy().tobytes() for o in outs})}))
"""


def test_bf16_conv_transpose_repeats():
    env = {k: v for k, v in os.environ.items() if not k.startswith("MIOPEN_DEBUG_CONV")}
    proc = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=600,
                          check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout.strip().splitlines()[-1])["distinct"] == 1
