"""qwen-tts reference audio, run in its own process with an overlay first on sys.path.

qwen-tts 0.1.1 pins transformers 4.57.3 and the serving venv has 5.x, so Qwen3Tts.make_reference
runs this file with that transformers in --overlay.

    python -I qwen_tts_ref.py --overlay DIR --request FILE
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path


def stub_unused_deps() -> None:
    """qwen_tts imports its 25 Hz tokenizer eagerly, which needs sox and onnxruntime. The 12 Hz
    checkpoints never call into it, so empty modules are enough when those aren't installed."""
    for name in ("sox", "onnxruntime"):
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            continue
        stub = types.ModuleType(name)
        stub.__spec__ = importlib.machinery.ModuleSpec(name, None)  # find_spec() callers expect one

        def missing(attr, name=name):
            # AttributeError, so hasattr() and inspect see a plain empty module
            raise AttributeError(f"{name} is not installed (only qwen_tts' 25 Hz tokenizer uses it), no {attr!r}")

        stub.__getattr__ = missing
        sys.modules[name] = stub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--request", required=True)
    args = ap.parse_args()
    req = json.loads(Path(args.request).read_text())
    sys.path[:0] = [args.overlay, req["runner_root"]]
    stub_unused_deps()

    import numpy as np
    import torch
    import transformers
    from qwen_tts import Qwen3TTSModel

    from benchmark.correctness.metrics import write_wav

    out = Path(req["out_dir"])
    model = Qwen3TTSModel.from_pretrained(req["hf_id"], dtype=torch.bfloat16, attn_implementation="sdpa",
                                          device_map=req["device"])
    for i, it in enumerate(req["items"]):
        torch.manual_seed(i)
        if req["mode"] == "design":
            wavs, sr = model.generate_voice_design(text=it["text"], language="English", instruct=req["instruct"])
        elif req["mode"] == "clone":
            wavs, sr = model.generate_voice_clone(text=it["text"], language="English", ref_audio=it["prompt_audio"],
                                                  ref_text=it["prompt_text"])
        else:
            wavs, sr = model.generate_custom_voice(text=it["text"], language="English", speaker=req["voice"])
        x = np.asarray(wavs[0], dtype=np.float32)
        write_wav(out / f"{it['id']}.wav", x, sr)
        print(f"  ref {it['id']}: {len(x) / sr:.2f}s", flush=True)
    (out.parent / "ref_env.json").write_text(json.dumps({"transformers": transformers.__version__,
                                                         "transformers_file": transformers.__file__,
                                                         "torch": torch.__version__}))


if __name__ == "__main__":
    main()
