#!/usr/bin/env python3
"""Record NVIDIA's GR00T N1.7 model code on ``record_dumps.py``'s requests, in mstar's environment.

The oracle GR00T's GPU tests and ``serve_check.py`` hold the port to. It runs NVIDIA's own
model classes (Isaac-GR00T), never any mstar code, but on mstar's torch, transformers and
diffusers, so that a difference between the port and this oracle is the port's, not the
stack's. Per request it writes ``req_XX.pt`` with:

    action_pred        the normalised action chunk ([1, 40, 132])
    noise              the initial noise the run started from
    vlln_out, vl_embeds, state_features
                       the VL bridge's LayerNorm and its output, and the state token
    backbone_features  the LLM's output (the bridge's input)
    vision_embeds, deepstack_embeds
                       the vision tower's merged output and its three DeepStack features

to ``<runs>/oracle_<bf16|fp32>[_seed<S>]``. The bf16 run against the fp32 one is the bf16
floor the tests measure the port against. ``--noise`` lists the runs to record, one model load
for all of them (loading takes minutes, recording 20 requests seconds): ``dump`` starts from
the noise NVIDIA's own run drew, and a number ``S`` from the noise mstar's server draws for
``seed: S`` (one CPU generator, seeded per request, at the weights' dtype:
``DenoiseLoopSubmodule`` and ``Gr00tN17DenoiseStepSubmodule.seed_latents``).

NVIDIA's package imports five small packages mstar's environment lacks; install them into
a folder of their own with ``--no-deps``, so nothing of mstar's is replaced:

    uv pip install --python <mstar>/.venv/bin/python --target ~/ref/oracle-overlay --no-deps \\
        tyro==0.9.17 typeguard==4.6.0 docstring-parser==0.18.0 shtab==1.12.1 dm-tree==0.1.10
    cd /tmp && PYTHONPATH=~/ref/oracle-overlay:~/ref/Isaac-GR00T \\
        <mstar>/.venv/bin/python <mstar>/test/gr00t_n17/record_oracle.py --runs ~/gr00t-runs [--fp32] \\
            --noise dump 7 8 9 10 11

``--runs`` is the folder that holds ``dump/`` and receives the recordings (default
``$GR00T_N17_ORACLE_DIR``, the setting the tests read). ``cd /tmp`` keeps a checkout's own
folders (``test``, ``gr00t``) from shadowing the packages on ``PYTHONPATH``.

The tests read the recorded noise and seeds 7-11 (``serve_check.py --n 200`` seeds 7-16 on
the larger set), in both dtypes. The fp32 model is 12 GB: ``--device cpu`` where it does not
fit. Record both dtypes on the GPU the tests will run on.

NVIDIA's code targets transformers 4.57; seven load-time shims bring it onto mstar's 5.x,
and none edits a forward. Two of them, ``mm_token_type_ids`` from ``input_ids`` and the LLM's
output taken before its final norm, are rules the port follows too, so this oracle cannot check
those two; ``record_dumps.py``'s dumps, from NVIDIA's own 4.57 stack, can
(``vlln_out`` against the dump's).
"""

from __future__ import annotations

import argparse
import glob
import importlib.abc
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

CONFIG_MODULE = "gr00t.configs.model.gr00t_n1d7"
TRAINING_MODULE = "gr00t.model.gr00t_n1d7.setup"
COLLATOR_MODULE = "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7"


class _NoCollator:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        raise RuntimeError("the oracle takes the dumps' collated inputs; nothing should collate")


class _ImportShims(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Shims 1 and 2, at import time.

    1. NVIDIA's ``Gr00tN1d7Config`` is a ``PretrainedConfig`` with an extra ``@dataclass``,
       which transformers 5.x refuses; it is imported with that decorator removed.
    2. The training pipeline (``setup``: datasets, scipy) and the image/text collator
       (albumentations, cv2) are stubbed: inference on collated inputs uses neither.
    """

    def find_spec(self, name, path, target=None):
        if name in (TRAINING_MODULE, COLLATOR_MODULE):
            return importlib.util.spec_from_loader(name, self)
        if name != CONFIG_MODULE:
            return None
        package = importlib.util.find_spec(name.rsplit(".", 1)[0]).submodule_search_locations
        self.config_file = os.path.join(list(package)[0], "gr00t_n1d7.py")
        return importlib.util.spec_from_loader(name, self, origin=self.config_file)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__name__ == TRAINING_MODULE:
            module.Gr00tN1d7Pipeline = None
            return
        if module.__name__ == COLLATOR_MODULE:
            module.Gr00tN1d7DataCollator = _NoCollator
            return
        src = Path(self.config_file).read_text()
        assert src.count("@dataclass\nclass Gr00tN1d7Config") == 1, "NVIDIA's config changed: re-check shim 1"
        module.__file__ = self.config_file
        exec(compile(src.replace("@dataclass\nclass Gr00tN1d7Config", "class Gr00tN1d7Config"),
                     self.config_file, "exec"), module.__dict__)


class _Help(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """The docstring as written, and each option's default."""


def build(device: str, dtype: torch.dtype):
    """NVIDIA's ``Gr00tN1d7`` with GR00T's own weights, on transformers 5.x (shims 3-7)."""
    sys.meta_path.insert(0, _ImportShims())
    # 3. transformers 5.x dropped Qwen3-VL's 4.57 aliases (``.language_model``, ``.visual``).
    #    NVIDIA's code uses them, and its RoPE repair looks them up with getattr, so a missing
    #    alias would skip that repair silently: restore the 4.57 definitions.
    from transformers import Qwen3VLForConditionalGeneration

    assert not hasattr(Qwen3VLForConditionalGeneration, "visual"), "transformers restored the alias: drop shim 3"
    Qwen3VLForConditionalGeneration.language_model = property(lambda self: self.model.language_model)
    Qwen3VLForConditionalGeneration.visual = property(lambda self: self.model.visual)

    import gr00t.model.gr00t_n1d7.gr00t_n1d7 as gr00t_model
    from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig

    # 4. transformers 5.x builds models on the meta device, where NVIDIA's nested backbone
    #    ``from_pretrained`` is refused: build on the CPU, then load GR00T's own checkpoint
    #    strictly, the end state NVIDIA's loader reaches (every weight from the GR00T file).
    # 5. NVIDIA's RoPE repair recomputes ``inv_freq`` buffers that 4.57's nested meta load left
    #    unset; built on the CPU, 5.x's constructor already computed them with the same formula,
    #    so the repair is skipped and the buffers are checked instead.
    Qwen3Backbone._reset_rotary_inv_freq = lambda self: None
    ckpt = snapshot_download("nvidia/GR00T-N1.7-3B")
    model = gr00t_model.Gr00tN1d7(AutoConfig.from_pretrained(ckpt))
    expected, seen = set(model.state_dict()), set()
    for shard in sorted(glob.glob(f"{ckpt}/model-*.safetensors")):  # one at a time: the fp32 CPU run's RAM
        weights = load_file(shard)
        _, unexpected = model.load_state_dict(weights, strict=False)
        assert not unexpected, unexpected
        seen |= set(weights)
        del weights
    assert seen == expected, f"weights never loaded: {sorted(expected - seen)}"
    rope = [b for n, b in model.named_buffers() if n.endswith("inv_freq")]
    assert rope and all(b.dtype == torch.float32 and b.isfinite().all() for b in rope)
    model.eval().to(device=device, dtype=dtype)

    qwen = model.backbone.model
    image_token_id = qwen.config.image_token_id

    # 6. transformers 5.x's Qwen3-VL needs ``mm_token_type_ids`` (0 text, 1 image) for its 3D
    #    RoPE, which 4.57 derived from ``input_ids``; NVIDIA's backbone passes four keys, so it is
    #    added at the Qwen3-VL call by 4.57's rule.
    def add_token_types(_module, args, kwargs):
        kwargs["mm_token_type_ids"] = (kwargs["input_ids"] == image_token_id).long()
        return args, kwargs

    qwen.register_forward_pre_hook(add_token_types, with_kwargs=True)
    # 7. 4.57's ``hidden_states[-1]`` is the last layer's output before the final norm, which
    #    GR00T was trained on; 5.x puts the normed output there, so the final norm is removed.
    qwen.model.language_model.norm = torch.nn.Identity()
    return model, qwen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=_Help)
    ap.add_argument("--runs", type=Path,
                    default=Path(os.environ.get("GR00T_N17_ORACLE_DIR", Path.home() / "gr00t-runs")),
                    help="holds dump/ and the recordings; $GR00T_N17_ORACLE_DIR when set")
    ap.add_argument("--dump", type=Path, help="the requests, if not <runs>/dump")
    ap.add_argument("--n", type=int, default=20, help="requests to record, the first n of the dump")
    ap.add_argument("--fp32", action="store_true", help="the fp32 run, not bf16")
    ap.add_argument("--noise", nargs="+", default=["dump"],
                    help="the runs: dump (NVIDIA's own draw) and/or seeds of mstar's server draw")
    ap.add_argument("--device", default="cuda", help="cpu where the fp32 model does not fit the GPU")
    args = ap.parse_args()
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    dump = args.dump or args.runs / "dump"
    if any(n != "dump" and not n.isdigit() for n in args.noise):
        ap.error(f"--noise takes dump and seed numbers, not {args.noise}")
    runs = {n: args.runs / (f"oracle_{'fp32' if args.fp32 else 'bf16'}" + ("" if n == "dump" else f"_seed{n}"))
            for n in args.noise}
    for out in runs.values():
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} already holds a recording; move it aside first")

    torch.set_grad_enabled(False)
    model, qwen = build(args.device, dtype)
    head = model.action_head
    captured: dict[str, torch.Tensor] = {}
    for name in ("vlln", "vl_self_attention", "state_encoder"):
        getattr(head, name).register_forward_hook(lambda _m, _i, o, name=name: captured.__setitem__(name, o.detach()))
    # the backbone's stages, for the per-stage bars: the LLM's output is vlln's input; the
    # vision tower's outputs are read inside its hook, because get_image_features replaces both
    # with per-image splits once the call returns
    head.vlln.register_forward_hook(lambda _m, i, _o: captured.__setitem__("backbone_features", i[0].detach().clone()))
    qwen.model.visual.register_forward_hook(lambda _m, _i, o: captured.update(
        vision_embeds=o.pooler_output.detach().clone(),
        deepstack_embeds=torch.stack(o.deepstack_features).detach().clone()))

    def cast(x):
        if torch.is_tensor(x):
            x = x.to(args.device)
            return x.to(dtype) if x.is_floating_point() else x
        if isinstance(x, dict):
            return {k: cast(v) for k, v in x.items()}
        return x

    real_randn = torch.randn
    requests = sorted(dump.glob("req_*.pt"))[: args.n]
    for which, out in runs.items():
        out.mkdir(parents=True, exist_ok=True)
        for path in requests:
            d = torch.load(path, weights_only=False)
            t0 = time.time()
            noise = d["noise"] if which == "dump" else torch.randn(
                (40, 132), generator=torch.Generator(device="cpu").manual_seed(int(which)), dtype=torch.bfloat16)[None]
            # the head's one torch.randn returns this run's noise instead
            torch.randn = lambda *a, noise=noise, **k: noise.to(device=k.get("device", args.device),
                                                               dtype=k.get("dtype", dtype))
            try:
                pred = model.get_action(**cast(d["collated"]))
            finally:
                torch.randn = real_randn
            rec = {"vlln_out": captured["vlln"], "vl_embeds": captured["vl_self_attention"],
                   "state_features": captured["state_encoder"], "action_pred": pred["action_pred"], "noise": noise,
                   **{k: captured[k] for k in ("backbone_features", "vision_embeds", "deepstack_embeds")}}
            torch.save({k: v.cpu() for k, v in rec.items()}, out / path.name)
            line = f"{out.name}/{path.name} [{time.time() - t0:.0f} s]"
            if which == "dump":  # the same computation as NVIDIA's own run, on another stack
                diff = (pred["action_pred"].float().cpu() - d["action_pred"].float()).abs()
                line += f" against NVIDIA's own stack: mean {diff.mean():.2e}, max {diff.max():.2e}"
            print(line)
        print("wrote", out)


if __name__ == "__main__":
    main()
