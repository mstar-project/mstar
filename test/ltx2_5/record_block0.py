#!/usr/bin/env python3
"""Record the reference DiT's block-0 intermediates at step 0, for localizing a
parity difference inside one block (oracle venv; see ``record_oracle.py``).

    .venv-ltx-oracle/bin/python test/ltx2_5/record_block0.py --oracle-dir <dir>/t2av
"""
import argparse
from pathlib import Path

import torch
from diffusers import LTX2VideoTransformer3DModel


class StopAfterBlock(Exception):  # noqa: N818 (control flow, not an error)
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle-dir", required=True)
    p.add_argument("--snapshot", required=True, help="local Lightricks/LTX-2.5-Diffusers snapshot dir")
    args = p.parse_args()
    out = Path(args.oracle_dir) / "block0"
    out.mkdir(exist_ok=True)
    model = LTX2VideoTransformer3DModel.from_pretrained(
        args.snapshot, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    inp = torch.load(Path(args.oracle_dir) / "dit_in_000.pt", weights_only=False)
    block = model.transformer_blocks[0]
    saved = {}
    for name in ("attn1", "audio_attn1", "attn2", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn",
                 "ff", "audio_ff"):
        def hook(m, a, kw, o, _n=name):
            saved[_n] = {"args": [t.cpu() if torch.is_tensor(t) else t for t in a],
                         "kwargs": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in kw.items()
                                    if k in ("hidden_states", "encoder_hidden_states")},
                         "out": o.cpu()}
        getattr(block, name).register_forward_hook(hook, with_kwargs=True)

    def block_pre(m, a, kw):
        saved["block_in"] = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in kw.items()
                             if k in ("hidden_states", "audio_hidden_states", "temb", "temb_audio", "temb_prompt",
                                      "temb_prompt_audio", "temb_ca_scale_shift", "temb_ca_audio_scale_shift",
                                      "temb_ca_gate", "temb_ca_audio_gate")}
    block.register_forward_pre_hook(block_pre, with_kwargs=True)

    def stop(m, a, o):
        saved["block_out"] = (o[0].cpu(), o[1].cpu())
        raise StopAfterBlock
    block.register_forward_hook(stop)
    kwargs = {k: (v.to("cuda:0") if torch.is_tensor(v) else v) for k, v in inp.items()}
    with torch.inference_mode():
        try:
            model(**kwargs)
        except StopAfterBlock:
            pass
    torch.save(saved, out / "block0.pt")
    print({k: list(v) if isinstance(v, dict) else type(v) for k, v in saved.items()})


if __name__ == "__main__":
    main()
