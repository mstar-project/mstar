"""Kimi K3.1's routed experts (MXFP4 x bf16, SiTU 4/25) through mstar.utils.fused_moe.w4a16 at a rank's
full geometry, against the experts dequantized in fp32. Runs on CUDA or in Triton's CPU interpreter
(TRITON_INTERPRET=1, set by the caller before import).

    python k31_experts_case.py --geom tp8 --tokens 2 [--h 3584 --i 384] [--config 64,256,256]
"""
import argparse
import json
import os
import sys
import time

import torch

if os.environ.get("TRITON_INTERPRET") == "1":  # triton 3.6's interpreter with numpy 2: 1-element arrays as indices
    import triton.runtime.interpreter as _ti
    _orig = _ti._patch_lang_tensor
    def _patched(tensor, scope):
        _orig(tensor, scope)
        scope.set_attr(tensor, "__index__", lambda self: int(self.handle.data.reshape(-1)[0]))
    _ti._patch_lang_tensor = _patched

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant(packed, scale, group=32):
    codes = torch.stack([packed & 0xF, packed >> 4], -1).reshape(*packed.shape[:-1], -1).long()
    vals = E2M1.to(packed.device)[codes]
    return (vals.view(*vals.shape[:-1], -1, group) * torch.exp2(scale.float() - 127.0)[..., None]).reshape(vals.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geom", default="tp8", choices=["tp8", "ep8"])
    p.add_argument("--tokens", type=int, default=2)
    p.add_argument("--e", type=int, default=None)
    p.add_argument("--h", type=int, default=3584)
    p.add_argument("--i", type=int, default=None)
    p.add_argument("--config", default="", help="BLOCK_M,BLOCK_N,BLOCK_K override (interpreter speed)")
    p.add_argument("--activation", default="situ")
    p.add_argument("--dtype", default=None, help="bf16 (GPU default) / fp16 (the interpreter's: its bf16 dot is wrong)")
    args = p.parse_args()
    from mstar.utils.fused_moe import w4a16 as moe
    from mstar.utils.situ import SiTU
    dev = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1" else "cpu")
    # a TP8 rank: all 896 experts, a 384-wide slice each; an EP8 rank: 112 whole experts, the rest of the slots
    # routed elsewhere (the sentinel id E, skip_invalid)
    e = args.e or (896 if args.geom == "tp8" else 112)
    inter = args.i or (384 if args.geom == "tp8" else 3072)
    h, top_k, m = args.h, 16, args.tokens
    g = torch.Generator().manual_seed(0)
    def rnd(*s, lo=0, hi=256):
        return torch.randint(lo, hi, s, dtype=torch.uint8, generator=g)
    w1, w2 = rnd(e, 2 * inter, h // 2).to(dev), rnd(e, h, inter // 2).to(dev)
    s1, s2 = rnd(e, 2 * inter, h // 32, lo=120, hi=125).to(dev), rnd(e, h, inter // 32, lo=120, hi=125).to(dev)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype or ("bf16" if dev.type == "cuda" else "fp16")]
    x = torch.randn(m, h, generator=g).to(dtype).to(dev)
    global_e = 896
    ids = torch.stack([torch.randperm(global_e, generator=g)[:top_k] for _ in range(m)])
    if args.geom == "ep8":  # this rank holds global experts [0, 112): the others become the sentinel
        ids = torch.where(ids < e, ids, torch.full_like(ids, e))
    ids = ids.to(torch.int32).to(dev)
    wts = torch.rand(m, top_k, generator=g).softmax(-1).to(dev)
    situ = SiTU(4.0, 25.0)
    act = situ.spec if args.activation == "situ" else args.activation
    if args.config:
        bm, bn, bk = map(int, args.config.split(","))
        cfg = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 1}
        moe.get_default_config = lambda **kw: dict(cfg)
    t0 = time.time()
    out = moe.fused_experts_w4a16_impl(x, w1, w2, s1, s2, None, None, wts, ids, 2, 32, act, args.geom == "ep8")
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_kernel = time.time() - t0
    # reference: the slots' experts dequantized in fp32
    ref = torch.zeros(m, h, dtype=torch.float32, device=dev)
    xf = x.float()
    for i in ids.unique().tolist():
        if i >= e:
            continue
        rows, slot = (ids == i).nonzero(as_tuple=True)
        gu = xf[rows] @ dequant(w1[i], s1[i]).T
        gate, up = gu.chunk(2, -1)
        hh = situ(gate, up) if args.activation == "situ" else torch.nn.functional.silu(gate) * up
        ref.index_add_(0, rows, (hh.float() @ dequant(w2[i], s2[i]).T) * wts[rows, slot, None].float())
    err = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    print(json.dumps({"geom": args.geom, "E": e, "H": h, "I": inter, "tokens": m, "rel_l2": round(err, 5),
                      "finite": bool(torch.isfinite(out).all()), "seconds": round(t_kernel, 1), "device": dev.type}),
          flush=True)
    sys.exit(0 if err < 2e-2 and torch.isfinite(out).all() else 1)


if __name__ == "__main__":
    main()
