"""
Test fused_qk_norm_rope against flashinfer's rmsnorm + rope.

Run: python test_fused_qk_norm_rope.py
"""
import torch
import flashinfer

from mminf.utils.attention import fused_qk_norm_rope



def torch_reference(x, w_norm, pos, eps, rope_theta, rope_style="split_half"):
    """
    Pure torch reference for QK norm + RoPE.
    x: (M, H, D) fp32
    w_norm: (D,) fp32
    pos: (M,) int
    """
    M, H, D = x.shape
    half = D // 2
    x = x.to(torch.float32)

    # RMS norm per (token, head) over D
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps) * w_norm.to(torch.float32)

    # RoPE
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, device=x.device, dtype=torch.float32) * 2.0 / D))
    angle = pos.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)  # (M, half)
    cos = angle.cos().unsqueeze(1)  # (M, 1, half)
    sin = angle.sin().unsqueeze(1)

    if rope_style == "split_half":
        x1 = x[..., :half]
        x2 = x[..., half:]
        out_first = x1 * cos - x2 * sin
        out_second = x2 * cos + x1 * sin
        out = torch.cat([out_first, out_second], dim=-1)
    else:  # interleaved
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        out = torch.stack([out1, out2], dim=-1).flatten(-2)
    return out


def flashinfer_reference(x, w_norm, pos, eps, rope_theta):
    """
    flashinfer path: rmsnorm (per-head, requires reshape) + apply_rope_pos_ids.
    Note: flashinfer rope kernels typically want bf16; we cast at the boundary.
    """
    M, H, D = x.shape
    # Per-head RMS norm: flatten to (M*H, D)
    x = x.to(torch.bfloat16)
    x_norm = flashinfer.norm.rmsnorm(x, w_norm.to(torch.bfloat16), eps)
    x_norm = x_norm.view(M, H, D)

    # flashinfer rope expects bf16. Cast in/out.
    # apply_rope_pos_ids signature: (q, k, pos_ids, rotary_dim, ...) — but here we only
    # have one tensor. Use the in-place variant on a single tensor by passing it twice
    # and discarding. Simpler: use the lower-level apply_rope_with_cos_sin_cache or
    # build cos/sin and rotate manually. Cleanest is apply_rope_pos_ids on (x, x_dummy).
    dummy = torch.zeros_like(x_norm)
    # Treat x as "q" and dummy as "k"; flashinfer applies rope to both.
    q_out, _ = flashinfer.rope.apply_rope_pos_ids(
        x_norm,
        dummy,
        pos.to(torch.int32),
        rotary_dim=D,
        rope_theta=rope_theta,
        interleave=False,  # split-half (Llama-style); set True for GPT-NeoX
    )
    return q_out.view(M, H, D).to(torch.float32)


def test_case(name, M, H, D, eps=1e-6, rope_theta=10000.0, max_pos=2048, seed=0):
    print(f"\n=== {name}: M={M}, H={H}, D={D} ===")
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.float32

    x = torch.randn(M, H, D, device=device, dtype=dtype)
    w_norm = torch.randn(D, device=device, dtype=dtype) * 0.1 + 1.0
    pos = torch.randint(0, max_pos, (M,), device=device, dtype=torch.int32)

    # Torch reference (split-half)
    out_torch = torch_reference(x, w_norm, pos, eps, rope_theta, rope_style="split_half")

    # Fused kernel (modifies in place)
    x_fused = x.clone()
    fused_qk_norm_rope(x_fused, w_norm, pos, eps, rope_theta)

    # flashinfer path (separate norm + rope, with bf16 round-trip on rope)
    try:
        out_fi = flashinfer_reference(x, w_norm, pos, eps, rope_theta)
        have_fi = True
    except Exception as e:
        print(f"  flashinfer path skipped: {e}")
        have_fi = False

    diff_fused_torch = (x_fused - out_torch).abs()
    print(f"  fused vs torch:       max={diff_fused_torch.max().item():.2e}  "
          f"mean={diff_fused_torch.mean().item():.2e}")

    if have_fi:
        diff_fi_torch = (out_fi - out_torch).abs()
        diff_fused_fi = (x_fused - out_fi).abs()
        print(f"  flashinfer vs torch:  max={diff_fi_torch.max().item():.2e}  "
              f"mean={diff_fi_torch.mean().item():.2e}  (bf16 rope expected ~1e-3)")
        print(f"  fused vs flashinfer:  max={diff_fused_fi.max().item():.2e}  "
              f"mean={diff_fused_fi.mean().item():.2e}")

    tol = 1e-3
    assert diff_fused_torch.max().item() < tol, \
        f"fused kernel too far from torch reference (max {diff_fused_torch.max().item():.2e})"
    print(f"  ✓ fused matches torch reference within {tol}")


def benchmark(M, H, D, eps=1e-6, rope_theta=10000.0, n_iter=200, n_warmup=20):
    print(f"\n--- benchmark: M={M}, H={H}, D={D} ---")
    device = "cuda"
    x = torch.randn(M, H, D, device=device, dtype=torch.float32)
    w_norm = torch.randn(D, device=device, dtype=torch.float32) * 0.1 + 1.0
    pos = torch.randint(0, 2048, (M,), device=device, dtype=torch.int32)

    # Warmup
    for _ in range(n_warmup):
        x_tmp = x.clone()
        fused_qk_norm_rope(x_tmp, w_norm, pos, eps, rope_theta)
        try:
            _ = flashinfer_reference(x, w_norm, pos, eps, rope_theta)
        except Exception:
            pass
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # fused
    start.record()
    for _ in range(n_iter):
        x_tmp = x.clone()  # since fused is in-place
        fused_qk_norm_rope(x_tmp, w_norm, pos, eps, rope_theta)
    end.record()
    torch.cuda.synchronize()
    t_fused = start.elapsed_time(end) / n_iter

    # flashinfer
    try:
        start.record()
        for _ in range(n_iter):
            _ = flashinfer_reference(x, w_norm, pos, eps, rope_theta)
        end.record()
        torch.cuda.synchronize()
        t_fi = start.elapsed_time(end) / n_iter
        print(f"  flashinfer norm+rope: {t_fi*1000:.1f} us")
    except Exception as e:
        print(f"  flashinfer benchmark skipped: {e}")
        t_fi = None

    print(f"  fused kernel:         {t_fused*1000:.1f} us  (incl. clone)")
    if t_fi:
        print(f"  speedup:              {t_fi/t_fused:.2f}x")


if __name__ == "__main__":
    # Correctness — typical decode/prefill shapes
    test_case("decode bs=1, q",   M=1,   H=16, D=128)
    test_case("decode bs=1, kv",  M=1,   H=4,  D=128)
    test_case("decode bs=32, q",  M=32,  H=16, D=128)
    test_case("decode bs=32, kv", M=32,  H=4,  D=128)
    test_case("prefill q",        M=128, H=16, D=128)
    test_case("head_dim=64",      M=32,  H=16, D=64)
    test_case("non-pow-2 M",      M=7,   H=16, D=128)

    # Benchmark
    benchmark(M=32,  H=16, D=128)
    benchmark(M=32,  H=4,  D=128)
    benchmark(M=128, H=16, D=128)