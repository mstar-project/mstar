// Decode-sized linears on AMD GPUs: out[rows, N] = x[rows, K] @ w[N, K]^T (+ bias), rows <= 16, as a weight
// stream. Each wave owns YTILE consecutive output columns (rows of w) at a time and reads their whole K in
// 16-byte loads, UNRL of them per column in flight, the next step's loads issued before this step's math. The
// grid is a fixed number of workgroups (of up to four waves) that loop over column tiles, so every CU streams
// until the weight is done. The activations (a few KB) come through L1. Products accumulate in fp32 per lane and are summed across
// the wave once per column tile; each output is rounded once, to nearest even. SILU_MUL takes a stacked
// [gate; up] weight and writes silu(gate) * up, both rounded to bf16 first as the unfused pair would be.
//
// ROWS (the rows padded to a power of two), YTILE and UNRL are template parameters and every combination below is
// instantiated in this one build, so a search picks a point at run time. Wave64 (AMD) only: the torch-side calls
// are spelled the CUDA way for torch's hipify, the kernel is HIP.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include <torch/all.h>
#include <torch/library.h>

namespace {

constexpr int kWave = 64;
constexpr int kVec = 8;  // bf16 per 16-byte load
// four waves per workgroup at most, so a wave may hold 512 VGPRs; occupancy comes from workgroups per CU
constexpr int kMaxThreads = 256;

__device__ inline float lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ inline float hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

__device__ inline float wave_sum(float v) {
#pragma unroll
  for (int offset = kWave / 2; offset >= 1; offset >>= 1) {
    v += __shfl_xor(v, offset, kWave);
  }
  return v;
}

__device__ inline float round_bf16(float v) { return __bfloat162float(__float2bfloat16(v)); }

__device__ inline uint4 load16(const __hip_bfloat16* p) { return *reinterpret_cast<const uint4*>(p); }

template <int ROWS, int YTILE, int UNRL, bool SILU_MUL>
__global__ void __launch_bounds__(kMaxThreads) decode_gemv_kernel(
    const __hip_bfloat16* __restrict__ x, const __hip_bfloat16* __restrict__ w, __hip_bfloat16* __restrict__ out,
    const __hip_bfloat16* __restrict__ bias, int rows, int n_out, int k, int64_t x_stride, int64_t out_stride,
    int64_t bias_stride, int up_rows) {
  constexpr int kWeights = SILU_MUL ? 2 : 1;
  constexpr int kStep = kWave * kVec * UNRL;
  const int lane = threadIdx.x;
  const int waves = gridDim.x * blockDim.y;
  const uint4 zero = {0u, 0u, 0u, 0u};

  for (int n0 = (blockIdx.x * blockDim.y + threadIdx.y) * YTILE; n0 < n_out; n0 += waves * YTILE) {
    float acc[kWeights][ROWS][YTILE];
#pragma unroll
    for (int g = 0; g < kWeights; ++g)
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int y = 0; y < YTILE; ++y) acc[g][r][y] = 0.f;

    // the weight rows this wave reads: [gate rows | up rows] when SILU_MUL
    const __hip_bfloat16* wrow[kWeights][YTILE];
#pragma unroll
    for (int g = 0; g < kWeights; ++g)
#pragma unroll
      for (int y = 0; y < YTILE; ++y) {
        const int n = min(n0 + y, n_out - 1);
        wrow[g][y] = w + (int64_t)(n + g * up_rows) * k;
      }

    uint4 wv[kWeights][YTILE][UNRL];
    auto load_w = [&](uint4 (&dst)[kWeights][YTILE][UNRL], int k0) {
#pragma unroll
      for (int u = 0; u < UNRL; ++u) {
        const int kk = k0 + (u * kWave + lane) * kVec;
#pragma unroll
        for (int g = 0; g < kWeights; ++g)
#pragma unroll
          for (int y = 0; y < YTILE; ++y) dst[g][y][u] = kk < k ? load16(wrow[g][y] + kk) : zero;
      }
    };
    load_w(wv, 0);
    for (int k0 = 0; k0 < k; k0 += kStep) {
      uint4 next[kWeights][YTILE][UNRL];
      if (k0 + kStep < k) load_w(next, k0 + kStep);
#pragma unroll
      for (int u = 0; u < UNRL; ++u) {
        const int kk = k0 + (u * kWave + lane) * kVec;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          const uint4 xv = (r < rows && kk < k) ? load16(x + r * x_stride + kk) : zero;
          const uint32_t* xs = reinterpret_cast<const uint32_t*>(&xv);
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            const float x0 = lo(xs[p]), x1 = hi(xs[p]);
#pragma unroll
            for (int g = 0; g < kWeights; ++g)
#pragma unroll
              for (int y = 0; y < YTILE; ++y) {
                const uint32_t ws = reinterpret_cast<const uint32_t*>(&wv[g][y][u])[p];
                acc[g][r][y] = fmaf(lo(ws), x0, fmaf(hi(ws), x1, acc[g][r][y]));
              }
          }
        }
      }
      if (k0 + kStep < k) {
#pragma unroll
        for (int g = 0; g < kWeights; ++g)
#pragma unroll
          for (int y = 0; y < YTILE; ++y)
#pragma unroll
            for (int u = 0; u < UNRL; ++u) wv[g][y][u] = next[g][y][u];
      }
    }

#pragma unroll
    for (int g = 0; g < kWeights; ++g)
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int y = 0; y < YTILE; ++y) acc[g][r][y] = wave_sum(acc[g][r][y]);
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        if (r >= rows) break;
#pragma unroll
        for (int y = 0; y < YTILE; ++y) {
          const int n = n0 + y;
          if (n >= n_out) break;
          float v = acc[0][r][y];
          if constexpr (SILU_MUL) {
            const float gate = round_bf16(v), up = round_bf16(acc[kWeights - 1][r][y]);
            v = gate / (1.f + __expf(-gate)) * up;
          } else if (bias != nullptr) {
            v += __bfloat162float(bias[r * bias_stride + n]);
          }
          out[r * out_stride + n] = __float2bfloat16(v);
        }
      }
    }
  }
}

template <int ROWS, int YTILE, int UNRL, bool SILU_MUL>
void launch(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, const c10::optional<at::Tensor>& bias,
            int n_out, int up_rows, int waves, int groups) {
  const int rows = x.size(0), k = x.size(1);
  int64_t bias_stride = 0;
  const __hip_bfloat16* bias_ptr = nullptr;
  if (bias.has_value()) {
    bias_ptr = reinterpret_cast<const __hip_bfloat16*>(bias->data_ptr());
    bias_stride = bias->dim() == 2 ? bias->stride(0) : 0;
  }
  TORCH_CHECK(waves >= 1 && waves * kWave <= kMaxThreads && groups >= 1, "1-4 waves per workgroup");
  const dim3 block(kWave, waves);
  const dim3 grid(groups);
  hipLaunchKernelGGL((decode_gemv_kernel<ROWS, YTILE, UNRL, SILU_MUL>), grid, block, 0,
                     at::cuda::getCurrentCUDAStream(), reinterpret_cast<const __hip_bfloat16*>(x.data_ptr()),
                     reinterpret_cast<const __hip_bfloat16*>(w.data_ptr()),
                     reinterpret_cast<__hip_bfloat16*>(out.data_ptr()), bias_ptr, rows, n_out, k, x.stride(0),
                     out.stride(0), bias_stride, up_rows);
}

template <int ROWS, bool SILU_MUL>
bool dispatch_tile(int ytile, int unrl, const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
                   const c10::optional<at::Tensor>& bias, int n_out, int up_rows, int waves, int groups) {
#define MSTAR_GEMV_CASE(Y, U)                                                       \
  if (ytile == Y && unrl == U) {                                                    \
    launch<ROWS, Y, U, SILU_MUL>(x, w, out, bias, n_out, up_rows, waves, groups);   \
    return true;                                                                    \
  }
  MSTAR_GEMV_CASE(1, 1) MSTAR_GEMV_CASE(1, 2) MSTAR_GEMV_CASE(1, 4)
  MSTAR_GEMV_CASE(2, 1) MSTAR_GEMV_CASE(2, 2) MSTAR_GEMV_CASE(2, 4)
  MSTAR_GEMV_CASE(4, 1) MSTAR_GEMV_CASE(4, 2)
#undef MSTAR_GEMV_CASE
  return false;
}

template <bool SILU_MUL>
bool dispatch_rows(int ytile, int unrl, const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
                   const c10::optional<at::Tensor>& bias, int n_out, int up_rows, int waves, int groups) {
  const int rows = x.size(0);
  if (rows <= 1) return dispatch_tile<1, SILU_MUL>(ytile, unrl, x, w, out, bias, n_out, up_rows, waves, groups);
  if (rows <= 2) return dispatch_tile<2, SILU_MUL>(ytile, unrl, x, w, out, bias, n_out, up_rows, waves, groups);
  if (rows <= 4) return dispatch_tile<4, SILU_MUL>(ytile, unrl, x, w, out, bias, n_out, up_rows, waves, groups);
  if (rows <= 8) return dispatch_tile<8, SILU_MUL>(ytile, unrl, x, w, out, bias, n_out, up_rows, waves, groups);
  if (rows <= 16) return dispatch_tile<16, SILU_MUL>(ytile, unrl, x, w, out, bias, n_out, up_rows, waves, groups);
  return false;
}

void check_inputs(const at::Tensor& x, const at::Tensor& w) {
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "x [rows, K] and w [N, K]");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "bf16 only");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % kVec == 0 && w.is_contiguous(), "rows of 16-byte vectors");
  TORCH_CHECK(x.size(1) % kVec == 0, "K a multiple of 8");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "16-byte aligned operands");
}

at::Tensor decode_gemv(const at::Tensor& x, const at::Tensor& w, const c10::optional<at::Tensor>& bias,
                       int64_t ytile, int64_t unrl, int64_t waves, int64_t groups) {
  check_inputs(x, w);
  TORCH_CHECK(!bias.has_value() || (bias->scalar_type() == at::kBFloat16 && bias->stride(-1) == 1), "a bf16 bias");
  auto out = at::empty({x.size(0), w.size(0)}, x.options());
  if (x.size(0) == 0) return out;
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(dispatch_rows<false>(ytile, unrl, x, w, out, bias, w.size(0), 0, waves, groups),
              "no decode_gemv variant for rows=", x.size(0), " ytile=", ytile, " unrl=", unrl);
  return out;
}

at::Tensor decode_gemv_silu_mul(const at::Tensor& x, const at::Tensor& w, int64_t ytile, int64_t unrl,
                                int64_t waves, int64_t groups) {
  check_inputs(x, w);
  const int inter = w.size(0) / 2;
  auto out = at::empty({x.size(0), inter}, x.options());
  if (x.size(0) == 0) return out;
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(dispatch_rows<true>(ytile, unrl, x, w, out, c10::nullopt, inter, inter, waves, groups),
              "no decode_gemv_silu_mul variant for rows=", x.size(0), " ytile=", ytile, " unrl=", unrl);
  return out;
}

}  // namespace

TORCH_LIBRARY(mstar_hip_gemv, m) {
  m.def("decode_gemv(Tensor x, Tensor w, Tensor? bias, int ytile, int unrl, int waves, int groups) -> Tensor");
  m.def("decode_gemv_silu_mul(Tensor x, Tensor w, int ytile, int unrl, int waves, int groups) -> Tensor");
}

TORCH_LIBRARY_IMPL(mstar_hip_gemv, CUDA, m) {
  m.impl("decode_gemv", &decode_gemv);
  m.impl("decode_gemv_silu_mul", &decode_gemv_silu_mul);
}
