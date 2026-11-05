// RMSNorm variants for the kernel harness's HIP family: fp32 math and one
// round-to-nearest-even store per output, one workgroup per row. THREADS (the
// workgroup size, i.e. waves per workgroup) and VEC (consecutive elements per
// thread per step) are template parameters, and every point of the knob space
// is instantiated in this one build, so the search picks a point at run time.
// Written in CUDA; on ROCm torch's cpp_extension hipifies it.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#include <torch/library.h>

namespace {

__device__ inline float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ inline float to_float(__half v) { return __half2float(v); }

template <typename T>
__device__ inline T from_float(float v);
template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <>
__device__ inline __half from_float<__half>(float v) { return __float2half_rn(v); }

template <typename T, int THREADS, int VEC>
__global__ void __launch_bounds__(THREADS) rms_norm_kernel(
    const T* __restrict__ x, const T* __restrict__ w, T* __restrict__ out, int hidden, int64_t x_stride,
    int64_t out_stride, float eps) {
  const T* xr = x + blockIdx.x * x_stride;
  T* outr = out + blockIdx.x * out_stride;
  float sum = 0.f;
  for (int i = threadIdx.x * VEC; i < hidden; i += THREADS * VEC) {
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
      if (i + v < hidden) {
        const float f = to_float(xr[i + v]);
        sum += f * f;
      }
    }
  }
  using BlockReduce = cub::BlockReduce<float, THREADS>;
  __shared__ typename BlockReduce::TempStorage scratch;
  __shared__ float rms_rcp;
  const float total = BlockReduce(scratch).Sum(sum);
  if (threadIdx.x == 0) {
    rms_rcp = rsqrtf(total / hidden + eps);
  }
  __syncthreads();
  for (int i = threadIdx.x * VEC; i < hidden; i += THREADS * VEC) {
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
      if (i + v < hidden) {
        outr[i + v] = from_float<T>(to_float(xr[i + v]) * rms_rcp * to_float(w[i + v]));
      }
    }
  }
}

template <typename T, int THREADS, int VEC>
void launch(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, double eps) {
  rms_norm_kernel<T, THREADS, VEC><<<x.size(0), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const T*>(w.data_ptr()),
      reinterpret_cast<T*>(out.data_ptr()), static_cast<int>(x.size(1)), x.stride(0), out.stride(0),
      static_cast<float>(eps));
}

template <typename T, int THREADS>
bool launch_vec(int64_t vec, const at::Tensor& x, const at::Tensor& w, at::Tensor& out, double eps) {
  switch (vec) {
    case 1: launch<T, THREADS, 1>(x, w, out, eps); return true;
    case 2: launch<T, THREADS, 2>(x, w, out, eps); return true;
    case 4: launch<T, THREADS, 4>(x, w, out, eps); return true;
    case 8: launch<T, THREADS, 8>(x, w, out, eps); return true;
    default: return false;
  }
}

template <typename T>
bool launch_threads(int64_t threads, int64_t vec, const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
                    double eps) {
  switch (threads) {
    case 64: return launch_vec<T, 64>(vec, x, w, out, eps);
    case 128: return launch_vec<T, 128>(vec, x, w, out, eps);
    case 256: return launch_vec<T, 256>(vec, x, w, out, eps);
    case 512: return launch_vec<T, 512>(vec, x, w, out, eps);
    case 1024: return launch_vec<T, 1024>(vec, x, w, out, eps);
    default: return false;
  }
}

at::Tensor rms_norm_variant(const at::Tensor& x, const at::Tensor& w, double eps, int64_t threads, int64_t vec) {
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 && w.is_contiguous(), "2-D input with contiguous rows");
  auto out = at::empty_like(x);
  if (x.size(0) == 0) {
    return out;
  }
  const c10::cuda::CUDAGuard guard(x.device());
  bool launched = false;
  if (x.scalar_type() == at::kBFloat16) {
    launched = launch_threads<__nv_bfloat16>(threads, vec, x, w, out, eps);
  } else if (x.scalar_type() == at::kHalf) {
    launched = launch_threads<__half>(threads, vec, x, w, out, eps);
  }
  TORCH_CHECK(launched, "no variant for threads=", threads, " vec=", vec, " dtype=", x.scalar_type());
  return out;
}

}  // namespace

TORCH_LIBRARY(mstar_harness_rms_norm, m) {
  m.def("rms_norm_variant(Tensor x, Tensor w, float eps, int threads, int vec) -> Tensor");
}

TORCH_LIBRARY_IMPL(mstar_harness_rms_norm, CUDA, m) { m.impl("rms_norm_variant", &rms_norm_variant); }
