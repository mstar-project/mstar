/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 *
 * Vendored for M* from vllm-project/vllm @ 4ba787ec9, csrc/quantization/marlin/marlin.cu. Changed here: the
 * thread-config, kernel-selection and launch code (namespace marlin) is verbatim; the torch stable-ABI wrapper
 * ``marlin_gemm`` became ``mstar_marlin_gemm`` below on the regular ATen API, for 16-bit activations only (no
 * activation scales, global scale, bias or float zero points), registered under the op namespace
 * MSTAR_MARLIN_LIB (one per built variant, see mstar/utils/marlin/__init__.py), so it never collides with
 * vLLM's _C. The kernel instantiations and kernel_selector.h are generated per variant at build time.
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include "kernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>


#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace marlin {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          int stages) {
  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  int tb_scales = tb_groups * tb_n * 2;
  return tb_scales * stages;
}

int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                          int prob_m, int prob_n, int prob_k, int num_bits,
                          int group_size, int has_zp, bool is_zp_float,
                          bool is_a_8bit, int stages) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size = get_scales_cache_size(th_config, prob_m, prob_n, prob_k,
                                        num_bits, group_size, stages);
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, int has_zp, bool is_zp_float,
                     bool is_a_8bit, int stages, int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  // Check that pipeline fits into cache
  int cache_size = get_kernel_cache_size(
      th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size,
      has_zp, is_zp_float, is_a_8bit, stages);
  return cache_size <= max_shared_mem;
}

MarlinFuncPtr get_marlin_kernel(const vllm::ScalarType a_type,
                                const vllm::ScalarType b_type,
                                const vllm::ScalarType c_type,
                                const vllm::ScalarType s_type,
                                int thread_m_blocks, int thread_n_blocks,
                                int thread_k_blocks, bool m_block_size_8,
                                bool has_zp, int group_blocks, int threads,
                                bool is_zp_float, int stages) {
  int num_bits = b_type.size_bits();
  auto kernel = MarlinDefault;

  #include "kernel_selector.h"

  return kernel;
}

exec_config_t determine_exec_config(
    const vllm::ScalarType& a_type, const vllm::ScalarType& b_type,
    const vllm::ScalarType& c_type, const vllm::ScalarType& s_type, int prob_m,
    int prob_n, int prob_k, int thread_m_blocks, bool m_block_size_8,
    int num_bits, int group_size, bool has_zp, bool is_zp_float, int is_a_8bit,
    int stages, int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
                         num_bits, group_size, has_zp, is_zp_float, is_a_8bit,
                         stages, max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits,
        group_size, has_zp, is_zp_float, is_a_8bit, stages);

    int group_blocks = group_size == -1 ? -1 : group_size / 16;

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks,
        th_config.thread_n / 16, th_config.thread_k / 16, m_block_size_8,
        has_zp, group_blocks, th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    return {1, th_config};
  }

  return exec_cfg;
}

void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, int prob_m,
               int prob_n, int prob_k, int lda, void* workspace,
               vllm::ScalarType const& a_type, vllm::ScalarType const& b_type,
               vllm::ScalarType const& c_type, vllm::ScalarType const& s_type,
               bool has_bias, bool has_zp, int num_groups, int group_size,
               int dev, cudaStream_t stream, int thread_k_init,
               int thread_n_init, int sms, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float) {
  bool is_a_8bit = a_type.size_bits() == 8;
  STD_TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [",
                  prob_m, ", ", prob_n, ", ", prob_k, "]");

  int group_blocks;
  if (group_size == -1) {
    group_blocks = -1;
  } else {
    group_blocks = group_size / 16;
    STD_TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                    " is not divisible by group_blocks = ", group_blocks);
  }

  int num_bits = b_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;

  const int4* bias_ptr = (const int4*)b_bias;
  const float* a_s_ptr = (const float*)a_s;
  const int4* b_s_ptr = (const int4*)b_s;
  const float* g_s_ptr = (const float*)g_s;

  const int4* zp_ptr = (const int4*)zp;
  int* locks = (int*)workspace;

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  STD_TORCH_CHECK(max_shared_mem > 0);

  int major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         dev);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         dev);
  STD_TORCH_CHECK(major_capability * 10 + minor_capability >= 75,
                  "marlin kernel only support Turing or newer GPUs.");
  int stages = 4;
  if (major_capability == 7 && minor_capability == 5) {
    stages = 2;
    STD_TORCH_CHECK(a_type == vllm::kFloat16 || a_type == vllm::kS8,
                    "Turing only support FP16 or INT8 activation.");
  }
  if (a_type == vllm::kFE4M3fn) {
    STD_TORCH_CHECK(major_capability * 10 + minor_capability >= 89,
                    "FP8 only support Ada Lovelace or newer GPUs.");
    STD_TORCH_CHECK(
        major_capability * 10 + minor_capability == 89 ||
            major_capability == 12,
        "Marlin W4A8-FP8 only support SM89 or SM12x device (It is slower than "
        "Marlin W4A16 on other devices).");
  }

  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int max_shared_mem_new = max_shared_mem;
  int rest_m = prob_m;
  int max_thread_m_blocks = 4;
  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split =
        par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_k = thread_k_init;
    int thread_n = thread_n_init;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    int m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16;

    // Set thread config
    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
      thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
      exec_cfg = exec_config_t{1, thread_tfg};
      STD_TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                      " is not divisible by thread_n = ", thread_n);
      STD_TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                      " is not divisible by thread_k = ", thread_k);
    } else {
      // Auto config
      exec_cfg = determine_exec_config(
          a_type, b_type, c_type, s_type, prob_m_split, prob_n, prob_k,
          thread_m_blocks, m_block_size_8, num_bits, group_size, has_zp,
          is_zp_float, is_a_8bit, stages, max_shared_mem, sms);
      thread_tfg = exec_cfg.tb_cfg;
      if (thread_tfg.thread_n != -1) {
        if (prob_n / thread_tfg.thread_n *
                div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <=
            sms) {
          if (is_valid_config({128, 64, 128}, thread_m_blocks, prob_m_split,
                              prob_n, prob_k, num_bits, group_size, has_zp,
                              is_zp_float, is_a_8bit, stages,
                              max_shared_mem_new)) {
            thread_tfg = {128, 64, 128};
            exec_cfg = {1, thread_tfg};
          }
        }
      }

      if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
        max_thread_m_blocks--;
        continue;
      }
    }

    int num_threads = thread_tfg.num_threads;
    thread_k = thread_tfg.thread_k;
    thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
      max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    STD_TORCH_CHECK(
        is_valid_config(thread_tfg, thread_m_blocks, prob_m_split, prob_n,
                        prob_k, num_bits, group_size, has_zp, is_zp_float,
                        is_a_8bit, stages, max_shared_mem_new),
        "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
        ", thread_k = ", thread_tfg.thread_k,
        ", thread_n = ", thread_tfg.thread_n,
        ", num_threads = ", thread_tfg.num_threads, " for MKN = [", prob_m,
        ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
        ", prob_m_split = ", prob_m_split, ", group_size = ", group_size,
        ", has_zp = ", has_zp, ", is_zp_float = ", is_zp_float,
        ", stages = ", stages, ", max_shared_mem_new = ", max_shared_mem_new);

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
        thread_k_blocks, m_block_size_8, has_zp, group_blocks, num_threads,
        is_zp_float, stages);

    if (kernel == MarlinDefault) {
      STD_TORCH_CHECK(
          false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n, ", ",
          prob_k, "]", ", num_groups = ", num_groups,
          ", group_size = ", group_size, ", prob_m_split = ", prob_m_split,
          ", thread_m_blocks = ", thread_m_blocks,
          ", thread_n_blocks = ", thread_n_blocks,
          ", thread_k_blocks = ", thread_k_blocks,
          ", num_threads = ", num_threads, ", num_bits = ", num_bits);
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_new);

    bool part_use_atomic_add =
        use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
        prob_m_split, prob_n, prob_k, lda, locks, has_bias, part_use_atomic_add,
        use_fp32_reduce, max_shared_mem_new);
    // clang-format on

    bool is_a_8bit = a_type.size_bits() == 8;
    A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8));
    a_s_ptr += prob_m_split;
    C_ptr += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
}

}  // namespace marlin

namespace {

// vllm's ScalarType of each weight kind the M* formats hand over (mstar/utils/marlin/__init__.py: KIND_*)
vllm::ScalarType weight_type(int64_t kind) {
  switch (kind) {
    case 0: return vllm::kU4B8;   // int4, symmetric (stored code - 8)
    case 1: return vllm::kU4;     // int4 with zero points
    case 2: return vllm::kFE2M1f; // e2m1 with e8m0 group scales (MXFP4)
  }
  TORCH_CHECK(false, "mstar marlin: unknown weight kind ", kind);
}

}  // namespace

// ``a [M, K] @ W``: W repacked by mstar/utils/marlin (``b_q_weight [K/16, N*16/8]`` int32), scales
// ``[K/group, N]`` in a's dtype (e8m0 for MXFP4), zero points ``[K/group, N/8]`` int32 for kind 1;
// ``workspace`` int32 zeros, at least one per SM (Marlin leaves them zero after each call).
torch::Tensor mstar_marlin_gemm(const torch::Tensor& a, const torch::Tensor& b_q_weight,
                                const torch::Tensor& b_scales, const c10::optional<torch::Tensor>& b_zeros,
                                torch::Tensor& workspace, int64_t kind, int64_t size_n, int64_t size_k) {
  TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.size(1) == size_k, "mstar marlin: a must be CUDA [M, K]");
  TORCH_CHECK(a.stride(1) == 1 && a.stride(0) % 8 == 0 && reinterpret_cast<uint64_t>(a.data_ptr()) % 16 == 0,
              "mstar marlin: a must be row-major, 16-byte aligned, its row stride a multiple of 8");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 || a.scalar_type() == at::kHalf,
              "mstar marlin: a must be bf16 or fp16");
  TORCH_CHECK(b_q_weight.is_contiguous() && b_q_weight.scalar_type() == at::kInt, "mstar marlin: b_q_weight");
  TORCH_CHECK(size_k % MARLIN_NAMESPACE_NAME::tile_size == 0 &&
                  b_q_weight.size(0) == size_k / MARLIN_NAMESPACE_NAME::tile_size,
              "mstar marlin: b_q_weight rows ", b_q_weight.size(0), " for K ", size_k);
  TORCH_CHECK(b_scales.is_contiguous() && b_scales.dim() == 2 && b_scales.size(1) == size_n,
              "mstar marlin: b_scales must be [K / group, N]");
  TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0, "mstar marlin: N ", size_n);
  const vllm::ScalarType b_type = weight_type(kind);
  const vllm::ScalarType a_type = a.scalar_type() == at::kHalf ? vllm::kFloat16 : vllm::kBFloat16;
  const vllm::ScalarType s_type = kind == 2 ? vllm::kFE8M0fnu : a_type;
  if (kind == 2) {
    TORCH_CHECK(b_scales.scalar_type() == at::kFloat8_e8m0fnu, "mstar marlin: MXFP4 scales must be e8m0");
  } else {
    TORCH_CHECK(b_scales.scalar_type() == a.scalar_type(), "mstar marlin: scales must have a's dtype");
  }
  const bool has_zp = kind == 1;
  const int pack_factor = 32 / b_type.size_bits();
  if (has_zp) {
    TORCH_CHECK(b_zeros.has_value() && b_zeros->is_contiguous() && b_zeros->size(0) == b_scales.size(0) &&
                    b_zeros->size(1) == size_n / pack_factor,
                "mstar marlin: zero points must be [K / group, N / 8] int32");
  }
  const int num_groups = b_scales.size(0);
  const int group_size = num_groups > 1 ? static_cast<int>(size_k / num_groups) : -1;

  const int dev = a.get_device();
  const c10::cuda::CUDAGuard guard(a.device());
  int sms = -1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type() == at::kInt && workspace.numel() >= sms,
              "mstar marlin: workspace needs one int32 per SM");

  const int64_t size_m = a.size(0);
  auto c = torch::empty({size_m, size_n}, a.options());
  if (size_m == 0) return c;
  // use_fp32_reduce (vLLM's default): partial sums reduced in fp32 scratch
  int max_m_block_size = std::min<int>((size_m + 15) / 16 * 16, 64);
  auto c_tmp = torch::empty({sms * max_m_block_size * MARLIN_NAMESPACE_NAME::max_thread_n},
                            a.options().dtype(at::kFloat));
  auto empty = torch::empty({0}, a.options());
  auto zeros = has_zp ? *b_zeros : torch::empty({0}, a.options().dtype(at::kInt));
  auto a_s = torch::empty({0}, a.options().dtype(at::kFloat));
  auto g_s = torch::empty({0}, a.options().dtype(at::kFloat));

  marlin::marlin_mm(a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(), c_tmp.data_ptr(), empty.data_ptr(),
                    a_s.data_ptr(), b_scales.data_ptr(), g_s.data_ptr(), zeros.data_ptr(), size_m, size_n, size_k,
                    a.stride(0), workspace.data_ptr(), a_type, b_type, a_type, s_type, /*has_bias=*/false, has_zp,
                    num_groups, group_size, dev, at::cuda::getCurrentCUDAStream(dev), /*thread_k_init=*/-1,
                    /*thread_n_init=*/-1, sms, /*use_atomic_add=*/false, /*use_fp32_reduce=*/true,
                    /*is_zp_float=*/false);
  return c;
}

#define MSTAR_LIBRARY(ns, m) TORCH_LIBRARY(ns, m)
#define MSTAR_LIBRARY_IMPL(ns, k, m) TORCH_LIBRARY_IMPL(ns, k, m)

MSTAR_LIBRARY(MSTAR_MARLIN_LIB, m) {
  m.def("gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, Tensor? b_zeros, Tensor(a!) workspace, int kind, "
        "int size_n, int size_k) -> Tensor");
}

MSTAR_LIBRARY_IMPL(MSTAR_MARLIN_LIB, CUDA, m) { m.impl("gemm", &mstar_marlin_gemm); }
