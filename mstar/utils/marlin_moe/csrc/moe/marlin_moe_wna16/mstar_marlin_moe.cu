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
 * Vendored for M* from vllm-project/vllm @ 4ba787ec9, csrc/moe/marlin_moe_wna16/ops.cu. Changed here: the
 * thread-config, kernel-selection and launch code (namespace MARLIN_NAMESPACE_NAME) is verbatim; the torch
 * stable-ABI wrapper ``moe_wna16_marlin_gemm`` became ``mstar_marlin_moe_gemm`` below on the regular ATen API, for
 * 16-bit activations only (no activation scales, global scale, bias or float zero points), registered under the op
 * namespace MSTAR_MARLIN_MOE_LIB (one per built variant, see mstar/utils/marlin_moe/__init__.py), so it never
 * collides with vLLM's _moe_C. The kernel instantiations and kernel_selector.h are generated per variant at build
 * time (after csrc/quantization/marlin/generate_kernels.py, which vLLM also runs for this directory).
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_moe_wna16
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

namespace MARLIN_NAMESPACE_NAME {

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

int get_kernel_cache_size(thread_config_t const& th_config, bool m_block_size_8,
                          int thread_m_blocks, int prob_m, int prob_n,
                          int prob_k, int num_bits, int group_size, int has_zp,
                          int is_zp_float, bool is_a_8bit, int stages) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;

  // shm size for block_sorted_ids/rd_block_sorted_ids/block_topk_weights
  // both of them requires tb_m * 4 bytes (tb_m * int32 or tb_m * float32)
  int sh_block_meta_size = tb_m * 16;
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

  int total_size =
      tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_block_meta_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, bool m_block_size_8,
                     int thread_m_blocks, int prob_m, int prob_n, int prob_k,
                     int num_bits, int group_size, int has_zp, int is_zp_float,
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
      th_config, m_block_size_8, thread_m_blocks, prob_m, prob_n, prob_k,
      num_bits, group_size, has_zp, is_zp_float, is_a_8bit, stages);
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
    int prob_n, int prob_k, int num_experts, int top_k, int thread_m_blocks,
    bool m_block_size_8, int num_bits, int group_size, bool has_zp,
    bool is_zp_float, bool is_a_8bit, int stages, int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  int count = 0;
  constexpr int device_max_reg_size = 255 * 1024;
  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, m_block_size_8, thread_m_blocks, prob_m,
                         prob_n, prob_k, num_bits, group_size, has_zp,
                         is_zp_float, is_a_8bit, stages,
                         max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config, m_block_size_8, thread_m_blocks, prob_m, prob_n, prob_k,
        num_bits, group_size, has_zp, is_zp_float, is_a_8bit, stages);

    int group_blocks = group_size == -1 ? -1 : (group_size / 16);

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks,
        th_config.thread_n / 16, th_config.thread_k / 16, m_block_size_8,
        has_zp, group_blocks, th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);
    int reg_size = max(attr.numRegs, 1) * th_config.num_threads * 4;
    int allow_count = min(device_max_reg_size / reg_size,
                          max_shared_mem / (cache_size + 1536));
    if (thread_m_blocks == 1)
      allow_count = max(min(allow_count, 4), 1);
    else
      allow_count = max(min(allow_count, 2), 1);

    if (prob_n / th_config.thread_n * prob_m * top_k * 4 < sms * allow_count) {
      allow_count =
          max(prob_n / th_config.thread_n * prob_m * top_k * 4 / sms, 1);
    }

    if (allow_count > count) {
      count = allow_count;
      exec_cfg = {count, th_config};
    };
  }

  return exec_cfg;
}

void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp,
               void* sorted_token_ids, void* expert_ids,
               void* num_tokens_past_padded, void* topk_weights,
               int moe_block_size, int num_experts, int top_k,
               bool mul_topk_weights, int prob_m, int prob_n, int prob_k,
               void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias, bool has_zp,
               int group_size, int dev, cudaStream_t stream, int thread_k,
               int thread_n, int sms, int blocks_per_sm, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float) {
  int thread_m_blocks = div_ceil(moe_block_size, 16);
  bool m_block_size_8 = moe_block_size == 8;
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
  const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;
  const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;
  const int32_t* num_tokens_past_padded_ptr =
      (const int32_t*)num_tokens_past_padded;
  const float* topk_weights_ptr = (const float*)topk_weights;
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

  // Set thread config
  exec_config_t exec_cfg;
  thread_config_t thread_tfg;
  if (thread_k != -1 && thread_n != -1) {
    thread_tfg = thread_config_t{thread_k, thread_n, thread_k * thread_n / 64};
    if (blocks_per_sm == -1) blocks_per_sm = 1;
    exec_cfg = exec_config_t{blocks_per_sm, thread_tfg};
    STD_TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                    " is not divisible by thread_n = ", thread_n);
    STD_TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                    " is not divisible by thread_k = ", thread_k);
  } else {
    // Auto config
    exec_cfg = determine_exec_config(
        a_type, b_type, c_type, s_type, prob_m, prob_n, prob_k, num_experts,
        top_k, thread_m_blocks, m_block_size_8, num_bits, group_size, has_zp,
        is_zp_float, is_a_8bit, stages, max_shared_mem, sms);
    thread_tfg = exec_cfg.tb_cfg;
  }

  int num_threads = thread_tfg.num_threads;
  thread_k = thread_tfg.thread_k;
  thread_n = thread_tfg.thread_n;
  int blocks = sms * exec_cfg.blocks_per_sm;
  if (exec_cfg.blocks_per_sm > 1)
    max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  STD_TORCH_CHECK(
      is_valid_config(thread_tfg, m_block_size_8, thread_m_blocks, prob_m,
                      prob_n, prob_k, num_bits, group_size, has_zp, is_zp_float,
                      is_a_8bit, stages, max_shared_mem),
      "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
      ", thread_k = ", thread_tfg.thread_k,
      ", thread_n = ", thread_tfg.thread_n,
      ", num_threads = ", thread_tfg.num_threads, " for MKN = [", prob_m, ", ",
      prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
      ", group_size = ", group_size, ", has_zp = ", has_zp,
      ", is_zp_float = ", is_zp_float, ", max_shared_mem = ", max_shared_mem);

  int sh_cache_size = get_kernel_cache_size(
      thread_tfg, m_block_size_8, thread_m_blocks, prob_m, prob_n, prob_k,
      num_bits, group_size, has_zp, is_zp_float, is_a_8bit, stages);

  auto kernel =
      get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks,
                        thread_n_blocks, thread_k_blocks, m_block_size_8,
                        has_zp, group_blocks, num_threads, is_zp_float, stages);

  if (kernel == MarlinDefault) {
    STD_TORCH_CHECK(false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n,
                    ", ", prob_k, "]", ", group_size = ", group_size,
                    ", thread_m_blocks = ", thread_m_blocks,
                    ", thread_n_blocks = ", thread_n_blocks,
                    ", thread_k_blocks = ", thread_k_blocks,
                    ", num_bits = ", num_bits);
  }

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_shared_mem);
  // avoid ">>>" being formatted to "> > >"
  // clang-format off
  kernel<<<blocks, num_threads, max_shared_mem, stream>>>(
      A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
      sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
      topk_weights_ptr, top_k, mul_topk_weights, prob_m, prob_n, prob_k, locks,
      has_bias, use_atomic_add, use_fp32_reduce);
  // clang-format on
}

}  // namespace MARLIN_NAMESPACE_NAME

namespace {

// vllm's ScalarType of each weight kind the M* formats hand over (mstar/utils/marlin_moe/__init__.py: KIND_*)
vllm::ScalarType weight_type(int64_t kind) {
  switch (kind) {
    case 0: return vllm::kU4B8;   // int4, symmetric (stored code - 8)
    case 1: return vllm::kU4;     // int4 with zero points
    case 2: return vllm::kFE2M1f; // e2m1 with e8m0 group scales (MXFP4)
  }
  TORCH_CHECK(false, "mstar marlin moe: unknown weight kind ", kind);
}

}  // namespace

// One grouped GEMM of the routed experts: ``a [size_m, K]`` (rows ``sorted_token_ids // top_k``) times each slot's
// expert ``W_e`` (``b_q_weight [E, K/16, N*16/8]`` int32 from mstar/utils/marlin_moe, scales ``[E, K/group, N]`` in
// a's dtype or e8m0, zero points ``[E, K/group, N/8]`` int32 for kind 1) into ``c [size_m * top_k, N]`` (slot rows),
// times the slot's routing weight when ``mul_topk_weights``. ``sorted_token_ids`` / ``expert_ids`` /
// ``num_tokens_past_padded`` are moe_align_block_size's at ``moe_block_size`` (8, 16, 32, 48 or 64); rows of slots
// it does not list (another rank's experts) are left as ``c`` had them.
void mstar_marlin_moe_gemm(const torch::Tensor& a, torch::Tensor& c, const torch::Tensor& b_q_weight,
                           const torch::Tensor& b_scales, const c10::optional<torch::Tensor>& b_zeros,
                           torch::Tensor& workspace, const torch::Tensor& sorted_token_ids,
                           const torch::Tensor& expert_ids, const torch::Tensor& num_tokens_past_padded,
                           const torch::Tensor& topk_weights, int64_t moe_block_size, int64_t top_k,
                           bool mul_topk_weights, int64_t kind, int64_t size_m, int64_t size_n, int64_t size_k,
                           bool use_atomic_add, bool use_fp32_reduce) {
  TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dim() == 2 && a.size(0) == size_m && a.size(1) == size_k,
              "mstar marlin moe: a must be contiguous CUDA [size_m, size_k]");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 || a.scalar_type() == at::kHalf,
              "mstar marlin moe: a must be bf16 or fp16");
  TORCH_CHECK(c.is_contiguous() && c.size(0) == size_m * top_k && c.size(1) == size_n &&
                  c.scalar_type() == a.scalar_type(),
              "mstar marlin moe: c must be contiguous [size_m * top_k, size_n] in a's dtype");
  TORCH_CHECK(moe_block_size == 8 || (moe_block_size % 16 == 0 && moe_block_size >= 16 && moe_block_size <= 64),
              "mstar marlin moe: moe_block_size ", moe_block_size);
  TORCH_CHECK(b_q_weight.is_contiguous() && b_q_weight.scalar_type() == at::kInt && b_q_weight.dim() == 3,
              "mstar marlin moe: b_q_weight");
  TORCH_CHECK(size_k % MARLIN_NAMESPACE_NAME::tile_size == 0 &&
                  b_q_weight.size(1) == size_k / MARLIN_NAMESPACE_NAME::tile_size,
              "mstar marlin moe: b_q_weight rows ", b_q_weight.size(1), " for K ", size_k);
  TORCH_CHECK(b_scales.is_contiguous() && b_scales.dim() == 3 && b_scales.size(2) == size_n,
              "mstar marlin moe: b_scales must be [E, K / group, N]");
  TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0, "mstar marlin moe: N ", size_n);
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "mstar marlin moe: topk_weights must be fp32");
  const vllm::ScalarType b_type = weight_type(kind);
  const vllm::ScalarType a_type = a.scalar_type() == at::kHalf ? vllm::kFloat16 : vllm::kBFloat16;
  const vllm::ScalarType s_type = kind == 2 ? vllm::kFE8M0fnu : a_type;
  if (kind == 2) {
    TORCH_CHECK(b_scales.scalar_type() == at::kFloat8_e8m0fnu, "mstar marlin moe: MXFP4 scales must be e8m0");
  } else {
    TORCH_CHECK(b_scales.scalar_type() == a.scalar_type(), "mstar marlin moe: scales must have a's dtype");
  }
  const bool has_zp = kind == 1;
  const int pack_factor = 32 / b_type.size_bits();
  if (has_zp) {
    TORCH_CHECK(b_zeros.has_value() && b_zeros->is_contiguous() && b_zeros->dim() == 3 &&
                    b_zeros->size(1) == b_scales.size(1) && b_zeros->size(2) == size_n / pack_factor,
                "mstar marlin moe: zero points must be [E, K / group, N / 8] int32");
  }
  const int num_groups = b_scales.size(1);
  const int group_size = num_groups > 1 ? static_cast<int>(size_k / num_groups) : -1;
  const int num_experts = b_q_weight.size(0);

  const int dev = a.get_device();
  const c10::cuda::CUDAGuard guard(a.device());
  int sms = -1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int max_n_tiles = size_n / MARLIN_NAMESPACE_NAME::min_thread_n;
  const int min_workspace = std::min<int64_t>(max_n_tiles * (sorted_token_ids.size(0) / moe_block_size), sms * 4);
  TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type() == at::kInt && workspace.numel() >= min_workspace,
              "mstar marlin moe: workspace needs ", min_workspace, " int32");
  if (size_m == 0) return;
  // use_fp32_reduce (vLLM's default): partial sums reduced in fp32 scratch, sized as vLLM sizes it
  at::Tensor c_tmp;
  if (use_fp32_reduce && !use_atomic_add) {
    int64_t n = std::min<int64_t>(size_n * sorted_token_ids.size(0),
                                  (int64_t)sms * 4 * moe_block_size * MARLIN_NAMESPACE_NAME::max_thread_n);
    if (moe_block_size == 8) n *= 2;
    c_tmp = torch::empty({n}, a.options().dtype(at::kFloat));
  } else {
    c_tmp = torch::empty({0}, a.options().dtype(at::kFloat));
  }
  auto empty = torch::empty({0}, a.options());
  auto zeros = has_zp ? *b_zeros : torch::empty({0}, a.options().dtype(at::kInt));
  auto a_s = torch::empty({0}, a.options().dtype(at::kFloat));
  auto g_s = torch::empty({0}, a.options().dtype(at::kFloat));

  MARLIN_NAMESPACE_NAME::marlin_mm(
      a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(), c_tmp.data_ptr(), empty.data_ptr(), a_s.data_ptr(),
      b_scales.data_ptr(), g_s.data_ptr(), zeros.data_ptr(), sorted_token_ids.data_ptr(), expert_ids.data_ptr(),
      num_tokens_past_padded.data_ptr(), topk_weights.data_ptr(), moe_block_size, num_experts, top_k,
      mul_topk_weights, size_m, size_n, size_k, workspace.data_ptr(), a_type, b_type, a_type, s_type,
      /*has_bias=*/false, has_zp, group_size, dev, at::cuda::getCurrentCUDAStream(dev), /*thread_k=*/-1,
      /*thread_n=*/-1, sms, /*blocks_per_sm=*/-1, use_atomic_add, use_fp32_reduce, /*is_zp_float=*/false);
}

#define MSTAR_LIBRARY(ns, m) TORCH_LIBRARY(ns, m)
#define MSTAR_LIBRARY_IMPL(ns, k, m) TORCH_LIBRARY_IMPL(ns, k, m)

MSTAR_LIBRARY(MSTAR_MARLIN_MOE_LIB, m) {
  m.def("gemm(Tensor a, Tensor(a!) c, Tensor b_q_weight, Tensor b_scales, Tensor? b_zeros, Tensor(b!) workspace, "
        "Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_past_padded, Tensor topk_weights, "
        "int moe_block_size, int top_k, bool mul_topk_weights, int kind, int size_m, int size_n, int size_k, "
        "bool use_atomic_add, bool use_fp32_reduce) -> ()");
}

MSTAR_LIBRARY_IMPL(MSTAR_MARLIN_MOE_LIB, CUDA, m) { m.impl("gemm", &mstar_marlin_moe_gemm); }
