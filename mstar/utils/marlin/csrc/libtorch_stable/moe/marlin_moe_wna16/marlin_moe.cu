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
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include "kernel.h"

// mstar: vendored from vLLM. The __global__ kernels + device host helpers
// (marlin_mm etc., lines below) are kept VERBATIM; only the top includes, the
// public host wrapper, and the op registration are ported from vLLM's
// torch::stable ABI to the classic torch ABI mstar's vendored ops use (see
// utils/fused_moe/csrc/moe_align_block_size.cu). STD_TORCH_CHECK, used by the
// verbatim device code, is provided by core/scalar_type.hpp's
// <torch/headeronly/util/Exception.h> include (reachable via kernel.h).
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace MARLIN_NAMESPACE_NAME {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
template <int moe_block_size>
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr, int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_past_padded_ptr, int size_m,
    int size_k, int top_k) {
  int num_tokens_past_padded = num_tokens_past_padded_ptr[0];
  int num_moe_blocks = div_ceil(num_tokens_past_padded, moe_block_size);
  int32_t block_sorted_ids[moe_block_size];
  int block_num_valid_tokens = 0;
  int64_t old_expert_id = 0;
  int64_t expert_id = 0;
  int row_stride = size_k * sizeof(half) / 16;

  auto read_moe_block_data = [&](int block_id) {
    block_num_valid_tokens = moe_block_size;
    int4* tmp_block_sorted_ids = reinterpret_cast<int4*>(block_sorted_ids);
    for (int i = 0; i < moe_block_size / 4; i++) {
      tmp_block_sorted_ids[i] =
          ((int4*)sorted_token_ids_ptr)[block_id * moe_block_size / 4 + i];
    }
    for (int i = 0; i < moe_block_size; i++) {
      if (block_sorted_ids[i] >= size_m * top_k) {
        block_num_valid_tokens = i;
        break;
      };
    }
  };

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int in_offset = (row / top_k) * row_stride;
    int out_offset = row * row_stride;

    half const* a_row_half =
        reinterpret_cast<half const*>(a_int4_ptr + in_offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + out_offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      auto cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        auto cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int index = blockIdx.x; index < num_moe_blocks; index += gridDim.x) {
    old_expert_id = expert_id;
    int tmp_expert_id = expert_ids_ptr[index];
    if (tmp_expert_id == -1) continue;
    expert_id = tmp_expert_id;
    perm_int_ptr += (expert_id - old_expert_id) * size_k;
    read_moe_block_data(index);

    for (int i = 0; i < block_num_valid_tokens; i++)
      permute_row(block_sorted_ids[i]);
  }
}

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
                          bool has_act_order, bool is_k_full, int stages) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups =
        tb_groups * stages * 2;          // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);  // We load at least 32 scale groups
    return load_groups * tb_n * 2;
  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * stages;
  }
}

int get_kernel_cache_size(thread_config_t const& th_config, bool m_block_size_8,
                          int thread_m_blocks, int prob_m, int prob_n,
                          int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full, int has_zp,
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

  int sh_s_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                            group_size, has_act_order, is_k_full, stages);
  int sh_g_idx_size = has_act_order && !is_k_full ? stages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size +
                   sh_g_idx_size + sh_block_meta_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, bool m_block_size_8,
                     int thread_m_blocks, int prob_m, int prob_n, int prob_k,
                     int num_bits, int group_size, bool has_act_order,
                     bool is_k_full, int has_zp, int is_zp_float,
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
  int cache_size =
      get_kernel_cache_size(th_config, m_block_size_8, thread_m_blocks, prob_m,
                            prob_n, prob_k, num_bits, group_size, has_act_order,
                            is_k_full, has_zp, is_zp_float, is_a_8bit, stages);
  return cache_size <= max_shared_mem;
}

MarlinFuncPtr get_marlin_kernel(
    const vllm::ScalarType a_type, const vllm::ScalarType b_type,
    const vllm::ScalarType c_type, const vllm::ScalarType s_type,
    int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
    bool m_block_size_8, bool has_act_order, bool has_zp, int group_blocks,
    int threads, bool is_zp_float, int stages) {
  int num_bits = b_type.size_bits();
  auto kernel = MarlinDefault;

#include "kernel_selector.h"

  return kernel;
}

exec_config_t determine_exec_config(
    const vllm::ScalarType& a_type, const vllm::ScalarType& b_type,
    const vllm::ScalarType& c_type, const vllm::ScalarType& s_type, int prob_m,
    int prob_n, int prob_k, int num_experts, int top_k, int thread_m_blocks,
    bool m_block_size_8, int num_bits, int group_size, bool has_act_order,
    bool is_k_full, bool has_zp, bool is_zp_float, bool is_a_8bit, int stages,
    int max_shared_mem, int sms) {
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
                         prob_n, prob_k, num_bits, group_size, has_act_order,
                         is_k_full, has_zp, is_zp_float, is_a_8bit, stages,
                         max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config, m_block_size_8, thread_m_blocks, prob_m, prob_n, prob_k,
        num_bits, group_size, has_act_order, is_k_full, has_zp, is_zp_float,
        is_a_8bit, stages);

    int group_blocks = 0;
    if (!has_act_order) {
      group_blocks = group_size == -1 ? -1 : (group_size / 16);
    }

    auto kernel =
        get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks,
                          th_config.thread_n / 16, th_config.thread_k / 16,
                          m_block_size_8, has_act_order, has_zp, group_blocks,
                          th_config.num_threads, is_zp_float, stages);

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
               void* a_s, void* b_s, void* g_s, void* zp, void* g_idx,
               void* perm, void* a_tmp, void* sorted_token_ids,
               void* expert_ids, void* num_tokens_past_padded,
               void* topk_weights, int moe_block_size, int num_experts,
               int top_k, bool mul_topk_weights, int prob_m, int prob_n,
               int prob_k, void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias,
               bool has_act_order, bool is_k_full, bool has_zp, int num_groups,
               int group_size, int dev, cudaStream_t stream, int thread_k,
               int thread_n, int sms, int blocks_per_sm, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float) {
  int thread_m_blocks = div_ceil(moe_block_size, 16);
  bool m_block_size_8 = moe_block_size == 8;
  bool is_a_8bit = a_type.size_bits() == 8;

  STD_TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [",
                  prob_m, ", ", prob_n, ", ", prob_k, "]");

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      STD_TORCH_CHECK(group_size != -1);
      group_blocks = group_size / 16;
      STD_TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                      " is not divisible by group_blocks = ", group_blocks);
    } else {
      STD_TORCH_CHECK(group_size == 0);
      group_blocks = 0;
    }
  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      STD_TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                      " is not divisible by group_blocks = ", group_blocks);
    }
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
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;
  const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;
  const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;
  const int32_t* num_tokens_past_padded_ptr =
      (const int32_t*)num_tokens_past_padded;
  const float* topk_weights_ptr = (const float*)topk_weights;
  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    auto kernel = permute_cols_kernel<8>;
    if (moe_block_size == 8) {
    } else if (moe_block_size == 16)
      kernel = permute_cols_kernel<16>;
    else if (moe_block_size == 32)
      kernel = permute_cols_kernel<32>;
    else if (moe_block_size == 48)
      kernel = permute_cols_kernel<48>;
    else if (moe_block_size == 64)
      kernel = permute_cols_kernel<64>;
    else
      STD_TORCH_CHECK(false, "unsupported moe_block_size ", moe_block_size);

    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    kernel<<<sms, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, sorted_token_ids_ptr, expert_ids_ptr,
        num_tokens_past_padded_ptr, prob_m, prob_k, top_k);
    // clang-format on
    A_ptr = a_tmp_ptr;
    prob_m = prob_m * top_k;
    top_k = 1;

    // If we have a full K, then we can run the non-act-order version of Marlin
    // (since the weight rows are reordered by increasing group ids, and by
    // having a full K, we have full original groups)
    if (is_k_full) has_act_order = false;
  }

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
        top_k, thread_m_blocks, m_block_size_8, num_bits, group_size,
        has_act_order, is_k_full, has_zp, is_zp_float, is_a_8bit, stages,
        max_shared_mem, sms);
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
                      prob_n, prob_k, num_bits, group_size, has_act_order,
                      is_k_full, has_zp, is_zp_float, is_a_8bit, stages,
                      max_shared_mem),
      "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
      ", thread_k = ", thread_tfg.thread_k,
      ", thread_n = ", thread_tfg.thread_n,
      ", num_threads = ", thread_tfg.num_threads, " for MKN = [", prob_m, ", ",
      prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
      ", group_size = ", group_size, ", has_act_order = ", has_act_order,
      ", is_k_full = ", is_k_full, ", has_zp = ", has_zp,
      ", is_zp_float = ", is_zp_float, ", max_shared_mem = ", max_shared_mem);

  int sh_cache_size =
      get_kernel_cache_size(thread_tfg, m_block_size_8, thread_m_blocks, prob_m,
                            prob_n, prob_k, num_bits, group_size, has_act_order,
                            is_k_full, has_zp, is_zp_float, is_a_8bit, stages);

  auto kernel = get_marlin_kernel(
      a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
      thread_k_blocks, m_block_size_8, has_act_order, has_zp, group_blocks,
      num_threads, is_zp_float, stages);

  if (kernel == MarlinDefault) {
    STD_TORCH_CHECK(
        false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n, ", ",
        prob_k, "]", ", has_act_order = ", has_act_order,
        ", num_groups = ", num_groups, ", group_size = ", group_size,
        ", thread_m_blocks = ", thread_m_blocks,
        ", thread_n_blocks = ", thread_n_blocks,
        ", thread_k_blocks = ", thread_k_blocks, ", num_bits = ", num_bits);
  }

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_shared_mem);
  // avoid ">>>" being formatted to "> > >"
  // clang-format off
  kernel<<<blocks, num_threads, max_shared_mem, stream>>>(
      A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr, g_idx_ptr,
      sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
      topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m,
      prob_n, prob_k, locks, has_bias, use_atomic_add, use_fp32_reduce);
  // clang-format on
}

}  // namespace MARLIN_NAMESPACE_NAME

// ---------------------------------------------------------------------------
// mstar classic-ABI host wrapper, trimmed to W4A16: fp16/bf16 activation,
// symmetric GPTQ INT4 (uint4b8), no zero-point / act-order / bias / global-scale
// / 8-bit-activation. Ports vLLM ops.cu::moe_wna16_marlin_gemm; every dropped
// optional is handed to the verbatim device ``marlin_mm`` as a 0-element tensor,
// exactly as upstream's "absent optional" branches did. Namespace of the device
// symbols is ``marlin_moe_wna16`` (set by kernel.h's MARLIN_NAMESPACE_NAME), so
// its Marlin<> instantiations never ODR-clash with the linear repack's ``marlin``.
// ---------------------------------------------------------------------------
torch::Tensor moe_wna16_marlin_gemm(
    torch::Tensor a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor b_q_weight, torch::Tensor b_scales, torch::Tensor workspace,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_past_padded, torch::Tensor topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
    int64_t b_type_id, int64_t size_m, int64_t size_n, int64_t size_k,
    bool is_k_full, bool use_atomic_add, bool use_fp32_reduce) {
  vllm::ScalarTypeId a_type_id, c_type_id;
  TORCH_CHECK(
      a.scalar_type() == torch::kHalf || a.scalar_type() == torch::kBFloat16,
      "moe_wna16_marlin_gemm (mstar W4A16): activation must be fp16 or bf16");
  if (a.scalar_type() == torch::kHalf) {
    a_type_id = vllm::kFloat16.id();
    c_type_id = vllm::kFloat16.id();
  } else {
    a_type_id = vllm::kBFloat16.id();
    c_type_id = vllm::kBFloat16.id();
  }
  auto c_dtype = a.scalar_type();
  vllm::ScalarTypeId s_type_id = c_type_id;

  vllm::ScalarType a_type = vllm::ScalarType::from_id(a_type_id);
  vllm::ScalarType b_type = vllm::ScalarType::from_id(b_type_id);
  vllm::ScalarType c_type = vllm::ScalarType::from_id(c_type_id);
  vllm::ScalarType s_type = vllm::ScalarType::from_id(s_type_id);
  TORCH_CHECK(b_type == vllm::kU4B8,
              "mstar Marlin MoE supports only symmetric INT4 (uint4b8); got ",
              b_type.str());

  int pack_factor = 32 / b_type.size_bits();
  int num_experts = b_q_weight.size(0);

  if (moe_block_size != 8) {
    TORCH_CHECK(moe_block_size % 16 == 0,
                "unsupported moe_block_size=", moe_block_size);
    TORCH_CHECK(moe_block_size >= 16 && moe_block_size <= 64,
                "unsupported moe_block_size=", moe_block_size);
  }

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B (Marlin-tiled: b_q_weight is (E, size_k/tile, size_n*pack/tile))
  TORCH_CHECK(size_k % marlin_moe_wna16::tile_size == 0, "size_k = ", size_k,
              " not divisible by tile_size = ", marlin_moe_wna16::tile_size);
  TORCH_CHECK((size_k / marlin_moe_wna16::tile_size) == b_q_weight.size(1),
              "b_q_weight.size(1) = ", b_q_weight.size(1), ", size_k = ", size_k);
  TORCH_CHECK(b_q_weight.size(2) % marlin_moe_wna16::tile_size == 0,
              "b_q_weight.size(2) = ", b_q_weight.size(2));
  int actual_size_n =
      (b_q_weight.size(2) / marlin_moe_wna16::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  TORCH_CHECK(a.is_cuda() && a.is_contiguous(), "A must be contiguous CUDA");
  TORCH_CHECK(b_q_weight.is_cuda() && b_q_weight.is_contiguous(),
              "b_q_weight must be contiguous CUDA");
  TORCH_CHECK(b_scales.is_cuda() && b_scales.is_contiguous(),
              "b_scales must be contiguous CUDA");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  int dev = a.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int sms = -1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

  auto opts_c = a.options().dtype(c_dtype);
  auto opts_f = a.options().dtype(torch::kFloat);

  torch::Tensor c;
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    TORCH_CHECK(c.is_cuda() && c.is_contiguous(), "c must be contiguous CUDA");
    TORCH_CHECK(c.size(0) == size_m * top_k && c.size(1) == size_n,
                "bad c shape");
  } else {
    c = torch::empty({size_m * top_k, size_n}, opts_c);
  }

  torch::Tensor c_tmp;
  if (use_fp32_reduce && !use_atomic_add) {
    long max_c_tmp_size = std::min(
        (long)size_n * sorted_token_ids.size(0),
        (long)sms * 4 * moe_block_size * marlin_moe_wna16::max_thread_n);
    if (moe_block_size == 8) max_c_tmp_size *= 2;
    c_tmp = torch::empty({max_c_tmp_size}, opts_f);
  } else {
    c_tmp = torch::empty({0}, opts_f);
  }

  // Grouping (no act-order): num_groups = b_scales.size(1).
  TORCH_CHECK(b_scales.dim() == 3, "b_scales must be rank 3");
  TORCH_CHECK(b_scales.size(2) == size_n, "b_scales dim2 != size_n");
  int num_groups = b_scales.size(1);
  int group_size;
  if (num_groups > 1) {
    TORCH_CHECK(size_k % num_groups == 0, "size_k not divisible by num_groups");
    group_size = size_k / num_groups;
  } else {
    group_size = -1;
  }

  // Dropped optionals -> 0-element tensors (upstream's absent-optional branches).
  torch::Tensor a_scales = torch::empty({0}, opts_f);
  torch::Tensor global_scale = torch::empty({0}, opts_f);
  torch::Tensor b_bias = torch::empty({0}, opts_c);
  torch::Tensor b_zeros = torch::empty({0}, opts_c);
  torch::Tensor g_idx = torch::empty({0}, opts_c);
  torch::Tensor perm = torch::empty({0}, opts_c);
  torch::Tensor a_tmp = torch::empty({0}, opts_c);

  TORCH_CHECK(size_n % marlin_moe_wna16::min_thread_n == 0, "size_n = ", size_n,
              " not divisible by min_thread_n");
  int max_n_tiles = size_n / marlin_moe_wna16::min_thread_n;
  int min_workspace_size = std::min(
      max_n_tiles * (int)(sorted_token_ids.size(0) / moe_block_size), sms * 4);
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " < min_workspace_size = ", min_workspace_size);

  marlin_moe_wna16::marlin_mm(
      a.const_data_ptr(), b_q_weight.const_data_ptr(), c.mutable_data_ptr(),
      c_tmp.mutable_data_ptr(), b_bias.mutable_data_ptr(),
      a_scales.mutable_data_ptr(), b_scales.mutable_data_ptr(),
      global_scale.mutable_data_ptr(), b_zeros.mutable_data_ptr(),
      g_idx.mutable_data_ptr(), perm.mutable_data_ptr(), a_tmp.mutable_data_ptr(),
      sorted_token_ids.mutable_data_ptr(), expert_ids.mutable_data_ptr(),
      num_tokens_past_padded.mutable_data_ptr(), topk_weights.mutable_data_ptr(),
      moe_block_size, num_experts, top_k, mul_topk_weights, size_m, size_n,
      size_k, workspace.mutable_data_ptr(), a_type, b_type, c_type, s_type,
      /*has_bias=*/false, /*has_act_order=*/false, is_k_full, /*has_zp=*/false,
      num_groups, group_size, dev, stream, /*thread_k=*/-1, /*thread_n=*/-1, sms,
      /*blocks_per_sm=*/-1, use_atomic_add, use_fp32_reduce,
      /*is_zp_float=*/false);

  return c;
}

// Single TORCH_LIBRARY block for the whole ``_mstar_marlin_C`` namespace (the
// repack impl lives in gptq_marlin_repack.cu as a TORCH_LIBRARY_IMPL).
TORCH_LIBRARY(_mstar_marlin_C, m) {
  m.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, int size_k, "
      "int size_n, int num_bits) -> Tensor");
  m.def(
      "moe_wna16_marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor b_scales, Tensor workspace, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_past_padded, Tensor topk_weights, "
      "int moe_block_size, int top_k, bool mul_topk_weights, int b_type_id, "
      "int size_m, int size_n, int size_k, bool is_k_full, bool use_atomic_add, "
      "bool use_fp32_reduce) -> Tensor");
}

TORCH_LIBRARY_IMPL(_mstar_marlin_C, CUDA, m) {
  m.impl("moe_wna16_marlin_gemm", &moe_wna16_marlin_gemm);
}
