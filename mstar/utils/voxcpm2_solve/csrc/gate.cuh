// The TMA/WGMMA pipeline follows NVIDIA CUTLASS's wgmma_tma_sm90.cu.
// See CUTLASS_LICENSE.txt for its BSD-3-Clause notice.
#pragma once
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

namespace vox {
using namespace cute;
using Element = cutlass::bfloat16_t;
using SmemLayout = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},
                                         Shape<_64, _64, _3>{}));
using Matrix = decltype(make_tensor(static_cast<Element const*>(nullptr),
                                    make_shape(int{}, int{}), make_stride(int{}, _1{})));
using Tma = decltype(make_tma_atom(SM90_TMA_LOAD{}, Matrix{}, SmemLayout{}(_, _, 0),
                                  Shape<_64, _64>{}));
using TokenSmemLayout = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<_32, _64, _8>{}));
using TokenTma = decltype(make_tma_atom(SM90_TMA_LOAD{}, Matrix{}, TokenSmemLayout{}(_, _, 0),
    Shape<_32, _64>{}));
struct Linear {
    Tma b;
    TokenTma token_a;
    int m, n, k;
};
// Packed groups of eight gate and eight up rows. WGMMA's M+8 accumulator
// partner belongs to the same thread, enabling a register-only activation.
#define GATE_STAGES 16
using GateTokenLayout = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<_32, _64, _16>{}));
using GatedSmemLayout = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<_64, _64, _16>{}));
using GatedMma = decltype(make_tiled_mma(
    SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
struct SharedGated {
    alignas(128) ArrayEngine<Element, cosize_v<GateTokenLayout>> a;
    alignas(128) ArrayEngine<Element, cosize_v<GatedSmemLayout>> b;
    uint64_t produced[GATE_STAGES];
};
__device__ __forceinline__ void gated_tile(Linear const& p, Element* activated,
    int tile_m, int tile_n, SharedGated& smem) {
    using TransposedTile = Shape<_64, _32, _64>;
    auto a = p.token_a.get_tma_tensor(make_shape(p.m, p.k));
    auto b = p.b.get_tma_tensor(make_shape(p.n, p.k));
    auto coord = make_coord(tile_n, tile_m, _);
    auto ga = local_tile(a, TransposedTile{}, coord, Step<X, _1, _1>{});
    auto gb = local_tile(b, TransposedTile{}, coord, Step<_1, X, _1>{});
    auto sa = make_tensor(make_smem_ptr(smem.a.begin()), GateTokenLayout{});
    auto sb = make_tensor(make_smem_ptr(smem.b.begin()), GatedSmemLayout{});
    auto [ta, tsa] = tma_partition(p.token_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sa), group_modes<0, 2>(ga));
    auto [tb, tsb] = tma_partition(p.b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sb), group_modes<0, 2>(gb));
    using Producer = cutlass::arch::ClusterTransactionBarrier;
    bool elected = cutlass::canonical_warp_idx_sync() == 0 && elect_one_sync();
    for (int i = 0; i < GATE_STAGES; ++i)
        if (elected) Producer::init(&smem.produced[i], 1);
    cutlass::arch::fence_barrier_init();
    __syncthreads();
    for (int i = 0; i < GATE_STAGES; ++i) {
        if (elected) {
            Producer::arrive_and_expect_tx(&smem.produced[i], (32 + 64) * 64 * sizeof(Element));
            copy(p.token_a.with(smem.produced[i]), ta(_, i), tsa(_, i));
            copy(p.b.with(smem.produced[i]), tb(_, i), tsb(_, i));
        }
    }
    auto mma = GatedMma{};
    auto thread = mma.get_thread_slice(threadIdx.x);
    auto rb = thread.make_fragment_A(thread.partition_A(sb));
    auto ra = thread.make_fragment_B(thread.partition_B(sa));
    auto indices = thread.partition_C(make_identity_tensor(Shape<_64, _32>{}));
    auto acc = thread.make_fragment_C(indices);
    auto acc1 = thread.make_fragment_C(indices);
    clear(acc); clear(acc1);
    for (int k = 0; k < GATE_STAGES; ++k) {
        Producer::wait(&smem.produced[k], 0);
        warpgroup_fence_operand(acc); warpgroup_fence_operand(acc1);
        warpgroup_arrive();
        if (k < 8) gemm(mma, rb(_, _, _, k), ra(_, _, _, k), acc);
        else gemm(mma, rb(_, _, _, k), ra(_, _, _, k), acc1);
        warpgroup_commit_batch();
        if (k >= 7) warpgroup_wait<7>();
    }
    warpgroup_wait<0>();
    warpgroup_fence_operand(acc); warpgroup_fence_operand(acc1);
    #pragma unroll
    for (int i = 0; i < size(acc); ++i) acc(i) += acc1(i);
    #pragma unroll
    for (int i = 0; i < size(acc); ++i) {
        auto [col, row] = indices(i);
        if (col % 16 < 8 && row + tile_m * 32 < p.m) {
            float up = 0.f;
            // The static fragment coordinates resolve this to the M+8
            // register partner; no cross-thread publication or shared exchange.
            #pragma unroll
            for (int j = 0; j < size(acc); ++j) {
                auto [other_col, other_row] = indices(j);
                if (other_col == col + 8 && other_row == row) up = acc(j);
            }
            float gate = float(Element(acc(i)));
            up = float(Element(up));
            int channel = tile_n * 32 + (col / 16) * 8 + col % 8;
            activated[(row + tile_m * 32) * 4096 + channel] =
                Element(gate / (1.f + expf(-gate)) * up);
        }
    }
}
} // namespace vox
