// M*: the CUTLASS implicit-GEMM configuration of conv.cu, timeconv.cu and attn.cu. An extension is built for one
// architecture, and __init__.py defines WAN22_VAE_SM90 when that is sm_90a.
//  - Blackwell (sm_100a / sm_103a), upstream's: tcgen05 MMA, 2-CTA clusters, 256x256x64 tiles over the pair (256x128
//    for the output head), CUTLASS's automatic epilogue.
//  - Hopper (sm_90a): wgmma. CUTLASS's SM90 implicit-GEMM convolution is one kernel, a TMA producer warpgroup and ONE
//    consumer warpgroup, so a CTA holds at most 128x128 or 64x256 fp32 accumulators (128x256 and 256x128 spill).
//    Measured on an H200 at the decode's shapes, 64x256 for custom_conv and 128x128 elsewhere were fastest, and
//    the epilogue without shared memory beat TMA stores at every tile; it gets time_upsample's destination (a frame
//    stride of 2) wrong, though, so that one keeps TMA stores. Neither the tile nor the epilogue changes a bit of the
//    video.
#pragma once
#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

// A CTA tile M x N x 64 (on Blackwell, the 2-CTA cluster's), in a cluster of `cl` CTAs along M.
struct ConvTile {
  int m, n, cl;
};

#ifdef WAN22_VAE_SM90
using ArchTag = cutlass::arch::Sm90;
using ConvEpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
using TimeEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
constexpr ConvTile kTileConv{64, 256, 1};    // conv.cu's custom_conv: the Winograd products, the direct convolutions
constexpr ConvTile kTilePhase{128, 128, 1};  // conv.cu's phase_conv
constexpr ConvTile kTileTime{128, 128, 1};   // timeconv.cu
constexpr ConvTile kTileHead{128, 128, 1};   // conv.cu's conv_n128 (the head's 112 channels)
constexpr ConvTile kTileAttn{128, 128, 1};   // attn.cu's two products
#else
using ArchTag = cutlass::arch::Sm100;
using ConvEpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
using TimeEpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
constexpr ConvTile kTileConv{256, 256, 2};
constexpr ConvTile kTilePhase{256, 256, 2};
constexpr ConvTile kTileTime{256, 256, 2};
constexpr ConvTile kTileHead{256, 128, 2};
constexpr ConvTile kTileAttn{256, 256, 2};
#endif
