// Vendored from FastVideo (Apache-2.0): hao-ai-lab/FastVideo @ ec9145a,
//   fastvideo-kernel/wan22_vae_decode/solution/prefix.h. Unchanged.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
// Exchange chronological full-frame prefix slices for chronological row bands.
// Source rank r owns [floor(T*r/world), floor(T*(r+1)/world)).
at::Tensor prefix_temporal_to_rows(const at::Tensor& local,int64_t total_frames,
    int rank,int world,const c10::intrusive_ptr<c10d::ProcessGroup>& pg);
