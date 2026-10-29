// Synchronisation and bandwidth probe for persistent kernels on CDNA3/CDNA4 (gfx942, gfx950).
//
// What a resident (persistent) step kernel pays to hand work from one workgroup to another, and what
// one XCD can stream, measured on the device:
//   placement   each workgroup's XCD (HW_REG_XCC_ID) and CU (HW_REG_HW_ID), to check the round-robin
//               dispatch the XCD-local designs assume
//   pingpong    one producer -> consumer handoff (flag plus an optional payload checked by the consumer),
//               between two workgroups on one XCD or on two
//   barrier     groups of workgroups looping over a barrier: the whole grid, or each XCD on its own
//   stream      wide reads of a buffer by the workgroups of some XCDs: each reading a slice (aggregate
//               bandwidth) or each XCD reading all of it (what a row-split loop does with its weights)
//
// Handoff scopes. AGENT uses the HIP memory model at agent scope: on gfx942 a release is buffer_wbl2 sc1 +
// s_waitcnt + a store sc1 and an acquire is a load sc1 + buffer_inv sc1, correct between any two
// workgroups. XCD keeps everything in the shared L2 of one XCD: the producer waits for its stores to reach
// L2 (s_waitcnt vmcnt(0)) and stores the flag; the consumer polls with an L2 atomic (atomics bypass the
// per-CU L1) and invalidates its L1 (buffer_inv sc0) before reading the payload. XCD is only meant to be
// correct inside one XCD; run across two it is the negative control of the payload check.
//
// Every spin is bounded by the wall clock (ctl timeout flag), so a wrong placement assumption ends the
// kernel with a recorded timeout instead of hanging the device.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int kAgent = 0;
constexpr int kXcd = 1;      // poll with an L2 atomic
constexpr int kXcdLoad = 2;  // poll with a load after invalidating the L1

// wall-clock ticks (s_memrealtime, constant rate) a spin may last before it reports a timeout
constexpr uint64_t kSpinTicks = 20000000ull;  // 0.2 s at 100 MHz

__device__ __forceinline__ uint64_t now() { return wall_clock64(); }

__device__ __forceinline__ unsigned xcc_id() {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
  return __builtin_amdgcn_s_getreg((3 << 11) | (0 << 6) | 20) & 0xfu;  // HW_REG_XCC_ID[3:0]
#else
  return 0u;
#endif
}

__device__ __forceinline__ unsigned hw_id() {
  return __builtin_amdgcn_s_getreg((31 << 11) | (0 << 6) | 4);  // HW_REG_HW_ID
}

__device__ __forceinline__ void wait_stores() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }

__device__ __forceinline__ void invalidate_l1() {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
  asm volatile("buffer_inv sc0" ::: "memory");
#else
  asm volatile("buffer_wbinvl1" ::: "memory");
#endif
}

// Store a flag after this wave's earlier stores (all waves must have called wait_stores() and met at a
// workgroup barrier before, for a workgroup-wide release).
__device__ __forceinline__ void release_store(unsigned* p, unsigned v, int scope) {
  if (scope == kAgent) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  } else {
    wait_stores();
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  }
}

// One poll of a flag, without ordering.
__device__ __forceinline__ unsigned poll(unsigned* p, int scope) {
  if (scope == kAgent) return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (scope == kXcd) return __hip_atomic_fetch_add(p, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  invalidate_l1();
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}

// After a successful poll: later loads see what the producer wrote before its release.
__device__ __forceinline__ void acquire_fence(int scope) {
  if (scope == kAgent) {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  } else {
    invalidate_l1();
  }
}

__device__ __forceinline__ unsigned arrive_add(unsigned* p, unsigned v, int scope) {
  if (scope == kAgent) return __hip_atomic_fetch_add(p, v, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
  wait_stores();
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}

// Spin until *p >= target (unsigned, monotonic). Returns false on timeout (and sets ctl[0]).
__device__ bool wait_geq(unsigned* p, unsigned target, int scope, unsigned* ctl) {
  const uint64_t t0 = now();
  while (true) {
    if (poll(p, scope) >= target) {
      acquire_fence(scope);
      return true;
    }
    if (__hip_atomic_load(ctl, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0u) return false;
    if (now() - t0 > kSpinTicks) {
      __hip_atomic_store(ctl, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      return false;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------
// placement

__global__ void placement_kernel(int* out) {
  if (threadIdx.x == 0) {
    out[2 * blockIdx.x] = (int)xcc_id();
    out[2 * blockIdx.x + 1] = (int)hw_id();
  }
}

// ---------------------------------------------------------------------------------------------------------
// pingpong: workgroups a and b take turns. Round r: a writes its payload (value r), releases flag_a = r;
// b waits for flag_a >= r, checks a's payload, writes its own, releases flag_b = r; a waits for flag_b >= r
// and checks b's payload. out (uint64[8]): t_start, t_end (a's wall clock), payload errors, timeouts,
// a's and b's XCC id, a's and b's HW_ID (CU, SE).

__global__ void pingpong_kernel(int rounds, int wg_a, int wg_b, int scope, int words, unsigned* flags,
                                unsigned* payload, unsigned* ctl, unsigned long long* out) {
  const int me = blockIdx.x;
  if (me != wg_a && me != wg_b) return;
  const bool is_a = me == wg_a;
  unsigned* mine = flags + (is_a ? 0 : 64);
  unsigned* theirs = flags + (is_a ? 64 : 0);
  unsigned* my_payload = payload + (is_a ? 0 : words);
  unsigned* their_payload = payload + (is_a ? words : 0);
  __shared__ unsigned errors;
  __shared__ int ok;
  if (threadIdx.x == 0) {
    errors = 0;
    ok = 1;
    out[is_a ? 4 : 5] = xcc_id();
    out[is_a ? 6 : 7] = hw_id();
  }
  __syncthreads();
  uint64_t t_start = now();
  for (int r = 1; r <= rounds; ++r) {
    for (int phase = 0; phase < 2; ++phase) {
      const bool produce = (phase == 0) == is_a;
      if (produce) {
        for (int i = threadIdx.x; i < words; i += blockDim.x) my_payload[i] = (unsigned)r;
        wait_stores();
        __syncthreads();
        if (threadIdx.x == 0) release_store(mine, (unsigned)r, scope);
      } else {
        if (threadIdx.x == 0) ok = wait_geq(theirs, (unsigned)r, scope, ctl) ? 1 : 0;
        __syncthreads();
        if (!ok) goto done;
        unsigned bad = 0;
        for (int i = threadIdx.x; i < words; i += blockDim.x) bad += their_payload[i] != (unsigned)r;
        if (bad) atomicAdd(&errors, bad);
      }
      __syncthreads();
    }
  }
done:
  __syncthreads();
  if (threadIdx.x == 0) {
    if (is_a) {
      out[0] = t_start;
      out[1] = now();
    }
    atomicAdd(&out[2], (unsigned long long)errors);
    if (!ok) atomicAdd(&out[3], 1ull);
  }
}

// ---------------------------------------------------------------------------------------------------------
// barrier: `groups` independent groups. group(g) = blockIdx.x % groups when per_xcd (with the round-robin
// dispatch that is the workgroup's XCD; the kernel records mismatches), else every workgroup is in group 0.
// Members of a group: blockIdx.x / groups (per_xcd) or blockIdx.x. Algorithms:
//   0 counter:  arrive on one counter (monotonic), everyone polls it for epoch * members
//   1 gather:   each member stores its own flag (epoch); member 0 polls all flags with one wave's lanes,
//               then stores a release flag every member polls
//   2 all:      each member stores its own flag; every member polls all of them
// state (per group, 4096 words): [0] counter, [64] release flag, [128 ...] member flags.
// out (uint64[4 + 2 * groups]): timeouts, xcc mismatches, then per group t_start, t_end of member 0.

__global__ void barrier_kernel(int iters, int algo, int scope, int groups, unsigned* state, unsigned* ctl,
                               unsigned long long* out) {
  const int g = groups > 1 ? (int)(blockIdx.x % groups) : 0;
  const int rank = groups > 1 ? (int)(blockIdx.x / groups) : (int)blockIdx.x;
  const int members = groups > 1 ? (int)(gridDim.x / groups) : (int)gridDim.x;
  unsigned* st = state + (size_t)g * 4096;
  unsigned* counter = st;
  unsigned* go = st + 64;
  unsigned* flag = st + 128;
  __shared__ int ok;
  if (threadIdx.x == 0) {
    ok = 1;
    if (groups == 8 && (int)xcc_id() != g) atomicAdd(&out[1], 1ull);
  }
  __syncthreads();
  const uint64_t t_start = now();
  for (unsigned e = 1; e <= (unsigned)iters; ++e) {
    __syncthreads();
    if (algo == 0) {
      if (threadIdx.x == 0) {
        arrive_add(counter, 1u, scope);
        ok = wait_geq(counter, e * (unsigned)members, scope, ctl);
      }
    } else if (algo == 1) {
      if (threadIdx.x == 0) release_store(flag + rank, e, scope);
      if (rank == 0 && threadIdx.x < 64) {
        // one wave polls every member's flag, lane l owning members l, l + 64, ...
        const uint64_t t0 = now();
        bool done = false;
        while (!done) {
          bool mine = true;
          for (int m = threadIdx.x; m < members; m += 64) mine &= poll(flag + m, scope) >= e;
          done = __all(mine);
          if (!done && now() - t0 > kSpinTicks) {
            if (threadIdx.x == 0) __hip_atomic_store(ctl, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            break;
          }
          if (!done && __hip_atomic_load(ctl, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0u) break;
        }
        if (done) {
          acquire_fence(scope);
          if (threadIdx.x == 0) release_store(go, e, scope);
        } else if (threadIdx.x == 0) {
          ok = 0;
        }
      } else if (rank != 0 && threadIdx.x == 0) {
        ok = wait_geq(go, e, scope, ctl);
      }
    } else {
      if (threadIdx.x == 0) release_store(flag + rank, e, scope);
      if (threadIdx.x < 64) {
        const uint64_t t0 = now();
        bool done = false;
        while (!done) {
          bool mine = true;
          for (int m = threadIdx.x; m < members; m += 64) mine &= poll(flag + m, scope) >= e;
          done = __all(mine);
          if (!done && now() - t0 > kSpinTicks) {
            if (threadIdx.x == 0) __hip_atomic_store(ctl, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            break;
          }
          if (!done && __hip_atomic_load(ctl, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0u) break;
        }
        if (done) {
          acquire_fence(scope);
        } else if (threadIdx.x == 0) {
          ok = 0;
        }
      }
    }
    __syncthreads();
    if (!ok) break;
  }
  if (threadIdx.x == 0) {
    if (!ok) atomicAdd(&out[0], 1ull);
    if (rank == 0) {
      out[4 + 2 * g] = t_start;
      out[5 + 2 * g] = now();
    }
  }
}

// ---------------------------------------------------------------------------------------------------------
// stream: active workgroups (xcd < active_xcds, xcd = blockIdx.x % 8 as the round-robin dispatch places
// them; workgroups whose XCC id disagrees are counted in sink[gridDim.x]) read 16-byte vectors.
// broadcast = 0: the active workgroups split the buffer between them; 1: each active XCD reads all of it,
// split among its own workgroups. One pass reads the buffer once per reader set; `passes` passes.

__global__ void __launch_bounds__(256) stream_kernel(const uint4* __restrict__ buf, long long n_vec, int passes,
                                                    int broadcast, int active_xcds, int wgs_per_xcd,
                                                    unsigned* sink) {
  const int xcd = (int)(blockIdx.x % 8);
  if (threadIdx.x == 0 && (int)xcc_id() != xcd) atomicAdd(&sink[gridDim.x], 1u);
  if (xcd >= active_xcds) return;
  const int local = (int)(blockIdx.x / 8);  // index among this XCD's workgroups
  long long readers, reader;
  if (broadcast) {
    readers = wgs_per_xcd;
    reader = local;
  } else {
    readers = (long long)wgs_per_xcd * active_xcds;
    reader = (long long)local * active_xcds + xcd;
  }
  const long long chunk = (n_vec + readers - 1) / readers;
  const long long begin = reader * chunk;
  const long long end = begin + chunk < n_vec ? begin + chunk : n_vec;
  unsigned acc = 0;
  for (int p = 0; p < passes; ++p) {
    long long i = begin + threadIdx.x;
    for (; i + 3 * 256 < end; i += 4 * 256) {
      const uint4 a = buf[i], b = buf[i + 256], c = buf[i + 512], d = buf[i + 768];
      acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < end; i += 256) {
      const uint4 a = buf[i];
      acc ^= a.x ^ a.y ^ a.z ^ a.w;
    }
  }
  if (acc == 0x9e3779b9u) sink[blockIdx.x] = acc;  // keeps the loads; practically never stores
}

hipStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }

void check_launch(const char* what) {
  hipError_t e = hipGetLastError();
  TORCH_CHECK(e == hipSuccess, what, ": ", hipGetErrorString(e));
}

unsigned* u32(const at::Tensor& t) { return reinterpret_cast<unsigned*>(t.data_ptr()); }
unsigned long long* u64(const at::Tensor& t) { return reinterpret_cast<unsigned long long*>(t.data_ptr()); }

void placement(int64_t grid, const at::Tensor& out) {
  TORCH_CHECK(out.scalar_type() == at::kInt && out.numel() >= 2 * grid, "out int32[2 * grid]");
  hipLaunchKernelGGL(placement_kernel, dim3(grid), dim3(64), 0, stream(), out.data_ptr<int>());
  check_launch("placement");
}

void pingpong(int64_t rounds, int64_t wg_a, int64_t wg_b, int64_t scope, int64_t words, const at::Tensor& flags,
              const at::Tensor& payload, const at::Tensor& ctl, const at::Tensor& out) {
  TORCH_CHECK(flags.numel() >= 128 && payload.numel() >= 2 * words && out.numel() >= 8, "buffer sizes");
  const int grid = (int)(wg_a > wg_b ? wg_a : wg_b) + 1;
  hipLaunchKernelGGL(pingpong_kernel, dim3(grid), dim3(256), 0, stream(), (int)rounds, (int)wg_a, (int)wg_b,
                     (int)scope, (int)words, u32(flags), u32(payload), u32(ctl), u64(out));
  check_launch("pingpong");
}

void barrier(int64_t iters, int64_t grid, int64_t algo, int64_t scope, int64_t groups, const at::Tensor& state,
             const at::Tensor& ctl, const at::Tensor& out) {
  TORCH_CHECK(state.numel() >= groups * 4096 && out.numel() >= 4 + 2 * groups, "buffer sizes");
  TORCH_CHECK(grid % groups == 0 && grid / groups <= 4096 - 128, "members per group");
  hipLaunchKernelGGL(barrier_kernel, dim3(grid), dim3(256), 0, stream(), (int)iters, (int)algo, (int)scope,
                     (int)groups, u32(state), u32(ctl), u64(out));
  check_launch("barrier");
}

void stream_read(const at::Tensor& buf, int64_t passes, int64_t broadcast, int64_t active_xcds,
                 int64_t wgs_per_xcd, const at::Tensor& sink) {
  TORCH_CHECK(buf.is_contiguous() && buf.numel() * buf.element_size() % 16 == 0, "buffer of 16-byte vectors");
  const long long n_vec = buf.numel() * buf.element_size() / 16;
  const int grid = 8 * (int)wgs_per_xcd;
  TORCH_CHECK(sink.numel() >= grid + 1, "sink[grid + 1]");
  hipLaunchKernelGGL(stream_kernel, dim3(grid), dim3(256), 0, stream(),
                     reinterpret_cast<const uint4*>(buf.data_ptr()), n_vec, (int)passes, (int)broadcast,
                     (int)active_xcds, (int)wgs_per_xcd, u32(sink));
  check_launch("stream");
}

int64_t wall_clock_khz() {
  int dev = 0, khz = 0;
  TORCH_CHECK(hipGetDevice(&dev) == hipSuccess, "hipGetDevice");
  TORCH_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) == hipSuccess,
              "hipDeviceAttributeWallClockRate");
  return khz;
}

}  // namespace

TORCH_LIBRARY(mstar_sync_probe, m) {
  m.def("placement(int grid, Tensor out) -> ()");
  m.def("pingpong(int rounds, int wg_a, int wg_b, int scope, int words, Tensor flags, Tensor payload, "
        "Tensor ctl, Tensor out) -> ()");
  m.def("barrier(int iters, int grid, int algo, int scope, int groups, Tensor state, Tensor ctl, Tensor out) -> ()");
  m.def("stream_read(Tensor buf, int passes, int broadcast, int active_xcds, int wgs_per_xcd, Tensor sink) -> ()");
  m.def("wall_clock_khz() -> int", &wall_clock_khz);
}

TORCH_LIBRARY_IMPL(mstar_sync_probe, CUDA, m) {
  m.impl("placement", &placement);
  m.impl("pingpong", &pingpong);
  m.impl("barrier", &barrier);
  m.impl("stream_read", &stream_read);
}
