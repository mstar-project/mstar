// One code-predictor depth-loop step (a residual-codebook decoder's whole frame) as one resident kernel.
//
// The host lowers a StepProgram (mstar/engine/step_program.py) to a table of phases: per depth position and
// layer the QKV GEMM, attention (q/k RMSNorm, split-half rope, the K/V append and attention over the
// frame's positions), the O GEMM with the residual add, the gate/up GEMM with SwiGLU, the down GEMM with
// the residual add; per position after the first the LM head GEMM and the sampler (top-k, top-p,
// temperature; or the teacher-forced code), which also gathers the next position's input embedding and
// adds it to the frame's embedding sum. The kernel walks the table; every phase ends at a barrier.
//
// Rows are split into `groups` groups; each group's workgroups (blockIdx.x % groups) run every phase for
// their rows, so the groups never synchronise with each other. With 8 groups on an 8-XCD part each group
// is one XCD (the round-robin dispatch puts workgroup i on XCD i % 8, which the kernel checks) and its
// barriers and data stay in that XCD's L2; with fewer groups a group spans XCDs and syncs at agent scope.
//
// Inside a group a GEMM phase is a list of 16-column tiles; workgroup m of M takes tiles m, m + M, ...;
// its four waves split K and add their partial tiles through LDS. Weights stream through registers one
// 16 x 1024 chunk ahead of the MFMAs, and the chunk after a barrier is requested before the barrier is
// waited on. The A operand (a norm of the fp32 residual, or a bf16 activation) is read from global
// memory, rows past the group's last are zeros. Residual stream and sums of squares are fp32; the
// GEMM operands are bf16, the accumulators fp32.
//
// All workgroups of a group must be resident at once (one launch on an otherwise idle device); every
// wait is bounded by the wall clock and ends the kernel with ctl[0] set instead of hanging.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

typedef short bf16x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef unsigned short bf16_t;

constexpr int kThreads = 256;
constexpr int kHidden = 1024;
constexpr int kHeadDim = 128;
constexpr int kKvHeads = 8;
constexpr int kQHeads = 16;
constexpr int kPositions = 16;
constexpr int kVocab = 2048;
constexpr int kInter = 3072;
constexpr int kQkv = (kQHeads + 2 * kKvHeads) * kHeadDim;  // 4096
constexpr int kChunkK = 1024;                              // K per weight chunk
constexpr int kMaxRows = 16;
constexpr uint64_t kSpinTicks = 20000000ull;               // 0.2 s at 100 MHz

enum Kind : int { PRO = 0, QKV = 1, ATTN = 2, OPROJ = 3, GATEUP = 4, DOWN = 5, INPUT = 6, LMHEAD = 7, SAMPLE = 8 };
enum Scope : int { AGENT = 0, XCD = 1 };

// One phase of the lowered program (int64 fields, see step_program.py: PHASE_FIELDS).
struct Phase {
  int64_t kind, pos, layer, n_tasks, chunks, ldw, task_begin, ss_in, ss_out;
  int64_t w, norm_w, norm_w2, aux;
};
constexpr int kPhaseFields = 13;

struct Params {
  const int64_t* program;  // [n_phases][kPhaseFields]
  const float* rope;       // [kPositions][128]: cos(64), sin(64)
  const bf16_t* last_hidden;  // [bs][1024]
  const bf16_t* embed0;       // [bs][1024]
  const bf16_t* codebook;     // [15][2048][1024]
  const int64_t* forced;      // [bs][16] or null
  float* emb_sum;             // [bs][1024]
  int64_t* codes;             // [bs][16]
  float* logits;              // [15][bs][2048]
  float* h;                   // [bs][1024]
  float* qkv;                 // [bs][4096]
  bf16_t* attn;               // [bs][2048]
  bf16_t* act;                // [bs][3072]
  bf16_t* kv;                 // [layers][bs][2][16][8][128]
  float* ss;                  // [slots][bs]
  unsigned* sync;             // [groups][64]
  unsigned* ctl;              // [0] abort, [1] XCC mismatches, [2] phases completed by group 0's member 0
  uint64_t* trace;            // [n_phases][2] or null: group 0 member 0's wall clock at each barrier's arrival and exit
  int n_phases, bs, groups, scope, sync_mode, stop_after, top_k;
  float top_p, inv_temperature, eps, attn_scale;
  uint64_t seed;
};

__device__ __forceinline__ float bf2f(bf16_t b) { return __uint_as_float(((unsigned)b) << 16); }

__device__ __forceinline__ bf16_t f2bf(float f) {
  unsigned u = __float_as_uint(f);
  if ((u & 0x7f800000u) == 0x7f800000u) return (bf16_t)((u >> 16) | ((u & 0xffffu) ? 0x40u : 0u));
  u += 0x7fffu + ((u >> 16) & 1u);
  return (bf16_t)(u >> 16);
}

__device__ __forceinline__ unsigned pack2(float a, float b) { return (unsigned)f2bf(a) | ((unsigned)f2bf(b) << 16); }

__device__ __forceinline__ uint64_t now() { return wall_clock64(); }

__device__ __forceinline__ unsigned xcc_id() {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
  return __builtin_amdgcn_s_getreg((3 << 11) | (0 << 6) | 20) & 0xfu;  // HW_REG_XCC_ID[3:0]
#else
  return 0u;
#endif
}

__device__ __forceinline__ void wait_stores() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }

__device__ __forceinline__ void invalidate_l1() {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
  asm volatile("buffer_inv sc0" ::: "memory");
#else
  asm volatile("buffer_wbinvl1" ::: "memory");
#endif
}

__device__ __forceinline__ bool aborted(const Params& P) {
  return __hip_atomic_load(P.ctl, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0u;
}

__device__ __forceinline__ void atomic_add_f32(float* p, float v, int scope) {
  if (scope == AGENT) {
    __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } else {
    __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  }
}

__device__ __forceinline__ bf16x4 half_of(const uint4& v, int hi) {
  bf16x4 r;
  const unsigned a = hi ? v.z : v.x, b = hi ? v.w : v.y;
  r[0] = (short)(a & 0xffffu);
  r[1] = (short)(a >> 16);
  r[2] = (short)(b & 0xffffu);
  r[3] = (short)(b >> 16);
  return r;
}

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
  return v;
}

__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 64));
  return v;
}

// sum across the 16 lanes that share lane / 16
__device__ __forceinline__ float sum16(float v) {
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
  return v;
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

__device__ __forceinline__ Phase load_phase(const Params& P, int i) {
  const int64_t* q = P.program + (int64_t)i * kPhaseFields;
  Phase ph;
  ph.kind = q[0]; ph.pos = q[1]; ph.layer = q[2]; ph.n_tasks = q[3]; ph.chunks = q[4]; ph.ldw = q[5];
  ph.task_begin = q[6]; ph.ss_in = q[7]; ph.ss_out = q[8]; ph.w = q[9]; ph.norm_w = q[10]; ph.norm_w2 = q[11];
  ph.aux = q[12];
  return ph;
}

__device__ __forceinline__ bool is_gemm(int64_t kind) {
  return kind == QKV || kind == OPROJ || kind == GATEUP || kind == DOWN || kind == LMHEAD;
}

// Group geometry and this workgroup's place in it.
struct Group {
  int g, member, members, row0, nrows;
  unsigned* sync;
};

// The next weight chunk this workgroup will multiply: phase, tile (index among the phase's tiles) and chunk.
struct ChunkIt {
  int ph, tile, chunk;
  bool valid;
};

// mode bit 3: the O and down GEMMs as one task per (tile, K chunk), each adding its partial product into the
// residual with an fp32 atomic, so their few tiles spread over more workgroups; every RMSNorm then sums its
// row's squares from the residual itself (the per-tile partial sums need the whole tile).
constexpr int kSplitK = 8;

// S: the kernel instance with the split (a template argument, so the default instance pays nothing for it)
template <bool S>
__device__ __forceinline__ bool split_k(const Phase& d) {
  return S && (d.kind == OPROJ || d.kind == DOWN);
}

// tasks a GEMM phase is cut into, and the chunks of each
template <bool S>
__device__ __forceinline__ int phase_tasks(const Phase& d) {
  return split_k<S>(d) ? (int)(d.n_tasks * d.chunks) : (int)d.n_tasks;
}

template <bool S>
__device__ __forceinline__ int task_chunks(const Phase& d) {
  return split_k<S>(d) ? 1 : (int)d.chunks;
}

// output tile and K chunk of a task's c-th chunk
template <bool S>
__device__ __forceinline__ int2 tile_chunk(const Phase& d, int task, int c) {
  if (split_k<S>(d)) return make_int2(task % (int)d.n_tasks, task / (int)d.n_tasks);
  return make_int2(task, c);
}

// First chunk at or after phase `ph` (`tile` holds the task index).
template <bool S>
__device__ ChunkIt first_chunk_from(const Params& P, const Group& G, int ph) {
  for (; ph < P.n_phases && ph < P.stop_after; ++ph) {
    const Phase d = load_phase(P, ph);
    if (is_gemm(d.kind) && G.member < phase_tasks<S>(d) - d.task_begin) {
      return {ph, (int)d.task_begin + G.member, 0, true};
    }
  }
  return {0, 0, 0, false};
}

template <bool S>
__device__ ChunkIt advance(const Params& P, const Group& G, ChunkIt it) {
  const Phase d = load_phase(P, it.ph);
  if (it.chunk + 1 < task_chunks<S>(d)) return {it.ph, it.tile, it.chunk + 1, true};
  if (it.tile + G.members < phase_tasks<S>(d)) return {it.ph, it.tile + G.members, 0, true};
  return first_chunk_from<S>(P, G, it.ph + 1);
}

// Weights of one chunk for this lane: 64 consecutive K elements of one of the tile's 16 rows.
// Chunk layout: rows n0..n0+15 of W (row stride ldw); K range [k0, k0 + 1024); wave w takes [256 w, 256 w + 256)
// of it and lane l the 64 elements at 64 * (l / 16) for row l % 16.
__device__ __forceinline__ const uint4* chunk_src(const Phase& d, int tile, int chunk) {
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  int n0 = tile * 16, k0 = 0;
  if (d.kind == GATEUP) {
    n0 += chunk ? kInter : 0;  // chunk 0: gate rows, chunk 1: the matching up rows
  } else {
    k0 = chunk * kChunkK;
  }
  const bf16_t* w = reinterpret_cast<const bf16_t*>(d.w);
  return reinterpret_cast<const uint4*>(w + (int64_t)(n0 + (lane & 15)) * d.ldw + k0 + 256 * wave + 64 * (lane >> 4));
}

template <bool S>
__device__ __forceinline__ void load_chunk(uint4 (&wb)[8], const Params& P, const ChunkIt& it) {
  if (!it.valid) return;
  const Phase d = load_phase(P, it.ph);
  const int2 tc = tile_chunk<S>(d, it.tile, it.chunk);
  const uint4* src = chunk_src(d, tc.x, tc.y);
#pragma unroll
  for (int i = 0; i < 8; ++i) wb[i] = src[i];
}

// A operand of a K = 1024 GEMM from the fp32 residual: bf16(h * rsqrt(ss / 1024 + eps) * norm_w), for this
// lane's row (lane % 16) and its 64 K elements; zeros past the group's rows.
template <bool S>
__device__ void load_a_normed(uint4 (&ab)[8], const Params& P, const Group& G, const Phase& d, float* scratch) {
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  const int m = lane & 15;
  const int kb = 256 * wave + 64 * (lane >> 4);
  const bool live = m < G.nrows;
  const int row = G.row0 + (live ? m : 0);
  const float4* hp = reinterpret_cast<const float4*>(P.h + (int64_t)row * kHidden + kb);
  float rs;
  if (S) {
    // the row's sum of squares from the residual: this lane's 64 elements, the wave's four lanes with this row,
    // then the four waves (the elements are read again below, from L1)
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float4 x = live ? hp[i] : make_float4(0.f, 0.f, 0.f, 0.f);
      sq += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
    }
    sq += __shfl_xor(sq, 16, 64);
    sq += __shfl_xor(sq, 32, 64);
    __syncthreads();
    if (lane < 16) scratch[wave * 16 + lane] = sq;
    __syncthreads();
    rs = rsqrtf((scratch[m] + scratch[16 + m] + scratch[32 + m] + scratch[48 + m]) * (1.f / kHidden) + P.eps);
  } else {
    rs = live ? rsqrtf(P.ss[d.ss_in * P.bs + row] * (1.f / kHidden) + P.eps) : 0.f;
  }
  if (!live) {
#pragma unroll
    for (int i = 0; i < 8; ++i) ab[i] = make_uint4(0, 0, 0, 0);
    return;
  }
  const uint4* wp = reinterpret_cast<const uint4*>(reinterpret_cast<const bf16_t*>(d.norm_w) + kb);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float4 x0 = hp[2 * i], x1 = hp[2 * i + 1];
    const uint4 w = wp[i];
    ab[i].x = pack2(x0.x * rs * bf2f(w.x & 0xffffu), x0.y * rs * bf2f(w.x >> 16));
    ab[i].y = pack2(x0.z * rs * bf2f(w.y & 0xffffu), x0.w * rs * bf2f(w.y >> 16));
    ab[i].z = pack2(x1.x * rs * bf2f(w.z & 0xffffu), x1.y * rs * bf2f(w.z >> 16));
    ab[i].w = pack2(x1.z * rs * bf2f(w.w & 0xffffu), x1.w * rs * bf2f(w.w >> 16));
  }
}

// A operand from a bf16 activation [bs][lda] at K offset k0 (the chunk's).
__device__ void load_a_bf16(uint4 (&ab)[8], const Group& G, const bf16_t* a, int lda, int k0) {
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  const int m = lane & 15;
  if (m >= G.nrows) {
#pragma unroll
    for (int i = 0; i < 8; ++i) ab[i] = make_uint4(0, 0, 0, 0);
    return;
  }
  const uint4* src = reinterpret_cast<const uint4*>(a + (int64_t)(G.row0 + m) * lda + k0 + 256 * wave + 64 * (lane >> 4));
#pragma unroll
  for (int i = 0; i < 8; ++i) ab[i] = src[i];
}

__device__ __forceinline__ f32x4 mfma_chunk(const uint4 (&ab)[8], const uint4 (&wb)[8], f32x4 acc) {
#pragma unroll
  for (int s = 0; s < 16; ++s) {
    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(half_of(ab[s >> 1], s & 1), half_of(wb[s >> 1], s & 1), acc, 0, 0, 0);
  }
  return acc;
}

// Sum the four waves' partial tiles; returns the full tile in wave 0 (lane l: rows 4 (l / 16) + i, column l % 16).
__device__ __forceinline__ f32x4 reduce_waves(f32x4 acc, f32x4* red) {
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  __syncthreads();
  red[wave * 64 + lane] = acc;
  __syncthreads();
  if (wave == 0) {
    const f32x4 b = red[64 + lane], c = red[128 + lane], e = red[192 + lane];
    acc = acc + b + c + e;
  }
  return acc;
}

// ---------------------------------------------------------------------------------------------------------
// barrier among a group's members; the chunk after it is requested between the arrival and the wait

// sync_mode bit 0: the barrier's algorithm, 0 = every member adds to one counter and polls it, 1 = every member
// stores its own epoch flag, member 0 polls all of them with one wave and stores a release flag the others poll.
// bit 1: inside one XCD, poll with a workgroup-scope load (sc0) instead of a compare-and-swap (both read L2).
// bit 2 (agent scope, a group over several XCDs): hierarchical. Members arrive on their XCD's counter in that
// XCD's L2; the XCD's first member waits for them, writes the XCD's L2 back once (an agent-scope release, which
// covers every member's stores since they are in that L2), exchanges flags with the other XCDs' first members at
// agent scope, invalidates (agent-scope acquire) and releases its XCD's members, which invalidate their L1.
constexpr int kSyncGather = 1;
constexpr int kSyncLoadPoll = 2;
constexpr int kSyncHier = 4;
// per group: [0] counter, [32] release flag, [64 + m] member m's flag (m < 448); hierarchical: [512 + 32 i] the
// i-th XCD's counter, [520 + 32 i] its release flag, [768 + 32 i] its flag among the XCDs
constexpr int kSyncWords = 1024;

__device__ __forceinline__ unsigned poll_value(unsigned* p, int scope, int mode) {
  if (scope == AGENT) return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (mode & kSyncLoadPoll) return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  // a returning atomic is performed in L2, past this CU's L1
  unsigned seen = 0xffffffffu;
  __hip_atomic_compare_exchange_strong(p, &seen, 0xffffffffu, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_WORKGROUP);
  return seen;
}

__device__ __forceinline__ void store_flag(unsigned* p, unsigned v, int scope) {
  if (scope == AGENT) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  } else {
    wait_stores();
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  }
}

// Spin (one lane) until *p >= target; false on abort or timeout. The abort flag and the clock are read every
// 32 polls, so a satisfied wait is seen one poll late at most.
__device__ bool spin_geq(const Params& P, unsigned* p, unsigned target, uint64_t t0) {
  for (unsigned spins = 1;; ++spins) {
    if (poll_value(p, P.scope, P.sync_mode) >= target) return true;
    if ((spins & 31u) == 0u) {
      if (aborted(P)) return false;
      if (now() - t0 > kSpinTicks) {
        __hip_atomic_store(P.ctl, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        return false;
      }
    }
  }
}

// spin_geq for a flag only this XCD writes, read in this XCD's L2 whatever the barrier's scope
__device__ bool spin_local(const Params& P, unsigned* p, unsigned target, uint64_t t0) {
  for (unsigned spins = 1;; ++spins) {
    if (poll_value(p, XCD, P.sync_mode) >= target) return true;
    if ((spins & 31u) == 0u) {
      if (aborted(P)) return false;
      if (now() - t0 > kSpinTicks) {
        __hip_atomic_store(P.ctl, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        return false;
      }
    }
  }
}

__device__ __forceinline__ void acquire(const Params& P) {
  if (P.scope == AGENT) {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  } else {
    invalidate_l1();
  }
}

// The hierarchical barrier (kSyncHier), for a group whose members span 8 / groups XCDs. Workgroup i is on XCD
// i % 8, so a group's members on one XCD are those with the same i % 8; the lowest of them leads the XCD.
template <bool S>
__device__ bool hier_barrier(const Params& P, const Group& G, unsigned epoch, uint4 (&wnext)[8], const ChunkIt& next,
                             bool prefetch, int* ok_flag) {
  const int xcds = 8 / P.groups;                // XCDs a group spans
  const int xi = (int)((blockIdx.x % 8) / P.groups);  // this workgroup's XCD among them
  const int local = G.members / xcds;           // members on each of them
  unsigned* xcounter = G.sync + 512 + 32 * xi;
  unsigned* xgo = G.sync + 520 + 32 * xi;
  const bool leader = (int)blockIdx.x < 8;      // the lowest workgroup id on its XCD
  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(xcounter, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  }
  if (prefetch) load_chunk<S>(wnext, P, next);
  const uint64_t t0 = now();
  if (threadIdx.x == 0) {
    bool ok;
    if (leader) {
      ok = spin_local(P, xcounter, epoch * (unsigned)local, t0);
      if (ok) {
        // one write-back of this XCD's L2 publishes every local member's stores
        __hip_atomic_store(G.sync + 768 + 32 * xi, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        for (int y = 0; y < xcds && ok; ++y) {
          if (y != xi) ok = spin_geq(P, G.sync + 768 + 32 * y, epoch, t0);
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        if (ok) {
          wait_stores();
          __hip_atomic_store(xgo, epoch, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        }
      }
    } else {
      ok = spin_local(P, xgo, epoch, t0);
      invalidate_l1();
    }
    *ok_flag = ok ? 1 : 0;
  }
  __syncthreads();
  return *ok_flag != 0;
}

template <bool S>
__device__ bool group_barrier(const Params& P, const Group& G, unsigned epoch, uint4 (&wnext)[8],
                              const ChunkIt& next, bool prefetch, int* ok_flag) {
  wait_stores();
  __syncthreads();
  if ((P.sync_mode & kSyncHier) != 0 && P.scope == AGENT && P.groups < 8) {
    return hier_barrier<S>(P, G, epoch, wnext, next, prefetch, ok_flag);
  }
  const bool gather = (P.sync_mode & kSyncGather) != 0;
  if (threadIdx.x == 0) {
    if (gather) {
      store_flag(G.sync + 64 + G.member, epoch, P.scope);
    } else if (P.scope == AGENT) {
      __hip_atomic_fetch_add(G.sync, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    } else {
      __hip_atomic_fetch_add(G.sync, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    }
  }
  if (prefetch) load_chunk<S>(wnext, P, next);
  const uint64_t t0 = now();
  if (gather && G.member == 0) {
    if (threadIdx.x < 64) {
      bool done = false, ok = true;
      for (unsigned spins = 1; !done; ++spins) {
        bool mine = true;
        for (int m = threadIdx.x; m < G.members; m += 64) {
          mine &= poll_value(G.sync + 64 + m, P.scope, P.sync_mode) >= epoch;
        }
        done = __all(mine);
        // decided by the whole wave, so the vote above stays uniform
        if (!done && (spins & 31u) == 0u && __any(aborted(P) || now() - t0 > kSpinTicks)) {
          if (threadIdx.x == 0) __hip_atomic_store(P.ctl, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          ok = false;
          break;
        }
      }
      acquire(P);
      if (threadIdx.x == 0) {
        if (ok) store_flag(G.sync + 32, epoch, P.scope);
        *ok_flag = ok ? 1 : 0;
      }
    }
  } else if (threadIdx.x == 0) {
    const bool ok = spin_geq(P, gather ? G.sync + 32 : G.sync, gather ? epoch : epoch * (unsigned)G.members, t0);
    acquire(P);
    *ok_flag = ok ? 1 : 0;
  }
  __syncthreads();
  return *ok_flag != 0;
}

// ---------------------------------------------------------------------------------------------------------
// GEMM phase: this workgroup's tiles, with the weights of the chunk after each one requested before it runs

struct Regs {
  uint4 wcur[8], wnext[8], a[8];
  bool requested;  // wnext holds (or has been asked for) the chunk `it` names
};

template <bool S>
__device__ void gemm_phase(const Params& P, const Group& G, int ph_idx, const Phase& d, ChunkIt& it, Regs& R,
                           f32x4* red) {
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  const bool shared_a = d.kind == QKV || d.kind == GATEUP || d.kind == LMHEAD;  // one A for the whole phase
  if (!(it.valid && it.ph == ph_idx)) return;
  if (shared_a) load_a_normed<S>(R.a, P, G, d, reinterpret_cast<float*>(red));
  const bool atomic_residual = split_k<S>(d);
  const int nch = task_chunks<S>(d);
  while (it.valid && it.ph == ph_idx) {
    const int task = it.tile;
    const int tile = tile_chunk<S>(d, task, 0).x;
    f32x4 acc0 = {0.f, 0.f, 0.f, 0.f}, acc1 = {0.f, 0.f, 0.f, 0.f};
    for (int c = 0; c < nch; ++c) {
      // this chunk's activations are requested before waiting for its weights (R.wnext, requested a chunk
      // earlier), so the two latencies overlap; then the following chunk's weights are requested
      if (!shared_a) {
        const bf16_t* a = d.kind == OPROJ ? P.attn : P.act;
        load_a_bf16(R.a, G, a, d.kind == OPROJ ? kQHeads * kHeadDim : kInter,
                    tile_chunk<S>(d, task, c).y * kChunkK);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) R.wcur[i] = R.wnext[i];
      it = advance<S>(P, G, it);
      R.requested = false;
      if (it.valid && it.ph == ph_idx) {
        load_chunk<S>(R.wnext, P, it);
        R.requested = true;
      }
      if (d.kind == GATEUP && c == 1) {
        acc1 = mfma_chunk(R.a, R.wcur, acc1);
      } else {
        acc0 = mfma_chunk(R.a, R.wcur, acc0);
      }
    }
    acc0 = reduce_waves(acc0, red);
    if (d.kind == GATEUP) acc1 = reduce_waves(acc1, red);
    if (wave == 0) {
      const int n = tile * 16 + (lane & 15);
      const int mb = 4 * (lane >> 4);
      float sq[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int m = mb + i;
        if (m >= G.nrows) continue;
        const int row = G.row0 + m;
        const float v = acc0[i];
        if (d.kind == QKV) {
          P.qkv[(int64_t)row * kQkv + n] = v;
        } else if (d.kind == GATEUP) {
          P.act[(int64_t)row * kInter + n] = f2bf(silu(v) * acc1[i]);
        } else if (d.kind == LMHEAD) {
          P.logits[((int64_t)d.aux * P.bs + row) * kVocab + n] = v;
        } else if (atomic_residual) {  // one K chunk's share of the residual add
          atomic_add_f32(P.h + (int64_t)row * kHidden + n, v, P.scope);
        } else {  // OPROJ, DOWN: residual add; sum of squares of the new row for the next norm
          float* hp = P.h + (int64_t)row * kHidden + n;
          const float hn = *hp + v;
          *hp = hn;
          sq[i] = hn * hn;
        }
      }
      if ((d.kind == OPROJ || d.kind == DOWN) && !atomic_residual) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float s = sum16(sq[i]);
          const int m = mb + i;
          if ((lane & 15) == 0 && m < G.nrows) atomic_add_f32(P.ss + d.ss_out * P.bs + G.row0 + m, s, P.scope);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------
// attention for one (row, kv head) per wave

__device__ void attn_phase(const Params& P, const Group& G, const Phase& d) {
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  const int slots = G.members * 4, slot = G.member * 4 + wave;
  const int p = (int)d.pos, layer = (int)d.layer;
  const bool need_out = d.aux != 0;  // the last layer at position 0 only fills the cache
  const bf16_t* qn = reinterpret_cast<const bf16_t*>(d.norm_w);
  const bf16_t* kn = reinterpret_cast<const bf16_t*>(d.norm_w2);
  const int d0 = 2 * lane;  // this lane's two dimensions
  const int dr = d0 & 63;   // rope frequency index
  const float c0 = P.rope[p * 128 + dr], c1 = P.rope[p * 128 + dr + 1];
  const float s0 = P.rope[p * 128 + 64 + dr], s1 = P.rope[p * 128 + 64 + dr + 1];
  const bool lo = lane < 32;
  for (int t = slot; t < G.nrows * kKvHeads; t += slots) {
    const int r = G.row0 + t / kKvHeads, kvh = t % kKvHeads;
    const float* base = P.qkv + (int64_t)r * kQkv;
    float x[3][2];
    const int off[3] = {2 * kvh * kHeadDim, (2 * kvh + 1) * kHeadDim, kQHeads * kHeadDim + kvh * kHeadDim};
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const float2 v = *reinterpret_cast<const float2*>(base + off[j] + d0);
      x[j][0] = v.x;
      x[j][1] = v.y;
      const float rs = rsqrtf(wave_sum(v.x * v.x + v.y * v.y) * (1.f / kHeadDim) + P.eps);
      const bf16_t* w = j < 2 ? qn : kn;
      const float a = x[j][0] * rs * bf2f(w[d0]), b = x[j][1] * rs * bf2f(w[d0 + 1]);
      const float pa = __shfl_xor(a, 32, 64), pb = __shfl_xor(b, 32, 64);
      x[j][0] = lo ? a * c0 - pa * s0 : a * c0 + pa * s0;
      x[j][1] = lo ? b * c1 - pb * s1 : b * c1 + pb * s1;
    }
    const float2 vv = *reinterpret_cast<const float2*>(base + kQHeads * kHeadDim + kKvHeads * kHeadDim + kvh * kHeadDim + d0);
    const bf16_t kb0 = f2bf(x[2][0]), kb1 = f2bf(x[2][1]), vb0 = f2bf(vv.x), vb1 = f2bf(vv.y);
    bf16_t* kvrow = P.kv + ((int64_t)(layer * P.bs + r) * 2) * kPositions * kKvHeads * kHeadDim;
    const int64_t pos_stride = kKvHeads * kHeadDim, kind_stride = (int64_t)kPositions * kKvHeads * kHeadDim;
    *reinterpret_cast<unsigned*>(kvrow + p * pos_stride + kvh * kHeadDim + d0) = (unsigned)kb0 | ((unsigned)kb1 << 16);
    *reinterpret_cast<unsigned*>(kvrow + kind_stride + p * pos_stride + kvh * kHeadDim + d0) = (unsigned)vb0 | ((unsigned)vb1 << 16);
    if (!need_out) continue;
    // every position's K and V requested at once (the scratch holds all of them; positions past p are masked),
    // so the reads cost one latency rather than one per position
    unsigned ku[kPositions], vu[kPositions];
#pragma unroll
    for (int j = 0; j < kPositions; ++j) {
      ku[j] = *reinterpret_cast<const unsigned*>(kvrow + j * pos_stride + kvh * kHeadDim + d0);
      vu[j] = *reinterpret_cast<const unsigned*>(kvrow + kind_stride + j * pos_stride + kvh * kHeadDim + d0);
    }
    float sc[2][kPositions];
    float mx0 = -3.0e38f, mx1 = -3.0e38f;
#pragma unroll
    for (int j = 0; j < kPositions; ++j) {
      const unsigned u = j == p ? ((unsigned)kb0 | ((unsigned)kb1 << 16)) : ku[j];
      const float k0v = bf2f(u & 0xffffu), k1v = bf2f(u >> 16);
      const float a0 = wave_sum(x[0][0] * k0v + x[0][1] * k1v) * P.attn_scale;
      const float a1 = wave_sum(x[1][0] * k0v + x[1][1] * k1v) * P.attn_scale;
      sc[0][j] = j <= p ? a0 : -3.0e38f;
      sc[1][j] = j <= p ? a1 : -3.0e38f;
      mx0 = fmaxf(mx0, sc[0][j]);
      mx1 = fmaxf(mx1, sc[1][j]);
    }
    float z0 = 0.f, z1 = 0.f, o00 = 0.f, o01 = 0.f, o10 = 0.f, o11 = 0.f;
#pragma unroll
    for (int j = 0; j < kPositions; ++j) {
      const unsigned u = j == p ? ((unsigned)vb0 | ((unsigned)vb1 << 16)) : vu[j];
      // masked positions contribute nothing even if the scratch there is not a number
      const float v0 = j <= p ? bf2f(u & 0xffffu) : 0.f, v1 = j <= p ? bf2f(u >> 16) : 0.f;
      const float e0 = j <= p ? __expf(sc[0][j] - mx0) : 0.f, e1 = j <= p ? __expf(sc[1][j] - mx1) : 0.f;
      z0 += e0;
      z1 += e1;
      o00 += e0 * v0;
      o01 += e0 * v1;
      o10 += e1 * v0;
      o11 += e1 * v1;
    }
    bf16_t* out = P.attn + (int64_t)r * (kQHeads * kHeadDim);
    *reinterpret_cast<unsigned*>(out + 2 * kvh * kHeadDim + d0) = pack2(o00 / z0, o01 / z0);
    *reinterpret_cast<unsigned*>(out + (2 * kvh + 1) * kHeadDim + d0) = pack2(o10 / z1, o11 / z1);
  }
}

// ---------------------------------------------------------------------------------------------------------
// per-row phases: load the position's input embedding into the residual (PRO, INPUT), or sample a code,
// gather its embedding as the next input and add it to the frame's sum (SAMPLE). One workgroup per row.

__device__ float block_sum(float v, float* scratch) {
  v = wave_sum(v);
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  __syncthreads();
  if (lane == 0) scratch[wave] = v;
  __syncthreads();
  return scratch[0] + scratch[1] + scratch[2] + scratch[3];
}

__device__ void set_input(const Params& P, int row, const bf16_t* src, int slot_out, bool add_to_sum, bool set_h,
                          float* scratch) {
  float sq = 0.f;
  for (int k = threadIdx.x * 4; k < kHidden; k += kThreads * 4) {
    const uint2 u = *reinterpret_cast<const uint2*>(src + k);
    const float e[4] = {bf2f(u.x & 0xffffu), bf2f(u.x >> 16), bf2f(u.y & 0xffffu), bf2f(u.y >> 16)};
    float4* hp = reinterpret_cast<float4*>(P.h + (int64_t)row * kHidden + k);
    if (set_h) *hp = make_float4(e[0], e[1], e[2], e[3]);
    if (add_to_sum) {
      float4* sp = reinterpret_cast<float4*>(P.emb_sum + (int64_t)row * kHidden + k);
      float4 s = *sp;
      s.x += e[0]; s.y += e[1]; s.z += e[2]; s.w += e[3];
      *sp = s;
    }
    sq += e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3];
  }
  const float total = block_sum(sq, scratch);
  if (threadIdx.x == 0 && slot_out >= 0) P.ss[slot_out * P.bs + row] = total;
}

__device__ __forceinline__ uint64_t order_key(float x, int idx) {
  // ascending order of the key = descending x, ties by ascending index
  unsigned u = __float_as_uint(x);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((uint64_t)(~u) << 32) | (unsigned)idx;
}

__device__ int sample_row(const Params& P, int row, int p, uint64_t* keys, float* scratch) {
  const float* lg = P.logits + ((int64_t)(p - 1) * P.bs + row) * kVocab;
  for (int i = threadIdx.x; i < kVocab; i += kThreads) keys[i] = order_key(lg[i] * P.inv_temperature, i);
  __syncthreads();
  // bitonic sort, ascending
  for (int size = 2; size <= kVocab; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int t = threadIdx.x; t < kVocab / 2; t += kThreads) {
        const int lo = 2 * t - (t & (stride - 1));
        const int hi = lo + stride;
        const bool up = (lo & size) == 0;
        const uint64_t a = keys[lo], b = keys[hi];
        if ((a > b) == up) {
          keys[lo] = b;
          keys[hi] = a;
        }
      }
      __syncthreads();
    }
  }
  int code = 0;
  if (threadIdx.x < 64) {
    const int lane = threadIdx.x;
    const bool in_k = lane < P.top_k;
    const int idx = (int)(keys[lane] & 0xffffffffu);
    const float x = in_k ? lg[idx] * P.inv_temperature : -3.0e38f;
    const float mx = lg[(int)(keys[0] & 0xffffffffu)] * P.inv_temperature;
    const float e = in_k ? __expf(x - mx) : 0.f;
    const float q = e / wave_sum(e);
    float c = q;  // inclusive prefix sum over lanes
#pragma unroll
    for (int o = 1; o < 64; o <<= 1) {
      const float y = __shfl_up(c, o, 64);
      if (lane >= o) c += y;
    }
    const bool kept = in_k && (c - q) < P.top_p;
    const float zk = wave_sum(kept ? q : 0.f);
    const uint64_t r = mix64(P.seed ^ mix64(((uint64_t)row << 32) ^ (uint64_t)p));
    const float u = (float)(r >> 40) * (1.0f / 16777216.0f);
    const bool hit = kept && c > u * zk;
    const unsigned long long hits = __ballot(hit);
    const unsigned long long keeps = __ballot(kept);
    const int pick = hits ? __ffsll((long long)hits) - 1 : 63 - __clzll((long long)keeps);
    code = __shfl(idx, pick, 64);
    if (lane == 0) scratch[8] = __int_as_float(code);
  }
  __syncthreads();
  return __float_as_int(scratch[8]);
}

__device__ void row_phase(const Params& P, const Group& G, const Phase& d, uint64_t* keys, float* scratch) {
  for (int t = G.member; t < G.nrows; t += G.members) {
    const int row = G.row0 + t;
    if (d.kind == PRO) {
      set_input(P, row, P.last_hidden + (int64_t)row * kHidden, (int)d.ss_out, false, true, scratch);
    } else if (d.kind == INPUT) {
      set_input(P, row, P.embed0 + (int64_t)row * kHidden, (int)d.ss_out, false, true, scratch);
    } else {  // SAMPLE
      const int p = (int)d.pos;
      int code;
      if (P.forced != nullptr) {
        code = (int)P.forced[row * kPositions + p];
      } else {
        code = sample_row(P, row, p, keys, scratch);
      }
      if (threadIdx.x == 0) P.codes[row * kPositions + p] = code;
      const bf16_t* e = P.codebook + ((int64_t)(p - 1) * kVocab + code) * kHidden;
      const bool last = p == kPositions - 1;
      set_input(P, row, e, last ? -1 : (int)d.ss_out, true, !last, scratch);
    }
    __syncthreads();
  }
}

template <bool S>
__global__ void __launch_bounds__(kThreads, 2) depth_step_kernel(Params P) {
  __shared__ f32x4 red[4 * 64];
  __shared__ uint64_t keys[kVocab];
  __shared__ float scratch[16];
  __shared__ int ok_flag;

  Group G;
  G.g = (int)(blockIdx.x % P.groups);
  G.member = (int)(blockIdx.x / P.groups);
  G.members = (int)(gridDim.x / P.groups);
  G.row0 = (G.g * P.bs) / P.groups;
  G.nrows = ((G.g + 1) * P.bs) / P.groups - G.row0;
  G.sync = P.sync + G.g * kSyncWords;
  const bool placed = P.scope == XCD || (P.sync_mode & kSyncHier) != 0;
  if (placed && (int)xcc_id() != (int)(blockIdx.x % 8)) {
    // the XCD-local protocols rely on workgroup i running on XCD i % 8
    if (threadIdx.x == 0) {
      atomicAdd(P.ctl + 1, 1u);
      __hip_atomic_store(P.ctl, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    return;
  }

  Regs R;
  ChunkIt it = first_chunk_from<S>(P, G, 0);
  load_chunk<S>(R.wnext, P, it);
  R.requested = true;
  unsigned epoch = 0;
  const int n = P.n_phases < P.stop_after ? P.n_phases : P.stop_after;
  for (int ph = 0; ph < n; ++ph) {
    const Phase d = load_phase(P, ph);
    if (is_gemm(d.kind)) {
      gemm_phase<S>(P, G, ph, d, it, R, red);
    } else if (d.kind == ATTN) {
      attn_phase(P, G, d);
    } else {
      row_phase(P, G, d, keys, scratch);
    }
    // the first chunk of a later phase is requested between this barrier's arrival and its wait
    const bool prefetch = it.valid && !R.requested;
    if (prefetch) R.requested = true;
    const bool tracer = P.trace != nullptr && G.g == 0 && G.member == 0 && threadIdx.x == 0;
    if (tracer) P.trace[2 * ph] = now();
    if (!group_barrier<S>(P, G, ++epoch, R.wnext, it, prefetch, &ok_flag)) return;
    if (tracer) P.trace[2 * ph + 1] = now();
    if (G.g == 0 && G.member == 0 && threadIdx.x == 0) P.ctl[2] = (unsigned)(ph + 1);
  }
}

// ---------------------------------------------------------------------------------------------------------

template <typename T>
T* ptr(const at::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

template <typename T>
T* ptr_or_null(const c10::optional<at::Tensor>& t) {
  return t.has_value() && t->defined() ? reinterpret_cast<T*>(t->data_ptr()) : nullptr;
}

void depth_step(const at::Tensor& program, const at::Tensor& rope, const at::Tensor& last_hidden,
                const at::Tensor& embed0, const at::Tensor& codebook, const c10::optional<at::Tensor>& forced,
                const at::Tensor& emb_sum, const at::Tensor& codes, const at::Tensor& logits, const at::Tensor& h,
                const at::Tensor& qkv, const at::Tensor& attn, const at::Tensor& act, const at::Tensor& kv,
                const at::Tensor& ss, const at::Tensor& sync, const at::Tensor& ctl,
                const c10::optional<at::Tensor>& trace, int64_t groups, int64_t scope, int64_t sync_mode, int64_t grid,
                int64_t stop_after, int64_t top_k, double top_p, double temperature, double eps,
                int64_t seed) {
  TORCH_CHECK(program.scalar_type() == at::kLong && program.dim() == 2 && program.size(1) == kPhaseFields,
              "program int64[n, ", kPhaseFields, "]");
  const int bs = (int)last_hidden.size(0);
  TORCH_CHECK(bs <= groups * kMaxRows, "at most ", kMaxRows, " rows per group");
  TORCH_CHECK(grid % groups == 0 && grid / groups >= kMaxRows, "members per group >= ", kMaxRows);
  TORCH_CHECK(top_k >= 1 && top_k <= 64, "top_k in [1, 64]");
  TORCH_CHECK(sync.numel() >= groups * kSyncWords && ctl.numel() >= 4, "sync[groups * ", kSyncWords, "], ctl[4]");
  TORCH_CHECK(grid / groups <= kSyncWords - 64, "at most ", kSyncWords - 64, " members per group");
  Params P;
  P.program = ptr<const int64_t>(program);
  P.rope = ptr<const float>(rope);
  P.last_hidden = ptr<const bf16_t>(last_hidden);
  P.embed0 = ptr<const bf16_t>(embed0);
  P.codebook = ptr<const bf16_t>(codebook);
  P.forced = ptr_or_null<const int64_t>(forced);
  P.emb_sum = ptr<float>(emb_sum);
  P.codes = ptr<int64_t>(codes);
  P.logits = ptr<float>(logits);
  P.h = ptr<float>(h);
  P.qkv = ptr<float>(qkv);
  P.attn = ptr<bf16_t>(attn);
  P.act = ptr<bf16_t>(act);
  P.kv = ptr<bf16_t>(kv);
  P.ss = ptr<float>(ss);
  P.sync = ptr<unsigned>(sync);
  P.ctl = ptr<unsigned>(ctl);
  P.trace = ptr_or_null<uint64_t>(trace);
  TORCH_CHECK(!P.trace || trace->numel() >= 2 * program.size(0), "trace int64[2 * n_phases]");
  P.n_phases = (int)program.size(0);
  P.bs = bs;
  P.groups = (int)groups;
  P.scope = (int)scope;
  P.sync_mode = (int)sync_mode;
  P.stop_after = stop_after < 0 ? (int)program.size(0) : (int)stop_after;
  P.top_k = (int)top_k;
  P.top_p = (float)top_p;
  P.inv_temperature = temperature > 0 ? (float)(1.0 / temperature) : 1.f;
  P.eps = (float)eps;
  P.attn_scale = 1.f / sqrtf((float)kHeadDim);
  P.seed = (uint64_t)seed;
  if ((sync_mode & kSplitK) != 0) {
    hipLaunchKernelGGL(depth_step_kernel<true>, dim3(grid), dim3(kThreads), 0, at::cuda::getCurrentCUDAStream().stream(),
                       P);
  } else {
    hipLaunchKernelGGL(depth_step_kernel<false>, dim3(grid), dim3(kThreads), 0, at::cuda::getCurrentCUDAStream().stream(),
                       P);
  }
  hipError_t e = hipGetLastError();
  TORCH_CHECK(e == hipSuccess, "depth_step launch: ", hipGetErrorString(e));
}

int64_t wall_clock_khz() {
  int dev = 0, khz = 0;
  TORCH_CHECK(hipGetDevice(&dev) == hipSuccess, "hipGetDevice");
  TORCH_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) == hipSuccess,
              "hipDeviceAttributeWallClockRate");
  return khz;
}

}  // namespace

TORCH_LIBRARY(mstar_persistent, m) {
  m.def("depth_step(Tensor program, Tensor rope, Tensor last_hidden, Tensor embed0, Tensor codebook, Tensor? forced, "
        "Tensor(a!) emb_sum, Tensor(b!) codes, Tensor(c!) logits, Tensor(d!) h, Tensor(e!) qkv, Tensor(f!) attn, "
        "Tensor(g!) act, Tensor(h!) kv, Tensor(i!) ss, Tensor(j!) sync, Tensor(k!) ctl, Tensor(l!)? trace, "
        "int groups, int scope, "
        "int sync_mode, int grid, "
        "int stop_after, int top_k, float top_p, float temperature, float eps, int seed) -> ()");
  m.def("wall_clock_khz() -> int", &wall_clock_khz);
}

TORCH_LIBRARY_IMPL(mstar_persistent, CUDA, m) { m.impl("depth_step", &depth_step); }
