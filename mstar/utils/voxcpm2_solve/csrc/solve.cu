// B1 VoxCPM2 solve: library contractions and fused consumers, captured by M*'s engine.
#include "gate.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <array>
#include <memory>
#include <unordered_map>

using vox::Element;
constexpr int WIDTH = 1024, SEQ = 11, HEADS = 16, DIM = 128;

struct Params {
    struct Bias { Element const* bias; } linear[9];
    Element const* norms[25];
    Element const *cond, *noise, *span;
    float const *cfg, *cos, *sin;
    Element *x, *cond_in, *x_in, *embedding, *time_hidden, *delta_embedding,
            *delta_hidden, *time, *delta, *hidden, *normed, *qkv, *attn,
            *activated, *slice, *mu0, *mu1, *cond_out, *x_out, *output;
    float* partial;
    int batch;
    float eps;
};

__device__ float bf(float x) { return float(Element(x)); }
__device__ int index() { return blockIdx.x * blockDim.x + threadIdx.x; }
__device__ int stride() { return gridDim.x * blockDim.x; }
__device__ float warp_sum(float x) {
    for (int d = 16; d; d /= 2) x += __shfl_down_sync(0xffffffff, x, d);
    return __shfl_sync(0xffffffff, x, 0);
}
// One CTA owns one query/head; eleven warps compute the eleven QK scores
// concurrently. Scores communicate only within this CTA, never across CTAs.
__global__ __launch_bounds__(384) void key_parallel_attention(__grid_constant__ const Params p) {
    int job = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    int q = job % SEQ, head = (job / SEQ) % HEADS;
    int sample = job / (SEQ * HEADS), kv = head / 8;
    __shared__ float scores[SEQ], probabilities[SEQ];
    if (warp < SEQ) {
        float dot = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int d = lane + 32 * j;
            dot += float(p.qkv[(sample * SEQ + q) * 2560 + head * DIM + d]) *
                   float(p.qkv[(sample * SEQ + warp) * 2560 + 2048 + kv * DIM + d]);
        }
        float score = warp_sum(dot) * 0.08838834764831845f;
        if (lane == 0) scores[warp] = score;
    }
    __syncthreads();
    if (warp == 0) {
        float score = lane < SEQ ? scores[lane] : -INFINITY;
        float maximum = score;
        #pragma unroll
        for (int d = 16; d; d /= 2) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, d));
        float ex = lane < SEQ ? expf(score - maximum) : 0.f;
        float sum = warp_sum(ex);
        if (lane < SEQ) probabilities[lane] = bf(ex) / sum;
    }
    __syncthreads();
    if (tid < DIM) {
        float acc = 0;
        #pragma unroll
        for (int k = 0; k < SEQ; ++k)
            acc += probabilities[k] * float(p.qkv[(sample * SEQ + k) * 2560 + 2304 + kv * DIM + tid]);
        p.attn[(sample * SEQ + q) * 2048 + head * DIM + tid] = Element(acc);
    }
}
__global__ __launch_bounds__(128) void prepare_inputs(__grid_constant__ const Params p) {
    int rows = 2 * p.batch;
    for (int i = index(); i < p.batch * 256; i += stride()) p.x[i] = p.noise[i];
    for (int i = index(); i < rows * 256; i += stride()) {
        int row = i / 256, patch = i % 256 / 64, feat = i % 64;
        p.x_in[i] = p.noise[(row % p.batch) * 256 + feat * 4 + patch];
    }
    // Inputs use [B, patch, feature]; the evolving state uses [B, feature, patch].
    for (int i = index(); i < rows * 256; i += stride()) p.cond_in[i] = p.cond[i % (p.batch * 256)];
}
__global__ __launch_bounds__(128) void prepare_time(__grid_constant__ const Params p) {
    int rows = 2 * p.batch;
    // Every replay refreshes all nine embeddings from the current schedule.
    for (int i = index(); i < 9 * rows * WIDTH; i += stride()) {
        int d = i % WIDTH, f = d % 512;
        int step_index = i / (rows * WIDTH) + 1;
        float frequency = expf(float(f) * -0.01802414945592208f);
        float angle = 1000.f * float(p.span[step_index]) * frequency;
        p.embedding[i] = Element(d < 512 ? sinf(angle) : cosf(angle));
    }
    for (int i = index(); i < rows * WIDTH; i += stride())
        p.delta_embedding[i] = Element(i % WIDTH < 512 ? 0.f : 1.f);
}
// Assemble all eleven tokens and immediately consume them in initial RMSNorm.
// No other CTA reads this token until the kernel boundary.
__global__ __launch_bounds__(256) void assemble_norm(__grid_constant__ const Params p, int step) {
    int row = blockIdx.x, token = row % SEQ, sample = row / SEQ;
    int tid = threadIdx.x, col = 4 * tid, i = row * WIDTH + col;
    int rows = 2 * p.batch;
    struct alignas(8) Four { Element lanes[4]; } hidden, normalized;
    float values[4], square = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int d = col + j;
        Element value;
        if (token < 2) value = sample < p.batch ? (token == 0 ? p.mu0[sample * WIDTH + d] : p.mu1[sample * WIDTH + d]) : Element(0.f);
        else if (token == 2) value = Element((float(p.time[((step - 1) * rows + sample) * WIDTH + d]) + float(p.linear[5].bias[d])) + (float(p.delta[sample * WIDTH + d]) + float(p.linear[7].bias[d])));
        else if (token < 7) value = Element(float(p.cond_out[(sample * 4 + token - 3) * WIDTH + d]) + float(p.linear[2].bias[d]));
        else value = Element(float(p.x_out[(sample * 4 + token - 7) * WIDTH + d]) + float(p.linear[3].bias[d]));
        hidden.lanes[j] = value;
        values[j] = float(value);
        square += values[j] * values[j];
    }
    reinterpret_cast<uint2*>(p.hidden)[i / 4] = *reinterpret_cast<uint2*>(&hidden);
    __shared__ float squares[8];
    square = warp_sum(square);
    if (tid % 32 == 0) squares[tid / 32] = square;
    __syncthreads();
    // Each warp redundantly reduces the eight partials, avoiding a second CTA barrier.
    float inv = rsqrtf(warp_sum(tid % 32 < 8 ? squares[tid % 32] : 0.f) / WIDTH + p.eps);
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        normalized.lanes[j] = Element(values[j] * inv * float(p.norms[0][col + j]));
    reinterpret_cast<uint2*>(p.normed)[i / 4] = *reinterpret_cast<uint2*>(&normalized);
}
// Final projection's BF16 velocity publication is consumed in-register by
// the original FP32 CFG/Euler equation. One warp owns one sample.
__global__ __launch_bounds__(32) void finish_cfg_euler(
    __grid_constant__ const Params p, float const* partial, int step) {
    int b = blockIdx.x, lane = threadIdx.x;
    float dt = float(p.span[step]) - float(p.span[step + 1]);
    float pos[8], neg[8], dot = 0.f, square = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int i = lane + j * 32;
        float bias = float(p.linear[8].bias[i % 64]);
        // Publish the projected velocity in BF16 before adding its bias.
        pos[j] = bf(partial[b * 256 + i]) + bias;
        neg[j] = bf(partial[(b + p.batch) * 256 + i]) + bias;
        dot += pos[j] * neg[j];
        square += neg[j] * neg[j];
    }
    float scale = warp_sum(dot) / (warp_sum(square) + 1e-8f);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int i = lane + j * 32;
        float scaled = neg[j] * scale;
        float velocity = scaled + p.cfg[b] * (pos[j] - scaled);
        int state = b * 256 + (i % 64) * 4 + i / 64;
        Element next = Element(float(p.x[state]) - bf(dt * velocity));
        p.x[state] = next;
        p.x_in[b * 256 + i] = next;
        p.x_in[(b + p.batch) * 256 + i] = next;
        if (step == 9) p.output[b * 256 + i] = next;
    }
}
__global__ void finish_time_silu(float const* partial, Element* out, Element const* bias,
                                 int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m * n) {
        float value = bf(partial[i]) + float(bias[i % n]);
        value = value / (1.f + expf(-value));
        out[i] = Element(value);
    }
}
// Each thread owns both halves of one projected Q/K/V pair. Round the
// projection before RoPE, then publish all Q/K/V at one kernel boundary.
__global__ void finish_qkv_rope(float const* partial, Element* out,
    int m, float const* cosine, float const* sine) {
    int pair_index = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (pair_index >= m * 20 * 64) return;
    int pair = pair_index % 64, head = pair_index / 64 % 20;
    int row = pair_index / (20 * 64), col = head * DIM + pair;
    int i = row * 2560 + col;
    float2 a = *reinterpret_cast<float2 const*>(partial + i);
    float2 b = *reinterpret_cast<float2 const*>(partial + i + 64);
    #pragma unroll
    for (int part = 1; part < 2; ++part) {
        float2 x = *reinterpret_cast<float2 const*>(partial + part * m * 2560 + i);
        float2 y = *reinterpret_cast<float2 const*>(partial + part * m * 2560 + i + 64);
        a.x += x.x; a.y += x.y; b.x += y.x; b.y += y.y;
    }
    struct alignas(4) Two { Element lane[2]; } lo, hi;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        float x = j == 0 ? a.x : a.y, y = j == 0 ? b.x : b.y;
        x = bf(x); y = bf(y);
        if (head < 18) {
            int pos = row % SEQ * DIM + pair + j;
            lo.lane[j] = Element(x * cosine[pos] - y * sine[pos]);
            hi.lane[j] = Element(y * cosine[pos+64] + x * sine[pos+64]);
        } else { lo.lane[j] = Element(x); hi.lane[j] = Element(y); }
    }
    *reinterpret_cast<unsigned*>(out+i) = *reinterpret_cast<unsigned*>(&lo);
    *reinterpret_cast<unsigned*>(out+i+64) = *reinterpret_cast<unsigned*>(&hi);
}
// Vector ownership: one 256-thread CTA per token, four contiguous channels
// per thread. All residual reads precede publication and each channel has a
// unique writer. GEMM planes and all original BF16 boundaries are retained.
template<int Parts>
__global__ __launch_bounds__(256) void finish_residual_norm(
    float const* partial, Element* out, Element const* residual,
    Element const* norm_weight, Element* normed, float eps, int partition_stride, Element* final_slice) {
    int row = blockIdx.x, tid = threadIdx.x, col = 4 * tid, i = row * WIDTH + col;
    float4 packed = reinterpret_cast<float4 const*>(partial)[i / 4];
    float values[4] = {packed.x, packed.y, packed.z, packed.w};
    #pragma unroll
    for (int part = 1; part < Parts; ++part) {
        float4 v = reinterpret_cast<float4 const*>(partial + part * partition_stride)[i / 4];
        values[0] += v.x; values[1] += v.y; values[2] += v.z; values[3] += v.w;
    }
    struct alignas(8) Four { Element lanes[4]; } res, weights, rounded, normalized;
    if (residual) *reinterpret_cast<uint2*>(&res) = reinterpret_cast<uint2 const*>(residual)[i / 4];
    *reinterpret_cast<uint2*>(&weights) = reinterpret_cast<uint2 const*>(norm_weight)[col / 4];
    float sum = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float value = values[j];
        if (residual) value += float(res.lanes[j]);
        value = bf(value);
        values[j] = value;
        rounded.lanes[j] = Element(value);
        sum += value * value;
    }
    reinterpret_cast<uint2*>(out)[i / 4] = *reinterpret_cast<uint2*>(&rounded);
    __shared__ float sums[8];
    sum = warp_sum(sum);
    if (tid % 32 == 0) sums[tid / 32] = sum;
    __syncthreads();
    float inv = rsqrtf(warp_sum(tid % 32 < 8 ? sums[tid % 32] : 0.f) / WIDTH + eps);
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        normalized.lanes[j] = Element(values[j] * inv * float(weights.lanes[j]));
    reinterpret_cast<uint2*>(normed)[i / 4] = *reinterpret_cast<uint2*>(&normalized);
    if (final_slice && row % SEQ >= 7) {
        int target_row = row / SEQ * 4 + row % SEQ - 7;
        reinterpret_cast<uint2*>(final_slice)[(target_row * WIDTH + col) / 4] = *reinterpret_cast<uint2*>(&normalized);
    }
}

// Tensor-map descriptors are launch arguments, not mutable global storage. CUDA
// retains their values in each graph node; no H2D parameter copy is replayed.
__global__ __launch_bounds__(128) void paired_gate_contract(
    __grid_constant__ const vox::Linear linear, Element* activated) {
    extern __shared__ __align__(128) unsigned char memory[];
    auto& smem = *reinterpret_cast<vox::SharedGated*>(memory);
    vox::gated_tile(linear, activated, blockIdx.y, blockIdx.x, smem);
}

namespace {
constexpr size_t WORKSPACE_BYTES = 16 * 1024 * 1024;
struct Shape { int m, n, k; };
Shape shape(int i) {
    constexpr Shape outer[] = {{1,1024,2048}, {1,1024,2048}, {8,1024,64},
        {8,1024,64}, {18,1024,1024}, {18,1024,1024}, {2,1024,1024},
        {2,1024,1024}, {8,64,1024}};
    constexpr Shape block[] = {{22,2560,1024}, {22,1024,2048},
        {22,8192,1024}, {22,1024,4096}};
    return i < 9 ? outer[i] : block[(i - 9) % 4];
}
bool gate(int i) { return i >= 9 && (i - 9) % 4 == 2; }
bool direct(int i) { return i == 0 || i == 1 || i == 2 || i == 3 || i == 5 || i == 7; }
int partitions(int i) {
    if (i < 9) return 1;
    constexpr int parts[] = {2,8,1,8};
    return parts[(i-9)%4];
}
void blas(cublasStatus_t status) { TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "Vox solve cuBLASLt status ", int(status)); }

// Only host algorithm metadata is reused. Bias pointers are rebound before each
// launch, on this thread. No activation, output or scratch address is cached.
struct Contraction {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    ~Contraction() {
        if (op) cublasLtMatmulDescDestroy(op);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
    }
};
struct Algorithms {
    std::array<Contraction,57> contraction;
    int blocks;
    Algorithms() {
        auto prop = at::cuda::getCurrentDeviceProperties();
        TORCH_CHECK(prop->major == 9 && prop->minor == 0, "Vox solve requires Hopper");
        blocks = prop->multiProcessorCount;
        C10_CUDA_CHECK(cudaFuncSetAttribute(paired_gate_contract,
            cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(vox::SharedGated)));
        auto lt = at::cuda::getCurrentCUDABlasLtHandle();
        // One heuristic, the same workspace limit and layouts as the qualified solve.
        struct Preference {
            cublasLtMatmulPreference_t value = nullptr;
            ~Preference() { if (value) cublasLtMatmulPreferenceDestroy(value); }
        } pref;
        blas(cublasLtMatmulPreferenceCreate(&pref.value));
        size_t bytes = WORKSPACE_BYTES;
        blas(cublasLtMatmulPreferenceSetAttribute(pref.value,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes)));
        for (int i=0; i<57; ++i) {
            if (gate(i)) continue;
            auto [m,n,k] = shape(i);
            auto& q = contraction[i];
            blas(cublasLtMatmulDescCreate(&q.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
            cublasOperation_t trans = CUBLAS_OP_T;
            blas(cublasLtMatmulDescSetAttribute(q.op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
            if (i < 2) {
                cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
                cudaDataType_t bt = CUDA_R_16BF;
                blas(cublasLtMatmulDescSetAttribute(q.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep)));
                blas(cublasLtMatmulDescSetAttribute(q.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
            }
            int parts = partitions(i), inner = k / parts;
            blas(cublasLtMatrixLayoutCreate(&q.a, CUDA_R_16BF, inner, n, k));
            blas(cublasLtMatrixLayoutCreate(&q.b, CUDA_R_16BF, inner, m, k));
            blas(cublasLtMatrixLayoutCreate(&q.c, direct(i) ? CUDA_R_16BF : CUDA_R_32F, n, m, n));
            if (parts > 1) {
                int64_t input_stride = inner, output_stride = int64_t(m)*n;
                for (auto layout : {q.a,q.b,q.c})
                    blas(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &parts, sizeof(parts)));
                for (auto layout : {q.a,q.b})
                    blas(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &input_stride, sizeof(input_stride)));
                blas(cublasLtMatrixLayoutSetAttribute(q.c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &output_stride, sizeof(output_stride)));
            }
            cublasLtMatmulHeuristicResult_t result{};
            int count=0;
            blas(cublasLtMatmulAlgoGetHeuristic(lt,q.op,q.a,q.b,q.c,q.c,pref.value,1,&result,&count));
            TORCH_CHECK(count==1 && result.state==CUBLAS_STATUS_SUCCESS, "No Vox cuBLASLt algorithm for projection ",i);
            q.algo=result.algo;
        }
    }
};
Algorithms& algorithms() {
    static thread_local std::unordered_map<int,std::unique_ptr<Algorithms>> cache;
    int device = c10::cuda::current_device();
    auto& entry=cache[device];
    if (!entry) {
        cudaStreamCaptureStatus status;
        C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &status));
        TORCH_CHECK(status==cudaStreamCaptureStatusNone, "Warm up Vox solve on the capture thread before capture");
        entry=std::make_unique<Algorithms>();
    }
    return *entry;
}
Element* bfptr(at::Tensor const& x) { return reinterpret_cast<Element*>(x.data_ptr<at::BFloat16>()); }
void check_tensor(at::Tensor const& x, at::Tensor const& like, at::IntArrayRef dims,
                  at::ScalarType dtype = at::kBFloat16) {
    TORCH_CHECK(x.device()==like.device() && x.scalar_type()==dtype && x.is_contiguous() && x.sizes()==dims,
                "Vox solve: invalid tensor shape/dtype/device/strides; expected ",dims);
}
} // namespace

at::Tensor vox_solve(at::Tensor lm, at::Tensor res, at::Tensor cond, at::Tensor noise,
    at::Tensor cfg, at::Tensor span, std::vector<at::Tensor> weights,
    std::vector<at::Tensor> biases, std::vector<at::Tensor> norms,
    at::Tensor cosine, at::Tensor sine, double eps) {
    TORCH_CHECK(lm.is_cuda(), "Vox solve requires CUDA tensors");
    c10::cuda::CUDAGuard guard(lm.device());
    constexpr int batch = 1, tokens = 2 * SEQ;
    check_tensor(lm,lm,{batch,2048}); check_tensor(res,lm,{batch,2048});
    check_tensor(cond,lm,{batch,4,64}); check_tensor(noise,lm,{batch,64,4});
    check_tensor(cfg,lm,{batch},at::kFloat); check_tensor(span,lm,{11});
    TORCH_CHECK(weights.size()==57 && biases.size()==9 && norms.size()==25, "Invalid Vox weight count");
    for (int i=0;i<57;++i) {
        auto d=shape(i); check_tensor(weights[i],lm,{d.n,d.k});
        if(i<9) check_tensor(biases[i],lm,{d.n});
    }
    for(auto const& w:norms) check_tensor(w,lm,{1024});
    check_tensor(cosine,lm,{11,128},at::kFloat); check_tensor(sine,lm,{11,128},at::kFloat);
    auto& alg=algorithms();
    auto st=at::cuda::getCurrentCUDAStream();
    auto lt=at::cuda::getCurrentCUDABlasLtHandle();
    // Allocator-backed invocation storage: captured by the engine's graph pool,
    // stream-ordered in eager execution, and never shared across graphs.
    std::vector<at::Tensor> scratch;
    auto buffer=[&](int64_t size) {
        scratch.push_back(at::empty({size},lm.options())); return bfptr(scratch.back());
    };
    auto output=at::empty({batch,4,64},lm.options());
    auto partial=at::empty({8*tokens*1024},lm.options().dtype(at::kFloat));
    auto workspace=at::empty({int64_t(WORKSPACE_BYTES)},lm.options().dtype(at::kByte));
    Params p{}; p.batch=batch; p.eps=float(eps);
    p.cond=bfptr(cond); p.noise=bfptr(noise); p.span=bfptr(span); p.cfg=cfg.data_ptr<float>();
    p.cos=cosine.data_ptr<float>(); p.sin=sine.data_ptr<float>();
    p.x=buffer(256); p.cond_in=buffer(8*64); p.x_in=buffer(8*64);
    p.embedding=buffer(18*1024); p.time_hidden=buffer(18*1024); p.time=buffer(18*1024);
    p.delta_embedding=buffer(2*1024); p.delta_hidden=buffer(2*1024); p.delta=buffer(2*1024);
    p.hidden=buffer(22*1024); p.normed=buffer(22*1024); p.qkv=buffer(22*2560);
    p.attn=buffer(22*2048); p.activated=buffer(22*4096); p.slice=buffer(8*1024);
    p.mu0=buffer(1024); p.mu1=buffer(1024); p.cond_out=buffer(8*1024); p.x_out=buffer(8*1024);
    p.output=bfptr(output); p.partial=partial.data_ptr<float>();
    for(int i=0;i<9;++i) p.linear[i].bias=bfptr(biases[i]);
    for(int i=0;i<25;++i) p.norms[i]=bfptr(norms[i]);
    struct Projection { Element const* input; Element* output; Element const* residual; };
    std::array<Projection,57> proj{};
    proj[0]={bfptr(lm),p.mu0,nullptr}; proj[1]={bfptr(res),p.mu1,nullptr};
    proj[2]={p.cond_in,p.cond_out,nullptr}; proj[3]={p.x_in,p.x_out,nullptr};
    proj[4]={p.embedding,p.time_hidden,nullptr}; proj[5]={p.time_hidden,p.time,nullptr};
    proj[6]={p.delta_embedding,p.delta_hidden,nullptr}; proj[7]={p.delta_hidden,p.delta,nullptr};
    proj[8]={p.slice,nullptr,nullptr};
    std::array<vox::Linear,12> gates;
    for(int l=0;l<12;++l) {
        proj[9+4*l]={p.normed,p.qkv,nullptr}; proj[10+4*l]={p.attn,p.hidden,p.hidden};
        proj[11+4*l]={p.normed,p.activated,nullptr}; proj[12+4*l]={p.activated,p.hidden,p.hidden};
        auto a=cute::make_tensor(static_cast<Element const*>(p.normed),cute::make_shape(tokens,1024),cute::make_stride(1024,cute::_1{}));
        auto b=cute::make_tensor(static_cast<Element const*>(bfptr(weights[11+4*l])),cute::make_shape(8192,1024),cute::make_stride(1024,cute::_1{}));
        gates[l]={cute::make_tma_atom(cute::SM90_TMA_LOAD{},b,vox::SmemLayout{}(cute::_,cute::_,0),cute::Shape<cute::_64,cute::_64>{}),
                  cute::make_tma_atom(cute::SM90_TMA_LOAD{},a,vox::TokenSmemLayout{}(cute::_,cute::_,0),cute::Shape<cute::_32,cute::_64>{}),tokens,8192,1024};
    }
    auto mm=[&](int i) {
        if(gate(i)) {
            paired_gate_contract<<<dim3(128,(tokens+31)/32,1),128,sizeof(vox::SharedGated),st>>>(gates[(i-9)/4],p.activated);
            return;
        }
        auto d=shape(i); auto const& v=proj[i]; auto& q=alg.contraction[i];
        auto bias=i<9 ? bfptr(biases[i]) : nullptr;
        if(i<2) blas(cublasLtMatmulDescSetAttribute(q.op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias,sizeof(bias)));
        float alpha=1.f,beta=0.f;
        void* dst=direct(i) ? static_cast<void*>(v.output) : static_cast<void*>(p.partial);
        blas(cublasLtMatmul(lt,q.op,&alpha,bfptr(weights[i]),q.a,v.input,q.b,&beta,dst,q.c,dst,q.c,
            &q.algo,workspace.data_ptr(),WORKSPACE_BYTES,st));
        if(direct(i) || i==8) return;
        if(i>=9 && (i-9)%4==0) {
            finish_qkv_rope<<<(d.m*640+127)/128,128,0,st>>>(p.partial,v.output,d.m,p.cos,p.sin);
        } else if(i>=9) {
            int ni=2*((i-9)/4)+((i-9)%4==1 ? 1 : 2);
            finish_residual_norm<8><<<d.m,256,0,st>>>(p.partial,v.output,v.residual,
                p.norms[ni],p.normed,p.eps,d.m*d.n,ni==24 ? p.slice : nullptr);
        } else {
            finish_time_silu<<<(d.m*d.n+255)/256,256,0,st>>>(p.partial,v.output,bias,d.m,d.n);
        }
    };
    prepare_inputs<<<alg.blocks,128,0,st>>>(p); mm(0); mm(1); mm(2);
    prepare_time<<<alg.blocks,128,0,st>>>(p); mm(4); mm(5); mm(6); mm(7);
    for(int step=1;step<10;++step) {
        mm(3); assemble_norm<<<tokens,256,0,st>>>(p,step);
        for(int layer=0;layer<12;++layer) {
            mm(9+4*layer);
            key_parallel_attention<<<2*batch*HEADS*SEQ,384,0,st>>>(p);
            mm(10+4*layer); mm(11+4*layer); mm(12+4*layer);
        }
        mm(8); finish_cfg_euler<<<batch,32,0,st>>>(p,p.partial,step);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
