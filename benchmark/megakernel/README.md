# Persistent decode kernels: microbenchmarks

Qwen3.5-0.8B's decode layer stack on `mstar/kernels/megakernel`, against the same ops as the engine's captured decode graph runs them: the model's own modules and the engine's resources (recurrent pool + FlashInfer GDN decode, paged KV + trtllm-gen fmha), compiled as `-O2` compiles a decode bucket (`max-autotune-no-cudagraphs`, `dynamic=False`) and captured in one CUDA graph. Every script uses the real checkpoint (`HF_HUB_OFFLINE=1`, a cached `Qwen/Qwen3.5-0.8B`) and needs a compute capability 10.x GPU.

| script | what |
|---|---|
| `mb_mlp.py` | one layer's MLP block (add+norm, gate/up, SiLU, down, residual + next norm), 24 layers chained |
| `mb_gdn.py` | one gated-delta-net mixer, the 18 GDN layers, state and conv window parity |
| `mb_attn.py` | one full-attention mixer at KV 512/2048, new KV entry parity |
| `mb_block.py` | the 4-layer block (3 GDN + 1 attention, each with its MLP): graph arm vs the stage kernels back to back vs `BlockMegakernel`, all 24 layers, plus block-0 parity |
| `greedy_block.py` | greedy tokens with block 0 on the fused kernel vs on the model's modules (eager and compiled), 32 prompts |
| `bw.py` | copy bandwidth (1-2 GiB) and cuBLAS GEMV times at the model's shapes |

Run from the repo root, e.g. `python -m benchmark.megakernel.mb_block --rows 1 --kv 512`. Times are sustained medians of graph replays (`common.timed_us`).
