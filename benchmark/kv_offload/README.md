# KV offload copy benchmark

How fast the paged KV cache moves pages to and from host memory (`CPUPagePool`), and what that costs a deployment whose pool is too small for its load.

## Copy microbenchmark

`copy_bench.py` drives `CPUPagePool` directly, on one GPU, through each copier in `mstar/engine/resources/kv/page_copier.py` that loads: one `cudaMemcpy2DAsync` per run of adjacent pages through `cuda.bindings` or through the libcudart torch loaded, one `hipMemcpy2DAsync` per run through torch's libamdhip64 on ROCm, and the per-page `copy_` that is the fallback (and was the only path before). The pages it copies are adjacent, so each 2D copier makes one call per copy. For each direction and page count it reports:

- **host ms**: from issuing the copies to `sync()` returning, with `--queued-ms` of compute already queued on the current stream. A path that makes the host wait for that compute reports at least `--queued-ms`.
- **copy ms / GB/s**: CUDA-event time of the copies alone on the pool's stream.

Defaults are one BAGEL-7B page per TP2 rank (28 layers, 2 KV heads of 128, 128 tokens, bf16: 3.7 MB); median of 5.

```bash
python benchmark/kv_offload/copy_bench.py --pages 1,16,48
```

## End to end

BAGEL with its LLM on TP2 and a deliberately small KV pool (`tight_pool.yaml`: 64 device pages, 256 host pages, 16 concurrent requests), driven by `load.py`: 384 text requests of ~0.3k–2.5k tokens with 16-token replies, at concurrency 32 and then 16. A request holds up to ~20 pages, so most admissions evict a running request to the host and reload another.

```bash
# two GPUs; needs the bagel extra
python -m mstar.api_server.entrypoint --config benchmark/kv_offload/tight_pool.yaml \
    --port 8000 --tensor-comm-protocol SHM 2>&1 | tee server.log
# once it reports ready
python benchmark/kv_offload/load.py --url http://localhost:8000 --concurrency 32
python benchmark/kv_offload/load.py --url http://localhost:8000 --concurrency 16
grep -c "Offloaded request" server.log   # evictions; only rank 0 picks victims and logs them
```

Compare runs with a fresh server each; wall time is the number to read, and the timeouts are requests that waited past the client's 180 s.
