# KV store benchmark

Time to first token for long text prompts with the KV store tier (`resources.<kv>.store`) on, against the same deployment with it off: a recompute, a GPU hit, a host hit, and a store hit read by a process that didn't write it. `ttft.py` sends the prompts: distinct filler text of about `--tokens` tokens each, the same for a given `--seed`, so two servers get byte-identical prompts. It streams each reply and records the time to the first content token, the prompt tokens the server counted and the greedy reply.

## Setup on one machine

Over TCP, Mooncake opens one connection to each peer unless told otherwise; `MC_TCP_LANES_PER_PEER=4` about doubled store reads on one H100 machine. Export it for every process below.

```bash
mooncake_master --rpc_port=50551 --metrics_port=9003 &
# memory for the store that outlives the M* processes: a Mooncake client that gives a segment and stays up,
# such as mooncake_store_service, or a short script that calls MooncakeDistributedStore.setup() with a
# global segment size and sleeps (keep a reference to the client: dropping it unmounts the segment)
```

The deployment is `configs/bagel_single_gpu.yaml` with a `kv` block. The store-off server is the same file without `store:`.

```yaml
resources:
  kv:
    max_num_pages: 2048
    cpu_offload_pages: 6144          # store reads and writes each hold up to an eighth
    prefix_cache_salt: kvstore-bench
    store:
      backend: mooncake
      mooncake:
        master: 127.0.0.1:50551
        protocol: tcp
```

## Run

Start each server with `mstar serve bagel --config <yaml> --gpus 0 --port <port>`, one at a time.

1. **Writer, store on.** Send one warmup prompt per length with a seed of its own, then the measured prompts (seed 0). Store misses: the time is a recompute plus the store lookup. After each prompt, send a decode-only request (`--filler 300`) until the master's `master_key_count` (on `localhost:9003/metrics`) stops growing: pages reach the host, and then the store, only after a step. Send the measured prompts again for GPU hits. Then send the prompts the reader warms up on (seed 200), drained the same way. Stop the server.

   ```bash
   python -m benchmark.kv_store.ttft --url http://127.0.0.1:8101 --tokens 2048 8192 16384 30720 --prompts 2 --label first --out results.jsonl
   python -m benchmark.kv_store.ttft --url http://127.0.0.1:8101 --filler 300
   ```

2. **Reader, store on, a fresh process.** Warm up on one store miss per length (a new seed) and one store hit per length (seed 200), then send the measured prompts: store hits.
3. **Store off, a fresh process.** Warm up, then send the measured prompts (recompute). Push them off the GPU with ten 30K-token prompts of another seed, run two `--filler 1500` requests so their pages reach the host, then send them again: host hits.

Each server's log has one line per request, `matched N of its M pages, K from the host`; pages read from the store count as from the host. Check those lines, not only the times, to tell a store hit from a recompute.
