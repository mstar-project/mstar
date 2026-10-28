# transport microbenchmark

Per-tensor latency and throughput of M\*'s tensor paths across tensor sizes, measured with M\*'s own transport managers, control mesh and wire codec. One process sends, one receives, either on one machine or on two.

```
python -m benchmark.transport.microbench --paths control,tcp,shm,arena --tcp-device lo
```

## What it measures

Each send is timed at the sender from the first byte of work to the ack, so every path includes one return hop.

| Path | What one send does |
|---|---|
| `control` | Serialize the tensor to bytes (a side-stream copy for a GPU tensor), send them in a control message, rebuild the tensor at the receiver, ack. What a tensor costs if it rides in the message itself |
| `tcp`, `rdma` | M\*'s Mooncake path: store, `register_for_send`, an `INPUT_SIGNALS` descriptor, `start_read_tensors` at the receiver, `get_ready_tensors` (which sends the real `TENSOR_RECEIVED` ack), and the sender's dereference and cleanup (unregister) |
| `shm` | M\*'s same-host file transport, same flow (file write, read, unlink) |
| `arena` | M\*'s same-host Rust shared-memory arena (`MSTAR_SHM_ARENA=1`), same flow |
| `tcp-pool`, `rdma-pool` | Mooncake reads from a buffer registered once at startup, copied into per send (`--pool-mode copy`) or produced there (`--pool-mode zerocopy`). Not an M\* path today; next to `tcp` it shows what per-tensor registration costs |

M\* has no CUDA IPC path: same-host tensors go through host shared memory (`shm`, `arena`). `--device cuda:N` puts the sender's tensors on a GPU (and `--receiver-device` the receiver's), so the device copies each path makes are included. RDMA paths are skipped when `/sys/class/infiniband` is empty.

Per path the tool prints p50 and p99 per size, throughput, the p50 of each phase (sender: store, register, send, cleanup or serialize; receiver: read or deserialize), a fit of the p50s to `latency(S) = fixed + S / bandwidth`, and the crossover size between every pair of paths, both from the fits and from the measured points. `--json FILE` writes all of it.

## Options worth knowing

- `--sizes 64:4M:x4` (log-spaced) or `--sizes 64,4K,1M`; `--iters`, `--warmup`, `--max-bytes-per-size`.
- `--inflight N`: N tensors in flight at once, for throughput under concurrency.
- Two machines: run the same command on both with `--role receiver` and `--role sender`, and `--sender-host` / `--receiver-host` set to each machine's reachable address. `--port` is the sender's ZMQ port, the receiver uses the next one; Mooncake picks its own ports.
- `--gpu-busy-ms N` (CUDA devices): each process keeps N ms of spin kernels queued on its default stream, as a worker's next step is while it moves tensors. Anything that waits on that stream, such as a read whose start event is recorded there, or a transport copying GPU memory on the legacy stream (Mooncake TCP, kvcache-ai/Mooncake#4323), then pays for it; side-stream copies do not.
- `--tcp-device lo` for Mooncake TCP over loopback; empty lets Mooncake choose.
- `--interference`: instead of the sweep, bursts of `--bg-streams` messages at `--bg-rate` bursts per second (one decode step's frames), each carrying a payload of each `--bg-sizes` size, with a small probe message every `--probe-interval-ms` and one right behind each burst (timed from before the burst, as a completion sent after a step's frames). The baseline is the same bursts with empty payloads, since a step sends those frames anyway; an idle run gives the scale. Reports the probes' p50 and p99, the extra p99 over the baseline, and the largest burst whose extra p99 stays within `--p99-budget-ms`.

## Use cases

- **Choosing inline thresholds**: the `control` vs `tcp` crossover is the size below which carrying a tensor in the message beats a transfer; `--interference` bounds how many bytes one step may put on the control channel.
- **Comparing Mooncake versions** (for example 0.3.13.post1 against 0.3.14 when it ships): run the same command in two environments and diff the JSON.
- **Validating RDMA or Kubernetes networking**: the two-machine mode between two nodes or pods shows the fixed cost and bandwidth M\* will see.
- **Regression-checking transport changes**: the same sweep before and after a change to `mstar/communication`.

## Reading the numbers

The fixed cost dominates small tensors and the bandwidth large ones, so the fit is weighted to treat them alike. The measured crossover is the first size at which the cheaper path changes; the fitted one can fall between measured sizes or outside the range. Python-side costs (codec, polling) are part of every path, as in a worker. The receiver polls without blocking while reads are in flight, the way a worker loop does. Host copies of 32K elements or more fan out to torch's intra-op thread pool, which on a busy host adds milliseconds; M\* workers keep torch's default pool, so the default here does too, and `--torch-threads 1` takes that out to see the transport alone. With CUDA buffers the receiver process may print `terminate called without an active exception` as it exits, after the results are written. Results on a busy host are noisy: the JSON records the host, devices, mesh and Mooncake version, and the load average is worth noting alongside.

## Related

The `*-pool` copy and zero-copy modes and the GPU-resident buffers follow a cross-node Mooncake throughput test that runs between two nodes in copy, zero-copy and GPUDirect modes. This tool adds per-size latency fitting, the inline crossover and control-channel interference for M\*'s own paths.
