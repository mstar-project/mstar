# Reference results

Metric definitions. ITL: per-request median inter-chunk gap, then the cell
median over requests. L: e2e latency, cell median (the image endpoint is not
streamed). SM and mem: mean of `nvidia-smi dmon -s u` columns 3 and 4 over the
run span. SM0: share of 1 Hz samples reading exactly 0% SM. Absolute latencies
depend on client NUMA placement (see `pinning.txt` in each cell); deltas within
one allocation are the comparison that holds.

## Original numbers, `5eeb434f` vs base `6666c623`

| config | cell | metric | base | `5eeb434f` | delta |
|---|---|---|---|---|---|
| bagel_single_gpu | c1_rep1 | ITL | 6.60 ms | 8.54 ms | +29.4% |
| bagel_single_gpu | c1_rep2 | ITL | 6.54 ms | 8.62 ms | +31.7% |
| bagel_single_gpu | c16 | ITL | 9.62 ms | 11.23 ms | +16.7% |
| bagel_single_gpu | c32 | req/s | 13.49 (200/200) | 0.62 (185/200) | -95.4% |
| bagel_pd_disaggregated | c1 / c6 / c16 | L | 4.738 / 27.199 / 72.560 s | 4.790 / 27.435 / 73.200 s | about +1% |
| bagel_cfg_parallel | c1 | L | 3.028 s | 3.188 s | +5.3% |
| bagel_cfg_parallel | c6 | L | 12.006 s | 24.722 s | +105.9% |
| bagel_cfg_parallel | c16 | L | 31.344 s | 45.804 s | +46.1% |

Chat gap distribution at c=1, pooled over every gap in the cell, computed from
the reference JSONs in this directory (rep1 / rep2):

| arm | share of gaps under 1 ms | p99 gap | median gap after a sub-1 ms gap |
|---|---|---|---|
| base | 0.1% / 0.0% | 11.2 / 8.8 ms | n/a (3 sub-ms gaps in 7620) |
| `5eeb434f` | 21.4% / 21.7% | 20.6 / 20.6 ms | 14.1 ms (mean 11.8 ms) |

## Utilisation, `5eeb434f` vs base

| config | cell | base SM | `5eeb434f` SM | base SM0 | `5eeb434f` SM0 |
|---|---|---|---|---|---|
| bagel_single_gpu | c1_rep1 | 68.3% | 74.5% | 18.2% | 13.9% |
| bagel_single_gpu | c1_rep2 | 70.5% | 73.0% | 18.8% | 16.2% |
| bagel_single_gpu | c16 | 49.1% | 46.9% | 20.8% | 23.1% |
| bagel_single_gpu | c32 | 34.0% | 3.7% | 25.0% | 94.1% |
| bagel_cfg_parallel | c1 | 47.8% | 45.7% | 17.1% | 15.8% |
| bagel_cfg_parallel | c6 | 81.1% | 42.5% | 4.0% | 6.2% |
| bagel_cfg_parallel | c16 | 79.5% | 58.0% | 5.2% | 3.4% |

At cfg c=6 the drop is symmetric across ranks: per-GPU mean SM is
83.2 / 80.3 / 79.9 at base and 43.5 / 41.7 / 42.3 at `5eeb434f`. In
`bagel_pd_disaggregated` both commits look the same per GPU: the prefill GPU
idles 95% of the time and the decode GPU holds 94% SM, so it acts as a control.

## The c=32 stall across commits

`bagel_single_gpu`, 200 requests at c=32, seed 1, fresh boot per row.

| commit | decode capture buckets | succeeded | req/s | wall | longest 0%-SM stretch |
|---|---|---|---|---|---|
| `6666c623` (base) | 1,2,4,8,16,32,64 | 200/200 | 13.49 | 14.8 s | 0 s |
| `5eeb434f` | 1,2,4,8,16 | 185/200 | 0.62 | 300.3 s | 285 s |
| `469105d5` | 1,2,4,8,16 | 200/200 | 10.12 | 19.8 s | 0 s |
| `485beeb7` | 1,2,4,8,16,32,64 | 200/200 | 13.48 | 14.8 s | 1 s |

The 15 lost requests at `5eeb434f` are the contiguous block [16]-[30], the
chunk that overflows the 16-wide decode cap. A 20-request warm-up at c=32 also
loses requests there, so the stall is a race, not a threshold. c=64 at
`485beeb7` completes 200/200 but at 8.95 req/s with mean SM 12.3%, which is a
separate problem.

## Re-measurement at `485beeb7`

The client shared the server's NUMA node in these runs, so compare the deltas,
not the absolutes, with the tables above.

| config | cell | metric | base | `485beeb7` | delta | `5eeb434f` delta |
|---|---|---|---|---|---|---|
| bagel_single_gpu | c1_rep1 | ITL | 6.58 ms | 8.64 ms | +31.2% | +29.4% |
| bagel_single_gpu | c1_rep2 | ITL | 6.61 ms | 8.51 ms | +28.7% | +31.7% |
| bagel_single_gpu | c16 | ITL | 9.12 ms | 11.49 ms | +26.0% | +16.7% |
| bagel_cfg_parallel | c1 | L | 2.404 s | 4.193 s | +74.4% | +5.3% |
| bagel_cfg_parallel | c6 | L | 15.283 s | 21.213 s | +38.8% | +105.9% |

Throughput on the same cells: c16 11.32 to 10.04 req/s (-11.3%), cfg c6 0.372
to 0.267 req/s (-28.2%). Share of sub-1 ms gaps at c=1: 0% to 19.4% (rep1) and
0% to 17.4% (rep2). Warm boot to `/health`, `bagel_single_gpu`: median 167.1 s
at base, 179.3 s at `485beeb7`.

## Autotune sensitivity of the c=1 gap

The reference runs above booted under a persistent Inductor cache, so they
replay whichever fused-MLP tile won autotuning once. Fresh caches re-run the
autotune on every first boot. Measured at `485beeb7` vs base, c1_rep1:

| arm | ITL p50 | new autotune configs written at boot |
|---|---|---|
| base, cached tile | 6.582 ms | 0 |
| base, fresh tune (x2) | 6.695 / 6.588 ms | 54 |
| `485beeb7`, cached tile | 8.636 ms | 0 |
| `485beeb7`, fresh tune (x2) | 7.586 / 7.610 ms | 61 |

On fresh tunes the gap is +14.4%; on the cached tiles it is +31.2%. The
scripts in this package default to a fresh cache per commit, so expect the
fresh-tune number when reproducing finding 2.

## Reference runner outputs in this directory

| file | cell |
|---|---|
| `base_6666c623_c1_rep{1,2}.json` | chat c=1, both reps, at the merge base |
| `pr_5eeb434f_c1_rep{1,2}.json` | chat c=1, both reps, at `5eeb434f` |
| `pr_5eeb434f_c32.json` | the stalled c=32 cell at `5eeb434f` |
| `head_485beeb7_c32.json` | the fixed c=32 cell at `485beeb7` |
