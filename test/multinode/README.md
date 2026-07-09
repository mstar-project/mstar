# Multi-host tests

`sim_two_hosts.sh` runs a complete two-host deployment on one machine: the
head (`mstar-serve`) and a node agent (`mstar-node`) each get their own
control-plane port base and GPU pin, so the BAGEL prefill/decode split in
`bagel_pd_two_hosts_sim.yaml` exercises the real multi-host path — agent
join/launch handshake, TCP control plane, cross-host KV transfer over the
Mooncake TCP transport — without needing a second machine.

```bash
# both workers on GPU 0 (fits one H100/H200):
test/multinode/sim_two_hosts.sh 0

# prefill worker on GPU 0, decode worker on GPU 1:
test/multinode/sim_two_hosts.sh 0 1
```

It serves two pinned-`request_id` prompts (fixed ids give fixed sampling
seeds, so byte hashes are comparable across runs and against a single-host
run of the same config), then kills the remote worker and asserts the next
request fails fast with HTTP 500 instead of hanging.

Requirements: BAGEL weights in the local HF cache (`HF_HUB_OFFLINE=1` is set
by default so nothing is downloaded), and a working Mooncake TCP device —
override with `MSTAR_TEST_TCP_DEVICE` if `0.0.0.0.0` does not suit the
machine.
