Session configurations for the `test_text_session` deployment, one per corner of
the session config the live matrix covers. They differ only in the `sessions:`
block; see `docs/sessions.rst` for what each knob means, and
`test/text_session/session_matrix.sbatch` for the scenarios run against each.

| config            | capacity | TTL          | scenarios                                   |
|-------------------|----------|--------------|---------------------------------------------|
| `capacity_keep`   | `keep`   | 600s idle    | happy path, ws, cap refusal, limits, budget |
| `capacity_evict`  | `evict`  | 600s idle    | happy path, LRU eviction, budget            |
| `ttl_idle`        | `keep`   | 20s idle     | idle expiry, refreshed by traffic           |
| `ttl_absolute`    | `keep`   | 30s absolute | expiry while busy                           |

All four keep the model's declared `kv` budget of 64 pages (~8k tokens): a
session that outgrows it is dropped and its next request is refused. The TTL
configs set a small `max_timeout_s` on purpose: a client asking for more than
the deployment allows is a 400, which the scenarios check.
