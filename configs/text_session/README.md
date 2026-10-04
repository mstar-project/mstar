Session configurations for the `test_text_session` deployment, one per corner of
the session config the live matrix covers. They differ only in the `sessions:`
block; see `docs/sessions.rst` for what each knob means, and
`test/text_session/session_matrix.sbatch` for the scenarios run against each.

| config            | capacity | kv overflow | TTL              | scenarios                                   |
|-------------------|----------|-------------|------------------|---------------------------------------------|
| `keep_error`      | `keep`   | `error`     | 600s idle        | happy path, ws, cap refusal, limits, budget |
| `evict_clear`     | `evict`  | `clear`     | 600s idle        | happy path, LRU eviction, budget            |
| `ttl_idle`        | `keep`   | `clear`     | 20s idle         | idle expiry, refreshed by traffic           |
| `ttl_absolute`    | `keep`   | `clear`     | 30s absolute     | expiry while busy                           |

The TTL configs set a small `max_timeout_s` on purpose: a client asking for more
than the deployment allows is a 400, which the scenarios check.
