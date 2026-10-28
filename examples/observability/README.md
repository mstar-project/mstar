# Prometheus and Grafana for M\*

A local Docker Compose stack that scrapes an M\* server's `/metrics` with Prometheus and
shows it in Grafana. The M\* server runs on the host, outside Docker. Every metric family
is described in [`docs/metrics.rst`](../../docs/metrics.rst).

## Run it

1. Start M\* on the host with metrics on (it needs the `metrics` extra,
   `pip install 'mstar[metrics]'`), and check that it answers:

   ```bash
   mstar serve kokoro --enable-metrics        # or: mstar-serve --config ... --enable-metrics
   curl -s localhost:8000/metrics | grep '^mstar_'
   ```

   `--rust-frontend` serves the same `/metrics` on the same port.

2. Start Prometheus and Grafana from the repository root:

   ```bash
   docker compose -f examples/observability/docker-compose.yml up -d
   ```

3. Open the UIs:

   - Prometheus: <http://localhost:9090>. Status, then Target health, should show the
     `mstar` job as up.
   - Grafana: <http://localhost:3000>. Prometheus is the default data source. Log in
     as `admin` with password `admin` (Grafana then asks for a new one), or with the
     value of `GRAFANA_ADMIN_PASSWORD` if it was set in the environment the first time
     the stack started.

4. Stop the stack with `docker compose -f examples/observability/docker-compose.yml down`.
   Prometheus data and Grafana settings stay in the `prometheus-data` and `grafana-data`
   volumes; add `-v` to delete them.

Both UIs listen on `127.0.0.1` only. Nothing in these files is a secret.

## What is where

- `prometheus.yml`: one job, `mstar`, scraping `host.docker.internal:8000/metrics`
  every 5 seconds. Change the target for a server on another port or host.
- `grafana/provisioning/datasources/prometheus.yml`: the Prometheus data source (uid
  `prom`), pointing at the `prometheus` service.
- `grafana/provisioning/dashboards/mstar.yml`: loads every dashboard JSON in
  `grafana/dashboards/` into an `M*` folder.
- `grafana/dashboards/mstar.json`: the `M* serving` dashboard, one row per component
  with a panel for every metric family.

## Reaching the host from the containers

Prometheus runs in a container and M\* on the host, so the scrape target is the host as
seen from the container, `host.docker.internal`. Docker Desktop (macOS, Windows) defines
that name. On Linux, `docker-compose.yml` maps it to the host with
`extra_hosts: host.docker.internal:host-gateway` (Docker 20.10 or later). Two things
can still stop the scrape on Linux:

- **M\* listens on localhost only.** The default `--host 0.0.0.0` accepts connections
  from the containers; `--host 127.0.0.1` does not.
- **A host firewall drops traffic from Docker networks** to the host (some `ufw` or
  `firewalld` setups). The target then shows as down with a connection timeout.

In either case, run both containers on the host network instead, so they reach M\* as
`localhost`. Put this in `examples/observability/docker-compose.host.yml`:

```yaml
services:
  prometheus:
    network_mode: host
    ports: !reset []
    extra_hosts: !reset []
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.path=/prometheus
      - --storage.tsdb.retention.time=7d
      - --web.listen-address=127.0.0.1:9090
  grafana:
    network_mode: host
    ports: !reset []
    environment:
      GF_SERVER_HTTP_ADDR: 127.0.0.1
```

then change the target in `prometheus.yml` to `localhost:8000` and the data source
`url` in `grafana/provisioning/datasources/prometheus.yml` to `http://localhost:9090`,
and start with both files:

```bash
docker compose -f examples/observability/docker-compose.yml \
  -f examples/observability/docker-compose.host.yml up -d
```

Host networking works on Linux only; `!reset` needs Docker Compose 2.24 or later.

## Optional: anonymous viewer

To let anyone who can reach port 3000 see the dashboards read-only without logging in,
uncomment `GF_AUTH_ANONYMOUS_ENABLED` and `GF_AUTH_ANONYMOUS_ORG_ROLE` in
`docker-compose.yml`. It is off by default. `/metrics` itself needs no API key, so keep
both the M\* port and Grafana on a network you trust.
