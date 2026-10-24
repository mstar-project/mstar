# Grafana dashboards

Grafana loads every `*.json` dashboard in this directory into its `M*` folder
(`../provisioning/dashboards/mstar.yml`) and rereads the directory every 30 seconds.
Panels should use the provisioned data source, uid `prom`:

```json
"datasource": {"type": "prometheus", "uid": "prom"}
```

The metric families are documented in `docs/metrics.rst`.

`mstar.json` is the `M* serving` dashboard: one row per component (requests, request
stages and output, data worker, conductor, engine workers, tensor transport, HTTP
refusals), filtered by its `model` variable. `test/modular/test_grafana_dashboard.py`
checks that it queries only documented families, shows every one of them, uses the
`prom` data source and has no overlapping panels. When exporting it from Grafana after
an edit, leave the option for sharing with other instances off: it replaces the `prom`
data source with an import placeholder.
