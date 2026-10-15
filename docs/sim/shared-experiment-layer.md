# Shared experiment layer (M\* + vibesim)

Two DES engines, one orchestration contract. Qwen3-TTS on **M\***; LLM presets on **vibesim**. Adapters and manifests were exercised in a local harness (`MSTAR_EXPERIMENT_LAB`); this fork carries **evidence docs** and small sim CLI fixes.

## Layering

```
experiment.manifest.yaml
        │
        ├─ backend: mstar  → simulate / validate + stepdb
        │
        └─ backend: sss    → launcher preset (dry-run or full)
        │
        ▼
run_report.json (v0) + acceptance YAML
```

## Manifest / report (v0)

| Manifest | Report |
|----------|--------|
| `backend`, deployment pointer, workload knobs | `gates` v1–v4, `coverage`, `human_review` |

## vibesim entrypoint (POC branch `feat/mstar-experiment-launcher-poc`)

```bash
export MSTAR_EXPERIMENT_LAB=/path/to/harness   # local scripts + examples
cd /path/to/servingstudiosim
uv run python -m launcher mstar-experiment --help
```

## Non-goals

Qwen3-TTS in vibesim arch; merged DES; cross-backend req/s comparison.
