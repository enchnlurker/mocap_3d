Per-clip overrides. `configs/<clip>/clean.json` sets any `CleanConfig` field (core/clean.py) for that clip, e.g.

```json
{"shrink_ratio": 1.8, "lean_fix": true}
```

`core.clean` reads it automatically (or `--config path.json`); `clean_summary.json` records the config used.
