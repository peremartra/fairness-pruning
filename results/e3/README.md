# results/e3 — Consolidated E3 Experiment Results

Structured, reproducible summary of all E3 neuron-zeroing interventions across models and languages, built directly from the verified per-item analysis outputs on Google Drive.

## Entry Points

- **[`E3_RESULTS.md`](E3_RESULTS.md)** — Master navigable report (Level 1 cross-unit overview, Level 2 unit index, Level 3 diagnostics & hardware noise controls).
- **[`units/`](units/)** — Eight homogeneous unit dossiers (one per model × language).
- **[`tables/`](tables/)** — Compact consolidated CSVs (single source of truth for both the Markdown reports and paper tables/figures) plus [`inputs_manifest.json`](tables/inputs_manifest.json) with SHA-256 hashes of every source file read on Drive.

## Measurement Conventions

- **Primary measure:** changed answers against the unpruned baseline (`changed / n_items`, `flips_pct` over 3-way predictions `{person0, person1, unknown}`, pooled across `ambig + disambig` items per category), contrasting selected neurons with layer-matched random neurons and measuring target-category concentration (`ratio = target / mean(other categories)`, `rank`).
- **Secondary measures:** BBQ/EsBBQ bias score deltas (with 95% item-bootstrap CIs), stereotype transitions (`A→S`, `S→A`), and accuracy transitions (`correct↔incorrect`).

## How to Regenerate

1. Rebuild CSV tables from Google Drive (Colab CPU runtime with Drive mounted; read-only on Drive):
```bash
python analysis/e3/build_e3_tables.py \
  --drive-root /content/drive/MyDrive/fair_pruning \
  --out-dir results/e3/tables
```

2. Re-render all Markdown files locally from `results/e3/tables/` (no Drive required):
```bash
python analysis/e3/render_e3_report.py \
  --tables-dir results/e3/tables \
  --out-dir results/e3
```

Last built from Drive: `2026-11-24T10:16:45+00:00` (`89` source files verified).
