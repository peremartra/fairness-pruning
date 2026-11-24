# results/e4 — Canonical Reference E4 Results

**Status: Canonical Reference Dataset (updated 2026-11-24).**
Use this directory for all paper tables, figures, and downstream capability comparisons. All **20/20** real-vs-random benchmark cells are **Matched (A100)** (`216/216` per-item checks passed, `0` problems), superseding earlier L4-mixed E4 outputs on Drive (such as `results/e4/analysis/20261115_123154_caps/`).

## Entry Points

- **[`E4_RESULTS.md`](E4_RESULTS.md)** — Master navigable report (Level 1 cross-model overview, Level 2 unit index, Level 3 WikiText multi-seed controls, MMLU micro-vs-macro reconciliation, and A100 hardware provenance).
- **[`units/`](units/)** — Five homogeneous unit dossiers (`llama_3.1_8b_en`, `qwen3_4b_base_en`, `salamandra_7b_en`, `llama_3.2_3b_en`, `llama_3.2_1b_en`).
- **[`tables/`](tables/)** — Full verified CSV table set (`raw_values.csv`, `deltas.csv`, `e4_capabilities_table.csv`, `e4_inventory.csv`, `e4_wikitext_random_summary.csv`, `mmlu_micro_macro.csv`, `checks.csv`, `provenance.csv`, `inputs.json`).
- **[`e4_capabilities_table.csv`](e4_capabilities_table.csv)** — Mirror of `tables/e4_capabilities_table.csv`: one row per unit, task, and comparison (`value_a`, `value_b`, `delta`, 95% paired-bootstrap CI with `B = 10,000`, `seed = 0`, equivalence margin `±1 pp`, chance exclusion `< 35%`).

## Canonical Drive Inputs (Read-Only)

- Base per-item tables: `MyDrive/fair_pruning/results/e4/<unit>/items/`
- Refreshed 2026-11-23 `llama_3.2_3b_en` A100 logs: `MyDrive/fair_pruning/checkpoints_e3_cap/results/lm_evals/`
- Overlaid 2026-11-24 A100 rerun logs (`rand_gender20_seed0` for `llama_3.1_8b_en`, `salamandra_7b_en`, `qwen3_4b_base_en`): `MyDrive/fair_pruning/checkpoints_e4_a100_rerun/results/lm_evals/`

## How to Regenerate

1. Rebuild CSV tables from Google Drive (read-only on Drive):
```bash
python analysis/e4/build_e4_tables.py \
  --drive-root /content/drive/MyDrive/fair_pruning \
  --provenance analysis/e4/provenance.csv \
  --out-dir results/e4/tables
cp results/e4/tables/e4_capabilities_table.csv results/e4/e4_capabilities_table.csv
```

2. Re-render all Markdown reports locally from `results/e4/tables/` (no Drive required):
```bash
python analysis/e4/render_e4_report.py \
  --tables-dir results/e4/tables \
  --out-dir results/e4
```
