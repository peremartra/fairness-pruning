# AGENTS.md — rules for coding agents working in this repository

Repo: `peremartra/fairness-pruning` (paper P0 "Fairness Pruning: Locating Demographic Bias in GLU-MLP Layers via Differential Activations", target TACL). Owner: Pere Martra. Conversation language with the owner: Spanish. Everything written in the repo (code, docs, reports): English.

## Purpose of the research (read before judging any result)

The paper LOCALIZES bias; it does not reduce it. The primary measure is the percentage of changed answers (`flips_pct`): real neurons vs random neurons of the same layer profile, and concentration of the changes in the target category. BBQ bias score and accuracy are secondary. Never frame a result as "bias reduction" or "cost".

## Canonical Reference Results (use these for all paper tables, figures, and follow-up analyses)

As of **2026-11-24**, all L4-vs-A100 GPU hardware mixing in E3 and E4 has been eliminated. The single canonical reference datasets for the paper are:

- **E3 (Localization & Interventions):** [`results/e3/`](results/e3/) — master report [`results/e3/E3_RESULTS.md`](results/e3/E3_RESULTS.md), unit dossiers [`results/e3/units/`](results/e3/units/), and verified CSV tables [`results/e3/tables/`](results/e3/tables/) (SHA-256 traced in [`results/e3/tables/inputs_manifest.json`](results/e3/tables/inputs_manifest.json)). Hardware status: all 5 EN units + `Salamandra-7B ES` + `Llama-3.1-8B ES` are **100% A100**; `Llama-3.2-3B ES` is **100% L4** (homogeneous baseline + intervened).
  - Canonical updated Drive folders (superseding earlier L4-mixed folders, which remain on Drive for provenance only):
    - `MyDrive/fair_pruning/results/e3/llama-3.2-3b_en/analysis/20261124_a100_items_3b_random/` (using A100 `rand_religion20_seed0..1` from `checkpoints_e2_a100_rerun/` and `items_religion_a100_seed01/`, plus A100 `rand_gender20_seed0` from `items_random_seed0_a100/`).
    - `MyDrive/fair_pruning/results/esbbq_zeroed_llama8b/analysis/20261124_141737_items_a100base/` (recomputed against the A100 K3 baseline).
- **E4 (General Capabilities):** [`results/e4/`](results/e4/) — master report [`results/e4/E4_RESULTS.md`](results/e4/E4_RESULTS.md), unit dossiers [`results/e4/units/`](results/e4/units/), consolidated CSV tables [`results/e4/tables/`](results/e4/tables/), and [`results/e4/e4_capabilities_table.csv`](results/e4/e4_capabilities_table.csv). Hardware status: **20/20** real-vs-random cells are **Matched (A100)** (`216/216` extraction checks with 0 problems; overlays A100 reruns in `MyDrive/fair_pruning/checkpoints_e4_a100_rerun/` and the 2026-11-23 `llama_3.2_3b_en` A100 reruns). Supersedes earlier L4-mixed E4 analysis folders (e.g., `results/e4/analysis/20261115_123154_caps/`).

## Hard rules

1. **Never commit, push, or change git state.** The owner does it. Do not run `git add/commit/push/reset/checkout/stash`. (*unless explicitly requested by the owner)*

1. **Do not modify tracked files** (notebooks, `utils.py`, `analysis/*`, `R1_exp.md`, `README.md`, existing CSVs). New analysis code goes into NEW files; notebook edits are made only in the Colab session copy and reported as an exact diff. If a tracked file must change, ask (see Communication).
2. **Never overwrite or delete anything on Google Drive** (`MyDrive/fair_pruning/...`). If a target file exists and is complete, skip it and say so. If something would overwrite, stop and ask.
3. **Do not invent data, paths or numbers.** If something is missing or ambiguous, ask. A failed assert/check = stop that task and report the full message; never work around it.
4. **Never read, print, copy or upload `.env`.**
5. **GPU:** write "A100" when A100 is meant. Do not substitute another GPU type; if A100 is unavailable, wait or ask. Record `nvidia-smi` name per task.
6. **Do not change any setting that the instructions do not list** (batch sizes, KS, seeds, caps).
7. Do not create files outside `.agent_comms/` and the output locations named in the instructions. No scratch files in the repo root.
8. **Measurement consistency:** before any new measurement or analysis, check what was measured before (metric, aggregation, configuration) so that results are comparable.

## Communication with Claude and the owner (git-ignored folder `.agent_comms/`)

- `INSTRUCTIONS.md` — written by Claude. Read it fully at start and re-read when `ANSWERS.md` changes.
- `STATUS.md` — YOU overwrite it every 15 minutes while working: timestamp (UTC), task id, state (RUNNING / WAITING-ON-ANSWER / BLOCKED / DONE / FAILED), what is running, elapsed time, errors, next step. Also overwrite it immediately when the state changes.
- `QUESTIONS.md` — YOU append one entry per question: `## Q<n> [OPEN] <title>`, date, task id, what you found, the options, your recommendation. Do not edit earlier entries except to mark them `[CLOSED]` once answered.
- `ANSWERS.md` — written by Claude/the owner, `## A<n>` matching `Q<n>`. Never edit it.
- `reports/` — deliverables as files, one per task: `T<id>_<short_name>.md` (what was done, exact commands/cell diffs, files created with Drive path and size, GPU per task, anything unexpected). Reports must be reusable files, not chat text.
- While waiting for an answer, continue with any task that does not depend on it; if none, set state WAITING-ON-ANSWER and poll `ANSWERS.md` every 5 minutes.
- Claude reviews `.agent_comms/` every 30 minutes to detect stalls. A stale `STATUS.md` (older than 20 minutes without WAITING/DONE state) is treated as a stall.