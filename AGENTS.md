# AGENTS.md — rules for coding agents working in this repository

Repo: `peremartra/fairness-pruning` (paper P0 "Fairness Pruning: Locating Demographic Bias in GLU-MLP Layers via Differential Activations", target TACL). Owner: Pere Martra. Conversation language with the owner: Spanish. Everything written in the repo (code, docs, reports): English.

## Purpose of the research (read before judging any result)
The paper LOCALIZES bias; it does not reduce it. The primary measure is the percentage of changed answers (`flips_pct`): real neurons vs random neurons of the same layer profile, and concentration of the changes in the target category. BBQ bias score and accuracy are secondary. Never frame a result as "bias reduction" or "cost".

## Hard rules
1. **Never commit, push, or change git state.** The owner does it. Do not run `git add/commit/push/reset/checkout/stash`.
2. **Do not modify tracked files** (notebooks, `utils.py`, `analysis/*`, `R1_exp.md`, `README.md`, existing CSVs). New analysis code goes into NEW files; notebook edits are made only in the Colab session copy and reported as an exact diff. If a tracked file must change, ask (see Communication).
3. **Never overwrite or delete anything on Google Drive** (`MyDrive/fair_pruning/...`). If a target file exists and is complete, skip it and say so. If something would overwrite, stop and ask.
4. **Do not invent data, paths or numbers.** If something is missing or ambiguous, ask. A failed assert/check = stop that task and report the full message; never work around it.
5. **Never read, print, copy or upload `.env`.**
6. **GPU:** write "A100" when A100 is meant. Do not substitute another GPU type; if A100 is unavailable, wait or ask. Record `nvidia-smi` name per task.
7. **Do not change any setting that the instructions do not list** (batch sizes, KS, seeds, caps).
8. Do not create files outside `.agent_comms/` and the output locations named in the instructions. No scratch files in the repo root.
9. **Measurement consistency:** before any new measurement or analysis, check what was measured before (metric, aggregation, configuration) so that results are comparable.

## Communication with Claude and the owner (git-ignored folder `.agent_comms/`)
- `INSTRUCTIONS.md` — written by Claude. Read it fully at start and re-read when `ANSWERS.md` changes.
- `STATUS.md` — YOU overwrite it every 15 minutes while working: timestamp (UTC), task id, state (RUNNING / WAITING-ON-ANSWER / BLOCKED / DONE / FAILED), what is running, elapsed time, errors, next step. Also overwrite it immediately when the state changes.
- `QUESTIONS.md` — YOU append one entry per question: `## Q<n> [OPEN] <title>`, date, task id, what you found, the options, your recommendation. Do not edit earlier entries except to mark them `[CLOSED]` once answered.
- `ANSWERS.md` — written by Claude/the owner, `## A<n>` matching `Q<n>`. Never edit it.
- `reports/` — deliverables as files, one per task: `T<id>_<short_name>.md` (what was done, exact commands/cell diffs, files created with Drive path and size, GPU per task, anything unexpected). Reports must be reusable files, not chat text.
- While waiting for an answer, continue with any task that does not depend on it; if none, set state WAITING-ON-ANSWER and poll `ANSWERS.md` every 5 minutes.
- Claude reviews `.agent_comms/` every 30 minutes to detect stalls. A stale `STATUS.md` (older than 20 minutes without WAITING/DONE state) is treated as a stall.
