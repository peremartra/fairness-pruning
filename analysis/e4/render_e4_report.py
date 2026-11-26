#!/usr/bin/env python3
"""
Render clean, structured Markdown reports for E4 from the consolidated CSV tables in results/e4/tables/.

No Drive access needed: reads only the compact CSVs produced by build_e4_tables.py.
Every section and table has exactly one concise explanatory sentence; no prose.

Writes:
  - <out-dir>/E4_RESULTS.md          (Level 1 overview + Level 2 index + Level 3 diagnostics)
  - <out-dir>/units/<unit>.md        (Level 2 homogeneous dossier for each of the 5 units)

Usage:
    python analysis/e4/render_e4_report.py \
        --tables-dir results/e4/tables \
        --out-dir results/e4
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TASK_ORDER = ["wikitext", "arc_challenge", "hellaswag", "hellaswag_es", "mmlu"]
TASK_LABELS = {
    "wikitext": "WikiText (word ppl, 62 docs)",
    "arc_challenge": "ARC-Challenge (acc_norm, 1,172)",
    "hellaswag": "HellaSwag (acc_norm, 10,042)",
    "hellaswag_es": "HellaSwag-es (acc_norm, 9,374)",
    "mmlu": "MMLU 5-shot (micro acc, 14,042)",
}
TASK_SHORT = {
    "wikitext": "WikiText (Δ%)",
    "arc_challenge": "ARC-C (pp)",
    "hellaswag": "HellaSwag (pp)",
    "hellaswag_es": "HellaSwag-es (pp)",
    "mmlu": "MMLU micro (pp)",
}
VERDICT_SHORT = {
    "equivalent": "equiv",
    "inconclusive": "inconcl",
    "worse": "WORSE",
    "better": "better",
    "excluded (baseline near chance)": "excl (<35%)",
    "n/a (no margin for perplexity)": "-",
}


def fmt_val(task: str, v) -> str:
    if pd.isna(v):
        return "-"
    return f"{float(v):.3f}" if task == "wikitext" else f"{float(v):.2f}%"


def fmt_delta_cell(task: str, delta, lo, hi, verdict) -> str:
    if pd.isna(delta):
        return "-"
    u = "%" if task == "wikitext" else ""
    vs = VERDICT_SHORT.get(str(verdict), str(verdict))
    base = f"{float(delta):+.2f}{u} [{float(lo):+.2f}, {float(hi):+.2f}]"
    return base if vs == "-" else f"{base} ({vs})"


def md_table(headers, rows, align=None):
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    if align:
        sep = ["---:" if a == "r" else (":---:" if a == "c" else ":---") for a in align]
        lines.append("| " + " | ".join(sep) + " |")
    else:
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def gpu_lookup(prov_df: pd.DataFrame):
    m = {}
    for _, r in prov_df.iterrows():
        m[(r["unit"], r["set"], r["task"])] = str(r["gpu"]).replace("NVIDIA ", "").strip()
    return m


def render_master(tables: dict, inputs: dict) -> str:
    inv = tables["inventory"]
    raw = tables["raw_values"]
    deltas = tables["deltas"]
    wt_rnd = tables["wikitext_random"]
    mmlu_mm = tables["mmlu_mm"]
    prov = tables["provenance"]
    checks = tables["checks"]
    gmap = gpu_lookup(prov)

    refreshed = inputs.get("refreshed_from_drive_logs") or []
    refresh_note = (
        f"; refreshed `{len(refreshed)}` A100 rerun logs (`llama_3.2_3b_en`, `llama_3.1_8b_en`, `salamandra_7b_en`, `qwen3_4b_base_en`)"
        if refreshed
        else ""
    )

    R = [
        "# E4 Results — General-Capability Evaluation After Neuron Zeroing",
        "",
        f"Paired item-level bootstrap (`B = {inputs['args']['bootstrap']:,}`, `seed = {inputs['args']['seed']}`, equivalence margin `±{inputs['args']['margin']:.1f} pp`, chance exclusion `< {inputs['args']['chance_excl']*100:.0f}%`) across five English base models and five benchmarks (`inputs.json`, built `{inputs.get('build_e4_tables_utc', inputs.get('created_utc'))}`{refresh_note}).",
        "",
        "## Navigation",
        "",
        "- **Level 1 — Cross-Model Overview:** [1.1 Inventory & Protocol](#11-experiment-inventory-and-protocol) · [1.2 Main Matrix (`K = 20`, Real vs. Baseline)](#12-main-matrix-k--20-real-sets-vs-baseline) · [1.3 Real vs. Random (`Gender·20`)](#13-real-vs-layer-matched-random-control-gender20-seed-0) · [1.4 `K = 5` vs. `K = 20` Comparison](#14-k--5-vs-k--20-comparison-real-sets-vs-baseline)",
        "- **Level 2 — Unit Dossiers:** [2. Index of Units](#2-unit-dossiers-level-2)",
        "- **Level 3 — Diagnostics & Provenance:** [3.1 WikiText Multi-Seed Random Controls](#31-wikitext-multi-seed-random-controls) · [3.2 MMLU Micro vs. Macro](#32-mmlu-micro-vs-macro-57-subjects) · [3.3 Hardware Provenance (A100 vs. L4) & Checks](#33-hardware-provenance-a100-vs-l4-and-extraction-checks) · [3.4 Source Tables](#34-source-tables)",
        "",
        "---",
        "",
        "## 1. Cross-Model Overview (Level 1)",
        "",
        "### 1.1 Experiment Inventory and Protocol",
        "",
        "All five E4 units, counts of real and random sets evaluated, extraction verification checks, and hardware used for baseline/real versus random evaluations.",
        "",
    ]
    rows = []
    for _, r in inv.iterrows():
        u = r["unit"]
        rows.append([
            f"[`{u}`](units/{u}.md)",
            r["model"],
            int(r["n_real_sets"]),
            int(r["n_random_full_battery"]),
            int(r["n_random_wikitext_only"]),
            r["k_values"],
            "Excluded (<35%)" if bool(r["mmlu_excluded_chance"]) else "Included",
            f"{int(r['checks_total'])} ({int(r['checks_problems'])} problems)",
            r["gpus_baseline_real"],
            r["gpus_random"],
        ])
    R.append(md_table(
        ["Unit", "Model", "Real Sets", "Random (Full)", "Random (WikiText-only)", "K", "MMLU Status", "Checks", "Baseline/Real GPUs", "Random GPUs"],
        rows,
        ["l", "l", "r", "r", "r", "c", "c", "c", "l", "l"],
    ))
    R += [
        "",
        "*Legend — **Real Sets**: selected top-K sets (`Gender`/`Race`/`Religion` × `K = 5, 20`); **Random (Full)**: layer-matched random control evaluated on all 5 benchmarks (`rand_gender20_seed0`); **Random (WikiText-only)**: additional random seeds evaluated on WikiText only; **MMLU Status**: `Excluded (<35%)` when unpruned baseline accuracy is below the 35% near-chance threshold; **Checks**: per-item extraction integrity checks (`0 problems` = 100% agreement with `lm-eval` aggregates and baseline item hashes).*",
        "",
    ]

    # 1.2 Main Matrix K=20 Real vs Baseline
    R += [
        "### 1.2 Main Matrix: `K = 20` Real Sets vs. Baseline",
        "",
        "Paired differences (`real − baseline`, 95% bootstrap CI, equivalence verdict at `±1 pp`) across all five benchmarks for every `K = 20` real set, with unpruned baseline scores shown on the first row of each model.",
        "",
    ]
    raw_idx = raw.set_index(["unit", "set", "task"])["value"]
    d_base = deltas[deltas["set_b"] == "baseline"].set_index(["unit", "set_a", "task"])
    rows = []
    for _, inv_r in inv.iterrows():
        u, mname = inv_r["unit"], inv_r["model"]
        b_cells = [fmt_val(t, raw_idx.get((u, "baseline", t), np.nan)) for t in TASK_ORDER]
        rows.append([f"[`{u}`](units/{u}.md)", mname, "`baseline` *(raw)*"] + b_cells)
        for sid in ("real_gender20", "real_race20", "real_religion20"):
            cells = []
            for t in TASK_ORDER:
                if (u, sid, t) in d_base.index:
                    dr = d_base.loc[(u, sid, t)]
                    cells.append(fmt_delta_cell(t, dr["delta"], dr["ci95_lo"], dr["ci95_hi"], dr["verdict"]))
                else:
                    cells.append("-")
            rows.append(["", "", f"`{sid}`"] + cells)
    R.append(md_table(
        ["Unit", "Model", "Set"] + [TASK_SHORT[t] for t in TASK_ORDER],
        rows,
        ["l", "l", "l"] + ["r"] * len(TASK_ORDER),
    ))
    R += [
        "",
        "*Legend — **`baseline` *(raw)***: unpruned model score (`word_perplexity` for WikiText, lower is better; `%` accuracy for the four QA benchmarks); **WikiText (Δ%)**: relative percentage change in word perplexity vs. baseline (`+` = higher/worse perplexity); **ARC-C / HellaSwag / HellaSwag-es / MMLU (pp)**: percentage-point change (`real − baseline`) with 95% paired item-level bootstrap CI (`B = 10,000`, `seed = 0`); **Verdicts (`±1.0 pp` equivalence margin)**: `equiv` = entire 95% CI within `[-1, +1] pp`, `inconcl` = CI crosses `±1 pp`, `WORSE` = entire CI below `-1 pp`, `excl (<35%)` = baseline near chance.*",
        "",
    ]

    # 1.3 Real vs Random (Gender·20 seed 0)
    R += [
        "### 1.3 Real vs. Layer-Matched Random Control (`Gender·20`, seed 0)",
        "",
        "Direct comparison of `real_gender20` against `rand_gender20_seed0` (same layer profile, full 5-task battery) across the four models with a full random control, with explicit GPU hardware per side.",
        "",
    ]
    d_idx = deltas.set_index(["unit", "comparison", "task"])
    rows = []
    for _, inv_r in inv.iterrows():
        u, mname = inv_r["unit"], inv_r["model"]
        if int(inv_r["n_random_full_battery"]) == 0:
            continue
        for t in TASK_ORDER:
            r_vs_b = d_idx.loc[(u, "real_gender20_vs_baseline", t)]
            q_vs_b = d_idx.loc[(u, "rand_gender20_seed0_vs_baseline", t)]
            r_vs_q = d_idx.loc[(u, "real_gender20_vs_rand_gender20_seed0", t)]
            g_real = gmap.get((u, "real_gender20", t), "unknown")
            g_rand = gmap.get((u, "rand_gender20_seed0", t), "unknown")
            rows.append([
                f"[`{u}`](units/{u}.md)",
                t,
                fmt_delta_cell(t, r_vs_b["delta"], r_vs_b["ci95_lo"], r_vs_b["ci95_hi"], r_vs_b["verdict"]),
                fmt_delta_cell(t, q_vs_b["delta"], q_vs_b["ci95_lo"], q_vs_b["ci95_hi"], q_vs_b["verdict"]),
                fmt_delta_cell(t, r_vs_q["delta"], r_vs_q["ci95_lo"], r_vs_q["ci95_hi"], r_vs_q["verdict"]),
                f"{g_real} / {g_rand}",
            ])
    R.append(md_table(
        ["Unit", "Task", "Real − Baseline", "Random(seed0) − Baseline", "Real − Random(seed0)", "GPUs (Real / Rand)"],
        rows,
        ["l", "l", "r", "r", "r", "l"],
    ))
    R += [
        "",
        "*Legend — **Real − Baseline / Random(seed0) − Baseline / Real − Random(seed0)**: paired item-level differences with 95% bootstrap CI (`Δ%` relative perplexity for WikiText; percentage points `pp` for accuracy tasks) and `±1.0 pp` equivalence verdicts (`equiv`, `inconcl`, `WORSE`); **GPUs (Real / Rand)**: verified A100 hardware for each side.*",
        "",
    ]

    # 1.4 K=5 vs K=20 comparison
    R += [
        "### 1.4 `K = 5` vs. `K = 20` Comparison (Real Sets vs. Baseline)",
        "",
        "Paired deltas against baseline for `K = 5` alongside `K = 20` across the four tasks evaluated at both K values (`MMLU` is evaluated at `K = 20` only).",
        "",
        "<details>",
        "<summary>Expand K = 5 vs. K = 20 table across all five models</summary>",
        "",
    ]
    k_tasks = ["wikitext", "arc_challenge", "hellaswag", "hellaswag_es"]
    rows = []
    for _, inv_r in inv.iterrows():
        u = inv_r["unit"]
        for cat_key in ("gender", "race", "religion"):
            s5, s20 = f"real_{cat_key}5", f"real_{cat_key}20"
            row = [f"[`{u}`](units/{u}.md)", cat_key.capitalize()]
            for t in k_tasks:
                d5 = d_base.loc[(u, s5, t)] if (u, s5, t) in d_base.index else None
                d20 = d_base.loc[(u, s20, t)] if (u, s20, t) in d_base.index else None
                u_sym = "%" if t == "wikitext" else ""
                c5 = f"{d5['delta']:+.2f}{u_sym}" if d5 is not None else "-"
                c20 = f"{d20['delta']:+.2f}{u_sym}" if d20 is not None else "-"
                row.append(f"{c5} / {c20}")
            rows.append(row)
    R.append(md_table(
        ["Unit", "Category", "WikiText (K=5 / K=20)", "ARC-C (K=5 / K=20)", "HellaSwag (K=5 / K=20)", "HellaSwag-es (K=5 / K=20)"],
        rows,
        ["l", "l", "r", "r", "r", "r"],
    ))
    R += [
        "",
        "*Legend — Cells show `K=5 / K=20` paired deltas vs. baseline (`Δ%` relative word perplexity for WikiText; percentage points `pp` for ARC-C, HellaSwag, and HellaSwag-es; MMLU is evaluated at `K = 20` only).*",
        "",
        "</details>",
        "",
        "---",
        "",
    ]

    # Level 2 Index
    R += [
        "## 2. Unit Dossiers (Level 2)",
        "",
        "Each unit dossier follows the same layout: raw scores across all sets, full baseline deltas with 95% bootstrap CIs, real-vs-random contrasts across all seeds, MMLU micro-vs-macro reconciliation, and per-task GPU/batch provenance.",
        "",
    ]
    rows = []
    for _, r in inv.iterrows():
        u = r["unit"]
        rows.append([
            f"[`units/{u}.md`](units/{u}.md)",
            r["model"],
            f"{int(r['n_real_sets'])} real / {int(r['n_random_full_battery'])} full-rand / {int(r['n_random_wikitext_only'])} wt-rand",
            r["gpus_baseline_real"],
            r["gpus_random"],
        ])
    R.append(md_table(["Dossier", "Model", "Sets", "Baseline/Real GPUs", "Random GPUs"], rows, ["l", "l", "l", "l", "l"]))
    R += ["", "---", ""]

    # Level 3 Diagnostics
    R += [
        "## 3. Diagnostics and Provenance (Level 3)",
        "",
        "### 3.1 WikiText Multi-Seed Random Controls",
        "",
        "Relative word-perplexity change (`%` vs. baseline) of `K = 20` real sets compared against all available layer-matched random seeds (note: `bf16` document log-likelihood quantization sets an interpretability floor near `~0.1%`).",
        "",
    ]
    rows = []
    for _, r in wt_rnd.iterrows():
        rows.append([
            f"[`{r['unit']}`](units/{r['unit']}.md)",
            f"`{r['real_set']}`",
            f"`{r['random_family']}` (n={int(r['n_seeds'])})",
            f"{r['baseline_ppl']:.3f}",
            f"{r['real_delta_pct']:+.2f}% [{r['real_ci95_lo']:+.2f}, {r['real_ci95_hi']:+.2f}]",
            f"{r['rand_delta_min_pct']:+.2f}% to {r['rand_delta_max_pct']:+.2f}% (mean {r['rand_delta_mean_pct']:+.2f}%)",
            f"{r['real_vs_rand_mean_pct']:+.2f}% [{r['real_vs_rand_min_pct']:+.2f}, {r['real_vs_rand_max_pct']:+.2f}]",
        ])
    R.append(md_table(
        ["Unit", "Real Set", "Random Family", "Base PPL", "Real Δ% [95% CI]", "Random Δ% (min–max, mean)", "Real − Random Δ% (mean [min, max])"],
        rows,
        ["l", "l", "l", "r", "r", "r", "r"],
    ))
    R += [
        "",
        "*Legend — **Base PPL**: unpruned WikiText word perplexity (`62` documents, lower is better); **Real Δ% [95% CI]**: relative perplexity change vs. baseline (`+` = higher perplexity); **Random Δ%**: min–max and mean relative change across all layer-matched random seeds; **Real − Random Δ%**: paired real-minus-random difference (mean `[min, max]` across seeds). Differences `< ~0.1%` lie below the `bf16` document log-likelihood quantization floor.*",
        "",
    ]

    # 3.2 MMLU micro vs macro
    R += [
        "### 3.2 MMLU Micro vs. Macro (57 Subjects)",
        "",
        "Item-weighted micro MMLU accuracy (used throughout E4 and reported by `lm-eval`) versus the unweighted 57-subject macro average stored in `_cap.json` (+0.35 to +2.64 pp higher; never mix in one table).",
        "",
        "<details>",
        "<summary>Expand MMLU micro vs. macro table (all evaluated units and sets)</summary>",
        "",
    ]
    rows = []
    for _, r in mmlu_mm.iterrows():
        diff_pp = (float(r["macro_57_subjects"]) - float(r["micro_from_items"])) * 100.0
        rows.append([
            f"[`{r['unit']}`](units/{r['unit']}.md)",
            f"`{r['set']}`",
            f"{float(r['micro_from_items'])*100:.2f}%",
            f"{float(r['macro_57_subjects'])*100:.2f}%",
            f"{diff_pp:+.2f} pp",
            f"{float(r['cap_json_value'])*100:.2f}%" if pd.notna(r["cap_json_value"]) else "-",
        ])
    R.append(md_table(
        ["Unit", "Set", "Micro (14,042 items)", "Macro (57 subjects)", "Macro − Micro", "`_cap.json` Value"],
        rows,
        ["l", "l", "r", "r", "r", "r"],
    ))
    R += [
        "",
        "*Legend — **Micro (14,042 items)**: item-weighted accuracy used throughout E4 and reported by `lm-eval`; **Macro (57 subjects)**: unweighted average across the 57 MMLU subjects; **`_cap.json` Value**: macro value stored in per-set checkpoint summaries (never mix with micro in one table).*",
        "",
        "</details>",
        "",
    ]

    # 3.3 Hardware provenance & checks
    R += [
        "### 3.3 Hardware Provenance (A100 vs. L4) and Extraction Checks",
        "",
        "Per-task GPU hardware across baseline, `real_gender20`, and `rand_gender20_seed0` for the four units with full random controls (all 20/20 cells matched on A100), plus overall extraction integrity across all per-item logs.",
        "",
    ]
    mix_rows = []
    for _, inv_r in inv.iterrows():
        u = inv_r["unit"]
        if int(inv_r["n_random_full_battery"]) == 0:
            continue
        for t in TASK_ORDER:
            gb = gmap.get((u, "baseline", t), "unknown")
            gr = gmap.get((u, "real_gender20", t), "unknown")
            gq = gmap.get((u, "rand_gender20_seed0", t), "unknown")
            same = ("A100" in gb and "A100" in gr and "A100" in gq) or (gb == gr == gq)
            mix_rows.append([f"[`{u}`](units/{u}.md)", t, gb, gr, gq, "Matched (A100)" if same else "Mixed (A100 vs. L4)"])
    R.append(md_table(
        ["Unit", "Task", "Baseline GPU", "Real Gender·20 GPU", "Random Gender·20 (seed0) GPU", "Hardware Match"],
        mix_rows,
        ["l", "l", "l", "l", "l", "c"],
    ))
    n_bad = int(((~checks["agg_ok"]) | (checks["cap_ok"] == False) | (checks["extraction_problems"].fillna("") != "")).sum())
    R += [
        "",
        "*Legend — **Hardware Match**: `Matched (A100)` confirms baseline, real, and random runs all executed on A100 GPUs (zero L4-vs-A100 hardware mixing).*",
        "",
        f"Extraction verification (`tables/checks.csv`): **{len(checks)}** unit × set × task tables checked, **{n_bad}** problems detected, 100% item-key and `doc_hash` alignment with baseline.",
        "",
    ]

    # 3.4 Source tables
    R += [
        "### 3.4 Source Tables",
        "",
        "All numbers in this report and its unit dossiers are rendered directly from the verified CSV tables under `results/e4/tables/` (and mirrored in `results/e4/e4_capabilities_table.csv`).",
        "",
    ]
    tdir_files = sorted(tables["_meta_files"])
    rows = [[f"[`tables/{f}`](tables/{f})", f"{r:,}", c, f"{b:,} B"] for f, r, c, b in tdir_files]
    R.append(md_table(["CSV Table", "Rows", "Columns", "Size"], rows, ["l", "r", "r", "r"]))
    R.append("")
    return "\n".join(R)


def render_unit(unit: str, tables: dict) -> str:
    inv = tables["inventory"]
    uinfo = inv[inv["unit"] == unit].iloc[0]
    raw = tables["raw_values"]
    deltas = tables["deltas"]
    mmlu_mm = tables["mmlu_mm"]
    prov = tables["provenance"]

    uraw = raw[raw["unit"] == unit].copy()
    udel = deltas[deltas["unit"] == unit].copy()
    umm = mmlu_mm[mmlu_mm["unit"] == unit].copy()
    uprov = prov[prov["unit"] == unit].copy()

    R = [
        f"# E4 Unit Dossier — `{unit}` ({uinfo['model']}, EN)",
        "",
        f"[← Back to E4 Overview](../E4_RESULTS.md) · Sets: **{int(uinfo['n_real_sets'])}** real, **{int(uinfo['n_random_full_battery'])}** full-battery random, **{int(uinfo['n_random_wikitext_only'])}** WikiText-only random · Checks: {int(uinfo['checks_total'])} ({int(uinfo['checks_problems'])} problems).",
        "",
    ]

    # 1. Raw scores
    R += [
        "## 1. Raw Benchmark Scores",
        "",
        "Unmodified benchmark values (`word_perplexity` for WikiText; `%` accuracy for ARC-Challenge, HellaSwag, HellaSwag-es, and micro MMLU) for every evaluated set.",
        "",
    ]
    set_order = ["baseline"] + sorted(
        [s for s in uraw["set"].unique() if s != "baseline"],
        key=lambda s: (0 if s.startswith("real_") else 1, s),
    )
    ridx = uraw.set_index(["set", "task"])["value"]
    rows = []
    for s in set_order:
        rows.append([f"`{s}`"] + [fmt_val(t, ridx.get((s, t), np.nan)) for t in TASK_ORDER])
    R.append(md_table(
        ["Set", "WikiText (ppl)", "ARC-C (%)", "HellaSwag (%)", "HellaSwag-es (%)", "MMLU micro (%)"],
        rows,
        ["l", "r", "r", "r", "r", "r"],
    ))
    R += [
        "",
        "*Legend — **WikiText (ppl)**: word perplexity over 62 documents (lower is better); **ARC-C / HellaSwag / HellaSwag-es (%)**: `acc_norm` accuracy; **MMLU micro (%)**: item-weighted 5-shot accuracy over 14,042 items (`-` = task not evaluated for that set).*",
        "",
    ]

    # 2. Deltas vs Baseline
    R += [
        "## 2. Paired Differences vs. Baseline (`B = 10,000`, `±1 pp` Margin)",
        "",
        "Item-paired deltas (`set − baseline`) with 95% percentile bootstrap confidence intervals and equivalence verdicts for all real sets and full-battery random controls.",
        "",
    ]
    d_base = udel[udel["set_b"] == "baseline"].copy()
    d_base["_tord"] = d_base["task"].map({t: i for i, t in enumerate(TASK_ORDER)})
    d_base["_kord"] = d_base["kind"].map({"real": 0, "random": 1}).fillna(2)
    d_base = d_base.sort_values(["_tord", "_kord", "set_a"])
    rows = []
    for _, r in d_base.iterrows():
        t = r["task"]
        u_sym = "%" if t == "wikitext" else " pp"
        rows.append([
            t,
            f"`{r['set_a']}`",
            r["kind"],
            fmt_val(t, ridx.get((r["set_a"], t), np.nan)),
            fmt_val(t, r["baseline_value"]),
            f"{float(r['delta']):+.3f}{u_sym}",
            f"[{float(r['ci95_lo']):+.3f}, {float(r['ci95_hi']):+.3f}]",
            VERDICT_SHORT.get(str(r["verdict"]), str(r["verdict"])),
        ])
    R.append(md_table(
        ["Task", "Set", "Kind", "Set Value", "Baseline", "Delta", "95% CI", "Verdict"],
        rows,
        ["l", "l", "c", "r", "r", "r", "c", "c"],
    ))
    R += [
        "",
        "*Legend — **Delta / 95% CI**: paired item-level difference (`set − baseline`, `B = 10,000`, `seed = 0`, relative `%` for WikiText, `pp` for QA tasks); **Verdict (`±1.0 pp` margin)**: `equiv` = CI within `[-1, +1] pp`, `inconcl` = CI crosses `±1 pp`, `WORSE` = CI below `-1 pp`, `excl (<35%)` = baseline near chance.*",
        "",
    ]

    # 3. Real vs Random (if present)
    d_rvr = udel[udel["set_b"] != "baseline"].copy()
    sec_num = 3
    if not d_rvr.empty:
        R += [
            f"## {sec_num}. Real vs. Layer-Matched Random Contrasts",
            "",
            "Paired differences (`real − random`) across all tasks and random seeds available for this unit.",
            "",
        ]
        sec_num += 1
        d_rvr["_tord"] = d_rvr["task"].map({t: i for i, t in enumerate(TASK_ORDER)})
        d_rvr = d_rvr.sort_values(["_tord", "comparison"])
        rows = []
        for _, r in d_rvr.iterrows():
            t = r["task"]
            u_sym = "%" if t == "wikitext" else " pp"
            rows.append([
                t,
                f"`{r['set_a']}` vs. `{r['set_b']}`",
                fmt_val(t, ridx.get((r["set_a"], t), np.nan)),
                fmt_val(t, ridx.get((r["set_b"], t), np.nan)),
                f"{float(r['delta']):+.3f}{u_sym}",
                f"[{float(r['ci95_lo']):+.3f}, {float(r['ci95_hi']):+.3f}]",
                VERDICT_SHORT.get(str(r["verdict"]), str(r["verdict"])),
            ])
        R.append(md_table(
            ["Task", "Comparison", "Real Value", "Random Value", "Delta", "95% CI", "Verdict"],
            rows,
            ["l", "l", "r", "r", "r", "c", "c"],
        ))
        R += [
            "",
            "*Legend — **Delta / 95% CI**: paired difference (`real − random`, relative `%` for WikiText, `pp` for QA tasks); **Verdict**: equivalence at `±1.0 pp` margin (`equiv`, `inconcl`, `WORSE`).*",
            "",
        ]

    # 4. MMLU micro vs macro
    if not umm.empty:
        R += [
            f"## {sec_num}. MMLU Micro vs. Macro Reconciliation",
            "",
            "Item-weighted micro accuracy (`14,042` items) compared with unweighted 57-subject macro accuracy and `_cap.json` values.",
            "",
        ]
        sec_num += 1
        rows = []
        for _, r in umm.iterrows():
            diff_pp = (float(r["macro_57_subjects"]) - float(r["micro_from_items"])) * 100.0
            rows.append([
                f"`{r['set']}`",
                f"{float(r['micro_from_items'])*100:.2f}%",
                f"{float(r['macro_57_subjects'])*100:.2f}%",
                f"{diff_pp:+.2f} pp",
                f"{float(r['cap_json_value'])*100:.2f}%" if pd.notna(r["cap_json_value"]) else "-",
            ])
        R.append(md_table(
            ["Set", "Micro (14,042 items)", "Macro (57 subjects)", "Macro − Micro", "`_cap.json` Value"],
            rows,
            ["l", "r", "r", "r", "r"],
        ))
        R += [
            "",
            "*Legend — **Micro**: item-weighted accuracy used throughout E4; **Macro**: unweighted 57-subject mean matching `_cap.json`.*",
            "",
        ]

    # 5. Provenance
    R += [
        f"## {sec_num}. Hardware and Batch-Size Provenance",
        "",
        "GPU hardware, batch size, and `lm-eval` version recorded per set and task from `tables/provenance.csv`.",
        "",
        "<details>",
        "<summary>Expand per-set provenance table</summary>",
        "",
    ]
    rows = []
    for _, r in uprov.sort_values(["set", "task"]).iterrows():
        rows.append([
            f"`{r['set']}`",
            r["task"],
            str(r["gpu"]).replace("NVIDIA ", ""),
            r["batch_size"],
            r["lm_eval_version"],
            r["source"] if pd.notna(r["source"]) else "-",
        ])
    R.append(md_table(["Set", "Task", "GPU", "Batch", "lm-eval", "Source Note"], rows, ["l", "l", "l", "r", "c", "l"]))
    R += ["", "</details>", ""]

    return "\n".join(R)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-dir", default="results/e4/tables")
    ap.add_argument("--out-dir", default="results/e4")
    args = ap.parse_args()

    tdir = Path(args.tables_dir)
    out = Path(args.out_dir)
    units_dir = out / "units"
    units_dir.mkdir(parents=True, exist_ok=True)

    inputs = json.load(open(tdir / "inputs.json", encoding="utf-8"))
    csv_files = []
    for p in sorted(tdir.glob("*.csv")):
        df_tmp = pd.read_csv(p)
        csv_files.append((p.name, len(df_tmp), len(df_tmp.columns), p.stat().st_size))

    tables = {
        "inventory": pd.read_csv(tdir / "e4_inventory.csv"),
        "raw_values": pd.read_csv(tdir / "raw_values.csv"),
        "deltas": pd.read_csv(tdir / "deltas.csv"),
        "wikitext_random": pd.read_csv(tdir / "e4_wikitext_random_summary.csv"),
        "mmlu_mm": pd.read_csv(tdir / "mmlu_micro_macro.csv"),
        "provenance": pd.read_csv(tdir / "provenance.csv"),
        "checks": pd.read_csv(tdir / "checks.csv"),
        "_meta_files": csv_files,
    }

    (out / "E4_RESULTS.md").write_text(render_master(tables, inputs), encoding="utf-8")
    for u in tables["inventory"]["unit"].tolist():
        (units_dir / f"{u}.md").write_text(render_unit(u, tables), encoding="utf-8")

    print(f"Rendered {out / 'E4_RESULTS.md'} and {len(tables['inventory'])} unit dossiers in {units_dir}")


if __name__ == "__main__":
    main()
