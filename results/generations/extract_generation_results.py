#!/usr/bin/env python3
"""
extract_generation_results.py
Extract base-model vs. zeroed-model generation comparison tables from
Jupyter notebook cell outputs and save as structured JSON files.

Usage (from repository root):
    python results/generations/extract_generation_results.py
"""

import json
import re
import html as html_module
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

# ══════════════════════════════════════════════════════════════════════════
# NOTEBOOK CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

NOTEBOOKS = [
    {
        "path":         "notebooks/07B_EvalPrunedModels_Llama3B.ipynb",
        "model_id":     "meta-llama/Llama-3.2-3B",
        "model_key":    "llama-3.2-3B",
        "language":     "en",
        "output_dir":   "results/generations/llama-3.2-3b/en",
        "has_fairness": True,
    },
    {
        "path":         "notebooks/07_EvalPrunedModels.ipynb",
        "model_id":     "meta-llama/Llama-3.2-1B",
        "model_key":    "llama-3.2-1B",
        "language":     "en",
        "output_dir":   "results/generations/llama-3.2-1b/en",
        "has_fairness": True,
    },
    {
        "path":         "notebooks/08_EvalSalamandraPrunedEsp.ipynb",
        "model_id":     "BSC-LT/salamandra-2b",
        "model_key":    "salamandra-2B",
        "language":     "es",
        "output_dir":   "results/generations/salamandra-2b/es",
        "has_fairness": False,
    },
]

# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def compute_diff(base: str, zeroed: str) -> int:
    """
    Difference score 0-10.
    0 = responses are identical.
    10 = responses are completely different.
    Uses SequenceMatcher on normalised lowercase text.
    """
    if not base and not zeroed:
        return 0
    if not base or not zeroed:
        return 10
    sim = SequenceMatcher(None, base.lower().strip(), zeroed.lower().strip()).ratio()
    return round((1 - sim) * 10)


def extract_categories_from_name(exp_name: str) -> list:
    """
    'Religion | down_bias | Top-5'  → ['Religion']
    'RaceEthnicity | fairness | Top-20' → ['RaceEthnicity']
    """
    parts = [p.strip() for p in exp_name.split("|")]
    return [parts[0]] if parts else ["Unknown"]


def _html_text(raw: str) -> str:
    """Strip HTML tags and unescape entities from a fragment."""
    return html_module.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def parse_generation_html_table(html_str: str) -> list:
    """
    Parse a pandas-Styler HTML table whose columns are exactly
    (prompt, base_model, zeroed_model) — in any order.

    Returns a list of dicts:
        [{"prompt": ..., "base_model_response": ...,
          "zeroed_model_response": ..., "diff": ...}, ...]
    or [] if the table doesn't match the expected schema.
    """
    # ── 1. Extract column headers ──────────────────────────────────────
    # Pandas Styler uses class="col_heading level0 colN"
    header_pat = re.compile(
        r'<th[^>]+class="[^"]*col_heading[^"]*\bcol(\d+)\b[^"]*"[^>]*>(.*?)</th>',
        re.DOTALL,
    )
    headers = {}
    for m in header_pat.finditer(html_str):
        col_idx = int(m.group(1))
        name    = _html_text(m.group(2))
        headers[col_idx] = name

    # Must contain the three generation columns
    header_names = set(headers.values())
    if not {"prompt", "base_model", "zeroed_model"}.issubset(header_names):
        # Fallback: if no headers found at all, assume col0/1/2 → prompt/base/zeroed
        if not headers:
            headers = {0: "prompt", 1: "base_model", 2: "zeroed_model"}
        else:
            return []  # wrong table type (e.g. detection prompts)

    # Build reverse map: column name → column index
    col_of = {v: k for k, v in headers.items()}

    # ── 2. Extract cell data ───────────────────────────────────────────
    cell_pat = re.compile(
        r'<td[^>]+id="T_[^"]+_row(\d+)_col(\d+)"[^>]*>(.*?)</td>',
        re.DOTALL,
    )
    rows_data: dict = {}
    for m in cell_pat.finditer(html_str):
        row_idx = int(m.group(1))
        col_idx = int(m.group(2))
        content = _html_text(m.group(3))
        rows_data.setdefault(row_idx, {})[col_idx] = content

    # ── 3. Build result list ───────────────────────────────────────────
    result = []
    for row_idx in sorted(rows_data.keys()):
        rd     = rows_data[row_idx]
        prompt = rd.get(col_of.get("prompt",       0), "")
        base   = rd.get(col_of.get("base_model",   1), "")
        zeroed = rd.get(col_of.get("zeroed_model", 2), "")
        result.append({
            "prompt":                prompt,
            "base_model_response":   base,
            "zeroed_model_response": zeroed,
            "diff":                  compute_diff(base, zeroed),
        })
    return result


def extract_exp_meta_from_stdout(text: str) -> dict | None:
    """
    Detect experiment header in stdout text, e.g.
        '  [exp1] Religion | down_bias | Top-5'
        '  Neurons to zero: 5 across 3 layers'
    Returns a metadata dict or None if no match.
    """
    exp_match     = re.search(r"\[\s*(f?exp\d+)\s*\]\s+(.+?)(?:\n|$)", text)
    neurons_match = re.search(r"Neurons to zero:\s*(\d+)", text)

    if not exp_match:
        return None

    exp_id    = exp_match.group(1).strip()
    exp_name  = exp_match.group(2).strip()
    n_neurons = int(neurons_match.group(1)) if neurons_match else None

    score_type = (
        "fairness"
        if exp_id.startswith("fexp") or "fairness" in exp_name.lower()
        else "bias"
    )

    return {
        "experiment_id":   exp_id,
        "experiment_name": exp_name,
        "categories":      extract_categories_from_name(exp_name),
        "score_type":      score_type,
        "n_neurons_zeroed": n_neurons,
    }


def process_cell_outputs(outputs: list) -> list:
    """
    Walk the outputs of a single cell and pair experiment metadata
    (from stdout) with the corresponding HTML generation table.

    Returns list of (meta_dict, rows_list).
    """
    results      = []
    current_meta = None

    for output in outputs:
        otype = output.get("output_type", "")

        # ── stdout → look for experiment header ───────────────────────
        if otype == "stream" and output.get("name") == "stdout":
            raw  = output.get("text", [])
            text = "".join(raw) if isinstance(raw, list) else raw
            meta = extract_exp_meta_from_stdout(text)
            if meta:
                current_meta = meta

        # ── display_data → look for generation results table ──────────
        elif otype == "display_data":
            html_data = output.get("data", {}).get("text/html", "")
            if isinstance(html_data, list):
                html_data = "".join(html_data)

            if not html_data:
                continue

            # Quick guard before full parse
            if "base_model" not in html_data or "zeroed_model" not in html_data:
                continue

            rows = parse_generation_html_table(html_data)
            if rows and current_meta:
                results.append((dict(current_meta), rows))
                current_meta = None  # consumed by this table

    return results


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def extract_from_notebook(nb_cfg: dict) -> dict:
    """
    Parse a notebook and extract all generation comparison tables.
    Returns {"bias": [...experiments...], "fairness": [...experiments...]}.
    """
    nb_path = Path(nb_cfg["path"])
    print(f"\n{'─'*60}")
    print(f"  Notebook : {nb_path.name}")
    print(f"  Model    : {nb_cfg['model_key']} | {nb_cfg['language'].upper()}")
    print(f"{'─'*60}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    bias_experiments     = []
    fairness_experiments = []

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs", [])
        if not outputs:
            continue

        # Quick check: does this cell have any HTML with generation columns?
        has_gen_html = any(
            "base_model" in "".join(
                o.get("data", {}).get("text/html", [])
                if isinstance(o.get("data", {}).get("text/html", []), list)
                else [o.get("data", {}).get("text/html", "")]
            )
            for o in outputs
            if o.get("output_type") == "display_data"
        )
        if not has_gen_html:
            continue

        cell_results = process_cell_outputs(outputs)
        for meta, rows in cell_results:
            entry = {
                "experiment_id":    meta["experiment_id"],
                "experiment_name":  meta["experiment_name"],
                "categories":       meta["categories"],
                "n_neurons_zeroed": meta["n_neurons_zeroed"],
                "prompts":          rows,
            }
            if meta["score_type"] == "fairness":
                fairness_experiments.append(entry)
            else:
                bias_experiments.append(entry)

        print(f"  Cell → {len(cell_results)} experiments "
              f"({cell_results[0][0]['score_type'] if cell_results else '—'})")

    total = len(bias_experiments) + len(fairness_experiments)
    print(f"  Total extracted: {len(bias_experiments)} bias + "
          f"{len(fairness_experiments)} fairness experiments "
          f"({total} total)")

    return {"bias": bias_experiments, "fairness": fairness_experiments}


def save_results(nb_cfg: dict, data: dict) -> list:
    """Write bias and fairness JSONs. Returns list of saved paths."""
    out_dir = Path(nb_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    common_meta = {
        "model_id":        nb_cfg["model_id"],
        "model_key":       nb_cfg["model_key"],
        "language":        nb_cfg["language"],
        "source_notebook": Path(nb_cfg["path"]).name,
        "extracted_at":    datetime.now().isoformat(),
    }

    saved = []

    for score_type in ("bias", "fairness"):
        experiments = data[score_type]
        if not experiments:
            if score_type == "fairness" and nb_cfg.get("has_fairness"):
                print(f"  ⚠  No {score_type} results found")
            continue

        payload = {
            "metadata":    {**common_meta, "score_type": score_type},
            "experiments": experiments,
        }
        out_path = out_dir / f"{score_type}_generation_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        n_prompts = sum(len(e["prompts"]) for e in experiments)
        print(f"  ✅  {out_path}  "
              f"({len(experiments)} experiments, {n_prompts} prompt rows)")
        saved.append(out_path)

    return saved


def main():
    print("=" * 60)
    print("  Generation Results Extractor — fairness-pruning")
    print("=" * 60)

    all_saved = []
    for nb_cfg in NOTEBOOKS:
        data  = extract_from_notebook(nb_cfg)
        saved = save_results(nb_cfg, data)
        all_saved.extend(saved)

    print(f"\n{'='*60}")
    print(f"  Done. {len(all_saved)} file(s) saved:")
    for p in all_saved:
        print(f"    → {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
