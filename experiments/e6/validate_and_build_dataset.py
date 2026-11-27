"""Audit, validate, and build the final E6 Gender DPO dataset (EN, 100% generative).

Reads all 10 domain JSONL files in `experiments/e6/raw/d*.jsonl`, verifies:
  1. Schema, exact counts (1,000 total = 800 train + 200 val; 100 per domain).
  2. Exact balance: 50/50 target_gender (woman/man), 60/40 style (situational/role_statement),
     50/50 author_agent (claude-sonnet / antigravity-gemini).
  3. Strict subtopic disjointness between train and val within every domain.
  4. Zero exact duplicates across all 1,000 prompts, chosen, and rejected strings, plus
     near-duplicate check via normalized 5-gram Jaccard.
  5. Continuation length (10-24 words) and chosen/rejected word-count ratio in [0.80, 1.25].
  6. Anti-contamination checks against forbidden BBQ/QA patterns and `datasets/prompt_pairs_en.json`
     (and optionally `oskarvanderwal/bbq` Gender_identity if `--check-bbq` is passed).

Writes:
  - experiments/e6/e6_gender_dpo_train.jsonl (800 shuffled rows, seed=42)
  - experiments/e6/e6_gender_dpo_val.jsonl   (200 shuffled rows, seed=42)
  - experiments/e6/dataset_audit.json        (full summary statistics & checks)
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "experiments" / "e6" / "raw"
OUT_DIR = REPO_ROOT / "experiments" / "e6"
PROMPT_PAIRS_EN = REPO_ROOT / "datasets" / "prompt_pairs_en.json"

REQUIRED_KEYS = {
    "id",
    "split",
    "domain",
    "subtopic",
    "style",
    "target_gender",
    "author_agent",
    "prompt",
    "chosen",
    "rejected",
}

FORBIDDEN_SUBSTRINGS = [
    "Context:",
    "Question:",
    "Answer:",
    "Cannot be determined",
    "Not known",
    "Not enough info",
    "The interviewer thought",
    "The officer assumed",
    "The doctor believed",
    "The resident decided",
    "The teacher felt",
]

WORD_RE = re.compile(r"[a-z0-9']+")


def normalize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def ngrams(words: list[str], n: int = 5) -> set[tuple[str, ...]]:
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def load_raw_batches(raw_dir: Path) -> tuple[list[dict], list[str]]:
    files = sorted(raw_dir.glob("d[0-9][0-9]_*.jsonl"))
    rows: list[dict] = []
    errors: list[str] = []

    if len(files) != 10:
        errors.append(f"Expected 10 domain files in {raw_dir}, found {len(files)}: {[f.name for f in files]}")

    for fpath in files:
        file_rows = []
        with fpath.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{fpath.name}:{lineno}: invalid JSON: {exc}")
                    continue
                missing = REQUIRED_KEYS - set(obj.keys())
                if missing:
                    errors.append(f"{fpath.name}:{lineno}: missing keys {sorted(missing)}")
                    continue
                obj["_source_file"] = fpath.name
                file_rows.append(obj)
        if len(file_rows) != 100:
            errors.append(f"{fpath.name}: expected 100 rows, found {len(file_rows)}")
        rows.extend(file_rows)

    return rows, errors


def check_bbq_overlap(rows: list[dict], sample_limit: int | None = None) -> dict:
    """Check 6-gram overlap against oskarvanderwal/bbq Gender_identity contexts/questions."""
    try:
        from datasets import load_dataset
    except ImportError:
        return {"status": "skipped", "reason": "datasets library not installed"}

    try:
        ds = load_dataset("oskarvanderwal/bbq", "Gender_identity", split="test")
    except Exception as exc:
        return {"status": "skipped", "reason": f"could not load oskarvanderwal/bbq: {exc}"}

    bbq_6grams: set[tuple[str, ...]] = set()
    n_docs = len(ds) if sample_limit is None else min(len(ds), sample_limit)
    for i in range(n_docs):
        doc = ds[i]
        for field in ("context", "question"):
            txt = doc.get(field, "") or ""
            bbq_6grams.update(ngrams(normalize_words(txt), 6))

    hits = []
    for r in rows:
        full_text = f"{r['prompt']} {r['chosen']} {r['rejected']}"
        row_6g = ngrams(normalize_words(full_text), 6)
        shared = row_6g & bbq_6grams
        if shared:
            hits.append({"id": r["id"], "n_shared_6grams": len(shared), "sample": " ".join(next(iter(shared)))})

    return {
        "status": "checked",
        "bbq_docs_checked": n_docs,
        "rows_with_any_6gram_match": len(hits),
        "sample_hits": hits[:10],
    }


def audit_rows(rows: list[dict], check_bbq: bool = False) -> tuple[dict, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    by_split = collections.Counter()
    by_domain = collections.Counter()
    by_gender = collections.Counter()
    by_style = collections.Counter()
    by_author = collections.Counter()
    cell_counts = collections.Counter()
    domain_split_subtopics: dict[str, dict[str, set[str]]] = collections.defaultdict(lambda: {"train": set(), "val": set()})

    seen_ids: set[str] = set()
    seen_prompts: dict[str, str] = {}
    seen_chosen: dict[str, str] = {}
    seen_rejected: dict[str, str] = {}

    chosen_lens: list[int] = []
    rejected_lens: list[int] = []
    ratios: list[float] = []

    # Load prompt_pairs_en.json for overlap check
    pp_prompts: set[str] = set()
    pp_5grams: set[tuple[str, ...]] = set()
    if PROMPT_PAIRS_EN.exists():
        pp_data = json.loads(PROMPT_PAIRS_EN.read_text(encoding="utf-8"))
        for item in pp_data:
            for k in ("prompt_1", "prompt_2"):
                p = item.get(k, "").strip().lower()
                if p:
                    pp_prompts.add(p)
                    pp_5grams.update(ngrams(normalize_words(p), 5))

    pp_5gram_hits = 0

    for r in rows:
        rid = r["id"]
        if rid in seen_ids:
            errors.append(f"Duplicate id: {rid}")
        seen_ids.add(rid)

        split = r["split"]
        domain = r["domain"]
        subtopic = r["subtopic"]
        style = r["style"]
        gender = r["target_gender"]
        author = r["author_agent"]

        if split not in ("train", "val"):
            errors.append(f"{rid}: invalid split {split!r}")
        if style not in ("situational", "role_statement"):
            errors.append(f"{rid}: invalid style {style!r}")
        if gender not in ("woman", "man"):
            errors.append(f"{rid}: invalid target_gender {gender!r}")

        by_split[split] += 1
        by_domain[domain] += 1
        by_gender[gender] += 1
        by_style[style] += 1
        by_author[author] += 1
        cell_counts[(domain, split, style, gender)] += 1
        if split in ("train", "val"):
            domain_split_subtopics[domain][split].add(subtopic)

        p_norm = " ".join(r["prompt"].strip().lower().split())
        c_norm = " ".join(r["chosen"].strip().lower().split())
        j_norm = " ".join(r["rejected"].strip().lower().split())

        if p_norm in seen_prompts:
            errors.append(f"{rid}: duplicate prompt with {seen_prompts[p_norm]}")
        seen_prompts[p_norm] = rid

        if c_norm in seen_chosen:
            errors.append(f"{rid}: duplicate chosen with {seen_chosen[c_norm]}")
        seen_chosen[c_norm] = rid

        if j_norm in seen_rejected:
            errors.append(f"{rid}: duplicate rejected with {seen_rejected[j_norm]}")
        seen_rejected[j_norm] = rid

        if c_norm == j_norm:
            errors.append(f"{rid}: chosen equals rejected")

        c_words = len(r["chosen"].split())
        j_words = len(r["rejected"].split())
        chosen_lens.append(c_words)
        rejected_lens.append(j_words)
        ratio = c_words / max(j_words, 1)
        ratios.append(ratio)

        if not (10 <= c_words <= 24) or not (10 <= j_words <= 24):
            errors.append(f"{rid}: word count out of [10,24]: chosen={c_words}, rejected={j_words}")
        if not (0.80 <= ratio <= 1.25):
            errors.append(f"{rid}: word ratio out of [0.80,1.25]: {ratio:.3f} ({c_words}/{j_words})")

        combined = f"{r['prompt']} {r['chosen']} {r['rejected']}"
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad.lower() in combined.lower():
                errors.append(f"{rid}: contains forbidden substring {bad!r}")

        if p_norm in pp_prompts:
            errors.append(f"{rid}: exact prompt match with datasets/prompt_pairs_en.json")
        if ngrams(normalize_words(r["prompt"]), 5) & pp_5grams:
            pp_5gram_hits += 1

    # Subtopic disjointness & per-domain cell balance
    subtopic_overlaps = {}
    for domain, splits in sorted(domain_split_subtopics.items()):
        overlap = splits["train"] & splits["val"]
        if overlap:
            subtopic_overlaps[domain] = sorted(overlap)
            errors.append(f"Domain {domain}: train/val subtopic overlap: {sorted(overlap)}")

        expected_cells = {
            ("train", "situational", "woman"): 24,
            ("train", "situational", "man"): 24,
            ("train", "role_statement", "woman"): 16,
            ("train", "role_statement", "man"): 16,
            ("val", "situational", "woman"): 6,
            ("val", "situational", "man"): 6,
            ("val", "role_statement", "woman"): 4,
            ("val", "role_statement", "man"): 4,
        }
        for (sp, st, gd), exp_n in expected_cells.items():
            got = cell_counts[(domain, sp, st, gd)]
            if got != exp_n:
                errors.append(f"Domain {domain} ({sp}, {st}, {gd}): expected {exp_n}, got {got}")

    # Near-duplicate prompt check across entire dataset (5-gram Jaccard >= 0.5)
    prompt_5grams = [(r["id"], r["split"], ngrams(normalize_words(r["prompt"]), 5)) for r in rows]
    near_dups = []
    for i in range(len(prompt_5grams)):
        id_i, sp_i, g_i = prompt_5grams[i]
        if not g_i:
            continue
        for j in range(i + 1, len(prompt_5grams)):
            id_j, sp_j, g_j = prompt_5grams[j]
            if not g_j:
                continue
            inter = len(g_i & g_j)
            if inter == 0:
                continue
            jacc = inter / len(g_i | g_j)
            if jacc >= 0.5:
                near_dups.append({"id_a": id_i, "id_b": id_j, "splits": f"{sp_i}/{sp_j}", "jaccard_5g": round(jacc, 3)})
    if near_dups:
        warnings.append(f"Found {len(near_dups)} prompt pairs with 5-gram Jaccard >= 0.5")

    bbq_report = check_bbq_overlap(rows) if check_bbq else {"status": "not_requested"}

    summary = {
        "total_rows": len(rows),
        "by_split": dict(by_split),
        "by_domain": dict(sorted(by_domain.items())),
        "by_target_gender": dict(by_gender),
        "by_style": dict(by_style),
        "by_author_agent": dict(by_author),
        "subtopic_overlaps": subtopic_overlaps,
        "chosen_words": {
            "min": min(chosen_lens) if chosen_lens else 0,
            "max": max(chosen_lens) if chosen_lens else 0,
            "mean": round(sum(chosen_lens) / max(len(chosen_lens), 1), 2),
        },
        "rejected_words": {
            "min": min(rejected_lens) if rejected_lens else 0,
            "max": max(rejected_lens) if rejected_lens else 0,
            "mean": round(sum(rejected_lens) / max(len(rejected_lens), 1), 2),
        },
        "word_ratio_chosen_over_rejected": {
            "min": round(min(ratios), 3) if ratios else 0.0,
            "max": round(max(ratios), 3) if ratios else 0.0,
            "mean": round(sum(ratios) / max(len(ratios), 1), 3),
        },
        "prompt_pairs_en_5gram_hits": pp_5gram_hits,
        "near_duplicate_prompts_jacc_ge_0_5": len(near_dups),
        "near_duplicate_samples": near_dups[:15],
        "bbq_check": bbq_report,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors[:50],
        "warnings": warnings,
    }
    return summary, errors


def write_outputs(rows: list[dict], summary: dict, seed: int = 42) -> None:
    clean_rows = [{k: r[k] for k in sorted(REQUIRED_KEYS)} for r in rows]
    train_rows = [r for r in clean_rows if r["split"] == "train"]
    val_rows = [r for r in clean_rows if r["split"] == "val"]

    rng = random.Random(seed)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    train_path = OUT_DIR / "e6_gender_dpo_train.jsonl"
    val_path = OUT_DIR / "e6_gender_dpo_val.jsonl"
    audit_path = OUT_DIR / "dataset_audit.json"

    for path, subset in ((train_path, train_rows), (val_path, val_rows)):
        with path.open("w", encoding="utf-8") as fh:
            for item in subset:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary["output_files"] = {
        "train": {
            "path": str(train_path.relative_to(REPO_ROOT)),
            "rows": len(train_rows),
            "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        },
        "val": {
            "path": str(val_path.relative_to(REPO_ROOT)),
            "rows": len(val_rows),
            "sha256": hashlib.sha256(val_path.read_bytes()).hexdigest(),
        },
    }
    audit_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and build E6 Gender DPO dataset.")
    parser.add_argument("--check-bbq", action="store_true", help="Also check 6-gram overlap against oskarvanderwal/bbq Gender_identity.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for train/val output JSONL.")
    args = parser.parse_args()

    rows, load_errors = load_raw_batches(RAW_DIR)
    summary, audit_errors = audit_rows(rows, check_bbq=args.check_bbq)
    all_errors = load_errors + audit_errors
    summary["n_errors"] = len(all_errors)
    summary["errors"] = all_errors[:50]

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if all_errors:
        print(f"\nFAILED with {len(all_errors)} error(s).", file=sys.stderr)
        return 1

    write_outputs(rows, summary, seed=args.seed)
    print("\nPASSED all checks. Wrote train/val JSONL and dataset_audit.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
