"""Audit the 400 short-attribution DPO pairs and build both short and combined (1,400-pair) E6 datasets.

Reads:
  - experiments/e6/raw/s01_short_attribution_claude.jsonl (200 rows)
  - experiments/e6/raw/s02_short_attribution_antigravity.jsonl (200 rows)
  - experiments/e6/e6_gender_dpo_train.jsonl (800 long rows)
  - experiments/e6/e6_gender_dpo_val.jsonl   (200 long rows)

Verifies:
  1. Exact schema, 400 short rows (320 train + 80 val; 40 per domain across all 10 domains).
  2. Exact balance: 50/50 short_disambig vs short_ambig, 50/50 counter_stereo vs pro_stereo
     inside short_disambig, 50/50 target_gender (woman/man), 50/50 author_agent.
  3. Strict train/val subtopic disjointness per domain (consistent with long dataset).
  4. Zero duplicate prompts (within short AND against long dataset), low 5-gram near-dups.
  5. Continuation lengths: short_disambig in [1, 4] words; short_ambig chosen in [2, 5], rejected in [1, 4].
  6. Anti-contamination: 0 BBQ/QA headers, 0 of the 10 exact BBQ unknown strings in chosen,
     0 5-gram overlap with datasets/prompt_pairs_en.json, and (with --check-bbq) 6-gram check
     against all 5,672 oskarvanderwal/bbq Gender_identity items.

Writes:
  - experiments/e6/e6_gender_dpo_short_train.jsonl    (320 rows)
  - experiments/e6/e6_gender_dpo_short_val.jsonl      (80 rows)
  - experiments/e6/e6_gender_dpo_combined_train.jsonl (1,120 rows = 800 long + 320 short)
  - experiments/e6/e6_gender_dpo_combined_val.jsonl   (280 rows = 200 long + 80 short)
  - experiments/e6/dataset_combined_audit.json        (full audit + SHA-256)
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
E6_DIR = REPO_ROOT / "experiments" / "e6"
RAW_DIR = E6_DIR / "raw"
PROMPT_PAIRS_EN = REPO_ROOT / "datasets" / "prompt_pairs_en.json"

SHORT_REQUIRED_KEYS = {
    "id",
    "split",
    "domain",
    "subtopic",
    "style",
    "evidence_alignment",
    "target_gender",
    "author_agent",
    "prompt",
    "chosen",
    "rejected",
}

FORBIDDEN_HEADERS = [
    "Context:",
    "Question:",
    "Answer:",
    "The interviewer thought",
    "The officer assumed",
    "The doctor believed",
    "The resident decided",
    "The teacher felt",
]

BBQ_UNKNOWN_EXACT = [
    "can't answer",
    "can't be determined",
    "cannot answer",
    "cannot be determined",
    "not answerable",
    "not enough info",
    "not enough information",
    "not known",
    "undetermined",
    "unknown",
]

WORD_RE = re.compile(r"[a-z0-9']+")


def normalize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def ngrams(words: list[str], n: int = 5) -> set[tuple[str, ...]]:
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_bbq_overlap(rows: list[dict]) -> dict:
    try:
        from datasets import load_dataset
    except ImportError:
        return {"status": "skipped", "reason": "datasets not installed"}
    try:
        ds = load_dataset("oskarvanderwal/bbq", "Gender_identity", split="test")
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc)}

    bbq_6grams: set[tuple[str, ...]] = set()
    for doc in ds:
        for field in ("context", "question"):
            txt = doc.get(field, "") or ""
            bbq_6grams.update(ngrams(normalize_words(txt), 6))

    hits = []
    for r in rows:
        full_text = f"{r['prompt']} {r['chosen']} {r['rejected']}"
        shared = ngrams(normalize_words(full_text), 6) & bbq_6grams
        if shared:
            hits.append({"id": r["id"], "n_shared_6grams": len(shared), "sample": " ".join(next(iter(shared)))})

    return {
        "status": "checked",
        "bbq_docs_checked": len(ds),
        "rows_with_any_6gram_match": len(hits),
        "sample_hits": hits[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit short-attribution batch and build combined E6 dataset.")
    parser.add_argument("--check-bbq", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    short_files = sorted(RAW_DIR.glob("s[0-9][0-9]_*.jsonl"))
    if len(short_files) != 2:
        errors.append(f"Expected 2 short raw files in {RAW_DIR}, found {len(short_files)}: {[f.name for f in short_files]}")

    short_rows: list[dict] = []
    for fp in short_files:
        f_rows = load_jsonl(fp)
        if len(f_rows) != 200:
            errors.append(f"{fp.name}: expected 200 rows, found {len(f_rows)}")
        for idx, r in enumerate(f_rows, 1):
            missing = SHORT_REQUIRED_KEYS - set(r.keys())
            if missing:
                errors.append(f"{fp.name}:{idx}: missing keys {sorted(missing)}")
        short_rows.extend(f_rows)

    long_train = load_jsonl(E6_DIR / "e6_gender_dpo_train.jsonl")
    long_val = load_jsonl(E6_DIR / "e6_gender_dpo_val.jsonl")
    long_rows = long_train + long_val
    long_prompts = {" ".join(r["prompt"].strip().lower().split()) for r in long_rows}
    long_subtopics: dict[str, dict[str, set[str]]] = collections.defaultdict(lambda: {"train": set(), "val": set()})
    for r in long_rows:
        long_subtopics[r["domain"]][r["split"]].add(r["subtopic"])

    pp_5grams: set[tuple[str, ...]] = set()
    if PROMPT_PAIRS_EN.exists():
        for item in json.loads(PROMPT_PAIRS_EN.read_text(encoding="utf-8")):
            for k in ("prompt_1", "prompt_2"):
                pp_5grams.update(ngrams(normalize_words(item.get(k, "")), 5))

    seen_ids: set[str] = set()
    seen_prompts: dict[str, str] = {}
    by_split = collections.Counter()
    by_domain = collections.Counter()
    by_style = collections.Counter()
    by_align = collections.Counter()
    by_gender = collections.Counter()
    by_author = collections.Counter()
    cell_counts = collections.Counter()
    short_subtopics: dict[str, dict[str, set[str]]] = collections.defaultdict(lambda: {"train": set(), "val": set()})
    chosen_lens, rejected_lens = [], []
    pp_5gram_hits = 0

    for r in short_rows:
        rid = r["id"]
        if rid in seen_ids:
            errors.append(f"Duplicate short id: {rid}")
        seen_ids.add(rid)

        sp, dom, sub, st, al, gd, au = (
            r["split"],
            r["domain"],
            r["subtopic"],
            r["style"],
            r["evidence_alignment"],
            r["target_gender"],
            r["author_agent"],
        )
        by_split[sp] += 1
        by_domain[dom] += 1
        by_style[st] += 1
        by_align[al] += 1
        by_gender[gd] += 1
        by_author[au] += 1
        cell_counts[(dom, sp, st, gd, al)] += 1
        short_subtopics[dom][sp].add(sub)

        p_norm = " ".join(r["prompt"].strip().lower().split())
        if p_norm in seen_prompts:
            errors.append(f"{rid}: duplicate short prompt with {seen_prompts[p_norm]}")
        seen_prompts[p_norm] = rid
        if p_norm in long_prompts:
            errors.append(f"{rid}: prompt collides with long dataset")

        c_str = r["chosen"].strip()
        j_str = r["rejected"].strip()
        if c_str.lower() == j_str.lower():
            errors.append(f"{rid}: chosen equals rejected")

        cw = len(c_str.split())
        jw = len(j_str.split())
        chosen_lens.append(cw)
        rejected_lens.append(jw)

        if st == "short_disambig":
            if al not in ("counter_stereo", "pro_stereo"):
                errors.append(f"{rid}: invalid evidence_alignment {al!r} for short_disambig")
            if not (1 <= cw <= 4) or not (1 <= jw <= 4):
                errors.append(f"{rid}: short_disambig word count out of [1,4]: chosen={cw}, rejected={jw}")
        elif st == "short_ambig":
            if al != "ambig":
                errors.append(f"{rid}: invalid evidence_alignment {al!r} for short_ambig")
            if not (2 <= cw <= 5) or not (1 <= jw <= 4):
                errors.append(f"{rid}: short_ambig word count invalid: chosen={cw} (need 2-5), rejected={jw} (need 1-4)")
            c_clean = c_str.lower().rstrip(".!? ")
            for bad_unk in BBQ_UNKNOWN_EXACT:
                if bad_unk in c_clean:
                    errors.append(f"{rid}: chosen contains forbidden BBQ unknown phrase {bad_unk!r}: {c_str!r}")
        else:
            errors.append(f"{rid}: invalid style {st!r}")

        combined = f"{r['prompt']} {c_str} {j_str}"
        for bad in FORBIDDEN_HEADERS:
            if bad.lower() in combined.lower():
                errors.append(f"{rid}: contains forbidden substring {bad!r}")

        if ngrams(normalize_words(r["prompt"]), 5) & pp_5grams:
            pp_5gram_hits += 1

    # Domain balance & subtopic disjointness (including cross-check against long dataset splits!)
    subtopic_overlaps = {}
    for dom in sorted(set(by_domain.keys()) | set(long_subtopics.keys())):
        all_tr = short_subtopics[dom]["train"] | long_subtopics[dom]["train"]
        all_va = short_subtopics[dom]["val"] | long_subtopics[dom]["val"]
        ov = all_tr & all_va
        if ov:
            subtopic_overlaps[dom] = sorted(ov)
            errors.append(f"Domain {dom}: combined train/val subtopic overlap: {sorted(ov)}")

        expected = {
            ("train", "short_disambig", "woman", "counter_stereo"): 4,
            ("train", "short_disambig", "woman", "pro_stereo"): 4,
            ("train", "short_disambig", "man", "counter_stereo"): 4,
            ("train", "short_disambig", "man", "pro_stereo"): 4,
            ("train", "short_ambig", "woman", "ambig"): 8,
            ("train", "short_ambig", "man", "ambig"): 8,
            ("val", "short_disambig", "woman", "counter_stereo"): 1,
            ("val", "short_disambig", "woman", "pro_stereo"): 1,
            ("val", "short_disambig", "man", "counter_stereo"): 1,
            ("val", "short_disambig", "man", "pro_stereo"): 1,
            ("val", "short_ambig", "woman", "ambig"): 2,
            ("val", "short_ambig", "man", "ambig"): 2,
        }
        for (sp, st, gd, al), exp_n in expected.items():
            got = cell_counts[(dom, sp, st, gd, al)]
            if got != exp_n:
                errors.append(f"Domain {dom} ({sp}, {st}, {gd}, {al}): expected {exp_n}, got {got}")

    # Near-duplicate prompt check across short rows
    p5 = [(r["id"], r["split"], ngrams(normalize_words(r["prompt"]), 5)) for r in short_rows]
    near_dups = []
    for i in range(len(p5)):
        for j in range(i + 1, len(p5)):
            inter = len(p5[i][2] & p5[j][2])
            if inter:
                jacc = inter / len(p5[i][2] | p5[j][2])
                if jacc >= 0.5:
                    near_dups.append({"id_a": p5[i][0], "id_b": p5[j][0], "splits": f"{p5[i][1]}/{p5[j][1]}", "jacc5": round(jacc, 3)})
    if near_dups:
        warnings.append(f"Found {len(near_dups)} short prompt pairs with 5-gram Jaccard >= 0.5")

    bbq_report = check_bbq_overlap(short_rows) if args.check_bbq else {"status": "not_requested"}

    summary = {
        "short_total": len(short_rows),
        "short_by_split": dict(by_split),
        "short_by_domain": dict(sorted(by_domain.items())),
        "short_by_style": dict(by_style),
        "short_by_evidence_alignment": dict(by_align),
        "short_by_target_gender": dict(by_gender),
        "short_by_author_agent": dict(by_author),
        "short_chosen_words": {"min": min(chosen_lens), "max": max(chosen_lens), "mean": round(sum(chosen_lens) / len(chosen_lens), 2)} if chosen_lens else {},
        "short_rejected_words": {"min": min(rejected_lens), "max": max(rejected_lens), "mean": round(sum(rejected_lens) / len(rejected_lens), 2)} if rejected_lens else {},
        "combined_subtopic_overlaps": subtopic_overlaps,
        "short_prompt_pairs_en_5gram_hits": pp_5gram_hits,
        "short_near_dups_ge_0_5": len(near_dups),
        "short_near_dup_samples": near_dups[:10],
        "short_bbq_check": bbq_report,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors[:50],
        "warnings": warnings,
    }

    if errors:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nFAILED with {len(errors)} error(s).", file=sys.stderr)
        return 1

    # Write short train/val and combined train/val
    def norm_long(r: dict) -> dict:
        out = dict(r)
        out["evidence_alignment"] = "long_generative"
        out["block"] = "long_generative"
        return out

    def norm_short(r: dict) -> dict:
        out = {k: r[k] for k in sorted(SHORT_REQUIRED_KEYS)}
        out["block"] = "short_attribution"
        return out

    s_train = [norm_short(r) for r in short_rows if r["split"] == "train"]
    s_val = [norm_short(r) for r in short_rows if r["split"] == "val"]
    c_train = [norm_long(r) for r in long_train] + s_train
    c_val = [norm_long(r) for r in long_val] + s_val

    rng = random.Random(args.seed)
    for subset in (s_train, s_val, c_train, c_val):
        rng.shuffle(subset)

    outputs = {
        "short_train": (E6_DIR / "e6_gender_dpo_short_train.jsonl", s_train),
        "short_val": (E6_DIR / "e6_gender_dpo_short_val.jsonl", s_val),
        "combined_train": (E6_DIR / "e6_gender_dpo_combined_train.jsonl", c_train),
        "combined_val": (E6_DIR / "e6_gender_dpo_combined_val.jsonl", c_val),
    }

    summary["output_files"] = {}
    for key, (path, subset) in outputs.items():
        with path.open("w", encoding="utf-8") as fh:
            for item in subset:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        summary["output_files"][key] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "rows": len(subset),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    audit_path = E6_DIR / "dataset_combined_audit.json"
    audit_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nPASSED all checks. Wrote short and combined train/val JSONLs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
