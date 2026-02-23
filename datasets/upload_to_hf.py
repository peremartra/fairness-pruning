"""
Upload to HuggingFace — Fairness Pruning TFM UIMP
==================================================
Splits prompt_pairs_en.json and prompt_pairs_es.json by category
and uploads them using the subdirectory structure that HuggingFace
reliably recognizes as separate configs:

    data/{Category}/test.json

This structure is auto-detected by HF without relying on README
YAML frontmatter, which can be ignored when HF auto-indexes files.

Usage:
    python upload_to_hf.py --lang en
    python upload_to_hf.py --lang es
    python upload_to_hf.py --lang both

Requirements:
    pip install huggingface_hub

Authentication:
    huggingface-cli login
    # or set HF_TOKEN environment variable
"""

import json
import argparse
import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ── Configuration ──────────────────────────────────────────────────────────────

REPOS = {
    "en": {
        "repo_id": "oopere/fairness-pruning-pairs-en",
        "source":  "prompt_pairs_en.json",
        "card":    "HF_DATASET_CARD_EN.md",
    },
    "es": {
        "repo_id": "oopere/fairness-pruning-pairs-es",
        "source":  "prompt_pairs_es.json",
        "card":    "HF_DATASET_CARD_ES.md",
    },
}


# ── Core functions ─────────────────────────────────────────────────────────────

def split_by_category(source_path: str) -> dict:
    """Load JSON and split records by category."""
    with open(source_path, encoding="utf-8") as f:
        records = json.load(f)

    by_category = {}
    for record in records:
        cat = record["category"]
        by_category.setdefault(cat, []).append(record)

    return by_category


def upload_dataset(lang: str, api: HfApi) -> None:
    config  = REPOS[lang]
    repo_id = config["repo_id"]
    source  = config["source"]
    card    = config["card"]

    print(f"\n{'═'*55}")
    print(f"  Uploading: {repo_id}")
    print(f"{'═'*55}")

    # ── Create repo if it doesn't exist ───────────────────────────────────────
    print(f"\n  Creating/verifying repository...")
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )
    print(f"  ✅ Repository ready: https://huggingface.co/datasets/{repo_id}")

    # ── Split by category ─────────────────────────────────────────────────────
    print(f"\n  Splitting {source} by category...")
    by_category = split_by_category(source)

    for category, records in by_category.items():
        print(f"    {category:<25} {len(records):>4} pairs")

    # ── Build folder structure and upload in a single commit ──────────────────
    #
    # HuggingFace auto-detects configs from subdirectory structure:
    #   data/{ConfigName}/test.json  →  config "ConfigName", split "test"
    #
    # This is more reliable than YAML frontmatter configs, which can be
    # ignored when HF indexes the repo before README is processed.
    #
    print(f"\n  Building subdirectory structure: data/{{Category}}/test.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Write data/{Category}/test.json for each category
        for category, records in by_category.items():
            cat_dir = tmp_path / "data" / category
            cat_dir.mkdir(parents=True)
            out_file = cat_dir / "test.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"    Prepared  data/{category}/test.json  ({len(records)} pairs)")

        # Copy dataset card as README.md
        if Path(card).exists():
            shutil.copy(card, tmp_path / "README.md")
            print(f"    Prepared  README.md  (from {card})")
        else:
            print(f"\n  ⚠️  Card file not found: {card} — uploading without README")

        # Single commit — all files at once
        print(f"\n  Uploading all files in a single commit...")
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Restructure: use data/{Category}/test.json for HF subset auto-detection",
        )

    print(f"\n  ✅ Done: https://huggingface.co/datasets/{repo_id}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Upload fairness pruning datasets to HuggingFace"
    )
    parser.add_argument(
        "--lang",
        choices=["en", "es", "both"],
        default="both",
        help="Which dataset to upload (default: both)",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    api   = HfApi(token=token)

    langs = ["en", "es"] if args.lang == "both" else [args.lang]

    for lang in langs:
        source = REPOS[lang]["source"]
        if not Path(source).exists():
            print(f"  ❌ Source file not found: {source} — skipping {lang}")
            continue
        upload_dataset(lang, api)

    print("All uploads complete.")


if __name__ == "__main__":
    main()
