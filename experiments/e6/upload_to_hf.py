"""Upload the E6 Gender DPO dataset to Hugging Face Hub.

Packages three configs (each with `train` and `validation` splits) using HF's
subdirectory auto-detection layout + explicit YAML frontmatter in README.md:

    data/combined/train.jsonl
    data/combined/validation.jsonl
    data/long_generative/train.jsonl
    data/long_generative/validation.jsonl
    data/short_attribution/train.jsonl
    data/short_attribution/validation.jsonl
    audit/dataset_audit.json
    audit/dataset_combined_audit.json
    README.md

Usage:
    python experiments/e6/upload_to_hf.py --private
    python experiments/e6/upload_to_hf.py --public
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, create_repo

E6_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ID = "oopere/fairness-dpo-gender-en"

FILE_MAP = {
    "data/combined/train.jsonl": E6_DIR / "e6_gender_dpo_combined_train.jsonl",
    "data/combined/validation.jsonl": E6_DIR / "e6_gender_dpo_combined_val.jsonl",
    "data/long_generative/train.jsonl": E6_DIR / "e6_gender_dpo_train.jsonl",
    "data/long_generative/validation.jsonl": E6_DIR / "e6_gender_dpo_val.jsonl",
    "data/short_attribution/train.jsonl": E6_DIR / "e6_gender_dpo_short_train.jsonl",
    "data/short_attribution/validation.jsonl": E6_DIR / "e6_gender_dpo_short_val.jsonl",
    "audit/dataset_audit.json": E6_DIR / "dataset_audit.json",
    "audit/dataset_combined_audit.json": E6_DIR / "dataset_combined_audit.json",
    "README.md": E6_DIR / "HF_DATASET_CARD.md",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload fairness-dpo-gender-en to Hugging Face Hub.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"HF dataset repo id (default: {DEFAULT_REPO_ID})")
    vis = parser.add_mutually_exclusive_group()
    vis.add_argument("--private", action="store_true", default=True, help="Create/keep repo private (default)")
    vis.add_argument("--public", action="store_true", help="Create/upload as public repo")
    parser.add_argument("--commit-message", default="Initial upload: fairness-dpo-gender-en (1,400 pairs, 3 configs)")
    args = parser.parse_args()

    private = not args.public

    for dst_rel, src_path in FILE_MAP.items():
        if not src_path.exists():
            raise FileNotFoundError(f"Missing required source file for {dst_rel}: {src_path}")

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    who = api.whoami()
    print(f"Authenticated as: {who.get('name')}")

    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=private,
        token=token,
    )
    print(f"Repository ready ({'private' if private else 'public'}): https://huggingface.co/datasets/{args.repo_id}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for dst_rel, src_path in FILE_MAP.items():
            out = tmp_path / dst_rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, out)
            print(f"  Staged {dst_rel} ({out.stat().st_size:,} bytes)")

        commit_info = api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=args.commit_message,
        )
        print(f"Uploaded successfully: {commit_info.commit_url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
