# PR: Repository improvements — bug fix, CaBBQ configs, and test infrastructure

## Bug Fix

- **Fixed silent data bug in `esbbq_age.yaml`** — both `task` and `dataset_name` were copy-pasted from `esbbq_gender.yaml` and never updated, so age bias evaluations were silently running on the Gender subset instead of Age

## New: CaBBQ (Catalan) Task Configs

- Added 10 lm-eval YAML configs in `custom_tasks/cabbq/` for the Catalan BBQ bias benchmark (`BSC-LT/CaBBQ`)
- Covers all bias categories: Age, Gender, Religion, Race/Ethnicity, SES, Disability Status, LGBTQIA, Nationality, Physical Appearance, Spanish Region
- Same structure as the existing EsBBQ configs

## New: Test Infrastructure

- Added `tests/test_task_configs.py` with automated validation for all 22 YAML task configs
- **Offline checks** (no internet needed, run with `pytest tests/ -m "not network"`):
  - YAML files parse without errors
  - Required fields are present (`task`, `dataset_path`, `output_type`, `test_split`, `metric_list`)
  - Filename matches `task:` value (e.g. `esbbq_age.yaml` must have `task: esbbq_age`) — this is exactly the bug we found and fixed
  - Task name prefix matches its parent directory (`esbbq/` files start with `esbbq_`, etc.)
  - `dataset_name` is present for esbbq/cabbq configs (HuggingFace subset selection)
- **Network checks** (run with `pytest tests/`):
  - Each dataset loads from HuggingFace
  - The `test` split exists
  - Required columns exist for BBQ datasets (`context`, `question`, `ans0`, `ans1`, `ans2`, `label`)
- Added `pytest.ini` to register the `network` marker

## New: requirements.txt

- Added project dependencies with `lm-eval` pinned to 0.4.8
- Added `pytest` for the test suite

## How to verify

```bash
# Run the offline tests (fast, no internet)
pytest tests/ -m "not network"

# Run everything including HuggingFace dataset checks
pytest tests/
```