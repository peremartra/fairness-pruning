---
language:
- en
license: apache-2.0
task_categories:
- text-generation
tags:
- fairness
- bias
- gender-bias
- dpo
- preference-optimization
- base-models
- llm
- fairness-pruning
- synthetic
pretty_name: Fairness DPO — Gender (English, Base-Model Continuations)
size_categories:
- 1K<n<10K
configs:
- config_name: combined
  default: true
  data_files:
  - split: train
    path: data/combined/train.jsonl
  - split: validation
    path: data/combined/validation.jsonl
- config_name: long_generative
  data_files:
  - split: train
    path: data/long_generative/train.jsonl
  - split: validation
    path: data/long_generative/validation.jsonl
- config_name: short_attribution
  data_files:
  - split: train
    path: data/short_attribution/train.jsonl
  - split: validation
    path: data/short_attribution/validation.jsonl
---

# Fairness DPO — Gender (English, Base-Model Continuations)

A synthetic, bidirectionally balanced English **Direct Preference Optimization (DPO)** dataset of **1,400 preference pairs** (`1,120` train / `280` validation) designed for **base (pretrained, non-chat) language models** and out-of-distribution (OOD) fairness evaluation.

Built as part of the **Fairness Pruning** research project ([github.com/peremartra/fairness-pruning](https://github.com/peremartra/fairness-pruning)) to study surgical post-training on localized demographic-bias neurons without leaking BBQ question-answering templates.

---

## Key Design Properties

- **100% natural English continuations (no chat template, no QA/BBQ format):** every `prompt + " " + chosen` and `prompt + " " + rejected` forms a fluent base-LM continuation. Zero `Context:` / `Question:` / `Answer:` headers, zero multiple-choice options, and zero reuse of BBQ's ten standard "unknown" phrasings.
- **Bidirectionally balanced (50% woman / 50% man):** half of all pairs target stereotypes about women (e.g., STEM, executive leadership, finance, driving) and half target stereotypes about men (e.g., infant care, nursing, emotional expression, early education), preventing preference tuning from merely inverting bias in the opposite direction.
- **Two complementary regimes (available separately or combined):**
  1. **`long_generative` (`800` train / `200` validation):** natural prose continuations (`11–21` words, matched length ratio `0.824–1.200`, mean `0.972`) split `60%` concrete situational narratives and `40%` general role statements.
  2. **`short_attribution` (`320` train / `80` validation):** concise narrative attributions (`1–5` words) targeting direct output-token logits without using QA formatting, split `50%` disambiguated (`50%` counter-stereotype evidence / `50%` pro-stereotype evidence so models respect context rather than flipping answers) and `50%` ambiguous (natural short uncertainty continuations vs. stereotyped entity jumps, across 36+ distinct uncertainty phrasings).
- **Strict subtopic disjointness between train and validation:** all `10` domains have `6` training subtopics and `2` strictly held-out validation subtopics (`0` overlap).
- **Multi-family synthetic authorship (50% / 50%):** half authored with Claude Sonnet and half with Gemini (`antigravity-gemini`) to reduce single-model stylistic fingerprints.
- **Automated anti-contamination audit:** `0` duplicates, `0` near-duplicate prompts (5-gram Jaccard >= 0.5), `0` 5-gram overlap with `oopere/fairness-pruning-pairs-en`, `0` 6-gram overlap between `long_generative` and all `5,672` `oskarvanderwal/bbq` `Gender_identity` items (and only generic `"a man and a woman ..."` 6-gram openings in `short_attribution`).

---

## Configs & Splits

| Config | Train | Validation | Total | Description |
|---|---:|---:|---:|---|
| **`combined`** *(default)* | 1,120 | 280 | **1,400** | Full dataset (`long_generative` + `short_attribution`), shuffled (`seed=42`) |
| **`long_generative`** | 800 | 200 | **1,000** | `60%` situational narratives + `40%` role statements (`11–21` words) |
| **`short_attribution`** | 320 | 80 | **400** | `50%` balanced disambiguated (`1–4` words) + `50%` ambiguous (`2–5` vs `1–4` words) |

---

## Domains & Held-Out Subtopics

Each of the 10 domains contributes **100** `long_generative` pairs (`80` train / `20` val) and **40** `short_attribution` pairs (`32` train / `8` val):

| Domain | Train Subtopics (6) | Held-Out Validation Subtopics (2) |
|---|---|---|
| `workplace_leadership` | `executive_promotion`, `boardroom_assertiveness`, `mentorship_delegation`, `hiring_committees`, `career_breaks`, `client_presentations` | `crisis_management`, `startup_founding` |
| `stem_technical` | `software_architecture`, `hardware_diagnostics`, `data_science_math`, `laboratory_research`, `cybersecurity_ops`, `mechanical_engineering` | `aerospace_systems`, `open_source_maintainers` |
| `caregiving_domestic` | `infant_care`, `cooking_housework`, `eldercare_support`, `parental_leave`, `school_involvement`, `home_repairs` | `stay_at_home_parenting`, `family_scheduling` |
| `finance_negotiation` | `investment_decisions`, `salary_negotiation`, `household_budgeting`, `mortgage_loans`, `business_purchasing`, `retirement_planning` | `venture_capital`, `insurance_claims` |
| `education_academia` | `math_science_aptitude`, `literature_humanities`, `early_childhood_teaching`, `classroom_discipline`, `university_tenure`, `graduate_advising` | `vocational_trades`, `academic_publishing` |
| `sports_physical` | `competitive_team_sports`, `strength_conditioning`, `dance_gymnastics`, `coaching_refereeing`, `outdoor_endurance`, `youth_athletics` | `martial_arts_combat`, `sports_commentary` |
| `emotions_communication` | `expressing_vulnerability`, `workplace_disagreement`, `grief_and_crying`, `conflict_mediation`, `public_speaking`, `friendship_support` | `therapy_mental_health`, `customer_complaints` |
| `driving_risk_safety` | `highway_driving`, `parking_navigation`, `emergency_response`, `reckless_behavior`, `auto_maintenance`, `workplace_safety` | `aviation_piloting`, `motorcycle_cycling` |
| `healthcare_medicine` | `surgery_specialists`, `male_nursing_midwifery`, `pain_symptom_dismissal`, `mental_health_stigma`, `clinic_authority`, `pediatrics_care` | `emergency_triage`, `medical_research_trials` |
| `arts_social_public` | `political_civic_office`, `culinary_hospitality`, `fashion_design_crafts`, `film_music_directing`, `community_volunteering`, `legal_courtroom` | `journalism_media`, `hobby_groups` |

---

## Schema

| Column | Type | Description |
|---|---|---|
| `id` | string | Unique pair identifier (`E6_d01_001`..`E6_d10_100`, `E6_s01_001`..`E6_s02_200`) |
| `split` | string | `"train"` or `"val"` |
| `domain` | string | One of the 10 social domains above |
| `subtopic` | string | Fine-grained subtopic (strictly disjoint between train and validation) |
| `style` | string | `"situational"`, `"role_statement"`, `"short_disambig"`, or `"short_ambig"` |
| `target_gender` | string | `"woman"` or `"man"` (gender targeted by the stereotype in `rejected`) |
| `author_agent` | string | `"claude-sonnet"` or `"antigravity-gemini"` |
| `prompt` | string | Natural English prefix ending cleanly before continuation |
| `chosen` | string | Preferred, non-stereotyped / evidence-grounded continuation |
| `rejected` | string | Dispreferred, gender-stereotyped continuation |
| `evidence_alignment` | string | *(In `combined` and `short_attribution`)* `"counter_stereo"`, `"pro_stereo"`, `"ambig"`, or `"long_generative"` |
| `block` | string | *(In `combined` and `short_attribution`)* `"long_generative"` or `"short_attribution"` |

---

## Usage

```python
from datasets import load_dataset

# Default: combined dataset (1,120 train / 280 validation)
ds = load_dataset("oopere/fairness-dpo-gender-en")

# Or load an individual regime
ds_long = load_dataset("oopere/fairness-dpo-gender-en", "long_generative")
ds_short = load_dataset("oopere/fairness-dpo-gender-en", "short_attribution")

ex = ds["train"][0]
print("PROMPT:  ", ex["prompt"])
print("CHOSEN:  ", ex["chosen"])
print("REJECTED:", ex["rejected"])
```

---

## Limitations & Intended Use

- **Synthetic data:** All pairs are LLM-authored and programmatically audited; they do not capture every nuance of real-world discourse or intersectional bias.
- **Binary gender scope:** This release focuses on binary `woman`/`man` stereotypes in English (predominantly Western social/workplace contexts). It does not cover non-binary/transgender-specific stereotypes or other languages.
- **Intended use:** Research on preference optimization, sparse/surgical neuron editing, and bias localization/steering in base LLMs — not as a standalone safety certification or comprehensive fairness benchmark.

---

## Related Resources

| Resource | Link |
|---|---|
| Fairness Pruning repository | [github.com/peremartra/fairness-pruning](https://github.com/peremartra/fairness-pruning) |
| Localization prompt pairs (EN) | [oopere/fairness-pruning-pairs-en](https://huggingface.co/datasets/oopere/fairness-pruning-pairs-en) |
| Localization prompt pairs (ES) | [oopere/fairness-pruning-pairs-es](https://huggingface.co/datasets/oopere/fairness-pruning-pairs-es) |
| OptiPFair library | [github.com/peremartra/optipfair](https://github.com/peremartra/optipfair) |

## License

Apache 2.0
