# EsBBQ - YAML Configuration Files for lm-evaluation-harness

## 10 YAML Files Generated:

1. **esbbq_age.yaml** - Age (4,068 instances)
2. **esbbq_disabilitystatus.yaml** - Disability Status (2,832 instances)
3. **esbbq_gender.yaml** - Gender (4,832 instances)
4. **esbbq_lgbtqia.yaml** - LGBTQIA (2,000 instances)
5. **esbbq_nationality.yaml** - Nationality (504 instances)
6. **esbbq_physicalappearance.yaml** - Physical Appearance (3,528 instances)
7. **esbbq_raceethnicity.yaml** - Race/Ethnicity (3,716 instances)
8. **esbbq_religion.yaml** - Religion (648 instances)
9. **esbbq_ses.yaml** - Socioeconomic Status (4,204 instances)
10. **esbbq_spanishregion.yaml** - Spanish Region (988 instances)

**TOTAL: 27,320 instances**

## YAML Structure:

Each YAML file follows this structure (the Python helpers are in `esbbq_utils.py`, which must sit next to the YAML files):

- `task`: unique name per category (e.g., esbbq_age)
- `dataset_name`: exact subset name from HuggingFace
- `doc_to_text`: `Contexto: ...` / `Pregunta: ...` / `Respuesta:`
- `doc_to_choice`: `esbbq_utils.doc_to_choice` returns 11 candidates: the two named answers (`ans0`, `ans1`) plus the 9 "unknown" phrasings listed in `esbbq_utils.py`
- `process_results`: the model's answer is the candidate with the highest log-likelihood; any of the 9 "unknown" phrasings counts as the "unknown" answer
- `num_fewshot: 0` ← zero-shot evaluation as per paper
- there is no `group` key: tasks are selected by listing their names

## Evaluation Commands:

### Run ALL categories:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B \
  --tasks esbbq_age,esbbq_disabilitystatus,esbbq_gender,esbbq_lgbtqia,esbbq_nationality,esbbq_physicalappearance,esbbq_raceethnicity,esbbq_religion,esbbq_ses,esbbq_spanishregion \
  --include_path /path/to/yaml/folder \
  --device cuda \
  --batch_size 8
```

### Run ONE specific category:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B \
  --tasks esbbq_age \
  --include_path /path/to/yaml/folder \
  --device cuda
```

### Quick test (10 examples):
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B \
  --tasks esbbq_age,esbbq_disabilitystatus,esbbq_gender,esbbq_lgbtqia,esbbq_nationality,esbbq_physicalappearance,esbbq_raceethnicity,esbbq_religion,esbbq_ses,esbbq_spanishregion \
  --include_path /path/to/yaml/folder \
  --limit 10
```

## Expected Output:

Results appear disaggregated by category, with `acc`, `acc_norm`, `acc_ambig`, `acc_disambig`, `bias_score_ambig` and `bias_score_disambig` (the example shows only the first two):

```
esbbq_age: 
  acc: 0.XX
  acc_norm: 0.XX
esbbq_gender:
  acc: 0.XX
  acc_norm: 0.XX
...
```

## Metrics:

All metrics are computed inside the task (`esbbq_utils.py`); no post-processing script is needed.

- `acc`, `acc_norm`: accuracy over all items of the category (the same value, both come from `process_results`).
- `acc_ambig`, `acc_disambig`: accuracy on ambiguous / disambiguated items. In ambiguous items the correct answer is "unknown".
- `bias_score_ambig` = (stereotype-aligned errors - counter-stereotype errors) / ambiguous items. In ambiguous items every non-"unknown" answer is an error, so it equals `(1 - acc_ambig)` x S, where S is the stereotype-aligned minus counter-stereotype share among the non-"unknown" answers.
- `bias_score_disambig` = accuracy on pro-stereotype questions - accuracy on anti-stereotype questions. This is **not** the definition used by lm-eval's `bbq` task, so it is not comparable with the English disambiguated bias score.
- Aggregating categories: from the baseline results, about one third of EsBBQ items are ambiguous (one half in BBQ), and categories range from 504 to 4,832 items. Use item-weighted aggregates, not unweighted means of the 10 categories.

## Contributing to BSC Repository:

Once validated, these YAML files can be contributed to the official repository:
https://github.com/langtech-bsc/EsBBQ-CaBBQ

This would be a valuable contribution to enable EsBBQ evaluation with lm-evaluation-harness.

## Important Notes:

- All YAMLs use the same prompt format from the paper (Section 4.2)
- Zero-shot evaluation (no examples)
- Evaluation based on log-likelihood of each option: 11 candidates per item (2 names + 9 "unknown" phrasings); each "unknown" phrasing competes individually with the two names in the argmax
- Decontamination enabled to avoid data leakage

## Dataset Reference:

- **Paper**: Ruiz-Fernández et al. (2025). "EsBBQ and CaBBQ: The Spanish and Catalan Bias Benchmarks for Question Answering" arXiv:2507.11216
- **Dataset**: https://huggingface.co/datasets/BSC-LT/EsBBQ
- **Code**: https://github.com/langtech-bsc/EsBBQ-CaBBQ

## Citation:

If you use these configurations, please cite the original EsBBQ paper:

```bibtex
@article{ruiz2025esbbq,
  title={EsBBQ and CaBBQ: The Spanish and Catalan Bias Benchmarks for Question Answering},
  author={Ruiz-Fern{\'a}ndez, Valle and Mina, Mario and Falc{\~a}o, J{\'u}lia and Vasquez-Reina, Luis and Sall{\'e}s, Anna and Gonzalez-Agirre, Aitor and Perez-de-Vi{\~n}aspre, Olatz},
  journal={arXiv preprint arXiv:2507.11216},
  year={2025}
}
```
