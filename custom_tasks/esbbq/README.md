# EsBBQ - YAML Configuration Files for lm-evaluation-harness

## 10 YAML Files Generated:

1. **esbbq_age_final.yaml** - Age (4,068 instances)
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

Each YAML file follows this structure:

- `task`: unique name per category (e.g., esbbq_age)
- `dataset_name`: exact subset name from HuggingFace
- `doc_to_choice`: ["{{ans0}}", "{{ans1}}", "{{ans2}}"] ← with double braces
- `group: esbbq` ← enables running all tasks together
- `num_fewshot: 0` ← zero-shot evaluation as per paper

## Evaluation Commands:

### Run ALL categories:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B \
  --tasks esbbq \
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
  --tasks esbbq \
  --include_path /path/to/yaml/folder \
  --limit 10
```

## Expected Output:

Results will automatically appear disaggregated by category:

```
esbbq_age: 
  acc: 0.XX
  acc_norm: 0.XX
esbbq_gender:
  acc: 0.XX
  acc_norm: 0.XX
...
```

## Post-processing Bias Scores:

The YAML files compute `acc` and `acc_norm`, but NOT the bias metrics from the paper.

To calculate `bias_score_ambig` and `bias_score_disambig` (Equations 3 and 4 from the paper), use the provided script:
```bash
python calculate_bias_scores.py --results results.json --dataset BSC-LT/EsBBQ
```

## Contributing to BSC Repository:

Once validated, these YAML files can be contributed to the official repository:
https://github.com/langtech-bsc/EsBBQ-CaBBQ

This would be a valuable contribution to enable EsBBQ evaluation with lm-evaluation-harness.

## Important Notes:

- All YAMLs use the same prompt format from the paper (Section 4.2)
- Zero-shot evaluation (no examples)
- Evaluation based on log-likelihood of each option
- Decontamination enabled to avoid data leakage

## Dataset Reference:

- **Paper**: Ruiz-Fernández et al. (2025). "EsBBQ: A Spanish Bias Benchmark for Question Answering" arXiv:2507.11216
- **Dataset**: https://huggingface.co/datasets/BSC-LT/EsBBQ
- **Code**: https://github.com/langtech-bsc/EsBBQ-CaBBQ

## Citation:

If you use these configurations, please cite the original EsBBQ paper:

```bibtex
@article{ruiz2025esbbq,
  title={EsBBQ: A Spanish Bias Benchmark for Question Answering},
  author={Ruiz-Fernández, et al.},
  journal={arXiv preprint arXiv:2507.11216},
  year={2025}
}
```
